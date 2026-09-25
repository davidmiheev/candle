use super::Config;
use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Conv1d, Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder};

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
    kv_cache: Option<(Tensor, Tensor)>,
    n_state: usize,
    /// Static self-attn KV [n_head, max_seq, head_dim] for graph decode.
    static_kv: Option<(Tensor, Tensor)>,
    /// Address-stable cross-attn KV [1, n_head, audio_ctx, head_dim];
    /// refreshed per request via copy_into so a captured graph stays valid.
    cross_static: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
            kv_cache: None,
            n_state,
            static_kv: None,
            cross_static: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let q = self.query.forward(x)?;
        let (k, v) = match xa {
            None => {
                let k = self.key.forward(x)?;
                let v = self.value.forward(x)?;
                (k, v)
            }
            Some(x) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = self.key.forward(x)?;
                    let v = self.value.forward(x)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        if let Some(mask) = mask {
            let mask = mask.i((0..n_ctx, 0..n_ctx))?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(&v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?;
        Ok(wv)
    }

    fn head_dim(&self) -> usize {
        self.n_state / self.n_head
    }

    fn enable_static_self(&mut self, max_seq: usize, dtype: DType, dev: &Device) -> Result<()> {
        if self.static_kv.is_none() {
            let shape = (self.n_head, max_seq, self.head_dim());
            self.static_kv = Some((
                Tensor::zeros(shape, dtype, dev)?,
                Tensor::zeros(shape, dtype, dev)?,
            ));
        }
        Ok(())
    }

    /// Compute cross K/V from encoder output and write into address-stable
    /// fixed-capacity buffers [n_head, max_actx, hd]. Segments may be shorter
    /// than capacity (the last window of an audio file usually is); rows
    /// beyond the segment length are masked at attention time.
    fn set_cross_static(&mut self, xa: &Tensor, max_actx: usize) -> Result<usize> {
        let hd = self.head_dim();
        let k = self.reshape_head(&self.key.forward(xa)?)?.contiguous()?; // [1, h, n, hd]
        let v = self.reshape_head(&self.value.forward(xa)?)?.contiguous()?;
        let n = k.dim(2)?;
        if n > max_actx {
            candle::bail!("encoder output {n} exceeds cross capacity {max_actx}");
        }
        if self.cross_static.is_none() {
            let shape = (self.n_head, max_actx, hd);
            self.cross_static = Some((
                Tensor::zeros(shape, k.dtype(), k.device())?,
                Tensor::zeros(shape, k.dtype(), k.device())?,
            ));
        }
        let (kb, vb) = self.cross_static.as_ref().unwrap();
        let kc = k.reshape((self.n_head, n, hd))?.contiguous()?;
        let vc = v.reshape((self.n_head, n, hd))?.contiguous()?;
        candle_nn::fused::static_decode::kv_write_chunk(kb, &kc, 0)?;
        candle_nn::fused::static_decode::kv_write_chunk(vb, &vc, 0)?;
        Ok(n)
    }

    /// One-token self-attention over the static buffer at the device position.
    fn forward_static_self(&mut self, x: &Tensor, pos: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let hd = self.head_dim();
        let scale = (hd as f64).powf(-0.25);
        let (kbuf, vbuf) = self
            .static_kv
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static self-attn not enabled".into()))?
            .clone();
        let max_seq = kbuf.dim(1)?;

        let q = self.query.forward(x)?; // [1, 1, n_state]
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        candle_nn::fused::static_decode::kv_write(
            &kbuf,
            &vbuf,
            &k.reshape((self.n_head, 1, hd))?.contiguous()?,
            &v.reshape((self.n_head, 1, hd))?.contiguous()?,
            pos,
        )?;
        // q scaled once; k left unscaled in the buffer -> apply the second
        // scale factor on the score (matches q*s @ (k*s)^T).
        let q = (q.reshape((1, self.n_head, 1, hd))? * scale)?;
        let qk = (q.matmul(&kbuf.unsqueeze(0)?.transpose(2, 3)?.contiguous()?)? * scale)?; // [1,h,1,max]
        let qk = qk.broadcast_add(&mask.reshape((1, 1, 1, max_seq))?)?;
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = w
            .matmul(&vbuf.unsqueeze(0)?.contiguous()?)? // [1,h,1,hd]
            .transpose(1, 2)?
            .flatten_from(2)?; // [1,1,n_state]
        self.out.forward(&wv)
    }

    /// One-token cross-attention over the fixed-capacity cross buffers with
    /// a length mask (rows past the segment's encoder length are -inf).
    fn forward_static_cross(&self, x: &Tensor, cross_mask: &Tensor) -> Result<Tensor> {
        let hd = self.head_dim();
        let scale = (hd as f64).powf(-0.25);
        let (kb, vb) = self
            .cross_static
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("cross static not prepared".into()))?;
        let max_actx = kb.dim(1)?;
        let q = (self.query.forward(x)?.reshape((1, self.n_head, 1, hd))? * scale)?;
        let qk = (q.matmul(&kb.unsqueeze(0)?.transpose(2, 3)?.contiguous()?)? * scale)?; // [1,h,1,cap]
        let qk = qk.broadcast_add(&cross_mask.reshape((1, 1, 1, max_actx))?)?;
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = w
            .matmul(&vb.unsqueeze(0)?.contiguous()?)?
            .transpose(1, 2)?
            .flatten_from(2)?;
        self.out.forward(&wv)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_attn_ln = layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn = self
            .attn
            .forward(&self.attn_ln.forward(x)?, None, mask, flush_kv_cache)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &mut self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None, flush_kv_cache)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        x + mlp
    }

    fn enable_static(&mut self, max_seq: usize, dtype: DType, dev: &Device) -> Result<()> {
        self.attn.enable_static_self(max_seq, dtype, dev)
    }

    fn set_cross_static(&mut self, xa: &Tensor, max_actx: usize) -> Result<usize> {
        match &mut self.cross_attn {
            Some((attn, _ln)) => attn.set_cross_static(xa, max_actx),
            None => Ok(0),
        }
    }

    fn forward_static(
        &mut self,
        x: &Tensor,
        pos: &Tensor,
        mask: &Tensor,
        cross_mask: &Tensor,
    ) -> Result<Tensor> {
        let attn = self
            .attn
            .forward_static_self(&self.attn_ln.forward(x)?, pos, mask)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &self.cross_attn {
            x = (&x + attn.forward_static_cross(&ln.forward(&x)?, cross_mask)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        x + mlp
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.reset_kv_cache();
        }
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(candle::DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x)?.gelu()?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x)?.gelu()?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, None, None, flush_kv_cache)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
    span: tracing::Span,
    span_final: tracing::Span,
    static_ctx: Option<WhisperStaticCtx>,
}

/// Device-resident state for the shape-static whisper decode path.
#[derive(Debug, Clone)]
pub struct WhisperStaticCtx {
    pub pos: Tensor,        // u32 [1]
    pub mask: Tensor,       // additive causal row [max_seq]
    pub cross_len: Tensor,  // u32 [1] — (segment encoder length - 1)
    pub cross_mask: Tensor, // additive length row [max_actx]
    pub max_seq: usize,
    pub max_actx: usize,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            span,
            span_final,
            static_ctx: None,
        })
    }

    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let _enter = self.span.enter();
        let last = x.dim(D::Minus1)?;
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, Some(xa), Some(&self.mask), flush_kv_cache)?;
        }
        self.ln.forward(&x)
    }

    /// Enable the shape-static decode path with a fixed sequence budget.
    pub fn enable_static_decode(&mut self, max_seq: usize, max_actx: usize) -> Result<()> {
        let dev = self.positional_embedding.device().clone();
        let dtype = self.positional_embedding.dtype();
        if self.static_ctx.is_none() {
            self.static_ctx = Some(WhisperStaticCtx {
                pos: Tensor::zeros(1, candle::DType::U32, &dev)?,
                mask: Tensor::zeros(max_seq, dtype, &dev)?,
                cross_len: Tensor::zeros(1, candle::DType::U32, &dev)?,
                cross_mask: Tensor::zeros(max_actx, dtype, &dev)?,
                max_seq,
                max_actx,
            });
        }
        for block in self.blocks.iter_mut() {
            block.enable_static(max_seq, dtype, &dev)?;
        }
        Ok(())
    }

    pub fn static_enabled(&self) -> bool {
        self.static_ctx.is_some()
    }

    /// Refresh the address-stable cross-attn K/V from a new encoder output
    /// (per request/segment; safe outside graph capture only).
    pub fn prepare_cross_static(&mut self, xa: &Tensor) -> Result<()> {
        let max_actx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?
            .max_actx;
        let mut n = 0usize;
        for block in self.blocks.iter_mut() {
            let m = block.set_cross_static(xa, max_actx)?;
            if m > 0 {
                n = m;
            }
        }
        let ctx = self.static_ctx.as_ref().unwrap();
        // Length mask: allow columns 0..n (mask_from_pos allows j <= pos).
        candle_nn::fused::static_decode::write_u32(&ctx.cross_len, (n.max(1) - 1) as u32)?;
        candle_nn::fused::static_decode::mask_from_pos(&ctx.cross_mask, &ctx.cross_len, 0)?;
        Ok(())
    }

    /// Rewind the static position (between segments; never inside a graph).
    pub fn reset_static(&mut self) -> Result<()> {
        match self.static_ctx.as_ref() {
            Some(ctx) => candle_nn::fused::static_decode::write_u32(&ctx.pos, 0),
            None => Ok(()),
        }
    }

    /// One graph-replayable decode step: [1,1] token -> post-ln hidden [1,1,d].
    /// Position enters via a device-side gather of the learned positional
    /// embedding row; the causal mask row is rebuilt from the device pos.
    pub fn forward_static(&mut self, token: &Tensor) -> Result<Tensor> {
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        candle_nn::fused::static_decode::mask_from_pos(&ctx.mask, &ctx.pos, 0)?;
        let tok = self.token_embedding.forward(token)?; // [1,1,d]
        let pe = self
            .positional_embedding
            .index_select(&ctx.pos, 0)?
            .unsqueeze(0)?; // [1,1,d]
        let mut x = tok.broadcast_add(&pe)?;
        let (pos, mask, cross_mask) = (ctx.pos.clone(), ctx.mask.clone(), ctx.cross_mask.clone());
        for block in self.blocks.iter_mut() {
            x = block.forward_static(&x, &pos, &mask, &cross_mask)?;
        }
        let x = self.ln.forward(&x)?;
        candle_nn::fused::static_decode::incr_u32(&pos)?;
        Ok(x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        let logits = {
            let _enter = self.span_final.enter();
            x.matmul(&w.t()?)?
        };
        Ok(logits)
    }

    pub fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
#[derive(Debug, Clone)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        self.encoder
            .blocks
            .iter_mut()
            .for_each(|b| b.reset_kv_cache());
        self.decoder.reset_kv_cache();
    }
}
