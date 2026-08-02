//! Qwen3.5 / Qwen3.6 text decoder (hybrid Gated-DeltaNet + full attention).
//!
//! Covers the dense checkpoints (Qwen3.5-0.8B/2B/4B/9B/27B, Qwen3.6-27B) and
//! carries an experimental MoE FFN variant for the A3B-class checkpoints.
//! Text stack only — the vision tower of the multimodal repos is skipped.
//!
//! STATUS: correctness-first scaffold (session item 4).
//! - Decode is O(1)-state recurrent (conv window + delta-rule state, f32).
//! - Prefill is NAIVE: the GDN recurrence is applied token-by-token, so the
//!   whole forward processes one position at a time. Chunked/parallel
//!   delta-rule prefill and a fused CUDA step kernel are the follow-ups.
//! - MTP layer of the 35B-A3B checkpoint is ignored (speculative-decoding
//!   on-ramp, later).
//!
//! Reference semantics were transcribed from the HF `qwen3_5` implementation
//! via kroggen/qwen3.5-c (structs, weight names, β/decay formulas, gated
//! attention with `[q|gate]`-packed q_proj, gemma-style `(1+w)` RMS norms).

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;

fn default_one() -> f64 {
    1.0
}
fn default_conv_kernel() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    pub rms_norm_eps: f64,
    /// When true (default), q_proj packs [q | gate] per head and the
    /// attention output is gated by sigmoid(gate).
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_one")]
    pub partial_rotary_factor: f64,

    /// "linear_attention" | "full_attention" per layer.
    pub layer_types: Vec<String>,
    // Gated DeltaNet dims.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    // MoE (A3B checkpoints); absent on dense models.
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
}

impl Qwen35TextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(1e7)
    }
    pub fn rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(self.partial_rotary_factor)
    }
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim() as f64) * self.rotary_factor()) as usize
    }
    fn is_linear(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|t| t == "linear_attention")
            .unwrap_or(false)
    }
}

// ── Norms ───────────────────────────────────────────────────────────────────

/// Gemma-style RMS norm: `x_hat * (1 + w)`. The `+1` is folded into the
/// weight at load so the hot path is a plain rms_norm.
#[derive(Debug, Clone)]
struct RmsNormPlus1 {
    weight: Tensor,
    eps: f64,
}

impl RmsNormPlus1 {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?.to_dtype(DType::F32)?;
        let weight = (w + 1.0)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let var = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x32.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)?.to_dtype(dt)
    }
}

fn l2norm_last(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    x.broadcast_div(&(n + 1e-6)?)
}

// ── Rotary (partial) ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rotary {
    cos: Tensor, // [max_pos, rot/2] f32
    sin: Tensor,
    rot: usize,
}

impl Rotary {
    fn new(cfg: &Qwen35TextConfig, device: &Device) -> Result<Self> {
        let rot = cfg.rotary_dim();
        let max_pos = 32768usize;
        let inv: Vec<f32> = (0..rot / 2)
            .map(|i| (1.0 / cfg.theta().powf(2.0 * i as f64 / rot as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, rot / 2), device)?;
        let pos: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (max_pos, 1), device)?;
        let freqs = pos.matmul(&inv)?; // [max_pos, rot/2]
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
            rot,
        })
    }

    /// Like [`Self::apply`] for a single token at a DEVICE-side position:
    /// the cos/sin rows are gathered by `pos` (u32 [1]) so the op chain is
    /// replayable inside a CUDA graph.
    fn apply_gather(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (_b, _h, _seq, hd) = x.dims4()?;
        let cos = self.cos.index_select(pos, 0)?.to_dtype(x.dtype())?; // [1, rot/2]
        let sin = self.sin.index_select(pos, 0)?.to_dtype(x.dtype())?;
        if self.rot == hd {
            candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
        } else {
            let xr = x.narrow(D::Minus1, 0, self.rot)?.contiguous()?;
            let xp = x.narrow(D::Minus1, self.rot, hd - self.rot)?;
            let xr = candle_nn::rotary_emb::rope_i(&xr, &cos, &sin)?;
            Tensor::cat(&[xr, xp], D::Minus1)
        }
    }

    /// x: [b, heads, seq, head_dim]; rotates the first `rot` dims
    /// (interleaved pairs, matching the reference), passes the rest through.
    fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, hd) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq)?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, pos, seq)?.to_dtype(x.dtype())?;
        if self.rot == hd {
            candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
        } else {
            let xr = x.narrow(D::Minus1, 0, self.rot)?.contiguous()?;
            let xp = x.narrow(D::Minus1, self.rot, hd - self.rot)?;
            let xr = candle_nn::rotary_emb::rope_i(&xr, &cos, &sin)?;
            Tensor::cat(&[xr, xp], D::Minus1)
        }
    }
}

// ── Full attention (gated) ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FullAttention {
    q_proj: Proj, // out = heads*head_dim (*2 when gated: [q | gate] per head)
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: RmsNormPlus1,
    k_norm: RmsNormPlus1,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    gated: bool,
    kv_cache: Option<(Tensor, Tensor)>,
    /// Preallocated [kv_heads, max_seq, head_dim] K/V for the shape-static
    /// (graph-replayable) decode path.
    static_kv: Option<(Tensor, Tensor)>,
}

fn linear_no_bias(inp: usize, out: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get((out, inp), "weight")?;
    Ok(Linear::new(w, None))
}

impl FullAttention {
    fn new(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: qproj(
                h,
                cfg.num_attention_heads * hd * if cfg.attn_output_gate { 2 } else { 1 },
                vb.pp("q_proj"),
            )?,
            k_proj: qproj(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: qproj(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o_proj: qproj(cfg.num_attention_heads * hd, h, vb.pp("o_proj"))?,
            q_norm: RmsNormPlus1::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNormPlus1::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            gated: cfg.attn_output_gate,
            kv_cache: None,
            static_kv: None,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn enable_static(&mut self, max_seq: usize, dtype: DType, dev: &Device) -> Result<()> {
        if self.static_kv.is_none() {
            let shape = (self.n_kv_heads, max_seq, self.head_dim);
            self.static_kv = Some((
                Tensor::zeros(shape, dtype, dev)?,
                Tensor::zeros(shape, dtype, dev)?,
            ));
        }
        Ok(())
    }

    /// Copy the dynamic KV cache (post-prefill) into the static buffers.
    fn migrate_kv_to_static(&mut self) -> Result<usize> {
        let (kbuf, vbuf) = self
            .static_kv
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        let (k, v) = match &self.kv_cache {
            Some((k, v)) => (k.clone(), v.clone()),
            None => return Ok(0),
        };
        let (_b, kvh, seq, hd) = k.dims4()?;
        let kc = k.reshape((kvh, seq, hd))?.contiguous()?;
        let vc = v.reshape((kvh, seq, hd))?.contiguous()?;
        candle_nn::fused::static_decode::kv_write_chunk(kbuf, &kc, 0)?;
        candle_nn::fused::static_decode::kv_write_chunk(vbuf, &vc, 0)?;
        Ok(seq)
    }

    /// One-token attention over the full static buffer at the device
    /// position. Mirrors `forward` exactly (gated q split, q/k norms,
    /// partial rotary, f32 softmax, sigmoid output gate).
    fn forward_static(
        &mut self,
        x: &Tensor, // [1, 1, hidden]
        rotary: &Rotary,
        pos: &Tensor,
        mask: &Tensor, // f32 additive row [max_seq]
    ) -> Result<Tensor> {
        let hd = self.head_dim;
        let (kbuf, vbuf) = self
            .static_kv
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?
            .clone();
        let max_seq = kbuf.dim(1)?;

        let qg = self.q_proj.forward(x)?;
        let (q, gate) = if self.gated {
            let qg = qg.reshape((1, 1, self.n_heads, 2, hd))?;
            (
                qg.i((.., .., .., 0, ..))?,
                Some(qg.i((.., .., .., 1, ..))?),
            )
        } else {
            (qg.reshape((1, 1, self.n_heads, hd))?, None)
        };
        let k = self
            .k_proj
            .forward(x)?
            .reshape((1, 1, self.n_kv_heads, hd))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((1, 1, self.n_kv_heads, hd))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = q.transpose(1, 2)?; // [1, heads, 1, hd]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let q = rotary.apply_gather(&q, pos)?;
        let k = rotary.apply_gather(&k, pos)?;

        candle_nn::fused::static_decode::kv_write(
            &kbuf,
            &vbuf,
            &k.reshape((self.n_kv_heads, 1, hd))?.contiguous()?,
            &v.reshape((self.n_kv_heads, 1, hd))?.contiguous()?,
            pos,
        )?;

        // Broadcast-GQA over the buffer; softmax in f32 like the dynamic path.
        let rep = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (hd as f64).sqrt();
        let qg4 = q.reshape((1, self.n_kv_heads, rep, hd))?;
        let kb = kbuf.unsqueeze(0)?;
        let vb = vbuf.unsqueeze(0)?;
        let att = (qg4.matmul(&kb.transpose(2, 3)?.contiguous()?)? * scale)?; // [1, kv, rep, max]
        let att = att
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.reshape((1, 1, 1, max_seq))?)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(q.dtype())?;
        let out = att.matmul(&vb.contiguous()?)?; // [1, kv, rep, hd]
        let out = out.reshape((1, self.n_heads, 1, hd))?;
        let out = if let Some(gate) = gate {
            let gate = gate.transpose(1, 2)?; // [1, heads, 1, hd]
            (out * candle_nn::ops::sigmoid(&gate.contiguous()?)?)?
        } else {
            out
        };
        let out = out.transpose(1, 2)?.reshape((1, 1, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }

    /// x: [b, seq, hidden] (naive path is called with seq == 1).
    fn forward(&mut self, x: &Tensor, rotary: &Rotary, pos: usize) -> Result<Tensor> {
        let (b, seq, _h) = x.dims3()?;
        let hd = self.head_dim;

        let qg = self.q_proj.forward(x)?;
        let (q, gate) = if self.gated {
            let qg = qg.reshape((b, seq, self.n_heads, 2, hd))?;
            (
                qg.i((.., .., .., 0, ..))?, // [b, seq, heads, hd]
                Some(qg.i((.., .., .., 1, ..))?),
            )
        } else {
            (qg.reshape((b, seq, self.n_heads, hd))?, None)
        };

        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, hd))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, hd))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q.transpose(1, 2)?; // [b, heads, seq, hd]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let q = rotary.apply(&q, pos)?;
        let k = rotary.apply(&k, pos)?.contiguous()?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2)?.contiguous()?,
                Tensor::cat(&[pv, &v], 2)?.contiguous()?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let rep = self.n_heads / self.n_kv_heads;
        let kx = repeat_kv(&k, rep)?;
        let vx = repeat_kv(&v, rep)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let q = q.contiguous()?;
        let att = (q.matmul(&kx.transpose(2, 3)?.contiguous()?)? * scale)?;
        // seq == 1 in the naive path → no causal mask needed; guard anyway.
        let att = if seq > 1 {
            let total = kx.dim(2)?;
            let mask = causal_mask(seq, total, att.device())?.to_dtype(att.dtype())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        let out = att.matmul(&vx)?; // [b, heads, seq, hd]

        // Elementwise output gate: attn * sigmoid(gate).
        let out = if let Some(gate) = gate {
            let gate = gate.transpose(1, 2)?; // [b, heads, seq, hd]
            (out * candle_nn::ops::sigmoid(&gate.contiguous()?)?)?
        } else {
            out
        };

        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, kvh, s, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kvh, rep, s, hd))?
        .contiguous()?
        .reshape((b, kvh * rep, s, hd))
}

fn causal_mask(q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
    let offset = k_len - q_len;
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..k_len).map(move |j| if j <= i + offset { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, q_len, k_len), device)
}

// ── Gated DeltaNet (linear attention) ───────────────────────────────────────

#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: Proj, // [2*key_dim + value_dim, hidden]
    in_proj_z: Proj,   // [value_dim, hidden]
    in_proj_b: Linear,   // [n_v_heads, hidden]
    in_proj_a: Linear,   // [n_v_heads, hidden]
    conv1d_weight: Tensor, // [conv_dim, kernel] f32 (depthwise)
    dt_bias: Tensor,     // [n_v_heads] f32
    neg_a: Tensor,       // [n_v_heads] f32: -exp(A_log)
    norm_weight: Tensor, // [d_v] f32
    out_proj: Proj,
    n_k_heads: usize,
    n_v_heads: usize,
    d_k: usize,
    d_v: usize,
    conv_kernel: usize,
    eps: f64,
    /// Rolling conv input window: [conv_dim, kernel] f32.
    conv_state: Option<Tensor>,
    /// Delta-rule state per v-head: [n_v_heads, d_v, d_k] f32
    /// (readout is `S · q`).
    s_state: Option<Tensor>,
    /// Address-stable buffers for the graph path (enable_static).
    conv_static: Option<Tensor>,
    s_static: Option<Tensor>,
}

impl GatedDeltaNet {
    fn new(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.linear_conv_kernel_dim;

        // conv1d checkpoint shape is [conv_dim, 1, kernel] (depthwise).
        let conv_w = vb
            .pp("conv1d")
            .get((conv_dim, 1, kernel), "weight")?
            .reshape((conv_dim, kernel))?
            .to_dtype(DType::F32)?;
        let a_log = vb.get(cfg.linear_num_value_heads, "A_log")?.to_dtype(DType::F32)?;
        let neg_a = a_log.exp()?.neg()?;
        Ok(Self {
            in_proj_qkv: qproj(h, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: qproj(h, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: linear_no_bias(h, cfg.linear_num_value_heads, vb.pp("in_proj_b"))?,
            in_proj_a: linear_no_bias(h, cfg.linear_num_value_heads, vb.pp("in_proj_a"))?,
            conv1d_weight: conv_w,
            dt_bias: vb.get(cfg.linear_num_value_heads, "dt_bias")?.to_dtype(DType::F32)?,
            neg_a,
            norm_weight: vb.pp("norm").get(cfg.linear_value_head_dim, "weight")?.to_dtype(DType::F32)?,
            out_proj: qproj(value_dim, h, vb.pp("out_proj"))?,
            n_k_heads: cfg.linear_num_key_heads,
            n_v_heads: cfg.linear_num_value_heads,
            d_k: cfg.linear_key_head_dim,
            d_v: cfg.linear_value_head_dim,
            conv_kernel: kernel,
            eps: cfg.rms_norm_eps,
            conv_state: None,
            s_state: None,
            conv_static: None,
            s_static: None,
        })
    }

    fn clear_state(&mut self) {
        self.conv_state = None;
        self.s_state = None;
    }

    /// Allocate address-stable state buffers for the graph-replayable path.
    /// s_state is already updated in place by gdn_scan; conv needs a fixed
    /// rolling-window buffer.
    fn enable_static(&mut self, dev: &Device) -> Result<()> {
        let key_dim = self.n_k_heads * self.d_k;
        let conv_dim = 2 * key_dim + self.n_v_heads * self.d_v;
        if self.conv_static.is_none() {
            self.conv_static = Some(Tensor::zeros(
                (conv_dim, self.conv_kernel),
                DType::F32,
                dev,
            )?);
        }
        if self.s_static.is_none() {
            self.s_static = Some(Tensor::zeros(
                (self.n_v_heads, self.d_v, self.d_k),
                DType::F32,
                dev,
            )?);
        }
        Ok(())
    }

    /// Copy prefill state into the address-stable buffers and REBIND the
    /// live state to them, so eager and graphed steps share storage.
    fn migrate_state_to_static(&mut self) -> Result<()> {
        let cbuf = self
            .conv_static
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("gdn static not enabled".into()))?
            .clone();
        let sbuf = self.s_static.as_ref().unwrap().clone();
        match &self.conv_state {
            Some(c) => candle_nn::fused::static_decode::copy_into(&cbuf, c)?,
            None => candle_nn::fused::static_decode::copy_into(
                &cbuf,
                &Tensor::zeros(cbuf.dims(), DType::F32, cbuf.device())?,
            )?,
        }
        match &self.s_state {
            Some(st) => candle_nn::fused::static_decode::copy_into(&sbuf, st)?,
            None => candle_nn::fused::static_decode::copy_into(
                &sbuf,
                &Tensor::zeros(sbuf.dims(), DType::F32, sbuf.device())?,
            )?,
        }
        self.conv_state = Some(cbuf);
        self.s_state = Some(sbuf);
        Ok(())
    }

    /// Graph-replayable single-token step: identical math to forward_seq at
    /// T == 1, with the conv window rolled IN PLACE on the fixed buffer
    /// (cat produces the new window, copy_into writes it back — recorded
    /// inside the graph, so replays keep state correct).
    fn forward_static_step(&mut self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let key_dim = self.n_k_heads * self.d_k;
        let value_dim = self.n_v_heads * self.d_v;
        let cbuf = self
            .conv_static
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("gdn static not enabled".into()))?
            .clone();
        let sbuf = self.s_static.as_ref().unwrap().clone();

        let x_flat = x.reshape((1, ()))?;
        let qkv = self.in_proj_qkv.forward(&x_flat)?.to_dtype(DType::F32)?; // [1, conv_dim]
        let z = self.in_proj_z.forward(&x_flat)?.to_dtype(DType::F32)?;
        let b_raw = self.in_proj_b.forward(&x_flat)?.to_dtype(DType::F32)?;
        let a_raw = self.in_proj_a.forward(&x_flat)?.to_dtype(DType::F32)?;
        let beta = candle_nn::ops::sigmoid(&b_raw)?;
        let sp = softplus(&b_cast(&a_raw, &self.dt_bias)?)?;
        let decay = sp.broadcast_mul(&self.neg_a.reshape((1, self.n_v_heads))?)?.exp()?;

        // Roll the conv window in place: new = [old[1..], x_t].
        let qkv_t = qkv.t()?.contiguous()?; // [conv_dim, 1]
        let prev = cbuf.narrow(1, 1, self.conv_kernel - 1)?;
        let padded = Tensor::cat(&[&prev, &qkv_t], 1)?.contiguous()?; // [conv_dim, kernel]
        candle_nn::fused::static_decode::copy_into(&cbuf, &padded)?;
        let mut conv = padded
            .narrow(1, self.conv_kernel - 1, 1)?
            .broadcast_mul(&self.conv1d_weight.narrow(1, self.conv_kernel - 1, 1)?)?;
        for j in 0..self.conv_kernel - 1 {
            conv = (conv
                + padded
                    .narrow(1, j, 1)?
                    .broadcast_mul(&self.conv1d_weight.narrow(1, j, 1)?)?)?;
        }
        let conved = silu_f32(&conv)?.t()?.contiguous()?; // [1, conv_dim]

        let q = conved.narrow(1, 0, key_dim)?.reshape((1, self.n_k_heads, self.d_k))?;
        let k = conved
            .narrow(1, key_dim, key_dim)?
            .reshape((1, self.n_k_heads, self.d_k))?;
        let v = conved
            .narrow(1, 2 * key_dim, value_dim)?
            .reshape((1, self.n_v_heads, self.d_v))?;
        let q = (l2norm_last(&q)? * (self.d_k as f64).powf(-0.5))?;
        let k = l2norm_last(&k)?;

        let o = candle_nn::fused::static_decode::gdn_scan(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &beta.contiguous()?,
            &decay.contiguous()?,
            &sbuf,
        )?; // [1, n_v, d_v]

        let var = o.sqr()?.mean_keepdim(D::Minus1)?;
        let o = o.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let o = o.broadcast_mul(&self.norm_weight.reshape((1, 1, self.d_v))?)?;
        let zg = silu_f32(&z.reshape((1, self.n_v_heads, self.d_v))?)?;
        let o = (o * zg)?;
        let o = o.reshape((1, 1, value_dim))?.to_dtype(dt)?;
        self.out_proj.forward(&o)
    }

    /// Single-token step. x: [b=1, 1, hidden]. All recurrence math in f32.
    fn forward_step(&mut self, x: &Tensor) -> Result<Tensor> {
        self.forward_seq(x)
    }

    /// Batched GDN over a whole [1, T, hidden] chunk: projections, causal
    /// depthwise conv, norms and gating are computed for all T positions at
    /// once; the sequential delta-rule recurrence runs inside ONE
    /// `gdn_scan` kernel launch with the state resident on-chip. T == 1 is
    /// the decode step (same path, graph-friendly: a handful of launches).
    fn forward_seq(&mut self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let dev = x.device().clone();
        let (_b, t_len, _h) = x.dims3()?;
        let key_dim = self.n_k_heads * self.d_k;
        let value_dim = self.n_v_heads * self.d_v;
        let conv_dim = 2 * key_dim + value_dim;

        let x_flat = x.reshape((t_len, ()))?; // [T, hidden]
        let qkv = self.in_proj_qkv.forward(&x_flat)?.to_dtype(DType::F32)?; // [T, conv_dim]
        let z = self.in_proj_z.forward(&x_flat)?.to_dtype(DType::F32)?; // [T, value_dim]
        let b_raw = self.in_proj_b.forward(&x_flat)?.to_dtype(DType::F32)?; // [T, n_v]
        let a_raw = self.in_proj_a.forward(&x_flat)?.to_dtype(DType::F32)?; // [T, n_v]

        // β = σ(b);   decay = exp(-exp(A_log) · softplus(a + dt_bias))
        let beta = candle_nn::ops::sigmoid(&b_raw)?; // [T, n_v]
        let sp = softplus(&b_cast(&a_raw, &self.dt_bias)?)?;
        let decay = sp.broadcast_mul(&self.neg_a.reshape((1, self.n_v_heads))?)?.exp()?;

        // Causal depthwise conv over the chunk: pad with the rolling state
        // (last kernel-1 inputs of the previous chunk, zeros at start).
        let qkv_t = qkv.t()?.contiguous()?; // [conv_dim, T]
        let prev = match &self.conv_state {
            Some(w) => w.narrow(1, 1, self.conv_kernel - 1)?,
            None => Tensor::zeros((conv_dim, self.conv_kernel - 1), DType::F32, &dev)?,
        };
        let padded = Tensor::cat(&[&prev, &qkv_t], 1)?; // [conv_dim, T + k - 1]
        self.conv_state = Some(padded.narrow(1, t_len.saturating_sub(1), self.conv_kernel)?.contiguous()?);
        let mut conv = padded
            .narrow(1, 0, t_len)?
            .broadcast_mul(&self.conv1d_weight.narrow(1, 0, 1)?)?;
        for j in 1..self.conv_kernel {
            conv = (conv
                + padded
                    .narrow(1, j, t_len)?
                    .broadcast_mul(&self.conv1d_weight.narrow(1, j, 1)?)?)?;
        }
        let conved = silu_f32(&conv)?.t()?.contiguous()?; // [T, conv_dim]

        let q = conved
            .narrow(1, 0, key_dim)?
            .reshape((t_len, self.n_k_heads, self.d_k))?;
        let k = conved
            .narrow(1, key_dim, key_dim)?
            .reshape((t_len, self.n_k_heads, self.d_k))?;
        let v = conved
            .narrow(1, 2 * key_dim, value_dim)?
            .reshape((t_len, self.n_v_heads, self.d_v))?;
        // Per-head L2 norm; q additionally scaled by d_k^-0.5 (validated
        // against the HF reference).
        let q = (l2norm_last(&q)? * (self.d_k as f64).powf(-0.5))?;
        let k = l2norm_last(&k)?;

        let st = match &self.s_state {
            Some(st) => st.clone(),
            None => {
                let z = Tensor::zeros(
                    (self.n_v_heads, self.d_v, self.d_k),
                    DType::F32,
                    &dev,
                )?;
                self.s_state = Some(z.clone());
                z
            }
        };
        // One launch for the whole recurrence; `st` is updated in place
        // (self.s_state shares storage, so decode continues seamlessly).
        let o = candle_nn::fused::static_decode::gdn_scan(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &beta.contiguous()?,
            &decay.contiguous()?,
            &st,
        )?; // [T, n_v, d_v]

        // Gated per-head RMS norm: w * x̂ * silu(z), batched over T.
        let var = o.sqr()?.mean_keepdim(D::Minus1)?;
        let o = o.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let o = o.broadcast_mul(&self.norm_weight.reshape((1, 1, self.d_v))?)?;
        let zg = silu_f32(&z.reshape((t_len, self.n_v_heads, self.d_v))?)?;
        let o = (o * zg)?;

        let o = o.reshape((1, t_len, value_dim))?.to_dtype(dt)?;
        self.out_proj.forward(&o)
    }
}

fn expand_heads(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (h, d) = x.dims2()?;
    x.unsqueeze(1)?.expand((h, group, d))?.reshape((h * group, d))
}

use crate::models::gemma4::text::qcache;
use candle::quantized::{GgmlDType, QMatMul, QTensor};
use std::sync::{Arc, Mutex, OnceLock};

fn quant_setting() -> &'static Mutex<Option<GgmlDType>> {
    static S: OnceLock<Mutex<Option<GgmlDType>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// K-quants need the reduction dim to be a multiple of 256; fall back to
/// q8_0 (32-wide blocks) when it is not.
fn quant_dtype_for(in_dim: usize, requested: GgmlDType) -> GgmlDType {
    let block = requested.block_size();
    if block > 32 && in_dim % 256 != 0 {
        GgmlDType::Q8_0
    } else if in_dim % 32 != 0 {
        // Should not happen for these checkpoints; keep unquantized-safe.
        requested
    } else {
        requested
    }
}

#[derive(Debug, Clone)]
enum Proj {
    Plain(Linear),
    Quant(QMatMul),
}

impl Module for Proj {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(x),
            Self::Quant(q) => q.forward(x),
        }
    }
}

/// Linear projection honoring the model-level quantize setting, with the
/// quantized bytes served from / persisted to the shared GGUF disk cache.
fn qproj(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Proj> {
    let quant = *quant_setting().lock().unwrap_or_else(|p| p.into_inner());
    match quant {
        None => Ok(Proj::Plain(linear_no_bias(in_dim, out_dim, vb)?)),
        Some(dtype) => {
            let dtype = quant_dtype_for(in_dim, dtype);
            let dev = vb.device().clone();
            let key = vb.prefix();
            let q = if let Some(q) = qcache::get(&key, &dev) {
                Arc::new(q)
            } else {
                let w = vb.get((out_dim, in_dim), "weight")?;
                let q = if dev.is_cuda() {
                    let cpu = w.to_device(&Device::Cpu)?;
                    QTensor::quantize_onto(&cpu, dtype, &dev)?
                } else {
                    QTensor::quantize(&w.to_dtype(DType::F32)?, dtype)?
                };
                let q = Arc::new(q);
                qcache::put(&key, &q);
                q
            };
            Ok(Proj::Quant(QMatMul::QTensor(q)))
        }
    }
}

fn b_cast(a: &Tensor, bias: &Tensor) -> Result<Tensor> {
    a.broadcast_add(&bias.reshape((1, bias.dim(0)?))?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

fn silu_f32(x: &Tensor) -> Result<Tensor> {
    x * candle_nn::ops::sigmoid(x)?
}

// ── MLP (dense) / MoE (experimental) ────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
}

fn anyhow_ok(cond: bool, msg: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        candle::bail!("{}", msg)
    }
}

fn qproj_cpu(w: &Tensor, dev: &Device, key: String) -> Result<Proj> {
    let quant = *quant_setting().lock().unwrap_or_else(|p| p.into_inner());
    match quant {
        None => Ok(Proj::Plain(Linear::new(w.to_device(dev)?, None))),
        Some(dtype) => {
            let dtype = quant_dtype_for(w.dim(1)?, dtype);
            let q = if let Some(q) = qcache::get(&key, dev) {
                Arc::new(q)
            } else {
                let q = if dev.is_cuda() {
                    QTensor::quantize_onto(w, dtype, dev)?
                } else {
                    QTensor::quantize(&w.to_dtype(DType::F32)?, dtype)?
                };
                let q = Arc::new(q);
                qcache::put(&key, &q);
                q
            };
            Ok(Proj::Quant(QMatMul::QTensor(q)))
        }
    }
}

fn qproj_tensor(w: Tensor, key: String) -> Result<Proj> {
    let quant = *quant_setting().lock().unwrap_or_else(|p| p.into_inner());
    match quant {
        None => Ok(Proj::Plain(Linear::new(w, None))),
        Some(dtype) => {
            let in_dim = w.dim(1)?;
            let dtype = quant_dtype_for(in_dim, dtype);
            let dev = w.device().clone();
            let q = if let Some(q) = qcache::get(&key, &dev) {
                Arc::new(q)
            } else {
                let q = if dev.is_cuda() {
                    let cpu = w.to_device(&Device::Cpu)?;
                    QTensor::quantize_onto(&cpu, dtype, &dev)?
                } else {
                    QTensor::quantize(&w.to_dtype(DType::F32)?, dtype)?
                };
                let q = Arc::new(q);
                qcache::put(&key, &q);
                q
            };
            Ok(Proj::Quant(QMatMul::QTensor(q)))
        }
    }
}

impl Mlp {
    /// Build from raw [out, in] weight slices (packed-expert checkpoints),
    /// routing each through the quantization/qcache path under classic
    /// per-expert key names so qcache files stay layout-agnostic.
    fn from_weights(gate_w: Tensor, up_w: Tensor, down_w: Tensor, key_prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: qproj_tensor(gate_w, format!("{key_prefix}.gate_proj"))?,
            up_proj: qproj_tensor(up_w, format!("{key_prefix}.up_proj"))?,
            down_proj: qproj_tensor(down_w, format!("{key_prefix}.down_proj"))?,
        })
    }

    /// Build from CPU-resident [out, in] slices (packed-expert checkpoints):
    /// quantize on the calling thread (CPU-bound, parallel-safe), upload the
    /// QTensor to `dev`, register in qcache under classic per-expert names.
    fn from_cpu_weights(
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        dev: &Device,
        key_prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: qproj_cpu(gate_w, dev, format!("{key_prefix}.gate_proj"))?,
            up_proj: qproj_cpu(up_w, dev, format!("{key_prefix}.up_proj"))?,
            down_proj: qproj_cpu(down_w, dev, format!("{key_prefix}.down_proj"))?,
        })
    }

    fn new(h: usize, inter: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: qproj(h, inter, vb.pp("gate_proj"))?,
            up_proj: qproj(h, inter, vb.pp("up_proj"))?,
            down_proj: qproj(inter, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?.silu()?;
        let u = self.up_proj.forward(x)?;
        self.down_proj.forward(&(g * u)?)
    }
}

/// Experimental sparse MoE FFN for the A3B checkpoints (qwen3_moe-style:
/// softmax router over `num_experts`, top-k renormalized, plus an
/// always-on shared expert). Naive per-expert loop — port to the fused
/// indexed-MoE kernel once validated.
#[derive(Debug, Clone)]
struct SparseMoe {
    gate: Linear, // router [num_experts, hidden]
    experts: Vec<Mlp>,
    shared_expert: Option<Mlp>,
    shared_expert_gate: Option<Linear>,
    top_k: usize,
}

impl SparseMoe {
    fn new(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let e = cfg.num_experts.unwrap_or(0);
        let inter = cfg
            .moe_intermediate_size
            .or(cfg.intermediate_size)
            .expect("either moe_intermediate_size or intermediate_size");
        let mut experts = Vec::with_capacity(e);
        // Modern checkpoints (Qwen3.6-A3B) pack experts into two stacked
        // tensors: gate_up_proj [E, 2I, H] (gate rows first) and down_proj
        // [E, H, I] — each expert slice is already [out, in] Linear layout.
        let evb = vb.pp("experts");
        if evb.contains_tensor("gate_up_proj") {
            let gu = evb.get((e, 2 * inter, h), "gate_up_proj")?;
            let dn = evb.get((e, h, inter), "down_proj")?;
            let kp = evb.prefix();
            let parallel = std::env::var("EXP1_PAR_EXPERT_QUANT")
                .map(|v| v == "1")
                .unwrap_or(false);
            if !parallel {
                // Batched (default): quantize each packed tensor in ONE call,
                // then split the quantized bytes at expert-row boundaries.
                // Rows are whole quantization blocks (H, I % block == 0), so
                // byte-slicing is exact and the per-expert records are
                // identical to slice-wise quantization. 2 quantize calls per
                // layer instead of 3*E; ~18x fewer per-tensor fixed costs.
                let quant = *quant_setting().lock().unwrap_or_else(|p| p.into_inner());
                match quant {
                    None => {
                        for i in 0..e {
                            let gui = gu.i(i)?;
                            experts.push(Mlp::from_weights(
                                gui.narrow(0, 0, inter)?.contiguous()?,
                                gui.narrow(0, inter, inter)?.contiguous()?,
                                dn.i(i)?.contiguous()?,
                                &format!("{kp}.{i}"),
                            )?);
                        }
                    }
                    Some(dtype) => {
                        use candle::quantized::ggml_file::qtensor_from_ggml;
                        let dev = vb.device().clone();
                        let cpu = Device::Cpu;
                        let d_gu = quant_dtype_for(h, dtype);
                        let d_dn = quant_dtype_for(inter, dtype);
                        anyhow_ok(h % d_gu.block_size() == 0, "gate/up rows not block-aligned")?;
                        anyhow_ok(inter % d_dn.block_size() == 0, "down rows not block-aligned")?;
                        // gate_up: [E, 2I, H] -> flat rows, one quantize.
                        let gu_flat = gu
                            .to_device(&cpu)?
                            .to_dtype(DType::F32)?
                            .reshape((e * 2 * inter, h))?;
                        let q_gu = QTensor::quantize(&gu_flat, d_gu)?;
                        let gu_bytes = q_gu.data()?;
                        let row_gu = h / d_gu.block_size() * d_gu.type_size();
                        // down: [E, H, I] -> flat rows, one quantize.
                        let dn_flat = dn
                            .to_device(&cpu)?
                            .to_dtype(DType::F32)?
                            .reshape((e * h, inter))?;
                        let q_dn = QTensor::quantize(&dn_flat, d_dn)?;
                        let dn_bytes = q_dn.data()?;
                        let row_dn = inter / d_dn.block_size() * d_dn.type_size();
                        for i in 0..e {
                            let g0 = i * 2 * inter * row_gu;
                            let gate = qtensor_from_ggml(
                                d_gu,
                                &gu_bytes[g0..g0 + inter * row_gu],
                                vec![inter, h],
                                &dev,
                            )?;
                            let u0 = g0 + inter * row_gu;
                            let up = qtensor_from_ggml(
                                d_gu,
                                &gu_bytes[u0..u0 + inter * row_gu],
                                vec![inter, h],
                                &dev,
                            )?;
                            let dn0 = i * h * row_dn;
                            let down = qtensor_from_ggml(
                                d_dn,
                                &dn_bytes[dn0..dn0 + h * row_dn],
                                vec![h, inter],
                                &dev,
                            )?;
                            let (gate, up, down) = (Arc::new(gate), Arc::new(up), Arc::new(down));
                            let kpref = format!("{kp}.{i}");
                            qcache::put(&format!("{kpref}.gate_proj"), &gate);
                            qcache::put(&format!("{kpref}.up_proj"), &up);
                            qcache::put(&format!("{kpref}.down_proj"), &down);
                            experts.push(Mlp {
                                gate_proj: Proj::Quant(QMatMul::QTensor(gate)),
                                up_proj: Proj::Quant(QMatMul::QTensor(up)),
                                down_proj: Proj::Quant(QMatMul::QTensor(down)),
                            });
                        }
                    }
                }
            } else {
            // Quantization of 3 x E slices is CPU-bound; fan it out across
            // threads (16 vCPU pod: ~10x). Slices move to CPU first so the
            // workers never touch the CUDA context concurrently.
            let mut slices = Vec::with_capacity(e);
            let cpu = Device::Cpu;
            for i in 0..e {
                let gui = gu.i(i)?;
                slices.push((
                    gui.narrow(0, 0, inter)?.contiguous()?.to_device(&cpu)?,
                    gui.narrow(0, inter, inter)?.contiguous()?.to_device(&cpu)?,
                    dn.i(i)?.contiguous()?.to_device(&cpu)?,
                ));
            }
            let dev_main = vb.device().clone();
            let n_workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .min(e.max(1));
            let results: Vec<Result<Mlp>> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let slices_ref = &slices;
                let dev_ref = &dev_main;
                let kp_ref: &str = &kp;
                for w in 0..n_workers {
                    let b = std::thread::Builder::new()
                        .name(format!("expert-quant-{w}"))
                        .stack_size(32 * 1024 * 1024);
                    handles.push(b.spawn_scoped(scope, move || {
                        let mut out = Vec::new();
                        let mut i = w;
                        while i < slices_ref.len() {
                            let (g, u, d) = &slices_ref[i];
                            out.push((
                                i,
                                Mlp::from_cpu_weights(g, u, d, dev_ref, &format!("{kp_ref}.{i}")),
                            ));
                            i += n_workers;
                        }
                        out
                    }).expect("spawn expert-quant worker"));
                }
                let mut all: Vec<(usize, Result<Mlp>)> = Vec::with_capacity(e);
                for h in handles {
                    all.extend(h.join().expect("expert quant worker panicked"));
                }
                all.sort_by_key(|(i, _)| *i);
                all.into_iter().map(|(_, r)| r).collect()
            });
            for r in results {
                experts.push(r?);
            }
            }
        } else {
            for i in 0..e {
                experts.push(Mlp::new(h, inter, vb.pp(format!("experts.{i}")))?);
            }
        }
        let (shared_expert, shared_expert_gate) =
            match cfg.shared_expert_intermediate_size {
                Some(si) if si > 0 => (
                    Some(Mlp::new(h, si, vb.pp("shared_expert"))?),
                    Some(linear_no_bias(h, 1, vb.pp("shared_expert_gate"))?),
                ),
                _ => (None, None),
            };
        Ok(Self {
            gate: linear_no_bias(h, e, vb.pp("gate"))?,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: cfg.num_experts_per_tok.unwrap_or(8),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Naive single-token routing (b=1, seq=1 in the scaffold path).
        let (b, s, h) = x.dims3()?;
        let flat = x.reshape((b * s, h))?;
        let logits = self.gate.forward(&flat)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs_v = probs.to_vec2::<f32>()?;
        let mut rows = Vec::with_capacity(b * s);
        for (t, row) in probs_v.iter().enumerate() {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_unstable_by(|&a, &c| row[c].total_cmp(&row[a]));
            idx.truncate(self.top_k);
            let denom: f32 = idx.iter().map(|&e| row[e]).sum();
            let xt = flat.narrow(0, t, 1)?;
            let mut acc: Option<Tensor> = None;
            for &e in &idx {
                let w = row[e] / denom;
                let y = (self.experts[e].forward(&xt)? * w as f64)?;
                acc = Some(match acc {
                    None => y,
                    Some(a) => (a + y)?,
                });
            }
            rows.push(acc.expect("top_k >= 1"));
        }
        let out = Tensor::cat(&rows, 0)?;
        let mut out = out.reshape((b, s, h))?;
        if let (Some(se), Some(sg)) = (&self.shared_expert, &self.shared_expert_gate) {
            let gate = candle_nn::ops::sigmoid(&sg.forward(x)?.to_dtype(DType::F32)?)?
                .to_dtype(x.dtype())?;
            let sh = se.forward(x)?.broadcast_mul(&gate)?;
            out = (out + sh)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
enum Ffn {
    Dense(Mlp),
    Moe(SparseMoe),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

// ── Decoder layer / model ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Mixer {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    mixer: Mixer,
    ffn: Ffn,
    input_layernorm: RmsNormPlus1,
    post_attention_layernorm: RmsNormPlus1,
}

impl DecoderLayer {
    fn new(cfg: &Qwen35TextConfig, layer: usize, vb: VarBuilder) -> Result<Self> {
        let mixer = if cfg.is_linear(layer) {
            Mixer::Linear(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
        } else {
            Mixer::Full(FullAttention::new(cfg, vb.pp("self_attn"))?)
        };
        let ffn = if cfg.num_experts.unwrap_or(0) > 0 {
            Ffn::Moe(SparseMoe::new(cfg, vb.pp("mlp"))?)
        } else {
            Ffn::Dense(Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size.expect("dense layer needs intermediate_size"),
                vb.pp("mlp"),
            )?)
        };
        Ok(Self {
            mixer,
            ffn,
            input_layernorm: RmsNormPlus1::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNormPlus1::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward_static(
        &mut self,
        x: &Tensor,
        rotary: &Rotary,
        pos: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let mixed = match &mut self.mixer {
            Mixer::Full(a) => a.forward_static(&normed, rotary, pos, mask)?,
            Mixer::Linear(g) => g.forward_static_step(&normed)?,
        };
        let x = (x + mixed)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let f = self.ffn.forward(&normed)?;
        x + f
    }

    fn forward_step(&mut self, x: &Tensor, rotary: &Rotary, pos: usize) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        if std::env::var("QWEN35_DEBUG_GDN").is_ok() {
            dbg_stats("  layer.input_ln_out", &normed)?;
        }
        let mixed = match &mut self.mixer {
            Mixer::Full(a) => a.forward(&normed, rotary, pos)?,
            Mixer::Linear(g) => g.forward_step(&normed)?,
        };
        if std::env::var("QWEN35_DEBUG_GDN").is_ok() {
            dbg_stats("  layer.mixer_out", &mixed)?;
        }
        let x = (x + mixed)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let f = self.ffn.forward(&normed)?;
        x + f
    }
}

fn dbg_stats(tag: &str, x: &Tensor) -> Result<()> {
    let v = x.flatten_all()?.to_dtype(DType::F32)?;
    let mean = v.mean_all()?.to_scalar::<f32>()?;
    let var = v.sqr()?.mean_all()?.to_scalar::<f32>()? - mean * mean;
    let f3 = v.narrow(0, 0, 3)?.to_vec1::<f32>()?;
    eprintln!(
        "{tag} mean={mean:.5} std={:.5} first3={:?}",
        var.max(0.0).sqrt(),
        f3.iter().map(|x| (x * 10000.0).round() / 10000.0).collect::<Vec<_>>()
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNormPlus1,
    lm_head: Proj,
    rotary: Rotary,
    device: Device,
    /// Absolute position of the next token (recurrent state cursor).
    pos: usize,
    static_ctx: Option<QwenStaticCtx>,
}

/// Device-resident state for the shape-static decode path.
#[derive(Debug, Clone)]
pub struct QwenStaticCtx {
    pub pos: Tensor,  // u32 [1]
    pub mask: Tensor, // f32 additive row [max_seq]
    pub max_seq: usize,
}

impl Model {
    pub fn new(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_quant(cfg, vb, None)
    }

    /// Like [`Self::new`], but with `quant` set every large projection is
    /// quantized to that GGML dtype at load (activations stay in the
    /// VarBuilder dtype); quantized bytes go through the shared GGUF disk
    /// cache so warm loads skip both the safetensors reads and quantization.
    pub fn new_with_quant(
        cfg: &Qwen35TextConfig,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        *quant_setting().lock().unwrap_or_else(|p| p.into_inner()) = quant;
        // Multimodal checkpoints prefix the text stack with `language_model`.
        let root = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model").pp("language_model")
        } else {
            vb.pp("model")
        };
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            root.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = root.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, vb_l.pp(i))?);
        }
        let norm = RmsNormPlus1::new(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?;
        let quant = *quant_setting().lock().unwrap_or_else(|p| p.into_inner());
        let lm_head = if cfg.tie_word_embeddings {
            match quant {
                None => Proj::Plain(Linear::new(embed_tokens.embeddings().clone(), None)),
                Some(dtype) => {
                    let emb = embed_tokens.embeddings();
                    let dtype = quant_dtype_for(cfg.hidden_size, dtype);
                    let key = "lm_head.tied";
                    let q = if let Some(q) = qcache::get(key, emb.device()) {
                        Arc::new(q)
                    } else {
                        let cpu = emb.to_device(&Device::Cpu)?;
                        let q = Arc::new(QTensor::quantize_onto(&cpu, dtype, emb.device())?);
                        qcache::put(key, &q);
                        q
                    };
                    Proj::Quant(QMatMul::QTensor(q))
                }
            }
        } else {
            qproj(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let rotary = Rotary::new(cfg, vb.device())?;
        if let Err(e) = qcache::flush() {
            eprintln!("[qwen3_5-qcache] flush failed: {e}");
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            device: vb.device().clone(),
            pos: 0,
            static_ctx: None,
        })
    }

    /// Enable the shape-static decode path (fixed [max_seq] KV budget on the
    /// full-attention layers; address-stable GDN state buffers).
    pub fn enable_static_decode(&mut self, max_seq: usize) -> Result<()> {
        let dtype = self.embed_tokens.embeddings().dtype();
        let dev = self.device.clone();
        if self.static_ctx.is_none() {
            self.static_ctx = Some(QwenStaticCtx {
                pos: Tensor::zeros(1, DType::U32, &dev)?,
                mask: Tensor::zeros(max_seq, DType::F32, &dev)?,
                max_seq,
            });
        }
        for layer in self.layers.iter_mut() {
            match &mut layer.mixer {
                Mixer::Full(a) => a.enable_static(max_seq, dtype, &dev)?,
                Mixer::Linear(g) => g.enable_static(&dev)?,
            }
        }
        Ok(())
    }

    pub fn static_enabled(&self) -> bool {
        self.static_ctx.is_some()
    }

    /// After dynamic prefill: migrate every layer's state into the
    /// address-stable buffers and set the device position. Returns the
    /// prefill length.
    pub fn migrate_prefill_to_static(&mut self) -> Result<usize> {
        let mut n = 0usize;
        for layer in self.layers.iter_mut() {
            match &mut layer.mixer {
                Mixer::Full(a) => {
                    let m = a.migrate_kv_to_static()?;
                    if m > 0 {
                        n = m;
                    }
                }
                Mixer::Linear(g) => g.migrate_state_to_static()?,
            }
        }
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        if n >= ctx.max_seq {
            candle::bail!("prefill length {n} exceeds static max_seq {}", ctx.max_seq);
        }
        // Trust the model cursor over the attention caches (they agree, but
        // the cursor also counts pure-GDN paths).
        let n = self.pos.max(n);
        candle_nn::fused::static_decode::write_u32(&ctx.pos, n as u32)?;
        Ok(n)
    }

    /// One graph-replayable decode step: [1,1] token ids -> [1, vocab] logits.
    pub fn forward_static(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?
            .clone();
        candle_nn::fused::static_decode::mask_from_pos(&ctx.mask, &ctx.pos, 0)?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let rotary = self.rotary.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward_static(&x, &rotary, &ctx.pos, &ctx.mask)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x.i((.., 0, ..))?.contiguous()?)?;
        candle_nn::fused::static_decode::incr_u32(&ctx.pos)?;
        self.pos += 1; // keep the host cursor coherent for mixed use
        Ok(logits)
    }

    /// Rewind static state between requests (never inside a graph).
    pub fn reset_static(&mut self) -> Result<()> {
        if let Some(ctx) = self.static_ctx.as_ref() {
            candle_nn::fused::static_decode::write_u32(&ctx.pos, 0)?;
        }
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            match &mut l.mixer {
                Mixer::Full(a) => a.clear_kv_cache(),
                Mixer::Linear(g) => g.clear_state(),
            }
        }
        self.pos = 0;
    }

    /// Executor-compatible forward. Processes the whole input chunk in one
    /// batched pass: projections/conv/norms/attention are computed for all
    /// positions at once and the GDN recurrence runs as a single on-chip
    /// scan kernel per layer. Returns lm_head logits of the final position:
    /// [1, 1, vocab].
    pub fn forward(&mut self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        if start_pos == 0 && self.pos != 0 {
            self.clear_kv_cache();
        }
        let (_b, seq) = input.dims2()?;
        let debug = std::env::var("QWEN35_DEBUG").is_ok();
        let mut x = self.embed_tokens.forward(input)?; // [1, seq, hidden]
        let pos = self.pos;
        if debug {
            dbg_stats("layer00", &x.narrow(1, seq - 1, 1)?)?;
        }
        for (li, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward_step(&x, &self.rotary, pos)?;
            if debug && li < 5 {
                dbg_stats(&format!("layer{:02}", li + 1), &x.narrow(1, seq - 1, 1)?)?;
            }
        }
        self.pos += seq;
        let x = x.narrow(1, seq - 1, 1)?;
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
