//! Unlimited-OCR text decoder (DeepSeek-OCR lineage).
//!
//! Plain llama-style MHA (no MLA: use_mla=false in the shipped config;
//! 10 heads == 10 kv heads, head_dim 128, hidden 1280, 12 layers, no
//! attention biases) + DeepSeek-V2-Lite-style MoE: 64 routed experts
//! (softmax gating, greedy top-6, scaling 1.0, no top-k renorm) + 2 shared
//! experts on layers 1..; layer 0 is a dense MLP. Standard rope
//! (theta 10000), rms_norm_eps 1e-6, vocab 129280.
//!
//! This file is the DYNAMIC path (prefill + cat-cache decode) used for
//! parity bring-up; the R-SWA ring-buffer static path lands on top of it
//! (see the session docs for the extracted algorithm).

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_eps() -> f64 {
    1e-6
}
fn default_max_pos() -> usize {
    32768
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct UnlimitedOcrTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    #[serde(default)]
    pub sliding_window_size: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
}

impl UnlimitedOcrTextConfig {
    pub fn head_dim(&self) -> usize {
        self.v_head_dim
    }
}

// ── Rotary (standard llama half-split) ──────────────────────────────────────

#[derive(Debug, Clone)]
struct Rotary {
    cos: Tensor, // [max_pos, hd/2] f32
    sin: Tensor,
}

impl Rotary {
    fn new(cfg: &UnlimitedOcrTextConfig, dev: &Device) -> Result<Self> {
        let hd = cfg.head_dim();
        let max_pos = cfg.max_position_embeddings;
        let inv: Vec<f32> = (0..hd / 2)
            .map(|i| (1.0 / cfg.rope_theta.powf(2.0 * i as f64 / hd as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, hd / 2), dev)?;
        let pos: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (max_pos, 1), dev)?;
        let freqs = pos.matmul(&inv)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, _hd) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq)?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, pos, seq)?.to_dtype(x.dtype())?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }
}

// ── Attention (llama-style, MHA) ────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(cfg: &UnlimitedOcrTextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: linear_no_bias(h, cfg.num_attention_heads * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.num_attention_heads * hd, h, vb.pp("o_proj"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            kv_cache: None,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn forward(&mut self, x: &Tensor, rotary: &Rotary, pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let hd = self.head_dim;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq, self.n_heads, hd))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, hd))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, hd))?
            .transpose(1, 2)?
            .contiguous()?;
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

        // n_heads == n_kv_heads (MHA): no repeat needed.
        let scale = 1.0 / (hd as f64).sqrt();
        let att = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = if seq > 1 {
            let total = k.dim(2)?;
            let offset = total - seq;
            let mask: Vec<f32> = (0..seq)
                .flat_map(|i| {
                    (0..total)
                        .map(move |j| if j <= i + offset { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, seq, total), att.device())?
                .to_dtype(att.dtype())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let out = att.matmul(&v)?;
        out.transpose(1, 2)?
            .reshape((b, seq, self.n_heads * hd))?
            .apply(&self.o_proj)
    }
}

// ── MLP / MoE ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(h: usize, inter: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(h, inter, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, inter, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(inter, h, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (self.gate_proj.forward(x)?.silu()? * self.up_proj.forward(x)?)?.apply(&self.down_proj)
    }
}

/// DeepSeek-V2-Lite MoE: softmax gate over 64 experts, greedy top-6,
/// scaling 1.0, no top-k renorm; plus shared experts always applied.
#[derive(Debug, Clone)]
struct MoE {
    gate_w: Tensor, // [n_experts, hidden] kept in F32 (reference gates in f32)
    experts: Vec<Mlp>,
    shared: Mlp,
    top_k: usize,
}

impl MoE {
    fn new(cfg: &UnlimitedOcrTextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        for i in 0..cfg.n_routed_experts {
            experts.push(Mlp::new(h, cfg.moe_intermediate_size, vb.pp("experts").pp(i))?);
        }
        Ok(Self {
            gate_w: vb
                .pp("gate")
                .get((cfg.n_routed_experts, h), "weight")?
                .to_dtype(DType::F32)?,
            experts,
            shared: Mlp::new(
                h,
                cfg.moe_intermediate_size * cfg.n_shared_experts,
                vb.pp("shared_experts"),
            )?,
            top_k: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, h) = x.dims3()?;
        let xf = x.reshape(((), h))?;
        let n_tok = b * seq;
        // Gating in f32 (mirrors the reference MoEGate: both operands cast).
        let logits = xf
            .to_dtype(DType::F32)?
            .matmul(&self.gate_w.t()?.contiguous()?)?; // [T, E]
        let scores = candle_nn::ops::softmax_last_dim(&logits)?;
        let scores_v: Vec<f32> = scores.flatten_all()?.to_vec1()?;
        let n_e = self.experts.len();
        // Greedy top-k per token (host-side; parity path).
        let mut routes: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_e];
        for t in 0..n_tok {
            let row = &scores_v[t * n_e..(t + 1) * n_e];
            let mut idx: Vec<usize> = (0..n_e).collect();
            // Descending by score; ties resolve to the LOWEST index, matching
            // torch.topk's first-occurrence semantics.
            idx.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap()
                    .then(a.cmp(&b))
            });
            for &e in idx.iter().take(self.top_k) {
                routes[e].push((t, row[e]));
            }
        }
        let mut out = Tensor::zeros((n_tok, h), x.dtype(), x.device())?;
        for (e, toks) in routes.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let ids: Vec<u32> = toks.iter().map(|(t, _)| *t as u32).collect();
            let ws: Vec<f32> = toks.iter().map(|(_, w)| *w).collect();
            let ids_t = Tensor::from_vec(ids, toks.len(), x.device())?;
            let sel = xf.index_select(&ids_t, 0)?;
            let y = self.experts[e].forward(&sel)?;
            let w = Tensor::from_vec(ws, (toks.len(), 1), x.device())?.to_dtype(x.dtype())?;
            let y = y.broadcast_mul(&w)?;
            out = out.index_add(&ids_t, &y, 0)?;
        }
        let out = (out + self.shared.forward(&xf)?)?;
        out.reshape((b, seq, h))
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Mlp),
    Moe(MoE),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FeedForward::Dense(m) => m.forward(x),
            FeedForward::Moe(m) => m.forward(x),
        }
    }
}

// ── Decoder layer / model ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    ffn: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &UnlimitedOcrTextConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let ffn = if layer_idx >= cfg.first_k_dense_replace {
            FeedForward::Moe(MoE::new(cfg, vb.pp("mlp"))?)
        } else {
            FeedForward::Dense(Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?)
        };
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            ffn,
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, x: &Tensor, rotary: &Rotary, pos: usize) -> Result<Tensor> {
        let residual = x;
        let h = self
            .self_attn
            .forward(&self.input_layernorm.forward(x)?, rotary, pos)?;
        let x = (residual + h)?;
        let h = self.ffn.forward(&self.post_attention_layernorm.forward(&x)?)?;
        x + h
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: Rotary,
    pub device: Device,
    pub dtype: DType,
}

impl TextModel {
    pub fn new(cfg: &UnlimitedOcrTextConfig, vb: VarBuilder) -> Result<Self> {
        let root = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, root.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, root.pp("layers").pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?,
            lm_head: linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?,
            rotary: Rotary::new(cfg, vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward over embeddings (vision embeds get merged upstream).
    pub fn forward_embeds(&mut self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq, _h) = xs.dims3()?;
        let mut x = xs.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x, &self.rotary, pos)?;
        }
        let x = self.norm.forward(&x)?;
        let last = x.i((.., seq - 1, ..))?.contiguous()?;
        self.lm_head.forward(&last)
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let xs = self.embed(input_ids)?;
        self.forward_embeds(&xs, pos)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.self_attn.clear_kv_cache();
        }
    }
}
