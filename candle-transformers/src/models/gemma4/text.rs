//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle::quantized::{GgmlDType, QMatMul, QTensor};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── Quantized-weight disk cache ─────────────────────────────────────────────
//
// Quantize-on-load of large checkpoints costs minutes (CPU quantize + full
// safetensors read). When `GEMMA4_QCACHE_FILE` is set, quantized tensors are
// served from / persisted to a standard GGUF file next to the model assets:
// the second load skips both the bf16 reads and the quantization entirely.
pub mod qcache {
    use candle::quantized::{gguf_file, QTensor};
    use candle::{Device, Result};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    struct State {
        path: std::path::PathBuf,
        reader: Option<(gguf_file::Content, std::fs::File)>,
        pending: Vec<(String, Arc<QTensor>)>,
    }

    static STATE: OnceLock<Mutex<Option<State>>> = OnceLock::new();

    fn state() -> &'static Mutex<Option<State>> {
        STATE.get_or_init(|| {
            let st = std::env::var("GEMMA4_QCACHE_FILE").ok().map(|p| {
                let path = std::path::PathBuf::from(p);
                let reader = std::fs::File::open(&path).ok().and_then(|mut f| {
                    match gguf_file::Content::read(&mut f) {
                        Ok(c) => {
                            eprintln!(
                                "[gemma4-qcache] using cache {} ({} tensors)",
                                path.display(),
                                c.tensor_infos.len()
                            );
                            Some((c, f))
                        }
                        Err(e) => {
                            eprintln!("[gemma4-qcache] unreadable cache ({e}), rebuilding");
                            None
                        }
                    }
                });
                State { path, reader, pending: Vec::new() }
            });
            Mutex::new(st)
        })
    }

    /// Fetch a cached quantized tensor straight onto `device`.
    pub fn get(name: &str, device: &Device) -> Option<QTensor> {
        let mut guard = state().lock().unwrap_or_else(|p| p.into_inner());
        let st = guard.as_mut()?;
        let (content, file) = st.reader.as_mut()?;
        if !content.tensor_infos.contains_key(name) {
            return None;
        }
        content.tensor(file, name, device).ok()
    }

    /// Record a freshly quantized tensor for persistence (no-op when the
    /// cache file already exists or caching is disabled).
    pub fn put(name: &str, qt: &Arc<QTensor>) {
        let mut guard = state().lock().unwrap_or_else(|p| p.into_inner());
        if let Some(st) = guard.as_mut() {
            if st.reader.is_none() {
                st.pending.push((name.to_string(), qt.clone()));
            }
        }
    }

    /// Write all pending tensors as a GGUF file (called once after load).
    pub fn flush() -> Result<()> {
        let mut guard = state().lock().unwrap_or_else(|p| p.into_inner());
        let st = match guard.as_mut() {
            Some(st) if st.reader.is_none() && !st.pending.is_empty() => st,
            _ => return Ok(()),
        };
        let t0 = std::time::Instant::now();
        let tmp = st.path.with_extension("tmp");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            let tensors: Vec<(&str, &QTensor)> = st
                .pending
                .iter()
                .map(|(n, t)| (n.as_str(), t.as_ref()))
                .collect();
            gguf_file::write(&mut f, &[], &tensors)?;
        }
        std::fs::rename(&tmp, &st.path)?;
        eprintln!(
            "[gemma4-qcache] wrote {} tensors to {} in {}ms",
            st.pending.len(),
            st.path.display(),
            t0.elapsed().as_millis()
        );
        st.pending.clear();
        let _ = HashMap::<(), ()>::new();
        Ok(())
    }
}

// ── Projection (dense or quantized-in-memory) ───────────────────────────────

/// Linear projection that can be quantized at load time: the checkpoint
/// weight is converted to the given GGML dtype and matmuls run through the
/// quantized kernels while activations stay in the model dtype. This trades a
/// one-time load cost for ~4x weight memory and integer-dot decode kernels —
/// no second (GGUF) checkpoint file is needed.
#[derive(Debug, Clone)]
enum Proj {
    Plain(Linear),
    Quant {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Proj {
    fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Self::Plain(linear_bias(in_dim, out_dim, bias, vb)?)),
            Some(dtype) => {
                // K-quants require the reduction dim % 256 == 0.
                let dtype = moe_quant_dtype_for(in_dim, dtype);
                if std::env::var("GEMMA4_QDEBUG").is_ok() {
                    eprintln!("[qdebug] Proj {} in={} out={} dtype={:?}", vb.prefix(), in_dim, out_dim, dtype);
                }
                let dev = vb.device().clone();
                let cache_key = vb.prefix();
                let weight = if let Some(q) = qcache::get(&cache_key, &dev) {
                    QMatMul::QTensor(Arc::new(q))
                } else {
                    let weight = vb.get((out_dim, in_dim), "weight")?;
                    // Quantize via the CPU: avoids an f32 copy of the full
                    // weight on the GPU (the tied lm_head alone is ~3 GB f32).
                    let weight = if dev.is_cuda() {
                        let cpu = weight.to_device(&Device::Cpu)?;
                        QTensor::quantize_onto(&cpu, dtype, &dev)?
                    } else {
                        QTensor::quantize(&weight.to_dtype(DType::F32)?, dtype)?
                    };
                    let weight = Arc::new(weight);
                    qcache::put(&cache_key, &weight);
                    QMatMul::QTensor(weight)
                };
                let bias = if bias {
                    Some(vb.get(out_dim, "bias")?.to_dtype(DType::F32)?)
                } else {
                    None
                };
                Ok(Self::Quant { weight, bias })
            }
        }
    }
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(xs),
            Self::Quant { weight, bias } => {
                let xs = weight.forward(xs)?;
                match bias {
                    None => Ok(xs),
                    Some(b) => xs.broadcast_add(b),
                }
            }
        }
    }
}

// ── RmsNorm (Gemma-style with +1 offset) ────────────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        // Unlike gemma1/2/3, gemma4 checkpoints store the norm scale directly
        // (weights initialized to ones) — no (1 + weight) offset.
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let gate_proj = Proj::new(hidden_size, intermediate_size, bias, vb.pp("gate_proj"), quant)?;
        let up_proj = Proj::new(hidden_size, intermediate_size, bias, vb.pp("up_proj"), quant)?;
        let down_proj = Proj::new(intermediate_size, hidden_size, bias, vb.pp("down_proj"), quant)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ── MoE (router + packed experts) ───────────────────────────────────────────

/// Token router: weightless RMS norm → learned scale (× hidden^-0.5) → linear
/// to expert logits → f32 softmax → top-k (renormalized, then multiplied by a
/// learned per-expert scale). Routing decisions are computed on the host —
/// [tokens, num_experts] is tiny and the expert dispatch needs host-side
/// grouping anyway.
#[derive(Debug, Clone)]
struct Router {
    proj: Proj,
    scale: Tensor,
    per_expert_scale: Vec<f32>,
    /// Same values kept on-device for the sync-free fused routing path.
    per_expert_scale_t: Tensor,
    top_k: usize,
    scalar_root: f64,
    eps: f64,
}

/// Per-token routing decision: (expert index, combine weight).
type RoutingPlan = Vec<Vec<(usize, f32)>>;

impl Router {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder, quant: Option<GgmlDType>) -> Result<Self> {
        let proj = Proj::new(cfg.hidden_size, cfg.num_experts, false, vb.pp("proj"), quant)?;
        let scale = vb.get(cfg.hidden_size, "scale")?;
        let per_expert_scale_t = vb
            .get(cfg.num_experts, "per_expert_scale")?
            .to_dtype(DType::F32)?;
        let per_expert_scale = per_expert_scale_t.to_vec1::<f32>()?;
        Ok(Self {
            proj,
            scale,
            per_expert_scale,
            per_expert_scale_t,
            top_k: cfg.top_k_experts,
            scalar_root: (cfg.hidden_size as f64).sqrt().recip(),
            eps: cfg.rms_norm_eps,
        })
    }

    /// `xs_flat`: [tokens, hidden] (the pre-feedforward residual stream).
    fn route(&self, xs_flat: &Tensor) -> Result<RoutingPlan> {
        let xs = v_norm(xs_flat, self.eps)?; // weightless RMS norm
        let xs = xs.broadcast_mul(&self.scale)?;
        let xs = (xs * self.scalar_root)?;
        let logits = self.proj.forward(&xs)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs = probs.to_vec2::<f32>()?;

        let mut plan = Vec::with_capacity(probs.len());
        for row in probs {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            idx.truncate(self.top_k);
            let sum: f32 = idx.iter().map(|&e| row[e]).sum();
            plan.push(
                idx.into_iter()
                    .map(|e| (e, row[e] / sum * self.per_expert_scale[e]))
                    .collect(),
            );
        }
        Ok(plan)
    }

    /// Sync-free routing for the fused CUDA expert path: everything stays on
    /// the device. Returns (`ids` [tokens, top_k] u32, `weights`
    /// [tokens, top_k] f32), where weights are renormalized over the top-k
    /// and multiplied by per_expert_scale — identical semantics to
    /// `route()`.
    fn route_gpu(&self, xs_flat: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs = v_norm(xs_flat, self.eps)?;
        let xs = xs.broadcast_mul(&self.scale)?;
        let xs = (xs * self.scalar_root)?;
        let logits = self.proj.forward(&xs)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?; // [T, E]
        let sorted = probs.arg_sort_last_dim(false)?; // descending, u32
        let ids = sorted.narrow(1, 0, self.top_k)?.contiguous()?; // [T, k]
        let topw = probs.gather(&ids, 1)?; // [T, k]
        let denom = topw.sum_keepdim(1)?;
        let w = topw.broadcast_div(&denom)?;
        let t = ids.dim(0)?;
        let pes = self
            .per_expert_scale_t
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((t, self.top_k))?;
        let w = (w * pes)?;
        Ok((ids, w))
    }
}

/// Expert FFN weights, stored packed as in the checkpoint:
/// `gate_up_proj` [E, 2*moe_intermediate, hidden], `down_proj`
/// [E, hidden, moe_intermediate]. With `quant` each expert slice is
/// quantized separately at load time.
#[derive(Debug, Clone)]
enum ExpertWeights {
    /// Dense experts, stored PRE-TRANSPOSED ([E, hidden, 2*inter] /
    /// [E, inter, hidden]) so per-token expert matmuls need no copy.
    Plain { gate_up_t: Tensor, down_t: Tensor },
    /// Legacy per-expert quantized matmuls (CPU / non-CUDA fallback).
    Quant {
        gate_up: Vec<QMatMul>,
        down: Vec<QMatMul>,
    },
    /// All experts stacked in single QTensors ([E, n, k]); dispatched with
    /// one fused indexed-MoE CUDA kernel per projection.
    QuantFused {
        gate_up: Arc<QTensor>,
        down: Arc<QTensor>,
    },
}

/// K-quants need the reduction dim to be a multiple of 256; fall back to a
/// 32-block dtype supported by the fused indexed-MoE kernel otherwise.
fn moe_quant_dtype_for(k: usize, requested: GgmlDType) -> GgmlDType {
    let needs_256 = matches!(
        requested,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    );
    if needs_256 && k % 256 != 0 {
        GgmlDType::Q8_0
    } else {
        requested
    }
}

#[derive(Debug, Clone)]
struct Experts {
    weights: ExpertWeights,
    act_fn: Activation,
}

impl Experts {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder, quant: Option<GgmlDType>) -> Result<Self> {
        let e = cfg.num_experts;
        let inter = cfg.moe_intermediate_size;
        let h = cfg.hidden_size;
        // Raw parameters — no ".weight" suffix in the checkpoint.
        let weights = match quant {
            None => {
                // Pre-transpose once at load: expert_forward then reads a
                // contiguous slice instead of materializing a transposed
                // copy of the expert weights on every token.
                let gate_up = vb.get((e, 2 * inter, h), "gate_up_proj")?;
                let down = vb.get((e, h, inter), "down_proj")?;
                let gate_up_t = gate_up.transpose(1, 2)?.contiguous()?;
                let down_t = down.transpose(1, 2)?.contiguous()?;
                ExpertWeights::Plain { gate_up_t, down_t }
            }
            Some(dtype) if vb.device().is_cuda() => {
                // Quantize on the CPU and upload only the quantized bytes:
                // avoids multi-GB f32 transients on the GPU during load.
                // One projection at a time keeps the peak small. Both stacks
                // go through the disk cache — on a warm cache the safetensors
                // for the experts (the bulk of the model) are never read.
                let dev = vb.device().clone();
                let gu_dtype = moe_quant_dtype_for(h, dtype);
                let dn_dtype = moe_quant_dtype_for(inter, dtype);
                let gu_key = format!("{}.gate_up_proj", vb.prefix());
                let dn_key = format!("{}.down_proj", vb.prefix());
                let gate_up = if let Some(q) = qcache::get(&gu_key, &dev) {
                    Arc::new(q)
                } else {
                    let t = vb
                        .get((e, 2 * inter, h), "gate_up_proj")?
                        .to_device(&Device::Cpu)?;
                    let q = Arc::new(QTensor::quantize_onto(&t, gu_dtype, &dev)?);
                    qcache::put(&gu_key, &q);
                    q
                };
                let down = if let Some(q) = qcache::get(&dn_key, &dev) {
                    Arc::new(q)
                } else {
                    let t = vb.get((e, h, inter), "down_proj")?.to_device(&Device::Cpu)?;
                    let q = Arc::new(QTensor::quantize_onto(&t, dn_dtype, &dev)?);
                    qcache::put(&dn_key, &q);
                    q
                };
                ExpertWeights::QuantFused { gate_up, down }
            }
            Some(dtype) => {
                let gate_up = vb.get((e, 2 * inter, h), "gate_up_proj")?;
                let down = vb.get((e, h, inter), "down_proj")?;
                let mut gu = Vec::with_capacity(e);
                let mut dn = Vec::with_capacity(e);
                for i in 0..e {
                    let g = gate_up.i(i)?.to_dtype(DType::F32)?.contiguous()?;
                    gu.push(QMatMul::from_qtensor(QTensor::quantize(&g, dtype)?)?);
                    let d = down.i(i)?.to_dtype(DType::F32)?.contiguous()?;
                    dn.push(QMatMul::from_qtensor(QTensor::quantize(&d, dtype)?)?);
                }
                ExpertWeights::Quant { gate_up: gu, down: dn }
            }
        };
        Ok(Self {
            weights,
            act_fn: cfg.hidden_activation,
        })
    }

    fn expert_forward(&self, expert: usize, xs: &Tensor) -> Result<Tensor> {
        let gate_up = match &self.weights {
            ExpertWeights::Plain { gate_up_t, .. } => xs.matmul(&gate_up_t.i(expert)?)?,
            ExpertWeights::Quant { gate_up, .. } => gate_up[expert].forward(xs)?,
            ExpertWeights::QuantFused { .. } => unreachable!("fused path handled in forward"),
        };
        let chunks = gate_up.chunk(2, D::Minus1)?;
        let hidden = (chunks[0].apply(&self.act_fn)? * &chunks[1])?;
        match &self.weights {
            ExpertWeights::Plain { down_t, .. } => hidden.matmul(&down_t.i(expert)?),
            ExpertWeights::Quant { down, .. } => down[expert].forward(&hidden),
            ExpertWeights::QuantFused { .. } => unreachable!("fused path handled in forward"),
        }
    }

    /// Fused CUDA path: one indexed-MoE kernel per projection, all selected
    /// experts of all tokens in a single launch. `ids` [tokens, top_k] u32
    /// and `weights` [tokens, top_k] f32 live on the device (route_gpu) —
    /// the whole path is free of host round-trips.
    fn forward_fused(
        &self,
        xs_flat: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        gate_up: &QTensor,
        down: &QTensor,
    ) -> Result<Tensor> {
        let out_dtype = xs_flat.dtype();
        let (tokens, hidden) = xs_flat.dims2()?;

        let x32 = xs_flat.to_dtype(DType::F32)?.reshape((tokens, 1, hidden))?;
        let gu = gate_up.indexed_moe_forward(&x32, ids)?; // [tokens, topk, 2*inter]
        // Single-launch act(gate) * up (item-3 fusion); falls back to the
        // chunk/act/mul composition off-CUDA.
        let act = match self.act_fn {
            Activation::Silu => candle_nn::fused::SwigluAct::Silu,
            _ => candle_nn::fused::SwigluAct::GeluTanh,
        };
        let hidden_act = candle_nn::fused::fused_swiglu(&gu, act)?;
        let dn = down.indexed_moe_forward(&hidden_act, ids)?; // [tokens, topk, hidden]

        let w_t = weights.unsqueeze(2)?; // [tokens, topk, 1]
        let out = dn.broadcast_mul(&w_t)?.sum(1)?; // [tokens, hidden]
        out.to_dtype(out_dtype)
    }

    /// `xs_flat`: [tokens, hidden] (already normalized). Groups tokens by
    /// expert, runs each hit expert once, and combines weighted outputs.
    fn forward(&self, xs_flat: &Tensor, plan: &RoutingPlan) -> Result<Tensor> {
        let device = xs_flat.device();
        let dtype = xs_flat.dtype();

        // expert -> (token indices, combine weights)
        let mut by_expert: std::collections::HashMap<usize, (Vec<u32>, Vec<f32>)> =
            std::collections::HashMap::new();
        for (tok, choices) in plan.iter().enumerate() {
            for &(e, w) in choices {
                let entry = by_expert.entry(e).or_default();
                entry.0.push(tok as u32);
                entry.1.push(w);
            }
        }

        let mut out = Tensor::zeros(xs_flat.shape(), dtype, device)?;
        let mut experts: Vec<_> = by_expert.into_iter().collect();
        experts.sort_unstable_by_key(|(e, _)| *e);
        for (expert, (toks, ws)) in experts {
            let idx = Tensor::from_vec(toks, (ws.len(),), device)?;
            let xs_e = xs_flat.index_select(&idx, 0)?;
            let ys = self.expert_forward(expert, &xs_e)?;
            let w = Tensor::from_vec(ws, (ys.dim(0)?, 1), device)?.to_dtype(dtype)?;
            let ys = ys.broadcast_mul(&w)?;
            out = out.index_add(&idx, &ys, 0)?;
        }
        Ok(out)
    }
}

// ── Flash attention ─────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

// ── Attention ───────────────────────────────────────────────────────────────

/// Per-forward-pass KV states shared from the last non-shared layer of each
/// layer type to the trailing `num_kv_shared_layers` layers (which have no
/// K/V projections of their own).
#[derive(Default)]
pub(crate) struct SharedKvStates {
    full: Option<(Tensor, Tensor)>,
    sliding: Option<(Tensor, Tensor)>,
}

/// Shared state for the CUDA-graph-replayable static decode path
/// (`GEMMA4_STATIC_DECODE=<max_seq>`): the decode position lives in a device
/// u32 scalar; masks over the full preallocated KV extent are rewritten from
/// it each token; per-layer KV buffers are fixed `[kv_heads, max, head_dim]`
/// allocations written in place. Everything a captured graph touches is
/// address- and shape-static, so one captured decode step replays for the
/// whole generation.
#[derive(Debug, Clone)]
pub struct StaticCtx {
    pub pos: Tensor,          // u32 [1]
    pub mask_global: Tensor,  // f32 [max_seq]
    pub mask_sliding: Tensor, // f32 [max_seq]
    pub max_seq: usize,
    pub sliding_window: usize,
    /// A2 chunked-prefill-graph state (allocated by enable_chunk_graph).
    pub chunk: Option<ChunkCtx>,
}

/// Device-resident chunked-prefill state: everything a captured prefill
/// chunk step reads or advances.
#[derive(Debug, Clone)]
pub struct ChunkCtx {
    pub ids: Tensor,          // u32 [1, C] — fed between replays (htod)
    pub rope_idx: Tensor,     // u32 [C] — iota(pos) refreshed in-graph
    pub mask_global: Tensor,  // f32 [C, max_seq]
    pub mask_sliding: Tensor, // f32 [C, max_seq]
    pub chunk: usize,
}

impl StaticCtx {
    pub fn new(max_seq: usize, sliding_window: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            pos: Tensor::zeros(1, DType::U32, dev)?,
            mask_global: Tensor::zeros(max_seq, DType::F32, dev)?,
            mask_sliding: Tensor::zeros(max_seq, DType::F32, dev)?,
            max_seq,
            sliding_window,
            chunk: None,
        })
    }

    /// Rewrite both mask rows for the current position (graph-replayable).
    pub fn refresh_masks(&self) -> Result<()> {
        candle_nn::fused::static_decode::mask_from_pos(&self.mask_global, &self.pos, 0)?;
        candle_nn::fused::static_decode::mask_from_pos(
            &self.mask_sliding,
            &self.pos,
            self.sliding_window,
        )
    }

    pub fn reset(&self) -> Result<()> {
        candle_nn::fused::static_decode::write_u32(&self.pos, 0)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Proj,
    // K/V projections and k_norm are absent on KV-shared layers.
    k_proj: Option<Proj>,
    v_proj: Option<Proj>,
    o_proj: Proj,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    is_kv_shared: bool,
    store_full_length_kv: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
    /// Preallocated `[kv_heads, max_seq, head_dim]` K/V for the static
    /// decode path (lazily allocated on first static forward).
    static_kv: Option<(Tensor, Tensor)>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;

        // Trailing `num_kv_shared_layers` layers reuse the KV states of the
        // last non-shared layer of the same layer type; they carry no K/V
        // projections (any such tensors in the checkpoint are ignored, as in
        // the reference implementation).
        let first_kv_shared_layer_idx = cfg
            .num_hidden_layers
            .saturating_sub(cfg.num_kv_shared_layers);
        let is_kv_shared = cfg.num_kv_shared_layers > 0 && layer_idx >= first_kv_shared_layer_idx;
        let layer_type = &cfg.layer_types[layer_idx];
        let store_full_length_kv = !is_kv_shared
            && cfg.layer_types[..first_kv_shared_layer_idx]
                .iter()
                .rposition(|t| t == layer_type)
                == Some(layer_idx);

        // K==V ("alternative attention", e.g. gemma-4-26B-A4B-it): global
        // layers have no v_proj tensor; V reuses the raw K projection (the
        // norms diverge afterwards: K gets k_norm+RoPE, V only the unscaled
        // v_norm).
        let k_eq_v = cfg.attention_k_eq_v && !is_sliding;
        let q_proj = Proj::new(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"), quant)?;
        let (k_proj, v_proj, k_norm) = if is_kv_shared {
            (None, None, None)
        } else {
            let v_proj = if k_eq_v {
                None
            } else {
                Some(Proj::new(
                    hidden_sz,
                    num_kv_heads * head_dim,
                    bias,
                    vb.pp("v_proj"),
                    quant,
                )?)
            };
            (
                Some(Proj::new(
                    hidden_sz,
                    num_kv_heads * head_dim,
                    bias,
                    vb.pp("k_proj"),
                    quant,
                )?),
                v_proj,
                Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?),
            )
        };
        let o_proj = Proj::new(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"), quant)?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        let kv_cache = if is_sliding {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.effective_sliding_window(),
            ))
        } else {
            // Capacity note: candle's KvCache allocates the FULL capacity on
            // first append. max_position_embeddings is 128k+ for these
            // checkpoints — with many full-attention layers that is
            // gigabytes of dead weight (gemma-4-31b: 1.07GB x 10 layers,
            // which OOM'd a 32GB card). Cap to a serving context, tunable
            // via GEMMA4_MAX_CONTEXT.
            let max_ctx = std::env::var("GEMMA4_MAX_CONTEXT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8192)
                .min(cfg.max_position_embeddings);
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(2, max_ctx))
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            is_kv_shared,
            store_full_length_kv,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn,
            static_kv: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        let (q, k, v) = if self.is_kv_shared {
            // Reuse the KV states stored by the last non-shared layer of the
            // same layer type; only Q is computed here.
            let (q, _) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &q, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &q, seqlen_offset)?
            };
            let slot = if self.is_sliding {
                shared_kv.sliding.as_ref()
            } else {
                shared_kv.full.as_ref()
            };
            let (k, v) = slot.ok_or_else(|| {
                candle::Error::Msg("kv-shared layer ran before any storing layer".to_string())
            })?;
            (q, k.clone(), v.clone())
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has k_proj");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has k_norm");

            let k_raw = k_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            // K==V layers reuse the raw (pre-norm) K projection as V.
            let v_raw = match self.v_proj.as_ref() {
                Some(v_proj) => v_proj
                    .forward(xs)?
                    .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                None => k_raw.clone(),
            };
            let k = k_norm.forward(&k_raw)?;
            // V norm (RMS without learned weight)
            let v = v_norm(&v_raw, self.rms_norm_eps)?;

            // Apply RoPE
            let (q, k) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            };

            let (k, v) = match &mut self.kv_cache {
                KvCache::Normal(cache) => cache.append(&k, &v)?,
                KvCache::Rotating(cache) => cache.append(&k, &v)?,
            };
            if self.store_full_length_kv {
                let slot = if self.is_sliding {
                    &mut shared_kv.sliding
                } else {
                    &mut shared_kv.full
                };
                *slot = Some((k.clone(), v.clone()));
            }
            (q, k, v)
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // The reference implementation uses softmax scale 1.0 for gemma4 (the
        // learned q_norm absorbs the usual 1/sqrt(head_dim)).
        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1.0, mask.is_some())?.transpose(1, 2)?
        } else {
            let attn_weights = q.contiguous()?.matmul(&k.transpose(2, 3)?)?;

            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    /// Chunked prefill into the static KV buffers: processes [1, T, hidden]
    /// at once (batched projections + rope with host offset), writes K/V
    /// rows [pos, pos+T) via kv_write_chunk, and attends over the buffer
    /// prefix with the same mask builder as the dynamic path. EAGER ONLY —
    /// host `pos` makes this non-replayable; decode uses forward_static.
    fn forward_static_chunk(
        &mut self,
        xs: &Tensor, // [1, T, hidden]
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
        pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, t_len, _) = xs.dims3()?;
        debug_assert_eq!(b_sz, 1);
        let klen = pos + t_len;

        let q = self.q_proj.forward(xs)?;
        let q = q
            .reshape((1, t_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let (kbuf, vbuf, q) = if self.is_kv_shared {
            // Rope Q only (the k output of the helper is discarded).
            let (q, _k) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &q.clone(), pos)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &q.clone(), pos)?
            };
            let slot = if self.is_sliding {
                shared_kv.sliding.as_ref()
            } else {
                shared_kv.full.as_ref()
            };
            let (k, v) = slot.ok_or_else(|| {
                candle::Error::Msg("kv-shared layer ran before any storing layer".to_string())
            })?;
            (k.clone(), v.clone(), q)
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has k_proj");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has k_norm");
            let k_raw = k_proj
                .forward(xs)?
                .reshape((1, t_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v_raw = match self.v_proj.as_ref() {
                Some(v_proj) => v_proj
                    .forward(xs)?
                    .reshape((1, t_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                None => k_raw.clone(),
            };
            let k = k_norm.forward(&k_raw)?;
            let v = v_norm(&v_raw, self.rms_norm_eps)?;
            let (q, k) = if self.is_sliding {
                self.rotary_emb_local.apply_rotary_emb_qkv(&q, &k, pos)?
            } else {
                self.rotary_emb_global.apply_rotary_emb_qkv(&q, &k, pos)?
            };

            if self.static_kv.is_none() {
                let dev = xs.device();
                let shape = (self.num_kv_heads, ctx.max_seq, self.head_dim);
                self.static_kv = Some((
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                ));
            }
            let (kbuf, vbuf) = self.static_kv.as_ref().unwrap();
            let kc = k
                .reshape((self.num_kv_heads, t_len, self.head_dim))?
                .contiguous()?;
            let vc = v
                .reshape((self.num_kv_heads, t_len, self.head_dim))?
                .contiguous()?;
            candle_nn::fused::static_decode::kv_write_chunk(kbuf, &kc, pos)?;
            candle_nn::fused::static_decode::kv_write_chunk(vbuf, &vc, pos)?;
            if self.store_full_length_kv {
                let slot = if self.is_sliding {
                    &mut shared_kv.sliding
                } else {
                    &mut shared_kv.full
                };
                *slot = Some((kbuf.clone(), vbuf.clone()));
            }
            (kbuf.clone(), vbuf.clone(), q)
        };

        // Attend over the written prefix with the dynamic-path mask builder.
        let kpre = kbuf.narrow(1, 0, klen)?.unsqueeze(0)?; // [1, kvh, klen, hd]
        let vpre = vbuf.narrow(1, 0, klen)?.unsqueeze(0)?;
        let kpre = crate::utils::repeat_kv(kpre, self.num_kv_groups)?.contiguous()?;
        let vpre = crate::utils::repeat_kv(vpre, self.num_kv_groups)?.contiguous()?;
        let window = if self.is_sliding {
            Some(ctx.sliding_window)
        } else {
            None
        };
        let mask =
            prepare_decoder_attention_mask(1, t_len, pos, window, xs.dtype(), xs.device())?;
        // Softmax scale 1.0 (q_norm absorbs it) — mirrors the dynamic path.
        let attn = q.contiguous()?.matmul(&kpre.transpose(2, 3)?)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&vpre)?;
        out.transpose(1, 2)?
            .reshape((1, t_len, ()))?
            .apply(&self.o_proj)
    }

    /// A2: chunk attention with EVERY positional input device-resident
    /// (rope indices, masks, kv offsets from ChunkCtx/StaticCtx) and all
    /// shapes fixed at the chunk size => replayable inside a captured graph.
    fn forward_static_chunk_dev(
        &mut self,
        xs: &Tensor, // [1, C, hidden]
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let (b_sz, t_len, _) = xs.dims3()?;
        debug_assert_eq!(b_sz, 1);
        let cctx = ctx
            .chunk
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("chunk graph not enabled".into()))?;

        let rope_gather = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Result<Tensor> {
            let cos_p = cos.index_select(&cctx.rope_idx, 0)?; // [C, half]
            let sin_p = sin.index_select(&cctx.rope_idx, 0)?;
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos_p, &sin_p)
        };
        let (cos_t, sin_t) = if self.is_sliding {
            (&self.rotary_emb_local.cos, &self.rotary_emb_local.sin)
        } else {
            (&self.rotary_emb_global.cos, &self.rotary_emb_global.sin)
        };

        let q = self.q_proj.forward(xs)?;
        let q = q
            .reshape((1, t_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;

        let (kbuf, vbuf, q) = if self.is_kv_shared {
            let q = rope_gather(&q, cos_t, sin_t)?;
            let slot = if self.is_sliding {
                shared_kv.sliding.as_ref()
            } else {
                shared_kv.full.as_ref()
            };
            let (k, v) = slot.ok_or_else(|| {
                candle::Error::Msg("kv-shared layer ran before any storing layer".to_string())
            })?;
            (k.clone(), v.clone(), q)
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has k_proj");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has k_norm");
            let k_raw = k_proj
                .forward(xs)?
                .reshape((1, t_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v_raw = match self.v_proj.as_ref() {
                Some(v_proj) => v_proj
                    .forward(xs)?
                    .reshape((1, t_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                None => k_raw.clone(),
            };
            let k = k_norm.forward(&k_raw)?;
            let v = v_norm(&v_raw, self.rms_norm_eps)?;
            let q = rope_gather(&q, cos_t, sin_t)?;
            let k = rope_gather(&k, cos_t, sin_t)?;

            if self.static_kv.is_none() {
                let dev = xs.device();
                let shape = (self.num_kv_heads, ctx.max_seq, self.head_dim);
                self.static_kv = Some((
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                ));
            }
            let (kbuf, vbuf) = self.static_kv.as_ref().unwrap();
            let kc = k
                .reshape((self.num_kv_heads, t_len, self.head_dim))?
                .contiguous()?;
            let vc = v
                .reshape((self.num_kv_heads, t_len, self.head_dim))?
                .contiguous()?;
            candle_nn::fused::static_decode::kv_write_chunk_at(kbuf, &kc, &ctx.pos)?;
            candle_nn::fused::static_decode::kv_write_chunk_at(vbuf, &vc, &ctx.pos)?;
            if self.store_full_length_kv {
                let slot = if self.is_sliding {
                    &mut shared_kv.sliding
                } else {
                    &mut shared_kv.full
                };
                *slot = Some((kbuf.clone(), vbuf.clone()));
            }
            (kbuf.clone(), vbuf.clone(), q)
        };

        // Fixed-shape attention over the FULL KV extent with the 2D device
        // chunk mask (rows = chunk, cols = capacity).
        let kfull = kbuf.unsqueeze(0)?; // [1, kvh, cap, hd]
        let vfull = vbuf.unsqueeze(0)?;
        let kfull = crate::utils::repeat_kv(kfull, self.num_kv_groups)?.contiguous()?;
        let vfull = crate::utils::repeat_kv(vfull, self.num_kv_groups)?.contiguous()?;
        let mask = if self.is_sliding {
            &cctx.mask_sliding
        } else {
            &cctx.mask_global
        };
        let attn = q.contiguous()?.matmul(&kfull.transpose(2, 3)?)?; // [1,h,C,cap]
        let attn = attn.to_dtype(DType::F32)?;
        let attn = attn.broadcast_add(&mask.reshape((1, 1, t_len, ctx.max_seq))?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.to_dtype(q.dtype())?;
        let out = attn.matmul(&vfull)?;
        out.transpose(1, 2)?
            .reshape((1, t_len, ()))?
            .apply(&self.o_proj)
    }

    /// One-token attention over the full preallocated KV extent, indexed by
    /// the device position scalar. Shape/address-static: replayable inside a
    /// captured CUDA graph.
    fn forward_static(
        &mut self,
        xs: &Tensor, // [1, 1, hidden]
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        debug_assert_eq!((b_sz, q_len), (1, 1));

        let q = self.q_proj.forward(xs)?;
        let q = q
            .reshape((1, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;

        // RoPE at the device position: gather one row of the tables.
        let rope_at_pos = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Result<Tensor> {
            let cos_p = cos.index_select(&ctx.pos, 0)?; // [1, half]
            let sin_p = sin.index_select(&ctx.pos, 0)?;
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos_p, &sin_p)
        };
        let (cos_t, sin_t) = if self.is_sliding {
            (&self.rotary_emb_local.cos, &self.rotary_emb_local.sin)
        } else {
            (&self.rotary_emb_global.cos, &self.rotary_emb_global.sin)
        };

        let (kbuf, vbuf, q) = if self.is_kv_shared {
            let q = rope_at_pos(&q, cos_t, sin_t)?;
            let slot = if self.is_sliding {
                shared_kv.sliding.as_ref()
            } else {
                shared_kv.full.as_ref()
            };
            let (k, v) = slot.ok_or_else(|| {
                candle::Error::Msg("kv-shared layer ran before any storing layer".to_string())
            })?;
            (k.clone(), v.clone(), q)
        } else {
            let k_proj = self.k_proj.as_ref().expect("non-shared layer has k_proj");
            let k_norm = self.k_norm.as_ref().expect("non-shared layer has k_norm");
            let k_raw = k_proj
                .forward(xs)?
                .reshape((1, 1, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v_raw = match self.v_proj.as_ref() {
                Some(v_proj) => v_proj
                    .forward(xs)?
                    .reshape((1, 1, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?,
                None => k_raw.clone(),
            };
            let k = k_norm.forward(&k_raw)?;
            let v = v_norm(&v_raw, self.rms_norm_eps)?;
            let q = rope_at_pos(&q, cos_t, sin_t)?;
            let k = rope_at_pos(&k, cos_t, sin_t)?;

            if self.static_kv.is_none() {
                let dev = xs.device();
                let shape = (self.num_kv_heads, ctx.max_seq, self.head_dim);
                self.static_kv = Some((
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                    Tensor::zeros(shape, xs.dtype(), dev)?,
                ));
            }
            let (kbuf, vbuf) = self.static_kv.as_ref().unwrap();
            candle_nn::fused::static_decode::kv_write(
                kbuf,
                vbuf,
                &k.reshape((self.num_kv_heads, self.head_dim))?.contiguous()?,
                &v.reshape((self.num_kv_heads, self.head_dim))?.contiguous()?,
                &ctx.pos,
            )?;
            if self.store_full_length_kv {
                let slot = if self.is_sliding {
                    &mut shared_kv.sliding
                } else {
                    &mut shared_kv.full
                };
                *slot = Some((kbuf.clone(), vbuf.clone()));
            }
            (kbuf.clone(), vbuf.clone(), q)
        };

        // Grouped attention over the full buffer.
        // q: [1, heads, 1, hd] -> [kvh, group, hd]
        let q3 = q.reshape((self.num_kv_heads, self.num_kv_groups, self.head_dim))?;
        let scores = q3
            .contiguous()?
            .matmul(&kbuf.transpose(1, 2)?.contiguous()?)?; // [kvh, group, max]
        let mask = if self.is_sliding {
            &ctx.mask_sliding
        } else {
            &ctx.mask_global
        };
        let scores = scores
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.reshape((1, 1, ctx.max_seq))?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(kbuf.dtype())?;
        let out = probs.matmul(&vbuf.contiguous()?)?; // [kvh, group, hd]
        out.reshape((1, 1, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// ── MoeBlock (per-layer MoE branch) ─────────────────────────────────────────

#[derive(Debug, Clone)]
struct MoeBlock {
    router: Router,
    experts: Experts,
    /// Normalizes the dense-MLP output before the branch sum.
    post_feedforward_layernorm_1: RmsNorm,
    /// Normalizes the expert input (off the raw residual).
    pre_feedforward_layernorm_2: RmsNorm,
    /// Normalizes the expert output before the branch sum.
    post_feedforward_layernorm_2: RmsNorm,
}

impl MoeBlock {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder, quant: Option<GgmlDType>) -> Result<Self> {
        Ok(Self {
            router: Router::new(cfg, vb.pp("router"), quant)?,
            experts: Experts::new(cfg, vb.pp("experts"), quant)?,
            post_feedforward_layernorm_1: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm_1"),
            )?,
            pre_feedforward_layernorm_2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm_2"),
            )?,
            post_feedforward_layernorm_2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm_2"),
            )?,
        })
    }

    /// `mlp_out`: dense MLP output. `residual`: the pre-feedforward residual
    /// stream (router and experts both read it, per the reference model).
    fn forward(&self, mlp_out: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let h1 = self.post_feedforward_layernorm_1.forward(mlp_out)?;

        let (b, s, h) = residual.dims3()?;
        let flat = residual.reshape((b * s, h))?;
        let h2 = self.pre_feedforward_layernorm_2.forward(&flat)?;
        let h2 = if let ExpertWeights::QuantFused { gate_up, down } = &self.experts.weights {
            // Fully on-device routing + fused expert dispatch: no host
            // round-trips anywhere in the MoE branch.
            let (ids, weights) = self.router.route_gpu(&flat)?;
            self.experts
                .forward_fused(&h2, &ids, &weights, gate_up, down)?
        } else {
            let plan = self.router.route(&flat)?;
            self.experts.forward(&h2, &plan)?
        };
        let h2 = h2.reshape((b, s, h))?;
        let h2 = self.post_feedforward_layernorm_2.forward(&h2)?;

        h1 + h2
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // Per-Layer Embeddings (PLE) integration, present when
    // cfg.hidden_size_per_layer_input > 0 (e.g. gemma-4-E2B-it).
    per_layer_input_gate: Option<Proj>,
    per_layer_projection: Option<Proj>,
    post_per_layer_input_norm: Option<RmsNorm>,
    // MoE branch, present when cfg.enable_moe_block (e.g. gemma-4-26B-A4B-it):
    // dense MLP and routed experts run in parallel off the same residual and
    // their (separately normalized) outputs are summed.
    moe: Option<MoeBlock>,
    act_fn: candle_nn::Activation,
    /// Learned scalar applied to the layer output (1.0 when absent).
    layer_scalar: f64,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
            quant,
        )?;
        // Models with `use_double_wide_mlp` (e.g. gemma-4-E2B-it) widen the MLP to
        // 2*intermediate_size in the trailing `num_kv_shared_layers` layers.
        let mut mlp_intermediate_size = cfg.intermediate_size;
        if cfg.use_double_wide_mlp
            && layer_idx >= cfg.num_hidden_layers.saturating_sub(cfg.num_kv_shared_layers)
        {
            mlp_intermediate_size *= 2;
        }
        let mlp = MLP::new(
            cfg.hidden_size,
            mlp_intermediate_size,
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
            quant,
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
            if cfg.hidden_size_per_layer_input > 0 {
                (
                    Some(Proj::new(
                        cfg.hidden_size,
                        cfg.hidden_size_per_layer_input,
                        false,
                        vb.pp("per_layer_input_gate"),
                        quant,
                    )?),
                    Some(Proj::new(
                        cfg.hidden_size_per_layer_input,
                        cfg.hidden_size,
                        false,
                        vb.pp("per_layer_projection"),
                        quant,
                    )?),
                    Some(RmsNorm::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_per_layer_input_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };
        // Learned per-layer output scalar (checkpoint buffer); 1.0 when absent.
        let layer_scalar = vb
            .get(1, "layer_scalar")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .map_or(1.0, |v| v[0] as f64);

        let moe = if cfg.enable_moe_block {
            Some(MoeBlock::new(cfg, vb.clone(), quant)?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            moe,
            act_fn: cfg.hidden_activation,
            layer_scalar,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        // Item-3 fusion: (residual add) + (pre-ffw rmsnorm) in one launch.
        let (xs_res, ffw_in) = candle_nn::fused::fused_add_rmsnorm(
            &xs,
            residual,
            &self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps as f32,
            false,
        )?;
        let residual = &xs_res;
        let xs = ffw_in.apply(&self.mlp)?;
        // MoE models sum the (normalized) dense and expert branches before the
        // shared post-feedforward norm; router/experts read the raw residual.
        let xs = match self.moe.as_ref() {
            Some(moe) => moe.forward(&xs, residual)?,
            None => xs,
        };
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;

        // Per-Layer Embeddings: gate the hidden state, multiply by this
        // layer's PLE slice, project back, normalize, residual-add.
        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let residual = &xs;
            let gated = xs.apply(gate)?.apply(&self.act_fn)?;
            let mixed = (gated * pli)?;
            let projected = mixed.apply(proj)?.apply(norm)?;
            xs = (residual + projected)?;
        }

        if self.layer_scalar != 1.0 {
            xs = (xs * self.layer_scalar)?;
        }
        Ok(xs)
    }

    /// One-token, shape-static step (graph-replayable): identical math to
    /// [`Self::forward`] with the attention going through the preallocated
    /// KV buffers and the device-side position/mask context.
    fn forward_static_chunk_dev(
        &mut self,
        xs: &Tensor, // [1, C, hidden]
        per_layer_input: Option<&Tensor>,
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_static_chunk_dev(&xs, ctx, shared_kv)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let (xs_res, ffw_in) = candle_nn::fused::fused_add_rmsnorm(
            &xs,
            residual,
            &self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps as f32,
            false,
        )?;
        let residual = &xs_res;
        let xs = ffw_in.apply(&self.mlp)?;
        let xs = match self.moe.as_ref() {
            Some(moe) => moe.forward(&xs, residual)?,
            None => xs,
        };
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let residual = &xs;
            let gated = xs.apply(gate)?.apply(&self.act_fn)?;
            let mixed = (gated * pli)?;
            let projected = mixed.apply(proj)?.apply(norm)?;
            xs = (residual + projected)?;
        }
        Ok(xs)
    }

    fn forward_static_chunk(
        &mut self,
        xs: &Tensor, // [1, T, hidden]
        per_layer_input: Option<&Tensor>,
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
        pos: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_static_chunk(&xs, ctx, shared_kv, pos)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let (xs_res, ffw_in) = candle_nn::fused::fused_add_rmsnorm(
            &xs,
            residual,
            &self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps as f32,
            false,
        )?;
        let residual = &xs_res;
        let xs = ffw_in.apply(&self.mlp)?;
        let xs = match self.moe.as_ref() {
            Some(moe) => moe.forward(&xs, residual)?,
            None => xs,
        };
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let residual = &xs;
            let gated = xs.apply(gate)?.apply(&self.act_fn)?;
            let mixed = (gated * pli)?;
            let projected = mixed.apply(proj)?.apply(norm)?;
            xs = (residual + projected)?;
        }
        if self.layer_scalar != 1.0 {
            xs = (xs * self.layer_scalar)?;
        }
        Ok(xs)
    }

    fn forward_static(
        &mut self,
        xs: &Tensor, // [1, 1, hidden]
        per_layer_input: Option<&Tensor>,
        ctx: &StaticCtx,
        shared_kv: &mut SharedKvStates,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_static(&xs, ctx, shared_kv)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let (xs_res, ffw_in) = candle_nn::fused::fused_add_rmsnorm(
            &xs,
            residual,
            &self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps as f32,
            false,
        )?;
        let residual = &xs_res;
        let xs = ffw_in.apply(&self.mlp)?;
        let xs = match self.moe.as_ref() {
            Some(moe) => moe.forward(&xs, residual)?,
            None => xs,
        };
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let residual = &xs;
            let gated = xs.apply(gate)?.apply(&self.act_fn)?;
            let mixed = (gated * pli)?;
            let projected = mixed.apply(proj)?.apply(norm)?;
            xs = (residual + projected)?;
        }
        if self.layer_scalar != 1.0 {
            xs = (xs * self.layer_scalar)?;
        }
        Ok(xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Proj,
    final_logit_softcapping: Option<f64>,
    // Per-Layer Embeddings (PLE) pipeline, present when
    // cfg.hidden_size_per_layer_input > 0.
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Proj>,
    per_layer_projection_norm: Option<RmsNorm>,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    static_ctx: Option<StaticCtx>,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_quant(cfg, vb, None)
    }

    /// Like [`Self::new`], but with `quant` set every linear projection is
    /// quantized to that GGML dtype at load time (activations stay in the
    /// VarBuilder dtype) and the gather-only embedding tables are stored in
    /// F16 — on gemma-4-E2B-it the PLE table alone is 2.35B parameters.
    pub fn new_with_quant(
        cfg: &Gemma4TextConfig,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_dtype = if quant.is_some() {
            DType::F16
        } else {
            vb.dtype()
        };
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens").set_dtype(embed_dtype),
        )?;

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
                quant,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            match quant {
                // Tied head: quantize a copy of the embedding table rather
                // than matmul against the F16 gather table.
                Some(dtype) => {
                    let emb = embed_tokens.embeddings();
                    let key = "lm_head.tied";
                    let q = if let Some(q) = qcache::get(key, emb.device()) {
                        Arc::new(q)
                    } else {
                        let q = if emb.device().is_cuda() {
                            let cpu = emb.to_device(&Device::Cpu)?;
                            QTensor::quantize_onto(&cpu, dtype, emb.device())?
                        } else {
                            QTensor::quantize(&emb.to_dtype(DType::F32)?, dtype)?
                        };
                        let q = Arc::new(q);
                        qcache::put(key, &q);
                        q
                    };
                    Proj::Quant {
                        weight: QMatMul::QTensor(q),
                        bias: None,
                    }
                }
                None => Proj::Plain(Linear::new(embed_tokens.embeddings().clone(), None)),
            }
        } else {
            Proj::new(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"), quant)?
        };

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if cfg.hidden_size_per_layer_input > 0 {
                (
                    Some(candle_nn::embedding(
                        cfg.vocab_size_per_layer_input,
                        cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
                        vb_m.pp("embed_tokens_per_layer").set_dtype(embed_dtype),
                    )?),
                    Some(Proj::new(
                        cfg.hidden_size,
                        cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
                        false,
                        vb_m.pp("per_layer_model_projection"),
                        quant,
                    )?),
                    Some(RmsNorm::new(
                        cfg.hidden_size_per_layer_input,
                        cfg.rms_norm_eps,
                        vb_m.pp("per_layer_projection_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };

        // Persist any freshly quantized tensors (no-op on warm cache).
        if let Err(e) = qcache::flush() {
            eprintln!("[gemma4-qcache] flush failed: {e}");
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            hidden_size_per_layer_input: cfg.hidden_size_per_layer_input,
            num_hidden_layers: cfg.num_hidden_layers,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            static_ctx: None,
        })
    }

    /// Combined Per-Layer Embeddings input: token-identity component (scaled
    /// per-layer embedding lookup) plus context component (projection of the
    /// scaled input embeddings), each normalized/scaled as in the reference
    /// implementation. Shape: [batch, seq, num_layers, per_layer_dim].
    fn per_layer_inputs(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Option<Tensor>> {
        let (Some(ple_embed), Some(proj), Some(norm)) = (
            self.embed_tokens_per_layer.as_ref(),
            self.per_layer_model_projection.as_ref(),
            self.per_layer_projection_norm.as_ref(),
        ) else {
            return Ok(None);
        };
        let (b_size, seq_len) = input_ids.dims2()?;
        let per_dim = self.hidden_size_per_layer_input;

        // Gather tables may be stored in F16 (quantized mode) — upcast after lookup.
        let ple = (ple_embed.forward(input_ids)?.to_dtype(self.dtype)? * (per_dim as f64).sqrt())?
            .reshape((b_size, seq_len, self.num_hidden_layers, per_dim))?;

        let projected = (inputs_embeds.apply(proj)? * (self.hidden_size as f64).sqrt().recip())?
            .reshape((b_size, seq_len, self.num_hidden_layers, per_dim))?
            .apply(norm)?;

        let combined = ((projected + ple)? * 2f64.sqrt().recip())?;
        Ok(Some(combined))
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        // The table may be stored in F16 (quantized mode) — upcast after lookup.
        let xs = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        let per_layer_inputs = self.per_layer_inputs(input_ids, &xs)?;
        self.forward_embeds(&xs, per_layer_inputs.as_ref(), seqlen_offset, b_size, seq_len)
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let mut shared_kv = SharedKvStates::default();
        let mut xs = xs.clone();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let pli = match per_layer_inputs {
                Some(p) => Some(p.i((.., .., layer_idx, ..))?),
                None => None,
            };
            xs = layer.forward(
                &xs,
                pli.as_ref(),
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                &mut shared_kv,
            )?;
            if std::env::var("GEMMA4_LDEBUG").is_ok() {
                let v = xs
                    .narrow(1, seq_len - 1, 1)?
                    .flatten_all()?
                    .to_dtype(candle::DType::F32)?;
                let mx = v.abs()?.max(0)?.to_scalar::<f32>()?;
                let mean = v.mean_all()?.to_scalar::<f32>()?;
                eprintln!("[ldbg] layer{layer_idx:02} mean={mean:.4} absmax={mx:.2}");
            }
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    /// Switch this model to the shape-static decode path with a fixed
    /// context budget. Allocates nothing until the first static step.
    pub fn enable_static_decode(&mut self, max_seq: usize) -> Result<()> {
        self.static_ctx = Some(StaticCtx::new(
            max_seq,
            self.sliding_window,
            &self.device,
        )?);
        Ok(())
    }

    pub fn static_enabled(&self) -> bool {
        self.static_ctx.is_some()
    }

    /// One-token decode step against the preallocated KV buffers. All
    /// position dependence lives on-device (position scalar, masks, rope
    /// row gather), so the whole call is CUDA-graph replayable; the
    /// position advances at the end of the step (inside the graph).
    /// Chunked static prefill: run the whole prompt through the static KV
    /// buffers in GEMMA4_PREFILL_CHUNK-token chunks (default 512) with
    /// batched attention, leaving the device position at prompt length and
    /// returning last-position logits. Replaces token-stepped static prefill
    /// (which measured ~64x slower on 1.6k-token prompts).
    /// Rewind the device position cursor to zero (prefill-graph capture).
    pub fn reset_static_pos(&mut self) -> Result<()> {
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        candle_nn::fused::static_decode::write_u32(&ctx.pos, 0)
    }

    /// The device position cursor (u32 [1]).
    pub fn static_pos(&self) -> Result<&Tensor> {
        Ok(&self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?
            .pos)
    }

    /// A2: allocate the device-resident chunk state for graphed prefill.
    pub fn enable_chunk_graph(&mut self, chunk: usize) -> Result<()> {
        let ctx = self
            .static_ctx
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        if ctx.chunk.is_none() {
            let dev = self.device.clone();
            ctx.chunk = Some(ChunkCtx {
                ids: Tensor::zeros((1, chunk), DType::U32, &dev)?,
                rope_idx: Tensor::zeros(chunk, DType::U32, &dev)?,
                mask_global: Tensor::zeros((chunk, ctx.max_seq), DType::F32, &dev)?,
                mask_sliding: Tensor::zeros((chunk, ctx.max_seq), DType::F32, &dev)?,
                chunk,
            });
        }
        Ok(())
    }

    /// A2: one full prefill chunk step reading ONLY device state — the body
    /// that gets captured into a CUDA graph. Refreshes rope indices and
    /// chunk masks from the position cursor, runs all layers, advances the
    /// cursor by the chunk size, and returns the last row's hidden->logits.
    pub fn prefill_chunk_step(&mut self) -> Result<(Tensor, Tensor)> {
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?
            .clone();
        let cctx = ctx
            .chunk
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("chunk graph not enabled".into()))?;
        use candle_nn::fused::static_decode as sd;
        sd::iota_add_u32(&cctx.rope_idx, &ctx.pos)?;
        sd::chunk_mask_from_pos(&cctx.mask_global, &ctx.pos, None)?;
        sd::chunk_mask_from_pos(&cctx.mask_sliding, &ctx.pos, Some(ctx.sliding_window))?;
        let ids = cctx.ids.clone();
        let xs = self.embed_tokens(&ids)?;
        let per_layer_inputs = self.per_layer_inputs(&ids, &xs)?;
        let mut shared_kv = SharedKvStates::default();
        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let pli = match per_layer_inputs.as_ref() {
                Some(p) => Some(p.i((.., .., layer_idx, ..))?),
                None => None,
            };
            xs = layer.forward_static_chunk_dev(&xs, pli.as_ref(), &ctx, &mut shared_kv)?;
        }
        sd::incr_add_u32(&ctx.pos, cctx.chunk as u32)?;
        // Return the full chunk hidden (cheap) + row C-1 logits; when the
        // final chunk is padded the driver derives logits from the REAL last
        // row of the hidden instead.
        let last = xs.narrow(1, cctx.chunk - 1, 1)?;
        let logits = self.logits_from_hidden(&last)?;
        Ok((xs, logits))
    }

    /// norm + lm_head (+softcap) on a [1, n, hidden] slice.
    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits = hidden.apply(&self.norm)?.apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    /// A2 driver (eager form; exp1 captures prefill_chunk_step and replays):
    /// pad the prompt to a chunk multiple, feed chunk ids between steps,
    /// then pin the cursor to the REAL length so padded KV rows fall outside
    /// every subsequent mask.
    pub fn prefill_static_graph(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, total) = input_ids.dims2()?;
        let chunk = {
            let ctx = self
                .static_ctx
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
            ctx.chunk
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("chunk graph not enabled".into()))?
                .chunk
        };
        let n_chunks = total.div_ceil(chunk);
        let mut ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        ids.resize(n_chunks * chunk, 0);
        let mut last: Option<(Tensor, Tensor)> = None;
        for c in 0..n_chunks {
            self.feed_chunk_ids(&ids[c * chunk..(c + 1) * chunk])?;
            last = Some(self.prefill_chunk_step()?);
        }
        // Pin cursor to the real length: padding rows now sit beyond every
        // subsequent validity mask (decode masks come from ctx.pos).
        let ctx = self.static_ctx.as_ref().unwrap();
        candle_nn::fused::static_decode::write_u32(&ctx.pos, total as u32)?;
        let (hidden, logits_row) = last.ok_or_else(|| candle::Error::Msg("empty prompt".into()))?;
        let pad = n_chunks * chunk - total;
        if pad == 0 {
            Ok(logits_row)
        } else {
            let real = hidden.narrow(1, chunk - 1 - pad, 1)?;
            self.logits_from_hidden(&real)
        }
    }

    /// Host->device feed of the next chunk's token ids (between replays).
    pub fn feed_chunk_ids(&mut self, ids: &[u32]) -> Result<()> {
        let ctx = self.static_ctx.as_ref().unwrap();
        let cctx = ctx.chunk.as_ref().unwrap();
        let src = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        candle_nn::fused::static_decode::copy_into(&cctx.ids, &src)
    }

    pub fn prefill_static_chunked(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, total) = input_ids.dims2()?;
        if self.static_ctx.is_none() {
            candle::bail!("static decode not enabled");
        }
        let chunk = std::env::var("GEMMA4_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512)
            .max(1);
        let mut pos = 0usize;
        let mut last: Option<Tensor> = None;
        while pos < total {
            let t = chunk.min(total - pos);
            let ids = input_ids.narrow(1, pos, t)?;
            let xs = self.embed_tokens(&ids)?;
            let per_layer_inputs = self.per_layer_inputs(&ids, &xs)?;
            let ctx = self.static_ctx.as_ref().unwrap();
            let mut shared_kv = SharedKvStates::default();
            let mut xs = xs;
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                let pli = match per_layer_inputs.as_ref() {
                    Some(p) => Some(p.i((.., .., layer_idx, ..))?),
                    None => None,
                };
                xs = layer.forward_static_chunk(&xs, pli.as_ref(), ctx, &mut shared_kv, pos)?;
            }
            last = Some(xs.narrow(1, t - 1, 1)?);
            pos += t;
        }
        let ctx = self.static_ctx.as_ref().unwrap();
        candle_nn::fused::static_decode::write_u32(&ctx.pos, total as u32)?;
        let xs = last.ok_or_else(|| candle::Error::Msg("empty prompt".into()))?;
        let logits = xs.apply(&self.norm)?.apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn forward_static(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens(input_ids)?;
        let per_layer_inputs = self.per_layer_inputs(input_ids, &xs)?;
        let ctx = self
            .static_ctx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("static decode not enabled".into()))?;
        ctx.refresh_masks()?;
        let mut shared_kv = SharedKvStates::default();
        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let pli = match per_layer_inputs.as_ref() {
                Some(p) => Some(p.i((.., .., layer_idx, ..))?),
                None => None,
            };
            xs = layer.forward_static(&xs, pli.as_ref(), ctx, &mut shared_kv)?;
        }
        let logits = xs.apply(&self.norm)?.apply(&self.lm_head)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };
        candle_nn::fused::static_decode::incr_u32(&ctx.pos)?;
        Ok(logits)
    }

    /// Rewind the static context to position 0 (host-side; call between
    /// requests, never inside a captured graph).
    pub fn reset_static(&mut self) -> Result<()> {
        match self.static_ctx.as_ref() {
            Some(ctx) => ctx.reset(),
            None => Ok(()),
        }
    }

    /// Debug helper: read back the device position scalar.
    pub fn static_pos_debug(&self) -> Result<u32> {
        match self.static_ctx.as_ref() {
            Some(ctx) => Ok(ctx.pos.to_vec1::<u32>()?[0]),
            None => Ok(u32::MAX),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
        let _ = self.reset_static();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny gemma-4-26B-A4B-style config: MoE on every layer, K==V global
    /// attention, distinct global head geometry. Zero-initialized weights —
    /// this validates shapes, tensor-name wiring, routing dispatch and both
    /// attention variants, not numerics.
    fn moe_test_config() -> Gemma4TextConfig {
        serde_json::from_str(
            r#"{
              "attention_k_eq_v": true,
              "enable_moe_block": true,
              "num_experts": 4,
              "top_k_experts": 2,
              "moe_intermediate_size": 8,
              "hidden_size": 32,
              "intermediate_size": 16,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "num_global_key_value_heads": 1,
              "head_dim": 8,
              "global_head_dim": 16,
              "num_hidden_layers": 3,
              "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
              "sliding_window": 4,
              "max_position_embeddings": 64,
              "vocab_size": 64,
              "final_logit_softcapping": 30.0,
              "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
              }
            }"#,
        )
        .expect("test config parses")
    }

    #[test]
    fn moe_k_eq_v_forward_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = moe_test_config();
        let mut model = TextModel::new(&cfg, vb)?;

        // Prefill 5 tokens, then decode 1 with offset — exercises both mask
        // paths, the rotating (sliding) and normal (global) KV caches, the
        // K==V global layer and the MoE dispatch.
        let ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &dev)?;
        let logits = model.forward(&ids, 0)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        let ids = Tensor::new(&[[6u32]], &dev)?;
        let logits = model.forward(&ids, 5)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        let v = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
        Ok(())
    }

    #[test]
    fn moe_quantized_load_path() -> Result<()> {
        // q8_0 requires the row length to be a multiple of the 32-wide GGML
        // block; hidden 32 / moe_intermediate 8 don't satisfy that for every
        // projection, so quantize only shapes that do by using hidden 64.
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut cfg = moe_test_config();
        cfg.hidden_size = 64;
        cfg.moe_intermediate_size = 32;
        cfg.intermediate_size = 32;
        let mut model = TextModel::new_with_quant(&cfg, vb, Some(GgmlDType::Q8_0))?;
        let ids = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        let logits = model.forward(&ids, 0)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        Ok(())
    }
}

#[cfg(test)]
mod config_null_tests {
    use super::config_null_tests_support::*;

    #[test]
    fn explicit_nulls_parse_as_defaults() {
        let cfg = parse_e2b_style_config();
        assert!(!cfg.enable_moe_block);
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.top_k_experts, 0);
        assert!(!cfg.attention_k_eq_v);
    }
}

#[cfg(test)]
mod config_null_tests_support {
    pub fn parse_e2b_style_config() -> super::Gemma4TextConfig {
        serde_json::from_str(
            r#"{
              "hidden_size": 32, "intermediate_size": 16, "num_hidden_layers": 1,
              "layer_types": ["sliding_attention"], "sliding_window": 4,
              "num_experts": null, "top_k_experts": null, "enable_moe_block": null,
              "attention_k_eq_v": null, "moe_intermediate_size": null,
              "num_global_key_value_heads": null, "use_bidirectional_attention": null
            }"#,
        )
        .expect("null-bearing config parses")
    }
}
