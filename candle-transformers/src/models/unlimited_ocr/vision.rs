//! Unlimited-OCR vision stack (DeepEncoder): SAM-ViT-B backbone (reused from
//! candle's segment_anything image encoder — identical ViTDet blocks and
//! weight names) + net_2/net_3 downsamples (256→512→1024, stride 2 each),
//! CHAINED into a CLIP-L-shaped NoTP tower that consumes the SAM features as
//! its patch embeddings, then concat(clip[1:], sam) → linear projector →
//! decoder embedding space with an image_newline column per grid row and a
//! trailing view separator.
//!
//! V1 scope: single global view at 1024×1024 (candidate_resolutions
//! [[1024,1024]]), where every positional embedding is used at its native
//! size so no bicubic interpolation paths are needed.

use candle::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use super::super::segment_anything::image_encoder::ImageEncoderViT;

fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    x * candle_nn::ops::sigmoid(&(x * 1.702f64)?)?
}

// ── CLIP-L NoTP tower ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NoTpBlock {
    ln1: LayerNorm,
    ln2: LayerNorm,
    qkv_proj: Linear,
    out_proj: Linear,
    fc1: Linear,
    fc2: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl NoTpBlock {
    fn new(hidden: usize, heads: usize, ffn: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(hidden, 1e-5, vb.pp("layer_norm1"))?,
            ln2: layer_norm(hidden, 1e-5, vb.pp("layer_norm2"))?,
            qkv_proj: linear(hidden, hidden * 3, vb.pp("self_attn").pp("qkv_proj"))?,
            out_proj: linear(hidden, hidden, vb.pp("self_attn").pp("out_proj"))?,
            fc1: linear(hidden, ffn, vb.pp("mlp").pp("fc1"))?,
            fc2: linear(ffn, hidden, vb.pp("mlp").pp("fc2"))?,
            n_heads: heads,
            head_dim: hidden / heads,
        })
    }

    fn attn(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        let qkv = self.qkv_proj.forward(x)?; // [b, seq, 3*hidden]
        let qkv = qkv.reshape((b, seq, 3, self.n_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?.contiguous()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let out = att.matmul(&v)?; // [b, heads, seq, hd]
        out.transpose(1, 2)?
            .reshape((b, seq, hidden))?
            .apply(&self.out_proj)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = (x + self.attn(&self.ln1.forward(x)?)?)?;
        let m = self
            .fc2
            .forward(&quick_gelu(&self.fc1.forward(&self.ln2.forward(&h)?)?)?)?;
        h + m
    }
}

#[derive(Debug, Clone)]
struct ClipTower {
    class_embedding: Tensor,    // [hidden]
    position_embedding: Tensor, // [num_positions, hidden]
    pre_ln: LayerNorm,
    blocks: Vec<NoTpBlock>,
}

impl ClipTower {
    fn new(vb: VarBuilder) -> Result<Self> {
        let (hidden, heads, ffn, layers, num_pos) = (1024, 16, 4096, 24, 257);
        let emb = vb.pp("embeddings");
        let class_embedding = emb.get(hidden, "class_embedding")?;
        let position_embedding = emb
            .pp("position_embedding")
            .get((num_pos, hidden), "weight")?;
        // patch_embedding exists in the checkpoint but is bypassed in the
        // chained configuration (SAM features arrive as patch embeds).
        let mut blocks = Vec::with_capacity(layers);
        let vb_l = vb.pp("transformer").pp("layers");
        for i in 0..layers {
            blocks.push(NoTpBlock::new(hidden, heads, ffn, vb_l.pp(i))?);
        }
        Ok(Self {
            class_embedding,
            position_embedding,
            pre_ln: layer_norm(hidden, 1e-5, vb.pp("pre_layrnorm"))?,
            blocks,
        })
    }

    /// patch_embeds: [b, 1024, gh, gw] (SAM features). V1 requires
    /// gh*gw + 1 == num_positions (native 16x16 grid, no pos interpolation).
    fn forward(&self, patch_embeds: &Tensor) -> Result<Tensor> {
        let (b, c, gh, gw) = patch_embeds.dims4()?;
        let seq = gh * gw;
        let pe = patch_embeds.reshape((b, c, seq))?.transpose(1, 2)?; // [b, seq, c]
        let cls = self
            .class_embedding
            .reshape((1, 1, c))?
            .expand((b, 1, c))?
            .to_dtype(pe.dtype())?;
        let x = Tensor::cat(&[cls, pe], 1)?;
        let pos_native = self.position_embedding.reshape((1, (), c))?;
        let pos = interp_pos_with_cls(&pos_native, gh)?.to_dtype(x.dtype())?;
        let x = x.broadcast_add(&pos)?;
        let mut x = self.pre_ln.forward(&x)?;
        for blk in self.blocks.iter() {
            x = blk.forward(&x)?;
        }
        Ok(x) // [b, seq+1, 1024] — no final norm in the reference
    }
}

// ── DeepEncoder ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DeepEncoder {
    sam: ImageEncoderViT,
    net_2: candle_nn::Conv2d,
    net_3: candle_nn::Conv2d,
    clip: ClipTower,
    projector: Linear,
    image_newline: Tensor,  // [n_embed]
    view_seperator: Tensor, // [n_embed] (sic — checkpoint spelling)
}

impl DeepEncoder {
    /// vb: the checkpoint ROOT VarBuilder (tensors live under `model.*`).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let m = vb.pp("model");
        let sam = ImageEncoderViT::new(
            1024,
            16,
            3,
            768,
            12,
            12,
            256,
            true,
            true,
            true,
            14,
            &[2, 5, 8, 11],
            m.pp("sam_model"),
        )?;
        let cfg2 = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let net_2 = candle_nn::conv2d_no_bias(256, 512, 3, cfg2, m.pp("sam_model").pp("net_2"))?;
        let net_3 = candle_nn::conv2d_no_bias(512, 1024, 3, cfg2, m.pp("sam_model").pp("net_3"))?;
        let clip = ClipTower::new(m.pp("vision_model"))?;
        let projector = linear(2048, 1280, m.pp("projector").pp("layers"))?;
        Ok(Self {
            sam,
            net_2,
            net_3,
            clip,
            projector,
            image_newline: m.get(1280, "image_newline")?,
            view_seperator: m.get(1280, "view_seperator")?,
        })
    }

    /// Parity helper: run only the CLIP tower on externally supplied
    /// patch features.
    pub fn clip_only(&self, f1: &Tensor) -> Result<Tensor> {
        self.clip.forward(f1)
    }

    /// Staged forward for parity work: (sam+nets, clip, final).
    pub fn forward_staged(&self, img: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let f1 = self.sam.forward(img)?;
        let f1 = self.net_3.forward(&self.net_2.forward(&f1)?)?;
        let f2 = self.clip.forward(&f1)?;
        let fin = self.assemble(&f1, &f2)?;
        Ok((f1, f2, fin))
    }

    /// Single global 1024×1024 view → decoder-space image embeddings:
    /// [gh*(gw+1) + 1, n_embed] (newline column per row + view separator).
    pub fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let f1 = self.sam.forward(img)?; // [b, 256, 64, 64]
        let f1 = self.net_3.forward(&self.net_2.forward(&f1)?)?; // [b, 1024, 16, 16]
        let f2 = self.clip.forward(&f1)?; // [b, 257, 1024]
        self.assemble(&f1, &f2)
    }

    /// Crop mode (their default for large images): global 1024 view + local
    /// 640 crops in a (width_crop_num x height_crop_num) grid. Feature order
    /// mirrors the reference exactly: [locals mosaic, global, separator].
    /// Locals: per-crop 10x10 grids arranged into an (hc*10, wc*10) mosaic
    /// (crop-row major), one image_newline per mosaic row.
    pub fn forward_crops(
        &self,
        global_img: &Tensor, // [1, 3, 1024, 1024]
        crops: &Tensor,      // [n, 3, 640, 640], n = wc * hc, row-major tiles
        wc: usize,
        hc: usize,
    ) -> Result<Tensor> {
        let global = self.forward(global_img)?; // [273, d] = grid+newline+sep
        let (g_len, d) = global.dims2()?;
        let global_body = global.i((..g_len - 1, ..))?; // strip separator
        let sep = global.i((g_len - 1.., ..))?;

        // Locals through the towers (chunked to bound activation memory).
        let n = crops.dim(0)?;
        let mut feats = Vec::new();
        let chunk = 8usize;
        let mut i = 0;
        while i < n {
            let m = chunk.min(n - i);
            let part = crops.narrow(0, i, m)?;
            let f1 = self.sam.forward(&part)?;
            let f1 = self.net_3.forward(&self.net_2.forward(&f1)?)?; // [m, d1, 10, 10]
            let f2 = self.clip.forward(&f1)?; // [m, 101, 1024]
            let (b, c, gh, gw) = f1.dims4()?;
            let sam_seq = f1.reshape((b, c, gh * gw))?.transpose(1, 2)?;
            // .contiguous(): sam_seq is a transpose and the cat of it keeps
            // non-standard strides, which the batched matmul in the projector
            // rejects. Harmless at m == 1, which is why the single-view path
            // never hit it.
            let merged = Tensor::cat(&[f2.i((.., 1.., ..))?, sam_seq], 2)?.contiguous()?;
            feats.push(self.projector.forward(&merged)?); // [m, 100, d]
            i += m;
        }
        let local = Tensor::cat(&feats, 0)?; // [n, 100, d]
        let g2 = local.dim(1)?;
        let gsz = (g2 as f64).sqrt() as usize; // 10
                                               // (hc, wc, g, g, d) -> permute(0,2,1,3,4) -> (hc*g, wc*g, d)
        let mosaic = local
            .reshape((hc, wc, gsz, gsz, d))?
            .permute((0, 2, 1, 3, 4))?
            .reshape((hc * gsz, wc * gsz, d))?;
        let newline = self
            .image_newline
            .to_dtype(mosaic.dtype())?
            .reshape((1, 1, d))?
            .expand((hc * gsz, 1, d))?;
        let mosaic = Tensor::cat(&[mosaic, newline], 1)?; // [hc*g, wc*g+1, d]
        let local_flat = mosaic.reshape((hc * gsz * (wc * gsz + 1), d))?;
        Tensor::cat(&[local_flat, global_body, sep], 0)
    }

    fn assemble(&self, f1: &Tensor, f2: &Tensor) -> Result<Tensor> {
        let (b, c, gh, gw) = f1.dims4()?;
        let sam_seq = f1.reshape((b, c, gh * gw))?.transpose(1, 2)?; // [b, 256, 1024]
        let feats = Tensor::cat(&[f2.i((.., 1.., ..))?, sam_seq], D_MINUS1)?; // [b, 256, 2048]
        let feats = self.projector.forward(&feats)?; // [b, 256, n_embed]
        let n_embed = feats.dim(2)?;
        let grid = feats.reshape((gh, gw, n_embed))?; // b == 1 in v1
        let newline = self
            .image_newline
            .to_dtype(grid.dtype())?
            .reshape((1, 1, n_embed))?
            .expand((gh, 1, n_embed))?;
        let grid = Tensor::cat(&[grid, newline], 1)?; // [gh, gw+1, n_embed]
        let flat = grid.reshape((gh * (gw + 1), n_embed))?;
        let sep = self
            .view_seperator
            .to_dtype(flat.dtype())?
            .reshape((1, n_embed))?;
        Tensor::cat(&[flat, sep], 0)
    }
}

use candle::D::Minus1 as D_MINUS1;

// Position-grid interpolation: shared torch-bicubic-antialias core lives in
// models::interpolation; this wrapper handles the cls-token-carrying CLIP
// layout [1, src*src+1, dim].
pub(crate) fn interp_pos_with_cls(pos: &Tensor, tgt_grid: usize) -> Result<Tensor> {
    let (one, n, dim) = pos.dims3()?;
    let src = ((n - 1) as f64).sqrt() as usize;
    if src * src + 1 != n || one != 1 {
        candle::bail!("unexpected pos shape {:?}", pos.dims());
    }
    if src == tgt_grid {
        return Ok(pos.clone());
    }
    let f = pos.to_dtype(DType::F32)?;
    let cls: Vec<f32> = f.i((0, 0, ..))?.to_vec1()?;
    let body: Vec<f32> = f.i((0, 1.., ..))?.flatten_all()?.to_vec1()?;
    let resized = crate::models::interpolation::resize_grid_f32(&body, src, tgt_grid, dim);
    let mut all = cls;
    all.extend(resized);
    Tensor::from_vec(all, (1, tgt_grid * tgt_grid + 1, dim), pos.device())?.to_dtype(pos.dtype())
}
