//! Torch-compatible bicubic (antialias) resampling of learned position
//! grids, shared by vision towers that support non-native input sizes.
//
// The reference resizes learned position grids with
// F.interpolate(mode='bicubic', antialias=True, align_corners=False) when the
// runtime grid differs from the pretrain grid (e.g. 640-pixel global views:
// CLIP 16x16 -> 10x10, SAM 64x64 -> 40x40). Computed once at load/first-use
// on host f32; exactness is checked by the parity harness against a torch
// dump of the same tensors.

pub(crate) fn cubic(a: f64, x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// One-dimensional resize weights for torch bicubic with antialias
/// (align_corners=false): support widened by the downscale factor.
pub(crate) fn bicubic_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f64>)> {
    const A: f64 = -0.75; // torch's cubic coefficient
    let scale = src as f64 / dst as f64;
    let support_scale = scale.max(1.0); // antialias widening on downscale
    let support = 2.0 * support_scale;
    let mut out = Vec::with_capacity(dst);
    for d in 0..dst {
        let center = (d as f64 + 0.5) * scale - 0.5;
        let lo = ((center - support).floor() as isize).max(0) as usize;
        let hi = ((center + support).ceil() as isize).min(src as isize - 1) as usize;
        let mut ws = Vec::with_capacity(hi - lo + 1);
        let mut sum = 0.0;
        for i in lo..=hi {
            let w = cubic(A, (i as f64 - center) / support_scale);
            ws.push(w);
            sum += w;
        }
        for w in ws.iter_mut() {
            *w /= sum;
        }
        out.push((lo, ws));
    }
    out
}

/// Bicubic-antialias resize of a [src, src, dim] grid to [dst, dst, dim]
/// (host f32; separable passes).
pub fn resize_grid_f32(grid: &[f32], src: usize, dst: usize, dim: usize) -> Vec<f32> {
    let wcols = bicubic_weights(src, dst);
    // Horizontal pass: [src, src, dim] -> [src, dst, dim]
    let mut tmp = vec![0f32; src * dst * dim];
    for r in 0..src {
        for (c_out, (lo, ws)) in wcols.iter().enumerate() {
            for d in 0..dim {
                let mut acc = 0f64;
                for (k, w) in ws.iter().enumerate() {
                    acc += *w * grid[(r * src + (lo + k)) * dim + d] as f64;
                }
                tmp[(r * dst + c_out) * dim + d] = acc as f32;
            }
        }
    }
    // Vertical pass: [src, dst, dim] -> [dst, dst, dim]
    let wrows = bicubic_weights(src, dst);
    let mut out = vec![0f32; dst * dst * dim];
    for (r_out, (lo, ws)) in wrows.iter().enumerate() {
        for c in 0..dst {
            for d in 0..dim {
                let mut acc = 0f64;
                for (k, w) in ws.iter().enumerate() {
                    acc += *w * tmp[((lo + k) * dst + c) * dim + d] as f64;
                }
                out[(r_out * dst + c) * dim + d] = acc as f32;
            }
        }
    }
    out
}

