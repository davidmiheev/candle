//! Fused decode-path ops (launch-overhead reduction, item 3).
//!
//! - [`fused_add_rmsnorm`]: one launch computing both the new residual
//!   stream (`x + residual`) and its RMS-norm (optionally gemma's `1 + w`
//!   weighting). Returns `(residual_sum, normed)`.
//! - [`fused_swiglu`]: one launch computing `act(gate) * up` from a packed
//!   `[..., 2*d]` tensor (gate = first half). `Silu` or `GeluTanh`.
//!
//! Non-CUDA devices (and unsupported dtypes) fall back to the equivalent
//! composition of standard ops, so callers can use these unconditionally.

use candle::{DType, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluAct {
    Silu,
    GeluTanh,
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::SwigluAct;
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{
        CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
    };
    use candle::cuda_backend::{kernel_name, kernels, Map1, Map3, WrapErr};
    use candle::{CpuStorage, CudaDevice, Layout, Result, Shape, WithDType};

    pub(super) struct FusedAddRmsNorm {
        pub eps: f32,
        pub plus_one: bool,
    }

    impl candle::CustomOp3 for FusedAddRmsNorm {
        fn name(&self) -> &'static str {
            "fused-add-rmsnorm"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle::bail!("fused-add-rmsnorm: cpu path is handled by the caller fallback")
        }

        fn cuda_fwd(
            &self,
            x: &candle::CudaStorage,
            x_l: &Layout,
            res: &candle::CudaStorage,
            res_l: &Layout,
            w: &candle::CudaStorage,
            w_l: &Layout,
        ) -> Result<(candle::CudaStorage, Shape)> {
            struct S {
                eps: f32,
                plus_one: i32,
            }
            impl Map3 for S {
                fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                    &self,
                    x: &CudaSlice<T>,
                    x_l: &Layout,
                    res: &CudaSlice<T>,
                    res_l: &Layout,
                    w: &CudaSlice<T>,
                    w_l: &Layout,
                    dev: &CudaDevice,
                ) -> Result<CudaSlice<T>> {
                    let (xo1, xo2) = x_l
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::Msg("x must be contiguous".into()))?;
                    let (ro1, ro2) = res_l
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::Msg("residual must be contiguous".into()))?;
                    let (wo1, wo2) = w_l
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::Msg("weight must be contiguous".into()))?;
                    let x = x.slice(xo1..xo2);
                    let res = res.slice(ro1..ro2);
                    let w = w.slice(wo1..wo2);

                    let dims = x_l.shape().dims();
                    let d = *dims.last().unwrap();
                    let el = x_l.shape().elem_count();
                    let rows = el / d;

                    let func = dev.get_or_load_func(
                        &kernel_name::<T>("fused_add_rmsnorm"),
                        &kernels::FUSED,
                    )?;
                    // Output holds [2, rows, d]: sum stream then normed.
                    let dst = unsafe { dev.alloc::<T>(2 * el)? };
                    let block = 256u32;
                    let cfg = LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: (block / 32) * 4,
                    };
                    let mut builder = func.builder();
                    builder.arg(&x);
                    builder.arg(&res);
                    builder.arg(&w);
                    builder.arg(&dst);
                    candle::builder_arg!(builder, rows as i32, d as i32, self.eps, self.plus_one);
                    unsafe { builder.launch(cfg) }.w()?;
                    Ok(dst)
                }
            }

            let dev = x.device();
            let slice = S {
                eps: self.eps,
                plus_one: if self.plus_one { 1 } else { 0 },
            }
            .map(&x.slice, x_l, &res.slice, res_l, &w.slice, w_l, dev)?;
            let dst = candle::cuda_backend::CudaStorage {
                slice,
                device: dev.clone(),
            };
            let mut out_shape = vec![2usize];
            out_shape.extend_from_slice(x_l.shape().dims());
            Ok((dst, out_shape.into()))
        }
    }

    pub(super) struct FusedSwiglu {
        pub act: SwigluAct,
    }

    impl candle::CustomOp1 for FusedSwiglu {
        fn name(&self) -> &'static str {
            "fused-swiglu"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle::bail!("fused-swiglu: cpu path is handled by the caller fallback")
        }

        fn cuda_fwd(
            &self,
            gu: &candle::CudaStorage,
            gu_l: &Layout,
        ) -> Result<(candle::CudaStorage, Shape)> {
            struct S {
                act: i32,
            }
            impl Map1 for S {
                fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                    &self,
                    gu: &CudaSlice<T>,
                    dev: &CudaDevice,
                    gu_l: &Layout,
                ) -> Result<CudaSlice<T>> {
                    let (o1, o2) = gu_l
                        .contiguous_offsets()
                        .ok_or_else(|| candle::Error::Msg("gate_up must be contiguous".into()))?;
                    let gu = gu.slice(o1..o2);
                    let dims = gu_l.shape().dims();
                    let two_d = *dims.last().unwrap();
                    if two_d % 2 != 0 {
                        candle::bail!("fused-swiglu: last dim must be even")
                    }
                    let d = two_d / 2;
                    let rows = gu_l.shape().elem_count() / two_d;

                    let func =
                        dev.get_or_load_func(&kernel_name::<T>("fused_swiglu"), &kernels::FUSED)?;
                    let dst = unsafe { dev.alloc::<T>(rows * d)? };
                    let total = (rows * d) as u32;
                    let block = 256u32;
                    let cfg = LaunchConfig {
                        grid_dim: (total.div_ceil(block), 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&gu);
                    builder.arg(&dst);
                    candle::builder_arg!(builder, rows as i32, d as i32, self.act);
                    unsafe { builder.launch(cfg) }.w()?;
                    Ok(dst)
                }
            }

            let dev = gu.device();
            let slice = S {
                act: match self.act {
                    SwigluAct::Silu => 0,
                    SwigluAct::GeluTanh => 1,
                },
            }
            .map(&gu.slice, dev, gu_l)?;
            let dst = candle::cuda_backend::CudaStorage {
                slice,
                device: dev.clone(),
            };
            let mut out_shape = gu_l.shape().dims().to_vec();
            *out_shape.last_mut().unwrap() /= 2;
            Ok((dst, out_shape.into()))
        }
    }
}

/// `x + residual`, then RMS-norm with `weight` (optionally gemma's `1 + w`).
/// Returns `(residual_sum, normed)`. Single kernel launch on CUDA.
pub fn fused_add_rmsnorm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    plus_one: bool,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && matches!(x.dtype(), DType::F32 | DType::BF16) {
        let op = cuda::FusedAddRmsNorm { eps, plus_one };
        let both = x
            .contiguous()?
            .apply_op3_no_bwd(&residual.contiguous()?, &weight.contiguous()?, &op)?;
        let sum = both.get(0)?;
        let normed = both.get(1)?;
        return Ok((sum, normed));
    }
    // Fallback composition (CPU / other dtypes / non-cuda builds).
    let sum = (x + residual)?;
    let x32 = sum.to_dtype(DType::F32)?;
    let variance = x32.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x32.broadcast_div(&(variance + eps as f64)?.sqrt()?)?;
    let w = if plus_one {
        (weight.to_dtype(DType::F32)? + 1.0)?
    } else {
        weight.to_dtype(DType::F32)?
    };
    let normed = normed.broadcast_mul(&w)?.to_dtype(sum.dtype())?;
    Ok((sum, normed))
}

/// `act(gate) * up` for a packed `[..., 2*d]` tensor (gate first). Single
/// kernel launch on CUDA.
pub fn fused_swiglu(gate_up: &Tensor, act: SwigluAct) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate_up.device().is_cuda() && matches!(gate_up.dtype(), DType::F32 | DType::BF16) {
        let op = cuda::FusedSwiglu { act };
        return gate_up.contiguous()?.apply_op1_no_bwd(&op);
    }
    let chunks = gate_up.chunk(2, D::Minus1)?;
    let g = match act {
        SwigluAct::Silu => chunks[0].silu()?,
        SwigluAct::GeluTanh => chunks[0].gelu()?,
    };
    &g * &chunks[1]
}

// ── Static-decode helpers (item 3 phase 2) ──────────────────────────────────
// In-place operations on preallocated buffers, all replayable inside a CUDA
// graph. These intentionally bypass the CustomOp machinery: they mutate
// existing device buffers (KV cache, mask row, device position scalar)
// rather than producing new tensors.

#[cfg(feature = "cuda")]
pub mod static_decode {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle::cuda_backend::{kernels, CudaStorageSlice, WrapErr};
    use candle::{DType, Result, Storage, Tensor};

    fn cuda_parts<'a>(
        t: &'a Tensor,
    ) -> Result<(std::sync::RwLockReadGuard<'a, Storage>, usize)> {
        let (s, l) = t.storage_and_layout();
        let off = l.start_offset();
        Ok((s, off))
    }

    macro_rules! with_slice {
        ($storage:expr, $off:expr, $dt:ident, $v:ident, $body:block) => {
            match &*$storage {
                Storage::Cuda(c) => match &c.slice {
                    CudaStorageSlice::$dt(s) => {
                        let $v = s.slice($off..);
                        $body
                    }
                    _ => candle::bail!("unexpected dtype for static-decode op"),
                },
                _ => candle::bail!("static-decode ops are CUDA only"),
            }
        };
    }

    /// Write `knew`/`vnew` (`[heads, dim]`, contiguous) into
    /// `kbuf`/`vbuf` (`[heads, max_seq, dim]`) at the position held in the
    /// device scalar `pos` (u32, `[1]`).
    pub fn kv_write(
        kbuf: &Tensor,
        vbuf: &Tensor,
        knew: &Tensor,
        vnew: &Tensor,
        pos: &Tensor,
    ) -> Result<()> {
        let (heads, max_seq, dim) = kbuf.dims3()?;
        let dev = match kbuf.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("cuda only"),
        };
        let dtype = kbuf.dtype();
        let kname = match dtype {
            DType::BF16 => "kv_write_bf16",
            DType::F32 => "kv_write_f32",
            dt => candle::bail!("kv_write: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_func(kname, &kernels::FUSED)?;
        let total = (heads * dim) as u32;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ks, ko) = cuda_parts(kbuf)?;
        let (vs, vo) = cuda_parts(vbuf)?;
        let (kns, kno) = cuda_parts(knew)?;
        let (vns, vno) = cuda_parts(vnew)?;
        let (ps, po) = cuda_parts(pos)?;
        let heads_i = heads as i32;
        let max_i = max_seq as i32;
        let dim_i = dim as i32;
        macro_rules! go {
            ($dt:ident) => {
                with_slice!(ks, ko, $dt, kb, {
                    with_slice!(vs, vo, $dt, vb, {
                        with_slice!(kns, kno, $dt, kn, {
                            with_slice!(vns, vno, $dt, vn, {
                                with_slice!(ps, po, U32, pp, {
                                    let mut b = func.builder();
                                    b.arg(&kb);
                                    b.arg(&vb);
                                    b.arg(&kn);
                                    b.arg(&vn);
                                    b.arg(&pp);
                                    candle::builder_arg!(b, heads_i, max_i, dim_i);
                                    unsafe { b.launch(cfg) }.w()?;
                                })
                            })
                        })
                    })
                })
            };
        }
        match dtype {
            DType::BF16 => go!(BF16),
            DType::F32 => go!(F32),
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Fill `mask` (`[max_seq]`) with the additive causal mask for the
    /// position in `pos`. `window == 0` means global causal; `> 0` sliding.
    pub fn mask_from_pos(mask: &Tensor, pos: &Tensor, window: usize) -> Result<()> {
        let max_seq = mask.elem_count();
        let dev = match mask.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("cuda only"),
        };
        let kname = match mask.dtype() {
            DType::BF16 => "mask_from_pos_bf16",
            DType::F32 => "mask_from_pos_f32",
            dt => candle::bail!("mask_from_pos: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_func(kname, &kernels::FUSED)?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((max_seq as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ms, mo) = cuda_parts(mask)?;
        let (ps, po) = cuda_parts(pos)?;
        let max_i = max_seq as i32;
        let win_i = window as i32;
        macro_rules! go {
            ($dt:ident) => {
                with_slice!(ms, mo, $dt, mb, {
                    with_slice!(ps, po, U32, pp, {
                        let mut b = func.builder();
                        b.arg(&mb);
                        b.arg(&pp);
                        candle::builder_arg!(b, max_i, win_i);
                        unsafe { b.launch(cfg) }.w()?;
                    })
                })
            };
        }
        match mask.dtype() {
            DType::BF16 => go!(BF16),
            DType::F32 => go!(F32),
            _ => unreachable!(),
        }
        Ok(())
    }

    /// `*pos += 1` on the device (single-thread kernel; graph-replayable).
    pub fn incr_u32(pos: &Tensor) -> Result<()> {
        let dev = match pos.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("cuda only"),
        };
        let func = dev.get_or_load_func("incr_u32", &kernels::FUSED)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ps, po) = cuda_parts(pos)?;
        with_slice!(ps, po, U32, pp, {
            let mut b = func.builder();
            b.arg(&pp);
            unsafe { b.launch(cfg) }.w()?;
        });
        Ok(())
    }

    /// Copy a [kv_heads, t, hd] bf16 chunk into the static KV buffer
    /// ([kv_heads, max_seq, hd]) at rows [pos, pos+t). Eager prefill path:
    /// `pos` is a host value (NOT graph-safe — decode keeps using kv_write
    /// with the device-resident position).
    pub fn kv_write_chunk(buf: &Tensor, src: &Tensor, pos: usize) -> Result<()> {
        let (kv_heads, max_seq, hd) = buf.dims3()?;
        let (h2, t, hd2) = src.dims3()?;
        if h2 != kv_heads || hd2 != hd {
            candle::bail!("kv_write_chunk: shape mismatch {:?} vs {:?}", buf.dims(), src.dims());
        }
        if pos + t > max_seq {
            candle::bail!("kv_write_chunk: pos {pos} + t {t} exceeds max_seq {max_seq}");
        }
        let dev = match buf.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("kv_write_chunk: cuda only"),
        };
        let kname = match buf.dtype() {
            DType::F32 => "kv_write_chunk_f32",
            _ => "kv_write_chunk_bf16",
        };
        let func = dev.get_or_load_func(kname, &kernels::FUSED)?;
        let cfg = LaunchConfig {
            grid_dim: (kv_heads as u32, t as u32, 1),
            block_dim: (hd.min(256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bs, bo) = cuda_parts(buf)?;
        let (ss, so) = cuda_parts(src)?;
        let (posu, ti, mi, hi) = (pos as u32, t as i32, max_seq as i32, hd as i32);
        match buf.dtype() {
            DType::BF16 => {
                with_slice!(bs, bo, BF16, bp, {
                    with_slice!(ss, so, BF16, sp, {
                        let mut b = func.builder();
                        b.arg(&bp);
                        b.arg(&sp);
                        b.arg(&posu);
                        b.arg(&ti);
                        b.arg(&mi);
                        b.arg(&hi);
                        unsafe { b.launch(cfg) }.w()?;
                    });
                });
            }
            DType::F32 => {
                with_slice!(bs, bo, F32, bp, {
                    with_slice!(ss, so, F32, sp, {
                        let mut b = func.builder();
                        b.arg(&bp);
                        b.arg(&sp);
                        b.arg(&posu);
                        b.arg(&ti);
                        b.arg(&mi);
                        b.arg(&hi);
                        unsafe { b.launch(cfg) }.w()?;
                    });
                });
            }
            dt => candle::bail!("kv_write_chunk: unsupported dtype {dt:?}"),
        }
        Ok(())
    }

    /// Run the whole Gated-DeltaNet recurrence for a layer in ONE kernel
    /// launch. `s` ([n_v, d_v, d_k] f32) is updated in place; returns
    /// `o` [t, n_v, d_v] f32. All inputs must be contiguous f32 on CUDA.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan(
        q: &Tensor,     // [t, n_k, d_k]
        k: &Tensor,     // [t, n_k, d_k]
        v: &Tensor,     // [t, n_v, d_v]
        beta: &Tensor,  // [t, n_v]
        decay: &Tensor, // [t, n_v]
        s: &Tensor,     // [n_v, d_v, d_k] (mutated)
    ) -> Result<Tensor> {
        let (t, n_k, d_k) = q.dims3()?;
        let (_, n_v, d_v) = v.dims3()?;
        let dev = match q.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("gdn_scan: cuda only"),
        };
        const ROWS: u32 = 64;
        let chunks = (d_v as u32).div_ceil(ROWS);
        let smem = (ROWS as usize * d_k + 2 * d_k) * std::mem::size_of::<f32>();
        if smem > 48 * 1024 {
            candle::bail!("gdn_scan: d_k={d_k} needs {smem}B smem (>48KB)");
        }
        let o = Tensor::zeros((t, n_v, d_v), DType::F32, q.device())?;
        let func = dev.get_or_load_func("gdn_scan_f32", &kernels::FUSED)?;
        let cfg = LaunchConfig {
            grid_dim: (n_v as u32 * chunks, 1, 1),
            block_dim: (ROWS, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        {
        let (qs, qo) = cuda_parts(q)?;
        let (ks, ko) = cuda_parts(k)?;
        let (vs, vo) = cuda_parts(v)?;
        let (bs, bo) = cuda_parts(beta)?;
        let (ds, do_) = cuda_parts(decay)?;
        let (ss, so) = cuda_parts(s)?;
        let (os, oo) = cuda_parts(&o)?;
        with_slice!(qs, qo, F32, qp, {
            with_slice!(ks, ko, F32, kp, {
                with_slice!(vs, vo, F32, vp, {
                    with_slice!(bs, bo, F32, bp, {
                        with_slice!(ds, do_, F32, dp, {
                            with_slice!(ss, so, F32, sp, {
                                with_slice!(os, oo, F32, op, {
                                    let (ti, nki, nvi, dki, dvi) = (
                                        t as i32,
                                        n_k as i32,
                                        n_v as i32,
                                        d_k as i32,
                                        d_v as i32,
                                    );
                                    let mut b = func.builder();
                                    b.arg(&qp);
                                    b.arg(&kp);
                                    b.arg(&vp);
                                    b.arg(&bp);
                                    b.arg(&dp);
                                    b.arg(&sp);
                                    b.arg(&op);
                                    b.arg(&ti);
                                    b.arg(&nki);
                                    b.arg(&nvi);
                                    b.arg(&dki);
                                    b.arg(&dvi);
                                    unsafe { b.launch(cfg) }.w()?;
                                });
                            });
                        });
                    });
                });
            });
        });
        }
        Ok(o)
    }

    /// Overwrite a device u32 scalar from the host (used OUTSIDE captured
    /// graphs: reset position, feed the sampled token id).
    pub fn write_u32(t: &Tensor, v: u32) -> Result<()> {
        // Write via a kernel taking the value as a launch parameter: unlike an
        // async h2d memcpy from a stack temporary, there is no host-memory
        // lifetime to race with, and the write is stream-ordered before any
        // subsequent work that reads it.
        let dev = match t.device() {
            candle::Device::Cuda(d) => d.clone(),
            _ => candle::bail!("cuda only"),
        };
        let func = dev.get_or_load_func("set_u32", &kernels::FUSED)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let (s, off) = cuda_parts(t)?;
        with_slice!(s, off, U32, slice, {
            let mut b = func.builder();
            b.arg(&slice);
            b.arg(&v);
            unsafe { b.launch(cfg) }.w()?;
        });
        Ok(())
    }

    /// Device-to-device copy of `src` into the preallocated `dst`
    /// (same dtype/element count). Graph-replayable.
    pub fn copy_into(dst: &Tensor, src: &Tensor) -> Result<()> {
        if dst.dtype() != src.dtype() || dst.elem_count() != src.elem_count() {
            candle::bail!("copy_into: mismatched tensors");
        }
        let dev_stream = match dst.device() {
            candle::Device::Cuda(d) => d.cuda_stream(),
            _ => candle::bail!("cuda only"),
        };
        let n = dst.elem_count();
        let (ds, doff) = cuda_parts(dst)?;
        let (ss, soff) = cuda_parts(src)?;
        macro_rules! go {
            ($dt:ident) => {
                with_slice!(ds, doff, $dt, db, {
                    with_slice!(ss, soff, $dt, sb, {
                        use candle::cuda_backend::cudarc::driver::DevicePtr;
                        let dv = db.slice(0..n);
                        let sv = sb.slice(0..n);
                        let (dp, _g1) = dv.device_ptr(&dev_stream);
                        let (sp, _g2) = sv.device_ptr(&dev_stream);
                        let bytes = n * dst.dtype().size_in_bytes();
                        unsafe {
                            candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async(
                                dp,
                                sp,
                                bytes,
                                dev_stream.cu_stream(),
                            )
                        }
                        .map_err(candle::Error::wrap)?;
                    })
                })
            };
        }
        match dst.dtype() {
            DType::BF16 => go!(BF16),
            DType::F32 => go!(F32),
            DType::U32 => go!(U32),
            dt => candle::bail!("copy_into: unsupported {dt:?}"),
        }
        Ok(())
    }
}
