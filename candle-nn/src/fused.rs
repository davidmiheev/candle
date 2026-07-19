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
