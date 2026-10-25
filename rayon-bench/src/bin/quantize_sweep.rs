//! Load-time quantization cost: QTensor::quantize on weight-sized f32 tensors,
//! checked byte-for-byte against a serial reference.
use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::time::Instant;

fn serial_ref<T: GgmlType>(xs: &[f32]) -> Vec<u8> {
    let mut ys = vec![T::zeros(); xs.len() / T::BLCK_SIZE];
    T::from_float(xs, &mut ys);
    let bytes = unsafe {
        std::slice::from_raw_parts(ys.as_ptr() as *const u8, ys.len() * std::mem::size_of::<T>())
    };
    bytes.to_vec()
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    // A 1.5B-ish layer: gate+up+down at 1536x8960, plus a 7B-class 4096x14336.
    let shapes = [(8960usize, 1536usize), (4096, 14336)];
    for (n, k) in shapes {
        let w = Tensor::randn(0f32, 1.0, (n, k), &dev)?;
        let xs = w.flatten_all()?.to_vec1::<f32>()?;
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let _ = QTensor::quantize(&w, dtype)?; // warmup
            let reps = 3;
            let t = Instant::now();
            let mut q = None;
            for _ in 0..reps {
                q = Some(QTensor::quantize(&w, dtype)?);
            }
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            let got = q.unwrap().data()?.to_vec();
            let want = match dtype {
                GgmlDType::Q4K => serial_ref::<BlockQ4K>(&xs),
                GgmlDType::Q6K => serial_ref::<BlockQ6K>(&xs),
                GgmlDType::Q8_0 => serial_ref::<BlockQ8_0>(&xs),
                _ => unreachable!(),
            };
            let mparams = (n * k) as f64 / 1e6;
            println!(
                "quantize,{dtype:?},{n}x{k},{:.1}ms,{:.0}Mparam/s,identical={}",
                dt * 1e3,
                mparams / dt,
                got == want
            );
        }
    }
    Ok(())
}
