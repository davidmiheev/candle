// Batch-4 kernel bisect: quantized matmul correctness, eager vs CUDA-graph
// replay vs legacy-DMMV reference, across batch sizes — isolates the
// in-capture NaN and the MMQ-512 numerics bug without model loads.
//
// Usage: capture-bisect [q4k|q8_0] [K] [N]
// Matrix of cells: batch in {1, 8, 256, 512} × {eager, captured} × dtype,
// each compared against the FORCE_DMMV legacy path on identical inputs.
use candle::cuda_backend::graph::CapturedGraph;
use candle::quantized::{GgmlDType, QMatMul, QTensor};
use candle::Module;
use candle::{DType, Device, Tensor};

fn max_rel_err(a: &Tensor, b: &Tensor) -> candle::Result<(f32, f32, usize)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_rel = 0f32;
    let mut rms_num = 0f64;
    let mut rms_den = 0f64;
    let mut nans = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if !y.is_finite() {
            nans += 1;
            continue;
        }
        let d = (x - y).abs();
        rms_num += (d * d) as f64;
        rms_den += (x * x) as f64;
        let rel = d / x.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let rms = (rms_num.sqrt() / rms_den.sqrt().max(1e-12)) as f32;
    Ok((max_rel, rms, nans))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dt = match args.get(1).map(|s| s.as_str()).unwrap_or("q4k") {
        "q8_0" | "q8" => GgmlDType::Q8_0,
        _ => GgmlDType::Q4K,
    };
    // Default dims: 31b sliding q_proj (in 5376 -> out 8192).
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5376);
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8192);

    let dev = Device::new_cuda(0)?;
    let cuda = match &dev {
        Device::Cuda(c) => c.clone(),
        _ => unreachable!(),
    };
    println!("dtype={dt:?} K={k} N={n} (weight [{n},{k}])");

    // Deterministic pseudo-random weight/inputs (no rand dep): sin-hash.
    fn f(i: usize, s: f32) -> f32 {
        let x = (i as f32) * s;
        (x.sin() * 43758.547).fract() - 0.5
    }
    let w_host: Vec<f32> = (0..n * k).map(|i| f(i, 0.7311)).collect();
    let w = Tensor::from_vec(w_host, (n, k), &Device::Cpu)?;
    let qw = QTensor::quantize(&w, dt)?;
    let qw_dev = {
        // quantize_onto keeps the CPU quantize + device upload path used in prod
        QTensor::quantize_onto(&w, dt, &dev)?
    };
    let qm = QMatMul::QTensor(std::sync::Arc::new(qw_dev));
    let _ = qw;

    for &b in &[1usize, 8, 256, 512] {
        let x_host: Vec<f32> = (0..b * k).map(|i| f(i, 1.309)).collect();
        let x = Tensor::from_vec(x_host, (1, b, k), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;

        // Reference: legacy DMMV path on identical input.
        candle::quantized::cuda::set_force_dmmv(true);
        let y_ref = qm.forward(&x)?;
        dev.synchronize()?;
        candle::quantized::cuda::set_force_dmmv(false);

        // Eager fast path.
        let y_eager = qm.forward(&x)?;
        dev.synchronize()?;
        let (rel_e, rms_e, nan_e) = max_rel_err(&y_ref, &y_eager)?;

        // Captured fast path: warmup once (sizes workspaces), then capture
        // one forward into stable in/out buffers and replay twice.
        let x_buf = x.zeros_like()?;
        x_buf.slice_set(&x, 0, 0)?;
        let _warm = qm.forward(&x_buf)?;
        dev.synchronize()?;
        let mut cap_out: Option<Tensor> = None;
        let g = CapturedGraph::capture(&cuda, || {
            cap_out = Some(qm.forward(&x_buf)?);
            Ok(())
        })?;
        let y_buf = cap_out.unwrap();
        g.replay()?;
        dev.synchronize()?;
        let y_replay1 = (&y_buf * 1.0)?; // detach copy
        g.replay()?;
        dev.synchronize()?;
        let y_replay2 = (&y_buf * 1.0)?;
        let (rel_c1, rms_c1, nan_c1) = max_rel_err(&y_ref, &y_replay1)?;
        let (rel_c2, rms_c2, nan_c2) = max_rel_err(&y_ref, &y_replay2)?;
        drop(g);
        dev.synchronize()?;

        println!(
            "b={b:4} eager: max_rel={rel_e:.4} rms={rms_e:.2e} nan={nan_e} | \
             replay1: max_rel={rel_c1:.4} rms={rms_c1:.2e} nan={nan_c1} | \
             replay2: max_rel={rel_c2:.4} rms={rms_c2:.2e} nan={nan_c2}"
        );
    }
    Ok(())
}
