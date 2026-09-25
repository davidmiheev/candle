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

    // Mode "alt": alternate two weight shapes eagerly at b and compare the
    // second shape's result against its isolated run — detects cross-call
    // workspace contamination (stream-k fixup residue).
    if args.get(1).map(|s| s.as_str()) == Some("alt") {
        let b: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(512);
        let k = 5376usize;
        let (n1, n2) = (8192usize, 4096usize);
        fn f(i: usize, s: f32) -> f32 {
            let x = (i as f32) * s;
            (x.sin() * 43758.547).fract() - 0.5
        }
        let mk = |n: usize, seed: f32| -> anyhow::Result<QMatMul> {
            let w: Vec<f32> = (0..n * k).map(|i| f(i, seed)).collect();
            let w = Tensor::from_vec(w, (n, k), &Device::Cpu)?;
            Ok(QMatMul::QTensor(std::sync::Arc::new(
                QTensor::quantize_onto(&w, GgmlDType::Q4K, &dev)?,
            )))
        };
        let qm1 = mk(n1, 0.7311)?;
        let qm2 = mk(n2, 0.4177)?;
        let x: Vec<f32> = (0..b * k).map(|i| f(i, 1.309)).collect();
        let x = Tensor::from_vec(x, (1, b, k), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;
        // Isolated qm2 result (fresh process state for qm2's shape).
        let y2_iso = qm2.forward(&x)?;
        dev.synchronize()?;
        let y2_iso = (&y2_iso * 1.0)?;
        // Interleaved: qm1 then qm2, repeatedly; compare qm2 each round.
        for round in 0..4 {
            let _y1 = qm1.forward(&x)?;
            let y2 = qm2.forward(&x)?;
            dev.synchronize()?;
            let (rel, rms, nan) = max_rel_err(&y2_iso, &y2)?;
            println!(
                "alt b={b} round={round}: qm2 vs isolated max_rel={rel:.4} rms={rms:.2e} nan={nan}"
            );
        }
        return Ok(());
    }

    // Mode "chain": capture TWO chained quantized matmuls in one graph and
    // replay — the minimal multi-op composition (prod chunk-step analog).
    if args.get(1).map(|s| s.as_str()) == Some("chain") {
        let b: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(512);
        let k = 5376usize;
        let mid = 4096usize;
        fn f(i: usize, s: f32) -> f32 {
            let x = (i as f32) * s;
            (x.sin() * 43758.547).fract() - 0.5
        }
        let mk = |n_out: usize, n_in: usize, seed: f32| -> anyhow::Result<QMatMul> {
            let w: Vec<f32> = (0..n_out * n_in).map(|i| f(i, seed) * 0.05).collect();
            let w = Tensor::from_vec(w, (n_out, n_in), &Device::Cpu)?;
            Ok(QMatMul::QTensor(std::sync::Arc::new(
                QTensor::quantize_onto(&w, GgmlDType::Q4K, &dev)?,
            )))
        };
        let qm1 = mk(mid, k, 0.7311)?;
        let qm2 = mk(k, mid, 0.4177)?;
        let x: Vec<f32> = (0..b * k).map(|i| f(i, 1.309)).collect();
        let x = Tensor::from_vec(x, (1, b, k), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;
        let chain = |inp: &Tensor| -> candle::Result<Tensor> {
            let y1 = qm1.forward(inp)?;
            let y1 = y1.to_dtype(DType::BF16)?;
            qm2.forward(&y1)
        };
        let y_eager = chain(&x)?;
        dev.synchronize()?;
        let y_eager = (&y_eager * 1.0)?;
        // Prod parity: the htod param cache guard is active during prod
        // capture (generate_static_gemma4) — CHAIN_HTOD=1 enables it here.
        let _guard = if std::env::var("CHAIN_HTOD")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            println!("htod param cache: ENABLED");
            Some(cuda.enable_cuda_graph_htod_cache())
        } else {
            None
        };
        // warmup then capture
        let _w = chain(&x)?;
        dev.synchronize()?;
        let mut cap: Option<Tensor> = None;
        let g = CapturedGraph::capture(&cuda, || {
            cap = Some(chain(&x)?);
            Ok(())
        })?;
        let y_buf = cap.unwrap();
        for r in 0..3 {
            g.replay()?;
            dev.synchronize()?;
            let y = (&y_buf * 1.0)?;
            let (rel, rms, nan) = max_rel_err(&y_eager, &y)?;
            println!("chain b={b} replay{r}: vs eager max_rel={rel:.4} rms={rms:.2e} nan={nan}");
        }
        std::mem::forget(g);
        std::mem::forget(y_buf);
        return Ok(());
    }

    let batches: Vec<usize> = args
        .get(4)
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 256, 512]);
    for &b in &batches {
        let cell = || -> anyhow::Result<()> {
            eprintln!("[cell b={b}] input");
            let scale: f32 = std::env::var("BISECT_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0);
            let x_host: Vec<f32> = (0..b * k).map(|i| f(i, 1.309) * scale).collect();
            let x = Tensor::from_vec(x_host, (1, b, k), &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .to_device(&dev)?;

            // Reference: legacy DMMV path on identical input.
            eprintln!("[cell b={b}] dmmv ref");
            candle::quantized::cuda::set_force_dmmv(true);
            let y_ref = qm.forward(&x)?;
            dev.synchronize()?;
            candle::quantized::cuda::set_force_dmmv(false);

            // Eager fast path.
            eprintln!("[cell b={b}] eager fast");
            let y_eager = qm.forward(&x)?;
            dev.synchronize()?;
            let (rel_e, rms_e, nan_e) = max_rel_err(&y_ref, &y_eager)?;

            // Captured fast path: warmup once (sizes workspaces), then capture
            // one forward into stable in/out buffers and replay twice.
            eprintln!("[cell b={b}] warm+capture");
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
            // Dropping the graph frees its capture-time (graph-owned) allocations;
            // the output tensor then double-frees -> INVALID_VALUE on the next
            // stream op (reproduced: cells following a drop crashed at input
            // upload). Leak both — this is a bisect bin.
            std::mem::forget(g);
            std::mem::forget(y_buf);
            dev.synchronize()?;

            println!(
                "b={b:4} eager: max_rel={rel_e:.4} rms={rms_e:.2e} nan={nan_e} | \
             replay1: max_rel={rel_c1:.4} rms={rms_c1:.2e} nan={nan_c1} | \
             replay2: max_rel={rel_c2:.4} rms={rms_c2:.2e} nan={nan_c2}"
            );
            Ok(())
        };
        if let Err(e) = cell() {
            println!("b={b:4} CELL-ERROR: {e}");
        }
    }
    Ok(())
}
