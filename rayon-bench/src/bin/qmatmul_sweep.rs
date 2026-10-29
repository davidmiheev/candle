//! Quantized matmul microbenchmark at Qwen2.5-1.5B layer shapes.
//! Thread count comes from the environment (RAYON_NUM_THREADS / CANDLE_NUM_THREADS),
//! so run one process per setting (see ../sweep.sh).
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

fn bench(dtype: GgmlDType, m: usize, k: usize, n: usize, label: &str) -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 1.0, (n, k), &dev)?;
    let q = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
    let x = Tensor::randn(0f32, 1.0, (1, m, k), &dev)?;
    // warmup (also triggers any lazy repacking)
    for _ in 0..3 {
        q.forward(&x)?;
    }
    let flops = 2.0 * (m * k * n) as f64;
    let target = 0.6; // seconds per case
    let t0 = Instant::now();
    let mut iters = 0usize;
    while t0.elapsed().as_secs_f64() < target || iters < 3 {
        q.forward(&x)?;
        iters += 1;
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "qmatmul,{dtype:?},{label},{m},{k},{n},{:.4},{:.2}",
        dt * 1e3,
        flops / dt / 1e9
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // Qwen2.5-1.5B: hidden 1536, intermediate 8960, kv dim 256.
    let shapes = [
        ("q_proj", 1536, 1536),
        ("kv_proj", 1536, 256),
        ("gate_up", 1536, 8960),
        ("down", 8960, 1536),
    ];
    let ms: Vec<usize> = std::env::var("BENCH_M")
        .unwrap_or_else(|_| "1,16,128,512".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let dtypes = [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0];
    for &dtype in &dtypes {
        for &m in &ms {
            for (label, k, n) in shapes {
                bench(dtype, m, k, n, label)?;
            }
        }
    }
    Ok(())
}
