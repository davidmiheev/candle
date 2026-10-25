# rayon-bench

CPU threading experiments for candle quantized inference (ignite `candle-llm-gguf` path).
Standalone crate (own `[workspace]`), not part of the candle build.

Binaries (one process per thread setting, because candle sizes its pools once per process):

| bin | what it measures |
|-----|------------------|
| `qmatmul_sweep` | `QMatMul::forward` at Qwen2.5-1.5B shapes, Q4K/Q6K/Q8_0, m = 1/16/128/512 |
| `llm_sweep` | quantized Qwen2 GGUF: load, prefill tok/s, greedy decode tok/s (`GGUF`, `PROMPT_TOKENS`, `DECODE_TOKENS`, `WITH_CONTEXT=1`) |
| `quantize_sweep` | `QTensor::quantize` throughput at load-time sizes, byte-checked against serial `from_float` |

Variants built by the commands below; `sweep.sh` iterates `RAYON_NUM_THREADS = CANDLE_NUM_THREADS = 1 2 4 8`.

```bash
# candle fork (this checkout)
cargo build --release                                                  # fork-base
RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir target-native   # fork-native / fork-ctx
# candle 0.10.2 from crates.io (what ignite-catalog pins today)
(cd v0102 && RUSTFLAGS="--cfg candle_0102" cargo build --release)
(cd v0102 && RUSTFLAGS="--cfg candle_0102 -C target-cpu=native" cargo build --release --target-dir target-native)
# Anthropic rayon-core fork (adaptive idle spinning)
git -C ../../rayon worktree add ../rayon-anthropic rayon-core-v1.13.5000
(cd anth && RUSTFLAGS="-C target-cpu=native" cargo build --release)

GGUF=qwen2.5-1.5b-instruct-q4_k_m.gguf PROMPT_TOKENS=256 DECODE_TOKENS=32 ./sweep.sh llm > results/llm.csv
./sweep.sh quantize > results/quantize.csv
REPS=1 ./sweep.sh qmatmul > results/qmatmul.csv
```

Notes
- candle >= 0.11 runs quantized matmul on its own barrier pool sized by `CANDLE_NUM_THREADS`;
  `RAYON_NUM_THREADS` only sizes rayon (gemm, softmax, etc.). 0.10.2 uses rayon for both.
- Quantized kernels are selected with `cfg(target_feature)`, so `RUSTFLAGS` decides whether AVX2/AVX-512 paths exist at all.
