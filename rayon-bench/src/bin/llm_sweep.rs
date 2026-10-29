//! End-to-end quantized Qwen2 GGUF: load time, prefill tok/s, decode tok/s.
//! Mirrors ignite's candle-llm-gguf loop (full-prompt prefill, then 1 token/step).
//! Env: GGUF, TOKENIZER, PROMPT_TOKENS (default 512), DECODE_TOKENS (default 32),
//! WITH_CONTEXT=1 wraps the run in Device::with_context (candle >= 0.11 only).
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn run() -> anyhow::Result<()> {
    let gguf = std::env::var("GGUF")?;
    let prompt_tokens = env_usize("PROMPT_TOKENS", 512);
    let decode_tokens = env_usize("DECODE_TOKENS", 32);
    let dev = Device::Cpu;

    let t = Instant::now();
    let mut f = std::fs::File::open(&gguf)?;
    let content = gguf_file::Content::read(&mut f)?;
    // ARCH=llama loads with quantized_llama, as ignite's candle-llm-gguf does for non-gemma families.
    if std::env::var("ARCH").as_deref() == Ok("llama") {
        let r = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(content, &mut f, &dev);
        println!("quantized_llama load: {}", match r { Ok(_) => "ok".to_string(), Err(e) => format!("ERROR: {e}") });
        return Ok(());
    }
    let mut model = ModelWeights::from_gguf(content, &mut f, &dev)?;
    let load_s = t.elapsed().as_secs_f64();

    // Deterministic synthetic prompt (token ids in the normal vocab range).
    let tokens: Vec<u32> = (0..prompt_tokens as u32).map(|i| 1000 + (i * 37) % 30000).collect();

    let t = Instant::now();
    let input = Tensor::new(tokens.as_slice(), &dev)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let mut next = logits.squeeze(0)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
    let prefill_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for i in 0..decode_tokens {
        let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens + i)?;
        next = logits.squeeze(0)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
    }
    let decode_s = t.elapsed().as_secs_f64();
    println!(
        "llm,load_s={load_s:.2},prefill_tok_s={:.1},decode_tok_s={:.2},last_token={next}",
        prompt_tokens as f64 / prefill_s,
        decode_tokens as f64 / decode_s
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    #[cfg(not(candle_0102))]
    if std::env::var("WITH_CONTEXT").as_deref() == Ok("1") {
        return Device::Cpu.with_context(run);
    }
    run()
}
