//! Qwen3.5-family decoders (Qwen3.6, Qwen3.8): a hybrid stack of Gated-DeltaNet
//! linear-attention layers with full attention every `full_attention_interval`.
//!
//! `candle-transformers` has had `qwen3_5.rs` for a while with no way to run it.
//! This is that binary. Quantization is not optional at these sizes: a 27B is
//! 55.6 GB in bf16, so `--quant q4k` is the difference between running the model
//! and reading its config.
//!
//! ```bash
//! cargo run --release --features cuda --example qwen3_5 -- \
//!     --model-id Qwen/Qwen3.6-27B --quant q4k --dtype f16 \
//!     --prompt "Why does grouped-query attention shrink the KV cache?"
//! ```
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{quantized::GgmlDType, DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen3_5::{Model, Qwen35TextConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "Qwen/Qwen3.6-27B")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(
        long,
        default_value = "Explain in two sentences why grouped-query attention reduces KV cache size."
    )]
    prompt: String,

    /// Wrap the prompt in Qwen's chat template. Instruction-tuned checkpoints
    /// need this; without it they continue the text instead of answering.
    #[arg(long, default_value_t = true)]
    chat_template: bool,

    #[arg(long, short = 'n', default_value_t = 64)]
    sample_len: usize,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Quantize every large projection at load: q8_0, q6k, q5k, q4k, q4_0,
    /// q3k or q2k. Without it a 27B needs 55.6 GB of VRAM.
    #[arg(long)]
    quant: Option<String>,

    /// f16, bf16 or f32. candle only builds its bf16 CUDA kernels for sm_80
    /// and newer, so pass f16 on older cards (a T4 is sm_75).
    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn parse_quant(name: &str) -> Result<GgmlDType> {
    Ok(match name {
        "q8_0" => GgmlDType::Q8_0,
        "q6k" => GgmlDType::Q6K,
        "q5k" => GgmlDType::Q5K,
        "q4k" => GgmlDType::Q4K,
        "q4_0" => GgmlDType::Q4_0,
        "q3k" => GgmlDType::Q3K,
        "q2k" => GgmlDType::Q2K,
        other => anyhow::bail!("unknown --quant {other}; expected q8_0 q6k q5k q4k q4_0 q3k q2k"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(other) => anyhow::bail!("unknown --dtype {other}; expected f16, bf16 or f32"),
        None if device.is_cuda() => DType::BF16,
        None => DType::F32,
    };
    let quant = args.quant.as_deref().map(parse_quant).transpose()?;

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;
    let weights = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!(
        "retrieved {} weight files in {:?}",
        weights.len(),
        start.elapsed()
    );

    // A multimodal checkpoint keeps the decoder's own config under text_config.
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_file)?)?;
    let config: Qwen35TextConfig = match raw.get("text_config") {
        Some(text_config) => serde_json::from_value(text_config.clone())?,
        None => serde_json::from_value(raw)?,
    };
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let mut model = Model::new_with_quant(&config, vb, quant)?;
    println!("loaded the model in {:?}", start.elapsed());

    let prompt = if args.chat_template {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    } else {
        args.prompt.clone()
    };

    let mut stream = TokenOutputStream::new(tokenizer);
    let mut tokens = stream
        .tokenizer()
        .encode(prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let eos = stream
        .tokenizer()
        .token_to_id("<|im_end|>")
        .or_else(|| stream.tokenizer().token_to_id("<|endoftext|>"));

    let mut logits_processor = if args.temperature < 1e-7 {
        LogitsProcessor::from_sampling(args.seed, Sampling::ArgMax)
    } else {
        LogitsProcessor::new(args.seed, Some(args.temperature), None)
    };

    let started = std::time::Instant::now();
    let mut generated = 0usize;
    let mut start_pos = 0usize;
    for index in 0..args.sample_len {
        let context = if index == 0 { tokens.len() } else { 1 };
        let input = Tensor::new(&tokens[tokens.len() - context..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if args.repeat_penalty == 1.0 {
            logits
        } else {
            let from = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[from..],
            )?
        };
        start_pos += context;
        let next = logits_processor.sample(&logits)?;
        if Some(next) == eos {
            break;
        }
        tokens.push(next);
        generated += 1;
        if let Some(text) = stream.next_token(next)? {
            print!("{text}");
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = stream.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let elapsed = started.elapsed();
    println!();
    println!(
        "{generated} tokens generated ({:.2} token/s)",
        generated as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}
