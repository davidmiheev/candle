//! Unlimited-OCR: document OCR with a DeepEncoder vision tower and a
//! DeepSeek-V2-style MoE text tower.
//!
//! The preprocessing here is not incidental — it is the model's contract.
//! A global view is padded to a 1024 square on mean-gray; images larger than
//! 640 on a side additionally get an InternVL-style tile grid chosen by aspect
//! ratio. Get that wrong and the model reads a different picture than the one
//! you meant, usually without saying so.
//!
//! ```bash
//! cargo run --release --features cuda --example unlimited-ocr -- \
//!     --image page.png --dtype f16
//! ```
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::unlimited_ocr::{DeepEncoder, TextModel, UnlimitedOcrTextConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const IMAGE_TOKEN_ID: u32 = 128815;
const EOS: u32 = 1;
/// Sliding-window no-repeat-ngram suppression, matching the reference
/// processor. Without it the model loops on repetitive page furniture.
const NGRAM_SIZE: usize = 35;
const NGRAM_WINDOW: usize = 128;

#[derive(Parser)]
struct Args {
    /// Document image(s) to read.
    #[arg(long)]
    image: Vec<String>,

    /// ocr | table | chart | formula
    #[arg(long, default_value = "ocr")]
    task: String,

    #[arg(long, default_value_t = 512)]
    max_length: usize,

    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// f16, bf16 or f32. candle only builds its bf16 CUDA kernels for sm_80
    /// and newer, so pass f16 on older cards (a T4 is sm_75).
    #[arg(long)]
    dtype: Option<String>,

    /// Use only the 1024 global view, skipping the tile grid. This is the
    /// reference default for images under 640px and the path the catalog's
    /// executor ships; the tiled path costs ~9x the image tokens.
    #[arg(long)]
    no_crops: bool,

    #[arg(long, default_value = "baidu/Unlimited-OCR")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,
}

/// InternVL-style grid search: 2..=32 tiles, closest aspect ratio wins, and on
/// a tie prefer more tiles when the source is big enough to fill them.
fn pick_crop_ratio(w: u32, h: u32) -> (usize, usize) {
    if w <= 640 && h <= 640 {
        return (1, 1);
    }
    let aspect = f64::from(w) / f64::from(h);
    let area = f64::from(w) * f64::from(h);
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for n in 2..=32usize {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= 32 && i * j >= 2 && !ratios.contains(&(i, j)) {
                    ratios.push((i, j));
                }
            }
        }
    }
    ratios.sort_by_key(|r| r.0 * r.1);
    let (mut best, mut best_diff) = ((1usize, 1usize), f64::INFINITY);
    for r in ratios {
        let diff = (aspect - r.0 as f64 / r.1 as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = r;
        } else if diff == best_diff && area > 0.5 * 640.0 * 640.0 * (r.0 * r.1) as f64 {
            best = r;
        }
    }
    best
}

fn to_chw_norm(
    img: &image::RgbImage,
    side: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; 3 * side * side];
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            data[c * side * side + (y as usize) * side + x as usize] =
                (f32::from(p.0[c]) / 255.0 - 0.5) / 0.5;
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, side, side), device)?.to_dtype(dtype)?)
}

/// Pad to a 1024 square on mean-gray, normalized to mean = std = 0.5.
/// CatmullRom is bicubic(a = -0.5), the closest match to PIL's resampler.
fn preprocess(img: &image::RgbImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let (w, h) = img.dimensions();
    let scale = (1024.0 / f64::from(w)).min(1024.0 / f64::from(h));
    let nw = ((f64::from(w) * scale).round() as u32).max(1);
    let nh = ((f64::from(h) * scale).round() as u32).max(1);
    let resized = image::imageops::resize(img, nw, nh, image::imageops::FilterType::CatmullRom);
    let mut canvas = image::RgbImage::from_pixel(1024, 1024, image::Rgb([127, 127, 127]));
    image::imageops::overlay(
        &mut canvas,
        &resized,
        i64::from((1024 - nw) / 2),
        i64::from((1024 - nh) / 2),
    );
    to_chw_norm(&canvas, 1024, device, dtype)
}

fn task_prompt(task: &str) -> &'static str {
    match task {
        "table" | "chart" | "formula" => "\n<|grounding|>Convert the document to markdown.",
        _ => "\nFree OCR.",
    }
}

fn banned_tokens(generated: &[u32]) -> Vec<u32> {
    if generated.len() < NGRAM_SIZE {
        return Vec::new();
    }
    let search_start = generated.len().saturating_sub(NGRAM_WINDOW);
    let search_end = generated.len() + 1 - NGRAM_SIZE;
    if search_end <= search_start {
        return Vec::new();
    }
    let prefix = &generated[generated.len() + 1 - NGRAM_SIZE..];
    let mut banned = Vec::new();
    for idx in search_start..search_end {
        if generated[idx..idx + NGRAM_SIZE - 1] == *prefix {
            banned.push(generated[idx + NGRAM_SIZE - 1]);
        }
    }
    banned
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.image.is_empty() {
        anyhow::bail!("pass at least one --image");
    }
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(other) => anyhow::bail!("unknown --dtype {other}; expected f16, bf16 or f32"),
        None if device.is_cuda() => DType::BF16,
        None => DType::F32,
    };

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;
    let weights = vec![repo.get("model-00001-of-000001.safetensors")?];
    println!("retrieved the files in {:?}", start.elapsed());

    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_file)?)?;
    let text_config: UnlimitedOcrTextConfig =
        serde_json::from_value(raw["language_config"].clone())?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let encoder = DeepEncoder::new(vb.clone())?;
    let mut model = TextModel::new(&text_config, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    for path in &args.image {
        let decoded = image::ImageReader::open(path)?.decode()?.to_rgb8();
        let (iw, ih) = decoded.dimensions();
        let (wc, hc) = if args.no_crops {
            (1, 1)
        } else {
            pick_crop_ratio(iw, ih)
        };
        let global = preprocess(&decoded, &device, dtype)?;
        let image_embeddings = if wc > 1 || hc > 1 {
            // Distort-resize to the tile grid, then cut it row-major.
            let resized = image::imageops::resize(
                &decoded,
                640 * wc as u32,
                640 * hc as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let mut tiles = Vec::with_capacity(wc * hc);
            for t in 0..(wc * hc) {
                let bx = ((t % wc) as u32) * 640;
                let by = ((t / wc) as u32) * 640;
                let tile = image::imageops::crop_imm(&resized, bx, by, 640, 640).to_image();
                tiles.push(to_chw_norm(&tile, 640, &device, dtype)?);
            }
            encoder.forward_crops(&global, &Tensor::cat(&tiles, 0)?, wc, hc)?
        } else {
            encoder.forward(&global)?
        };
        let n_image = image_embeddings.dim(0)?;

        let mut tail: Vec<u32> = tokenizer
            .encode(task_prompt(&args.task), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        // The reference text_encode drops a trailing standalone-space token.
        if tail.last() == Some(&223) {
            tail.pop();
        }
        let mut ids: Vec<u32> = vec![0];
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_image));
        ids.extend(&tail);
        let seq = ids.len();
        println!("{path}: {iw}x{ih}, crops {wc}x{hc}, {n_image} image tokens, prompt {seq}");

        model.clear_kv_cache();
        let embedded = model.embed(&Tensor::from_vec(ids, (1, seq), &device)?)?;
        // Splice the vision embeddings over the image-token placeholders.
        let spliced = Tensor::cat(
            &[
                embedded.i((.., ..1, ..))?,
                image_embeddings.unsqueeze(0)?.to_dtype(embedded.dtype())?,
                embedded.i((.., 1 + n_image.., ..))?,
            ],
            1,
        )?;

        let started = std::time::Instant::now();
        let mut logits = model.forward_embeds(&spliced, 0)?;
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..args.max_length {
            let mut values: Vec<f32> = logits.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
            for token in banned_tokens(&generated) {
                values[token as usize] = f32::NEG_INFINITY;
            }
            let mut best = 0usize;
            for i in 1..values.len() {
                if values[i] > values[best] {
                    best = i;
                }
            }
            let next = best as u32;
            if next == EOS {
                break;
            }
            generated.push(next);
            logits = model.forward(&Tensor::from_vec(vec![next], (1, 1), &device)?, seq + step)?;
        }
        let elapsed = started.elapsed();
        let text = tokenizer.decode(&generated, false).map_err(E::msg)?;
        println!("{:-<60}", "");
        println!("{text}");
        println!("{:-<60}", "");
        println!(
            "{} tokens in {:.2}s ({:.1} tok/s)",
            generated.len(),
            elapsed.as_secs_f64(),
            generated.len() as f64 / elapsed.as_secs_f64()
        );
    }
    Ok(())
}
