//! baidu/Unlimited-OCR: DeepSeek-OCR-lineage document parsing model.
//! Text decoder in `text`; vision towers (SAM-ViT-B + CLIP-L) and the
//! R-SWA static path land in follow-up modules.
pub mod text;
pub use text::{TextModel, UnlimitedOcrTextConfig};
