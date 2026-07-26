// Streaming q8_0 quantizer: one tensor resident at a time (fits small
// cgroup limits where tensor-tools' eager full-load OOMs). Mirrors
// tensor-tools' rules exactly: rank==2 && dim1 % block == 0 -> target
// dtype, else F32.
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (inp, out) = (&args[1], &args[2]);
    let dtype = GgmlDType::Q8_0;
    let block = dtype.block_size();
    let st = unsafe { candle::safetensors::MmapedSafetensors::new(inp)? };
    let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let t = st.load(name, &Device::Cpu)?.to_dtype(candle::DType::F32)?;
        let should_q = t.rank() == 2 && t.dim(1)? % block == 0;
        let q = if should_q {
            QTensor::quantize(&t, dtype)?
        } else {
            QTensor::quantize(&t, GgmlDType::F32)?
        };
        if i % 50 == 0 {
            eprintln!("[{i}/{}] {name} q={should_q}", names.len());
        }
        qtensors.push((name.clone(), q));
    }
    let mut f = std::fs::File::create(out)?;
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    gguf_file::write(&mut f, &[], &refs)?;
    eprintln!("wrote {out} ({} tensors)", refs.len());
    Ok(())
}
