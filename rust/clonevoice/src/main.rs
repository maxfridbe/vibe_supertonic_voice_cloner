//! clonevoice — turn a recording into a Supertonic style JSON, the same way the
//! Android app does it: two ONNX forward passes, no training, no autograd.
//!
//!   clonevoice <input.wav> <output.json> [--models <dir>] [--name <voice name>]
//!
//! Pipeline:  wav → spk_encoder.onnx → 192-d speaker embedding
//!                → style_encoder.onnx → (style_ttl [1,50,256], style_dp [1,8,16])
//!                → JSON that Supertonic / TTS Runner plays directly.

use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

mod refine;

const RATE: u32 = 16_000; // the ECAPA speaker encoder eats 16 kHz mono
const MAX_SECS: usize = 12; // identity saturates after a few seconds

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut positional = Vec::new();
    let mut models = PathBuf::from("../../models");
    let mut name = String::new();
    let mut supertonic: Option<PathBuf> = None;
    let mut do_refine = false;
    let mut iters = 120usize;
    let mut pop = 20usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--models" => { models = PathBuf::from(&args[i + 1]); i += 2; }
            "--name" => { name = args[i + 1].clone(); i += 2; }
            "--supertonic" => { supertonic = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--refine" => { do_refine = true; i += 1; }
            "--iters" => { iters = args[i + 1].parse().unwrap_or(120); i += 2; }
            "--pop" => { pop = args[i + 1].parse().unwrap_or(20); i += 2; }
            other => { positional.push(other.to_string()); i += 1; }
        }
    }
    if positional.len() < 2 {
        eprintln!("usage: clonevoice <input.wav> <output.json> [--models <dir>] [--name <voice>]");
        std::process::exit(2);
    }
    let (input, output) = (&positional[0], &positional[1]);
    if name.is_empty() {
        name = Path::new(input).file_stem().and_then(|s| s.to_str()).unwrap_or("cloned").to_string();
    }

    let audio = read_wav_mono_16k(input)?;
    let n = audio.len().min(RATE as usize * MAX_SECS);
    let audio = &audio[..n];

    // 1) recording → speaker embedding
    let mut spk = Session::builder()?.commit_from_file(models.join("spk_encoder.onnx"))?;
    let wav_t = Tensor::from_array(([1_i64, n as i64], audio.to_vec()))?;
    let spk_out = spk.run(ort::inputs!["wav" => wav_t])?;
    let emb: Vec<f32> = spk_out[0].try_extract_tensor::<f32>()?.1.to_vec();
    eprintln!("speaker embedding: {} dims", emb.len());

    // 2) embedding → style tensors
    let mut style = Session::builder()?.commit_from_file(models.join("style_encoder.onnx"))?;
    let emb_t = Tensor::from_array(([1_i64, emb.len() as i64], emb))?;
    let style_out = style.run(ort::inputs!["embedding" => emb_t])?;
    let mut ttl: Vec<f32> = style_out[0].try_extract_tensor::<f32>()?.1.to_vec();
    let mut dp: Vec<f32> = style_out[1].try_extract_tensor::<f32>()?.1.to_vec();
    if ttl.len() != 50 * 256 || dp.len() != 8 * 16 {
        return Err(format!("unexpected style shape: ttl={} dp={}", ttl.len(), dp.len()).into());
    }

    // 2b) optional refine: search the style basis to close the encoder's gap
    let mut source = "clonevoice (encoder path)";
    if do_refine {
        let sup = supertonic.as_ref().ok_or("--refine requires --supertonic <dir>")?;
        let r = refine::refine(sup, &models, audio, &ttl, &dp, iters, pop)?;
        eprintln!("refined: {:.3} -> {:.3} over {} evals", r.start_cos, r.end_cos, r.evals);
        ttl = r.ttl; dp = r.dp;
        source = "clonevoice (refined)";
    }

    // 3) write the style JSON Supertonic plays
    let json = serde_json::json!({
        "style_ttl": { "dims": [1, 50, 256], "data": ttl },
        "style_dp":  { "dims": [1, 8, 16],   "data": dp },
        "metadata":  { "name": name, "source": source, "reference": input }
    });
    std::fs::write(output, serde_json::to_string(&json)?)?;
    eprintln!("wrote {output}");
    // ONNX Runtime (loaded dynamically) can fault while tearing itself down at
    // process exit; the work is done, so exit now and let the OS reclaim it
    // rather than run the sessions' destructors.
    std::io::Write::flush(&mut std::io::stderr()).ok();
    std::process::exit(0);
}

/// 16-bit PCM WAV → mono f32, linearly resampled to 16 kHz.
fn read_wav_mono_16k(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    // downmix to mono
    let frames = raw.len() / ch;
    let mut mono = vec![0f32; frames];
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..ch { acc += raw[i * ch + c]; }
        mono[i] = acc / ch as f32;
    }
    if spec.sample_rate == RATE || frames <= 1 {
        return Ok(mono);
    }
    // linear resample
    let out_n = (frames as u64 * RATE as u64 / spec.sample_rate as u64) as usize;
    let mut out = vec![0f32; out_n];
    for i in 0..out_n {
        let x = i as f64 * (frames - 1) as f64 / (out_n - 1) as f64;
        let i0 = (x as usize).min(frames - 1);
        let i1 = (i0 + 1).min(frames - 1);
        let t = (x - i0 as f64) as f32;
        out[i] = mono[i0] * (1.0 - t) + mono[i1] * t;
    }
    Ok(out)
}
