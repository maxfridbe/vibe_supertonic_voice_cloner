//! On-device-style refine (CMA polish), ported from the Android app's
//! RefineEngine + SupertonicEngine. Encoder path lands near a speaker; this
//! searches the PCA style basis with a gradient-free optimiser (separable
//! CMA-ES) to close the gap — decode a candidate, synthesise a probe, embed it,
//! score cosine to the reference. Forward passes only, no autograd.

use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

const PROBE: &str = "A cup of coffee on the desk had long since gone cold.";
const RATE: u32 = 16_000;

type Err = Box<dyn std::error::Error>;

// ---- PCA style basis (style_basis.bin) -------------------------------------

pub struct Basis {
    pub k: usize, d: usize, split: usize,
    ttl_r: usize, ttl_c: usize, dp_r: usize, dp_c: usize,
    scale: Vec<f32>, mean: Vec<f32>, basis: Vec<f32>, // basis: k*d row-major
}

impl Basis {
    pub fn load(path: &Path) -> Result<Basis, Err> {
        let b = std::fs::read(path)?;
        let i = |o: usize| i32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) as usize;
        let (k, d, split) = (i(0), i(4), i(8));
        let (ttl_r, ttl_c, dp_r, dp_c) = (i(12), i(16), i(20), i(24));
        let mut o = 28;
        let take = |o: &mut usize, n: usize, b: &[u8]| {
            let v: Vec<f32> = (0..n).map(|j| f32::from_le_bytes(
                [b[*o+4*j], b[*o+4*j+1], b[*o+4*j+2], b[*o+4*j+3]])).collect();
            *o += 4 * n; v
        };
        let scale = take(&mut o, k, &b);
        let mean = take(&mut o, d, &b);
        let basis = take(&mut o, k * d, &b);
        Ok(Basis { k, d, split, ttl_r, ttl_c, dp_r, dp_c, scale, mean, basis })
    }

    /// coeffs -> (ttl_flat, dp_flat), per-row L2 normalised (what the synth wants).
    pub fn decode(&self, c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut flat = self.mean.clone();
        for i in 0..self.k {
            let cs = c[i] * self.scale[i];
            if cs == 0.0 { continue; }
            let base = i * self.d;
            for j in 0..self.d { flat[j] += cs * self.basis[base + j]; }
        }
        let mut ttl = flat[..self.split].to_vec();
        let mut dp = flat[self.split..].to_vec();
        row_norm(&mut ttl, self.ttl_r, self.ttl_c);
        row_norm(&mut dp, self.dp_r, self.dp_c);
        (ttl, dp)
    }

    /// (ttl_flat, dp_flat) -> coeffs = ((flat - mean) @ basis^T) / scale.
    pub fn encode(&self, ttl: &[f32], dp: &[f32]) -> Vec<f32> {
        let mut flat = vec![0f32; self.d];
        flat[..self.split].copy_from_slice(ttl);
        flat[self.split..].copy_from_slice(dp);
        for j in 0..self.d { flat[j] -= self.mean[j]; }
        (0..self.k).map(|i| {
            let base = i * self.d;
            let mut acc = 0f32;
            for j in 0..self.d { acc += self.basis[base + j] * flat[j]; }
            acc / self.scale[i]
        }).collect()
    }
}

fn row_norm(m: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let s = &mut m[r * cols..(r + 1) * cols];
        let mut n = 0f32;
        for v in s.iter() { n += v * v; }
        n = n.sqrt() + 1e-8;
        for v in s.iter_mut() { *v /= n; }
    }
}

// ---- Supertonic synth ------------------------------------------------------

pub struct Synth {
    dp: Session, text_enc: Session, vec_est: Session, vocoder: Session,
    indexer: Vec<i32>, sr: u32, chunk: i64, l_dim: i64,
}

impl Synth {
    pub fn load(dir: &Path) -> Result<Synth, Err> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("tts.json"))?)?;
        let sr = cfg["ae"]["sample_rate"].as_u64().unwrap() as u32;
        let base = cfg["ae"]["base_chunk_size"].as_i64().unwrap();
        let ccf = cfg["ttl"]["chunk_compress_factor"].as_i64().unwrap();
        let ld = cfg["ttl"]["latent_dim"].as_i64().unwrap();
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("unicode_indexer.json"))?)?;
        let indexer = idx.as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        let s = |n: &str| -> Result<Session, Err> {
            Ok(Session::builder()?.commit_from_file(dir.join(n))?)
        };
        Ok(Synth {
            dp: s("duration_predictor.onnx")?, text_enc: s("text_encoder.onnx")?,
            vec_est: s("vector_estimator.onnx")?, vocoder: s("vocoder.onnx")?,
            indexer, sr, chunk: base * ccf, l_dim: ld * ccf,
        })
    }

    fn ids(&self, text: &str) -> Vec<i64> {
        preprocess(text).iter()
            .map(|&cp| *self.indexer.get(cp as usize).unwrap_or(&0) as i64)
            .collect()
    }

    /// One probe synthesis -> f32 PCM at self.sr.
    pub fn generate(&mut self, ids: &[i64], ttl: &[f32], dp: &[f32],
                    steps: usize, speed: f32, rng: &mut Rng) -> Result<Vec<f32>, Err> {
        let len = ids.len() as i64;
        let mask = vec![1f32; ids.len()];
        // duration
        let dout = self.dp.run(ort::inputs![
            "text_ids"  => Tensor::from_array(([1_i64, len], ids.to_vec()))?,
            "style_dp"  => Tensor::from_array(([1_i64, 8, 16], dp.to_vec()))?,
            "text_mask" => Tensor::from_array(([1_i64, 1, len], mask.clone()))?
        ])?;
        let duration = dout[0].try_extract_tensor::<f32>()?.1[0] / speed;
        // text embedding
        let eout = self.text_enc.run(ort::inputs![
            "text_ids"  => Tensor::from_array(([1_i64, len], ids.to_vec()))?,
            "style_ttl" => Tensor::from_array(([1_i64, 50, 256], ttl.to_vec()))?,
            "text_mask" => Tensor::from_array(([1_i64, 1, len], mask.clone()))?
        ])?;
        let (te_shape, te_slice) = eout[0].try_extract_tensor::<f32>()?;
        let te: Vec<f32> = te_slice.to_vec();
        let te_dims: Vec<i64> = te_shape.iter().map(|&x| x as i64).collect();

        let wav_len = (duration * self.sr as f32) as i64;
        let latent_len = ((wav_len + self.chunk - 1) / self.chunk).max(1);
        let ld = self.l_dim;
        let mut xt: Vec<f32> = (0..(ld * latent_len) as usize).map(|_| rng.gauss()).collect();
        let lat_mask = vec![1f32; latent_len as usize];

        for step in 0..steps {
            let out = self.vec_est.run(ort::inputs![
                "noisy_latent" => Tensor::from_array(([1_i64, ld, latent_len], xt.clone()))?,
                "text_emb"     => Tensor::from_array((te_dims.clone(), te.clone()))?,
                "style_ttl"    => Tensor::from_array(([1_i64, 50, 256], ttl.to_vec()))?,
                "text_mask"    => Tensor::from_array(([1_i64, 1, len], mask.clone()))?,
                "latent_mask"  => Tensor::from_array(([1_i64, 1, latent_len], lat_mask.clone()))?,
                "current_step" => Tensor::from_array(([1_i64], vec![step as f32]))?,
                "total_step"   => Tensor::from_array(([1_i64], vec![steps as f32]))?
            ])?;
            xt = out[0].try_extract_tensor::<f32>()?.1.to_vec();
        }
        let vout = self.vocoder.run(ort::inputs![
            "latent" => Tensor::from_array(([1_i64, ld, latent_len], xt.clone()))?
        ])?;
        let wav = vout[0].try_extract_tensor::<f32>()?.1.to_vec();
        let n = (wav.len() as i64).min(wav_len.max(1)) as usize;
        Ok(wav[..n].to_vec())
    }
}

/// Mirrors the Kotlin preprocessor (enough for the ASCII probe): NFKD, drop
/// astral chars, a few punctuation swaps, ensure a trailing stop, wrap <en>…</en>.
fn preprocess(raw: &str) -> Vec<u32> {
    let mut t: String = raw.nfkd().collect();
    t.retain(|c| (c as u32) <= 0xFFFF);
    for (a, b) in [("–","-"),("—","-"),("_"," "),("["," "),("]"," "),("|"," "),("/"," "),("#"," ")] {
        t = t.replace(a, b);
    }
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let ends = t.chars().last().map(|c| ".!?;:,'\")]}".contains(c)).unwrap_or(false);
    if !ends { t.push('.'); }
    t = format!("<en>{t}</en>");
    t.chars().map(|c| c as u32).collect()
}

// ---- ECAPA embedding -------------------------------------------------------

pub fn embed(spk: &mut Session, wav16k: &[f32]) -> Result<Vec<f32>, Err> {
    let n = wav16k.len();
    let out = spk.run(ort::inputs!["wav" => Tensor::from_array(([1_i64, n as i64], wav16k.to_vec()))?])?;
    let mut e = out[0].try_extract_tensor::<f32>()?.1.to_vec();
    let mut norm = 0f32;
    for v in &e { norm += v * v; }
    norm = norm.sqrt() + 1e-8;
    for v in &mut e { *v /= norm; }
    Ok(e)
}

pub fn resample_to_16k(wav: &[f32], src: u32) -> Vec<f32> {
    if src == RATE || wav.len() < 2 { return wav.to_vec(); }
    let frames = wav.len();
    let out_n = (frames as u64 * RATE as u64 / src as u64) as usize;
    (0..out_n).map(|i| {
        let x = i as f64 * (frames - 1) as f64 / (out_n - 1) as f64;
        let i0 = (x as usize).min(frames - 1);
        let i1 = (i0 + 1).min(frames - 1);
        let t = (x - i0 as f64) as f32;
        wav[i0] * (1.0 - t) + wav[i1] * t
    }).collect()
}

// ---- separable CMA-ES (port of the Kotlin sepCma) --------------------------

pub struct Rng { s: u64 }
impl Rng {
    pub fn new(seed: u64) -> Rng { Rng { s: seed } }
    fn next(&mut self) -> f32 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.s >> 33) as u32 as f64 as f32 / 2_147_483_648.0
    }
    fn gauss(&mut self) -> f32 {
        let mut u = self.next(); let v = self.next();
        if u < 1e-7 { u = 1e-7; }
        ((-2.0 * (u as f64).ln()).sqrt() * (2.0 * std::f64::consts::PI * v as f64).cos()) as f32
    }
}

fn sep_cma<F: FnMut(&[f32]) -> f32>(x0: &[f32], sigma0: f32, iters: usize, pop: usize,
                                    mut f: F, rng: &mut Rng) -> Vec<f32> {
    let n = x0.len();
    let mu = pop / 2;
    let mut w: Vec<f32> = (0..mu).map(|i| ((mu as f32 + 0.5).ln()) - ((i + 1) as f32).ln()).collect();
    let wsum: f32 = w.iter().sum();
    for x in &mut w { *x /= wsum; }
    let mu_eff = 1.0 / w.iter().map(|x| x * x).sum::<f32>();
    let cs = (mu_eff + 2.0) / (n as f32 + mu_eff + 5.0);
    let ds = 1.0 + cs + 2.0 * (0f32).max(((mu_eff - 1.0) / (n as f32 + 1.0)).sqrt() - 1.0);
    let cc = (4.0 + mu_eff / n as f32) / (n as f32 + 4.0 + 2.0 * mu_eff / n as f32);
    let c1 = 2.0 / ((n as f32 + 1.3).powi(2) + mu_eff);
    let cmu = (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f32 + 2.0).powi(2) + mu_eff));
    let chi_n = (n as f32).sqrt() * (1.0 - 1.0 / (4.0 * n as f32) + 1.0 / (21.0 * (n * n) as f32));

    let mut m = x0.to_vec();
    let mut sigma = sigma0;
    let mut cvec = vec![1f32; n];
    let mut ps = vec![0f32; n];
    let mut pc = vec![0f32; n];
    let mut best_x = x0.to_vec();
    let mut best_f = f(x0);

    for g in 0..iters {
        let z: Vec<Vec<f32>> = (0..pop).map(|_| (0..n).map(|_| rng.gauss()).collect()).collect();
        let x: Vec<Vec<f32>> = (0..pop).map(|p| (0..n)
            .map(|j| (m[j] + sigma * cvec[j].sqrt() * z[p][j]).clamp(-3.0, 3.0)).collect()).collect();
        let scores: Vec<f32> = (0..pop).map(|p| f(&x[p])).collect();
        let mut order: Vec<usize> = (0..pop).collect();
        order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
        if scores[order[0]] < best_f { best_f = scores[order[0]]; best_x = x[order[0]].clone(); }
        let mut zmean = vec![0f32; n];
        let mut new_m = vec![0f32; n];
        for r in 0..mu {
            let idx = order[r];
            for j in 0..n { zmean[j] += w[r] * z[idx][j]; new_m[j] += w[r] * x[idx][j]; }
        }
        m.copy_from_slice(&new_m);
        let mut ps_norm = 0f32;
        for j in 0..n { ps[j] = (1.0 - cs) * ps[j] + (cs * (2.0 - cs) * mu_eff).sqrt() * zmean[j]; ps_norm += ps[j] * ps[j]; }
        ps_norm = ps_norm.sqrt();
        sigma *= ((cs / ds) * (ps_norm / chi_n - 1.0)).exp();
        let hs = if ps_norm / (1.0 - (1.0 - cs).powi(2 * (g as i32 + 1))).sqrt() < (1.4 + 2.0 / (n as f32 + 1.0)) * chi_n { 1.0 } else { 0.0 };
        for j in 0..n {
            pc[j] = (1.0 - cc) * pc[j] + hs * (cc * (2.0 - cc) * mu_eff).sqrt() * cvec[j].sqrt() * zmean[j];
            let mut cmu_term = 0f32;
            for r in 0..mu { let d = cvec[j].sqrt() * z[order[r]][j]; cmu_term += w[r] * d * d; }
            cvec[j] = ((1.0 - c1 - cmu) * cvec[j] + c1 * pc[j] * pc[j] + cmu * cmu_term).max(1e-8);
        }
        eprintln!("  gen {}/{}: best cosine so far {:.3}", g + 1, iters, -best_f);
    }
    best_x
}

// ---- entry -----------------------------------------------------------------

pub struct Refined { pub ttl: Vec<f32>, pub dp: Vec<f32>, pub start_cos: f32, pub end_cos: f32, pub evals: usize }

/// Everything one candidate evaluation touches, so the CMA closure can borrow it
/// mutably without fighting the borrow checker over separate pieces.
struct Eval {
    basis: Basis, synth: Synth, spk: Session,
    target: Vec<f32>, probe_ids: Vec<i64>, rng: Rng, evals: usize,
}
impl Eval {
    fn cosine(&mut self, c: &[f32]) -> f32 {
        let (ttl, dp) = self.basis.decode(c);
        let wav = match self.synth.generate(&self.probe_ids, &ttl, &dp, 8, 1.05, &mut self.rng) {
            Ok(w) => w, Err(_) => return -1.0,
        };
        let w16 = resample_to_16k(&wav, self.synth.sr);
        let emb = match embed(&mut self.spk, &w16) { Ok(e) => e, Err(_) => return -1.0 };
        emb.iter().zip(&self.target).map(|(a, b)| a * b).sum()
    }
    fn objective(&mut self, c: &[f32]) -> f32 {
        self.evals += 1;
        let l2: f32 = c.iter().map(|v| v * v).sum();
        -(self.cosine(c) - 0.02 * l2 / c.len() as f32)
    }
}

/// Refine an encoder seed (seed_ttl/seed_dp) against a reference recording.
pub fn refine(supertonic_dir: &Path, models_dir: &Path, ref_wav16k: &[f32],
              seed_ttl: &[f32], seed_dp: &[f32], iters: usize, pop: usize) -> Result<Refined, Err> {
    let basis = Basis::load(&models_dir.join("style_basis.bin"))?;
    let mut synth = Synth::load(supertonic_dir)?;
    let mut spk = Session::builder()?.commit_from_file(models_dir.join("spk_encoder.onnx"))?;
    let target = embed(&mut spk, ref_wav16k)?;
    let probe_ids = synth.ids(PROBE);
    let x0 = basis.encode(seed_ttl, seed_dp);
    let k = basis.k;

    let mut ev = Eval { basis, synth, spk, target, probe_ids, rng: Rng::new(1234567), evals: 0 };
    let start_cos = ev.cosine(&x0);
    eprintln!("start cosine {:.3}; searching k={} basis, {} gens x {} pop", start_cos, k, iters, pop);

    let mut cma_rng = Rng::new(987654321);
    let best = sep_cma(&x0, 0.35, iters, pop, |c| ev.objective(c), &mut cma_rng);
    let end_cos = ev.cosine(&best);
    let (ttl, dp) = if end_cos >= start_cos { ev.basis.decode(&best) } else { ev.basis.decode(&x0) };
    Ok(Refined { ttl, dp, start_cos, end_cos, evals: ev.evals })
}
