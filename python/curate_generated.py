"""Curate a diverse, clean subset of the 6000 manufactured (generated) styles
into app-ready style JSONs with an auto description. No import provenance —
each is a random sample in the refit basis, re-synthesised so its audio matches
its JSON exactly."""
import os, sys, json, numpy as np, torch
WORK = os.path.expanduser("~/supertonic-experiment")
os.environ.setdefault("SB_CACHE", os.path.join(WORK, ".sb_cache"))
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.join(WORK, "tooling"))
from train_encoder import StyleBasis
from pipeline import DiffSynth

dev = "cuda:0"; O = os.path.join(WORK, "clone_out/overnight")
OUT = os.path.join(O, "curated_generated"); os.makedirs(OUT, exist_ok=True)
K = 20
SENT = "Hello. This is a generated voice you can use in the app. I hope it sounds clear and natural."

pairs = np.load(os.path.join(O, "all_pairs3.npz"), allow_pickle=True)
bp = np.load(os.path.join(WORK, "clone_out/pairs_p1.npz"), allow_pickle=True)
rep = max(1, len(bp["ttl"]) // max(len(pairs["ttl"]), 1))
basis = StyleBasis(np.concatenate([bp["ttl"], np.repeat(pairs["ttl"], rep, 0)]),
                   np.concatenate([bp["dp"], np.repeat(pairs["dp"], rep, 0)]), 128, dev)
synth = DiffSynth(os.path.join(WORK, "assets/onnx"), device=str(dev))
dp0 = bp["dp"][:1].astype(np.float32)
pt, pl = synth.pad_bounds([SENT], "en", dp0, 1.05)
prep = synth.prepare(SENT, "en", pad_text=pt, pad_latent=pl)

C = np.load(os.path.join(O, "mfg_wavs/coeffs.npz"), allow_pickle=True)["coeff"].astype(np.float32)
# numpy k-means (Lloyd) for diversity, then medoid nearest each centroid
rng = np.random.default_rng(7)
cen = C[rng.choice(len(C), K, replace=False)]
for _ in range(18):
    a = ((C[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
    for j in range(K):
        m = C[a == j]
        if len(m): cen[j] = m.mean(0)
medoids = []
for j in range(K):
    idx = np.where(a == j)[0]
    if not len(idx): continue
    medoids.append(int(idx[((C[idx] - cen[j]) ** 2).sum(-1).argmin()]))

def f0_median(w, sr=16000):
    w = w / (np.abs(w).max() + 1e-8)
    fl = int(0.04 * sr); vals = []
    for s in range(0, len(w) - fl, fl):
        fr = w[s:s + fl]
        if (fr ** 2).mean() < 1e-3: continue
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        lo, hi = sr // 400, sr // 70
        if hi >= len(ac): continue
        lag = lo + np.argmax(ac[lo:hi])
        if ac[lag] > 0.3 * ac[0]: vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0

def describe(f0):
    if f0 == 0: return "neutral voice"
    if f0 < 130: return "deep low male voice"
    if f0 < 165: return "warm male voice"
    if f0 < 200: return "light male / low female voice"
    if f0 < 245: return "clear female voice"
    return "bright high female voice"

rows = []
for n, mi in enumerate(medoids, 1):
    c = C[mi]
    with torch.no_grad():
        ttl, dp = basis.decode(torch.tensor(c[None], dtype=torch.float32, device=dev))
        w, _, _ = synth.synth(ttl, dp, prep)
    wav = w.squeeze().detach().cpu().numpy()
    if np.abs(wav).max() < 0.02 or len(wav) < synth.sr:   # silent / too short -> skip
        continue
    import torchaudio
    w16 = torchaudio.functional.resample(torch.tensor(wav)[None], synth.sr, 16000).squeeze().numpy()
    f0 = f0_median(w16); desc = describe(f0)
    tarr = ttl.squeeze(0).detach().cpu().numpy(); darr = dp.squeeze(0).detach().cpu().numpy()
    name = f"Generated {n:02d}"
    js = {
        "style_ttl": {"dims": [1, *tarr.shape], "data": tarr.reshape(-1).tolist()},
        "style_dp":  {"dims": [1, *darr.shape], "data": darr.reshape(-1).tolist()},
        "metadata": {"name": name, "source": "generated (manufactured basis sample)",
                     "description": desc, "f0_hz": round(f0, 1), "sample_text": SENT},
    }
    slug = f"gen-{n:02d}"
    json.dump(js, open(os.path.join(OUT, slug + ".json"), "w"))
    import soundfile as sf
    sf.write(os.path.join(OUT, slug + ".wav"), wav, synth.sr)
    rows.append((slug, name, desc, round(f0, 1)))
    print(f"{slug}: {desc}  (f0 {f0:.0f} Hz)")

json.dump(rows, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(f"\n{len(rows)} curated -> {OUT}")
