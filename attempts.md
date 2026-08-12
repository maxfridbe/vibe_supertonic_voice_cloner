# Attempts log

Running lab notebook: everything tried, what it found, what it changed.
Newest entries at the bottom. Statuses: ✅ done · 🔄 running · 📋 planned.

## 2026-08-11 — Refine speedup levers, desktop (✅)

**Tried:** four search-side levers in `rust/clonevoice` (`refine.rs`), no model
changes: (1) search at 4 flow steps + ~2 s probe, full fidelity only for
seed/winner scoring; (2) duration frozen from the seed → whole CMA generation
synthesised as one batched ONNX pass; (3) one shared noise tensor per run
(common random numbers); (4) early stop after 15 idle generations.

**Findings:** identical 49-eval budget on desktop CPU: 64.1 s → 16.1 s
end-to-end (4×); search loop ~1.1 s → ~0.13 s per evaluation (~8×). Full
120×20 refine: ~45 min → ~5 min. Verified via ONNX graph metadata that every
Supertonic input has a dynamic batch dim — `current_step`/`total_step` must be
tiled to `[B]`, not passed as `[1]`. Commit `4dd4dce`.

## 2026-08-11 — Kotlin port + on-device A/B, Galaxy S24 FE (✅)

**Tried:** same four levers ported to the app (`SupertonicEngine.generateBatch`,
`RefineEngine`); `RefineService` exported so adb can drive it (same rationale
as `TtsService`). A/B on the same Dale reference at 120×20: new build to
completion, old build sampled for 10 min then stopped.

**Findings:** old ~2.5 s/eval (projected ~93 min); new ~0.83 s/eval (~3× —
batching buys less on mobile than desktop, memory-bandwidth-bound), early stop
at gen 65/120 → 18.5 min wall, **~5× end-to-end**, 0.50 → 0.62 held-out.
Batched path never fell back. Commit `7540245` (app repo).

**Open question:** Dale reached 0.62 on-device vs 0.716 desktop-reference —
protocol differences account for some, but patience 15 may be cutting hard
voices short. → optimizer sweep below.

## 2026-08-11 — Optimizer sweep on brainiac-nvidia GPU 0 (🔄)

**Trying:** `tooling/refine_experiments.py` (new): 20 of the 40 generated
named-library voices (two instances × 10 disjoint voices — second instance
added after observing the 5070 at ~28% util; one instance doesn't saturate it,
two put it at ~94%). Six conditions, shared seeds/budget (120×20, k=384):
`base-short` (no patience), `pat15`, `pat25`, `c2f-pat25` (top-128
coefficients first third, then all 384), `rich-pat25` (phonetically rich
Rainbow-Passage probe), `dual-pat25` (two short probes averaged, 2× cost).
Metrics: held-out ECAPA cosine, FRESH, style-space recovery (targets are
generated, so ground-truth styles are known).

**Caveat by design:** named-library voices were sampled *from* the basis —
in-span by construction — so this measures the optimizer, not the basis
ceiling.

**Findings (✅, 120 runs / 20 voices complete):**

| condition | held-out | fresh | synths | voice wins |
|---|---|---|---|---|
| base-short (no patience) | 0.743 | 0.690 | 2,405 | 0 |
| pat15 | 0.708 | 0.662 | 1,631 | 0 |
| pat25 | 0.712 | 0.656 | 1,790 | 0 |
| c2f-pat25 | 0.711 | 0.666 | 2,115 | 0 |
| **rich-pat25** | **0.804** | 0.722 | 2,133 | **11** |
| dual-pat25 | 0.798 | **0.738** | 4,334 | 9 |

1. **The probe dominates everything.** All 20 per-voice wins went to a probe
   variant; the phonetically rich probe beats the short-probe baseline on
   19/20 voices (mean **+0.061**, max +0.177, worst −0.010) at *less* cost —
   ECAPA gets more of the timbre spectrum to match per evaluation.
2. **Dual probes ≈ rich probe** on quality (slightly better on fresh
   sentences) at 2× the synthesis cost — not worth it on-device.
3. **Patience alone loses quality** (−0.03 vs baseline at 15 or 25); it's only
   acceptable when paired with a probe good enough to converge early.
4. **Coarse-to-fine won nothing here** — expected in hindsight: these targets
   are in-basis by construction, so the coefficient tail carries little. It
   may still matter for out-of-span real voices (Dale); the basis-refit
   benchmark will tell.

**Adopted:** search probe → Rainbow-Passage opener, PATIENCE → 25, in both
`refine.rs` and `RefineEngine.kt`. Honest cost note: the rich probe is ~2.5×
the short probe's audio length, so phone runs get proportionally slower per
evaluation than the short-probe build — still far faster than the original
full-fidelity loop, and worth +0.06 held-out.

## 2026-08-11 — VCTK inversion night on brainiac-nvidia GPU 1 (🔄)

**Why:** the basis is the ceiling for *real* voices, and the current one was
fit on LibriSpeech inversions only — nothing resembling character voices in
the bank (likely part of why Dale lags). VCTK adds 110 accented speakers;
refit follows once banked.

**Attempt 1 (❌, fixed):** launched with a relative `--out`; the invert
subprocess runs with `cwd=src/`, so every reference path resolved to a
nonexistent file — all 110 speakers "produced nothing" in ~7 minutes.
Lesson recorded: `invert_corpus.py` needs an **absolute** `--out` (its default
is absolute for this reason).

**Attempt 2 (🔄):** relaunched with absolute paths; refs staged once, reused.
GPU 1 at ~56%/8.9 GB, first speakers inverting. Output:
`clone_out/overnight/inversions_vctk/aux_pairs.npz` (~110 speakers, expected
overnight).

## 2026-08-12 — Generic, documented basis builder (✅)

**Tried:** `python/make_basis.py` — portable (numpy-only) replacement for the
workstation-hardcoded `export_k384.py`: documented methodology in the
docstring (flatten → balance → center → SVD → top-k, scale = s/√(N−1)),
generic inputs (`--npz` ttl/dp archives, `--styles-dir` of style JSONs),
writes the 7-int32-header binary.

**Findings:** refitting with the original inputs (1200 manufactured pairs +
198 LibriSpeech-inversion styles, `--balance` → ×6) reproduces the shipped
basis exactly: 96.1 % variance at k=384, byte-count-identical output
(19,910,684 B), round-trip err 1e-5. This is now the canonical path for the
VCTK-extended refit.

## 2026-08-12 — Winner config on real phones, Dale (✅)

**Tried:** the adopted config (rich probe + patience 25) end-to-end on two
phones, same Dale reference, 120×20 budget.

**Findings:** Galaxy S24 FE (Exynos 2400e): 0.44 → **0.68** in 78.5 min.
Galaxy S24 (Snapdragon 8 Gen 3): 0.51 → **0.65** in 85.4 min. vs yesterday's
short-probe build on the FE: 0.62 in 18.5 min. The sweep's predicted ~+0.06
held-out gain transfers to a real out-of-basis voice; cost is ~4× wall (longer
probe audio + patience using the full budget). Snapdragon ≈ Exynos on the CPU
path — chip-agnostic, so further speed must come from GPU delegation (route-1
experiment queued). Product note: worth exposing quick (short-probe) vs best
(rich-probe) refine modes until the GPU work lands.

## 2026-08-12 — VCTK basis refit (🔄)

**Done:** 109/110 VCTK speakers inverted overnight on the 3060 (~8.4 min each,
label cos 0.59–0.90). New k=384 basis via checked-in `make_basis.py` from
pairs_p1 + LibriSpeech aux ×6 + VCTK aux ×11: **97.5 % variance (was 96.1 %)**.

**Findings (✅, old-vs-new benchmark complete):** 4 real voices, shipping
config, both GPUs:

| voice (held-out) | old basis | VCTK basis | ceiling old→new |
|---|---|---|---|
| Dale | 0.757 | 0.750 | 0.49 → 0.54 |
| Fireside Narrator | 0.797 | 0.793 | 0.43 → 0.51 |
| Soothing | 0.824 | **0.837** | 0.55 → 0.60 |
| Stephen Fry | 0.759 | **0.769** | 0.50 → 0.55 |

1. Final refined quality is a wash (+0.003 mean) — the search compensates for
   the old basis on these voices.
2. The new basis wins everywhere else: ceilings +0.05–0.08 on all four,
   starts +0.03–0.09 (direct encoder-path benefit), convergence ~25% faster.
3. **Retro-finding:** the *old* basis under the rich probe scored 0.757 on
   Dale vs the historical 0.716 — much of what looked like a basis ceiling
   yesterday was actually the short probe. The probe win compounds.

**Shipped:** `models/style_basis.bin` replaced with the VCTK-extended fit
(97.5 % variance, same k=384 byte layout, drop-in — the encoder head emits
styles, not coefficients).

## 2026-08-12 — GPU route 1: static shapes + ORT NNAPI (✅ answered: dead end)

**Tried:** the refine's frozen shapes make static graphs possible, so: fixed
the three hot-loop graphs to batch 20 / 92 probe tokens / 128 mask-padded
latent frames (`onnxruntime.tools.make_dynamic_shape_fixed`); app loads them
as optional accelerator sessions with per-generation CPU fallback. Findings
by step, on the Galaxy S24 (Snapdragon 8 Gen 3):

1. Dynamic graphs + NNAPI: places nothing (the app already knew this).
2. **Static graphs + NNAPI: real progress — the EP now tries to compile** and
   dies on a builder bug (`AddNnapiSplit count [0] does not evenly divide
   dimension`), identical on ORT 1.20 and 1.29 (bumped the app to 1.29).
3. **Split→Slice graph surgery** (`split_to_slice.py`, 24 nodes rewritten,
   bit-exact parity vs originals): sessions compile and run under NNAPI…
4. …at **2.7× slower than plain CPU** (projected ~230 min vs 85 min): the
   driver's placements + partition-boundary copies + NNAPI's reference-CPU
   fallback lose to ORT's multithreaded CPU kernels.
5. `USE_FP16 + CPU_DISABLED` flags (genuine accelerator partitions only,
   fp16): still ~2.6× slower (~223 min). **ORT's NNAPI EP is a dead end for
   these graphs.**

**Kept:** the static-graph infrastructure (loader, mask-padded batch path,
fallback) — it's exactly what any other accelerator backend needs, and the
static+surgered graphs are the input for route 2. NNAPI stays opt-in
(`backend=nnapi` extra), CPU remains the default.

**Next for GPU:** route 2 — convert the static graphs to LiteRT and use its
GPU delegate (real OpenCL/Vulkan, no NNAPI HAL in the middle); or QNN EP
direct on Snapdragon. The Split→Slice + static-shape work carries over.

## 2026-08-12 — GPU route 2 opened and parked (🅿)

**Tried:** onnx2tf conversion of the static graphs to LiteRT. text_encoder
converted only after onnxsim pre-folding, and the produced flatbuffer is
**invalid** (XNNPACK reshape prepare failure; a 5-D tensor with a 4-element
transpose perm) — per-op replacement-JSON debugging required, on the easiest
graph. Parked by decision: improve quality numbers first, revisit
acceleration when there's something worth speeding up. Noted for later:
`onnxruntime-android-qnn` 1.29.0 exists on Maven and consumes our
static+surgered ONNX directly — likely the cheaper path than converter
archaeology when we return.

## 2026-08-12 — Translation-head capacity sweep (✅)

**Question (user's):** would a bigger embedding→style head (9 MB → 100 MB)
learn the inversion mapping better?

**Setup:** 307 real pairs (198 LibriSpeech + 109 fresh VCTK inversions),
targets in the new k=384 basis, ECAPA 192-d input, dropout 0.1 + feat-noise
0.1, same val split (46 refs; 43 scored), judged by synthesize→embed→cosine
(`score_styles.py`).

| head | params | val MSE | held-out audio cos |
|---|---|---|---|
| 512×2 (shipped arch) | 0.4M | 2.305 | 0.446 |
| 1024×3 | 2.5M | 2.368 | 0.481 |
| **2048×4** | **13M** | 2.487 | **0.494** |
| 4096×4 | 55M | *2.210 best* | *0.317 worst* |

1. **The shipped head is undersized**: ~13M params buys +0.048 held-out audio
   cosine. The user's intuition was right up to that scale.
2. **The 100 MB class collapses** (overfits 307 noisy pairs), despite the
   *best* coefficient MSE —
3. — confirming coefficient MSE and audio quality are uncorrelated across
   capacities; only the audio score ranks runs.

**Next:** export the 2048×4 head as the new `style_encoder.onnx` (needs
`export_cloner.py` to honor hidden/depth), and repeat with Qwen features
(extract on VCTK refs first) for the qwen-variant answer.

## 📋 Planned next

- **Pick winner config** from the sweep → update Kotlin `RefineEngine` +
  Rust `refine.rs` constants (patience/probe/c2f).
- **Basis refit** on pairs_p1 + LibriSpeech aux + VCTK aux → re-export
  `style_basis.bin` (k=384 via `export_k384.py` recipe) → re-run the
  4 real benchmark voices (Dale especially) against the new ceiling.
- **Latent-space scorer** (distill vocoder+ECAPA out of the search loop) and
  **surrogate-assisted CMA** — the two big remaining speed levers; need
  training runs.
- **Self-distillation refiner head**: recycle finished refine runs as
  `(embedding, seed → refined coeffs)` training pairs so the encoder's guess
  improves permanently.
