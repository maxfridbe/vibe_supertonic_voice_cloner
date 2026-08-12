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

**Running:** old-vs-new basis benchmark — 4 real voices (Dale, Fireside,
Soothing, Stephen Fry), shipping config, both GPUs, incl. per-voice ceiling
projections.

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
