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

**Findings so far:** first row: aurora-belle `base-short` 0.379 → 0.736
held-out (style-cos 0.974, 5.8 min). Full sweep ~5 h; results land in
`clone_out/overnight/refine_experiments{,_b}.json`.

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
