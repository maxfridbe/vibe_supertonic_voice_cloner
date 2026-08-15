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

**Shipped:** the ECAPA 2048×4 head is the new `models/style_encoder.onnx`
(75 MB fp32: 13M head + folded k=384 VCTK basis; smoke-tested through
`clonevoice`). `export_translation.py` honors hidden/depth now.

**Qwen variant (✅ answered):** extracted Qwen features for the 109 VCTK refs
(via the audiobook-maker venv — `qwen_tts` lives there, not in the supertonic
venv), merged with the 188 LibriSpeech features, same protocol:

| head (input) | held-out audio cos |
|---|---|
| Qwen 512×2 | 0.426 |
| Qwen 2048×4 | 0.344 |

For the **qwen head, bigger hurts immediately**: the 2048-d input makes every
layer ~10× wider than the ECAPA twin, so 13M-class capacity memorises 297
pairs (the trainer's own `--pca` warning, confirmed). And ECAPA beats Qwen at
every size on this corpus-speaker val set — consistent with the old finding
that Qwen features earn their keep on *character* voices, which this
benchmark contains none of. Verdict: don't grow the qwen head on current
data; grow the *dataset* (or add a PCA front) first.

## 2026-08-12 — Qwen-space similarity metric + referee flip (✅)

**Direction change (user):** focus the Qwen path; the ECAPA-path output is
judged useless by ear. Idea: converge training on a *Qwen* similarity metric.

**Metric built** (`python/qwen_similarity.py`): cosine over mean-centered Qwen
speaker features. Raw features share a dominant common component (different
speakers cosine at 0.942!); after centering on the reference population:
same-speaker 0.836 ± 0.061 vs different-speaker −0.005 ± 0.270, **d′ 4.30,
4.1 % err** on 61 half-vs-half reference pairs — a valid, well-scaled metric.

**Referee flip, measured** (same synthesized eval audio, two judges):

| head | ECAPA cos | centered Qwen cos |
|---|---|---|
| **Qwen 512×2** | 0.426 (5th) | **0.369 (1st)** |
| ECAPA 2048×4 | **0.494 (1st)** | 0.308 (2nd) |
| Qwen 2048×4 | 0.344 | 0.303 |
| ECAPA 1024×3 / 512×2 | 0.481 / 0.446 | 0.264 / 0.236 |
| ECAPA 4096×4 | 0.317 | 0.149 |

Each analyzer's head wins under its own referee — quantified referee bias.
Capacity conclusions survive the flip (55M collapses under both; growing the
qwen head still doesn't pay). Under the ear-aligned judge, **q512×2 is the
best head produced today**.

**Shipped:** `models/style_encoder_qwen.onnx` (26 MB) — the q512×2 head
(new k=384 VCTK basis, LS+VCTK training) exported for the app's qwen analyzer
variant.

**Next for qwen-first:** select head checkpoints by centered-qwen audio score
(not coeff MSE); a qwen-scored refine option (the app already ships
qwen_spk_encoder.onnx, so the on-device loop can converge on this metric);
character-voice benchmark set; qwen-loss inversion nights.

## 2026-08-12 — Full-stack Dale on S24: seed-choice lesson (✅)

**Tried:** end-to-end clone with everything new (2048×4 ECAPA head seed via
desktop clonevoice + VCTK basis + rich probe) on the S24: 0.25 → 0.53.
Morning's run with the same phone but seeded from the curated `Dale.json`
(previously refined style): 0.51 → 0.65.

**Lesson:** the test conflated "new head" with "replaced a good seed." A raw
encoder guess on an out-of-domain character voice (0.25 — though better than
the old head's ~0.1–0.2 on Dale) loses to a curated seed (0.51), and the
refine can't fully recover the difference. Keep seeding from the best
available style when one exists (the app's normal flow does). Character
voices stay the encoder path's blind spot under the ECAPA referee — the
overnight qwen-first program targets exactly this.

## 2026-08-12 — Qwen-first overnight program (🔄)

Running on brainiac: (A ✅) 4,000 manufactured (audio, coeff) pairs in the
merged k=384 basis — the qwen path's first manufactured supervision (ECAPA
always had ~1,200; qwen had 0); (B 🔄) qwen features for all 4,000, two
parallel extractors on GPU 0; (C queued) six qwen-head configs — ±extra
pairs, ±PCA-256 front, capacity retests at 1024/2048 — each scored under both
referees, two parallel lanes; (E 🔄, GPU 1) **qwen-scored refine** prototype:
10 voices searched against centered-qwen cosine via qwen_spk_encoder.onnx,
finals scored in both spaces — the go/no-go for porting the qwen objective
into the app's RefineEngine.

**Ops lesson (twice-burned):** `pkill -f <script>` inside an ssh one-liner
matches the ssh command line itself and kills its own shell — use
`[c]haracter-class` patterns or skip pkill.

## 2026-08-12 — Qwen-first night program: results (✅)

All metrics are **centered qwen cosine only** (per direction; ECAPA retired).
**Goal line calibrated first**: 40 desktop-inverted styles re-scored under the
qwen metric = **0.4905** — the qwen-space equivalent of the old "~0.82 ECAPA."

**Head program** (6 configs, 297 real + 4,000 manufactured qwen pairs):

| config | qwen-cos |
|---|---|
| **ex512pca** (512×2, PCA-256 front, +pairs) | **0.5047 — above the goal line** |
| ex1024pca | 0.4568 |
| ex512 (+pairs, no PCA) | 0.4548 |
| ex2048 | 0.4175 |
| ex1024 | 0.3777 |
| base512 (no pairs) | 0.3685 |

Manufactured pairs: +0.086. PCA front on top: +0.050 more. Capacity: still
never pays. **The instant encoder guess now exceeds the desktop inversions'
own similarity** — shipped as `models/style_encoder_qwen.onnx` (24 MB,
PCA + basis folded in) and deployed to the S24.

**Qwen-scored refine** (18 voices, objective = centered qwen via
qwen_spk_encoder.onnx): mean **0.840**, range 0.664–0.917 (grizzled-gus
0.917, boomer-hank 0.909, frosty-fern 0.903); real voices avg 0.725 (Dale
0.717, Fireside 0.800), generated voices avg 0.873. Every voice lands far
above the 0.49 goal line. Ear-check demos rendered (deacon-gray,
grizzled-gus A/B m4a) — Goodhart guard: the listen is the final judge.

**Also banked:** 7,000 manufactured pairs with qwen features (3,000 extracted
for the next round).

**Ship list next:** port the qwen objective into the app's RefineEngine
(embed probe with qwen_spk_encoder.onnx + centering); retrain-select heads by
qwen audio score in-loop; character-voice reference set for the app voices.

## 2026-08-12 — Whisper-class case study: "my mistress" (✅ diagnosis, 🔄 verdict by ear)

**Report:** on-device qwen clone of a whispered, heavily stylised recording
sounded completely wrong. Server clone with identical models sounded the same
→ **not an app bug: the encoder head is out-of-domain** (training bank =
read speech + manufactured samples; zero whispered voices — the Dale pattern
from the original research, again).

**Escalation ladder, all delivered to the phone + Telegram demos:**
| method | compute | centered qwen |
|---|---|---|
| instant encoder | ~1 s | (wrong by ear) |
| qwen-scored refine | ~7 min | 0.575 → **0.812** |
| gradient inversion (500 steps) | ~10 min | 0.946 (own metric) |

**Also shipped meanwhile:** the app's refine now judges by centered qwen
cosine when qwen_spk_encoder.onnx + qwen_center.bin are staged (app commit
`c19888c`; center vector = qwen_feats_all population mean, 2048×f32); style
JSON metadata now records analyzer/objective/scores/budget/version/date;
RefineService takes iters/pop extras.

**If the ear confirms the refine/inversion:** whisper-class is *reachable* —
the fix is a data night inverting expressive/whispered recordings into the
bank so the basis, head, and manufactured pairs all learn the class.

## 2026-08-13 — Expressive bank + full retrain (✅)

**Data acquired** (agent-fetched, verified on disk, ~28 GB): EARS subset
(25 speakers, 1.37 h pure whisper), EmoV-DB, AniSpeech (MIT character
voices), ASVP-ESD, JL-Corpus, CREMA-D, TESS. (The 15 Mozilla Data Collective
sets are login-walled; at least one bans voice cloning — skipped.)

**Inversions:** wave 1: 62/80 banked (RAVDESS happy/angry/fearful ×24 actors
+ 8 Thorsten styles incl. the first whisper, cross-lingual, 0.651). Wave 2:
49/50 EARS refs — English whispers labeled up to **0.880**. Bank: 307 → **418
real pairs**, whisper/emotion represented for the first time.

**Retrains (all centered-qwen, old center kept for comparability):**
| head | val set | qwen |
|---|---|---|
| shipped yesterday (ex512pca) | 43 corpus voices | 0.5047 |
| interim (wave-1 bank) 512×2+PCA | 52 voices (+expressive) | 0.5234 |
| deeper variants 768×3 / 1024×3 | same | 0.4878 / 0.4876 |
| **final fh_d10 (418-pair bank)** | **58 voices (+whisper refs)** | **0.5240** |

Depth re-tested with 14× data and expressive coverage: **still negative**
(third and fourth confirmations). Small head + PCA front remains optimal.

**Shipped:** `style_encoder_qwen.onnx` (fh_d10) + its bound
`style_basis.bin` (v3, fit incl. all expressive pairs). "My Mistress"
instant-clone v1→v2→v3 demos delivered for the ear test — the coverage
thesis's before/after.

## 2026-08-13 — Instant-v3 ear tests: whisper fixed, bubbly-class fails (🔄)

**Ear test 1 — "My Mistress" instant v3** (fh_d10 head, expressive bank):
user verdict **"that sounds good"**. The coverage fix worked — a whispered,
heavily stylised voice that the instant encoder previously got *completely
wrong* is now acceptable with zero refine. The expressive-bank thesis is
confirmed end-to-end (data → basis v3 → head → ear).

**Ear test 2 — "Bubbly british girl" instant v3**: user verdict **"not even
close"**. Same failure signature as pre-fix My Mistress: a delivery class
the bank still doesn't cover. The 418-pair bank's expressive additions are
emotion-acted (RAVDESS/CREMA/EmoV: angry/happy/sad/fearful) and whisper
(EARS/Thorsten) — there is essentially **no high-energy bright/bubbly young
female** delivery in it. AniSpeech (character voices) was downloaded but not
yet inverted; TESS is older-female; JL-Corpus (young NZ adults, "excited"
label) also uninverted — those are the candidate coverage for this class.

**Running now** (same playbook as the My Mistress case): gradient inversion
of `bubbly british girl.wav` on GPU0 (invert.py, muon, 500 steps, instant-v3
seed) + qwen-objective refine on GPU1 (refine_qwen.py, k=384 v3-basis
sources, instant-v3 seed) → ladder demos to Telegram for the ear.

**Note:** render_demo on server now needs `TORCHDYNAMO_DISABLE=1` — torch
inductor started failing in the vocoder compile ("vr must not be None for
symbol q1"); eager is fine and barely slower for one-off renders. Same bug
killed the first bubbly invert.py run; eager invert measured **faster**
anyway (594 vs 863 ms/iter) — use `TORCHDYNAMO_DISABLE=1` everywhere now.

**Bubbly escalation ladder results (all delivered to Telegram):**
| method | score | ear |
|---|---|---|
| instant v3 | qwen 0.510 | "not even close" |
| qwen refine (8.5 min) | qwen 0.510 → **0.720** | better |
| gradient inversion (500 steps) | ecapa 0.734 (vs mm's 0.946) | **worse than refine** |

Two findings: (1) refine stalls at 0.72 vs the usual ~0.84 → the v3 basis
itself under-spans this class, exactly the coverage signature; (2) the
ECAPA+mel-loss *inversion* lost to the qwen-objective *refine* by ear — the
referee flip now shows up even against the gradient method. Inversion labels
remain useful as bank data, but qwen-refine is the quality path for delivery.

## 2026-08-13 — Wave 3: bubbly/high-energy coverage night (🔄)

**Hypothesis test that "we just need more voice types":** confirmed enough to
act on (My Mistress = the causal experiment; bubbly = same signature).

**Staged 94 refs** (flat layout, 47 per GPU, `stage_wave3.py`):
- **AniSpeech** 52 character voices (parquet-packed; pyarrow-extracted,
  caption transcripts) — the energetic/bright class itself
- **JL-Corpus** 4 speakers × {excited, happy, encouraging, assertive,
  anxious, apologetic, concerned, sad} with its own transcripts
- **EmoV-DB** amused + sleepy × 4 speakers (whisper-small transcripts)
- **TESS** 2 speakers, pleasant-surprise ("Say the word X" transcripts)

**Pipeline** (`run_wave3.sh`, fully chained, Telegram notifies at each phase):
invert both halves (GPU0+GPU1, eager) → merged_aux4 bank → basis v4 (k=384)
→ mfg_v4 3000 pairs (GPU0) ∥ wave3 qwen features (GPU1) → qwen_feats_all4
(OLD center kept) → fh4_d10/d15 heads → qwen-score → export_v4 → regenerate
bubbly + mymistress instant v4 + demos. ETA ~5h invert + ~2h retrain.

## 2026-08-13 — Ray Porter case (🔄) + models deployed to the app

**New voice: Ray Porter** (Bobiverse Audible sample, clean narration,
whisper-verified). Ladder so far, all delivered to Telegram:
| method | centered qwen | ear |
|---|---|---|
| instant v3 | 0.524 (exactly the head's par) | "does not sound good" |
| qwen refine, 21s ref | 0.513 → 0.819 (9.4 m) | "not amazing" |
| refine round 2: 90s ref + 200×24 budget, seeded from round 1 | 0.815 → **0.829**, early-stopped at 93 gens | — |

Round 2's tiny gain despite 2.4× budget and a 4× longer reference = **the v3
basis ceiling for this voice**, not a search failure. Queued for tonight
(auto-runs after wave 3): gradient inversion + instant/refine v4 over the new
basis. GPU-borrowing trick worked twice: SIGSTOP the wave-3 B-half driver,
let its in-flight inversion finish, refine on GPU1, SIGCONT — wave cost
≈ refine time only.

**Deployed to production** (user request): app repo `08a67ef` ships fh_d10 +
basis v3 + qwen_center.bin via the in-app downloader; the basis is served as
`style_basis_v3.bin` because a k=384 refit never changes byte size and size
is the downloader's only cache-buster (RefineEngine falls back to a
side-loaded `style_basis.bin`). CI APK build green (run 31760098760). Cloner
repo `084c486`: qwen_center.bin checked in, READMEs to current numbers.

## 2026-08-14 — Wave 3 results: bank 511, v4 heads, Ray Porter ladder (✅ run, 🔄 ear)

**Inversions: 93/94 banked** (one quality-gated), bank 418 → **511 real
pairs**. Basis v4 written (round-trip err 6e-6).

**Pipeline bug + fix:** the v4 train() omitted the `score_styles` step — which
*renders* the val styles that the qwen scorer reads — so scoring/export/demos
cascaded (heads themselves trained fine). Fix script ran the missed tail;
lesson: the escore step is not optional even though ECAPA numbers are retired,
because it produces `score_work/`.

**v4 heads** (511-pair bank, 3000 mfg v4 pairs, features_all4, old center):
val mse 2.42 → **2.09**; centered-qwen fh4_d10 **0.5163** / fh4_d15 0.5103
over 71 voices (v3: 0.5240/58 — larger, harder val set incl. the new
character/excited classes; not directly comparable). Goal line 0.4905.

**Paired instant-clone scores** (demo vs ref, centered qwen — the real
before/after):
| voice | instant v3 | instant v4 |
|---|---|---|
| bubbly | 0.463 | **0.535** |
| my mistress | ~0.52 | **0.629** |
| ray porter | 0.513 | 0.526 |

Coverage wave moved its target class (+0.07 bubbly) and *improved* whisper
(+0.11) — no catastrophic forgetting.

**Ray Porter ladder complete:** v4 refine 0.479→0.813 (21s ref); inversion
best **0.9447 ECAPA @300** — but its held-out demo scores only **0.398
centered-qwen**: the referee flip again, now on the strongest inversion yet.
ECAPA-optimal ≠ qwen-optimal, and the ear has sided with qwen every time.
Delivered to Telegram: Ray INVERTED + bubbly/mm INSTANT v4 (JSONs + demos);
qwen-preferred Ray remains REFINED v2 (0.829, v3-basis ceiling).

**Ear verdict on Ray INVERTED: "does not sound right"** — the third
consecutive ear-loss for ECAPA-optimizing inversion (bubbly, and by score
mymistress's inversion was never ear-preferred over its refine either).
**Referee flip is conclusive, and it indicts the bank labels themselves:**
every (embedding, style) training pair comes from ECAPA+mel inversion, so
the labels are pulled toward ECAPA-optimal rather than sounds-right styles.
Next structural move: make the label generator qwen-scored — either the
refine flywheel (qwen-CMA refined styles as labels; forward-only, already
proven) or a differentiable qwen speaker loss inside invert.py.

**Ray refine rounds 3–4 (v4 basis):** round 3 (200×24, seeded from v2's
0.829) → **0.847**, full 200 gens, no plateau; ecapa *fell* 0.554→0.501 as
qwen rose — the objectives actively diverge. Round 4 (400 gens, seeded from
round 3) early-stopped at 26 gens / 0.840 → **converged; ~0.85 is the
v4-basis ceiling for this voice** (v3 basis: 0.829 — wave 3 measurably
raised a specific voice's ceiling). Accidental control: a mis-seeded round 4
searched from scratch and reached only 0.781 in 172 gens — seeding from the
best available style is worth ~0.07 (Dale lesson, quantified).

**App:** speech-speed slider shipped (`ce7223c`, CI green): Jobs-tab SeekBar
0.6×–1.6× persisted as speech_speed_pct; TtsService multiplies the engine's
published 1.05 pace at synthesis time (jobs, live, hosted alike). The
engine's generate() had the parameter all along; nothing exposed it.

## 2026-08-14 — Flywheel night launched (🔄)

**The structural response to the referee-flip finding**: stop using ECAPA
inversions as labels for expressive voices. Running now (`run_flywheel.sh`):

1. **Qwen-relabel** all 224 expressive refs (exp/ears/wave3) + the 3 user
   voices (Ray Porter 90s, bubbly, mymistress — their qwen-refined styles
   are the seeds, so the search continues from the best known point; bank
   styles seed the rest). Both GPUs, ~113 refs each, iters 120 patience 25.
2. **Bank v5**: relabeled styles replace the ECAPA-inversion labels
   (`relabel_to_bank.py`; cos ← qwen_held, the label-quality weight
   train_translation already uses); user voices join as new pairs.
   LibriSpeech/VCTK read-speech labels stay ECAPA for now (least harmed;
   full relabel is a future night).
3. Basis v5, mfg v5 (3000), features_all5 (old center kept), fh5_d10/d15
   **with the score_styles render step** (the wave-3 lesson), export_v5,
   instant-v5 demos for the 3 test voices, Telegram notify per phase.

Known limitation, accepted: relabeled styles live inside span(v4 basis), so
v5's basis can re-weight but not expand the span — span growth still comes
from inversion waves; label *quality* is what this night buys.

**Expresso re-download started** (38 GB; the earlier tar didn't survive on
disk) → staging + qwen-labeling is the next wave. TED-LIUM skipped:
LibriSpeech already covers read narration.

## 2026-08-15 — Flywheel results: the biggest jump yet (✅, 🔄 ear)

**All 227 refs qwen-relabeled** (4 workers, 2 per GPU after mid-run
rebalances; ~5 min avg each). Bank v5 = 514 pairs: 224 expressive labels
replaced (ECAPA-inversion → qwen-CMA), user voices (rayporter_long, bubbly,
mymistress) added as first-class pairs. val mse 2.09 → **2.04**.

**Centered-qwen, val heads (old center):**
| head | score | val set |
|---|---|---|
| v3 (fh_d10) | 0.5240 | 58 |
| v4 (fh4_d10) | 0.5163 | 71 |
| **v5 (fh5_d10)** | **0.5883** | **77** |

Label quality was worth **+0.072** — more than any data wave. The referee
flip finding cashed out: training on ear-endorsed labels moves the encoder
more than adding voices did.

**Paired instant-clone scores** (demo vs ref):
| voice | v3 | v4 | **v5** |
|---|---|---|---|
| ray porter | 0.513 | 0.526 | **0.669** |
| bubbly | 0.463 | 0.535 | 0.536 |
| my mistress | ~0.52 | 0.629 | **0.746** |

Ray +0.14 and mymistress +0.12 in one night (both are IN the bank now —
partly memorization, but that is the fold-in working as designed: known
voices clone near-perfectly instantly). Bubbly flat: her label improved
only 0.69→0.74 — the voice itself sits at the basis edge.

Instant-v5 JSONs + demos delivered to Telegram. Expresso tar (38 GB)
fully downloaded, unstaged. GPUs idle.

**Ear verdict: Ray instant v5 "not better"** despite the paired metric's
+0.14 — the demo-vs-ref qwen score and the ear diverge at the instant tier
for this voice (the metric may be partly rewarding bank memorization).
User shipped v5 anyway for the CI build. Deployed: app `8e25be2` — and the
HEAD filename is now versioned too (style_encoder_qwen_v5.onnx): the
architecture is fixed, so every retrain is byte-size-identical, and v5 would
never have reached existing installs under the old name (same trap as the
basis, caught at deploy time). Fallback chains: head v5→legacy, basis
v5→v3→legacy. Cloner repo `5fe45b8`.

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
