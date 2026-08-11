# Method: WAV → style JSON

How a recording becomes a Supertonic **style** (`style_ttl [1,50,256]` +
`style_dp [1,8,16]`), the way `clonevoice` and the Android app do it.
Supertonic's published models can synthesise *from* a style, but nothing turns a
*recording into* one — this pipeline fills that gap. The [README](README.md) has
the higher-level map; this is the detailed walk-through.

The whole method is one idea: **guess, then listen and correct.** A trained head
makes a fast guess at the style — but it never hears its own output, so it lands
*near* the voice, not *on* it. A search loop then closes the gap the only way a
phone can: synthesise a candidate, listen to it, score it, adjust.

## The five steps

```
        ┌────────────── encoder path (sub-second) ──────────────┐   ┌── refine (minutes → ~1 hr) ──┐
 WAV ─► ① prepare ─► ② embed ─► ③ translate ─► first JSON ─► ④ search ─► ⑤ decode ─► refined JSON
```

| # | Step | In → out | Cost | What you have after |
|---|---|---|---|---|
| ① | **Prepare** | WAV → 16 kHz mono, first ~12 s | ~0 | audio the encoder expects |
| ② | **Embed** — `spk_encoder.onnx` | audio → `[1,192]` identity | 1 forward pass | *who* is speaking, nothing else |
| ③ | **Translate** — `style_encoder.onnx` | `[1,192]` → `style_ttl` + `style_dp` | 1 forward pass | **first JSON** (~0.43–0.49) |
| ④ | **Search** — separable CMA-ES | first JSON + WAV → best basis coefficients | ~2,400 syntheses | coefficients that *sound* like the target |
| ⑤ | **Decode** | coefficients → style tensors | ~0 | **refined JSON** (0.72–0.83) |

Steps ①–③ are the **encoder path**: two ONNX forward passes, no search, a
playable voice in under a second. Steps ④–⑤ are the **refine**: forward-only
search, minutes multi-core on a desktop, ~an hour on a phone. Scores are
held-out ECAPA speaker cosine to the reference.

---

## ① Prepare the audio

Downmix to mono, resample to 16 kHz (what the speaker encoder expects), keep the
first ~12 s. Identity information saturates after a few seconds of speech, so
more audio only costs time.

## ② Embed — recording → a 192-number identity

`spk_encoder.onnx` is an **ECAPA-TDNN**, a speaker-*verification* network (the
kind used to tell whether two clips are the same person). It turns the waveform
into mel-spectrogram features, runs them through time-delay convolutions with
attention, and attentively pools the whole clip into one fixed vector:

```
16 kHz waveform  ──►  [1, 192]   (unit-normalized)
```

Those 192 numbers are **pure identity** — trained to keep only what separates
one speaker from another (pitch register, vocal-tract timbre, accent, delivery)
and discard the words, the room, the noise. Same person → close on the 192-d
unit sphere; different people → far apart.

> A second analyzer option — Qwen3-TTS's own 2048-d speaker encoder — trades a
> little cosine score for a listening-preferred result on character voices. Same
> contract, different "ears."

## ③ Translate — identity → first JSON

`style_encoder.onnx` is a **small trained MLP, the "translation head."** It maps
the 192-d identity to the two style tensors the synthesiser speaks with:

```
[1, 192]  ──►  style_ttl [1, 50, 256]   (timbre — conditions the flow + vocoder)
          ──►  style_dp  [1, 8, 16]      (pace/rhythm — conditions the duration predictor)
```

Each output row is L2-normalized, because Supertonic's real style vectors live
on a per-row unit sphere and the synthesiser assumes it. The tensors serialize
straight into the Supertonic style format — a playable voice immediately:

```json
{ "style_ttl": { "dims":[1,50,256], "data":[ …12800 floats… ] },
  "style_dp":  { "dims":[1,8,16],   "data":[ …128 floats… ] },
  "metadata":  { "source": "encoder path", … } }
```

**How the head learned the mapping:** it was trained on `(embedding → style)`
pairs where the styles come from desktop gradient inversions of a speech corpus
(real speakers, ~0.82 styles) plus thousands of manufactured pairs — sample a
point in the style basis, decode it, synthesise it, embed *that*. So it
regresses "given this identity, what style would a real inversion have
produced?"

**Why it stops at ~0.43–0.49** (stylised character voices can start lower): it
is a single feed-forward regression with no feedback. It can only emit styles
inside its basis (the ceiling), and it never *hears* its own output, so it
can't tell the voice is a bit off and correct it. It lands in the right
neighborhood — recognizably that kind of voice — but not on the person.

## ④ Search — listen and correct

The first style is one point in a **384-dimension PCA "style basis."** The
refine searches that basis for a point whose *synthesized audio actually sounds
like the target*:

1. **Embed the reference WAV once** → the target embedding (what to match).
2. **Encode** the first-JSON style into 384 basis coefficients → the start point.
3. **Separable CMA-ES loop** (a gradient-free optimizer). Each generation
   samples ~20 candidate coefficient vectors near the current best; each
   candidate is decoded to a style, **synthesized** as a short fixed probe with
   the full Supertonic pipeline, **embedded** with ECAPA, and **scored** by
   cosine to the target. The optimizer walks toward the higher-scoring
   candidates and tightens its search each generation. ~120 generations ≈
   **2,400 evaluations**.

**Why it runs on a phone:** every evaluation is a *forward* pass — synthesise,
embed, score. No gradients, no autograd (which the desktop gradient-inversion
method needs and a phone can't do). It reaches the answer by listening to its
own output and adjusting.

## ⑤ Decode — best coefficients → refined JSON

Take the best coefficients from the search, decode them through the basis back
into `style_ttl` + `style_dp`, and write the same JSON format as step ③ —
schema-identical to Supertonic's published styles.

---

## The basis is the ceiling

The refine can only reach styles the basis can *express*:

| Basis | Style variance captured | Refine reaches |
|---|---|---|
| k = 128 | 89.4 % | ~0.60–0.77 (plateau) |
| **k = 384** (shipped) | **96.1 %** | **0.72–0.83** |

At k=384 the best voices match the desktop gradient inversion (~0.82): Soothing
0.826, Fireside 0.802; harder voices land lower (Stephen Fry 0.758), and one
(Dale, 0.716) did better in the smaller basis — a wider space is harder to
search on the same budget. Beyond the basis, only the desktop gradient
inversion (unconstrained, autograd + GPU) does better — which is why that stays
a desktop step.

## Models in play

| File | Used in step | Role | Shapes |
|---|---|---|---|
| `spk_encoder.onnx` | ②, ④ | ECAPA speaker encoder | `wav [1,N]` (16 kHz) → `[1,192]` |
| `style_encoder.onnx` | ③ | translation head | `[1,192]` → `ttl [1,50,256]`, `dp [1,8,16]` |
| `style_basis.bin` | ④, ⑤ | PCA style basis (the search space) | k=384, D=12928 (split 12800) |
| Supertonic 3 graphs | ④ | duration predictor / text encoder / vector estimator / vocoder — the synth that *voices* each candidate | dynamic `batch_size` |
