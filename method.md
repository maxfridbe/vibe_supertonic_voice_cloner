# Method: WAV → first JSON → refined JSON

How a recording becomes a Supertonic **style** (`style_ttl [1,50,256]` +
`style_dp [1,8,16]`), the way `clonevoice` and the Android app do it. It's a
two-stage pipeline: a fast rough guess (the **encoder path**), then a slow
polish (the **refine**). This document is the detailed walk-through; the
[README](README.md) has the higher-level map.

```
                       Stage 1 (sub-second)                 Stage 2 (minutes → ~1 hr)
 recording ─► spk_encoder ─► 192-d id ─► style_encoder ─► first JSON ─► CMA search over the basis ─► refined JSON
              (ECAPA)                    (trans. head)     (~0.3–0.5)      (synthesise→embed→score)     (0.80–0.83)
```

Supertonic's published models contain **no speaker encoder** — they can
synthesise *from* a style but nothing turns a *recording into* one. Everything
below exists to fill that gap.

---

## Stage 1 — WAV → first JSON (encoder path)

Two ONNX models, two forward passes, no search.

### 1. Prepare the audio
Read the WAV, **downmix to mono**, **resample to 16 kHz** (what the speaker
encoder expects), keep the **first ~12 s**. Identity information saturates after
a few seconds of speech, so more audio only costs time.

### 2. `spk_encoder.onnx` — recording → a 192-number identity
An **ECAPA-TDNN**, a speaker-*verification* network (the kind used to tell
whether two clips are the same person). It turns the waveform into
mel-spectrogram features, runs them through time-delay convolutions with
attention, and **attentively pools** the whole clip into one fixed vector:

```
16 kHz waveform  ──►  [1, 192]   (unit-normalized)
```

Those 192 numbers are **pure identity** — trained to keep only what separates
one speaker from another (pitch register, vocal-tract timbre, accent, delivery)
and discard the words, the room, the noise. Same person → close on the 192-d
unit sphere; different people → far apart. Nothing about *what* was said
survives.

> A second analyzer option — Qwen3-TTS's own 2048-d speaker encoder — trades a
> little cosine score for a listening-preferred result on character voices. Same
> contract, different "ears."

### 3. `style_encoder.onnx` — identity → Supertonic style
A **small trained MLP, the "translation head."** It maps the 192-d identity to
the two style tensors the synthesiser speaks with:

```
[1, 192]  ──►  style_ttl [1, 50, 256]   (timbre — conditions the flow + vocoder)
          ──►  style_dp  [1, 8, 16]      (pace/rhythm — conditions the duration predictor)
```

Each output row is L2-normalized, because Supertonic's real style vectors live
on a per-row unit sphere and the synthesiser assumes it.

**How it learned that mapping** (why the first JSON is what it is): the head was
trained on `(embedding → style)` pairs where the styles come from **desktop
gradient inversions of a speech corpus** (real speakers, ~0.82 styles) plus
thousands of **manufactured pairs** — sample a point in the style basis, decode
it, synthesise it, embed *that*. So it regresses "given this identity, what's the
style a real inversion would have produced?"

### 4. Write the first JSON
The two tensors serialize straight into the Supertonic style format — a playable
voice immediately:

```json
{ "style_ttl": { "dims":[1,50,256], "data":[ …12800 floats… ] },
  "style_dp":  { "dims":[1,8,16],   "data":[ …128 floats… ] },
  "metadata":  { "source": "encoder path", … } }
```

### Why it's only ~0.3–0.5
A **single feed-forward regression with no feedback.** It predicts the *most
likely* style for that identity, but (a) it can only emit styles inside its basis
(the ceiling), and (b) it never *hears* its own output, so it can't tell the
voice is a bit off and correct it. It lands "in the right neighborhood" —
recognizably that kind of voice — but not on the person. Closing that gap is
Stage 2.

---

## Stage 2 — first JSON → refined JSON (forward-only CMA search)

The first style is one point in a **384-dimension PCA "style basis."** The refine
searches that basis for a point whose *synthesized audio actually sounds like the
target*.

1. **Embed the reference WAV once** → the target embedding (what to match).
2. **Encode** the first-JSON style into 384 basis coefficients → the start point.
3. **Separable CMA-ES loop** (a gradient-free optimizer). Each generation samples
   ~20 candidate coefficient vectors near the current best, and for each
   candidate:
   - decode coefficients → a style,
   - **synthesize a short fixed probe** with the full Supertonic pipeline
     (duration predictor → text encoder → flow steps → vocoder),
   - **embed** that synthesized audio with ECAPA,
   - **score cosine** to the target.
   The optimizer walks toward the higher-scoring candidates and tightens its
   search each generation. ~120 generations ≈ **2,400 evaluations**.
4. Take the best coefficients → decode → the **refined JSON**.

**Why it runs on a phone:** every step is a *forward* pass — synthesise, embed,
score. No gradients, no autograd (which the desktop gradient-inversion method
needs and a phone can't do). It reaches the same answer by *listening to its own
output* and adjusting. Costs ~an hour on a phone, a few minutes multi-core on a
desktop, forward passes only.

### The basis is the ceiling
The refine can only reach styles the basis can *express*:

| Basis | Style variance captured | Refine reaches |
|---|---|---|
| k = 128 | 89.4 % | ~0.77 (plateau) |
| **k = 384** (shipped) | **96.1 %** | **0.80–0.83** |

Raising `k` recovers the residual where a real voice lives. Beyond the basis,
only the desktop gradient inversion (~0.82, unconstrained) does better — and it
needs autograd + a GPU, which is why it stays a desktop step.

---

## Models in play

| File | Role | Shapes |
|---|---|---|
| `spk_encoder.onnx` | ECAPA speaker encoder | `wav [1,N]` (16 kHz) → `[1,192]` |
| `style_encoder.onnx` | translation head (the first JSON) | `[1,192]` → `ttl [1,50,256]`, `dp [1,8,16]` |
| `style_basis.bin` | PCA style basis (the refine's search space) | k=384, D=12928 (split 12800) |
| Supertonic 3 graphs | `duration_predictor` / `text_encoder` / `vector_estimator` / `vocoder` — the synth used to *score* each candidate | dynamic `batch_size` |

The style JSON: `style_ttl` conditions the flow/vocoder (timbre), `style_dp` the
duration predictor (pace); both are L2-normalized per row, schema-identical to
Supertonic's published styles.
