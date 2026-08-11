# Supertonic Voice Cloner

Turn a voice recording into a [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
**style** — the `style_ttl [1,50,256]` + `style_dp [1,8,16]` tensors the
synthesiser speaks with — and do it cheaply enough to run on a phone.

This repo is the methodology, the trained models, the Python that produced them,
and a small **Rust `clonevoice`** program that takes a `.wav` and writes the
style `.json`.

> Supertonic ships no speaker encoder: its published weights can synthesise from
> a style, but nothing turns a *recording* into one. Everything here exists to
> fill that gap, at three different points on the quality/compute curve.

## The three paths

| Path | Where it runs | Similarity* | How |
|---|---|---|---|
| **Gradient inversion** | desktop GPU, autograd | **~0.82** | optimise the style tensors directly against the recording |
| **Encoder** (this repo's Rust tool) | phone, 2 forward passes | ~0.43–0.49 | a net predicts the style from a speaker embedding |
| **Refine** (CMA polish) | phone, forward-only search | ~0.60–0.77 | gradient-free search over a style basis, seeded by the encoder |

<sub>*held-out ECAPA speaker cosine to the reference recording, not the sentence being fitted.*</sub>

### 1. Gradient inversion — the quality reference

Freeze the whole Supertonic pipeline (duration predictor → text encoder → flow →
vocoder), make the style tensors the only trainable parameters, and minimise a
speaker-similarity loss (ECAPA cosine) between the synthesiser's output and the
target recording. A few hundred Adam steps land at **~0.82** held-out. It needs
autograd and a GPU — thousands of times more compute than *using* the result —
so it stays a desktop step. `python/invert_corpus.py` and
`python/clone_library.py` drive it.

### 2. Encoder — sub-second, on-device

Gradient inversion in two forward passes instead of hundreds of backward ones:

```
recording ──► spk_encoder.onnx ──► 192-d speaker embedding ──► style_encoder.onnx ──► (style_ttl, style_dp)
```

- **`spk_encoder.onnx`** is an ECAPA-TDNN speaker-verification network exported to
  ONNX: 16 kHz mono in, a 192-number identity embedding out. (A Qwen3-TTS
  variant, 2048-d, trades cosine for a listening-preferred result.)
- **`style_encoder.onnx`** is the *translation head*: a small MLP trained to map
  that embedding to the style tensors.

The head is trained on `(embedding, style)` pairs from two sources:
- **Real inversions** — run path 1 over a speech corpus (`invert_corpus.py`) to
  get real-speaker styles, and embed the same recordings.
- **Manufactured pairs** — `gen_style_pairs2.py` samples a coefficient in a PCA
  **style basis**, decodes it to a style, synthesises a sentence, and embeds
  *that*. Thousands of `(audio, coefficient)` pairs teach the head the whole
  basis, not just the handful of real speakers.

The PCA basis (`style_basis.bin`, `train_encoder.py`'s `StyleBasis`) is fit on
the inverted real-speaker styles; at k=128 it captures 89.4 % of their variance.
**It is the ceiling**: a style outside the basis span can't be expressed, so the
encoder's held-out score (~0.49) is bounded by how much of a real voice the
basis can represent.

**Raising k breaks the ceiling.** At k=384 the basis captures 96.1 % of the
variance, and the refine search over it reaches the desktop reference:

| Voice | refine @k=128 | refine @k=384 |
|---|---|---|
| Soothing female british | 0.771 | **0.826** |
| Fireside Narrator | 0.756 | **0.802** |
| Stephen Fry | 0.652 | 0.758 |
| Dale | 0.759 | 0.716 |

`export_k384.py` writes the wider basis; `basis_k_sweep.py` and
`basis_ceiling.py` are the diagnostics. The shipped `models/style_basis.bin`
here is the k=384 one.

### 3. Refine — forward-only search, on-device

The encoder lands *near* a speaker, not on them. Gradient inversion would close
the gap but needs autograd; a phone has neither. So search the same basis with a
gradient-free optimiser — **separable CMA-ES** (`cma_polish.py`): decode a
candidate coefficient vector → synthesise a short probe → embed it → score
cosine to the reference → hill-climb. Every evaluation is one forward synthesis
plus one embedding, both of which a phone already runs. ~1300 evaluations lift
character voices from ~0.2 to ~0.65–0.77 — most of the way to the desktop
reference, using only forward passes. This is the loop the TTS Runner app runs
in the background as "Refine this voice".

## Repo layout

```
models/          the trained on-device graphs
  spk_encoder.onnx     ECAPA speaker encoder (wav → 192-d)
  style_encoder.onnx   translation head (embedding → style tensors)
  style_basis.bin      PCA style basis for the refine search
python/          the scripts that produce all of the above (see python/README.md)
rust/clonevoice/ a sample CLI: wav in, style JSON out (the encoder path)
```

## Quick start (Rust `clonevoice`)

```sh
cd rust/clonevoice
cargo build --release
# encoder path (~0.5, sub-second):
./target/release/clonevoice reference.wav out.json --models ../../models --name "My Voice"
# + refine to ~0.8 (needs the Supertonic graphs; see rust/clonevoice/README.md):
./target/release/clonevoice reference.wav out.json --models ../../models \
    --supertonic /path/to/supertonic-3 --refine --iters 120 --pop 20
```

`out.json` is schema-identical to Supertonic's published styles, so
[TTS Runner](https://github.com/maxfridbe/vibe_android_tts_runner) (or any
Supertonic client) plays it with no changes: **Speakers → Import style**.

## The style JSON

```json
{
  "style_ttl": { "dims": [1, 50, 256], "data": [ ... 12800 floats ... ] },
  "style_dp":  { "dims": [1, 8, 16],   "data": [ ... 128 floats ... ] },
  "metadata":  { "name": "...", "source": "...", "reference": "..." }
}
```

`style_ttl` conditions the flow/vocoder (timbre), `style_dp` the duration
predictor (pace/rhythm); both are L2-normalised per row.

## License

Apache-2.0. The models are derived from Supertonic 3 (its own license) and an
ECAPA-TDNN speaker encoder ([SpeechBrain](https://github.com/speechbrain/speechbrain),
Apache-2.0). No cloned voice here is a real person.
