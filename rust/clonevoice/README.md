# clonevoice

Clone a voice from a WAV into a Supertonic style JSON — the encoder path (two
ONNX forward passes, no training, no autograd), the same pipeline the Android
app runs.

```
wav ──► spk_encoder.onnx ──► 192-d embedding ──► style_encoder.onnx ──► style JSON
```

## Build

`ort` loads ONNX Runtime dynamically (its bundled-download build script is
currently broken against ureq 3.4), so fetch the runtime once and point at it:

```sh
# ONNX Runtime (Linux x64 shown; use the build for your OS)
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz | tar xz
export ORT_DYLIB_PATH="$PWD/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so"

cargo build --release
```

## Run

```sh
./target/release/clonevoice reference.wav out.json --models ../../models --name "My Voice"
```

- `reference.wav` — any WAV (mono/stereo, any rate; it downmixes + resamples to
  16 kHz and uses the first ~12 s).
- `out.json` — a Supertonic style, schema-identical to the published styles.
  Import it in TTS Runner: **Speakers → Import style**.

This is the *encoder* path (~0.43–0.49 held-out). For higher fidelity, refine it
on-device in the app, or use the desktop gradient inversion (`../../python`).

## Refine (reach ~0.8, like the app)

The encoder path lands near a speaker (~0.5). `--refine` then runs the forward-
only CMA search over the k=384 basis to close the gap — the same loop the app
runs in the background. It needs the Supertonic graphs:

```sh
# the four Supertonic ONNX graphs + tts.json + unicode_indexer.json
# (from https://huggingface.co/Supertone/supertonic-3, or the app's model cache)
./target/release/clonevoice reference.wav out.json \
    --models ../../models --supertonic /path/to/supertonic-3 \
    --refine --iters 120 --pop 20
```

Each generation prints the best cosine so far. The full budget (iters 120 ×
pop 20 ≈ 2400 evaluations) reaches 0.80–0.83 held-out on many voices; a small
budget (`--iters 4 --pop 8`) is a quick smoke test. Everything is forward passes
— no autograd, no GPU.
