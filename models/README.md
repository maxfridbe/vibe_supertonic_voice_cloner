# Trained models

The on-device graphs the encoder path and the refine search use. All produced by
the scripts in `../python`.

| File | What | Shapes |
|---|---|---|
| `spk_encoder.onnx` | ECAPA-TDNN speaker encoder | `wav [1, N]` (16 kHz mono) → `[1, 192]` |
| `style_encoder.onnx` | translation head | `embedding [1, 192]` → `style_ttl [1, 50, 256]`, `style_dp [1, 8, 16]` |
| `style_basis.bin` | PCA style basis for the refine search | little-endian: `int32` header (k, D, split, ttl_r, ttl_c, dp_r, dp_c) then `scale[k]`, `mean[D]`, `basis[k·D]` |

A Qwen3-TTS analyzer variant (2048-d embedding) also exists in the app repo
(`qwen_spk_encoder.onnx` + `style_encoder_qwen.onnx`); it trades a little cosine
for a listening-preferred result on character voices.

Held-out speaker cosine (encoder alone): 0.43–0.49. With the on-device refine on
top: 0.60–0.77. The desktop gradient inversion these are trained toward: ~0.82.

No model here is a real person: the encoder was trained on inverted corpus
speakers and manufactured basis samples, and it emits a style, not a recording.
