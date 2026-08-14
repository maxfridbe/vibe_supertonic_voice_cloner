# Trained models

The on-device graphs the encoder path and the refine search use. All produced by
the scripts in `../python`.

| File | What | Shapes |
|---|---|---|
| `style_encoder_qwen.onnx` | **the shipped head** — "fh_d10": PCA-256 front + 512×2 MLP, trained on the 418-pair expressive bank (24 MB) | `embedding [1, 2048]` (Qwen3-TTS speaker features) → `style_ttl [1, 50, 256]`, `style_dp [1, 8, 16]` |
| `qwen_center.bin` | population mean of Qwen speaker features — subtract before cosine (the project metric) | `2048 × f32`, little-endian |
| `style_basis.bin` | v3 PCA style basis (k=384, expressive refit) for the refine search | little-endian: `int32` header (k, D, split, ttl_r, ttl_c, dp_r, dp_c) then `scale[k]`, `mean[D]`, `basis[k·D]` |
| `spk_encoder.onnx` | ECAPA-TDNN speaker encoder (comparison variant) | `wav [1, N]` (16 kHz mono) → `[1, 192]` |
| `style_encoder.onnx` | ECAPA translation head (2048×4 MLP + folded basis, 75 MB) | `embedding [1, 192]` → the same style tensors |

The Qwen waveform-to-embedding graph (`qwen_spk_encoder.onnx`, 49 MB) lives in
the app repo's `models/cloner`; `python/export_qwen_spk.py` produces it.

Numbers under the project metric (centered Qwen cosine, `python/qwen_similarity.py`):
the shipped head scores **0.524** held-out over 58 voices — above the **0.4905**
the desktop gradient inversions score on themselves; the qwen-objective refine
reaches mean **0.840** over 18 voices. (Legacy ECAPA-cosine framing: encoder
0.43–0.49, refine 0.72–0.83, inversion ~0.82.)

No model here is a real person: the encoder was trained on inverted corpus
speakers and manufactured basis samples, and it emits a style, not a recording.
