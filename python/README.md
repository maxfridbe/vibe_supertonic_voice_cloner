# Pipeline scripts

The Python that produces the models in `../models`. They run against a frozen
Supertonic 3 (its ONNX graphs + the `DiffSynth` wrapper from
[Mimocro/supertonic-voice-cloning](https://github.com/Mimocro/supertonic-voice-cloning)),
PyTorch (autograd for the inversion), [SpeechBrain](https://github.com/speechbrain/speechbrain)
for the ECAPA speaker embedding, and — for the Qwen analyzer — a Qwen3-TTS
checkpoint. Paths and corpora are wired for the original workstation; treat
these as the reference implementation, not a turnkey `make`.

## The order they run in

1. **`invert_corpus.py`** — gradient inversion over a speech corpus. Freezes
   Supertonic, makes the style tensors trainable, minimises ECAPA cosine to each
   reference recording. Output: real-speaker styles (~0.82 held-out) — the
   ground truth everything else is trained toward. `clone_library.py` is the
   same for a hand-picked set of voices.

2. **`train_encoder.py`** — fits the PCA **`StyleBasis`** on the inverted styles
   (SVD; k=128 ≈ 89.4 % variance) and, optionally, trains an encoder head. The
   basis is the search space for refine and the ceiling for the whole approach.

3. **`gen_style_pairs2.py`** — manufactures `(audio, coefficient)` supervision:
   sample a coefficient in the basis, decode to a style, synthesise a sentence.
   Thousands of these teach the head the *whole* basis, not just the real
   speakers. (`gen_speaker_bank.py` / `gen_style_pairs.py` are earlier variants.)

4. **`extract_qwen_features.py`** / **`embed_wavs.py`** — speaker embeddings for
   the recordings: Qwen3-TTS's 2048-d conditioning, or ECAPA's 192-d.

5. **`train_translation.py`** — trains the **translation head**: embedding →
   style (via basis coefficients). Dropout + feature noise so it generalises off
   the manufactured manifold.

6. **`export_cloner.py`** / **`export_basis.py`** / **`export_translation.py`** —
   bake the trained pieces into the on-device artifacts: `spk_encoder.onnx`,
   `style_encoder.onnx`, `style_basis.bin`.

7. **`cma_polish.py`** — the forward-only **refine**: separable CMA-ES over the
   basis coefficients, seeded by the head, scored by synthesise→embed→cosine.
   This exact loop is ported to Kotlin in the Android app. Raising `--k`
   (e.g. 384) searches a bigger basis and lifts several voices past 0.8.

## Evaluation

- **`eval_style.py`** / **`eval_encoder_real.py`** — held-out speaker cosine on
  sentences the styles were *not* fitted on (the only score that ranks runs).
- **`basis_ceiling.py`** — the key diagnostic: project a known-good inversion
  through the basis and measure what survives. If projection destroys
  similarity, the basis is the ceiling and the fix is a bigger/refit basis, not
  more training.
- **`score_styles.py`** — batch scoring helper.

## Measured

| Stage | Held-out cosine |
|---|---|
| Encoder (2 forward passes) | 0.43–0.49 |
| + Refine, ~1300 evals (k=128) | 0.60–0.77 |
| + Refine over a k=384 basis | Soothing 0.826, Fireside 0.802, Stephen Fry 0.758 |
| Desktop gradient inversion | ~0.82 |
