#!/usr/bin/env bash
# The pipeline, in order — a recipe, not a turnkey build. Each step assumes the
# workstation setup (a frozen Supertonic 3 + DiffSynth, PyTorch/CUDA for the
# inversion, SpeechBrain ECAPA, and — for the qwen analyzer — a Qwen3-TTS
# checkpoint). Paths are placeholders; wire them to your corpora/models.
set -euo pipefail
PY=${PY:-python}
TOOL=python

# 1) real-speaker styles: gradient-invert a corpus (the ~0.82 ground truth)
$PY $TOOL/invert_corpus.py --corpus /path/to/LibriSpeech --out inversions/ --gpu 0
#    a curated set of named voices, same idea
$PY $TOOL/clone_library.py --refs /path/to/refs --out clone_out/library/

# 2) fit the PCA style basis on the inverted styles (+ presets)
#    k=128 captures 89.4% of variance; k=384 captures 96.1% and is what reaches
#    0.80-0.83 held-out (matching the desktop inversion). Export the phone basis:
$PY $TOOL/export_k384.py            # -> style_basis_k384.bin  (k=384, 20 MB)
#    (k-sweep that measured the ceiling vs k: 128->0.44, 256->0.50, 384->0.53 projection)
$PY $TOOL/basis_k_sweep.py

# 3) manufacture (audio, coefficient) supervision across the whole basis
$PY $TOOL/gen_style_pairs2.py --pairs-npz all_pairs.npz --count 6000 --out-dir mfg_wavs/

# 4) speaker embeddings for the real recordings (ECAPA and/or Qwen)
$PY $TOOL/embed_wavs.py --wavs inversions/refs --out real_ecapa.npz
$PY $TOOL/extract_qwen_features.py --wavs inversions/refs --out real_qwen.npz

# 5) train the translation head (embedding -> style coefficients)
$PY $TOOL/train_translation.py --pairs-npz all_pairs.npz --k 128 --steps 10000

# 6) bake the on-device artifacts
$PY $TOOL/export_cloner.py --head trans_ecapa.pt --out ../models/style_encoder.onnx
$PY $TOOL/export_basis.py  --head trans_ecapa.pt --out ../models/style_basis.bin

# 7) forward-only refine (the on-device loop; ported to Kotlin in the app).
#    Over the k=384 basis this reaches 0.80-0.83:
$PY $TOOL/cma_polish.py --refs ref.wav --basis-extra aux_pairs.npz \
     --start-styles clone_out/library/styles --k 384 --iters 150 --pop 24

# 8) evaluation
$PY $TOOL/eval_encoder_real.py     # held-out speaker cosine
$PY $TOOL/basis_ceiling.py         # how much of an inversion the basis expresses

# Generating a *new* voice (no target): sample the basis, decode, synthesise.
$PY $TOOL/gen_named_library.py     # a named 20M/20F library
