#!/usr/bin/env python3
"""Fit the PCA style basis and export style_basis.bin, generically.

This is the checked-in, portable version of what `export_k384.py` did on the
original workstation with hardcoded paths. It needs only numpy.

## Methodology

The basis is a plain PCA over a bank of Supertonic styles:

1. Every input style is flattened to one row of D = 12928 floats
   (`style_ttl` 50x256 = 12800, then `style_dp` 8x16 = 128).
2. The rows are stacked (optionally balanced, below), the mean is taken, and
   the centered matrix is SVD'd.
3. `basis`  = the top-k right singular vectors (k rows of D)
   `scale`  = s_i / sqrt(N-1)  — the per-component standard deviation, so a
   coefficient of 1.0 means "one std along this component"
   `decode(c) = mean + (c * scale) @ basis`, with per-row L2 normalisation
   applied by the consumer (the synthesiser assumes styles live on per-row
   unit spheres).

The exact same math as `train_encoder.StyleBasis`; the shipped consumers
(`rust/clonevoice/src/refine.rs::Basis`, the app's `RefineEngine.loadBasis`)
read the binary this writes.

## Input sources

Two forms, mix freely, each repeatable:

  --npz FILE          an archive with `ttl` [N,50,256] and `dp` [N,8,16]
                      arrays — the format `invert_corpus.py` banks
                      (aux_pairs.npz) and `gen_style_pairs2.py` produces
  --styles-dir DIR    a directory of Supertonic style .json files

## Balancing

With `--balance`, each source after the first is repeated
round(N_first / N_source) times (at least once), so a small set of precious
real-speaker inversions carries roughly the same total weight as thousands of
manufactured pairs. The shipped k=384 basis was fit this way: ~6000
manufactured pairs balanced against ~70 LibriSpeech inversions.

## Binary layout (matches models/README.md)

  int32 x7 little-endian: k, D, split, ttl_r, ttl_c, dp_r, dp_c
  float32 scale[k], mean[D], basis[k*D]   (basis row-major, k rows of D)

## Example — the k=384 recipe with the VCTK-extended bank

  make_basis.py --npz pairs_p1.npz \
                --npz inversions/aux_pairs.npz \
                --npz inversions_vctk/aux_pairs.npz \
                --balance --k 384 --out style_basis.bin
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

TTL_SHAPE = (50, 256)
DP_SHAPE = (8, 16)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    ttl = np.asarray(d["ttl"], dtype=np.float32)
    dp = np.asarray(d["dp"], dtype=np.float32)
    return ttl.reshape(len(ttl), *TTL_SHAPE), dp.reshape(len(dp), *DP_SHAPE)


def load_styles_dir(path):
    ttls, dps = [], []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        o = json.load(open(f))
        ttls.append(np.array(o["style_ttl"]["data"], dtype=np.float32).reshape(TTL_SHAPE))
        dps.append(np.array(o["style_dp"]["data"], dtype=np.float32).reshape(DP_SHAPE))
    if not ttls:
        raise SystemExit(f"no style .json files in {path}")
    return np.stack(ttls), np.stack(dps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", action="append", default=[],
                    help="style archive with ttl/dp arrays (repeatable)")
    ap.add_argument("--styles-dir", action="append", default=[],
                    help="directory of style .json files (repeatable)")
    ap.add_argument("--k", type=int, default=384)
    ap.add_argument("--balance", action="store_true",
                    help="repeat each later source to roughly the first source's size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources = [(p, load_npz(p)) for p in args.npz]
    sources += [(p, load_styles_dir(p)) for p in args.styles_dir]
    if not sources:
        raise SystemExit("give at least one --npz or --styles-dir")

    parts = []
    n_first = len(sources[0][1][0])
    for i, (name, (ttl, dp)) in enumerate(sources):
        rep = 1 if (i == 0 or not args.balance) else max(1, round(n_first / len(ttl)))
        parts.append((np.repeat(ttl, rep, 0), np.repeat(dp, rep, 0)))
        print(f"  {name}: {len(ttl)} styles x{rep}")
    ttl = np.concatenate([t for t, _ in parts])
    dp = np.concatenate([d for _, d in parts])

    n = len(ttl)
    X = np.concatenate([ttl.reshape(n, -1), dp.reshape(n, -1)], 1).astype(np.float32)
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(args.k, Vt.shape[0])
    basis = Vt[:k].astype(np.float32)
    scale = (S[:k] / np.sqrt(max(n - 1, 1))).astype(np.float32)
    var = (S ** 2 / (S ** 2).sum()).cumsum()
    print(f"fit on {n} rows: k={k} captures {var[k - 1] * 100:.1f}% of style variance")

    split = int(np.prod(TTL_SHAPE))
    with open(args.out, "wb") as f:
        f.write(struct.pack("<7i", k, split + int(np.prod(DP_SHAPE)), split,
                            *TTL_SHAPE, *DP_SHAPE))
        f.write(scale.tobytes())
        f.write(mean.astype(np.float32).tobytes())
        f.write(basis.tobytes())
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")

    # self-check: encode(decode(c)) == c when basis rows are orthonormal
    rng = np.random.default_rng(0)
    c = rng.standard_normal(k).astype(np.float32)
    flat = mean + (c * scale) @ basis
    back = ((flat - mean) @ basis.T) / scale
    err = float(np.abs(back - c).max())
    print(f"round-trip max abs err {err:.2e}")
    assert err < 1e-3, "basis round-trip failed"


if __name__ == "__main__":
    main()
