#!/usr/bin/env python3
"""Centered Qwen-space speaker similarity — the metric the project optimizes
for character voices.

Raw Qwen speaker-conditioning features share a dominant common component
(different speakers still cosine at ~0.94), so cosine is computed after
subtracting a population mean ("center"), which restores a full-range metric:
measured on 61 half-vs-half reference pairs, same-speaker 0.836 +/- 0.061 vs
different-speaker -0.005 +/- 0.270, d-prime 4.30, 4.1 % midpoint error.

Inputs are feature npz files (spk, feat[N,2048]) as produced by
extract_qwen_features.py. The center is estimated from --center-feats
(typically the reference-population features) and must be the SAME whenever
scores are compared.

  # is the metric discriminative on your data? (spk names <name>__a/<name>__b)
  qwen_similarity.py calib --feats halves.npz --center-feats refs.npz
  # score synthesized eval wavs (<spk>_<i>) against their references
  qwen_similarity.py score --feats synth.npz --ref-feats refs.npz --out scores.json
"""
import argparse
import json

import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["spk"]).astype(str), np.asarray(d["feat"], dtype=np.float32)


def fmap(path, center):
    s, f = load(path)
    f = f - center
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return dict(zip(s, f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calib", "score"])
    ap.add_argument("--feats", required=True)
    ap.add_argument("--ref-feats", default=None)
    ap.add_argument("--center-feats", default=None,
                    help="population npz for the center; defaults to --ref-feats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _, cf = load(args.center_feats or args.ref_feats or args.feats)
    center = cf.mean(0)
    m = fmap(args.feats, center)

    if args.mode == "calib":
        spks = sorted(k[:-3] for k in m if k.endswith("__a") and k[:-3] + "__b" in m)
        A = np.stack([m[s + "__a"] for s in spks])
        B = np.stack([m[s + "__b"] for s in spks])
        cross = A @ B.T
        same, diff = np.diag(cross), cross[~np.eye(len(spks), dtype=bool)]
        thr = (same.mean() + diff.mean()) / 2
        err = ((same < thr).mean() + (diff > thr).mean()) / 2
        print(f"same {same.mean():.3f}+/-{same.std():.3f}  diff {diff.mean():.3f}+/-{diff.std():.3f}  "
              f"d-prime {(same.mean() - diff.mean()) / np.sqrt((same.var() + diff.var()) / 2):.2f}  "
              f"err {err * 100:.1f}% (n={len(spks)})")
    else:
        refs = fmap(args.ref_feats, center)
        per = {}
        for k, v in m.items():
            spk = k.rsplit("_", 1)[0]
            if spk in refs:
                per.setdefault(spk, []).append(float(v @ refs[spk]))
        means = {s: float(np.mean(c)) for s, c in per.items()}
        print(f"mean centered qwen-cos {np.mean(list(means.values())):.4f} over {len(means)} voices")
        if args.out:
            json.dump(means, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
