"""Generate a corpus of CSV datasets spanning the HETEROGENEITY axis.

Why: the OpenML API is firewalled in our environments, so we cannot pull real tabular
tasks through it. To still run the at-scale + significance pipeline across MANY
datasets, this writes a controlled corpus whose datasets vary in how heterogeneous
their feature pathologies are (fraction of features that are sparse / skewed / heavy-
tailed vs. clean). That turns the single-dataset anecdotes into a gain-vs-heterogeneity
curve and gives the Wilcoxon tests enough datasets to be meaningful.

HONEST SCOPE: these are SYNTHETIC. They validate (a) that the guarantee property holds
across many tables, and (b) that per-feature compilation's accuracy benefit tracks
heterogeneity. They do NOT establish real-world external validity -- for that, drop
manually-downloaded OpenML/UCI/Kaggle CSVs into the same folder and re-run.

Usage:
    python examples/make_dataset_corpus.py --out ./my_datasets --n 18
    PYTHONPATH=. python examples/run_openml_at_scale.py --data-dir ./my_datasets --seeds 5
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def _sparse(n, rng, p_zero=0.7, lam=3.0):
    return np.where(rng.random(n) < p_zero, 0.0, rng.poisson(lam, n).astype(float))


def _skewed(n, rng):
    return np.exp(rng.normal(0, 1, n))                      # lognormal


def _heavy(n, rng):
    x = rng.standard_t(3, n)
    spikes = rng.random(n) < 0.04
    x[spikes] += rng.normal(0, 30, spikes.sum())            # corrupted extremes
    return x


def _clean(n, rng):
    return rng.normal(rng.uniform(-3, 3), rng.uniform(0.5, 4), n)


_MAKERS = {"sparse": _sparse, "skewed": _skewed, "heavy": _heavy, "clean": _clean}


def make_one(name, n, d, mix, task, seed):
    """mix: dict like {'sparse':0.25,'skewed':0.25,'heavy':0.0,'clean':0.5} (fractions).

    UNBIASED DESIGN (v2): the target is a function of an independent LATENT per feature,
    NOT of the observed (pathological) feature or its 'correctly-transformed' view. We
    then render the OBSERVED feature as a monotone-but-pathological encoding of the same
    latent (lognormal for skew, t-tailed for heavy, thresholded for sparse). Thus:
      * the predictive information is recoverable in principle from the observed feature,
      * but NO specific scaler is handed the DGP -- the right transform must be found,
      * and we never define y from log1p(x)/clip(x)/(x>0), which previously let a method
        that applied exactly those transforms match the generator by construction.
    """
    rng = np.random.default_rng(seed)
    kinds = []
    for k, frac in mix.items():
        kinds += [k] * int(round(frac * d))
    while len(kinds) < d:
        kinds.append("clean")
    kinds = kinds[:d]
    rng.shuffle(kinds)

    cols, latents = [], []
    for k in kinds:
        z = rng.normal(0, 1, n)                  # the latent that actually drives y
        latents.append(z)
        # observed feature = a pathological, monotone-ish encoding of z (+ independent
        # nuisance), so recovering z requires undoing the pathology but the DGP target
        # never sees the 'correct' transform.
        if k == "sparse":
            # zero-inflate: below a threshold the feature reads exactly 0 (structural),
            # else a positive count rising with z. Presence correlates with z but the
            # mapping is not (x>0) of any transform we score.
            mask = z < rng.normal(0.4, 0.2)
            x = np.where(mask, 0.0, np.maximum(1, np.round(np.exp(0.6 * z) + rng.normal(0, 0.5, n))))
        elif k == "skewed":
            x = np.exp(0.8 * z + rng.normal(0, 0.3, n))      # lognormal in z
        elif k == "heavy":
            x = z + rng.standard_t(3, n) * 0.6               # heavy nuisance tail
            spk = rng.random(n) < 0.04
            x[spk] += rng.normal(0, 25, spk.sum())
        else:
            x = 2.5 * z + rng.normal(0, 0.4, n)              # clean affine in z
        cols.append(np.asarray(x, dtype=float))

    X = np.column_stack(cols)
    Zlat = np.column_stack(latents)
    # target: nonlinear function of the LATENTS (held out from the observed encoding).
    w = rng.normal(size=d)
    wq = rng.normal(size=d) * 0.4
    signal = Zlat @ w + (Zlat ** 2 - 1.0) @ wq               # linear + mild nonlinearity
    names = [f"{k}{j}" for j, k in enumerate(kinds)]
    df = pd.DataFrame(X, columns=names)
    if task == "classification":
        y = (signal + rng.normal(0, 0.6, n) > np.median(signal)).astype(int)
    else:
        y = signal + rng.normal(0, 0.6, n)
    df["target"] = y
    return df


def build_corpus(out_dir, n_datasets=18, n=800, seed0=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed0)
    # sweep heterogeneity: fraction of features that are "pathological" (not clean)
    written = []
    for i in range(n_datasets):
        het = (i + 1) / (n_datasets + 1)                    # 0..1 sweep
        d = int(rng.integers(6, 16))
        # split the pathological budget across sparse/skewed/heavy
        path = het
        sp, sk, hv = rng.dirichlet([1, 1, 1]) * path
        mix = {"sparse": sp, "skewed": sk, "heavy": hv, "clean": max(0.0, 1 - path)}
        task = "classification" if i % 3 != 0 else "regression"
        name = f"corpus_{i:02d}_het{int(het*100):02d}_{task[:3]}"
        df = make_one(name, n, d, mix, task, seed=int(rng.integers(1e9)))
        fp = os.path.join(out_dir, name + ".csv")
        df.to_csv(fp, index=False)
        written.append((name, d, task, round(het, 2)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./my_datasets")
    ap.add_argument("--n", type=int, default=18, help="number of datasets")
    ap.add_argument("--rows", type=int, default=800)
    args = ap.parse_args()
    written = build_corpus(args.out, n_datasets=args.n, n=args.rows)
    print(f"Wrote {len(written)} datasets to {os.path.abspath(args.out)}:")
    for name, d, task, het in written:
        print(f"  {name:34} feats={d:>2} task={task:<14} target_heterogeneity≈{het}")
    print("\nNext:")
    print(f"  PYTHONPATH=. python examples/run_openml_at_scale.py --data-dir {args.out} --seeds 5")
    print("\nNOTE: synthetic corpus -> validates mechanism + heterogeneity-dependence,")
    print("not real-world external validity. Add manually-downloaded real CSVs to the")
    print("same folder to strengthen external validity.")


if __name__ == "__main__":
    main()
