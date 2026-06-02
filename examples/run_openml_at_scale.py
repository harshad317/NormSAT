"""At-scale OpenML evaluation with significance testing.

Run locally (needs network for OpenML; falls back to sklearn-bundled data otherwise):

    PYTHONPATH=. python examples/run_openml_at_scale.py --seeds 5
    PYTHONPATH=. python examples/run_openml_at_scale.py --seeds 10 --models logreg knn
    PYTHONPATH=. python examples/run_openml_at_scale.py --datasets credit-g kc1 --seeds 5

Outputs (written next to the repo root):
    openml_scale_report.txt    -- per-dataset scores + guarantee violations
    openml_significance.txt     -- Wilcoxon + Holm significance vs each baseline
    openml_results.json         -- machine-readable everything
    paper/figures/figdata.pkl   -- cached (profiles,res) for make_figures.py

What it does, per dataset x seed:
    * profiles the dataset (pathology mix),
    * fits every baseline + RANC-SAT on the train fold (leakage-safe),
    * records held-out AUROC/RMSE and declared-invariant violations,
    * accumulates paired score differences for significance testing.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

# Silence the sklearn ConvergenceWarning flood (it does not affect AUROC ranking and
# made the run look hung). Set RANCSAT_VERBOSE_WARNINGS=1 to re-enable.
if not os.environ.get("RANCSAT_VERBOSE_WARNINGS"):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from rancsat import realdata as RD
from rancsat.realdata import (
    RealDataset, profile_dataset, auto_contract, _global, _eval, _SKLEARN_BUNDLED,
)
from rancsat.profiling import build_regime_cards
from rancsat.sklearn import RANCSATTransformer
from rancsat.ablation import RuleBasedSelector, CVBruteForceSelector
from rancsat.heterogeneous import count_guarantee_violations
from rancsat.schemas import NormalizationPolicy, PolicyKind
from rancsat import policies as P
from rancsat.stats import compare_method_to_baselines, format_significance


# A larger, heterogeneity-rich OpenML list. Many of these have count/sparse features
# with structural zeros (kc1, pc1, jm1: software-defect metrics with many zeros) where
# Standard/Quantile DENSIFY -> this is what strengthens the guarantee figure.
OPENML_AT_SCALE = [
    ("credit-g", 1, "classification"),
    ("kc1", 1, "classification"),
    ("kc2", 1, "classification"),
    ("pc1", 1, "classification"),
    ("pc3", 1, "classification"),
    ("pc4", 1, "classification"),
    ("jm1", 1, "classification"),
    ("phoneme", 1, "classification"),
    ("spambase", 1, "classification"),       # word-frequency features: many zeros
    ("wdbc", 1, "classification"),
    ("ionosphere", 1, "classification"),
    ("diabetes", 1, "classification"),
    ("blood-transfusion-service-center", 1, "classification"),
    ("banknote-authentication", 1, "classification"),
    ("qsar-biodeg", 1, "classification"),
]

GLOBALS = {
    "none": PolicyKind.IDENTITY, "standard": PolicyKind.AFFINE_STANDARD,
    "robust": PolicyKind.AFFINE_ROBUST, "minmax": PolicyKind.MINMAX,
    "maxabs": PolicyKind.MAXABS, "quantile_normal": PolicyKind.QUANTILE_NORMAL,
}
BASELINES_FOR_SIG = ["none", "standard", "robust", "minmax", "quantile_normal", "rule_based"]


def load_local_dir(data_dir: str, target: str = "auto",
                   max_features: int = 200) -> List[RealDataset]:
    """Load every .csv / .arff in `data_dir` as a dataset. The target column is the
    last column, or the one named by --target. Use this when OpenML's API is blocked
    but you can download dataset files manually (e.g. from openml.org or UCI) and drop
    them in a folder.

    CSV: standard comma-separated, a header row, numeric features. ARFF: read via scipy.
    """
    import glob
    import pandas as pd
    out: List[RealDataset] = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")) +
                   glob.glob(os.path.join(data_dir, "*.arff")))
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            if fp.endswith(".arff"):
                from scipy.io import arff
                data, _ = arff.loadarff(fp)
                df = pd.DataFrame(data)
                # decode byte-string columns (ARFF nominal attrs come as bytes)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].str.decode("utf-8", errors="ignore")
            else:
                df = pd.read_csv(fp)
            tcol = df.columns[-1] if target == "auto" else target
            y_raw = df[tcol]
            X = df.drop(columns=[tcol]).select_dtypes("number").dropna(axis=1, how="any")
            if X.shape[1] < 3 or X.shape[1] > max_features:
                print(f"  [skip {name}: {X.shape[1]} usable numeric features]")
                continue
            # infer task: few unique non-float target values -> classification
            uniq = y_raw.dropna().unique()
            is_cls = (y_raw.dtype == object) or (len(uniq) <= max(20, 0.05 * len(y_raw)))
            if is_cls:
                yv = pd.Categorical(y_raw).codes.astype(float)
                if len(set(yv)) > 2:
                    # binarize as MOST-FREQUENT class vs rest (balanced enough to split),
                    # not rarest-vs-rest which can yield a near-empty positive class.
                    vals, counts = np.unique(yv, return_counts=True)
                    majority = vals[counts.argmax()]
                    yv = (yv == majority).astype(float)
                task = "classification"
                # guard: need >=2 classes and the minority class big enough that a 30%
                # stratified split leaves >=2 of each in train AND test.
                _, c = np.unique(yv, return_counts=True)
                if len(c) < 2 or c.min() < 8:
                    print(f"  [skip {name}: degenerate binary target (class counts {list(c)})]")
                    continue
            else:
                yv = y_raw.to_numpy(float)
                task = "regression"
            out.append(RealDataset(name, X.to_numpy(float), yv,
                                   list(map(str, X.columns)), task, "local"))
            print(f"  loaded {name}: X={X.shape} task={task}")
        except Exception as e:
            print(f"  [skip {name}: {type(e).__name__}: {e}]")
    return out


def load_openml_scale(names: Optional[List[str]] = None,
                      max_features: int = 60) -> List[RealDataset]:
    from sklearn import datasets as skd
    want = OPENML_AT_SCALE if names is None else [(n, 1, "classification") for n in names]
    out: List[RealDataset] = []
    for name, ver, task in want:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y = skd.fetch_openml(name, version=ver, return_X_y=True,
                                        as_frame=True, parser="auto")
            X = X.select_dtypes("number").dropna(axis=1, how="any")
            if X.shape[1] < 3 or X.shape[1] > max_features:
                continue
            yv = y.astype("category").cat.codes.to_numpy().astype(float)
            if len(set(yv)) > 2:
                yv = (yv == yv.min()).astype(float)
            out.append(RealDataset(name, X.to_numpy(float), yv,
                                   list(map(str, X.columns)), "classification", "openml"))
            print(f"  loaded {name}: {X.shape}")
        except Exception as e:
            print(f"  [skip {name}: {type(e).__name__}]")
    return out


def evaluate(ds: RealDataset, seeds, model) -> Dict:
    # map downstream model -> contract family: distance-based models (knn, svm_rbf)
    # request distribution matching; trees are scale-invariant -> 'tree' prefers no-op;
    # linear is the scale-sensitive default.
    family = {"knn": "knn", "svm_rbf": "rbf_svm", "tree": "tree"}.get(model, "linear")
    base, ov = auto_contract(ds, model_family=family)
    per_seed = defaultdict(list)          # method -> [score per seed]
    viol = defaultdict(int)               # method -> total violations
    rancsat_by_seed = []

    # stratify only for classification, and only when every class has >=2 members
    strat = None
    if ds.task == "classification":
        _, counts = np.unique(ds.y, return_counts=True)
        if counts.min() >= 2:
            strat = ds.y
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            ds.X, ds.y, test_size=0.3, random_state=int(s), stratify=strat)
        cards = build_regime_cards(Xtr, ds.feature_names)
        for label, kind in GLOBALS.items():
            try:
                Ztr, Zte = _global(Xtr, Xte, ds.feature_names, kind)
                if np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte)):
                    per_seed[label].append(_eval(Ztr, Zte, ytr, yte, ds.task, model))
                pols = [NormalizationPolicy(feature=cc.name, index=cc.index, kind=kind,
                        params=P.fit_params(kind, Xtr[:, cc.index], cc),
                        invertible=P.CAPABILITIES[kind].invertible) for cc in cards]
                viol[label] += sum(count_guarantee_violations(
                    Xtr, Xte, ds.feature_names, pols, ov)["counts"].values())
            except Exception:
                pass
        # skip degenerate folds (a single-class train fold can occur on imbalanced data)
        if ds.task == "classification" and len(np.unique(ytr)) < 2:
            continue
        try:
            rb = RuleBasedSelector(feature_names=ds.feature_names).fit(Xtr, ytr)
            per_seed["rule_based"].append(_eval(rb.transform(Xtr), rb.transform(Xte), ytr, yte, ds.task, model))
            t = RANCSATTransformer(contract=base, overrides=ov,
                                   feature_names=ds.feature_names).fit(Xtr, ytr)
            sc = _eval(t.transform(Xtr), t.transform(Xte), ytr, yte, ds.task, model)
            per_seed["RANC-SAT"].append(sc); rancsat_by_seed.append(sc)
            viol["RANC-SAT"] += sum(count_guarantee_violations(
                Xtr, Xte, ds.feature_names, t.policies_, ov)["counts"].values())
        except Exception as e:
            print(f"    [skip seed {s} of {ds.name}: {type(e).__name__}]")

    means = {k: float(np.mean(v)) for k, v in per_seed.items() if v}
    return {"name": ds.name, "task": ds.task, "model": model,
            "means": means, "per_seed": {k: list(map(float, v)) for k, v in per_seed.items()},
            "violations": dict(viol)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["logreg", "knn"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--max-features", type=int, default=60)
    ap.add_argument("--data-dir", default=None,
                    help="Folder of local .csv/.arff files to use instead of OpenML "
                         "(target = last column, or --target). Best option when the "
                         "OpenML API is firewalled.")
    ap.add_argument("--target", default="auto", help="Target column name for --data-dir.")
    args = ap.parse_args()
    seeds = range(args.seeds)

    if args.data_dir:
        print(f"Loading local datasets from {args.data_dir} ...")
        dsets = load_local_dir(args.data_dir, target=args.target,
                               max_features=args.max_features)
        if not dsets:
            # Hard stop: silently falling back to bundled data when the user explicitly
            # asked for a folder is misleading. Tell them exactly what's wrong.
            import glob
            found = glob.glob(os.path.join(args.data_dir, "*"))
            raise SystemExit(
                f"\nERROR: --data-dir '{args.data_dir}' produced no usable datasets.\n"
                f"  Directory exists: {os.path.isdir(args.data_dir)}\n"
                f"  Files seen: {[os.path.basename(f) for f in found][:10] or 'NONE'}\n"
                f"  Expected: .csv or .arff files, each with >=3 numeric feature columns\n"
                f"  and a target column (last column, or pass --target NAME).\n"
                f"  Download datasets from openml.org/UCI/Kaggle into that folder, then "
                f"re-run.\n  (To deliberately use the built-in datasets, omit --data-dir.)")
    else:
        print("Loading datasets (OpenML; falls back to sklearn-bundled if no network)...")
        dsets = load_openml_scale(args.datasets, max_features=args.max_features)
        if not dsets:
            print("No external datasets -> using sklearn-bundled datasets "
                  "(includes 'digits', which has many structural zeros).")
            dsets = RD.load_bundled()

    profiles = {d.name: profile_dataset(d) for d in dsets}
    results = {"profiles": profiles, "eval": {}, "significance": {}}
    higher_better = True   # AUROC for classification; regression handled via -RMSE in _eval

    for model in args.models:
        rows = [evaluate(d, seeds, model) for d in dsets]
        results["eval"][model] = rows
        # Accumulate paired diffs for significance, SEPARATED BY TASK so we never pool
        # AUROC (classification) with -RMSE (regression) -- those live on different
        # scales and pooling them corrupts the median/CI/effect estimates.
        by_task = defaultdict(lambda: defaultdict(list))   # task -> baseline -> [diffs]
        for r in rows:
            rs = r["per_seed"].get("RANC-SAT", [])
            for b in BASELINES_FOR_SIG:
                bs = r["per_seed"].get(b, [])
                for a, c in zip(rs, bs):
                    by_task[r["task"]][b].append(a - c)   # _eval orients higher=better
        results["significance"][model] = {
            task: compare_method_to_baselines({k: v for k, v in paired.items() if v})
            for task, paired in by_task.items()
        }

    # ---- write reports ----
    _write_scale_report(results)
    _write_significance(results)
    json.dump(_jsonable(results), open("openml_results.json", "w"), indent=2)
    os.makedirs("paper/figures", exist_ok=True)
    # cache (profiles,res) so paper/make_figures.py can render without recompute
    fig_res = {m: [ {"name": r["name"], "task": r["task"], "model": r["model"],
                     "scores": r["means"], "violations": r["violations"]}
                    for r in results["eval"][m] ] for m in results["eval"]}
    pickle.dump((profiles, fig_res), open("paper/figures/figdata.pkl", "wb"))
    print("\n[wrote openml_scale_report.txt, openml_significance.txt, openml_results.json,"
          " paper/figures/figdata.pkl]")


def _write_scale_report(R):
    L = ["OpenML AT-SCALE RESULTS", "", "Pathology profiles:",
         f"    {'dataset':<34}{'feats':>6}{'samp':>7}{'skew':>6}{'sparse':>7}{'heterog':>9}"]
    for n, p in R["profiles"].items():
        L.append(f"    {n:<34}{p['n_features']:>6}{p['n_samples']:>7}{p['skewed']:>6}"
                 f"{p['sparse']:>7}{p['heterogeneity']:>9}")
    for model, rows in R["eval"].items():
        L += ["", f"=== model = {model} (AUROC, higher=better) ===",
              f"    {'dataset':<34}{'RANC-SAT':>10}{'best_global':>12}{'rank':>6}{'viol g/ours':>14}"]
        for r in rows:
            best_g = max((v for k, v in r["means"].items()
                          if k not in ("RANC-SAT", "rule_based")), default=float("nan"))
            rk = sum(1 for k, v in r["means"].items()
                     if k != "RANC-SAT" and v > r["means"].get("RANC-SAT", -1)) + 1
            gv = sum(v for k, v in r["violations"].items() if k != "RANC-SAT")
            ov = r["violations"].get("RANC-SAT", 0)
            L.append(f"    {r['name']:<34}{r['means'].get('RANC-SAT', float('nan')):>10.4f}"
                     f"{best_g:>12.4f}{str(rk)+'/'+str(len(r['means'])):>6}{f'{gv}/{ov}':>14}")
    open("openml_scale_report.txt", "w").write("\n".join(L) + "\n")


def _write_significance(R):
    blocks = []
    for model, by_task in R["significance"].items():
        for task, sig in by_task.items():
            metric = "AUROC" if task == "classification" else "-RMSE"
            blocks.append(format_significance(
                sig, title=f"(model={model}, task={task}, metric={metric})"))
    open("openml_significance.txt", "w").write("\n\n".join(blocks) + "\n")
    print("\n".join(blocks))


def _jsonable(R):
    out = {"profiles": R["profiles"], "eval": R["eval"], "significance": {}}
    for m, by_task in R["significance"].items():
        out["significance"][m] = {}
        for task, sig in by_task.items():
            out["significance"][m][task] = {
                k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                    for kk, vv in v.items()} for k, v in sig.items()}
    return out


if __name__ == "__main__":
    main()
