"""Download REAL tabular datasets from stable direct-CSV mirrors into ./my_datasets/.

Why this exists: the OpenML *API* (api.openml.org) is firewalled in our environments,
but the underlying datasets are available as plain CSV from stable public mirrors
(GitHub raw, UCI). This script fetches a curated set chosen for the pathologies that
matter to NormSAT -- structural zeros, skew, heavy tails -- normalizes them to
"numeric features + target as the LAST column", and writes them where
run_openml_at_scale.py --data-dir expects them.

It runs on YOUR machine (standard urllib), so it uses your normal internet access.
Each dataset is independent: a failed download is skipped with a clear message.

Usage:
    python examples/fetch_real_datasets.py --out ./my_datasets
    PYTHONPATH=. python examples/run_openml_at_scale.py --data-dir ./my_datasets --seeds 5
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
from typing import List, Optional

import numpy as np
import pandas as pd

# Each entry: (name, url_or_list, read_kwargs, target_col, note)
# url_or_list may be a single URL or a list of fallback URLs tried in order.
# target_col=None means "last column". Sources are stable public CSV mirrors.
CATALOG = [
    # Pima diabetes: glucose/insulin/skin-thickness have STRUCTURAL ZEROS (encoded
    # missing) -> the cleanest real example for the zero-densification guarantee.
    ("pima_diabetes",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
     {"header": None}, None, "structural zeros in several columns"),
    # Sonar: 60 continuous features, mild skew; binary target (last col, M/R).
    ("sonar",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv",
     {"header": None}, None, "60 continuous features"),
    # Ionosphere: 34 features, some near-constant; binary target last col (g/b).
    ("ionosphere",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv",
     {"header": None}, None, "near-constant + continuous mix"),
    # Wine quality (red): 11 skewed physico-chemical features; quality score target.
    ("wine_quality_red",
     "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
     {"sep": ","}, "quality", "right-skewed chemical features"),
    # Banknote authentication: 4 continuous features, heavy-tailed; binary target.
    ("banknote",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv",
     {"header": None}, None, "heavy-tailed continuous"),
    # Glass identification: skewed oxide composition features; class target.
    ("glass",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv",
     {"header": None}, None, "skewed composition + structural zeros"),
    # Haberman survival: small counts; binary target.
    ("haberman",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv",
     {"header": None}, None, "count-like features"),
    # Wheat seeds: 7 continuous geometric features, mild skew; class target.
    ("seeds",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv",
     {"header": None}, None, "continuous geometric features"),
    # Abalone (age via rings): skewed measurements; first col is sex (dropped as non-numeric).
    ("abalone",
     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
     {"header": None}, None, "skewed physical measurements"),
    # --- additional datasets (fallback URL lists for reliability) ---------------
    # Breast cancer Wisconsin (diagnostic): 30 skewed cell-nucleus features.
    ("breast_cancer_wisc",
     ["https://raw.githubusercontent.com/datasciencedojo/datasets/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20Data%20Set/data.csv"],
     {}, "diagnosis", "30 skewed continuous features"),
    # Heart disease (Cleveland processed), via UCI mirror.
    ("heart_cleveland",
     ["https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
      "https://raw.githubusercontent.com/educatorsRlearners/heart-disease-uci/master/data/processed.cleveland.data"],
     {"header": None}, None, "mixed-scale clinical features"),
    # Yeast localization (whitespace-separated), via UCI mirror.
    ("yeast",
     ["https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"],
     {"header": None, "sep": r"\s+"}, None, "skewed + structural zeros (first col = id, dropped)"),
    # Ecoli localization (whitespace-separated), via UCI mirror.
    ("ecoli",
     ["https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"],
     {"header": None, "sep": r"\s+"}, None, "skewed, some structural zeros (first col = id)"),
    # ---- REGRESSION datasets (continuous target) -------------------------------
    # Housing (Boston-style), whitespace-separated; median value target (last col).
    ("housing",
     ["https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"],
     {}, "medv", "REGRESSION: skewed housing features"),
    # Concrete compressive strength: skewed mixture features; strength target (last).
    ("concrete",
     ["https://raw.githubusercontent.com/eflume/datasets/master/concrete.csv"],
     {}, None, "REGRESSION: skewed mixture features"),
]
# REGRESSION NOTE: external regression CSV mirrors are less stable than the UCI
# classification ones; some URLs above may 404. The harness ALSO handles regression on
# sklearn's bundled 'diabetes' dataset out of the box (no network), and any local
# regression CSV (continuous target as last column) dropped into --out works. The
# at-scale runner auto-detects task type and reports -RMSE for regression.
# NOTE: a few mirror URLs (ecoli/yeast whitespace formats, wdbc/heart/german) were
# pruned because their public paths 404'd or needed bespoke parsing. To add more, drop
# any CSV (target = last column) into the --out folder; the at-scale harness reads it.


def _download(url: str, timeout: int = 30) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [download failed: {type(e).__name__}: {e}]")
        return None


def _normalize(df: pd.DataFrame, target_col, name: str) -> Optional[pd.DataFrame]:
    """Coerce to numeric features + a 'target' column (last). Drop non-numeric features;
    map a categorical/string target to codes. Returns None if too few usable columns."""
    if target_col is None:
        target_col = df.columns[-1]
    if target_col not in df.columns:
        print(f"    [skip {name}: target column '{target_col}' not found]")
        return None
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # the first column of some headerless UCI files is an id/string -> let numeric coercion drop it
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="any")
    if X.shape[1] < 3:
        print(f"    [skip {name}: only {X.shape[1]} usable numeric features]")
        return None
    out = X.copy()
    out["target"] = pd.Categorical(y).codes if (y.dtype == object) else y.values
    out.columns = [f"f{i}" for i in range(X.shape[1])] + ["target"]
    return out


def fetch_all(out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, url, kw, target, note in CATALOG:
        print(f"  {name}  ({note})")
        urls = url if isinstance(url, list) else [url]
        text = None
        for u in urls:
            text = _download(u)
            if text is not None:
                break
        if text is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(text), engine="python",
                             na_values=["?", "NA", ""], **kw)
            df = df.dropna(axis=0, how="any")   # drop rows with missing markers
        except Exception as e:
            print(f"    [parse failed: {type(e).__name__}: {e}]")
            continue
        norm = _normalize(df, target, name)
        if norm is None:
            continue
        fp = os.path.join(out_dir, f"{name}.csv")
        norm.to_csv(fp, index=False)
        n_zero_cols = int((norm.iloc[:, :-1] == 0).any(axis=0).sum())
        print(f"    wrote {fp}  shape={norm.shape}  (cols with structural zeros: {n_zero_cols})")
        written.append(fp)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./my_datasets")
    args = ap.parse_args()
    print(f"Fetching real datasets into {os.path.abspath(args.out)} ...")
    written = fetch_all(args.out)
    print(f"\nDone: {len(written)}/{len(CATALOG)} datasets downloaded.")
    if not written:
        print("\nNothing downloaded -- your network may also block these mirrors.\n"
              "Fallback: download any .csv datasets manually into the folder, with the\n"
              "target as the last column, then run the at-scale script with --data-dir.")
        sys.exit(1)
    print("\nNext:")
    print(f"  PYTHONPATH=. python examples/run_openml_at_scale.py --data-dir {args.out} --seeds 5")


if __name__ == "__main__":
    main()
