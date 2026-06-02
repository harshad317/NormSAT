"""Real-data evaluation harness.

Validates the synthetic findings on REAL tabular data. Two data sources:

  * OpenML (preferred, used when run locally with network access) -- a curated list of
    heterogeneous tabular tasks with mixed feature pathologies.
  * sklearn bundled datasets (no network) -- real-world tables (breast-cancer, wine,
    diabetes, digits) used as a fallback so the harness runs anywhere.

For each dataset we:
  1. PROFILE it -- report the pathology mix (skew, heavy tails, sparsity, bounded
     ratios) so we know whether per-feature headroom should even exist.
  2. GUARANTEE AUDIT -- declare a conservative auto-contract per feature and count how
     often each method violates it on a held-out fold. RANC-SAT should be 0.
  3. DOWNSTREAM -- compare RANC-SAT vs global scalers + rule-based + CV search.

The guarantee audit is the claim most likely to transfer, because zero-densification
and out-of-bound breaches are structural properties of the transform, not fitted wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import warnings
import numpy as np

from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from .schemas import InvarianceContract, NormalizationPolicy, PolicyKind
from .profiling import build_regime_cards
from .sklearn import RANCSATTransformer
from .ablation import RuleBasedSelector, CVBruteForceSelector
from .heterogeneous import count_guarantee_violations
from . import policies as P


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
@dataclass
class RealDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    task: str
    source: str


_SKLEARN_BUNDLED = {
    "breast_cancer": (skd.load_breast_cancer, "classification"),
    "wine": (skd.load_wine, "classification"),
    "diabetes": (skd.load_diabetes, "regression"),
    "digits": (skd.load_digits, "classification"),
}

# heterogeneous OpenML tasks (used only when network is available)
_OPENML_TASKS = [
    ("credit-g", 1, "classification"),
    ("diabetes", 1, "classification"),
    ("kc1", 1, "classification"),
    ("pc1", 1, "classification"),
    ("phoneme", 1, "classification"),
    ("wdbc", 1, "classification"),
]


def load_bundled() -> List[RealDataset]:
    out = []
    for name, (loader, task) in _SKLEARN_BUNDLED.items():
        b = loader()
        y = b.target.astype(float)
        if task == "classification" and len(set(y)) > 2:
            # binarize multi-class to keep AUROC well-defined: class-0 vs rest
            y = (y == y.min()).astype(float)
        names = list(getattr(b, "feature_names", [f"f{i}" for i in range(b.data.shape[1])]))
        names = [str(n) for n in names]
        out.append(RealDataset(name, np.asarray(b.data, float), y, names, task, "sklearn"))
    return out


def load_openml(max_datasets: int = 6) -> List[RealDataset]:
    """Try OpenML; returns [] if network is unavailable (caller falls back)."""
    out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, ver, task in _OPENML_TASKS[:max_datasets]:
                X, y = skd.fetch_openml(name, version=ver, return_X_y=True,
                                        as_frame=True, parser="auto")
                X = X.select_dtypes("number").dropna(axis=1, how="any")
                if X.shape[1] < 3:
                    continue
                yv = y.astype("category").cat.codes.to_numpy().astype(float)
                if len(set(yv)) > 2:
                    yv = (yv == yv.min()).astype(float)
                out.append(RealDataset(name, X.to_numpy(float), yv,
                                       list(map(str, X.columns)), task, "openml"))
    except Exception as e:
        print(f"[openml unavailable: {type(e).__name__}; using bundled datasets]")
        return []
    return out


def load_real_datasets() -> List[RealDataset]:
    ds = load_openml()
    return ds if ds else load_bundled()


def load_local_dir(data_dir: str, target: str = "auto",
                   max_features: int = 200) -> List[RealDataset]:
    """Load every .csv / .arff in ``data_dir`` as a RealDataset. Target is the last
    column, or the column named by ``target``. Numeric features only; the task
    (classification vs regression) is inferred from target cardinality. Multiclass
    targets are binarized as most-frequent-class vs rest; degenerate binary targets are
    skipped. Lives in the package so it is importable as
    ``from rancsat.realdata import load_local_dir`` without importing ``examples``.
    """
    import os
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
            uniq = y_raw.dropna().unique()
            is_cls = (y_raw.dtype == object) or (len(uniq) <= max(20, 0.05 * len(y_raw)))
            if is_cls:
                yv = pd.Categorical(y_raw).codes.astype(float)
                if len(set(yv)) > 2:
                    vals, counts = np.unique(yv, return_counts=True)
                    yv = (yv == vals[counts.argmax()]).astype(float)
                task = "classification"
                _, cc = np.unique(yv, return_counts=True)
                if len(cc) < 2 or cc.min() < 8:
                    print(f"  [skip {name}: degenerate binary target {list(cc)}]")
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


# --------------------------------------------------------------------------- #
# 1. Pathology profile
# --------------------------------------------------------------------------- #
def profile_dataset(ds: RealDataset) -> Dict:
    cards = build_regime_cards(ds.X, ds.feature_names)
    n = len(cards)
    skewed = sum(1 for c in cards if abs(c.skew) > 1.0)
    heavy = sum(1 for c in cards if c.tail.value == "heavy")
    sparse = sum(1 for c in cards if c.zero_mass > 0.2)
    signed = sum(1 for c in cards if c.sign.value == "signed")
    positive = sum(1 for c in cards if c.sign.value in ("positive", "nonnegative"))
    return {"name": ds.name, "n_features": n, "n_samples": ds.X.shape[0],
            "task": ds.task, "skewed": skewed, "heavy_tail": heavy,
            "sparse": sparse, "signed": signed, "positive": positive,
            "heterogeneity": round((skewed + heavy + sparse) / max(n, 1), 2)}


# --------------------------------------------------------------------------- #
# 2. Auto-contract -- conservative, per-feature, derived from the data only
# --------------------------------------------------------------------------- #
def auto_contract(ds: RealDataset, model_family: Optional[str] = None
                  ) -> Tuple[InvarianceContract, Dict[int, InvarianceContract]]:
    """Declare conservative invariants from feature structure (no labels, no leakage):
       sparse feature -> preserve_zero; otherwise enforce scale-invariance.

    When the downstream model is distance-based (model_family in knn/kmeans/rbf_svm),
    every per-feature override additionally requests prefer_distribution_match so the
    compiler equalizes feature distributions (the fix for KNN under-performance)."""
    distance = model_family in ("knn", "kmeans", "rbf_svm")
    # tree/ensemble models are scale-invariant: the correct policy is (near) no-op, so
    # we do NOT impose scale-invariance and only protect structural zeros. This is the
    # 'does no harm' setting -- NormSAT should leave features essentially untouched.
    tree_like = model_family in ("tree", "naive_bayes")
    cards = build_regime_cards(ds.X, ds.feature_names)
    ov: Dict[int, InvarianceContract] = {}
    for c in cards:
        if tree_like:
            ov[c.index] = InvarianceContract(preserve_zero=(c.zero_mass > 0.2))
        elif c.zero_mass > 0.2:
            ov[c.index] = InvarianceContract(preserve_zero=True,
                                             enforce_scale_invariance=True,
                                             prefer_distribution_match=distance)
        elif abs(c.skew) > 1.5 and c.sign.value in ("positive", "nonnegative"):
            ov[c.index] = InvarianceContract(preserve_monotonicity=True,
                                             prefer_distribution_match=distance)
        else:
            ov[c.index] = InvarianceContract(enforce_scale_invariance=True,
                                             enforce_shift_invariance=True,
                                             prefer_distribution_match=distance)
    return InvarianceContract(), ov


# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #
def _make_estimator(task, model):
    """Downstream estimator for a given model family.
      logreg  -- linear, scale-sensitive
      knn     -- distance-based, scale-sensitive (wants commensurate features)
      svm_rbf -- kernel distance-based, scale-sensitive
      tree    -- RandomForest, scale-INVARIANT (normalization should be a no-op here;
                 this is the 'does no harm' control)
    """
    if task == "classification":
        if model == "knn":
            return KNeighborsClassifier(15)
        if model == "svm_rbf":
            from sklearn.svm import SVC
            return SVC(probability=True)
        if model == "tree":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        return LogisticRegression(max_iter=2000, solver="liblinear")
    if model == "knn":
        return KNeighborsRegressor(15)
    if model == "svm_rbf":
        from sklearn.svm import SVR
        return SVR()
    if model == "tree":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    return Ridge()


def _eval(Ztr, Zte, ytr, yte, task, model):
    est = _make_estimator(task, model)
    est.fit(Ztr, ytr)
    if task == "classification":
        try:
            return roc_auc_score(yte, est.predict_proba(Zte)[:, 1])
        except Exception:
            try:
                return roc_auc_score(yte, est.decision_function(Zte))  # SVM w/o proba
            except Exception:
                return accuracy_score(yte, est.predict(Zte))
    return -float(np.sqrt(mean_squared_error(yte, est.predict(Zte))))


def _global(Xtr, Xte, names, kind):
    cards = build_regime_cards(Xtr, names)
    Ztr = np.empty_like(Xtr); Zte = np.empty_like(Xte)
    for c in cards:
        pol = NormalizationPolicy(feature=c.name, index=c.index, kind=kind,
                                  params=P.fit_params(kind, Xtr[:, c.index], c),
                                  invertible=P.CAPABILITIES[kind].invertible)
        Ztr[:, c.index] = P.apply(pol, Xtr[:, c.index])
        Zte[:, c.index] = P.apply(pol, Xte[:, c.index])
    return Ztr, Zte


def evaluate_dataset(ds: RealDataset, seeds=range(3), model="logreg") -> Dict:
    # map the downstream model to a contract model-family so the auto-contract can
    # request distribution matching for distance-based models.
    family = "knn" if model == "knn" else "linear"
    base, ov = auto_contract(ds, model_family=family)
    globals_ = {"none": PolicyKind.IDENTITY, "standard": PolicyKind.AFFINE_STANDARD,
                "robust": PolicyKind.AFFINE_ROBUST, "minmax": PolicyKind.MINMAX,
                "maxabs": PolicyKind.MAXABS, "quantile_normal": PolicyKind.QUANTILE_NORMAL}
    scores: Dict[str, List[float]] = {}
    viol_tot: Dict[str, int] = {}

    for s in seeds:
        strat = ds.y if ds.task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.3,
                                              random_state=int(s), stratify=strat)
        cards = build_regime_cards(Xtr, ds.feature_names)
        for label, kind in globals_.items():
            try:
                Ztr, Zte = _global(Xtr, Xte, ds.feature_names, kind)
                if np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte)):
                    scores.setdefault(label, []).append(_eval(Ztr, Zte, ytr, yte, ds.task, model))
                pols = [NormalizationPolicy(feature=c.name, index=c.index, kind=kind,
                        params=P.fit_params(kind, Xtr[:, c.index], c),
                        invertible=P.CAPABILITIES[kind].invertible) for c in cards]
                r = count_guarantee_violations(Xtr, Xte, ds.feature_names, pols, ov)
                viol_tot[label] = viol_tot.get(label, 0) + sum(r["counts"].values())
            except Exception:
                pass
        # rule-based
        rb = RuleBasedSelector(feature_names=ds.feature_names).fit(Xtr, ytr)
        scores.setdefault("rule_based", []).append(_eval(rb.transform(Xtr), rb.transform(Xte), ytr, yte, ds.task, model))
        # RANC-SAT
        t = RANCSATTransformer(contract=base, overrides=ov, feature_names=ds.feature_names).fit(Xtr, ytr)
        scores.setdefault("RANC-SAT", []).append(_eval(t.transform(Xtr), t.transform(Xte), ytr, yte, ds.task, model))
        r = count_guarantee_violations(Xtr, Xte, ds.feature_names, t.policies_, ov)
        viol_tot["RANC-SAT"] = viol_tot.get("RANC-SAT", 0) + sum(r["counts"].values())

    agg = {k: float(np.mean(v)) for k, v in scores.items() if v}
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    rval = agg.get("RANC-SAT")
    rank = (sum(1 for k, v in agg.items() if k != "RANC-SAT" and v > rval) + 1) if rval is not None else None
    return {"name": ds.name, "task": ds.task, "model": model, "scores": agg,
            "ranked": ranked, "rancsat_rank": rank, "n_methods": len(agg),
            "violations": viol_tot}


def run_realdata_suite(seeds=range(3), verbose=True) -> Dict:
    dsets = load_real_datasets()
    profiles = [profile_dataset(d) for d in dsets]
    results = {"profiles": profiles, "eval": {}}
    for model in ["logreg", "knn"]:
        results["eval"][model] = [evaluate_dataset(d, seeds=seeds, model=model) for d in dsets]
    if verbose:
        print(format_realdata(results))
    return results


def format_realdata(R: Dict) -> str:
    L = ["REAL-DATA VALIDATION", "", "Dataset pathology profiles:",
         f"    {'name':<16}{'feats':>6}{'samp':>7}{'skew':>6}{'heavy':>6}{'sparse':>7}"
         f"{'heterog':>9}  task"]
    for p in R["profiles"]:
        L.append(f"    {p['name']:<16}{p['n_features']:>6}{p['n_samples']:>7}"
                 f"{p['skewed']:>6}{p['heavy_tail']:>6}{p['sparse']:>7}"
                 f"{p['heterogeneity']:>9}  {p['task']}")
    for model, rows in R["eval"].items():
        L += ["", f"=== Downstream model = {model} ===",
              f"    {'dataset':<16}{'RANC-SAT':>10}{'best_global':>13}{'rank':>6}"
              f"{'viol(global/ours)':>20}"]
        for r in rows:
            best_g = max((v for k, v in r["scores"].items()
                          if k not in ("RANC-SAT", "rule_based")), default=float("nan"))
            gviol = sum(v for k, v in r["violations"].items() if k != "RANC-SAT")
            oviol = r["violations"].get("RANC-SAT", 0)
            rs = r["scores"].get("RANC-SAT", float("nan"))
            L.append(f"    {r['name']:<16}{rs:>10.4f}{best_g:>13.4f}"
                     f"{str(r['rancsat_rank'])+'/'+str(r['n_methods']):>6}"
                     f"{str(gviol)+' / '+str(oviol):>20}")
    L += ["", "Note: scores are AUROC (classification) or -RMSE (regression), higher=better.",
          "viol = total declared-invariant breaches summed over seeds (global baselines",
          "combined vs RANC-SAT). RANC-SAT should be 0 by construction."]
    return "\n".join(L)
