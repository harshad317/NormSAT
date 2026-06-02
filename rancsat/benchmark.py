"""Benchmark harness.

Two entry points:

    run_oracle_benchmark()  -- on synthetic regimes with known-correct policies, report
                               how often RANC-SAT recovers the oracle policy ("policy
                               accuracy") -- this is the correctness proof.

    run_downstream_benchmark() -- on any (X, y), compare downstream model quality of
                               RANC-SAT vs classical scalers vs the rule-based and CV
                               selectors, plus model-fit count and leakage status.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional
import numpy as np

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.base import clone

from .schemas import InvarianceContract, PolicyKind
from .sklearn import RANCSATTransformer
from .ablation import RuleBasedSelector, CVBruteForceSelector, policy_agreement
from . import synthetic
from . import policies as P
from .profiling import build_regime_cards
from .schemas import NormalizationPolicy


# --------------------------------------------------------------------------- #
# Oracle (policy-correctness) benchmark
# --------------------------------------------------------------------------- #
def run_oracle_benchmark(regimes: Optional[List[Callable]] = None,
                         use_overrides: bool = True) -> List[Dict]:
    regimes = regimes or synthetic.ALL_REGIMES
    rows = []
    for gen in regimes:
        reg = gen()
        overrides = None
        if use_overrides and reg.name == "signal_vs_noise_outliers":
            overrides = synthetic.per_feature_contracts_for_outliers()
        t = RANCSATTransformer(contract=reg.contract,
                               feature_names=reg.feature_names,
                               overrides=overrides,
                               inverse_required=reg.contract.allow_inverse_transform)
        t.fit(reg.X, reg.y)
        chosen = [p.kind for p in t.policies_]
        match = np.mean([c == o for c, o in zip(chosen, reg.oracle_kinds)])
        rows.append({
            "regime": reg.name,
            "policy_accuracy": float(match),
            "chosen": [c.value for c in chosen],
            "oracle": [o.value for o in reg.oracle_kinds],
            "noop_fraction": t.audit_report().noop_fraction,
        })
    return rows


# --------------------------------------------------------------------------- #
# Downstream-quality benchmark
# --------------------------------------------------------------------------- #
def _fit_eval(Ztr, Zte, ytr, yte, task, model="logreg"):
    if task == "classification":
        if model == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            est = KNeighborsClassifier(n_neighbors=15)
        elif model == "svm_rbf":
            from sklearn.svm import SVC
            est = SVC(probability=True)
        else:
            est = LogisticRegression(max_iter=500)
        est.fit(Ztr, ytr)
        pred = est.predict(Zte)
        out = {"accuracy": accuracy_score(yte, pred)}
        try:
            out["auroc"] = roc_auc_score(yte, est.predict_proba(Zte)[:, 1])
        except Exception:
            out["auroc"] = float("nan")
        return out
    if model == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        est = KNeighborsRegressor(n_neighbors=15)
    else:
        est = Ridge()
    est.fit(Ztr, ytr)
    pred = est.predict(Zte)
    return {"rmse": float(np.sqrt(mean_squared_error(yte, pred)))}


def _classical(X, kind, names):
    cards = build_regime_cards(X, names)
    Z = np.empty_like(X)
    for card in cards:
        pol = NormalizationPolicy(feature=card.name, index=card.index, kind=kind,
                                  params=P.fit_params(kind, X[:, card.index], card),
                                  invertible=P.CAPABILITIES[kind].invertible)
        Z[:, card.index] = P.apply(pol, X[:, card.index])
    return Z


def run_downstream_benchmark(X, y, task="classification",
                             contract: Optional[InvarianceContract] = None,
                             overrides=None, feature_names=None,
                             seed=0, model="logreg") -> List[Dict]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y if task == "classification" else None)
    rows = []

    # classical scalers (fit on train fold only -> leakage-safe)
    classical = {
        "none": PolicyKind.IDENTITY,
        "standard": PolicyKind.AFFINE_STANDARD,
        "robust": PolicyKind.AFFINE_ROBUST,
        "minmax": PolicyKind.MINMAX,
        "maxabs": PolicyKind.MAXABS,
        "quantile_normal": PolicyKind.QUANTILE_NORMAL,
    }
    for label, kind in classical.items():
        try:
            from .profiling import build_regime_cards as _b
            cards = _b(Xtr, names)
            pols = [NormalizationPolicy(feature=c.name, index=c.index, kind=kind,
                    params=P.fit_params(kind, Xtr[:, c.index], c),
                    invertible=P.CAPABILITIES[kind].invertible) for c in cards]
            Ztr = np.column_stack([P.apply(p, Xtr[:, p.index]) for p in pols])
            Zte = np.column_stack([P.apply(p, Xte[:, p.index]) for p in pols])
            if not (np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))):
                continue
            m = _fit_eval(Ztr, Zte, ytr, yte, task, model=model)
            m.update({"method": label, "n_model_fits": 0, "leakage_safe": True})
            rows.append(m)
        except Exception as e:
            rows.append({"method": label, "error": str(e)})

    # rule-based selector
    rb = RuleBasedSelector(feature_names=names).fit(Xtr, ytr)
    m = _fit_eval(rb.transform(Xtr), rb.transform(Xte), ytr, yte, task, model=model)
    m.update({"method": "rule_based", "n_model_fits": 0, "leakage_safe": True})
    rows.append(m)

    # CV brute-force (AutoML-style) selector
    cvs = CVBruteForceSelector(task=task, feature_names=names).fit(Xtr, ytr)
    m = _fit_eval(cvs.transform(Xtr), cvs.transform(Xte), ytr, yte, task, model=model)
    m.update({"method": f"cv_search({cvs.best_kind_.value})",
              "n_model_fits": cvs.n_model_fits, "leakage_safe": True})
    rows.append(m)

    # RANC-SAT
    t = RANCSATTransformer(contract=contract, feature_names=names, overrides=overrides)
    t.fit(Xtr, ytr)
    m = _fit_eval(t.transform(Xtr), t.transform(Xte), ytr, yte, task, model=model)
    m.update({"method": "RANC-SAT", "n_model_fits": 0, "leakage_safe": True,
              "noop_fraction": t.audit_report().noop_fraction})
    rows.append(m)
    return rows


def format_table(rows: List[Dict]) -> str:
    if not rows:
        return "(no results)"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    widths = {k: max(len(k), *(len(_fmt(r.get(k, ""))) for r in rows)) for k in keys}
    head = "  ".join(k.ljust(widths[k]) for k in keys)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("  ".join(_fmt(r.get(k, "")).ljust(widths[k]) for k in keys))
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list):
        return "[" + ",".join(map(str, v)) + "]"
    return str(v)


# --------------------------------------------------------------------------- #
# Multi-seed ablation -- the make-or-break honesty check
# --------------------------------------------------------------------------- #
def _primary_metric(row: Dict, task: str):
    if task == "classification":
        return row.get("auroc", row.get("accuracy"))
    return row.get("rmse")


def run_multiseed_ablation(regimes: Optional[List[Callable]] = None,
                           seeds=range(5), model="knn",
                           verbose: bool = True) -> Dict:
    """For every regime, across several seeds, compare RANC-SAT against a heuristic
    menu (rule_based), an AutoML CV search, and strong global scalers (incl. the
    tough quantile-normal). Reports mean +/- std of the primary metric, RANC-SAT's
    rank, model-fit counts, and policy agreement vs the rule-based selector.

    Higher-is-better for classification (AUROC); lower-is-better for regression (RMSE).
    """
    from . import synthetic as S
    regimes = regimes or S.ALL_REGIMES
    methods = ["none", "standard", "robust", "quantile_normal",
               "rule_based", "cv_search", "RANC-SAT"]
    results: Dict = {}

    for gen in regimes:
        ref = gen(seed=0)
        task = ref.task
        per_method = {m: [] for m in methods}
        fit_counts = {m: [] for m in methods}
        agreements = []

        for s in seeds:
            reg = gen(seed=int(s))
            overrides = (S.per_feature_contracts_for_outliers()
                         if reg.name == "signal_vs_noise_outliers" else None)
            rows = run_downstream_benchmark(
                reg.X, reg.y, task=task, contract=reg.contract, overrides=overrides,
                feature_names=reg.feature_names, seed=int(s), model=model)
            for row in rows:
                label = row.get("method", "")
                key = "cv_search" if label.startswith("cv_search") else label
                if key in per_method and "error" not in row:
                    per_method[key].append(_primary_metric(row, task))
                    fit_counts[key].append(row.get("n_model_fits", 0))

            # policy agreement: RANC-SAT vs the heuristic menu
            t = RANCSATTransformer(contract=reg.contract, overrides=overrides,
                                   feature_names=reg.feature_names).fit(reg.X, reg.y)
            rb = RuleBasedSelector(feature_names=reg.feature_names).fit(reg.X, reg.y)
            agreements.append(policy_agreement(t.policies_, rb.policies_))

        agg = {}
        for m in methods:
            vals = [v for v in per_method[m] if v is not None]
            if vals:
                agg[m] = (float(np.mean(vals)), float(np.std(vals)),
                          float(np.mean(fit_counts[m])))
        # rank RANC-SAT among methods (1 = best). Tie-robust: count strictly-better.
        higher_better = (task == "classification")
        ranked = sorted(agg.items(),
                        key=lambda kv: kv[1][0], reverse=higher_better)
        order = [m for m, _ in ranked]
        if "RANC-SAT" in agg:
            rval = agg["RANC-SAT"][0]
            better = sum(1 for m, v in agg.items()
                         if m != "RANC-SAT" and ((v[0] > rval) if higher_better
                                                 else (v[0] < rval)))
            rank = better + 1
        else:
            rank = None

        results[reg.name] = {
            "task": task, "agg": agg, "rank_of_rancsat": rank,
            "n_methods": len(agg), "best_method": order[0] if order else None,
            "policy_agreement_vs_rule": float(np.mean(agreements)),
        }

    if verbose:
        print(format_ablation(results, model))
    return results


def format_ablation(results: Dict, model: str = "") -> str:
    lines = [f"MULTI-SEED ABLATION  (downstream model = {model};  "
             f"classification=AUROC higher-better, regression=RMSE lower-better)", ""]
    for regime, r in results.items():
        better = "lower" if r["task"] == "regression" else "higher"
        lines.append(f"### {regime}   [{r['task']}, {better}-is-better]")
        lines.append(f"    RANC-SAT rank: {r['rank_of_rancsat']}/{r['n_methods']}   "
                     f"best={r['best_method']}   "
                     f"policy-agreement vs rule-based menu={r['policy_agreement_vs_rule']:.0%}")
        for m, (mu, sd, fits) in sorted(
                r["agg"].items(), key=lambda kv: kv[1][0],
                reverse=(r["task"] == "classification")):
            star = "  <-- RANC-SAT" if m == "RANC-SAT" else ""
            fitstr = f"  fits={fits:.0f}" if fits else ""
            lines.append(f"      {m:<16} {mu:.4f} +/- {sd:.4f}{fitstr}{star}")
        lines.append("")
    return "\n".join(lines)
