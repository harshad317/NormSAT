"""Efficiency + audit-coverage accounting.

The third defensible axis (after accuracy-in-regime and guarantees) is COST: how a
constraint-compiled policy is selected without spending downstream model fits, and how
much of its decision is machine-auditable.

Two tables:

  run_efficiency_table()   -- for each selection method, count the number of DOWNSTREAM
                              MODEL FITS consumed to choose the normalization policy.
                              RANC-SAT and rule-based use 0 (they compile from data
                              statistics); AutoML/CV search scales with grid x folds.

  audit_coverage()         -- fraction of RANC-SAT's decision that is serialized into
                              the certificate: policies justified, clauses checked,
                              rejected candidates with reasons, falsification outcomes.
                              A proxy for "can a reviewer/regulator reconstruct WHY".
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional
import numpy as np

from sklearn.model_selection import train_test_split

from .sklearn import RANCSATTransformer
from .ablation import RuleBasedSelector, CVBruteForceSelector, _SEARCH_KINDS
from . import synthetic as S
from .heterogeneous import make_heterogeneous


# --------------------------------------------------------------------------- #
# 1. Model-fit accounting
# --------------------------------------------------------------------------- #
def run_efficiency_table(seeds=range(5), cv_folds=3, verbose=True) -> Dict:
    """Count downstream model fits spent on POLICY SELECTION per method.

    The comparison is mechanism-level and dataset-independent for the compiled methods:
      * RANC-SAT        : 0 model fits  (compiles from Regime Cards + contract)
      * rule_based      : 0 model fits  (heuristic over statistics)
      * cv_search       : |grid| x folds per fit() call  (AutoML-style)
    We also report wall-clock-independent "selection complexity": number of candidate
    evaluations the selector performs (a hardware-neutral proxy).
    """
    grid = len(_SEARCH_KINDS)
    rows: Dict[str, Dict] = {}

    # measured fit counts on a representative dataset (heterogeneous)
    cv_fits = []
    for s in seeds:
        ds = make_heterogeneous(seed=int(s))
        Xtr, _, ytr, _ = train_test_split(ds.X, ds.y, test_size=0.3,
                                          random_state=int(s), stratify=ds.y)
        cvs = CVBruteForceSelector(task="classification", cv=cv_folds,
                                   feature_names=ds.feature_names).fit(Xtr, ytr)
        cv_fits.append(cvs.n_model_fits)

    rows["RANC-SAT"] = {
        "model_fits": 0, "selection_evals": grid,  # enumerates DSL, no model fit
        "mechanism": "compile from Regime Cards + contract (constraint solve)",
    }
    rows["rule_based"] = {
        "model_fits": 0, "selection_evals": 1,
        "mechanism": "heuristic if/else over feature statistics",
    }
    rows["cv_search (AutoML)"] = {
        "model_fits": int(np.mean(cv_fits)),
        "selection_evals": grid * cv_folds,
        "mechanism": f"grid({grid}) x cv({cv_folds}) downstream model fits",
    }
    # how AutoML cost scales with the search space
    rows["cv_search@grid=20,cv=5"] = {
        "model_fits": 20 * 5, "selection_evals": 20 * 5,
        "mechanism": "illustrates scaling: 20 candidates x 5 folds",
    }

    if verbose:
        print(_fmt_efficiency(rows, cv_folds))
    return rows


def _fmt_efficiency(rows: Dict, cv_folds: int) -> str:
    L = [f"MODEL-FIT EFFICIENCY (downstream fits spent SELECTING the normalization; "
         f"lower=better)", ""]
    w = max(len(m) for m in rows)
    L.append(f"    {'method':<{w}}   model_fits   sel_evals   mechanism")
    L.append("    " + "-" * (w + 48))
    for m, d in rows.items():
        L.append(f"    {m:<{w}}   {d['model_fits']:>10d}   {d['selection_evals']:>9d}   "
                 f"{d['mechanism']}")
    L.append("")
    L.append("    Note: RANC-SAT's selection is O(|DSL| x n_features) statistic "
             "evaluations,")
    L.append("    independent of the downstream model. AutoML search cost grows with "
             "the")
    L.append("    candidate grid x CV folds and must refit the model each time.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 2. Audit coverage
# --------------------------------------------------------------------------- #
def audit_coverage(transformer: RANCSATTransformer) -> Dict:
    """Quantify how much of the decision is machine-auditable in the certificate."""
    rep = transformer.audit_report()
    n = len(rep.policies)
    justified = sum(1 for p in rep.policies if p.get("notes"))
    with_rejected = sum(1 for p in rep.policies if p.get("rejected"))
    with_costs = sum(1 for p in rep.policies if p.get("cost_vector"))
    falsified = len(rep.falsification)
    # total candidate decisions recorded (chosen + rejected-with-reason)
    total_candidates = sum(1 + len(p.get("rejected", [])) for p in rep.policies)
    reasoned_candidates = sum(
        (1 if p.get("notes") else 0) + len(p.get("rejected", []))
        for p in rep.policies)
    return {
        "n_features": n,
        "policies_justified": _frac(justified, n),
        "policies_with_rejected_alternatives": _frac(with_rejected, n),
        "policies_with_cost_vector": _frac(with_costs, n),
        "features_falsification_tested": _frac(falsified, n),
        "decision_traceability": _frac(reasoned_candidates, total_candidates),
        "leakage_guard": rep.leakage_guard,
        "certificate_serializable": _is_json(rep),
    }


def run_audit_coverage(seeds=range(3), verbose=True) -> Dict:
    covs = []
    for s in seeds:
        ds = make_heterogeneous(seed=int(s))
        t = RANCSATTransformer(contract=ds.contract, overrides=ds.overrides,
                               feature_names=ds.feature_names).fit(ds.X, ds.y)
        covs.append(audit_coverage(t))
    agg = {}
    for k in covs[0]:
        vals = [c[k] for c in covs]
        if isinstance(vals[0], (int, float)):
            agg[k] = float(np.mean(vals))
        else:
            agg[k] = vals[0]
    if verbose:
        print(_fmt_coverage(agg))
    return agg


def _fmt_coverage(c: Dict) -> str:
    L = ["AUDIT COVERAGE (fraction of the decision that is machine-auditable)", ""]
    for k, v in c.items():
        if k == "n_features":
            vs = f"{v:.0f}"
        elif isinstance(v, float):
            vs = f"{v:.0%}"
        else:
            vs = str(v)
        L.append(f"    {k:<38} {vs}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def _frac(a, b):
    return float(a) / float(b) if b else 0.0


def _is_json(rep) -> bool:
    try:
        rep.to_json()
        return True
    except Exception:
        return False
