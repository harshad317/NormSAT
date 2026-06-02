"""Component ablation: does each mechanism of NormSAT earn its keep?

We toggle four mechanisms OFF, one at a time, and measure the effect versus the full
system on real data:

  (A) hard clauses    -- disable contract feasibility predicates (selection becomes a
                         pure soft-cost ranking over ALL policies). Tests whether the
                         constraints actually change decisions.
  (B) F term          -- drop the fit-for-purpose cost (back to "least transform").
                         Tests the under-transformation fix.
  (C) model-aware     -- ignore the downstream model family in contract derivation.
                         Tests the distance-model distribution-match logic.
  (D) falsification   -- skip the falsification+repair pass. Tests whether repair ever
                         fires / matters.

Metrics per ablation:
  * policy_change_rate : fraction of features whose chosen policy kind differs from full
  * guarantee_violations : declared-invariant breaches on held-out data (full should be 0)
  * mean_auroc          : downstream accuracy (linear model)

A mechanism "earns its keep" if turning it off measurably degrades guarantees or
changes a non-trivial fraction of decisions. Reported honestly -- if an ablation makes
no difference on these data, we say so.
"""
from __future__ import annotations

from typing import Dict, List
import copy
import numpy as np

from sklearn.model_selection import train_test_split

from .schemas import InvarianceContract, NormalizationPolicy, PolicyKind
from .profiling import build_regime_cards
from .contracts import derive_invariance_contracts
from .ledger import build_signal_risk_ledger
from .compiler import compile_policy, soft_costs, hard_violations, _drift_threshold
from .falsification import run_falsification_tests
from .sklearn import RANCSATTransformer
from .heterogeneous import count_guarantee_violations
from .realdata import auto_contract, _eval, RealDataset, load_real_datasets
from . import policies as P
from . import compiler as _C


# --------------------------------------------------------------------------- #
# Variant transformers (each disables one mechanism)
# --------------------------------------------------------------------------- #
def _compile_no_hard(card, contract, ledger, col):
    """Ablation A: ignore hard clauses -> rank ALL per-feature policies by soft cost."""
    from .policies import PER_FEATURE_KINDS, CAPABILITIES
    feasible = [(soft_costs(k, card, contract, ledger, col=col), k) for k in PER_FEATURE_KINDS]
    feasible.sort(key=lambda t: t[0])
    best_cost, best_kind = feasible[0]
    cap = CAPABILITIES[best_kind]
    return NormalizationPolicy(
        feature=card.name, index=card.index, kind=best_kind,
        params=P.fit_params(best_kind, col, card), invertible=cap.invertible,
        is_noop=(best_kind == PolicyKind.IDENTITY), cost_vector=best_cost,
        drift_threshold=_drift_threshold(card))


def _soft_costs_no_F(kind, card, contract, ledger, col=None):
    """Ablation B: zero out the F (fit-for-purpose) term."""
    R, F, U, D, K, C, I = soft_costs(kind, card, contract, ledger, col=col)
    return (R, 0.0, U, D, K, C, I)


def fit_variant(X, contract, model_family, ablation: str):
    """Return (policies, cards, contracts) under a given ablation on training data X."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cards = build_regime_cards(X, [f"f{i}" for i in range(X.shape[1])])

    fam = None if ablation == "no_model_aware" else model_family
    contracts = derive_invariance_contracts(cards, contract, fam, None)
    ledger = build_signal_risk_ledger(cards, contracts, X_train=X)

    policies = []
    for card, con in zip(cards, contracts):
        col = X[:, card.index]
        if ablation == "no_hard":
            pol = _compile_no_hard(card, con, ledger, col)
        elif ablation == "no_F":
            orig = _C.soft_costs
            _C.soft_costs = _soft_costs_no_F
            try:
                pol = compile_policy(card, con, ledger, col=col)
            finally:
                _C.soft_costs = orig
        else:
            pol = compile_policy(card, con, ledger, col=col)
        policies.append(pol)

    if ablation != "no_falsification":
        fres = run_falsification_tests(policies, cards, contracts, X)
        # minimal repair: fall back failed policies to identity (mirrors transformer)
        for i, f in enumerate(fres):
            if not f.passed:
                c = cards[i]
                policies[i] = NormalizationPolicy(
                    feature=c.name, index=c.index, kind=PolicyKind.IDENTITY,
                    params={}, invertible=True, is_noop=True,
                    drift_threshold=_drift_threshold(c))
    return policies, cards, contracts


def _apply(policies, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Z = np.empty_like(X)
    for pol in policies:
        Z[:, pol.index] = P.apply(pol, X[:, pol.index])
    return Z


# --------------------------------------------------------------------------- #
# Run the study
# --------------------------------------------------------------------------- #
ABLATIONS = ["FULL", "no_hard", "no_F", "no_model_aware", "no_falsification"]


def run_ablation_study(datasets: List[RealDataset] = None, seeds=range(5),
                       model="logreg") -> Dict:
    datasets = datasets or load_real_datasets()
    rows = {a: {"policy_change": [], "violations": 0, "auroc": []} for a in ABLATIONS}

    for ds in datasets:
        base, ov = auto_contract(ds, model_family="linear")
        # the per-feature overrides ARE the contract intent; for ablation we apply the
        # global contract + model family so toggles have something to act on.
        fam = "knn" if model == "knn" else "linear"
        strat = ds.y if ds.task == "classification" and np.unique(ds.y).size > 1 else None
        for s in seeds:
            if ds.task == "classification" and strat is not None:
                cnt = np.unique(ds.y, return_counts=True)[1]
                if cnt.min() < 2:
                    continue
            Xtr, Xte, ytr, yte = train_test_split(
                ds.X, ds.y, test_size=0.3, random_state=int(s), stratify=strat)
            # FULL reference via the real transformer (uses overrides)
            full = RANCSATTransformer(contract=base, overrides=ov,
                                      feature_names=ds.feature_names).fit(Xtr, ytr)
            full_kinds = [p.kind for p in full.policies_]
            for ab in ABLATIONS:
                if ab == "FULL":
                    pols = full.policies_
                else:
                    pols, cards, contracts = fit_variant(Xtr, base, fam, ab)
                # policy change vs FULL
                change = np.mean([a.kind != b for a, b in zip(pols, full_kinds)]) \
                    if ab != "FULL" else 0.0
                rows[ab]["policy_change"].append(change)
                # violations on held-out (use the override contract as declared intent)
                v = count_guarantee_violations(Xtr, Xte, ds.feature_names, pols, ov)
                rows[ab]["violations"] += sum(v["counts"].values())
                # accuracy
                try:
                    Ztr, Zte = _apply(pols, Xtr), _apply(pols, Xte)
                    if np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte)):
                        rows[ab]["auroc"].append(_eval(Ztr, Zte, ytr, yte, ds.task, model))
                except Exception:
                    pass

    summary = {}
    for ab in ABLATIONS:
        summary[ab] = {
            "policy_change_rate": float(np.mean(rows[ab]["policy_change"])) if rows[ab]["policy_change"] else 0.0,
            "guarantee_violations": rows[ab]["violations"],
            "mean_auroc": float(np.mean(rows[ab]["auroc"])) if rows[ab]["auroc"] else float("nan"),
        }
    return summary


def format_ablation_study(summary: Dict) -> str:
    L = ["COMPONENT ABLATION (effect of disabling each mechanism vs FULL system)",
         "policy_change = frac. of features whose chosen transform differs from FULL", ""]
    L.append(f"    {'variant':<18}{'policy_change':>15}{'violations':>12}{'mean_auroc':>12}")
    L.append("    " + "-" * 57)
    for ab, r in summary.items():
        L.append(f"    {ab:<18}{r['policy_change_rate']:>15.2f}"
                 f"{r['guarantee_violations']:>12d}{r['mean_auroc']:>12.4f}")
    L.append("")
    L.append("    Reading: a mechanism earns its keep if disabling it raises violations")
    L.append("    or changes a non-trivial fraction of policy decisions.")
    return "\n".join(L)
