"""Per-regime diagnosis: WHY does RANC-SAT underperform?

For each synthetic regime we separate three distinct failure modes:

  (A) WRONG POLICY  -- the compiler did not pick the oracle/best-available policy.
  (B) RIGHT POLICY, WEAK -- it picked a good policy but the best global scaler still
      beats it (i.e. the transform family itself isn't the lever in this regime).
  (C) CONTRACT MISMATCH -- the contract we declared doesn't match what the downstream
      model actually wants (e.g. KNN wants rank-equalisation; we declared magnitude
      preservation).

Method: for every regime + downstream model, compute
  * the policy RANC-SAT chose (per feature)
  * the SINGLE best global scaler ("oracle-global") by held-out metric
  * the per-feature "oracle-applied" score = apply each candidate policy per feature,
    keep the best-scoring assignment found by exhaustive per-feature search
    (an upper bound on what ANY per-feature compiler could achieve with this DSL)
  * gap_to_global  = RANC-SAT  - best_global
  * gap_to_oraclePF = RANC-SAT - oracle_per_feature
If oracle_per_feature also fails to beat best_global -> mode (B): per-feature handling
is not the lever. If oracle_per_feature DOES beat global but RANC-SAT doesn't ->
mode (A) or (C): the lever exists but our selection/contract missed it.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional
import numpy as np

from sklearn.model_selection import train_test_split

from .schemas import NormalizationPolicy, PolicyKind
from .profiling import build_regime_cards
from .sklearn import RANCSATTransformer
from . import policies as P
from . import synthetic as S
from .benchmark import _fit_eval, _primary_metric


# candidate per-feature policies the oracle search is allowed to use
_ORACLE_KINDS = [
    PolicyKind.IDENTITY, PolicyKind.AFFINE_STANDARD, PolicyKind.AFFINE_ROBUST,
    PolicyKind.MINMAX, PolicyKind.MAXABS, PolicyKind.ROBUST_MAXABS,
    PolicyKind.CLIP_ROBUST, PolicyKind.LOG1P, PolicyKind.QUANTILE_NORMAL,
]
_GLOBAL_KINDS = [PolicyKind.IDENTITY, PolicyKind.AFFINE_STANDARD,
                 PolicyKind.AFFINE_ROBUST, PolicyKind.MINMAX, PolicyKind.MAXABS,
                 PolicyKind.QUANTILE_NORMAL, PolicyKind.LOG1P]


def _apply_kinds(Xtr, Xte, names, kinds):
    """Fit per-feature `kinds[j]` on train, transform train+test. Returns (Ztr, Zte)
    or (None, None) if any column is non-finite (illegal domain)."""
    cards = build_regime_cards(Xtr, names)
    Ztr = np.empty_like(Xtr); Zte = np.empty_like(Xte)
    for j, card in enumerate(cards):
        k = kinds[j]
        try:
            pol = NormalizationPolicy(feature=card.name, index=j, kind=k,
                                      params=P.fit_params(k, Xtr[:, j], card),
                                      invertible=P.CAPABILITIES[k].invertible)
            Ztr[:, j] = P.apply(pol, Xtr[:, j]); Zte[:, j] = P.apply(pol, Xte[:, j])
        except Exception:
            return None, None
    if not (np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))):
        return None, None
    return Ztr, Zte


def _score(Xtr, Xte, ytr, yte, names, kinds, task, model):
    Ztr, Zte = _apply_kinds(Xtr, Xte, names, kinds)
    if Ztr is None:
        return None
    return _primary_metric(_fit_eval(Ztr, Zte, ytr, yte, task, model=model), task)


def diagnose_regime(gen, model="knn", seeds=range(5)) -> Dict:
    task = gen(seed=0).task
    higher = (task == "classification")
    rows = {"rancsat": [], "best_global": [], "oracle_pf": [],
            "global_name": [], "rancsat_kinds": [], "oracle_kinds": []}

    for s in seeds:
        reg = gen(seed=int(s))
        names = reg.feature_names
        Xtr, Xte, ytr, yte = train_test_split(
            reg.X, reg.y, test_size=0.3, random_state=int(s),
            stratify=reg.y if task == "classification" else None)

        # RANC-SAT
        overrides = (S.per_feature_contracts_for_outliers()
                     if reg.name == "signal_vs_noise_outliers" else None)
        t = RANCSATTransformer(contract=reg.contract, overrides=overrides,
                               feature_names=names).fit(Xtr, ytr)
        rk = [p.kind for p in t.policies_]
        rows["rancsat"].append(_score(Xtr, Xte, ytr, yte, names, rk, task, model))
        rows["rancsat_kinds"].append([k.value for k in rk])

        # best single GLOBAL scaler
        best_g, best_gname = None, None
        for k in _GLOBAL_KINDS:
            sc = _score(Xtr, Xte, ytr, yte, names, [k] * reg.X.shape[1], task, model)
            if sc is None:
                continue
            if best_g is None or (sc > best_g if higher else sc < best_g):
                best_g, best_gname = sc, k.value
        rows["best_global"].append(best_g); rows["global_name"].append(best_gname)

        # ORACLE per-feature: GREEDY coordinate ascent over per-feature kinds. Start
        # from the best global, then for each feature pick the kind that most improves
        # the held-out metric, holding others fixed; 2 passes. This is a tractable
        # lower-bound on the true per-feature oracle (and thus a conservative estimate
        # of available headroom -- if even this beats global, the lever is real).
        d = reg.X.shape[1]
        cur = [PolicyKind(best_gname)] * d if best_gname else [PolicyKind.IDENTITY] * d
        cur_score = _score(Xtr, Xte, ytr, yte, names, cur, task, model)
        for _ in range(2):
            for j in range(d):
                for k in _ORACLE_KINDS:
                    trial = list(cur); trial[j] = k
                    sc = _score(Xtr, Xte, ytr, yte, names, trial, task, model)
                    if sc is None:
                        continue
                    if cur_score is None or (sc > cur_score if higher else sc < cur_score):
                        cur_score, cur = sc, trial
        rows["oracle_pf"].append(cur_score)
        rows["oracle_kinds"].append([k.value for k in cur])

    def _m(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    rancsat, glob, oracle = _m(rows["rancsat"]), _m(rows["best_global"]), _m(rows["oracle_pf"])
    sign = 1 if higher else -1
    gap_global = (sign * (rancsat - glob)) if (rancsat and glob) else None
    headroom = (sign * (oracle - glob)) if (oracle and glob) else None     # does PF help at all?
    left_on_table = (sign * (oracle - rancsat)) if (oracle and rancsat) else None

    # classify failure mode
    EPS = 0.002
    if headroom is not None and headroom <= EPS:
        mode = "B: per-feature handling is NOT the lever (oracle-PF can't beat global)"
    elif left_on_table is not None and left_on_table > EPS:
        mode = "A/C: lever EXISTS but RANC-SAT missed it (wrong policy or contract)"
    else:
        mode = "OK: RANC-SAT ~= oracle per-feature"

    return {
        "regime": gen(seed=0).name, "task": task, "model": model,
        "rancsat": rancsat, "best_global": glob, "best_global_name":
            max(set(rows["global_name"]), key=rows["global_name"].count),
        "oracle_pf": oracle,
        "gap_to_global": gap_global, "pf_headroom_over_global": headroom,
        "left_on_table_vs_oracle": left_on_table,
        "mode": mode,
        "rancsat_kinds_seed0": rows["rancsat_kinds"][0],
        "oracle_kinds_seed0": rows["oracle_kinds"][0],
    }


def run_diagnosis(model="knn", seeds=range(5), regimes=None) -> List[Dict]:
    regimes = regimes or S.ALL_REGIMES
    out = [diagnose_regime(g, model=model, seeds=seeds) for g in regimes]
    print(format_diagnosis(out))
    return out


def format_diagnosis(rows: List[Dict]) -> str:
    L = [f"DIAGNOSIS (model={rows[0]['model'] if rows else '?'}; "
         f"gaps in metric units, +=RANC-SAT better)", ""]
    for r in rows:
        L.append(f"### {r['regime']}  [{r['task']}]   {r['mode']}")
        L.append(f"    RANC-SAT={_n(r['rancsat'])}   best_global={_n(r['best_global'])}"
                 f" ({r['best_global_name']})   oracle_PF={_n(r['oracle_pf'])}")
        L.append(f"    gap_to_global={_n(r['gap_to_global'])}   "
                 f"PF_headroom_over_global={_n(r['pf_headroom_over_global'])}   "
                 f"left_on_table_vs_oracle={_n(r['left_on_table_vs_oracle'])}")
        L.append(f"    RANC-SAT chose : {r['rancsat_kinds_seed0']}")
        L.append(f"    oracle-PF chose: {r['oracle_kinds_seed0']}")
        L.append("")
    return "\n".join(L)


def _n(x):
    return f"{x:+.4f}" if isinstance(x, float) else "n/a"
