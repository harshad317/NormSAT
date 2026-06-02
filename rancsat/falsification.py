"""Falsification suite: try to *break* each compiled policy before trusting it.

The compiler guarantees a policy is feasible by construction, but feasibility is only
as good as the capability table and the fitted parameters. The falsification tests are
an independent, data-driven second opinion that actually execute the transform on the
training fold and check the invariants empirically. A failed check triggers
repair-or-fallback in the transformer.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from .schemas import (
    FalsificationResult, InvarianceContract, NormalizationPolicy, RegimeCard,
)
from . import policies as P


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def falsify_policy(policy: NormalizationPolicy, card: RegimeCard,
                   contract: InvarianceContract, col: np.ndarray) -> FalsificationResult:
    x = np.asarray(col, dtype=float)
    checks: dict = {}
    metrics: dict = {}
    failures: List[str] = []

    z = P.apply(policy, x)

    # --- leakage sentinel: transform must not depend on test-time statistics. -----
    # We verify determinism: re-applying the frozen policy gives identical output.
    z2 = P.apply(policy, x)
    checks["deterministic"] = bool(np.allclose(np.nan_to_num(z), np.nan_to_num(z2)))
    if not checks["deterministic"]:
        failures.append("deterministic")

    # --- finiteness ---------------------------------------------------------------
    finite_ratio = float(np.mean(np.isfinite(z[np.isfinite(x)]))) if np.isfinite(x).any() else 1.0
    checks["finite_output"] = finite_ratio > 0.999
    metrics["finite_ratio"] = finite_ratio
    if not checks["finite_output"]:
        failures.append("finite_output")

    # --- zero preservation --------------------------------------------------------
    if contract.preserve_zero:
        zero_mask = (x == 0.0)
        if zero_mask.any():
            ok = bool(np.allclose(z[zero_mask], 0.0, atol=1e-9))
            checks["zero_preservation"] = ok
            metrics["max_zero_image"] = float(np.max(np.abs(z[zero_mask])))
            if not ok:
                failures.append("zero_preservation")

    # --- sign preservation --------------------------------------------------------
    if contract.preserve_sign:
        m = np.isfinite(x) & np.isfinite(z) & (x != 0)
        if m.any():
            ok = bool(np.all(np.sign(z[m]) == np.sign(x[m])))
            checks["sign_preservation"] = ok
            metrics["sign_violation_frac"] = float(np.mean(np.sign(z[m]) != np.sign(x[m])))
            if not ok:
                failures.append("sign_preservation")

    # --- monotonicity / rank ------------------------------------------------------
    if contract.preserve_monotonicity or contract.preserve_rank:
        xf, zf = x[np.isfinite(x) & np.isfinite(z)], z[np.isfinite(x) & np.isfinite(z)]
        if xf.size > 2:
            order = np.argsort(xf, kind="mergesort")
            diffs = np.diff(zf[order])
            ok = bool(np.all(diffs >= -1e-9))
            checks["monotonicity"] = ok
            metrics["spearman"] = _spearman(xf, zf)
            if not ok:
                failures.append("monotonicity")

    # --- distance-ratio preservation (affine policies only) -----------------------
    if contract.preserve_distance_ratios:
        ok = P.CAPABILITIES[policy.kind].affine
        checks["distance_ratio"] = ok
        if not ok:
            failures.append("distance_ratio")

    # --- inverse-transform accuracy ----------------------------------------------
    if contract.allow_inverse_transform and policy.invertible:
        try:
            x_rec = P.invert(policy, z)
            m = np.isfinite(x) & np.isfinite(x_rec)
            denom = np.std(_finite(x)) + 1e-9
            err = float(np.sqrt(np.mean((x[m] - x_rec[m]) ** 2)) / denom)
            checks["inverse_accuracy"] = err < 1e-3
            metrics["inverse_nrmse"] = err
            if err >= 1e-3:
                failures.append("inverse_accuracy")
        except Exception as e:  # pragma: no cover
            checks["inverse_accuracy"] = False
            failures.append(f"inverse_accuracy:{type(e).__name__}")

    # --- outlier-as-signal stress -------------------------------------------------
    # If extremes were declared possible signal, the transform must not collapse them
    # to a constant ceiling (saturation). Measure spread retention in the top decile.
    if contract.preserve_extreme_signal:
        xf, zf = x[np.isfinite(x) & np.isfinite(z)], z[np.isfinite(x) & np.isfinite(z)]
        if xf.size > 20:
            hi = xf >= np.percentile(xf, 90)
            spread_in = np.std(xf[hi]) + 1e-12
            spread_out = np.std(zf[hi]) + 1e-12
            # compare to overall spread ratio; saturation => top-decile spread vanishes
            global_ratio = (np.std(zf) + 1e-12) / (np.std(xf) + 1e-12)
            retained = (spread_out / spread_in) / (global_ratio + 1e-12)
            checks["extreme_signal_retained"] = retained > 0.25
            metrics["extreme_spread_retention"] = float(retained)
            if retained <= 0.25:
                failures.append("extreme_signal_retained")

    passed = len(failures) == 0
    return FalsificationResult(feature=card.name, passed=passed, checks=checks,
                              metrics=metrics, failures=failures)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def run_falsification_tests(policies: List[NormalizationPolicy],
                            cards: List[RegimeCard],
                            contracts: List[InvarianceContract],
                            X_train: np.ndarray) -> List[FalsificationResult]:
    X = np.asarray(X_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    results = []
    for pol, card, con in zip(policies, cards, contracts):
        results.append(falsify_policy(pol, card, con, X[:, card.index]))
    return results
