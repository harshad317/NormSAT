"""The Policy DSL: capability metadata + fit / transform / inverse for each PolicyKind.

Two things live here:

1. ``CAPABILITIES`` -- a static table of *what each transform guarantees* (does it
   preserve zero? sign? rank? is it affine? invertible? what domain does it need?).
   The compiler reads this table to decide hard feasibility against a contract.

2. ``apply`` / ``invert`` -- the numeric execution of a *fitted* NormalizationPolicy.

All fitting consumes only the RegimeCard (train-only statistics) or the training
column itself; nothing here ever sees test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import stats, special

from .schemas import NormalizationPolicy, PolicyKind, RegimeCard, SignPattern


# --------------------------------------------------------------------------- #
# Capability table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Capability:
    affine: bool                 # linear map -> preserves distance ratios
    monotone: bool               # order preserving -> preserves rank
    preserves_zero: bool         # f(0) == 0
    preserves_sign: bool         # sign(f(x)) == sign(x)
    invertible: bool
    removes_location: bool       # makes output shift-invariant
    removes_scale: bool          # makes output scale-invariant
    bounded_output: bool
    robust_to_outliers: bool     # damps extremes
    keeps_extreme_signal: bool   # does NOT clip/saturate extremes
    interpretable: bool
    requires_positive: bool      # domain x > 0
    requires_nonnegative: bool   # domain x >= 0
    densifies: bool              # turns structural zeros into nonzeros
    complexity: float            # 0..1 model/audit complexity
    runtime: float               # 0..1 relative cost


CAPABILITIES: Dict[PolicyKind, Capability] = {
    PolicyKind.IDENTITY: Capability(
        affine=True, monotone=True, preserves_zero=True, preserves_sign=True,
        invertible=True, removes_location=False, removes_scale=False,
        bounded_output=False, robust_to_outliers=False, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=False, complexity=0.0, runtime=0.0),
    PolicyKind.AFFINE_STANDARD: Capability(
        affine=True, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=True, removes_location=True, removes_scale=True,
        bounded_output=False, robust_to_outliers=False, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.15, runtime=0.1),
    PolicyKind.AFFINE_ROBUST: Capability(
        affine=True, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=True, removes_location=True, removes_scale=True,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.2, runtime=0.1),
    PolicyKind.MINMAX: Capability(
        affine=True, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=True, removes_location=True, removes_scale=True,
        bounded_output=True, robust_to_outliers=False, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.15, runtime=0.1),
    PolicyKind.MAXABS: Capability(
        affine=True, monotone=True, preserves_zero=True, preserves_sign=True,
        invertible=True, removes_location=False, removes_scale=True,
        bounded_output=True, robust_to_outliers=False, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=False, complexity=0.1, runtime=0.05),
    PolicyKind.ROBUST_MAXABS: Capability(
        affine=True, monotone=True, preserves_zero=True, preserves_sign=True,
        invertible=True, removes_location=False, removes_scale=True,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=False, complexity=0.2, runtime=0.05),
    PolicyKind.CLIP_ROBUST: Capability(
        affine=False, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=False, removes_location=True, removes_scale=True,
        bounded_output=True, robust_to_outliers=True, keeps_extreme_signal=False,
        interpretable=True, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.3, runtime=0.1),
    PolicyKind.SIGNED_LOG: Capability(
        affine=False, monotone=True, preserves_zero=True, preserves_sign=True,
        invertible=True, removes_location=False, removes_scale=False,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=False, requires_positive=False, requires_nonnegative=False,
        densifies=False, complexity=0.35, runtime=0.15),
    PolicyKind.LOG1P: Capability(
        affine=False, monotone=True, preserves_zero=True, preserves_sign=False,
        invertible=True, removes_location=False, removes_scale=False,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=False, requires_positive=False, requires_nonnegative=True,
        densifies=False, complexity=0.3, runtime=0.1),
    PolicyKind.BOXCOX: Capability(
        affine=False, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=True, removes_location=False, removes_scale=False,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=False, requires_positive=True, requires_nonnegative=False,
        densifies=True, complexity=0.5, runtime=0.3),
    PolicyKind.YEOJOHNSON: Capability(
        affine=False, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=True, removes_location=False, removes_scale=False,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=True,
        interpretable=False, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.5, runtime=0.3),
    PolicyKind.QUANTILE_UNIFORM: Capability(
        affine=False, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=False, removes_location=True, removes_scale=True,
        bounded_output=True, robust_to_outliers=True, keeps_extreme_signal=False,
        interpretable=False, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.6, runtime=0.4),
    PolicyKind.QUANTILE_NORMAL: Capability(
        affine=False, monotone=True, preserves_zero=False, preserves_sign=False,
        invertible=False, removes_location=True, removes_scale=True,
        bounded_output=False, robust_to_outliers=True, keeps_extreme_signal=False,
        interpretable=False, requires_positive=False, requires_nonnegative=False,
        densifies=True, complexity=0.6, runtime=0.4),
}

PER_FEATURE_KINDS = list(CAPABILITIES.keys())


# --------------------------------------------------------------------------- #
# Fitting -- produce params for a chosen kind from the training column
# --------------------------------------------------------------------------- #
def fit_params(kind: PolicyKind, col: np.ndarray, card: RegimeCard) -> Dict[str, float]:
    x = np.asarray(col, dtype=float)
    x = x[np.isfinite(x)]
    eps = 1e-12
    if kind == PolicyKind.IDENTITY:
        return {}
    if kind == PolicyKind.AFFINE_STANDARD:
        return {"center": card.mean, "scale": card.std if card.std > eps else 1.0}
    if kind == PolicyKind.AFFINE_ROBUST:
        scale = card.iqr if card.iqr > eps else (card.mad if card.mad > eps else 1.0)
        return {"center": card.median, "scale": scale}
    if kind == PolicyKind.MINMAX:
        rng = card.max - card.min
        return {"min": card.min, "range": rng if rng > eps else 1.0}
    if kind == PolicyKind.MAXABS:
        return {"scale": card.abs_max if card.abs_max > eps else 1.0}
    if kind == PolicyKind.ROBUST_MAXABS:
        s = card.robust_abs_max if card.robust_abs_max > eps else (
            card.abs_max if card.abs_max > eps else 1.0)
        return {"scale": s}
    if kind == PolicyKind.CLIP_ROBUST:
        qlo, qhi = np.percentile(x, [1, 99])
        med = card.median
        scale = card.iqr if card.iqr > eps else 1.0
        return {"qlo": float(qlo), "qhi": float(qhi), "center": med, "scale": scale}
    if kind == PolicyKind.SIGNED_LOG:
        s = card.robust_abs_max if card.robust_abs_max > eps else 1.0
        return {"s": s}
    if kind == PolicyKind.LOG1P:
        return {}
    if kind == PolicyKind.BOXCOX:
        xp = x[x > 0]
        # boxcox requires >=3 non-constant positive values; fall back to identity-power
        # (lmbda=1) on degenerate input rather than raising.
        if xp.size < 3 or np.ptp(xp) < 1e-12:
            return {"lmbda": 1.0}
        _, lmbda = stats.boxcox(xp)
        return {"lmbda": float(lmbda)}
    if kind == PolicyKind.YEOJOHNSON:
        if x.size < 3 or np.ptp(x) < 1e-12:
            return {"lmbda": 1.0}
        _, lmbda = stats.yeojohnson(x)
        return {"lmbda": float(lmbda)}
    if kind in (PolicyKind.QUANTILE_UNIFORM, PolicyKind.QUANTILE_NORMAL):
        # store a grid of training quantiles for interpolation. Cap the number of knots
        # at min(256, n) -- 1001 knots is far more than the empirical CDF supports for
        # typical sample sizes and dominates fit time on wide tables.
        n_knots = int(min(256, max(16, x.size)))
        qs = np.linspace(0, 1, n_knots)
        knots = np.quantile(x, qs)
        return {"_quantile_knots": knots, "_quantile_levels": qs}  # arrays stored in params
    return {}


# --------------------------------------------------------------------------- #
# Apply / invert a fitted policy to a column
# --------------------------------------------------------------------------- #
def apply(policy: NormalizationPolicy, col: np.ndarray) -> np.ndarray:
    x = np.asarray(col, dtype=float)
    p = policy.params
    k = policy.kind
    eps = 1e-12
    if k == PolicyKind.IDENTITY:
        return x
    if k == PolicyKind.AFFINE_STANDARD or k == PolicyKind.AFFINE_ROBUST:
        return (x - p["center"]) / (p["scale"] + eps)
    if k == PolicyKind.MINMAX:
        return (x - p["min"]) / (p["range"] + eps)
    if k == PolicyKind.MAXABS or k == PolicyKind.ROBUST_MAXABS:
        return x / (p["scale"] + eps)
    if k == PolicyKind.CLIP_ROBUST:
        xc = np.clip(x, p["qlo"], p["qhi"])
        return (xc - p["center"]) / (p["scale"] + eps)
    if k == PolicyKind.SIGNED_LOG:
        return np.sign(x) * np.log1p(np.abs(x) / (p["s"] + eps))
    if k == PolicyKind.LOG1P:
        return np.log1p(np.clip(x, 0, None))
    if k == PolicyKind.BOXCOX:
        lm = p["lmbda"]
        xs = np.clip(x, eps, None)
        return stats.boxcox(xs, lmbda=lm)
    if k == PolicyKind.YEOJOHNSON:
        return stats.yeojohnson(x, lmbda=p["lmbda"])
    if k in (PolicyKind.QUANTILE_UNIFORM, PolicyKind.QUANTILE_NORMAL):
        knots = np.asarray(p["_quantile_knots"], dtype=float)
        levels = np.asarray(p["_quantile_levels"], dtype=float)
        u = np.interp(x, knots, levels, left=0.0, right=1.0)
        if k == PolicyKind.QUANTILE_UNIFORM:
            return u
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return special.ndtri(u)  # inverse normal CDF
    raise ValueError(f"unknown policy kind {k}")


def invert(policy: NormalizationPolicy, z: np.ndarray) -> np.ndarray:
    if not policy.invertible:
        raise ValueError(f"policy {policy.kind} is not invertible")
    x = np.asarray(z, dtype=float)
    p = policy.params
    k = policy.kind
    eps = 1e-12
    if k == PolicyKind.IDENTITY:
        return x
    if k in (PolicyKind.AFFINE_STANDARD, PolicyKind.AFFINE_ROBUST):
        return x * (p["scale"] + eps) + p["center"]
    if k == PolicyKind.MINMAX:
        return x * (p["range"] + eps) + p["min"]
    if k in (PolicyKind.MAXABS, PolicyKind.ROBUST_MAXABS):
        return x * (p["scale"] + eps)
    if k == PolicyKind.SIGNED_LOG:
        return np.sign(x) * (np.expm1(np.abs(x))) * (p["s"] + eps)
    if k == PolicyKind.LOG1P:
        return np.expm1(x)
    if k == PolicyKind.BOXCOX:
        return special.inv_boxcox(x, p["lmbda"])
    if k == PolicyKind.YEOJOHNSON:
        return _yeojohnson_inv(x, p["lmbda"])
    raise ValueError(f"inverse not implemented for {k}")


def _yeojohnson_inv(y: np.ndarray, lmbda: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y >= 0
    if abs(lmbda) < 1e-8:
        out[pos] = np.expm1(y[pos])
    else:
        out[pos] = np.power(y[pos] * lmbda + 1.0, 1.0 / lmbda) - 1.0
    if abs(lmbda - 2.0) < 1e-8:
        out[~pos] = -np.expm1(-y[~pos])
    else:
        out[~pos] = 1.0 - np.power(-(2.0 - lmbda) * y[~pos] + 1.0, 1.0 / (2.0 - lmbda))
    return out
