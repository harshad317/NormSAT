"""Leakage-safe profiling: turn X_train into a list of RegimeCards.

CRITICAL INVARIANT: every statistic here is computed from the *training fold only*.
The compiler and the falsification suite consume RegimeCards; they never touch raw
test data. This is what lets RANC-SAT treat "no leakage" as a hard contract.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from .schemas import RegimeCard, SignPattern, TailCategory


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else default


def _classify_sign(col: np.ndarray, zero_mass: float) -> SignPattern:
    finite = col[np.isfinite(col)]
    if finite.size == 0 or np.allclose(np.nanstd(finite), 0.0):
        return SignPattern.CONSTANT
    has_neg = np.any(finite < 0)
    has_pos = np.any(finite > 0)
    if has_neg and has_pos:
        return SignPattern.SIGNED
    if not has_neg and zero_mass > 0:
        return SignPattern.NONNEGATIVE
    if not has_neg and not (finite == 0).any():
        return SignPattern.POSITIVE
    return SignPattern.NONNEGATIVE


def _classify_tail(excess_kurtosis: float) -> TailCategory:
    if excess_kurtosis > 3.0:
        return TailCategory.HEAVY
    if excess_kurtosis < -0.5:
        return TailCategory.LIGHT
    return TailCategory.NORMAL


def _skewness(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _drift_estimate(x: np.ndarray) -> float:
    """Cheap train-internal drift proxy: standardized mean shift between the first
    and second half of the (assumed time-ordered) training rows, squashed to 0..1."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 0.0
    half = x.size // 2
    a, b = x[:half], x[half:]
    pooled = np.std(x) + 1e-9
    shift = abs(a.mean() - b.mean()) / pooled
    return float(np.clip(shift, 0.0, 1.0))


def build_regime_cards(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    *,
    bounded_features: Optional[Sequence[int]] = None,
    distance_sensitive: bool = False,
    interpretability_required: bool = False,
    inverse_required: bool = False,
    covariance: Optional[np.ndarray] = None,
) -> List[RegimeCard]:
    """Profile each column of ``X`` (train fold) into a RegimeCard.

    Parameters
    ----------
    X : (n_samples, n_features) array. NaNs allowed (treated as missing).
    feature_names : optional column names.
    bounded_features : indices whose finite bounds are *semantic* (e.g. percentages).
    covariance : optional precomputed covariance for covariance-reliability scoring.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(d)]
    bounded = set(bounded_features or [])

    cards: List[RegimeCard] = []
    for j in range(d):
        col_full = X[:, j]
        missing_frac = float(np.mean(~np.isfinite(col_full)))
        col = col_full[np.isfinite(col_full)]
        if col.size == 0:
            col = np.zeros(1)

        mean = float(np.mean(col))
        std = float(np.std(col))
        median = float(np.median(col))
        q1, q3 = np.percentile(col, [25, 75])
        iqr = float(q3 - q1)
        mad = float(np.median(np.abs(col - median))) * 1.4826  # ~std under normality
        cmin, cmax = float(np.min(col)), float(np.max(col))
        abs_max = float(np.max(np.abs(col)))
        robust_abs_max = float(np.percentile(np.abs(col), 99))

        zero_mass = float(np.mean(col_full == 0.0))
        skew = _skewness(col)
        kurt = _excess_kurtosis(col)
        tail = _classify_tail(kurt)
        sign = _classify_sign(col, zero_mass)

        # outliers beyond median +/- 3 robust-sigma
        rsig = mad if mad > 0 else (iqr / 1.349 if iqr > 0 else std)
        if rsig > 0:
            outlier_frac = float(np.mean(np.abs(col - median) > 3 * rsig))
        else:
            outlier_frac = 0.0

        # scale stability: penalize small n and heavy tails (std is fragile then)
        n_factor = np.clip(col.size / 200.0, 0.0, 1.0)
        tail_factor = {TailCategory.LIGHT: 1.0, TailCategory.NORMAL: 0.9,
                       TailCategory.HEAVY: 0.4}[tail]
        scale_stability = float(np.clip(n_factor * tail_factor * (1 - outlier_frac), 0.0, 1.0))

        drift = _drift_estimate(col)

        # covariance reliability: need many more rows than features, and a usable cov
        if covariance is not None:
            cov_rel = float(np.clip((n - d) / max(n, 1), 0.0, 1.0))
        else:
            cov_rel = float(np.clip((n - d) / max(n, 1), 0.0, 1.0)) * 0.5

        cards.append(
            RegimeCard(
                name=names[j], index=j, n=int(col.size),
                mean=mean, std=std, median=median, iqr=iqr, mad=mad,
                min=cmin, max=cmax, abs_max=abs_max, robust_abs_max=robust_abs_max,
                skew=skew, excess_kurtosis=kurt, tail=tail, sign=sign,
                zero_mass=zero_mass, sparsity=zero_mass, missing_frac=missing_frac,
                outlier_frac=outlier_frac, bounded_semantic=(j in bounded),
                scale_stability=scale_stability, covariance_reliability=cov_rel,
                drift=drift,
                distance_sensitive=distance_sensitive,
                interpretability_required=interpretability_required,
                inverse_required=inverse_required,
            )
        )
    return cards
