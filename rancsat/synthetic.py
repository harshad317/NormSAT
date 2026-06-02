"""Synthetic regimes with KNOWN-CORRECT policies.

Each generator returns ``(X, y, oracle_contract, oracle_kinds)`` where ``oracle_kinds``
is the per-feature policy a correct compiler *should* pick given the oracle contract.
These are the ground-truth tests that prove the compiler selects the right transform
for the right reason -- the core scientific claim -- before touching real datasets.

The headline regime is ``make_signal_vs_noise_outliers``: two features whose extremes
are respectively predictive signal and pure noise, which no single classical scaler
handles correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .schemas import InvarianceContract, PolicyKind


@dataclass
class SyntheticRegime:
    name: str
    X: np.ndarray
    y: np.ndarray
    contract: InvarianceContract
    oracle_kinds: List[PolicyKind]
    feature_names: List[str]
    task: str  # 'regression' | 'classification'


def make_sparse_semantic_zeros(n=2000, d=6, seed=0) -> SyntheticRegime:
    """Count-like sparse features; zeros are structural. Correct: keep zeros."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.4, size=(n, d)).astype(float)
    # inflate sparsity
    mask = rng.random((n, d)) < 0.6
    X[mask] = 0.0
    w = rng.normal(size=d)
    y = (X @ w + rng.normal(scale=0.5, size=n) > 0).astype(float)
    contract = InvarianceContract(preserve_zero=True, enforce_scale_invariance=True)
    oracle = [PolicyKind.MAXABS] * d
    return SyntheticRegime("sparse_semantic_zeros", X, y, contract, oracle,
                           [f"count{i}" for i in range(d)], "classification")


def make_signal_vs_noise_outliers(n=4000, seed=0) -> SyntheticRegime:
    """THE decisive regime.

    f_signal : heavy-tailed; its extremes ARE the predictive signal -> must NOT damp.
    f_noise  : heavy-tailed; its extremes are corrupted measurements -> SHOULD damp.
    A single global scaler cannot do both; RANC-SAT can, via per-feature contracts.
    """
    rng = np.random.default_rng(seed)
    # signal feature: student-t heavy tail, target depends on its magnitude
    f_signal = rng.standard_t(df=2.5, size=n)
    # noise feature: heavy tail too, but injected spikes uncorrelated with y
    f_noise = rng.standard_t(df=2.5, size=n)
    spikes = rng.random(n) < 0.05
    f_noise[spikes] += rng.normal(0, 50, size=spikes.sum())
    # two clean covariates
    f_a = rng.normal(size=n)
    f_b = rng.normal(size=n)

    logit = 1.3 * f_signal + 0.8 * f_a - 0.6 * f_b  # noise feature deliberately absent
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(float)

    X = np.column_stack([f_signal, f_noise, f_a, f_b])
    # oracle: protect signal extremes; damp noise extremes; standardize clean ones
    contract = InvarianceContract(enforce_scale_invariance=True)
    oracle = [PolicyKind.ROBUST_MAXABS,   # keeps extreme signal, robust scale
              PolicyKind.CLIP_ROBUST,      # damps noise extremes
              PolicyKind.AFFINE_STANDARD,
              PolicyKind.AFFINE_STANDARD]
    names = ["f_signal", "f_noise", "f_a", "f_b"]
    return SyntheticRegime("signal_vs_noise_outliers", X, y, contract, oracle, names,
                           "classification")


def per_feature_contracts_for_outliers() -> dict:
    """Explicit per-feature overrides matching make_signal_vs_noise_outliers oracle."""
    return {
        0: InvarianceContract(enforce_scale_invariance=True,
                              preserve_extreme_signal=True, damp_outliers=False),
        1: InvarianceContract(enforce_scale_invariance=True,
                              damp_outliers=True, preserve_extreme_signal=False),
        2: InvarianceContract(enforce_scale_invariance=True),
        3: InvarianceContract(enforce_scale_invariance=True),
    }


def make_positive_skew(n=2000, d=4, seed=0) -> SyntheticRegime:
    """Positive, right-skewed (multiplicative) features. Correct: monotone compression."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, d))
    X = np.exp(latent)  # lognormal, strictly positive, heavy right tail
    w = rng.normal(size=d)
    y = latent @ w + rng.normal(scale=0.3, size=n)
    contract = InvarianceContract(preserve_monotonicity=True,
                                  allow_inverse_transform=True)
    oracle = [PolicyKind.LOG1P] * d
    return SyntheticRegime("positive_skew", X, y, contract, oracle,
                           [f"pos{i}" for i in range(d)], "regression")


def make_drift(n=3000, d=4, seed=0) -> SyntheticRegime:
    """Time-ordered features with mean/scale drift. Correct: drift-robust, not min-max."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = np.empty((n, d))
    for j in range(d):
        X[:, j] = rng.normal(size=n) + 4 * t * (j + 1)  # growing mean
    w = rng.normal(size=d)
    y = X @ w + rng.normal(scale=0.5, size=n)
    contract = InvarianceContract(enforce_scale_invariance=True,
                                  enforce_shift_invariance=True)
    oracle = [PolicyKind.AFFINE_ROBUST] * d  # robust beats minmax/standard under drift
    return SyntheticRegime("drift", X, y, contract, oracle,
                           [f"drift{i}" for i in range(d)], "regression")


def make_constant_and_mixed(n=1500, seed=0) -> SyntheticRegime:
    """Mixed bag incl. a constant column (must be no-op) and a clean gaussian."""
    rng = np.random.default_rng(seed)
    const = np.full(n, 3.0)
    clean = rng.normal(size=n)
    sparse = rng.poisson(0.3, size=n).astype(float)
    X = np.column_stack([const, clean, sparse])
    y = clean + 0.5 * sparse + rng.normal(scale=0.3, size=n)
    contract = InvarianceContract(enforce_scale_invariance=True)
    oracle = [PolicyKind.IDENTITY, PolicyKind.AFFINE_STANDARD, PolicyKind.MAXABS]
    return SyntheticRegime("constant_and_mixed", X, y, contract, oracle,
                           ["const", "clean", "sparse"], "regression")


ALL_REGIMES = [
    make_sparse_semantic_zeros,
    make_signal_vs_noise_outliers,
    make_positive_skew,
    make_drift,
    make_constant_and_mixed,
]
