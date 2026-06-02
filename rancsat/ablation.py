"""Ablation baselines -- the comparators that decide whether the compiler earns its keep.

The single most important experiment for this paper is the *rule-based selector*: if a
hand-written ruleset over the same Policy DSL matches RANC-SAT, the constraint machinery
adds nothing. We provide:

    RuleBasedSelector      -- shallow if/else scaler picker (the "is it just a menu?" test)
    CVBruteForceSelector   -- AutoML-style: pick the scaler with best CV score (model fits!)

Both expose ``fit``/``transform`` and report ``n_model_fits`` so you can show RANC-SAT
reaches comparable quality with ZERO model fits during policy selection.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from .schemas import NormalizationPolicy, PolicyKind, RegimeCard
from .profiling import build_regime_cards
from . import policies as P


# --------------------------------------------------------------------------- #
# Rule-based selector (no solver, no contracts -- just heuristics)
# --------------------------------------------------------------------------- #
class RuleBasedSelector(BaseEstimator, TransformerMixin):
    """Conventional 'pick a scaler by rules of thumb' baseline."""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.n_model_fits = 0

    def _rule(self, card: RegimeCard) -> PolicyKind:
        if card.sign.value == "constant":
            return PolicyKind.IDENTITY
        if card.zero_mass > 0.2:
            return PolicyKind.MAXABS          # keep sparsity
        if card.tail.value == "heavy":
            return PolicyKind.AFFINE_ROBUST   # robust to outliers
        if card.sign.value == "positive" and card.skew > 1.0:
            return PolicyKind.LOG1P
        return PolicyKind.AFFINE_STANDARD

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.cards_ = build_regime_cards(X, self.feature_names)
        self.policies_ = []
        for card in self.cards_:
            kind = self._rule(card)
            self.policies_.append(NormalizationPolicy(
                feature=card.name, index=card.index, kind=kind,
                params=P.fit_params(kind, X[:, card.index], card),
                invertible=P.CAPABILITIES[kind].invertible,
                is_noop=(kind == PolicyKind.IDENTITY)))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z = np.empty_like(X)
        for pol in self.policies_:
            Z[:, pol.index] = P.apply(pol, X[:, pol.index])
        return Z


# --------------------------------------------------------------------------- #
# CV brute-force selector (AutoML-style search over the same DSL)
# --------------------------------------------------------------------------- #
_SEARCH_KINDS = [
    PolicyKind.IDENTITY, PolicyKind.AFFINE_STANDARD, PolicyKind.AFFINE_ROBUST,
    PolicyKind.MINMAX, PolicyKind.MAXABS, PolicyKind.QUANTILE_NORMAL,
]


class CVBruteForceSelector(BaseEstimator, TransformerMixin):
    """Select ONE global scaler by cross-validated downstream score (counts model fits)."""

    def __init__(self, task="classification", cv=3, feature_names=None):
        self.task = task
        self.cv = cv
        self.feature_names = feature_names
        self.n_model_fits = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).ravel()
        self.cards_ = build_regime_cards(X, self.feature_names)
        est = LogisticRegression(max_iter=500) if self.task == "classification" else Ridge()
        scoring = "accuracy" if self.task == "classification" else "neg_mean_squared_error"

        best_kind, best_score = PolicyKind.IDENTITY, -np.inf
        for kind in _SEARCH_KINDS:
            Z = self._apply_global(X, kind)
            if not np.all(np.isfinite(Z)):
                continue
            try:
                scores = cross_val_score(clone(est), Z, y, cv=self.cv, scoring=scoring)
            except Exception:
                continue
            self.n_model_fits += self.cv
            if scores.mean() > best_score:
                best_score, best_kind = scores.mean(), kind
        self.best_kind_ = best_kind
        self.policies_ = []
        for card in self.cards_:
            self.policies_.append(NormalizationPolicy(
                feature=card.name, index=card.index, kind=best_kind,
                params=P.fit_params(best_kind, X[:, card.index], card),
                invertible=P.CAPABILITIES[best_kind].invertible))
        return self

    def _apply_global(self, X, kind):
        cards = build_regime_cards(X, self.feature_names)
        Z = np.empty_like(X)
        for card in cards:
            pol = NormalizationPolicy(
                feature=card.name, index=card.index, kind=kind,
                params=P.fit_params(kind, X[:, card.index], card),
                invertible=P.CAPABILITIES[kind].invertible)
            Z[:, card.index] = P.apply(pol, X[:, card.index])
        return Z

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z = np.empty_like(X)
        for pol in self.policies_:
            Z[:, pol.index] = P.apply(pol, X[:, pol.index])
        return Z


def policy_agreement(a: List[NormalizationPolicy], b: List[NormalizationPolicy]) -> float:
    """Fraction of features where two selectors chose the same policy kind."""
    if not a:
        return 1.0
    return float(np.mean([pa.kind == pb.kind for pa, pb in zip(a, b)]))
