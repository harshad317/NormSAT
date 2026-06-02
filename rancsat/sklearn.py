"""RANCSATTransformer: scikit-learn compatible estimator.

Pipeline implemented in ``fit``:

    validate split / no test stats
        -> build Regime Cards         (train-only)
        -> derive Invariance Contracts
        -> build Signal Risk Ledger
        -> compile one policy per feature (enumeration solver)
        -> run falsification suite
        -> repair / fallback on any failed feature
        -> (optional) compile group policies: whitening / vector-norm
        -> freeze + emit audit certificate

``transform`` only *applies* frozen policies and records drift; it never refits.
Use inside ``sklearn.Pipeline`` / ``ColumnTransformer``; cross-validation clones the
whole compiler per fold, so there is no leakage across folds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import hashlib
import copy
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
try:
    from sklearn.utils.validation import check_is_fitted
except Exception:  # pragma: no cover
    def check_is_fitted(est, attrs=None):
        if not hasattr(est, "policies_"):
            raise RuntimeError("not fitted")

from .schemas import (
    AuditReport, InvarianceContract, NormalizationPolicy, PolicyKind, SignPattern,
)
from .profiling import build_regime_cards
from .contracts import derive_invariance_contracts
from .ledger import build_signal_risk_ledger
from .compiler import RANCCompiler, compile_policy, _drift_threshold
from .falsification import run_falsification_tests, falsify_policy
from . import policies as P


# --- ordered "safer" fallbacks: try these in order if a policy is falsified ------
_FALLBACK_ORDER = [
    PolicyKind.AFFINE_ROBUST,
    PolicyKind.ROBUST_MAXABS,
    PolicyKind.MAXABS,
    PolicyKind.IDENTITY,
]


class RANCSATTransformer(BaseEstimator, TransformerMixin):
    """Constraint-satisfying normalization compiler.

    Parameters
    ----------
    contract : InvarianceContract, optional
        Global invariance contract. Per-feature contracts are specialized from it.
    model_family : {'linear','svm','knn','neural','tree','naive_bayes', None}
        Downstream model prior (e.g. trees -> prefer no-op).
    feature_names : sequence of str, optional
    bounded_features : sequence of int, optional
        Indices whose finite bounds are semantic (enables safe min-max).
    overrides : dict[int, InvarianceContract], optional
        Hard per-feature contract overrides.
    enable_whitening : bool
        Allow a group ZCA whitening policy when reduce_covariance is contractual and
        covariance is reliable.
    distance_sensitive, interpretability_required, inverse_required : bool
        Forwarded into every Regime Card / contract inference.
    """

    def __init__(self, contract: Optional[InvarianceContract] = None,
                 model_family: Optional[str] = None,
                 feature_names: Optional[Sequence[str]] = None,
                 bounded_features: Optional[Sequence[int]] = None,
                 overrides: Optional[Dict[int, InvarianceContract]] = None,
                 enable_whitening: bool = False,
                 distance_sensitive: bool = False,
                 interpretability_required: bool = False,
                 inverse_required: bool = False,
                 split_type: str = "iid"):
        self.contract = contract
        self.model_family = model_family
        self.feature_names = feature_names
        self.bounded_features = bounded_features
        self.overrides = overrides
        self.enable_whitening = enable_whitening
        self.distance_sensitive = distance_sensitive
        self.interpretability_required = interpretability_required
        self.inverse_required = inverse_required
        self.split_type = split_type

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = self._validate(X)
        n, d = X.shape
        contract = self.contract or InvarianceContract(
            allow_inverse_transform=self.inverse_required)

        # 1. profile (train-only)
        cov = np.cov(np.nan_to_num(X), rowvar=False) if d > 1 else None
        self.cards_ = build_regime_cards(
            X, self.feature_names,
            bounded_features=self.bounded_features,
            distance_sensitive=self.distance_sensitive,
            interpretability_required=self.interpretability_required,
            inverse_required=self.inverse_required,
            covariance=cov,
        )
        # 2. contracts
        self.contracts_ = derive_invariance_contracts(
            self.cards_, contract, self.model_family, self.overrides)
        # 3. ledger
        self.ledger_ = build_signal_risk_ledger(
            self.cards_, self.contracts_, X_train=X,
            y=np.asarray(y) if y is not None else None)
        # 4. compile
        compiler = RANCCompiler(X)
        self.policies_ = compiler.compile(self.cards_, self.contracts_, self.ledger_)
        # 5. falsify + 6. repair/fallback
        self.falsification_ = run_falsification_tests(
            self.policies_, self.cards_, self.contracts_, X)
        self._repair(X)
        # 7. optional group whitening
        self.whitening_ = None
        if self.enable_whitening and d > 1 and contract.reduce_covariance:
            self._fit_whitening(X)

        self.n_features_in_ = d
        self.labels_used_ = y is not None
        self.audit_ = self._build_audit(X)
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X):
        check_is_fitted(self, "policies_")
        X = self._validate(X, fitting=False)
        Z = np.empty_like(X, dtype=float)
        self.last_drift_ = {}
        for pol, card in zip(self.policies_, self.cards_):
            col = X[:, pol.index]
            Z[:, pol.index] = P.apply(pol, col)
            self.last_drift_[pol.feature] = self._drift_alarm(card, col)
        if self.whitening_ is not None:
            Z = self._apply_whitening(Z)
        return Z

    def inverse_transform(self, Z):
        check_is_fitted(self, "policies_")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if self.whitening_ is not None:
            Z = self._invert_whitening(Z)
        X = np.empty_like(Z, dtype=float)
        for pol in self.policies_:
            if not pol.invertible:
                raise ValueError(
                    f"feature '{pol.feature}' uses non-invertible policy "
                    f"{pol.kind.value}; set allow_inverse_transform=True to forbid it.")
            X[:, pol.index] = P.invert(pol, Z[:, pol.index])
        return X

    # ------------------------------------------------------------- accessors
    def get_policy(self, feature) -> NormalizationPolicy:
        for pol in self.policies_:
            if pol.feature == feature or pol.index == feature:
                return pol
        raise KeyError(feature)

    def audit_report(self) -> AuditReport:
        check_is_fitted(self, "policies_")
        return self.audit_

    def get_feature_names_out(self, input_features=None):
        return np.asarray([c.name for c in self.cards_], dtype=object)

    # ------------------------------------------------------------- internals
    def _validate(self, X, fitting=True):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not fitting and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X

    def _repair(self, X):
        """Replace any falsified policy with the first fallback that passes."""
        for i, (pol, card, con, fres) in enumerate(
                zip(self.policies_, self.cards_, self.contracts_, self.falsification_)):
            if fres.passed:
                continue
            col = X[:, pol.index]
            repaired = None
            for kind in _FALLBACK_ORDER:
                # respect hard feasibility of the fallback too
                from .compiler import hard_violations
                if hard_violations(kind, card, con):
                    continue
                cand = NormalizationPolicy(
                    feature=card.name, index=card.index, kind=kind,
                    params=P.fit_params(kind, col, card),
                    invertible=P.CAPABILITIES[kind].invertible,
                    is_noop=(kind == PolicyKind.IDENTITY),
                    notes=f"repair: fell back from {pol.kind.value} after falsification",
                    drift_threshold=_drift_threshold(card),
                )
                if falsify_policy(cand, card, con, col).passed:
                    repaired = cand
                    break
            if repaired is None:
                repaired = NormalizationPolicy(
                    feature=card.name, index=card.index, kind=PolicyKind.IDENTITY,
                    params={}, invertible=True, is_noop=True,
                    notes="repair: no falsifiable-safe transform; identity fallback",
                    unsat_core=fres.failures,
                    drift_threshold=_drift_threshold(card))
            self.policies_[i] = repaired
            self.falsification_[i] = falsify_policy(repaired, card, con, col)

    # --- group whitening (ZCA) --------------------------------------------------
    def _fit_whitening(self, X):
        Xs = np.nan_to_num(X)
        mu = Xs.mean(axis=0)
        Xc = Xs - mu
        cov = np.cov(Xc, rowvar=False)
        U, S, _ = np.linalg.svd(cov + 1e-6 * np.eye(cov.shape[0]))
        W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
        Winv = U @ np.diag(np.sqrt(S)) @ U.T
        self.whitening_ = {"mu": mu, "W": W, "Winv": Winv}

    def _apply_whitening(self, Z):
        w = self.whitening_
        return (np.nan_to_num(Z) - w["mu"]) @ w["W"].T

    def _invert_whitening(self, Z):
        w = self.whitening_
        return Z @ w["Winv"].T + w["mu"]

    def _drift_alarm(self, card, col):
        col = col[np.isfinite(col)]
        if col.size == 0:
            return 0.0
        scale = card.std + 1e-9
        shift = abs(np.mean(col) - card.mean) / scale
        return float(shift > card_drift_thr(card))

    def _build_audit(self, X) -> AuditReport:
        h = hashlib.sha256(np.ascontiguousarray(np.nan_to_num(X)).tobytes()).hexdigest()[:16]
        noop = np.mean([p.is_noop for p in self.policies_]) if self.policies_ else 0.0
        rej: Dict[str, int] = {}
        for p in self.policies_:
            for _, reason in p.rejected:
                for clause in reason.split(","):
                    rej[clause] = rej.get(clause, 0) + 1
        return AuditReport(
            data_hash=h, n_features=X.shape[1], n_samples=X.shape[0],
            split_type=self.split_type, labels_used=self.labels_used_,
            contract=(self.contract or InvarianceContract()).to_dict(),
            cards=[c.to_dict() for c in self.cards_],
            policies=[p.to_dict() for p in self.policies_],
            ledger=[r.to_dict() for r in self.ledger_],
            falsification=[f.to_dict() for f in self.falsification_],
            noop_fraction=float(noop), rejected_summary=rej,
        )


def card_drift_thr(card) -> float:
    return _drift_threshold(card)
