"""Signal Risk Ledger.

For every feature where the contract would remove some variation (outliers, location,
scale, sign, magnitude), record what *true signal* might be destroyed, how severe and
how confident the risk is, the mitigation, and the falsification test that would catch
it. The compiler uses the aggregated risk score R(pi) as the top soft-cost term, and
the falsification suite later executes the named tests.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from .schemas import InvarianceContract, RegimeCard, SignalRiskRow, SignPattern


def build_signal_risk_ledger(
    cards: List[RegimeCard],
    contracts: List[InvarianceContract],
    X_train: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> List[SignalRiskRow]:
    rows: List[SignalRiskRow] = []
    X = np.asarray(X_train, dtype=float) if X_train is not None else None
    y = np.asarray(y, dtype=float).ravel() if y is not None else None

    for card, contract in zip(cards, contracts):
        # ---- risk: damping/removing extremes may delete predictive signal --- #
        # Only a RISK when the extremes might actually be signal: either the user
        # declared preserve_extreme_signal, or extremes empirically correlate with y.
        # If the user explicitly set damp_outliers AND preserve_extreme_signal=False,
        # they have asserted extremes are nuisance -> no signal-risk row (don't fight
        # the contract).
        if card.outlier_frac > 0.0:
            corr = _extreme_target_corr(X, y, card.index)
            extremes_may_be_signal = (
                contract.preserve_extreme_signal
                or (corr is not None and corr > 0.1)
            )
            if extremes_may_be_signal:
                severity = float(np.clip(card.outlier_frac * 5, 0.0, 1.0))
                confidence = 0.8 if y is not None else 0.4
                if corr is not None and corr > 0.1:
                    severity = float(np.clip(severity + corr, 0.0, 1.0))
                evidence = (f"outlier_frac={card.outlier_frac:.3f}"
                            + (f", |corr(extremes,y)|={corr:.3f}" if corr is not None
                               else ""))
                rows.append(SignalRiskRow(
                    feature=card.name,
                    proposed_removal="damp / clip extreme values",
                    possible_true_signal="rare-event / heavy-tail predictive signal",
                    evidence=evidence, severity=severity, confidence=confidence,
                    mitigation="keep extremes (robust scale, no clip) when corr>thr",
                    falsification_test="outlier_as_signal",
                ))

        # ---- risk: centering destroys structural zeros / sparsity ----------- #
        if (contract.enforce_shift_invariance and card.zero_mass > 0.2
                and not contract.preserve_zero):
            rows.append(SignalRiskRow(
                feature=card.name,
                proposed_removal="subtract location (center)",
                possible_true_signal="structural zeros / sparsity pattern",
                evidence=f"zero_mass={card.zero_mass:.3f}",
                severity=float(np.clip(card.zero_mass, 0.0, 1.0)), confidence=0.9,
                mitigation="use zero-preserving scaling (maxabs / robust_maxabs)",
                falsification_test="zero_preservation",
            ))

        # ---- risk: rank/quantile transform destroys magnitude ratios -------- #
        if contract.preserve_distance_ratios is False and card.distance_sensitive:
            rows.append(SignalRiskRow(
                feature=card.name,
                proposed_removal="nonlinear monotone remap",
                possible_true_signal="distance / magnitude ratios used downstream",
                evidence="distance_sensitive feature",
                severity=0.5, confidence=0.5,
                mitigation="restrict to affine policies",
                falsification_test="distance_ratio",
            ))

        # ---- risk: sign-altering transform on signed feature ---------------- #
        if card.sign == SignPattern.SIGNED and not contract.preserve_sign:
            rows.append(SignalRiskRow(
                feature=card.name,
                proposed_removal="possible sign distortion",
                possible_true_signal="direction / signed semantics",
                evidence="signed feature, sign not contractually protected",
                severity=0.4, confidence=0.5,
                mitigation="set preserve_sign or use sign-preserving policy",
                falsification_test="sign_preservation",
            ))
    return rows


def _extreme_target_corr(X, y, j) -> Optional[float]:
    """|correlation| between 'is-this-row-an-extreme-in-feature-j' and the target."""
    if X is None or y is None:
        return None
    col = X[:, j].astype(float)
    finite = np.isfinite(col) & np.isfinite(y)
    if finite.sum() < 10:
        return None
    col, yy = col[finite], y[finite]
    med = np.median(col)
    mad = np.median(np.abs(col - med)) * 1.4826 + 1e-12
    is_ext = (np.abs(col - med) > 3 * mad).astype(float)
    if is_ext.std() == 0 or yy.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(is_ext, yy)[0, 1]))
