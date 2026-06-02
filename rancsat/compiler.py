"""The RANC-SAT compiler (pure-Python enumeration backend).

For each feature group we enumerate the finite Policy DSL, evaluate the HARD predicate
H_g(pi, contract) == 0 (legal domain + every contractual invariant), discard infeasible
policies, then rank the feasible set lexicographically by the SOFT cost vector

        (R, U, D, K, C, I)
         |  |  |  |  |  +-- interpretability penalty
         |  |  |  |  +----- complexity
         |  |  |  +-------- runtime
         |  |  +----------- drift sensitivity
         |  +-------------- estimation uncertainty
         +----------------- signal-removal risk   (top priority to minimize)

and emit the minimal-risk legal transform. If the feasible set is empty we return the
UNSAT CORE (the clauses that eliminated every candidate) and fall back to the
least-transforming safe policy -- almost always IDENTITY / no-op.

The search space is tiny (~13 candidates / feature), so exhaustive enumeration is exact
and fully transparent -- no black-box solver required. This is deliberate: it keeps the
"why was this policy chosen / rejected" question answerable for the audit certificate.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .schemas import (
    InvarianceContract, NormalizationPolicy, PolicyKind, RegimeCard,
    SignalRiskRow, SignPattern, TailCategory,
)
from .policies import (
    CAPABILITIES, PER_FEATURE_KINDS, Capability, fit_params,
    apply as _policy_apply,
)


# --------------------------------------------------------------------------- #
# HARD predicates: return list of violated clause names (empty == feasible)
# --------------------------------------------------------------------------- #
def hard_violations(kind: PolicyKind, card: RegimeCard,
                    contract: InvarianceContract) -> List[str]:
    cap: Capability = CAPABILITIES[kind]
    v: List[str] = []

    # --- legal domain --------------------------------------------------------
    if cap.requires_positive and card.min <= 0:
        v.append("legal_domain:requires_positive")
    if cap.requires_nonnegative and card.min < 0:
        v.append("legal_domain:requires_nonnegative")

    # --- contractual invariants ---------------------------------------------
    if contract.preserve_zero and (cap.densifies or not cap.preserves_zero):
        v.append("preserve_zero")
    if contract.preserve_sign and not cap.preserves_sign:
        v.append("preserve_sign")
    if contract.preserve_rank and not cap.monotone:
        v.append("preserve_rank")
    if contract.preserve_monotonicity and not cap.monotone:
        v.append("preserve_monotonicity")
    if contract.preserve_distance_ratios and not cap.affine:
        v.append("preserve_distance_ratios")
    if contract.allow_inverse_transform and not cap.invertible:
        v.append("allow_inverse_transform")
    if contract.preserve_extreme_signal and not cap.keeps_extreme_signal:
        v.append("preserve_extreme_signal")
    if contract.preserve_interpretability and not cap.interpretable:
        v.append("preserve_interpretability")

    # --- positive obligations: the transform must actually do its job --------
    # if scale-invariance is required, IDENTITY (and pure-location maps) is illegal
    if contract.enforce_scale_invariance and not cap.removes_scale:
        v.append("enforce_scale_invariance")
    if contract.enforce_shift_invariance and not cap.removes_location:
        v.append("enforce_shift_invariance")
    if contract.damp_outliers and not cap.robust_to_outliers:
        v.append("damp_outliers")

    # --- deployment-safety: a transform that PROMISES a fixed output bound (minmax,
    # quantile_uniform) cannot keep that promise on unseen data UNLESS the input bounds
    # are semantic/certified. Otherwise future values fall outside the fitted range and
    # silently breach the bound. Hard-reject bounded-range maps on non-semantic,
    # drift-prone features so RANC-SAT's bound guarantee is actually sound.
    if kind in (PolicyKind.MINMAX, PolicyKind.QUANTILE_UNIFORM):
        if not card.bounded_semantic:
            v.append("unsafe_bound:non_semantic_range")

    return v


# --------------------------------------------------------------------------- #
# Fit-for-purpose deficiency: how much distributional pathology REMAINS after the
# transform. This is the term that fixes systematic under-transformation: a policy
# that leaves a lognormal feature wildly skewed is "legal but inadequate".
#
# Activates ONLY for pathological features (high skew, heavy tail, or huge dynamic
# range); for well-behaved features every policy scores F=0, so the minimal-transform
# principle is preserved. Measured empirically on the TRAIN column (no leakage).
# --------------------------------------------------------------------------- #
def _residual_pathology(kind: PolicyKind, card: RegimeCard,
                        col: Optional[np.ndarray]) -> float:
    # is the RAW feature pathological enough to demand correction?
    raw_skew = abs(card.skew)
    heavy = card.tail == TailCategory.HEAVY
    dyn_range = (abs(card.max) + 1e-12) / (abs(card.median) + 1e-9)
    pathological = (raw_skew > 1.0) or heavy or (card.sign != SignPattern.SIGNED
                                                 and dyn_range > 50)
    if not pathological or col is None:
        return 0.0

    x = np.asarray(col, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0
    try:
        params = fit_params(kind, x, card)
        probe = NormalizationPolicy(feature=card.name, index=card.index, kind=kind,
                                    params=params,
                                    invertible=CAPABILITIES[kind].invertible)
        z = _policy_apply(probe, x)
        z = z[np.isfinite(z)]
        if z.size < 8:
            return 1.0
    except Exception:
        return 1.0

    # residual skew + excess-kurtosis magnitude of the TRANSFORMED feature
    m, s = z.mean(), z.std()
    if s < 1e-12:
        return 1.0
    zr = (z - m) / s
    res_skew = abs(float(np.mean(zr ** 3)))
    res_kurt = abs(float(np.mean(zr ** 4) - 3.0))
    residual = 0.6 * res_skew + 0.4 * min(res_kurt / 3.0, 2.0)
    # bucket: below an acceptable threshold counts as "adequate" (F contribution 0),
    # so among adequate policies the cheaper/robust tiebreakers (U,D,C) decide.
    return 0.0 if residual < 0.4 else round(float(residual), 6)


# --------------------------------------------------------------------------- #
# SOFT costs: lower is better, evaluated only on feasible policies
# --------------------------------------------------------------------------- #
def soft_costs(kind: PolicyKind, card: RegimeCard, contract: InvarianceContract,
               ledger: List[SignalRiskRow],
               col: Optional[np.ndarray] = None) -> Tuple[float, ...]:
    cap = CAPABILITIES[kind]

    # R: signal-removal risk -- aggregate ledger severity*confidence for THIS feature,
    #    weighted by whether this policy actually performs the risky removal.
    R = 0.0
    for row in ledger:
        if row.feature != card.name:
            continue
        weight = row.severity * row.confidence
        if row.falsification_test == "outlier_as_signal":
            # Only policies that actually DROP extreme signal (clipping / quantile
            # saturation) incur this risk. A robust *scale* estimator keeps the
            # extremes (they are merely rescaled), so it is NOT penalized here --
            # conflating "robust scale" with "damps signal" is a category error.
            #
            # EXCEPTION: if the contract explicitly prefers distribution matching for a
            # distance-based model, the user has decided commensurability across
            # features matters more than preserving a single feature's extreme tail.
            # Don't let this mild risk veto rank/quantile equalization.
            if not cap.keeps_extreme_signal and not contract.prefer_distribution_match:
                R += weight * 1.0
        elif row.falsification_test == "zero_preservation":
            if cap.densifies or not cap.preserves_zero:
                R += weight * 1.0
        elif row.falsification_test == "distance_ratio":
            if not cap.affine:
                R += weight * 1.0
        elif row.falsification_test == "sign_preservation":
            if not cap.preserves_sign:
                R += weight * 1.0

    # nuisance-retention penalty: if the contract declares extremes are NUISANCE to be
    # damped (damp_outliers and NOT preserve_extreme_signal), then a policy that merely
    # rescales but leaves the extremes unbounded fails the contract's intent -- the
    # nuisance survives into the output. Penalize un-bounded / extreme-keeping policies
    # so a genuinely damping transform (e.g. CLIP_ROBUST) is preferred.
    if contract.damp_outliers and not contract.preserve_extreme_signal:
        if not cap.bounded_output:
            R += 0.5
        if cap.keeps_extreme_signal:
            R += 0.5

    # U: estimation uncertainty -- standard (moment) scaling is fragile when scale is
    #    unstable; quantile/power transforms overfit small / unstable samples.
    inst = 1.0 - card.scale_stability
    if kind == PolicyKind.AFFINE_STANDARD:
        U = inst * 1.0
    elif kind in (PolicyKind.QUANTILE_UNIFORM, PolicyKind.QUANTILE_NORMAL,
                  PolicyKind.BOXCOX, PolicyKind.YEOJOHNSON):
        U = inst * 0.8 + 0.2
    elif kind in (PolicyKind.AFFINE_ROBUST, PolicyKind.ROBUST_MAXABS,
                  PolicyKind.CLIP_ROBUST):
        U = inst * 0.4
    else:
        U = inst * 0.3

    # spike-domination penalty: MAXABS / MINMAX divide by the single most extreme
    # value. If that value is a spike (abs_max >> robust q99), the divisor collapses
    # the rest of the distribution and destroys relative magnitude / signal. Penalize
    # these non-robust extreme-based scalers when the max is spike-dominated.
    spike_ratio = card.abs_max / (card.robust_abs_max + 1e-12)
    if kind in (PolicyKind.MAXABS, PolicyKind.MINMAX) and spike_ratio > 3.0:
        U += float(min((spike_ratio - 3.0) / 7.0, 1.0))

    # D: drift sensitivity -- min/max & fixed-quantile maps extrapolate badly under drift
    if kind in (PolicyKind.MINMAX, PolicyKind.QUANTILE_UNIFORM, PolicyKind.QUANTILE_NORMAL):
        D = card.drift * 1.0
    elif kind == PolicyKind.AFFINE_STANDARD:
        D = card.drift * 0.6
    else:
        D = card.drift * 0.3

    # F: fit-for-purpose deficiency -- residual pathology after the transform. Sits
    # just below signal-risk R in priority: among low-risk policies, prefer the one
    # that actually NORMALIZES a pathological feature. Zero for well-behaved features.
    F = _residual_pathology(kind, card, col)

    # distribution-match preference (distance models): reward policies that map every
    # feature onto a COMMON, commensurate distribution so no feature dominates the
    # distance metric. Rank/quantile maps equalize distributions exactly; bounded
    # affine maps (min-max-like) partially; raw/unbounded affine does not. Implemented
    # as an F-tier penalty for NON-matching policies so it only breaks ties among
    # otherwise-acceptable transforms, and only when the contract asks for it.
    if contract.prefer_distribution_match:
        if kind in (PolicyKind.QUANTILE_NORMAL, PolicyKind.QUANTILE_UNIFORM):
            F += 0.0                       # ideal: identical distribution per feature
        elif kind in (PolicyKind.MINMAX, PolicyKind.MAXABS, PolicyKind.ROBUST_MAXABS):
            F += 0.3                       # bounded range, but shape not equalized
        elif kind in (PolicyKind.AFFINE_STANDARD, PolicyKind.AFFINE_ROBUST):
            F += 0.5                       # commensurate scale, but tails/shape remain
        else:
            F += 0.8                       # identity / skewed maps leave features incommensurate

    # K: runtime ; C: complexity ; I: interpretability penalty
    K = cap.runtime
    C = cap.complexity
    I = (0.0 if cap.interpretable else 0.3)
    if contract.preserve_interpretability and not cap.interpretable:
        I += 0.5

    return (round(R, 6), round(F, 6), round(U, 6), round(D, 6),
            round(K, 6), round(C, 6), round(I, 6))


# --------------------------------------------------------------------------- #
# Per-feature compile
# --------------------------------------------------------------------------- #
def compile_policy(card: RegimeCard, contract: InvarianceContract,
                   ledger: List[SignalRiskRow],
                   col: Optional[np.ndarray] = None) -> NormalizationPolicy:
    feasible: List[Tuple[Tuple[float, ...], PolicyKind]] = []
    rejected: List[Tuple[str, str]] = []
    all_violations: List[str] = []

    for kind in PER_FEATURE_KINDS:
        viol = hard_violations(kind, card, contract)
        if viol:
            rejected.append((kind.value, ",".join(viol)))
            all_violations.extend(viol)
            continue
        feasible.append((soft_costs(kind, card, contract, ledger, col=col), kind))

    if not feasible:
        # UNSAT: report the core (clauses that killed every candidate) and fall back.
        core = sorted(set(all_violations))
        policy = NormalizationPolicy(
            feature=card.name, index=card.index, kind=PolicyKind.IDENTITY,
            params={}, invertible=True, is_noop=True,
            cost_vector=(), rejected=rejected, unsat_core=core,
            notes="UNSAT under contract; fell back to identity (least-transforming).",
            drift_threshold=_drift_threshold(card),
        )
        return policy

    feasible.sort(key=lambda t: t[0])
    best_cost, best_kind = feasible[0]
    cap = CAPABILITIES[best_kind]
    params = fit_params(best_kind, col if col is not None else np.zeros(1), card)

    return NormalizationPolicy(
        feature=card.name, index=card.index, kind=best_kind, params=params,
        invertible=cap.invertible, is_noop=(best_kind == PolicyKind.IDENTITY),
        cost_vector=best_cost, rejected=rejected, unsat_core=[],
        notes=f"selected {best_kind.value} with cost(R,F,U,D,K,C,I)={best_cost}",
        drift_threshold=_drift_threshold(card),
    )


def _drift_threshold(card: RegimeCard) -> float:
    """A simple deployment drift alarm threshold in standardized units."""
    return float(max(0.25, 3.0 * (1.0 - card.scale_stability)))


# --------------------------------------------------------------------------- #
# Orchestrator object
# --------------------------------------------------------------------------- #
class RANCCompiler:
    """Compile a list of per-feature policies from cards + contracts + ledger."""

    def __init__(self, X_train: np.ndarray):
        self.X_train = np.asarray(X_train, dtype=float)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

    def compile(self, cards: List[RegimeCard], contracts: List[InvarianceContract],
                ledger: List[SignalRiskRow]) -> List[NormalizationPolicy]:
        policies = []
        for card, contract in zip(cards, contracts):
            col = self.X_train[:, card.index]
            policies.append(compile_policy(card, contract, ledger, col=col))
        return policies
