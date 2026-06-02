"""Derive a per-feature InvarianceContract.

A user can pass one global contract; this module specializes/augments it per feature
using the RegimeCard and model-family priors, and -- crucially -- biases toward
*conservative* defaults (preserve structure, prefer no-op) when the data semantics
are uncertain. This implements the dossier's "contract elicitation can encode wrong
assumptions -> default to identity" risk mitigation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional
import copy

from .schemas import InvarianceContract, RegimeCard, SignPattern


# model-family priors: what normalization the downstream model actually needs
MODEL_FAMILY_PRIORS: Dict[str, Dict[str, bool]] = {
    "linear":        {"enforce_scale_invariance": True, "enforce_shift_invariance": True},
    "svm":           {"enforce_scale_invariance": True, "enforce_shift_invariance": True},
    # Distance models (KNN, k-means) need COMMENSURATE feature distributions, not
    # preserved per-feature distance ratios. preserve_distance_ratios=True was wrong:
    # it forces affine-only and blocks the rank/quantile equalization these models
    # benefit from. Instead set the soft prefer_distribution_match flag.
    "knn":           {"enforce_scale_invariance": True, "prefer_distribution_match": True},
    "kmeans":        {"enforce_scale_invariance": True, "prefer_distribution_match": True},
    "rbf_svm":       {"enforce_scale_invariance": True, "prefer_distribution_match": True},
    "neural":        {"enforce_scale_invariance": True, "enforce_shift_invariance": True},
    "tree":          {},  # trees are scale-invariant -> prefer no-op
    "naive_bayes":   {},
    # No model family declared: do NOT impose a hard scale-invariance clause. Forcing
    # enforce_scale_invariance here makes every non-scale-removing policy (log/power/
    # signed-log) infeasible, which systematically blocks the correct fix for skewed
    # data. Leave it to the soft fit-for-purpose term + base contract instead.
    None:            {},
}


def derive_invariance_contracts(
    cards: List[RegimeCard],
    base_contract: Optional[InvarianceContract] = None,
    model_family: Optional[str] = None,
    overrides: Optional[Dict[int, InvarianceContract]] = None,
) -> List[InvarianceContract]:
    """Return one contract per feature.

    Resolution order (later wins):
        global default -> model-family prior -> base_contract -> per-card inference
        -> explicit per-feature override.
    """
    base = base_contract or InvarianceContract()
    prior = MODEL_FAMILY_PRIORS.get(model_family, MODEL_FAMILY_PRIORS[None])
    overrides = overrides or {}

    out: List[InvarianceContract] = []
    for card in cards:
        c = copy.deepcopy(base)

        # apply model-family prior only where the user left the default (False)
        for k, v in prior.items():
            if not getattr(c, k):
                setattr(c, k, v)

        # --- conservative per-card inference -------------------------------- #
        # sparse / structural-zero features: protect the zeros unless told otherwise
        if card.zero_mass > 0.2 and not base.preserve_zero:
            c.preserve_zero = True
        # NOTE: we deliberately do NOT auto-infer preserve_sign here. Centering a
        # signed feature legitimately flips signs; forcing sign-preservation on every
        # signed column would make ordinary standardization infeasible. Sign semantics
        # must be declared explicitly (base contract or per-feature override).
        #
        # heavy tails with no explicit damping stance: ambiguous whether extremes are
        # signal -> bias toward NOT destroying them. BUT only when the heaviness is a
        # symmetric/abrupt tail, not when the feature is simply highly SKEWED -- skew
        # is a distributional pathology that monotone compression should fix, and
        # preserve_extreme_signal would wrongly forbid that compression. So gate on
        # "heavy AND not strongly skewed".
        if (card.tail.value == "heavy" and abs(card.skew) < 2.0
                and not base.damp_outliers and not base.preserve_extreme_signal):
            c.preserve_extreme_signal = True
        # constant / near-constant features: nothing to do -> force no-op via contract
        if card.sign == SignPattern.CONSTANT:
            c = InvarianceContract(preserve_zero=card.zero_mass > 0,
                                   preserve_interpretability=True)
        # tree models: do not impose scaling
        if model_family in ("tree", "naive_bayes"):
            c.enforce_scale_invariance = False
            c.enforce_shift_invariance = False

        if card.index in overrides:
            c = overrides[card.index]

        out.append(c)
    return out
