"""Test suite for rancsat. Run: pytest -q"""
import numpy as np
import pytest

from rancsat import (
    RANCSATTransformer, InvarianceContract, PolicyKind, build_regime_cards,
)
from rancsat.compiler import hard_violations, compile_policy
from rancsat.contracts import derive_invariance_contracts
from rancsat.ledger import build_signal_risk_ledger
from rancsat import synthetic, policies as P
from rancsat.schemas import NormalizationPolicy


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def test_profiling_detects_structure():
    rng = np.random.default_rng(0)
    X = np.column_stack([
        rng.normal(0, 1, 1000),
        np.where(rng.random(1000) < 0.7, 0.0, rng.poisson(2, 1000)),
        np.exp(rng.normal(0, 1, 1000)),
    ])
    cards = build_regime_cards(X, ["g", "sparse", "pos"])
    assert cards[1].zero_mass > 0.5
    assert cards[2].sign.value == "positive"
    assert cards[2].skew > 0.5


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #
def test_preserve_zero_rejects_centering():
    rng = np.random.default_rng(1)
    X = np.where(rng.random((500, 1)) < 0.6, 0.0, rng.normal(5, 1, (500, 1)))
    cards = build_regime_cards(X, ["s"])
    contract = InvarianceContract(preserve_zero=True, enforce_scale_invariance=True)
    v = hard_violations(PolicyKind.AFFINE_STANDARD, cards[0], contract)
    assert "preserve_zero" in v
    pol = compile_policy(cards[0], contract, [], col=X[:, 0])
    assert P.CAPABILITIES[pol.kind].preserves_zero


def test_positive_domain_rejects_log_on_negative():
    X = np.random.default_rng(2).normal(0, 1, (300, 1))  # has negatives
    cards = build_regime_cards(X, ["x"])
    v = hard_violations(PolicyKind.LOG1P, cards[0],
                        InvarianceContract(preserve_monotonicity=True))
    assert any("requires_nonnegative" in c for c in v)


def test_scale_invariance_forbids_identity():
    X = np.random.default_rng(3).normal(0, 5, (300, 1))
    cards = build_regime_cards(X, ["x"])
    contract = InvarianceContract(enforce_scale_invariance=True)
    v = hard_violations(PolicyKind.IDENTITY, cards[0], contract)
    assert "enforce_scale_invariance" in v


# --------------------------------------------------------------------------- #
# Compiler end-to-end + UNSAT
# --------------------------------------------------------------------------- #
def test_unsat_falls_back_to_identity():
    # "center the data (remove location) but keep structural zeros" is impossible:
    # every location-removing policy densifies, so the feasible set is empty.
    rng = np.random.default_rng(4)
    X = np.where(rng.random((300, 1)) < 0.6, 0.0, rng.normal(5, 1, (300, 1)))
    cards = build_regime_cards(X, ["x"])
    contract = InvarianceContract(enforce_shift_invariance=True, preserve_zero=True)
    pol = compile_policy(cards[0], contract, [], col=X[:, 0])
    assert pol.is_noop and pol.kind == PolicyKind.IDENTITY
    assert pol.unsat_core  # non-empty core explaining infeasibility
    assert "enforce_shift_invariance" in pol.unsat_core or "preserve_zero" in pol.unsat_core


# --------------------------------------------------------------------------- #
# Transformer API + leakage safety
# --------------------------------------------------------------------------- #
def test_fit_transform_shapes_and_determinism():
    X = np.random.default_rng(5).normal(size=(400, 5))
    t = RANCSATTransformer(contract=InvarianceContract(enforce_scale_invariance=True))
    Z1 = t.fit_transform(X)
    Z2 = t.transform(X)
    assert Z1.shape == X.shape
    assert np.allclose(Z1, Z2)  # frozen -> deterministic


def test_transform_does_not_refit_on_new_data():
    rng = np.random.default_rng(6)
    Xtr = rng.normal(0, 1, (400, 3))
    t = RANCSATTransformer(contract=InvarianceContract(enforce_scale_invariance=True))
    t.fit(Xtr)
    params_before = [dict(p.params) for p in t.policies_]
    Xte = rng.normal(5, 3, (100, 3))  # shifted/scaled distribution
    t.transform(Xte)
    params_after = [dict(p.params) for p in t.policies_]
    for a, b in zip(params_before, params_after):
        assert a.keys() == b.keys()
        for k in a:
            assert np.allclose(np.ravel(a[k]), np.ravel(b[k]))


def test_inverse_round_trip():
    reg = synthetic.make_positive_skew()
    t = RANCSATTransformer(contract=reg.contract, feature_names=reg.feature_names,
                           inverse_required=True)
    Z = t.fit_transform(reg.X)
    Xrec = t.inverse_transform(Z)
    assert np.max(np.abs(reg.X - Xrec)) < 1e-6


def test_tree_model_prefers_noop():
    X = np.random.default_rng(7).normal(0, 3, (300, 4))
    t = RANCSATTransformer(model_family="tree")
    t.fit(X)
    assert t.audit_report().noop_fraction == 1.0


# --------------------------------------------------------------------------- #
# Oracle recovery
# --------------------------------------------------------------------------- #
def test_oracle_signal_vs_noise():
    # Hard-constraint guarantees (deterministic): the compiler MUST keep extreme
    # signal on the signal feature and MUST be robust on the noise feature.
    reg = synthetic.make_signal_vs_noise_outliers()
    overrides = synthetic.per_feature_contracts_for_outliers()
    t = RANCSATTransformer(contract=reg.contract, feature_names=reg.feature_names,
                           overrides=overrides).fit(reg.X, reg.y)
    caps = [P.CAPABILITIES[p.kind] for p in t.policies_]
    assert caps[0].keeps_extreme_signal and caps[0].removes_scale   # f_signal
    assert caps[1].robust_to_outliers                               # f_noise
    assert np.all(np.isfinite(t.transform(reg.X)))


def test_oracle_sparse_preserves_zero():
    reg = synthetic.make_sparse_semantic_zeros()
    t = RANCSATTransformer(contract=reg.contract,
                           feature_names=reg.feature_names).fit(reg.X, reg.y)
    for p in t.policies_:
        assert P.CAPABILITIES[p.kind].preserves_zero  # guaranteed by preserve_zero clause


def test_guarantee_audit_rancsat_is_zero():
    # RANC-SAT must NOT violate any declared invariant on the heterogeneous benchmark:
    # no zero-densification, no sign-flip, no unsafe-bound breach. Provable by the hard
    # clauses (preserve_zero, preserve_sign, unsafe_bound rejection).
    from rancsat.heterogeneous import (
        make_heterogeneous, count_guarantee_violations)
    from sklearn.model_selection import train_test_split
    ds = make_heterogeneous(seed=0)
    Xtr, Xte, ytr, yte = train_test_split(
        ds.X, ds.y, test_size=0.3, random_state=0, stratify=ds.y)
    t = RANCSATTransformer(contract=ds.contract, overrides=ds.overrides,
                           feature_names=ds.feature_names).fit(Xtr, ytr)
    r = count_guarantee_violations(Xtr, Xte, ds.feature_names, t.policies_, ds.overrides)
    assert sum(r["counts"].values()) == 0, r["counts"]


def test_unsafe_bound_rejected_on_nonsemantic_feature():
    # minmax / quantile_uniform must be hard-rejected on a drift-prone, non-semantic
    # feature (they cannot keep their bound promise out-of-sample).
    from rancsat.compiler import hard_violations
    X = np.random.default_rng(11).normal(0, 5, (400, 1))
    cards = build_regime_cards(X, ["x"])  # bounded_semantic defaults to False
    for k in (PolicyKind.MINMAX, PolicyKind.QUANTILE_UNIFORM):
        v = hard_violations(k, cards[0], InvarianceContract())
        assert any("unsafe_bound" in c for c in v)
    # but ALLOWED when the feature's bounds are declared semantic
    cards_sem = build_regime_cards(X, ["x"], bounded_features=[0])
    assert not any("unsafe_bound" in c
                   for c in hard_violations(PolicyKind.MINMAX, cards_sem[0],
                                            InvarianceContract()))


def test_positive_skew_compiles_power_transform():
    # regression guard for the under-transformation bug: lognormal data must get a
    # monotone power/log compression, NOT a no-op maxabs.
    reg = synthetic.make_positive_skew()
    t = RANCSATTransformer(contract=reg.contract,
                           feature_names=reg.feature_names).fit(reg.X, reg.y)
    compressors = {PolicyKind.BOXCOX, PolicyKind.LOG1P, PolicyKind.YEOJOHNSON,
                   PolicyKind.SIGNED_LOG, PolicyKind.QUANTILE_NORMAL}
    assert all(p.kind in compressors for p in t.policies_), \
        [p.kind.value for p in t.policies_]


def test_model_aware_distance_contract_prefers_distribution_match():
    # A distance-model contract (prefer_distribution_match) should route features to
    # rank/quantile equalization rather than plain affine scaling, so features become
    # commensurate. Compare against the same data WITHOUT the preference.
    rng = np.random.default_rng(21)
    # several features with different shapes/scales
    X = np.column_stack([
        rng.normal(0, 1, 800),
        rng.normal(100, 30, 800),
        np.exp(rng.normal(0, 1, 800)),
        rng.standard_t(3, 800),
    ])
    names = ["a", "b", "c", "d"]
    plain = RANCSATTransformer(
        contract=InvarianceContract(enforce_scale_invariance=True),
        feature_names=names).fit(X)
    dist = RANCSATTransformer(
        contract=InvarianceContract(enforce_scale_invariance=True,
                                    prefer_distribution_match=True),
        feature_names=names).fit(X)
    n_quant_plain = sum(p.kind in (PolicyKind.QUANTILE_NORMAL,
                                   PolicyKind.QUANTILE_UNIFORM) for p in plain.policies_)
    n_quant_dist = sum(p.kind in (PolicyKind.QUANTILE_NORMAL,
                                  PolicyKind.QUANTILE_UNIFORM) for p in dist.policies_)
    assert n_quant_dist > n_quant_plain
    # the distance-matched output should be MORE commensurate than the plain one
    # (lower spread-ratio across features), even if not perfectly equal.
    rng2 = dist.transform(X).std(axis=0)
    plain_std = plain.transform(X).std(axis=0)
    ratio_dist = rng2.max() / (rng2.min() + 1e-9)
    ratio_plain = plain_std.max() / (plain_std.min() + 1e-9)
    assert ratio_dist <= ratio_plain


def test_repair_on_falsification():
    # build a case where chosen policy could fail a check, ensure repair leaves valid state
    reg = synthetic.make_drift()
    t = RANCSATTransformer(contract=reg.contract, feature_names=reg.feature_names)
    t.fit(reg.X, reg.y)
    assert all(f.passed for f in t.falsification_)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
