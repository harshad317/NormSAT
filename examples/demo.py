"""End-to-end demo of RANC-SAT / NormSAT.

Run:  python examples/demo.py
"""
import numpy as np

from rancsat import RANCSATTransformer, InvarianceContract
from rancsat.benchmark import (
    run_oracle_benchmark, run_downstream_benchmark, format_table,
)
from rancsat import synthetic


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    # ------------------------------------------------------------------ 1
    section("1. Compile a transform from an Invariance Contract")
    rng = np.random.default_rng(0)
    X = np.column_stack([
        rng.normal(10, 2, 500),                       # clean gaussian
        np.maximum(0, rng.poisson(0.3, 500).astype(float)),  # sparse counts
        np.exp(rng.normal(0, 1, 500)),                # positive skew
    ])
    contract = InvarianceContract(enforce_scale_invariance=True, preserve_zero=True)
    t = RANCSATTransformer(contract=contract,
                           feature_names=["clean", "sparse", "skewed"])
    t.fit(X)
    print(t.audit_report().summary())

    # ------------------------------------------------------------------ 2
    section("2. Oracle benchmark -- does the compiler recover known-correct policies?")
    for row in run_oracle_benchmark():
        print(f"{row['regime']:>26}: policy_acc={row['policy_accuracy']:.2f}  "
              f"chosen={row['chosen']}")

    # ------------------------------------------------------------------ 3
    section("3. Decisive regime: signal-vs-noise outliers")
    reg = synthetic.make_signal_vs_noise_outliers()
    overrides = synthetic.per_feature_contracts_for_outliers()
    print("--- linear model (scale-equivariant -> scaling barely matters) ---")
    print(format_table(run_downstream_benchmark(
        reg.X, reg.y, task="classification", contract=reg.contract,
        overrides=overrides, feature_names=reg.feature_names, model="logreg")))
    print("\n--- KNN (distance-sensitive -> per-feature handling is decisive) ---")
    print(format_table(run_downstream_benchmark(
        reg.X, reg.y, task="classification", contract=reg.contract,
        overrides=overrides, feature_names=reg.feature_names, model="knn")))
    print("\nHONEST READ: RANC-SAT compiles a sensible per-feature program with ZERO "
          "model\nfits, but it does NOT dominate here. Under a linear model it ties; "
          "under KNN a\nplain global quantile-normal wins, because KNN needs COMMENSURATE "
          "feature spreads\nand 'preserve extreme signal' leaves f_signal's tail "
          "un-bounded, dominating\ndistances. Lesson: the contract must reflect the "
          "DOWNSTREAM MODEL's real needs\n(a distance model wants rank-equalisation, which "
          "conflicts with magnitude\npreservation). This tension is the actual research "
          "question -- not a bug to tune away.")

    # ------------------------------------------------------------------ 4
    section("4. Inverse transform round-trip (positive-skew regime)")
    reg = synthetic.make_positive_skew()
    t = RANCSATTransformer(contract=reg.contract, feature_names=reg.feature_names,
                           inverse_required=True)
    Z = t.fit_transform(reg.X)
    Xrec = t.inverse_transform(Z)
    print(f"max round-trip error: {np.max(np.abs(reg.X - Xrec)):.2e}")

    # ------------------------------------------------------------------ 5
    section("5. Neural mode (if torch installed)")
    try:
        import torch
        from rancsat.torch_mode import RANCSATNorm, ActivationRegime
        layer = RANCSATNorm(256, InvarianceContract(avoid_batch_dependence=True),
                            ActivationRegime(block="mlp", mean_is_signal=False))
        out = layer(torch.randn(8, 256))
        print("compiled layer:", layer.extra_repr())
        print("stability:", layer.stability_report())
        print("output shape:", tuple(out.shape))
    except ImportError:
        print("torch not installed -- skipping neural mode.")


if __name__ == "__main__":
    main()
