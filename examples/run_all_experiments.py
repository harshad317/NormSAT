"""Reproduce every experiment in the paper. Writes results/ tables to stdout + files.

Run:  PYTHONPATH=. python examples/run_all_experiments.py
"""
import io
import contextlib

from rancsat.benchmark import run_oracle_benchmark, run_multiseed_ablation, format_ablation
from rancsat.diagnose import run_diagnosis
from rancsat.heterogeneous import run_heterogeneous_benchmark, run_guarantee_audit
from rancsat.efficiency import run_efficiency_table, run_audit_coverage


def banner(t):
    print("\n" + "#" * 78 + f"\n# {t}\n" + "#" * 78)


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        banner("E1. Oracle policy-recovery (does the compiler pick the right transform?)")
        for r in run_oracle_benchmark():
            print(f"{r['regime']:>26}: policy_acc={r['policy_accuracy']:.2f} "
                  f"chosen={r['chosen']}")

        banner("E2. Multi-seed ablation vs heuristic menu + AutoML + global scalers")
        for mdl in ["knn", "logreg"]:
            run_multiseed_ablation(seeds=range(5), model=mdl)

        banner("E3. Failure-mode diagnosis (is per-feature handling even the lever?)")
        for mdl in ["knn", "logreg"]:
            run_diagnosis(model=mdl, seeds=range(3))

        banner("E4. Heterogeneous-pathology benchmark")
        run_heterogeneous_benchmark(seeds=range(5), model="knn")
        print()
        run_heterogeneous_benchmark(seeds=range(5), model="logreg")

        banner("E5. Guarantee-violation audit (declared invariants broken)")
        run_guarantee_audit(seeds=range(5))

        banner("E6. Model-fit efficiency + audit coverage")
        run_efficiency_table(seeds=range(5))
        print()
        run_audit_coverage(seeds=range(3))

    out = buf.getvalue()
    print(out)
    with open("results_all.txt", "w") as f:
        f.write(out)
    print("\n[wrote results_all.txt]")


if __name__ == "__main__":
    main()
