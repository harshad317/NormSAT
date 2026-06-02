"""Run the component ablation study (standalone; imports only from the rancsat package).

Usage:
    PYTHONPATH=. python examples/run_ablation.py --data-dir ./my_datasets --seeds 3
    PYTHONPATH=. python examples/run_ablation.py --seeds 3          # uses bundled data

Outputs the ablation table to stdout and writes ablation_study_report.txt.
"""
import argparse

from rancsat.ablation_study import run_ablation_study, format_ablation_study
from rancsat.realdata import load_local_dir, load_real_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Folder of CSV/ARFF datasets (target = last column). "
                         "If omitted, uses sklearn-bundled real datasets.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="logreg", choices=["logreg", "knn"])
    args = ap.parse_args()

    if args.data_dir:
        print(f"Loading datasets from {args.data_dir} ...")
        datasets = load_local_dir(args.data_dir)
    else:
        print("Using sklearn-bundled real datasets ...")
        datasets = load_real_datasets()
    if not datasets:
        raise SystemExit("No datasets loaded. Check --data-dir contents.")

    summary = run_ablation_study(datasets, seeds=range(args.seeds), model=args.model)
    text = format_ablation_study(summary)
    print("\n" + text)
    with open("ablation_study_report.txt", "w") as f:
        f.write(text + "\n")
    print("\n[wrote ablation_study_report.txt]")


if __name__ == "__main__":
    main()
