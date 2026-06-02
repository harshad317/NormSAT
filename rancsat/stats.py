"""Significance testing for the benchmark.

Given per-(dataset, seed) paired scores for RANC-SAT vs each baseline, we report:
  * win / tie / loss counts,
  * median paired difference and a bootstrap CI,
  * Wilcoxon signed-rank p-value (paired, non-parametric),
  * Holm-Bonferroni correction across the family of baseline comparisons,
  * a rank-based effect size (matched-pairs rank-biserial correlation).

We keep classification (AUROC, higher-better) and regression (RMSE) separate and
always orient differences so that POSITIVE means "RANC-SAT better".
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _oriented_diff(rancsat: float, other: float, higher_better: bool) -> float:
    """Positive => RANC-SAT better."""
    return (rancsat - other) if higher_better else (other - rancsat)


def bootstrap_ci(diffs: Sequence[float], n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial(diffs: Sequence[float]) -> float:
    """Matched-pairs rank-biserial effect size in [-1, 1]; >0 favors RANC-SAT."""
    d = np.asarray([x for x in diffs if x != 0], dtype=float)
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    rpos = ranks[d > 0].sum()
    rneg = ranks[d < 0].sum()
    total = ranks.sum()
    return float((rpos - rneg) / total)


def paired_test(diffs: Sequence[float]) -> Dict:
    d = np.asarray(diffs, dtype=float)
    n = d.size
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    out = {
        "n": n, "wins": wins, "losses": losses, "ties": ties,
        "median_diff": float(np.median(d)) if n else float("nan"),
        "mean_diff": float(np.mean(d)) if n else float("nan"),
        "ci95": bootstrap_ci(d),
        "effect_size_rrb": rank_biserial(d),
        "p_value": float("nan"),
    }
    nz = d[d != 0]
    if _HAVE_SCIPY and nz.size >= 6:
        try:
            out["p_value"] = float(wilcoxon(nz, alternative="two-sided").pvalue)
        except Exception:
            pass
    return out


def holm_correction(pairs: Dict[str, Dict]) -> Dict[str, Dict]:
    """Apply Holm-Bonferroni across the baseline comparisons (in-place add fields)."""
    items = [(k, v["p_value"]) for k, v in pairs.items() if np.isfinite(v["p_value"])]
    m = len(items)
    items.sort(key=lambda kv: kv[1])
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        # enforce monotonicity of Holm-adjusted p-values
        pairs[k]["p_holm"] = adj
    # monotone non-decreasing in sorted order
    running = 0.0
    for k, _ in items:
        running = max(running, pairs[k]["p_holm"])
        pairs[k]["p_holm"] = running
        pairs[k]["significant_05"] = bool(running < 0.05)
    for k, v in pairs.items():
        v.setdefault("p_holm", float("nan"))
        v.setdefault("significant_05", False)
    return pairs


def compare_method_to_baselines(
    paired: Dict[str, List[float]],
) -> Dict[str, Dict]:
    """paired: {baseline_name: [oriented diffs across (dataset, seed)]}.
    Returns per-baseline test dict with Holm correction applied."""
    res = {name: paired_test(diffs) for name, diffs in paired.items()}
    return holm_correction(res)


def format_significance(res: Dict[str, Dict], title: str = "") -> str:
    L = [f"SIGNIFICANCE: RANC-SAT vs baselines  {title}".rstrip(),
         "(positive diff => RANC-SAT better; p_holm = Holm-corrected Wilcoxon)", ""]
    hdr = f"    {'baseline':<18}{'n':>4}{'W/T/L':>10}{'median Δ':>11}{'95% CI':>20}{'eff':>7}{'p_holm':>9}  sig"
    L.append(hdr)
    L.append("    " + "-" * (len(hdr) - 4))
    for name, r in sorted(res.items(), key=lambda kv: kv[1].get("p_holm", 1.0)):
        ci = r["ci95"]
        ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        ph = r.get("p_holm", float("nan"))
        sig = "***" if (np.isfinite(ph) and ph < 0.05) else ""
        L.append(f"    {name:<18}{r['n']:>4}{wtl:>10}{r['median_diff']:>+11.4f}"
                 f"{ci_s:>20}{r['effect_size_rrb']:>+7.2f}{ph:>9.3f}  {sig}")
    return "\n".join(L)
