"""Executable schema + Policy DSL for RANC-SAT.

Everything the compiler reasons over is declared here as a dataclass so it can be
serialized into an audit certificate. Nothing in this module touches data; it only
defines vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class PolicyKind(str, Enum):
    """The finite Policy DSL: candidate per-feature transforms the compiler may emit."""

    IDENTITY = "identity"                 # no-op
    AFFINE_STANDARD = "affine_standard"   # (x - mean) / std         (z-score)
    AFFINE_ROBUST = "affine_robust"       # (x - median) / IQR
    MINMAX = "minmax"                     # (x - min) / (max - min)  -> [0, 1]
    MAXABS = "maxabs"                     # x / max|x|               (keeps zeros)
    ROBUST_MAXABS = "robust_maxabs"       # x / q99(|x|)             (keeps zeros, spike-robust)
    CLIP_ROBUST = "clip_robust"           # winsorize to [qlo, qhi], then robust affine
    SIGNED_LOG = "signed_log"             # sign(x) * log1p(|x| / s)
    LOG1P = "log1p"                       # log1p(x)                 (x >= 0)
    BOXCOX = "boxcox"                     # power transform          (x > 0)
    YEOJOHNSON = "yeojohnson"             # power transform          (signed ok)
    QUANTILE_UNIFORM = "quantile_uniform"  # empirical CDF -> [0, 1]
    QUANTILE_NORMAL = "quantile_normal"    # rank-gaussian
    # group-level policies (handled by the transformer, not per-feature loop)
    VECTOR_L2 = "vector_l2"               # row-wise L2 normalization
    WHITENING = "whitening"               # ZCA whitening over a feature group


class SignPattern(str, Enum):
    POSITIVE = "positive"        # strictly > 0
    NONNEGATIVE = "nonnegative"  # >= 0, zeros present
    SIGNED = "signed"            # both signs present
    CONSTANT = "constant"        # ~zero variance


class TailCategory(str, Enum):
    LIGHT = "light"
    NORMAL = "normal"
    HEAVY = "heavy"


# --------------------------------------------------------------------------- #
# Regime Card -- a structured statistical profile of one feature (train-only)
# --------------------------------------------------------------------------- #
@dataclass
class RegimeCard:
    name: str
    index: int
    n: int

    # location / scale
    mean: float
    std: float
    median: float
    iqr: float
    mad: float

    # support
    min: float
    max: float
    abs_max: float
    robust_abs_max: float            # q99 of |x|

    # shape
    skew: float
    excess_kurtosis: float
    tail: TailCategory
    sign: SignPattern

    # structure
    zero_mass: float                 # fraction exactly == 0
    sparsity: float                  # alias of zero_mass for sparse features
    missing_frac: float
    outlier_frac: float              # fraction beyond median +/- 3*MAD
    bounded_semantic: bool           # caller declared finite semantic bounds

    # reliability of estimates (drives the soft costs)
    scale_stability: float           # 0..1 ; high => mean/std trustworthy
    covariance_reliability: float    # 0..1
    drift: float                     # 0..1 estimated train-internal drift

    # requirements forwarded from metadata
    distance_sensitive: bool
    interpretability_required: bool
    inverse_required: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tail"] = self.tail.value
        d["sign"] = self.sign.value
        return d


# --------------------------------------------------------------------------- #
# Invariance Contract -- what must hold. Boolean / ordinal.
# --------------------------------------------------------------------------- #
@dataclass
class InvarianceContract:
    enforce_shift_invariance: bool = False      # remove location
    enforce_scale_invariance: bool = False      # remove scale
    preserve_rank: bool = False                 # monotone transforms only
    preserve_distance_ratios: bool = False      # affine-only (no nonlinear)
    preserve_zero: bool = False                 # 0 -> 0 ; no densification
    preserve_sign: bool = False                 # sign(f(x)) == sign(x)
    preserve_monotonicity: bool = True          # default: don't reorder values
    damp_outliers: bool = False                 # robustify against extremes
    preserve_extreme_signal: bool = False       # extremes may be signal -> don't clip
    reduce_covariance: bool = False             # whiten correlated features
    avoid_batch_dependence: bool = True         # (neural) no batch stats
    avoid_sequence_length_dependence: bool = True  # (neural)
    preserve_interpretability: bool = False     # keep human-readable units
    allow_inverse_transform: bool = False       # transform must be invertible
    prefer_distribution_match: bool = False     # SOFT: prefer making features
    #   commensurate across the table (rank/quantile equalization) -- what a
    #   distance-based downstream model (KNN, k-means, RBF-SVM) actually needs.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Signal Risk Ledger row -- one declared nuisance-removal and its danger
# --------------------------------------------------------------------------- #
@dataclass
class SignalRiskRow:
    feature: str
    proposed_removal: str            # what nuisance the policy removes
    possible_true_signal: str        # what real signal could be lost
    evidence: str
    severity: float                  # 0..1
    confidence: float                # 0..1
    mitigation: str
    falsification_test: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Compiled policy -- a frozen, fitted transform program for one feature/group
# --------------------------------------------------------------------------- #
@dataclass
class NormalizationPolicy:
    feature: str
    index: int
    kind: PolicyKind
    params: Dict[str, float] = field(default_factory=dict)
    invertible: bool = False
    is_noop: bool = False
    cost_vector: Tuple[float, ...] = ()      # (R, U, D, K, C, I)
    rejected: List[Tuple[str, str]] = field(default_factory=list)  # (kind, reason)
    unsat_core: List[str] = field(default_factory=list)            # violated clauses if fallback
    notes: str = ""
    drift_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


# --------------------------------------------------------------------------- #
# Falsification result
# --------------------------------------------------------------------------- #
@dataclass
class FalsificationResult:
    feature: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Audit report -- the certificate emitted with the frozen transformer
# --------------------------------------------------------------------------- #
@dataclass
class AuditReport:
    data_hash: str
    n_features: int
    n_samples: int
    split_type: str
    labels_used: bool
    contract: Dict[str, Any]
    cards: List[Dict[str, Any]] = field(default_factory=list)
    policies: List[Dict[str, Any]] = field(default_factory=list)
    ledger: List[Dict[str, Any]] = field(default_factory=list)
    falsification: List[Dict[str, Any]] = field(default_factory=list)
    noop_fraction: float = 0.0
    rejected_summary: Dict[str, int] = field(default_factory=dict)
    leakage_guard: str = "fit-on-train-fold-only"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Concise human-readable certificate."""
        lines = [
            f"RANC-SAT audit  |  {self.n_features} features x {self.n_samples} samples",
            f"split={self.split_type}  labels_used={self.labels_used}  "
            f"no-op fraction={self.noop_fraction:.0%}",
            f"leakage guard: {self.leakage_guard}",
            "policies:",
        ]
        for p in self.policies:
            tag = " [NO-OP]" if p.get("is_noop") else ""
            core = f"  unsat_core={p['unsat_core']}" if p.get("unsat_core") else ""
            lines.append(f"  - {p['feature']:>14}: {p['kind']}{tag}{core}")
        n_fail = sum(1 for f in self.falsification if not f.get("passed", True))
        lines.append(f"falsification: {n_fail} feature(s) failed and were repaired/fallback")
        return "\n".join(lines)
