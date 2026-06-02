"""rancsat: Regime-Aware Normalization Compiler with SAT-certified invariance contracts.

NormSAT / RANC-SAT treats normalization not as a moment-estimation task but as a
*constraint-satisfaction / compilation* problem: you declare what must stay invariant
and what signal must survive (an Invariance Contract), the compiler profiles the data
into Regime Cards, enumerates a finite Policy DSL, rejects policies that violate hard
clauses, scores feasible policies lexicographically, fits parameters *train-only*, and
falsifies the result before freezing an auditable transformer.

Public API
----------
    from rancsat import RANCSATTransformer, InvarianceContract
    t = RANCSATTransformer(contract=InvarianceContract(preserve_zero=True))
    Xt = t.fit_transform(X_train)
    report = t.audit_report()

Optional neural mode (requires torch):
    from rancsat.torch_mode import RANCSATNorm, compile_neural_policy
"""

from .schemas import (
    RegimeCard,
    InvarianceContract,
    SignalRiskRow,
    NormalizationPolicy,
    FalsificationResult,
    AuditReport,
    PolicyKind,
    SignPattern,
    TailCategory,
)
from .profiling import build_regime_cards
from .contracts import derive_invariance_contracts
from .ledger import build_signal_risk_ledger
from .compiler import compile_policy, RANCCompiler
from .falsification import run_falsification_tests
from .sklearn import RANCSATTransformer

__version__ = "0.1.0"

__all__ = [
    "RegimeCard",
    "InvarianceContract",
    "SignalRiskRow",
    "NormalizationPolicy",
    "FalsificationResult",
    "AuditReport",
    "PolicyKind",
    "SignPattern",
    "TailCategory",
    "build_regime_cards",
    "derive_invariance_contracts",
    "build_signal_risk_ledger",
    "compile_policy",
    "RANCCompiler",
    "run_falsification_tests",
    "RANCSATTransformer",
    "__version__",
]
