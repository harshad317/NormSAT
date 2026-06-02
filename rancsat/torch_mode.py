"""Neural mode: RANCSATNorm -- a normalization *layer* compiled from a contract.

Instead of hard-coding BatchNorm/LayerNorm/RMSNorm, we compile the layer's structure
from an InvarianceContract plus an activation regime ("Regime Card" for activations):

    y = gamma * phi( (x - c * mean_A(x)) / (r_A(x) + eps) ) + beta

where
    c        compiled centering BIT     (0 -> RMS-style, no mean removal; 1 -> centered)
    mean_A   mean over the contracted axes
    r_A      scale statistic: rms | std | robust | 1   (compiled)
    phi      identity | clipped | tanh | erf            (compiled, only if stable)
    gamma,beta   learned affine restore params

This reproduces LayerNorm (c=1, r=std), RMSNorm (c=0, r=rms), and bounded variants
(phi=tanh/erf) as *special cases* of one compiled layer, with the choice justified by
the contract and by stability probes -- not picked by hand.

Requires PyTorch. Import this module only when you need neural mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise ImportError(
        "rancsat.torch_mode requires PyTorch. Install it with `pip install torch`."
    ) from e

import numpy as np

from .schemas import InvarianceContract


# --------------------------------------------------------------------------- #
# Activation regime profile (the neural analogue of a Regime Card)
# --------------------------------------------------------------------------- #
@dataclass
class ActivationRegime:
    block: str = "mlp"               # 'mlp' | 'attention' | 'conv'
    batch_reliable: bool = False     # large stable batches available?
    mean_is_signal: bool = False     # is the activation mean informative?
    activation_range: float = 5.0    # typical |activation| magnitude
    saturation_risk: float = 0.3     # 0..1 risk that a bounded map saturates
    depth_sensitive: bool = False    # deep net -> avoid depth-fragile bounded maps
    precision: str = "fp32"          # 'fp16'|'bf16'|'fp32'
    seq_len_varies: bool = False


@dataclass
class NeuralPolicy:
    center: bool                     # c bit
    scale_kind: str                  # 'rms' | 'std' | 'robust' | 'none'
    phi: str                         # 'identity' | 'clip' | 'tanh' | 'erf'
    axes: Tuple[int, ...]            # normalization axes
    reason: str
    rejected: List[Tuple[str, str]]


# --------------------------------------------------------------------------- #
# Compile the neural policy from contract + regime
# --------------------------------------------------------------------------- #
def compile_neural_policy(contract: InvarianceContract, regime: ActivationRegime,
                          normalized_axes: Sequence[int] = (-1,)) -> NeuralPolicy:
    rejected: List[Tuple[str, str]] = []

    # batch dependence: contract forbids batch stats unless batches are reliable
    if contract.avoid_batch_dependence:
        rejected.append(("batchnorm", "avoid_batch_dependence"))
    if contract.avoid_sequence_length_dependence and regime.seq_len_varies:
        rejected.append(("seq_len_stat", "avoid_sequence_length_dependence"))

    # centering bit: remove mean unless the mean is declared signal, or unless the
    # contract explicitly wants shift-invariance (then we MUST center).
    if contract.enforce_shift_invariance:
        center = True
    elif regime.mean_is_signal:
        center = False
        rejected.append(("centering", "mean_is_signal"))
    else:
        # default: RMSNorm-style (cheaper) unless attention block prefers centering
        center = regime.block == "conv"

    # scale statistic
    if contract.damp_outliers:
        scale_kind = "robust"
    elif center:
        scale_kind = "std"
    else:
        scale_kind = "rms"

    # bounded map phi: only adopt if it WON'T saturate / isn't depth-fragile, and the
    # contract permits losing extreme signal.
    phi = "identity"
    if not contract.preserve_extreme_signal:
        if regime.saturation_risk < 0.25 and not regime.depth_sensitive:
            phi = "tanh"
        elif regime.saturation_risk < 0.4 and regime.precision == "fp32":
            phi = "erf"
        else:
            rejected.append(("bounded_map", "saturation_or_depth_probe_failed"))
    else:
        rejected.append(("bounded_map", "preserve_extreme_signal"))
        if contract.damp_outliers:  # clip is the compromise: robust but keeps order
            phi = "clip"

    reason = (f"center={center} (shift_inv={contract.enforce_shift_invariance}, "
              f"mean_signal={regime.mean_is_signal}); scale={scale_kind}; phi={phi}")
    return NeuralPolicy(center=center, scale_kind=scale_kind, phi=phi,
                        axes=tuple(normalized_axes), reason=reason, rejected=rejected)


# --------------------------------------------------------------------------- #
# The compiled layer
# --------------------------------------------------------------------------- #
class RANCSATNorm(nn.Module):
    """A normalization layer whose structure is compiled from an invariance contract.

    Examples
    --------
    >>> contract = InvarianceContract(avoid_batch_dependence=True)
    >>> regime = ActivationRegime(block="mlp", mean_is_signal=False)
    >>> layer = RANCSATNorm(normalized_shape=256, contract=contract, regime=regime)
    >>> y = layer(torch.randn(8, 256))
    """

    def __init__(self, normalized_shape, contract: InvarianceContract,
                 regime: Optional[ActivationRegime] = None,
                 normalized_axes: Sequence[int] = (-1,), eps: float = 1e-5,
                 elementwise_affine: bool = True, clip_value: float = 3.0):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.clip_value = clip_value
        self.contract = contract
        self.regime = regime or ActivationRegime()
        self.policy = compile_neural_policy(contract, self.regime, normalized_axes)
        self.axes = self.policy.axes

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # diagnostics buffers (logged, not used in train/inference math)
        self.register_buffer("_saturation", torch.zeros(1))
        self.register_buffer("_act_norm", torch.zeros(1))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        dims = tuple(a if a >= 0 else x.dim() + a for a in self.axes)

        if self.policy.center:
            mean = x.mean(dim=dims, keepdim=True)
            xc = x - mean
        else:
            xc = x

        sk = self.policy.scale_kind
        if sk == "rms":
            scale = torch.sqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        elif sk == "std":
            scale = torch.sqrt(xc.var(dim=dims, keepdim=True, unbiased=False) + self.eps)
        elif sk == "robust":
            med = xc.median(dim=dims[-1], keepdim=True).values if len(dims) == 1 else \
                xc.flatten(start_dim=min(dims)).median(dim=-1, keepdim=True).values.view(
                    *([1] * x.dim()))
            mad = (xc - med).abs().mean(dim=dims, keepdim=True) + self.eps
            scale = 1.4826 * mad
        else:  # 'none'
            scale = torch.ones_like(x.mean(dim=dims, keepdim=True))

        h = xc / scale

        phi = self.policy.phi
        if phi == "tanh":
            h = torch.tanh(h)
        elif phi == "erf":
            h = torch.erf(h)
        elif phi == "clip":
            h = torch.clamp(h, -self.clip_value, self.clip_value)

        # diagnostics
        with torch.no_grad():
            self._act_norm = h.detach().abs().mean().reshape(1)
            if phi in ("tanh", "erf"):
                self._saturation = (h.detach().abs() > 0.95).float().mean().reshape(1)

        if self.gamma is not None:
            h = self.gamma * h + self.beta
        return h

    def extra_repr(self) -> str:
        return (f"shape={self.normalized_shape}, center={self.policy.center}, "
                f"scale={self.policy.scale_kind}, phi={self.policy.phi}, "
                f"axes={self.axes}")

    def stability_report(self) -> dict:
        return {
            "policy": self.policy.reason,
            "rejected": self.policy.rejected,
            "saturation_frac": float(self._saturation.item()),
            "mean_abs_activation": float(self._act_norm.item()),
        }
