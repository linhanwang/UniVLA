"""Apply Liger kernels to Emu3 custom modules.

HF's `use_liger_kernel=True` no-ops on Emu3 because its `model_type="Emu3"` is
not in Liger's registry, and the RMSNorm/MLP classes are re-defined locally
rather than imported from `transformers.models.llama`. This patches the
local classes directly.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from liger_kernel.transformers.functional import liger_rms_norm, liger_swiglu

from . import modeling_emu3

logger = logging.getLogger(__name__)

_PATCHED = False


def _liger_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return liger_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _liger_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.config.pretraining_tp > 1:
        return getattr(modeling_emu3.Emu3MLP, "_orig_forward")(self, x)
    return self.down_proj(liger_swiglu(self.gate_proj(x), self.up_proj(x)))


def apply_liger_kernel_to_emu3(rms_norm: bool = True, swiglu: bool = True) -> None:
    global _PATCHED
    if _PATCHED:
        return

    if rms_norm:
        setattr(modeling_emu3.Emu3RMSNorm, "forward", _liger_rms_norm_forward)

    if swiglu:
        setattr(modeling_emu3.Emu3MLP, "_orig_forward", modeling_emu3.Emu3MLP.forward)
        setattr(modeling_emu3.Emu3MLP, "forward", _liger_mlp_forward)

    _PATCHED = True
    logger.info(
        "Applied Liger kernels to Emu3 (rms_norm=%s, swiglu=%s)", rms_norm, swiglu
    )
