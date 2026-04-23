"""Convert Emu3MLP's Linear layers to torchao float8 for training speedup.

Targets only the MLP projections (gate/up/down). Attention, embeddings, and
lm_head stay in bf16 — they are either shape-unfriendly (head_dim=128 splits)
or numerically sensitive. MLPs dominate FLOPs, so fp8 there captures most of
the speedup with minimal accuracy risk.
"""

from __future__ import annotations

import torch.nn as nn


_MLP_LINEAR_NAMES = ("gate_proj", "up_proj", "down_proj")


def _is_mlp_linear(mod: nn.Module, fqn: str) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    leaf = fqn.rsplit(".", 1)[-1]
    if leaf not in _MLP_LINEAR_NAMES:
        return False
    # Dims must be multiples of 16 for fp8 scaled matmul.
    return mod.in_features % 16 == 0 and mod.out_features % 16 == 0


def convert_emu3_mlp_to_fp8(model: nn.Module) -> int:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    config = Float8LinearConfig.from_recipe_name("tensorwise")
    convert_to_float8_training(model, config=config, module_filter_fn=_is_mlp_linear)

    converted = sum(
        1
        for name, mod in model.named_modules()
        if mod.__class__.__name__ == "Float8Linear" and name.rsplit(".", 1)[-1] in _MLP_LINEAR_NAMES
    )
    return converted
