"""A fused output shape cannot hide different gate and up widths."""
import numpy as np
import pytest
import torch

from prismaquant.aqua_activation_cost import packed_act_dloss_per_expert, per_expert_weight_keys
from test_aqua_per_expert_checkpoint import E, M, N, _unit


@pytest.mark.parametrize("shift", [-1, 0, 1])
def test_each_gate_up_projection_matches_its_role(shift):
    unit = _unit()
    lookup = {}
    for expert in range(E):
        stem = f"model.layers.0.mlp.experts.{expert}"
        lookup[f"{stem}.gate_proj.weight"] = torch.ones(M // 2 + shift, N)
        lookup[f"{stem}.up_proj.weight"] = torch.ones(M // 2 - shift, N)
    keys = per_expert_weight_keys(unit.topology.name, lookup, n_experts=E)
    assert keys is not None
    if shift:
        with pytest.raises(RuntimeError, match="shape|projection|materialized"):
            packed_act_dloss_per_expert(unit, keys, "unused", np.ones((E, N)), handles=lookup.__getitem__)
    else:
        assert packed_act_dloss_per_expert(unit, keys, "unused", np.ones((E, N)), handles=lookup.__getitem__) > 0
