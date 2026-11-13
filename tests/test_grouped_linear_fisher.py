"""Grouped-BMM Linears: the `wo_a` shape gets a correctly-shaped probe row.

DSv4's `attn.wo_a` is an `nn.Linear` subclass whose forward consumes its
`[G*R, D]` weight plane as `W[g, r, d]` through a view + bmm. These tests pin:

  - the probe inventory prices a declared grouped class as one
    `grouped_linear` unit (its probe row is pinned in
    tests/test_body_fisher_sweep.py);
  - `finalize_fisher_stats` divides the per-group decomposition by the same
    GLOBAL calibration token count as its scalar;
  - dispatch is explicit (profile-declared classes only) and fails fast
    when a declared class lacks its group count.

The synthetic `GroupedBMMLinear` mirrors the vendored
`DeepseekV4GroupedLinear.forward` line for line; the slow test at the
bottom runs the REAL vendored class end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from prismaquant.probe_inventory import build_probe_inventory
from prismaquant.sensitivity_probe import grouped_linear_groups
from prismaquant.probe_statistics import finalize_fisher_stats

G, R, D, T = 2, 3, 4, 4


class GroupedBMMLinear(nn.Linear):
    """Mirror of DeepseekV4GroupedLinear: standard [out, in] storage,
    per-group bmm consumption."""

    def __init__(self, in_features_per_group: int, out_features: int,
                 n_groups: int):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        xx = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(xx, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class _DeclaringProfile:
    """Minimal profile stand-in: declares GroupedBMMLinear, nothing else."""

    def probe_grouped_module_class_names(self):
        return ("GroupedBMMLinear",)


class _SilentProfile:
    def probe_grouped_module_class_names(self):
        return ()


def test_finalize_normalizes_the_per_group_decomposition():
    """`h_trace_per_group_raw` is a [G] decomposition of h_trace_raw; it
    must divide by the same global count as its scalar."""
    stats = {"grouped": {
        "h_trace_raw": 96.0, "h_w2_sum_raw": 24.0,
        "h_trace_per_group_raw": [40.0, 56.0],
        "n_tokens_seen": 4,
    }}
    finalize_fisher_stats(stats, global_tokens=4)
    s = stats["grouped"]
    assert s["h_trace"] == pytest.approx(24.0)
    assert s["h_trace_norm_tokens"] == 4
    assert s["h_trace_per_group"] == pytest.approx([10.0, 14.0])
    assert sum(s["h_trace_per_group"]) == pytest.approx(s["h_trace"])


def test_inventory_prices_a_declared_grouped_class_as_one_grouped_unit():
    model = nn.Module()
    model.grouped = GroupedBMMLinear(D, G * R, G)
    model.dense = nn.Linear(D, R, bias=False)

    class _Profile(_DeclaringProfile):
        def should_probe_linear(self, name, module):
            return isinstance(module, nn.Linear)

    units = {unit.module_name: unit for unit in
             build_probe_inventory(model, _Profile()).priced_units}
    assert units["grouped"].kind == "grouped_linear"
    assert units["grouped"].parameters == ("grouped.weight",)
    assert units["dense"].kind == "linear"


def test_dispatch_is_explicit_and_fails_fast():
    mod = GroupedBMMLinear(D, G * R, G)
    plain = nn.Linear(D, R)
    assert grouped_linear_groups(mod, _DeclaringProfile()) == G
    # Undeclared class -> dense path (explicit declaration only).
    assert grouped_linear_groups(mod, _SilentProfile()) is None
    assert grouped_linear_groups(mod, None) is None
    assert grouped_linear_groups(plain, _DeclaringProfile()) is None
    # Declared class without its group count must fail LOUD, not fall
    # through to numbers the dense flatten would get wrong.
    bare = type("GroupedBMMLinear", (nn.Linear,), {})(D, G * R)
    with pytest.raises(ValueError, match="n_groups"):
        grouped_linear_groups(bare, _DeclaringProfile())


@pytest.mark.own_process
@pytest.mark.slow
def test_real_dsv4_wo_a_gets_a_priced_probe_row():
    """End to end on the REAL vendored DSv4 modeling code (toy dims): the
    probe tracks `DeepseekV4GroupedLinear` as a priced grouped unit with the
    right group structure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_model_walk import _shrunken_dsv4

    profile, model = _shrunken_dsv4()
    tracked = sorted(
        n for n, m in model.named_modules()
        if type(m).__name__ == "DeepseekV4GroupedLinear")
    assert tracked, "shrunken DSv4 must contain wo_a modules"
    for name in tracked:
        assert profile.should_probe_linear(name, model.get_submodule(name))
    units = {unit.module_name: unit
             for unit in build_probe_inventory(model, profile).priced_units}

    o_groups, o_lora_rank = 4, 32
    heads_x_head_dim = 8 * 32
    d_in = heads_x_head_dim // o_groups
    for name in tracked:
        module = model.get_submodule(name)
        assert units[name].kind == "grouped_linear"
        assert grouped_linear_groups(module, profile) == o_groups
        assert module.out_features == o_groups * o_lora_rank
        assert module.in_features == d_in
