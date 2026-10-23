"""Probe inventory must retain exclusions and profile declaration failures."""
import pytest
import torch
from torch import nn

from prismaquant import probe_inventory
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_walk import ClaimRule, evaluate_walk_gate, find_decided_but_unpriced, walk_model


class SharedGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.shared_expert_gate = nn.Linear(8, 8, bias=False, device='meta')

    def forward(self, value):
        return self.mlp.shared_expert_gate(value)


def test_end_anchored_module_exclusion_refuses_a_decided_parameter():
    model = SharedGateModel()
    result = walk_model(model, (torch.empty(2, 8, device='meta'),),
                        claim_rules=[ClaimRule('decide', 'linear weight', leaf='weight',
                                               module_class='Linear')], strict=False)
    entries = find_decided_but_unpriced(result, model, DefaultProfile())
    assert entries == ({'node': 'mlp.shared_expert_gate.weight',
                        'owner_module': 'mlp.shared_expert_gate',
                        'module_class': 'Linear', 'reason_code': 'probe_linear_excluded'},)
    assert evaluate_walk_gate(result, unpriced_decides=entries).refused


class BrokenPackedProfile(DefaultProfile):
    def packed_expert_param_names(self):
        raise RuntimeError('declared inventory unavailable')

    def packed_expert_parent_for_projection(self, name):
        raise RuntimeError('declared parent unavailable')


@pytest.mark.parametrize('query,args', [
    (probe_inventory.packed_expert_param_name_set, ()),
    (probe_inventory.packed_expert_parent_for_projection, ('gate_proj',)),
])
def test_declared_inventory_failures_propagate(query, args):
    with pytest.raises(RuntimeError, match='declared .* unavailable'):
        query(BrokenPackedProfile(), *args)


class BrokenProjectionProfile(DefaultProfile):
    def packed_expert_projection_names(self, name):
        raise RuntimeError('declared projections unavailable')


def test_declared_projection_failure_propagates():
    with pytest.raises(RuntimeError, match='declared projections unavailable'):
        probe_inventory.packed_expert_projection_candidate_names(BrokenProjectionProfile())


class PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(2, 4, 8, device='meta'))
        self.down_proj = nn.Parameter(torch.empty(2, 8, 2, device='meta'))


class GroupedLinear(nn.Linear):
    pass


class InventoryProfile(DefaultProfile):
    def probe_grouped_module_class_names(self):
        return ('GroupedLinear',)


def test_inventory_publishes_actual_priced_kinds_and_excluded_module_names():
    model = nn.Module()
    model.dense = nn.Linear(8, 4, bias=False, device='meta')
    model.grouped = GroupedLinear(8, 4, bias=False, device='meta')
    model.experts = PackedExperts()
    model.mlp = nn.Module()
    model.mlp.shared_expert_gate = nn.Linear(8, 2, bias=False, device='meta')
    inventory = probe_inventory.build_probe_inventory(model, InventoryProfile())
    assert [(unit.module_name, unit.kind, unit.parameters) for unit in inventory.priced_units] == [
        ('dense', 'linear', ('dense.weight',)),
        ('grouped', 'grouped_linear', ('grouped.weight',)),
        ('experts', 'packed_experts', ('experts.down_proj', 'experts.gate_up_proj')),
    ]
    assert inventory.linear_names == ('dense', 'grouped')
    assert [(entry.module_name, entry.reason_code) for entry in inventory.excluded_linears] == [
        ('mlp.shared_expert_gate', 'probe_linear_excluded'),
    ]


def test_missing_optional_declarations_are_an_empty_inventory():
    profile = object()
    assert probe_inventory.packed_expert_param_name_set(profile) == set()
    assert probe_inventory.packed_expert_parent_for_projection(profile, 'gate_proj') is None
    assert probe_inventory.packed_expert_projection_candidate_names(profile) == ()
