from dataclasses import FrozenInstanceError

import pytest

from prismaquant import allocator_candidates as assembly, format_registry as formats
from prismaquant.priced_cell import CostPricingPolicy, price_cell


def test_explicit_policy_keeps_numeric_price_and_source_consistent_after_environment_changes(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '2')
    policy = CostPricingPolicy.from_environment()
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '99')
    proposal = price_cell({},
        {'predicted_dloss': 3.0, 'predicted_dloss_stderr': 0.5},
        format_name='BF16', policy=policy)
    assert proposal.predicted_dloss == 4.0
    assert proposal.source == 'predicted_dloss'
    with pytest.raises(FrozenInstanceError):
        policy.ucb_z = 0.0


def test_candidate_build_uses_one_policy_even_if_an_adapter_changes_the_environment(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '2')
    original = assembly.check_stats_format_applicability
    def changing_adapter(*args, **kwargs):
        monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '99')
        return original(*args, **kwargs)
    monkeypatch.setattr(assembly, 'check_stats_format_applicability', changing_adapter)
    stats = {name: {'n_params': 512 * 1024,
        'in_features': 1024, 'out_features': 512} for name in ('a', 'b')}
    costs = {name: {'BF16': {'predicted_dloss': 3.0,
        'predicted_dloss_stderr': 0.5}} for name in stats}
    candidates = assembly.build_candidates(stats, costs, [formats.get_format('BF16')]).candidates
    assert [candidates[name][0].predicted_dloss for name in stats] == [4.0, 4.0]
