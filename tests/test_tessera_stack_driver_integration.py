"""Check that checkpoint identity binds Fisher and sampling values."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace


from prismaquant import tessera_campaign_wire as campaign_wire
from prismaquant import tessera_campaign as campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def test_checkpoint_binds_fisher_and_probability_values_with_same_measured_units(monkeypatch):
    from prismaquant import tessera_campaign as campaign
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    monkeypatch.setattr(campaign_wire, 'require_checkpoint_identity_api', lambda: SimpleNamespace(
        encoder_source_sha256=lambda: 'encoder', tensor_identity=lambda t: t))
    monkeypatch.setattr(campaign.th, 'encoder_recipe', lambda: {})
    monkeypatch.setattr('prismaquant.production_weight_cache._production_cache_source_sha256', lambda: 'pq')
    common = dict(weights={'a': 'tensor'}, acts={}, hessians={}, menus={'a': []},
        settings=dict(units='/a.json'), calibration_identity={}, serving_scope=None,
        static_scales={}, static_scale_policy='policy')
    record = {'stack': {'probe_row': {'h_trace': 4.0}, 'inclusion_prob': {'0': 0.5}}}
    first = campaign.campaign_checkpoint_identity(**common, stack_sampling_identity=record)
    changed = copy.deepcopy(record)
    changed['stack']['inclusion_prob']['0'] = 0.75
    second = campaign.campaign_checkpoint_identity(**common, stack_sampling_identity=changed)
    assert first['units'] == second['units']
    assert canonical_json_sha256(first, where="first") != canonical_json_sha256(second, where="second")
