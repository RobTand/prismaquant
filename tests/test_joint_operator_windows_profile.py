"""CPU instrumentation checks; native operator qualification runs separately."""
import copy

import pytest
import torch

from experiments.joint_operator_windows_profile import (
    ATOL, PROBES, StorageObserver, make_policy, require_cost_parity,
    require_cotangent_parity,
)


def test_matrix_observer_tracks_backing_storage_through_alias():
    observer = StorageObserver()
    matrix = torch.ones(4, 16)
    alias = matrix[:1]
    observer.register('statistics', 'unit', matrix)
    del matrix
    assert observer.live() == {'statistics': 256}
    del alias
    assert observer.live() == {}
    assert observer.peaks == {'statistics': 256}


def _rows():
    return [dict(name='unit', format='FP8_E4M3',
        components=[dict(weight=1., activation=-0.25, mixed=0.125, total=0.875)] * PROBES,
        signed=[0.875] * PROBES,
        probe=dict(calibration='frozen', arithmetic='legacy'),
        operator=dict(source_weight='original', rendered_weight='frozen',
                      arithmetic='legacy', probe_identity_sha256='legacy'))]


def test_signed_parity_accepts_stated_rounding_with_distinct_arithmetic_identity():
    before = _rows()
    after = copy.deepcopy(before)
    after[0]['probe']['arithmetic'] = 'operator-statistics'
    after[0]['operator'].update(arithmetic='operator-statistics', probe_identity_sha256='statistics')
    after[0]['components'][0]['weight'] += ATOL / 2
    result = require_cost_parity(before, after)
    assert result['compared_components'] == 4 * PROBES


@pytest.mark.parametrize('field', ['weight', 'activation', 'mixed', 'total'])
def test_signed_parity_refuses_component_drift_and_nonfinite(field):
    before = _rows()
    for value in (10., float('nan')):
        after = copy.deepcopy(before)
        after[0]['components'][0][field] = value
        with pytest.raises(RuntimeError, match=f'signed {field}'):
            require_cost_parity(before, after)


@pytest.mark.parametrize('field', ['source_weight', 'rendered_weight'])
def test_signed_parity_does_not_excuse_changed_source_or_render_identity(field):
    before = _rows()
    after = copy.deepcopy(before)
    after[0]['operator'][field] = 'changed'
    with pytest.raises(RuntimeError, match='input identity'):
        require_cost_parity(before, after)


def test_signed_parity_requires_exact_candidate_roster():
    before = _rows()
    with pytest.raises(RuntimeError, match='roster'):
        require_cost_parity(before, [])


def _cotangents():
    return [dict(layer=1, batch=0, incoming=dict(sha256=str(probe)),
                 outgoing=dict(sha256=f'output-{probe}')) for probe in range(PROBES)]


def test_cotangent_comparison_allows_reordered_replay_but_checks_every_copy():
    baseline = _cotangents()
    replay = copy.deepcopy(list(reversed(baseline)) * 3)
    assert require_cotangent_parity(baseline, replay) == dict(
        unique_cotangents=PROBES, observed_cotangents=PROBES * 3)
    replay[-1]['outgoing']['sha256'] = 'changed'
    with pytest.raises(RuntimeError, match='bytes changed'):
        require_cotangent_parity(baseline, replay)


def test_cotangent_comparison_refuses_missing_or_ambiguous_probe():
    rows = _cotangents()
    with pytest.raises(RuntimeError, match='omitted'):
        require_cotangent_parity(rows, rows[1:])
    with pytest.raises(RuntimeError, match='uniquely'):
        require_cotangent_parity(rows * 2, rows)


def test_policy_admits_largest_whole_target_and_forces_routed_windows():
    from prismaquant import format_registry as fr
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    modules = {f'model.layers.1.mlp.experts.{index}.up_proj':
               torch.nn.Linear(64, 32, bias=False, device='meta') for index in range(12)}
    shapes = {name: list(module.weight.shape) for name, module in modules.items()}
    fixture = dict(shapes={**shapes, 'model.layers.0.mlp.up_proj': [128, 64]},
                   entries=[dict(serialized_bytes=20000)])
    policy = make_policy(fixture)
    plan = plan_joint_statistics_target_windows(modules,
        {name: {fmt: fr.get_format(fmt) for fmt in ('FP8_E4M3', 'NVFP4A16')} for name in modules},
        max_statistics_bytes=policy['max_statistics_bytes'])
    assert len(plan.windows) == 3
    assert sorted(name for window in plan.windows for name in window) == sorted(modules)
    assert policy['max_candidate_bytes'] == 128 * 64 * 4
