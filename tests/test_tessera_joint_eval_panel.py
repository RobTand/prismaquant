"""The original encoding draw and diagnostic joint draw have different owners."""
import copy

import pytest
import torch

from prismaquant.tessera_joint_eval_panel import make_panel, select_panel, observation_status


def test_legacy_full_draw_and_nested_deterministic_prefix():
    ids = torch.arange(512 * 4, dtype=torch.int64).reshape(512, 4)
    calibration = {'artifact_sha256': 'a' * 64, 'shape': [512, 4]}
    legacy, selected = select_panel(ids, calibration, None)
    assert legacy is ids and selected is None
    small = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=237, size=16)
    large = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=237, size=32)
    assert large['selection']['indices'][:16] == small['selection']['indices']
    assert small['selection']['indices'] != list(range(16))
    evaluated, replayed = select_panel(ids, calibration, small)
    assert evaluated.shape == (16, 4) and replayed == small
    assert calibration['shape'] == [512, 4] and torch.equal(ids, legacy)
    assert small['eval_ids_sha256'] != make_panel(ids,
        artifact_sha256=calibration['artifact_sha256'], seed=238, size=16)['eval_ids_sha256']


@pytest.mark.parametrize('field', ['indices', 'eval_ids_sha256', 'shape', 'artifact', 'seed'])
def test_panel_refuses_changed_selection_and_token_identity(field):
    ids = torch.arange(48, dtype=torch.int64).reshape(12, 4)
    calibration = {'artifact_sha256': 'a'*64}
    panel = make_panel(ids, artifact_sha256=calibration['artifact_sha256'], seed=7, size=4)
    changed = copy.deepcopy(panel)
    if field == 'indices': changed['selection']['indices'][0] ^= 1
    elif field == 'eval_ids_sha256': changed[field] = 'b'*64
    elif field == 'shape': changed[field] = [5, 4]
    elif field == 'artifact': changed['calibration_input_sha256'] = 'b'*64
    else: changed['selection']['seed'] += 1
    with pytest.raises(ValueError, match='joint evaluation'):
        select_panel(ids, calibration, changed)
    ids[panel['selection']['indices'][0], 0] += 1
    with pytest.raises(ValueError, match='joint evaluation'):
        select_panel(ids, calibration, panel)


def test_pilot_handoff_and_allocator_refuse_even_observed_zero():
    from prismaquant.cost_currency import CostCurrencyError, require_run_currency
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    from test_tessera_joint_allocation import fixture
    joint, data, prepared, metadata, kwargs = fixture()
    joint['provenance']['joint_eval'] = {'status': 'diagnostic_pilot'}
    joint['provenance']['tessera_joint_anchors']['joint_eval'] = {'status': 'diagnostic_pilot'}
    name = next(iter(joint['costs']))
    joint['stats'][name]['joint_eval_observations'] = {'tokens': 0, 'calls': 0}
    joint['stats'][name]['joint_eval_status'] = 'unknown_unobserved'
    assert observation_status({'tokens': 0, 'calls': 0}) == 'unknown_unobserved'
    assert observation_status({'tokens': 0, 'calls': 1}) == 'unknown_unobserved'
    assert observation_status({'tokens': 3, 'calls': 1}) == 'observed'
    with pytest.raises(CostCurrencyError, match='separate sampled-proposal path'):
        require_run_currency(joint)
    with pytest.raises(ValueError, match='separate sampled-proposal path'):
        bind_allocation_payload(joint, data, prepared, metadata, **kwargs)


@pytest.mark.parametrize('alias', ['dotdot', 'symlink'])
def test_plan_generator_refuses_alias_of_original_output_root(tmp_path, monkeypatch, alias):
    import json
    from prismaquant import tessera_joint_eval_panel as panel

    baseline = tmp_path / 'original-output'
    baseline.mkdir()
    base_plan = tmp_path / 'base-plan.json'
    base_plan.write_text(json.dumps({
        'schema': 'prismaquant.tessera_joint_aura.plan.v1',
        'output_root': str(baseline),
        'execution': {'n_calib_samples': 4, 'calib_seqlen': 2,
                      'boundary_storage': {'directory': str(baseline / 'exact-boundaries')}},
        'calibration_input': {'path': '/fixture/tokens.safetensors', 'sha256': 'a'*64},
    }))
    monkeypatch.setattr(panel, 'load_calibration_input', lambda *_args, **_kwargs:
                        (torch.arange(8, dtype=torch.int64).reshape(4, 2),
                         {'artifact_sha256': 'a'*64}))
    if alias == 'dotdot':
        requested = baseline / '..' / baseline.name
    else:
        requested = tmp_path / 'alias'
        requested.symlink_to(baseline, target_is_directory=True)
    assert str(requested) != str(baseline)
    assert requested.resolve() == baseline.resolve()
    output = tmp_path / 'pilot-plan.json'
    with pytest.raises(ValueError, match='distinct output root'):
        panel.main(['--base-plan', str(base_plan), '--output', str(output),
                    '--output-root', str(requested), '--seed', '7', '--size', '2'])
    assert not output.exists()


def test_observer_distinguishes_invoked_exact_zero_from_unobserved():
    from prismaquant.joint_aura import JointOperatorStatisticsLease
    from prismaquant.format_registry import get_format
    a = torch.nn.Linear(2, 2, bias=False).eval()
    b = torch.nn.Linear(2, 2, bias=False).eval()
    modules = {'invoked': a, 'unobserved': b}
    specs = {name: {'FP8_E4M3': get_format('FP8_E4M3')} for name in modules}
    with JointOperatorStatisticsLease(modules, specs,
            max_statistics_bytes=2048, max_candidate_bytes=2048) as lease:
        lease.begin_probe()
        x = torch.zeros(3, 2, requires_grad=True)
        a(x).sum().backward()
        lease.finish_observations()
        diagnostics = lease.operator_diagnostics(collect_col_energy=False)
        assert diagnostics['invoked']['observed_calls'] == 1
        assert diagnostics['invoked']['observed_tokens'] == 3
        assert diagnostics['invoked']['g_trace'] == 0
        assert diagnostics['unobserved'] == {'g_trace': 0., 'observed_tokens': 0, 'observed_calls': 0}


def test_pilot_counts_are_per_probe_and_survive_resume(tmp_path, monkeypatch):
    from prismaquant import aura_cost as aura
    from test_joint_aura_streamed import _fixture, _run
    from test_joint_operator_windows import policy
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1'*40)
    panel = {'schema': 'prismaquant.tessera_joint_eval_panel.v1', 'status': 'diagnostic_pilot'}
    def run(resume):
        _, _, runner, cache = _fixture()
        return _run(runner, cache, operator_windows=policy(), checkpoint_dir=tmp_path,
                    checkpoint_identity_extra={'joint_eval': panel}, resume=resume)
    first = run(False)
    resumed = run(True)
    assert first['costs'] == resumed['costs']
    for name, stat in first['stats'].items():
        observations = stat['joint_eval_observations']
        assert observations['count_scope'] == 'summed_over_probes'
        assert observations['n_probes'] == 3
        assert len(observations['per_probe']) == 3
        assert observations['tokens'] == sum(row['tokens'] for row in observations['per_probe'])
        assert observations['calls'] == sum(row['calls'] for row in observations['per_probe'])
        assert stat['joint_eval_status'] == 'observed'
        assert resumed['stats'][name] == stat
