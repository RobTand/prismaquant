"""Selected anchors reuse canonical inputs without loading the whole source."""
import pytest


@pytest.mark.parametrize('extra', [[], ['--calibration-cache', '/capture'],
    ['--calibration-cache-sha256', 'a'*64]])
def test_selected_source_requires_complete_hash_bound_capture_before_loading(tmp_path, extra):
    from prismaquant.tessera_campaign import main
    with pytest.raises(SystemExit) as caught:
        main(['--model', '/missing', '--out', str(tmp_path/'out'),
            '--cache-dir', str(tmp_path/'cache'), '--streaming',
            '--units', '/units', *extra])
    assert caught.value.code == 2
    assert not (tmp_path/'cache').exists()


@pytest.fixture
def selected_runner(monkeypatch):
    import torch
    from types import SimpleNamespace
    from prismaquant.cost_streaming import StreamedCausalLM
    from prismaquant import routed_experts
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
    for layer in model.layers:
        layer.proj = torch.nn.Linear(4, 3, bias=False, device='meta')
    calls, pending = [], set()
    source = {i: torch.arange(12, dtype=torch.float32).reshape(3, 4)+i for i in range(4)}
    def schedule(layer):
        calls.append(('schedule', layer)); pending.add(layer)
    def install(layer, *, require_prefetched, prefetch_following):
        assert require_prefetched and not prefetch_following
        assert layer in pending
        pending.remove(layer)
        model.layers[layer].proj.weight = torch.nn.Parameter(source[layer])
        calls.append(('install', layer))
        return 'wait'
    def release(layer):
        model.layers[layer].proj.weight = torch.nn.Parameter(torch.empty(3, 4, device='meta'))
        calls.append(('release', layer))
    context = SimpleNamespace(model=model, base_model=model, layers=model.layers,
        layers_prefix='layers.', num_layers=4, device='cpu', dtype=torch.float32,
        max_cache_slots=2, schedule_prefetch=schedule, install=install,
        release_completed_layer=release)
    monkeypatch.setattr(routed_experts, 'profile_declared_packed_expert_projections', lambda *_: [])
    monkeypatch.setattr(routed_experts, 'refresh_packed_expert_projections', lambda *_: [])
    runner = StreamedCausalLM(context, profile=SimpleNamespace(),
        prefetch_lookahead=1, require_prefetched_residency=True)
    return runner, calls, source


def test_selected_source_prefetches_only_selected_layers_and_releases_them(selected_runner):
    import torch
    runner, calls, source = selected_runner
    values, receipt = runner.snapshot_selected_weights(['layers.3.proj', 'layers.1.proj'],
                                                       max_resident_bytes=96)
    assert [layer for kind, layer in calls if kind == 'schedule'] == [1, 3]
    assert [layer for kind, layer in calls if kind == 'release'] == [1, 3]
    assert receipt['resident_bytes'] == 96 and receipt['source_forward_count'] == 0
    for i in (1, 3):
        assert torch.equal(values[f'layers.{i}.proj'], source[i])
        assert values[f'layers.{i}.proj'].untyped_storage().data_ptr() != source[i].untyped_storage().data_ptr()
    assert all(p.is_meta for p in runner.model.parameters())


def test_selected_source_refuses_budget_before_any_source_read(selected_runner):
    runner, calls, _source = selected_runner
    with pytest.raises(RuntimeError, match='resident byte budget'):
        runner.snapshot_selected_weights(['layers.1.proj'], max_resident_bytes=47)
    assert calls == []


def test_selected_source_releases_layer_when_copy_guard_refuses(selected_runner):
    runner, calls, _source = selected_runner
    def refuse(*args, **kwargs):
        raise RuntimeError('budget refusal')
    with pytest.raises(RuntimeError, match='budget refusal'):
        runner.snapshot_selected_weights(['layers.1.proj'], max_resident_bytes=48,
                                         resource_check=refuse)
    assert calls[-1] == ('release', 1)
    assert all(p.is_meta for p in runner.model.parameters())


def test_selected_admission_excludes_unselected_source_and_forward_owners(monkeypatch):
    from prismaquant import autoscale
    monkeypatch.setattr(autoscale, 'streamed_calibration_resources', lambda *a, **k: dict(
        live_layer_prefix='layers.', terms=dict(nonbody_source_bytes=100, declared_headroom_bytes=200),
        body_layer_bytes={'0': 1000, '1': 10000, '2': 2000},
        body_loader_transient_bytes={'0': 100, '1': 1000, '2': 200},
        body_source_file_bytes={'0': 900, '1': 9000, '2': 1800},
        full_hessian_bytes=128, full_prefix_bytes=64, source_header_sha256='a'*64))
    plan = autoscale.selected_anchor_resources('/source',
        unit_shapes={'layers.0.proj': [3, 4], 'layers.2.proj': [3, 4]},
        counts={'layers.0.proj': 9, 'layers.2.proj': 9}, max_act_rows=2,
        cache_slots=2, prefetch_workers=1, headroom_gb=0)
    assert plan['selected_layers'] == ['0', '2']
    source, anchors = plan['phases'].values()
    assert source['source_window_bytes'] == 3000
    assert source['loader_transient_bytes'] == 200
    assert anchors['selected_hessian_bytes'] == 128
    assert anchors['selected_prefix_bytes'] == 64
    assert 'nonbody_source_bytes' not in anchors
    assert all('boundary' not in key for phase in plan['phases'].values() for key in phase)
    assert plan['memory_bytes'] == max(map(lambda phase: sum(phase.values()), plan['phases'].values()))


def test_streaming_planner_requires_capture_and_stamps_selected_phase_plan(monkeypatch, tmp_path):
    import json
    from tools import dispatch_tessera_campaign as dispatch
    census = dict(model='/source', anchor_groups={'u:layers.0.proj': ['layers.0.proj']},
        layer_stride=1, unit_shapes={'layers.0.proj': [3, 4]}, counts={'layers.0.proj': 9})
    (tmp_path/'census.json').write_text(json.dumps(census))
    spec = dict(model='/source', campaign_argv=['--streaming'], cwd=str(tmp_path),
                python='python3', env={}, cpus=1)
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    common = ['plan', '--spec', str(tmp_path/'spec.json'), '--workspace', str(tmp_path)]
    with pytest.raises(RuntimeError, match='complete calibration cache'):
        dispatch.main(common)
    monkeypatch.setattr(dispatch, '_calibration_cache_binding', lambda *a: dict(path='/capture', sha256='a'*64))
    def resources(spec, census, members, *, selected_source):
        assert selected_source and members == ['layers.0.proj']
        return dict(memory_bytes=3*1024**3, selected_layers=['0'])
    monkeypatch.setattr(dispatch, '_streamed_resource_plan', resources)
    assert dispatch.main([*common, '--calibration-cache', '/capture']) == 0
    rows = json.loads((tmp_path/'manifest.json').read_text())
    assert rows[0]['demand']['mem_gb'] == 3
    assert rows[0]['env']['MIMALLOC_PURGE_DELAY'] == '0'
    assert rows[0]['env']['PRISMAQUANT_RELEASE_SOURCE_PAGES'] == '1'
    assert '--calibration-cache-sha256' in rows[0]['argv']
    plan = json.loads((tmp_path/'plan.json').read_text())
    assert plan['rows'][0]['resources']['selected_layers'] == ['0']


def test_selected_capture_cli_reaches_existing_streamed_source(monkeypatch, tmp_path):
    from test_tessera_campaign_resume import _main_fixture
    from prismaquant import cost_streaming
    campaign, _, argv, _, _ = _main_fixture(monkeypatch, tmp_path)
    argv[argv.index('--hessian') + 1] = 'require'

    class SelectedSourceReached(Exception):
        pass

    def build(*args, **kwargs):
        assert kwargs['require_prefetched_residency'] is True
        raise SelectedSourceReached

    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', build)
    with pytest.raises(SelectedSourceReached):
        campaign.main([*argv, '--streaming', '--units', str(tmp_path/'units.json'),
            '--calibration-census', str(tmp_path/'census.json'),
            '--calibration-cache', str(tmp_path/'capture_manifest.json'),
            '--calibration-cache-sha256', 'a'*64,
            '--attention-implementation', 'eager'])
