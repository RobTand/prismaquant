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
        unit_source_weight_bytes={'layers.0.proj': 24, 'layers.2.proj': 24},
        full_hessian_bytes=128, full_prefix_bytes=64, source_header_sha256='a'*64))
    plan = autoscale.selected_anchor_resources('/source',
        unit_shapes={'layers.0.proj': [3, 4], 'layers.2.proj': [3, 4]},
        counts={'layers.0.proj': 9, 'layers.2.proj': 9}, max_act_rows=2,
        cache_slots=2, prefetch_workers=1, headroom_gb=0)
    assert plan['selected_layers'] == ['0', '2']
    source, export, anchors = plan['phases'].values()
    assert source['source_window_bytes'] == 3000
    assert source['loader_transient_bytes'] == 200
    assert anchors['selected_hessian_bytes'] == 128
    assert anchors['selected_prefix_bytes'] == 64
    assert export['export_input_page_window_bytes'] == 64+2*16384
    assert plan['schema'] == 'prismaquant.selected_anchor_resources.v2'
    assert plan['export_input_writer_policy'] == 'verified-tensor-record-prefix'
    assert export['serialization_scratch_bytes'] == 4**2*4
    assert anchors['encoder_memo_bytes'] == 4*4*4+4*8
    assert plan['encoder_memo_capacity'] == 1
    assert 'nonbody_source_bytes' not in anchors
    assert all('boundary' not in key for phase in plan['phases'].values() for key in phase)
    assert plan['memory_bytes'] == max(map(lambda phase: sum(phase.values()), plan['phases'].values()))


def test_selected_plan_terms_follow_the_allocations_they_bound(monkeypatch):
    """Each charged term is the shape and dtype of a traced allocation.

    The plan is a delta over a process floor it cannot see, so a term that is
    a chosen multiplier is not conservative, it is unattributable
    (RobTand/prismaquant#390). Every number here is derived in
    ``selected_anchor_resources``' comments from a named allocating line.
    """
    from prismaquant import autoscale
    monkeypatch.setattr(autoscale, 'streamed_calibration_resources', lambda *a, **k: dict(
        live_layer_prefix='layers.', terms=dict(nonbody_source_bytes=100, declared_headroom_bytes=200),
        body_layer_bytes={'0': 1000, '2': 2000}, body_loader_transient_bytes={'0': 100, '2': 200},
        body_source_file_bytes={'0': 900, '2': 1800},
        unit_source_weight_bytes={'layers.0.proj': 24, 'layers.2.wide': 48},
        full_hessian_bytes=128, full_prefix_bytes=64, source_header_sha256='a'*64))
    shapes = {'layers.0.proj': [3, 4], 'layers.2.wide': [3, 6]}
    # Different counts put the widest H and the widest X on different
    # units, so a sum of separate maxima is distinguishable from the
    # widest single entry the loader actually holds.
    counts = {'layers.0.proj': 9, 'layers.2.wide': 1}
    plan = autoscale.selected_anchor_resources('/source', unit_shapes=shapes,
        counts=counts, max_act_rows=2, cache_slots=2, prefetch_workers=1,
        headroom_gb=0, anchor_batch_size=3)
    anchors = plan['phases']['resident_anchors']
    export = plan['phases']['export_inputs']
    widest_h = 6**2*4
    # torch.serialization._save stages exactly one CPU copy of a device
    # storage per data/ record, not two.
    assert export['serialization_scratch_bytes'] == widest_h
    # The seal/unit content hash and the regularise-plus-factorise stage are
    # sequential, and each peaks at two fp32 copies of the widest H.
    assert anchors['factorization_scratch_bytes'] == 2*widest_h
    # One capture entry's CPU payload and its device copy coexist; the entry
    # is H plus X for ONE unit, so the widest entry is not the sum of the
    # widest H and the widest X measured on different units.
    assert anchors['entry_validation_bytes'] == 2*max(
        4*(cols**2 + min(counts[name], 2)*cols)
        for name, (_rows, cols) in shapes.items())
    # The retained keywords are the [in, in] fp32 LDL factor and two fp32
    # [in] refit metrics, once per memo entry the memo is built with.
    assert plan['encoder_memo_capacity'] == 3
    assert anchors['encoder_memo_bytes'] == 3*(6**2*4 + 6*8)
    # No process baseline can be derived before the row runs, so the plan
    # says which pre-run term it has instead of inventing one.
    assert plan['baseline_policy'] == 'declared-headroom-pre-run-measured-in-row'
    assert anchors['declared_headroom_bytes'] == 200


def test_selected_guard_measures_its_own_process_floor(tmp_path, monkeypatch):
    """The floor a delta plan is admitted against is read, never assumed."""
    from prismaquant import memory_management as memory
    gib = 1024**3
    root = tmp_path/'cgroup'
    child = root/'job'
    child.mkdir(parents=True)
    (root/'memory.max').write_text(str(16*gib))
    (root/'memory.current').write_text(str(gib))
    (child/'memory.max').write_text('max')
    membership = tmp_path/'membership'
    membership.write_text('0::/job\n')
    monkeypatch.setattr(memory.torch.cuda, 'memory_reserved', lambda _device: 2*gib)
    monkeypatch.setattr(memory, '_host_memory_info', lambda: (20*gib, 32*gib))
    guard = memory.CaptureMemoryGuard('cuda', cgroup_root=root, membership=membership)
    with pytest.raises(RuntimeError, match='baseline is unmeasured'):
        guard.baseline_bytes()
    guard.check('before_selected_capture_identity')
    assert guard.baseline_bytes() == 3*gib
    (root/'memory.current').write_text(str(4*gib))
    guard.check('before_selected_encoder_factors:layers.0.proj')
    snapshot = guard.snapshot()
    # The floor stays the first reading; the peak moves and says where.
    assert snapshot['baseline'] == dict(label='before_selected_capture_identity',
        bytes=3*gib, measured_in_process=True, cgroup_current_bytes=gib,
        cuda_reserved_bytes=2*gib)
    assert snapshot['peak_conservative_bytes'] == 6*gib
    assert snapshot['peak_checkpoint'] == 'before_selected_encoder_factors:layers.0.proj'
    assert snapshot['peak_by_checkpoint_prefix'] == {
        'before_selected_capture_identity': 3*gib,
        'before_selected_encoder_factors': 6*gib}


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


def test_bounded_encoder_memo_releases_evicted_factors_and_recomputes_identically(monkeypatch):
    import weakref
    import torch
    from prismaquant import tessera_campaign as campaign
    refs = []
    def encoder(source, name, columns, device, *, scale_plane):
        result = dict(ldl=torch.eye(columns)*(int(name)+1),
                      refit_metric=torch.full((columns,), float(scale_plane)))
        refs.append(weakref.ref(result['ldl']))
        return result
    monkeypatch.setattr(campaign.th, 'encoder_kwargs', encoder)
    weights = {str(i): torch.empty(2, 4) for i in range(6)}
    results = []
    for capacity, expected_live in ((None, 12), (2, 2)):
        memo = campaign._activation_kwargs_memo(None, weights, 'cpu', max_entries=capacity)
        outputs = []
        for repeat in range(2):
            for plane in (1, 2):
                for name in weights:
                    kwargs = memo(name, plane)
                    outputs.append((kwargs['ldl'].tolist(), kwargs['refit_metric'].tolist()))
                    del kwargs
                    assert sum(ref() is not None for ref in refs) <= (12 if capacity is None else 2)
        assert sum(ref() is not None for ref in refs) == expected_live
        results.append(outputs)
        memo.cache_clear()
        assert all(ref() is None for ref in refs)
        del memo
    assert results[0] == results[1]


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='real encoder memo qualification requires CUDA')
def test_native_bounded_encoder_memo_keeps_wire_and_price_bytes(tmp_path, monkeypatch):
    import json
    import os
    import weakref
    from pathlib import Path
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_campaign as campaign
    generator = torch.Generator().manual_seed(370)
    weights = {f'unit{i}': torch.randn(16, 256, generator=generator).to('cuda', torch.bfloat16)
               for i in range(3)}
    acts = {name: torch.randn(7, 256, generator=generator).cuda() for name in weights}
    hs = {name: torch.eye(256, device='cuda')*(i+1) for i, name in enumerate(weights)}
    identity = campaign.th.calibration_identity('bounded memo', [torch.ones(1, 256, dtype=torch.long)],
                                               fit_tokens=256)
    source = campaign.th.activation_source(hs, identity)
    original = campaign.th.encoder_kwargs
    shared = {value.untyped_storage().data_ptr() for value in hs.values()}
    refs = []
    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        refs.extend(weakref.ref(value) for value in result.values()
                    if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in shared)
        return result
    monkeypatch.setattr(campaign.th, 'encoder_kwargs', tracked)
    outputs = []
    measurements = []
    for capacity in (None, 1):
        memo = campaign._activation_kwargs_memo(source, weights, 'cuda', max_entries=capacity)
        root = tmp_path/str(capacity)
        root.mkdir()
        cache = SimpleNamespace(weights={}, cache_dir=str(root))
        rows = []
        peak_owned_bytes = 0
        for fmt in ('TESSERA_E4M3_K1_R1024', 'TESSERA_E4M3_K1_R1280'):
            for name, weight in weights.items():
                anchor = campaign._measure_anchor(qname=name, weight=weight, activations=acts[name],
                    format_name=fmt, cache=cache, wire_dir=root,
                    activation_kwargs_for=memo, hessian_required=True)
                row = vars(anchor).copy()
                row.pop('seconds')
                rows.append(row)
                alive = [ref() for ref in refs]
                live = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                        for value in alive if value is not None}
                peak_owned_bytes = max(peak_owned_bytes, sum(live.values()))
                del alive, live
        wires = {path.name: path.read_bytes() for path in root.glob('*.tessera')}
        outputs.append((rows, wires))
        assert memo.cache_info().currsize == (3 if capacity is None else 1)
        measurements.append(dict(capacity=capacity, retained_entries=memo.cache_info().currsize,
                                 peak_owned_factor_bytes_after_anchor=peak_owned_bytes))
        memo.cache_clear()
    assert outputs[0] == outputs[1]
    assert measurements[1]['peak_owned_factor_bytes_after_anchor'] < measurements[0]['peak_owned_factor_bytes_after_anchor']
    if os.environ.get('PRISMAQUANT_SELECTED_SOURCE_PROFILE'):
        path = Path(os.environ['PRISMAQUANT_SELECTED_SOURCE_PROFILE'])/'memo-parity.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(measurements=measurements, wire_and_price_parity=True), indent=2)+'\n')


def test_selected_capture_cli_refuses_missing_capture_before_streamed_source(monkeypatch, tmp_path):
    from test_tessera_campaign_resume import _main_fixture
    from prismaquant import cost_streaming
    from prismaquant.autoscale import BOUNDED_CAPTURE_ENV
    campaign, _, argv, _, _ = _main_fixture(monkeypatch, tmp_path)
    argv[argv.index('--hessian') + 1] = 'require'
    # On a CUDA host the CLI checks the bounded-capture release policy before
    # it reaches the streamed source; the dispatcher sets this for real runs.
    for name, value in BOUNDED_CAPTURE_ENV.items():
        monkeypatch.setenv(name, value)

    def build(*args, **kwargs):
        pytest.fail('source construction preceded complete-capture authentication')

    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', build)
    with pytest.raises(FileNotFoundError, match='capture_manifest.json'):
        campaign.main([*argv, '--streaming', '--units', str(tmp_path/'units.json'),
            '--calibration-census', str(tmp_path/'census.json'),
            '--calibration-cache', str(tmp_path/'capture_manifest.json'),
            '--calibration-cache-sha256', 'a'*64,
            '--attention-implementation', 'eager'])
