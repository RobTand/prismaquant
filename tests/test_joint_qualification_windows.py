"""Qualification bounds preserve the canonical per-unit and render checks."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from prismaquant import tessera_joint_aura as bridge


def policy(**changes):
    return dict(schema='prismaquant.joint_anchor_qualification.v1',
                max_capture_resident_bytes=128, max_load_buffer_bytes=10000,
                workspace_reserve_bytes=10000, **changes)


@pytest.mark.parametrize('field,value', [
    ('max_capture_resident_bytes', 0), ('max_load_buffer_bytes', True),
    ('workspace_reserve_bytes', -1), ('schema', 'unknown')])
def test_policy_requires_complete_finite_positive_budgets(field, value):
    config = policy(); config[field] = value
    with pytest.raises(ValueError):
        bridge.normalize_qualification_window(config)


def test_policy_is_detached_and_legacy_none_is_preserved():
    config = policy()
    normalized = bridge.normalize_qualification_window(config)
    assert normalized == config and normalized is not config
    assert bridge.normalize_qualification_window(None) is None
    with pytest.raises(ValueError):
        bridge.normalize_qualification_window({**config, 'extra': 1})


def fixture(tmp_path, monkeypatch, *, fail_cell=False, fail_unit=None):
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc
    from prismaquant import tessera_hessian as th, joint_aura
    names = ['model.layers.0.a', 'model.layers.0.b']
    formats = ['TESSERA_E4M3_K1_R1024', 'TESSERA_E4M3_K1_R768']
    modules = {name: torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16) for name in names}
    cells = {}
    for name in names:
        for fmt in formats:
            path = tmp_path / (name + fmt + '.pt')
            torch.save(modules[name].weight.detach(), path)
            wire = tmp_path / (name + fmt + '.wire')
            blob = (name + fmt).encode()
            wire.write_bytes(blob)
            cells[name, fmt] = dict(render=str(path), render_origin='encoded',
                                    wire=str(wire), record={'blob_bytes': len(blob),
                                    'blob_sha256': hashlib.sha256(blob).hexdigest()},
                                    anchor={'qname': name, 'format_name': fmt})
    capture_path = tmp_path / 'capture.json'
    capture_path.write_text('{}')
    capture = dict(path=str(capture_path), sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest())
    from prismaquant.perturbed_x_cache import activation_cache_filename
    capture_entries = {}
    (tmp_path / 'inputs').mkdir()
    for name in names:
        filename = activation_cache_filename(name)
        entry = tmp_path / 'inputs' / filename
        entry.write_bytes(('canonical-X-and-H:' + name).encode())
        capture_entries[name] = {'path': 'inputs/' + filename,
                                 'sha256': hashlib.sha256(entry.read_bytes()).hexdigest()}
    expected = dict(max_act_rows=4, calibration={'draw': 'same'}, units=names)
    data = SimpleNamespace(payload={'provenance': {'calibration_cache': capture,
             'hessian': {'calibration_identity': expected['calibration']}}},
        inputs={'census': {'path': 'census'}}, census={'model_load_contract': {},
            'attention_implementation': 'eager', 'unit_shapes': {name: [4, 4] for name in names},
            'counts': {name: 4 for name in names}},
        manifest={'identity': {'calibration': expected['calibration']},
                  'identity_sha256': 'j' * 64},
        cells=cells, formats_by_qname={name: formats for name in names})
    events, live, observed = [], [], []
    context = SimpleNamespace(schedule_prefetch=lambda depth: None,
        install=lambda layer, **kw: events.append(('install', layer)),
        unload=lambda layer: events.append(('unload', layer)),
        settle_prefetched_layers=lambda indices: events.append(('settled', tuple(indices))))
    runner = SimpleNamespace(model=torch.nn.Module(), context=context, num_layers=1, prefetch_lookahead=1,
        require_prefetched_residency=True, profile=object(), device='cpu',
        layer_index_for_qname=lambda name: 0)
    monkeypatch.setattr(bridge, '_bound', lambda record, label: tmp_path / 'capture.json')
    monkeypatch.setattr(cc, 'require_capture_contract', lambda *args, **kw:
                        {'identity': expected, 'entries': capture_entries})
    monkeypatch.setattr(cc, 'open_capture_metadata', lambda *args, **kw: object())
    monkeypatch.setattr(cc, 'capture_identity', lambda *args, **kw: expected)
    monkeypatch.setattr(bridge, 'calibrated_maxima', lambda *args: ({}, {}))
    monkeypatch.setattr(bridge, '_live_targets', lambda *args: modules)
    def prefetch(*args, names, release_file_pages=False, **kwargs):
        assert len(names) == 1 and release_file_pages
        assert all(ref() is None for ref in live), 'previous unit capture still owned'
        name, = names
        x, h = torch.ones(4, 4), torch.eye(4)
        live.extend((weakref.ref(x), weakref.ref(h)))
        events.append(('capture', name))
        return ({name: x}, {name: h}, {}, {}), {}
    monkeypatch.setattr(cc, 'prefetch_capture', prefetch)
    monkeypatch.setattr(th, 'activation_source', lambda h, identity: SimpleNamespace(hessians=h))
    monkeypatch.setattr(tc, 'CampaignAnchor', lambda **kw: SimpleNamespace(**kw))
    @contextmanager
    def bound(anchors, **kwargs):
        observed.append(('bind', tuple(anchor.format_name for anchor in anchors)))
        yield object()
    monkeypatch.setattr(tc, 'bind_checkpoint_unit_identity', bound)
    monkeypatch.setattr(joint_aura, 'activation_identity', lambda *args: {'input_global_scale': None})
    def verify(cell, source, rendered, **kwargs):
        assert torch.equal(source, rendered)
        assert kwargs['calibration_source'].hessians
        observed.append(('verify', cell['anchor']['qname'], cell['anchor']['format_name']))
        if fail_cell or cell['anchor']['qname'] == fail_unit:
            raise RuntimeError('intentional verification failure')
        return {'source_weight': {'sha256': 's' * 64},
                'rendered_weight': {'sha256': 'r' * 64},
                'encoding_identity_sha256': 'e' * 64,
                'render_file_sha256': cell['render_file_sha256'],
                'wire_sha256': cell['record']['blob_sha256'],
                'render_origin': cell['render_origin'],
                'render_comparison':
                    bridge.RENDER_COMPARISON_BY_ORIGIN[cell['render_origin']]}
    monkeypatch.setattr(bridge, 'verify_anchor_render', verify)
    return runner, data, capture, events, live, observed


def test_qualification_releases_each_capture_and_window_preserving_roster(tmp_path, monkeypatch):
    runner, data, capture, events, live, observed = fixture(tmp_path, monkeypatch)
    cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=1, qualification_window=policy())
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    assert cache.metadata['render_origins'] == {'encoded': 4, 'synthesized_from_wire': 0}
    assert cache.metadata['render_comparisons'] == {
        'independent_render_vs_wire': 4, 'wire_round_trip_only': 0}
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert all(ref() is None for ref in live)
    assert len([row for row in observed if row[0] == 'bind']) == 2
    assert len([row for row in observed if row[0] == 'verify']) == 4
    assert [row[0] for row in events] == ['install', 'settled', 'capture', 'capture', 'unload']


def test_oversize_capture_refuses_before_any_capture_load(tmp_path, monkeypatch):
    runner, data, capture, events, live, observed = fixture(tmp_path, monkeypatch)
    config = policy(); config['max_capture_resident_bytes'] = 127
    with pytest.raises(ValueError, match='capture.*budget'):
        bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
            file_load_workers=1, qualification_window=config)
    assert not any(row[0] == 'capture' for row in events)


def test_failed_qualification_releases_unit_and_source(tmp_path, monkeypatch):
    runner, data, capture, events, live, observed = fixture(tmp_path, monkeypatch, fail_cell=True)
    with pytest.raises(RuntimeError, match='intentional'):
        bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
            file_load_workers=1, qualification_window=policy())
    assert all(ref() is None for ref in live)
    assert events[-1] == ('unload', 0)


def test_smaller_serialized_budget_plans_more_windows(tmp_path, monkeypatch):
    from pathlib import Path
    runner, data, capture, events, live, observed = fixture(tmp_path, monkeypatch)
    config = policy()
    config['max_load_buffer_bytes'] = max(Path(cell['render']).stat().st_size for cell in data.cells.values())
    cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=2, qualification_window=config)
    windows = cache.metadata['prefetch'][0]['windows']
    assert len(windows) == 4 and all(len(window['keys']) == 1 for window in windows)


def test_wire_read_ahead_reserves_current_and_pending_blobs(tmp_path, monkeypatch):
    runner, data, capture, _events, _live, _observed = fixture(tmp_path, monkeypatch)
    reserved = []
    guard = SimpleNamespace(check=lambda where, reserve_bytes=0:
                            reserved.append((where, reserve_bytes)), snapshot=lambda: {})
    bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=1, qualification_window=policy(), qualification_guard=guard)
    maximum = max(cell['record']['blob_bytes'] for cell in data.cells.values())
    units = [amount for where, amount in reserved
             if where.startswith('before_joint_qualification_unit:')]
    assert len(units) == 2
    assert units == [2 * 128 + 10000 + 10000 + 10000 + 2 * maximum] * 2


def test_prepared_metadata_counts_a_synthesized_rung_apart(tmp_path, monkeypatch):
    """A shard written from its own wire is never counted as independently
    compared, and a receipt that disagrees with its cell refuses."""
    runner, data, capture, _events, _live, _observed = fixture(tmp_path, monkeypatch)
    pair = next(iter(data.cells))
    data.cells[pair]['render_origin'] = 'synthesized_from_wire'
    cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=1, qualification_window=policy())
    assert cache.metadata['render_origins'] == {'encoded': 3, 'synthesized_from_wire': 1}
    assert cache.metadata['render_comparisons'] == {
        'independent_render_vs_wire': 3, 'wire_round_trip_only': 1}
    assert (cache.metadata['verified_cells'][pair]['render_comparison']
            == 'wire_round_trip_only')


def test_qualification_journal_restarts_from_durable_unit(tmp_path, monkeypatch):
    from prismaquant.cost_stage_checkpoint import unit_path
    runner, data, capture, events, _live, observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    journal = tmp_path / 'qualification'
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=journal,
                   qualification_identity={'plan_sha256': 'p' * 64})
    progress = []
    def committed(units, phase, unit=None):
        assert unit_path(journal, unit).is_file(), 'PB progress preceded durable receipt'
        progress.append((units, phase, unit))
    monkeypatch.setattr(bridge, '_pb_commit', committed)
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    assert unit_path(journal, 'model.layers.0.a').is_file()
    assert not unit_path(journal, 'model.layers.0.b').exists()
    assert len([row for row in observed if row[0] == 'verify']) == 3
    assert progress == [(1, 'qualification', 'model.layers.0.a')]
    events.clear()
    monkeypatch.setattr(bridge, 'verify_anchor_render', lambda cell, source, rendered, **kw:
        {'source_weight': {'sha256': 's' * 64},
         'rendered_weight': {'sha256': 'r' * 64},
         'encoding_identity_sha256': 'e' * 64,
         'render_file_sha256': cell['render_file_sha256'],
         'wire_sha256': cell['record']['blob_sha256'],
         'render_origin': cell['render_origin'],
         'render_comparison': 'independent_render_vs_wire'})
    cache = bridge.prepare_cache(runner, data, **options, qualification_resume=True)
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    assert [row for row in events if row[0] == 'capture'] == [
        ('capture', 'model.layers.0.b')]
    assert len(cache.metadata['prefetch'][0]['windows']) == 4
    assert unit_path(journal, 'model.layers.0.b').is_file()
    assert progress == [(1, 'qualification', 'model.layers.0.a'),
                        (2, 'qualification', 'model.layers.0.b')]


def test_replay_reads_only_completed_capture_entries_before_resume(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import activation_cache_filename
    runner, data, capture, _events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=tmp_path/'qualification',
                   qualification_identity={'plan_sha256': 'p' * 64})
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    unfinished = tmp_path / 'inputs' / activation_cache_filename('model.layers.0.b')
    original_hash = bridge._qualification_file_sha
    seen = []
    def checked(path):
        assert Path(path) != unfinished, 'replay eagerly reread uncommitted X/H'
        seen.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(bridge, '_qualification_file_sha', checked)
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True)
    assert tmp_path / 'inputs' / activation_cache_filename('model.layers.0.a') in seen


@pytest.mark.parametrize('changed', ['plan', 'source', 'reader', 'implementation',
                                     'capture', 'wire', 'render', 'symlink_wire',
                                     'symlink_render'])
def test_qualification_replay_refuses_changed_upstream(tmp_path, monkeypatch, changed):
    runner, data, capture, events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=tmp_path/'qualification',
                   qualification_identity={'plan_sha256': 'p' * 64})
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    events.clear()
    if changed in ('plan', 'source', 'reader', 'implementation'):
        key = {'plan': 'plan_sha256', 'source': 'source_model_identity',
               'reader': 'reader_identity', 'implementation': 'implementation_sha256'}[changed]
        options['qualification_identity'] = {**options['qualification_identity'], key: 'q' * 64}
    elif changed == 'capture':
        from prismaquant.perturbed_x_cache import activation_cache_filename
        (tmp_path / 'inputs' / activation_cache_filename('model.layers.0.a')).write_bytes(b'changed H')
    elif changed.startswith('symlink_'):
        key = (data.formats_by_qname['model.layers.0.a'][0])
        original = Path(data.cells['model.layers.0.a', key][changed.removeprefix('symlink_')])
        target = original.with_name(original.name + '.saved')
        original.rename(target)
        original.symlink_to(target)
    else:
        Path(data.cells['model.layers.0.a', next(iter(data.formats_by_qname.values()))[0]][changed]).write_bytes(b'changed')
    with pytest.raises((ValueError, RuntimeError), match='mismatch|changed|regular'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True)
    assert not any(row[0] == 'capture' for row in events)
