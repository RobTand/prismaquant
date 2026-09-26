"""Qualification bounds preserve the canonical per-unit and render checks."""
from contextlib import contextmanager
import hashlib
import json
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


def fixture(tmp_path, monkeypatch, *, fail_cell=False, fail_unit=None,
            layer_of=(0, 0)):
    """A two-unit joint preparation.

    ``layer_of`` says which layer each unit belongs to, in unit order, so a
    case can put a fully-qualified unit in an earlier layer than one still to
    qualify. The default keeps both units in layer 0, which is what the
    single-layer cases below hold.
    """
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc
    from prismaquant import tessera_hessian as th, joint_aura
    names = [f'model.layers.{layer}.{suffix}'
             for layer, suffix in zip(layer_of, ('a', 'b'))]
    assert len(names) == len(set(names)), 'layer_of must name two distinct units'
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
        settle_prefetched_layers=lambda indices, *, retry_availability=False: events.append(('settled', tuple(indices))))
    runner = SimpleNamespace(model=torch.nn.Module(), context=context,
        num_layers=max(layer_of) + 1, prefetch_lookahead=1,
        source_layers=tuple(range(max(layer_of) + 1)),
        require_prefetched_residency=True, profile=object(), device='cpu',
        layer_index_for_qname=lambda name: layer_of[names.index(name)])
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
    """The serialized budget alone splits the windows, at one loader.

    A quantum's width is set by the two byte budgets, not by the loader count
    (#693), so this needs one loader. It asked for two, which made it assert on
    the shard's CPU count: ``_window_limits`` refuses more loaders than the
    process's CPU affinity holds, and a one-CPU shard failed (PQ #1032). Both
    plans run at one loader, so the default budget's two keys per window
    against the smaller budget's one key per window is the budget's doing.
    """
    def key_counts(root, *, smaller):
        root.mkdir()
        runner, data, capture, _events, _live, _observed = fixture(root, monkeypatch)
        config = policy()
        if smaller:
            config['max_load_buffer_bytes'] = max(
                Path(cell['render']).stat().st_size for cell in data.cells.values())
        cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
            file_load_workers=1, qualification_window=config)
        return [len(window['keys']) for window in cache.metadata['prefetch'][0]['windows']]

    assert key_counts(tmp_path / 'default', smaller=False) == [2, 2]
    assert key_counts(tmp_path / 'smaller', smaller=True) == [1, 1, 1, 1]


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
    # The reservation the guard is handed is the CPU side plus the device side,
    # and this stub guard is not split-aware, so it sees their sum:
    #   CPU    capture 128 (the payload) + render 10000 (PWC backing storages)
    #          + load buffers 10000 + wire read-ahead 2 * maximum
    #   device capture 128 (X/H on the device) + render 10000 (the copy the
    #          verifier receives) + workspace 10000
    # The render is the term that appears on BOTH sides: its backing storages
    # are the cgroup's and the tensor handed to the verifier is a copy of them
    # on the device, so the old single number was counting one of the two.
    assert units == [2 * 128 + 4 * 10000 + 2 * maximum] * 2


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
    # One window per unit, holding that unit's whole render set: the loader
    # count no longer decides a quantum's width, so file_load_workers=1 no
    # longer means one key per window (#693).
    windows = cache.metadata['prefetch'][0]['windows']
    assert [len(window['keys']) for window in windows] == [2, 2]
    assert [window['unit'] for window in windows] == [
        'model.layers.0.a', 'model.layers.0.b']
    assert unit_path(journal, 'model.layers.0.b').is_file()
    assert progress == [(1, 'qualification', 'model.layers.0.a'),
                        (2, 'qualification', 'model.layers.0.b')]


def test_qualification_journal_binds_cells_without_copying_full_roster(tmp_path, monkeypatch):
    runner, data, capture, _events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    journal = tmp_path / 'qualification'
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=journal,
                   qualification_identity={'plan_sha256': 'p' * 64})
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    from json import loads
    identity = loads((journal / 'manifest.json').read_text())['identity']
    assert 'cells' not in identity, 'journal duplicated the complete input records'
    assert identity['cell_count'] == len(data.cells)
    assert len(identity['cells_sha256']) == 64
    first = ('model.layers.0.a', data.formats_by_qname['model.layers.0.a'][0])
    data.cells[first]['anchor']['format_name'] = 'changed'
    with pytest.raises((RuntimeError, ValueError), match='mismatch'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True)


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


def _passing_verify(cell, source, rendered, **kwargs):
    return {'source_weight': {'sha256': 's' * 64},
            'rendered_weight': {'sha256': 'r' * 64},
            'encoding_identity_sha256': 'e' * 64,
            'render_file_sha256': cell['render_file_sha256'],
            'wire_sha256': cell['record']['blob_sha256'],
            'render_origin': cell['render_origin'],
            'render_comparison': bridge.RENDER_COMPARISON_BY_ORIGIN[cell['render_origin']]}


def _first_pass(tmp_path, monkeypatch, *, fail_unit, layer_of=(0, 0)):
    """Run once and leave whatever journal that run wrote.

    ``fail_unit=None`` completes the pass, so the journal holds every unit; a
    unit name stops the pass on that unit, so the journal holds the units
    before it.
    """
    runner, data, capture, events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit=fail_unit, layer_of=layer_of)
    journal = tmp_path / 'qualification'
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=journal,
                   qualification_identity={'plan_sha256': 'p' * 64})
    if fail_unit is None:
        bridge.prepare_cache(runner, data, **options)
    else:
        with pytest.raises(RuntimeError, match='intentional verification failure'):
            bridge.prepare_cache(runner, data, **options)
    events.clear()
    return runner, data, options, journal, events


def _sealed_over(journal, data, roster, *, phase_of=None, identity_sha256='JOURNAL'):
    """The frontier a submission seals over ``roster`` (#607).

    Bytes and per-unit cells are the runtime's own view of them, which is what
    the submission's producer derives from the same checkpoint.
    """
    from prismaquant import joint_replay_frontier as replay

    names = sorted(roster)
    cells = {name: [fmt for fmt in data.formats_by_qname[name] if fmt != 'BF16']
             for name in names}
    starts = ({name: replay.replay_phase_name(0) for name in names}
              if phase_of is None else dict(phase_of))
    if identity_sha256 == 'JOURNAL':
        identity_sha256 = json.loads(
            (journal / 'manifest.json').read_text())['identity_sha256']
    return replay.seal_frontier(names, cells, phase_start_units=starts,
                                journal_identity_sha256=identity_sha256)


def test_a_sealed_replay_frontier_announces_only_the_phases_it_reads(
    tmp_path, monkeypatch,
):
    runner, data, options, journal, events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    sealed = _sealed_over(journal, data, ['model.layers.0.a'])
    from prismaquant import joint_replay_frontier as replay
    starts = {**sealed[replay.PHASE_START_UNITS_KEY],
              'model.layers.0.b': 'layer-0-part-0'}
    monkeypatch.setattr(bridge, 'verify_anchor_render', _passing_verify)
    progress = []
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        progress.append((units, phase, unit)))
    cache = bridge.prepare_cache(runner, data, **options, qualification_resume=True,
                                 prewarm_phase_starts=starts, sealed_replay=sealed,
                                 prewarm_phases=('head', 'replay-0000',
                                                 'layer-0-part-0'))
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    # The journal's unit is re-authenticated, not re-qualified.
    assert [row for row in events if row[0] == 'capture'] == [
        ('capture', 'model.layers.0.b')]
    # Its replay phase is announced inside the replay, once its own bytes are
    # on the end of a completed read. The walk then names only the unit it
    # still has to qualify -- never the replayed unit's own phase again.
    assert progress == [(0, 'replay-0000', 'model.layers.0.a'),
                        (1, 'replay-0000', 'model.layers.0.a'),
                        (1, 'layer-0-part-0', 'model.layers.0.b'),
                        (2, 'layer-0-part-0', 'model.layers.0.b')]


def test_a_sealed_replay_frontier_refuses_a_journal_that_moved(tmp_path, monkeypatch):
    from prismaquant import joint_replay_frontier as replay

    runner, data, options, journal, events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    sealed = _sealed_over(journal, data, ['model.layers.0.a'],
                          identity_sha256='e' * 64)
    starts = {**sealed[replay.PHASE_START_UNITS_KEY],
              'model.layers.0.b': 'layer-0-part-0'}
    roster = sealed[replay.ROSTER_KEY]
    with pytest.raises(RuntimeError, match='journal identity changed'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True,
                             prewarm_phase_starts=starts, sealed_replay=sealed,
                             prewarm_phases=('head', 'replay-0000', 'layer-0-part-0'))
    # Refused before the first replayed byte, not after reporting a prefix.
    assert not any(row[0] == 'capture' for row in events)


def test_sealed_phases_without_a_replay_frontier_are_refused(tmp_path, monkeypatch):
    runner, data, options, journal, events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    from prismaquant import joint_replay_frontier as replay
    sealed = _sealed_over(journal, data, ['model.layers.0.a'])
    starts = {**sealed[replay.PHASE_START_UNITS_KEY],
              'model.layers.0.b': 'layer-0-part-0'}
    with pytest.raises(ValueError, match='sealed replay frontier'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True,
                             prewarm_phase_starts=starts,
                             prewarm_phases=('head', 'replay-0000', 'layer-0-part-0'))
    assert not any(row[0] == 'capture' for row in events)


def test_a_fully_completed_resume_verifies_and_publishes(tmp_path, monkeypatch):
    """Every unit is journaled: nothing is re-qualified and nothing errors."""
    from prismaquant import joint_replay_frontier as replay

    runner, data, options, journal, events = _first_pass(
        tmp_path, monkeypatch, fail_unit=None)
    sealed = _sealed_over(journal, data, data.formats_by_qname)
    progress = []
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        progress.append((units, phase, unit)))
    cache = bridge.prepare_cache(
        runner, data, **options, qualification_resume=True,
        prewarm_phase_starts=sealed[replay.PHASE_START_UNITS_KEY],
        sealed_replay=sealed,
        prewarm_phases=('head', 'replay-0000', 'layer-0-part-0'))
    # The replay re-reads and verifies every unit, so the walk has nothing left
    # to qualify: no capture is loaded again and no unit draws a phase of its
    # own.
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    assert not any(row[0] == 'capture' for row in events)
    # One window per unit, holding that unit's whole render set: the loader
    # count no longer decides a quantum's width, so file_load_workers=1 no
    # longer means one key per window (#693).
    windows = cache.metadata['prefetch'][0]['windows']
    assert [len(window['keys']) for window in windows] == [2, 2]
    assert [window['unit'] for window in windows] == [
        'model.layers.0.a', 'model.layers.0.b']
    # One announcement per part plus one count update per durable unit; both
    # units sit in the same part, so only the first draws a transition. The
    # layer's own phase is still announced (unitless) before the walk reads the
    # source extents it declares.
    assert progress == [(0, 'replay-0000', 'model.layers.0.a'),
                        (1, 'replay-0000', 'model.layers.0.a'),
                        (2, 'replay-0000', 'model.layers.0.b'),
                        (2, 'layer-0-part-0', None)]


def test_the_counter_continues_from_the_head_walk_rather_than_restarting(
    tmp_path, monkeypatch,
):
    """The head walk reports the units it resolved, so replay and the layer
    walk continue from that count. Restarting at zero here would hand PB a
    counter that goes backwards, which renews no allowance (#678).
    """
    from prismaquant import joint_replay_frontier as replay

    runner, data, options, journal, _events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    sealed = _sealed_over(journal, data, ['model.layers.0.a'])
    starts = {**sealed[replay.PHASE_START_UNITS_KEY],
              'model.layers.0.b': 'layer-0-part-0'}
    monkeypatch.setattr(bridge, 'verify_anchor_render', _passing_verify)
    progress = []
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        progress.append((units, phase, unit)))
    cache = bridge.prepare_cache(runner, data, **options, qualification_resume=True,
                                 prewarm_phase_starts=starts, sealed_replay=sealed,
                                 prewarm_phases=('head', 'replay-0000',
                                                 'layer-0-part-0'),
                                 progress_base=2)
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    assert progress == [(2, 'replay-0000', 'model.layers.0.a'),
                        (3, 'replay-0000', 'model.layers.0.a'),
                        (3, 'layer-0-part-0', 'model.layers.0.b'),
                        (4, 'layer-0-part-0', 'model.layers.0.b')]
    counts = [units for units, _, _ in progress]
    assert counts == sorted(counts) and counts[0] >= 2


def test_an_empty_sealed_frontier_is_accepted_and_a_later_unit_is_refused(
    tmp_path, monkeypatch,
):
    """A resume submitted before anything was journaled seals an empty roster."""
    runner, data, options, journal, events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.0.a')
    empty = _sealed_over(journal, data, [], identity_sha256=None)
    starts = {name: 'layer-0-part-0' for name in data.formats_by_qname}
    monkeypatch.setattr(bridge, 'verify_anchor_render', _passing_verify)
    progress = []
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        progress.append((units, phase, unit)))
    cache = bridge.prepare_cache(
        runner, data, **options, qualification_resume=True,
        prewarm_phase_starts=starts, sealed_replay=empty,
        prewarm_phases=('head', 'layer-0-part-0'))
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    # Nothing was replayed, so the walk is the fresh order.
    assert progress == [(0, 'layer-0-part-0', 'model.layers.0.a'),
                        (1, 'layer-0-part-0', 'model.layers.0.a'),
                        (2, 'layer-0-part-0', 'model.layers.0.b')]

    # The same frontier over a journal that has since gained a unit is refused
    # before that unit is announced: the empty roster bound nothing on disk.
    (tmp_path / 'second').mkdir()
    runner, data, options, journal, events = _first_pass(
        tmp_path / 'second', monkeypatch, fail_unit='model.layers.0.b')
    with pytest.raises(RuntimeError, match='no longer holds the sealed replay roster'):
        bridge.prepare_cache(
            runner, data, **options, qualification_resume=True,
            prewarm_phase_starts=starts, sealed_replay=empty,
            prewarm_phases=('head', 'layer-0-part-0'))
    assert not any(row[0] == 'capture' for row in events)


def test_a_source_only_layer_phase_is_announced_before_its_reads(
    tmp_path, monkeypatch,
):
    """A resumed layer with no unit left still reads its source extents.

    Announcing it late would hand PB a prefix one phase too far: the bytes
    ``layer-0-part-0`` declares are read by the prefetch window this walk opens
    before ``install(0)``, and they must not be released before that read.
    """
    runner, data, options, journal, _events = _first_pass(
        tmp_path, monkeypatch, fail_unit='model.layers.1.b', layer_of=(0, 1))

    order = []
    runner.context.install = lambda layer, **kw: order.append(('install', layer))
    runner.context.schedule_prefetch = lambda depth: order.append(('prefetch', depth))
    runner.context.unload = lambda layer: order.append(('unload', layer))
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        order.append(('announce', units, phase, unit)))
    monkeypatch.setattr(bridge, 'verify_anchor_render', _passing_verify)

    sealed = _sealed_over(journal, data, ['model.layers.0.a'])
    from prismaquant import joint_replay_frontier as replay
    starts = {**sealed[replay.PHASE_START_UNITS_KEY],
              'model.layers.1.b': 'layer-1-part-0'}
    bridge.prepare_cache(
        runner, data, **options, qualification_resume=True,
        prewarm_phase_starts=starts, sealed_replay=sealed,
        prewarm_phases=('head', 'replay-0000', 'layer-0-part-0', 'layer-1-part-0'))

    # Layer 0 has no unit left to qualify, so its phase carries only source
    # extents: still announced, before the reads it describes, and with no
    # committed unit invented for it.
    assert order == [
        ('announce', 0, 'replay-0000', 'model.layers.0.a'),
        ('announce', 1, 'replay-0000', 'model.layers.0.a'),
        ('announce', 1, 'layer-0-part-0', None),
        ('prefetch', 0),
        ('prefetch', 1),
        ('install', 0),
        ('prefetch', 1),
        ('unload', 0),
        ('announce', 1, 'layer-1-part-0', 'model.layers.1.b'),
        ('install', 1),
        ('prefetch', 2),
        ('announce', 2, 'layer-1-part-0', 'model.layers.1.b'),
        ('unload', 1),
    ]


def test_a_fully_completed_resume_announces_every_declared_source_phase(
    tmp_path, monkeypatch,
):
    """Every layer the walk still reads announces its own phase, in order."""
    runner, data, options, journal, _events = _first_pass(
        tmp_path, monkeypatch, fail_unit=None, layer_of=(0, 1))

    order = []
    runner.context.install = lambda layer, **kw: order.append(('install', layer))
    runner.context.schedule_prefetch = lambda depth: order.append(('prefetch', depth))
    runner.context.unload = lambda layer: order.append(('unload', layer))
    monkeypatch.setattr(bridge, '_pb_commit',
                        lambda units, phase, unit=None:
                        order.append(('announce', units, phase, unit)))

    sealed = _sealed_over(journal, data, data.formats_by_qname)
    from prismaquant import joint_replay_frontier as replay
    bridge.prepare_cache(
        runner, data, **options, qualification_resume=True,
        prewarm_phase_starts=sealed[replay.PHASE_START_UNITS_KEY],
        sealed_replay=sealed,
        prewarm_phases=('head', 'replay-0000', 'layer-0-part-0'))

    assert order == [
        ('announce', 0, 'replay-0000', 'model.layers.0.a'),
        ('announce', 1, 'replay-0000', 'model.layers.0.a'),
        ('announce', 2, 'replay-0000', 'model.layers.1.b'),
        ('announce', 2, 'layer-0-part-0', None),
        ('prefetch', 0),
        ('prefetch', 1),
        ('install', 0),
        ('prefetch', 1),
        ('unload', 0),
        ('install', 1),
        ('prefetch', 2),
        ('unload', 1),
    ]


def _reuse_record(recorded='c' * 64, current='d' * 64):
    """The exact record `resolve_encoder_source_reuse` returns for a substitution."""
    return {'schema': bridge.HISTORICAL_ENCODER_REUSE_SCHEMA,
            'status': bridge.ENCODER_REUSE_STATUS,
            'recorded_encoder_source_sha256': recorded,
            'observed_current_encoder_source_sha256': current,
            'allowlist_entry': {'encoder_source_sha256': recorded,
                                'reason': 'the historical package that priced this checkpoint',
                                'evidence': '/dev/null',
                                'recorded_unix': 1789625518.0,
                                'recorded_by': 'test'}}


def _verify_with_reuse(reuse, inner):
    """Wrap the fixture's own verifier so its failure injection still fires."""
    def verify(cell, source, rendered, **kwargs):
        return {**inner(cell, source, rendered, **kwargs),
                'current_encoding_identity_sha256': 'f' * 64,
                'encoder_source_reuse': reuse}
    return verify


def test_a_replay_accepts_the_reuse_fields_its_own_writer_emits(tmp_path, monkeypatch):
    """The receipt grammar is required-plus-optional, not set equality.

    ``_verify_cell`` adds ``current_encoding_identity_sha256`` and
    ``encoder_source_reuse`` whenever the encoder source seal was substituted.
    The replay compared ``set(record) == required``, so under an encoder-reuse
    allowlist every receipt the writer durably journalled was refused on the
    next resume -- the GLM-5.3-Flash complete-512 campaign lost 19,442 units
    to it on 2026-09-17.
    """
    reuse = _reuse_record()
    runner, data, capture, _events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    data.encoder_source_reuse = reuse
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=tmp_path/'qualification',
                   qualification_identity={'plan_sha256': 'p' * 64})
    monkeypatch.setattr(bridge, 'verify_anchor_render',
                        _verify_with_reuse(reuse, bridge.verify_anchor_render))
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    # The first pass durably journalled model.layers.0.a under the reuse
    # fields. The resume replays that receipt and verifies only the unit the
    # injected failure left unfinished, so the verifier stops failing here --
    # exactly as the neighbouring completed-resume test does.
    monkeypatch.setattr(bridge, 'verify_anchor_render',
                        _verify_with_reuse(reuse, _passing_verify))
    cache = bridge.prepare_cache(runner, data, **options, qualification_resume=True)
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    first = next(iter(cache.metadata['verified_cells'].values()))
    assert first['encoder_source_reuse'] == reuse


@pytest.mark.parametrize('damage', ['other_allowlist', 'no_reuse_now', 'half_a_pair',
                                    'unknown_field'])
def test_a_replay_binds_the_journalled_reuse_to_this_run(tmp_path, monkeypatch, damage):
    """Optional does not mean unchecked: the pair is bound, not merely allowed."""
    reuse = _reuse_record()
    runner, data, capture, _events, _live, _observed = fixture(
        tmp_path, monkeypatch, fail_unit='model.layers.0.b')
    data.encoder_source_reuse = reuse
    options = dict(capture=capture, max_render_bytes=10000, file_load_workers=1,
                   qualification_window=policy(), qualification_journal=tmp_path/'qualification',
                   qualification_identity={'plan_sha256': 'p' * 64})
    base = _verify_with_reuse(reuse, bridge.verify_anchor_render)
    if damage == 'half_a_pair':
        def verify(cell, source, rendered, **kwargs):
            row = base(cell, source, rendered, **kwargs)
            row.pop('current_encoding_identity_sha256')
            return row
    elif damage == 'unknown_field':
        def verify(cell, source, rendered, **kwargs):
            return {**base(cell, source, rendered, **kwargs),
                    'a_field_no_writer_emits': 1}
    else:
        verify = base
    monkeypatch.setattr(bridge, 'verify_anchor_render', verify)
    with pytest.raises(RuntimeError, match='intentional verification failure'):
        bridge.prepare_cache(runner, data, **options)
    monkeypatch.setattr(bridge, 'verify_anchor_render',
                        _verify_with_reuse(reuse, _passing_verify))
    if damage == 'other_allowlist':
        data.encoder_source_reuse = _reuse_record(recorded='e' * 64)
    elif damage == 'no_reuse_now':
        data.encoder_source_reuse = None
    with pytest.raises(ValueError, match='incomplete qualification receipt|half a reuse receipt|'
                                         'journalled encoder reuse'):
        bridge.prepare_cache(runner, data, **options, qualification_resume=True)


def test_a_scoped_source_walks_only_its_own_layers(tmp_path, monkeypatch):
    """A source scope (PQ #1338) reads only ``runner.source_layers``.

    GLM's MTP layer is layer 45 of a 46-slot index whose layers 0..44 the
    scope cannot read: installing, prefetching or settling any of them would
    read a layer the scope does not hold.
    """
    runner, data, capture, events, _live, observed = fixture(
        tmp_path, monkeypatch, layer_of=(2, 2))
    runner.source_layers = (2,)
    scheduled = []
    runner.context.schedule_prefetch = lambda layer: scheduled.append(layer)
    cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=1, qualification_window=policy())
    assert set(cache.metadata['verified_cells']) == set(data.cells)
    assert [event for event in events if event[0] == 'install'] == [('install', 2)]
    assert [event for event in events if event[0] == 'unload'] == [('unload', 2)]
    # Past the last index a schedule is the no-op the body walk has always
    # issued; no in-range layer outside the scope is ever scheduled.
    assert [layer for layer in scheduled if layer < runner.num_layers] == [2]
    assert all(set(event[1]) <= {2} for event in events if event[0] == 'settled')
