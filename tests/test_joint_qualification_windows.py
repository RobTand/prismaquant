"""Qualification bounds preserve the canonical per-unit and render checks."""
from contextlib import contextmanager
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


def fixture(tmp_path, monkeypatch, *, fail_cell=False):
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
            cells[name, fmt] = dict(render=str(path), anchor={'qname': name, 'format_name': fmt})
    capture = dict(path=str(tmp_path / 'capture.json'), sha256='a' * 64)
    expected = dict(max_act_rows=4, calibration={'draw': 'same'}, units=names)
    data = SimpleNamespace(payload={'provenance': {'calibration_cache': capture,
             'hessian': {'calibration_identity': expected['calibration']}}},
        inputs={'census': {'path': 'census'}}, census={'model_load_contract': {},
            'attention_implementation': 'eager', 'unit_shapes': {name: [4, 4] for name in names},
            'counts': {name: 4 for name in names}},
        manifest={'identity': {'calibration': expected['calibration']}},
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
    monkeypatch.setattr(cc, 'require_capture_contract', lambda *args, **kw: {'identity': expected})
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
        if fail_cell:
            raise RuntimeError('intentional verification failure')
        return {'render_file_sha256': cell['render_file_sha256']}
    monkeypatch.setattr(bridge, 'verify_anchor_render', verify)
    return runner, data, capture, events, live, observed


def test_qualification_releases_each_capture_and_window_preserving_roster(tmp_path, monkeypatch):
    runner, data, capture, events, live, observed = fixture(tmp_path, monkeypatch)
    cache = bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=10000,
        file_load_workers=1, qualification_window=policy())
    assert set(cache.metadata['verified_cells']) == set(data.cells)
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
