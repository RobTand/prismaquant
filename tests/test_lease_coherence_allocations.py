"""Coherence must not collect the model heap just to snapshot unrelated imports."""
import gc
import sys
from types import SimpleNamespace

import pytest

from prismaquant import staged_lease as lease


def _modules(monkeypatch, root):
    modules = {k: v for k, v in sys.modules.items()
               if k != 'prismabuild' and not k.startswith('prismabuild.')}
    modules.update({f'unrelated_loaded_module_{i}': None for i in range(6000)})
    modules['prismabuild'] = SimpleNamespace(__file__=str(root / '__init__.py'))
    monkeypatch.setattr(sys, 'modules', modules)
    return modules


def test_coherence_does_not_trigger_heap_collection_for_unrelated_imports(tmp_path, monkeypatch):
    _modules(monkeypatch, tmp_path)
    was_enabled, thresholds = gc.isenabled(), gc.get_threshold()
    collections = []
    def observe(phase, info):
        if phase == 'start':
            collections.append(info['generation'])
    try:
        gc.enable()
        gc.collect()
        gc.set_threshold(1000, 1000000, 1000000)
        gc.callbacks.append(observe)
        lease._check_package_coherence(tmp_path)
    finally:
        gc.callbacks.remove(observe)
        gc.set_threshold(*thresholds)
        if not was_enabled:
            gc.disable()
    assert collections == [], 'coherence snapshot caused unrelated model-heap collection'


def test_new_divergent_module_still_refuses_on_next_check(tmp_path, monkeypatch):
    modules = _modules(monkeypatch, tmp_path / 'expected')
    lease._check_package_coherence(tmp_path / 'expected')
    modules['prismabuild.new_module'] = SimpleNamespace(__file__=str(tmp_path / 'elsewhere.py'))
    with pytest.raises(lease.LeaseRefused, match='resolves elsewhere'):
        lease._check_package_coherence(tmp_path / 'expected')


def test_symlink_target_change_is_rechecked(tmp_path, monkeypatch):
    expected = tmp_path / 'expected'
    expected.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = expected / 'linked'
    link.symlink_to(expected, target_is_directory=True)
    modules = _modules(monkeypatch, expected)
    modules['prismabuild.extra'] = SimpleNamespace(__file__=str(link / 'extra.py'))
    lease._check_package_coherence(expected)
    link.unlink()
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(lease.LeaseRefused, match='resolves elsewhere'):
        lease._check_package_coherence(expected)
