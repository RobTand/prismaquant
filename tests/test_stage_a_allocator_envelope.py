"""Stage A enforces its explicit allocation ceiling before any backend work."""
import json
from types import SimpleNamespace

import pytest
import torch

from tests.test_joint_cost_quantum_runtime import _stage_a_run_stub, _prefetch_budget


def test_explicit_envelope_precedes_backend_and_is_in_receipts(tmp_path, monkeypatch):
    import prismaquant.joint_projection_backend as backend
    calls = []
    _, run = _stage_a_run_stub(tmp_path, monkeypatch, tmp_path / 'out')
    monkeypatch.setenv('PRISMAQUANT_MAX_GPU_MEM_GB', '72')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(total_memory=128 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction',
                        lambda fraction, device: calls.append((fraction, device)))
    def prewarm(*args, **kwargs):
        assert calls == [(72 / 128, 2)], 'backend started before allocator enforcement'
        return SimpleNamespace(identity=None)
    monkeypatch.setattr(backend, 'prewarm_projection_backend', prewarm)
    result, results, counters = run(plan_budget=_prefetch_budget())
    stamp = result['device_envelope']
    assert stamp['enforced'] is True
    assert stamp['device_envelope_bytes'] == 72 * 1024 ** 3
    assert stamp['allocator_device_index'] == 2
    assert results['device_envelope'] == counters['device_envelope'] == stamp
    receipt = json.loads((tmp_path / 'out/layer-quanta/adjoint/adjoint-capture.json').read_text())
    assert receipt['device_envelope'] == stamp


@pytest.mark.parametrize('raw', ['0', '-1', 'nan', 'inf', 'off'])
def test_explicit_invalid_envelope_refuses_before_backend(tmp_path, monkeypatch, raw):
    import prismaquant.joint_projection_backend as backend
    _, run = _stage_a_run_stub(tmp_path, monkeypatch, tmp_path / 'out')
    monkeypatch.setenv('PRISMAQUANT_MAX_GPU_MEM_GB', raw)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid explicit envelope reached backend')
    monkeypatch.setattr(backend, 'prewarm_projection_backend', forbidden)
    with pytest.raises((ValueError, RuntimeError), match='envelope'):
        run(plan_budget=_prefetch_budget())
