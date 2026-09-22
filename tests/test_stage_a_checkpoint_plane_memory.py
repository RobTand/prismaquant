"""Full capture must not retain an unbudgeted cotangent tensor plane."""
import weakref

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from test_joint_cost_quantum_runtime import _execution, _stage_a
from test_layer_major_boundary_capture import draw
from test_streamed_cost_checkpoints import _model_identity


def _capture(tmp_path, monkeypatch):
    runner, _ = _stage_a(tmp_path, monkeypatch)
    runner.context.settle_prefetched_layers = lambda layers, **kwargs: None
    return stage_a.run_adjoint_capture_core(
        runner, draw(), execution=_execution(tmp_path),
        output_root=tmp_path / 'capture', stride=1,
        source_model_identity=_model_identity('joint-source'),
        unit_roster_sha256='a' * 64, plan_sha256='b' * 64,
        prepared_sha256='c' * 64, read_manifest_sha256='d' * 64,
        implementation_sha256='e' * 64)


def test_capture_checkpoint_planes_retain_descriptors_only(tmp_path, monkeypatch):
    original = stage_a.write_checkpoint_with_snapshot
    planes = []

    def checked(*args, plane, **kwargs):
        assert not any(isinstance(value, torch.Tensor) for value in plane.values()), (
            'checkpoint plane retains all probe/batch tensors outside resident budget')
        planes.append(len(plane))
        return original(*args, plane=plane, **kwargs)

    monkeypatch.setattr(stage_a, 'write_checkpoint_with_snapshot', checked)
    receipt = _capture(tmp_path, monkeypatch)
    assert planes == [20, 20]
    assert receipt['status'] == 'complete'


def test_reverse_roll_releases_each_cpu_result_after_write(tmp_path, monkeypatch):
    original = stage_a.render_free_layer_roll
    peaks = []

    def checked(*args, roll, **kwargs):
        refs = []

        def observed(tensor, batch_index, probe_index):
            refs.append(weakref.ref(tensor))
            roll(tensor, batch_index, probe_index)
            live = sum(ref() is not None for ref in refs)
            peaks.append(live)
            assert live == 1, 'reverse roll retains earlier CPU cotangents'

        return original(*args, roll=observed, **kwargs)

    monkeypatch.setattr(stage_a, 'render_free_layer_roll', checked)
    _capture(tmp_path, monkeypatch)
    assert peaks and max(peaks) == 1


def test_descriptor_checkpoint_matches_tensor_bytes_with_bounded_windows(
        tmp_path, monkeypatch):
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.joint_adjoint_checkpoints import write_adjoint_checkpoint
    from test_checkpoint_artifact_budget import _owner, _session

    owner = _owner(tmp_path / 'entries', cap=3 * 4096, n_probes=4)
    tensors = {(probe, batch): torch.arange(1024, dtype=torch.float32) + probe + batch
               for probe in range(4) for batch in range(17)}
    references = {key: owner.write(tensor, probe_index=key[0], batch_index=key[1],
                                   boundary_index=5)
                  for key, tensor in tensors.items()}
    legacy = write_adjoint_checkpoint(
        tmp_path / 'legacy', boundary=5, session=_session(), cotangents=tensors,
        shared_adjoint={}, shared_pass={})
    windows = []
    original = owner.prefetch

    from contextlib import contextmanager
    @contextmanager
    def watched(entries):
        windows.append(len(entries))
        with original(entries) as window:
            yield window

    monkeypatch.setattr(owner, 'prefetch', watched)
    record = write_adjoint_checkpoint(
        tmp_path / 'streamed', boundary=5, session=_session(), cotangents=references,
        shared_adjoint={}, shared_pass={}, owner=owner)
    assert windows == ([2] * 8 + [1]) * 4
    assert owner.telemetry['peak_resident_tensor_bytes'] <= 3 * 4096
    assert owner.telemetry['resident_tensor_bytes'] == 0
    assert owner.telemetry['read_tensor_bytes'] == 68 * 4096
    assert [entry['sha256'] for entry in record['activation_entries']] == [
        entry['sha256'] for entry in legacy['activation_entries']]
    assert owner.checkpoint_commitment(record['cotangent_sha256'])['actual_bytes'] > 0


def test_descriptor_budget_refuses_before_any_read(tmp_path, monkeypatch):
    from prismaquant.joint_adjoint_checkpoints import write_adjoint_checkpoint
    from test_checkpoint_artifact_budget import _owner, _session

    owner = _owner(tmp_path / 'entries', disk=100000)
    ref = owner.write(torch.zeros(1024), batch_index=0, boundary_index=5,
                      probe_index=0)
    monkeypatch.setattr(owner, 'prefetch', lambda _: pytest.fail('read before admission'))
    with pytest.raises(RuntimeError, match='budget exceeded'):
        write_adjoint_checkpoint(
            tmp_path / 'streamed', boundary=5, session=_session(),
            cotangents={(0, 0): ref}, shared_adjoint={}, shared_pass={}, owner=owner)
    assert not (tmp_path / 'streamed' / 'checkpoints').exists()


def test_descriptor_write_failure_releases_lease_window_and_keeps_budget(
        tmp_path, monkeypatch):
    from prismaquant import joint_adjoint_checkpoints as checkpoints
    from test_checkpoint_artifact_budget import _owner, _session

    owner = _owner(tmp_path / 'entries')
    ref = owner.write(torch.zeros(1024), batch_index=0, boundary_index=5,
                      probe_index=0)

    def fail(*args, **kwargs):
        assert owner._active_window is not None
        raise RuntimeError('serializer failed')

    monkeypatch.setattr(checkpoints, 'write_checkpoint_cotangent_entry', fail)
    with pytest.raises(RuntimeError, match='serializer failed'):
        checkpoints.write_adjoint_checkpoint(
            tmp_path / 'streamed', boundary=5, session=_session(),
            cotangents={(0, 0): ref}, shared_adjoint={}, shared_pass={}, owner=owner)
    assert owner._active_window is None
    assert owner.telemetry['resident_tensor_bytes'] == 0
    assert any(row['state'] == 'retained' for row in owner._checkpoint_reservations.values())


def test_retired_descriptor_refuses_before_checkpoint_reservation(tmp_path):
    from prismaquant.joint_adjoint_checkpoints import write_adjoint_checkpoint
    from test_checkpoint_artifact_budget import _owner, _session

    owner = _owner(tmp_path / 'entries')
    ref = owner.write(torch.zeros(1024), batch_index=0, boundary_index=5,
                      probe_index=0)
    owner.retire(ref)
    with pytest.raises(RuntimeError, match='reference is stale'):
        write_adjoint_checkpoint(
            tmp_path / 'streamed', boundary=5, session=_session(),
            cotangents={(0, 0): ref}, shared_adjoint={}, shared_pass={}, owner=owner)
    assert owner.telemetry['checkpoint_reservations'] == 0
    assert not (tmp_path / 'streamed' / 'checkpoints').exists()
