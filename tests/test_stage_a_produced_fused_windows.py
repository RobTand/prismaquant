"""Fused exact boundary/probe windows keep one existing local alias per group."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from prismaquant import io_engine
from produced_native_fixtures import owner
GROUP_SIZE, PROBES = 4, 2
def _value(boundary, probe, batch):
    return torch.arange(8, dtype=torch.float32) + 100 * boundary + 10 * (probe + 1) + batch


def _write_planes(storage):
    """Boundary 1's group and both probes' incoming groups at boundary 2."""
    boundary = [storage.write(_value(1, -1, batch), batch_index=batch, boundary_index=1)
                for batch in range(GROUP_SIZE)]
    incoming = [[storage.write(_value(2, probe, batch), batch_index=batch,
                               boundary_index=2, probe_index=probe)
                 for batch in range(GROUP_SIZE)] for probe in range(PROBES)]
    batches = [SimpleNamespace(activations_cpu={1: reference}) for reference in boundary]
    return batches, incoming


def _fused_read(storage, batches, incoming, *, window_batches):
    from prismaquant.streamed_boundary_artifacts import prefetched_fused_boundary_windows

    seen = []
    with prefetched_fused_boundary_windows(
            storage, batches, 1, incoming=incoming,
            window_batches=window_batches) as windows:
        for indices, boundary, incoming_of in windows:
            for batch in indices:
                assert torch.equal(boundary(batch), _value(1, -1, batch))
                for probe in range(PROBES):
                    assert torch.equal(incoming_of(probe, batch), _value(2, probe, batch))
                seen.append(batch)
    return seen




def test_a_window_that_cannot_hold_one_fused_read_refuses_at_bind(tmp_path):
    with pytest.raises(ValueError, match='required groups'):
        owner(tmp_path, n_batches=GROUP_SIZE, n_probes=PROBES,
            group_size=GROUP_SIZE, tensor_bytes=32, read_order='sample_major', window_groups=2)


@pytest.mark.parametrize('retention', [True, False])
def test_fused_windows_retain_exact_local_aliases_across_the_pass(tmp_path, monkeypatch, retention):
    from prismaquant.streamed_boundary_artifacts import prefetched_fused_boundary_windows
    storage, pub = owner(tmp_path, n_batches=GROUP_SIZE, n_probes=PROBES,
        group_size=GROUP_SIZE, tensor_bytes=32, read_order='sample_major')
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    if not retention:
        monkeypatch.setattr(storage, 'retain_produced_reads', lambda references: 0)
    with storage:
        batches, incoming = _write_planes(storage)
        refs = [batch.activations_cpu[1] for batch in batches] + [ref for plane in incoming for ref in plane]
        storage._local_output_spool.join_copies()
        restored = []
        actual_copy = io_engine.publish_copy
        def copy(source, destination, **kwargs):
            result = actual_copy(source, destination, **kwargs)
            restored.append(destination)
            return result
        monkeypatch.setattr(io_engine, 'publish_copy', copy)
        seen = []
        with prefetched_fused_boundary_windows(storage, batches, 1,
                incoming=incoming, window_batches=2) as windows:
            for indices, boundary, incoming_of in windows:
                for batch in indices:
                    assert torch.equal(boundary(batch), _value(1, -1, batch))
                    for probe in range(PROBES):
                        assert torch.equal(incoming_of(probe, batch), _value(2, probe, batch))
                    seen.append(batch)
                if indices.start == 0:
                    for ref in refs:
                        storage.release_produced_group(ref)
        assert seen == list(range(GROUP_SIZE))
        assert len(restored) == (0 if retention else (1 + PROBES) * GROUP_SIZE)
        assert len(pub._groups) == 1 + PROBES
        assert storage._produced_retained_reads == frozenset()
        assert all(not Path(row[0].path).exists() for group in storage._local_output_spool._groups.values()
                   for row in group['references'])
