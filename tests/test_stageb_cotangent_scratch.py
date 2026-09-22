"""Bound checkpoint tensor lifetimes without changing their exact values."""
import weakref

import torch

from prismaquant import joint_adjoint_checkpoints as checkpoints


def test_checkpoint_load_releases_each_exact_window_before_next(tmp_path, monkeypatch):
    tensors = {(probe, batch): torch.arange(8, dtype=torch.float32).reshape(2, 4) + probe + batch
               for probe in range(2) for batch in range(3)}
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(
        space, boundary=5, session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'},
        cotangents=tensors, shared_adjoint={}, shared_pass={})
    original = checkpoints.read_exact_entry_tensors
    seen = []

    def bounded(records, **kwargs):
        assert len(records) == 1, 'checkpoint loader retains the complete cotangent plane in one read window'
        seen.append(records[0]['name'])
        return original(records, **kwargs)

    monkeypatch.setattr(checkpoints, 'read_exact_entry_tensors', bounded)
    actual, shared, passed = checkpoints.load_adjoint_checkpoint(space, record)
    assert len(seen) == 6 and shared == passed == {}
    assert all(torch.equal(actual[key], value) for key, value in tensors.items())


def _records(dtype=torch.float32):
    size = torch.empty((), dtype=dtype).element_size() * 8
    return [{'name': f'cotangent-0-{i}', 'shape': [2, 4], 'dtype': str(dtype),
             'tensor_bytes': size} for i in range(3)]


def test_fixed_slots_are_exact_owned_and_released(tmp_path):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        records = _records(dtype)
        arena = ExactCotangentScratch(records, directory=tmp_path, max_bytes=1024)
        original = torch.arange(8, dtype=dtype).reshape(2, 4)
        arena[0, 0] = original
        owned = arena[0, 0]
        arena[0, 0] = original + 1
        assert torch.equal(owned, original)
        assert torch.equal(arena[0, 0], original + 1)
        strided = torch.arange(16, dtype=dtype).reshape(2, 8)[:, ::2]
        arena[0, 0] = strided
        assert torch.equal(arena[0, 0], strided)
        assert arena.tensor_bytes == sum(r['tensor_bytes'] for r in records)
        arena.close()
        assert torch.equal(owned, original)
        assert list(tmp_path.iterdir()) == []


def test_disk_ceiling_and_io_failures_refuse(tmp_path, monkeypatch):
    import os
    import pytest
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    with pytest.raises(RuntimeError, match='ceiling'):
        ExactCotangentScratch(_records(), directory=tmp_path, max_bytes=1)
    arena = ExactCotangentScratch(_records(), directory=tmp_path, max_bytes=1024)
    try:
        def failed(*args):
            raise OSError('sync failed')
        monkeypatch.setattr(os, 'fdatasync', failed)
        with pytest.raises(OSError, match='sync failed'):
            arena[0, 0] = torch.ones(2, 4)
        with pytest.raises(RuntimeError, match='not ready'):
            arena[0, 0]
    finally:
        arena.close()
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_sink_drops_prior_tensors_and_owner_closes(tmp_path, monkeypatch):
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    import test_joint_cost_quantum_runtime as rt
    values = {(0, i): torch.full((2, 4), float(i)) for i in range(3)}
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(
        space, boundary=5, session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'},
        cotangents=values, shared_adjoint={}, shared_pass={})
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(scratch))
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', '1024')
    original = checkpoints.read_exact_entry_tensors
    prior = []
    def bounded(records, **kwargs):
        assert all(ref() is None for ref in prior), 'previous checkpoint tensor retained'
        tensors = original(records, **kwargs)
        prior.extend(weakref.ref(value) for value in tensors.values())
        return tensors
    monkeypatch.setattr(checkpoints, 'read_exact_entry_tensors', bounded)
    owner = StreamedBoundaryArtifacts(rt._boundary_policy(tmp_path / 'boundaries'))
    with owner:
        arena, _, _ = checkpoints.load_adjoint_checkpoint(
            space, record, cotangent_factory=owner.checkpoint_cotangent_sink)
        assert all(ref() is None for ref in prior)
        assert all(torch.equal(arena[key], value) for key, value in values.items())
    assert arena._file is None and list(scratch.iterdir()) == []


def test_scratch_quantum_matches_single_run_bitwise(tmp_path, monkeypatch):
    import test_joint_cost_quantum_runtime as rt
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(scratch))
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', str(1 << 20))
    rt.test_quantum_matches_single_run_bitwise(tmp_path, monkeypatch)
    assert list(scratch.iterdir()) == []


def test_scratch_quantum_preserves_executable_phase_and_resume_order(tmp_path, monkeypatch):
    import test_quantum_executable_readset as phases
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(scratch))
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', str(1 << 20))
    phases.test_acceptance_real_quantum_reports_before_reads(tmp_path, monkeypatch)
    assert list(scratch.iterdir()) == []


def test_local_mount_and_resident_slot_bounds_refuse_before_allocation(tmp_path):
    import pytest
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    with pytest.raises(ValueError, match='local disk'):
        ExactCotangentScratch(_records(), directory='/dev/shm', max_bytes=1024)
    with pytest.raises(RuntimeError, match='resident tensor ceiling'):
        ExactCotangentScratch(_records(), directory=tmp_path, max_bytes=1024,
                              max_tensor_bytes=4)
    assert list(tmp_path.iterdir()) == []


def test_scratch_preserves_float_bit_patterns(tmp_path):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    bits = torch.tensor([0, -2147483648, 2139095040, -8388608,
                         2143289345, 1, -1, 1065353216], dtype=torch.int32).reshape(2, 4)
    arena = ExactCotangentScratch(_records(), directory=tmp_path, max_bytes=1024)
    try:
        arena[0, 0] = bits.view(torch.float32)
        assert torch.equal(arena[0, 0].view(torch.int32), bits)
    finally:
        arena.close()


def test_page_release_failure_never_publishes_slot(tmp_path, monkeypatch):
    import os
    import pytest
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    arena = ExactCotangentScratch(_records(), directory=tmp_path, max_bytes=1024)
    def failed(*args):
        raise OSError('page release failed')
    monkeypatch.setattr(os, 'posix_fadvise', failed)
    try:
        with pytest.raises(OSError, match='page release failed'):
            arena[0, 0] = torch.ones(2, 4)
        with pytest.raises(RuntimeError, match='not ready'):
            arena[0, 0]
    finally:
        arena.close()


def test_shared_payload_ceiling_precedes_deserialization(tmp_path, monkeypatch):
    import pytest
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(
        space, boundary=5, session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'},
        cotangents={(0, 0): torch.ones(2, 4)}, shared_adjoint={},
        shared_pass={0: {'value': torch.ones(2, 4)}})
    def no_read(*args, **kwargs):
        raise AssertionError('payload read before aggregate shared-state admission')
    monkeypatch.setattr(checkpoints, 'read_exact_entry_tensors', no_read)
    with pytest.raises(RuntimeError, match='auxiliary byte ceiling'):
        checkpoints.load_adjoint_checkpoint(space, record, shared_state_max_bytes=1)


def test_checkpoint_failure_closes_private_scratch_preserves_originals(tmp_path, monkeypatch):
    import pytest
    from pathlib import Path
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    import test_joint_cost_quantum_runtime as rt
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(
        space, boundary=5, session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'},
        cotangents={(0, i): torch.ones(2, 4) for i in range(2)},
        shared_adjoint={}, shared_pass={})
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(scratch))
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', '1024')
    original = checkpoints.read_exact_entry_tensors
    calls = 0
    def fails_second(records, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('checkpoint payload failed')
        return original(records, **kwargs)
    monkeypatch.setattr(checkpoints, 'read_exact_entry_tensors', fails_second)
    owner = StreamedBoundaryArtifacts(rt._boundary_policy(tmp_path / 'boundaries'))
    with pytest.raises(RuntimeError, match='checkpoint payload failed'):
        with owner:
            checkpoints.load_adjoint_checkpoint(space, record,
                cotangent_factory=owner.checkpoint_cotangent_sink)
    assert owner._cotangent_scratch is None and list(scratch.iterdir()) == []
    assert all(Path(entry['path']).is_file() for entry in record['activation_entries'])
