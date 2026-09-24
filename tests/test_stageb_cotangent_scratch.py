"""Bound checkpoint tensor lifetimes without changing their exact values."""
import weakref

import pytest
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


def _fake_cap(monkeypatch, tmp_path, *, cap, current):
    """Make ``cgroup_memory_cap`` report ``cap`` over a scope charged ``current``."""
    from prismaquant import memory_management as memory
    scope = tmp_path / 'cgroup-scope'; scope.mkdir(exist_ok=True)
    (scope / 'memory.current').write_text(str(current))
    monkeypatch.setattr(memory, 'cgroup_memory_cap', lambda **_: (cap, scope))


def test_sink_without_a_scratch_refuses_a_plane_the_cgroup_cannot_hold(tmp_path, monkeypatch):
    """No scratch pair and no room: refuse before the first entry is read (#1141)."""
    import pytest
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.memory_management import CaptureMemoryGuard
    import test_joint_cost_quantum_runtime as rt
    values = {(0, i): torch.full((2, 4), float(i)) for i in range(3)}
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(
        space, boundary=5, session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'},
        cotangents=values, shared_adjoint={}, shared_pass={})
    plane = sum(row['tensor_bytes'] for row in record['activation_entries'])
    assert plane == 3 * 8 * 4
    monkeypatch.delenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', raising=False)
    monkeypatch.delenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', raising=False)
    original = checkpoints.read_exact_entry_tensors
    reads = []

    def counted(records, **kwargs):
        reads.append(len(records))
        return original(records, **kwargs)

    monkeypatch.setattr(checkpoints, 'read_exact_entry_tensors', counted)
    current = 5 << 30
    room = CaptureMemoryGuard.MARGIN_BYTES + current
    _fake_cap(monkeypatch, tmp_path, cap=room + plane - 1, current=current)
    owner = StreamedBoundaryArtifacts(rt._boundary_policy(tmp_path / 'boundaries'))
    with owner, pytest.raises(RuntimeError, match='cotangent plane without a scratch') as refused:
        checkpoints.load_adjoint_checkpoint(
            space, record, cotangent_factory=owner.checkpoint_cotangent_sink)
    assert reads == []
    message = str(refused.value)
    assert f'{plane} bytes in host memory' in message
    assert f'{room + plane - 1}-byte cap' in message
    assert f'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES >= {plane}' in message
    # Exactly enough room: the plane is a host dict, as before.
    _fake_cap(monkeypatch, tmp_path, cap=room + plane, current=current)
    owner = StreamedBoundaryArtifacts(rt._boundary_policy(tmp_path / 'boundaries-fit'))
    with owner:
        actual, _, _ = checkpoints.load_adjoint_checkpoint(
            space, record, cotangent_factory=owner.checkpoint_cotangent_sink)
    assert type(actual) is dict and len(reads) == 3
    assert all(torch.equal(actual[key], value) for key, value in values.items())


def test_cgroup_room_is_held_against_the_tightest_ancestor(tmp_path):
    """The cap and charge come from the scope that sets the smallest limit."""
    import pytest
    from prismaquant.memory_management import (
        CaptureMemoryGuard, cgroup_memory_cap, require_cgroup_room)
    root = tmp_path / 'cgroup'
    outer, inner = root / 'job', root / 'job' / 'container'
    inner.mkdir(parents=True)
    (outer / 'memory.max').write_text(str(94 << 30))
    (outer / 'memory.current').write_text(str(40 << 30))
    (inner / 'memory.max').write_text(str(28 << 30))
    (inner / 'memory.current').write_text(str(4 << 30))
    membership = tmp_path / 'membership'; membership.write_text('0::/job/container\n')
    kwargs = dict(cgroup_root=root, membership=membership)
    assert cgroup_memory_cap(**kwargs) == (28 << 30, inner)
    room = (28 << 30) - (4 << 30) - CaptureMemoryGuard.MARGIN_BYTES
    assert require_cgroup_room(room, owner='plane', remedy='r', **kwargs) == {
        'cap_bytes': 28 << 30, 'current_bytes': 4 << 30,
        'margin_bytes': CaptureMemoryGuard.MARGIN_BYTES}
    with pytest.raises(RuntimeError, match='plane: 34359738368 bytes in host memory'):
        require_cgroup_room(32 << 30, owner='plane', remedy='r', **kwargs)
    # No finite cap anywhere, or no cgroup v2 membership: nothing to hold.
    (inner / 'memory.max').write_text('max'); (outer / 'memory.max').write_text('max')
    assert cgroup_memory_cap(**kwargs) is None
    assert require_cgroup_room(32 << 30, owner='plane', remedy='r', **kwargs) is None
    membership.write_text('1:name=systemd:/job\n')
    assert require_cgroup_room(32 << 30, owner='plane', remedy='r', **kwargs) is None


@pytest.mark.parametrize('scratch', [True, False], ids=['scratch', 'no-scratch'])
def test_the_quantum_holds_its_plane_in_the_scratch(tmp_path, monkeypatch, scratch):
    """With the pair set, the quantum's checkpoint plane lives in the scratch.

    The sink is observed around the real quantum: with the pair it hands the
    loader an ExactCotangentScratch sized to the whole plane, and every plane
    row is written into it. Without the pair the same quantum holds a host
    dict, which is what the observation must be able to tell apart.
    """
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    import test_joint_cost_quantum_runtime as rt
    sinks, written = [], {}
    original_sink = StreamedBoundaryArtifacts.checkpoint_cotangent_sink
    original_write = ExactCotangentScratch.__setitem__

    def observed(self, records):
        sink = original_sink(self, records)
        sinks.append((sink, sum(row['tensor_bytes'] for row in records), len(records)))
        return sink

    def write(self, key, value):
        original_write(self, key, value)
        written.setdefault(id(self), set()).add(key)

    monkeypatch.setattr(StreamedBoundaryArtifacts, 'checkpoint_cotangent_sink', observed)
    monkeypatch.setattr(ExactCotangentScratch, '__setitem__', write)
    root = tmp_path / 'scratch'; root.mkdir()
    if scratch:
        monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(root))
        monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', str(1 << 20))
    else:
        monkeypatch.delenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', raising=False)
        monkeypatch.delenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', raising=False)
    rt.test_quantum_matches_single_run_bitwise(tmp_path, monkeypatch)
    assert sinks
    for sink, plane, rows in sinks:
        if not scratch:
            assert type(sink) is dict
            continue
        assert isinstance(sink, ExactCotangentScratch)
        assert sink.tensor_bytes == plane > 0
        assert len(sink) == rows and written[id(sink)] == set(sink)
    assert bool(written) is scratch
    assert list(root.iterdir()) == []
