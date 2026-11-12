"""Writer/prewrite, exact native publication and bounded own-session readback.

The former fleet simulator, funding, mover and receipt fixtures are retired.
Native execution/dependency authority is proved by the recorded CPU controller.
These fixtures exercise the actual tensor and IO owners without a scheduler.
"""
import dataclasses
import os
from pathlib import Path
import pytest
import torch
from produced_native_fixtures import owner
from prismaquant.stage_a_produced_output import BoundaryProducedPrewriteRefused
from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA

GROUP_SIZE = 4


@pytest.mark.parametrize('lifetime', ['retain', 'consumed'])
def test_referenced_checkpoint_requires_retained_application_lifetime(tmp_path, monkeypatch, lifetime):
    from produced_native_fixtures import selected_publications
    from prismaquant.stage_a_produced_output import read_bound_document
    storage, pub = owner(tmp_path, n_batches=2, group_size=2, tensor_bytes=32,
        spool_bytes=None, write_only=True, origin_lifetime=lifetime)
    register = selected_publications(monkeypatch)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: register(pub))
    with storage:
        refs = [storage.write(torch.arange(8, dtype=torch.float32) + b,
            batch_index=b, boundary_index=3, probe_index=0, read_back=False) for b in range(2)]
        group = read_bound_document(next(iter(pub._groups.values())))
        assert group['lifetime'] == lifetime
        assert [Path(ref.path).is_file() for ref in refs] == [True, True]
        if lifetime == 'consumed':
            with pytest.raises(RuntimeError, match='lifetime consumed'):
                storage.await_checkpoint_references(refs)
        else:
            storage.await_checkpoint_references(refs)
        assert [Path(ref.path).is_file() for ref in refs] == [True, True]


def write_group(storage, *, boundary_index=0, count=GROUP_SIZE, first=0):
    return [storage.write(torch.arange(8, dtype=torch.float32) + batch,
        batch_index=batch, boundary_index=boundary_index)
        for batch in range(first, first + count)]


def read_group(storage, refs, *, first=0):
    with storage.prefetch(refs) as window:
        for batch, ref in enumerate(refs, first):
            assert torch.equal(storage.get(window, ref), torch.arange(8, dtype=torch.float32) + batch)


def _owner(tmp_path, monkeypatch, **kwargs):
    storage, pub = owner(tmp_path, group_size=GROUP_SIZE, tensor_bytes=32, **kwargs)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    return storage, pub


@pytest.mark.parametrize('spool', [False, True])
def test_own_boundary_group_publishes_and_reads_exact_tensors(tmp_path, monkeypatch, spool):
    from prismaquant.staged_tier_policy import staged_tier_policy_test_context
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE,
        spool_bytes=(1 << 24) if spool else None)
    with storage:
        refs = write_group(storage)
        with staged_tier_policy_test_context('ssd'):
            read_group(storage, refs)
        storage.settle_local_output()
        assert len(pub._groups) == 1
        assert len(storage.produced_origin_batches()) == 1
    assert all(Path(ref.path).is_file() for ref in refs)
    assert storage.telemetry['resident_tensor_bytes'] == 0


def test_strict_unpublished_input_has_no_own_session_exemption(tmp_path, monkeypatch):
    from prismaquant.staged_tier_policy import staged_tier_policy_test_context
    from prismaquant.staged_lease import LeaseRefused
    from prismaquant.residency_map import ENV_VAR, reset_residency_resolver_for_tests
    storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
        'directory': str(tmp_path / 'unpublished'), 'max_resident_bytes': 1024,
        'max_auxiliary_bytes': 1024, 'max_artifact_bytes': 1 << 20, 'prefetch_batches': 1})
    storage.bind({'fixture': 'unpublished'}, n_probes=1)
    ref = storage.write(torch.arange(8, dtype=torch.float32), batch_index=0, boundary_index=0)
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    try:
        with storage, staged_tier_policy_test_context('ssd'):
            with pytest.raises(LeaseRefused, match='^readset-not-staged$'):
                with storage.prefetch([ref]):
                    pass
    finally:
        reset_residency_resolver_for_tests()


def test_repeated_distinct_group_turnover_and_same_group_rematerialization(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE, spool_bytes=2 * GROUP_SIZE * (32 + 65536))
    with storage:
        for boundary in range(8):
            refs = write_group(storage, boundary_index=boundary)
            read_group(storage, refs)
            storage._local_output_spool.join_copies()
            for ref in refs:
                storage.release_produced_group(ref)
            assert all(storage._local_output_spool.released(group['batch_id'])
                for group in storage._produced_groups.values())
        before = dict(pub._groups)
        read_group(storage, refs)
        assert pub._groups == before and len(pub._groups) == 8
        assert not storage._deferred_unlinks


def test_a_foreign_session_reference_is_refused(tmp_path, monkeypatch):
    storage, _ = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE)
    foreign, _ = _owner(tmp_path / 'foreign', monkeypatch, n_batches=GROUP_SIZE)
    with storage, foreign:
        ref = write_group(foreign)[0]
        with pytest.raises(RuntimeError, match='stale or belongs to another'):
            with storage.prefetch([ref]):
                pass


def test_an_entry_directory_outside_the_prefix_is_refused(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE)
    escaped = StreamedBoundaryArtifacts({**storage.config, 'directory': str(tmp_path / 'elsewhere')})
    escaped.bind({'source_model': 'fixture'}, n_probes=1)
    with storage, escaped:
        with pytest.raises(RuntimeError, match='escapes declared output prefix'):
            escaped.bind_produced_output(pub, group_size=GROUP_SIZE, n_batches=GROUP_SIZE,
                max_entry_tensor_bytes=32)


def test_a_mutated_origin_is_refused_before_direct_publication(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE, spool_bytes=None)
    refs = write_group(storage, count=1)
    path = Path(refs[0].path)
    content = path.read_bytes()
    before = path.stat()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 255]))
    # File timestamps use a coarse clock: a rewrite inside the same tick keeps
    # the writer's stat identity. Move the timestamp so the mutation is visible.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert path.stat().st_size == refs[0].file_bytes
    with pytest.raises(RuntimeError, match='writer destination identity changed'):
        write_group(storage, count=GROUP_SIZE - 1, first=1)
    assert not pub._groups and path.exists()
    storage.__exit__(RuntimeError, RuntimeError('fixture mutation'), None)


def test_a_read_only_attachment_cannot_declare_owner_prewrites(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE, spool_bytes=None)
    with storage:
        write_group(storage)
        attached = StreamedBoundaryArtifacts(dict(storage.config))
        attached.attach(storage.session, n_probes=1)
        with attached:
            with pytest.raises(RuntimeError, match='fresh writable bound session'):
                attached.bind_produced_output(pub, group_size=GROUP_SIZE, n_batches=GROUP_SIZE,
                    max_entry_tensor_bytes=32)


def test_a_group_ceiling_refuses_before_first_tensor_or_temporary(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, n_batches=GROUP_SIZE, payload_max=100)
    with pytest.raises(BoundaryProducedPrewriteRefused):
        write_group(storage, count=1)
    assert not pub._prewrites and not list(storage.directory.rglob('*.pt*'))
