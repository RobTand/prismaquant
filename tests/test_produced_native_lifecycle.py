"""Known bytes, exact tensors and ownership barriers through the real IO owner."""
import dataclasses
import hashlib
from pathlib import Path
import threading

import pytest
import torch

from prismaquant import io_engine
from prismaquant.stage_a_produced_output import (BoundaryProducedPrewriteRefused,
    BoundaryProducedPublicationFailed, read_bound_document)
from prismaquant.produced_output_spool import ProducedOutputSpool, ProducedExportRefused, ProducedWindowRefused
from prismaquant.streamed_boundary_artifacts import ProducedFileReference
from produced_native_fixtures import publication, owner


def raw_group(pub, spool, batch='known', payload=b'\x00independent\xffbytes'):
    destination = Path(pub.output_prefix) / (batch + '.bin')
    pub.require_prewrite(batch_id=batch, payload_ceiling_bytes=len(payload), paths=[str(destination)])
    directory = spool.reserve(batch, len(payload))
    local = directory / destination.name
    local.write_bytes(payload)
    ref = spool.record(batch, ProducedFileReference(str(local), local.name,
        len(payload), hashlib.sha256(payload).hexdigest()), destination.parent, read_back=True)
    return ref, local


def test_prewrite_ceiling_refuses_before_any_byte(tmp_path):
    pub = publication(tmp_path, payload_max=7)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    filename = Path(pub.output_prefix) / 'not-written'
    with pytest.raises(BoundaryProducedPrewriteRefused):
        pub.require_prewrite(batch_id='overflow', payload_ceiling_bytes=8, paths=[str(filename)])
    assert list(Path(pub.output_prefix).iterdir()) == []
    assert not pub._prewrites


def test_known_bytes_live_reader_retention_and_rematerialization(tmp_path):
    pub = publication(tmp_path, spool_bytes=1024)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    spool = ProducedOutputSpool.from_publication(pub)
    ref, local = raw_group(pub, spool)
    spool.submit('known')
    spool.await_group('known')
    assert Path(ref.path).read_bytes() == b'\x00independent\xffbytes'
    with spool.local_reads([ref]) as held:
        assert held[ref].read_bytes() == b'\x00independent\xffbytes'
        spool.retire_entry('known', ref)
        assert local.is_file() and not spool.released('known')
    assert not local.exists() and spool.released('known')
    binding = dict(spool._groups['known']['binding'])
    spool.prepare_reads([ref])
    with spool.local_reads([ref]) as held:
        assert held[ref].read_bytes() == b'\x00independent\xffbytes'
    assert spool._groups['known']['binding'] == binding
    assert len(pub._groups) == 1


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'replaced'])
def test_destination_mutation_retains_last_local_copy(tmp_path, mutation):
    pub = publication(tmp_path)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    spool = ProducedOutputSpool.from_publication(pub)
    ref, local = raw_group(pub, spool)
    spool.submit('known'); spool.await_group('known')
    target = Path(ref.path)
    if mutation == 'changed':
        target.chmod(0o600); target.write_bytes(b'wrong')
    elif mutation == 'missing':
        target.unlink()
    else:
        alternate = target.with_suffix('.new')
        alternate.write_bytes(b'\x00independent\xffbytes')
        alternate.replace(target)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        spool.retire_entry('known', ref)
    assert local.read_bytes() == b'\x00independent\xffbytes'
    assert not spool.released('known')


def test_copy_failure_preserves_original_error_and_sources(tmp_path, monkeypatch):
    pub = publication(tmp_path)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    spool = ProducedOutputSpool.from_publication(pub)
    ref, local = raw_group(pub, spool)
    entered, finish = threading.Event(), threading.Event()
    actual_copy = io_engine.publish_copy
    def interrupted(*args, **kwargs):
        entered.set()
        assert finish.wait(30)
        raise OSError('injected disk failure')
    monkeypatch.setattr(io_engine, 'publish_copy', interrupted)
    spool.submit('known')
    assert entered.wait(30)
    assert local.exists() and not Path(ref.path).exists()
    finish.set()
    with pytest.raises(ProducedExportRefused, match='injected disk failure'):
        spool.await_group('known')
    spool.join_copies()
    assert local.read_bytes() == b'\x00independent\xffbytes'
    assert not pub._groups and not spool.released('known')


def test_tensor_same_box_and_deferred_group_unlink(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path)
    # Do not issue selected authority from this unit seam. Runtime and actual
    # launch publication are exercised by the separate real native proof.
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        refs = [storage.write(torch.tensor([[1., 2., 3., 4.]]) + i,
            batch_index=i, boundary_index=0) for i in range(2)]
        storage.settle_local_output()
        with storage.prefetch(refs) as window:
            assert torch.equal(storage.get(window, refs[0]), torch.tensor([[1., 2., 3., 4.]]))
            assert torch.equal(storage.get(window, refs[1]), torch.tensor([[2., 3., 4., 5.]]))
        storage._retire(refs[0])
        assert Path(refs[0].path).exists()
        with storage.prefetch([refs[1]]) as window:
            assert torch.equal(storage.get(window, refs[1]), torch.tensor([[2., 3., 4., 5.]]))
        storage._retire(refs[1])
        assert all(not Path(ref.path).exists() for ref in refs)
    assert not storage._deferred_unlinks


def test_four_step_two_probe_chain_matches_independent_in_memory_arithmetic(tmp_path, monkeypatch):
    from test_stage_a_same_box_readback import _reference_chain, _run_chain, _payload_digests, N_BATCHES
    storage, pub = owner(tmp_path, n_batches=N_BATCHES, n_probes=2, tensor_bytes=1 << 14,
                         spool_bytes=2 * 2 * N_BATCHES * ((1 << 14) + 65536))
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        actual, _references = _run_chain(storage)
    assert _payload_digests(actual) == _payload_digests(_reference_chain())
    assert storage.telemetry['produced_local_reads'] > 0
    assert not storage._deferred_unlinks


@pytest.mark.parametrize('writer', ['document', 'copy'])
def test_exclusive_temporary_collision_never_removes_foreign_file(tmp_path, monkeypatch, writer):
    import secrets
    from prismaquant.staged_manifest import publish_staged_json
    monkeypatch.setattr(secrets, 'token_hex', lambda n: 'fixed')
    temporary = tmp_path / ('.publish-fixed' if writer == 'document' else '.copy-fixed')
    temporary.write_bytes(b'foreign-owner-bytes')
    target = tmp_path / 'destination'
    with pytest.raises(FileExistsError):
        if writer == 'document':
            publish_staged_json(target, {'known': 'input'})
        else:
            from prismaquant.stage_a_produced_output import produced_file_identity
            source = tmp_path / 'source'
            source.write_bytes(b'known')
            io_engine.publish_copy(source, target, length=5,
                sha256=hashlib.sha256(b'known').hexdigest(), source_identity=produced_file_identity(source))
    assert temporary.read_bytes() == b'foreign-owner-bytes'
    assert not target.exists()


def test_replaced_spool_root_retains_original_and_foreign_files(tmp_path):
    pub = publication(tmp_path)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    spool = ProducedOutputSpool.from_publication(pub)
    ref, local = raw_group(pub, spool)
    spool.submit('known'); spool.await_group('known')
    saved = spool.root.with_name('saved-owned-root')
    spool.root.rename(saved)
    foreign = spool.root / 'known'
    foreign.mkdir(parents=True)
    (foreign / local.name).write_bytes(b'foreign-owner-bytes')
    with pytest.raises(RuntimeError, match='root changed'):
        spool.retire_entry('known', ref)
    assert (saved / 'known' / local.name).read_bytes() == b'\x00independent\xffbytes'
    assert (foreign / local.name).read_bytes() == b'foreign-owner-bytes'
