"""Known bytes, exact tensors and ownership barriers through the real IO owner."""
import dataclasses
import copy
import hashlib
from pathlib import Path
import threading

import pytest
import torch

from prismaquant import io_engine
from prismaquant.stage_a_produced_output import (BoundaryProducedPrewriteRefused,
    BoundaryProducedPublication, BoundaryProducedPublicationFailed, read_bound_document,
    publish_document)
from prismaquant.produced_output_spool import ProducedOutputSpool, ProducedExportRefused, ProducedWindowRefused
from prismaquant.streamed_boundary_artifacts import (ProducedFileReference,
    StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA)
from produced_native_fixtures import publication, owner, selected_publications


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


@pytest.mark.parametrize('spool', [None, 1 << 20])
@pytest.mark.parametrize('budget', [0, 64])
def test_metadata_temporary_ceiling_precedes_writes(tmp_path, monkeypatch, spool, budget):
    original = publication(tmp_path, spool_bytes=spool, write_only=True)
    declaration = copy.deepcopy(original.declaration)
    declaration['durable_maxima']['temp_max_bytes'] = budget
    binding = publish_document(tmp_path / 'temp-declaration.json', declaration)
    producer = copy.deepcopy(original.producer)
    producer['argv'] = ['fixture', '--produced-declaration', binding['path'],
                       '--produced-declaration-sha256', binding['sha256']]
    pub = BoundaryProducedPublication(declaration=declaration, binding=binding, producer=producer)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
        'directory': str(Path(pub.output_prefix) / 'boundaries'),
        'max_resident_bytes': 1024, 'max_auxiliary_bytes': 1024,
        'max_artifact_bytes': 1 << 24, 'prefetch_batches': 2})
    storage.bind({'fixture': 'temp-budget'}, n_probes=1, published=True)
    storage.bind_produced_output(pub, group_size=2, n_batches=2,
        max_entry_tensor_bytes=1024, origin_lifetime='retain')
    payload = b'independently known metadata bytes'
    with storage:
        if budget == 0:
            with pytest.raises(BoundaryProducedPrewriteRefused, match='temp_max_bytes'):
                storage.write_produced_files([('control.json', payload)], kind='metadata', boundary_index=0)
            assert not (storage.directory / 'control.json').exists()
            assert not list(storage.directory.rglob('*.tmp')) and not pub._prewrites
        else:
            refs = storage.write_produced_files([('control.json', payload)], kind='metadata', boundary_index=0)
            assert len(refs) == 1 and Path(refs[0].path).read_bytes() == payload


@pytest.mark.parametrize('spool', [None, 1 << 20])
@pytest.mark.parametrize('ceiling', [4, 32])
def test_direct_and_spooled_tensor_ceiling_precedes_writes(tmp_path, monkeypatch, spool, ceiling):
    storage, pub = owner(tmp_path, tensor_bytes=ceiling, spool_bytes=spool)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        if ceiling < 32:
            with pytest.raises(RuntimeError, match='tensor ceiling'):
                storage.write(torch.arange(8, dtype=torch.float32), batch_index=0, boundary_index=0)
            assert not list(storage.directory.rglob('*.pt*')) and not pub._prewrites
        else:
            refs = [storage.write(torch.arange(8, dtype=torch.float32) + batch,
                batch_index=batch, boundary_index=0) for batch in range(2)]
            with storage.prefetch(refs) as window:
                for batch, ref in enumerate(refs):
                    assert torch.equal(storage.get(window, ref), torch.arange(8, dtype=torch.float32) + batch)


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
@pytest.mark.parametrize('occupied', [False, True])
def test_checkpoint_cancel_releases_native_reservation_only_on_absence(tmp_path, monkeypatch, occupied):
    register = selected_publications(monkeypatch)
    storage, pub = owner(tmp_path, spool_bytes=None)
    directory = Path(pub.output_prefix) / 'checkpoints' / 'attempt'
    plan = {'files': [{'name': 'f', 'path': str(directory / 'f.pt'), 'envelope_bytes': 100}],
            'manifest_bytes': 100, 'temp_overlap_bytes': 100, 'envelope_bytes': 300}
    def reserve():
        return storage.reserve_checkpoint_artifact(label='known checkpoint', envelope_bytes=300,
            file_plan=plan, checkpoint_dir=directory)
    reservation = reserve()
    batch = storage._checkpoint_reservations[reservation]['produced_batch']
    if occupied:
        directory.mkdir(parents=True)
        (directory / 'f.pt').write_bytes(b'foreign occupied bytes')
        occupied_path = directory / 'f.pt'
        with pytest.raises((OSError, ValueError, RuntimeError)):
            storage.cancel_checkpoint_artifact(reservation)
        assert storage._checkpoint_active == reservation
        assert reservation in storage._checkpoint_reservations and batch not in pub._released
        assert set(directory.iterdir()) == {occupied_path}
        assert occupied_path.read_bytes() == b'foreign occupied bytes'
        return
    storage.cancel_checkpoint_artifact(reservation)
    assert batch in pub._released, 'native reservation remained active after application cancellation'
    again = reserve()
    storage.cancel_checkpoint_artifact(again)
    refs = [storage.write(torch.arange(8, dtype=torch.float32) + i, batch_index=i, boundary_index=0)
            for i in range(2)]
    with storage.prefetch(refs) as window:
        assert torch.equal(storage.get(window, refs[1]), torch.arange(8, dtype=torch.float32) + 1)
    register(pub)
