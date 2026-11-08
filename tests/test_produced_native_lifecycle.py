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
    StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA, _write_new_file)
from produced_native_fixtures import publication, owner, selected_publications


def raw_group(pub, spool, batch='known', payload=b'\x00independent\xffbytes'):
    destination = Path(pub.output_prefix) / (batch + '.bin')
    pub.require_prewrite(batch_id=batch, payload_ceiling_bytes=len(payload), paths=[str(destination)])
    directory = spool.reserve(batch, len(payload))
    local = directory / destination.name
    identity = _write_new_file(local, payload)
    ref = spool.record(batch, ProducedFileReference(str(local), local.name,
        len(payload), hashlib.sha256(payload).hexdigest(), identity), destination.parent, read_back=True)
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
def _owned_group_roster(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('spool', [None, 1 << 20])
@pytest.mark.parametrize('replacement', ['intact', 'directory', 'symlink'])
def test_group_directory_is_still_owned_before_second_write(tmp_path, spool, replacement):
    storage, pub = owner(tmp_path, spool_bytes=spool)
    first = storage.write(torch.arange(8, dtype=torch.float32), batch_index=0, boundary_index=0)
    group = next(iter(storage._produced_groups.values()))
    directory = (Path(first.path).parent if spool is None else
                 storage._local_output_spool.directory(group['batch_id']))
    if replacement == 'intact':
        second = storage.write(torch.arange(8, dtype=torch.float32) + 10,
                               batch_index=1, boundary_index=0)
        storage._join_produced_copies()
        with storage.prefetch([first, second]) as window:
            assert torch.equal(storage.get(window, first), torch.arange(8, dtype=torch.float32))
            assert torch.equal(storage.get(window, second), torch.arange(8, dtype=torch.float32) + 10)
        assert len(pub._groups) == 1
        return
    original = tmp_path / 'original-group'
    directory.rename(original)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'sentinel').write_bytes(b'independent foreign bytes')
    if replacement == 'symlink':
        directory.symlink_to(foreign, target_is_directory=True)
        changed = foreign
    else:
        directory.mkdir()
        (directory / 'sentinel').write_bytes(b'independent replacement bytes')
        changed = directory
    before_original, before_foreign, before_changed = _owned_group_roster(original), _owned_group_roster(foreign), _owned_group_roster(changed)
    refused = False
    try:
        storage.write(torch.arange(8, dtype=torch.float32) + 10, batch_index=1, boundary_index=0)
    except (OSError, ValueError, RuntimeError):
        refused = True
    assert _owned_group_roster(original) == before_original
    assert _owned_group_roster(foreign) == before_foreign
    assert _owned_group_roster(changed) == before_changed, 'replacement received second payload or temporary bytes'
    assert refused and not pub._groups


@pytest.mark.parametrize('replace', [False, True])
def test_direct_publication_keeps_original_writer_inode(tmp_path, replace):
    storage, pub = owner(tmp_path, spool_bytes=None)
    first = storage.write(torch.arange(8, dtype=torch.float32), batch_index=0, boundary_index=0)
    path = Path(first.path)
    payload = path.read_bytes()
    if replace:
        original = tmp_path / 'original-first.pt'
        path.rename(original)
        path.write_bytes(payload)
    refused = False
    try:
        storage.write(torch.arange(8, dtype=torch.float32) + 10, batch_index=1, boundary_index=0)
    except (OSError, ValueError, RuntimeError):
        refused = True
    if replace:
        assert refused and not pub._groups, 'same-byte replacement was adopted as writer ownership'
        assert original.read_bytes() == path.read_bytes() == payload
    else:
        assert not refused and len(pub._groups) == 1


@pytest.mark.parametrize('replace', [False, True])
def test_copy_publication_keeps_returned_destination_inode(tmp_path, monkeypatch, replace):
    pub = publication(tmp_path)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    spool = ProducedOutputSpool.from_publication(pub)
    ref, local = raw_group(pub, spool)
    reached, proceed = threading.Event(), threading.Event()
    real = io_engine.publish_copy
    def copy(*args, **kwargs):
        identity = real(*args, **kwargs)
        reached.set()
        assert proceed.wait(30)
        return identity
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    spool.submit('known')
    assert reached.wait(30)
    path = Path(ref.path)
    payload = path.read_bytes()
    if replace:
        original = tmp_path / 'original-copy'
        path.rename(original)
        path.write_bytes(payload)
    proceed.set()
    try:
        if replace:
            with pytest.raises(ProducedExportRefused):
                spool.await_group('known')
            assert not pub._groups and not spool.released('known')
            assert original.read_bytes() == path.read_bytes() == local.read_bytes() == payload
        else:
            spool.await_group('known')
            assert path.read_bytes() == payload and len(pub._groups) == 1
    finally:
        spool.join_copies()


@pytest.mark.parametrize('replace', [False, True])
def test_metadata_origin_retains_writer_identity_until_group_seal(tmp_path, monkeypatch, replace):
    from prismaquant import streamed_boundary_artifacts as artifacts
    storage, pub = owner(tmp_path, write_only=True, spool_bytes=None)
    actual = artifacts._link_new_file
    preserved = []
    def publish(path, payload, **kwargs):
        identity = actual(path, payload, **kwargs)
        if replace and path.name == 'first.json':
            saved = path.with_name('original-first')
            path.rename(saved)
            path.write_bytes(payload)
            preserved.append(saved)
        return identity
    monkeypatch.setattr(artifacts, '_link_new_file', publish)
    files = [('first.json', b'independent first metadata'), ('second.json', b'second')]
    if replace:
        with pytest.raises(BoundaryProducedPublicationFailed, match='writer destination identity changed'):
            storage.write_produced_files(files, kind='metadata', boundary_index=0)
        assert not pub._groups
        assert preserved[0].read_bytes() == (storage.directory / 'first.json').read_bytes() == files[0][1]
    else:
        refs = storage.write_produced_files(files, kind='metadata', boundary_index=0)
        assert [Path(ref.path).read_bytes() for ref in refs] == [raw for _, raw in files]
        rows = read_bound_document(next(iter(pub._groups.values())))['entries']
        assert [row['file_identity'] for row in rows] == [list(ref.writer_file_identity) for ref in refs]


@pytest.mark.parametrize('packed,replace', [(False, 'intact'), (False, 'cotangent'),
    (False, 'shared'), (False, 'manifest'), (True, 'intact'), (True, 'shared'), (True, 'manifest')])
def test_checkpoint_group_keeps_each_writer_identity(tmp_path, monkeypatch, packed, replace):
    from prismaquant import joint_adjoint_checkpoints as checkpoint
    storage, pub = owner(tmp_path, spool_bytes=None)
    known = torch.tensor([[1., 3.], [5., 7.]])
    session = {**storage.session, 'kind': 'adjoint_checkpoint'}
    cotangents = {(0, 0): known}
    if packed:
        cotangents = {(0, batch): storage.write(known + batch,
            probe_index=0, batch_index=batch, boundary_index=5) for batch in range(2)}
        storage.settle_local_output()
    prior_groups = dict(pub._groups)
    original_tensor = checkpoint.write_checkpoint_cotangent_entry
    original_shared = checkpoint._publish_streamed_file
    original_manifest = checkpoint.atomic_write_bytes
    preserved = []
    def replacement(path):
        payload = path.read_bytes()
        saved = tmp_path / (path.name + '.original')
        path.rename(saved)
        path.write_bytes(payload)
        preserved.append((saved, path, payload))
    def tensor(*args, **kwargs):
        row = original_tensor(*args, **kwargs)
        if replace == 'cotangent':
            replacement(Path(row['path']))
        return row
    def shared(*args, **kwargs):
        row = original_shared(*args, **kwargs)
        if replace == 'shared' and not preserved:
            replacement(Path(row['path']))
        return row
    def manifest(path, payload, **kwargs):
        result = original_manifest(path, payload, **kwargs)
        if replace == 'manifest':
            replacement(path)
        return result
    monkeypatch.setattr(checkpoint, 'write_checkpoint_cotangent_entry', tensor)
    monkeypatch.setattr(checkpoint, '_publish_streamed_file', shared)
    monkeypatch.setattr(checkpoint, 'atomic_write_bytes', manifest)
    space = Path(pub.output_prefix) / 'checkpoint-space'
    def write():
        return checkpoint.write_adjoint_checkpoint(space, boundary=5, session=session,
            cotangents=cotangents, shared_adjoint={(0, 0): {'w': known + 2}},
            shared_pass={0: {'known': 'independent shared metadata'}}, owner=storage,
            packed=packed, referenced=packed)
    if replace != 'intact':
        with pytest.raises(BoundaryProducedPublicationFailed, match='writer destination identity changed'):
            write()
        assert pub._groups == prior_groups and preserved
        for saved, path, payload in preserved:
            assert saved.read_bytes() == path.read_bytes() == payload
        return
    record = write()
    tensors = checkpoint.read_exact_entry_tensors(record['activation_entries'],
        expected_session=storage.session if packed else session)
    assert torch.equal(tensors[record['activation_entries'][0]['name']], known)
    adjoint, shared_pass = checkpoint.load_checkpoint_shared_states(space, record)
    assert torch.equal(adjoint[(0, 0)]['w'], known + 2)
    assert shared_pass == {0: {'known': 'independent shared metadata'}}
    new_groups = set(pub._groups) - set(prior_groups)
    assert len(new_groups) == 1
    rows = read_bound_document(pub._groups[new_groups.pop()])['entries']
    assert {row['path'] for row in rows} == {
        *(row['path'] for row in ([] if packed else record['activation_entries'])),
        *(row['path'] for row in record['shared_state_entries']),
        str(space / 'checkpoints' / 'boundary-005' / 'checkpoint.json')}


def test_generic_descriptor_refuses_missing_writer_identity(tmp_path):
    pub = publication(tmp_path)
    target = Path(pub.output_prefix) / 'generic'
    _write_new_file(target, b'known generic bytes')
    ref = ProducedFileReference(str(target), target.name, 19, hashlib.sha256(b'known generic bytes').hexdigest())
    with pytest.raises(BoundaryProducedPublicationFailed, match='writer destination identity required'):
        pub.descriptor_for(ref, producer_generation='unowned')
    assert target.read_bytes() == b'known generic bytes' and not pub._groups


@pytest.mark.parametrize('spool', [None, 1 << 20])
@pytest.mark.parametrize('replacement', ['directory', 'symlink'])
def test_metadata_parent_replacement_refuses_before_next_file(tmp_path, monkeypatch, spool, replacement):
    from prismaquant import streamed_boundary_artifacts as artifacts
    storage, pub = owner(tmp_path, write_only=True, spool_bytes=spool)
    actual = artifacts._write_new_file
    original, foreign = tmp_path / 'original-metadata-parent', tmp_path / 'foreign-metadata-parent'
    foreign.mkdir()
    (foreign / 'sentinel').write_bytes(b'independent foreign file')
    snapshots = {}
    def write(path, payload, **kwargs):
        identity = actual(path, payload, **kwargs)
        if path.name in ('first.json', 'first.json.tmp'):
            parent = path.parent
            parent.rename(original)
            if replacement == 'directory':
                parent.mkdir()
                (parent / 'sentinel').write_bytes(b'independent replacement file')
            else:
                parent.symlink_to(foreign, target_is_directory=True)
            snapshots['original'] = _owned_group_roster(original)
            snapshots['foreign'] = _owned_group_roster(foreign)
            snapshots['replacement'] = _owned_group_roster(parent)
            snapshots['path'] = parent
        return identity
    monkeypatch.setattr(artifacts, '_write_new_file', write)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        storage.write_produced_files([('first.json', b'independent first'), ('second.json', b'second')],
                                    kind='metadata', boundary_index=0)
    assert not pub._groups and snapshots
    assert _owned_group_roster(original) == snapshots['original']
    assert _owned_group_roster(foreign) == snapshots['foreign']
    assert _owned_group_roster(snapshots['path']) == snapshots['replacement']


@pytest.mark.parametrize('boundary', ['entries', 'manifest'])
@pytest.mark.parametrize('replacement', ['intact', 'directory', 'symlink'])
def test_checkpoint_parent_replacement_precedes_next_write(tmp_path, boundary, replacement):
    from prismaquant import joint_adjoint_checkpoints as checkpoint
    storage, pub = owner(tmp_path, spool_bytes=None)
    session = {**storage.session, 'kind': 'adjoint_checkpoint'}
    attempt = checkpoint.open_adjoint_checkpoint(Path(pub.output_prefix) / 'space',
        boundary=5, session=session, specs={(0, batch): (16, [2, 2], 'torch.float32') for batch in range(2)},
        shared_adjoint_keys=[], shared_pass_keys=[], owner=storage)
    known = torch.tensor([[2., 4.], [6., 8.]])
    attempt.write_activation(0, 0, known)
    if boundary == 'manifest':
        attempt.write_activation(0, 1, known + 1)
        attempt.write_shared_states({}, {})
    directory = attempt.checkpoint_dir / 'entries' if boundary == 'entries' else attempt.checkpoint_dir
    original, foreign = tmp_path / 'original-checkpoint-parent', tmp_path / 'foreign-checkpoint-parent'
    foreign.mkdir()
    (foreign / 'sentinel').write_bytes(b'independent foreign file')
    if replacement != 'intact':
        directory.rename(original)
        if replacement == 'directory':
            directory.mkdir()
            (directory / 'sentinel').write_bytes(b'independent replacement file')
        else:
            directory.symlink_to(foreign, target_is_directory=True)
        before = [_owned_group_roster(root) for root in (original, foreign, directory)]
    def finish():
        if boundary == 'entries':
            attempt.write_activation(0, 1, known + 1)
            attempt.write_shared_states({}, {})
        return attempt.seal()
    if replacement == 'intact':
        record = finish()
        tensors = checkpoint.read_exact_entry_tensors(record['activation_entries'], expected_session=session)
        assert torch.equal(tensors['cotangent-0-0'], known)
        assert torch.equal(tensors['cotangent-0-1'], known + 1)
        assert len(pub._groups) == 1
    else:
        with pytest.raises((OSError, ValueError, RuntimeError)):
            finish()
        assert [_owned_group_roster(root) for root in (original, foreign, directory)] == before
        assert not pub._groups


@pytest.mark.parametrize('writer', ['manifest', 'shared'])
@pytest.mark.parametrize('collision', [False, True])
def test_native_small_file_publication_never_replaces_a_collision(tmp_path, monkeypatch, writer, collision):
    import os
    from prismaquant import cost_stage_checkpoint as journal
    from prismaquant import joint_adjoint_checkpoints as checkpoint
    info = tmp_path.stat()
    directories, identities = {str(tmp_path): (info.st_dev, info.st_ino)}, {}
    path = tmp_path / 'known.bin'
    actual = os.link
    foreign = b'independently known foreign bytes'
    def link(source, destination, **kwargs):
        if collision and destination == path.name:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o600, dir_fd=kwargs['dst_dir_fd'])
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(foreign)
        return actual(source, destination, **kwargs)
    monkeypatch.setattr(os, 'link', link)
    def publish():
        if writer == 'manifest':
            journal.atomic_write_bytes(path, b'known writer bytes',
                writer_identities=identities, writer_directories=directories)
        else:
            checkpoint._publish_streamed_file(path, 'known', lambda sink: sink.write(b'known writer bytes'),
                max_file_bytes=100, label='known', writer_identities=identities, writer_directories=directories)
    if collision:
        with pytest.raises(FileExistsError):
            publish()
        assert path.read_bytes() == foreign and not identities
    else:
        publish()
        assert path.read_bytes() == b'known writer bytes'
        from prismaquant.stage_a_produced_output import produced_file_identity
        assert list(identities[str(path)]) == produced_file_identity(path)
