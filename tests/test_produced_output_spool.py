"""Application spool ownership, exact writer bytes and IO completion barriers."""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
import threading
import pytest
import torch
from prismaquant import io_engine
from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry
from prismaquant.produced_output_spool import (ProducedOutputSpool,
    ProducedOutputSpoolRefused, ProducedExportRefused, ProducedWindowRefused)
from prismaquant.stage_a_produced_output import read_bound_document
from prismaquant.streamed_boundary_artifacts import ProducedFileReference
from produced_native_fixtures import publication, owner
from test_produced_native_lifecycle import raw_group


def _spool(tmp_path, *, window=1024):
    pub = publication(tmp_path, spool_bytes=window)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    return pub, ProducedOutputSpool.from_publication(pub)

def test_preallocation_uses_serialized_inode_and_preserves_exact_bytes(tmp_path, monkeypatch):
    calls = []
    original = os.posix_fallocate
    def allocate(fd, offset, length):
        original(fd, offset, length)
        calls.append((os.fstat(fd).st_ino, length))
    monkeypatch.setattr(os, "posix_fallocate", allocate)
    tensor = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    kwargs = dict(identity={"session": "fixture"}, max_tensor_bytes=2048,
                  max_file_bytes=2048 + 65536)
    normal = write_exact_activation_cache_entry(tmp_path / "normal", "same", tensor, **kwargs)
    local = write_exact_activation_cache_entry(tmp_path / "local", "same", tensor,
                                             preallocate=True, **kwargs)
    assert calls == [(Path(local.path).stat().st_ino, 2048 + 65536)]
    assert Path(normal.path).read_bytes() == Path(local.path).read_bytes()
    assert normal.sha256 == local.sha256
    assert Path(local.path).stat().st_size == local.file_bytes < calls[0][1]


def test_serialization_cannot_overrun_preallocated_file(tmp_path):
    with pytest.raises(RuntimeError, match="preallocated ceiling|unexpected pos"):
        write_exact_activation_cache_entry(
            tmp_path, "too-large", torch.arange(8), identity={"session": "fixture"},
            max_tensor_bytes=64, max_file_bytes=128, preallocate=True)
    assert not list(tmp_path.glob("*.pt*"))


def test_local_atomic_publication_never_overwrites_or_deletes_foreign_file(tmp_path, monkeypatch):
    real_link = os.link
    foreign = b"unrelated file appeared before publication"
    def raced_link(source, destination, **kwargs):
        Path(destination).write_bytes(foreign)
        return real_link(source, destination, **kwargs)
    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        write_exact_activation_cache_entry(
            tmp_path, "raced", torch.arange(8), identity={"session": "fixture"},
            max_tensor_bytes=64, max_file_bytes=65536, preallocate=True)
    assert (tmp_path / "raced.pt").read_bytes() == foreign
    assert not (tmp_path / "raced.pt.tmp").exists()


def test_local_write_does_not_publish_before_actual_durable_copy(tmp_path, monkeypatch):
    pub, spool = _spool(tmp_path)
    ref, local = raw_group(pub, spool)
    entered, release = threading.Event(), threading.Event()
    actual = io_engine.publish_copy
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return actual(*args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', gated)
    spool.submit('known')
    assert entered.wait(30)
    try:
        assert local.read_bytes() == b'\x00independent\xffbytes'
        assert not Path(ref.path).exists() and not pub._groups
        assert spool.durable_entries() == []
    finally:
        release.set()
        spool.join_copies()
    spool.await_group('known')
    assert Path(ref.path).read_bytes() == b'\x00independent\xffbytes'
    assert spool.durable_entries() == [ref]
    assert spool.durable_entries() == []
    assert len(read_bound_document(pub._groups['known'])['entries']) == 1


def test_incomplete_group_cannot_be_awaited_or_release_its_sources(tmp_path):
    pub, spool = _spool(tmp_path)
    ref, local = raw_group(pub, spool)
    with pytest.raises(ProducedOutputSpoolRefused, match='incomplete group'):
        spool.await_group('known')
    assert not pub._groups and local.read_bytes() == b'\x00independent\xffbytes'
    assert not spool.released('known') and not Path(ref.path).exists()


def test_capacity_release_rematerializes_exact_same_publication(tmp_path):
    pub, spool = _spool(tmp_path, window=32)
    first, local = raw_group(pub, spool, 'first', b'independent-first')
    spool.submit('first'); spool.await_group('first')
    binding = dict(pub._groups['first'])
    second, _ = raw_group(pub, spool, 'second', b'independent-second')
    assert not local.exists() and spool.released('first')
    spool.submit('second'); spool.await_group('second')
    spool.prepare_reads([first])
    with spool.local_reads([first]) as held:
        assert held[first].read_bytes() == b'independent-first'
    assert pub._groups['first'] == binding and len(pub._groups) == 2
    assert Path(second.path).read_bytes() == b'independent-second'


def test_a_live_reader_keeps_capacity_and_aliases(tmp_path):
    pub, spool = _spool(tmp_path, window=32)
    first, local = raw_group(pub, spool, 'first', b'independent-first')
    spool.submit('first'); spool.await_group('first')
    with spool.local_reads([first]) as held:
        with pytest.raises(ProducedWindowRefused, match='no releasable'):
            spool.reserve('second', 32, wait=False)
        spool.retire_entry('first', first)
        assert local.exists() and held[first].read_bytes() == b'independent-first'
    assert not local.exists() and spool.released('first')
    assert Path(first.path).read_bytes() == b'independent-first'


def test_checkpoint_and_payload_classes_survive_actual_copy(tmp_path):
    pub, spool = _spool(tmp_path)
    root = Path(pub.output_prefix)
    paths = [root / 'payload.bin', root / 'checkpoint.bin']
    pub.require_prewrite(batch_id='mixed', payload_ceiling_bytes=7,
        checkpoint_ceiling_bytes=10, paths=list(map(str, paths)))
    local = spool.reserve('mixed', 17)
    for target, content, kind in zip(paths, [b'payload', b'checkpoint'], ['payload', 'checkpoint']):
        path = local / target.name; path.write_bytes(content)
        spool.record('mixed', ProducedFileReference(str(path), path.name,
            len(content), hashlib.sha256(content).hexdigest()), root, artifact_class=kind)
    spool.submit('mixed'); spool.await_group('mixed')
    group = read_bound_document(pub._groups['mixed'])
    assert [(row['artifact_class'], Path(row['path']).read_bytes()) for row in group['entries']] == [
        ('payload', b'payload'), ('checkpoint', b'checkpoint')]
    assert spool.released('mixed')


def test_failed_export_retains_original_sources_and_error(tmp_path, monkeypatch):
    pub, spool = _spool(tmp_path)
    ref, local = raw_group(pub, spool)
    def fail(*args, **kwargs):
        raise OSError('known failed export')
    monkeypatch.setattr(io_engine, 'publish_copy', fail)
    spool.submit('known'); spool.join_copies()
    for _ in range(2):
        with pytest.raises(ProducedExportRefused, match='known failed export'):
            spool.await_group('known')
    assert local.read_bytes() == b'\x00independent\xffbytes'
    assert not Path(ref.path).exists() and not pub._groups
    report = spool.report()
    assert report['retained_ceiling_bytes'] == 18
    assert len(report['refusals']) == 1 and report['exports'][0]['publication_error']


def test_write_only_groups_publish_independently_without_later_copy_completion(tmp_path, monkeypatch):
    pub, spool = _spool(tmp_path)
    first, _ = raw_group(pub, spool, 'first', b'independent-first')
    second, _ = raw_group(pub, spool, 'second', b'independent-second')
    entered, release = threading.Event(), threading.Event()
    actual = io_engine.publish_copy
    def gated(source, *args, **kwargs):
        if Path(source).name == 'second.bin':
            entered.set()
            assert release.wait(30)
        return actual(source, *args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', gated)
    spool.submit('first', lifetime='consumed')
    spool.await_group('first')
    spool.submit('second', lifetime='consumed')
    assert entered.wait(30)
    try:
        assert Path(first.path).read_bytes() == b'independent-first'
        assert 'first' in pub._groups and 'second' not in pub._groups
    finally:
        release.set(); spool.join_copies()
    assert Path(second.path).read_bytes() == b'independent-second'


def test_unsubmitted_disposal_refuses_foreign_alias_before_any_unlink(tmp_path):
    pub, spool = _spool(tmp_path)
    _, local = raw_group(pub, spool)
    foreign = local.parent / 'foreign'; foreign.write_bytes(b'foreign bytes')
    with pytest.raises(ProducedOutputSpoolRefused, match='unknown local aliases'):
        spool.discard_unsubmitted('known')
    assert local.read_bytes() == b'\x00independent\xffbytes'
    assert foreign.read_bytes() == b'foreign bytes'


def _synced_paths(monkeypatch):
    paths = []
    actual = os.fsync
    def fsync(fd):
        paths.append(os.readlink(f'/proc/self/fd/{fd}'))
        return actual(fd)
    monkeypatch.setattr(os, 'fsync', fsync)
    return paths


def test_a_spool_entry_is_written_without_fsync(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    synced = _synced_paths(monkeypatch)
    with storage:
        for batch in range(2):
            storage.write(torch.arange(8, dtype=torch.float32), batch_index=batch, boundary_index=0)
    root = str(storage._local_output_spool.root)
    assert not [path for path in synced if path.startswith(root)]
    assert any('/.copy-' in path for path in synced)


def test_an_entry_outside_the_spool_is_still_fsynced(tmp_path, monkeypatch):
    synced = _synced_paths(monkeypatch)
    directory = tmp_path / 'canonical'
    ref = write_exact_activation_cache_entry(directory, 'entry', torch.arange(8),
        identity={'session': 'fixture'}, max_tensor_bytes=64, max_file_bytes=65536)
    assert str(ref.path) + '.tmp' in synced
    assert str(directory) in synced
