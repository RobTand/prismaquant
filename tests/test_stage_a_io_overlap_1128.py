"""Existing IO futures publish copied groups independently of the writer."""
import threading
import pytest
from prismaquant import io_engine
from prismaquant.produced_output_spool import ProducedExportRefused
import test_stage_a_same_box_readback as readback


def _cotangent_groups(storage, groups):
    return [storage.write(readback._tail(0, b), batch_index=b,
        boundary_index=readback.TOP, probe_index=0) for b in range(groups * readback.GROUP_SIZE)]


def test_the_writer_submits_existing_io_and_progress_follows_durable_bytes(tmp_path, monkeypatch):
    storage, pub = readback._spool_owner(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    writer = threading.get_ident()
    copies, progress = [], []
    class Progress:
        def entry(self, **kwargs):
            progress.append(kwargs)
    storage.watch_progress(Progress())
    original = io_engine.publish_copy
    def copy(*args, **kwargs):
        copies.append(threading.get_ident())
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    refs = _cotangent_groups(storage, 2)
    try:
        assert entered.wait(30)
        assert not progress and not pub._groups
        assert all(thread != writer for thread in copies)
    finally:
        release.set(); storage._local_output_spool.join_copies()
    with storage:
        storage.settle_local_output()
    assert len(copies) == len(refs) == 2 * readback.GROUP_SIZE
    assert sorted(progress, key=lambda row: row['partition']) == [
        {'layer': readback.TOP, 'partition': b, 'kind': 'cotangent'}
        for b in range(2 * readback.GROUP_SIZE)]


def test_an_asynchronous_copy_failure_surfaces_at_the_next_owner_call_once(tmp_path, monkeypatch):
    storage, _ = readback._spool_owner(tmp_path, monkeypatch)
    def copy(*args, **kwargs):
        raise OSError('fixture: exact copy failure')
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    _cotangent_groups(storage, 1)
    storage._local_output_spool.join_copies()
    with pytest.raises(ProducedExportRefused, match='exact copy failure'):
        storage.write(readback._tail(0, readback.GROUP_SIZE), batch_index=readback.GROUP_SIZE,
            boundary_index=readback.TOP, probe_index=0)
    with pytest.raises(ProducedExportRefused, match='exact copy failure'):
        storage.settle_local_output()
    records = storage.produced_output_report()['local_spool']['refusals']
    assert len(records) == 1 and 'exact copy failure' in records[0]['publication_error']
