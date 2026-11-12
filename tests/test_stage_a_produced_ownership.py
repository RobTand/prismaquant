"""Closing cannot outrun owned copies, and failures preserve their evidence."""
from pathlib import Path
import threading
import pytest
import torch
from prismaquant import io_engine
from produced_native_fixtures import owner


@pytest.mark.parametrize('failing', [False, True])
def test_close_joins_live_publication_before_releasing_sources(tmp_path, monkeypatch, failing):
    storage, pub = owner(tmp_path)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    copying, release, joined, finished = (threading.Event() for _ in range(4))
    actual_copy = io_engine.publish_copy
    def copy(*args, **kwargs):
        copying.set()
        assert release.wait(30)
        return actual_copy(*args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    refs = [storage.write(torch.arange(8, dtype=torch.float32) + i,
        batch_index=i, boundary_index=0) for i in range(2)]
    assert copying.wait(30)
    spool = storage._local_output_spool
    original_join = spool.join_copies
    def join():
        joined.set()
        original_join()
    monkeypatch.setattr(spool, 'join_copies', join)
    errors = []
    original = RuntimeError('independent producer failure')
    def close():
        try:
            storage.__exit__(RuntimeError if failing else None, original if failing else None, None)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    worker = threading.Thread(target=close)
    worker.start()
    try:
        assert joined.wait(30)
        assert not finished.is_set() and not pub._groups
        assert all(Path(row[0].path).is_file() for group in spool._groups.values()
            for row in group['references'])
    finally:
        release.set(); worker.join(30)
    assert not worker.is_alive() and finished.is_set() and not errors
    assert all(Path(ref.path).is_file() for ref in refs)
    assert all(group['released'] for group in spool._groups.values())
    assert storage._status == ('failed' if failing else 'complete')


def test_retirement_inside_live_window_defers_canonical_unlinks(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        refs = [storage.write(torch.arange(8, dtype=torch.float32) + i,
            batch_index=i, boundary_index=0) for i in range(2)]
        storage._local_output_spool.join_copies()
        with storage.prefetch(refs) as window:
            for i, ref in enumerate(refs):
                tensor = storage.get(window, ref)
                storage._retire(ref)
                assert torch.equal(tensor, torch.arange(8, dtype=torch.float32) + i)
                with pytest.raises(RuntimeError, match='stale or belongs to another'):
                    storage.get(window, ref)
            assert all(Path(ref.path).is_file() for ref in refs)
            assert storage._deferred_unlinks
        storage._produced_flush_deferred_unlinks()
        assert all(not Path(ref.path).exists() for ref in refs)
        assert not storage._deferred_unlinks


def test_partial_group_retirement_keeps_every_origin_for_rematerialization(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        refs = [storage.write(torch.arange(8, dtype=torch.float32) + i,
            batch_index=i, boundary_index=0) for i in range(2)]
        storage._local_output_spool.join_copies()
        storage._retire(refs[0])
        storage._local_output_spool.release_landed()
        storage._produced_flush_deferred_unlinks()
        assert all(Path(ref.path).is_file() for ref in refs)
        with storage.prefetch([refs[1]]) as window:
            assert torch.equal(storage.get(window, refs[1]), torch.arange(8, dtype=torch.float32) + 1)
        storage._retire(refs[1])
        storage._produced_flush_deferred_unlinks()
        assert all(not Path(ref.path).exists() for ref in refs)
