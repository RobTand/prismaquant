"""Actual copy completion precedes progress and strict exact readback."""
from pathlib import Path
import threading
import torch
from prismaquant import io_engine
from produced_native_fixtures import owner
from prismaquant.staged_tier_policy import staged_tier_policy_test_context


def test_actual_durable_copy_precedes_progress_and_strict_read(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    copying, release = threading.Event(), threading.Event()
    original = io_engine.publish_copy
    def copy(*args, **kwargs):
        copying.set()
        assert release.wait(30)
        return original(*args, **kwargs)
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    refs = [storage.write(torch.arange(8, dtype=torch.float32) + i,
        batch_index=i, boundary_index=0) for i in range(2)]
    assert copying.wait(30)
    try:
        assert not pub._groups and not storage._local_output_spool.durable_entries()
        with staged_tier_policy_test_context('ram,ssd'), storage.prefetch(refs) as window:
            for i, ref in enumerate(refs):
                assert torch.equal(storage.get(window, ref), torch.arange(8, dtype=torch.float32) + i)
    finally:
        release.set(); storage._local_output_spool.join_copies()
    assert storage._local_output_spool.durable_entries() == refs
    with storage:
        storage.settle_local_output()
    assert all(Path(ref.path).is_file() for ref in refs)
