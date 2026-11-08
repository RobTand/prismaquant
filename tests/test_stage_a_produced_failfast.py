"""Conclusive copy failure, cancellation and publication remain distinct facts."""
from pathlib import Path
import threading
import pytest
import torch
from prismaquant import io_engine
from prismaquant.produced_output_spool import ProducedExportRefused, ProducedOutputWaitCancelled
from produced_native_fixtures import owner


@pytest.mark.parametrize('cancel', [False, True])
def test_a_copy_failure_or_cancelled_wait_never_authorizes_release(tmp_path, monkeypatch, cancel):
    storage, pub = owner(tmp_path)
    copying, release = threading.Event(), threading.Event()
    def copy(*args, **kwargs):
        copying.set()
        assert release.wait(30)
        raise OSError('independent IO failure')
    monkeypatch.setattr(io_engine, 'publish_copy', copy)
    refs = [storage.write(torch.arange(8, dtype=torch.float32) + i,
        batch_index=i, boundary_index=0) for i in range(2)]
    assert copying.wait(30)
    spool = storage._local_output_spool
    group = next(iter(storage._produced_groups.values()))
    try:
        if cancel:
            spool.cancel_waits()
            with pytest.raises(ProducedOutputWaitCancelled):
                spool.await_group(group['batch_id'])
        else:
            release.set(); spool.join_copies()
            with pytest.raises(ProducedExportRefused, match='independent IO failure'):
                spool.await_group(group['batch_id'])
    finally:
        release.set(); spool.join_copies()
    assert not pub._groups and not any(Path(ref.path).exists() for ref in refs)
    assert all(Path(row[0].path).is_file() for g in spool._groups.values() for row in g['references'])
    assert len(spool.refusals) == 1 and 'independent IO failure' in spool.refusals[0]['publication_error']
    assert spool.report()['retained_ceiling_bytes'] > 0
