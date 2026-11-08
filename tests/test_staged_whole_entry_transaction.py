"""Whole-entry payload fences and exact native descriptor release."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from staged_native_fixtures import bind, native_staged_state, publish_inputs
from prismaquant.residency_map import StagedReadRefused
from prismaquant.staged_tier_policy import staged_tier_policy_test_context


pytestmark = pytest.mark.usefixtures('native_staged_state')


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("fault", [
    None, "size-divergent", "changed-before-read", "truncated", "grew-during-read",
    "changed-under-pin", "unreadable", "open-refused",
])
def test_checkpoint_owned_payload_transaction(tmp_path, monkeypatch, manifest, fault):
    from prismaquant import joint_adjoint_checkpoints as checkpoints, staged_lease

    blob = b"owned-checkpoint-payload\x00\xff"
    path = tmp_path / "pool" / "payload.bin"
    path.parent.mkdir()
    path.write_bytes(blob)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [path]))
    staged = Path(resolver.staged_read(path)['stage_path'])
    staged.chmod(0o600)
    path.unlink()
    events, held = [], []
    label = "checkpoint-manifest" if manifest else "shared-state"
    entry = {"name": checkpoints.CHECKPOINT_MANIFEST_NAME if manifest else "state",
             "file_bytes": len(blob) + (fault == "size-divergent"),
             "sha256": hashlib.sha256(blob).hexdigest()}
    real_open = staged_lease.LeaseWindow.open
    real_exit = staged_lease.LeaseWindow.__exit__
    real_preadv = os.preadv
    real_serving, real_account = resolver.record_serving_tier, resolver.record_stage_read

    def opened(window, key):
        if fault == "open-refused":
            events.append("open-refused")
            raise staged_lease.LeaseRefused("unpublished", kind="availability")
        fd, serving = real_open(window, key)
        held.append(fd)
        events.append("open")
        with pytest.raises(RuntimeError, match='live reader windows'):
            resolver.close()
        if fault == "changed-before-read":
            os.truncate(staged, len(blob) - 1)
        return fd, serving

    def serving(*args, **kwargs):
        events.append("serving")
        assert kwargs["pin_id"] and kwargs["range_ref"]
        return real_serving(*args, **kwargs)

    def preadv(fd, buffers, offset, *args, **kwargs):
        if fd not in held:
            return real_preadv(fd, buffers, offset, *args, **kwargs)
        events.append("fill")
        if fault == "truncated":
            return 0
        if fault == "unreadable":
            raise OSError(errno.EIO, "injected payload I/O failure")
        moved = real_preadv(fd, buffers, offset, *args, **kwargs)
        if fault == "grew-during-read":
            with staged.open('ab') as stream:
                stream.write(b'x')
        if fault == "changed-under-pin":
            before = staged.stat()
            os.utime(staged, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        return moved

    def exited(window, *args):
        events.append("exit")
        try:
            return real_exit(window, *args)
        finally:
            for fd in held:
                with pytest.raises(OSError):
                    os.fstat(fd)
            assert resolver._windows == 0
            events.append("closed-and-released")

    def accounted(*args, **kwargs):
        assert events[-1] == "closed-and-released"
        events.append("account")
        return real_account(*args, **kwargs)

    monkeypatch.setattr(staged_lease.LeaseWindow, "open", opened)
    monkeypatch.setattr(staged_lease.LeaseWindow, "__exit__", exited)
    monkeypatch.setattr(resolver, "record_serving_tier", serving)
    monkeypatch.setattr(resolver, "record_stage_read", accounted)
    monkeypatch.setattr(os, "preadv", preadv)
    with staged_tier_policy_test_context('ssd'):
        _check_transaction(checkpoints, staged_lease, resolver, path, entry, blob,
                           events, fault, label)
    assert resolver.report()["bytes_from_pool"] == 0
    assert resolver._windows == 0


def _check_transaction(checkpoints, staged_lease, resolver, path, entry, blob,
                       events, fault, label):
    if fault is None:
        raw = checkpoints._read_shared_state_payload(path, entry)
        assert type(raw) is bytearray
        assert raw == blob
        assert events == ["open", "serving", "fill", "exit",
                          "closed-and-released", "account"]
        assert resolver.report()["bytes_from_stage"] == len(blob)
    else:
        changed = fault in {'changed-before-read', 'changed-under-pin', 'grew-during-read'}
        reason = ('object changed during reader lifetime' if changed else
                  'unpublished' if fault == 'open-refused' else f'{label}-{fault}')
        with pytest.raises(StagedReadRefused if changed else staged_lease.LeaseRefused) as failure:
            checkpoints._read_shared_state_payload(path, entry)
        assert str(failure.value).startswith(reason)
        if not changed:
            assert failure.value.kind == ("availability" if fault in {
                "unreadable", "open-refused"} else "integrity")
        assert events[-2:] == ["exit", "closed-and-released"]
        assert "account" not in events
        assert resolver.report()["bytes_from_stage"] == 0
