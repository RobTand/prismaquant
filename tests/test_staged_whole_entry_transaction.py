"""Whole-entry payload fences and exact release using real PB material."""
from __future__ import annotations

import errno
import hashlib
import os

import pytest

from test_strict_reader_tier_enforcement import (
    _forget_state, _leased_fixture, _pins_live, _stage_root, _stage_whole,
)


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
    staged = _stage_whole(_stage_root(tmp_path), path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {"payload": (path, staged, None)})
    events, held = [], []
    label = "checkpoint-manifest" if manifest else "shared-state"
    entry = {"name": checkpoints.CHECKPOINT_MANIFEST_NAME if manifest else "state",
             "file_bytes": len(blob) + (fault == "size-divergent"),
             "sha256": hashlib.sha256(blob).hexdigest()}
    real_open = staged_lease.LeaseWindow.open
    real_exit = staged_lease.LeaseWindow.__exit__
    real_preadv, real_pread = os.preadv, os.pread
    real_serving, real_account = resolver.record_serving_tier, resolver.record_stage_read

    def opened(window, key):
        if fault == "open-refused":
            events.append("open-refused")
            raise staged_lease.LeaseRefused("unpublished", kind="availability")
        fd, serving = real_open(window, key)
        held.append(fd)
        events.append("open")
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
        if fault == "changed-under-pin":
            before = staged.stat()
            os.utime(staged, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        return moved

    def pread(fd, size, offset):
        if fd in held and offset == len(blob) and size == 1:
            events.append("probe")
            if fault == "grew-during-read":
                return b"x"
        return real_pread(fd, size, offset)

    def exited(window, *args):
        events.append("exit")
        result = real_exit(window, *args)
        for fd in held:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert _pins_live(tmp_path, consumer) == []
        events.append("closed-and-released")
        return result

    def accounted(*args, **kwargs):
        assert events[-1] == "closed-and-released"
        events.append("account")
        return real_account(*args, **kwargs)

    monkeypatch.setattr(staged_lease.LeaseWindow, "open", opened)
    monkeypatch.setattr(staged_lease.LeaseWindow, "__exit__", exited)
    monkeypatch.setattr(resolver, "record_serving_tier", serving)
    monkeypatch.setattr(resolver, "record_stage_read", accounted)
    monkeypatch.setattr(os, "preadv", preadv)
    monkeypatch.setattr(os, "pread", pread)
    if fault is None:
        raw = checkpoints._read_shared_state_payload(path, entry)
        assert type(raw) is bytearray
        assert raw == blob
        assert events == ["open", "serving", "fill", "probe", "exit",
                          "closed-and-released", "account"]
        assert resolver.report()["bytes_from_stage"] == len(blob)
    else:
        reason = ("unpublished" if fault == "open-refused" else
                  f"{label}-changed-under-pin" if fault == "changed-before-read" else
                  f"{label}-{fault}")
        with pytest.raises(staged_lease.LeaseRefused) as failure:
            checkpoints._read_shared_state_payload(path, entry)
        assert failure.value.reason.startswith(reason)
        assert failure.value.kind == ("availability" if fault in {
            "unreadable", "open-refused"} else "integrity")
        assert events[-2:] == ["exit", "closed-and-released"]
        assert "account" not in events
        assert resolver.report()["bytes_from_stage"] == 0
    assert resolver.report()["bytes_from_pool"] == 0
    assert _pins_live(tmp_path, consumer) == []
