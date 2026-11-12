"""A strict bulk read refuses on the first window refusal.

The immutable manifest cannot republish an entry, so an availability or an
integrity refusal fails clear on the first attempt, and the pool is never
read. Fakes only: no SDK is needed.
"""
import time

import pytest

from prismaquant import perturbed_x_cache as pxc
from prismaquant import staged_lease
from prismaquant.staged_lease import LeaseRefused

PATH = "/mnt/shared/fake/stage.bin"


class FakeResolver:
    def __init__(self):
        self.fallbacks = []
        self.tiers = []
        self.pool_reads = []

    def staged_read(self, path, expected_sha256=None):
        return {"path": path}

    def record_fallback(self, path, reason):
        self.fallbacks.append((path, reason))

    def record_serving_tier(self, path, tier, *, pin_id="", range_ref=""):
        self.tiers.append((path, tier))

    def record_pool_read(self, path, nbytes):
        self.pool_reads.append((path, nbytes))


class FakeWindow:
    def __init__(self, refusal):
        self.refusal = refusal
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        if self.refusal is not None:
            raise self.refusal
        return self

    def open(self, key):
        return 7, {"pin_id": "p", "range_ref": "r"}

    def __exit__(self, *exc):
        self.exited += 1


def _open(resolver):
    return pxc._acquire_open_bulk_window(PATH, "0" * 64, resolver=resolver)


@pytest.fixture
def harness(monkeypatch):
    windows = []
    script = []  # one refusal (or None) per acquired window

    def fake_acquire(resolver, declared, entry):
        refusal = script.pop(0) if script else None
        w = FakeWindow(refusal)
        windows.append(w)
        return w, "key"

    monkeypatch.setattr(staged_lease, "acquire_entry_window", fake_acquire)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return windows, script


def test_unpublished_immutable_entry_refuses_without_retry(harness):
    windows, script = harness
    script.append(LeaseRefused("unpublished", kind="availability"))
    res = FakeResolver()
    with pytest.raises(LeaseRefused, match="unpublished"):
        _open(res)
    assert len(windows) == 1
    assert res.pool_reads == [] and len(res.fallbacks) == 1


def test_integrity_refusal_fails_first_attempt(harness):
    windows, script = harness
    script.append(LeaseRefused("digest-mismatch", kind="integrity"))
    res = FakeResolver()
    with pytest.raises(LeaseRefused):
        _open(res)
    assert len(windows) == 1
    assert len(res.fallbacks) == 1 and res.pool_reads == []


def test_availability_refusal_never_sleeps(harness, monkeypatch):
    windows, script = harness
    script.append(LeaseRefused("unpublished", kind="availability"))
    def forbidden_sleep(_seconds):
        raise AssertionError('immutable reader must not sleep')
    monkeypatch.setattr(time, 'sleep', forbidden_sleep)
    with pytest.raises(LeaseRefused, match='unpublished'):
        _open(FakeResolver())
    assert len(windows) == 1
