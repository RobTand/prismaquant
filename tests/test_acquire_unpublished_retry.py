"""PQ #1248: a strict bulk read rides out one retired cover.

PrismaBuild can retire a cover's fragment after the map answered and before
``acquire_for`` ran, so the window's enter refuses ``unpublished``
(availability). The reader waits for the republish, builds a NEW window and
enters again. Integrity refusals still fail clear on the first attempt, and
the pool is never read. Fakes only: no SDK is needed.
"""
import time

import pytest

from prismaquant import perturbed_x_cache as pxc
from prismaquant import residency_map, staged_lease
from prismaquant.staged_lease import LeaseRefused

PATH = "/mnt/shared/fake/stage.bin"
SIZE = 4096


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
    serving_tier = "stage"

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


def _open(resolver, **kw):
    fn = getattr(pxc, "_acquire_open_bulk_window", None)
    if fn is not None:
        return fn(PATH, "0" * 64, resolver=resolver, declared_size=SIZE, **kw)
    # Before the fix the caller composed these two calls by hand.
    window, key, staged, res = pxc._acquire_bulk_window(
        PATH, "0" * 64, resolver=resolver, declared_size=SIZE)
    fd, serving, tier = pxc._enter_and_open_window(res, window, key, PATH)
    return window, key, staged, res, fd, serving, tier


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
    monkeypatch.setattr(pxc, "_await_entry_landing",
                        lambda *a, **k: True)
    return windows, script


def test_unpublished_then_republish_serves_stage_bytes(harness):
    windows, script = harness
    script.append(LeaseRefused("unpublished", kind="availability"))
    res = FakeResolver()
    window, _key, _staged, _r, fd, _serving, tier = _open(res)
    assert fd == 7 and tier == "stage"
    assert len(windows) == 2 and windows[1] is window
    assert windows[0].entered == 1 and windows[1].entered == 1
    assert res.pool_reads == [] and res.fallbacks == []
    assert res.tiers == [(PATH, "stage")]


def test_integrity_refusal_fails_first_attempt(harness):
    windows, script = harness
    script.append(LeaseRefused("digest-mismatch", kind="integrity"))
    res = FakeResolver()
    with pytest.raises(LeaseRefused):
        _open(res)
    assert len(windows) == 1
    assert len(res.fallbacks) == 1 and res.pool_reads == []


def test_landing_timeout_records_fallback_and_raises(harness, monkeypatch):
    windows, script = harness
    script.append(LeaseRefused("unpublished", kind="availability"))
    monkeypatch.setattr(pxc, "_await_entry_landing", lambda *a, **k: False)
    res = FakeResolver()
    with pytest.raises(LeaseRefused):
        _open(res)
    assert len(windows) == 1
    assert len(res.fallbacks) == 1 and res.pool_reads == []


def test_retry_is_bounded_by_the_deadline(harness):
    windows, script = harness
    script.extend(LeaseRefused("unpublished", kind="availability")
                  for _ in range(100000))
    res = FakeResolver()
    start = time.monotonic()
    with pytest.raises(LeaseRefused):
        _open(res, deadline=start + 0.2) if hasattr(
            pxc, "_acquire_open_bulk_window") else _open(res)
    assert time.monotonic() - start < 5
    assert len(res.fallbacks) == 1 and res.pool_reads == []
