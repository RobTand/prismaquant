"""Ownership at the stager's edges: unlink, exit, and shutdown.

The background stager (RobTand/prismaquant#895) moves PrismaBuild calls off
the compute thread, but the ORIGINS stay the owner's to dispose of: an entry
file may go only once no mover will ever open it again, and closing must
never mutate state a live worker still owns. These tests drive the real
bound owner over a real queue with the stager's work held on events, so the
interleavings are forced, not raced.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
# Autouse, and it must apply HERE too: it drops the outer PrismaBuild launch
# tuple before each test and deactivates the strict tier policy after it, so
# nothing these tests activate reaches the test that runs next.
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

GROUP_SIZE = chain.GROUP_SIZE
STAGER_THREAD = "stagea-produced-stager"


def _stager_threads():
    return [thread for thread in threading.enumerate()
            if thread.name == STAGER_THREAD and thread.is_alive()]


@pytest.fixture
def closing():
    """Stop every stager a test started, then prove none outlived it."""

    owners = []
    yield owners.append
    for storage in owners:
        storage._produced_stop_stager()
    assert _stager_threads() == []


def _wide_owner(tmp_path, *, staging_timeout_s):
    storage, publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=2 * GROUP_SIZE, window_gib=8,
        gib=16, payload_max_bytes=1 << 22,
        staging_timeout_s=staging_timeout_s)
    assert storage._produced_plan["ahead_groups"] > 0
    assert storage._stager is not None, (
        "a wide window starts the background stager")
    return storage, publication, q, env, pb_repo


def _gate_publish(storage, monkeypatch):
    """Hold the owner's real publication on an event; release to let go."""

    entered = threading.Event()
    release = threading.Event()
    orig = storage._produced.publish

    def gated(**kwargs):
        # This is the real PB-call boundary: the owner lock is yielded.
        entered.set()
        assert release.wait(20.0), "gated publication never released"
        return orig(**kwargs)

    monkeypatch.setattr(storage._produced, "publish", gated)
    return entered, release


def test_a_timed_out_publication_wait_retains_the_origin(
        tmp_path, monkeypatch, closing):
    """A publication queued behind a stalled stager still owns its origins.

    Group H's publish-ahead is queued (never ran: H is not staged-ahead)
    while the stager sits in group G's gated publication. Retiring one of
    H's entries waits out the short staging bound, which expires. The
    origin must be retained and reported -- never unlinked while the
    queued publication may still read it.
    """

    storage, _publication, _q, _env, _pb = _wide_owner(
        tmp_path, staging_timeout_s=0.1)
    closing(storage)
    entered, release = _gate_publish(storage, monkeypatch)
    try:
        g_refs = chain._write_group(storage, count=1, first=0)
        h_refs = chain._write_group(storage, count=1, first=GROUP_SIZE)
        g_refs += chain._write_group(storage, count=GROUP_SIZE - 1, first=1)
        assert entered.wait(10.0), "G's publication runs gated on the stager"
        h_refs += chain._write_group(
            storage, count=GROUP_SIZE - 1, first=GROUP_SIZE + 1)
        (h_key, h_group), = [
            (key, group) for key, group in storage._produced_groups.items()
            if key[3] == 1]
        assert h_key not in storage._produced_ahead, (
            "H's publication is queued behind G's gated one: never ran")
        with pytest.raises(TimeoutError, match="publication.*idle"):
            storage.retire(h_refs[0])
        assert Path(h_refs[0].path).exists(), (
            "the wait timed out with H's publication still queued: "
            "its origin must be retained, not unlinked")
        assert storage._references.get(h_refs[0].name) is h_refs[0]
        assert h_refs[0] in storage._slots.values()
        assert any(entry["step"] == "retain-origin-copy-unresolved"
                   for entry in storage._produced_release_errors), (
            "a retained origin is reported debt, not a silent keep")
    finally:
        release.set()

