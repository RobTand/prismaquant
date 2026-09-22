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


@pytest.mark.parametrize("primary_failure", [False, True])
def test_a_stuck_stager_retains_origins_and_reports_debt(
        tmp_path, monkeypatch, primary_failure):
    """__exit__ with a live worker retains origins/credit and keeps primary.

    The stager thread sits gated inside group G's publication, so the
    close's join runs out its bound and the worker may still publish
    beside teardown. The exit must not unlink origins, must not release
    prewrite credit a live publication may claim, must report both as
    debt, and must let the primary failure propagate unchanged.
    """

    storage, _publication, _q, _env, _pb = _wide_owner(
        tmp_path, staging_timeout_s=1.0)
    entered, release = _gate_publish(storage, monkeypatch)
    stager = storage._stager
    close = stager.close
    monkeypatch.setattr(stager, "close",
                        lambda **kw: close(timeout=0.1))
    aborts = []
    abort = storage._produced.abort_prewrite
    def tracked_abort(**kw):
        aborts.append(kw)
        return abort(**kw)
    monkeypatch.setattr(storage._produced, "abort_prewrite", tracked_abort)
    try:
        refs = chain._write_group(storage)
        assert entered.wait(10.0), "G's publication runs gated on the stager"
        checkpoint_dir = tmp_path / "retained-checkpoint"
        payload = checkpoint_dir / "payload"
        reservation = storage.reserve_checkpoint_artifact(
            label="abandoned checkpoint", envelope_bytes=2,
            file_plan={"files": [{"name": "payload", "path": str(payload),
                                  "envelope_bytes": 1}],
                       "manifest_bytes": 1, "temp_overlap_bytes": 0,
                       "envelope_bytes": 2}, checkpoint_dir=checkpoint_dir)
        checkpoint_dir.mkdir()
        payload.write_bytes(b"x")
        storage.abandon_checkpoint_artifact(reservation)
        from prismabuild import produced_output
        prewrites = produced_output.instance_dir(
            _q.root, _publication.instance) / "prewrites"
        reserved = {path: path.read_bytes() for path in prewrites.glob("*.json")}
        assert reserved, "the actual PB prewrite reservations are held"
        live_bytes = storage.telemetry["live_artifact_bytes"]
        primary = RuntimeError("primary") if primary_failure else None
        if primary is not None:
            with pytest.raises(RuntimeError) as caught:
                with storage:
                    raise primary
            assert caught.value is primary
            assert any("retaining origins" in note for note in primary.__notes__)
        else:
            with storage:
                pass
        assert payload.read_bytes() == b"x"
        assert storage.checkpoint_reservation_state(reservation) == "retained"
        assert {path: path.read_bytes() for path in reserved} == reserved
        assert storage.telemetry["live_artifact_bytes"] == live_bytes
        assert storage._stager_stuck, "the join ran out its bound"
        assert storage._stager is stager, "retain the live worker ownership handle"
        assert stager.alive()
        assert aborts == [], "live publication still owns its prewrite"
        for ref in refs:
            assert Path(ref.path).exists(), (
                "a live worker may still publish from these origins: "
                "the stuck exit retains them")
        steps = [entry["step"] for entry in storage._produced_release_errors]
        assert "stager-close" in steps
        assert "stuck-stager-exit-retain" in steps, (
            "retained origins and unreleased credit are reported debt")
    finally:
        release.set()
    assert close(timeout=10.0)
    storage._produced_stop_stager()
    assert not storage._stager_stuck and storage._stager is None
    # Once actual termination is established, a later explicit close can
    # release the retained local resources. The failed-run path asks for
    # stage retirement without waiting on the fixture's absent movers.
    storage.__exit__(RuntimeError, RuntimeError("cleanup"), None)
    assert not checkpoint_dir.exists()
    assert all(not Path(ref.path).exists() for ref in refs)
    assert storage._references == {}
    assert _stager_threads() == []
