"""The Stage A owner's background stager (RobTand/prismaquant#895).

Measured on the first production run with read-ahead: every PrismaBuild call
ran on the thread that drives the GPU, and the GPU was busy 18 percent of the
forward pass. With a sealed window wider than two groups the owner now runs
those calls on one stager thread. These tests pin what that must and must not
change.

The first half drives ``ProducedStager`` and ``OwnerLock`` alone: order,
backpressure, close and failure routing need no queue. The second half drives
the real bound owner over a real queue, as the read-ahead tests do, and is
asynchronous on purpose: nothing here waits for a task where it is submitted.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
# Autouse, and it must apply HERE too (RobTand/prismaquant#889): it drops the
# outer PrismaBuild launch tuple before each test and deactivates the strict
# tier policy after it.
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

GROUP_SIZE = chain.GROUP_SIZE
STAGER_THREAD = "stagea-produced-stager"


def _stager_threads():
    return [thread for thread in threading.enumerate()
            if thread.name == STAGER_THREAD and thread.is_alive()]


# --------------------------------------------------------------------------
# The executor alone
# --------------------------------------------------------------------------


def _stager(**kwargs):
    from prismaquant.produced_stager import ProducedStager
    kwargs.setdefault("name", "test-stager")
    kwargs.setdefault("capacity", 8)
    return ProducedStager(**kwargs)


def test_urgent_and_ordered_work_runs_before_optional_work_in_order():
    from prismaquant.produced_stager import OPTIONAL, ORDERED, URGENT

    gate, ran = threading.Event(), []
    stager = _stager()
    try:
        # Hold the thread on one task so everything below is queued behind it.
        stager.submit(gate.wait, kind=OPTIONAL, label="hold")
        for label, kind in (("opt-1", OPTIONAL), ("opt-2", OPTIONAL),
                            ("release", ORDERED), ("read", URGENT)):
            stager.submit(lambda label=label: ran.append(label),
                          kind=kind, label=label)
        gate.set()
        assert stager.drain(10.0)
    finally:
        gate.set()
        stager.close(timeout=10.0)
    assert ran == ["release", "read", "opt-1", "opt-2"], (
        "a read never jumps a retirement ask queued before it, and both run "
        "before work nobody is waiting for", ran)


def test_a_full_optional_lane_blocks_the_submitter_until_there_is_room():
    from prismaquant.produced_stager import OPTIONAL

    gate = threading.Event()
    stager = _stager(capacity=1)
    try:
        stager.submit(gate.wait, kind=OPTIONAL, label="hold")
        stager.submit(lambda: None, kind=OPTIONAL, label="fills-the-lane")
        accepted = threading.Event()

        def third():
            stager.submit(lambda: None, kind=OPTIONAL, label="blocked")
            accepted.set()

        submitter = threading.Thread(target=third, daemon=True)
        submitter.start()
        assert not accepted.wait(0.5), "the lane is full: that is backpressure"
        gate.set()
        assert accepted.wait(10.0)
        assert stager.drain(10.0)
    finally:
        gate.set()
        stager.close(timeout=10.0)


def test_close_drops_optional_work_but_finishes_what_must_finish():
    from prismaquant.produced_stager import (
        OPTIONAL, ORDERED, StagerClosed)

    gate, ran, dropped = threading.Event(), [], []
    stager = _stager()
    stager.submit(gate.wait, kind=OPTIONAL, label="hold")
    publish = stager.submit(lambda: ran.append("publish"), kind=OPTIONAL,
                            label="publish-ahead",
                            on_drop=lambda: dropped.append("publish"))
    stager.submit(lambda: ran.append("reclaim"), kind=OPTIONAL,
                  label="reclaim-origin", keep_on_close=True)
    stager.submit(lambda: ran.append("release"), kind=ORDERED, label="release")
    closer = threading.Thread(
        target=lambda: stager.close(timeout=10.0), daemon=True)
    closer.start()
    time.sleep(0.2)
    gate.set()
    closer.join(10.0)
    assert not closer.is_alive() and not stager.alive()
    assert ran == ["release", "reclaim"] and dropped == ["publish"]
    with pytest.raises(StagerClosed):
        publish.wait(1.0)
    with pytest.raises(StagerClosed):
        stager.submit(lambda: None, kind=OPTIONAL, label="late")


def test_a_waited_failure_is_raised_as_itself_and_an_unwaited_one_is_kept():
    from prismaquant.produced_stager import ORDERED, URGENT

    class Refused(RuntimeError):
        pass

    kept = []
    stager = _stager(on_error=lambda task, exc: kept.append((task.label, exc)))
    try:
        def fail():
            raise Refused("no")

        waited = stager.submit(fail, kind=URGENT, label="read", waited=True)
        with pytest.raises(Refused):
            waited.wait(10.0)
        assert kept == [], "the waiter has it; keeping it too would raise twice"
        stager.submit(fail, kind=ORDERED, label="release")
        assert stager.drain(10.0)
    finally:
        stager.close(timeout=10.0)
    assert [(label, type(exc)) for label, exc in kept] == [("release", Refused)]


def test_wait_keys_idle_sees_queued_and_running_tasks():
    from prismaquant.produced_stager import OPTIONAL

    gate = threading.Event()
    stager = _stager()
    try:
        stager.submit(gate.wait, kind=OPTIONAL, label="hold")
        stager.submit(lambda: None, kind=OPTIONAL, label="publish-ahead",
                      keys=("group-a",))
        assert not stager.wait_keys_idle(("group-a",), timeout=0.3)
        assert stager.wait_keys_idle(("group-b",), timeout=0.3)
        assert stager.wait_keys_idle(
            ("group-a",), labels=("stage-ahead",), timeout=0.3)
        gate.set()
        assert stager.wait_keys_idle(("group-a",), timeout=10.0)
    finally:
        gate.set()
        stager.close(timeout=10.0)


def test_the_owner_lock_is_given_up_around_a_call_and_taken_back():
    from prismaquant.produced_stager import OwnerLock

    lock, seen = OwnerLock(), []

    def other():
        with lock.held():
            seen.append("other")

    with lock.held():
        with lock.held():                       # nests on its own thread
            thread = threading.Thread(target=other, daemon=True)
            thread.start()
            thread.join(0.3)
            assert seen == [], "held: bookkeeping is never torn"
            with lock.yielded():
                thread.join(10.0)
                assert seen == ["other"], "yielded: a call blocks nobody"
        thread = threading.Thread(target=other, daemon=True)
        thread.start()
        thread.join(0.3)
        assert seen == ["other"], "and it is held again afterwards"
    thread.join(10.0)
    assert seen == ["other", "other"]
    with lock.yielded():                        # not held: does nothing
        pass


# --------------------------------------------------------------------------
# The bound owner, asynchronous
# --------------------------------------------------------------------------


def _owner(tmp_path, *, groups: int, window_gib: int):
    return chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=window_gib,
        gib=max(2 * window_gib, 4), payload_max_bytes=1 << 22)


def _expected(first: int = 0, count: int = GROUP_SIZE):
    return [torch.arange(8, dtype=torch.float32) + index
            for index in range(first, first + count)]


def _read(storage, references, expected):
    with storage.prefetch(references) as window:
        for reference, want in zip(references, expected):
            assert torch.equal(storage.get(window, reference), want)


@pytest.fixture
def closing():
    """Close every owner a test made, so no stager thread outlives it."""

    owners = []
    yield owners.append
    for storage in owners:
        storage._produced_stop_stager()
    assert _stager_threads() == []


def test_the_default_window_starts_no_thread(tmp_path, closing):
    """Inert at two groups: synchronous, and exactly as it was."""

    before = len(_stager_threads())
    storage, _publication, _q, _env, _pb = chain._bound_owner(tmp_path)
    closing(storage)
    assert storage._produced_plan["window_groups"] == 2
    assert storage._stager is None
    assert len(_stager_threads()) == before
    chain._write_group(storage)
    (group,) = storage._produced_groups.values()
    assert group["published"] is None
    assert storage.telemetry["produced_stager_tasks"] == 0


def test_the_inline_override_keeps_a_wide_window_on_the_calling_thread(
        tmp_path, monkeypatch, closing):
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    monkeypatch.setenv(StreamedBoundaryArtifacts.PRODUCED_STAGER_ENV, "inline")
    storage, _publication, _q, _env, _pb = _owner(
        tmp_path, groups=1, window_gib=8)
    closing(storage)
    assert storage._stager is None
    chain._write_group(storage)
    (group,) = storage._produced_groups.values()
    assert group["published"] is not None, (
        "the synchronous read-ahead loop of #887, on the calling thread")
    assert storage.telemetry["produced_compute_blocked_publish_ahead_s"] > 0.0


def test_a_slow_publication_does_not_hold_the_writer(
        tmp_path, monkeypatch, closing):
    storage, publication, _q, _env, _pb = _owner(
        tmp_path, groups=1, window_gib=8)
    closing(storage)
    assert storage._stager is not None and len(_stager_threads()) == 1
    real, threads = publication.publish, []

    def slow_publish(**kwargs):
        threads.append(threading.current_thread().name)
        time.sleep(1.5)
        return real(**kwargs)

    monkeypatch.setattr(publication, "publish", slow_publish)
    partial = chain._write_group(storage, count=GROUP_SIZE - 1)
    started = time.monotonic()
    chain._write_group(storage, count=1, first=GROUP_SIZE - 1)
    last_write_s = time.monotonic() - started
    assert partial and last_write_s < 0.75, (
        "the write that completes a group returns without waiting for its "
        "publication", last_write_s)
    assert storage.drain_produced_stager(60.0)
    (group,) = storage._produced_groups.values()
    assert group["published"] is not None
    assert threads == [STAGER_THREAD]
    assert storage.telemetry["produced_groups_published_ahead"] == 1
    assert storage.telemetry["produced_compute_blocked_publish_ahead_s"] == 0.0
    assert storage.telemetry["produced_stager_busy_s"] >= 1.5


def test_the_next_groups_prewrite_is_claimed_ahead_of_the_writer(
        tmp_path, closing):
    storage, publication, _q, _env, _pb = _owner(
        tmp_path, groups=2, window_gib=8)
    closing(storage)
    chain._write_group(storage, count=1)
    assert storage.drain_produced_stager(60.0)
    assert len(storage._produced_groups) == 2, (
        "the first entry of a group claims the next group of its plane")
    assert storage.telemetry["produced_groups_prewritten_ahead"] == 1
    urgent = storage.telemetry["produced_stager_urgent_tasks"]
    chain._write_group(storage, count=GROUP_SIZE - 1, first=1)
    chain._write_group(storage, first=GROUP_SIZE)
    assert storage.drain_produced_stager(60.0)
    assert storage.telemetry["produced_stager_urgent_tasks"] == urgent, (
        "the writer found the second group claimed and waited for nothing")
    assert storage.telemetry["produced_groups_prewritten"] == 2
    assert len(storage._produced_groups) == 2, "and the plane ends there"


def test_a_staged_group_is_read_without_asking_the_stager(
        tmp_path, monkeypatch, closing):
    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    closing(storage)
    references = chain._write_group(storage)
    assert storage.drain_produced_stager(60.0)
    urgent = storage.telemetry["produced_stager_urgent_tasks"]
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    assert storage.telemetry["produced_group_fast_reads"] == 1
    assert storage.telemetry["produced_compute_blocked_read_fund_s"] == 0.0, (
        "published, funded and not being retired: the read waits on the "
        "mover's receipt and on nothing else")
    assert storage.telemetry["produced_stager_urgent_tasks"] == urgent + 1, (
        "only the settle was urgent")
    assert storage.telemetry["produced_compute_blocked_s"] == pytest.approx(
        sum(storage.telemetry[f"produced_compute_blocked_{reason}_s"]
            for reason in storage.PRODUCED_BLOCKED_REASONS))
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_refused_publication_is_logged_when_it_happens_and_the_read_stages_it(
        tmp_path, monkeypatch, capsys, closing):
    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    closing(storage)
    real, calls = publication.publish, []

    def refuse_once(**kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            raise RuntimeError("the fleet is busy")
        return real(**kwargs)

    monkeypatch.setattr(publication, "publish", refuse_once)
    references = chain._write_group(storage)
    assert storage.drain_produced_stager(60.0)
    assert storage.telemetry["produced_group_ahead_refusals"] == 1
    logged = capsys.readouterr().out
    assert "read-ahead publication of" in logged and "the fleet is busy" in logged
    (refusal,) = storage.produced_ahead_refusals()
    assert "the fleet is busy" in refusal["reason"]

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    assert calls == [STAGER_THREAD, STAGER_THREAD], (
        "the read's own publication runs on the stager too: the compute "
        "thread makes no PrismaBuild call that can wait on a lock", calls)
    assert storage.telemetry["produced_groups_published"] == 1
    assert storage.telemetry["produced_compute_blocked_read_fund_s"] > 0.0
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_failure_nobody_waited_for_is_raised_by_the_next_call_as_itself(
        tmp_path, monkeypatch, capsys, closing):
    class EgressWentWrong(RuntimeError):
        pass

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=2, window_gib=8)
    closing(storage)
    references = chain._write_group(storage)
    assert storage.drain_produced_stager(60.0)

    def fail(ask, wait):
        raise EgressWentWrong("the retirement ask failed")

    monkeypatch.setattr(storage, "_produced_ask_releases", fail)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())       # the exit only asks
        assert storage._stager.drain(60.0)
    assert "will be raised by the owner's next call" in capsys.readouterr().out
    with pytest.raises(EgressWentWrong) as raised:
        chain._write_group(storage, first=GROUP_SIZE)
    assert any("stager thread" in note for note in raised.value.__notes__)
    assert storage.telemetry["produced_stager_failures"] == 1
    chain._write_group(storage, first=GROUP_SIZE)     # raised once, not twice


def test_closing_drops_a_queued_publication_and_leaves_no_thread(
        tmp_path, monkeypatch, closing):
    storage, publication, _q, _env, _pb = _owner(
        tmp_path, groups=2, window_gib=8)
    closing(storage)
    gate, real = threading.Event(), publication.publish

    def held_publish(**kwargs):
        gate.wait(30.0)
        return real(**kwargs)

    monkeypatch.setattr(publication, "publish", held_publish)
    chain._write_group(storage)                       # in flight, held
    chain._write_group(storage, first=GROUP_SIZE)     # queued behind it
    closer = threading.Thread(target=storage._produced_stop_stager, daemon=True)
    closer.start()
    time.sleep(0.3)
    gate.set()
    closer.join(60.0)
    assert not closer.is_alive()
    assert storage._stager is None and _stager_threads() == []
    published = [group["published"] is not None
                 for group in storage._produced_groups.values()]
    assert published == [True, False], (
        "the publication in flight finished; the one still queued was not "
        "started for an owner that is closing", published)
    assert storage.telemetry["produced_stager_dropped"] == 1
    assert [entry["reason"] for entry in storage.produced_ahead_refusals()] == [
        "owner-closing"]
    # With no thread, the owner is the synchronous code again.
    assert storage._produced_on_compute_with_stager() is False


def test_close_releases_every_waiter_and_joins_when_a_drop_callback_raises(monkeypatch):
    from prismaquant.produced_stager import OPTIONAL, StagerClosed

    gate, entered = threading.Event(), threading.Event()
    stager = _stager()
    joins = []
    join = stager._thread.join
    def tracked_join(timeout=None):
        joins.append(timeout)
        gate.set()
        return join(timeout)
    monkeypatch.setattr(stager._thread, "join", tracked_join)
    first_error = RuntimeError("first bookkeeping failure")
    drops = []
    try:
        def hold():
            entered.set()
            assert gate.wait(10.0)
        stager.submit(hold, kind=OPTIONAL, label="hold")
        assert entered.wait(5.0)
        def bad_drop():
            drops.append("first")
            raise first_error
        def second_drop():
            drops.append("second")
            raise ValueError("second bookkeeping failure")
        first = stager.submit(lambda: None, kind=OPTIONAL, label="first",
                              on_drop=bad_drop)
        second = stager.submit(lambda: None, kind=OPTIONAL, label="second",
                               on_drop=second_drop)
        with pytest.raises(RuntimeError) as caught:
            stager.close(timeout=5.0)
        assert caught.value is first_error
        assert drops == ["first", "second"]
        assert len(joins) == 1 and 0 <= joins[0] <= 5.0
        assert not stager.alive()
        for task in (first, second):
            with pytest.raises(StagerClosed):
                task.wait(0.0)
    finally:
        gate.set()
        stager.close(timeout=10.0)
