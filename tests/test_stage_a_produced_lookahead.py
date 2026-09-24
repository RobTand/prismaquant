"""Keep the GPU fed in Stage A (RobTand/prismaquant#989).

R12 left the GPU idle about 20 s of every 60 s window, for two reasons this
module pins:

1. **Head-of-line blocking.** An optional ``publish-ahead`` on the stager's
   one lane sleep-polled its group's local export receipt, so a read the
   compute thread was waiting for queued behind it. An optional task now
   looks once and gives the lane back (``produced_stager.Requeue``).
2. **No read-side lookahead.** Nothing staged the group a window reads
   before that window opened: publication at write time fills the share
   with groups read a layer later, and a plane read again after its
   retirement was restaged window by window at its read. Each open window
   now asks for the next window's groups, and write-time publication
   leaves that request room.

The first half drives ``ProducedStager`` alone. The second half drives the
real bound owner over a real queue and a real asynchronous fleet, as the
read-ahead tests do.
"""
from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
import test_produced_output_spool as spool_tests
# Autouse, and it must apply HERE too (RobTand/prismaquant#889).
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

GROUP_SIZE = chain.GROUP_SIZE
STAGER_THREAD = "stagea-produced-stager"


# --------------------------------------------------------------------------
# The executor alone
# --------------------------------------------------------------------------


def _stager(**kwargs):
    from prismaquant.produced_stager import ProducedStager
    kwargs.setdefault("name", "test-stager")
    kwargs.setdefault("capacity", 8)
    return ProducedStager(**kwargs)


def _waiting_on(event, *, delay_s=0.05, runs=None):
    """An optional step that gives the lane back until ``event`` is set."""

    from prismaquant.produced_stager import Requeue

    def step():
        if runs is not None:
            runs.append(time.monotonic())
        if not event.is_set():
            return Requeue(delay_s)
        return "landed"

    return step


def test_a_requeued_optional_task_gives_the_lane_to_an_urgent_one():
    from prismaquant.produced_stager import OPTIONAL, URGENT

    landed, started = threading.Event(), threading.Event()
    stager = _stager()
    try:
        waiting = stager.submit(_waiting_on(landed), kind=OPTIONAL,
                                label="publish-ahead", keys=("group-a",))
        time.sleep(0.2)
        read = stager.submit(started.set, kind=URGENT, label="read")
        assert started.wait(2.0), (
            "an urgent read ran while the optional step waited off the lane")
        read.wait(2.0)
        assert not waiting._done.is_set()
        # Still queued for everything that waits on the lanes.
        assert stager.pending() == 1
        assert not stager.drain(0.3)
        assert not stager.wait_keys_idle(("group-a",), timeout=0.3)
        landed.set()
        assert waiting.wait(5.0) == "landed"
        assert stager.drain(5.0)
        assert stager.wait_keys_idle(("group-a",), timeout=1.0)
    finally:
        landed.set()
        stager.close(timeout=10.0)
    assert waiting.requeues >= 1 and stager.requeues == waiting.requeues


def test_a_requeued_task_waits_its_delay_and_is_busy_only_while_it_runs():
    from prismaquant.produced_stager import OPTIONAL

    landed, runs, done = threading.Event(), [], []
    stager = _stager(on_done=done.append)
    try:
        task = stager.submit(_waiting_on(landed, delay_s=0.2, runs=runs),
                             kind=OPTIONAL, label="publish-ahead")
        time.sleep(0.7)
        landed.set()
        assert task.wait(5.0) == "landed"
        assert stager.drain(5.0)
    finally:
        landed.set()
        stager.close(timeout=10.0)
    gaps = [later - earlier for earlier, later in zip(runs, runs[1:])]
    assert gaps and min(gaps) >= 0.19, (
        "a requeued task is not taken again before its delay", gaps)
    assert done == [task], "finished once, not once per run"
    assert task.busy_s < 0.2, (
        "the deferrals are not busy time", task.busy_s)
    assert task.requeues == len(runs) - 1


def test_optional_work_queued_behind_a_requeued_task_is_not_held_by_it():
    from prismaquant.produced_stager import OPTIONAL

    landed, ran = threading.Event(), []
    stager = _stager()
    try:
        stager.submit(_waiting_on(landed, delay_s=5.0), kind=OPTIONAL,
                      label="publish-ahead")
        time.sleep(0.2)
        stager.submit(lambda: ran.append("stage-ahead"), kind=OPTIONAL,
                      label="stage-ahead")
        deadline = time.monotonic() + 2.0
        while not ran and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ran == ["stage-ahead"]
    finally:
        landed.set()
        stager.close(timeout=10.0)


def test_close_drops_a_requeued_task_with_its_callback():
    from prismaquant.produced_stager import OPTIONAL, StagerClosed

    landed, dropped = threading.Event(), []
    stager = _stager()
    task = stager.submit(_waiting_on(landed, delay_s=0.1), kind=OPTIONAL,
                         label="publish-ahead",
                         on_drop=lambda: dropped.append("publish-ahead"))
    time.sleep(0.3)
    assert stager.close(timeout=10.0)
    assert dropped == ["publish-ahead"]
    with pytest.raises(StagerClosed):
        task.wait(1.0)


def test_a_task_that_asks_to_requeue_after_close_began_is_dropped():
    """In flight when close begins: it is not put back on a closing lane."""

    from prismaquant.produced_stager import OPTIONAL, Requeue, StagerClosed

    entered, gate, dropped = threading.Event(), threading.Event(), []

    def step():
        entered.set()
        assert gate.wait(10.0)
        return Requeue(0.1)

    stager = _stager()
    task = stager.submit(step, kind=OPTIONAL, label="publish-ahead",
                         on_drop=lambda: dropped.append("publish-ahead"))
    assert entered.wait(5.0)
    closer = threading.Thread(target=lambda: stager.close(timeout=10.0),
                              daemon=True)
    closer.start()
    time.sleep(0.2)
    gate.set()
    closer.join(10.0)
    assert not closer.is_alive() and not stager.alive()
    assert dropped == ["publish-ahead"]
    with pytest.raises(StagerClosed):
        task.wait(1.0)


def test_a_kept_task_that_requeues_during_close_still_runs():
    from prismaquant.produced_stager import OPTIONAL

    landed = threading.Event()
    stager = _stager()
    task = stager.submit(_waiting_on(landed, delay_s=0.05), kind=OPTIONAL,
                         label="reclaim-origin", keep_on_close=True)
    time.sleep(0.2)
    timer = threading.Timer(0.3, landed.set)
    timer.start()
    try:
        assert stager.close(timeout=10.0)
    finally:
        timer.cancel()
        landed.set()
    assert task.wait(1.0) == "landed"


def test_a_dead_worker_strands_a_requeued_task_with_its_callback():
    from prismaquant.produced_stager import OPTIONAL, StagerClosed

    landed, dropped = threading.Event(), []
    boom = RuntimeError("bookkeeping died on the stager")

    def on_done(task):
        if task.label == "hold":
            raise boom

    stager = _stager(on_done=on_done)
    try:
        waiting = stager.submit(_waiting_on(landed, delay_s=5.0),
                                kind=OPTIONAL, label="publish-ahead",
                                on_drop=lambda: dropped.append("publish-ahead"))
        time.sleep(0.2)
        stager.submit(lambda: None, kind=OPTIONAL, label="hold")
        with pytest.raises(StagerClosed):
            waiting.wait(10.0)
    finally:
        landed.set()
    assert dropped == ["publish-ahead"]
    assert stager.death() is boom
    assert stager.stranded() == ("publish-ahead",)


def test_only_an_optional_task_may_requeue():
    from prismaquant.produced_stager import Requeue, URGENT

    stager = _stager()
    try:
        task = stager.submit(lambda: Requeue(0.1), kind=URGENT, label="read",
                             waited=True)
        with pytest.raises(TypeError, match="only an optional task"):
            task.wait(5.0)
    finally:
        stager.close(timeout=10.0)


# --------------------------------------------------------------------------
# The bound owner
# --------------------------------------------------------------------------


@pytest.fixture
def closing():
    """Close every owner a test made, so no stager thread outlives it."""

    owners = []
    yield owners.append
    for storage in owners:
        storage._produced_stop_stager()


def _expected(index):
    return torch.arange(8, dtype=torch.float32) + index


def test_a_group_held_on_this_box_is_never_published_ahead(
        tmp_path, closing):
    """Defect 1, and what PQ #1110 made of it.

    Before #1110 an optional ``publish-ahead`` waited on the stager for its
    group's local export, and PQ #989 made it give the lane back while it
    waited. A group this box still holds is now read from its local copy,
    so it is never published ahead at all: nothing waits on its export on
    the stager, and an urgent read runs at once. The one step that asks
    about the export there is the per-write look (PQ #1128), which reads its
    state once and returns.
    """

    from prismaquant.produced_output_spool import ProducedOutputSpool
    from prismaquant.produced_stager import URGENT

    storage, _publication, _q, _env, _pb = chain._bound_owner(
        tmp_path, n_batches=GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22, staging_timeout_s=30)
    closing(storage)
    assert storage._stager is not None
    # PB's asynchronous exporter is replaced by a transport whose receipt
    # the test controls; the writer and the publisher are the real ones.
    storage._published = True
    backend = spool_tests.ControlledExport(tmp_path / "local")
    storage._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=spool_tests.CapacityDeferred)
    polled_on_stager = []
    real_poll = backend.poll_group

    def poll_group(batch_id):
        if threading.current_thread().name == STAGER_THREAD:
            # The step running on the stager is the one asking.
            polled_on_stager.append(storage._stager._inflight.label)
        return real_poll(batch_id)

    backend.poll_group = poll_group
    chain._write_group(storage)
    assert storage.drain_produced_stager(30.0)
    (key, group), = storage._produced_groups.items()
    started = threading.Event()
    storage._stager.submit(started.set, kind=URGENT, label="read")
    assert started.wait(5.0)
    assert set(polled_on_stager) <= {"export-poll"}, (
        "nothing on the stager but the per-write look asks about the export "
        "of a group held here", polled_on_stager)
    assert group["published"] is None
    assert key not in storage._produced_held
    assert key not in storage._produced_ahead

    backend.acknowledge(group["batch_id"])
    assert storage.drain_produced_stager(30.0)
    assert group["published"] is None, "landing publishes nothing either"
    assert storage._local_output_spool.holds(group["batch_id"])
    telemetry = storage.telemetry
    assert telemetry["produced_groups_published_ahead"] == 0
    assert telemetry["produced_group_ahead_local_skips"] == 1
    assert telemetry["produced_group_ahead_refusals"] == 0
    assert telemetry["produced_group_ahead_export_deferrals"] == 0
    assert telemetry["produced_compute_blocked_publish_ahead_s"] == 0.0


def test_a_group_whose_export_never_lands_is_not_staged_ahead(
        tmp_path, closing):
    """No staging step waits out an export any more (PQ #1110).

    Before #1110 this ended in a counted ``export has not landed`` refusal
    once the step's budget ran out. A group held here is skipped instead,
    so there is no step to refuse; the export's own barrier is what waits
    for it (``test_stage_a_same_box_readback``).
    """

    from prismaquant.produced_output_spool import ProducedOutputSpool

    storage, _publication, _q, _env, _pb = chain._bound_owner(
        tmp_path, n_batches=GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22, staging_timeout_s=1.5)
    closing(storage)
    storage._published = True
    backend = spool_tests.ControlledExport(tmp_path / "local")
    storage._local_output_spool = ProducedOutputSpool(
        backend, capacity_deferred=spool_tests.CapacityDeferred)
    chain._write_group(storage)
    assert storage.drain_produced_stager(30.0)
    (key, group), = storage._produced_groups.items()
    assert group["published"] is None
    assert not storage._produced_held and not storage._produced_ahead
    assert storage.telemetry["produced_group_ahead_refusals"] == 0
    assert storage.produced_ahead_refusals() == []
    assert storage.telemetry["produced_group_ahead_local_skips"] == 1


def _plane(storage, groups):
    references = chain._write_group(storage, count=groups * GROUP_SIZE)
    return references, [SimpleNamespace(activations_cpu={0: reference})
                        for reference in references]


def _read_plane(storage, batches, *, then=None):
    """Read one plane window by window, as the reverse roll does.

    The stager is drained once per window, after the window opened: the
    lookahead that window asked for has then run, so what the next window
    finds is what was asked, not a race with the fleet.
    """

    from prismaquant.cost_streaming import prefetched_boundary_batches

    # ``then`` only when asked for: the parent's signature has no such
    # argument, and the red run must fail on staging, not on a keyword.
    extra = {} if then is None else {"then": then}
    with prefetched_boundary_batches(storage, batches, 0, **extra) as windows:
        for index, _batch, tensor, _incoming in windows:
            assert torch.equal(tensor, _expected(index))
            if index % GROUP_SIZE == 0:
                assert storage.drain_produced_stager(120.0)


def test_every_window_of_a_plane_wider_than_the_share_opens_staged(
        tmp_path, monkeypatch, closing):
    """Defect 2. Red on the parent: windows past the share read cold."""

    groups = 10
    storage, _publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22)
    closing(storage)
    share = storage._produced_plan["ahead_groups"]
    assert groups > share, "the plane must not fit the read-ahead share"
    _references, batches = _plane(storage, groups)
    assert storage.drain_produced_stager(60.0)
    assert storage.telemetry["produced_groups_published_ahead"] == share
    urgent = storage.telemetry["produced_stager_urgent_tasks"]
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read_plane(storage, batches)
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    telemetry = storage.telemetry
    assert telemetry["produced_group_fast_reads"] == groups, (
        "every window opens on groups staged while the window before it "
        "computed", telemetry["produced_group_fast_reads"])
    assert telemetry["produced_stager_urgent_tasks"] == urgent + 1, (
        "only the settle")
    assert telemetry["produced_compute_blocked_read_fund_s"] == 0.0
    assert telemetry["produced_read_ahead_missed"] == 0
    assert telemetry["produced_groups_read_ahead"] == groups - share
    assert telemetry["produced_groups_published"] == groups, (
        "one publication per group, whoever asked for it")
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_plane_read_again_after_retirement_opens_staged(
        tmp_path, monkeypatch, closing):
    """Defect 2, the pattern after a checkpoint: a plane read, then read again.

    Every group of the plane was read and retired, so nothing is staged
    when the second pass begins. Its first window restages at its read;
    every later window asked for the next as it opened, and opens staged.
    Red on the parent: every window of the second pass restaged at its read.
    """

    groups = 6
    storage, _publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22)
    closing(storage)
    _references, batches = _plane(storage, groups)
    assert storage.drain_produced_stager(60.0)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read_plane(storage, batches)
        assert storage.drain_produced_stager(60.0)
        first = storage.telemetry["produced_group_fast_reads"]
        _read_plane(storage, batches)
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    telemetry = storage.telemetry
    assert telemetry["produced_group_fast_reads"] - first == groups - 1, (
        "the second pass restaged each group one window ahead of its read",
        telemetry["produced_group_fast_reads"] - first)
    assert telemetry["produced_groups_rematerialized"] == groups
    assert telemetry["produced_groups_published"] == groups, (
        "a restage is never a second publication")
    assert telemetry["produced_read_ahead_missed"] == 0
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_the_last_window_of_a_pass_asks_for_the_first_of_the_next(
        tmp_path, monkeypatch, closing):
    """``then``: the next probe pass, or the next layer, opens staged too."""

    groups = 6
    storage, _publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22)
    closing(storage)
    _references, batches = _plane(storage, groups)
    assert storage.drain_produced_stager(60.0)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read_plane(storage, batches, then=(0, None))
        assert storage.drain_produced_stager(60.0)
        first = storage.telemetry["produced_group_fast_reads"]
        _read_plane(storage, batches)
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    telemetry = storage.telemetry
    assert telemetry["produced_group_fast_reads"] - first == groups
    assert telemetry["produced_groups_rematerialized"] == groups
    assert telemetry["produced_read_ahead_missed"] == 0
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_write_time_publication_leaves_the_lookahead_its_room(
        tmp_path, monkeypatch, closing):
    """The share never fills with groups read later than the next window.

    Once a window has asked for the next one, a write-time publication that
    would take the room that request needs is skipped (counted), and the
    group is staged by its own read's lookahead instead.
    """

    groups = 4
    storage, _publication, q, env, pb_repo = chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=8, gib=16,
        payload_max_bytes=1 << 22)
    closing(storage)
    share = storage._produced_plan["ahead_groups"]
    _references, batches = _plane(storage, groups)
    assert storage.drain_produced_stager(60.0)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        from prismaquant.cost_streaming import prefetched_boundary_batches
        with prefetched_boundary_batches(storage, batches, 0) as windows:
            for index, _batch, _tensor, _incoming in windows:
                if index != 0:
                    continue
                assert storage.drain_produced_stager(60.0)
                reserve = storage._produced_lookahead_reserve
                assert reserve == 2, "the next window's group and the last one's"
                # Fill what is left of the share at write time.
                for other in range(share + 2):
                    storage.write(_expected(other), batch_index=other,
                                  boundary_index=1)
                assert storage.drain_produced_stager(60.0)
                assert len(storage._produced_ahead) + reserve <= share
                assert storage.telemetry["produced_group_ahead_no_room"] >= 1
        assert storage.drain_produced_stager(60.0)
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT
