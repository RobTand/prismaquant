"""The IO engine's ordered read stream (PQ #1294, #1291).

A stream reads an ordered sequence of files ahead of its consumer on the
engine's one pool. The caller supplies a budget and the entries, never a
depth or a worker count: how far the stream runs ahead is the budget's
headroom, and how many reads run at once follows the measured rates. Every
entry is read once, hashed once, held to its digest and decoded before the
consumer sees it, and groups are delivered whole and in order.

These tests run real files through the real read path with no residency map
bound, so every read is the declared file's.
"""
from __future__ import annotations

import hashlib
import threading
import time

import pytest
import torch

from prismaquant import io_engine
from prismaquant import memory_management as mm
from prismaquant.joint_statistics_replay import GuardReadBudget

SIZE = 4096


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


def _decode(raw, receipt, staged):
    return bytes(raw), {"staged": staged}


def _stream_files(tmp_path, groups=3, per_group=2, *, decoder=_decode):
    """``groups`` groups of ``per_group`` distinct files, one entry each."""
    entries, contents = [], {}
    for group in range(groups):
        for index in range(per_group):
            key = f"g{group}e{index}"
            data = bytes([(group * 31 + index * 7 + offset) % 251
                          for offset in range(SIZE)])
            path = tmp_path / f"{key}.bin"
            path.write_bytes(data)
            contents[key] = data
            entries.append(io_engine.ReadEntry(
                key=key, path=str(path), size=SIZE, limit=SIZE, held_bytes=SIZE,
                expected_sha256=hashlib.sha256(data).hexdigest(),
                decoder=decoder, group=group))
    return entries, contents


def _quiet(stream, timeout=30.0):
    """What the stream holds once it has nothing more it may start.

    Every read and every group gate pumps the stream again under its lock
    when it finishes, so once none is in flight nothing starts until the
    consumer takes, releases or reclaims.
    """
    deadline = time.monotonic() + timeout
    with stream._cond:
        while stream._active or stream._gating:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "the stream never went quiet"
            stream._cond.wait(remaining)
        return stream._held_actual, stream.counters["entries_read"]


def test_groups_arrive_whole_in_order_verified_and_decoded(tmp_path):
    entries, contents = _stream_files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30)
    with io_engine.read_stream(entries, budget=budget) as stream:
        with pytest.raises(RuntimeError, match="not the next in order"):
            stream.take(1)
        for group in range(3):
            delivered = stream.take(group)
            assert [item.entry.key for item in delivered] == list(stream.group_keys(group))
            for item in delivered:
                assert item.value == contents[item.entry.key]
                receipt = item.observed[0]
                assert receipt["sha256"] == item.entry.expected_sha256
                assert receipt["bytes"] == SIZE
            stream.release()
    counters = stream.counters
    assert counters["entries_read"] == 6 and counters["rereads"] == 0
    assert [taken["group"] for taken in counters["groups_taken"]] == [0, 1, 2]
    assert counters["pool_width"] == io_engine.ENGINE.width
    assert counters["per_stream_bytes_per_s"] > 0


def test_without_headroom_only_the_demanded_group_is_read(tmp_path):
    entries, contents = _stream_files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=SIZE, headroom=0)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert _quiet(stream) == (0, 0)
        assert stream.demand(0) == (2 * SIZE, 2 * SIZE)
        delivered = stream.take(0)
        assert [item.value for item in delivered] == [contents["g0e0"], contents["g0e1"]]
        # Nothing of group 1 is read until it is asked for.
        assert _quiet(stream) == (0, 2)
        assert stream.demand(1) == (2 * SIZE, 2 * SIZE)


def test_the_headroom_is_the_read_ahead_depth(tmp_path):
    """Read ahead only while the next entry fits; taking frees room again."""
    entries, _contents = _stream_files(tmp_path)
    # Each entry charges its serialized buffer plus its decoded bytes while it
    # reads, and holds its decoded bytes once read: 3 * SIZE of headroom holds
    # two entries read ahead and has no room for a third's read.
    headroom = 3 * SIZE
    budget = io_engine.FixedBudget(buffer_bytes=SIZE, headroom=headroom)
    with io_engine.read_stream(entries, budget=budget) as stream:
        held, read = _quiet(stream)
        assert (held, read) == (2 * SIZE, 2)
        assert stream.demand(0) == (0, 0)
        stream.take(0)
        held, read = _quiet(stream)
        assert (held, read) == (2 * SIZE, 4)
        assert stream.demand(1) == (0, 0)


def test_reclaim_drops_the_farthest_ahead_first_and_they_are_read_again(tmp_path):
    entries, contents = _stream_files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert _quiet(stream) == (6 * SIZE, 6)
        stream.take(0)
        # The taken group is gone; the farthest entries are dropped first.
        with stream.paused():
            assert stream.reclaim(SIZE + 1) == 2 * SIZE
            assert stream.held_bytes() == 2 * SIZE
            assert stream.counters["evictions"] == 2
        assert [item.value for item in stream.take(1)] == [contents["g1e0"], contents["g1e1"]]
        delivered = stream.take(2)
        assert [item.value for item in delivered] == [contents["g2e0"], contents["g2e1"]]
        assert stream.counters["groups_taken"][2]["reread_entries"] == 2
        assert stream.counters["rereads"] == 2


def test_a_changed_file_fails_its_group_by_name_and_is_never_decoded(tmp_path):
    decoded = []

    def recording(raw, receipt, staged):
        decoded.append(bytes(raw))
        return _decode(raw, receipt, staged)

    entries, contents = _stream_files(tmp_path, decoder=recording)
    victim = tmp_path / "g1e0.bin"
    victim.write_bytes(bytes(SIZE))  # same length, other bytes
    budget = io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert [item.value for item in stream.take(0)] == [contents["g0e0"], contents["g0e1"]]
        with pytest.raises(io_engine.EntryError, match="checksum changed") as caught:
            stream.take(1)
    assert caught.value.key == "g1e0"
    assert bytes(SIZE) not in decoded


def test_a_group_that_is_not_staged_yet_waits_for_its_consumer(tmp_path):
    """``ready`` says a group's bytes have not landed: nothing past it is read ahead."""
    entries, contents = _stream_files(tmp_path)
    asked = []

    def ready(group, cancel):
        asked.append(group)
        return group != 1

    budget = io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30)
    with io_engine.read_stream(entries, budget=budget, ready=ready) as stream:
        assert _quiet(stream) == (2 * SIZE, 2)
        assert stream.counters["ahead_deferrals"] == 1
        stream.take(0)
        assert _quiet(stream) == (0, 2)
        # Asked for, the group is read whatever ``ready`` says: the read owns
        # every check.
        assert [item.value for item in stream.take(1)] == [contents["g1e0"], contents["g1e1"]]
        stream.take(2)
    assert asked[:2] == [0, 1] and asked.count(1) == 2


def test_serialized_buffers_in_flight_stay_within_the_budget(tmp_path, monkeypatch):
    entries, _contents = _stream_files(tmp_path, groups=2, per_group=4)
    live, peaks = [], []
    lock = threading.Lock()
    original = io_engine.load_file

    def counted(path, limit, **kwargs):
        with lock:
            live.append(SIZE)
            peaks.append(sum(live))
        try:
            time.sleep(0.01)
            return original(path, limit, **kwargs)
        finally:
            with lock:
                live.pop()

    monkeypatch.setattr(io_engine, "load_file", counted)
    budget = io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30)
    with io_engine.read_stream(entries, budget=budget) as stream:
        stream.take(0)
        stream.take(1)
    assert peaks and max(peaks) <= 2 * SIZE
    assert stream.counters["peak_workers"] <= io_engine.ENGINE.width


def test_workers_follow_the_measured_rates():
    """The next group must land within the consumer's shortest measured group."""
    engine = io_engine.IOEngine()
    engine.width = 8
    entries = [io_engine.ReadEntry(key=i, path="unused", size=900, limit=900,
                                   held_bytes=900, expected_sha256=None,
                                   decoder=_decode, group=i // 4)
               for i in range(8)]
    stream = io_engine.ReadStream(
        engine, entries, io_engine.FixedBudget(buffer_bytes=8000), None)
    with stream._cond:
        # Nothing measured yet: the whole pool.
        assert stream._workers() == 8
        stream.counters["bytes_read"], stream.counters["read_s"] = 100, 1.0
        stream._busy_min_s = 10.0
        stream._took_at = time.monotonic()
        stream._cursor = 4
        # 3600 bytes at 100 B/s per read must land within 10 s: 4 reads.
        assert stream._workers() == 4
        stream._consumer_waiting = True
        assert stream._workers() == 8


def test_a_closed_stream_refuses_and_holds_nothing(tmp_path):
    entries, _contents = _stream_files(tmp_path)
    stream = io_engine.read_stream(
        entries, budget=io_engine.FixedBudget(buffer_bytes=2 * SIZE, headroom=1 << 30))
    stream.close()
    assert stream.held_bytes() == 0
    with pytest.raises(RuntimeError, match="closed"):
        stream.take(0)


def test_entries_are_checked_before_anything_is_read(tmp_path):
    entries, _contents = _stream_files(tmp_path)
    with pytest.raises(ValueError, match="not contiguous"):
        io_engine.read_stream([entries[0], entries[2], entries[1]],
                              budget=io_engine.FixedBudget(buffer_bytes=SIZE))
    with pytest.raises(ValueError, match="exceeds the budget"):
        io_engine.read_stream(entries, budget=io_engine.FixedBudget(buffer_bytes=SIZE - 1))
    with pytest.raises(ValueError, match="unique"):
        io_engine.read_stream([entries[0], entries[0]],
                              budget=io_engine.FixedBudget(buffer_bytes=SIZE))


# --------------------------------------------------------------------------
# The capture guard as a stream budget
# --------------------------------------------------------------------------

GiB = 1 << 30


def _guard(tmp_path, monkeypatch, *, cap, current, reserved, available):
    root = tmp_path / "cgroup"
    scope = root / "scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(cap))
    (scope / "memory.current").write_text(str(current))
    (scope / "memory.stat").write_text(
        "anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/scope\n")
    state = {"reserved": reserved, "available": available}
    monkeypatch.setattr(torch.cuda, "memory_reserved",
                        lambda *args, **kwargs: state["reserved"])
    monkeypatch.setattr(mm, "_host_memory_info",
                        lambda: (state["available"], 121 * GiB))
    guard = mm.CaptureMemoryGuard("cuda", device_bytes=68 * GiB, aggregate_envelope=True,
                                  cgroup_root=root, membership=membership)
    return guard, scope, state


def test_the_guard_headroom_leaves_the_phase_its_own_reservation(
        tmp_path, monkeypatch):
    """Allocating the headroom on the host keeps the phase's check passing.

    The headroom is what the host side can still take with the phase's own
    host reservation also held against the cgroup cap: the row's cgroup is
    26 GiB after the margin, 13 GiB are committed and the phase reserves
    1 GiB on the host, so 12 GiB may be read ahead. The device reservation
    lands on the device envelope and takes nothing from it.
    """
    guard, scope, _state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=80 * GiB)
    guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)
    headroom = guard.headroom_bytes()
    assert headroom == 28 * GiB - guard.margin_bytes - 13 * GiB - GiB
    # The stream's own read-ahead is in the guard's reading already.
    assert GuardReadBudget(guard, buffer_bytes=GiB).headroom_bytes(5 * GiB) == headroom
    (scope / "memory.current").write_text(str(13 * GiB + headroom))
    guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)
    assert guard.headroom_bytes() == 0
    # Past the headroom by more than the phase's host reservation, the check
    # itself refuses.
    (scope / "memory.current").write_text(str(13 * GiB + headroom + GiB + 1))
    assert guard.headroom_bytes() == -(GiB + 1)
    with pytest.raises(RuntimeError, match="refusal"):
        guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)
    assert guard.headroom_bytes() == 0  # a failed guard admits nothing more


def test_a_check_reclaims_read_ahead_bytes_before_it_would_refuse(tmp_path, monkeypatch):
    guard, scope, _state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=80 * GiB)
    limit = 28 * GiB - guard.margin_bytes
    asked = []

    def reclaim(shortfall):
        asked.append(shortfall)
        (scope / "memory.current").write_text(str(limit - GiB))
        return shortfall

    remove = guard.add_reclaimer(reclaim)
    (scope / "memory.current").write_text(str(limit + GiB))
    record = guard.check("phase", reserve_bytes=GiB)
    # The cgroup term is exceeded by the gigabyte committed past the limit.
    assert asked == [GiB]
    assert record["cgroup_current_bytes"] == limit - GiB
    remove()
    (scope / "memory.current").write_text(str(limit + GiB))
    with pytest.raises(RuntimeError, match="refusal"):
        guard.check("phase", reserve_bytes=GiB)
    assert len(asked) == 1
