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

import dataclasses
import hashlib
import threading
import time

import pytest
import torch

from prismaquant import io_engine
from prismaquant import memory_management as mm
from prismaquant import joint_statistics_replay as replay
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
    """Read ahead only while the next entry fits; a release frees room, a take does not.

    A taken group's values are the consumer's until it releases them, and
    they hold memory until then (a sealed read's pages live as long as the
    value that maps them), so the stream charges them to its budget until the
    release. Counting them free at the take would read a further group into
    memory the consumer still holds.
    """
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
        taken = stream.take(0)
        # Taken, not released: nothing more fits beside the consumer's group.
        held, read = _quiet(stream)
        assert (held, read) == (0, 2)
        taken = None
        stream.release()
        held, read = _quiet(stream)
        assert (held, read) == (2 * SIZE, 4)
        assert stream.demand(1) == (0, 0)
        # The same for the next group.
        stream.take(1)
        assert _quiet(stream) == (0, 4)
        stream.release()
        assert _quiet(stream) == (2 * SIZE, 6)
    assert taken is None


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

    remove = guard.add_reclaimer(reclaim, lowers={"committed"})
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


# -- range entries and a stream read beside another (PQ #1348) ------------


def _range_entries(groups=4, held=SIZE):
    """One range entry per group; its reader returns bytes it makes itself."""
    calls = []

    def reader(group):
        calls.append(group)
        return bytes([group]) * held, ({"group": group},)

    entries = [io_engine.ReadEntry(
        key=("chunk", group), path=None, size=held, limit=held, held_bytes=held,
        expected_sha256=None, decoder=None, group=("chunk", group),
        reader=lambda group=group: reader(group)) for group in range(groups)]
    return entries, calls


def test_range_entries_are_read_by_their_reader_and_charged_their_held_bytes():
    """A range entry pins nothing and holds no serialized buffer.

    Its depth is its held bytes against the headroom, so a budget whose
    serialized-buffer bound is one byte still reads ahead, and order,
    delivery and the counters are the stream's.
    """
    entries, calls = _range_entries()
    assert all(entry.raw_bytes == 0 for entry in entries)
    budget = io_engine.FixedBudget(buffer_bytes=1, headroom=2 * SIZE)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert _quiet(stream) == (2 * SIZE, 2)
        for group in range(4):
            delivered = stream.take(("chunk", group))
            assert [item.value for item in delivered] == [bytes([group]) * SIZE]
            assert delivered[0].observed == ({"group": group},)
            stream.release()
    assert calls == [0, 1, 2, 3]
    assert stream.counters["entries_read"] == 4
    assert stream.counters["bytes_read"] == 4 * SIZE
    assert stream.counters["read_s"] > 0


def test_a_range_entry_names_no_file_and_a_file_entry_needs_one():
    entries, _calls = _range_entries(groups=1)
    budget = io_engine.FixedBudget(buffer_bytes=SIZE, headroom=0)
    for field, value in (("path", "/x"), ("decoder", _decode), ("expected_sha256", "0" * 64)):
        bad = [dataclasses.replace(entries[0], **{field: value})]
        with pytest.raises(ValueError, match="reader reads and verifies its own bytes"):
            io_engine.read_stream(bad, budget=budget)
    bare = [dataclasses.replace(entries[0], reader=None)]
    with pytest.raises(ValueError, match="needs a path and a decoder"):
        io_engine.read_stream(bare, budget=budget)


def test_a_failed_range_read_surfaces_at_its_consumer_by_key():
    def reader():
        raise OSError("short read")

    entries = [io_engine.ReadEntry(
        key="bad", path=None, size=SIZE, limit=SIZE, held_bytes=SIZE,
        expected_sha256=None, decoder=None, group=0, reader=reader)]
    with io_engine.read_stream(entries, budget=io_engine.FixedBudget(
            buffer_bytes=SIZE, headroom=0)) as stream:
        with pytest.raises(io_engine.EntryError) as caught:
            stream.take(0)
    assert caught.value.key == "bad"


def test_next_group_and_unread_bytes_read_the_stream_and_mark_nothing(tmp_path):
    entries, _contents = _stream_files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=SIZE, headroom=0)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert _quiet(stream) == (0, 0)
        assert stream.next_group() == 0
        # Two entries not read yet: each will hold its decoded bytes.
        assert stream.unread_bytes(0) == 2 * SIZE
        # Asking marks nothing: group 0 is still not demanded, so not read.
        assert _quiet(stream) == (0, 0)
        stream.take(0)
        assert stream.unread_bytes(0) == 0
        assert stream.next_group() == 1
        stream.take(1)
        stream.take(2)
        assert stream.next_group() is None
        assert stream.unread_bytes("absent") == 0


def test_a_yielding_budget_leaves_the_other_streams_next_group_its_room(tmp_path):
    """The spill replay's budget beside the render stream (PQ #1348).

    The live reading less what the render stream's next group will still
    hold, and never below the floor the phase reserved for the reader.
    """
    from types import SimpleNamespace

    entries, _contents = _stream_files(tmp_path)
    reading = {"headroom": 10 * SIZE}
    guard = SimpleNamespace(headroom_bytes=lambda: reading["headroom"])
    budget = io_engine.FixedBudget(buffer_bytes=SIZE, headroom=0)
    with io_engine.read_stream(entries, budget=budget) as renders:
        assert _quiet(renders) == (0, 0)
        spill = GuardReadBudget(guard, buffer_bytes=SIZE, yield_to=renders,
                                floor_bytes=3 * SIZE)
        # The reading already holds what the spill holds: only the render
        # stream's next group (two unread entries) comes off it.
        assert spill.headroom_bytes(5 * SIZE) == 10 * SIZE - 2 * SIZE
        reading["headroom"] = SIZE
        # Below the floor the reader keeps what its phase reserved for it.
        assert spill.headroom_bytes(0) == 3 * SIZE
        assert spill.headroom_bytes(2 * SIZE) == SIZE
        assert spill.headroom_bytes(4 * SIZE) == -SIZE
        renders.take(0)
        reading["headroom"] = 10 * SIZE
        # Group 1 is next now, still unread.
        assert spill.headroom_bytes(0) == 8 * SIZE
        renders.take(1)
        renders.take(2)
        # Every group taken: nothing to yield to.
        assert spill.headroom_bytes(0) == 10 * SIZE
    # Without a stream to yield to it is the guard's reading, as before.
    assert GuardReadBudget(guard, buffer_bytes=SIZE).headroom_bytes(7 * SIZE) == 10 * SIZE
    with pytest.raises(ValueError, match="floor"):
        GuardReadBudget(guard, buffer_bytes=SIZE, floor_bytes=-1)



def test_the_device_headroom_is_every_limit_a_cuda_allocation_lands_in(
        tmp_path, monkeypatch):
    """PQ #1348: the device side's headroom, for a cache of CUDA tensors.

    The aggregate guard: 20 GiB reserved on the device and the phase's
    4 GiB device reservation leave 44 GiB of the 68 GiB envelope; the
    aggregate envelope and the host floor are the other two limits.
    """
    guard, _scope, state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=80 * GiB)
    guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)
    limit = 28 * GiB - guard.margin_bytes
    expected = min(68 * GiB - 20 * GiB - 4 * GiB,
                   80 * GiB - guard.host_floor_bytes - GiB - 4 * GiB,
                   limit + 68 * GiB - (13 * GiB + 20 * GiB + GiB + 4 * GiB))
    assert guard.device_headroom_bytes() == expected
    state["reserved"] += expected
    assert guard.device_headroom_bytes() == 0


def test_a_device_shortfall_asks_only_device_side_reclaimers(tmp_path, monkeypatch):
    guard, _scope, state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=66 * GiB,
                                  available=80 * GiB)
    asked = {"host": [], "device": []}

    def host(shortfall):
        asked["host"].append(shortfall)
        return shortfall

    def device(shortfall):
        asked["device"].append(shortfall)
        state["reserved"] -= 6 * GiB
        return 6 * GiB

    remove_host = guard.add_reclaimer(host, lowers={"committed", "available"})
    remove_device = guard.add_reclaimer(device, lowers={"reserved"})
    # 66 GiB reserved and 4 GiB more asked of a 68 GiB envelope: 2 GiB short
    # on the device alone.
    guard.check("phase", reserve_device_bytes=4 * GiB)
    assert asked == {"host": [], "device": [2 * GiB]}
    assert guard.last["cuda_reserved_bytes"] == 60 * GiB
    remove_device()
    remove_host()
    state["reserved"] = 66 * GiB
    with pytest.raises(RuntimeError, match="device memory refusal"):
        guard.check("phase", reserve_device_bytes=4 * GiB)
    assert asked["device"] == [2 * GiB]


def test_a_host_shortfall_asks_only_host_reclaimers(tmp_path, monkeypatch):
    guard, scope, _state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=80 * GiB)
    limit = 28 * GiB - guard.margin_bytes
    asked = {"host": [], "device": []}

    def host(shortfall):
        asked["host"].append(shortfall)
        (scope / "memory.current").write_text(str(limit - GiB))
        return shortfall

    guard.add_reclaimer(lambda shortfall: asked["device"].append(shortfall) or 0,
                        lowers={"reserved"})
    guard.add_reclaimer(host, lowers={"committed", "available"})
    (scope / "memory.current").write_text(str(limit + GiB))
    guard.check("phase", reserve_bytes=GiB)
    assert asked == {"host": [GiB], "device": []}


def test_the_device_headroom_reads_the_host_pool_not_the_cuda_driver(
        tmp_path, monkeypatch):
    """On unified memory the host floor bounds a CUDA allocation (PQ #1348).

    The device envelope is loose here, so MemAvailable decides: the reading
    comes from ``/proc/meminfo`` less the host floor and the phase's
    reservations, the same term ``check`` refuses on, and the torch
    reservation only against the device envelope. The CUDA driver's own
    free-memory figure is never read.
    """
    guard, _scope, state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=30 * GiB)
    guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *args, **kwargs: pytest.fail("mem_get_info is read"))
    headroom = guard.device_headroom_bytes()
    assert headroom == 30 * GiB - guard.host_floor_bytes - GiB - 4 * GiB
    # A CUDA allocation takes MemAvailable and the torch reservation alike:
    # after one of the headroom's size the reading is zero, and the check at
    # that point still passes.
    state["available"] -= headroom
    state["reserved"] += headroom
    assert guard.device_headroom_bytes() == 0
    guard.check("phase", reserve_bytes=GiB, reserve_device_bytes=4 * GiB)


# -- the reclaim pass reads what each reclaimer freed (PQ #1383) ----------

#: prof-2 of #1348 at its refusal: the cgroup 60 MB past its cap less the
#: margin, the aggregate envelope 17 GB clear, and 31 GB more requested.
PROF2_CAP = 30064771072
PROF2_COMMITTED = 27977605120
PROF2_RESERVED = 24511512576
PROF2_REQUESTED = 31083986944
RENDER = 16 << 20


class _Held:
    """A reclaimer on the mocked guard, holding ``held`` bytes of one kind.

    ``reclaim(need)`` drops whole ``RENDER``-sized values until ``need`` is
    covered and reports them. It lowers the mocked readings named in
    ``lowers`` by what it dropped, the way those bytes do on the GB10
    (``tests/test_reclaim_readings_gb10.py``), and no other reading.
    """

    def __init__(self, scope, state, held, lowers):
        self.scope, self.state, self.held, self.lowers = scope, state, held, lowers
        self.asked = []

    def reclaim(self, need):
        self.asked.append(need)
        dropped = min(self.held, -(-need // RENDER) * RENDER)
        self.held -= dropped
        if "committed" in self.lowers:
            current = int((self.scope / "memory.current").read_text())
            (self.scope / "memory.current").write_text(str(current - dropped))
        if "reserved" in self.lowers:
            self.state["reserved"] -= dropped
        if "available" in self.lowers:
            self.state["available"] += dropped
        return dropped


def _replay_reclaimers(scope, state, *, spill=0, cache=4 * GiB, stream=4 * GiB):
    return (_Held(scope, state, spill, replay.SPILL_REPLAY_RECLAIM_LOWERS),
            _Held(scope, state, cache, replay.RENDER_CACHE_RECLAIM_LOWERS),
            _Held(scope, state, stream, replay.RENDER_STREAM_RECLAIM_LOWERS))


def _register(guard, spill, cache, stream):
    return replay.register_replay_reclaimers(
        guard, spill=type("Spill", (), {"reclaim_replay": staticmethod(spill.reclaim)})(),
        render_cache=cache, render_stream=stream)


def test_renders_kept_on_the_device_do_not_stop_a_cgroup_reclaim(tmp_path, monkeypatch):
    """prof-2's refusal (PQ #1383): a cgroup shortfall reclaims renders read ahead.

    The cgroup is 60 MB past its cap less the margin, and the device render
    cache holds gigabytes. Freeing a CUDA allocation does not lower the
    cgroup's committed bytes on the GB10, so the cache is not asked and its
    renders stay. The renders read ahead are memfd pages the cgroup counts:
    they are asked for the 60 MB, and the check passes on the reading after.
    ``main`` asked the cache first, stopped once its report covered the
    60 MB, and refused on the same committed bytes.
    """
    guard, scope, state = _guard(tmp_path, monkeypatch, cap=PROF2_CAP,
                                 current=PROF2_COMMITTED, reserved=PROF2_RESERVED,
                                 available=80 * GiB)
    spill, cache, stream = _replay_reclaimers(scope, state)
    _register(guard, spill, cache, stream)
    need = PROF2_COMMITTED - (PROF2_CAP - guard.margin_bytes)
    assert 0 < need < RENDER * 4
    record = guard.check("spill-capture", reserve_bytes=GiB,
                         reserve_device_bytes=PROF2_REQUESTED - GiB)
    assert cache.asked == [] and cache.held == 4 * GiB
    assert stream.asked == [need]
    assert record["cgroup_committed_bytes"] == PROF2_COMMITTED - 4 * RENDER
    assert [step["reclaimer"] for step in record["reclaims"]] == (
        ["spill_replay"] if "committed" in replay.SPILL_REPLAY_RECLAIM_LOWERS else []
    ) + ["render_stream"]
    assert record["reclaims"][-1]["committed_drop_bytes"] == 4 * RENDER
    assert record["reclaims"][-1]["exceeded"] == ["cgroup"]
    assert guard.reclaim_counters["render_cache"]["skipped"] == 1
    assert guard.snapshot()["reclaims"]["render_stream"]["asked"] == 1


def test_a_device_shortfall_asks_the_render_cache_alone(tmp_path, monkeypatch):
    guard, scope, state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                 current=13 * GiB, reserved=66 * GiB,
                                 available=80 * GiB)
    spill, cache, stream = _replay_reclaimers(scope, state, spill=GiB)
    _register(guard, spill, cache, stream)
    guard.check("phase", reserve_device_bytes=4 * GiB)
    assert spill.asked == [] and stream.asked == []
    assert cache.asked == [2 * GiB]
    assert guard.last["cuda_reserved_bytes"] == 64 * GiB


def test_a_reported_free_the_reading_does_not_show_does_not_end_the_pass(
        tmp_path, monkeypatch):
    """What a reclaimer reports is recorded, never counted (PQ #1383).

    The first reclaimer reports the whole shortfall and moves nothing, as a
    free the allocator keeps would (the mimalloc lesson of #1291). The pass
    reads the process again, finds the term still exceeded, and asks the
    next.
    """
    guard, scope, _state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                  current=13 * GiB, reserved=20 * GiB,
                                  available=80 * GiB)
    limit = 28 * GiB - guard.margin_bytes
    asked = []

    def kept(need):
        asked.append(("kept", need))
        return need

    def returned(need):
        asked.append(("returned", need))
        (scope / "memory.current").write_text(str(limit - GiB))
        return need

    guard.add_reclaimer(kept, lowers={"committed"}, name="kept")
    guard.add_reclaimer(returned, lowers={"committed"}, name="returned")
    (scope / "memory.current").write_text(str(limit + GiB))
    record = guard.check("phase", reserve_bytes=GiB)
    assert asked == [("kept", GiB), ("returned", GiB)]
    assert [(step["reclaimer"], step["reported_bytes"], step["committed_drop_bytes"])
            for step in record["reclaims"]] == [("kept", GiB, 0), ("returned", GiB, 2 * GiB)]


def test_a_pass_that_frees_too_little_refuses_and_says_what_it_asked(
        tmp_path, monkeypatch):
    guard, scope, state = _guard(tmp_path, monkeypatch, cap=PROF2_CAP,
                                 current=PROF2_COMMITTED, reserved=PROF2_RESERVED,
                                 available=80 * GiB)
    spill, cache, stream = _replay_reclaimers(scope, state, stream=RENDER)
    (scope / "memory.current").write_text(str(PROF2_COMMITTED + RENDER))
    _register(guard, spill, cache, stream)
    with pytest.raises(RuntimeError, match="aggregate memory refusal"):
        guard.check("spill-capture", reserve_bytes=GiB,
                    reserve_device_bytes=PROF2_REQUESTED - GiB)
    assert stream.held == 0 and cache.held == 4 * GiB
    assert guard.last["reclaims"][-1]["reclaimer"] == "render_stream"
    assert guard.last["reclaims"][-1]["committed_drop_bytes"] == RENDER


@pytest.mark.parametrize("lowers", [set(), {"committed", "rss"}])
def test_a_reclaimer_names_what_the_guard_reads(tmp_path, monkeypatch, lowers):
    guard, _scope, _state = _guard(tmp_path, monkeypatch, cap=28 * GiB,
                                   current=13 * GiB, reserved=20 * GiB,
                                   available=80 * GiB)
    with pytest.raises(ValueError, match="nonempty subset"):
        guard.add_reclaimer(lambda need: 0, lowers=lowers)
