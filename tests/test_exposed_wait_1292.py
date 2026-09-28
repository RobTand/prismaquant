"""Exposed wait: the consumer blocked on a load while the GPU idles (PQ #1292).

The pure functions get synthetic power traces and intervals whose answers
are known by hand. The stream and counters tests run the real read path.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from prismaquant import io_engine
from prismaquant import io_spans

ENVELOPE = 140.0


# --------------------------------------------------------------------------
# The idle threshold is derived from the row's own samples
# --------------------------------------------------------------------------


def test_otsu_threshold_splits_a_two_band_trace():
    samples = [12.0, 13.0, 12.5, 11.5] * 5 + [95.0, 100.0, 105.0, 98.0] * 5
    split = io_spans.otsu_idle_threshold(samples)
    assert split["threshold_w"] is not None
    assert 13.0 < split["threshold_w"] < 95.0
    assert split["low_mean_w"] == pytest.approx(12.25)
    assert split["high_mean_w"] == pytest.approx(99.5)


def test_otsu_threshold_is_none_when_the_trace_has_one_band():
    assert io_spans.otsu_idle_threshold([50.0] * 30)["threshold_w"] is None
    assert io_spans.otsu_idle_threshold([])["threshold_w"] is None
    assert io_spans.otsu_idle_threshold([40.0])["threshold_w"] is None


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def test_ledger_records_intervals_and_takes_thread_safely():
    ledger = io_spans.ExposedWaitLedger()

    def worker(base):
        for i in range(50):
            ledger.add("spill-hook", base + i, base + i + 0.5)

    threads = [threading.Thread(target=worker, args=(1000.0 * n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ledger.take("window-load", 10.0, 12.0, {"bytes": 100, "work_before_s": 1.0,
                                            "load_bytes_per_s": 50.0,
                                            "first_fill": False})
    snap = ledger.snapshot()
    assert len(snap["intervals"]) == 200 + 1
    assert len(snap["takes"]) == 1
    assert snap["takes"][0]["kind"] == "window-load"
    assert snap["takes"][0]["wait_s"] == pytest.approx(2.0)
    kinds = {row["kind"] for row in snap["intervals"]}
    assert kinds == {"spill-hook", "window-load"}


# --------------------------------------------------------------------------
# The bound is derived: bytes / measured_rate - overlappable compute
# --------------------------------------------------------------------------


def test_bound_is_zero_when_the_load_rate_covers_the_consume_rate():
    # 1000 B consumed over 2 s of work is 500 B/s; loading at 800 B/s keeps up.
    bound = io_spans.derive_wait_bound(wait_s=0.3, nbytes=1000,
                                       work_before_s=2.0, load_bytes_per_s=800.0)
    assert bound["regime"] == "load_ge_consume"
    assert bound["bound_s"] == 0.0
    assert bound["excess_s"] == pytest.approx(0.3)
    assert bound["consume_bytes_per_s"] == pytest.approx(500.0)
    assert bound["load_bytes_per_s"] == pytest.approx(800.0)


def test_bound_is_bytes_over_rate_minus_compute_when_load_is_slower():
    # 1000 B at 100 B/s takes 10 s; 2 s of it overlaps compute, so 8 s may wait.
    bound = io_spans.derive_wait_bound(wait_s=8.0, nbytes=1000,
                                       work_before_s=2.0, load_bytes_per_s=100.0)
    assert bound["regime"] == "load_lt_consume"
    assert bound["bound_s"] == pytest.approx(8.0)
    assert bound["excess_s"] == pytest.approx(0.0)
    over = io_spans.derive_wait_bound(wait_s=9.5, nbytes=1000,
                                      work_before_s=2.0, load_bytes_per_s=100.0)
    assert over["excess_s"] == pytest.approx(1.5)


def test_bound_is_unmeasured_without_a_load_rate():
    for rate in (None, 0.0):
        bound = io_spans.derive_wait_bound(wait_s=1.0, nbytes=1000,
                                           work_before_s=2.0, load_bytes_per_s=rate)
        assert bound["regime"] == "unmeasured"
        assert bound["bound_s"] is None and bound["excess_s"] is None


# --------------------------------------------------------------------------
# The report: phases, power bands, union, first fill
# --------------------------------------------------------------------------


def _trace(start, idle_until, end, *, idle_w=12.0, busy_w=100.0):
    """1 Hz samples: idle before ``idle_until``, busy after."""
    times = [float(t) for t in range(int(start), int(end))]
    return times, [idle_w if t < idle_until else busy_w for t in times]


def _report(intervals, takes=(), *, times, samples, phases, threshold=None):
    return io_spans.exposed_wait_report(
        intervals, list(takes), power_times=times, power_samples=samples,
        interval_s=1.0, phase_windows=phases, envelope_w=ENVELOPE,
        idle_threshold_w=threshold)


def test_wait_is_split_by_power_band_per_phase():
    times, samples = _trace(100, 110, 120)
    intervals = [
        {"kind": "window-wait", "start_unix": 102.0, "end_unix": 106.0},
        {"kind": "window-wait", "start_unix": 112.0, "end_unix": 115.0},
    ]
    report = _report(intervals, times=times, samples=samples,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 120.0}])
    assert report["schema"] == "prismaquant.exposed_wait.v1"
    assert report["gpu_power_envelope_w"] == ENVELOPE
    assert report["idle_threshold_source"] == "otsu"
    assert report["sample_interval_s"] == 1.0
    phase = report["phases"]["p0"]
    assert phase["wait_s"] == pytest.approx(7.0)
    assert phase["idle_band_s"] == pytest.approx(4.0)
    assert phase["busy_band_s"] == pytest.approx(3.0)
    assert phase["unsampled_s"] == pytest.approx(0.0)
    assert phase["by_kind"]["window-wait"]["wait_s"] == pytest.approx(7.0)
    assert phase["by_kind"]["window-wait"]["count"] == 2
    assert report["total"]["idle_band_s"] == pytest.approx(4.0)


def test_wait_outside_the_sampled_span_is_unsampled_not_guessed():
    times, samples = _trace(100, 110, 120)
    intervals = [{"kind": "window-wait", "start_unix": 118.0, "end_unix": 125.0}]
    report = _report(intervals, times=times, samples=samples,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 130.0}])
    phase = report["phases"]["p0"]
    assert phase["wait_s"] == pytest.approx(7.0)
    assert phase["unsampled_s"] > 0
    assert (phase["idle_band_s"] + phase["busy_band_s"] + phase["unsampled_s"]
            == pytest.approx(7.0))


def test_no_power_samples_reports_wait_unclassified():
    intervals = [{"kind": "checkpoint-load", "start_unix": 1.0, "end_unix": 4.0}]
    report = _report(intervals, times=[], samples=[],
                     phases=[{"name": "p0", "start_unix": 0.0, "end_unix": 10.0}])
    assert report["idle_threshold_source"] == "unavailable"
    assert report["idle_threshold_w"] is None
    assert report["phases"]["p0"]["wait_s"] == pytest.approx(3.0)
    assert report["phases"]["p0"]["idle_band_s"] == 0.0
    assert report["phases"]["p0"]["unsampled_s"] == pytest.approx(3.0)


def test_overlapping_kinds_are_counted_once_and_the_overlap_reported():
    times, samples = _trace(100, 120, 130)
    intervals = [
        {"kind": "own-source", "start_unix": 102.0, "end_unix": 108.0},
        {"kind": "window-load", "start_unix": 104.0, "end_unix": 110.0},
    ]
    report = _report(intervals, times=times, samples=samples,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 130.0}])
    phase = report["phases"]["p0"]
    assert phase["wait_s"] == pytest.approx(8.0)  # union 102..110
    assert phase["overlap_s"] == pytest.approx(4.0)
    assert phase["by_kind"]["own-source"]["wait_s"] == pytest.approx(6.0)
    assert phase["by_kind"]["window-load"]["wait_s"] == pytest.approx(6.0)


def test_wait_spanning_a_phase_boundary_is_clipped_into_each_phase():
    times, samples = _trace(100, 130, 140)
    intervals = [{"kind": "window-wait", "start_unix": 108.0, "end_unix": 114.0}]
    report = _report(intervals, times=times, samples=samples, phases=[
        {"name": "a", "start_unix": 100.0, "end_unix": 110.0},
        {"name": "b", "start_unix": 110.0, "end_unix": 140.0}])
    assert report["phases"]["a"]["wait_s"] == pytest.approx(2.0)
    assert report["phases"]["b"]["wait_s"] == pytest.approx(4.0)
    assert report["total"]["wait_s"] == pytest.approx(6.0)


def test_phases_never_entered_are_reported_at_zero():
    times, samples = _trace(100, 110, 120)
    report = _report([], times=times, samples=samples, phases=[
        {"name": "a", "start_unix": 100.0, "end_unix": 120.0},
        {"name": "never", "start_unix": None, "end_unix": None}])
    assert report["phases"]["never"]["wait_s"] == 0.0


def test_bound_block_exempts_the_first_fill_and_sums_the_rest():
    times, samples = _trace(100, 130, 140)
    intervals = [
        {"kind": "window-load", "start_unix": 101.0, "end_unix": 104.0},
        {"kind": "window-load", "start_unix": 110.0, "end_unix": 112.0},
        {"kind": "window-load", "start_unix": 120.0, "end_unix": 120.5},
    ]
    takes = [
        {"kind": "window-load", "start_unix": 101.0, "end_unix": 104.0,
         "wait_s": 3.0, "bytes": 1000, "work_before_s": None,
         "load_bytes_per_s": 400.0, "first_fill": True},
        # Load 100 B/s, 1000 B, 2 s compute: the bound is 8 s and the wait 2 s.
        {"kind": "window-load", "start_unix": 110.0, "end_unix": 112.0,
         "wait_s": 2.0, "bytes": 1000, "work_before_s": 2.0,
         "load_bytes_per_s": 100.0, "first_fill": False},
        # Load 800 B/s beats the 500 B/s consume rate: the bound is 0.
        {"kind": "window-load", "start_unix": 120.0, "end_unix": 120.5,
         "wait_s": 0.5, "bytes": 1000, "work_before_s": 2.0,
         "load_bytes_per_s": 800.0, "first_fill": False},
        {"kind": "window-load", "start_unix": 130.0, "end_unix": 131.0,
         "wait_s": 1.0, "bytes": 1000, "work_before_s": 2.0,
         "load_bytes_per_s": None, "first_fill": False},
    ]
    report = _report(intervals, takes, times=times, samples=samples,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 140.0}])
    bound = report["bound"]
    assert bound["takes"] == 4
    assert bound["first_fill_wait_s"] == pytest.approx(3.0)
    assert bound["steady_wait_s"] == pytest.approx(3.5)
    assert bound["unmeasured_takes"] == 1
    assert bound["bound_s"] == pytest.approx(8.0)
    assert bound["excess_s"] == pytest.approx(0.5)
    rows = bound["per_take"]
    assert [row["regime"] for row in rows] == [
        "first_fill", "load_lt_consume", "load_ge_consume", "unmeasured"]
    assert rows[2]["excess_s"] == pytest.approx(0.5)
    assert rows[2]["load_bytes_per_s"] == pytest.approx(800.0)
    assert rows[2]["consume_bytes_per_s"] == pytest.approx(500.0)


# --------------------------------------------------------------------------
# The read stream measures what the bound needs, in every row
# --------------------------------------------------------------------------

SIZE = 4096


@pytest.fixture(autouse=True)
def _no_residency_map(monkeypatch):
    monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)


def _decode(raw, receipt, staged):
    return bytes(raw), {"staged": staged}


def _files(tmp_path, groups=3, per_group=2):
    entries = []
    for group in range(groups):
        for index in range(per_group):
            data = bytes([(group * 31 + index * 7 + o) % 251 for o in range(SIZE)])
            path = tmp_path / f"g{group}e{index}.bin"
            path.write_bytes(data)
            entries.append(io_engine.ReadEntry(
                key=f"g{group}e{index}", path=str(path), size=SIZE, limit=SIZE,
                held_bytes=SIZE, expected_sha256=hashlib.sha256(data).hexdigest(),
                decoder=_decode, group=group))
    return entries


def test_read_stream_records_take_timing_and_load_rate(tmp_path):
    entries = _files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=SIZE * 4, headroom=SIZE)
    seen = []
    with io_engine.read_stream(entries, budget=budget) as stream:
        stream.wait_sink = lambda kind, start, end, info: seen.append(
            (kind, start, end, info))
        before = time.time()
        for group in range(3):
            stream.take(group)
            time.sleep(0.02)  # the consumer's work
        after = time.time()
        counters = stream.counters
    taken = counters["groups_taken"]
    assert [row["first_fill"] for row in taken] == [True, False, False]
    assert taken[0]["work_before_s"] is None
    for row in taken[1:]:
        assert row["work_before_s"] >= 0.02
    for row in taken:
        assert before - 1 <= row["at_unix"] <= after + 1
    assert counters["read_wall_s"] > 0
    assert counters["read_wall_s"] <= (after - before) + 1.0
    assert any(row["load_bytes_per_s"] and row["load_bytes_per_s"] > 0
               for row in taken)
    # The sink saw every take, with the interval the consumer waited on.
    assert len(seen) == 3
    for kind, start, end, info in seen:
        assert kind == "window-load"
        assert start <= end
        assert info["bytes"] == 2 * SIZE
    assert seen[0][3]["first_fill"] is True


def test_read_stream_without_a_sink_is_unchanged(tmp_path):
    entries = _files(tmp_path)
    budget = io_engine.FixedBudget(buffer_bytes=SIZE * 4, headroom=SIZE)
    with io_engine.read_stream(entries, budget=budget) as stream:
        assert stream.wait_sink is None
        for group in range(3):
            stream.take(group)


# --------------------------------------------------------------------------
# The row's counters carry the block, no profiler needed
# --------------------------------------------------------------------------


def _counters(tmp_path, sampler):
    from prismaquant.joint_cost_quantum import ChunkFrontier, QuantumCounters
    chunks = [{"name": "layer-001-chunk-000", "start_bytes": 0, "end_bytes": 100},
              {"name": "layer-001-chunk-001", "start_bytes": 100, "end_bytes": 200}]
    windows = [{"window_index": 0, "names": ["a"], "statistics_bytes": 1,
                "render_file_upper_bound_bytes": 100, "candidate_count": 1}]
    frontier = ChunkFrontier(chunks=chunks, windows=windows)
    return QuantumCounters(quantum_id="layer-001", identity_sha256="1" * 64,
                           chunks=chunks, frontier=frontier, sampler=sampler)


class _FakeSampler:
    def __init__(self, times, samples):
        self.times, self.samples, self.interval_s = times, samples, 1.0

    def start(self):
        return self

    def stop(self):
        return {"sample_count": len(self.samples), "interval_s": 1.0,
                "gpu_joules": 0.0, "gpu_power_w_p50": 0.0,
                "gpu_power_w_p95": 0.0, "gpu_power_w_max": 0.0}


def test_counters_carry_the_exposed_wait_block_per_phase(tmp_path):
    now = time.time()
    times = [now - 20 + t for t in range(0, 40)]
    samples = [12.0 if t < now - 5 else 100.0 for t in times]
    counters = _counters(tmp_path, _FakeSampler(times, samples))
    counters.open()
    counters.enter_phase()
    counters.exposed_wait.add("window-wait", now - 12.0, now - 10.0)
    counters.exposed_wait.take("window-load", now - 12.0, now - 10.0, {
        "bytes": 1000, "work_before_s": 1.0, "load_bytes_per_s": 100.0,
        "first_fill": False})
    block = counters.finish(units_done=1, units_total=1)
    wait = block["exposed_wait"]
    assert wait["schema"] == "prismaquant.exposed_wait.v1"
    assert "layer-001-chunk-000" in wait["phases"]
    assert wait["total"]["wait_s"] == pytest.approx(2.0)
    assert wait["total"]["idle_band_s"] == pytest.approx(2.0, abs=1.01)
    assert wait["bound"]["takes"] == 1
    text = json.dumps(block)
    assert "gpu_utilization" not in text and "utilization_percent" not in text


def test_counters_read_spans_as_wait_kinds(tmp_path):
    from prismaquant.joint_layer_quanta import CHECKPOINT_LOAD_PHASE
    counters = _counters(tmp_path, _FakeSampler([], []))
    counters.open()
    counters.enter_phase()
    with counters.io.span(CHECKPOINT_LOAD_PHASE):
        time.sleep(0.02)
    block = counters.finish(units_done=1, units_total=1)
    # The load precedes every chunk phase (the frontier has not advanced), so
    # the row total carries it and ``outside_phases_s`` names the remainder.
    total = block["exposed_wait"]["total"]
    kinds = total["by_kind"]
    assert kinds[CHECKPOINT_LOAD_PHASE]["count"] == 1
    assert kinds[CHECKPOINT_LOAD_PHASE]["wait_s"] >= 0.02
    assert total["outside_phases_s"] == pytest.approx(total["wait_s"])


# --------------------------------------------------------------------------
# The spill's waits reach the sink with their counters unchanged
# --------------------------------------------------------------------------


def test_spill_wait_lands_in_its_counter_and_the_sink():
    from prismaquant.joint_replay_spill import StageBReplaySpill

    spill = object.__new__(StageBReplaySpill)
    spill.telemetry = {"reader_wait_s": 0.0, "hook_wait_s": 0.0}
    seen = []
    spill.wait_sink = lambda kind, start, end: seen.append((kind, start, end))
    started = time.time() - 0.5
    spill._note_wait("spill-reader", started)
    spill._note_wait("spill-hook", started)
    assert [kind for kind, _, _ in seen] == ["spill-reader", "spill-hook"]
    assert all(end - start >= 0.5 for _, start, end in seen)
    assert spill.telemetry["reader_wait_s"] >= 0.5
    assert spill.telemetry["hook_wait_s"] >= 0.5
    spill.wait_sink = None
    spill._note_wait("spill-hook", started)  # no sink: the counter still counts
    assert len(seen) == 2
