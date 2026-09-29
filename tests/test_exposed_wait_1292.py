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


def _report(intervals, takes=(), *, times, samples, phases, baseline_end=None):
    """``baseline_end``: the instant the row first touched the GPU."""
    return io_spans.exposed_wait_report(
        intervals, list(takes), power_times=times, power_samples=samples,
        interval_s=1.0, phase_windows=phases, envelope_w=ENVELOPE,
        baseline_end_unix=baseline_end)


def test_wait_is_split_by_power_band_per_phase():
    times, samples = _trace(100, 110, 120)
    intervals = [
        {"kind": "window-wait", "start_unix": 102.0, "end_unix": 106.0},
        {"kind": "window-wait", "start_unix": 112.0, "end_unix": 115.0},
    ]
    report = _report(intervals, times=times, samples=samples, baseline_end=105.0,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 120.0}])
    assert report["schema"] == "prismaquant.exposed_wait.v1"
    assert report["gpu_power_envelope_w"] == ENVELOPE
    assert report["idle_ceiling_w"] == 12.0
    assert report["baseline"]["n"] == 5
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
    report = _report(intervals, times=times, samples=samples, baseline_end=105.0,
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
    assert report["idle_ceiling_w"] is None
    assert report["baseline"] is None
    assert report["phases"]["p0"]["wait_s"] == pytest.approx(3.0)
    assert report["phases"]["p0"]["idle_band_s"] == 0.0
    assert report["phases"]["p0"]["unsampled_s"] == pytest.approx(3.0)


def test_overlapping_kinds_are_counted_once_and_the_overlap_reported():
    times, samples = _trace(100, 120, 130)
    intervals = [
        {"kind": "own-source", "start_unix": 102.0, "end_unix": 108.0},
        {"kind": "window-load", "start_unix": 104.0, "end_unix": 110.0},
    ]
    report = _report(intervals, times=times, samples=samples, baseline_end=105.0,
                     phases=[{"name": "p0", "start_unix": 100.0, "end_unix": 130.0}])
    phase = report["phases"]["p0"]
    assert phase["wait_s"] == pytest.approx(8.0)  # union 102..110
    assert phase["overlap_s"] == pytest.approx(4.0)
    assert phase["by_kind"]["own-source"]["wait_s"] == pytest.approx(6.0)
    assert phase["by_kind"]["window-load"]["wait_s"] == pytest.approx(6.0)


def test_wait_spanning_a_phase_boundary_is_clipped_into_each_phase():
    times, samples = _trace(100, 130, 140)
    intervals = [{"kind": "window-wait", "start_unix": 108.0, "end_unix": 114.0}]
    report = _report(intervals, times=times, samples=samples, baseline_end=105.0, phases=[
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
    report = _report(intervals, takes, times=times, samples=samples, baseline_end=105.0,
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
    counters.gpu_work_started_unix = now - 15.0  # 12 W samples before this
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
    assert wait["idle_ceiling_w"] == 12.0
    assert wait["baseline"]["n"] == 5
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


# --------------------------------------------------------------------------
# The idle ceiling is measured before the first CUDA call, never derived
# from the trace being classified
# --------------------------------------------------------------------------

_P0 = [{"name": "p0", "start_unix": 100.0, "end_unix": 200.0}]


def test_a_unimodal_busy_trace_reports_no_idle_seconds():
    # 70-90 W the whole row: the GPU was never idle. A cut through the
    # middle of a single band must not relabel half the busy seconds idle.
    times = [float(t) for t in range(100, 160)]
    samples = [70.0 + 20.0 * ((t * 7) % 10) / 9.0 for t in range(100, 160)]
    # the row's pre-work window really was idle at 5 W
    times = [float(t) for t in range(90, 100)] + times
    samples = [5.0] * 10 + samples
    intervals = [{"kind": "window-wait", "start_unix": 110.0, "end_unix": 150.0}]
    report = _report(intervals, times=times, samples=samples,
                     phases=_P0, baseline_end=100.0)
    assert report["idle_ceiling_w"] == 5.0
    assert report["total"]["wait_s"] == pytest.approx(40.0)
    assert report["total"]["idle_band_s"] == 0.0
    assert report["total"]["busy_band_s"] == pytest.approx(40.0)


def test_the_ceiling_is_the_baseline_max_even_when_the_baseline_ran_warm():
    # Even when the baseline itself reads 70-90 W (another tenant, a warm
    # device), the ceiling is what was observed: samples above it are busy.
    times = [float(t) for t in range(100, 160)]
    samples = [70.0 + (t % 5) * 5.0 for t in range(100, 160)]  # 70..90
    intervals = [{"kind": "window-wait", "start_unix": 120.0, "end_unix": 150.0}]
    report = _report(intervals, times=times, samples=samples,
                     phases=_P0, baseline_end=110.0)
    assert report["idle_ceiling_w"] == 90.0
    # nothing exceeds the ceiling, so every second is at or below it
    assert report["total"]["busy_band_s"] == 0.0
    assert report["total"]["idle_band_s"] == pytest.approx(30.0)


def test_idle_seconds_at_the_baseline_level_are_reported():
    times, samples = _trace(100, 130, 160, idle_w=12.0, busy_w=95.0)
    intervals = [
        {"kind": "window-load", "start_unix": 110.0, "end_unix": 118.0},  # idle
        {"kind": "window-load", "start_unix": 140.0, "end_unix": 145.0},  # busy
    ]
    report = _report(intervals, times=times, samples=samples,
                     phases=_P0, baseline_end=105.0)
    assert report["total"]["wait_s"] == pytest.approx(13.0)
    assert report["total"]["idle_band_s"] == pytest.approx(8.0)
    assert report["total"]["busy_band_s"] == pytest.approx(5.0)


def test_idle_ceiling_is_the_max_of_a_noisy_baseline():
    times = [float(t) for t in range(100, 130)]
    samples = [11.0, 12.5, 13.0, 11.5, 12.0] * 2 + [60.0] * 20
    # A sample's cell is half an interval either side, so the cell-aligned
    # window is 99.5..129.5: ten idle cells, twenty busy.
    intervals = [{"kind": "window-load", "start_unix": 99.5, "end_unix": 129.5}]
    report = _report(intervals, times=times, samples=samples,
                     phases=_P0, baseline_end=110.0)
    assert report["idle_ceiling_w"] == 13.0
    assert report["total"]["idle_band_s"] == pytest.approx(10.0)
    assert report["total"]["busy_band_s"] == pytest.approx(20.0)


def test_the_receipt_records_the_baseline():
    times, samples = _trace(100, 130, 160, idle_w=12.0, busy_w=95.0)
    samples[0], samples[1] = 10.0, 14.0
    report = _report([], times=times, samples=samples,
                     phases=_P0, baseline_end=104.0)
    base = report["baseline"]
    assert base["n"] == 4
    assert base["min_w"] == 10.0 and base["max_w"] == 14.0
    assert base["mean_w"] == pytest.approx((10.0 + 14.0 + 12.0 + 12.0) / 4)
    assert base["start_unix"] == 100.0 and base["end_unix"] == 103.0
    assert base["span_s"] == pytest.approx(3.0)
    assert base["window_end_unix"] == 104.0
    assert report["idle_ceiling_w"] == 14.0


def test_no_baseline_puts_the_wait_in_unsampled_never_guessed():
    times, samples = _trace(100, 130, 160)
    intervals = [{"kind": "window-load", "start_unix": 110.0, "end_unix": 120.0}]
    # no stamp at all
    stamped_none = _report(intervals, times=times, samples=samples,
                           phases=_P0, baseline_end=None)
    # a stamp with no sample before it
    stamped_early = _report(intervals, times=times, samples=samples,
                            phases=_P0, baseline_end=100.0)
    for report in (stamped_none, stamped_early):
        assert report["idle_ceiling_w"] is None
        assert report["baseline"] is None
        total = report["total"]
        assert total["wait_s"] == pytest.approx(10.0)
        assert total["idle_band_s"] == 0.0 and total["busy_band_s"] == 0.0
        assert total["unsampled_s"] == pytest.approx(10.0)


def test_a_unimodal_busy_trace_without_a_baseline_is_unsampled_not_split():
    # The coordinator's case: 70-90 W the whole row and nothing to say what
    # idle looks like on this device. A data-driven cut through the middle of
    # the band would relabel about half of these busy seconds idle; with no
    # measured ceiling the wait is unsampled and nothing is guessed.
    times = [float(t) for t in range(100, 160)]
    samples = [70.0 + 20.0 * ((t * 7) % 10) / 9.0 for t in range(100, 160)]
    intervals = [{"kind": "window-wait", "start_unix": 110.0, "end_unix": 150.0}]
    for stamp in (None, 100.0):
        report = _report(intervals, times=times, samples=samples,
                         phases=_P0, baseline_end=stamp)
        total = report["total"]
        assert report["idle_ceiling_w"] is None
        assert total["idle_band_s"] == 0.0
        assert total["busy_band_s"] == 0.0
        assert total["unsampled_s"] == pytest.approx(40.0)
