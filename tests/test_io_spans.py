"""Per-phase IO spans and read-rate lines (``prismaquant.io_spans``).

The module is shared instrumentation: Stage B's quantum opens its spans, and
Stage A's split runs are to open the same ones. These tests drive it with
injected clocks and counters, so every number checked is exact.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prismaquant import io_spans
from prismaquant.io_spans import (
    IO_SPAN_MARKER,
    READ_RATE_MARKER,
    GpuPowerSpanSource,
    IoSpanLog,
    ReadRateReporter,
    failure_outcome,
    read_proc_io,
    stage_span_log,
)


class _Ticker:
    """A clock that returns the values it is given, then holds the last."""

    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def _proc(readings):
    """A ``/proc/self/io`` reader that returns each reading in turn."""
    it = iter(readings)
    return lambda: next(it)


def _io(rchar=0, read_bytes=0, wchar=0, write_bytes=0, syscr=0, syscw=0):
    return {"rchar": rchar, "wchar": wchar, "syscr": syscr, "syscw": syscw,
            "read_bytes": read_bytes, "write_bytes": write_bytes,
            "cancelled_write_bytes": 0}


def _log(lines, **kwargs):
    return IoSpanLog(scope="layer-044", log=lines.append, **kwargs)


def test_read_proc_io_parses_the_kernel_format(tmp_path):
    path = tmp_path / "io"
    path.write_text("rchar: 10\nwchar: 20\nsyscr: 3\nsyscw: 4\n"
                    "read_bytes: 4096\nwrite_bytes: 8192\n"
                    "cancelled_write_bytes: 0\n")
    assert read_proc_io(path) == {
        "rchar": 10, "wchar": 20, "syscr": 3, "syscw": 4,
        "read_bytes": 4096, "write_bytes": 8192, "cancelled_write_bytes": 0}


def test_read_proc_io_reads_this_process():
    values = read_proc_io()
    assert set(io_spans.PROC_IO_FIELDS) <= set(values)


def test_a_span_prints_one_line_with_its_deltas_and_wall_time():
    lines = []
    log = _log(lines, proc_io=_proc([_io(100, 4096, 7, 0, 2, 1),
                                     _io(1100, 8192, 9, 512, 5, 2)]),
               clock=_Ticker(10.0, 12.5), wall=_Ticker(1000.0, 1002.5))
    with log.span("checkpoint-load", entries=2048):
        pass
    (record,) = log.records
    assert record["schema"] == io_spans.IO_SPAN_SCHEMA
    assert record["scope"] == "layer-044"
    assert record["span"] == "checkpoint-load"
    assert record["parent"] is None
    assert record["entries"] == 2048
    assert record["outcome"] == "ok"
    assert record["wall_s"] == 2.5
    assert (record["start_unix"], record["end_unix"]) == (1000.0, 1002.5)
    assert record["proc_io"] == {"rchar": 1000, "wchar": 2, "syscr": 3,
                                 "syscw": 1, "read_bytes": 4096,
                                 "write_bytes": 512, "cancelled_write_bytes": 0}
    (line,) = lines
    assert line.startswith(IO_SPAN_MARKER + " ")
    assert json.loads(line[len(IO_SPAN_MARKER) + 1:]) == record


def test_spans_nest_and_name_their_parent():
    lines = []
    log = _log(lines, proc_io=lambda: _io())
    with log.span("window", window=0):
        with log.span("replay", window=0, probe=1):
            pass
    replay, window = log.records
    assert (replay["span"], replay["parent"], replay["probe"]) == ("replay", "window", 1)
    assert (window["span"], window["parent"]) == ("window", None)


def test_open_and_close_span_callbacks_that_are_not_one_block():
    log = _log([], proc_io=lambda: _io())
    handle = log.open("window", window=3)
    assert log.open_names == ["window"]
    record = log.close(handle)
    assert record["window"] == 3 and log.open_names == []
    assert log.close(handle) is None  # a second close is a no-op
    assert len(log.records) == 1


def test_a_workload_error_passes_through_and_is_the_outcome():
    log = _log([], proc_io=lambda: _io())
    with pytest.raises(ValueError, match="boom"):
        with log.span("own-source", layer=44):
            raise ValueError("boom")
    (record,) = log.records
    assert record["outcome"] == "error"
    assert record["error"] == "ValueError: boom"


def test_unreadable_counters_and_a_broken_log_never_raise():
    def unreadable():
        raise OSError("no /proc here")

    def broken_source():
        raise RuntimeError("resolver gone")

    def broken_log(line):
        raise OSError("stdout closed")

    log = IoSpanLog(scope="s", log=broken_log, proc_io=unreadable,
                    sources={"residency": broken_source})
    with log.span("head"):
        pass
    (record,) = log.records
    assert record["proc_io"] is None
    assert "no /proc here" in record["proc_io_error"]
    assert record["sources"] == {"residency": None}
    assert "resolver gone" in record["source_errors"]["residency"]


def test_close_open_marks_every_open_span_interrupted_innermost_first():
    log = _log([], proc_io=lambda: _io())
    log.open("head")
    log.open("window", window=2)
    error = RuntimeError("kernel refused")
    closed = log.close_open(error=error)
    assert [r["span"] for r in closed] == ["window", "head"]
    assert all(r["outcome"] == "interrupted" for r in closed)
    assert all(r["error"] == "RuntimeError: kernel refused" for r in closed)
    assert log.open_names == []


def test_span_attributes_may_not_replace_record_keys():
    log = _log([], proc_io=lambda: _io())
    with pytest.raises(ValueError, match="reserved"):
        log.open("window", outcome="fake")


def test_a_source_with_summarize_reports_power_against_the_envelope():
    sampler = SimpleNamespace(samples=[10.0], interval_s=1.0)
    log = _log([], proc_io=lambda: _io(),
               sources={"gpu_power": GpuPowerSpanSource(sampler)})
    with log.span("replay", window=0, probe=0):
        sampler.samples.extend([60.0, 80.0])
    power = log.records[0]["sources"]["gpu_power"]
    assert power == {"samples": 2, "joules": 140.0, "mean_w": 70.0,
                     "envelope_w": 140.0, "envelope_fraction": 0.5}


def test_a_span_shorter_than_one_power_sample_reports_no_watts():
    sampler = SimpleNamespace(samples=[10.0], interval_s=1.0)
    source = GpuPowerSpanSource(sampler)
    log = _log([], proc_io=lambda: _io(), sources={"gpu_power": source})
    with log.span("records-out"):
        pass
    assert log.records[0]["sources"]["gpu_power"] == {"samples": 0,
                                                      "envelope_w": 140.0}


def test_stage_span_log_reads_the_residency_tiers(monkeypatch):
    import prismaquant.residency_map as residency

    reports = iter([{"bytes_from_ram": 5, "bytes_from_stage": 100,
                     "bytes_from_pool": 0, "hits": 9},
                    {"bytes_from_ram": 5, "bytes_from_stage": 400,
                     "bytes_from_pool": 7, "hits": 12}])
    monkeypatch.setattr(residency, "residency_report", lambda: next(reports))
    log = stage_span_log("layer-044", log=lambda line: None)
    with log.span("window", window=0):
        pass
    tiers = log.records[0]["sources"]["residency"]
    assert tiers == {"bytes_from_ram": 0, "bytes_from_stage": 300,
                     "bytes_from_pool": 7, "fallback_count": 0,
                     "ram_fallback_count": 0}


def test_stage_span_log_without_a_map_records_none(monkeypatch):
    import prismaquant.residency_map as residency

    monkeypatch.setattr(residency, "residency_report", lambda: None)
    log = stage_span_log("q", log=lambda line: None)
    with log.span("head"):
        pass
    assert log.records[0]["sources"] == {"residency": None}


def _rates(lines):
    return [json.loads(line[len(READ_RATE_MARKER) + 1:]) for line in lines]


def test_rate_lines_every_64_entries():
    lines = []
    clock = [0.0]
    reporter = ReadRateReporter(
        "checkpoint-load", total_entries=130, total_bytes=130 * 16,
        log=lines.append, clock=lambda: clock[0], proc_io=lambda: _io())
    for _ in range(130):
        clock[0] += 0.1
        reporter.entry(16)
    reporter.done()
    rates = _rates(lines)
    assert [r["entries"] for r in rates] == [64, 128, 130]
    assert [r["final"] for r in rates] == [False, False, True]
    first = rates[0]
    assert first["label"] == "checkpoint-load"
    assert first["bytes"] == 64 * 16 and first["bytes_total"] == 130 * 16
    assert first["interval_mb_s"] == pytest.approx(64 * 16 / 6.4 / 1e6)
    # ETA is the remaining bytes over the last interval's rate.
    assert first["eta_s"] == pytest.approx((130 - 64) * 16 / (64 * 16 / 6.4))
    assert rates[-1]["eta_s"] == 0.0


def test_rate_lines_every_30_seconds_when_entries_are_slow():
    lines = []
    clock = [0.0]
    reporter = ReadRateReporter(
        "handoff-load", total_entries=10, total_bytes=10,
        log=lines.append, clock=lambda: clock[0], proc_io=lambda: _io())
    clock[0] = 29.0
    reporter.entry(1)
    assert lines == []
    clock[0] = 31.0
    reporter.entry(1)
    (rate,) = _rates(lines)
    assert rate["entries"] == 2 and rate["interval_s"] == 31.0


def test_rate_lines_carry_this_process_io_since_the_start():
    lines = []
    reporter = ReadRateReporter(
        "checkpoint-load", total_entries=1, total_bytes=1, log=lines.append,
        proc_io=_proc([_io(rchar=10, read_bytes=4096),
                       _io(rchar=110, read_bytes=8192)]))
    reporter.entry(1)
    reporter.done()
    (rate,) = _rates(lines)
    assert rate["proc_io"]["rchar"] == 100
    assert rate["proc_io"]["read_bytes"] == 4096


def test_rate_lines_report_no_prismabuild_units(monkeypatch):
    """A read into a disposable scratch is not durable work (PB #480)."""
    import prismaquant.prismabuild_progress as progress

    def refuse(*args, **kwargs):
        raise AssertionError("a rate line reported PrismaBuild units")

    monkeypatch.setattr(progress, "report", refuse)
    reporter = ReadRateReporter("checkpoint-load", total_entries=200,
                                total_bytes=200, log=lambda line: None)
    for _ in range(200):
        reporter.entry(1)
    reporter.done()
    assert reporter.lines == 4


def test_rate_line_thresholds_must_be_positive():
    with pytest.raises(ValueError):
        ReadRateReporter("x", total_entries=1, total_bytes=1, every_entries=0)


def test_failure_outcome_names_the_error_and_the_open_spans():
    outcome = failure_outcome(TypeError("missing reserve_bytes"),
                              open_spans=["head"])
    assert outcome == {"status": "failed", "error_type": "TypeError",
                       "error": "TypeError: missing reserve_bytes",
                       "open_spans": ["head"]}
