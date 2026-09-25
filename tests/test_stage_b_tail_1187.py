"""The work after a Stage B quantum's last window is spanned and truthfully logged (PQ #1187).

Stage B v7 (layer-044, PB 2063ab925db1) ran its last 796 s with no io-span:
py-spy put 4,068 of 4,116 main-thread samples in that stretch on the final
row-validation loop of ``run_layer_quantum_core``, at 12.3 W of the 140 W
envelope. PQ #1184 cut that loop to about 1.7 ms per row, so on main the
stretch is seconds. It still needs a span, so the phase table can say where
it went, and so does the runner teardown after it. Both run under the last
render phase's stall allowance, and neither commits units: the loop checks
rows, and the rows are written later, under ``records-out``.

The quantum also logged ``phase 'layer-044-chunk-000' was not declared;
committing nothing under it`` 15 times, while ``commit()`` kept reporting
under the last declared phase. The line now says once where units commit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import joint_aura
from prismaquant.io_spans import IO_SPAN_MARKER

import test_joint_cost_quantum_runtime as runtime
from test_quantum_failure_counters import _Stop, failing_head  # noqa: F401
from test_quantum_probe_identity_once_1183 import _campaign


def _validations(monkeypatch):
    """Record when each joint row is validated, from inside the quantum."""
    import time

    stamps = []
    validate = joint_aura.validate_joint_aura_entry

    def stamped(entry):
        stamps.append(time.time())
        return validate(entry)

    monkeypatch.setattr(joint_aura, "validate_joint_aura_entry", stamped)
    return stamps


def _quantum(tmp_path, monkeypatch, *, handoff_emitter=None):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    return runtime._run_quantum(
        tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
        output_root=output_root, plan_sha=runtime._hex("d"),
        prepared_sha=runtime._hex("e"), handoff_emitter=handoff_emitter)


def test_the_row_validation_after_the_last_window_runs_under_its_span(
        tmp_path, monkeypatch):
    stamps = _validations(monkeypatch)
    payload, _record, block = _quantum(tmp_path, monkeypatch)
    spans = block["io_spans"]
    rows = sum(len(per_unit) for per_unit in payload["costs"].values())
    assert rows >= 3
    tails = [span for span in spans if span["span"] == "payload"]
    assert len(tails) == 1, [span["span"] for span in spans]
    (tail,) = tails
    assert tail["parent"] is None and tail["outcome"] == "ok"
    assert tail["rows"] == rows and tail["units"] == len(payload["costs"])
    last_window = max(span["end_unix"] for span in spans if span["span"] == "window")
    assert tail["start_unix"] >= last_window
    # Every row the payload carries is checked once more inside the span:
    # that loop is the stretch v7 spent 796 s in.
    inside = [t for t in stamps if tail["start_unix"] <= t <= tail["end_unix"]]
    assert len(inside) == rows


def test_a_band_serial_handoff_is_written_under_its_own_span(tmp_path, monkeypatch):
    """The handoff writes n_probes x n_batches cotangent entries after the
    last window; they are the one sizable write in the tail."""
    calls = []

    def emitter(record, adjoint_slice, execution):
        from prismaquant.joint_replay_regime import normalize_replay_regime

        def emit(**kwargs):
            calls.append(sorted(kwargs))
            return {}
        # The core refuses an emitter built for another capture batch, so the
        # stub carries the launch regime's, as HandoffEmitter does.
        capture_batch = normalize_replay_regime(
            execution.get("replay_regime"))["capture_batch"]
        return SimpleNamespace(emit=emit, published={}, capture_batch=capture_batch)

    _payload, _record, block = _quantum(tmp_path, monkeypatch, handoff_emitter=emitter)
    assert calls, "the fixture emitter was never called"
    handoffs = [span for span in block["io_spans"] if span["span"] == "handoff-out"]
    assert len(handoffs) == 1
    assert handoffs[0]["parent"] == "payload" and handoffs[0]["outcome"] == "ok"


def _span_lines(text):
    return [json.loads(line[len(IO_SPAN_MARKER) + 1:])
            for line in text.splitlines() if line.startswith(IO_SPAN_MARKER + " {")]


def test_the_runner_teardown_has_its_span(failing_head, tmp_path, capsys):  # noqa: F811
    """``runner.shutdown()`` and ``residency_report()`` run in the quantum's
    ``finally``, between the core and ``records-out``. The failing-head
    fixture reaches that ``finally`` without a GPU."""
    run = failing_head
    config = {"model": "fixture", "inputs": {},
              "execution": {"production_act_scales": "0"}}
    with pytest.raises(_Stop):
        run.quantum.run_layer_quantum(
            config, record=run.record, adjoint_slice={}, plan_sha256="c" * 64,
            prepared={"path": str(run.prepared),
                      "sha256": hashlib.sha256(run.prepared.read_bytes()).hexdigest()},
            output_root=tmp_path)
    spans = _span_lines(capsys.readouterr().out)
    teardown = [span for span in spans if span["span"] == "teardown"]
    assert len(teardown) == 1, [span["span"] for span in spans]
    assert teardown[0]["parent"] is None and teardown[0]["outcome"] == "ok"
    # The failure document was written before the teardown, as before.
    document = json.loads(Path(run.record["output_space"]["counters"]).read_text())
    assert [span["span"] for span in document["io_spans"]] == ["head"]


def _stage_b_progress(tmp_path, monkeypatch, declared):
    from prismaquant.joint_cost_quantum import ChunkFrontier, QuantumProgress

    chunks = [{"name": "layer-001-chunk-000", "start_bytes": 0, "end_bytes": 50},
              {"name": "layer-001-chunk-001", "start_bytes": 50, "end_bytes": 100}]
    windows = [{"window_index": index, "names": [f"u{index}"], "statistics_bytes": 1,
                "render_file_upper_bound_bytes": 20, "candidate_count": 1}
               for index in range(5)]
    progress_file = tmp_path / "progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress_file))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "token")
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PHASES", json.dumps(declared))
    lines = []
    progress = QuantumProgress(frontier=ChunkFrontier(chunks=chunks, windows=windows),
                               base_units=3, log=lines.append)
    return progress, windows, lines, progress_file


def test_an_undeclared_chunk_is_logged_once_under_the_phase_that_commits(
        tmp_path, monkeypatch):
    progress, windows, lines, progress_file = _stage_b_progress(
        tmp_path, monkeypatch, ["head", "render-00"])
    progress.enter_head(0)
    progress.enter_read_phase("render-00")
    for count, window in enumerate(windows, start=1):
        progress.priced(count)
        progress.window_done(window)
        progress.commit()
    record = json.loads(progress_file.read_text())
    # What commit() did: every window's units, under the last declared phase.
    assert record["phase"] == "render-00" and record["units_completed"] == 3 + 5
    about = {name: [line for line in lines if repr(name) in line]
             for name in ("layer-001-chunk-000", "layer-001-chunk-001")}
    # Once per name, not once per window, and naming where units commit.
    assert [len(found) for found in about.values()] == [1, 1], lines
    for (found,) in about.values():
        assert "'render-00'" in found, found
        assert "committing nothing" not in found, found


def test_an_undeclared_phase_before_any_declared_one_says_nothing_commits(
        tmp_path, monkeypatch):
    progress, windows, lines, _progress_file = _stage_b_progress(
        tmp_path, monkeypatch, ["render-00"])
    progress.window_done(windows[0])
    progress.window_done(windows[1])
    assert progress.commit() is False
    assert len(lines) == 1, lines
    (line,) = lines
    assert "'layer-001-chunk-000'" in line and "until a declared phase" in line, line


def test_the_single_run_reporter_names_where_its_units_commit():
    from prismaquant.joint_run_progress import JointRunProgress

    lines = []
    reporter = JointRunProgress(layers=2, partitions=1, log=lines.append,
                                interval_s=3600, phases=["head", "layer-001"])
    reporter.enter("head")
    reporter.enter("layer-009")
    reporter.enter("layer-009")
    notes = [line for line in lines if "'layer-009'" in line]
    assert len(notes) == 1, lines
    assert "'head'" in notes[0] and "committing nothing" not in notes[0], notes[0]


def test_the_handoff_export_record_is_kept_in_the_counters(tmp_path, monkeypatch):
    """PQ #1225: the handoff's export keys, bytes, times and drain wait are
    in the row's counters, so PrismaBuild's record of each export can be
    read after its spool namespace is retired."""
    report = {"local_spool": {"exports": [
        {"batch_id": "g0", "export_key": "e" * 64, "entries": 64,
         "bytes": 1 << 30, "reserved_unix": 1.0, "submitted_unix": 2.0,
         "landed_unix": 3.0, "released_unix": 4.0, "export_wait_s": 1.0,
         "drain_wait_s": 1.0}]}}

    def emitter(record, adjoint_slice, execution):
        from prismaquant.joint_replay_regime import normalize_replay_regime

        stub = SimpleNamespace(published={}, export_report=None,
                               capture_batch=normalize_replay_regime(
                                   execution.get("replay_regime"))["capture_batch"])

        def emit(**kwargs):
            stub.export_report = report
            return {}
        stub.emit = emit
        return stub

    _payload, _record, block = _quantum(tmp_path, monkeypatch, handoff_emitter=emitter)
    assert block.get("handoff_export") == report
    json.dumps(block)
