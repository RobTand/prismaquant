"""PACT replay stops on silence, not total duration."""
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"
sys.path.insert(0, str(REPLAY_ROOT))
import multi_stream_replay as replay


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_main_completes_past_3500_with_steady_progress(tmp_path, monkeypatch):
    """Exercise the command's supervision around a short replay window."""
    monkeypatch.syspath_prepend(str(REPLAY_ROOT / "accepted"))
    import time
    import band_continuation
    import band_replay
    import energy_inputs
    import g3_residency
    import mixed_injection
    import stage1a_energy
    import prismaquant.cost_streaming
    import prismaquant.model_profiles

    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    args = SimpleNamespace(
        device="cpu", require_staged_inputs=False, prepare_readset=None,
        boundaries=tmp_path / "receipt.json", stream_manifest=tmp_path / "roster.json",
        teacher_frontier=tmp_path / "teacher.json", teacher_content=tmp_path / "content.json",
        teacher_content_sha256="a" * 64, energy_rows=tmp_path / "energy.jsonl",
        output=tmp_path / "energy-out.jsonl", run_manifest_out=tmp_path / "run.json",
        output_dir=tmp_path / "streams", dry_run_cpu=False, startup_need_gib=3,
        rendered_resident_budget_gib=1, source_root=tmp_path, boundary_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints", band="39:42", deadline_seconds=3500,
        stall_seconds=1800,
    )
    for path in (args.boundaries, args.stream_manifest, args.teacher_frontier, args.teacher_content):
        path.write_text("{}")
    monkeypatch.setattr(replay, "cli_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    run = dict(band_start=39, band_stop=42, replay_start=39, replay_stop=45,
               window_start=39, window_stop=42, window_layers=3, plan=[],
               manifest_streams=[], generations={})
    monkeypatch.setattr(replay, "resolve_cli_run", lambda *a: run)
    monkeypatch.setattr(replay, "supervise", lambda work, allowance: work(lambda event: None), raising=False)
    monkeypatch.setattr(stage1a_energy, "setup", lambda a: (None, {"token_sha256": "b" * 64}, [], []))
    monkeypatch.setattr(stage1a_energy, "available", lambda: 10 * 2**30)
    monkeypatch.setattr(stage1a_energy, "guard", lambda label: None)
    monkeypatch.setattr(g3_residency, "read_file", lambda path: path.read_bytes())
    monkeypatch.setattr(band_continuation, "bind_teacher_content", lambda *a, **kw: {})
    monkeypatch.setattr(band_continuation, "teacher_bindings", lambda *a: {
        "teacher_digest": "c" * 64, "teacher_receipt_sha256": "d" * 64})
    monkeypatch.setattr(band_replay, "load_teacher", lambda *a, **kw: {})
    monkeypatch.setattr(replay, "prepare_streams", lambda *a, **kw: {"null": {}})
    monkeypatch.setattr(energy_inputs, "source_reader", lambda a: nullcontext(SimpleNamespace(stats={})))
    runner = SimpleNamespace(model=SimpleNamespace(eval=lambda: None), shutdown=lambda: None)
    monkeypatch.setattr(prismaquant.cost_streaming, "build_streamed_causal_lm", lambda *a, **kw: runner)
    monkeypatch.setattr(prismaquant.model_profiles, "detect_profile", lambda *a: None)
    monkeypatch.setattr(mixed_injection, "interceptable_packed_experts", lambda m: nullcontext({}))

    def window(*a, check, progress, **kw):
        for sample in range(384, 390):
            clock.now += 800
            check("resident perturbed call")
            progress("layer-41-stream-null", "complete band sequence %d" % sample)
        return {"checkpoint_path": "fixture-checkpoint", "next": {"complete": False}}

    monkeypatch.setattr(replay, "run_band_stream_window", window)
    replay.main()
    manifest = json.loads(args.run_manifest_out.read_text())
    assert manifest["elapsed_seconds"] == 4800
    assert manifest["complete"] is False
    assert manifest["checkpoints"] == ["fixture-checkpoint"]


def test_fake_clock_stall_names_last_completed_coordinates(tmp_path):
    from replay_progress import ReplayProgress

    clock = Clock()
    progress = ReplayProgress(tmp_path / "history.jsonl", 1800, clock=clock)
    clock.now = 100
    progress("layer-41-stream-A8-a4", "complete band sequence 399")
    clock.now = 1899
    progress.check("a call with no completed work")
    clock.now = 1900
    with pytest.raises(TimeoutError, match=r"stalled.*layer=41 sequence=399 stream=A8-a4"):
        progress.check("blocked call")


def test_duplicate_progress_does_not_reset_silence(tmp_path):
    from replay_progress import ReplayProgress

    clock = Clock()
    progress = ReplayProgress(tmp_path / "history.jsonl", 10, clock=clock)
    progress("layer-41-stream-null", "complete band sequence 384")
    clock.now = 9
    progress("layer-41-stream-null", "complete band sequence 384")
    clock.now = 10
    with pytest.raises(TimeoutError):
        progress.check()


def test_every_step_passes_the_real_prismabuild_watch(tmp_path, monkeypatch):
    from prismabuild.pool import ProgressPhase, ProgressPolicy, ProgressWatch
    from replay_progress import ReplayProgress

    destination = tmp_path / "pb-progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(destination))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "fixture-token")
    phases = ["layer-41-stream-null", "layer-41-stream-A8-a4", "energy"]
    policy = ProgressPolicy(tuple(ProgressPhase(phase, 1800, None) for phase in phases), None)
    watch = ProgressWatch(destination, "fixture-token", policy, started=0)
    clock = Clock()
    history = tmp_path / "history.jsonl"
    progress = ReplayProgress(history, 1800, clock=clock)
    steps = [(phases[0], "complete band sequence 384"),
             (phases[0], "complete band sequence 385"),
             (phases[0], "complete band replay layer stream"),
             (phases[1], "complete teacher score 384"),
             (phases[1], "complete teacher score 385"),
             (phases[1], "complete band replay layer stream"),
             (phases[2], "durable complete band streams")]
    for count, (phase, unit) in enumerate(steps, 1):
        clock.now += 800
        progress(phase, unit)
        assert watch.sample(now=clock.now), watch.last_rejection
        assert watch.last_accepted["units_completed"] == count
        assert "layer=41 sequence=" in watch.last_accepted["unit"]
        assert "stream=" in watch.last_accepted["unit"]
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert [(r["phase"], r["units_completed"], r["layer"], r["sequence"], r["stream"])
            for r in records[1:]] == [
        (phases[0], 1, 41, 384, "null"), (phases[0], 2, 41, 385, "null"),
        (phases[0], 3, 41, 385, "null"), (phases[1], 4, 41, 384, "A8-a4"),
        (phases[1], 5, 41, 385, "A8-a4"), (phases[1], 6, 41, 385, "A8-a4"),
        (phases[2], 7, 41, 385, "A8-a4")]
    assert [b["monotonic_seconds"] - a["monotonic_seconds"]
            for a, b in zip(records, records[1:])] == [800] * 7
    assert watch.rejected == 0


def test_supervisor_stops_a_blocked_native_operation(tmp_path):
    import os
    from replay_progress import ReplayProgress, supervise

    pid_path = tmp_path / "worker.pid"

    def blocked(notify):
        pid_path.write_text(str(os.getpid()))
        progress = ReplayProgress(tmp_path / "history.jsonl", 1, notify=notify)
        progress("layer-43-stream-null", "complete band sequence 399")
        reader, writer = os.pipe()
        # This native read has no next Python check and no completed work.
        os.read(reader, 1)

    with pytest.raises(TimeoutError, match=r"stalled.*layer=43 sequence=399 stream=null"):
        supervise(blocked, 1)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_supervisor_keeps_a_worker_error():
    from replay_progress import supervise

    def fail(notify):
        raise SystemExit(7)

    with pytest.raises(SystemExit) as error:
        supervise(fail, 10)
    assert error.value.code == 7


def test_supervisor_drains_progress_after_a_parent_delay(tmp_path, monkeypatch):
    import replay_progress as rp
    from multiprocessing.connection import wait as real_wait

    clock = Clock()
    monkeypatch.setattr(rp.time, "monotonic", clock)

    def delayed_wait(objects, timeout):
        if clock.now == 0:
            # Let the worker finish before the parent's artificial clock jump.
            assert real_wait([objects[-1]], timeout=10)
            clock.now = 8000
        return real_wait(objects, timeout=0)

    monkeypatch.setattr(rp, "wait", delayed_wait)

    def steady(notify):
        progress = rp.ReplayProgress(tmp_path / "delayed.jsonl", 1800, notify=notify)
        for sample in range(384, 390):
            clock.now += 800
            progress("layer-41-stream-null", "complete band sequence %d" % sample)

    rp.supervise(steady, 1800)
    rows = [json.loads(line) for line in (tmp_path / "delayed.jsonl").read_text().splitlines()]
    assert rows[-1]["monotonic_seconds"] == 4800
    assert rows[-1]["sequence"] == 389
