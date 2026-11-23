"""The scoped head walk measures I/O width on a bounded, read-only sample (#1492, #1247)."""
from __future__ import annotations

import threading

import pytest

from prismaquant import io_spans, tessera_joint_aura as bridge
from tools import profile_stage_b_head as head


def _mountstats(monkeypatch, fields):
    rows = {"/mnt/shared": {"bytes": None, "ops": {} if fields is None else {"READ": fields}}}
    monkeypatch.setattr(io_spans, "read_mountstats", lambda _path=None: rows)


def test_read_rtt_reads_the_ops_and_round_trip_columns(monkeypatch):
    _mountstats(monkeypatch, [10, 10, 0, 100, 200, 3, 40, 50])
    assert head.read_rtt() == (10, 40)


def test_read_rtt_of_a_mount_without_reads_is_zero(monkeypatch):
    _mountstats(monkeypatch, None)
    assert head.read_rtt() == (0, 0)
    assert head.read_rtt("/absent") == (0, 0)


def test_interval_mean_is_round_trip_per_read():
    assert head.interval_read_rtt_ms((100, 1000), (220, 1600)) == 5.0


def test_interval_mean_ignores_a_quiet_interval():
    assert head.interval_read_rtt_ms((100, 1000), (110, 9000), min_ops=20) is None


def _guard(samples, limit_ms, stops):
    feed = iter(samples)
    last = [samples[-1]]

    def sample():
        last[0] = next(feed, last[0])
        return last[0]

    return head.start_read_guard(limit_ms, interval_s=0.01, min_ops=20,
                                 sample=sample, on_stop=stops.append)


def test_guard_stops_when_the_round_trip_rises_past_the_limit():
    stopped = []
    event = threading.Event()
    stops = type("S", (), {"append": lambda self, v: (stopped.append(v), event.set())})()
    _guard([(0, 0), (100, 500), (200, 2500)], 10.0, stops)
    assert event.wait(5), "the guard never fired"
    assert stopped == [20.0]


def test_guard_does_not_stop_at_a_normal_round_trip_and_records_what_it_saw():
    stopped = []
    stop, trace = _guard([(0, 0), (100, 500), (200, 1000), (300, 1500)], 10.0, stopped)
    deadline = threading.Event()
    deadline.wait(0.3)
    stop.request_stop()
    assert stopped == []
    assert trace[:3] == [5.0, 5.0, 5.0]


def test_scope_over_the_cap_is_refused_before_any_read(monkeypatch):
    monkeypatch.setattr(bridge, "load_measured_anchor_input",
                        lambda *a, **k: pytest.fail("the loader must not run"))
    with pytest.raises(SystemExit, match="the limit is 2000"):
        head.scoped_walk_intake({"inputs": {}}, scope=(0, head.SCOPED_WALK_MAX_UNITS + 1),
                                workers=4)
    with pytest.raises(SystemExit, match="empty or reversed"):
        head.scoped_walk_intake({"inputs": {}}, scope=(5, 5), workers=4)


def test_scoped_walk_passes_the_scope_and_width_and_stays_read_only(monkeypatch):
    from prismaquant import tessera_reader
    seen = {}

    class Data:
        head_walk_workers = 4
        cells = [("a", "F"), ("a", "G"), ("b", "F")]

    def fake_load(inputs, **kwargs):
        seen.update(inputs=inputs, **kwargs)
        return Data()

    monkeypatch.setattr(bridge, "load_measured_anchor_input", fake_load)
    monkeypatch.setattr(tessera_reader, "load_declared_reader", lambda spec: "reader")
    result = head.scoped_walk_intake({"inputs": {"k": 1}, "reader": None},
                                     scope=(100, 600), workers=4)
    assert seen["unit_scope"] == (100, 600)
    assert seen["head_walk_workers"] == 4
    assert seen["head_checkpoint"] is None and seen["head_resume"] is False
    assert seen["verify_payloads"] is False
    assert seen["require_existing_renders"] is True
    assert seen["progress_phase"] is None
    assert seen["inputs"] == {"k": 1}
    assert result["scope"] == [100, 600] and result["scope_units"] == 500
    assert result["head_walk_workers"] == 4
    assert result["measured_cells"] == 3 and result["units_with_cells"] == 2


def _argv(tmp_path, *extra, mode="scoped-walk"):
    return ["--mode", mode, "--quantum", str(tmp_path / "q.json"), "--quantum-sha256", "0" * 64,
            "--plan", str(tmp_path / "p.json"), "--plan-sha256", "0" * 64,
            "--scratch", str(tmp_path / "scratch"), *extra]


@pytest.mark.parametrize("extra,message", [
    ((), "needs either --unit-scope"),
    (("--unit-scope", "0:2001", "--head-walk-workers", "4"), "at most 2000"),
    (("--unit-scope", "0:", "--head-walk-workers", "4"), "explicit end"),
    (("--unit-scope", "5:5", "--head-walk-workers", "4"), "empty or reversed"),
    (("--unit-scope", "0:100"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "17"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "0"), "head-walk-workers"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--stop-read-rtt-ms", "0"),
     "guard limits"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--stop-read-rtt-ms", "nan"),
     "positive and finite"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--stop-read-rtt-ms", "inf"),
     "positive and finite"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--guard-interval-s", "nan"),
     "positive and finite"),
    (("--unit-scope", "0:100", "--head-walk-workers", "4", "--guard-interval-s", "inf"),
     "positive and finite"),
    (("--sweep-start", "0", "--slice-units", "100", "--sweep-workers", "4",
      "--stop-read-rtt-ms", "nan"), "positive and finite"),
])
def test_main_refuses_a_bad_scoped_call_before_reading_anything(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exit_info:
        head.main(_argv(tmp_path, *extra))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scratch").exists(), "a refused call creates nothing"


def test_the_scoped_options_belong_to_the_scoped_mode(tmp_path, capsys):
    with pytest.raises(SystemExit):
        head.main(_argv(tmp_path, "--unit-scope", "0:100", mode="walk"))
    assert "belong to --mode scoped-walk" in capsys.readouterr().err


def test_sweep_plan_puts_a_one_unit_baseline_first_and_keeps_the_worker_order():
    scopes = head.sweep_plan(1000, 250, [4, 1, 8, 2])
    assert scopes[0] == {"label": "baseline", "scope": (1000, 1001), "workers": 4}
    assert [item["workers"] for item in scopes[1:]] == [4, 1, 8, 2]
    assert [item["label"] for item in scopes] == ["baseline", "run0", "run1", "run2", "run3"]


def test_sweep_slices_are_disjoint_and_adjacent():
    scopes = head.sweep_plan(0, 249, [4, 1, 8, 2, 2, 8, 1, 4])
    edges = [item["scope"] for item in scopes]
    for (_, high), (low, _) in zip(edges, edges[1:]):
        assert low == high
    assert len({unit for low, high in edges for unit in range(low, high)}) == 1 + 8 * 249
    assert 1 + 8 * 249 <= head.SCOPED_WALK_MAX_UNITS


def test_sweep_holds_one_budget_for_the_whole_sweep():
    assert len(head.sweep_plan(0, 250, [1, 2, 4, 8, 8, 4, 2])) == 8
    with pytest.raises(SystemExit, match="limit is 2000 for the whole sweep"):
        head.sweep_plan(0, 250, [1, 2, 4, 8, 8, 4, 2, 1])
    with pytest.raises(SystemExit, match="limit is 2000 for the whole sweep"):
        head.sweep_plan(0, 500, [4, 1, 8, 2])
    with pytest.raises(SystemExit, match="at least one worker count"):
        head.sweep_plan(0, 250, [])


def test_sweep_runs_each_scope_once_in_order_and_reports_read_load(monkeypatch):
    calls = []
    counters = iter([(0, 0), (10, 100), (10, 100), (30, 700)])
    monkeypatch.setattr(head, "mount_ops", lambda mount_point="/mnt/shared": {"READ": 1})
    monkeypatch.setattr(head, "read_rtt", lambda mount_point="/mnt/shared": next(counters))

    def fake_walk(config, *, scope, workers, metadata_memo):
        calls.append((scope, workers))
        return {"units_with_cells": scope[1] - scope[0], "measured_cells": 7}

    report = head.scoped_walk_sweep({"inputs": {}}, start=0, slice_units=250,
                                    workers_order=[4], walk=fake_walk)
    assert calls == [((0, 1), 4), ((1, 251), 4)]
    assert [run["label"] for run in report["runs"]] == ["baseline", "run0"]
    assert report["runs"][0]["read_ops"] == 10
    assert report["runs"][0]["read_rtt_ms_mean"] == 10.0
    assert report["runs"][1]["read_ops"] == 20
    assert report["runs"][1]["read_rtt_ms_mean"] == 30.0
    assert report["sweep"]["units_total"] == 251


@pytest.mark.parametrize("extra,message", [
    (("--sweep-start", "0", "--slice-units", "500", "--sweep-workers", "4,1,8,2"),
     "limit is 2000 for the whole sweep"),
    (("--sweep-start", "0", "--slice-units", "250"), "needs --sweep-start, --slice-units and --sweep-workers"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4,x"), "integers"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4,17"), "counts in 1:16"),
    (("--sweep-start", "0", "--slice-units", "250", "--sweep-workers", "4",
      "--unit-scope", "0:10", "--head-walk-workers", "2"), "either"),
])
def test_main_refuses_a_bad_sweep_before_reading_anything(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exit_info:
        head.main(_argv(tmp_path, *extra))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scratch").exists()


def test_sigterm_handler_reports_once_and_leaves_without_waiting(monkeypatch):
    import os
    import signal

    exits, reasons = [], []
    monkeypatch.setattr(os, "_exit", exits.append)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        handler = head.install_stop_signal(reasons.append)
        assert signal.getsignal(signal.SIGTERM) is handler
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert reasons == ["SIGTERM"]
    assert exits == [75]


def _quiet_counters(monkeypatch):
    monkeypatch.setattr(head, "mount_ops", lambda mount_point="/mnt/shared": {})
    monkeypatch.setattr(head, "read_rtt", lambda mount_point="/mnt/shared": (0, 0))


def _count_loads(monkeypatch):
    real, loads = bridge._read_campaign_metadata, []

    def counting(inputs, reuse_policy):
        loads.append(1)
        return real(inputs, reuse_policy)

    monkeypatch.setattr(bridge, "_read_campaign_metadata", counting)
    return loads


def test_a_sweep_over_a_real_campaign_loads_the_metadata_once(tmp_path, monkeypatch):
    from tests.test_tessera_joint_aura import fixture

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    _quiet_counters(monkeypatch)
    loads = _count_loads(monkeypatch)
    report = head.scoped_walk_sweep({"inputs": config}, start=0, slice_units=1,
                                    workers_order=[1])
    assert [run["scope"] for run in report["runs"]] == [[0, 1], [1, 2]]
    assert [run["units_with_cells"] for run in report["runs"]] == [1, 1]
    assert len(loads) == 1
    assert report["metadata_loads"] == 1


def test_without_a_memo_every_load_reads_the_metadata_again(tmp_path, monkeypatch):
    from tests.test_tessera_joint_aura import fixture

    config, *_ = fixture(tmp_path)
    loads = _count_loads(monkeypatch)
    for scope in ((0, 1), (1, 2)):
        bridge.load_measured_anchor_input(config, verify_payloads=False, unit_scope=scope)
    assert len(loads) == 2


def test_a_memo_does_not_hide_a_changed_metadata_file(tmp_path):
    from tests.test_tessera_joint_aura import fixture

    config, *_ = fixture(tmp_path)
    memo = {}
    bridge.load_measured_anchor_input(config, verify_payloads=False, unit_scope=(0, 1),
                                      metadata_memo=memo)
    census = tmp_path / "campaign/census.json"
    census.write_text(census.read_text() + " ")
    with pytest.raises(ValueError, match="checksum changed"):
        bridge.load_measured_anchor_input(config, verify_payloads=False, unit_scope=(1, 2),
                                          metadata_memo=memo)


def test_a_scoped_walk_leaves_the_candidate_overlay_out(tmp_path, monkeypatch):
    from prismaquant import joint_catalog_extension as extension
    from tests.test_tessera_joint_aura import fixture

    config, *_ = fixture(tmp_path)
    config["candidate_overlay"] = {"path": str(tmp_path / "absent-overlay.json"),
                                   "sha256": "0" * 64}
    attached = []

    def attach(*args, **kwargs):
        attached.append(kwargs)
        raise RuntimeError("the overlay fence ran")

    monkeypatch.setattr(extension, "attach_candidate_overlay", attach)
    result = head.scoped_walk_intake({"inputs": config}, scope=(0, 1), workers=1)
    assert attached == []
    assert result["candidate_overlay_left_out"] is True
    assert result["units_with_cells"] == 1
    # The control: the same inputs through the plain loader do reach the overlay.
    with pytest.raises(RuntimeError, match="the overlay fence ran"):
        bridge.load_measured_anchor_input(config, verify_payloads=False, unit_scope=(0, 1))
    assert len(attached) == 1


def test_sigterm_stops_a_process_with_blocked_workers_after_one_report(tmp_path):
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(head.__file__).resolve().parents[1]
    script = tmp_path / "blocked_sigterm.py"
    script.write_text(textwrap.dedent("""
        import os, signal, sys, threading
        sys.path.insert(0, sys.argv[1])
        from tools import profile_stage_b_head as head

        head.install_stop_signal(lambda reason: print("REPORT " + reason, flush=True))
        never = threading.Event()
        for _ in range(4):
            threading.Thread(target=never.wait).start()
        os.kill(os.getpid(), signal.SIGTERM)
        never.wait()
    """))
    done = subprocess.run([sys.executable, str(script), str(root)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 75, done.stderr
    assert done.stdout.splitlines() == ["REPORT SIGTERM"]


@pytest.fixture
def render_sample(tmp_path, monkeypatch):
    return _make_render_sample(tmp_path, monkeypatch)


def _make_render_sample(tmp_path, monkeypatch, variant=None):
    sample = getattr(head, "render_sample_intake", None)
    if sample is None:
        return None, None, None, None

    import builtins
    import io
    import json
    import os
    from pathlib import Path

    from tests import campaign_sample_fixtures
    from prismaquant.render_cache_paths import cache_weight_filename

    inputs, names, fmt, payload, states = campaign_sample_fixtures.fixture(tmp_path)
    cells = [[name, fmt] for name in sorted(names)]
    paths = [tmp_path / "campaign/rows/row-0000/cache" / cache_weight_filename(name, fmt)
             for name, fmt in cells]
    for path, body in zip(paths, (b"abc", b"hello"), strict=True):
        path.write_bytes(body)
    name = cells[0][0]
    record = states[name]["wire_records"][fmt]
    marker = bridge._render_origin_marker_path(paths[0])
    wire = Path(payload["provenance"]["wire_dir"]) / record["file"]
    if variant == "symlink":
        marker.symlink_to(wire)
    elif variant == "hardlink":
        marker.hardlink_to(wire)
    elif variant in {"marked", "missing"}:
        marker.write_text(json.dumps({
            "schema": bridge.RENDER_ORIGIN_SCHEMA, "render_origin": "synthesized_from_wire",
            "unit": name, "format_name": fmt, "wire_file": record["file"],
            "wire_sha256": record["blob_sha256"],
        }))
    scope = (0, len(paths))
    manifest = {"schema": "prismaquant.render_sample.v1", "inputs": inputs,
                "scope": list(scope), "cells": cells}
    config = {"inputs": {**inputs, "candidate_overlay": {"path": str(tmp_path / "absent")}}}
    opened = []

    def watch(real_open):
        def record(path, *args, **kwargs):
            if not isinstance(path, int):
                path = Path(os.fsdecode(path))
                assert path.suffix != ".tessera", "The reader opened a wire payload."
                if path == marker:
                    if variant == "missing":
                        paths[0].unlink(missing_ok=True)
                    if path.exists():
                        assert not path.samefile(wire), "The marker opened a wire payload."
                if path in paths:
                    opened.append(path)
            return real_open(path, *args, **kwargs)
        return record

    for module in (builtins, io, os):
        monkeypatch.setattr(module, "open", watch(module.open))

    def run(*, reader=None, **changes):
        return sample(config | {"reader": reader},
                      **({"manifest": manifest, "scope": scope, "workers": 1,
                          "hash_workers": 1, "trials": 2, "max_trials": 2,
                          "max_files": 4, "max_bytes": 16} | changes))

    return run, manifest, paths, opened


def test_render_sample_admits_marker_metadata_before_hashing(tmp_path, monkeypatch):
    run, manifest, paths, opened = _make_render_sample(tmp_path, monkeypatch, "marked")
    assert run is not None, "The tool has no render sample reader."
    with pytest.raises(ValueError, match="manifest does not match admitted inputs"):
        run(manifest={**manifest, "inputs": {}})
    assert opened == []
    with pytest.raises(ValueError, match="max_bytes"):
        run(max_bytes=15)
    assert opened == []
    result = run()
    assert result["trials"][0][0]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert opened == paths * 2


def test_render_sample_opens_markers_only_after_plan_admission(tmp_path, monkeypatch):
    import builtins
    import io
    import os
    from contextlib import contextmanager
    from pathlib import Path
    from unittest.mock import Mock

    run, manifest, paths, opened = _make_render_sample(tmp_path, monkeypatch, "marked")
    marker = bridge._render_origin_marker_path(paths[0])
    marker_opens, reads = [], []
    wires = (tmp_path / "merged/cache/wire").iterdir()
    payload_ids = {(info.st_dev, info.st_ino) for info in (path.stat() for path in [*paths, *wires])}
    fdopen = os.fdopen

    def watch(real_open):
        def record(path, *args, **kwargs):
            if not isinstance(path, int) and Path(os.fsdecode(path)) == marker:
                marker_opens.append(marker)
            return real_open(path, *args, **kwargs)
        return record

    @contextmanager
    def count_reads(*args, **kwargs):
        with fdopen(*args, **kwargs) as handle:
            info = os.fstat(handle.fileno())
            if (info.st_dev, info.st_ino) not in payload_ids:
                yield handle
                return

            def read(size):
                chunk = handle.read(size)
                reads.append(len(chunk))
                return chunk

            yield Mock(wraps=handle, read=read)

    for module in (builtins, io, os):
        monkeypatch.setattr(module, "open", watch(module.open))
    monkeypatch.setattr(os, "fdopen", count_reads)

    for changes, message in [
        ({"manifest": manifest | {"inputs": {}}}, "admitted inputs"),
        ({"manifest": manifest | {"cells": [[manifest["cells"][0][0], "absent"]]}},
         "unadmitted cell"),
        ({"max_files": 3}, "max_files"),
        ({"max_bytes": 15}, "max_bytes"),
    ]:
        with pytest.raises(ValueError, match=message):
            run(**changes)
        assert marker_opens == opened == reads == []

    result = run()
    assert marker_opens == [marker]
    assert opened == paths * 2
    assert result["payload_files"] == len(opened)
    assert result["payload_bytes"] == sum(reads) == 16


def test_render_sample_refuses_a_marker_link_to_wire_before_reading(tmp_path, monkeypatch):
    for link in ("symlink", "hardlink"):
        with monkeypatch.context() as patch:
            run, _, _, opened = _make_render_sample(tmp_path / link, patch, link)
            assert run is not None, "The tool has no render sample reader."
            with pytest.raises(ValueError, match="render origin marker must be an unlinked regular file"):
                run()
            assert opened == []


def test_render_sample_refuses_disappearance_while_reading_a_marker(tmp_path, monkeypatch):
    run, manifest, paths, opened = _make_render_sample(tmp_path, monkeypatch, "missing")
    assert run is not None, "The tool has no render sample reader."
    with pytest.raises((ValueError, FileNotFoundError)) as refusal:
        run(manifest={**manifest, "cells": manifest["cells"][1:]})
    assert "prepared render is missing" in str(refusal.value)
    assert opened == []
    assert not paths[0].exists()


def test_render_sample_does_not_load_declared_reader(render_sample, monkeypatch):
    from prismaquant import tessera_reader

    run, _, _, _ = render_sample
    assert run is not None, "The tool has no render sample reader."

    def check_reader(record):
        assert record is None, "The sample tried to load the declared wire reader."

    monkeypatch.setattr(tessera_reader, "load_declared_reader", check_reader)
    assert run(reader={"path": "unused"})["payload_files"] == 4


def test_render_sample_reports_stable_measured_digests(render_sample, tmp_path, monkeypatch):
    from itertools import count
    from types import SimpleNamespace
    from prismaquant.digests import canonical_json_sha256
    from prismaquant.file_identity import file_stat_signature

    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    ticks = count(step=2)
    monkeypatch.setattr(head, "time", SimpleNamespace(monotonic=ticks.__next__))
    before = {path: file_stat_signature(path.stat()) for path in tmp_path.rglob("*")}
    first, second = run(), run()
    assert first == second
    assert first["manifest_sha256"] == canonical_json_sha256(manifest, where="fixture")
    assert first["inputs"] == manifest["inputs"]
    assert first["verification"] == "sample-only"
    expected = ["ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"]
    assert [[row["sha256"] for row in trial] for trial in first["trials"]] == [expected] * 2
    assert [[row["cell"] for row in trial] for trial in first["trials"]] == [manifest["cells"]] * 2
    assert first["payload_files"] == 4 and first["payload_bytes"] == 16
    assert first["marks"] == {"anchor_input_s": 2, "render_hash_s": 2}
    assert opened == paths * 4
    assert {path: file_stat_signature(path.stat()) for path in tmp_path.rglob("*")} == before


def test_render_sample_checks_supplied_digests(render_sample):
    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    assert run().get("digest_verification") == "not-requested"
    opened.clear()
    expected = ["ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"]
    with pytest.raises(ValueError, match="expected_sha256"):
        run(manifest=manifest | {"expected_sha256": expected[:1]})
    assert opened == []
    report = run(manifest=manifest | {"expected_sha256": expected})
    assert report["verification"] == "sample-only"
    assert report["digest_verification"] == "verified"
    assert opened == paths * 2
    with pytest.raises(ValueError, match="render digest mismatch"):
        run(manifest=manifest | {"expected_sha256": expected[::-1]})


def test_render_sample_reads_only_the_selected_cell(render_sample):
    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    selected = manifest["cells"][1:]
    report = run(manifest={**manifest, "cells": selected}, max_files=2, max_bytes=10)
    assert [trial[0]["cell"] for trial in report["trials"]] == selected * 2
    assert report["payload_files"] == 2 and report["payload_bytes"] == 10
    assert opened == paths[1:] * 2


@pytest.mark.parametrize("case", ["valid", "invalid_manifest", "over_budget",
                                  "wire_symlink", "wire_hardlink", "render_symlink"])
def test_render_sample_separates_marker_metadata_from_payload(render_sample, tmp_path, monkeypatch, case):
    import builtins
    import hashlib
    import io
    import json
    import os
    from pathlib import Path

    run, manifest, paths, opened = render_sample
    cell = manifest["cells"][0]
    name, fmt = cell
    wire = tmp_path / "merged/cache/wire" / (name + ".tessera")
    marker = bridge._render_origin_marker_path(paths[0])
    if case == "wire_symlink":
        marker.symlink_to(wire)
    elif case == "wire_hardlink":
        marker.hardlink_to(wire)
    elif case == "render_symlink":
        marker.symlink_to(paths[0])
    else:
        marker.write_text(json.dumps({"schema": bridge.RENDER_ORIGIN_SCHEMA,
            "render_origin": "synthesized_from_wire", "unit": name, "format_name": fmt,
            "wire_file": wire.name, "wire_sha256": hashlib.sha256(f"inert {name}".encode()).hexdigest()}))

    def guard(real_open):
        def open_file(path, *args, **kwargs):
            if not isinstance(path, int) and Path(os.fsdecode(path)) == marker:
                assert case in {"valid", "invalid_manifest", "over_budget"}, (
                    "The sample opened a payload through the marker.")
            return real_open(path, *args, **kwargs)
        return open_file

    for module in (builtins, io, os):
        monkeypatch.setattr(module, "open", guard(module.open))

    selected = [[name, "unadmitted"]] if case == "invalid_manifest" else [cell]
    limits = dict(manifest={**manifest, "cells": selected}, max_files=1, trials=1,
                  max_bytes=2 if case == "over_budget" else 3)
    if case == "valid":
        assert run(**limits)["trials"] == [[{"cell": cell, "bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]]
        assert opened == paths[:1]
    else:
        message = {"invalid_manifest": "manifest", "over_budget": "max_bytes"}.get(
            case, "render origin marker")
        with pytest.raises(ValueError, match=message):
            run(**limits)
        assert opened == []


def test_render_sample_stops_reading_at_the_planned_size(render_sample, monkeypatch):
    import os
    from contextlib import contextmanager
    from unittest.mock import Mock, call

    from prismaquant import digests

    run, manifest, _, _ = render_sample
    assert run is not None, "The tool has no render sample reader."
    fdopen = os.fdopen
    reads = []

    @contextmanager
    def record_reads(*args, **kwargs):
        with fdopen(*args, **kwargs) as handle:
            stream = Mock(wraps=handle)
            reads.append(stream.read)
            yield stream

    monkeypatch.setattr(os, "fdopen", record_reads)
    monkeypatch.setattr(digests, "FILE_BLOCK_BYTES", 2)
    run(manifest={**manifest, "cells": manifest["cells"][:1]}, trials=1, max_bytes=3)
    assert [read.call_args_list for read in reads] == [[call(2), call(1)]]


@pytest.mark.parametrize("budget", ["max_files", "max_bytes", "max_trials"])
@pytest.mark.parametrize("limit", ["absent", "too_low"])
def test_render_sample_refuses_over_budget_before_payload(render_sample, budget, limit):
    run, _, _, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    too_low = {"max_files": 3, "max_bytes": 15, "max_trials": 1}
    with pytest.raises(ValueError, match=budget):
        run(**{budget: None if limit == "absent" else too_low[budget]})
    assert opened == []


@pytest.mark.parametrize("field", ["cells", "inputs"])
def test_render_sample_refuses_an_invalid_manifest_before_payload(render_sample, field):
    run, manifest, _, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    bad = {"cells": [*manifest["cells"][:1], ["not-admitted", manifest["cells"][-1][1]]],
           "inputs": {}}
    with pytest.raises(ValueError, match="manifest"):
        run(manifest={**manifest, field: bad[field]})
    assert opened == []


def test_render_sample_refuses_a_missing_later_render_before_hashing(render_sample, monkeypatch):
    run, _, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    missing = paths[-1]
    missing.unlink()
    monkeypatch.setattr(bridge, "_synthesize_render_from_wire",
                        lambda *a, **k: pytest.fail("The loader tried to synthesize a render."))
    with pytest.raises(ValueError, match="prepared render is missing"):
        run()
    assert opened == []
    assert not missing.exists()


def test_render_sample_refuses_an_unselected_render_removed_at_resolution(render_sample, monkeypatch):
    from pathlib import Path

    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    resolve = bridge._resolve_render_origin

    def remove_before_resolve(render, **kwargs):
        if Path(render) == paths[-1]:
            paths[-1].unlink()
        return resolve(render, **kwargs)

    monkeypatch.setattr(bridge, "_resolve_render_origin", remove_before_resolve)
    monkeypatch.setattr(bridge, "_synthesize_render_from_wire",
                        lambda *a, **k: pytest.fail("The loader tried to synthesize a render."))
    with pytest.raises(ValueError, match="prepared render is missing"):
        run(manifest={**manifest, "cells": manifest["cells"][:1]})
    assert opened == []
    assert not paths[-1].exists()


def test_render_sample_refuses_a_render_link_to_a_wire(render_sample, tmp_path):
    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    name, _ = manifest["cells"][0]
    paths[0].unlink()
    paths[0].symlink_to(tmp_path / "merged/cache/wire" / (name + ".tessera"))
    with pytest.raises(ValueError, match="not a regular file"):
        run(max_bytes=128)
    assert opened == []


def test_render_sample_refuses_a_render_hard_link_to_a_wire(render_sample, tmp_path):
    run, manifest, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    name, _ = manifest["cells"][-1]
    paths[-1].unlink()
    paths[-1].hardlink_to(tmp_path / "merged/cache/wire" / (name + ".tessera"))
    with pytest.raises(ValueError, match="hard link"):
        run(max_bytes=128)
    assert opened == []


def test_render_sample_overlaps_reads_within_the_hash_worker_limit(render_sample, monkeypatch):
    from threading import Barrier, Lock
    from prismaquant import cost_stage_checkpoint, digests

    run, manifest, _, _ = render_sample
    assert run is not None, "The tool has no render sample reader."
    rendezvous, lock = Barrier(2, timeout=10), Lock()
    shared_digest = digests.chunks_sha256hex
    drive = cost_stage_checkpoint._drive_ordered_units
    widths = []
    active = peak = 0

    def record_width(roster, walk, commit, *, workers):
        widths.append(workers)
        assert workers == 2
        return drive(roster, walk, commit, workers=workers)

    def overlap(chunks):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            rendezvous.wait()
            return shared_digest(chunks)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(digests, "chunks_sha256hex", overlap)
    monkeypatch.setattr(cost_stage_checkpoint, "_drive_ordered_units", record_width)
    result = run(hash_workers=2)
    assert widths == [2, 2] and peak == 2
    assert [[row["cell"] for row in trial] for trial in result["trials"]] == [manifest["cells"]] * 2


def test_render_sample_queues_a_read_until_a_hash_slot_is_free(tmp_path, monkeypatch):
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import Mock

    sample = getattr(head, "render_sample_intake", None)
    assert sample is not None, "The tool has no render sample reader."
    paths = [tmp_path / name for name in ("a", "b", "c")]
    for path, body in zip(paths, (b"abc", b"hello", b"x")):
        path.write_bytes(body)
    inputs = {"fixture": "admitted"}
    cells = {(path.name, "F"): {"render": str(path)} for path in paths}
    manifest = {"schema": "prismaquant.render_sample.v1", "inputs": inputs,
                "scope": [0, 3], "cells": [list(cell) for cell in cells]}
    data = SimpleNamespace(cells=cells, inputs=inputs, head_walk_workers=3)
    monkeypatch.setattr(head, "_load_scoped_walk", lambda *args: (data, time.monotonic()))
    slots = {path.stat().st_ino: slot for slot, path in enumerate(paths)}
    entered = [threading.Event() for _ in paths]
    release = [threading.Event() for _ in paths[:2]]
    lock = threading.Lock()
    active = peak = 0
    fdopen, submit = os.fdopen, ThreadPoolExecutor.submit

    @contextmanager
    def hold_reads(*args, **kwargs):
        with fdopen(*args, **kwargs) as handle:
            slot = slots.get(os.fstat(handle.fileno()).st_ino)
            if slot is None:
                yield handle
                return

            def read(size):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    body = handle.read(size)
                    entered[slot].set()
                    if slot < len(release):
                        assert release[slot].wait(10), "The test did not release the read."
                    return body
                finally:
                    with lock:
                        active -= 1

            stream = Mock(wraps=handle)
            stream.read.side_effect = read
            yield stream

    def check_queue(pool, walk, item):
        if not isinstance(item, tuple) or item[0] != tuple(manifest["cells"][2]):
            return submit(pool, walk, item)
        try:
            assert entered[0].wait(10) and entered[1].wait(10)
            queued = submit(pool, walk, item)
            assert pool._max_workers == 2
            assert not queued.running() and not queued.done() and not entered[2].is_set()
            release[1].set()
            assert entered[2].wait(10), "The queued read did not start."
            queued.result(timeout=10)
            return queued
        finally:
            for event in release:
                event.set()

    monkeypatch.setattr(os, "fdopen", hold_reads)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", check_queue)
    report = sample({"inputs": inputs}, manifest=manifest, scope=(0, 3), workers=3,
                    hash_workers=2, trials=1, max_trials=1, max_files=3, max_bytes=9)
    assert peak == 2
    assert [row["cell"] for row in report["trials"][0]] == manifest["cells"]


@pytest.mark.parametrize("change", ["extend", "replace"])
def test_render_sample_refuses_a_file_changed_during_hashing(render_sample, monkeypatch, change):
    import os

    from prismaquant import digests

    run, _, paths, opened = render_sample
    assert run is not None, "The tool has no render sample reader."
    shared_digest = digests.chunks_sha256hex

    def mutate(chunks):
        digest = shared_digest(chunks)
        path = paths[0]
        if change == "extend":
            os.truncate(path, 4)
        else:
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b"abc")
            replacement.replace(path)
        return digest

    monkeypatch.setattr(digests, "chunks_sha256hex", mutate)
    with pytest.raises(ValueError, match="render changed"):
        run()
    assert opened == paths[:1]
