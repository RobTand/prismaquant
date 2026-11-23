"""Check the render sample CLI with local inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, current_thread, main_thread

import pytest

from prismaquant import cost_stage_checkpoint, io_spans, tessera_joint_aura
from tests import campaign_sample_fixtures
from tools import profile_stage_b_head as head


@pytest.fixture
def sample_args(tmp_path):
    inputs, names, fmt, _, _ = campaign_sample_fixtures.fixture(tmp_path)
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({
        "schema": "prismaquant.render_sample.v1", "inputs": inputs,
        "scope": [0, 2], "cells": [[name, fmt] for name in sorted(names)],
    }))
    missing = {"path": str(tmp_path / "absent"), "sha256": "0" * 64}
    plan = tmp_path / "joint.json"
    plan.write_text(json.dumps({
        "schema": tessera_joint_aura.SCHEMA, "inputs": inputs,
        "canonical_capture": missing, "source_identity_cache": missing,
        "source_digest_cache": missing, "profile_tool": "cprofile",
        "max_render_bytes": 64, "max_gpu_bytes": 1, "min_free_gib": 0,
        "source_prefetch": {"max_cache_slots": 2, "prefetch_workers": 1,
                            "prefetch_lookahead": 1, "cache_headroom_gb": 1,
                            "prefetch_min_available_gb": 1, "require_prefetched_residency": True},
        "execution": {"n_calib_samples": 1, "calib_seqlen": 1, "probe_microbatch": 1,
                      "n_probes": 2, "seed_base": 0, "token_scope": "all",
                      "temperature": 1.0, "production_act_scales": "0"},
    }))
    return ["--mode", "render-sample", "--plan", str(plan),
            "--plan-sha256", campaign_sample_fixtures.sha(plan),
            "--sample-manifest", str(manifest), "--unit-scope", "0:2",
            "--head-walk-workers", "2", "--file-hash-workers", "3",
            "--trials", "1", "--max-trials", "1", "--max-files", "2", "--max-bytes", "64",
            "--storage-mount", str(tmp_path), "--deadline-s", "120"]


@pytest.mark.parametrize("nfs", [True, False])
def test_cli_reports_local_content_and_selected_storage(sample_args, tmp_path, monkeypatch,
                                                        capsys, nfs):
    empty = {"bytes": None, "ops": {}}
    selected = {"bytes": [0] * 8, "ops": {"READ": [0] * 8}} if nfs else empty
    other = {"bytes": [0] * 8, "ops": {"WRITE": [0] * 8}}
    monkeypatch.setattr(io_spans, "read_mountstats",
                        lambda: {str(tmp_path): selected, str(tmp_path / "other"): other})
    widths = []
    drive = cost_stage_checkpoint._drive_ordered_units

    def measure(*args, workers, **kwargs):
        widths.append(workers)
        if nfs:
            selected["bytes"] = [24] + [0] * 7
            selected["ops"]["READ"] = [2, 2, 0, 0, 24, 0, 8, 8]
        return drive(*args, workers=workers, **kwargs)

    monkeypatch.setattr(cost_stage_checkpoint, "_drive_ordered_units", measure)
    io = iter([{"rchar": 10}, {"rchar": 34}])
    monkeypatch.setattr(io_spans, "read_proc_io", lambda: next(io))
    monkeypatch.setattr(io_spans, "read_proc_status", lambda: {"VmHWM": 8192})
    try:
        status = head.main(sample_args)
    except SystemExit as exc:
        status = exc.code
    assert status == 0, capsys.readouterr().err
    prefix, payload = capsys.readouterr().out.strip().split(" ", 1)
    assert prefix == "STAGE_B_HEAD_PROFILE"
    report = json.loads(payload)
    assert report["mode"] == "render-sample"
    assert report["verification"] == "sample-only"
    assert report["payload_files"] == 2 and report["payload_bytes"] == 24
    assert [row["sha256"] for row in report["trials"][0]] == [
        hashlib.sha256(b"inert render").hexdigest()] * 2
    assert report["head_walk_workers"] == 2
    assert report["file_hash_workers"] == 3 and widths == [3]
    assert report["cpu_s"] >= 0 and report["peak_rss_kib"] == 8
    assert report["proc_io"] == {"rchar": 24}
    assert report["storage_mount"] == str(tmp_path)
    assert report["nfs_status"] == ("available" if nfs else "unavailable")
    assert report["nfs_ops"] == ({"_bytes": [24] + [0] * 7, "READ": 2} if nfs else None)
    assert report["read_rtt_ms_mean"] == (4.0 if nfs else None)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("deadline", [[], ["--deadline-s", "nan"]])
def test_cli_requires_a_finite_positive_deadline(sample_args, capsys, deadline):
    with pytest.raises(SystemExit) as stopped:
        head.main(sample_args[:-2] + deadline)
    assert stopped.value.code == 2
    assert "--deadline-s" in capsys.readouterr().err


def test_cli_requires_a_storage_mount(sample_args, capsys):
    with pytest.raises(SystemExit) as stopped:
        head.main(sample_args[:-4] + sample_args[-2:])
    assert stopped.value.code == 2
    assert "--storage-mount" in capsys.readouterr().err


def test_cli_rejects_sample_options_without_opt_in(sample_args, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        head.main(["--mode", "walk", *sample_args[2:], "--scratch", str(tmp_path / "scratch"),
                   "--quantum", str(tmp_path / "quantum"), "--quantum-sha256", "0" * 64])
    assert stopped.value.code == 2
    assert "require --mode render-sample" in capsys.readouterr().err
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("reason", ["deadline", "read-guard", "SIGTERM"])
def test_cli_reports_stop_while_plan_read_blocks(sample_args, reason):
    done = subprocess.run([sys.executable, "-m", "tests.test_profile_render_sample", reason,
                           *sample_args[:-2], "--deadline-s", "0.1" if reason == "deadline" else "120",
                           "--guard-interval-s", "0.01"], capture_output=True, text=True, timeout=45)
    assert done.returncode == 75, done.stdout + done.stderr
    blocked, record = done.stdout.splitlines()
    assert blocked == "The plan read is blocked."
    prefix, payload = record.split(" ", 1)
    assert prefix == "STAGE_B_HEAD_PROFILE_STOPPED"
    report = json.loads(payload)
    assert report["reason"] == reason
    assert report["storage_mount"] == sample_args[-3]


@pytest.mark.parametrize("payload_kind", ["render", "wire"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_cli_reads_no_payload_given_as_the_manifest_before_the_checks(
        sample_args, tmp_path, monkeypatch, capsys, payload_kind, alias):
    directory, pattern = {"render": ("campaign/rows/row-0000/cache", "*.pt"),
                          "wire": ("merged/cache/wire", "*.tessera")}[payload_kind]
    payload = min((tmp_path / directory).glob(pattern))
    manifest = tmp_path / "alias.json"
    if alias == "direct":
        manifest = payload
    elif alias == "symlink":
        manifest.symlink_to(payload)
    else:
        manifest.hardlink_to(payload)
    sample_args[sample_args.index("--sample-manifest") + 1] = str(manifest)

    def guard(real_open):
        def open_file(file, *args, **kwargs):
            path = Path(os.fsdecode(file))
            assert not path.exists() or not path.samefile(payload), "The sample opened a payload."
            return real_open(file, *args, **kwargs)
        return open_file

    for owner in (Path, os):
        monkeypatch.setattr(owner, "open", guard(owner.open))
    with pytest.raises(ValueError, match="manifest"):
        head.main(sample_args)
    assert capsys.readouterr().out == ""


def _block_plan(reason, argv):
    entered, baseline = Event(), Event()
    mount = argv[argv.index("--storage-mount") + 1]
    sampler = io_spans.PeriodicSampler

    def wait_for_read(tick, **kwargs):
        if kwargs["name"] != "render-sample-deadline":
            return sampler(tick, **kwargs)
        assert kwargs["interval_s"] == float(argv[argv.index("--deadline-s") + 1])
        assert kwargs["tick_first"] is False

        def stop():
            entered.wait()
            return tick()

        return sampler(stop, **kwargs)

    def read_mountstats():
        reading = entered.is_set()
        ops = 20 if reading and reason == "read-guard" else 0
        if current_thread().name == "read-rtt-guard":
            baseline.set()
        if reading and reason == "SIGTERM":
            signal.pthread_kill(main_thread().ident, signal.SIGTERM)
        return {mount: {"bytes": None, "ops": {"READ": [ops, ops, 0, 0, 0, 0, ops * 100, 0]}}}

    def read_plan(*args, **kwargs):
        if reason != "deadline":
            baseline.wait()
        print("The plan read is blocked.", flush=True)
        entered.set()
        Event().wait()

    io_spans.PeriodicSampler = wait_for_read
    io_spans.read_mountstats = read_mountstats
    tessera_joint_aura._read_bound = read_plan
    return head.main(argv)


if __name__ == "__main__":
    sys.exit(_block_plan(sys.argv[1], sys.argv[2:]))
