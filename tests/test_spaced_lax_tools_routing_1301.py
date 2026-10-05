"""Actual tool outputs frozen from the pre-refactor consumer tree (PQ #2320).

These tests intentionally pass before and after the serializer consolidation.
The fixture is recorded from 99a578e0, not from the shared profile. The separate
consumer evidence matrix states which remaining outputs are not exercised here.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from golden_table import GoldenTable

GOLDEN = GoldenTable("spaced_lax_tools_consumers_2320")
ROOT = Path(__file__).resolve().parents[1]
LAUNCH_CONTRACTS = json.loads((Path(__file__).parent / "fixtures" /
    "spaced_lax_tools_launch_contracts_2320.json").read_text())["rows"]


def test_state_append_preserves_existing_bytes_and_nonfinite_event(tmp_path, monkeypatch):
    from tools.dispatch_joint_quanta import _append_state
    import time

    monkeypatch.setattr(time, "time", lambda: 1234.5)
    path = tmp_path / "state" / "events.jsonl"
    path.parent.mkdir()
    path.write_bytes(b"previous event\n")
    _append_state(path, {"quantum_id": "q-é", "values": [float("nan"), {}, []]})
    GOLDEN.value(path.read_bytes())


def test_state_encoding_failure_does_not_append_a_partial_event(tmp_path):
    from tools.dispatch_joint_quanta import _append_state

    path = tmp_path / "events.jsonl"
    path.write_bytes(b"previous event\n")
    row = GOLDEN.call(lambda: _append_state(path, {"not_serializable": object()}))
    assert row["raised"] == "builtins.TypeError"
    assert path.read_bytes() == b"previous event\n"


@pytest.mark.parametrize("runs", [[3.0, 1.0, 2.0], []], ids=["median", "empty"])
def test_fence_finish_reports_calculated_median_and_removes_scratch(tmp_path, capsys, runs):
    from tools.bench_fence_rehash import _finish

    root = tmp_path / "scratch"
    root.mkdir()
    report = {"runs": [{"wall_s": wall} for wall in runs], "label": "é"}
    GOLDEN.call(lambda: _finish(report, str(root), 100.0))
    GOLDEN.value(capsys.readouterr().out)
    GOLDEN.value(report)
    assert not root.exists()


@pytest.mark.parametrize("spy", [None, "/tools/spy-é"], ids=["no-spy", "spy"])
def test_chain_spec_removes_spool_environment_and_mounts(tmp_path, capsys, spy):
    from tools.chain_roll_bench import cmd_spec

    spec = {"container": {"mounts": [
        {"source": "/spool", "target": "/spool", "readonly": False},
        {"source": "/model-é", "target": "/model", "readonly": True}]},
        "env": {"PRISMABUILD_PRODUCED_SPOOL_ROOT": "/spool",
                "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": "100", "KEEP": "é"}}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(spec))
    before = path.read_bytes()
    args = SimpleNamespace(base_spec=path, scratch="/scratch-é", py_spy=spy)
    GOLDEN.call(lambda: cmd_spec(args))
    GOLDEN.value(capsys.readouterr().out)
    assert path.read_bytes() == before


@pytest.mark.parametrize("digests", [["same", "same"], ["left", "right"], []],
                         ids=["equal", "different", "empty"])
def test_render_analysis_publishes_reductions_and_exit_status(tmp_path, capsys, digests):
    from tools.render_window_bench import cmd_analyze

    for index, digest in enumerate(digests):
        row = {"arm": "arm-é", "label": f"arm-r{index}", "window_count": 2,
               "projection_digest": digest, "gpu_power": {"mean_w": float("nan")},
               "windows": {"0": {"wall_s": 2.0, "timers": {
                   "main.hash": {"seconds": 4.0}}},
                           "1": {"wall_s": 4.0, "timers": {}}}}
        (tmp_path / f"arm-r{index}.json").write_text(json.dumps(row))
    GOLDEN.call(lambda: cmd_analyze(SimpleNamespace(out=tmp_path, py_spy_rate=100)))
    GOLDEN.value(capsys.readouterr().out)
    GOLDEN.value((tmp_path / "analysis.json").read_bytes())


def test_checkpoint_parse_reads_identity_and_counts_interned_menu(tmp_path, capsys, monkeypatch):
    from tools import checkpoint_parse_probe as probe
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256_normalized

    identity = {"units": {"a": {"menu": ["é", "x"]}, "b": {"menu": ["é"]}}}
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(probe, "_gib", lambda field: 0.5)
    monkeypatch.setattr(probe.time, "time", lambda: 100.0)
    GOLDEN.call(lambda: probe.main([
        "--checkpoint", str(path), "--phase", "parse", "--sample-units", "2",
        "--expect-digest", canonical_json_sha256_normalized(identity, where="fixture")]))
    GOLDEN.value(capsys.readouterr().out, tmp=tmp_path)


def test_container_wrapper_emits_transformed_spec_and_preserves_input(tmp_path):
    from tools.dispatch_joint_quanta import _container_wrap

    spec = {"container": {"image": "sha256:" + "a" * 64},
            "env": {"LABEL": "é"}}
    original = copy.deepcopy(spec)
    result = GOLDEN.call(lambda: _container_wrap(
        tmp_path / "unused.json", ["payload", "é"], spec=spec, progress=[("head", 1800)]))
    assert "returned" in result
    assert spec == original


def test_campaign_submission_preserves_inline_spec_bytes_and_envelope(tmp_path):
    from tools.dispatch_tessera_campaign import _pbrun_argv

    args = SimpleNamespace(spec=tmp_path / "unused.json", pbrun="/published/pbrun.py",
        demand="gpu=1,mem_gb=16", cpus=2, tag="gb10", priority=0,
        timeout_s=None, container_arg=[])
    spec = {"env": {"LABEL": "é"}, "container": {"image": "sha256:" + "a" * 64}}
    GOLDEN.call(lambda: _pbrun_argv(args, manifest=tmp_path / "manifest.json",
        inner=["payload", "é"], container_spec=spec), tmp=tmp_path)


def test_campaign_row_embeds_the_resolved_container_spec():
    from tools.dispatch_tessera_campaign import _row

    spec = {"cwd": "/work", "python": "python3", "cpus": 2, "tags": ["gb10"],
            "env": {"LABEL": "é"}, "container": {"image": "sha256:" + "a" * 64}}
    GOLDEN.call(lambda: _row(spec, ["--value", "é"], mem_gb=16,
                            timeout_s=30, module="prismaquant.tessera_campaign"))


def test_standalone_worker_atomic_publication_keeps_ascii_and_final_newline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "worker_2320", root / "tools/tessera_fleet/model_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    path = tmp_path / "prepared" / "identity.json"
    GOLDEN.call(lambda: worker.atomic_json(path, {"b": [], "a": "é"}))
    GOLDEN.value(path.read_bytes())
    assert not list(path.parent.glob("*.tmp"))


def _standalone(name, args, tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    script = Path(__file__).resolve().parents[1] / "tools" / name
    return subprocess.run(["/usr/bin/python3", "-S", str(script), *args],
                          cwd=tmp_path, env=env, capture_output=True, check=False)


@pytest.mark.parametrize("name", [row["file"].removeprefix("tools/")
    for row in LAUNCH_CONTRACTS if row["kind"] == "Standard-library argument/help entry"])
def test_lightweight_cli_help_works_without_scientific_dependencies(name, tmp_path):
    result = _standalone(name, ["--help"], tmp_path)
    assert result.returncode == 0, result.stderr.decode()


def test_checkpoint_cli_keeps_argument_refusal_before_scientific_import(tmp_path):
    result = _standalone("checkpoint_parse_probe.py", [], tmp_path)
    assert result.returncode == 2, result.stderr.decode()


def test_chain_spec_cli_works_without_scientific_dependencies(tmp_path):
    spec = tmp_path / "base.json"
    spec.write_text(json.dumps({"container": {"mounts": []}, "env": {"LABEL": "é"}}))
    result = _standalone("chain_roll_bench.py", ["spec", "--base-spec", str(spec),
        "--scratch", "/scratch-é"], tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == (
        b'{"container": {"mounts": [{"readonly": false, "source": "/scratch-\\u00e9", '
        b'"target": "/scratch-\\u00e9"}]}, "env": {"LABEL": "\\u00e9"}}\n')


def test_render_analysis_cli_publishes_without_scientific_dependencies(tmp_path):
    row = {"arm": "base", "label": "base-r0", "window_count": 1,
           "windows": {"0": {"timers": {}, "wall_s": 1.0}},
           "gpu_power": {}, "projection_digest": "a" * 64}
    (tmp_path / "base-r0.json").write_text(json.dumps(row))
    result = _standalone("render_window_bench.py", ["analyze", "--out", str(tmp_path)], tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads((tmp_path / "analysis.json").read_bytes())["identical_projections"] is True


def _scientific_environment():
    env = dict(os.environ)
    dependencies = env.pop("PQ_TEST_DEPENDENCY_PATH", None)
    if dependencies:
        env["PYTHONPATH"] = dependencies
    else:
        env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("row", [row for row in LAUNCH_CONTRACTS
    if row["kind"] == "Package-dependent argument/help entry"], ids=lambda row: row["file"])
def test_package_dependent_entrypoint_help_uses_its_supported_context(row):
    env = _scientific_environment()
    result = subprocess.run([sys.executable, *row["supported_probe"][1:]],
        cwd=ROOT, env=env, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_split_forward_subcommand_keeps_its_supported_bootstrap():
    env = _scientific_environment()
    result = subprocess.run([sys.executable, "-m", "tools.build_stagea_split_package",
        "forward", "--help"], cwd=ROOT, env=env, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_source_adoption_batch_reaches_only_its_actual_input_boundary():
    """Real imports and input open, not source adoption or 120-header execution."""
    expected = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-joint-01/prepare/source-identity.json"
    probe = r'''
import json, os, runpy, sys
expected = sys.argv[1]
output_root = "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/cpu-export-codex-20260922"
denied = []
publications = []
class ApplicationInputDenied(Exception):
    pass
def audit(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if path == expected:
        denied.append(path)
        raise ApplicationInputDenied("bootstrap application input deliberately denied")
    flags = args[2]
    if path.startswith(output_root + "/") and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
        publications.append(path)
        raise RuntimeError("unexpected output publication before the input boundary")
sys.addaudithook(audit)
try:
    runpy.run_module("tools.prove_glm_source_adoption", run_name="__main__")
except ApplicationInputDenied:
    print(json.dumps({"scope": "bootstrap through first application input only",
        "denied_open": denied, "publications": publications,
        "reason": "bootstrap application input deliberately denied"}, sort_keys=True))
    raise SystemExit(86)
raise SystemExit("script did not reach the actual denied input")
'''
    env = _scientific_environment()
    result = subprocess.run([sys.executable, "-c", probe, expected],
        cwd=ROOT, env=env, capture_output=True, check=False)
    assert result.returncode == 86, result.stderr.decode()
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["denied_open"] == [expected]
    assert observed["publications"] == []
    assert observed["reason"] == "bootstrap application input deliberately denied"


@pytest.mark.parametrize("name", ["check_stageb_a4_quantizer.py", "check_stageb_a4_group_quantizer.py"])
def test_native_batch_entrypoint_runs_its_real_tiny_cuda_checker(name):
    import hashlib
    import torch
    if not torch.cuda.is_available():
        pytest.skip("actual tiny registered-operator batch bootstrap requires CUDA")
    args = []
    if name == "check_stageb_a4_group_quantizer.py":
        policy = Path("/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/r13-stageb-20260923/a4/served-activation-policy.json")
        args = ["--policy", str(policy), "--policy-sha256", hashlib.sha256(policy.read_bytes()).hexdigest()]
    result = subprocess.run([sys.executable, str(ROOT / "tools" / name), *args],
        cwd=ROOT, env=_scientific_environment(), capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "passed"


def test_separately_staged_model_worker_keeps_its_standard_library_help(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_bytes((ROOT / "tools/tessera_fleet/model_worker.py").read_bytes())
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(["/usr/bin/python3", "-S", str(worker), "--help"],
        cwd=tmp_path, env=env, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
