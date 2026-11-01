"""Public driver memory-budget flow through the actual allocator.

The shipped preflight and allocator shell sections run unchanged. Tiny real
header/cost fixtures avoid model execution; no allocator or serving gate is
mocked. This covers knob admission, not a full GPU pipeline or served claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

import test_allocator_serve_constraints as fixture


SCRIPT = Path.cwd() / "prismaquant" / "run-pipeline.sh"


def _driver_sections():
    text = SCRIPT.read_text()
    preflight = text[text.index(': "${SERVE_DISPATCH_TABLE:=}"'):
                     text.index('PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"')]
    start = text.index("# 3. Allocator")
    allocator = text[start:text.index("# 4. Production cache", start)]
    return preflight, allocator


def _run_driver(tmp_path, *, budget=None, context="none"):
    model, probe, costs, stats = fixture._fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6,
    )
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    fp8_bytes, _ = fixture._artifact_bytes(model, "FP8_E4M3", stats)
    nvfp4_bytes, _ = fixture._artifact_bytes(model, "NVFP4", stats)
    table = fixture._dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.0)
    work = tmp_path / "work"
    (work / "artifacts").mkdir(parents=True)
    (work / "logs").mkdir()
    copied_probe = work / "artifacts" / "probe.pkl"
    shutil.copyfile(probe, copied_probe)
    trace = tmp_path / "allocator-argv.json"
    preflight, allocator = _driver_sections()
    arrays = sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\[@\]", allocator)))
    declarations = "\n".join(f"{name}=()" for name in arrays)
    python = shlex.quote(sys.executable)
    recorder = (
        "import json,os,sys; "
        "open(os.environ['TEST_TRACE'],'w').write(json.dumps(sys.argv[1:]))"
    )
    preamble = f"""set -euo pipefail
{declarations}
python3() {{
  command {python} -c {shlex.quote(recorder)} "$@"
  command {python} "$@" --threads 1
}}
"""
    env = os.environ.copy()
    referenced = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", preflight + allocator))
    for name in referenced - {"HOME", "PATH", "CODEX_HOME", "PYTHONPATH", "TMPDIR"}:
        env[name] = ""
    requested_budget = (nvfp4_bytes + 3010 if budget == "fits_nvfp4" else budget)
    env.update({
        "WORK_DIR": str(work), "MODEL_PATH": str(model),
        "PROBE_PATH": str(copied_probe), "COST_PATH": str(costs),
        "SELECTION_MODE": "surrogate", "CALIBRATION_MODALITY": "text-only",
        "FORMATS": "NVFP4,FP8_E4M3", "TARGET_BITS": "16",
        "TARGET_PROFILE": "vllm_packed_moe", "TARGET_PROFILE_DEFAULT": "vllm_packed_moe",
        "TARGET_DISK_GB": repr((fp8_bytes + 10_000) / fixture.fp.GB),
        "ARTIFACT_OVERHEAD_RESERVE_BYTES": str(fixture._OVERHEAD_RESERVE),
        "PARETO_TARGETS": "4.6,8.2", "VISUAL_FORMAT": "BF16", "MTP_FORMAT": "BF16",
        "LM_HEAD_FORMAT_CANONICAL": "BF16", "TEST_TRACE": str(trace),
        "SERVE_DISPATCH_TABLE": str(table) if context in {"table", "complete"} else "",
        "SERVE_WORKLOAD_MIX": "prefill:dense=1.0" if context in {"mix", "complete"} else "",
        "SERVE_DEVICE_BUDGET_BYTES": "" if requested_budget is None else str(requested_budget),
        "SERVE_KV_BYTES": "1000", "SERVE_PEAK_SCRATCH_BYTES": "2000",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
    })
    result = subprocess.run(
        ["bash", "-c", preamble + preflight + allocator], cwd=SCRIPT.parents[1],
        env=env, text=True, capture_output=True, timeout=180, check=False,
    )
    argv = json.loads(trace.read_text()) if trace.exists() else None
    return result, argv, work, requested_budget


@pytest.mark.parametrize("context", ["none", "mix"])
def test_driver_refuses_explicit_budget_without_dispatch_table(tmp_path, context):
    result, argv, work, _ = _run_driver(tmp_path, budget=1, context=context)

    assert result.returncode != 0, (result.stdout, result.stderr, argv)
    assert "budget" in (result.stdout + result.stderr).lower()
    assert not (work / "artifacts" / "layer_config.json").exists()


def test_driver_without_budget_preserves_actual_unconstrained_selection(tmp_path):
    result, argv, work, _ = _run_driver(tmp_path)

    assert result.returncode == 0, (result.stdout, result.stderr, argv)
    assert "--serve-device-budget-bytes" not in argv
    selection = json.loads((work / "artifacts" / "selection.json").read_text())
    assignment = json.loads((work / "artifacts" / "layer_config.json").read_text())
    assert selection["serve_constraints"]["active"] is False
    assert {row["data_type"] for name, row in assignment.items()
            if name != "__prismaquant__"} == {"fp8_e4m3"}


def test_driver_keeps_existing_table_without_mix_budget_refusal(tmp_path):
    result, argv, work, _ = _run_driver(tmp_path, budget=1, context="table")

    assert result.returncode != 0, (result.stdout, result.stderr, argv)
    assert argv[argv.index("--serve-device-budget-bytes") + 1] == "1"
    assert "budget" in (result.stdout + result.stderr).lower()
    assert not (work / "artifacts" / "layer_config.json").exists()


def test_driver_complete_context_enforces_actual_memory_budget(tmp_path):
    result, argv, work, budget = _run_driver(tmp_path, budget="fits_nvfp4", context="complete")

    assert result.returncode == 0, (result.stdout, result.stderr, argv)
    assert argv[argv.index("--serve-device-budget-bytes") + 1] == str(budget)
    selection = json.loads((work / "artifacts" / "selection.json").read_text())
    assignment = json.loads((work / "artifacts" / "layer_config.json").read_text())
    verdict = selection["serve_constraints"]
    assert verdict["active"] is True and verdict["feasible"] is True
    assert verdict["predicted"]["device_memory_bytes"] <= budget
    assert verdict["coverage"]["memory"]["kv_bytes"] == 1000
    assert verdict["coverage"]["memory"]["peak_scratch_bytes"] == 2000
    assert {row["data_type"] for name, row in assignment.items()
            if name != "__prismaquant__"} == {"nv_fp"}
