"""Execute the real cost-path, cost-stage and allocator shell wiring.

Only expensive worker commands are intercepted. The driver sections and the
prepriced-input validator run unchanged under PrismaBuild's CPU reservation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "prismaquant" / "run-pipeline.sh"


def _sections():
    text = SCRIPT.read_text()
    paths = text[text.index('PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"'):
                 text.index('case "${HADAMARD_DUQUANT:-}"')]
    cost_start = text.index("# 2. Cost measurement")
    allocation_start = text.index("# 3. Allocator")
    allocation_end = text.index("# 4. Production cache", allocation_start)
    return paths, text[cost_start:allocation_start], text[allocation_start:allocation_end]


def _input(tmp_path, mode="aura"):
    path = tmp_path / "external cost with spaces.pkl"
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    payload = {
        "costs": {"model.layers.0.self_attn.o_proj": {"BF16": {"predicted_dloss": 0.0}}},
        "formats": ["BF16"], "provenance": {"cost_mode": mode},
        "meta": {"model": str(model)},
    }
    path.write_bytes(pickle.dumps(payload))
    return path, model


def _run(tmp_path, *, mode="aura", override=None,
         mutate_before_allocator=False):
    paths, costs, allocator = _sections()
    work = tmp_path / "work"
    (work / "artifacts").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    trace = tmp_path / "workers.jsonl"
    # The scoped driver has arrays that are constructed by earlier stages.
    # They are irrelevant to the cost-input edge but remain real empty arrays.
    arrays = sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\[@\]", paths + costs + allocator)))
    declarations = "\n".join(f"{name}=()" for name in arrays)
    python = shlex.quote(sys.executable)
    recorder = (
        "import json,os,sys; "
        "open(os.environ['TEST_TRACE'],'a').write(json.dumps(sys.argv[1:])+'\\n')"
    )
    preamble = f"""set -euo pipefail
{declarations}
require_stage_settings() {{ return 0; }}
cost_table_reusable() {{ return 1; }}
harvest_cb_col_weights() {{ return 0; }}
python3() {{
  if [[ "$1" == '-m' && "$2" == 'prismaquant.prepriced_cost' ]]; then
    command {python} "$@"
    return $?
  fi
  command {python} -c {shlex.quote(recorder)} "$@"
  if [[ "$1" == '-m' && "$2" == 'prismaquant.allocator' ]]; then
    return 88
  fi
  return 77
}}
"""
    mutation = ""
    if mutate_before_allocator:
        mutation = '\nprintf "changed after validation" >> "$COST_PATH_OVERRIDE"\n'
    env = os.environ.copy()
    # Set only script-local knobs. Preserve system paths/home/interpreter state.
    referenced = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", paths + costs + allocator))
    for name in referenced - {"HOME", "PATH", "CODEX_HOME", "PYTHONPATH", "TMPDIR"}:
        env.setdefault(name, "")
    env.update({
        "WORK_DIR": str(work), "MODEL_PATH": str(tmp_path / "model"),
        "COST_MODE": mode, "COST_PATH_OVERRIDE": str(override or ""),
        "CB_LADDER_INTERP": "0", "COST_CACHE_COL_WEIGHTS_REQUIRED": "0",
        "SELECTION_MODE": "surrogate",
        "FORMATS": "BF16", "COST_FORMATS": "BF16",
        "TARGET_BITS": "16", "TARGET_PROFILE_DEFAULT": "research",
        "LM_HEAD_FORMAT_CANONICAL": "BF16", "MTP_FORMAT": "BF16",
        "VISUAL_FORMAT": "BF16", "TEST_TRACE": str(trace),
        "CUDA_VISIBLE_DEVICES": "",
    })
    result = subprocess.run(
        ["bash", "-c", preamble + paths + costs + mutation + allocator],
        cwd=SCRIPT.parents[1], env=env, text=True, capture_output=True,
        timeout=60, check=False,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return result, calls, work


@pytest.mark.parametrize("mode", ["aura"])
def test_actual_driver_passes_exact_override_to_allocator_without_builders(tmp_path, mode):
    supplied, _ = _input(tmp_path, mode)
    before = hashlib.sha256(supplied.read_bytes()).hexdigest()
    result, calls, work = _run(tmp_path, mode=mode, override=supplied)

    assert result.returncode == 88, (result.stdout, result.stderr, calls)
    assert len(calls) == 1 and calls[0][:2] == ["-m", "prismaquant.allocator"]
    assert calls[0][calls[0].index("--costs") + 1] == str(supplied)
    assert hashlib.sha256(supplied.read_bytes()).hexdigest() == before
    receipt = json.loads((work / "artifacts" / "prepriced_cost_input.json").read_text())
    assert receipt["sha256"] == before


@pytest.mark.parametrize("condition", ["missing", "mode-mismatch", "malformed"])
def test_override_refusal_precedes_any_cost_worker(tmp_path, condition):
    # A table stamped by another producer mode than the run's.
    supplied, _ = _input(tmp_path, "local" if condition == "mode-mismatch" else "aura")
    if condition == "missing":
        supplied = tmp_path / "missing.pkl"
    elif condition == "malformed":
        supplied.write_bytes(pickle.dumps({"not_costs": {}}))
    result, calls, _ = _run(tmp_path, mode="aura", override=supplied)
    assert result.returncode == 2, (result.stdout, result.stderr, calls)
    assert calls == []


def test_override_changed_after_preflight_never_reaches_allocator(tmp_path):
    supplied, _ = _input(tmp_path)
    result, calls, _ = _run(tmp_path, override=supplied, mutate_before_allocator=True)
    assert result.returncode == 2, (result.stdout, result.stderr, calls)
    assert calls == []


def test_no_override_preserves_the_actual_cost_builder_path(tmp_path):
    result, calls, _ = _run(tmp_path)
    assert result.returncode == 77, (result.stdout, result.stderr, calls)
    assert calls[0][:2] == ["-m", "prismaquant.incremental_measure_quant_cost"]
