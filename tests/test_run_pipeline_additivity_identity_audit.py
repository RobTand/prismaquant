"""Check the pipeline shell contract without a model load."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "prismaquant/run-pipeline.sh"


def section(start, end):
    source = SCRIPT.read_text()
    first = source.index(start)
    return source[first:source.index(end, first)]


def stage_request():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS

    return {source: "unchanged" for keys in STAGE_SETTINGS_KEYS.values()
            for _, source in keys}


def old_artifact(path, stage, settings, content):
    from prismaquant.pipeline import check_stage_settings, stage_settings_document

    rc, messages = check_stage_settings(path, stage, stage_settings_document(settings))
    assert rc == 0, messages
    path.write_text(json.dumps(content))


def run_shell(tmp_path, body, values):
    env = dict(os.environ, **{key: str(value) for key, value in values.items()})
    env["AUDIT_PYTHON"] = sys.executable
    env["AUDIT_CALLS"] = str(tmp_path / "calls.txt")
    wrapper = r'''
set -euo pipefail
python3() {
  if [[ "${1:-}" == "-m" && ( "${2:-}" == "prismaquant.validate_assignments_kl" || "${2:-}" == "prismaquant.aura_cost" ) ]]; then
    printf '%s\n' "$*" >> "$AUDIT_CALLS"
    if [[ "${AUDIT_VALIDATE_FAIL:-0}" == "1" ]]; then return 1; fi
    "$AUDIT_PYTHON" - "$@" <<'PYSTUB'
import json, sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index("--output") + 1]).write_text(json.dumps({
    "results": [{"label": "selected", "kl_mean": 0.75, "kl_stderr": 0.02}]
}))
PYSTUB
    return 0
  fi
  "$AUDIT_PYTHON" "$@"
}
'''
    return subprocess.run(["bash", "-c", wrapper + body], env=env,
                          text=True, capture_output=True)



@pytest.mark.parametrize("mode", ["validated-surrogate", "surrogate"])
def test_additivity_does_not_reuse_old_frontier_in_surrogate_run(tmp_path, mode):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "validated_frontier_kl.json").write_text(json.dumps({
        "results": [{"label": "old", "kl_mean": 0.125, "kl_stderr": 0.01}],
    }))
    (artifacts / "validated_frontier_selection.json").write_text(json.dumps({
        "selected": {"label": "old"},
    }))
    (artifacts / "layer_config.json").write_text(json.dumps({"q": "FP8_DYNAMIC"}))
    block = section('  AURA_ADDITIVITY_JSON=', '  if [[ -n "$AURA_MEASURED_KL" ]]; then')
    result = run_shell(tmp_path, block + '\nprintf "%s" "$AURA_MEASURED_KL"', {
        "WORK_DIR": tmp_path, "SELECTION_MODE": mode, "AURA_ADDITIVITY_GATE": "report",
    })
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", "The report adopted an old frontier KL."


@pytest.mark.parametrize("fails", [False, True])
def test_additivity_remeasures_old_single_assignment(tmp_path, fails):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (tmp_path / "logs").mkdir()
    (artifacts / "layer_config.json").write_text('{"q": "FP8_DYNAMIC"}')
    (artifacts / "aura_additivity_kl.json").write_text(json.dumps({
        "results": [{"label": "selected", "kl_mean": 0.125}],
    }))
    block = section('  AURA_ADDITIVITY_JSON=', '  if [[ -n "$AURA_MEASURED_KL" ]]; then')
    values = {
        "WORK_DIR": tmp_path, "SELECTION_MODE": "surrogate", "AURA_ADDITIVITY_GATE": "measure",
        "MODEL_PATH": "unused", "PROBE_PATH": "unused", "COST_PATH": "unused",
        "FORMATS": "FP8_DYNAMIC", "DATASET": "unused", "AURA_COST_NSAMPLES": 8,
        "AURA_COST_SEQLEN": 128, "AURA_COST_CALIB_SEED": 42, "DEVICE": "cuda",
        "PRODUCTION_RENDER_COST_CACHE_PATH": "unused", "PRODUCTION_RENDER_COST_CACHE_DIR": "unused",
        "AUDIT_VALIDATE_FAIL": int(fails),
    }
    result = run_shell(tmp_path, block + '\nprintf "AUDIT_RESULT=%s" "$AURA_MEASURED_KL"', values)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls.txt").exists(), "The report reused an old single-assignment KL."
    assert result.stdout.endswith("AUDIT_RESULT=" + ("" if fails else "0.75"))
