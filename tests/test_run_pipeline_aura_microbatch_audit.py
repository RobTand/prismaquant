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
    return 0
  fi
  "$AUDIT_PYTHON" "$@"
}
'''
    return subprocess.run(["bash", "-c", wrapper + body], env=env,
                          text=True, capture_output=True)



@pytest.mark.parametrize("microbatch", [8, 1])
def test_aura_raw_reuse_binds_requested_microbatch(tmp_path, microbatch):
    from prismaquant.pipeline import stage_settings_document

    settings = stage_request()
    settings["AURA_COST_PROBE_MICROBATCH"] = "8"
    raw = tmp_path / "cost_aura.pkl"
    old_artifact(raw, "aura-cost", settings, {"recorded_microbatch": 8})
    settings["AURA_COST_PROBE_MICROBATCH"] = str(microbatch)
    request = tmp_path / "stage_settings.json"
    request.write_text(json.dumps(stage_settings_document(settings)))
    guard = section("require_stage_settings() {", "# CB / GGUF col-weights")
    block = section('  require_stage_settings "$AURA_COST_RAW" aura-cost',
                    "  # [2d] Hybrid finalize")
    result = run_shell(tmp_path, guard + "\n" + block, {
        "AURA_COST_RAW": raw, "STAGE_SETTINGS_PATH": request,
        "AURA_COST_PROBE_MICROBATCH": microbatch,
    })
    if microbatch != 8:
        assert result.returncode != 0 or (tmp_path / "calls.txt").exists(), (
            "The raw AURA cost reused microbatch 8 after a request for 1.",
            result.stdout, result.stderr,
        )
    else:
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "calls.txt").exists()

@pytest.mark.parametrize("stage", ["aura-cost", "aura-hybrid-cost"])
@pytest.mark.parametrize("changed", [False, True])
def test_resident_microbatch_keys_raw_and_final_cost(tmp_path, stage, changed):
    from prismaquant.pipeline import check_stage_settings, stage_settings_document

    settings = dict(stage_request(), AURA_COST_STREAMING="0", AURA_COST_PROBE_MICROBATCH="8")
    artifact = tmp_path / "cost.pkl"
    old_artifact(artifact, stage, settings, {})
    if changed:
        settings["AURA_COST_PROBE_MICROBATCH"] = "1"
    code, messages = check_stage_settings(artifact, stage, stage_settings_document(settings))
    assert code == (2 if changed else 0), messages


def test_streamed_cost_ignores_unused_resident_microbatch(tmp_path):
    from prismaquant.pipeline import check_stage_settings, stage_settings_document

    settings = dict(stage_request(), AURA_COST_STREAMING="1", AURA_COST_PROBE_MICROBATCH="8")
    artifact = tmp_path / "cost.pkl"
    old_artifact(artifact, "aura-cost", settings, {})
    settings["AURA_COST_PROBE_MICROBATCH"] = "1"
    code, messages = check_stage_settings(artifact, "aura-cost", stage_settings_document(settings))
    assert code == 0, messages


def test_final_cost_binds_raw_cost_content(tmp_path):
    from prismaquant.pipeline import check_stage_settings, stage_settings_document

    settings = dict(stage_request(), AURA_RAW_DIGEST="raw-before")
    artifact = tmp_path / "cost.pkl"
    old_artifact(artifact, "aura-hybrid-cost", settings, {})
    settings["AURA_RAW_DIGEST"] = "raw-after"
    code, messages = check_stage_settings(artifact, "aura-hybrid-cost", stage_settings_document(settings))
    assert code == 2, messages
