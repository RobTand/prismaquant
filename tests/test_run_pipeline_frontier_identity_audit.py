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



@pytest.mark.parametrize("changed", [False, True])
def test_inplace_point_reuse_binds_current_assignment(tmp_path, changed):
    from prismaquant.pipeline import stage_settings_document

    settings = stage_request()
    request = tmp_path / "stage_settings.json"
    request.write_text(json.dumps(stage_settings_document(settings)))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (tmp_path / "logs").mkdir()
    base = artifacts / "layer_config.json"
    base.write_text(json.dumps({"base": "BF16"}))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({"q": "NVFP4"}))
    point = tmp_path / "vak_row.json"
    guard = section("require_stage_settings() {", "# CB / GGUF col-weights")
    loop = section("      VAK_PART_FILES=()", '      python3 - "$VALIDATION_JSON"')
    body = guard + "\nVAK_COMMON_ARGS=()\nVALIDATED_ASSIGNMENT_ARGS=(--assignment " + shlex.quote(
        f"row={assignment}") + ")\n" + loop
    values = {"VAK_PART_DIR": tmp_path, "STAGE_SETTINGS_PATH": request,
              "WORK_DIR": tmp_path, "PROBE_PATH": base, "COST_PATH": base,
              "PROD_CACHE_RAW": base, "VALIDATED_FRONTIER_DATASET": "corpus"}
    first = run_shell(tmp_path, body, values)
    assert first.returncode == 0, first.stderr
    point.write_text(json.dumps({"results": [{"label": "row"}]}))
    (tmp_path / "calls.txt").unlink()
    if changed:
        assignment.write_text(json.dumps({"q": "FP8_DYNAMIC"}))
    result = run_shell(tmp_path, body, values)
    if changed:
        assert result.returncode != 0 or (tmp_path / "calls.txt").exists(), (
            "The loop reused old KL after the candidate assignment changed.",
            result.stdout, result.stderr,
        )
    else:
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "calls.txt").exists()
