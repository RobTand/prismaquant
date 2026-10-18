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
from prismaquant.layer_config import load_assignment
from prismaquant.footprint import assignment_serialization_sha256
import os
assignment = load_assignment(args[args.index("--base-assignment") + 1])
Path(args[args.index("--output") + 1]).write_text(json.dumps({
    "calibration": {"calib_hash": os.environ["AUDIT_DRAW_HASH"], "kl_scope": "full_sequence"},
    "results": [{"label": "selected", "kl_mean": 0.75, "kl_stderr": 0.02,
                 "assignment_sha256": assignment_serialization_sha256(assignment)}]
}))
PYSTUB
    return 0
  fi
  if [[ "${1:-}" == "-m" && "${2:-}" == "prismaquant.pipeline" ]]; then
    "$AUDIT_PYTHON" -c '
import runpy, sys, torch, transformers
from types import SimpleNamespace
class Tokenizer:
    eos_token_id = 0
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[int(token) for token in text.split()]]))
transformers.AutoTokenizer.from_pretrained = lambda *args, **kwargs: Tokenizer()
sys.argv = ["prismaquant.pipeline", *sys.argv[3:]]
runpy.run_module("prismaquant.pipeline", run_name="__main__")
' "$@"
    return $?
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
    values, _settings = measurement_environment(tmp_path)
    values.update(SELECTION_MODE=mode, AURA_ADDITIVITY_GATE="report")
    result = run_shell(tmp_path, block + '\nprintf "%s" "$AURA_MEASURED_KL"', values)
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
    values, _settings = measurement_environment(tmp_path)
    values["AUDIT_VALIDATE_FAIL"] = int(fails)
    result = run_shell(tmp_path, block + '\nprintf "AUDIT_RESULT=%s" "$AURA_MEASURED_KL"', values)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls.txt").exists(), "The report reused an old single-assignment KL."
    assert result.stdout.endswith("AUDIT_RESULT=" + ("" if fails else "0.75"))


def measurement_environment(tmp_path):
    import torch
    from prismaquant.pipeline import stage_settings_document
    from prismaquant.calibration_data import load_calibration
    from prismaquant.perturbed_x_cache import calibration_data_hash

    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[int(t) for t in text.split()]]))

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    assignment = artifacts / "layer_config.json"
    assignment.write_text('{"q": "FP8_DYNAMIC"}')
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": " ".join(str(1000*r+i) for i in range(128))}) + "\n" for r in range(40)))
    cache = artifacts / "cache.pkl"
    cache.write_bytes(b"cache")
    settings = dict(stage_request(), MODEL_PATH="unused", FORMATS="FP8_DYNAMIC")
    request = artifacts / "stage_settings.json"
    request.write_text(json.dumps(stage_settings_document(settings)))
    draw_hash = calibration_data_hash(load_calibration(Tokenizer(), str(corpus), 8, 128))
    return {
        "WORK_DIR": tmp_path, "STAGE_SETTINGS_PATH": request,
        "SELECTION_MODE": "surrogate", "AURA_ADDITIVITY_GATE": "measure",
        "MODEL_PATH": "unused", "PROBE_PATH": cache, "COST_PATH": cache,
        "FORMATS": "FP8_DYNAMIC", "DATASET": corpus, "AURA_COST_NSAMPLES": 8,
        "AURA_COST_SEQLEN": 128, "AURA_COST_CALIB_SEED": 42, "DEVICE": "cuda",
        "PRODUCTION_RENDER_COST_CACHE_PATH": cache, "PRODUCTION_RENDER_COST_CACHE_DIR": artifacts,
        "AUDIT_DRAW_HASH": draw_hash,
    }, settings


@pytest.mark.parametrize("record_name", ["validated_frontier_kl.json", "aura_additivity_kl.json"])
def test_bound_assignment_and_draw_reuses_measurement(tmp_path, record_name):
    from prismaquant import pipeline
    from prismaquant.digests import file_sha256hex
    from prismaquant.footprint import assignment_serialization_sha256

    values, settings = measurement_environment(tmp_path)
    assignment = tmp_path / "artifacts/layer_config.json"
    record = tmp_path / "artifacts" / record_name
    record.write_text(json.dumps({
        "calibration": {"calib_hash": values["AUDIT_DRAW_HASH"], "kl_scope": "full_sequence"},
        "results": [{"label": "selected", "kl_mean": 0.125, "kl_stderr": 0.01,
                     "assignment_sha256": assignment_serialization_sha256({"q": "FP8_DYNAMIC"})}],
    }))
    projection, missing = pipeline.stage_settings_projection(
        "assignment-kl", dict(settings,
            ASSIGNMENT_DIGEST=file_sha256hex(assignment),
            DRAW_DIGEST=values["AUDIT_DRAW_HASH"],
            MEASUREMENT_DIGEST=file_sha256hex(record),
            MEASUREMENT_CACHE_DIGEST=file_sha256hex(values["PRODUCTION_RENDER_COST_CACHE_PATH"]))) if "assignment-kl" in pipeline.STAGE_SETTINGS_KEYS else ({}, [])
    assert not missing
    Path(str(record) + ".settings.json").write_text(json.dumps({
        "schema": pipeline.STAGE_MANIFEST_SCHEMA, "stages": {"assignment-kl": projection},
    }))
    block = section('  AURA_ADDITIVITY_JSON=', '  if [[ -n "$AURA_MEASURED_KL" ]]; then')
    result = run_shell(tmp_path, block + '\nprintf "AUDIT_RESULT=%s" "$AURA_MEASURED_KL"', values)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "calls.txt").exists(), "The bound record triggered a new KL measurement."
    assert result.stdout.endswith("AUDIT_RESULT=0.125")


@pytest.mark.parametrize("drift", ["assignment_bytes", "draw", "result_bytes", "cache", "render"])
def test_bound_record_rejects_changed_inputs(tmp_path, drift):
    from prismaquant import pipeline
    from prismaquant.footprint import assignment_serialization_sha256

    values, settings = measurement_environment(tmp_path)
    assignment = tmp_path / "artifacts/layer_config.json"
    record = tmp_path / "artifacts/aura_additivity_kl.json"
    record.write_text(json.dumps({
        "calibration": {"calib_hash": values["AUDIT_DRAW_HASH"], "kl_scope": "full_sequence"},
        "results": [{"label": "selected", "kl_mean": 0.125, "kl_stderr": 0.01,
                     "assignment_sha256": assignment_serialization_sha256({"q": "FP8_DYNAMIC"})}],
    }))
    document = pipeline.stage_settings_document(settings)
    kwargs = dict(draw_digest=values["AUDIT_DRAW_HASH"], cache_path=values["PRODUCTION_RENDER_COST_CACHE_PATH"])
    assert pipeline.assignment_kl_measurement(record, assignment, document, record=True, **kwargs) == (0.125, 0.01)
    if drift == "assignment_bytes":
        assignment.write_text(assignment.read_text() + "\n")
    elif drift == "draw":
        kwargs["draw_digest"] = "another-draw"
    elif drift == "result_bytes":
        record.write_text(record.read_text() + "\n")
    elif drift == "cache":
        Path(values["PRODUCTION_RENDER_COST_CACHE_PATH"]).write_bytes(b"new-cache")
    else:
        settings["PRISMAQUANT_GPTQ_DAMP"] = "new-render"
        document = pipeline.stage_settings_document(settings)
    assert pipeline.assignment_kl_measurement(record, assignment, document, **kwargs) is None
