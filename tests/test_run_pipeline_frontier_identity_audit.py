"""Check the pipeline shell contract without a model load."""

import json
import os
import pickle
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



REUSE_CHANGES = [
    "unchanged", "changed-candidate", "rewritten-metadata", "base-solver-seconds",
    "changed-base", "changed-corpus", "candidate-metadata",
]


@pytest.mark.parametrize(("shape", "change"), [
    (shape, change) for shape in ("flat", "pareto-v2") for change in REUSE_CHANGES
] + [("pareto-v2", "wrapper-metadata")])
def test_inplace_point_reuse_binds_current_assignment(tmp_path, shape, change):
    from prismaquant.pipeline import stage_settings_document

    settings = stage_request()
    request = tmp_path / "stage_settings.json"
    request.write_text(json.dumps(stage_settings_document(settings)))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (tmp_path / "logs").mkdir()
    base = artifacts / "layer_config.json"
    base_payload = {"base": "BF16", "__prismaquant__": {
        "solve_diagnostics": {"4.75": {"solver_seconds": 1.0}},
    }}
    base.write_text(json.dumps(base_payload))
    assignment = tmp_path / "assignment.json"
    candidate_assignment = {"q": "NVFP4"}
    candidate_payload = candidate_assignment
    if shape == "pareto-v2":
        candidate_payload = {
            "schema": "prismaquant.allocator.pareto_assignment.v2",
            "label": "allocator_target_4p7500_achieved_4p5000_fixture",
            "source": "allocator_pareto",
            "target_bits": 4.75,
            "achieved_bits": 4.5,
            "bits_total": 576.0,
            "bits_total_with_aux": 576.0,
            "predicted_dloss": 0.125,
            "total_predicted_dloss_with_aux": 0.125,
            "format_counts": {"NVFP4": 1},
            "artifact_bytes": None,
            "artifact_byte_scope": None,
            "artifact_tensor_payload_bytes": None,
            "artifact_tensor_payload_scope": None,
            "whole_artifact_upper_bound_bytes": None,
            "target_profile": "vllm_packed_moe",
            "assignment": candidate_assignment,
        }
    assignment.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"calibration text\n")
    point = tmp_path / "vak_row.json"
    from prismaquant.production_weight_cache import ProductionWeightCache
    probe, costs, cache = (tmp_path / name for name in ("probe.pkl", "cost.pkl", "cache.pkl"))
    probe.write_bytes(pickle.dumps({"stats": {}}))
    costs.write_bytes(pickle.dumps({"costs": {}, "provenance": {}}))
    cache.write_bytes(pickle.dumps(ProductionWeightCache(weights={}, levers={}, metadata={})))
    guard = section("require_stage_settings() {", "# CB / GGUF col-weights")
    loop = section("      VAK_PART_FILES=()", '      python3 - "$VALIDATION_JSON"')
    body = guard + "\nVAK_COMMON_ARGS=()\nVALIDATED_ASSIGNMENT_ARGS=(--assignment " + shlex.quote(
        f"row={assignment}") + ")\n" + loop
    values = {"VAK_PART_DIR": tmp_path, "STAGE_SETTINGS_PATH": request,
              "WORK_DIR": tmp_path, "PROBE_PATH": probe, "COST_PATH": costs,
              "PROD_CACHE_RAW": cache, "VALIDATED_FRONTIER_DATASET": corpus}
    first = run_shell(tmp_path, body, values)
    assert first.returncode == 0, first.stderr
    point.write_text(json.dumps({"results": [{"label": "row"}]}))
    (tmp_path / "calls.txt").unlink()
    if change == "changed-candidate":
        candidate_assignment["q"] = "FP8_DYNAMIC"
        assignment.write_text(json.dumps(candidate_payload))
    if change == "base-solver-seconds":
        base_payload["__prismaquant__"]["solve_diagnostics"]["4.75"]["solver_seconds"] = 2.0
        base.write_text(json.dumps(base_payload))
    if change == "changed-base":
        base_payload["base"] = "FP8_DYNAMIC"
        base.write_text(json.dumps(base_payload))
    if change == "changed-corpus":
        corpus.write_bytes(b"Calibration text\n")
    if change == "candidate-metadata":
        candidate_assignment["__prismaquant__"] = {
            "solve_diagnostics": {"4.75": {"solver_seconds": 2.0}},
        }
        assignment.write_text(json.dumps(candidate_payload))
    if change == "wrapper-metadata":
        candidate_payload["predicted_dloss"] = 0.25
        assignment.write_text(json.dumps(candidate_payload))
    if change == "rewritten-metadata":
        payload = pickle.loads(costs.read_bytes())
        payload["provenance"]["additivity"] = {"measured_kl": 0.125}
        costs.write_bytes(pickle.dumps(payload))
        payload = pickle.loads(cache.read_bytes())
        payload.metadata["lazy_expert_renders"] = 1
        cache.write_bytes(pickle.dumps(payload))
    result = run_shell(tmp_path, body, values)
    if change in {"changed-candidate", "changed-base", "changed-corpus"}:
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "DIFFERENT settings" in result.stdout + result.stderr
        assert not (tmp_path / "calls.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "calls.txt").exists()


def test_assignment_reader_refuses_nonexportable_selection_request(tmp_path):
    from prismaquant.layer_config import load_assignment
    from prismaquant.tessera_materialization import REQUEST_SCHEMA

    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "schema": REQUEST_SCHEMA,
        "status": "selection_only_nonexportable",
        "assignment": {"q": "NVFP4"},
    }))
    with pytest.raises(ValueError):
        load_assignment(path)


def test_assignment_reader_preserves_flat_assignment_module(tmp_path):
    from prismaquant.layer_config import load_assignment

    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({"assignment": {"data_type": "nv_fp", "bits": 4}}))
    assert load_assignment(path) == {"assignment": "NVFP4"}


@pytest.mark.parametrize("payload", [
    {"schema": "prismaquant.allocator.pareto_assignment.v3", "assignment": {"q": "BF16"}},
    {"schema": "prismaquant.allocator.pareto_assignment.v2"},
    *({"schema": "prismaquant.allocator.pareto_assignment.v2", "assignment": value}
      for value in (None, [], "BF16", 1)),
    {"schema": "prismaquant.allocator.pareto_assignment.v2", "assignment": {"q": 0.5}},
    {"schema": "prismaquant.allocator.pareto_assignment.v2", "assignment": {
        "schema": "prismaquant.allocator.pareto_assignment.v2", "assignment": {"q": "BF16"},
    }},
])
def test_assignment_reader_refuses_malformed_pareto_envelope(tmp_path, payload):
    from prismaquant.layer_config import load_assignment

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_assignment(path)
