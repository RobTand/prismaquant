"""Current request-document intake, separate from legacy artifact reuse."""
from __future__ import annotations

import json

import pytest

from prismaquant import pipeline


def _request():
    return pipeline.stage_settings_document({
        "MODEL_PATH": "source", "DATASET": "wikitext", "NSAMPLES": "1",
        "SEQLEN": "8", "CALIBRATION_MODALITY": "text",
    })


def _check(tmp_path, document, *, stage="probe", overrides=()):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(document))
    artifact = tmp_path / "probe.pkl"
    code = pipeline.main([
        "--check-stage-settings", "--stage-settings", str(request),
        "--artifact", str(artifact), "--stage", stage,
        *(argument for value in overrides for argument in ("--setting", value)),
    ])
    return code, artifact, tmp_path / "probe.pkl.settings.json"


@pytest.mark.parametrize("damage", ["unknown_stage", "missing_stage", "partial_stage"])
def test_public_guard_refuses_unbound_current_request(tmp_path, damage):
    document = _request()
    stage = "probe"
    if damage == "unknown_stage":
        stage = "probe-typo"
    elif damage == "missing_stage":
        document["artifacts"].pop(stage)
    else:
        document["artifacts"][stage].pop("SEQLEN")
    code, _artifact, manifest = _check(tmp_path, document, stage=stage)
    assert code == 2
    assert not manifest.exists()


def test_complete_current_request_remains_admitted(tmp_path):
    document = _request()
    code, artifact, manifest = _check(tmp_path, document)
    assert code == 0
    assert json.loads(manifest.read_text())["stages"]["probe"] == document["artifacts"]["probe"]
    artifact.write_bytes(b"acknowledged artifact")
    assert _check(tmp_path, document)[0] == 0


def test_complete_request_keeps_legacy_unknown_artifact_reuse(tmp_path):
    (tmp_path / "probe.pkl").write_bytes(b"historical artifact")
    code, _artifact, manifest = _check(tmp_path, _request())
    assert code == 0
    stored = json.loads(manifest.read_text())["stages"]["probe"]
    assert stored[pipeline.UNVERIFIED_SETTINGS_KEY]["settings_identity"] == "unknown"


@pytest.mark.parametrize("damage", [
    "schema", "artifacts_mapping", "unresolved_mapping", "stage_mapping",
    "unresolved_sources", "setting_value",
])
def test_public_guard_refuses_malformed_current_request(tmp_path, damage):
    document = _request()
    if damage == "schema":
        document["schema"] = "unknown"
    elif damage == "artifacts_mapping":
        document["artifacts"] = []
    elif damage == "unresolved_mapping":
        document["unresolved"] = []
    elif damage == "stage_mapping":
        document["artifacts"]["probe"] = []
    elif damage == "unresolved_sources":
        document["unresolved"]["probe"] = None
    else:
        document["artifacts"]["probe"]["NSAMPLES"] = True
    code, _artifact, manifest = _check(tmp_path, document)
    assert code == 2
    assert not manifest.exists()


def test_public_guard_resolves_aliased_late_override(tmp_path):
    stage = "aura-dw-cache"
    settings = {source: "value" for _key, source in pipeline.STAGE_SETTINGS_KEYS[stage]
                if source != "NSAMPLES"}
    document = pipeline.stage_settings_document(settings)
    assert document["unresolved"][stage] == ["NSAMPLES"]
    code, _artifact, manifest = _check(
        tmp_path, document, stage=stage, overrides=["NSAMPLES=8"])
    assert code == 0
    assert json.loads(manifest.read_text())["stages"][stage]["NS"] == "8"


def test_public_guard_preserves_optional_tessera_plan_scope(tmp_path):
    document = pipeline.stage_settings_document({
        "MODEL_PATH": "source", "TESSERA_PLAN_COVER": "as-allocated", "ASSIGNMENT_DIGEST": "a" * 64,
    })
    code, _artifact, manifest = _check(tmp_path, document, stage="tessera-plan")
    assert code == 0
    assert json.loads(manifest.read_text())["stages"]["tessera-plan"] == {
        "MODEL_PATH": "source", "COVER": "as-allocated", "ASSIGNMENT_DIGEST": "a" * 64,
    }
