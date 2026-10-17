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


def _check(tmp_path, document, *, stage="probe"):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(document))
    artifact = tmp_path / "probe.pkl"
    code = pipeline.main([
        "--check-stage-settings", "--stage-settings", str(request),
        "--artifact", str(artifact), "--stage", stage,
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
