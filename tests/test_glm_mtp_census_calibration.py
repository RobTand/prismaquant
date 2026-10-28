from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prismaquant import glm_mtp_capture, tessera_campaign
from prismaquant.calibration_data import load_calibration_input


@pytest.fixture
def census_inputs(tmp_path):
    from safetensors.torch import save_file

    tokens = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    provenance = {
        "source": "synthetic", "seed": 0, "nsamples": 2, "seqlen": 4,
        "fit_tokens": 8,
        "fit_ids_sha256": hashlib.sha256(tokens.to(torch.int32).numpy().tobytes()).hexdigest(),
    }
    artifact = tmp_path / "calibration.safetensors"
    save_file({"calibration_ids": tokens}, str(artifact),
              metadata={"calibration_provenance": json.dumps(provenance)})
    _, receipt = load_calibration_input(
        artifact, expected_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        n_samples=2, seqlen=4)
    identity = {
        "text_sha256": hashlib.sha256(b"synthetic corpus").hexdigest(),
        "fit_ids_sha256": provenance["fit_ids_sha256"],
        "calibration_input": receipt,
    }
    base = tessera_campaign.calibration_census(
        {"body": 8}, {"body": 1.0},
        args=SimpleNamespace(model="synthetic", nsamples=2, seqlen=4, seed=0,
                             layer_stride=1),
        groups={"body": ["body"]}, dense_targets=["body"], expert_targets=[],
        shapes={"body": [4, 4]}, identity=identity)
    base_path = tmp_path / "base-census.json"
    base_path.write_text(json.dumps(base))
    unit = "model.language_model.layers.45.mlp.shared_experts.down_proj"
    return {
        "base_census": base,
        "base_census_ref": {"path": str(base_path),
                            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest()},
        "canonical_capture_ref": {"path": "capture_manifest.json", "sha256": "c" * 64},
        "final_hidden_ref": {"path": "final-hidden.json", "sha256": "d" * 64},
        "layer": 45, "units": {unit: [4, 4]}, "counts": {unit: 6},
        "max_abs": {unit: 1.0}, "groups": {"mtp": [unit]},
        "model_load_contract": None, "attention_implementation": "eager",
        "capture_runtime": {}, "expert_projection": {},
    }, identity


def test_mtp_census_preserves_bound_calibration_input(census_inputs):
    arguments, identity = census_inputs
    original = copy.deepcopy(arguments)
    census = glm_mtp_capture.mtp_census(**arguments)

    tessera_campaign.require_census_draw(census, identity, where="calibration capture")
    assert census["calibration_input"] == identity["calibration_input"]
    assert json.loads(json.dumps(census))["calibration_input"] == identity["calibration_input"]
    assert arguments == original


@pytest.mark.parametrize("change", ["changed", "dropped"])
def test_mtp_census_reader_refuses_changed_or_dropped_receipt(census_inputs, change):
    arguments, identity = census_inputs
    original = copy.deepcopy(arguments)
    census = copy.deepcopy(glm_mtp_capture.mtp_census(**arguments))
    if change == "changed":
        census["calibration_input"]["artifact_sha256"] = "e" * 64
    else:
        census.pop("calibration_input", None)

    with pytest.raises(RuntimeError, match="calibration input receipt differs"):
        tessera_campaign.require_census_draw(census, identity, where="calibration capture")
    assert arguments == original


@pytest.mark.parametrize("field", ["text_sha256", "fit_ids_sha256"])
def test_mtp_census_reader_refuses_different_draw(census_inputs, field):
    arguments, identity = census_inputs
    census = glm_mtp_capture.mtp_census(**arguments)
    changed_identity = dict(identity, **{field: hashlib.sha256(b"different draw").hexdigest()})

    with pytest.raises(RuntimeError, match=field):
        tessera_campaign.require_census_draw(
            census, changed_identity, where="calibration capture")


def test_mtp_census_without_optional_receipt_keeps_legacy_draw(census_inputs):
    arguments, identity = census_inputs
    arguments["base_census"].pop("calibration_input")
    legacy_identity = {key: value for key, value in identity.items()
                       if key != "calibration_input"}
    original = copy.deepcopy(arguments)
    census = glm_mtp_capture.mtp_census(**arguments)

    assert "calibration_input" not in census
    tessera_campaign.require_census_draw(census, legacy_identity, where="calibration capture")
    with pytest.raises(RuntimeError, match="calibration input receipt differs"):
        tessera_campaign.require_census_draw(census, identity, where="calibration capture")
    assert arguments == original
