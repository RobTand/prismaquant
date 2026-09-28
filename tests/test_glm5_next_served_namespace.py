"""GLM-5.3's served module namespace, as the profile states it (PQ #1490).

The route.trace gate compares priced checkpoint targets with the module names
a serve traced. ``Glm5NextProfile.served_module_name`` is the one place that
map lives: the MTP draft range (read from ``config.json``) goes to the draft
model's bare ``model.layers.N.…`` namespace, everything else to the body's
``language_model.model.…``. A profile that attests no map is the identity.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.model_profiles.registry import profile_from_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
BAL_CONFIG = ROOT / "tests" / "fixtures" / "tessera_route_trace_1490" / "config.json"
M44E1_CONFIG = ROOT / "tests" / "fixtures" / "tessera_route_trace_m44e1" / "config.json"


def _text(first, count):
    return {"text_config": {"num_hidden_layers": first, "num_nextn_predict_layers": count}}


def test_the_bal_config_puts_the_draft_at_layer_45():
    config = json.loads(BAL_CONFIG.read_text())
    profile = profile_from_config(config)
    assert isinstance(profile, Glm5NextProfile)
    assert profile.mtp_draft_layer_range(config) == range(45, 46)
    assert profile.served_module_name(
        "model.language_model.layers.45.mlp.experts", config) == "model.layers.45.mlp.experts"
    assert profile.served_module_name(
        "model.language_model.layers.44.mlp.experts", config
    ) == "language_model.model.layers.44.mlp.experts"
    assert profile.served_module_name(
        "model.language_model.layers.0.mlp.down_proj", config
    ) == "language_model.model.layers.0.mlp.down_proj"


def test_no_nextn_layers_is_no_draft_range():
    """The m44e1 4-layer artifact states ``num_nextn_predict_layers: 0``."""
    config = json.loads(M44E1_CONFIG.read_text())
    profile = Glm5NextProfile()
    assert profile.mtp_draft_layer_range(config) == range(4, 4)
    for layer in (0, 3, 4):
        name = f"model.language_model.layers.{layer}.mlp.experts"
        assert profile.to_mtp_draft_module_name(name, config) is None
        assert profile.served_module_name(name, config) == (
            f"language_model.model.layers.{layer}.mlp.experts")


def test_the_range_is_derived_from_the_config_and_not_a_literal_45():
    config = _text(10, 2)
    profile = Glm5NextProfile()
    assert profile.mtp_draft_layer_range(config) == range(10, 12)
    assert profile.served_module_name(
        "model.language_model.layers.9.mlp.experts", config
    ) == "language_model.model.layers.9.mlp.experts"
    for layer in (10, 11):
        assert profile.served_module_name(
            f"model.language_model.layers.{layer}.mlp.shared_experts.down_proj", config
        ) == f"model.layers.{layer}.mlp.shared_experts.down_proj"
    assert profile.served_module_name(
        "model.language_model.layers.45.mlp.experts", config
    ) == "language_model.model.layers.45.mlp.experts"


def test_a_config_with_no_text_config_is_read_at_the_top_level():
    config = {"num_hidden_layers": 45, "num_nextn_predict_layers": 1}
    assert Glm5NextProfile.mtp_draft_layer_range(config) == range(45, 46)


def test_only_checkpoint_names_are_read_as_draft_targets():
    config = _text(45, 1)
    profile = Glm5NextProfile()
    # Recipe space and the served spellings are not config_groups targets.
    assert profile.to_mtp_draft_module_name("model.layers.45.mlp.experts", config) is None
    assert profile.to_mtp_draft_module_name(
        "language_model.model.layers.45.mlp.experts", config) is None


@pytest.mark.parametrize("config", [
    pytest.param({"text_config": {"num_hidden_layers": 45}}, id="no-nextn-count"),
    pytest.param({"text_config": {"num_nextn_predict_layers": 1}}, id="no-layer-count"),
    pytest.param(_text(45, -1), id="negative"),
    pytest.param(_text(45, True), id="bool"),
    pytest.param(_text("45", 1), id="string"),
    pytest.param({}, id="empty"),
])
def test_a_range_the_config_does_not_state_raises(config):
    with pytest.raises(ValueError, match="cannot be derived"):
        Glm5NextProfile.mtp_draft_layer_range(config)


def test_a_profile_with_no_map_is_the_identity():
    name = "model.language_model.layers.45.mlp.experts"
    assert DefaultProfile().served_module_name(name, {}) == name
    assert profile_from_config({}).served_module_name(name, {}) == name


def test_the_map_imports_no_serving_runtime():
    """The attestation travels as cited data; the test never imports vLLM."""
    before = "vllm" in sys.modules
    config = json.loads(BAL_CONFIG.read_text())
    profile = profile_from_config(config)
    for target in ("model.language_model.layers.45.mlp.experts",
                   "model.language_model.layers.3.mlp.experts"):
        profile.served_module_name(target, config)
    assert ("vllm" in sys.modules) == before
