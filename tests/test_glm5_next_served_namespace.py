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


# ---------------------------------------------------------------------------
# One source for "which layers are MTP": the streamed loader's drop rule
# ---------------------------------------------------------------------------
def _glm_config(first, count):
    return {"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
            **_text(first, count)}


def test_detection_declares_the_config_and_the_loader_drops_its_draft_range(tmp_path):
    from prismaquant.model_profiles.registry import detect_profile

    (tmp_path / "config.json").write_text(json.dumps(_glm_config(10, 1)))
    profile = detect_profile(str(tmp_path))
    assert isinstance(profile, Glm5NextProfile)
    draft = "model.language_model.layers.10.mlp.experts.0.down_proj.weight"
    body = "model.language_model.layers.45.mlp.experts.0.down_proj.weight"
    assert profile.is_mtp_checkpoint_key(draft)
    assert profile.checkpoint_to_live_name(draft) is None
    # Layer 45 is body here: the literal no longer decides for a detected profile.
    assert not profile.is_mtp_checkpoint_key(body)
    assert profile.checkpoint_to_live_name(body) == body


def test_a_config_given_to_detection_is_the_one_declared(tmp_path):
    from prismaquant.model_profiles.registry import detect_profile

    (tmp_path / "config.json").write_text(json.dumps(_glm_config(45, 1)))
    profile = detect_profile(str(tmp_path), config=_glm_config(3, 2))
    assert profile.is_mtp_checkpoint_key("model.language_model.layers.4.self_attn.o_proj.weight")
    assert not profile.is_mtp_checkpoint_key("model.language_model.layers.45.mlp.gate.weight")


def test_config_only_detection_declares_the_config():
    profile = profile_from_config(_glm_config(45, 0))
    assert not profile.is_mtp_checkpoint_key("model.language_model.layers.45.mlp.gate.weight")


def test_a_declared_config_that_states_no_range_raises_rather_than_guessing():
    profile = profile_from_config({"model_type": "glm5_next",
                                   "text_config": {"num_hidden_layers": 45}})
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        profile.checkpoint_to_live_name("model.language_model.layers.45.mlp.gate.weight")


def test_a_hand_built_profile_falls_back_to_the_glm53_flash_layer():
    """No config.json was declared, so the documented literal decides."""
    profile = Glm5NextProfile()
    assert profile.is_mtp_checkpoint_key("model.language_model.layers.45.mlp.gate.weight")
    assert not profile.is_mtp_checkpoint_key("model.language_model.layers.44.mlp.gate.weight")
