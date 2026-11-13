"""Text-only skeleton construction for vision-language *wrapper* configs.

Issue #12: a user running a text-only calibration on MiniMax-M3 hit

    ValueError: Unrecognized configuration class MiniMaxM3VLConfig for this
    kind of AutoModel: AutoModelForCausalLM. Model type should be one of
    ... MiniMaxM3VLTextConfig ...

i.e. the *text* config is natively supported and only the VL wrapper that
sits above it is not. The calibration data must not decide whether a
skeleton can be built at all.

Everything here is synthetic and CPU-only: configs are constructed from
dicts and skeletons under `init_empty_weights()` (meta tensors), so no
checkpoint is read and no weights are allocated.
"""
from __future__ import annotations

import json

import pytest
import transformers
from transformers import AutoModelForCausalLM, PreTrainedConfig

from prismaquant.streaming_model import (
    _auto_causal_lm_can_resolve,
    _config_rebuilt_as,
    _resolve_declared_model_cls,
    _resolve_text_only_skeleton,
    _skeleton_config_and_class,
)

try:
    from accelerate import init_empty_weights
except ModuleNotFoundError:  # pragma: no cover - accelerate is a hard dep here
    init_empty_weights = None


# --------------------------------------------------------------------------
# Synthetic checkpoints (config.json only — nothing loads weights here)
# --------------------------------------------------------------------------
_TEXT_BODY = {
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 1000,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}

# `Ovis2Config` is a real transformers 5.x VL wrapper with exactly the
# issue-#12 shape: absent from AutoModelForCausalLM's mapping, while its
# `text_config` class (Qwen2Config) is present. Using a real wrapper keeps
# the test honest about transformers' actual behaviour; nothing about this
# test is Ovis2-specific and no Ovis2 checkpoint is touched.
_VL_WRAPPER_CONFIG = {
    "architectures": ["Ovis2ForConditionalGeneration"],
    "model_type": "ovis2",
    # Multimodal-level value that text-only staging must overwrite with the
    # text_config value (64). If it survives, the skeleton is wrong.
    "hidden_size": 4096,
    "visual_indicator_layers": 1,
    "vision_config": {
        "model_type": "siglip_vision_model",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 2,
    },
    "text_config": {"model_type": "qwen2", **_TEXT_BODY},
}

_PLAIN_TEXT_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    **_TEXT_BODY,
}


def _write_ckpt_dir(tmp_path, name, config_dict):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config_dict, indent=2))
    return str(d)


def _staged_config(tmp_path, monkeypatch, name, config_dict):
    """Run the real text-only staging step, then AutoConfig — exactly what
    `_build_streaming_context` does before it builds the skeleton."""
    from transformers import AutoConfig

    from prismaquant.sensitivity_probe import stage_text_only

    # Keep staging's scratch dirs inside pytest's tmp_path (never /tmp,
    # never the repo).
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "stage"))
    (tmp_path / "stage").mkdir(parents=True, exist_ok=True)
    staged = stage_text_only(_write_ckpt_dir(tmp_path, name, config_dict))
    return AutoConfig.from_pretrained(staged, trust_remote_code=True)


class _ThrowawayConfig(PreTrainedConfig):
    """A config class registered with nothing at all."""

    model_type = "pq_test_throwaway"


# --------------------------------------------------------------------------
# The reported failure
# --------------------------------------------------------------------------
def test_vl_wrapper_config_reproduces_issue_12(tmp_path, monkeypatch):
    """Pin the failure this change fixes, on the staged config itself."""
    config = _staged_config(tmp_path, monkeypatch, "vl", _VL_WRAPPER_CONFIG)
    assert type(config).__name__ == "Ovis2Config"
    assert not _auto_causal_lm_can_resolve(config)
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def test_vl_wrapper_text_only_resolves_to_text_sub_config(tmp_path, monkeypatch):
    config = _staged_config(tmp_path, monkeypatch, "vl", _VL_WRAPPER_CONFIG)

    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)

    # Text sub-config class, resolved through the auto class as usual.
    assert type(resolved_cfg).__name__ == "Qwen2Config"
    assert model_cls is AutoModelForCausalLM
    assert _auto_causal_lm_can_resolve(resolved_cfg)
    # model_type is the text class's own, not the wrapper's.
    assert resolved_cfg.model_type == "qwen2"


def test_vl_wrapper_text_only_takes_dims_from_staged_top_level(
        tmp_path, monkeypatch):
    """The values must come from the staged TOP LEVEL, not from the
    wrapper's nested sub-config object.

    `stage_text_only` pops `text_config` and lifts its keys up, so the
    sub-config object left on the wrapper is default-constructed. A
    refactor that reads `config.text_config` instead would silently build
    a 32-layer / 4096-hidden skeleton here.
    """
    config = _staged_config(tmp_path, monkeypatch, "vl", _VL_WRAPPER_CONFIG)
    # The trap, made explicit: the nested object is at class defaults.
    assert config.text_config.hidden_size != _TEXT_BODY["hidden_size"]
    assert (config.text_config.num_hidden_layers
            != _TEXT_BODY["num_hidden_layers"])

    resolved_cfg, _ = _skeleton_config_and_class(config, multimodal=False)

    assert resolved_cfg.hidden_size == _TEXT_BODY["hidden_size"]
    assert resolved_cfg.num_hidden_layers == _TEXT_BODY["num_hidden_layers"]
    assert resolved_cfg.num_attention_heads == _TEXT_BODY["num_attention_heads"]
    assert resolved_cfg.num_key_value_heads == _TEXT_BODY["num_key_value_heads"]
    assert resolved_cfg.intermediate_size == _TEXT_BODY["intermediate_size"]
    assert resolved_cfg.vocab_size == _TEXT_BODY["vocab_size"]
    # Derived fields are recomputed from the real depth, not inherited from
    # the default-constructed sub-config (which would be 32 long).
    assert len(resolved_cfg.layer_types) == _TEXT_BODY["num_hidden_layers"]
    # Nested sub-configs are not smuggled through.
    assert not hasattr(resolved_cfg, "vision_config")


@pytest.mark.skipif(init_empty_weights is None, reason="accelerate missing")
def test_vl_wrapper_text_only_builds_body_only_skeleton(tmp_path, monkeypatch):
    """The whole point: a skeleton gets built, and it has no visual tower."""
    config = _staged_config(tmp_path, monkeypatch, "vl", _VL_WRAPPER_CONFIG)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)

    with init_empty_weights():
        if model_cls is AutoModelForCausalLM:
            skeleton = AutoModelForCausalLM.from_config(
                resolved_cfg, trust_remote_code=True)
        else:  # pragma: no cover - not this config's path
            skeleton = model_cls._from_config(resolved_cfg)

    assert type(skeleton).__name__ == "Qwen2ForCausalLM"
    assert len(skeleton.model.layers) == _TEXT_BODY["num_hidden_layers"]
    assert (tuple(skeleton.model.layers[0].mlp.gate_proj.weight.shape)
            == (_TEXT_BODY["intermediate_size"], _TEXT_BODY["hidden_size"]))
    # Text-only means text-only: no tower anywhere in the tree.
    assert not hasattr(skeleton, "visual")
    assert not hasattr(skeleton.model, "visual")
    assert not any("visual" in n or "vision" in n
                   for n, _ in skeleton.named_modules())


def test_second_unrelated_wrapper_family_needs_no_code_change(tmp_path,
                                                              monkeypatch):
    """Genericity check: a different real wrapper family (AriaConfig, whose
    text class is AriaTextConfig rather than a shared Qwen2Config) goes
    through the same code with nothing family-specific added."""
    aria = {
        "architectures": ["AriaForConditionalGeneration"],
        "model_type": "aria",
        "hidden_size": 4096,
        "vision_config": {
            "model_type": "idefics3_vision",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
        },
        "text_config": {"model_type": "aria_text", "moe_num_experts": 4,
                        "moe_topk": 2, **_TEXT_BODY},
    }
    config = _staged_config(tmp_path, monkeypatch, "aria", aria)
    assert not _auto_causal_lm_can_resolve(config)

    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)

    assert type(resolved_cfg).__name__ == "AriaTextConfig"
    assert model_cls is AutoModelForCausalLM
    assert resolved_cfg.hidden_size == _TEXT_BODY["hidden_size"]
    assert resolved_cfg.num_hidden_layers == _TEXT_BODY["num_hidden_layers"]


# --------------------------------------------------------------------------
# Plain text models: byte-identical old path
# --------------------------------------------------------------------------
def test_plain_text_config_takes_the_old_path_untouched(tmp_path, monkeypatch):
    """Same config object, same class, and the fallback never runs."""
    config = _staged_config(tmp_path, monkeypatch, "plain", _PLAIN_TEXT_CONFIG)

    def _must_not_run(*a, **k):  # pragma: no cover - asserted not called
        raise AssertionError("wrapper fallback ran for a plain text config")

    monkeypatch.setattr("prismaquant.streaming_model._resolve_text_only_skeleton",
                        _must_not_run)

    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)

    assert resolved_cfg is config          # identical object, not a rebuild
    assert model_cls is AutoModelForCausalLM
    assert _auto_causal_lm_can_resolve(config)


@pytest.mark.skipif(init_empty_weights is None, reason="accelerate missing")
def test_plain_text_skeleton_unchanged(tmp_path, monkeypatch):
    config = _staged_config(tmp_path, monkeypatch, "plain", _PLAIN_TEXT_CONFIG)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(
            resolved_cfg, trust_remote_code=True)
        reference = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    assert type(skeleton) is type(reference)
    assert type(skeleton).__name__ == "Qwen2ForCausalLM"
    assert ([n for n, _ in skeleton.named_modules()]
            == [n for n, _ in reference.named_modules()])


def test_remote_code_config_is_left_to_the_auto_class(tmp_path):
    """A trust_remote_code model resolves via `auto_map`, not the static
    mapping — the predicate must not divert it into the fallback."""
    config = _ThrowawayConfig()
    config.auto_map = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    assert _auto_causal_lm_can_resolve(config)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)
    assert resolved_cfg is config
    assert model_cls is AutoModelForCausalLM


def test_multimodal_path_unchanged(tmp_path, monkeypatch):
    """`multimodal=True` still builds the declared architecture from the
    unmodified config (that is what materializes the visual tower)."""
    config = _staged_config(tmp_path, monkeypatch, "plain", _PLAIN_TEXT_CONFIG)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=True)
    assert resolved_cfg is config
    assert model_cls is _resolve_declared_model_cls(
        config, AutoModelForCausalLM)
    assert model_cls is transformers.Qwen2ForCausalLM


# --------------------------------------------------------------------------
# Declared-architecture fallback (wrapper with no usable text sub-config)
# --------------------------------------------------------------------------
def test_declared_architecture_fallback_when_no_text_sub_config():
    """A wrapper-ish config with no text sub-config but an importable
    declared architecture builds from that architecture's own config
    class, valued from the top level."""
    config = _ThrowawayConfig()
    config.architectures = ["Qwen2ForCausalLM"]
    for k, v in _TEXT_BODY.items():
        setattr(config, k, v)

    assert not _auto_causal_lm_can_resolve(config)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)

    assert model_cls is transformers.Qwen2ForCausalLM
    assert type(resolved_cfg) is transformers.Qwen2Config
    assert resolved_cfg.hidden_size == _TEXT_BODY["hidden_size"]
    assert resolved_cfg.num_hidden_layers == _TEXT_BODY["num_hidden_layers"]

    if init_empty_weights is not None:
        with init_empty_weights():
            skeleton = model_cls._from_config(resolved_cfg)
        assert len(skeleton.model.layers) == _TEXT_BODY["num_hidden_layers"]


# --------------------------------------------------------------------------
# Genuinely unresolvable: fail loudly, naming what was tried
# --------------------------------------------------------------------------
def test_unresolvable_config_raises_naming_every_attempt():
    config = _ThrowawayConfig()
    config.architectures = ["ThisArchDoesNotExistForCausalLM"]

    with pytest.raises(RuntimeError) as exc:
        _skeleton_config_and_class(config, multimodal=False)

    msg = str(exc.value)
    assert "cannot build a text-only skeleton" in msg
    assert "_ThrowawayConfig" in msg                  # what failed
    assert "pq_test_throwaway" in msg                 # its model_type
    assert "no text sub-config" in msg                # attempt 1
    assert "ThisArchDoesNotExistForCausalLM" in msg   # attempt 2
    assert "stage_text_only_promote_inner_model_type" in msg  # the fix seam


def test_unresolvable_config_with_no_architectures_still_names_attempts():
    config = _ThrowawayConfig()
    with pytest.raises(RuntimeError, match="cannot build a text-only skeleton"):
        _resolve_text_only_skeleton(config)


# --------------------------------------------------------------------------
# Detection is derived, not a name list
# --------------------------------------------------------------------------
def test_detection_follows_the_auto_mapping_not_a_name_list(monkeypatch):
    """Register a throwaway config→model pair and the predicate flips to
    True with no change here — the same way a future transformers release
    or a `prismaquant/vendored` override flips it."""
    config = _ThrowawayConfig()
    assert not _auto_causal_lm_can_resolve(config)

    mapping = AutoModelForCausalLM._model_mapping
    extra = mapping._extra_content
    monkeypatch.setitem(extra, _ThrowawayConfig, transformers.Qwen2ForCausalLM)

    assert _auto_causal_lm_can_resolve(config)
    resolved_cfg, model_cls = _skeleton_config_and_class(
        config, multimodal=False)
    assert resolved_cfg is config
    assert model_cls is AutoModelForCausalLM


def test_config_rebuilt_as_drops_sub_configs_and_model_type(tmp_path,
                                                            monkeypatch):
    config = _staged_config(tmp_path, monkeypatch, "vl", _VL_WRAPPER_CONFIG)
    rebuilt = _config_rebuilt_as(transformers.Qwen2Config, config)
    assert rebuilt.model_type == "qwen2"
    assert not hasattr(rebuilt, "vision_config")
    assert not hasattr(rebuilt, "text_config")
    # Non-text top-level keys ride along harmlessly but must not displace
    # the text schema.
    assert rebuilt.hidden_size == _TEXT_BODY["hidden_size"]
