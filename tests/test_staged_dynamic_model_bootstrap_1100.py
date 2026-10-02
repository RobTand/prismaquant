"""A staged config cannot activate an undeclared model Python module."""
from __future__ import annotations

from pathlib import Path

import pytest

from prismaquant import streaming_model
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from test_autoconfig_staged_1922 import _config
from test_streamed_metadata_staged_reads import (  # noqa: F401
    checkpoint, _activate, _deny_pool_opens, _forget_state,
)

pytestmark = pytest.mark.own_process


@pytest.mark.parametrize("multimodal", [False, True])
def test_active_auto_model_route_refuses_before_dynamic_pool_open(
        checkpoint, tmp_path, monkeypatch, multimodal):
    from transformers.models.auto import auto_factory

    root, _cfg, manifest = _config(checkpoint, {
        "architectures": [],
        "auto_map": {"AutoModelForCausalLM": "modeling_fixture.RemoteModel"},
    })
    module_path = root / "modeling_fixture.py"
    module_path.write_text("# deliberately outside the declared metadata readset\n")
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root, names=(
        "config.json", "model.safetensors.index.json", "modeling_fixture.py"))
    config = streaming_model.load_streaming_auto_config(str(root), str(root))
    calls = []

    def dynamic(class_ref, source_root, *args, **kwargs):
        calls.append((class_ref, source_root))
        # The real HF dynamic branch acquires this local module. The guard
        # records that boundary without executing or caching arbitrary code.
        Path(source_root, "modeling_fixture.py").read_bytes()
        raise AssertionError("dynamic model code was reached")

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", dynamic)
    with pytest.raises(RuntimeError, match="undeclared dynamic AutoModelForCausalLM"):
        streaming_model.build_streaming_skeleton(config, multimodal=multimodal)
    assert calls == []


def test_inactive_dynamic_model_route_retains_legacy_dispatch(tmp_path, monkeypatch):
    from transformers import LlamaConfig
    from transformers.models.auto import auto_factory

    config = LlamaConfig(hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2)
    config.auto_map = {"AutoModelForCausalLM": "modeling_fixture.RemoteModel"}
    config.name_or_path = str(tmp_path)
    calls = []

    def dynamic(class_ref, source_root, *args, **kwargs):
        calls.append((class_ref, source_root))
        raise RuntimeError("legacy dynamic boundary")

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", dynamic)
    with pytest.raises(RuntimeError, match="legacy dynamic boundary"):
        streaming_model.build_streaming_skeleton(config, multimodal=False)
    assert calls == [(config.auto_map["AutoModelForCausalLM"], str(tmp_path))]


@pytest.mark.parametrize("auto_map", [None, {}, {"AutoModel": "unused.Module"}])
def test_active_native_model_route_retains_stock_meta_geometry(auto_map):
    from transformers import LlamaConfig

    config = LlamaConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32)
    if auto_map is not None:
        config.auto_map = auto_map
    baseline = streaming_model.build_streaming_skeleton(config, multimodal=False)
    with staged_tier_policy_test_context("ram,ssd"):
        actual = streaming_model.build_streaming_skeleton(config, multimodal=False)
    assert type(actual) is type(baseline)
    assert {name: (tuple(value.shape), value.dtype, value.device.type)
            for name, value in actual.named_parameters()} == {
        name: (tuple(value.shape), value.dtype, value.device.type)
        for name, value in baseline.named_parameters()}


def test_active_explicit_stock_architecture_does_not_use_dynamic_map():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32, architectures=["LlamaForCausalLM"])
    config.auto_map = {"AutoModelForCausalLM": "unused.RemoteModel"}
    with staged_tier_policy_test_context("ram,ssd"):
        actual = streaming_model.build_streaming_skeleton(config, multimodal=True)
    assert type(actual) is LlamaForCausalLM
    assert all(value.device.type == "meta" for value in actual.parameters())
