"""A failed serving declaration cannot become an empty coupling roster."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from prismaquant.allocator_solver import promote_serving_units
import prismaquant.export_native_compressed as exporter
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles import vllm_registry
from prismaquant.production_weight_cache import _fused_sibling_leaf_mapping_from_profile


class DeclarationFailed(RuntimeError):
    pass


class FailedProfile(DefaultProfile):
    def fused_sibling_group(self, name):
        raise DeclarationFailed("fused declaration unavailable")

    def fused_sibling_leaf_mapping(self):
        raise DeclarationFailed("fused declaration unavailable")


@pytest.mark.parametrize("consumer", [
    lambda profile: exporter._compute_nvfp4_joint_global(
        torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False)), {"0": "NVFP4"}, profile=profile),
    lambda profile: exporter._unify_input_global_scales_across_fused_siblings(
        {"model.layers.0.self_attn.q_proj": 1.0}, profile=profile),
    lambda profile: _fused_sibling_leaf_mapping_from_profile(profile),
])
def test_failed_fused_declaration_propagates_to_every_render_consumer(consumer):
    with pytest.raises(DeclarationFailed, match="unavailable"):
        consumer(FailedProfile())


def test_failed_packed_declaration_cannot_skip_allocator_coupling():
    class Profile(DefaultProfile):
        def packed_expert_format_group(self, name):
            raise DeclarationFailed("packed declaration unavailable")
    with pytest.raises(DeclarationFailed, match="packed"):
        promote_serving_units({"a": "NVFP4", "b": "BF16"},
                              {"NVFP4": 4, "BF16": 16}, profile=Profile())


def test_registry_load_failure_cannot_be_cached_as_absent(monkeypatch):
    def fail(*args):
        raise DeclarationFailed("registered model cannot load")
    model_registry = SimpleNamespace(
        get_supported_archs=lambda: ["Qwen3ForCausalLM"], resolve_model_cls=fail)
    module = ModuleType("vllm.model_executor.models")
    module.registry = SimpleNamespace(ModelRegistry=model_registry)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", module)
    profile = DefaultProfile()
    profile._architectures = ["Qwen3ForCausalLM"]
    for _ in range(2):
        with pytest.raises(DeclarationFailed, match="cannot load"):
            profile.fused_sibling_leaf_mapping()


def test_unregistered_architecture_remains_an_explicit_absence(monkeypatch):
    module = ModuleType("vllm.model_executor.models")
    module.registry = SimpleNamespace(ModelRegistry=SimpleNamespace(
        get_supported_archs=lambda: ["Registered"],
        resolve_model_cls=lambda names: pytest.fail("unknown architecture was loaded")))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", module)
    assert vllm_registry.vllm_class_for_architecture("Unregistered") is None


def test_registry_metadata_loader_does_not_require_runtime_model_config(monkeypatch):
    class QwenMetadata:
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    entry = SimpleNamespace(load_model_cls=lambda: QwenMetadata)
    def runtime_resolution(architectures, model_config):
        pytest.fail("metadata discovery initialized a runtime model config")
    module = ModuleType("vllm.model_executor.models")
    module.registry = SimpleNamespace(ModelRegistry=SimpleNamespace(
        get_supported_archs=lambda: ["Qwen3ForCausalLM"],
        models={"Qwen3ForCausalLM": entry},
        resolve_model_cls=runtime_resolution))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", module)
    resolved = vllm_registry.vllm_class_for_architecture("Qwen3ForCausalLM")
    assert vllm_registry.packed_modules_mapping_from_class(resolved) == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"]}


def test_registered_metadata_loader_failure_propagates(monkeypatch):
    def fail():
        raise DeclarationFailed("registered metadata cannot load")
    module = ModuleType("vllm.model_executor.models")
    module.registry = SimpleNamespace(ModelRegistry=SimpleNamespace(
        get_supported_archs=lambda: ["Qwen3ForCausalLM"],
        models={"Qwen3ForCausalLM": SimpleNamespace(load_model_cls=fail)}))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", module)
    with pytest.raises(DeclarationFailed, match="metadata cannot load"):
        vllm_registry.vllm_class_for_architecture("Qwen3ForCausalLM")
