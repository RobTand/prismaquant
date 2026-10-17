"""Actual resident source ownership and configured dense-shape admission.

Root submits these tiny CPU fixtures through admitted PrismaBuild. Observers
retain the actual factory model/context to inspect cleanup; they do not replace
source reads, decode, installation, memory admission or initialization witnesses.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import streaming_model as owner
from tests.test_resident_source_decode_audit import (
    _check_values, _cpu_guard, _load, _source,
)


def _rewrite_source(root, tensors):
    temporary = root / "model.replacement.safetensors"
    save_file(tensors, temporary)
    temporary.replace(root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))


def _real_parameters(model):
    return [(name, tuple(value.shape), str(value.device))
            for name, value in model.named_parameters(remove_duplicate=False)
            if not value.is_meta]


@pytest.mark.parametrize("missing_head", [False, True],
                         ids=["complete-head", "missing-untied-head"])
def test_factory_failure_releases_its_allocated_head(
    tmp_path, monkeypatch, missing_head,
):
    root, expected = _source(tmp_path)
    if missing_head:
        tensors = load_file(root / "model.safetensors")
        assert "lm_head.weight" in tensors
        del tensors["lm_head.weight"]
        _rewrite_source(root, tensors)
        del tensors
    models = []
    materialize = owner._materialize

    def observe_materialize(model, *args, **kwargs):
        result = materialize(model, *args, **kwargs)
        models.append(model)
        return result

    monkeypatch.setattr(owner, "_materialize", observe_materialize)
    guard = _cpu_guard(tmp_path, monkeypatch)
    if missing_head:
        with pytest.raises(RuntimeError, match="missing its LM head"):
            _load(root, tmp_path, guard)
        for model in models:
            assert not _real_parameters(model), (
                "Factory raised before returning a context and left actual source "
                f"parameters installed: {_real_parameters(model)}")
    else:
        model = _load(root, tmp_path, guard)
        assert models == [model]
        _check_values(model, expected, torch.bfloat16, False)


@pytest.mark.parametrize("missing_projection", [False, True],
                         ids=["complete-tied-model", "tied-head-body-failure"])
def test_body_failure_releases_every_tied_head_alias(
    tmp_path, monkeypatch, missing_projection,
):
    root, expected = _source(tmp_path, tied=True, omit_projection=missing_projection)
    contexts = []
    models = []
    build = owner._build_streaming_context
    materialize = owner._materialize

    def observe_materialize(model, *args, **kwargs):
        result = materialize(model, *args, **kwargs)
        models.append(model)
        return result

    def observe_context(*args, **kwargs):
        context = build(*args, **kwargs)
        assert (context.model.get_output_embeddings().weight
                is context.model.get_input_embeddings().weight)
        contexts.append(context)
        return context

    monkeypatch.setattr(owner, "_materialize", observe_materialize)
    monkeypatch.setattr(owner, "_build_streaming_context", observe_context)
    guard = _cpu_guard(tmp_path, monkeypatch)
    if missing_projection:
        with pytest.raises(ValueError, match="missing expert"):
            _load(root, tmp_path, guard)
        assert all(not thread.is_alive()
                   for context in contexts for thread in context.prefetch_pool._threads)
        for model in models:
            assert not _real_parameters(model), (
                "Failed resident delivery retained an installed parameter alias: "
                f"{_real_parameters(model)}")
    else:
        model = _load(root, tmp_path, guard)
        _check_values(model, expected, torch.bfloat16, False)
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


@pytest.mark.parametrize("oversized_dense", [False, True],
                         ids=["configured-dense-shape", "oversized-dense-source"])
@pytest.mark.parametrize("native", [False, True], ids=["bf16", "native-fp8"])
def test_dense_source_shape_must_match_the_configured_slot(
    tmp_path, monkeypatch, oversized_dense, native,
):
    root, expected = _source(tmp_path, native=native)
    key = next(name for name in sorted(expected) if name.endswith(".self_attn.o_proj.weight"))
    configured = tuple(expected[key].shape)
    if oversized_dense:
        tensors = load_file(root / "model.safetensors")
        assert tuple(tensors[key].shape) == configured
        actual_shape = (configured[0] * 2, configured[1])
        tensors[key] = torch.full(actual_shape, 1.875).to(
            torch.float8_e4m3fn if native else torch.bfloat16)
        if native:
            config = json.loads((root / "config.json").read_text())
            block_r, block_c = config["quantization_config"]["weight_block_size"]
            grid = ((actual_shape[0] + block_r - 1) // block_r,
                    (actual_shape[1] + block_c - 1) // block_c)
            tensors[key + "_scale_inv"] = (
                torch.arange(grid[0] * grid[1], dtype=torch.float32).reshape(grid) * 0.51 + 1.003)
        _rewrite_source(root, tensors)
        del tensors
    guard = _cpu_guard(tmp_path, monkeypatch)
    payload_calls = []
    materialize = owner._materialize

    def observe_payload(*args, **kwargs):
        payload_calls.append(True)
        return materialize(*args, **kwargs)

    monkeypatch.setattr(owner, "_materialize", observe_payload)
    if oversized_dense:
        try:
            model = _load(root, tmp_path, guard)
        except (ValueError, RuntimeError) as error:
            assert key in str(error) and "shape" in str(error).lower(), (
                f"A different failure is not dense-slot shape admission: {error}")
        else:
            actual = dict(model.named_parameters())[key]
            pytest.fail(
                "Resident delivery accepted a source/config shape mismatch: "
                f"{key}, configured={configured}, installed={tuple(actual.shape)}, "
                f"witness={model._prismaquant_resident_source_initialization}")
        assert not payload_calls
        assert guard.last is None
    else:
        model = _load(root, tmp_path, guard)
        _check_values(model, expected, torch.bfloat16, False)
        assert payload_calls == [True]


@pytest.mark.parametrize("missing", [False, True], ids=["persistent-head-buffer", "missing-head-buffer"])
def test_factory_owns_persistent_head_buffer_failure(tmp_path, monkeypatch, missing):
    root, expected = _source(tmp_path)
    value = torch.arange(4, dtype=torch.float32) + 0.137
    if not missing:
        tensors = load_file(root / "model.safetensors")
        tensors["model.norm.audit_state"] = value
        _rewrite_source(root, tensors)
        del tensors
    models = []
    build = owner.build_streaming_skeleton

    def with_persistent_head_buffer(*args, **kwargs):
        model = build(*args, **kwargs)
        model.model.norm.register_buffer(
            "audit_state", torch.empty(4, dtype=torch.float32, device="meta"), persistent=True)
        models.append(model)
        return model

    monkeypatch.setattr(owner, "build_streaming_skeleton", with_persistent_head_buffer)
    guard = _cpu_guard(tmp_path, monkeypatch)
    if missing:
        with pytest.raises(RuntimeError, match="head buffers left on meta"):
            _load(root, tmp_path, guard)
        assert models
        assert not _real_parameters(models[0])
        assert models[0].model.norm.audit_state.is_meta
    else:
        model = _load(root, tmp_path, guard)
        _check_values(model, expected, torch.bfloat16, False)
        assert model.model.norm.audit_state.dtype == torch.float32
        assert torch.equal(model.model.norm.audit_state, value)


@pytest.mark.parametrize("malformed", [False, True], ids=["declared-concat", "wrong-concat-shape"])
def test_resident_header_geometry_uses_the_declared_concat_bridge(tmp_path, malformed):
    from functools import partial

    from prismaquant.layer_streaming import _merge_concat_sources
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.conv1d = torch.nn.Conv1d(9, 9, 4, groups=9, bias=False, device="meta")
    model.model.language_model.layers = torch.nn.ModuleList([layer])
    prefix = "model.language_model.layers."
    sources = {f"{prefix}0.self_attn.{leaf}_conv1d.weight": torch.ones(
        4 if malformed and leaf == "v" else 3, 1, 4, dtype=torch.bfloat16)
        for leaf in "qkv"}
    path = tmp_path / "split-conv.safetensors"
    save_file(sources, path)
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    merge = partial(_merge_concat_sources, groups=Glm5NextProfile().concat_merge_groups(),
                    live_param_shape=shapes.get)

    def demand():
        return owner._resident_source_demand(
            model, dtype=torch.bfloat16, heads=(), layers_prefix=prefix, num_layers=1,
            weight_shard=dict.fromkeys(sources, str(path)),
            weight_ckpt={name: name for name in sources}, fp8_scale_inv_map={},
            pack_experts=None, merge_concat=merge)

    if malformed:
        with pytest.raises(ValueError, match="concat merge: assembled.*shape"):
            demand()
    else:
        result = demand()
        assert result.final_bytes["0"] == 9 * 4 * 2
        assert result.device_transient_bytes > 0
    assert all(value.is_meta for value in model.parameters())


def test_factory_cleanup_retains_the_primary_head_error(tmp_path, monkeypatch):
    root, _ = _source(tmp_path)
    tensors = load_file(root / "model.safetensors")
    del tensors["lm_head.weight"]
    _rewrite_source(root, tensors)
    del tensors
    unload = owner._unload

    def release_then_report(*args, **kwargs):
        unload(*args, **kwargs)
        raise RuntimeError("cleanup diagnostic")

    monkeypatch.setattr(owner, "_unload", release_then_report)
    with pytest.raises(RuntimeError, match="missing its LM head") as error:
        _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    assert any("cleanup diagnostic" in note for note in error.value.__notes__)


def test_factory_owns_its_pool_until_context_return(tmp_path, monkeypatch):
    root, _ = _source(tmp_path)
    contexts = []
    context_type = owner.StreamingContext

    def construct_then_fail(*args, **kwargs):
        contexts.append(context_type(*args, **kwargs))
        raise RuntimeError("context delivery failed")

    monkeypatch.setattr(owner, "StreamingContext", construct_then_fail)
    with pytest.raises(RuntimeError, match="context delivery failed"):
        _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    assert contexts
    assert contexts[0].prefetch_pool._shutdown
    assert all(not thread.is_alive() for thread in contexts[0].prefetch_pool._threads)
    assert not _real_parameters(contexts[0].model)
