"""Native decode identity cannot adopt old guarded or explicitly trusted data.

Legacy shards below are real public-cache output from the existing TinyChain
fixture. They are deliberately unattributed to the native checkpoint. The
tests exercise admission, never claim those old shards have native provenance.
"""
from __future__ import annotations

from copy import deepcopy
import json
import pickle

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from prismaquant import streamed_model_identity as identity
from tests.test_resident_source_decode_audit import _cpu_guard, _load, _source
from tests.test_production_cache_source_model_binding import _fill as _legacy_fill
from tests.test_production_weight_cache import _TinyChain


V1 = "prismaquant.source_model_binding.v1"
V2 = "prismaquant.source_model_binding.v2"
DECODE = "prismaquant.native_fp8_source_decode.v1"


def _old_binding(current):
    return {"schema": V1, "checkpoint": deepcopy(current["checkpoint"]),
            "effective_dtype": current["effective_dtype"]}


def _empty_fill(model, directory, binding):
    return pwc.fill_production_weight_cache(
        model, torch.tensor([[1, 2]], dtype=torch.long), [], formats=["FP8_E4M3"],
        levers={"gptq": False, "scale_sweep": False}, cache_dir=directory,
        source_binding=binding, max_act_rows=8, progress=False)


@pytest.mark.parametrize("native", [True, False], ids=["native-v2", "bf16-v1-control"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16-load", "fp32-load"])
def test_source_binding_names_actual_native_decode_contract(tmp_path, native, dtype):
    root, _expected = _source(tmp_path, native=native)
    binding = identity.build_source_model_binding(root, dtype=dtype)
    expected = {"schema": V2 if native else V1, "checkpoint": binding["checkpoint"],
                "effective_dtype": str(dtype)}
    if native:
        expected["source_decode"] = DECODE
    assert binding == expected
    assert identity.validate_source_model_binding(binding) == expected


def test_native_independent_pickle_refuses_legacy_decoder_binding(tmp_path):
    root, _expected = _source(tmp_path)
    current = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    old = pwc.ProductionWeightCache(weights={}, levers={}, metadata={"source_binding": _old_binding(current)})
    restored = pickle.loads(pickle.dumps(old))
    with pytest.raises(ValueError, match="(?i)(decode|binding|identity).*(rebuild|fresh)"):
        restored.require_source_binding(current, where="native attachment")


@pytest.mark.parametrize("native", [True, False], ids=["native-refuses", "bf16-trust-control"])
def test_matching_pickle_does_not_erase_pre_guard_history(tmp_path, native):
    root, _expected = _source(tmp_path, native=native)
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    cache = pwc.ProductionWeightCache(weights={}, levers={}, metadata={
        "source_binding": binding,
        pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY: pwc.build_pre_guard_admission(["old.pt"]),
    })
    restored = pickle.loads(pickle.dumps(cache))
    if native:
        with pytest.raises(ValueError, match="(?i)(pre.guard|trust|decode).*(rebuild|fresh)"):
            restored.require_source_binding(binding, where="pre-guard attachment")
    else:
        restored.require_source_binding(binding, where="pre-guard BF16 attachment")
    assert restored.metadata == cache.metadata


@pytest.mark.parametrize("native", [True, False], ids=["native-refuses", "bf16-trust-control"])
@pytest.mark.parametrize("sidecar", ["missing", "append-only"])
def test_directory_cannot_reattest_unbound_old_shards(tmp_path, monkeypatch, native, sidecar):
    root, _expected = _source(tmp_path, native=native)
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    directory = tmp_path / "cache"
    legacy = _legacy_fill(_TinyChain().eval(), directory, None)
    assert legacy.weights
    path = directory / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    if sidecar == "missing":
        path.unlink()
        original = None
    else:
        section = pwc.build_mtp_append_identity(
            max_act_rows=8, activation_source_hash="0" * 64, activation_rows={"mtp.fc": 2},
            source_prefix="mtp.", source_tensor_count=1, profile_name="test")
        path.write_text(json.dumps({pwc.MTP_APPEND_SIDECAR_KEY: section}))
        original = path.read_bytes()
    shard_bytes = {item.name: item.read_bytes() for item in directory.glob("*.pt")}
    assert shard_bytes
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    if native:
        with pytest.raises(ValueError, match="(?i)(pre.guard|trust|decode|binding).*(rebuild|fresh)"):
            _empty_fill(model, directory, binding)
        assert (path.read_bytes() if path.exists() else None) == original
    else:
        admitted = _empty_fill(model, directory, binding)
        history = admitted.metadata[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY]
        assert history["trusted_shards"]
        admitted.require_source_binding(binding, where="named BF16 trust control")
    assert {item.name: item.read_bytes() for item in directory.glob("*.pt")} == shard_bytes


@pytest.mark.parametrize("native", [True, False], ids=["native-v2", "bf16-v1-control"])
def test_fresh_bound_directory_and_pickle_keep_matching_binding(tmp_path, monkeypatch, native):
    root, _expected = _source(tmp_path, native=native)
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    cache = _empty_fill(model, tmp_path / "cache", binding)
    restored = pickle.loads(pickle.dumps(cache))
    restored.require_source_binding(binding, where="fresh decode attachment")
    assert restored.metadata["source_binding"] == binding
    assert pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY not in restored.metadata


def test_bf16_payload_with_stale_fp8_config_retains_v1_binding(tmp_path):
    from safetensors import safe_open

    root, _expected = _source(tmp_path, native=False)
    config_path = root / "config.json"
    document = json.loads(config_path.read_text())
    document["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [48, 48]}
    config_path.write_text(json.dumps(document))
    with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert all(str(handle.get_slice(name).get_dtype()) == "BF16" for name in handle.keys())
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    assert binding == {"schema": V1, "checkpoint": binding["checkpoint"],
                       "effective_dtype": "torch.bfloat16"}
    cache = pwc.ProductionWeightCache(weights={}, levers={}, metadata={"source_binding": binding})
    cache.require_source_binding(binding, where="actual BF16 payload with stale native config")


@pytest.mark.parametrize("defect", ["missing-grid", "malformed-grid", "missing-scale", "per-tensor"])
def test_actual_native_headers_cannot_fall_back_to_ordinary_binding(tmp_path, defect):
    from safetensors.torch import load_file, save_file

    root, _expected = _source(tmp_path)
    config_path = root / "config.json"
    document = json.loads(config_path.read_text())
    if defect in {"missing-grid", "per-tensor"}:
        document["quantization_config"].pop("weight_block_size")
    elif defect == "malformed-grid":
        document["quantization_config"]["weight_block_size"] = [0, 48]
    config_path.write_text(json.dumps(document))
    if defect in {"missing-scale", "per-tensor"}:
        tensors = {name: value.clone() for name, value in
                   load_file(str(root / "model.safetensors")).items()}
        scale_keys = [name for name in tensors if name.endswith(".weight_scale_inv")]
        assert scale_keys
        if defect == "missing-scale":
            key = next(name for name in scale_keys if ".self_attn.o_proj." in name)
            del tensors[key]
        else:
            for name in scale_keys:
                tensors[name] = torch.tensor(1.0, dtype=torch.float32)
        fresh = root / "replacement.safetensors"
        save_file(tensors, str(fresh))
        fresh.replace(root / "model.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(fp8|float8|scale|block|grid)"):
        identity.build_source_model_binding(root, dtype=torch.bfloat16)


@pytest.mark.parametrize("damage", ["missing-contract", "unknown-contract", "v1-with-contract"])
def test_binding_parser_requires_versioned_native_contract(tmp_path, damage):
    root, _expected = _source(tmp_path)
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    assert binding["schema"] == V2
    if damage == "missing-contract":
        del binding["source_decode"]
    elif damage == "unknown-contract":
        binding["source_decode"] = "unknown-native-decoder"
    else:
        binding["schema"] = V1
    with pytest.raises(ValueError, match="binding"):
        identity.validate_source_model_binding(binding, where="native contract admission")


@pytest.mark.parametrize("auxiliary", ["mtp.fc.weight", "model.visual.proj.weight"])
def test_decoder_binding_does_not_attest_excluded_auxiliary_namespace(tmp_path, auxiliary):
    from safetensors.torch import load_file, save_file

    root, _expected = _source(tmp_path, native=False)
    tensors = {name: value.clone() for name, value in
               load_file(str(root / "model.safetensors")).items()}
    tensors[auxiliary] = torch.ones((2, 2)).to(torch.float8_e4m3fn)
    fresh = root / "replacement.safetensors"
    save_file(tensors, str(fresh))
    fresh.replace(root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    binding = identity.build_source_model_binding(root, dtype=torch.bfloat16)
    assert binding["schema"] == V1
    assert "source_decode" not in binding
