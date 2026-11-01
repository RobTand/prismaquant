"""Research MoE resume binds completed staging to the current requested plan.

Tiny real safetensors and probe receipts exercise the exported planner. These
CPU policy controls provide no MoE fold-fidelity or serving qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
from transformers import AutoConfig

from prismaquant import prismasnap_checkpoint as common
from prismaquant import prismasnap_moe_checkpoint as moe
from prismaquant.streamed_model_identity import build_streamed_model_identity
from test_prismasnap_moe import _config, _packed_qwen_graph_inputs


def _write_inputs(root, *, seed):
    root.mkdir()
    source = root / "source"
    source.mkdir()
    profile, one_stats, one_rows, hidden, experts = _packed_qwen_graph_inputs()
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text", "hidden_size": hidden,
            "intermediate_size": 32, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "vocab_size": 32, "num_experts": experts,
            "num_experts_per_tok": 1,
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("{}\n")
    shapes, stats = {}, {}
    for layer in range(2):
        shapes.update({name.replace(".layers.0.", f".layers.{layer}."): row["shape"]
                       for name, row in one_rows.items()})
        stats.update({name.replace(".layers.0.", f".layers.{layer}."): deepcopy(row)
                      for name, row in one_stats.items()})
    for recipe, shape in (("model.embed_tokens.weight", [32, hidden]),
                          ("model.norm.weight", [hidden]), ("lm_head.weight", [32, hidden])):
        shapes[profile.source_tensor_name(recipe)] = shape
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        name: (torch.zeros(shape) if "layernorm.weight" in name or name.endswith(".norm.weight")
               else torch.randn(shape, generator=generator) * 0.125).to(torch.bfloat16)
        for name, shape in shapes.items()
    }
    shard = "model.safetensors"
    save_file(tensors, str(source / shard), metadata={"format": "pt"})
    weight_map = {name: shard for name in sorted(tensors)}
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    identity_config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    identity_config._commit_hash = "synthetic-revision"
    runner = SimpleNamespace(model=SimpleNamespace(config=identity_config), context=SimpleNamespace(
        weight_shard={name: source / shard for name in weight_map},
        weight_ckpt={name: name for name in weight_map}))
    identity = build_streamed_model_identity(runner, str(source.resolve()))
    identity_path = root / "identity.json"
    identity_path.write_text(json.dumps(identity))
    probe = {"stats": stats, "meta": {
        "model": str(source.resolve()), "calib_hash": "synthetic-calibration",
        "dataset": "synthetic", "nsamples": 2, "seqlen": 512,
        "dtype": "bf16", "device_map": "streaming-layerwise",
        "execution_device": "cuda:0", "calibration_modality": "text-only",
    }}
    probe_path = root / "probe.pkl"
    probe_path.write_bytes(pickle.dumps(probe, protocol=pickle.HIGHEST_PROTOCOL))
    return {"source": source, "identity": identity_path, "probe": probe_path}


def _plan(inputs, output, *, layers=(0,), resume=False, search_config=None,
          tensor_metadata_manifest_path=None):
    return moe.plan_moe_checkpoint(inputs["source"], inputs["probe"], inputs["identity"], output,
                                   device="cpu", layers=layers,
                                   search_config=_config() if search_config is None else search_config,
                                   tensor_metadata_manifest_path=tensor_metadata_manifest_path,
                                   resume=resume)


@pytest.fixture(scope="module")
def genuine_plan(tmp_path_factory):
    root = tmp_path_factory.mktemp("moe-resume")
    inputs = _write_inputs(root / "original", seed=17)
    plan_dir = root / "plan"
    plan = _plan(inputs, plan_dir)
    loaded, _scales = common.load_plan(plan_dir)
    assert loaded == plan and plan["schema"] == moe.MOE_PLAN_SCHEMA
    return root, inputs, plan_dir, plan


def _changed_request(tmp_path, original, change):
    current = dict(original)
    layers = (0,)
    if change == "source":
        current = _write_inputs(tmp_path / "other", seed=23)
    elif change == "probe":
        payload = pickle.loads(original["probe"].read_bytes())
        payload["meta"]["calib_hash"] = "another-synthetic-calibration"
        current["probe"] = tmp_path / "other-probe.pkl"
        current["probe"].write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        layers = (1,)
    return current, layers


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("change", ["source", "probe", "layers"])
@pytest.mark.parametrize("location", ["output", "staging"])
def test_moe_resume_refuses_a_valid_plan_for_other_requested_inputs(
    tmp_path, genuine_plan, change, location,
):
    _root, original, plan_dir, _plan_record = genuine_plan
    output = tmp_path / "requested-plan"
    retained = output if location == "output" else output.with_name(output.name + ".prismasnap-plan-incomplete")
    shutil.copytree(plan_dir, retained)
    before = _snapshot(retained)
    current, layers = _changed_request(tmp_path, original, change)

    with pytest.raises(RuntimeError, match="different.*inputs|different requested"):
        _plan(current, output, layers=layers, resume=True)

    assert _snapshot(retained) == before
    if location == "staging":
        assert not output.exists()


def test_moe_resume_publishes_matching_completed_staging(tmp_path, genuine_plan):
    _root, inputs, plan_dir, plan_record = genuine_plan
    output = tmp_path / "requested-plan"
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    shutil.copytree(plan_dir, staging)

    assert _plan(inputs, output, resume=True) == plan_record
    assert common.load_plan(output)[0] == plan_record
    assert not staging.exists()


@pytest.mark.parametrize("change", ["search", "tensor_metadata_manifest"])
@pytest.mark.parametrize("location", ["output", "staging"])
def test_moe_resume_retains_current_search_and_metadata_request(
    tmp_path, genuine_plan, change, location,
):
    _root, inputs, plan_dir, _plan_record = genuine_plan
    output = tmp_path / "requested-plan"
    retained = output if location == "output" else output.with_name(output.name + ".prismasnap-plan-incomplete")
    shutil.copytree(plan_dir, retained)
    before = _snapshot(retained)
    request = {}
    if change == "search":
        request["search_config"] = replace(_config(), max_rounds=2)
    else:
        manifest = tmp_path / "tensor-metadata.json"
        common.scan_tensor_metadata_manifest(inputs["source"], inputs["identity"], manifest)
        request["tensor_metadata_manifest_path"] = manifest

    with pytest.raises(RuntimeError, match="different requested inputs"):
        _plan(inputs, output, resume=True, **request)

    assert _snapshot(retained) == before
    if location == "staging":
        assert not output.exists()


def test_moe_resume_returns_matching_existing_output(tmp_path, genuine_plan):
    _root, inputs, plan_dir, plan_record = genuine_plan
    output = tmp_path / "requested-plan"
    shutil.copytree(plan_dir, output)
    before = _snapshot(output)

    assert _plan(inputs, output, resume=True) == plan_record
    assert _snapshot(output) == before


def test_moe_resume_keeps_matching_explicit_metadata_binding(tmp_path, genuine_plan):
    _root, inputs, _plan_dir, _plan_record = genuine_plan
    manifest = tmp_path / "tensor-metadata.json"
    common.scan_tensor_metadata_manifest(inputs["source"], inputs["identity"], manifest)
    output = tmp_path / "requested-plan"
    plan_record = _plan(inputs, output, tensor_metadata_manifest_path=manifest)
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    output.rename(staging)

    assert _plan(inputs, output, resume=True, tensor_metadata_manifest_path=manifest) == plan_record
    assert common.load_plan(output)[0] == plan_record
    assert plan_record["tensor_metadata_binding"]["mode"] == "external_manifest"
    assert not staging.exists()
