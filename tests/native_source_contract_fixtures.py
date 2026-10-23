"""Real GLM source contracts with synthetic prices and calibration labels.

Checkpoint writes, header census, streamed delivery and identities are real.
The price factories do not run AURA or establish quality or serving behavior.
"""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from prismaquant import glm53_stock_reprice as stock
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.glm53_stock_harvest import (
    HARVEST_SCHEMA, build_arm_identity, build_dense_plan,
)
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.streamed_model_identity import (
    build_source_model_binding, build_streamed_model_identity,
    source_model_decode_contract, validate_streamed_model_identity,
)
from test_glm5_next_streamed_forward_parity import _tiny_config, _build_model
from tests.test_glm_campaign_streaming import write_original_layout_checkpoint

CALIB_HASH = "feedbeef" * 4


def _probe_unit(out_features, in_features, *, experts=None, packed=False):
    return {
        "n_params": (experts or 1) * out_features * in_features,
        "in_features": in_features,
        "out_features": out_features,
        "num_experts": experts,
        "is_packed": packed,
    }


def _anchor_row(value):
    return {
        "predicted_dloss": value,
        "dw_source": "production_render",
        "production_anchor_measured": True,
        "cost_source": "aura",
        "output_mse_measured": False,
    }


def _harvest(probe, checkpoint, **overrides):
    planned = build_dense_plan(probe, checkpoint)
    arm = build_arm_identity(
        probe_calib_hash=CALIB_HASH,
        observed_calib_hash=CALIB_HASH,
        dataset="/data/diverse-v1.jsonl",
        n_calib_samples=8,
        calib_seqlen=512,
        calib_seed=42,
        n_probes=32,
        max_act_rows=256,
    )
    costs = {
        qname: {fmt: _anchor_row(1e-4 * (idx + 1)) for fmt in fmts}
        for idx, (qname, fmts) in enumerate(sorted(planned["plan"].items()))
    }
    wrapper = {
        "schema": HARVEST_SCHEMA,
        "plan_scope": planned["plan_scope"],
        "unit_filter": None,
        "max_units": 0,
        "plan": {q: list(f) for q, f in planned["plan"].items()},
        "packed_expert_units_excluded": planned["packed_expert_units_excluded"],
        "pinned_units": planned["pinned_units"],
        "ladder_refusals": planned["ladder_refusals"],
        "arm_identity": arm,
        "checkpoint_dir": "/durable/ckpt",
        "aura_payload": {
            "schema": "aura",
            "n_probes": 32,
            "formats": ["NVFP4"],
            "costs": costs,
            "provenance": {
                "git_commit": "test",
                "calib_hash": CALIB_HASH,
                "dw_rtn_fallback_rows": 0,
                "dw_production_anchor_rows": len(costs),
                "streaming": True,
            },
        },
    }
    wrapper.update(overrides)
    return wrapper


def _expert_payload(probe):
    packed = [
        name for name, row in probe["units"].items() if row["is_packed"]
    ]
    return {
        "schema": "prismaquant.expert_empirical_cost.v1",
        "costs": {
            name: {
                "NVFP4": {"predicted_dloss": 3e-2, "cost_source": "expert_kl"},
                "FP8_E4M3": {
                    "predicted_dloss": 2e-2, "cost_source": "expert_kl",
                },
            }
            for name in packed
        },
        "provenance": {"eval_driver": "forked-stream/1", "calib_batch": 16},
    }


def _source(tmp_path, *, native=True):
    config = _tiny_config()
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 256
    config.text_config.moe_intermediate_size = 128
    config.vision_config.out_hidden_size = 128
    config = type(config).from_dict(config.to_dict())
    model = _build_model(config).to(torch.bfloat16)
    source = tmp_path / "glm-source"
    write_original_layout_checkpoint(model, source)
    names = dict(model.named_parameters())
    selected = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.1.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.1.mlp.experts.gate_up_proj",
        "model.language_model.layers.1.self_attn.q_b_proj.weight",
        "lm_head.weight",
    ]
    assert all(name in names for name in selected), sorted(names)
    units = {}
    for name in selected:
        shape = tuple(names[name].shape)
        packed = len(shape) == 3
        qname = name.removesuffix(".weight")
        units[qname] = _probe_unit(
            shape[-2], shape[-1], experts=shape[0] if packed else None,
            packed=packed)
    probe = {"schema": stock.PROBE_CENSUS_SCHEMA,
             "meta": {"calib_hash": CALIB_HASH},
             "n_stats": len(units), "units": units}
    if native:
        raw = {name: value.clone() for name, value in
               load_file(str(source / "model.safetensors")).items()}
        encoded = []
        for name, value in list(raw.items()):
            if (".mlp." not in name or ".shared_experts." in name
                    or not name.endswith(".weight")
                    or not name.endswith(("gate_proj.weight", "up_proj.weight",
                                          "down_proj.weight"))):
                continue
            assert value.ndim == 2, (name, value.shape)
            rows, cols = value.shape
            raw[name] = torch.full(value.shape, 0.5).to(torch.float8_e4m3fn)
            raw[name + "_scale_inv"] = torch.full(
                ((rows + 127) // 128, (cols + 127) // 128), 2.003)
            encoded.append(name)
        assert encoded and any(".experts." in name for name in encoded)
        fresh = source / "replacement.safetensors"
        save_file(raw, str(fresh))
        fresh.replace(source / "model.safetensors")
        document = json.loads((source / "config.json").read_text())
        document["quantization_config"] = {
            "quant_method": "fp8", "weight_block_size": [128, 128]}
        (source / "config.json").write_text(json.dumps(document))
    else:
        raw = load_file(str(source / "model.safetensors"))
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw}}))
    census = stock.build_checkpoint_census(source)
    assert set(units) <= set(census["units"])
    for qname in units:
        expected = ("fp8" if native and ".mlp." in qname
                    and ".shared_experts." not in qname else "bf16")
        assert census["units"][qname]["source_kind"] == expected
    assert bool(source_model_decode_contract(source)) is native
    return source, probe, census


def _actual_streamed_identity(source, tmp_path):
    runner = build_streamed_causal_lm(
        str(source), device=torch.device("cpu"), dtype=torch.bfloat16,
        offload_folder=str(tmp_path / "identity-offload"),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager")
    try:
        runner.context.begin_source_initialization_audit()
        for layer in range(runner.num_layers):
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=True,
                                   prefetch_following=False)
            assert all(not value.is_meta and torch.isfinite(value).all()
                       for value in runner.layers[layer].parameters())
        witness = runner.context.source_initialization_contract()
        assert witness["schema"] == "prismaquant.streaming_initialization.v1"
        identity = build_streamed_model_identity(runner, str(source))
        assert identity == validate_streamed_model_identity(
            identity, where="actual stock harvest identity fixture")
        assert identity["checkpoint_weight_map"]
        assert identity["weight_map"]
        return identity
    finally:
        runner.shutdown()


def coherent_glm_campaign(tmp_path):
    """Bind synthetic prices only after the actual streamed source load."""
    source, probe, census = _source(tmp_path)
    model_identity = _actual_streamed_identity(source, tmp_path)
    binding = build_source_model_binding(source, dtype=torch.bfloat16)
    harvest = _harvest(
        probe, census, model_identity=model_identity, source_binding=binding)
    expert = _expert_payload(probe)
    expert["provenance"]["model"] = str(source)
    for rows in expert["costs"].values():
        for row in rows.values():
            row["source_binding"] = binding
    return probe, census, harvest, expert


def producer_source(tmp_path, *, native=True, tied=False, omit_projection=False):
    from prismaquant.model_runtime import initialize_model_runtime
    from test_mtp_module_arch import _minimal_moe_text_config

    initialize_model_runtime()
    from transformers import AutoModelForCausalLM

    config = _minimal_moe_text_config()
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    config.tie_word_embeddings = tied
    template = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    raw, expected = {}, {}
    block = 48

    def encoded(key, shape, code, base):
        weight = torch.full(shape, code).to(torch.float8_e4m3fn)
        rows, cols = ((size + block - 1) // block for size in shape)
        scale = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) * 0.51 + base + 0.003
        expanded = scale.to(torch.bfloat16).repeat_interleave(block, 0)[:shape[0]]
        expanded = expanded.repeat_interleave(block, 1)[:, :shape[1]]
        decoded = weight.to(torch.bfloat16) * expanded
        raw[key] = weight if native else decoded
        if native:
            raw[key + "_scale_inv"] = scale
        return decoded

    for name, tensor in template.state_dict().items():
        if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            prefix, leaf = name.rsplit(".", 1)
            wanted = torch.empty_like(tensor)
            for expert in range(tensor.shape[0]):
                if leaf == "gate_up_proj":
                    half = tensor.shape[1] // 2
                    parts = [("gate_proj", wanted[expert, :half], 0.5, 2),
                             ("up_proj", wanted[expert, half:], 1.875, 6)]
                else:
                    parts = [("down_proj", wanted[expert], 0.25, 10)]
                for projection, target, code, base in parts:
                    key = f"{prefix}.{expert}.{projection}.weight"
                    target.copy_(encoded(key, tuple(target.shape), code, base + expert))
                    if omit_projection and expert == 0 and projection == "up_proj":
                        raw.pop(key)
                        raw.pop(key + "_scale_inv", None)
            expected[name] = wanted
        elif name.endswith(".self_attn.o_proj.weight"):
            expected[name] = encoded(name, tuple(tensor.shape), 1.875, 1)
        else:
            raw[name] = torch.full_like(tensor, 0.125)
            expected[name] = raw[name]
    if tied:
        raw.pop("lm_head.weight")
    root = tmp_path / "source"
    root.mkdir()
    document = config.to_dict()
    if native:
        document["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [block, block]}
    (root / "config.json").write_text(json.dumps(document))
    save_file(raw, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in raw},
    }))
    return root, expected


def producer_cpu_guard(tmp_path, monkeypatch, *, capacity=4 << 30):
    from prismaquant import memory_management as memory

    root = tmp_path / "cgroup"
    scope = root / "task"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(capacity))
    (scope / "memory.current").write_text("0")
    (scope / "memory.stat").write_text("anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "membership"
    membership.write_text("0::/task\n")
    monkeypatch.setattr(memory, "_host_memory_info", lambda: (64 << 30, 128 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    return memory.CaptureMemoryGuard(torch.device("cpu"), cgroup_root=root, membership=membership)



def producer_cli_inputs(monkeypatch):
    """Double CUDA admission, tokenizer and calibration sampling only."""
    from prismaquant import calibration_data, gpu_guard
    from transformers import AutoTokenizer

    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_args: torch.device('cpu'))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *_args, **_kwargs: object())
    monkeypatch.setattr(calibration_data, 'load_calibration', lambda *_args, **_kwargs: tokens)
    return tokens
