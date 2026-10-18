"""Offline stock source admission with actual GLM checkpoint/identity facts.

The GLM source, source-kind census, model loading, streamed identity and stock
CLI are real. Prices and calibration-hash label are synthetic. No AURA/KL,
full harvest, artifact or serving qualification follows. Model sources are
removed before pricing to expose any accidental live-checkpoint reread.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import glm53_stock_reprice as stock
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.streamed_model_identity import (
    build_source_model_binding, build_streamed_model_identity,
    source_model_decode_contract, validate_streamed_model_identity,
)
from test_glm5_next_streamed_forward_parity import _tiny_config, _build_model
from tests.test_glm_campaign_streaming import write_original_layout_checkpoint
from tests.test_glm53_stock_campaign import (
    CALIB_HASH, _expert_payload, _harvest, _probe_unit,
)
from tests.test_resident_source_decode_audit import _cpu_guard, _load


CASES = ["unmarked", "legacy-v1", "same-source-v2", "foreign-source-v2",
         "config-only-v2", "root-code-metadata-v2", "legacy-native-harvest",
         "missing-harvest-identity", "corrupt-harvest-identity"]


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
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.1.mlp.experts.gate_up_proj",
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
            if (".mlp." not in name or not name.endswith(".weight")
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
        expected = "fp8" if native and ".mlp." in qname else "bf16"
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
        # Build through the actual owner rather than hand-serializing a hash.
        identity = build_streamed_model_identity(runner, str(source))
        assert identity == validate_streamed_model_identity(
            identity, where="actual stock harvest identity fixture")
        assert identity["checkpoint_weight_map"]
        assert identity["weight_map"]
        return identity
    finally:
        runner.shutdown()


def _actual_resident_binding(source, tmp_path, monkeypatch):
    model = _load(source, tmp_path, _cpu_guard(tmp_path, monkeypatch))
    witness = model._prismaquant_resident_source_initialization
    assert witness["status"] == "completed"
    weights = dict(model.named_parameters())
    for name in (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.1.mlp.experts.gate_up_proj"):
        value = weights[name]
        assert value.dtype == torch.bfloat16 and not value.is_meta
        # Code 0.5 times the declared scale's BF16 rounding 2.0 is exactly1.
        assert torch.equal(value, torch.ones_like(value)), name
    return build_source_model_binding(source, dtype=torch.bfloat16)


def _write_inputs(tmp_path, probe, census, harvest, expert):
    paths = {name: tmp_path / name for name in (
        "probe.json", "census.json", "harvest.pkl", "expert.pkl")}
    paths["probe.json"].write_text(json.dumps(probe))
    paths["census.json"].write_text(json.dumps(census))
    paths["harvest.pkl"].write_bytes(pickle.dumps(harvest))
    paths["expert.pkl"].write_bytes(pickle.dumps(expert))
    output = tmp_path / "output.pkl"
    arguments = ["campaign", "--probe-census", str(paths["probe.json"]),
                 "--checkpoint-census", str(paths["census.json"]),
                 "--harvest", str(paths["harvest.pkl"]),
                 "--expert-empirical", str(paths["expert.pkl"]),
                 "--output", str(output)]
    return paths, arguments, output


@pytest.mark.parametrize("case", CASES)
def test_real_offline_stock_source_identity_admission(tmp_path, monkeypatch, case):
    source, probe, census = _source(tmp_path)
    harvest = _harvest(probe, census)
    harvest["model_identity"] = _actual_streamed_identity(source, tmp_path)
    expert = _expert_payload(probe)
    expert["provenance"]["model"] = str(source)
    source_roots = [source]
    binding = _actual_resident_binding(source, tmp_path, monkeypatch)
    # Proposed producer evidence, only after the real streamed and resident
    # source delivery controls. Never manufacture this on an old harvest.
    harvest["source_binding"] = deepcopy(binding)
    if case in {"foreign-source-v2", "config-only-v2", "root-code-metadata-v2"}:
        other = tmp_path / "other"
        other.mkdir()
        foreign = other / "glm-source"
        shutil.copytree(source, foreign)
        if case == "foreign-source-v2":
            raw = {name: value.clone() for name, value in
                   load_file(str(foreign / "model.safetensors")).items()}
            raw["lm_head.weight"].add_(0.25)
            fresh = foreign / "replacement.safetensors"
            save_file(raw, str(fresh))
            fresh.replace(foreign / "model.safetensors")
        elif case == "config-only-v2":
            document = json.loads((foreign / "config.json").read_text())
            document["text_config"]["rms_norm_eps"] *= 2
            (foreign / "config.json").write_text(json.dumps(document))
        else:
            # v2 deliberately binds every root Python file, including one
            # not imported. This checks that exact metadata contract, not a
            # claim that the toy executes custom remote code.
            (foreign / "checkpoint_policy.py").write_text("SOURCE_POLICY = 2\n")
        foreign_binding = _actual_resident_binding(foreign, other, monkeypatch)
        assert foreign_binding["checkpoint"] != binding["checkpoint"]
        if case != "foreign-source-v2":
            assert foreign_binding["checkpoint"]["shards"] == binding["checkpoint"]["shards"]
            assert foreign_binding["checkpoint"]["metadata"] != binding["checkpoint"]["metadata"]
        binding = foreign_binding
        source_roots.append(foreign)
    elif case == "legacy-v1":
        binding = {key: deepcopy(value) for key, value in binding.items()
                   if key != "source_decode"}
        binding["schema"] = "prismaquant.source_model_binding.v1"
    elif case == "legacy-native-harvest":
        del harvest["source_binding"]
    elif case == "missing-harvest-identity":
        del harvest["model_identity"]
    elif case == "corrupt-harvest-identity":
        harvest["model_identity"]["shards"][0]["sha256"] = "f" * 64
    if case != "unmarked":
        for rows in expert["costs"].values():
            for entry in rows.values():
                entry["source_binding"] = binding
    paths, arguments, output = _write_inputs(tmp_path, probe, census, harvest, expert)
    before = {name: path.read_bytes() for name, path in paths.items()}
    for root in source_roots:
        shutil.rmtree(root)
        assert not root.exists()
    if case == "same-source-v2":
        assert stock.main(arguments) == 0
        actual = pickle.loads(output.read_bytes())
        from prismaquant.prepriced_cost import require_native_cost_source_binding

        require_native_cost_source_binding(
            actual, harvest["source_binding"], where="offline generated stock output")
        for name, rows in expert["costs"].items():
            assert actual["costs"][name] == rows
    else:
        with pytest.raises((ValueError, RuntimeError, stock.Glm53StockError),
                           match="(?i)(native|decode|source|model.identity)"):
            stock.main(arguments)
        assert not output.exists()
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_real_bf16_packed_stock_scope_keeps_existing_ladder_refusal(tmp_path):
    source, probe, census = _source(tmp_path, native=False)
    assert all(row["source_kind"] == "bf16" for row in census["units"].values())
    harvest = _harvest(probe, census)
    harvest["model_identity"] = _actual_streamed_identity(source, tmp_path)
    paths, arguments, output = _write_inputs(
        tmp_path, probe, census, harvest, _expert_payload(probe))
    before = {name: path.read_bytes() for name, path in paths.items()}
    with pytest.raises(stock.Glm53StockError, match="non-dense declarations survived"):
        stock.main(arguments)
    assert not output.exists()
    assert {name: path.read_bytes() for name, path in paths.items()} == before
