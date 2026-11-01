"""Synthetic CPU cache/intake evidence, NOT production wires or serving qualification.

The receipt fixtures deliberately contain non-decodable sentinel bytes. They
exercise closure and authentication, never a numerical encoder. The passthrough
artifact test uses the installed pinned producer and real BF16 safetensors.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import tessera_plan_writer as writer
from prismaquant.cost_stage_checkpoint import write_unit
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.model_profiles import detect_profile
from prismaquant.tessera_census_cache import (
    CensusCacheError, census_selected_cached_units_manifest,
    load_selected_wire_records,
)
from prismaquant.tessera_export_lane import (
    TesseraExportLaneError, _write_plan_assignment, packaged_contract_path,
)
from prismaquant.tessera_joint_aura import STAGE
from prismaquant.tessera_serving_runtime_pin import load_tessera_serving_runtime_pin
from test_tessera_census_cache import (
    DENSE, ENCODING_SCHEMA, FMT, INPUT_SCHEMA, SEAL, _control,
)
from test_tessera_expert_projection import _record

VISION = "model.visual.blocks.0.attn.qkv"
MERGER = "model.visual.merger.mlp.0"
BODY = "model.layers.0.mlp.down_proj"
VISUALS = (VISION, MERGER)


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "num_hidden_layers": 1,
                        "intermediate_size": 32, "moe_intermediate_size": 32,
                        "num_experts": 1},
        "vision_config": {"hidden_size": 32, "intermediate_size": 32},
    }))
    tensors = {name + ".weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
               for name in (BODY, *VISUALS)}
    # Fixed/non-Linear controls are legal SOURCE-PRECISION passthrough, not
    # admission to the quantizable visual Linear roster.
    tensors["model.visual.pos_embed.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    tensors["model.visual.blocks.0.norm1.weight"] = torch.ones(2, dtype=torch.bfloat16)
    save_file(tensors, str(source / "model.safetensors"))
    return source, detect_profile(source), tensors


def _vision_cache(tmp_path):
    names, experts, records, cost, roster, shapes, assignment, metadata = _control(tmp_path)
    # Replace the text dense fixture by two independent visual dense receipts;
    # keep the expert controls unchanged to detect accidental coverage loss.
    template = copy.deepcopy(records[DENSE])
    for mapping in (cost["costs"], roster["units"], shapes, assignment, records):
        mapping.pop(DENSE)
    wire = Path(cost["provenance"]["wire_dir"])
    for name in VISUALS:
        record = _record(wire, name, {"tensor": name + ".weight", "rows": 2, "cols": 2,
            "source_tensor": name + ".weight", "source_layout": "whole", "source_slice": {},
            "expert": 0, "projection": "down_proj", "group": "w2"})
        record["identity"] = copy.deepcopy(template["identity"])
        record["identity"]["unit"] = name
        records[name] = record
        roster["units"][name] = {"weight": copy.deepcopy(record["identity"]["source"]),
                                "hessian": copy.deepcopy(record["identity"]["calibration"]["hessian"])}
        shapes[name] = [2, 2]
        cost["costs"][name] = {FMT: {"output_mse": 1e-3, "output_mse_measured": True,
            "wire_bytes": record["blob_bytes"], "hessian_identity": {"applied": True}}}
        assignment[name] = FMT
    return experts, records, cost, roster, shapes, assignment, metadata


def _close(cost, roster, shapes, assignment, metadata, records, *, hash_blobs=False):
    return census_selected_cached_units_manifest(
        assignment, metadata, cost, roster, shapes, records,
        input_schema=INPUT_SCHEMA, encoding_input_schema=ENCODING_SCHEMA,
        cache_schema="tessera.cached_units.v1", blob_workers=2, hash_blobs=hash_blobs)


def _plan(tmp_path, profile, source, assignment):
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps(assignment))
    return _write_plan_assignment(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  profile=profile, model_path=source)


def test_visual_selected_cache_preserves_exact_population_and_receipts(tmp_path):
    from tessera.cached_unit import CachedUnitBundle
    experts, records, cost, roster, shapes, assignment, metadata = _vision_cache(tmp_path)
    source, profile, _ = _source(tmp_path)
    admitted = profile.visual_linear_roster(source)
    assert set(admitted) == set(VISUALS)
    assert all(admitted[name]["shape"] == shapes[name] and
               admitted[name]["visual_root"] == "model.visual" for name in VISUALS)
    expected = copy.deepcopy(records)
    # Exercise the journal reader as well as the manifest builder.
    for name in VISUALS:
        write_unit(tmp_path / "parts", stage=STAGE, qname=name, identity_sha256=SEAL,
                   state={"anchors": [], "wire_records": {FMT: records[name]}})
    loaded = load_selected_wire_records(tmp_path / "parts", assignment,
                                        identity_sha256=SEAL, workers=2)
    manifest = _close(cost, roster, shapes, assignment, metadata, loaded, hash_blobs=True)
    assert set(manifest["units"]) == set(assignment) == set(experts) | set(VISUALS)
    assert manifest["units"] == expected
    before = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    bundle = CachedUnitBundle(manifest, Path(cost["provenance"]["wire_dir"]),
                              set(assignment), manifest["source"])
    for name in VISUALS:
        blob, receipt = bundle.read(name)
        assert receipt == expected[name]
        assert hashlib.sha256(blob).hexdigest() == receipt["blob_sha256"]
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == before


@pytest.mark.parametrize("name", VISUALS)
@pytest.mark.parametrize("change,match", [
    ("absent_assignment", "full source roster"),
    ("absent_record", "no exact measured census wire"),
    ("wrong_shape", "geometry differs"),
    ("unmeasured", "no exact measured census wire"),
    ("source", "source differs"), ("encoder", "encoder differs"),
    ("recipe", "identity differs"), ("unit", "identity differs"),
    ("wire_bytes", "wire_bytes differs"), ("wire_missing", "dense wire escapes|missing|No such file"),
    ("wire_changed", "dense wire differs"),
])
def test_visual_selected_cache_refuses_missing_or_mismatched_evidence(tmp_path, name, change, match):
    _, records, cost, roster, shapes, assignment, metadata = _vision_cache(tmp_path)
    record = records[name]
    if change == "absent_assignment": assignment.pop(name)
    elif change == "absent_record": records.pop(name)
    elif change == "wrong_shape": shapes[name] = [3, 2]
    elif change == "unmeasured": cost["costs"][name][FMT]["output_mse_measured"] = False
    elif change == "source": record["identity"]["source"]["sha256"] = "d" * 64
    elif change == "encoder": record["identity"]["encoder_source_sha256"] = "d" * 64
    elif change == "recipe": record["identity"]["recipe"]["q256"] = 896
    elif change == "unit": record["identity"]["unit"] = "vis_attn_qkv"
    elif change == "wire_bytes": cost["costs"][name][FMT]["wire_bytes"] += 1
    elif change == "wire_missing": (Path(cost["provenance"]["wire_dir"]) / record["file"]).unlink()
    elif change == "wire_changed":
        path = Path(cost["provenance"]["wire_dir"]) / record["file"]
        raw = path.read_bytes(); path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises((CensusCacheError, FileNotFoundError), match=match):
        _close(cost, roster, shapes, assignment, metadata, records, hash_blobs=True)


@pytest.mark.parametrize("name", VISUALS)
def test_visual_same_size_substitution_refuses_at_original_wire_intake(tmp_path, name):
    from tessera.cached_unit import CachedUnitBundle, verify_cached_unit
    _, records, cost, roster, shapes, assignment, metadata = _vision_cache(tmp_path)
    manifest = _close(cost, roster, shapes, assignment, metadata, records)
    bundle = CachedUnitBundle(manifest, Path(cost["provenance"]["wire_dir"]),
                              set(assignment), manifest["source"])
    path = Path(cost["provenance"]["wire_dir"]) / records[name]["file"]
    raw = path.read_bytes(); path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    blob, receipt = bundle.read(name)
    with pytest.raises(ValueError, match="blob size/sha256 mismatch"):
        verify_cached_unit(blob, receipt, receipt["identity"])


@pytest.mark.parametrize("name", [*VISUALS, "vis_attn_qkv", "merger_fc1",
                                   "model.visual.pos_embed", "unowned.visual.linear"])
def test_nonbody_lowbit_keeps_existing_named_refusal(tmp_path, name):
    source, profile, tensors = _source(tmp_path)
    if name in ("vis_attn_qkv", "merger_fc1", "unowned.visual.linear"):
        # Even a real rank-2 tensor is not permission to use an unowned
        # namespace or an allocator role tag as a source/serving module name.
        tensors[name + ".weight"] = tensors[VISION + ".weight"].clone()
        save_file(tensors, str(source / "model.safetensors"))
    with pytest.raises(TesseraExportLaneError) as error:
        _plan(tmp_path, profile, source,
              {name: {"data_type": "tessera", "tessera_format": FMT}})
    assert name in str(error.value)
    if name in (*VISUALS, "model.visual.pos_embed"):
        assert "outside the text graph" in str(error.value)
    assert not (tmp_path / "layer_config.tessera-source-units.json").exists()


@pytest.mark.parametrize("name", ["model.visual.blocks.9.attn.qkv", "model.visual.merger.mlp.9",
                                   "vis_attn_qkv", "merger_fc1", "unowned.visual.linear"])
def test_explicit_bf16_source_name_must_exist_before_plan_projection(tmp_path, name):
    # RED on main: projection silently drops even a misspelled BF16 assignment.
    # This must be a source-name refusal, not permission to encode nonbody wires.
    source, profile, _ = _source(tmp_path)
    with pytest.raises(TesseraExportLaneError) as error:
        _plan(tmp_path, profile, source, {name: {"bits": 16, "data_type": "float"}})
    assert name in str(error.value)
    assert not (tmp_path / "layer_config.tessera-source-units.json").exists()


@pytest.mark.parametrize("name", VISUALS)
def test_source_header_geometry_must_agree_with_selected_visual_receipt(tmp_path, name):
    _, records, cost, roster, shapes, assignment, metadata = _vision_cache(tmp_path)
    source, profile, tensors = _source(tmp_path)
    tensors[name + ".weight"] = torch.zeros(3, 2, dtype=torch.bfloat16)
    save_file(tensors, str(source / "model.safetensors"))
    for unit, entry in profile.visual_linear_roster(source).items():
        shapes[unit] = entry["shape"]
    with pytest.raises(CensusCacheError, match="geometry differs") as error:
        _close(cost, roster, shapes, assignment, metadata, records)
    assert name in str(error.value)


def test_installed_pin_exports_real_bf16_source_names_not_visual_lowbit_claims(tmp_path):
    pin = load_tessera_serving_runtime_pin()
    assert pin.producer_commit == "aaa417003f0ee690e8ace138acd18df7f9531369"
    contract_path = packaged_contract_path()
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == pin.contract_sha256
    contract = json.loads(contract_path.read_text())
    source, profile, tensors = _source(tmp_path)
    receipt = _plan(tmp_path, profile, source, {
        name: {"bits": 16, "data_type": "float"} for name in (BODY, *VISUALS)})
    projected = Path(receipt["plan_assignment"])
    view = json.loads(projected.read_text())
    assert view[LAYER_CONFIG_META_KEY]["tessera_export_assignment"][
        "source_precision_outside_graph"] == sorted(VISUALS)
    assert set(view) - {LAYER_CONFIG_META_KEY} == {BODY}
    plan = tmp_path / "plan.json"
    assert writer.main([str(projected), str(source), str(plan), "--cover", "as-allocated",
                        "--no-uniform-control"]) == 0
    expected_plan = {"schema": "tessera.serving_plan.v1", BODY + ".weight": "BF16"}
    assert json.loads(plan.read_text()) == expected_plan
    out = tmp_path / "export"
    # --layers 0 is the producer's explicit passthrough-copy control. No
    # encoding, gates, source authentication, shard writer or manifest is mocked.
    # Run the real installed CLI in a fresh process: other allocator tests
    # install synthetic lane contracts and can populate Tessera module caches.
    completed = subprocess.run([sys.executable, "-m", "tessera.export_serving", str(source), str(out),
        "--device", "cpu", "--grid", "E4M3", "--q256", "1024", "--layers", "0",
        "--plan-json", str(plan)], cwd=source, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        assert set(handle.keys()) == set(tensors)
        for name, tensor in tensors.items():
            assert handle.get_tensor(name).dtype == torch.bfloat16
            assert torch.equal(handle.get_tensor(name), tensor)
    config = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert "quantization_config" not in config
    # A passthrough copy makes NO serving-cell claim. Inspect actual emitted
    # names and the pinned contract instead of assuming a newer Tessera
    # checkout's visual recognition qualifies this pin.
    # The producer's git stamp is recorded, not used to qualify its code:
    # git_hash() may use ambient checkout/install provenance on CPU arms.
    # The check above binds recognition to PQ's exact packaged CONTRACT hash;
    # this BF16-copy test makes no production-producer/source qualification.
    assert isinstance(manifest["git"], str) and manifest["git"]
    assert contract["schema"] == pin.runtime_contract_schema
    assert manifest["serving_gate"]["contract_version"] == contract["contract_version"]
    assert manifest["serving_gate"]["allow_unserveable"] is False
    assert manifest["serving_gate"]["allow_unrouted"] is False
    assert manifest["modules"] == {}
    assert manifest["plan"] == {BODY + ".weight": "BF16"}
    assert manifest["totals"]["quantized_params"] == 0
    assert manifest["totals"]["passthrough_bytes"] == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values())
