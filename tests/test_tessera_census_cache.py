"""Census wires close for an assignment without a joint handoff, or refuse by name."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, canonical_json_sha256, unit_path, write_unit,
)
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.tessera_census_cache import (
    ROSTER_SCHEMA, CensusCacheError, canonical_json_sha256_of_loaded,
    census_layer_config, census_selected_cached_units_manifest,
    load_selected_wire_records, seal_roster, uniform_assignment,
)
from prismaquant.tessera_joint_aura import STAGE
from test_tessera_expert_projection import STACK, _declared, _projection, _record

FMT = "TESSERA_E4M3_K1_R1024"
DENSE = "model.layers.2.mlp.down_proj"
SEAL = "e" * 64
INPUT_SCHEMA = "tessera.cached_unit_inputs.v1"
ENCODING_SCHEMA = "tessera.encoding_inputs.v1"


def _world(tmp_path: Path):
    wire = tmp_path / "wire"
    wire.mkdir()
    projection = _projection(experts=(0,))
    carried = tep.carried_projection(
        projection, tep.bind_expert_projection(projection, declared=_declared(experts=(0,))),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")
    _source, experts, _ = tep.carried_units(carried)
    names = sorted({DENSE, *experts})
    records, roster_units, rows = {}, {}, {}
    for name in names:
        routed = name in experts
        unit = experts[name] if routed else {
            "tensor": name + ".weight", "rows": 2, "cols": 2, "source_tensor": name + ".weight",
            "source_layout": "whole", "source_slice": {}, "expert": 0,
            "projection": "down_proj", "group": "w2"}
        record = _record(wire, name, unit)
        identity = record["identity"]
        identity["schema"] = INPUT_SCHEMA if routed else ENCODING_SCHEMA
        identity["source"] = {"dtype": "torch.bfloat16", "shape": [2, 2], "sha256": "a" * 64}
        identity["calibration"] = {"hessian": {"sha256": "b" * 64}}
        identity["encoder_source_sha256"] = "c" * 64
        if not routed:
            identity.pop("projection")
        records[name] = record
        roster_units[name] = {"weight": copy.deepcopy(identity["source"]),
                              "hessian": copy.deepcopy(identity["calibration"]["hessian"])}
        rows[name] = {FMT: {"output_mse": 1e-3, "output_mse_measured": True,
                            "wire_bytes": record["blob_bytes"],
                            "hessian_identity": {"applied": True}}}
        write_unit(tmp_path / "parts", stage=STAGE, qname=name, identity_sha256=SEAL,
                   state={"anchors": [], "wire_records": {FMT: copy.deepcopy(record)}})
    cost = {"costs": rows,
            "provenance": {tep.PROJECTION_KEY: carried, "wire_dir": str(wire.resolve())},
            tep.EXPERT_WIRES_KEY: {name: {FMT: copy.deepcopy(records[name])} for name in experts}}
    roster = {"schema": ROSTER_SCHEMA, "identity_sha256": SEAL,
              "encoder_source_sha256": "c" * 64, "units": roster_units}
    shapes = {name: [2, 2] for name in names}
    return names, experts, records, cost, roster, shapes


def _control(tmp_path: Path):
    names, experts, records, cost, roster, shapes = _world(tmp_path)
    assignment = uniform_assignment(cost, FMT)
    config = census_layer_config(
        cost, assignment, target_profile="glm_packed_research_sm121",
        entry_for=lambda fmt: {"data_type": "tessera", "tessera_format": fmt})
    return names, experts, records, cost, roster, shapes, assignment, config[LAYER_CONFIG_META_KEY]


def _build(tmp_path, assignment, metadata, cost, roster, shapes, *, parts=None, seal=SEAL):
    loaded = load_selected_wire_records(parts or tmp_path / "parts", assignment,
                                        identity_sha256=seal, workers=2)
    return census_selected_cached_units_manifest(
        assignment, metadata, cost, roster, shapes, loaded, input_schema=INPUT_SCHEMA,
        encoding_input_schema=ENCODING_SCHEMA, cache_schema="tessera.cached_units.v1",
        hash_workers=2)


def test_uniform_control_closes_every_dense_and_routed_wire(tmp_path):
    names, experts, records, cost, roster, shapes, assignment, metadata = _control(tmp_path)
    assert metadata[tep.STACK_FORMATS_KEY] == {STACK: FMT}
    assert metadata[tep.EXPERT_WIRES_KEY] == {name: records[name] for name in experts}
    assert metadata[tep.WIRE_DIR_KEY] == cost["provenance"]["wire_dir"]
    manifest = _build(tmp_path, assignment, metadata, cost, roster, shapes)
    assert set(manifest["units"]) == set(names)
    assert manifest["units"][DENSE]["identity"]["schema"] == ENCODING_SCHEMA
    from tessera.cached_unit import CachedUnitBundle
    bundle = CachedUnitBundle(manifest, Path(cost["provenance"]["wire_dir"]), set(names),
                              manifest["source"])
    assert set(bundle.units) == set(names)


def test_uniform_control_refuses_an_unmeasured_cell(tmp_path):
    _names, _experts, _records, cost, *_ = _world(tmp_path)
    cost["costs"][DENSE][FMT]["output_mse_measured"] = False
    with pytest.raises(CensusCacheError, match="no measured TESSERA_E4M3_K1_R1024 cell"):
        uniform_assignment(cost, FMT)


def test_allocator_sidecar_outside_the_census_passes_through_at_bf16(tmp_path):
    # A real allocator layer config also names the Linears it kept outside the
    # priced population; GLM-5.3 Flash's carries 124 visual-tower BF16 rows.
    names, _experts, _records, cost, roster, shapes, assignment, metadata = _control(tmp_path)
    assignment["model.visual.blocks.0.attn.proj"] = "BF16"
    manifest = _build(tmp_path, assignment, metadata, cost, roster, shapes)
    assert set(manifest["units"]) == set(names)


@pytest.mark.parametrize("fmt", [FMT, "NVFP4"])
def test_a_wired_rung_outside_the_census_refuses(tmp_path, fmt):
    *_, cost, roster, shapes, assignment, metadata = _control(tmp_path)
    # Records come from the census names: the journal holds nothing for the
    # outside unit, so the refusal under test is the manifest's own.
    loaded = load_selected_wire_records(tmp_path / "parts", assignment,
                                        identity_sha256=SEAL, workers=2)
    assignment["model.visual.blocks.0.attn.proj"] = fmt
    with pytest.raises(CensusCacheError, match="outside the census roster are not BF16"):
        census_selected_cached_units_manifest(
            assignment, metadata, cost, roster, shapes, loaded, input_schema=INPUT_SCHEMA,
            encoding_input_schema=ENCODING_SCHEMA, cache_schema="tessera.cached_units.v1",
            hash_workers=2)


def _routed(experts):
    return sorted(experts)[0]


@pytest.mark.parametrize("change,match", [
    ("roster", "does not cover the full source roster"),
    ("projection", "exact carried producer projection"),
    ("wire_dir", "wire directory differs"),
    ("stack_formats", "stack formats differ"),
    ("unmeasured", "no exact measured census wire"),
    ("source", "source differs from checkpoint seal"),
    ("hessian", "Hessian differs from checkpoint seal"),
    ("encoder", "encoder differs from checkpoint seal"),
    ("wire_bytes", "measured wire_bytes differs"),
    ("receipt", "selected expert receipt differs"),
    ("cost_receipt", "cost table expert receipt differs"),
    ("dense_bytes", "dense wire differs from measured receipt"),
    ("routed_bytes", "does not match its receipt"),
    ("journal_seal", "identity_sha256"),
    ("census_roster", "complete census roster"),
])
def test_changed_census_evidence_refuses(tmp_path, change, match):
    names, experts, records, cost, roster, shapes, assignment, metadata = _control(tmp_path)
    routed = _routed(experts)
    seal = SEAL
    wire = Path(cost["provenance"]["wire_dir"])
    if change == "roster":
        assignment.pop(DENSE)
    elif change == "projection":
        metadata.pop(tep.PROJECTION_KEY)
    elif change == "wire_dir":
        metadata[tep.WIRE_DIR_KEY] = str(tmp_path)
    elif change == "stack_formats":
        metadata[tep.STACK_FORMATS_KEY] = {}
    elif change == "unmeasured":
        cost["costs"][DENSE][FMT]["output_mse_measured"] = False
    elif change == "source":
        roster["units"][DENSE]["weight"]["sha256"] = "d" * 64
    elif change == "hessian":
        roster["units"][DENSE]["hessian"]["sha256"] = "d" * 64
    elif change == "encoder":
        roster["encoder_source_sha256"] = "d" * 64
    elif change == "wire_bytes":
        cost["costs"][DENSE][FMT]["wire_bytes"] += 1
    elif change == "receipt":
        metadata[tep.EXPERT_WIRES_KEY][routed] = {**records[routed], "blob_sha256": "0" * 64}
    elif change == "cost_receipt":
        cost[tep.EXPERT_WIRES_KEY][routed][FMT]["blob_sha256"] = "0" * 64
    elif change == "dense_bytes":
        (wire / records[DENSE]["file"]).write_bytes(b"changed")
    elif change == "routed_bytes":
        (wire / records[routed]["file"]).write_bytes(b"changed")
    elif change == "journal_seal":
        seal = "f" * 64
    elif change == "census_roster":
        shapes.pop(DENSE)
    with pytest.raises(CensusCacheError, match=match):
        _build(tmp_path, assignment, metadata, cost, roster, shapes, seal=seal)


def _manifest_for(identity: dict) -> dict:
    return {"schema": MANIFEST_SCHEMA, "stage": STAGE, "identity": identity,
            "identity_sha256": canonical_json_sha256(identity, where="t"),
            "units": [{"qname": name, "file": unit_path(Path("."), name).as_posix()}
                      for name in identity["units"]]}


def _identity() -> dict:
    return json.loads(json.dumps({
        "encoder_source_sha256": "c" * 64, "calibration": {"seed": 0, "tokens": 2.5e-300},
        "text": "naïve – ünïcode", "list": [1, [2.0, {"z": None, "a": True}], "x"],
        "units": {
            "b.proj": {"weight": {"sha256": "1" * 64, "shape": [2, 3]}, "hessian": None,
                       "menu": ["TESSERA_E4M3_K1_R1024", "BF16"], "input_global_scale": 0.1},
            "a.proj": {"weight": {"sha256": "2" * 64, "shape": [3, 2]},
                       "hessian": {"sha256": "3" * 64}, "menu": []},
        }}))


def test_streamed_seal_digest_is_the_checkpoint_digest():
    identity = _identity()
    assert canonical_json_sha256_of_loaded(identity) == canonical_json_sha256(identity, where="t")
    roster = seal_roster(_manifest_for(identity))
    assert roster["identity_sha256"] == canonical_json_sha256(identity, where="t")
    assert roster["units"]["a.proj"] == {"weight": identity["units"]["a.proj"]["weight"],
                                         "hessian": {"sha256": "3" * 64}}


@pytest.mark.parametrize("change,match", [
    ("digest", "differs from its stored seal"),
    ("journals", "journal list differs"),
    ("weight", "no weight and Hessian identity"),
    ("stage", "not a Tessera campaign checkpoint"),
])
def test_seal_roster_refuses_a_drifted_manifest(change, match):
    identity = _identity()
    manifest = _manifest_for(identity)
    if change == "digest":
        manifest["identity"]["units"]["a.proj"]["weight"]["sha256"] = "9" * 64
    elif change == "journals":
        manifest["units"].pop()
    elif change == "weight":
        del identity["units"]["a.proj"]["weight"]
        manifest = _manifest_for(identity)
    elif change == "stage":
        manifest["stage"] = "other"
    with pytest.raises(CensusCacheError, match=match):
        seal_roster(manifest)
