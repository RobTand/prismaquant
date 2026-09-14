"""Selected dense/expert bundle keeps exact measured wire identities."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.tessera_export_lane import TesseraExportLaneError, selected_cached_units_manifest
from test_tessera_expert_projection import (
    STACK, _declared, _projection, _record,
)

FMT = "TESSERA_E4M3_K1_R1024"
DENSE = "model.layers.2.mlp.down_proj"


def fixture(tmp_path: Path):
    projection = _projection(experts=(0,))
    carried = tep.carried_projection(
        projection, tep.bind_expert_projection(projection, declared=_declared(experts=(0,))),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")
    source, experts, _ = tep.carried_units(carried)
    names = {DENSE, *experts}
    records, cells, units, costs = {}, {}, {}, {}
    for name in names:
        expert = name in experts
        record = (_record(tmp_path, name, experts[name]) if expert else
                  _record(tmp_path, name, {"tensor": name + ".weight", "rows": 2, "cols": 2,
                                           "source_tensor": name + ".weight", "source_layout": "whole",
                                           "source_slice": {}, "expert": 0, "projection": "down_proj", "group": "w2"}))
        record["identity"]["schema"] = (
            "tessera.cached_unit_inputs.v1" if expert else "tessera.encoding_inputs.v1")
        record["identity"]["source"] = {"dtype": "torch.bfloat16", "shape": [2, 2],
                                          "sha256": "a" * 64}
        record["identity"]["calibration"] = {"hessian": {"sha256": "b" * 64}}
        record["identity"]["encoder_source_sha256"] = "c" * 64
        if not expert:
            record["identity"].pop("projection")
        records[name] = record
        cells[name, FMT] = {"record": record}
        units[name] = {"weight": copy.deepcopy(record["identity"]["source"]),
                       "hessian": copy.deepcopy(record["identity"]["calibration"]["hessian"])}
        costs[name] = {FMT: {"joint_operator_identity": {"source_weight": {"shape": [2, 2]}}}}
    handoff = {"costs": costs, "provenance": {
        "tessera_joint_allocation": {"status": "research_metadata_handoff"},
        tep.PROJECTION_KEY: carried, "wire_dir": str(tmp_path.resolve())}}
    metadata = {tep.PROJECTION_KEY: carried, tep.WIRE_DIR_KEY: str(tmp_path.resolve()),
                tep.EXPERT_WIRES_KEY: {name: records[name] for name in experts}}
    data = SimpleNamespace(unit_scope=None, census={"unit_shapes": {name: [2, 2] for name in names}},
        manifest={"identity": {"units": units, "encoder_source_sha256": "c" * 64}},
        payload={"provenance": {tep.PROJECTION_KEY: carried, "wire_dir": str(tmp_path.resolve())},
                 "costs": {name: {FMT: {"output_mse_measured": True,
                                      "hessian_identity": {"applied": True}}} for name in names}},
        cells=cells)
    return source, names, records, handoff, metadata, data


def test_full_selected_bundle_includes_dense_and_expert(tmp_path):
    source, names, _, handoff, metadata, data = fixture(tmp_path)
    manifest = selected_cached_units_manifest(
        {name: FMT for name in names}, metadata, handoff, data,
        schema="tessera.cached_units.v1")
    from tessera.cached_unit import CachedUnitBundle
    bundle = CachedUnitBundle(manifest, tmp_path, set(names), source)
    assert set(bundle.units) == names
    assert manifest["units"][DENSE]["identity"]["schema"] == "tessera.encoding_inputs.v1"


def test_sampled_research_bundle_keeps_same_source_hessian_encoder_and_wire_gates(tmp_path):
    from prismaquant.tessera_sampled_stack_proposal import (
        SCHEMA, selected_assignment_sha256)
    source, names, records, handoff, metadata, data = fixture(tmp_path)
    assignment = {name: FMT for name in names}
    panel = {'status': 'diagnostic_pilot'}
    handoff['provenance']['joint_eval'] = panel
    handoff['provenance']['tessera_joint_allocation'].update(
        status='research_sampled_joint_panel', plan_sha256='a'*64,
        prepared={'path': '/prepared', 'sha256': 'b'*64})
    proposal = {'schema': SCHEMA, 'status': 'research_proposal',
        'production_export_authority': False, 'validation_export_eligible': None,
        'research_validation_permitted': True, 'pilot': panel,
        'expanded_assignment': assignment,
        'selected_assignment_sha256': selected_assignment_sha256(assignment),
        'original_joint_plan_sha256': 'a'*64,
        'original_prepared': handoff['provenance']['tessera_joint_allocation']['prepared']}
    manifest = selected_cached_units_manifest(
        assignment, metadata, handoff, data, schema='tessera.cached_units.v1',
        research_proposal=proposal)
    assert set(manifest['units']) == names
    records[DENSE]['identity']['encoder_source_sha256'] = 'd'*64
    with pytest.raises(TesseraExportLaneError, match='encoder differs'):
        selected_cached_units_manifest(assignment, metadata, handoff, data,
            schema='tessera.cached_units.v1', research_proposal=proposal)


@pytest.mark.parametrize("change,match", [
    ("interpolated", "no exact measured joint wire"),
    ("source", "source differs from checkpoint seal"),
    ("hessian", "Hessian differs from checkpoint seal"),
    ("encoder", "encoder differs from checkpoint seal"),
    ("wire", "dense wire differs from measured receipt"),
    ("coverage", "does not cover the full source roster"),
])
def test_missing_or_changed_selected_evidence_refuses(tmp_path, change, match):
    _, names, records, handoff, metadata, data = fixture(tmp_path)
    selected = {name: FMT for name in names}
    if change == "interpolated":
        data.cells.pop((DENSE, FMT))
    elif change == "source":
        records[DENSE]["identity"]["source"]["sha256"] = "d" * 64
    elif change == "hessian":
        records[DENSE]["identity"]["calibration"]["hessian"]["sha256"] = "d" * 64
    elif change == "encoder":
        records[DENSE]["identity"]["encoder_source_sha256"] = "d" * 64
    elif change == "wire":
        (tmp_path / records[DENSE]["file"]).write_bytes(b"changed")
    elif change == "coverage":
        selected.pop(DENSE)
    with pytest.raises(TesseraExportLaneError, match=match):
        selected_cached_units_manifest(selected, metadata, handoff, data,
                                       schema="tessera.cached_units.v1")
