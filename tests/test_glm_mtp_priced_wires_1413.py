"""MTP selections retain the exact measured wires across rate merges (#1413)."""

import hashlib
import pickle
from pathlib import Path

import pytest

from prismaquant import tessera_expert_projection as tep
from test_glm_mtp_selection import CONSTANTS, _probe, _row
from test_tessera_expert_projection import (
    STACK, _declared, _projection, _record,
)

FMT = "TESSERA_E4M3_K1_R1024"
FMT_LOW = "TESSERA_E4M3_K1_R896"
DENSE = "model.layers.2.feed_forward.shared_experts.w1"


def _write(path, payload):
    raw = pickle.dumps(payload)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def bound_parts(tmp_path):
    return _bound_parts_for_rates(tmp_path, (1024, 896))


def _bound_parts_for_rates(tmp_path, rates, *, probe=None):
    producer = _projection(experts=(0,))
    declared = _declared(experts=(0,))
    carried = tep.carried_projection(
        producer, tep.bind_expert_projection(producer, declared=declared),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="fixture")
    _source, units, _stacks = tep.carried_units(carried)
    probe = probe or _probe()
    parts = []
    for index, rate in enumerate(rates):
        root = tmp_path / f"r{index}"
        root.mkdir()
        fmt = f"TESSERA_E4M3_K1_R{rate}"
        records = {}
        for name, unit in units.items():
            record = _record(root, name, unit, q256=rate)
            if rate != 1024:
                old = root / record["file"]
                record["file"] = record["file"].replace("R1024", f"R{rate}")
                old.rename(root / record["file"])
            records[name] = record
        m3_costs = {name: {fmt: {"wire_bytes": record["blob_bytes"],
                                 "output_mse_measured": True}}
                    for name, record in records.items()}
        m3_costs[DENSE] = {fmt: {"wire_bytes": 96, "output_mse_measured": True}}
        m3 = {"schema": "prismaquant.tessera_campaign_cost.v1",
              "costs": m3_costs,
              tep.EXPERT_WIRES_KEY: {name: {fmt: record}
                                     for name, record in records.items()},
              "provenance": {tep.PROJECTION_KEY: carried, "wire_dir": str(root)}}
        m3_ref = _write(tmp_path / f"m3-{index}.pkl", m3)
        costs = {name: {fmt: _row(name, fmt, [0.01 + index * 0.01] * 4, probe)}
                 for name in (*units, DENSE)}
        probe_sha256 = next(iter(costs.values()))[fmt]["probe_identity_sha256"]
        m4 = {"schema": "prismaquant.glm_mtp_cost.v1", "mtp_layer": 45,
              "costs": costs,
              "wire_bytes": {name: {fmt: records[name]["blob_bytes"]} for name in units}
                            | {DENSE: {fmt: 96}},
              "params": {name: 8192 for name in (*units, DENSE)},
              "source_dtype": {name: "bfloat16" for name in (*units, DENSE)},
              "groups": {STACK: sorted(units), "shared": [DENSE]},
              "provenance": {"probe_identity_sha256": probe_sha256,
                             "tessera_joint_anchors": {"inputs": {"merged_cost": m3_ref}}}}
        # The M4 part is immutable evidence in the merge's source list.
        parts.append((m4, _write(tmp_path / f"m4-{index}.pkl", m4)))
    return parts, units, carried


def test_bound_m3_receipts_survive_m4_merge_and_selected_rung(bound_parts):
    from prismaquant.glm_mtp_selection import merge_mtp_costs, select_mtp_rungs
    parts, units, carried = bound_parts
    merged = merge_mtp_costs([part for part, _ in parts],
                             sources=[source for _, source in parts])
    selected = select_mtp_rungs(merged, byte_budget=17000, constants=CONSTANTS)
    assert set(selected["assignment"]) == set(units) | {DENSE}
    assert {selected["assignment"][name] for name in units} == {FMT}
    assert selected["assignment"][DENSE] == "BF16"
    assert selected["mtp_expert_projection"] == carried
    assert set(selected["mtp_expert_wires"]) == set(units)
    with open(parts[0][0]["provenance"]["tessera_joint_anchors"]["inputs"][
            "merged_cost"]["path"], "rb") as handle:
        priced = pickle.load(handle)
    for name in units:
        assert selected["mtp_expert_wires"][name] == priced[tep.EXPERT_WIRES_KEY][name][FMT]
        assert selected["mtp_expert_wire_roots"][name] == priced["provenance"]["wire_dir"]


def test_completed_allocation_backfills_only_selected_metadata(bound_parts, tmp_path):
    from prismaquant import format_registry as fr
    from prismaquant.glm_mtp_selection import (
        backfill_mtp_selection_wires, merge_mtp_costs, select_mtp_rungs,
    )

    parts, units, _carried = bound_parts
    merged = merge_mtp_costs([part for part, _ in parts],
                             sources=[source for _, source in parts])
    cost_path = tmp_path / "merged.pkl"
    _write(cost_path, merged)
    selection = select_mtp_rungs(merged, byte_budget=17000, constants=CONSTANTS)
    assignment = selection.pop("assignment")
    for key in ("mtp_expert_projection", "mtp_expert_wires", "mtp_expert_wire_roots",
                "mtp_expert_source_bindings", "mtp_expert_wire_binding_schema"):
        selection.pop(key)
    config = {name: fr.get_format(fmt).autoround_config()
              for name, fmt in assignment.items()}
    config["body.weight"] = {"format": "unchanged"}
    config["__prismaquant__"] = {"mtp_selection": {
        **selection, "cost_path": str(cost_path), "units": len(assignment)},
        "body_bit_accounting": "unchanged"}
    result = backfill_mtp_selection_wires(config, cost_path)
    assert result["__prismaquant__"]["mtp_selection"]["mtp_joint_cost_sha256"] == hashlib.sha256(
        cost_path.read_bytes()).hexdigest()
    assert config["__prismaquant__"]["mtp_selection"] == {
        **selection, "cost_path": str(cost_path), "units": len(assignment)}
    assert result["body.weight"] == config["body.weight"]
    assert result["__prismaquant__"]["body_bit_accounting"] == "unchanged"
    assert set(result["__prismaquant__"]["mtp_selection"]["mtp_expert_wires"]) == set(units)
    assert {name: result[name] for name in assignment} == {
        name: config[name] for name in assignment}
    config[next(iter(units))] = {"format": "changed"}
    with pytest.raises(ValueError, match="config differs"):
        backfill_mtp_selection_wires(config, cost_path)


def test_selected_dense_wire_without_existing_receipt_path_refuses(bound_parts):
    from prismaquant.glm_mtp_selection import merge_mtp_costs, select_mtp_rungs

    parts, _units, _carried = bound_parts
    for part, source in parts:
        part["source_dtype"][DENSE] = "float32"
        source.update(_write(Path(source["path"]), part))
    merged = merge_mtp_costs([part for part, _ in parts],
                             sources=[source for _, source in parts])
    with pytest.raises(ValueError, match="has no mtp_expert_wires"):
        select_mtp_rungs(merged, byte_budget=10**9, constants=CONSTANTS)


@pytest.mark.parametrize("problem", ["missing_m3", "changed_m3", "missing_receipt", "wrong_bytes"])
def test_enrichment_refuses_unbound_or_unpriced_wires(bound_parts, problem):
    from prismaquant.glm_mtp_selection import enrich_mtp_cost_wires, merge_mtp_costs
    parts, units, _carried = bound_parts
    merged = merge_mtp_costs([part for part, _ in parts],
                             sources=[source for _, source in parts])
    source = parts[0][0]["provenance"]["tessera_joint_anchors"]["inputs"]["merged_cost"]
    if problem == "missing_m3":
        source["path"] = str(source["path"] + ".absent")
    elif problem == "changed_m3":
        with open(source["path"], "ab") as handle:
            handle.write(b"changed")
    elif problem in {"missing_receipt", "wrong_bytes"}:
        with open(source["path"], "rb") as handle:
            m3 = pickle.load(handle)
        name = next(iter(units))
        if problem == "missing_receipt":
            del m3[tep.EXPERT_WIRES_KEY][name][FMT]
        else:
            m3["costs"][name][FMT]["wire_bytes"] += 1
        source.update(_write(Path(source["path"]), m3))
    # Refresh the M4 source digest only when its own bound anchor was edited;
    # the M3 mutation itself is what enrichment must catch next.
    if problem != "changed_m3":
        source_path = Path(parts[0][1]["path"])
        parts[0] = (parts[0][0], _write(source_path, parts[0][0]))
        merged = merge_mtp_costs([part for part, _ in parts],
                                 sources=[bound for _, bound in parts])
    with pytest.raises((ValueError, FileNotFoundError)):
        enrich_mtp_cost_wires(merged)


def test_anchor_only_quality_selects_and_exports_the_bound_fractional_wire(tmp_path, monkeypatch):
    from prismaquant import format_registry as registry
    from prismaquant.glm_mtp_selection import merge_mtp_costs, select_mtp_rungs, backfill_mtp_selection_wires
    from canonical_quality_fixtures import scope_for
    from test_canonical_quality_quantity import owner_for_rows
    probe = _probe()
    probe.update(calibration_shape=[512, 512], token_scope="all")
    parts, units, carried = _bound_parts_for_rates(tmp_path, (1024, 768, 896), probe=probe)
    half = "TESSERA_E4M3_K1_R896"
    for index, (part, source) in enumerate(parts):
        if index == 2:
            part["costs"] = {name: {} for name in part["costs"]}
        else:
            fmt = next(iter(next(iter(part["costs"].values()))))
            value = 1.0 if index == 0 else 2.0
            for name in part["costs"]:
                row = _row(name, fmt, [value] * 4, probe)
                row["quality_scope"] = scope_for(row)
                part["costs"][name][fmt] = row
        source.update(_write(Path(source["path"]), part))
    merged = merge_mtp_costs([part for part, _ in parts], sources=[source for _, source in parts])
    assert all(half not in rows for rows in merged["costs"].values())
    owner = owner_for_rows(monkeypatch)
    budget = 16384 + sum(parts[2][0]["wire_bytes"][name][half] for name in units)
    selection = select_mtp_rungs(merged, byte_budget=budget, constants=CONSTANTS,
        formats=[half, "BF16"], rung_allowability={"TESSERA_E4M3_K1": owner})
    assignment = selection.pop("assignment")
    assert {assignment[name] for name in units} == {half}
    assert assignment[DENSE] == "BF16"
    assert selection["resident_bytes"] == budget
    assert selection["E"] == 1.25 * len(units)
    cost_path = tmp_path / "anchor-only.pkl"
    _write(cost_path, merged)
    for key in ("mtp_expert_projection", "mtp_expert_wires", "mtp_expert_wire_roots",
                "mtp_expert_source_bindings", "mtp_expert_wire_binding_schema"):
        selection.pop(key)
    config = {name: registry.get_format(fmt).autoround_config() for name, fmt in assignment.items()}
    config["__prismaquant__"] = {"mtp_selection": {**selection, "cost_path": str(cost_path),
                                                "units": len(assignment)}}
    exported = backfill_mtp_selection_wires(config, cost_path)
    record = exported["__prismaquant__"]["mtp_selection"]
    assert record["mtp_expert_projection"] == carried
    for name in units:
        receipt = record["mtp_expert_wires"][name]
        assert receipt["identity"]["recipe"]["q256"] == 896
        assert receipt["blob_bytes"] == parts[2][0]["wire_bytes"][name][half]
        assert record["mtp_expert_source_bindings"][name]["m4"] == parts[2][1]
