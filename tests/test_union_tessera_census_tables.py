"""Two census tables over one census union into a table every census reader accepts."""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, _load_unit, canonical_json_sha256, unit_path, write_unit,
)
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.tessera_campaign import SCHEMA, ExpertPopulation, campaign_population_block
from prismaquant.tessera_census_cache import (
    census_layer_config, census_selected_cached_units_manifest, load_selected_wire_records,
    seal_roster,
)
from prismaquant.tessera_formats import parse_tessera_format_name
from prismaquant.tessera_hessian import HESSIAN_IDENTITY_FIELDS
from prismaquant.tessera_joint_aura import STAGE
from prismaquant.tessera_menu import assert_uniform_hessian_identity
from test_tessera_expert_projection import STACK, _declared, _projection


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "union_tessera_census_tables.py"
    spec = importlib.util.spec_from_file_location("union_tessera_census_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()

FMT_A = "TESSERA_E4M3_K1_R1024"
FMT_A_UNPRICED = "TESSERA_E4M3_K1_R960"
FMT_B = "TESSERA_E2M1_K2_R896"
DENSE = "model.layers.2.mlp.down_proj"
INPUT_SCHEMA = "tessera.cached_unit_inputs.v1"
ENCODING_SCHEMA = "tessera.encoding_inputs.v1"
ENCODER = "c" * 64
CAPTURE_A = "3" * 64
CAPTURE_B = "4" * 64
TRIPLE = {field: (3 if "tokens" in field else f"{index}" * 64)
          for index, field in enumerate(HESSIAN_IDENTITY_FIELDS, start=5)}


def _hessian(name):
    return {"algorithm": "sha256.dtype_shape_contiguous.v1", "dtype": "torch.float32",
            "sha256": hashlib.sha256(name.encode()).hexdigest(), "shape": [2, 2]}


def _carried():
    projection = _projection(experts=(0,))
    return tep.carried_projection(
        projection, tep.bind_expert_projection(projection, declared=_declared(experts=(0,))),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")


def _manifest_bytes(manifest):
    # The spelling the campaign and dispatch_tessera_campaign.merge_checkpoint write.
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _record(wire, name, fmt, projected):
    family, q256 = parse_tessera_format_name(fmt)
    blob = hashlib.sha256((name + fmt).encode()).digest() * 3
    file = name.replace(".", "__") + "__" + fmt + ".tessera"
    (wire / file).write_bytes(blob)
    identity = {
        "schema": INPUT_SCHEMA if projected is not None else ENCODING_SCHEMA,
        "unit": name, "recipe": {"grid": family.payload_grid().name, "q256": int(q256)},
        "source": {"dtype": "torch.bfloat16", "shape": [2, 2], "sha256": "a" * 64},
        "calibration": {"hessian": _hessian(name)}, "encoder_source_sha256": ENCODER,
    }
    if projected is not None:
        identity["projection"] = {key: projected[key] for key in tep.UNIT_IDENTITY_KEYS}
    return {"file": file, "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "blob_bytes": len(blob), "identity": identity}


BINDING = {"schema": "tessera.canonical_hessian_binding.v1",
           "canonical_capture_sha256": "f" * 64, "census_sha256": "9" * 64}


def _mse(name, fmt):
    # A price is a function of the cell, not of the campaign that measured it.
    return int(hashlib.sha256((name + fmt).encode()).hexdigest()[:6], 16) * 1e-9


def _table(root, *, cells, menus, groups, capture, settings, rate_band, clock,
           carried, experts, all_units, restriction=None, coverage=None, batch=8):
    """One merged census table; ``cells`` is ``{unit: [format, ...]}``."""
    wire = root / "cache" / "wire"
    wire.mkdir(parents=True)
    parts = root / "cost.anchors.json.parts"
    units = sorted(cells)
    records, costs, loo, counts, surfaces, wires, refusals = {}, {}, {}, {}, {}, {}, []
    anchors = {}
    for name in units:
        records[name], costs[name], loo[name], counts[name], surfaces[name] = {}, {}, {}, {}, {}
        anchors[name] = []
        for fmt in cells[name]:
            family = fmt.rsplit("_R", 1)[0]
            q256 = int(fmt.rsplit("_R", 1)[1])
            record = records[name][fmt] = _record(wire, name, fmt, experts.get(name))
            costs[name][fmt] = {
                "output_mse": _mse(name, fmt), "output_mse_measured": True,
                "wire_bytes": record["blob_bytes"], "tessera_family": family,
                "tessera_body_rate_q256": q256, "encode_seconds": clock,
                # The three batch-derived fields exactly as ``tessera_campaign.py``
                # stamps them (:1777-1780): the accounting spelling follows the width.
                "encode_seconds_accounting": ("unit" if batch == 1
                                              else "batch_wall_time_divided_by_batch_size"),
                "encoding_batch_size": batch, "activation_quantized": True,
                "hessian_identity": {"applied": True, "supplied": True, "capture_sha256": capture,
                                     "reference_binding": dict(BINDING), **TRIPLE}}
            loo[name][family] = {"interior_anchors": 0, "max_abs_log2_error": None}
            counts[name][family] = 1
            surfaces[name][family] = {"anchors": 1, "rungs": [q256], "non_interpolable": False,
                                      "encode_seconds": clock}
            refusals.append({"qname": name, "family": family, "reason": "non_interpolable_anchors",
                             "anchor_q256": [q256]})
            anchors[name].append({"qname": name, "family": family, "format_name": fmt,
                                  "body_rate_q256": q256, "dloss": _mse(name, fmt),
                                  "seconds": clock, "encoding_batch_size": batch})
        if name in experts:
            wires[name] = dict(records[name])
    identity = {
        "campaign_schema": "prismaquant.tessera_campaign.v1",
        "currency": "output_mse_under_route_activation_contract",
        "settings": settings, "calibration": {"fit_tokens": 3}, "serving_scope": None,
        "encoder_recipe": {"body": "window"}, "prismaquant_source_sha256": "d" * 64,
        "encoder_source_sha256": ENCODER, "input_global_scale_policy": "static",
        "expert_projection": {"stacks": [STACK]},
        "units": {name: {"weight": {"dtype": "torch.bfloat16", "shape": [2, 2], "sha256": "a" * 64},
                         "hessian": _hessian(name), "scoring_rows": None,
                         "input_global_scale": None, "menu": list(menus[name])} for name in units},
    }
    if restriction is not None:
        identity["family_restriction"] = restriction
    seal = canonical_json_sha256(identity, where="test identity")
    (root / "cost.anchors.json").write_bytes(_manifest_bytes({
        "schema": MANIFEST_SCHEMA, "stage": STAGE, "identity_sha256": seal, "identity": identity,
        "units": [{"qname": name, "file": str(unit_path(parts, name).relative_to(parts))}
                  for name in sorted(units)]}))
    for name in units:
        write_unit(parts, stage=STAGE, qname=name, identity_sha256=seal, state={
            "anchors": anchors[name], "wire_records": dict(records[name])})
    references = root / "cache" / "hessian_capture.references.json"
    references.write_text(json.dumps({
        "schema": "tessera.hessian_capture.references.v1", "capture_sha256": capture,
        "canonical_capture": {"path": "/capture", "sha256": "f" * 64},
        "census": {"path": str(root / "census.json"), "sha256": "9" * 64},
        "counts": {name: 4 for name in all_units},
        "hessians": {name: _hessian(name) for name in units},
        "load_policy": {"schema": "tessera.hessian_reference_load.v1"},
        "provenance": {"model": "m"},
        "rows": [{"capture_sha256": capture, "units": sorted(units)}]}))
    scales = root / "cache" / "input_scales.safetensors"
    scales.write_bytes(b"one static scale file")
    anchor_groups = {"g:dense": [DENSE], "g:stack": sorted(experts)}
    scope = {"dense_targets": [DENSE], "expert_targets": sorted(experts), "dense_all": [DENSE],
             "pinned": [], "declared_stacks": {STACK: {n: list(s) for n, s in
                                                      _declared(experts=(0,))[STACK].items()}},
             "packed_in_scope": {}, "packed_outside_layer_stride": {},
             "anchor_groups": anchor_groups, "calibration_census": {}}
    menu_sizes = {name: len(menus[name]) for name in units}
    provenance = {
        "model": "m", "nsamples": 4, "seqlen": 8, "layer_stride": 1, "max_act_rows": 4,
        "calibration_cache": {"path": "/capture", "sha256": "f" * 64},
        "cost_mode": "production-render-score", "rate_band": rate_band,
        tep.PROJECTION_KEY: carried, "wire_dir": str(wire.resolve()),
        "cache_dir": str((root / "cache").resolve()),
        "hessian": {"calibration_identity": {"model": "m"}, "capture_path": str(references),
                    "capture_sha256": capture, "supplied": True,
                    "reference_binding": dict(BINDING), **TRIPLE},
        "activation_static_scales": {"policy": "p", "source": "s", "path": str(scales),
                                     "units": {name: 1.0 for name in units}},
        "surfaces": surfaces, "anchor_groups": {k: anchor_groups[k] for k in groups},
        "unservable": {},
        "unit_selection": {"schema": "prismaquant.tessera_campaign_units.v1",
                           "selected": coverage is not None,
                           "groups": [{"key": k, "members": anchor_groups[k]} for k in groups]},
        "unit_selection_sample": {"audit_units": [], "inclusion_probability": {}},
        "no_admitted_rung": [], "stopped_early": False, "wall_seconds": 1.5, "rounds_run": 1,
        "campaign_scope": scope, "seed_checkpoint": {"manifest": str(root)},
        "campaign_fanout": {"schema": "prismaquant.tessera_campaign_plan.v1",
                            "rows": {"row-0000": list(groups)}, "seed_checkpoints": []},
        tep.POPULATION_KEY: campaign_population_block(
            dense_targets=[DENSE], expert_targets=sorted(experts), dense_all=[DENSE], pinned=[],
            population=ExpertPopulation(
                members=(), declared={STACK: dict(_declared(experts=(0,))[STACK])},
                packed_in_scope={}, omitted_outside_layer_stride={}),
            layer_stride=1, costs=costs, menus=menu_sizes),
    }
    if restriction is not None:
        provenance["family_restriction"] = restriction
    if coverage is not None:
        provenance["coverage"] = coverage
    formats = sorted({fmt for listed in cells.values() for fmt in listed})
    cost = {"schema": SCHEMA, "currency": "output_mse_under_route_activation_contract",
            "costs": costs, "formats": formats, "leave_one_anchor_out": loo,
            "non_interpolable": refusals, "menu_sizes": menu_sizes, "anchor_counts": counts,
            "provenance": provenance, tep.EXPERT_WIRES_KEY: wires}
    (root / "cost.pkl").write_bytes(pickle.dumps(cost))
    return root


def _census(tmp_path, shape="routed_subset", *, b_clock=22.0, b_batch=8):
    """Table A prices the whole scope; table B is one of two shapes of the E2M1 extension.

    ``routed_subset``: B's plan excluded the dense rows, so B prices only the
    routed stack, and its selection is a strict subset (``selected`` True, a
    ``coverage`` block) -- the merge-side fix for the 42-of-132-group plan.

    ``whole_scope``: B restored the dense rows and re-measured the dense E2M1
    cell A already priced: same price and wire bytes, different clock, a
    different inode; B's selection is the whole scope (``selected`` False).
    """
    carried = _carried()
    _source, experts, _stack_of = tep.carried_units(carried)
    all_units = sorted({DENSE, *experts})
    dense_a = [FMT_A] if shape == "routed_subset" else [FMT_A, FMT_B]
    a = _table(tmp_path / "r1024",
               cells={DENSE: dense_a, **{name: [FMT_A] for name in experts}},
               menus={DENSE: sorted({FMT_A, FMT_A_UNPRICED, *dense_a}),
                      **{name: [FMT_A, FMT_A_UNPRICED] for name in experts}},
               groups=["g:dense", "g:stack"], capture=CAPTURE_A, clock=11.0,
               settings={"nsamples": 4, "rate_band": "1024,1024"}, rate_band=[1024, 1024],
               carried=carried, experts=experts, all_units=all_units)
    b_cells = {name: [FMT_B] for name in experts}
    # The block ``dispatch_tessera_campaign.merge_payloads`` writes for a plan
    # that declared ``dense_rows_excluded`` (PrismaQuant #621, merged 47c5c28934).
    coverage = {"schema": "prismaquant.tessera_campaign_coverage.v1",
                "scope_groups": 2, "priced_groups": 1, "unpriced_groups": ["g:dense"],
                "excluded_rows": ["row-0001"],
                "reason": "the dense cells already exist in table A"}
    if shape == "whole_scope":
        b_cells[DENSE] = [FMT_B]
        coverage = None
    b = _table(tmp_path / "e2m1", cells=b_cells, menus={name: [FMT_B] for name in b_cells},
               groups=["g:stack"] if shape == "routed_subset" else ["g:dense", "g:stack"],
               capture=CAPTURE_B, clock=b_clock, batch=b_batch, coverage=coverage,
               settings={"nsamples": 4, "rate_band": "896,896", "family_restriction": "E2M1_K2"},
               restriction={"policy": "E2M1_K2",
                            "structure_by_unit": {name: ("dense" if name == DENSE else "routed_moe")
                                                  for name in b_cells}},
               rate_band=[896, 896], carried=carried, experts=experts, all_units=all_units)
    return a, b, sorted(experts), all_units


def _run(a, b, out, *extra):
    return tool.main(["--table", f"r1024={a}", "--table", f"e2m1={b}", "--out", str(out),
                      *map(str, extra)])


def _union(tmp_path):
    a, b, experts, all_units = _census(tmp_path)
    out = tmp_path / "union"
    assert _run(a, b, out) == 0
    manifest = json.loads((out / "cost.anchors.json").read_bytes())
    cost = pickle.loads((out / "cost.pkl").read_bytes())
    return a, b, experts, all_units, out, manifest, cost


def _tree(root):
    return {p: p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# (1) the seal, (2) the envelopes
# ---------------------------------------------------------------------------
def test_union_manifest_is_sealed_and_seal_roster_accepts_it(tmp_path):
    a, b, experts, all_units, out, manifest, _cost = _union(tmp_path)
    raw = (out / "cost.anchors.json").read_bytes()
    assert raw == _manifest_bytes(manifest)
    roster = seal_roster(manifest)
    assert roster["identity_sha256"] == canonical_json_sha256(manifest["identity"], where="union")
    assert set(roster["units"]) == set(all_units)
    identity = manifest["identity"]
    assert identity["units"][DENSE]["menu"] == [FMT_A, FMT_A_UNPRICED]
    for name in experts:
        assert identity["units"][name]["menu"] == sorted([FMT_A, FMT_A_UNPRICED, FMT_B])
        assert roster["units"][name] == {"weight": identity["units"][name]["weight"],
                                         "hessian": _hessian(name)}
    # The differing settings live only per table; the shared ones stay readable.
    assert identity["settings"] == {"nsamples": 4}
    assert "family_restriction" not in identity
    union = identity["union"]
    assert union["schema"] == tool.UNION_SCHEMA
    sources = {name: json.loads((root / "cost.anchors.json").read_bytes())
               for name, root in (("r1024", a), ("e2m1", b))}
    for name, source in sources.items():
        assert union["tables"][name]["identity_sha256"] == source["identity_sha256"]
        assert union["tables"][name]["settings"] == source["identity"]["settings"]
    assert union["tables"]["e2m1"]["family_restriction"]["policy"] == "E2M1_K2"
    assert "family_restriction" not in union["tables"]["r1024"]


def test_every_union_envelope_loads_under_the_union_seal(tmp_path):
    _a, _b, experts, all_units, out, manifest, _cost = _union(tmp_path)
    parts = out / "cost.anchors.json.parts"
    assert sorted(p.name for p in (parts / "units").iterdir()) == sorted(
        unit_path(parts, name).name for name in all_units)
    for entry in manifest["units"]:
        state = _load_unit(parts / entry["file"], stage=STAGE, qname=entry["qname"],
                           identity_sha256=manifest["identity_sha256"])
        expected = [FMT_A, FMT_B] if entry["qname"] in experts else [FMT_A]
        assert [anchor["format_name"] for anchor in state["anchors"]] == expected
        assert sorted(state["wire_records"]) == sorted(expected)


# ---------------------------------------------------------------------------
# (3) the selected-wire manifest over a rung from each table
# ---------------------------------------------------------------------------
def test_selected_cache_closes_a_rung_from_each_table_under_the_union_wire_dir(tmp_path):
    a, b, experts, all_units, out, manifest, cost = _union(tmp_path)
    roster = seal_roster(manifest)
    assignment = {DENSE: FMT_A, **{name: FMT_B for name in experts}}
    config = census_layer_config(
        cost, assignment, target_profile="glm_packed_research_sm121",
        entry_for=lambda fmt: {"data_type": "tessera", "tessera_format": fmt})
    metadata = config[LAYER_CONFIG_META_KEY]
    records = load_selected_wire_records(out / "cost.anchors.json.parts", assignment,
                                         identity_sha256=roster["identity_sha256"], workers=2)
    selected = census_selected_cached_units_manifest(
        assignment, metadata, cost, roster, {name: [2, 2] for name in all_units}, records,
        input_schema=INPUT_SCHEMA, encoding_input_schema=ENCODING_SCHEMA,
        cache_schema="tessera.cached_units.v1", blob_workers=2)
    assert set(selected["units"]) == set(all_units)
    wire_dir = out / "cache" / "wire"
    assert cost["provenance"]["wire_dir"] == str(wire_dir.resolve())
    for name, record in records.items():
        blob = wire_dir / record["file"]
        source = (a if assignment[name] == FMT_A else b) / "cache" / "wire" / record["file"]
        assert not blob.is_symlink() and os.path.samefile(blob, source)
    cached_unit = pytest.importorskip("tessera.cached_unit")
    bundle = cached_unit.CachedUnitBundle(selected, wire_dir.resolve(), set(all_units),
                                          selected["source"])
    assert set(bundle.units) == set(all_units)


# ---------------------------------------------------------------------------
# (5) side tables and menu sizes
# ---------------------------------------------------------------------------
def test_cost_side_tables_and_menu_sizes_are_consistent(tmp_path):
    a, b, experts, all_units, out, manifest, cost = _union(tmp_path)
    units = manifest["identity"]["units"]
    assert set(cost["costs"]) == set(units) == set(cost["menu_sizes"])
    for name in all_units:
        # The campaign's meaning: the legal menu, not the priced rungs.
        assert cost["menu_sizes"][name] == len(units[name]["menu"])
    assert cost["menu_sizes"][DENSE] == 2 and len(cost["costs"][DENSE]) == 1
    assert cost["formats"] == sorted([FMT_A, FMT_B])
    for name in experts:
        assert sorted(cost["costs"][name]) == sorted([FMT_A, FMT_B])
        assert cost["anchor_counts"][name] == {"TESSERA_E4M3_K1": 1, "TESSERA_E2M1_K2": 1}
        assert set(cost["leave_one_anchor_out"][name]) == {"TESSERA_E4M3_K1", "TESSERA_E2M1_K2"}
        assert set(cost["provenance"]["surfaces"][name]) == {"TESSERA_E4M3_K1", "TESSERA_E2M1_K2"}
        assert sorted(cost[tep.EXPERT_WIRES_KEY][name]) == sorted([FMT_A, FMT_B])
    assert len(cost["non_interpolable"]) == len(all_units) + len(experts)
    # One Hessian handoff, one capture digest on every row.
    assert {row["hessian_identity"]["capture_sha256"]
            for rows in cost["costs"].values() for row in rows.values()} == {CAPTURE_A}
    assert assert_uniform_hessian_identity(cost["costs"]) is not None
    hessian = cost["provenance"]["hessian"]
    references = out / "cache" / "hessian_capture.references.json"
    assert hessian["capture_sha256"] == CAPTURE_A and hessian["capture_path"] == str(references)
    assert os.path.samefile(references, a / "cache" / "hessian_capture.references.json")
    assert os.path.samefile(out / "cache" / "input_scales.safetensors",
                            a / "cache" / "input_scales.safetensors")
    provenance = cost["provenance"]
    assert provenance["rate_band"] is None and "family_restriction" not in provenance
    assert provenance["activation_static_scales"]["path"] == str(out / "cache" / "input_scales.safetensors")
    assert set(provenance["activation_static_scales"]["units"]) == set(all_units)
    assert set(provenance["campaign_fanout"]["rows"]) == {"r1024/row-0000", "e2m1/row-0000"}
    assert [g["key"] for g in provenance["unit_selection"]["groups"]] == ["g:dense", "g:stack"]
    assert provenance["wall_seconds"] == 3.0
    population = provenance[tep.POPULATION_KEY]
    assert population["priced"]["dense"] == [DENSE]
    assert population["priced"]["routed_experts"] == experts
    assert population["unpriced"] == {"dense": {}, "routed_experts": {}}
    tables = provenance["union"]["tables"]
    assert tables["e2m1"]["provenance"]["rate_band"] == [896, 896]
    assert tables["e2m1"]["provenance"]["family_restriction"]["policy"] == "E2M1_K2"
    assert "tessera_expert_projection" in tables["r1024"]["carried_provenance_sha256"]
    for name, root in (("r1024", a), ("e2m1", b)):
        assert tables[name]["cost_pkl_sha256"] == hashlib.sha256(
            (root / "cost.pkl").read_bytes()).hexdigest()
        assert tables[name]["cost_anchors_json_sha256"] == hashlib.sha256(
            (root / "cost.anchors.json").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# (6) dry run
# ---------------------------------------------------------------------------
def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    a, b, experts, all_units = _census(tmp_path)
    before = _tree(tmp_path)
    out = tmp_path / "union"
    assert _run(a, b, out, "--dry-run") == 0
    assert not out.exists()
    assert _tree(tmp_path) == before
    plan = json.loads(capsys.readouterr().out)
    assert plan["writes"] == 0
    assert plan["tables"]["r1024"]["units"] == len(all_units)
    assert plan["tables"]["e2m1"]["units"] == len(experts)
    assert plan["union"]["units"] == len(all_units)
    assert plan["union"]["rungs_by_family"] == {"TESSERA_E2M1_K2": len(experts),
                                                 "TESSERA_E4M3_K1": len(all_units)}
    assert plan["union"]["wire_blobs"] == len(all_units) + len(experts)
    assert plan["collisions"]["unit_format_collisions"] == 0
    assert plan["collisions"]["wire_names"] == {"identical_inode": 0, "identical_bytes": 0}


def test_the_same_table_twice_is_all_identical_collisions(tmp_path, capsys):
    a, _b, _experts, all_units = _census(tmp_path)
    assert tool.main(["--table", f"one={a}", "--table", f"two={a}", "--out",
                      str(tmp_path / "union"), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["union"]["units"] == len(all_units)
    assert plan["collisions"]["unit_format_collisions"] == len(all_units)
    assert plan["collisions"]["journal_anchor_duplicates"] == len(all_units)
    assert plan["collisions"]["wire_names"]["identical_inode"] == len(all_units)


# ---------------------------------------------------------------------------
# (4) refusals
# ---------------------------------------------------------------------------
def _reseal(root, mutate):
    path = root / "cost.anchors.json"
    manifest = json.loads(path.read_bytes())
    mutate(manifest["identity"])
    manifest["identity_sha256"] = canonical_json_sha256(manifest["identity"], where="test")
    path.write_bytes(_manifest_bytes(manifest))


def _recost(root, mutate):
    path = root / "cost.pkl"
    cost = pickle.loads(path.read_bytes())
    mutate(cost)
    path.write_bytes(pickle.dumps(cost))


def _unit_input(key):
    def mutate(root, experts):
        def change(identity):
            identity["units"][experts[0]][key] = {"sha256": "0" * 64}
        _reseal(root, change)
    return mutate


def _collide_row(root, experts):
    def change(cost):
        row = dict(next(iter(cost["costs"][experts[0]].values())))
        cost["costs"][experts[0]][FMT_A] = {**row, "output_mse": 9.0}
    _recost(root, change)


def _collide_wire(root, experts):
    file = DENSE.replace(".", "__") + "__" + FMT_A + ".tessera"
    (root / "cache" / "wire" / file).write_bytes(b"other bytes under the same wire name")


@pytest.mark.parametrize("mutate,match", [
    (_unit_input("weight"), "bind different weight"),
    (_unit_input("hessian"), "bind different hessian"),
    (_collide_row, rf"\({{unit}}, {FMT_A}\) is priced by tables r1024 and e2m1 with different rows"),
    (_collide_wire, "hold different bytes"),
])
def test_refusals_fire_before_any_write(tmp_path, capsys, mutate, match):
    a, b, experts, _all_units = _census(tmp_path)
    mutate(b, experts)
    out = tmp_path / "union"
    assert _run(a, b, out) == 2
    err = capsys.readouterr().err
    import re
    assert re.search(match.replace("{unit}", re.escape(experts[0])), err), err
    assert not out.exists()


def test_a_failed_hardlink_refuses_and_copies_nothing(tmp_path, capsys, monkeypatch):
    a, b, _experts, _all_units = _census(tmp_path)

    def cross_device(source, target, *args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(tool.os, "link", cross_device)
    out = tmp_path / "union"
    assert _run(a, b, out) == 2
    assert "refusing to copy wires" in capsys.readouterr().err
    assert [p for p in out.rglob("*") if not p.is_dir()] == []


def test_existing_out_needs_resume_and_resume_verifies_every_file(tmp_path, capsys):
    a, b, experts, all_units = _census(tmp_path)
    out = tmp_path / "union"
    assert _run(a, b, out) == 0
    first = _tree(out)
    capsys.readouterr()
    assert _run(a, b, out) == 2
    assert "pass --resume" in capsys.readouterr().err
    assert _run(a, b, out, "--resume") == 0
    written = json.loads(capsys.readouterr().out)["written"]
    assert written["envelopes_kept"] == len(all_units) and written["envelopes_written"] == 0
    assert written["wires_kept"] == len(all_units) + len(experts)
    assert _tree(out) == first
    manifest = json.loads((out / "cost.anchors.json").read_bytes())
    write_unit(out / "cost.anchors.json.parts", stage=STAGE, qname=DENSE,
               identity_sha256=manifest["identity_sha256"], state={"anchors": [], "wire_records": {}})
    assert _run(a, b, out, "--resume") == 2
    assert "holds a different state" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The streamed manifest is the JSON the campaign writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block", [7, 64 << 20])
def test_streamed_manifest_read_digest_and_write_are_the_json_spellings(tmp_path, monkeypatch, block):
    a, _b, _experts, _all_units = _census(tmp_path)
    monkeypatch.setattr(tool, "_READ_BLOCK", block)
    raw = (a / "cost.anchors.json").read_bytes()
    manifest, sha, size = tool.read_manifest(a / "cost.anchors.json", tool.MenuInterner())
    assert manifest == json.loads(raw)
    assert (sha, size) == (hashlib.sha256(raw).hexdigest(), len(raw))
    assert tool.identity_sha256(manifest["identity"]) == manifest["identity_sha256"]
    assert b"".join(tool.manifest_chunks(manifest)) == raw
    compact = tmp_path / "compact.json"
    compact.write_bytes(json.dumps(json.loads(raw)).encode())
    assert tool.read_manifest(compact, tool.MenuInterner())[0] == json.loads(raw)


# ---------------------------------------------------------------------------
# The two shapes table B can take
# ---------------------------------------------------------------------------
def test_routed_subset_table_b_unions_back_to_the_whole_scope(tmp_path):
    # Shape routed_subset: B's selection is a strict subset (selected True, coverage block).
    _a, _b, _experts, _all_units, _out, _manifest, cost = _union(tmp_path)
    provenance = cost["provenance"]
    assert provenance["unit_selection"]["selected"] is False
    assert "coverage" not in provenance
    tables = provenance["union"]["tables"]
    assert tables["e2m1"]["provenance"]["coverage"]["unpriced_groups"] == ["g:dense"]
    assert "coverage" not in tables["r1024"]["provenance"]


def test_whole_scope_table_b_duplicate_cells_agree_when_price_and_bytes_agree(tmp_path, capsys):
    # Shape whole_scope: B re-measured A's dense E2M1 cell; only the clock differs.
    a, b, experts, all_units = _census(tmp_path, "whole_scope")
    out = tmp_path / "union"
    assert _run(a, b, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["collisions"]["unit_format_collisions"] == 1
    assert summary["collisions"]["journal_anchor_duplicates"] == 1
    assert summary["collisions"]["wire_names"] == {"identical_inode": 0, "identical_bytes": 1}
    cost = pickle.loads((out / "cost.pkl").read_bytes())
    assert sorted(cost["costs"][DENSE]) == sorted([FMT_A, FMT_B])
    assert cost["costs"][DENSE][FMT_B]["encode_seconds"] == 11.0  # the first table's row is kept
    assert cost["provenance"]["unit_selection"]["selected"] is False
    manifest = json.loads((out / "cost.anchors.json").read_bytes())
    roster = seal_roster(manifest)
    assignment = {name: FMT_B for name in all_units}
    config = census_layer_config(
        cost, assignment, target_profile="glm_packed_research_sm121",
        entry_for=lambda fmt: {"data_type": "tessera", "tessera_format": fmt})
    records = load_selected_wire_records(out / "cost.anchors.json.parts", assignment,
                                         identity_sha256=roster["identity_sha256"], workers=2)
    selected = census_selected_cached_units_manifest(
        assignment, config[LAYER_CONFIG_META_KEY], cost, roster,
        {name: [2, 2] for name in all_units}, records, input_schema=INPUT_SCHEMA,
        encoding_input_schema=ENCODING_SCHEMA, cache_schema="tessera.cached_units.v1",
        blob_workers=2)
    assert set(selected["units"]) == set(all_units)


def _dense_b_blob(b):
    return b / "cache" / "wire" / (DENSE.replace(".", "__") + "__" + FMT_B + ".tessera")


def test_whole_scope_duplicate_with_a_different_price_refuses(tmp_path, capsys):
    a, b, _experts, _all_units = _census(tmp_path, "whole_scope")

    def change(cost):
        cost["costs"][DENSE][FMT_B]["output_mse"] *= 2
    _recost(b, change)
    assert _run(a, b, tmp_path / "union") == 2
    assert f"({DENSE}, {FMT_B}) is priced by tables r1024 and e2m1 with different rows" \
        in capsys.readouterr().err
    assert not (tmp_path / "union").exists()


def test_whole_scope_duplicate_at_another_anchor_batch_width_merges(tmp_path, capsys):
    # Only the width differs: same clock, same price, same bytes; batch 8 in A, 32 in B.
    a, b, _experts, _all_units = _census(tmp_path, "whole_scope", b_clock=11.0, b_batch=32)
    rows = {name: pickle.loads((root / "cost.pkl").read_bytes())["costs"][DENSE][FMT_B]
            for name, root in (("r1024", a), ("e2m1", b))}
    assert {k for k in rows["r1024"] if rows["r1024"][k] != rows["e2m1"][k]} == {
        "encoding_batch_size", "hessian_identity"}  # the digest is restamped by the union
    out = tmp_path / "union"
    assert _run(a, b, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["collisions"]["unit_format_collisions"] == 1
    assert summary["collisions"]["journal_anchor_duplicates"] == 1
    cost = pickle.loads((out / "cost.pkl").read_bytes())
    assert cost["costs"][DENSE][FMT_B]["encoding_batch_size"] == 8  # the first table's row is kept


def test_whole_scope_duplicate_priced_at_batch_1_and_batch_32_merges(tmp_path, capsys):
    # Every batch-derived field differs at once -- ``encode_seconds`` (the batch
    # wall time divided by the width), ``encode_seconds_accounting`` ("unit"
    # against "batch_wall_time_divided_by_batch_size") and
    # ``encoding_batch_size`` -- and the journal anchors' ``seconds`` with them.
    # Same price, same bytes: one cell, one price.  This is the shape the
    # census takes when its remaining rows move to a wider encoder batch.
    a, b, _experts, _all_units = _census(tmp_path, "whole_scope", b_clock=11.0 / 32, b_batch=32)

    def unit_width(cost):
        row = cost["costs"][DENSE][FMT_B]
        row["encoding_batch_size"], row["encode_seconds_accounting"] = 1, "unit"
    _recost(a, unit_width)
    rows = {name: pickle.loads((root / "cost.pkl").read_bytes())["costs"][DENSE][FMT_B]
            for name, root in (("r1024", a), ("e2m1", b))}
    assert {k for k in rows["r1024"] if rows["r1024"][k] != rows["e2m1"][k]} == {
        "encode_seconds", "encode_seconds_accounting", "encoding_batch_size",
        "hessian_identity"}  # the digest is restamped by the union
    out = tmp_path / "union"
    assert _run(a, b, out) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["collisions"]["unit_format_collisions"] == 1
    assert summary["collisions"]["journal_anchor_duplicates"] == 1
    kept = pickle.loads((out / "cost.pkl").read_bytes())["costs"][DENSE][FMT_B]
    assert (kept["encoding_batch_size"], kept["encode_seconds_accounting"],
            kept["encode_seconds"]) == (1, "unit", 11.0)  # the first table's row is kept


def test_width_equivalence_does_not_extend_to_other_row_fields(tmp_path, capsys):
    # Same width-only setup, plus one non-procedure field that differs: refused.
    a, b, _experts, _all_units = _census(tmp_path, "whole_scope", b_clock=11.0, b_batch=32)

    def change(cost):
        cost["costs"][DENSE][FMT_B]["activation_quantized"] = False
    _recost(b, change)
    assert _run(a, b, tmp_path / "union") == 2
    assert f"({DENSE}, {FMT_B}) is priced by tables r1024 and e2m1 with different rows" \
        in capsys.readouterr().err
    assert not (tmp_path / "union").exists()


def test_whole_scope_duplicate_with_different_wire_bytes_refuses(tmp_path, capsys):
    a, b, _experts, _all_units = _census(tmp_path, "whole_scope")
    blob = _dense_b_blob(b)
    blob.write_bytes(bytes(len(blob.read_bytes())))  # same size, same receipt, other bytes
    assert _run(a, b, tmp_path / "union") == 2
    assert "hold different bytes" in capsys.readouterr().err
    assert not (tmp_path / "union").exists()


def test_tables_must_bind_one_canonical_capture(tmp_path, capsys):
    a, b, _experts, _all_units = _census(tmp_path)
    other = {**BINDING, "canonical_capture_sha256": "e" * 64}

    def change(cost):
        cost["provenance"]["hessian"]["reference_binding"] = dict(other)
        for rows in cost["costs"].values():
            for row in rows.values():
                row["hessian_identity"]["reference_binding"] = dict(other)
    _recost(b, change)
    assert _run(a, b, tmp_path / "union") == 2
    assert "does not name its handoff's canonical capture" in capsys.readouterr().err
