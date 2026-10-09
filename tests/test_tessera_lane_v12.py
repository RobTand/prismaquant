"""Lane schema v12 parses the authoritative v66 launches and scopes each rung.

The table below is the two routed ``resident`` cells of Tessera contract v66
(branch ``kernels/t8-routed-cells-20261007``, PR #1045): the three historical
launches cover census rungs 832-1088, the class decoder covers 768 alone.
No pin moves here: the serving and export pins stay at lane schema v11.
"""
import copy
import json
from pathlib import Path

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_runtime_contract as runtime

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tessera_lane_v12_v66_routed.json"

CLASS_DECODER = "native_routed_window_classes_e4m3mma"
HISTORICAL = ("native_window_moe_compact", "native_routed_fused_window",
              "native_routed_fused_window_e4m3mma")
HISTORICAL_RUNGS = (832, 864, 896, 928, 944, 960, 1024, 1088)


def _payload():
    return json.loads(FIXTURE.read_bytes())


def _table(payload=None):
    payload = _payload() if payload is None else payload
    return lane._parse_table(payload["lane_eligibility"], payload["formats"],
                             "fixture", "fixture", "fixture",
                             native_extensions=payload["native_extensions"])


def _parsed(payload=None):
    payload = _payload() if payload is None else payload
    return runtime._parse(payload, commit="fixture", sha="fixture", path="fixture")


def test_v12_fixture_is_the_authoritative_v66_pair():
    payload = _payload()
    assert payload["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v12"
    assert payload["contract_version"] == 66
    cells = payload["lane_eligibility"]["cells"]
    assert {c["id"] for c in cells} == {
        "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
        "tessera_e4m3_k1_routed_moe_sm121_batch_resident"}
    for cell in cells:
        assert cell["rungs_q256"] == [768] + list(HISTORICAL_RUNGS)
        scopes = {e["decoder"]: e.get("rungs_q256") for e in cell["executes"]}
        assert scopes[CLASS_DECODER] == [768]
        for decoder in HISTORICAL:
            assert scopes[decoder] == list(HISTORICAL_RUNGS)


def test_v12_parses_and_keeps_the_launch_scope_on_the_record():
    table = _table()
    assert table.schema == "tessera.lane-eligibility.v12"
    for cell in table.cells:
        assert len(cell.executes) == 4
        assert len(cell.launch_rungs_q256) == 4
        scopes = dict(zip([d for _, d in cell.executes], cell.launch_rungs_q256))
        assert scopes[CLASS_DECODER] == (768,)
        for decoder in HISTORICAL:
            assert scopes[decoder] == HISTORICAL_RUNGS
    parsed = _parsed()
    assert parsed.lane_schema == "tessera.lane-eligibility.v12"
    route = next(c for c in parsed.cells
                 if c.cell_id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    assert dict(zip([d for _, d in route.executes],
                    route.launch_rungs_q256))[CLASS_DECODER] == (768,)


def test_v12_r768_selects_only_the_class_decoder():
    table = _table()
    cell = next(c for c in table.cells
                if c.id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    for index, (_, decoder) in enumerate(cell.executes):
        assert cell.launch_covers_rate(index, 768) == (decoder == CLASS_DECODER)
    scoped = [pair for index, pair in enumerate(cell.executes)
              if cell.launch_covers_rate(index, 768)]
    assert {decoder for _, decoder in scoped} == {CLASS_DECODER}


@pytest.mark.parametrize("rung", list(HISTORICAL_RUNGS))
def test_v12_listed_r832_to_r1088_select_only_historical_decoders(rung):
    table = _table()
    cell = next(c for c in table.cells
                if c.id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    for index, (_, decoder) in enumerate(cell.executes):
        assert cell.launch_covers_rate(index, rung) == (decoder in HISTORICAL)
    scoped = [pair for index, pair in enumerate(cell.executes)
              if cell.launch_covers_rate(index, rung)]
    assert {decoder for _, decoder in scoped} == set(HISTORICAL)


@pytest.mark.parametrize("cell_id", [
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident"])
def test_v12_batch_cell_scopes_match_the_decode_cell(cell_id):
    table = _table()
    cell = next(c for c in table.cells if c.id == cell_id)
    scoped_768 = {decoder for index, (_, decoder) in enumerate(cell.executes)
                  if cell.launch_covers_rate(index, 768)}
    assert scoped_768 == {CLASS_DECODER}
    scoped_896 = {decoder for index, (_, decoder) in enumerate(cell.executes)
                  if cell.launch_covers_rate(index, 896)}
    assert scoped_896 == set(HISTORICAL)


def test_v12_rung_launches_narrow_by_scope_before_the_lane_gate():
    """A lenient reader admits the wrong decoder; this one does not.

    The scope narrows first, so the lane gate never sees an out-of-scope
    launch: at R768 only the class launch is left, at R832 only the three
    historical ones. The test calls the gate (not only the scope
    predicate), so a fix that parses the key and ignores its meaning fails
    here. It needs the v66 serving runtime for the lane predicate; the
    scope-only tests above run on any runtime.
    """
    tessera_export = pytest.importorskip(
        "tessera.export", reason="needs the v66 serving runtime")
    if not hasattr(tessera_export, "served_recipe"):
        pytest.skip("needs the v66 serving runtime")
    table = _table()
    cell = next(c for c in table.cells
                if c.id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    admits, why, launches = lane.cell_rung_launches(cell, 768, table.lanes)
    assert admits, why
    assert {decoder for _, decoder in launches} == {CLASS_DECODER}
    admits, why, launches = lane.cell_rung_launches(cell, 896, table.lanes)
    assert admits, (896, why)
    assert {decoder for _, decoder in launches} == set(HISTORICAL)


def test_v12_unscoped_launch_keeps_the_scope_of_its_cell():
    payload = _payload()
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    del cell["executes"][0]["rungs_q256"]
    table = _table(payload)
    parsed_cell = next(c for c in table.cells if c.id == cell["id"])
    assert parsed_cell.launch_rungs_q256[0] is None
    assert parsed_cell.launch_covers_rate(0, 768)
    assert parsed_cell.launch_covers_rate(0, 832)

def test_v12_route_cell_keeps_the_derived_launch_coverage():
    parsed = _parsed()
    table = _table()
    route = next(c for c in parsed.cells
                 if c.cell_id == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    cell = next(c for c in table.cells if c.id == route.cell_id)
    assert tuple(route.launch_covered_rungs_q256) == tuple(
        cell.launch_covered_rungs_q256)
    index = next(i for i, (_, decoder) in enumerate(route.executes)
                 if decoder in HISTORICAL)
    assert 800 in route.launch_covered_rungs_q256[index]
    assert route.launch_covers_rate(index, 800)
    assert route.launch_covers_rate(index, 832)
    class_index = next(i for i, (_, decoder) in enumerate(route.executes)
                       if decoder == CLASS_DECODER)
    assert not route.launch_covers_rate(class_index, 800)


def test_v12_answer_keeps_v11_row_shape_without_a_scope():
    payload = _payload()
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    for cell in payload["lane_eligibility"]["cells"]:
        for launch in cell["executes"]:
            launch.pop("rungs_q256", None)
    before = runtime.contract_answer(_parsed())
    answer = runtime.contract_answer(runtime._parse(
        payload, commit="fixture", sha="fixture", path="fixture"))
    assert runtime._answer_drift(before, before) == []
    row = next(r for r in answer["cells"]
               if r[0] == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    assert len(row) == 18
    assert isinstance(row[13], dict) and set(row[13]) >= {
        "image", "execution_modes"}
    assert "launch_scopes" not in row


@pytest.mark.parametrize("mutation,match", [
    (lambda e: e.update(rungs_q256=[]), "at least one rung"),
    (lambda e: e.update(rungs_q256=[832, 768]), "ascending order"),
    (lambda e: e.update(rungs_q256=[832, 832]), "repeat a rung"),
    (lambda e: e.update(rungs_q256=[True]), "integer rung"),
    (lambda e: e.update(rungs_q256=[[832]]), "integer rung"),
    (lambda e: e.update(rungs_q256=[640]), "does not publish"),
    (lambda e: e.update(rungs_q256="832"), "JSON array"),
])
def test_v12_refuses_a_malformed_launch_scope(mutation, match):
    payload = _payload()
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    mutation(cell["executes"][0])
    with pytest.raises(lane.LaneEligibilityError, match=match):
        _table(payload)


def test_v12_key_is_refused_under_v11():
    payload = _payload()
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    with pytest.raises(lane.LaneEligibilityError, match="rungs_q256"):
        _table(payload)


def test_v12_scope_change_moves_the_contract_answer():
    before = _parsed()
    answer = runtime.contract_answer(before)
    row = next(r for r in answer["cells"]
               if r[0] == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    entry = next(e for e in row[13]["launch_scopes"]
                 if e["decoder"] == CLASS_DECODER)
    assert entry["rungs_q256"] == [768]
    payload = _payload()
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_routed_moe_sm121_decode_resident")
    scoped = next(e for e in cell["executes"] if e["decoder"] == CLASS_DECODER)
    scoped["rungs_q256"] = [768, 832]
    after = _parsed(payload)
    drift = runtime._answer_drift(answer, runtime.contract_answer(after))
    assert any("tessera_e4m3_k1_routed_moe_sm121_decode_resident" in line
               for line in drift)


def test_v12_keeps_the_pin_at_v11():
    assert runtime.TESSERA_DEV_PIN_ANSWER["lane_schema"] == (
        "tessera.lane-eligibility.v11")
