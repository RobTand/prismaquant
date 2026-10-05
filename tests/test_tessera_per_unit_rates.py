"""PrismaQuant #2319: the allocator spends the byte budget at unit granularity.

The price surface is per routed unit (the corrected T-8 derivation: 11,232
priced R1024->R1088 rows for the 13 up-only layers, 262,144 B per upgrade);
the export path used to express picks whole per layer stack only.  These tests
pin the consumer slice: the priced per-unit upgrade allocator under a hard
serialized-byte cap, the v57 per-unit capability block of the installed Tessera
contract, the complete per-unit assignment owner that replaces the stack
uniform restriction, the plan writer's ``unit_q256`` overrides, and the exact
per-unit receipts the allocation and the export lane carry.  Stack-uniform
worlds must keep every spelling they had.

Fixture: ``tests/fixtures/pq2319-predicted-table-units.csv`` is the real
corrected-derivation price table (byte-identical copy; sha256 pinned below),
so the 718-pick outcome is the derivation's own arithmetic, not a synthetic
stand-in.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

from prismaquant import tessera_census_cache as census
from prismaquant import tessera_expert_projection as tep
from prismaquant import tessera_export_lane as export
from prismaquant import tessera_plan_writer as writer
from prismaquant import tessera_runtime_contract as trc
from prismaquant.tessera_expert_projection import (
    ExpertProjectionError,
    bind_expert_projection,
    carried_projection,
    carried_units,
    stack_plan_request,
)
from test_allocator_expert_projection import (
    FMT as ALLOC_FMT,
    DENSE as ALLOC_DENSE,
    STACK as ALLOC_STACK,
    _allocator_argv as _alloc_argv,
    _assignment as _alloc_assignment,
    _cost_payload as _alloc_cost_payload,
    _units as _alloc_units,
    _v5_contract as _alloc_v5_contract,
)
from test_allocator_byte_budget_selection import _write_safetensors
from test_tessera_census_cache import FMT as CENSUS_FMT, _world as _census_world
from test_tessera_export_projection import (
    _blob as _lane_blob,
    _carried as _lane_carried,
    _receipt as _lane_receipt,
    _units as _lane_units,
    case,  # noqa: F401  -- pytest fixture
)
from test_tessera_expert_projection import STACK, _declared, _projection
from test_tessera_plan_writer import (
    STACK as WRITER_STACK,
    STACK_TENSORS,
    _context,
    _stack_surface,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pq2319-predicted-table-units.csv"
FIXTURE_SHA256 = "43144ce99456223a0a3acf098e7fb47563a53fe0c92a5dee64fdb9d204be229c"
FIXTURE_SOURCE = ("/mnt/shared/tessera-measurements/t8-corrected-alloc-20261005"
                  "/derivation/predicted-table-units.csv")
BASE_FMT = "TESSERA_E4M3_K1_R1024"
UP_FMT = "TESSERA_E4M3_K1_R1088"
HEADROOM_BYTES = 188_331_767
UNIT_DELTA_BYTES = 262_144
EXPECTED_PICKS = 718
EXPECTED_SPENT = EXPECTED_PICKS * UNIT_DELTA_BYTES        # 188,219,392
EXPECTED_LEFT = HEADROOM_BYTES - EXPECTED_SPENT           # 112,375


# ---------------------------------------------------------------------------
# The real price table
# ---------------------------------------------------------------------------
def _fixture_rows() -> list[dict]:
    with FIXTURE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fixture_costs() -> dict:
    """The campaign-table shape the allocator consumes, from the real rows.

    Only the price deltas are load-bearing: the absolute wire bytes are set to
    the R1024 rate the CSV's own ``n_params`` column implies (q256/8 bytes per
    parameter), and the R1088 row adds the row's real ``byte_delta``, so every
    upgrade costs exactly the priced 262,144 B wire delta.
    """
    costs: dict[str, dict] = {}
    for row in _fixture_rows():
        assert row["from_format"] == BASE_FMT and row["to_format"] == UP_FMT
        base = int(row["n_params"]) // 2
        costs[row["unit"]] = {
            BASE_FMT: {"predicted_dloss": float(row["predicted_dloss_r1024"]),
                       "wire_bytes": base},
            UP_FMT: {"predicted_dloss": float(row["predicted_dloss_r1088"]),
                     "wire_bytes": base + int(row["byte_delta"])},
        }
    return costs


# ---------------------------------------------------------------------------
# The priced per-unit upgrade allocator
# ---------------------------------------------------------------------------
def test_allocator_fills_188331767_headroom_with_718_priced_upgrades():
    # The fixture is the real corrected-derivation price table, and the
    # provenance assertions live here so this test proves them only with the
    # allocator that consumes them (it fails whole on the pre-#2319 code).
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    rows = _fixture_rows()
    assert len(rows) == 11_232
    assert list(rows[0]) == ["unit", "layer", "expert", "role", "from_format",
                             "to_format", "n_params", "byte_delta",
                             "predicted_dloss_r1024", "predicted_dloss_r1088",
                             "predicted_dloss_delta"]
    assert {row["byte_delta"] for row in rows} == {str(UNIT_DELTA_BYTES)}
    assert len({row["unit"] for row in rows}) == len(rows)
    assert FIXTURE_SOURCE.startswith("/mnt/shared/tessera-measurements/")
    assert raw.startswith(b"unit,layer,expert,role")
    costs = _fixture_costs()
    assignment = {unit: BASE_FMT for unit in costs}
    picks, record = tep.select_priced_unit_upgrades(
        costs, assignment, byte_budget=HEADROOM_BYTES)
    assert len(picks) == EXPECTED_PICKS
    assert set(picks.values()) == {UP_FMT}
    assert record["byte_budget"] == HEADROOM_BYTES
    assert record["reserve_bytes"] == 0
    assert record["spent_wire_delta_bytes"] == EXPECTED_SPENT
    assert record["remaining_wire_delta_bytes"] == EXPECTED_LEFT
    assert EXPECTED_LEFT < UNIT_DELTA_BYTES, "the cap is filled to within one unit's cost"
    # Independent expectation: the 718 cheapest priced rows by quality price
    # per added wire byte, ties by unit -- the derivation's own priced order.
    priced = sorted((float(row["predicted_dloss_delta"]), row["unit"])
                    for row in _fixture_rows())
    expected = {unit for _price, unit in priced[:EXPECTED_PICKS]}
    assert set(picks) == expected
    picked_prices = [price for price, unit in priced[:EXPECTED_PICKS]]
    unpicked_prices = [price for price, unit in priced[EXPECTED_PICKS:]]
    assert max(picked_prices) < min(unpicked_prices), "no tie straddles the cut"
    assert record["whole_artifact_bytes_claimed"] is False
    assert record["currency"] == "price_row_wire_delta_bytes"


def test_allocator_is_deterministic_across_row_insertion_order():
    costs = _fixture_costs()
    shuffled = {}
    keys = list(costs)
    random.Random(2319).shuffle(keys)
    for key in keys:
        menus = list(costs[key].items())
        random.Random(hash(key) & 0xFFFF).shuffle(menus)
        shuffled[key] = dict(menus)
    assignment = {unit: BASE_FMT for unit in costs}
    picks, record = tep.select_priced_unit_upgrades(
        costs, assignment, byte_budget=HEADROOM_BYTES)
    shuffled_picks, shuffled_record = tep.select_priced_unit_upgrades(
        shuffled, assignment, byte_budget=HEADROOM_BYTES)
    assert shuffled_picks == picks
    assert shuffled_record == record


def test_allocator_honors_an_explicit_reserve_off_the_hard_cap():
    costs = _fixture_costs()
    assignment = {unit: BASE_FMT for unit in costs}
    picks, record = tep.select_priced_unit_upgrades(
        costs, assignment, byte_budget=HEADROOM_BYTES, reserve_bytes=EXPECTED_LEFT)
    assert len(picks) == EXPECTED_PICKS
    assert record["spend_cap_bytes"] == EXPECTED_SPENT
    assert record["remaining_wire_delta_bytes"] == 0
    tight, tight_record = tep.select_priced_unit_upgrades(
        costs, assignment, byte_budget=HEADROOM_BYTES, reserve_bytes=EXPECTED_LEFT + 1)
    assert len(tight) == EXPECTED_PICKS - 1
    assert tight_record["spent_wire_delta_bytes"] == (EXPECTED_PICKS - 1) * UNIT_DELTA_BYTES
    with pytest.raises(ExpertProjectionError, match="reserve"):
        tep.select_priced_unit_upgrades(costs, assignment, byte_budget=HEADROOM_BYTES,
                                        reserve_bytes=HEADROOM_BYTES + 1)
    with pytest.raises(ExpertProjectionError, match="reserve"):
        tep.select_priced_unit_upgrades(costs, assignment, byte_budget=HEADROOM_BYTES,
                                        reserve_bytes=-1)


def test_allocator_stops_at_the_infeasible_floor_without_inventing_a_pick():
    costs = _fixture_costs()
    assignment = {unit: BASE_FMT for unit in costs}
    for budget in (0, UNIT_DELTA_BYTES - 1):
        picks, record = tep.select_priced_unit_upgrades(costs, assignment,
                                                        byte_budget=budget)
        assert picks == {}
        assert record["spent_wire_delta_bytes"] == 0
        assert record["remaining_wire_delta_bytes"] == budget
    # A world with no priced upgrade rows at all buys nothing either.
    stock, stock_record = tep.select_priced_unit_upgrades({}, {}, byte_budget=1 << 30)
    assert stock == {} and stock_record["spent_wire_delta_bytes"] == 0


def test_allocator_refuses_rows_without_exact_price_or_wire_bytes():
    costs = _fixture_costs()
    unit = next(iter(costs))
    assignment = {u: BASE_FMT for u in costs}
    # Present-but-corrupt prices refuse: a row that claims a field this
    # allocator spends must not be guessed into one it can trust.
    for damage in (
        lambda c: c[unit][UP_FMT].update(wire_bytes="262144.5"),
        lambda c: c[unit][UP_FMT].update(predicted_dloss=None),
        lambda c: c[unit].update({UP_FMT: "not a row"}),
    ):
        damaged = json.loads(json.dumps(costs))
        damage(damaged)
        with pytest.raises(ExpertProjectionError, match=unit):
            tep.select_priced_unit_upgrades(damaged, assignment, byte_budget=1 << 30)
    # Absent price fields are the sampled/stack-only case: the unit stays
    # grouped and the rest of the table still fills the cap.
    for damage in (
        lambda c: c[unit][UP_FMT].pop("wire_bytes"),
        lambda c: c[unit][UP_FMT].pop("predicted_dloss"),
        lambda c: c[unit][BASE_FMT].pop("wire_bytes"),
        lambda c: c[unit].update({BASE_FMT: {}}),
        lambda c: c.pop(unit),
    ):
        damaged = json.loads(json.dumps(costs))
        damage(damaged)
        picks, _record = tep.select_priced_unit_upgrades(
            damaged, assignment, byte_budget=HEADROOM_BYTES)
        assert unit not in picks
        assert len(picks) == EXPECTED_PICKS


def test_allocator_refuses_non_integer_budget_or_reserve():
    costs = _fixture_costs()
    assignment = {unit: BASE_FMT for unit in costs}
    for budget in (-1, float(HEADROOM_BYTES), True, "188331767"):
        with pytest.raises(ExpertProjectionError, match="byte_budget"):
            tep.select_priced_unit_upgrades(costs, assignment, byte_budget=budget)
    with pytest.raises(ExpertProjectionError, match="reserve_bytes"):
        tep.select_priced_unit_upgrades(costs, assignment, byte_budget=HEADROOM_BYTES,
                                        reserve_bytes=1.5)


def test_allocator_climbs_multi_rung_ladders_until_no_single_upgrade_fits():
    # A: R1024 -100B-> R768 (best price), then R768 -300B-> R1088.
    # B: R1024 -150B-> R1088 at a worse price.
    costs = {
        "s.0.w1": {BASE_FMT: {"predicted_dloss": 9.0, "wire_bytes": 1000},
                   "TESSERA_E4M3_K1_R768": {"predicted_dloss": 4.0, "wire_bytes": 1100},
                   UP_FMT: {"predicted_dloss": 3.0, "wire_bytes": 1400}},
        "s.0.w2": {BASE_FMT: {"predicted_dloss": 5.0, "wire_bytes": 1000},
                   UP_FMT: {"predicted_dloss": 2.0, "wire_bytes": 1150}},
    }
    assignment = {"s.0.w1": BASE_FMT, "s.0.w2": BASE_FMT}
    picks, record = tep.select_priced_unit_upgrades(costs, assignment, byte_budget=550)
    assert picks == {"s.0.w1": UP_FMT, "s.0.w2": UP_FMT}
    assert record["spent_wire_delta_bytes"] == 550
    assert record["remaining_wire_delta_bytes"] == 0
    # One byte short of the second step: A stops at R768 and nothing is invented.
    picks, record = tep.select_priced_unit_upgrades(costs, assignment, byte_budget=549)
    assert picks == {"s.0.w1": "TESSERA_E4M3_K1_R768", "s.0.w2": UP_FMT}
    assert record["spent_wire_delta_bytes"] == 250
    assert record["remaining_wire_delta_bytes"] == 299
    # Downgrades and same-rung rows are not upgrades and are never bought.
    costs["s.0.w3"] = {BASE_FMT: {"predicted_dloss": 1.0, "wire_bytes": 2000},
                       UP_FMT: {"predicted_dloss": 0.5, "wire_bytes": 1000}}
    picks, _record = tep.select_priced_unit_upgrades(
        costs, assignment, byte_budget=1 << 30)
    assert "s.0.w3" not in picks


def test_allocator_keeps_stack_only_units_grouped_and_refuses_cross_grid_rows():
    costs = _fixture_costs()
    unit = next(iter(costs))
    other = "model.language_model.layers.4.mlp.experts.0.down_proj"
    assignment = {u: BASE_FMT for u in costs}
    assignment[other] = BASE_FMT
    # A unit priced only at another grid is not an upgrade of this stack's grid.
    costs[other] = {"TESSERA_E2M1x2_K2_R896": {"predicted_dloss": -1.0, "wire_bytes": 1},
                    BASE_FMT: {"predicted_dloss": 0.0, "wire_bytes": 4_000_000}}
    picks, record = tep.select_priced_unit_upgrades(costs, assignment,
                                                    byte_budget=HEADROOM_BYTES)
    assert other not in picks
    assert len(picks) == EXPECTED_PICKS
    # A BF16 baseline has no priced wire row to upgrade from: left grouped.
    bf16_unit = "model.language_model.layers.4.mlp.experts.1.up_proj"
    assignment[bf16_unit] = "BF16"
    costs[bf16_unit] = {UP_FMT: {"predicted_dloss": -1.0, "wire_bytes": 4_000_000}}
    picks, _record = tep.select_priced_unit_upgrades(costs, assignment,
                                                     byte_budget=HEADROOM_BYTES)
    assert bf16_unit not in picks


# ---------------------------------------------------------------------------
# The installed contract's per-unit capability block (v57)
# ---------------------------------------------------------------------------
def _capability_contract(tmp_path: Path, *, version=57, block=None,
                         producer_interface=None) -> Path:
    payload = {"schema": trc.TESSERA_CONTRACT_SCHEMA, "contract_version": version}
    if producer_interface is not None:
        payload["producer_interface"] = producer_interface
    elif block is not None:
        payload["producer_interface"] = {"routed_units": block}
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(payload))
    return path


def test_packaged_capability_reader_requires_v57_and_the_exact_block(tmp_path, monkeypatch):
    exact = {"schema": "tessera.routed-unit-assignment.v1",
             "plannable_unit": "expert_projection",
             "plan_field": "unit_q256",
             "q256_spelling": "int_or_per_role_or_expert_role_matrix",
             "production_admission": "requires_lane_qualification"}
    assert trc.ROUTED_UNIT_ASSIGNMENT_BLOCK == exact
    assert trc.ROUTED_UNIT_ASSIGNMENT_CONTRACT_VERSION == 57

    def read(path):
        monkeypatch.setattr(trc, "contract_path", lambda: path)
        return trc.packaged_routed_unit_capability()

    sha, block = read(_capability_contract(tmp_path))
    assert block == exact
    assert len(sha) == 64
    # The block is checked field for field: a renamed or reworded field is not
    # the capability the loader published.
    for key, value in (("plan_field", "unit_rates"), ("plannable_unit", "stack"),
                       ("q256_spelling", "int"), ("production_admission", "admitted"),
                       ("schema", "tessera.routed-unit-assignment.v2")):
        damaged = dict(exact, **{key: value})
        with pytest.raises(trc.TesseraContractError, match=key):
            read(_capability_contract(tmp_path, block=damaged))
    with pytest.raises(trc.TesseraContractError, match="v57"):
        read(_capability_contract(tmp_path, version=56, block=exact))
    with pytest.raises(trc.TesseraContractError, match="v57"):
        read(_capability_contract(tmp_path, version=45))
    with pytest.raises(trc.TesseraContractError, match="v57"):
        read(_capability_contract(tmp_path, producer_interface={}))
    with pytest.raises(trc.TesseraContractError, match="v57"):
        read(tmp_path / "absent.json")


def test_mixed_stack_without_capability_refuses_by_unit_stack_and_v57():
    _source, units, stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    uniform = {name: BASE_FMT for name in units}
    mixed = dict(uniform)
    mixed[f"{STACK}.0.w2"] = "TESSERA_E4M3_K1_R768"
    with pytest.raises(ExpertProjectionError,
                       match=rf"{STACK}.*0\.w2.*differ.*v57"):
        tep.require_unit_assignment(mixed, stack_of, units, capability=None)
    # The installed runtime is consulted only when a stack is actually mixed:
    # a uniform world never resolves the capability and never imports tessera.
    stack_formats, unit_rungs = tep.require_unit_assignment(uniform, stack_of, units)
    assert stack_formats == {STACK: BASE_FMT} and unit_rungs == {}


def test_mixed_bf16_assignment_uses_the_published_window_capability():
    _source, units, stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("BF16", 1024)}), tool="t"))
    selected = {name: "TESSERA_BF16_K1_R1024" for name in units}
    selected[f"{STACK}.0.w2"] = "TESSERA_BF16_K1_R1088"
    stacks, per_unit = tep.require_unit_assignment(
        selected, stack_of, units, capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))
    assert stacks == {}
    assert per_unit == {STACK: selected}

def test_uniform_assignment_keeps_the_stack_uniform_stamps_and_emits_no_unit_map():
    _source, units, stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    uniform = {name: BASE_FMT for name in units}
    stack_formats, unit_rungs = tep.require_unit_assignment(
        uniform, stack_of, units, capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))
    assert stack_formats == {STACK: BASE_FMT}
    assert unit_rungs == {}
    # Partial stacks and unprojected units refuse by name exactly as before:
    # the plan still serves a stack whole.
    partial = dict(uniform)
    partial.pop(f"{STACK}.1.w3")
    with pytest.raises(ExpertProjectionError, match=r"executes the stack whole.*1\.w3"):
        tep.require_unit_assignment(partial, stack_of, units,
                                    capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))
    with pytest.raises(ExpertProjectionError, match="not in the carried producer projection"):
        tep.require_unit_assignment(
            {**uniform, "model.layers.6.feed_forward.experts.0.w1": BASE_FMT},
            stack_of, units, capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))


# ---------------------------------------------------------------------------
# The allocation block: per-unit rungs beside the stack formats
# ---------------------------------------------------------------------------
def test_mixed_allocation_block_emits_per_unit_rungs_and_exact_receipts(
        tmp_path, monkeypatch):
    monkeypatch.setattr(trc, "packaged_routed_unit_capability",
                        lambda: ("d" * 64, dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK)))
    payload = _alloc_cost_payload(tmp_path, formats=(ALLOC_FMT, "TESSERA_E4M3_K1_R768"))
    mixed = _alloc_assignment()
    mixed[f"{ALLOC_STACK}.0.w2"] = "TESSERA_E4M3_K1_R768"
    block = tep.allocation_expert_projection_block(payload, mixed)
    assert block["tessera_expert_stack_formats"] == {}
    assert block[tep.UNIT_RUNGS_KEY] == {
        "schema": tep.UNIT_RUNGS_SCHEMA,
        "stacks": {ALLOC_STACK: {name: mixed[name] for name in _alloc_units()}},
    }
    for name in _alloc_units():
        assert block[tep.EXPERT_WIRES_KEY][name] == \
            payload[tep.EXPERT_WIRES_KEY][name][mixed[name]]
    # Receipt correctness survives per-unit selection: a receipt sealed for
    # another rung but filed under the selected one is still refused.
    swapped = json.loads(json.dumps(payload))
    swapped[tep.EXPERT_WIRES_KEY][f"{ALLOC_STACK}.0.w1"]["TESSERA_E4M3_K1_R768"] = \
        payload[tep.EXPERT_WIRES_KEY][f"{ALLOC_STACK}.0.w1"][ALLOC_FMT]
    split = _alloc_assignment()
    split[f"{ALLOC_STACK}.0.w1"] = "TESSERA_E4M3_K1_R768"
    with pytest.raises(ExpertProjectionError, match="not the selected rung"):
        tep.allocation_expert_projection_block(swapped, split)


def test_mixed_allocation_block_without_the_installed_capability_refuses(
        tmp_path, monkeypatch):
    payload = _alloc_cost_payload(tmp_path, formats=(ALLOC_FMT, "TESSERA_E4M3_K1_R768"))
    mixed = _alloc_assignment()
    mixed[f"{ALLOC_STACK}.0.w2"] = "TESSERA_E4M3_K1_R768"
    with pytest.raises(ExpertProjectionError, match="v57"):
        tep.allocation_expert_projection_block(payload, mixed)


def test_uniform_allocation_block_is_byte_identical_to_the_stack_uniform_artifact(
        tmp_path):
    payload = _alloc_cost_payload(tmp_path)
    block = tep.allocation_expert_projection_block(payload, _alloc_assignment())
    assert block["tessera_expert_stack_formats"] == {ALLOC_STACK: ALLOC_FMT}
    assert tep.UNIT_RUNGS_KEY not in block
    # A stack kept whole at BF16 says so, without receipts, exactly as before.
    bf16 = tep.allocation_expert_projection_block(payload, _alloc_assignment("BF16"))
    assert bf16["tessera_expert_stack_formats"] == {ALLOC_STACK: "BF16"}
    assert bf16[tep.EXPERT_WIRES_KEY] == {}
    assert tep.UNIT_RUNGS_KEY not in bf16


# ---------------------------------------------------------------------------
# The plan writer: unit_q256 overrides on the stack entry
# ---------------------------------------------------------------------------
def _leaf(rung: int):
    return {"grid": "E4M3", "q256": rung}


def _logical_plan(rungs: dict) -> dict:
    return {tensor: _leaf(rung) for tensor, rung in rungs.items()}


def _members_layouts():
    tensors = list(STACK_TENSORS)
    return {WRITER_STACK: tensors}, {WRITER_STACK: "unpacked_per_expert"}


def test_stack_plan_emits_unit_q256_overrides_and_normalizes_uniform_stacks():
    members, layouts = _members_layouts()
    tensors = members[WRITER_STACK]
    # Uniform: the exact stack-uniform spelling, no unit map.
    uniform = writer.stack_plan(_logical_plan({t: 1024 for t in tensors}),
                                dict(members), dict(layouts))
    assert uniform[WRITER_STACK] == {"grid": "E4M3", "q256": 1024,
                                     "source_layout": "unpacked_per_expert"}
    # Mixed: the entry keeps the producer-planned rung and names the overrides.
    rungs = {t: 1024 for t in tensors}
    rungs[tensors[1]] = 768
    mixed = writer.stack_plan(_logical_plan(rungs), dict(members), dict(layouts),
                              baseline_q256={WRITER_STACK: 1024})
    override = tensors[1][: -len(".weight")]
    assert mixed[WRITER_STACK] == {"grid": "E4M3", "q256": 1024,
                                   "source_layout": "unpacked_per_expert",
                                   "unit_q256": {override: 768}}
    # Every member above the baseline: complete overrides, entry rung kept.
    all_up = writer.stack_plan(_logical_plan({t: 768 for t in tensors}),
                               dict(members), dict(layouts),
                               baseline_q256={WRITER_STACK: 1024})
    assert all_up[WRITER_STACK]["unit_q256"] == {
        t[: -len(".weight")]: 768 for t in tensors}
    assert all_up[WRITER_STACK]["q256"] == 1024


def test_stack_plan_refuses_mixed_grids_bf16_mixing_and_missing_baseline():
    members, layouts = _members_layouts()
    tensors = members[WRITER_STACK]
    rungs = {t: 1024 for t in tensors}
    mixed_plan = _logical_plan(rungs)
    mixed_plan[tensors[1]] = {"grid": "E2M1x2", "q256": 896}
    with pytest.raises(SystemExit, match=rf"{WRITER_STACK}.*grid"):
        writer.stack_plan(mixed_plan, dict(members), dict(layouts),
                          baseline_q256={WRITER_STACK: 1024})
    bf16_plan = _logical_plan(rungs)
    bf16_plan[tensors[1]] = "BF16"
    with pytest.raises(SystemExit, match=rf"{WRITER_STACK}.*BF16"):
        writer.stack_plan(bf16_plan, dict(members), dict(layouts),
                          baseline_q256={WRITER_STACK: 1024})
    mixed = _logical_plan({**rungs, tensors[1]: _leaf(768)})
    with pytest.raises(SystemExit, match=rf"{WRITER_STACK}.*planned"):
        writer.stack_plan(mixed, dict(members), dict(layouts))
    with pytest.raises(SystemExit, match=rf"{WRITER_STACK}.*planned"):
        writer.stack_plan(mixed, dict(members), dict(layouts), baseline_q256={})


def test_plan_from_assignment_threads_the_carried_request_baseline(
        tmp_path, monkeypatch):
    surface = _stack_surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    (tmp_path / "config.json").write_text("{}")
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    up = {"tessera_format": "TESSERA_E4M3_K1_R768"}
    config = {
        "model.layers.0.self_attn.q_proj": fmt,
        "model.layers.0.self_attn.k_proj": fmt,
        "model.layers.0.self_attn.v_proj": fmt,
        WRITER_STACK + ".expert_0.w1": fmt,
        WRITER_STACK + ".expert_0.w3": up,
        WRITER_STACK + ".expert_0.w2": fmt,
    }
    shapes, members, layouts = _context(surface, config, tmp_path)
    carried = {"schema": tep.CARRIED_PROJECTION_SCHEMA,
               "producer": {"schema": tep.PROJECTION_SCHEMA},
               "request": {WRITER_STACK: {"grid": "E4M3", "q256": 1024,
                                          "source_layout": "unpacked_per_expert"}}}
    monkeypatch.setattr(writer, "read_carried_projection", lambda config: carried)
    plan, provenance, _logical = writer.plan_from_assignment(
        config, shapes, members, layouts, model=tmp_path, cover="as-allocated",
        allow_disagreement=False, control_rule="nearest", with_control=False,
        surface=surface)
    planned = plan[WRITER_STACK]
    assert planned["q256"] == 1024
    assert planned["unit_q256"] == {WRITER_STACK + ".expert_0.w3": 768}
    assert provenance["expert_stacks"][WRITER_STACK]["planned_as"] == planned


# ---------------------------------------------------------------------------
# The export lane and the census close: the per-unit stamp must agree
# ---------------------------------------------------------------------------
def test_export_lane_mixed_selection_agrees_with_its_per_unit_stamp(case, monkeypatch):
    monkeypatch.setattr(trc, "packaged_routed_unit_capability",
                        lambda: ("d" * 64, dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK)))
    source_tensors = carried["producer"]["source"]["tensors"]
    shards = {name + ".weight": source_tensors[name + ".weight"]
              for name in _lane_units()}
    switched = sorted(_lane_units())[0]
    up = "TESSERA_E4M3_K1_R768"
    receipt = _lane_receipt(switched, case.units[switched], fmt=up)
    (case.wire_dir / receipt["file"]).write_bytes(_lane_blob(switched, up))
    selected = {name: ALLOC_FMT for name in _lane_units()}
    selected[switched] = up
    meta = {tep.PROJECTION_KEY: carried,
            tep.EXPERT_WIRES_KEY: {**case.receipts, switched: receipt},
            tep.STACK_FORMATS_KEY: {},
            tep.UNIT_RUNGS_KEY: {
                "schema": tep.UNIT_RUNGS_SCHEMA,
                "stacks": {STACK: dict(selected)}},
            tep.WIRE_DIR_KEY: str(case.wire_dir)}
    _mode, bundle = export._carried_expert_projection(meta, selected, shards)
    assert bundle["stacks"] == {}
    assert bundle["unit_rungs"] == {STACK: dict(selected)}
    assert bundle["units"][switched] == receipt
    # A hand-edited or stale stamp refuses instead of shipping.
    stale = dict(meta, **{tep.UNIT_RUNGS_KEY: {"schema": tep.UNIT_RUNGS_SCHEMA,
                                               "stacks": {}}})
    with pytest.raises(export.TesseraExportLaneError, match=tep.UNIT_RUNGS_KEY):
        export._carried_expert_projection(stale, selected, shards)
    absent = dict(meta)
    absent.pop(tep.UNIT_RUNGS_KEY)
    with pytest.raises(export.TesseraExportLaneError, match=tep.UNIT_RUNGS_KEY):
        export._carried_expert_projection(absent, selected, shards)
    # Without the installed capability the same mixed selection refuses.
    monkeypatch.setattr(trc, "packaged_routed_unit_capability",
                        lambda: (_ for _ in ()).throw(
                            trc.TesseraContractError("contract v45 publishes no "
                                                     "routed_units; requires v57")))
    with pytest.raises(export.TesseraExportLaneError, match="v57"):
        export._carried_expert_projection(meta, selected, shards)


def test_selected_census_assignment_checks_the_per_unit_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(trc, "packaged_routed_unit_capability",
                        lambda: ("d" * 64, dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK)))
    _names, experts, records, cost, _roster, _shapes = _census_world(tmp_path)
    _source, units, stack_of = carried_units(cost["provenance"][tep.PROJECTION_KEY])
    routed = sorted(experts)[0]
    up = "TESSERA_E4M3_K1_R768"
    blob = b"r768-wire" * 8
    record = dict(records[routed])
    record.update(file=f"{routed.replace('.', '__')}__{up}.tessera",
                  blob_sha256=hashlib.sha256(blob).hexdigest(), blob_bytes=len(blob))
    record["identity"] = {**record["identity"],
                          "recipe": {"grid": "E4M3", "q256": 768}}
    cost["costs"][routed][up] = {"output_mse": 1e-3, "output_mse_measured": True,
                                 "wire_bytes": len(blob),
                                 "hessian_identity": {"applied": True}}
    cost.setdefault(tep.EXPERT_WIRES_KEY, {}).setdefault(routed, {})[up] = record
    names = sorted(units)
    assignment = {name: CENSUS_FMT for name in names}
    assignment[routed] = up
    stack_formats, unit_rungs = tep.require_unit_assignment(
        {name: assignment[name] for name in units}, stack_of, units,
        capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))
    metadata = {tep.PROJECTION_KEY: cost["provenance"][tep.PROJECTION_KEY],
                tep.STACK_FORMATS_KEY: stack_formats,
                tep.UNIT_RUNGS_KEY: {"schema": tep.UNIT_RUNGS_SCHEMA,
                                     "stacks": unit_rungs}}
    selected, _source, _units2, _stack_of = census.selected_census_assignment(
        assignment, metadata, cost)
    assert selected[routed] == up
    drifted = json.loads(json.dumps(metadata))
    drifted[tep.UNIT_RUNGS_KEY]["stacks"][STACK][routed] = CENSUS_FMT
    with pytest.raises(census.CensusCacheError, match=tep.UNIT_RUNGS_KEY):
        census.selected_census_assignment(assignment, drifted, cost)
    # Uniform worlds stay exactly as they were: no unit stamp required.
    uniform_assignment_map = {name: CENSUS_FMT for name in names}
    uniform_meta = {tep.PROJECTION_KEY: metadata[tep.PROJECTION_KEY],
                    tep.STACK_FORMATS_KEY: {STACK: CENSUS_FMT}}
    selected, _s, _u, _so = census.selected_census_assignment(
        uniform_assignment_map, uniform_meta, cost)
    assert set(selected) == set(names)


# ---------------------------------------------------------------------------
# Mixed-buy CLI acceptance: synthetic two-rung eligibility, real entrypoint
# ---------------------------------------------------------------------------
def _synthetic_two_rung_contract(monkeypatch):
    """Test-local v5 contract attesting a second routed-E4M3 rung.

    SYNTHETIC ELIGIBILITY FIXTURE, NO SERVED QUALIFICATION: the packaged
    contract attests R1024 for routed E4M3, so no second rung can be
    candidate-legal in this world without fabricating attestation.  This
    helper extends ONLY the test-local parsed contract -- the packaged
    bytes, the pins and the shared fixtures are untouched -- with R1088 on
    the routed-E4M3 cells and the attested set, through the same
    down-conversion the shared fixture uses.  It models eligibility (which
    rungs the allocator may consider), never measurement (no served KL is
    claimed for anything here).
    """
    import dataclasses
    from prismaquant import tessera_menu as menu
    from prismaquant import tessera_runtime_contract as contract
    from conftest import (
        down_convert_lane_table, lane_cells_on_one_image,
        project_lane_cells_onto_structures)
    from prismaquant.lane_eligibility import LANE_ELIGIBILITY_SCHEMAS
    from test_allocator_expert_projection import IMAGE as _ALLOC_IMAGE
    payload = json.loads(contract.contract_path().read_text())
    block = payload["lane_eligibility"]
    if block.get("schema") not in LANE_ELIGIBILITY_SCHEMAS:
        pytest.skip("packaged lane table is not readable by this checkout's "
                    "lane_eligibility")
    payload = project_lane_cells_onto_structures(
        lane_cells_on_one_image(payload), ("dense", "routed_moe"))
    payload = down_convert_lane_table(payload, "tessera.lane-eligibility.v5")
    for cell in payload["lane_eligibility"]["cells"]:
        cell["runtime"] = {"image": _ALLOC_IMAGE, "execution_modes": ["eager"]}
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    cells = []
    for cell in parsed.cells:
        if cell.family == "TESSERA_E4M3" and cell.structure == "routed_moe":
            cell = dataclasses.replace(
                cell, rungs_q256=cell.rungs_q256 | {1088},
                covered_rungs_q256=cell.covered_rungs_q256 | {1088})
        cells.append(cell)
    extended = dataclasses.replace(
        parsed, cells=tuple(cells),
        attested_rungs={**parsed.attested_rungs,
                        "TESSERA_E4M3": parsed.attested_rungs.get("TESSERA_E4M3",
                                                                 frozenset()) | {1088}})
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: extended)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")


# ---------------------------------------------------------------------------
# The allocator CLI hook: the user-runnable per-unit entrypoint
# ---------------------------------------------------------------------------
def test_allocator_cli_routed_unit_rates_entrypoint(tmp_path, monkeypatch):
    """``--routed-unit-rates`` is the runnable allocation entrypoint.

    One ID, four subcases, all through real ``allocator.main`` runs.  (1) The
    flag needs a serialized-byte cap: without ``--target-disk-gb`` it
    refuses by name.  (2) Admission-noop: where the priced R768 rows are not
    candidate-legal (the synthetic v5 contract attests R1024 for routed
    E4M3), the pass buys nothing instead of mislabeling a menu-excluded rung
    as legitimate -- priced is not eligible -- while dense rows without exact
    price fields stay grouped and a priced-but-never-admitted third rung is
    never bought.  (3) Legal mixed-buy: under the synthetic two-rung
    eligibility fixture the DP baselines R1024 (the R1088 rows are priced
    worse, so the min-loss body never takes them) and the pass buys five
    legal R1024->R1088 upgrades inside the disk headroom under the explicit
    reserve, one unit keeping its baseline -- full new pass and export wiring
    executed.  (4) Without the flag the noop run is byte-identical to today
    minus the record block.  The reservation rides the CLI's own explicit
    ``--artifact-overhead-reserve-bytes``; the whole-artifact claim stays the
    export's exact owner and every record says so.
    """
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    _alloc_v5_contract(monkeypatch)
    # The PB box's installed SDK predates v57: the capability gate is pinned
    # by its own tests; this test pins the entrypoint wiring.
    monkeypatch.setattr(trc, "packaged_routed_unit_capability",
                        lambda: ("d" * 64, dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK)))

    def _disk_argv(root, payload, *extra):
        argv = _alloc_argv(root, payload)
        bits = argv.index("--target-bits")
        del argv[bits:bits + 2]
        argv += ["--target-disk-gb", "16",
                 "--artifact-overhead-reserve-bytes", "1048576",
                 *extra]
        return argv

    # 1. The flag needs a serialized-byte cap.
    nocap = tmp_path / "nocap"
    nocap.mkdir()
    monkeypatch.setattr(
        sys, "argv",
        _alloc_argv(nocap, _alloc_cost_payload(nocap)) + ["--routed-unit-rates"])
    with pytest.raises(SystemExit, match="needs --target-disk-gb"):
        allocator.main()

    # 2. Admission-noop: priced but inapplicable rows are never bought.
    noop = tmp_path / "noop"
    noop.mkdir()
    payload = _alloc_cost_payload(noop, formats=(ALLOC_FMT, "TESSERA_E4M3_K1_R768"))
    ineligible = f"{ALLOC_STACK}.1.w3"
    for name in _alloc_units():
        payload["costs"][name][ALLOC_FMT].update(predicted_dloss=1.0, wire_bytes=4096)
        payload["costs"][name]["TESSERA_E4M3_K1_R768"].update(
            predicted_dloss=0.5, wire_bytes=4608)
    # A priced third rung no menu ever admitted: profile-ineligible by
    # construction, and never bought even with exact price fields.
    payload["costs"][ineligible]["TESSERA_E4M3_K1_R896"] = {
        "predicted_dloss": -1.0, "wire_bytes": 4096}
    payload[tep.EXPERT_WIRES_KEY][ineligible]["TESSERA_E4M3_K1_R896"] = \
        payload[tep.EXPERT_WIRES_KEY][ineligible][ALLOC_FMT]
    monkeypatch.setattr(sys, "argv", _disk_argv(noop, payload, "--routed-unit-rates"))
    allocator.main()
    meta = json.loads((noop / "layer.json").read_text())["__prismaquant__"]
    placed = load_assignment(noop / "layer.json")
    assert placed[ALLOC_DENSE] == ALLOC_FMT
    assert {name: placed[name] for name in _alloc_units()} == {
        name: ALLOC_FMT for name in _alloc_units()}
    assert meta["tessera_expert_stack_formats"] == {ALLOC_STACK: ALLOC_FMT}
    assert tep.UNIT_RUNGS_KEY not in meta
    assert meta[tep.EXPERT_WIRES_KEY] == {
        name: payload[tep.EXPERT_WIRES_KEY][name][ALLOC_FMT]
        for name in _alloc_units()}
    record = meta["tessera_routed_unit_rates"]
    assert record["currency"] == "price_row_wire_delta_bytes"
    assert record["upgrades"] == 0
    assert record["spent_wire_delta_bytes"] == 0
    assert record["byte_budget"] >= 0
    assert record["reserve_bytes"] == 1_048_576
    assert record["whole_artifact_bytes_claimed"] is False

    # 3. Legal mixed-buy under the synthetic two-rung eligibility fixture
    # (SYNTHETIC ELIGIBILITY, NO SERVED QUALIFICATION -- see
    # _synthetic_two_rung_contract).
    _synthetic_two_rung_contract(monkeypatch)
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    up = "TESSERA_E4M3_K1_R1088"
    payload = _alloc_cost_payload(mixed, formats=(ALLOC_FMT, up))
    stays = f"{ALLOC_STACK}.0.w2"
    for name in _alloc_units():
        payload["costs"][name][ALLOC_FMT].update(predicted_dloss=1.0, wire_bytes=4096)
        payload["costs"][name][up].update(
            predicted_dloss=1.5, wire_bytes=4096 if name == stays else 4608,
            output_mse=4e-3)
    monkeypatch.setattr(sys, "argv", _disk_argv(mixed, payload, "--routed-unit-rates"))
    allocator.main()
    placed = load_assignment(mixed / "layer.json")
    upgraded = [name for name in _alloc_units() if name != stays]
    assert {name: placed[name] for name in _alloc_units()} == {
        **{name: up for name in upgraded}, stays: ALLOC_FMT}
    meta = json.loads((mixed / "layer.json").read_text())["__prismaquant__"]
    assert meta[tep.UNIT_RUNGS_KEY] == {
        "schema": tep.UNIT_RUNGS_SCHEMA,
        "stacks": {ALLOC_STACK: {name: placed[name] for name in _alloc_units()}},
    }
    assert meta["tessera_expert_stack_formats"] == {}
    assert meta[tep.EXPERT_WIRES_KEY] == {
        name: payload[tep.EXPERT_WIRES_KEY][name][placed[name]]
        for name in _alloc_units()}
    record = meta["tessera_routed_unit_rates"]
    assert record["currency"] == "price_row_wire_delta_bytes"
    assert record["upgrades"] == len(upgraded)
    assert record["spent_wire_delta_bytes"] == 512 * len(upgraded)
    assert record["byte_budget"] >= record["spent_wire_delta_bytes"]
    assert record["reserve_bytes"] == 1_048_576
    assert record["whole_artifact_bytes_claimed"] is False

    # 4. Without the flag: the noop run minus the record block, byte for byte.
    off = tmp_path / "off"
    off.mkdir()
    payload = _alloc_cost_payload(off, formats=(ALLOC_FMT, "TESSERA_E4M3_K1_R768"))
    for name in _alloc_units():
        payload["costs"][name][ALLOC_FMT].update(predicted_dloss=1.0, wire_bytes=4096)
        payload["costs"][name]["TESSERA_E4M3_K1_R768"].update(
            predicted_dloss=0.5, wire_bytes=4608)
    monkeypatch.setattr(sys, "argv", _disk_argv(off, payload))
    allocator.main()
    plain = json.loads((off / "layer.json").read_text())["__prismaquant__"]
    placed = load_assignment(off / "layer.json")
    assert {name: placed[name] for name in _alloc_units()} == {
        name: ALLOC_FMT for name in _alloc_units()}
    assert plain["tessera_expert_stack_formats"] == {ALLOC_STACK: ALLOC_FMT}
    assert plain == {k: v for k, v in
                     json.loads((noop / "layer.json").read_text())["__prismaquant__"].items()
                     if k != "tessera_routed_unit_rates"}
    assert tep.UNIT_RUNGS_KEY not in plain
    assert "tessera_routed_unit_rates" not in plain


# ---------------------------------------------------------------------------
# The allocator record is the export accounting's price-row leg, honestly
# ---------------------------------------------------------------------------
def test_allocator_record_does_not_claim_whole_artifact_compliance():
    costs = _fixture_costs()
    assignment = {unit: BASE_FMT for unit in costs}
    _picks, record = tep.select_priced_unit_upgrades(costs, assignment,
                                                     byte_budget=HEADROOM_BYTES)
    assert record["whole_artifact_bytes_claimed"] is False
    assert "price_row_wire_delta_bytes" == record["currency"]
    assert record["spend_cap_bytes"] == HEADROOM_BYTES
    assert record["rule"].startswith("ascending predicted_dloss delta per wire-delta byte")


# ---------------------------------------------------------------------------
# Review regressions (parent REQUEST_CHANGES, pre-fix red)
# ---------------------------------------------------------------------------
def test_allocator_recomputes_marginals_after_every_single_upgrade():
    """One pick per recomputed margin, not one bulk pass over a stale sort.

    A climbs 768->896 at -0.10 loss/byte, then 896->1024 at -0.08; B climbs
    768->896 at -0.07.  With 200 B of cap the priced order buys A twice (A at
    1024, 200 spent, 0 left).  A pass that sorts once and buys the whole
    snapshot buys A-896 then B-896 instead -- the same spend at worse loss.
    """
    costs = {
        "s.0.w1": {"TESSERA_E4M3_K1_R768": {"predicted_dloss": 100.0, "wire_bytes": 400},
                   "TESSERA_E4M3_K1_R896": {"predicted_dloss": 90.0, "wire_bytes": 500},
                   "TESSERA_E4M3_K1_R1024": {"predicted_dloss": 82.0, "wire_bytes": 600}},
        "s.0.w2": {"TESSERA_E4M3_K1_R768": {"predicted_dloss": 100.0, "wire_bytes": 400},
                   "TESSERA_E4M3_K1_R896": {"predicted_dloss": 93.0, "wire_bytes": 500}},
    }
    assignment = {"s.0.w1": "TESSERA_E4M3_K1_R768", "s.0.w2": "TESSERA_E4M3_K1_R768"}
    picks, record = tep.select_priced_unit_upgrades(costs, assignment, byte_budget=200)
    assert picks == {"s.0.w1": "TESSERA_E4M3_K1_R1024"}
    assert record["spent_wire_delta_bytes"] == 200
    assert record["remaining_wire_delta_bytes"] == 0


def test_mixed_nvfp4_stack_refuses_by_name():
    """Per-unit rungs are served on the producer's E4M3 grid only (v57).

    A stack mixed across two E2M1x2 rungs is one grid and capability-granted,
    but the v57 producer does not serve mixed NVFP4 stacks -- it refuses by
    stack, grid and the served rung surface instead of emitting an
    unservable plan.
    """
    _source, units, stack_of = carried_units(carried_projection(
        _projection(), bind_expert_projection(_projection(), declared=_declared()),
        request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t"))
    mixed = {}
    for name in sorted(units):
        rung = "TESSERA_E2M1_K2_R896" if ".0." in name else "TESSERA_E2M1_K2_R700"
        mixed[name] = rung
    assert len(set(mixed.values())) == 2
    with pytest.raises(ExpertProjectionError, match=rf"{STACK}.*E2M1x2.*E4M3"):
        tep.require_unit_assignment(
            mixed, stack_of, units,
            capability=dict(trc.ROUTED_UNIT_ASSIGNMENT_BLOCK))
