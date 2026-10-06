"""PQ #1618: the lane roster mirror learns Tessera v45's structure-scoped
``column_rates_routed_moe`` requirement; PQ #1702 pins v45.

Tessera v45 (tessera#694) moves the fused routed lanes' ``column_rates`` to
[1..8] and adds ``column_rates_routed_moe: [1..6]``: the routed-expert
gate/up launch (two tables plus word stages) does not fit sm_121's shared
memory above rate 6, while the dense / one-table launch reads every rate.
The decision is Tessera's (``scheme.decide_lane_requirements``); the mirror
learns the NAME, holds it to a subset of ``column_rates`` as Tessera's
validator does, and states the unit's STRUCTURE as a plan fact, taken from
the decision unit's cell and never inferred from bytes.

The installed v56 pin publishes routed rates1..8 and the MMA extension.
Every leg runs the real unpatched decision core. Historical refusal legs
use an explicit v45 six-rate predicate and a v44 predicate without the
routed field; neither is relabelled as the current publisher answer.
The v56 widening is separately asserted for rates7/8.
"""
import copy
import dataclasses
from types import SimpleNamespace

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render

from tests.test_tessera_lane_requires import (
    COMPACT_E4M3, E4M3, E4M3_RATE, FUSED_E4M3, FUSED_LANES, GATED_CARRIER, _cell,
    _parsed_cell, _raw, _row, _table)

FIELD = "column_rates_routed_moe"
FUSED_LANE = FUSED_LANES[0]
ROUTED_RATES = [1, 2, 3, 4, 5, 6, 7, 8]
ALL_RATES = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def payload():
    return _raw()[0]


def _fused_only(payload):
    """A copy whose routed E4M3 decode cell launches ONLY through the fused
    lane, so a refusal is the cell's refusal.  The pinned cell also names the
    compact launch, which a refusing lane leaves in place."""
    moved = copy.deepcopy(payload)
    cell = _cell(moved, GATED_CARRIER)
    cell["executes"] = [{"symbol": FUSED_E4M3[0], "decoder": FUSED_E4M3[1]}]
    return moved


def _v44(payload):
    """A copy whose fused lanes publish v44's predicate: ``column_rates`` [4]
    and no routed set.  The pop has no default, so a pin whose fused lanes do
    not publish the field fails here rather than passing vacuously."""
    moved = copy.deepcopy(payload)
    for name in FUSED_LANES:
        requires = _row(moved, name)["lane"]["requires"]
        requires.pop(FIELD)
        requires["column_rates"] = [4]
    # The MMA extension did not exist at v44; it is not a historical launch.
    moved["native_extensions"] = [
        row for row in moved["native_extensions"]
        if row["module_name_prefix"] != "tessera_routed_fused_mma_e4m3"]
    for cell in moved["lane_eligibility"]["cells"]:
        cell["executes"] = [launch for launch in cell["executes"]
                            if launch["decoder"] != "native_routed_fused_window_e4m3mma"]
    return moved


def _plan_at_rate(monkeypatch, rate):
    """The real planned facts (structure plumbing included) with the rate set
    the render would plan pinned to ``{rate}``."""
    real = render.planned_wire_facts

    def planned(family, rung, **kw):
        facts = dict(real(family, rung, **kw))
        facts["rates"] = (rate,)
        return facts

    monkeypatch.setattr(render, "planned_wire_facts", planned)


def _fused_only_cell_and_lanes(payload):
    table = _table(_fused_only(payload))
    cell = _parsed_cell(table, GATED_CARRIER)
    return cell, table.lanes


def test_the_pinned_fused_lanes_publish_the_field(payload):
    assert FIELD in lane.LANE_REQUIREMENT_FIELDS
    assert FIELD in lane.LANE_REQUIREMENT_LISTS
    table = _table(payload)
    for name in FUSED_LANES:
        claim = next(c for c in table.lanes if c.extension == name)
        assert claim.requires[FIELD] == tuple(ROUTED_RATES)
        expected = _row(payload, name)["lane"]["requires"]["column_rates"]
        assert claim.requires["column_rates"] == tuple(expected)


@pytest.mark.parametrize("routed,base,needle", [
    ([1, 2, 9], ALL_RATES, "column_rates"),          # not a subset
    (ROUTED_RATES, None, "column_rates"),            # no column_rates to narrow
    ([6, 5, 4], ALL_RATES, "ascending"),             # the list grammar still holds
])
def test_the_field_is_held_to_a_subset_of_column_rates(payload, routed, base, needle):
    moved = copy.deepcopy(payload)
    requires = _row(moved, FUSED_LANE)["lane"]["requires"]
    requires[FIELD] = routed
    if base is None:
        del requires["column_rates"]
    with pytest.raises(lane.LaneEligibilityError, match=needle) as info:
        _table(moved)
    assert FIELD in str(info.value) or needle == "ascending"


def _v45_routed_limit(payload):
    """Retain the original structure-only refusal on an explicit v45 predicate."""
    moved = copy.deepcopy(payload)
    _row(moved, FUSED_LANE)["lane"]["requires"][FIELD] = [1, 2, 3, 4, 5, 6]
    return moved


@pytest.mark.parametrize("rate", [7, 8])
def test_current_v56_routed_rate_widening_is_explicit(payload, monkeypatch, rate):
    _plan_at_rate(monkeypatch, rate)
    cell, lanes = _fused_only_cell_and_lanes(payload)
    assert lane.cell_lane_admits(cell, E4M3_RATE, lanes) == (True, "")


def test_a_routed_unit_at_rate_7_is_refused_with_the_field_named(payload, monkeypatch):
    _plan_at_rate(monkeypatch, 7)
    cell, lanes = _fused_only_cell_and_lanes(_v45_routed_limit(payload))
    assert cell.structure == "routed_moe"
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, lanes)
    assert not admits
    assert FIELD in why and "[7]" in why and "compact adapter" in why, why


def test_the_same_rung_as_a_dense_unit_passes(payload, monkeypatch):
    _plan_at_rate(monkeypatch, 7)
    cell, lanes = _fused_only_cell_and_lanes(_v45_routed_limit(payload))
    dense = dataclasses.replace(cell, structure="dense")
    assert lane.cell_lane_admits(dense, E4M3_RATE, lanes) == (True, "")
    # ... and a routed unit inside the routed set passes too: the refusal is
    # the rate, not the lane.
    _plan_at_rate(monkeypatch, 6)
    assert lane.cell_lane_admits(cell, E4M3_RATE, lanes) == (True, "")


def test_a_unit_with_no_structure_fact_is_refused_by_name(payload, monkeypatch):
    _plan_at_rate(monkeypatch, 4)
    cell, lanes = _fused_only_cell_and_lanes(payload)
    bare = SimpleNamespace(id=cell.id, family=cell.family, executes=cell.executes)
    admits, why = lane.cell_lane_admits(bare, E4M3_RATE, lanes)
    assert not admits
    assert FIELD in why and "structure" in why, why


def test_planned_wire_facts_states_structure_only_when_told(payload):
    assert "structure" not in render.planned_wire_facts(E4M3, E4M3_RATE)
    for structure in ("dense", "routed_moe"):
        assert render.planned_wire_facts(
            E4M3, E4M3_RATE, structure=structure)["structure"] == structure
    with pytest.raises(ValueError, match="structure"):
        render.planned_wire_facts(E4M3, E4M3_RATE, structure="sparse")


def test_a_v44_shaped_table_still_reads_and_decides_as_v44_did(payload):
    """The real decision core on the pinned table with its fused lanes rewound
    to v44's predicate: no lane carries the field, the cells parse, and the
    routed E4M3 cell makes the fused launch at q1024 only.  The structure fact
    the gate states is read by no requirement of a v44-shaped lane."""
    table = _table(_v44(payload))
    for claim in table.lanes:
        assert FIELD not in (claim.requires or {})
    cell = _parsed_cell(table, GATED_CARRIER)
    for rung, want in ((1024, {COMPACT_E4M3, FUSED_E4M3}), (896, {COMPACT_E4M3})):
        admits, why, launches = lane.cell_rung_launches(cell, rung, table.lanes)
        assert admits, why
        assert set(launches) == want, (rung, launches)
