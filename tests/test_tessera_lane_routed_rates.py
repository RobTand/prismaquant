"""PQ #1618: the lane roster mirror learns Tessera v45's structure-scoped
``column_rates_routed_moe`` requirement.

Tessera v45 (tessera#694) moves the fused routed lanes' ``column_rates`` to
[1..8] and adds ``column_rates_routed_moe: [1..6]``: the routed-expert
gate/up launch (two tables plus word stages) does not fit sm_121's shared
memory above rate 6, while the dense / one-table launch reads every rate.
The decision is Tessera's (``scheme.decide_lane_requirements``); the mirror
learns the NAME, holds it to a subset of ``column_rates`` as Tessera's
validator does, and states the unit's STRUCTURE as a plan fact, taken from
the decision unit's cell and never inferred from bytes.

The installed Tessera pin predates v45, so its decision core does not decide
the new name.  The v45 legs install a stand-in that wraps the real core and
adds the v45 clause verbatim (``every_in_routed_moe``: structure absent ->
refuse by name, dense -> skip, routed_moe -> every planned rate must be in
the set); the v44 leg runs the real, unpatched core on the real pinned table.
"""
import copy
import dataclasses
from types import SimpleNamespace

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render

from tests.test_tessera_lane_requires import (
    E4M3, E4M3_RATE, FUSED_E4M3, FUSED_LANES, GATED_CARRIER, _cell, _parsed_cell,
    _raw, _row, _table)

FIELD = "column_rates_routed_moe"
FUSED_LANE = FUSED_LANES[0]
ROUTED_RATES = [1, 2, 3, 4, 5, 6]
ALL_RATES = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def payload():
    return _raw()[0]


def _v45(payload):
    """The pinned v44 payload reshaped as Tessera v45 publishes it: each fused
    lane reads ``column_rates`` [1..8] and gates the routed launch to 1..6, and
    the routed E4M3 decode cell launches ONLY through the fused lane so a
    refusal is the cell's refusal."""
    moved = copy.deepcopy(payload)
    for name in FUSED_LANES:
        requires = _row(moved, name)["lane"]["requires"]
        requires["column_rates"] = list(ALL_RATES)
        requires[FIELD] = list(ROUTED_RATES)
    cell = _cell(moved, GATED_CARRIER)
    cell["executes"] = [{"symbol": FUSED_E4M3[0], "decoder": FUSED_E4M3[1]}]
    return moved


def _install_v45_decision(monkeypatch):
    import tessera.serving.scheme as scheme
    from tessera.structure import STRUCTURES

    real = scheme.decide_lane_requirements

    def decide(lane_name, requires, facts):
        requires = dict(requires)
        supported = requires.pop(FIELD, None)
        refusals = list(real(lane_name, requires, facts))
        if supported is None:
            return refusals
        structure = facts.get("structure")
        if structure is None:
            refusals.append(
                f"{FIELD}: the unit's structure was not read, so the lane's "
                f"{FIELD} requirement ({supported!r}) cannot be decided; "
                "absent evidence is not a pass")
        elif structure not in STRUCTURES:
            refusals.append(f"{FIELD}: unknown structure {structure!r}")
        elif structure == "routed_moe":
            offending = sorted(set(facts["rates"]) - set(supported))
            if offending:
                refusals.append(
                    f"{FIELD} {offending} are outside the rates this lane's "
                    f"routed-expert launch reaches ({sorted(supported)}); the "
                    "lane reads the wire at these rates but the gate/up "
                    "launch's two tables and word stages do not fit the "
                    "target's shared memory at them, so the stack keeps the "
                    "compact adapter")
        return refusals

    monkeypatch.setattr(scheme, "decide_lane_requirements", decide)


def _plan_at_rate(monkeypatch, rate):
    """The real planned facts (structure plumbing included) with the rate set
    the render would plan pinned to ``{rate}``."""
    real = render.planned_wire_facts

    def planned(family, rung, **kw):
        facts = dict(real(family, rung, **kw))
        facts["rates"] = (rate,)
        return facts

    monkeypatch.setattr(render, "planned_wire_facts", planned)


def _v45_cell_and_lanes(payload):
    table = _table(_v45(payload))
    cell = _parsed_cell(table, GATED_CARRIER)
    return cell, table.lanes


def test_the_mirror_reads_a_v45_shaped_lane(payload):
    assert FIELD in lane.LANE_REQUIREMENT_FIELDS
    assert FIELD in lane.LANE_REQUIREMENT_LISTS
    table = _table(_v45(payload))
    claim = next(c for c in table.lanes if c.extension == FUSED_LANE)
    assert claim.requires[FIELD] == tuple(ROUTED_RATES)
    assert claim.requires["column_rates"] == tuple(ALL_RATES)


@pytest.mark.parametrize("routed,base,needle", [
    ([1, 2, 9], ALL_RATES, "column_rates"),          # not a subset
    (ROUTED_RATES, None, "column_rates"),            # no column_rates to narrow
    ([6, 5, 4], ALL_RATES, "ascending"),             # the list grammar still holds
])
def test_the_field_is_held_to_a_subset_of_column_rates(payload, routed, base, needle):
    moved = _v45(payload)
    requires = _row(moved, FUSED_LANE)["lane"]["requires"]
    requires[FIELD] = routed
    if base is None:
        del requires["column_rates"]
    with pytest.raises(lane.LaneEligibilityError, match=needle) as info:
        _table(moved)
    assert FIELD in str(info.value) or needle == "ascending"


def test_a_routed_unit_at_rate_7_is_refused_with_the_field_named(payload, monkeypatch):
    _install_v45_decision(monkeypatch)
    _plan_at_rate(monkeypatch, 7)
    cell, lanes = _v45_cell_and_lanes(payload)
    assert cell.structure == "routed_moe"
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, lanes)
    assert not admits
    assert FIELD in why and "[7]" in why and "compact adapter" in why, why


def test_the_same_rung_as_a_dense_unit_passes(payload, monkeypatch):
    _install_v45_decision(monkeypatch)
    _plan_at_rate(monkeypatch, 7)
    cell, lanes = _v45_cell_and_lanes(payload)
    dense = dataclasses.replace(cell, structure="dense")
    assert lane.cell_lane_admits(dense, E4M3_RATE, lanes) == (True, "")
    # ... and a routed unit inside the routed set passes too: the refusal is
    # the rate, not the lane.
    _plan_at_rate(monkeypatch, 6)
    assert lane.cell_lane_admits(cell, E4M3_RATE, lanes) == (True, "")


def test_a_unit_with_no_structure_fact_is_refused_by_name(payload, monkeypatch):
    _install_v45_decision(monkeypatch)
    _plan_at_rate(monkeypatch, 4)
    cell, lanes = _v45_cell_and_lanes(payload)
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


def test_a_v44_contract_without_the_field_still_reads(payload):
    """The real pinned table and the real, unpatched decision core: no lane
    publishes the field, the cells parse, and the routed E4M3 cell decides the
    same way as before -- the structure fact the gate now states is ignored."""
    table = _table(payload)
    for claim in table.lanes:
        assert FIELD not in (claim.requires or {})
    cell = _parsed_cell(table, GATED_CARRIER)
    admits, why, launches = lane.cell_rung_launches(cell, E4M3_RATE, table.lanes)
    assert admits, why
    assert launches
