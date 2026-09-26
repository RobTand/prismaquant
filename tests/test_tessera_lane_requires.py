"""The lane PREDICATE: ``native_extensions[].lane.requires``, consumed.

Tessera's contract v20 (its #264) publishes, on every ``native_extensions[]``
row, a ``lane`` block: the ``decoder`` the extension serves and -- for the
window-GEMV kernel -- ``requires``, the predicate a unit's WIRE must satisfy
for that kernel to read it: column rates, window bits, body, plane, no release
overrides, no diagonals, no rotation, no start state, a scalar grid.  The
loader refuses a unit that fails it.  A producer that SELECTED such a unit
would ship an artifact whose serve substitutes another decoder or refuses,
and it would find out at serve time -- the defect ``lane_eligibility`` exists
to stop, one field over.

One rule, one home.  The DECISION is Tessera's
(``tessera.serving.scheme.decide_lane_requirements``) and is called, never
restated: nothing in this repository spells "rates must be in [1, 2, 4]".
PrismaQuant's half is the FACTS -- ``tessera_render.planned_wire_facts``,
the wire this producer's encoder is going to write for a family at a rung,
read off the same recipe and the same decoration constants
``render_tessera_weight`` encodes with -- and the READER, which parses the
predicate closed at Tessera's own vocabulary and refuses by name whatever it
does not understand.  ``lane_eligibility.cell_lane_admits`` is the gate, and
every admission leg reads it: the menu (``tessera_attesting_cells``), the
development contract (``TesseraContract.native_cells``) and the per-unit
export gate (``resolve_unit_route``).

Consequence on the pinned table, stated because it is what a reviewer needs:
since the v31 withdrawals NO pinned cell launches through an extension -- the
streamed E4M3 cells that executed the ``window_gemv`` lane are withdrawn, and
the dense E2M1 pair's span-2 provider retired with the CUDA decoder -- so the
pinned table's every cell reads the gate as admitted and the lane-gated shapes
these tests decide are synthesised (see ``_claiming_lane``/``_gated_carrier``
below).  Nothing is refused today.  The BF16 rung the lane's ``routes`` also
names would NOT pass -- rung 1792 plans rate 7 -- and no BF16 cell exists to
claim that launch at all; the test below pins that a cell which did would be
refused by name at all three legs, on the E2M1 dense pair, whose rung 896
plans rate 7 for the same reason.
"""
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render
from prismaquant import tessera_runtime_contract as contract


WINDOW_LANE = "tessera_window_gemv"
E4M3 = "TESSERA_E4M3_K1"
E4M3_NAME = "TESSERA_E4M3_K1_R1024"
E4M3_RATE = 1024
BF16 = "TESSERA_BF16_K1"
BF16_NAME = "TESSERA_BF16_K1_R1792"
BF16_RATE = 1792
#: Since the v31 withdrawals no pinned cell launches through an extension:
#: the routed rows decode ``torch_materialize_stock`` and the dense E2M1 pair
#: rides ``torch._scaled_mm``/``native_span2``, whose providing row retired
#: with the span-2 CUDA decoder.  The lane-gated shape the tests below need
#: is therefore SYNTHESISED two ways: a carrier whose plan passes (the routed
#: E4M3 decode cell claiming the window lane at rate 4) and a claimer whose
#: plan refuses (the dense E2M1 decode cell claiming it at rate 7).
GATED_CARRIER = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"
CLAIMING = "tessera_e2m1_k2_dense_sm121_decode"
CLAIMING_BATCH = "tessera_e2m1_k2_dense_sm121_batch"
CLAIMING_FAMILY = "TESSERA_E2M1_K2"
CLAIMING_NAME = "TESSERA_E2M1_K2_R896"
CLAIMING_RATE = 896
WINDOW_LAUNCH = {"symbol": "tessera_window_gemv::gemv", "decoder": "window_gemv"}
#: Contract v34's dense window-GEMM launch, verbatim: a qualified symbol
#: that is NOT an extension launch -- no native_extensions row declares
#: `tessera`, and no lane serves `native_window_gemm`.
IN_PLUGIN_LAUNCH = {"symbol": "tessera::window_gemm_dense",
                    "decoder": "native_window_gemm"}


def _claiming_lane(payload):
    """A copy of the pinned payload whose dense E2M1 decode cell claims the
    window-GEMV lane: rung 896 plans rate 7, which the predicate refuses, so
    the cell is a claim the lane will not honour."""
    moved = copy.deepcopy(payload)
    _cell(moved, CLAIMING)["executes"] = [WINDOW_LAUNCH]
    return moved


def _gated_carrier(payload):
    """A copy whose routed E4M3 decode cell claims the window lane at a rung
    whose plan (rate 4) PASSES the predicate -- the shape the streamed E4M3
    cells were on the pre-v31 table.

    The rung is set to q1024 because the window-GEMV lane's published
    ``column_rates`` are [1, 2, 4] and q896 plans rate 3.5, which that lane
    refuses. That refusal applies ONLY to a cell that launches through
    ``window_gemv``. The shipped v38 routed E4M3 cells launch through
    ``native_window_moe_compact``, a decoder no lane publishes a predicate
    for, so they are not lane-gated at all; see
    ``test_the_shipped_routed_e4m3_q896_cells_are_not_lane_gated`` and the
    q896 leg of ``test_a_window_lane_graft_is_refused_by_the_published_column_rates``."""
    moved = copy.deepcopy(payload)
    cell = _cell(moved, GATED_CARRIER)
    cell["executes"] = [WINDOW_LAUNCH]
    cell["rungs_q256"] = [1024]
    return moved


def _raw() -> tuple[dict, str]:
    with as_file(contract.contract_path()) as path:
        raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


@pytest.fixture
def payload():
    return _raw()[0]


def _table(payload):
    return lane._parse_table(payload["lane_eligibility"], payload["formats"],
                             "", "", "x", native_extensions=payload["native_extensions"])


@pytest.fixture
def table(payload):
    return _table(payload)


def _formats(payload):
    return {row["family"]: row for row in payload["formats"]}


def _row(payload, prefix):
    for row in payload["native_extensions"]:
        if row["module_name_prefix"] == prefix:
            return row
    raise AssertionError(f"the installed contract publishes no extension {prefix!r}")


def _cell(payload, cell_id):
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise AssertionError(f"the installed contract publishes no cell {cell_id!r}")


def _parsed_cell(table, cell_id):
    for cell in table.cells:
        if cell.id == cell_id:
            return cell
    raise AssertionError(cell_id)


def _context(payload, cell_id, residency):
    cell = _cell(payload, cell_id)
    return lane.ServingContext(
        platform=cell["platform"], structure=cell["structure"], residency=residency,
        runtime_image=cell["runtime"]["image"],
        execution_mode=cell["runtime"]["execution_modes"][0])


def _canonical(requires):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in requires.items()}


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
def test_the_installed_predicate_is_read_closed_at_tesseras_vocabulary(table, payload):
    claims = {claim.extension: claim for claim in table.lanes}
    assert set(claims) == {WINDOW_LANE}, (
        "since the v32 withdrawals retired the span-2 decoder's row, the "
        "window-GEMV row is the pinned table's only lane")
    window = claims[WINDOW_LANE]
    assert window.decoder == _row(payload, WINDOW_LANE)["lane"]["decoder"]
    assert window.requires == _canonical(_row(payload, WINDOW_LANE)["lane"]["requires"])
    # A lane that publishes no predicate is the route's own eligibility, not
    # an empty predicate.  The nvfp4 row witnessed it on the pinned table
    # until its retirement; the branch is now witnessed by moving the window
    # row's own lane to the no-``requires`` shape.
    bare = copy.deepcopy(payload)
    _row(bare, WINDOW_LANE)["lane"] = {"decoder": "window_gemv"}
    bare_claims = {claim.extension: claim for claim in _table(bare).lanes}
    assert bare_claims[WINDOW_LANE].requires is None
    # The vocabulary this reader closes is exactly what the pinned lane
    # publishes today: a requirement the lane grows is refused by name below,
    # never skipped.
    assert set(lane.LANE_REQUIREMENT_FIELDS) == set(window.requires)
    assert table.provenance()["lanes"] == [
        claim.answer() | {"extension": claim.extension} for claim in table.lanes]


def test_a_table_is_not_readable_without_the_extension_table(payload):
    block, formats = payload["lane_eligibility"], payload["formats"]
    with pytest.raises(TypeError):
        lane._parse_table(block, formats, "", "", "x")
    # The refusal is about launches the reader cannot DECIDE, and no pinned
    # cell names one since the v31 withdrawals -- so the gated shape is
    # synthesised on a copy before the table is withheld.
    gated = _gated_carrier(payload)
    with pytest.raises(lane.LaneEligibilityError, match="native_extensions"):
        lane._parse_table(gated["lane_eligibility"], gated["formats"],
                          "", "", "x", native_extensions=None)


def test_a_table_whose_cells_launch_only_through_torch_needs_no_extension_table(payload):
    """The refusal above is about launches the reader cannot DECIDE. A table
    whose cells never launch through an extension -- every ``executes`` symbol
    a torch/vLLM path -- has no lane to decide and reads ``()`` lanes, as a
    v3 table does; the minimal contracts this repository's export-scope and
    route-receipt tests build are exactly that shape. Only a cell that names
    ``extension::symbol`` makes the missing table a refusal, and the refusal
    names the launch."""
    moved = copy.deepcopy(payload)
    for cell in moved["lane_eligibility"]["cells"]:
        cell["executes"] = [{"symbol": "torch.mm", "decoder": "torch_window"}]
    block, formats = moved["lane_eligibility"], moved["formats"]
    table = lane._parse_table(block, formats, "", "", "x", native_extensions=None)
    assert table.present and table.lanes == ()
    # ... and the refusal, when it fires, says WHICH launch it cannot decide.
    # The pinned table no longer carries an extension-qualified launch, so
    # the claiming copy supplies one.
    gated = _claiming_lane(payload)
    with pytest.raises(lane.LaneEligibilityError, match="tessera_window_gemv::gemv"):
        lane._parse_table(gated["lane_eligibility"], gated["formats"],
                          "", "", "x", native_extensions=None)


def test_an_in_plugin_qualified_launch_is_not_an_extension_launch(payload):
    """A ``::``-qualified symbol is not by itself a launch through an extension.

    Tessera's contract v34 mints four dense cells on
    ``tessera::window_gemm_dense`` under decoder ``native_window_gemm`` and
    says, in the same changelog entry, that the launch "carries lane null
    because it is a launch, not an extension lane"; its ``native_extensions``
    still publishes the one ``tessera_window_gemv`` row and nothing else.
    (Tessera ``e42c0593d257c374315a4ace22db1907921337b0``,
    ``src/tessera/serving/runtime_contract.json``, sha256
    ``d37c9448a751feb3e65db1807a7dff1fbacc767a2ce419dfee70f458dbf03472``.)
    The reader used to read every prefix as an extension name and refuse the
    whole contract. Such a launch stands where ``torch._scaled_mm`` stands:
    the route's own path, no wire predicate to read.
    """
    moved = copy.deepcopy(payload)
    _cell(moved, CLAIMING)["executes"] = [IN_PLUGIN_LAUNCH]
    table = _table(moved)
    cell = _parsed_cell(table, CLAIMING)
    assert cell.executes == ((IN_PLUGIN_LAUNCH["symbol"],
                              IN_PLUGIN_LAUNCH["decoder"]),)
    assert lane.lane_claim_for_cell(cell, table.lanes) is None
    admits, why = lane.cell_lane_admits(cell, CLAIMING_RATE, table.lanes)
    assert admits and why == ""


def test_a_qualified_launch_under_a_lanes_decoder_must_name_its_extension(payload):
    """The guard the prefix rule really carried, on the field the gate keys on.

    A cell that takes a decoder some lane SERVES while naming an extension no
    row declares would be read by that lane at serve time and escape its
    predicate here. That is still refused, and the refusal names the cell, the
    launch, the decoder and the extension whose lane serves it."""
    moved = copy.deepcopy(payload)
    _cell(moved, CLAIMING)["executes"] = [
        {"symbol": "tessera::window_gemm_dense", "decoder": "window_gemv"}]
    with pytest.raises(lane.LaneEligibilityError) as refused:
        _table(moved)
    message = str(refused.value)
    for name in (CLAIMING, "tessera::window_gemm_dense", "window_gemv",
                 WINDOW_LANE, "tessera"):
        assert name in message, (name, message)


def test_a_launch_through_an_extension_under_another_decoder_is_refused(payload):
    """The unchanged leg: a symbol whose prefix IS a declared extension must
    carry that extension's decoder, or the cell would slip past the lane."""
    moved = copy.deepcopy(payload)
    _cell(moved, CLAIMING)["executes"] = [
        {"symbol": "tessera_window_gemv::gemv", "decoder": "native_window_gemm"}]
    with pytest.raises(lane.LaneEligibilityError) as refused:
        _table(moved)
    message = str(refused.value)
    for name in (CLAIMING, "tessera_window_gemv::gemv", "native_window_gemm",
                 "window_gemv"):
        assert name in message, (name, message)


def _mutate(payload, **changes):
    moved = copy.deepcopy(payload)
    block = _row(moved, WINDOW_LANE)["lane"]
    for key, value in changes.items():
        if key == "lane":
            _row(moved, WINDOW_LANE)["lane"] = value
        elif key.startswith("requires."):
            block["requires"][key.split(".", 1)[1]] = value
        elif key == "requires":
            block["requires"] = value
        else:
            block[key] = value
    return moved


@pytest.mark.parametrize("changes, names", [
    ({"requires.span": [2]}, ("span", "cannot decide")),
    ({"requires": {}}, ("requires", "empty")),
    ({"requires": [1, 2, 4]}, ("requires", "object")),
    ({"requires.column_rates": [4, 2, 1]}, ("column_rates", "ascending")),
    ({"requires.column_rates": [1, 1, 2]}, ("column_rates", "ascending")),
    ({"requires.window_bits": [0]}, ("window_bits", "positive")),
    ({"requires.grid_arities": []}, ("grid_arities", "non-empty")),
    ({"requires.release_overrides": "false"}, ("release_overrides", "JSON boolean")),
    ({"requires.rotation": ["two_sided"]}, ("rotation", "two_sided")),
    ({"requires.rotation": []}, ("rotation", "non-empty")),
    ({"requires.body": "trellis"}, ("body", "trellis")),
    ({"requires.plane": "lut"}, ("plane", "lut")),
    ({"kernel": "gemv"}, ("unknown field", "kernel")),
    ({"lane": {"requires": {"column_rates": [4]}}}, ("missing field", "decoder")),
    ({"lane": "window_gemv"}, ("lane", "object")),
])
def test_a_predicate_this_reader_does_not_understand_is_refused_by_name(
        payload, changes, names):
    moved = _mutate(payload, **changes)
    with pytest.raises(lane.LaneEligibilityError) as refused:
        _table(moved)
    for name in names:
        assert name in str(refused.value), (name, str(refused.value))
    assert WINDOW_LANE in str(refused.value)


def test_the_development_contract_reads_the_same_grammar(payload):
    moved = _mutate(payload, **{"requires.span": [2]})
    with pytest.raises(contract.TesseraContractError, match="span"):
        contract._parse(moved, commit="fixture", sha="fixture", path="fixture")


# ---------------------------------------------------------------------------
# The facts: what this producer plans is what its encoder writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, family, rate", [
    (E4M3_NAME, E4M3, E4M3_RATE),
    (BF16_NAME, BF16, BF16_RATE),
])
def test_planned_wire_facts_are_the_facts_the_encoded_bytes_carry(name, family, rate):
    """Principle 8: the gate decides on the plan, the export writes the bytes,
    and the two must be one object.  Read the facts off a unit encoded by the
    render path itself, in Tessera's OWN byte-side vocabulary."""
    from tessera.decode import reconstruct_unit
    from tessera.serving.scheme import wire_facts_of_parsed

    torch.manual_seed(0)
    weight = torch.randn(4, 512, dtype=torch.bfloat16)
    unit, forests = render._encode_planned_unit(weight, name)
    spec, _rung = render.parse_tessera_format_name(name)
    on_bytes = wire_facts_of_parsed(SimpleNamespace(unit=unit, grid=render._grid_for(spec)))
    planned = render.planned_wire_facts(family, rate)
    assert set(planned) == set(on_bytes)
    assert set(on_bytes["rates"]) == set(planned["rates"])
    for key in set(planned) - {"rates"}:
        assert planned[key] == on_bytes[key], key
    assert None not in planned.values()
    # And the render IS that unit's reconstruction, not a second encode.
    rendered = render.render_tessera_weight(weight, name)
    assert torch.equal(
        rendered, reconstruct_unit(unit, forests, render._tessera_export.DEFAULT_CODE)
        .to(dtype=weight.dtype))


def test_the_planned_facts_speak_the_decision_cores_whole_vocabulary(payload):
    """Every published requirement is decided on a fact this producer
    supplies: an 'was not read' refusal would mean the plan is silent on a
    condition the loader enforces."""
    from tessera.serving.scheme import decide_lane_requirements

    requires = _row(payload, WINDOW_LANE)["lane"]["requires"]
    assert decide_lane_requirements(
        WINDOW_LANE, requires, render.planned_wire_facts(E4M3, E4M3_RATE)) == []
    refusals = decide_lane_requirements(
        WINDOW_LANE, requires, render.planned_wire_facts(BF16, BF16_RATE))
    assert refusals and all("was not read" not in line for line in refusals)
    assert all(line.startswith("column_rates") for line in refusals), refusals


def test_the_planned_decoration_is_the_render_decoration():
    """The constants the facts are read from are the constants the encoder
    is called with -- one home, by name."""
    planned = render.planned_wire_facts(E4M3, E4M3_RATE)
    assert planned["rotation"] == render.TESSERA_PLANNED_ROTATION.name
    assert planned["diagonals"] is render.TESSERA_PLANNED_DIAGONALS
    assert planned["release_overrides"] == render.TESSERA_PLANNED_RELEASE_OVERRIDES
    assert planned["start_state"] is False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_no_pinned_cell_is_lane_gated_and_every_plan_admits(table):
    """Since the v31 withdrawals, no pinned cell launches through an extension.

    The streamed E4M3 cells that executed the window-GEMV lane are withdrawn
    and the dense E2M1 pair's span-2 provider retired with the CUDA decoder,
    so the gated set is empty and the gate admits every surviving cell --
    the machinery's honest state at this pin, and the reason the gated shapes
    the rest of this section decides are synthesised.
    """
    gated = {cell.id for cell in table.cells
             if lane.lane_claim_for_cell(cell, table.lanes) is not None}
    assert gated == set(), (
        "the cells subject to the window-GEMV predicate are exactly the ones "
        "that execute it; a decoder no lane names is the route's own path")
    for cell in table.cells:
        for rung in cell.rungs_q256:
            admits, why = lane.cell_lane_admits(cell, rung, table.lanes)
            assert admits, (cell.id, why)
            assert why == ""


def test_every_lane_gated_cell_on_a_synthesised_table_admits_this_producers_plan(
        payload):
    """The gated shape the pinned table no longer carries, kept exercised.

    A cell claiming the window lane at a rung whose plan passes (the routed
    E4M3 decode cell at rate 4) is subject to the predicate and admitted.
    This is the pre-v31 pinned behaviour, synthesised rather than lost."""
    table = _table(_gated_carrier(payload))
    gated = {cell.id for cell in table.cells
             if lane.lane_claim_for_cell(cell, table.lanes) is not None}
    assert gated == {GATED_CARRIER}
    for cell_id in gated:
        cell = _parsed_cell(table, cell_id)
        for rung in cell.rungs_q256:
            admits, why = lane.cell_lane_admits(cell, rung, table.lanes)
            assert admits, (cell_id, why)
            assert why == ""


def test_the_shipped_routed_e4m3_q896_cells_are_not_lane_gated(payload):
    """The cells GLM's layer-43 pick rides (contract v38) admit q896.

    They execute ``native_window_moe_compact``; the only lane in the table
    that publishes a predicate is ``tessera_window_gemv`` (decoder
    ``window_gemv``), so ``lane_claim_for_cell`` answers ``None`` and
    ``cell_lane_admits`` passes on the rung the cells name.
    """
    table = _table(payload)
    for cell_id in (GATED_CARRIER, GATED_CARRIER.replace("_decode_", "_batch_")):
        cell = _parsed_cell(table, cell_id)
        assert tuple(cell.rungs_q256) == (896,)
        assert {decoder for _symbol, decoder in cell.executes} == {
            "native_window_moe_compact"}
        assert lane.lane_claim_for_cell(cell, table.lanes) is None
        assert lane.cell_lane_admits(cell, 896, table.lanes) == (True, "")
    gated_decoders = {claim.decoder for claim in table.lanes
                      if claim.requires is not None}
    assert gated_decoders == {"window_gemv"}, gated_decoders


@pytest.mark.parametrize("rung, admitted", [(1024, True), (896, False)])
def test_a_window_lane_graft_is_refused_by_the_published_column_rates(
        payload, rung, admitted):
    """What refuses q896 is Tessera's window-GEMV predicate, not a PQ list.

    The same synthetic graft at q1024 (rate 4) passes and at q896 (rate 3.5)
    is refused on ``column_rates``, the field the contract publishes for
    that lane. A real routed E4M3 q896 unit never meets this predicate,
    because its cells do not launch through ``window_gemv``.
    """
    moved = _gated_carrier(payload)
    _cell(moved, GATED_CARRIER)["rungs_q256"] = [rung]
    table = _table(moved)
    cell = _parsed_cell(table, GATED_CARRIER)
    admits, why = lane.cell_lane_admits(cell, rung, table.lanes)
    assert admits is admitted, why
    if not admitted:
        assert "column_rates" in why and "tessera_window_gemv" in why, why


def _claiming_bf16(payload):
    """The claiming shape on the surviving dense E2M1 pair: rung 896 plans
    rate 7 exactly as the withdrawn BF16 rung 1792 did, so the claim refuses
    for the same reason.  Kept under its historical name."""
    moved = copy.deepcopy(payload)
    _cell(moved, CLAIMING)["executes"] = [WINDOW_LAUNCH]
    return moved


def test_a_cell_claiming_the_lane_for_a_rung_it_refuses_is_refused_by_name(payload):
    moved = _claiming_bf16(payload)
    table = _table(moved)
    cell = _parsed_cell(table, CLAIMING)
    admits, why = lane.cell_lane_admits(cell, CLAIMING_RATE, table.lanes)
    assert not admits
    for name in (CLAIMING, WINDOW_LANE, "window_gemv", "column_rates", "[7]"):
        assert name in why, (name, why)
    # Sibling cell on the same family is untouched: refusal is per launch.
    batch = _parsed_cell(table, CLAIMING_BATCH)
    assert lane.cell_lane_admits(batch, CLAIMING_RATE, table.lanes) == (True, "")


def test_all_three_admission_legs_read_the_one_gate(payload, monkeypatch):
    moved = _claiming_bf16(payload)
    table, formats = _table(moved), _formats(moved)
    context = _context(moved, CLAIMING, "resident")

    # 1. The menu.
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    assert render.tessera_attesting_cells(CLAIMING_NAME, serving_context=context) == ()
    admitted, reason = render.tessera_lane_admission(CLAIMING_NAME, serving_context=context)
    assert not admitted
    assert "column_rates" in reason and CLAIMING in reason and WINDOW_LANE in reason
    assert not render.tessera_lane_attested(CLAIMING_NAME, serving_context=context)

    # 2. The per-unit export gate names the cell AND the reason.
    facts = lane.UnitStructuralFacts(
        qname="fixture.weight", format_name=CLAIMING_NAME,
        payload_family=CLAIMING_FAMILY,
        k=None, n_sub=None, rate_q256=CLAIMING_RATE, structure="dense",
        role_split=False, in_features=1024, out_features=1024)
    route = lane.resolve_unit_route(
        facts, table, platform=context.platform, residency="resident",
        runtime_image=context.runtime_image, execution_mode=context.execution_mode)
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
    decode = {r.regime: r for r in route.regimes}["decode"]
    assert decode.cell_id == CLAIMING
    assert "column_rates" in decode.detail

    # 3. The development contract.
    parsed = contract._parse(moved, commit="fixture", sha="fixture", path="fixture")
    assert parsed.native_cells(CLAIMING_FAMILY, CLAIMING_RATE,
                               serving_context=context) == ()


def test_the_untouched_table_admits_the_same_rung_at_every_leg(payload, monkeypatch):
    """The control for the test above: the refusal is the launch claim, not
    the family or the rung."""
    table, formats = _table(payload), _formats(payload)
    context = _context(payload, CLAIMING, "resident")
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    assert render.tessera_lane_admission(CLAIMING_NAME, serving_context=context) == (True, "")
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    assert parsed.native_cells(CLAIMING_FAMILY, CLAIMING_RATE, serving_context=context)


def test_a_decorated_plan_is_refused_at_every_requirement_it_breaks(payload, monkeypatch):
    """The predicate is the whole predicate: the loader reads all nine, so
    the gate cannot decide fewer."""
    decorated = dict(render.planned_wire_facts(E4M3, E4M3_RATE))
    decorated.update(rates=(5,), window_bits=12, release_overrides=3, diagonals=True,
                     rotation="R_IN_ONLY", start_state=True, grid_arity=2)
    monkeypatch.setattr(render, "planned_wire_facts", lambda family, rung: decorated)
    table = _table(_gated_carrier(payload))
    cell = _parsed_cell(table, GATED_CARRIER)
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, table.lanes)
    assert not admits
    for name in ("column_rates", "window_bits", "release_overrides", "diagonals",
                 "rotation", "start_state", "grid_arities"):
        assert name in why, (name, why)


def test_a_requirement_the_decision_core_cannot_decide_refuses_rather_than_skips(
        payload, table):
    claim = dataclasses.replace(
        [c for c in table.lanes if c.extension == WINDOW_LANE][0],
        requires={"span": (2,)})
    cell = _parsed_cell(_table(_gated_carrier(payload)), GATED_CARRIER)
    with pytest.raises(lane.LaneEligibilityError, match="span"):
        lane.cell_lane_admits(cell, E4M3_RATE, (claim,))


def test_a_family_this_producer_cannot_plan_is_refused_not_passed(payload, table):
    cell = dataclasses.replace(
        _parsed_cell(_table(_gated_carrier(payload)), GATED_CARRIER),
        family="TESSERA_E5M2_K1")
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, table.lanes)
    assert not admits
    assert "TESSERA_E5M2_K1" in why and "plan" in why


def test_a_lane_gated_cell_without_a_rung_is_refused(payload, table):
    cell = _parsed_cell(_table(_gated_carrier(payload)), GATED_CARRIER)
    admits, why = lane.cell_lane_admits(cell, None, table.lanes)
    assert not admits and "rung" in why


# ---------------------------------------------------------------------------
# The reviewed answer
# ---------------------------------------------------------------------------
def test_the_lane_predicate_is_part_of_the_reviewed_answer(payload):
    """A widened predicate must re-stale the pin: it changes what this
    producer may select."""
    before = contract.contract_answer(
        contract._parse(payload, commit="fixture", sha="fixture", path="fixture"))
    rows = {row["module_name_prefix"]: row for row in before["native_extensions"]}
    assert rows[WINDOW_LANE]["lane"] == {
        "decoder": "window_gemv",
        "requires": _row(payload, WINDOW_LANE)["lane"]["requires"]}
    # The retired nvfp4 row also carried {"decoder": "native_span2",
    # "requires": None} until the v32 withdrawals; the answer's rows are the
    # published ones, and there is exactly one now.
    assert set(rows) == {WINDOW_LANE}
    moved = _mutate(payload, **{"requires.column_rates": [1, 2, 4, 8]})
    after = contract.contract_answer(
        contract._parse(moved, commit="fixture", sha="fixture", path="fixture"))
    assert before != after
    drift = contract._answer_drift(before, after)
    assert any(f"native_extensions[{WINDOW_LANE}].lane" in line for line in drift), drift
