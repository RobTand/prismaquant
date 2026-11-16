"""Contract v69 admits whole TP1 E2M1 R896 units within each cell's scope."""
import json
from dataclasses import replace

from tests.tessera_metadata_fixture import install_metadata

from prismaquant.tessera_runtime_contract import installed_runtime_snapshot

import pytest

from prismaquant import tessera_menu as menu, tessera_render as render
from prismaquant import tessera_runtime_contract as contract
from prismaquant.lane_eligibility import ServingContext, cell_lane_admits
from prismaquant.tessera_formats import get_tessera_family
from prismaquant.tessera_footprint import tessera_exact_bits_for_shape
from prismaquant.tessera_serving_runtime_pin import require_pinned_tessera_runtime

RUNG = "TESSERA_E2M1_K2_R896"
FAMILY = "TESSERA_E2M1_K2"
ROUTED = "routed_moe"


def _packaged_reference():
    """The dense and the routed E2M1 cells the pinned contract carries."""
    snapshot = installed_runtime_snapshot()
    table, formats = snapshot.eligibility, snapshot.published_formats
    dense = [cell for cell in table.cells if cell.family == FAMILY and cell.structure == "dense"]
    routed = [cell for cell in table.cells if cell.family == FAMILY and cell.structure == ROUTED]
    assert dense, "the pinned contract attests no dense E2M1_K2 cell; this test's premise moved"
    assert routed, (
        "the pinned contract attests no routed E2M1_K2 cell (it did from contract "
        "v28); this test's premise moved")
    return table, formats, dense, routed


def _context(cell, structure):
    """The serving context one attested cell answers under, re-targeted at a structure."""
    return ServingContext(platform=cell.platform, structure=structure,
                          residency=cell.residency_modes[0],
                          runtime_image=cell.runtime_image,
                          execution_mode=cell.execution_modes[0])


@pytest.mark.parametrize("dev_pin", [False, True])
def test_routed_e2m1_admits_without_start_state_on_the_pinned_contract(monkeypatch, dev_pin):
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()  # The real pin, no substitute.
    _table, _formats, dense, routed = _packaged_reference()
    dense_context = _context(dense[0], "dense")
    routed_context = _context(routed[0], ROUTED)
    if dev_pin:
        monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "1")

    on_dense = menu.route_admission(RUNG, serving_context=dense_context)
    assert on_dense.attested
    assert on_dense.route_status == dense[0].route_status
    # PQ names the priced contract in its own route vocabulary; the cell names
    # the executed one in the runtime's.  ``route_admission`` already refused
    # if the two did not project to the same (act_bits, group), so what is
    # pinned here is the projection, not the spelling.
    assert on_dense.act_bits == 4

    on_routed = menu.route_admission(RUNG, serving_context=routed_context)
    assert on_routed.attested
    assert on_routed.route_status == routed[0].route_status
    admitted, reason = cell_lane_admits(routed[0], 896, _table.lanes)
    assert admitted and reason == ""
    assert on_routed.readable is True
    assert on_routed.act_bits == on_dense.act_bits == 4


@pytest.mark.parametrize("dev_pin", [False, True])
def test_a_contract_without_routed_e2m1_refuses_it_with_no_producer_change(monkeypatch, dev_pin):
    """Drop the routed E2M1 cells and the same read answers unattested."""
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()
    table, formats, _dense, routed = _packaged_reference()
    routed_context = _context(routed[0], ROUTED)

    def is_routed_e2m1(cell):
        return cell.family == FAMILY and cell.structure == ROUTED

    if dev_pin:
        monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "1")
        parsed = contract.load_tessera_contract()
        kept = tuple(cell for cell in parsed.cells if not is_routed_e2m1(cell))
        assert len(kept) < len(parsed.cells)
        monkeypatch.setattr(menu, "tessera_runtime_contract",
                            lambda: replace(parsed, cells=kept))
    else:
        kept = tuple(cell for cell in table.cells if not is_routed_e2m1(cell))
        assert len(kept) < len(table.cells)
        altered = replace(table, cells=kept)
        payload = json.loads(installed_runtime_snapshot().raw)
        payload["lane_eligibility"]["cells"] = [
            cell for cell in payload["lane_eligibility"]["cells"]
            if not (cell["family"] == FAMILY and cell["structure"] == ROUTED)]
        install_metadata(monkeypatch, payload)

    admission = menu.route_admission(RUNG, serving_context=routed_context)
    assert not admission.attested
    assert admission.route_status == menu.ROUTE_STATUS_UNATTESTED
    # The decoder still accepts the bytes: the rung stays priceable in a
    # readable census, stamped unattested, and the export gate refuses it.
    assert admission.readable is True
    assert admission.requires_serve_flags == ()
    assert admission.act_bits == 4


def test_readable_menu_spans_a4_a8_a16_on_a_routed_unit_and_stamps_the_gap(monkeypatch):
    """A4/A8/A16 side by side on one routed shape, each priced under its own
    activation contract; only the attestation differs, and it is stamped."""
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()
    table, _formats, _dense, routed = _packaged_reference()
    routed_context = _context(routed[0], ROUTED)
    families = [get_tessera_family(name)
                for name in (FAMILY, "TESSERA_E4M3_K1", "TESSERA_BF16_K1")]
    shape = (2048, 4096)  # one GLM-5.3-Flash routed expert projection

    readable = menu.expand_tessera_menu(shape, mode=menu.MENU_READABLE, families=families,
                                        serving_context=routed_context)
    by_name = {rung.format_name: rung for rung in readable}
    a4 = by_name[RUNG]
    a8 = by_name["TESSERA_E4M3_K1_R1024"]
    a16 = by_name["TESSERA_BF16_K1_R1023"]
    assert (a4.admission.act_bits, a8.admission.act_bits, a16.admission.act_bits) == (4, 8, 16)
    assert len({a4.admission.activation_contract, a8.admission.activation_contract,
                a16.admission.activation_contract}) == 3
    # Byte accounting grants no serving admission.
    assert a4.bits_per_param == tessera_exact_bits_for_shape(
        FAMILY, 896, shape, structure=ROUTED) / (shape[0] * shape[1])
    assert a4.admission.attested
    assert a16.route_status == menu.ROUTE_STATUS_UNATTESTED
    assert a16.admission.readable
    routed_e4m3_here = any(
        cell.family == "TESSERA_E4M3_K1" and cell.structure == ROUTED
        and cell.runtime_image == routed_context.runtime_image
        for cell in table.cells)
    assert a8.admission.attested == routed_e4m3_here

    attested = menu.expand_tessera_menu(shape, mode=menu.MENU_ATTESTED, families=families,
                                        serving_context=routed_context)
    names = {rung.format_name for rung in attested}
    assert RUNG in names
    assert "TESSERA_BF16_K1_R1024" not in names
    assert "TESSERA_BF16_K1_R1023" not in names
