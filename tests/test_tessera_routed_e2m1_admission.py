"""Routed E2M1_K2 admission is derived from the pinned contract, never asserted.

The pinned Tessera contract attests ``TESSERA_E2M1_K2`` at R896 for ``dense``
units only; its ``routed_moe`` cells cover ``TESSERA_E4M3_K1`` alone, on the
routed serving image.  So the rung a readable-menu census prices on a routed
expert stack resolves ``unattested`` on the same read that admits it on a dense
unit -- a serving gap the allocator reports (principle 1), not a rung anything
removes from the menu -- and export fails closed on it exactly as on any
unattested unit.

The second half is what makes this a derivation and not a ban: a contract that
carries routed E2M1 cells flips the verdict with no producer change (principle
14), on both the dev-pin and the packaged-table paths.
"""
from dataclasses import replace

import pytest

from prismaquant import tessera_menu as menu, tessera_render as render
from prismaquant import tessera_runtime_contract as contract
from prismaquant.lane_eligibility import ServingContext
from prismaquant.tessera_formats import get_tessera_family
from prismaquant.tessera_serving_runtime_pin import require_pinned_tessera_runtime

RUNG = "TESSERA_E2M1_K2_R896"
FAMILY = "TESSERA_E2M1_K2"
ROUTED = "routed_moe"


def _packaged_reference():
    """The dense E2M1 cells and the routed cells the pinned contract carries."""
    table, formats = render._pinned_serving_table()
    dense = [cell for cell in table.cells if cell.family == FAMILY and cell.structure == "dense"]
    routed = [cell for cell in table.cells if cell.structure == ROUTED]
    assert dense, "the pinned contract attests no dense E2M1_K2 cell; this test's premise moved"
    assert routed, "the pinned contract attests no routed_moe cell; this test's premise moved"
    assert not [cell for cell in routed if cell.family == FAMILY], (
        "the pinned contract now attests routed E2M1_K2: the gap this test pins is closed, "
        "retire the unattested half and keep the attested one")
    return table, formats, dense, routed


def _context(cell, structure):
    """The serving context one attested cell answers under, re-targeted at a structure."""
    return ServingContext(platform=cell.platform, structure=structure, residency="resident",
                          runtime_image=cell.runtime_image,
                          execution_mode=cell.execution_modes[0])


@pytest.mark.parametrize("dev_pin", [False, True])
def test_routed_e2m1_is_readable_but_unattested_on_the_pinned_contract(monkeypatch, dev_pin):
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
    assert not on_routed.attested
    assert on_routed.route_status == menu.ROUTE_STATUS_UNATTESTED
    # The decoder accepts the bytes: the rung stays priceable in a readable
    # census, stamped unattested, and the export gate is what refuses it.
    assert on_routed.readable is True
    assert on_routed.requires_serve_flags == ()
    # Same wire, same priced A-side contract on both structures: the routed
    # verdict is about attestation, not about a different activation contract.
    assert on_routed.activation_contract == on_dense.activation_contract
    assert on_routed.act_bits == on_dense.act_bits == 4
    # The routed image DOES attest a routed rung: the gap is E2M1's, not the context's.
    assert menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=routed_context).attested


@pytest.mark.parametrize("dev_pin", [False, True])
def test_a_contract_attesting_routed_e2m1_admits_it_with_no_producer_change(monkeypatch, dev_pin):
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()
    table, formats, dense, routed = _packaged_reference()
    routed_context = _context(routed[0], ROUTED)
    # A routed E2M1 cell as the sibling serving work would publish it: the
    # dense E2M1 cell's route, on the routed image, under the routed flags.
    retarget = dict(structure=ROUTED, runtime_image=routed[0].runtime_image,
                    execution_modes=routed[0].execution_modes,
                    residency_modes=routed[0].residency_modes,
                    requires_serve_flags=routed[0].requires_serve_flags)
    if dev_pin:
        monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "1")
        parsed = contract.load_tessera_contract()
        clones = tuple(
            replace(cell, cell_id=cell.cell_id.replace("dense", ROUTED), **retarget)
            for cell in parsed.cells if cell.family == FAMILY and cell.structure == "dense")
        assert {cell.regime for cell in clones} == set(parsed.regimes)
        monkeypatch.setattr(menu, "tessera_runtime_contract",
                            lambda: replace(parsed, cells=parsed.cells + clones))
    else:
        clones = tuple(replace(cell, id=cell.id.replace("dense", ROUTED), **retarget)
                       for cell in dense)
        altered = replace(table, cells=table.cells + clones)
        monkeypatch.setattr(render, "_pinned_serving_table", lambda: (altered, formats))

    admission = menu.route_admission(RUNG, serving_context=routed_context)
    assert admission.attested
    assert admission.route_status == dense[0].route_status
    assert admission.requires_serve_flags == tuple(routed[0].requires_serve_flags)
    assert admission.act_bits == 4
    assert admission.readable is True


def test_readable_menu_spans_a4_a8_a16_on_a_routed_unit_and_stamps_the_gap(monkeypatch):
    """A4/A8/A16 side by side on one routed shape, each priced under its own
    activation contract; only the attestation differs, and it is stamped."""
    monkeypatch.delenv(contract.TESSERA_DEV_PIN_ENV, raising=False)
    require_pinned_tessera_runtime()
    _table, _formats, _dense, routed = _packaged_reference()
    routed_context = _context(routed[0], ROUTED)
    families = [get_tessera_family(name)
                for name in (FAMILY, "TESSERA_E4M3_K1", "TESSERA_BF16_K1")]
    shape = (2048, 4096)  # one GLM-5.3-Flash routed expert projection

    readable = menu.expand_tessera_menu(shape, mode=menu.MENU_READABLE, families=families,
                                        serving_context=routed_context)
    by_name = {rung.format_name: rung for rung in readable}
    a4 = by_name[RUNG]
    a8 = by_name["TESSERA_E4M3_K1_R1024"]
    a16 = by_name["TESSERA_BF16_K1_R1024"]
    assert (a4.admission.act_bits, a8.admission.act_bits, a16.admission.act_bits) == (4, 8, 16)
    assert len({a4.admission.activation_contract, a8.admission.activation_contract,
                a16.admission.activation_contract}) == 3
    # 3.5 bpp body plus the attested 0.5 bpp UE4M3 block-scale plane.
    assert float(a4.bits_per_param) == pytest.approx(4.0, abs=1e-2)
    # The pinned contract's routed cells attest E4M3_K1 only.
    assert a8.admission.attested
    assert a4.route_status == a16.route_status == menu.ROUTE_STATUS_UNATTESTED
    assert a4.admission.readable and a16.admission.readable

    attested = menu.expand_tessera_menu(shape, mode=menu.MENU_ATTESTED, families=families,
                                        serving_context=routed_context)
    names = {rung.format_name for rung in attested}
    assert "TESSERA_E4M3_K1_R1024" in names
    assert RUNG not in names and "TESSERA_BF16_K1_R1024" not in names
