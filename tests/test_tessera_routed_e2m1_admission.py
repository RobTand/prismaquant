"""Routed E2M1_K2 admission is derived from the pinned contract, never asserted.

Through contract v27 the pinned Tessera contract attested ``TESSERA_E2M1_K2``
at R896 for ``dense`` units only, so the rung a readable-menu census priced on
a routed expert stack resolved ``unattested``.  Contract v28 (Tessera #517's
pin, PrismaQuant #632) publishes two ``routed_moe`` E2M1_K2 cells at q896 on
``sm_121`` -- decode and batch, resident, eager, grade ``route_only``,
``smoke.status: not_recorded`` -- and the same read now admits it on a routed
unit with no producer change.  That flip IS the principle-14 property this file
exists for, observed on a real pin move rather than on a constructed table;
whether ``not_recorded`` evidence should admit a routed-MoE rung is a
principle-9 decision held in #198, not this file's.

The second half keeps it a derivation and not a grant: a contract that drops
the routed E2M1 cells flips the verdict back to ``unattested``, on both the
dev-pin and the packaged-table paths.
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
    """The dense and the routed E2M1 cells the pinned contract carries."""
    table, formats = render._pinned_serving_table()
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
def test_routed_e2m1_is_attested_on_the_pinned_contract(monkeypatch, dev_pin):
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
    # Read off the routed cell, not typed.
    assert on_routed.route_status == routed[0].route_status
    assert on_routed.requires_serve_flags == tuple(sorted(routed[0].requires_serve_flags))
    assert on_routed.readable is True
    # Same wire, same priced A-side contract on both structures.
    assert on_routed.activation_contract == on_dense.activation_contract
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
        monkeypatch.setattr(render, "_pinned_serving_table", lambda: (altered, formats))

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
    a16 = by_name["TESSERA_BF16_K1_R1024"]
    assert (a4.admission.act_bits, a8.admission.act_bits, a16.admission.act_bits) == (4, 8, 16)
    assert len({a4.admission.activation_contract, a8.admission.activation_contract,
                a16.admission.activation_contract}) == 3
    # 3.5 bpp body plus the attested 0.5 bpp UE4M3 block-scale plane.
    assert float(a4.bits_per_param) == pytest.approx(4.0, abs=1e-2)
    # Since contract v28 the routed E2M1 cell attests A4 at this scope; no
    # routed BF16 cell exists, so A16 stays a stamped gap.
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
