"""The priced Tessera route, asked about a platform -- and answered from the contract.

Until this module's change ``tessera_serving_route`` was platform-blind and
expressed capability as ``min_capability_sm``: an NVIDIA compute capability,
resolved by the allocator's gate out of the profile's ``target_platform``
string with a regex.  That worked while every target had an SM number.  It has
no answer for ``gfx1151``, and worse, it was a producer ASSERTING what another
runtime executes -- derived from an id rather than read from the runtime's own
table, which is what principle 14 refuses.

Tessera contract v23 publishes the answer directly:
``lane_eligibility.platforms[target].executes[family]`` is the family's own
route contract, or ``null``.  ``null`` is a claim somebody looked: the pinned
runtime has no native route for these bytes on that device.  A platform key
the table does not carry is a THIRD state -- the document declined to answer --
and this reader keeps it apart from ``null``, because reporting an unread
question as a measured refusal is the same error read backwards.

**No new priced route.**  The contract string, the A-side projection and the
terminal format are unchanged; what a target adds is one boolean.  An unbacked
rung stays PRICED and stays on the menu (principle 1): an allocator that
reaches for it is reporting a serving gap, and that signal is the point.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_formats as tf
from prismaquant import tessera_render as tr
from prismaquant.tessera_formats import (
    get_tessera_family, platform_backs, tessera_serving_route,
)

FAMILIES = ("TESSERA_BF16_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1")


def _packaged() -> dict:
    with as_file(tr.tessera_serving_contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _table():
    with as_file(tr.tessera_serving_contract_path()) as path:
        return lane.load_eligibility_table(contract_path=path)


def _executes() -> dict:
    """``{platform: {family: contract-or-None}}``, read off the real file."""
    return {
        key: dict(entry["executes"])
        for key, entry in _packaged()["lane_eligibility"]["platforms"].items()
    }


# ---------------------------------------------------------------------------
# The blind form does not move
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_no_target_platform_leaves_the_route_exactly_as_it_was(family):
    route = tessera_serving_route(family)
    assert route.platform_backed is True
    assert route.target_platform is None
    # Every field a caller prices on is a defaulted addition away from the
    # object this function returned before the platform axis existed.
    stripped = dataclasses.replace(
        route, platform_backed=True, target_platform=None)
    assert stripped == route


# ---------------------------------------------------------------------------
# The table test the issue asks for: per (family, target_platform)
# ---------------------------------------------------------------------------
def test_every_family_on_every_declared_platform():
    """Non-null -> today's route byte-identical; null -> unbacked.

    Driven by the contract rather than by a literal roster, so a platform the
    runtime adds is covered the day it lands and a family it backs later
    widens this test without an edit.
    """
    table = _table()
    executes = _executes()
    assert set(executes) >= {"sm_121", "gfx1151", "gfx1201"}, sorted(executes)
    seen_backed = seen_unbacked = 0
    for platform, per_family in executes.items():
        for family, contract in per_family.items():
            blind = tessera_serving_route(family)
            route = tessera_serving_route(
                family, target_platform=platform, table=table)
            # The PRICED half never moves: same contract string, same A side,
            # same terminal. A platform answers whether it is served, not how.
            assert dataclasses.replace(
                route, platform_backed=True, target_platform=None) == blind, (
                f"{family} on {platform} changed its priced route")
            assert route.target_platform == platform
            assert route.platform_backed is (contract is not None), (
                f"{family} on {platform}: executes={contract!r}")
            assert platform_backs(family, platform, table=table) is (
                contract is not None)
            seen_backed += contract is not None
            seen_unbacked += contract is None
    # Both arms are exercised, or the parametrisation proves nothing.
    assert seen_backed and seen_unbacked, (seen_backed, seen_unbacked)


def test_the_amd_lane_is_tessera_16_only():
    """Rob's ruling, and the contract states it as a measured platform fact."""
    executes = _executes()
    for amd in ("gfx1151", "gfx1201"):
        backed = {f for f, c in executes[amd].items() if c is not None}
        assert backed == {"TESSERA_BF16_K1"}, (amd, sorted(backed))
    assert not {f for f, c in executes["sm_121"].items() if c is None}


def test_an_undeclared_platform_is_unstated_and_fails_closed():
    """A third state, kept apart from ``null`` and still refusing."""
    table = _table()
    for family in FAMILIES:
        assert table.platform_executes(family, "gfx906") == (
            lane.PLATFORM_EXECUTES_UNSTATED)
        assert platform_backs(family, "gfx906", table=table) is False
        assert tessera_serving_route(
            family, target_platform="gfx906", table=table
        ).platform_backed is False


def test_an_absent_table_backs_nothing():
    absent = lane.EligibilityTable(
        present=False, runtime_version="", runtime_commit="",
        contract_sha256="", absent_reason="none packaged")
    for family in FAMILIES:
        assert platform_backs(family, "sm_121", table=absent) is False


# ---------------------------------------------------------------------------
# The priced A side agrees with what the cells execute -- on the BF16 lane
# ---------------------------------------------------------------------------
def test_the_bf16_route_agrees_with_bf16_unquantized():
    """The reason the AMD lane needs no new priced route.

    ``w16a16-bf16-channel`` prices ``act_bits=16, act_group_size=0``; the
    contract's ``bf16_unquantized`` projects to the same pair.  So the route
    PrismaQuant already prices is the route Tessera-16 executes, on sm_121 and
    on an AMD device alike, and the agreement check that guards the currency
    passes rather than being widened for the new platform.
    """
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_runtime_contract import cell_activation_projection

    route = tessera_serving_route("TESSERA_BF16_K1")
    assert route.contract == "w16a16-bf16-channel"
    assert (route.act_bits, route.act_group_size) == (16, 0)
    assert cell_activation_projection("bf16_unquantized") == (16, 0)
    # Through the real seam, not by restating its rule here.
    tm.check_tessera_activation_agreement(
        "TESSERA_BF16_K1_R1792", route, ["bf16_unquantized"])
    # and the same route resolved for the AMD target still agrees: the
    # platform answers backing, never the executed A side.
    amd = tessera_serving_route(
        "TESSERA_BF16_K1", target_platform="gfx1151", table=_table())
    tm.check_tessera_activation_agreement(
        "TESSERA_BF16_K1_R1792", amd, ["bf16_unquantized"])


def test_a_disagreeing_a_side_still_raises():
    """The guard bites; the test above is not passing on a dead check."""
    from prismaquant import tessera_menu as tm

    route = tessera_serving_route("TESSERA_BF16_K1")
    with pytest.raises(tm.TesseraMenuError):
        tm.check_tessera_activation_agreement(
            "TESSERA_BF16_K1_R1792", route, ["fp8_per_token_dynamic"])


# ---------------------------------------------------------------------------
# The gate reads the contract, not an SM number
# ---------------------------------------------------------------------------
def test_the_capability_gate_reads_the_contract():
    from prismaquant.tessera_allocator import _capability_gate

    for family in FAMILIES:
        spec = get_tessera_family(family)
        legal, detail = _capability_gate(spec, "sm_121")
        assert legal, detail
        legal, detail = _capability_gate(spec, "gfx1151")
        assert legal is (family == "TESSERA_BF16_K1"), (family, detail)
        if not legal:
            assert "executes: null" in detail, detail
    spec = get_tessera_family("TESSERA_BF16_K1")
    legal, detail = _capability_gate(spec, "sm_89")
    assert legal is False
    assert "declares no platform" in detail, detail
    legal, detail = _capability_gate(spec, None)
    assert legal is True


def test_the_allocator_no_longer_parses_a_capability_out_of_a_platform_id():
    """The mechanism, not only the verdict.

    ``sm_121`` still resolves legal and ``sm_89`` still refuses E2M1, so the
    verdicts alone cannot tell whether the gate reads the contract or the
    string.  What distinguishes them is that the regex is gone.
    """
    import inspect

    from prismaquant import tessera_allocator

    source = inspect.getsource(tessera_allocator)
    assert "_SM_PLATFORM" not in source
    assert "minimum_capability_sm" not in inspect.getsource(
        tessera_allocator._capability_gate)
