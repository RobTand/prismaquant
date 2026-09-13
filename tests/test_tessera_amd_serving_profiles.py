"""The AMD Tessera profiles: a derived menu, and an export that fails closed.

Rob's ruling is that the RDNA3.5 (Strix Halo, ``gfx1151``) lane is Tessera-16
WnA16 only.  The interesting part is WHERE that lives.  It is not a rule in
these profiles: a hardcoded format ban is what principle 1 vetoes, and it would
go stale the day the runtime backs one more family.  It is the pinned
contract's own ``executes: null`` for the two quantized families on that device
-- a measured platform fact, which is principle 9's carve-out and the one thing
allowed to remove a rung.  So the profiles declare no ``format_rules`` at all,
and what this module asserts is that the menu the platform derives EQUALS the
set of families the contract backs there.  A contract that later backs E4M3 on
``gfx1151`` widens the menu with no edit to either profile, and this test
follows it.

The second half is the part a reader is most likely to get wrong. The contract
says ``gfx1151`` EXECUTES ``bf16_unquantized`` for ``TESSERA_BF16_K1``. That is
not a receipt, and this side does not treat it as one: no cell names either AMD
platform, so ``route_status_for`` answers ``unattested`` with source
``serving_runtime_contract:<v>:no_cell`` for every family including the backed
one, and export fails closed without an explicit override. Backing is
permission to PRICE; a cell is permission to ship.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from importlib.resources import as_file

from prismaquant import serving_profiles as sp
from prismaquant import tessera_render as tr
from prismaquant.lane_eligibility import ROUTE_STATUS_UNATTESTED
from prismaquant.tessera_formats import platform_backs

AMD_PROFILES = {
    "tessera_strix_halo_gfx1151": "gfx1151",
    "tessera_research_gfx1201": "gfx1201",
}
SPEC_DIR = Path(sp.__file__).resolve().parent / "serving_profile_specs"


def _packaged() -> dict:
    with as_file(tr.tessera_serving_contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _families() -> tuple[str, ...]:
    return tuple(str(row["family"]) for row in _packaged()["formats"])


def _backed_by_contract(platform: str) -> set[str]:
    entry = _packaged()["lane_eligibility"]["platforms"][platform]["executes"]
    return {family for family, contract in entry.items() if contract is not None}


# ---------------------------------------------------------------------------
# The profiles exist and declare what they are
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile_id,platform", sorted(AMD_PROFILES.items()))
def test_the_profile_is_emulation_only_and_targets_an_exact_device(
        profile_id, platform):
    profile = sp.load_serving_profile(profile_id)
    assert profile.target_platform == platform
    assert profile.emulation_only is True
    assert profile.export_lane is None, (
        "an emulation-only profile declares no export lane; declaring one "
        "would make this a serving claim")
    assert profile.tensor_parallel.world_size == 1
    assert profile_id in sp.serving_profile_names()


@pytest.mark.parametrize("profile_id", sorted(AMD_PROFILES))
def test_the_profile_declares_no_format_ban(profile_id):
    """The menu is derived. A ban here would be the band-aid principle 1 vetoes."""
    payload = json.loads(
        (SPEC_DIR / f"{profile_id}.json").read_text(encoding="utf-8"))
    assert not payload.get("format_rules"), payload.get("format_rules")
    assert not payload.get("formats")
    for family in _families():
        assert family not in json.dumps(payload.get("format_rules", []))


# ---------------------------------------------------------------------------
# The menu equals what the contract backs for the target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile_id,platform", sorted(AMD_PROFILES.items()))
def test_the_menu_equals_the_families_the_contract_backs_there(
        profile_id, platform):
    profile = sp.load_serving_profile(profile_id)
    derived = {
        family for family in _families()
        if platform_backs(family, profile.target_platform)
    }
    assert derived == _backed_by_contract(platform)
    assert derived == {"TESSERA_BF16_K1"}, sorted(derived)


@pytest.mark.parametrize("profile_id,platform", sorted(AMD_PROFILES.items()))
def test_the_capability_gate_is_what_removes_the_quantized_families(
        profile_id, platform):
    """Through the seam the allocator actually consults, not a restatement.

    ``_capability_gate`` is what the candidate builder calls with the
    profile's ``target_platform``; since PrismaQuant #528 it reads the
    contract. So the families that leave this profile's menu leave it because
    the runtime says it has no native route for them there, and the refusal
    says so in words a reader can act on.
    """
    from prismaquant.tessera_allocator import _capability_gate
    from prismaquant.tessera_formats import get_tessera_family

    profile = sp.load_serving_profile(profile_id)
    backed = _backed_by_contract(platform)
    for family in _families():
        legal, detail = _capability_gate(
            get_tessera_family(family), profile.target_platform)
        assert legal is (family in backed), (family, detail)
        if not legal:
            assert "executes: null" in detail, detail


def test_bf16_passthrough_stays_on_the_menu():
    """Principle 11's scalar rung is untouched by any of this."""
    for profile_id in AMD_PROFILES:
        profile = sp.load_serving_profile(profile_id)
        assert profile.check_format(None, "BF16").legal


# ---------------------------------------------------------------------------
# Export fails closed: backing is not a receipt
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", sorted(set(AMD_PROFILES.values())))
def test_every_family_resolves_unattested_no_cell_on_the_amd_targets(platform):
    lane = sp.ServingLaneSpec(
        id="tessera",
        formats=tuple(_families()),
        activation_contract="W16A16",
        fallback_route="none",
        route_status_structures=("dense", "routed_moe"),
    )
    version = _packaged()["versions"]["tessera"]
    for family in _families():
        name = f"{family}_R1792" if family == "TESSERA_BF16_K1" else f"{family}_R896"
        status, flags, source = lane.route_status_for(name, platform=platform)
        assert status == ROUTE_STATUS_UNATTESTED, (family, status, source)
        assert source == f"serving_runtime_contract:{version}:no_cell", source
        assert flags == ()


def test_the_same_lane_still_attests_the_sm121_cells():
    """The control: `no_cell` above is about the platform, not a dead lane."""
    lane = sp.ServingLaneSpec(
        id="tessera",
        formats=tuple(_families()),
        activation_contract="W16A16",
        fallback_route="none",
        route_status_structures=("dense", "routed_moe"),
    )
    status, _flags, source = lane.route_status_for(
        "TESSERA_BF16_K1_R1792", platform="sm_121")
    assert status != ROUTE_STATUS_UNATTESTED, (status, source)
    assert "no_cell" not in source, source


@pytest.mark.parametrize("platform", sorted(set(AMD_PROFILES.values())))
def test_the_contract_declares_the_platform_but_ships_no_cell(platform):
    """Stated once, because both halves above depend on it."""
    lane_block = _packaged()["lane_eligibility"]
    assert platform in lane_block["platforms"]
    assert not [c for c in lane_block["cells"] if c["platform"] == platform]
    assert lane_block["platforms"][platform]["serve_image"] is None
