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
platform, so the resolver answers ``unattested`` with source
``serving_runtime_contract:<v>:no_cell`` for every family including the backed
one, and export fails closed without an explicit override. Backing is
permission to PRICE; a cell is permission to ship.

That answer is asserted twice, through both resolvers, because neither alone
says it: ``ServingLaneSpec.route_status_for`` speaks the ``no_cell`` vocabulary
but reads no default table (absence, by design, since the Gridbook lane
retired), so it is handed the pinned contract through the module's declared
test seam -- and ``tessera_render.tessera_attesting_cells``, the predicate
behind ``tessera_menu.route_admission`` and therefore the path production runs,
resolves the pinned table itself and finds no cell. Each has its own sm_121
control, so ``unattested`` here is never allowed to mean "nothing was read".
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


def _rung_name(family: str) -> str:
    """The family's own attested rung, spelled the contract's way.

    Read rather than typed: ``name_pattern`` and ``attested_rungs_q256`` both
    come off the pinned file, so a release that moves a rung moves this test
    with it instead of leaving a stale literal that resolves to ``rate=None``
    and fails for the wrong reason.
    """
    row = next(r for r in _packaged()["formats"] if r["family"] == family)
    return str(row["name_pattern"]).replace(
        "{k}", str(int(row["attested_rungs_q256"][0])))


def _version() -> str:
    return str(_packaged()["versions"]["tessera"])


@pytest.fixture
def pinned_table(monkeypatch):
    """Hand ``route_status_for`` the pinned contract through its own seam.

    ``load_eligibility_table()`` with no ``contract_path`` is an honest
    ABSENCE, by design since the Gridbook lane retired: there is no default
    table any more (``lane_eligibility.py`` docstring), and every live caller
    passes the pinned runtime's own file --
    ``tessera_render._pinned_serving_table`` is the one that does it for
    Tessera. ``ServingLaneSpec.route_status_for`` is the resolver that speaks
    the ``serving_runtime_contract:<v>:no_cell`` vocabulary, and it reads the
    table through a per-process cache with no argument, so asking it about a
    real platform means supplying the real table first. That is what this
    fixture does, through the module's declared test seam
    (``_reset_eligibility_table_cache``). It substitutes the CONTRACT, never
    the verdict: the table below is parsed from Tessera's own packaged
    ``runtime_contract.json`` by the same parser the export gate uses.
    """
    from prismaquant import lane_eligibility as lane

    real_table = lane.load_eligibility_table
    real_formats = lane.load_published_formats
    with as_file(tr.tessera_serving_contract_path()) as path:
        table = real_table(_version(), contract_path=path)
        formats = real_formats(_version(), contract_path=path)
    assert table.present, table.absent_reason
    # Both loaders, from the same file. The rung vocabulary is the second half
    # of the same absence: with no contract in hand `resolve_payload_rung`
    # cannot name a family either, so it returns the raw format string, no cell
    # matches it, and the resolver reports `no_cell` for sm_121 too -- the
    # right answer to the wrong question. A test that patched only the table
    # would read that as a platform fact. (`tessera_render` supplies the same
    # pair from the same file for the live path.)
    monkeypatch.setattr(lane, "load_eligibility_table", lambda *a, **k: table)
    monkeypatch.setattr(lane, "load_published_formats", lambda *a, **k: formats)
    sp._reset_eligibility_table_cache()
    yield table
    sp._reset_eligibility_table_cache()


def _tessera_lane() -> "sp.ServingLaneSpec":
    return sp.ServingLaneSpec(
        id="tessera",
        formats=tuple(_families()),
        activation_contract="W16A16",
        fallback_route="none",
        route_status_structures=("dense", "routed_moe"),
    )


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
def test_every_family_resolves_unattested_no_cell_on_the_amd_targets(
        platform, pinned_table):
    lane = _tessera_lane()
    for family in _families():
        status, flags, source = lane.route_status_for(
            _rung_name(family), platform=platform)
        assert status == ROUTE_STATUS_UNATTESTED, (family, status, source)
        assert source == f"serving_runtime_contract:{_version()}:no_cell", (
            family, source)
        assert flags == ()


def test_the_same_lane_still_attests_the_sm121_cells(pinned_table):
    """The control: `no_cell` above is about the platform, not a dead lane."""
    lane = _tessera_lane()
    status, _flags, source = lane.route_status_for(
        _rung_name("TESSERA_BF16_K1"), platform="sm_121")
    assert status != ROUTE_STATUS_UNATTESTED, (status, source)
    assert "no_cell" not in source, source


def test_the_resolver_says_absent_when_nobody_supplies_a_contract():
    """Why the fixture above exists, asserted rather than asserted-in-prose.

    With no contract in hand this resolver answers ``unattested`` with source
    ``...:absent`` for EVERY platform, sm_121 included. That is the designed
    fail-closed default and it is also the reason the two tests above are not
    evidence without the fixture: an ``unattested`` that means "nobody handed
    me a table" must never be read as "the runtime has no cell here".
    """
    sp._reset_eligibility_table_cache()
    try:
        status, _flags, source = _tessera_lane().route_status_for(
            _rung_name("TESSERA_BF16_K1"), platform="sm_121")
    finally:
        sp._reset_eligibility_table_cache()
    assert status == ROUTE_STATUS_UNATTESTED
    assert source.endswith(":absent"), source


@pytest.mark.parametrize("platform", sorted(set(AMD_PROFILES.values())))
def test_the_live_seam_finds_no_attesting_cell_on_the_amd_targets(platform):
    """The same fact through the path production actually runs.

    ``tessera_render.tessera_attesting_cells`` is the match predicate behind
    ``tessera_menu.route_admission`` -- the one seam that reads a serving
    contract on Tessera's behalf -- and it resolves the pinned table itself.
    The contract gives neither AMD platform a ``serve_image``, so no serving
    context can honestly be built for one; this test LENDS it the release's
    default image and still gets nothing, which is the stronger statement:
    the absence is the absence of a cell, not of an image.
    """
    from prismaquant.lane_eligibility import ServingContext

    # Other files in this process repoint ``tessera_runtime_contract.
    # contract_path`` to a rewritten copy; the table behind this seam is
    # ``lru_cache``d, so read it fresh rather than inherit what ran first.
    tr._pinned_serving_table.cache_clear()
    image = str(_packaged()["versions"]["default_serve_image"])
    assert _packaged()["lane_eligibility"]["platforms"][platform][
        "serve_image"] is None
    for family in _families():
        for structure in ("dense", "routed_moe"):
            context = ServingContext(
                platform=platform, structure=structure, residency="resident",
                runtime_image=image, execution_mode="eager")
            assert tr.tessera_attesting_cells(
                _rung_name(family), serving_context=context) == (), (
                    family, structure)


def test_the_live_seam_still_attests_a_sm121_rung():
    """The control for the live seam, so `()` above is about the platform."""
    from prismaquant.lane_eligibility import ServingContext

    tr._pinned_serving_table.cache_clear()
    context = ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=str(_packaged()["versions"]["default_serve_image"]),
        execution_mode="eager")
    assert tr.tessera_attesting_cells(
        _rung_name("TESSERA_BF16_K1"), serving_context=context) != ()


@pytest.mark.parametrize("platform", sorted(set(AMD_PROFILES.values())))
def test_the_contract_declares_the_platform_but_ships_no_cell(platform):
    """Stated once, because both halves above depend on it."""
    lane_block = _packaged()["lane_eligibility"]
    assert platform in lane_block["platforms"]
    assert not [c for c in lane_block["cells"] if c["platform"] == platform]
    assert lane_block["platforms"][platform]["serve_image"] is None
