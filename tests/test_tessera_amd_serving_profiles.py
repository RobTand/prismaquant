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

The second half is the part a reader is most likely to get wrong, and contract
v31 is what makes it concrete. The contract says BOTH AMD platforms EXECUTE
``bf16_unquantized`` for ``TESSERA_BF16_K1``. That was never a receipt, and
this side does not treat it as one -- backing is permission to PRICE, a cell is
permission to ship. At v24 exactly one of the two had acquired cells
(``gfx1201`` shipped two ``TESSERA_BF16_K1`` dense cells at rung
``q256 = 1792``); at v31 the eight dense cells withdraw with the retired
window-GEMV dispatch (Tessera #538, PrismaQuant #699) and NEITHER platform
ships a cell:

* ``gfx1201`` (RDNA4) answers ``unattested`` with ``:no_cell`` for every
  family, and its platform ``serve_image`` returns to ``null`` -- v10's rule
  is a digest iff the platform has at least one cell.
* ``gfx1151`` (Strix Halo) still ships none, and still answers ``:no_cell``
  for every family including the backed one. Nobody here owns the hardware;
  ``docs/strix-halo-tester-protocol.md`` in the Tessera tree is what a receipt
  must satisfy.

Each answer is asserted twice, through both resolvers, because neither alone
says it: ``ServingLaneSpec.route_status_for``, which speaks the ``no_cell``
vocabulary, and ``tessera_render.tessera_attesting_cells``, the predicate
behind ``tessera_menu.route_admission`` and therefore the path production runs.
Since #537 both resolve the pinned contract themselves -- the tracked serving
pin plus the ``runtime_contract.json`` it names -- so no fixture hands either
one a table, and the answers below are the ones the pinned runtime gives.
Each has its own sm_121 control, so ``unattested`` here is never allowed to
mean "nothing was read".
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


def _cells_on(platform: str) -> list[dict]:
    """The packaged cells on one platform. Read, never typed.

    A later contract that mints a ``gfx1151`` cell, or a second rung on
    ``gfx1201``, moves these tests with it instead of leaving them asserting a
    roster the runtime has left behind.
    """
    return [c for c in _packaged()["lane_eligibility"]["cells"]
            if c["platform"] == platform]


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
def pinned_table():
    """Only a cache reset: since #537 the resolver reads the pin itself.

    Until then ``route_status_for`` called ``load_eligibility_table()`` with no
    ``contract_path``, which has had no default table since the Gridbook lane
    retired, so asking it about a real platform meant substituting the contract
    first and the ``no_cell`` assertions below were evidence only with that
    fixture in place. The resolver now resolves the tracked serving pin and the
    packaged ``runtime_contract.json`` on its own, so all that is left to do is
    drop a memo another module in this process may have filled.
    """
    sp._reset_eligibility_table_cache()
    yield
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
def test_every_family_resolves_unattested_no_cell_on_gfx1151(pinned_table):
    """The platform with no receipt: backing prices, it does not ship."""
    assert not _cells_on("gfx1151"), "gfx1151 has cells; this test moved"
    lane = _tessera_lane()
    for family in _families():
        status, flags, source = lane.route_status_for(
            _rung_name(family), platform="gfx1151")
        assert status == ROUTE_STATUS_UNATTESTED, (family, status, source)
        assert source == f"serving_runtime_contract:{_version()}:no_cell", (
            family, source)
        assert flags == ()


def test_gfx1201_attests_nothing_after_the_dense_withdrawal(pinned_table):
    """Contract v31: the withdrawal returns gfx1201 to no-cell.

    The two ``TESSERA_BF16_K1`` dense cells v24 minted withdraw with the
    retired window-GEMV dispatch, so the resolver answers ``unattested``
    with ``:no_cell`` for every family -- the same vocabulary as ``gfx1151``,
    and for the same reason: a cell widens exactly one (family, rung), and
    removing the cell removes the widening.
    """
    assert not _cells_on("gfx1201"), [c["id"] for c in _cells_on("gfx1201")]
    lane = _tessera_lane()
    for family in _families():
        status, flags, source = lane.route_status_for(
            _rung_name(family), platform="gfx1201")
        assert status == ROUTE_STATUS_UNATTESTED, (family, status, source)
        assert source == f"serving_runtime_contract:{_version()}:no_cell", (
            family, source)
        assert flags == ()


def test_a_bf16_rung_on_gfx1201_is_unattested_for_want_of_a_cell(pinned_table):
    """With no cell on the platform there is no rung left to list.

    The control for the test above: at v24, with cells present, an unlisted
    rung answered ``rung_not_listed``; with the cells withdrawn the refusal
    is ``no_cell`` instead, precisely so a reader can tell which fact was
    absent.
    """
    lane = _tessera_lane()
    for rung in ("TESSERA_BF16_K1_R1024", _rung_name("TESSERA_BF16_K1")):
        status, flags, source = lane.route_status_for(
            rung, platform="gfx1201")
        assert status == ROUTE_STATUS_UNATTESTED, (rung, status, source)
        assert source == (
            f"serving_runtime_contract:{_version()}:no_cell"), (rung, source)
        assert flags == ()


def test_the_same_lane_still_attests_the_sm121_dense_cells(pinned_table):
    """The control: `no_cell` above is about the platform, not a dead lane."""
    lane = _tessera_lane()
    status, flags, source = lane.route_status_for(
        _rung_name("TESSERA_E2M1_K2"), platform="sm_121")
    assert status == "backed_with_serve_flag", (status, source)
    assert "no_cell" not in source, source
    assert flags == ("TESSERA_SERVE_MODE=resident|streamed",), flags


def test_the_resolver_reads_the_pin_with_nothing_supplied(pinned_table):
    """#537, from this file's side: no fixture supplies the table any more.

    This used to assert the opposite -- that with no contract in hand the
    resolver answered ``...:absent`` for EVERY platform, sm_121 included -- and
    that was the reason the two tests above needed a fixture at all. It is now
    the defect: an ``unattested`` meaning "nobody handed me a table" is not a
    fact about a runtime, and principle 9's structured ``route_status`` must
    not carry one. ``tests/test_route_status_reads_the_pinned_contract.py``
    owns the full census; this keeps the AMD file honest about its own control.
    """
    status, _flags, source = _tessera_lane().route_status_for(
        _rung_name("TESSERA_E2M1_K2"), platform="sm_121")
    assert status != ROUTE_STATUS_UNATTESTED, (status, source)
    assert "absent" not in source, source


def test_the_live_seam_finds_no_attesting_cell_on_gfx1151():
    """The same fact through the path production actually runs.

    ``tessera_render.tessera_attesting_cells`` is the match predicate behind
    ``tessera_menu.route_admission`` -- the one seam that reads a serving
    contract on Tessera's behalf -- and it resolves the pinned table itself.
    The contract gives ``gfx1151`` no ``serve_image``, so no serving context
    can honestly be built for it; this test LENDS it the release's default
    image and still gets nothing, which is the stronger statement: the absence
    is the absence of a cell, not of an image.
    """
    from prismaquant.lane_eligibility import ServingContext

    # Other files in this process repoint ``tessera_runtime_contract.
    # contract_path`` to a rewritten copy; the table behind this seam is
    # ``lru_cache``d, so read it fresh rather than inherit what ran first.
    tr._pinned_serving_table.cache_clear()
    image = str(_packaged()["versions"]["default_serve_image"])
    assert _packaged()["lane_eligibility"]["platforms"]["gfx1151"][
        "serve_image"] is None
    for family in _families():
        for structure in ("dense", "routed_moe"):
            context = ServingContext(
                platform="gfx1151", structure=structure, residency="resident",
                runtime_image=image, execution_mode="eager")
            assert tr.tessera_attesting_cells(
                _rung_name(family), serving_context=context) == (), (
                    family, structure)


def test_the_live_seam_finds_no_cell_on_gfx1201_either():
    """The withdrawn platform through the path production actually runs.

    ``tessera_render.tessera_attesting_cells`` is the match predicate behind
    ``tessera_menu.route_admission``. At v24 this test found the gfx1201
    cells on their OWN ROCm image; at v31 the platform publishes
    ``serve_image: null``, so no serving context can honestly be built for
    it. This test LENDS it the release default (sm_121's image) and still
    gets nothing for every family -- the absence is the absence of a cell,
    not of an image.
    """
    from prismaquant.lane_eligibility import ServingContext

    tr._pinned_serving_table.cache_clear()
    platforms = _packaged()["lane_eligibility"]["platforms"]
    assert platforms["gfx1201"]["serve_image"] is None
    assert not _cells_on("gfx1201")

    image = str(_packaged()["versions"]["default_serve_image"])
    for family in _families():
        for structure in ("dense", "routed_moe"):
            context = ServingContext(
                platform="gfx1201", structure=structure, residency="resident",
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
        _rung_name("TESSERA_E2M1_K2"), serving_context=context) != ()


def test_the_contract_declares_gfx1151_and_ships_no_cell_for_it():
    """Stated once, because the gfx1151 halves above depend on it."""
    lane_block = _packaged()["lane_eligibility"]
    assert "gfx1151" in lane_block["platforms"]
    assert not _cells_on("gfx1151")
    assert lane_block["platforms"]["gfx1151"]["serve_image"] is None


def test_gfx1201_ships_no_cell_and_serve_image_stays_null():
    """Stated once, because the gfx1201 halves above depend on it.

    v10's rule: a platform's ``serve_image`` is a digest iff it has at least
    one cell, and the image must be one its OWN cells attest. At v23 gfx1201
    had no cell and ``null``; v24 gave it both halves at once; v31 withdraws
    the dense cells and returns it to the null side, which is why the two
    move in the same contract bump in both directions.
    """
    lane_block = _packaged()["lane_eligibility"]
    assert "gfx1201" in lane_block["platforms"]
    assert not _cells_on("gfx1201")
    assert lane_block["platforms"]["gfx1201"]["serve_image"] is None
    # The AMD lane is Tessera-16 only, and the withdrawal does not change
    # that: the platform's ``executes`` map is what prices, and it still
    # backs TESSERA_BF16_K1 alone.
    assert _backed_by_contract("gfx1201") == {"TESSERA_BF16_K1"}
