"""``ServingLaneSpec.route_status_for`` answers from the pin, not from absence.

The defect this module pins (#537). ``route_status_for`` resolved its
eligibility table through ``load_eligibility_table()`` with no
``contract_path``. That loader has had no default table since the Gridbook
lane retired on 2026-09-02, so on a normal checkout the resolver answered
``unattested`` with source ``serving_runtime_contract::absent`` for every lane
on every platform, sm_121 included. The gate still failed closed, but the
structured ``route_status`` principle 9 requires ("never in prose a gate cannot
read") carried the value "nobody handed me a table" -- which is not a fact
about any runtime, and therefore not something principle 14 lets a producer
field say.

The fix reads the tracked serving pin
(``prismaquant/tessera_runtime/tessera_serving_runtime_pin.json``) and the
``runtime_contract.json`` the ``tessera.serving`` package it names actually
packages. So ``absent`` now means one thing only: there is no pin on disk.

Every expectation below is READ off that same pinned file rather than typed
here, so a re-pin that moves a platform, a family or a rung moves this test
with it instead of leaving a literal that fails for the wrong reason. Nothing
here substitutes a table or a verdict: the one monkeypatch in the file points
the pin RESOLVER at a path that does not exist, which is the driver's input,
not its answer.
"""
from __future__ import annotations

import json
from importlib.resources import as_file

import pytest

from prismaquant import serving_profiles as sp
from prismaquant.lane_eligibility import ROUTE_STATUS_UNATTESTED
from prismaquant.tessera_runtime_contract import contract_path
from prismaquant import tessera_serving_runtime_pin as pin_module

NATIVE_STATUSES = ("backed", "backed_with_serve_flag")


def _packaged() -> dict:
    with as_file(contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _cells() -> list[dict]:
    return list(_packaged()["lane_eligibility"]["cells"])


def _pin_version() -> str:
    return pin_module.load_tessera_serving_runtime_pin().version


def _format_row(family: str) -> dict:
    return next(r for r in _packaged()["formats"] if r["family"] == family)


def _name_at(family: str, rung_q256: int) -> str:
    """The family's name at one rate, spelled the contract's own way."""
    return str(_format_row(family)["name_pattern"]).replace(
        "{k}", str(int(rung_q256)))


def _lane(structures: tuple[str, ...]) -> sp.ServingLaneSpec:
    """A lane spec exactly as a serving profile would declare one.

    Constructed rather than loaded because no live serving profile declares
    ``serving_lanes`` today, so this class IS the production entry to the
    resolver; ``ServingProfile.serving_lane_for`` only picks which instance to
    ask. No live profile means no fixture to borrow, not a seam.
    """
    return sp.ServingLaneSpec(
        id="tessera",
        formats=tuple(str(row["family"]) for row in _packaged()["formats"]),
        activation_contract="W16A16",
        fallback_route="none",
        route_status_structures=structures,
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The resolver memoizes the pinned read; other modules may have filled it."""
    sp._reset_eligibility_table_cache()
    yield
    sp._reset_eligibility_table_cache()


# ---------------------------------------------------------------------------
# The measured answer
# ---------------------------------------------------------------------------
def _cell_groups() -> dict[tuple[str, str, str, int], list[dict]]:
    """Cells keyed the way the resolver filters them.

    The lane answer is per (platform, family, structure, rung), and the pinned
    contract ships several cells per key (one per regime, one per residency).
    ``route_status_for`` ranks the covering cells and reports the best one, so
    the census below is grouped the same way rather than asserted cell by cell.
    """
    groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for cell in _cells():
        rungs = [int(r) for r in (cell.get("rungs_q256") or cell.get("rungs") or ())]
        assert rungs, cell["id"]
        for rung in rungs:
            groups.setdefault(
                (str(cell["platform"]), str(cell["family"]),
                 str(cell["structure"]), rung), []).append(cell)
    return groups


def test_every_pinned_cell_resolves_to_its_own_measured_route():
    """The regression: on the pre-#537 resolver every one of these was ``absent``.

    One case per route the pinned contract ships, so the assertion is the
    contract's census and not a hand-picked example. Each must come back with
    a route status the contract itself carries and the id of one of the cells
    that carry it.
    """
    groups = _cell_groups()
    assert groups, "the pinned contract ships no eligibility cells"
    for (platform, family, structure, rung), cells in sorted(groups.items()):
        status, _flags, source = _lane((structure,)).route_status_for(
            _name_at(family, rung), platform=platform)
        statuses = {str(c["route_status"]) for c in cells}
        assert status in statuses, (platform, family, rung, status, source)
        assert source in {
            f"serving_runtime_contract:{_pin_version()}:{c['id']}"
            for c in cells}, source
        assert "absent" not in source, source


def test_the_sm121_control_lane_is_natively_backed():
    """The issue's acceptance case, stated on its own so a regression is legible."""
    sm121 = [k for k in _cell_groups() if k[0] == "sm_121"]
    assert sm121, "the pinned contract ships no sm_121 cell"
    for _platform, family, structure, rung in sorted(sm121):
        status, _flags, source = _lane((structure,)).route_status_for(
            _name_at(family, rung), platform="sm_121")
        assert status in NATIVE_STATUSES, (family, rung, status, source)
        assert "absent" not in source, source


def test_the_published_formats_come_off_the_same_pinned_file():
    """The second half of the same bug, which a table-only fix would leave.

    ``resolve_payload_rung`` needs the pinned ``formats`` rows to turn a format
    NAME into the runtime's family and rate. Called without them it hands back
    the raw name, no cell matches it, and the lane reports ``no_cell`` -- which
    reads as a platform fact and is not one. So ask for a rate the contract's
    own reader range allows but no cell attests: the honest answer is
    ``rung_not_listed`` (the family resolved, the rate is not in a cell), and
    ``no_cell`` there would mean the formats table was never read.
    """
    for (platform, family, structure, _rung), cells in sorted(
            _cell_groups().items()):
        if platform != "sm_121":
            continue
        attested = {
            int(r) for c in _cells() if c["family"] == family
            for r in (c.get("rungs_q256") or c.get("rungs") or ())}
        lo, hi = (int(v) for v in _format_row(family)["reader_rate_range_q256"])
        spare = next((r for r in range(lo, hi + 1) if r not in attested), None)
        if spare is None:  # a family the runtime reads at exactly one rate
            continue
        status, _flags, source = _lane((structure,)).route_status_for(
            _name_at(family, spare), platform="sm_121")
        assert status == ROUTE_STATUS_UNATTESTED, (family, spare, source)
        assert source == (
            f"serving_runtime_contract:{_pin_version()}:rung_not_listed"), source
        return
    pytest.fail("no pinned family reads a rate that no cell attests")


# ---------------------------------------------------------------------------
# `absent` means one thing
# ---------------------------------------------------------------------------
def test_absent_is_reachable_only_when_the_pin_file_is_missing(
        monkeypatch, tmp_path):
    """Point the pin RESOLVER at a path with no file; the verdict is untouched."""
    monkeypatch.setattr(
        pin_module, "tessera_serving_runtime_pin_path",
        lambda: tmp_path / "no_such_tessera_serving_runtime_pin.json")
    sp._reset_eligibility_table_cache()
    cell = next(c for c in _cells() if c["platform"] == "sm_121")
    rung = int((cell.get("rungs_q256") or cell.get("rungs"))[0])
    status, flags, source = _lane((str(cell["structure"]),)).route_status_for(
        _name_at(str(cell["family"]), rung), platform="sm_121")
    assert status == ROUTE_STATUS_UNATTESTED
    assert source == sp.ROUTE_STATUS_SOURCE_NO_PIN, source
    assert "serving_runtime_pin_missing" in source
    assert flags == ()


def test_the_tracked_pin_exists_so_absent_is_not_the_production_answer():
    """Stated once: the branch above is unreachable on a normal checkout."""
    assert pin_module.tessera_serving_runtime_pin_path().exists()
    _table, _formats, refusal = sp._load_pinned_lane_tables()
    assert refusal == "", refusal


def test_no_other_refusal_borrows_the_absent_token():
    """Every miss says WHICH fact was absent; only a missing pin says ``absent``.

    The four non-cell answers a live lane can reach, each asked for on purpose.
    """
    cell = next(c for c in _cells() if c["platform"] == "sm_121")
    rung = int((cell.get("rungs_q256") or cell.get("rungs"))[0])
    name = _name_at(str(cell["family"]), rung)
    structure = str(cell["structure"])
    sources = [
        _lane(())  # a lane that names no attestation
        .route_status_for(name, platform="sm_121")[2],
        _lane((structure,))  # no target platform
        .route_status_for(name, platform=None)[2],
        _lane((structure,))  # a platform the contract ships no cell for
        .route_status_for(name, platform="sm_90")[2],
        _lane(("no_such_structure",))  # a structure class with no cell
        .route_status_for(name, platform="sm_121")[2],
    ]
    assert sources == [
        "lane_declares_no_route_status_source",
        f"serving_runtime_contract:{_pin_version()}:no_target_platform",
        f"serving_runtime_contract:{_pin_version()}:no_cell",
        f"serving_runtime_contract:{_pin_version()}:no_cell",
    ], sources
    assert not any("absent" in s for s in sources), sources
