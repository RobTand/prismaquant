"""Synthetic D41 tables exercise admission mechanics, never measured qualification."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from prismaquant import tessera_lane, tessera_menu

FIXTURE = Path(__file__).parent / "fixtures" / "rung_allowability"
FAMILY = "TESSERA_E4M3_K1"
BUILD = json.loads((FIXTURE / FAMILY / "fixture-t8" / "v0001.json").read_text())["kernel_build"]


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "publication"
    shutil.copytree(FIXTURE, root)
    return root


def _load(root, **kwargs):
    from prismaquant.rung_allowability import load_rung_allowability
    from prismaquant.lane_eligibility import load_published_formats
    return load_rung_allowability(root, format_entry=load_published_formats()[FAMILY],
                                 expected_kernel_build=kwargs.pop("build", BUILD), **kwargs)


def _mutate(root, change):
    path = root / FAMILY / "fixture-t8" / "v0001.json"
    table = json.loads(path.read_text())
    change(table)
    path.write_text(json.dumps(table))


@pytest.mark.parametrize("rung,allowed,reason", [
    (896, True, ""),
    (897, False, "anomaly"),
    (898, False, "missing_measurement"),
    (899, False, "unlisted"),
])
def test_fixture_exclusions_join_existing_producer_admission(publication, rung, allowed, reason):
    table = _load(publication)
    admission = tessera_lane.rung_admission(f"{FAMILY}_R{rung}", allowability={FAMILY: table},
                                           require_allowability=True)
    assert admission.admits(tessera_menu.MENU_RESEARCH) is allowed
    assert reason in admission.detail
    if rung == 898:
        assert table.refusal(rung) == "missing_measurement"
        assert not hasattr(table, "seal")


def test_production_hook_refuses_without_table():
    with pytest.raises(tessera_menu.TesseraMenuError, match="allowability.*required"):
        tessera_lane.rung_admission(f"{FAMILY}_R896", require_allowability=True)


def test_production_candidate_builder_requires_table_even_in_research_menu(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    spec = registry.get_format(f"{FAMILY}_R896")
    stats = {"unit": {"in_features": 256, "out_features": 256, "n_params": 65536}}
    costs = {"unit": {spec.name: {"predicted_dloss": 0.1}}}
    with pytest.raises(tessera_menu.TesseraMenuError, match="allowability.*required"):
        candidates.build_candidates(stats, costs, [spec], target_profile="vllm_packed_moe")


def test_index_selects_current_version_not_directory_order(publication):
    path = publication / FAMILY / "fixture-t8" / "v0001.json"
    table = json.loads(path.read_text())
    table["table_version"] = 2
    table["rungs"][0]["measurement_status"] = "pending"
    table["rungs"][0]["supported"] = None
    (path.parent / "v0002.json").write_text(json.dumps(table))
    assert _load(publication).allows(896)
    index = publication / "index.json"
    payload = json.loads(index.read_text())
    payload["formats"][FAMILY]["kernel_builds"][BUILD["id"]]["current_version"] = 2
    payload["formats"][FAMILY]["kernel_builds"][BUILD["id"]]["versions"]["2"] = {
        "path": f"{FAMILY}/fixture-t8/v0002.json", "table_schema": table["schema"],
        "table_status": table["table_status"]}
    index.write_text(json.dumps(payload))
    assert not _load(publication).allows(896)


@pytest.mark.parametrize("change,match", [
    (lambda t: t.pop("table_version"), "version"),
    (lambda t: t.update(table_version=2), "version"),
    (lambda t: t["kernel_build"].update(source_commit="other"), "kernel_build"),
    (lambda t: t["scope"].update(grid_step_q256=64), "grid|step"),
    (lambda t: t["rungs"].append(t["rungs"][0]), "duplicate"),
    (lambda t: t["rungs"][0]["measurements"][0].update(kernel_time_us=None), "measured|missing"),
    (lambda t: t["rungs"][0]["measurements"][0]["geometry"].update(alignment=None), "missing"),
    (lambda t: t["rungs"][0].update(quality={}), "quality"),
    (lambda t: t["rungs"][0].update(excluded=True, dominating_rung=960), "adjacent|step"),
])
def test_malformed_or_stale_publication_refuses(publication, change, match):
    _mutate(publication, change)
    with pytest.raises(ValueError, match=match):
        _load(publication)


def test_unreadable_current_table_refuses(publication):
    (publication / FAMILY / "fixture-t8" / "v0001.json").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        _load(publication)


def test_observations_do_not_manufacture_exclusion(publication):
    _mutate(publication, lambda t: t["rungs"][0].update(
        observations=[{"kind": "slow"}, {"kind": "missing_census"}]))
    assert _load(publication).allows(896)


def test_table_cannot_bypass_existing_run_table_rule(publication):
    from prismaquant.lane_eligibility import load_published_formats
    from prismaquant.rung_allowability import load_rung_allowability
    row = load_published_formats()[FAMILY]
    row["allowable_rungs"]["excluded_q256"] = [896]
    assert not load_rung_allowability(publication, format_entry=row,
                                     expected_kernel_build=BUILD).allows(896)
