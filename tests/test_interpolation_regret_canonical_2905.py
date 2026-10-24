"""Production canonical-price API regressions, not W2 or serving evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from prismaquant import allocator_candidates, footprint, format_registry
from prismaquant.interpolation_regret import replay_three_arms
from prismaquant.model_profiles import DefaultProfile
from prismaquant.rung_allowability import CANONICAL_CHORD_SOURCE
from test_canonical_quality_quantity import FAMILY, anchor_rows, owner_for_rows


def canonical_plan(monkeypatch):
    owner = owner_for_rows(monkeypatch)
    stats = {n: {"out_features": 64, "in_features": 128, "n_params": 8192,
                 "h_trace": 9.0, "unit_structure": "dense"} for n in ("u", "v")}
    costs = {n: anchor_rows(currency="served_kl") for n in stats}
    for n, rows in costs.items():
        for row in rows.values():
            row["quality_scope"]["unit"] = n
    formats = [format_registry.get_format(f"{FAMILY}_R{rate}")
               for rate in (768, 896, 1024)]
    built = allocator_candidates.build_candidates(
        stats, costs, formats, target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_m=8)
    for n in stats:
        chord = next(c for c in built.candidates[n] if c.fmt == f"{FAMILY}_R896")
        assert chord.activation_pricing == CANONICAL_CHORD_SOURCE
        assert built.cost_rows[n][chord.fmt]["cost_source"] == CANONICAL_CHORD_SOURCE
        assert chord.predicted_dloss == 0.75
    manifest = footprint.SourceByteManifest(
        {n: 2 * 8192 for n in stats}, spans={n: frozenset({n}) for n in stats})
    return dict(stats=stats, costs=built.cost_rows, candidates=built.candidates,
                formats=formats, profile=DefaultProfile(), currency="served_kl",
                source_total_bytes=sum(manifest.values()), source_manifest=manifest,
                overhead_reserve_bytes=0, refused_cells=frozenset(),
                measured_only_units=frozenset(), bit_precision=0.00001)


@pytest.mark.parametrize("provenance", ["both", "candidate_only", "row_only"])
def test_canonical_chords_are_interpolation_in_production_menus(monkeypatch, provenance):
    plan = canonical_plan(monkeypatch)
    chord = f"{FAMILY}_R896"
    if provenance == "candidate_only":
        for rows in plan["costs"].values():
            rows[chord].pop("cost_source")
    elif provenance == "row_only":
        plan["candidates"] = {n: [replace(c, activation_pricing="") for c in cs]
                              for n, cs in plan["candidates"].items()}
    report = replay_three_arms(**plan)
    assert report["eligible_interpolated_count"] == 2
    for row in report["targets"]:
        assert all(chord in fs for fs in row["arms"]["full"]["candidate_formats"].values())
        assert all(chord not in fs for fs in row["arms"]["meas"]["candidate_formats"].values())
        # This small production fixture's measured anchor is above all three
        # fixed budgets. Preserve that fact, rather than alter the denominator
        # or pretend this is an actual campaign/fullm selection. Successful
        # three-arm selections are exercised by the separate portable tests.
        assert row["arms"]["meas"]["status"] == "infeasible"
        assert row["arms"]["fullm"]["status"] == "unavailable"


def test_canonical_chord_measured_only_and_refusal_exclusions(monkeypatch):
    plan = canonical_plan(monkeypatch)
    chord = f"{FAMILY}_R896"
    plan["measured_only_units"] = frozenset({"u"})
    report = replay_three_arms(**plan)
    assert report["eligible_interpolated_count"] == 1
    for row in report["targets"]:
        assert chord not in row["arms"]["full"]["candidate_formats"]["u"]
        assert chord in row["arms"]["full"]["candidate_formats"]["v"]
    plan["refused_cells"] = frozenset({("v", chord)})
    with pytest.raises(ValueError, match="no eligible interpolated"):
        replay_three_arms(**plan)
    # The same production-qualified input is reusable after both exclusions.
    assert all(any(c.fmt == chord for c in cs) for cs in plan["candidates"].values())
