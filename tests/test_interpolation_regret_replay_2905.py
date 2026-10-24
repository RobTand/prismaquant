"""CPU contract for #2905's portable three-arm replay, not campaign evidence.

The proposed public seam consumes already production-qualified Candidate menus;
its caller must build those through the current contract/currency/shape gates.
These toy registered formats stand in for measured/interpolated rate rungs. They
must never qualify a Tessera wire, substitute for W2, or imply served quality.

Budget convention pinned here: nominal bpp uses the fixed mutable-parameter
count, target bytes = floor(Decimal(bpp) * N / 8), ALL IN (retained spans and
explicit non-tensor reserve included). fullm receives meas's achieved integer
all-in bytes directly, never a rounded bpp. Other reporting conventions may be
added, but cannot silently change these three nominal comparison targets.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import importlib
import importlib.util
import itertools
import re

import pytest

from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from prismaquant.allocator_solver import Candidate
from prismaquant.model_profiles import DefaultProfile


LOW, MID, HIGH = "IQ2_XXS", "IQ4_XS", "Q6_K"
CURRENCY = "synthetic_common_reference_dloss"
TARGETS = ("3.5", "4.5", "6.0")
HEX = re.compile(r"^[0-9a-f]{64}$")


def _replay(**kwargs):
    # Explicit feature failure rather than a collection/import error on main.
    name = "prismaquant.interpolation_regret"
    assert importlib.util.find_spec(name) is not None, (
        "#2905: missing portable production-menu three-arm byte-budget replay"
    )
    module = importlib.import_module(name)
    run = getattr(module, "replay_three_arms", None)
    assert callable(run), "#2905: replay_three_arms public acceptance seam is absent"
    # Give the real aggregation APIs their full reference cost rows, rather
    # than requiring an adapter to reverse-engineer prices from Candidate.
    costs = {n: {c.fmt: {
        "output_mse": 2 * c.predicted_dloss / kwargs["stats"][n]["h_trace"],
        "output_mse_measured": c.activation_pricing == "measured_output_mse",
        "cost_source": ("tessera_campaign_interpolated"
                        if c.activation_pricing == "interpolated_output_mse"
                        else "tessera_campaign_measured"),
        "cost_currency": CURRENCY,
    } for c in cs} for n, cs in kwargs["candidates"].items()}
    formats = [fr.get_format(f) for f in sorted({c.fmt for cs in
               kwargs["candidates"].values() for c in cs})]
    return run(**kwargs, costs=costs, formats=formats)


def _candidate(fmt, loss, *, measured):
    shape = (256, 256)
    size = fr.get_format(fmt).memory_bytes_for_shape(shape)
    return Candidate(
        fmt=fmt, bits_per_param=size * 8 / (256 * 256),
        memory_bytes=size, predicted_dloss=float(loss),
        activation_pricing=("measured_output_mse" if measured
                            else "interpolated_output_mse"),
    )


def _fixture(names=None, profile=None):
    names = names or ["model.layers.0.self_attn.o_proj",
                      "model.layers.1.self_attn.o_proj"]
    stats = {n: {"n_params": 256 * 256, "in_features": 256,
                 "out_features": 256, "h_trace": 1.0} for n in names}
    # A real tradeoff: distributing bytes across the interpolated middle rungs
    # improves loss, but a discrete menu need not spend the same byte total.
    menus = {n: [_candidate(LOW, 100 + i, measured=True),
                 _candidate(MID, 8 + i, measured=False),
                 _candidate(HIGH, 1 + i, measured=True)]
             for i, n in enumerate(names)}
    manifest = fp.SourceByteManifest(
        {n: 2 * 256 * 256 for n in names},
        spans={n: frozenset({n}) for n in names},
    )
    manifest["model.norm"] = 258  # retained, absent from mutable stats
    manifest.spans["model.norm"] = frozenset({"model.norm"})
    return dict(
        stats=stats, candidates=menus, profile=profile or DefaultProfile(),
        currency=CURRENCY, targets=TARGETS, bit_precision=0.00001,
        source_total_bytes=sum(manifest.values()), source_manifest=manifest,
        overhead_reserve_bytes=84, refused_cells=set(), measured_only_units=set(),
    )


def _bytes(plan, assignment):
    priced = fp.assignment_artifact_bytes(
        assignment, plan["stats"], source_total_bytes=plan["source_total_bytes"],
        source_manifest=plan["source_manifest"],
    )
    assert not priced["missing_stats_names"]
    return priced["artifact_payload_bytes"] + plan["overhead_reserve_bytes"]


def _allowed(plan, arm):
    return {
        n: [c for c in cs
            if (n, c.fmt) not in plan["refused_cells"]
            and not (c.activation_pricing == "interpolated_output_mse"
                     and (arm == "meas" or n in plan["measured_only_units"]))]
        for n, cs in plan["candidates"].items()
    }


def _oracle(plan, arm, budget):
    """Only a tiny test oracle; the adapter must use production DP, not this."""
    menus = _allowed(plan, arm)
    names = list(menus)
    feasible = []
    for choices in itertools.product(*(menus[n] for n in names)):
        assignment = {n: c.fmt for n, c in zip(names, choices)}
        size = _bytes(plan, assignment)
        if size <= budget:
            feasible.append((sum(c.predicted_dloss for c in choices), size))
    return min(feasible) if feasible else None


def _validate_pick(plan, row, arm, expected_budget):
    pick = row["arms"][arm]
    assert pick["budget_bytes"] == expected_budget
    allowed = _allowed(plan, arm)
    assert pick["candidate_formats"] == {
        n: sorted(c.fmt for c in cs) for n, cs in allowed.items()
    }
    for field in ("candidate_sha256", "assignment_sha256"):
        assert HEX.fullmatch(pick[field]), field
    assert set(pick["assignment"]) == set(plan["stats"])
    for n, fmt in pick["assignment"].items():
        assert fmt in {c.fmt for c in allowed[n]}
    achieved = _bytes(plan, pick["assignment"])
    assert pick["achieved_bytes"] == achieved <= expected_budget
    assert pick["residual_bytes"] == expected_budget - achieved
    loss = sum(next(c.predicted_dloss for c in plan["candidates"][n]
                    if c.fmt == fmt) for n, fmt in pick["assignment"].items())
    assert pick["predicted_dloss"] == pytest.approx(loss)
    assert pick["status"] == "ok"
    return pick


def test_three_targets_and_integer_fullm_budget_distinguish_reach_from_ranking():
    plan = _fixture()
    report = _replay(**plan)
    assert report["status"] == "surrogate_only_pending_w4"
    assert report["currency"] == CURRENCY
    assert report["eligible_interpolated_count"] == 2
    assert [r["target_bpp"] for r in report["targets"]] == list(TARGETS)
    nparams = sum(s["n_params"] for s in plan["stats"].values())
    saw_non_equal_sizes = saw_strict_ranking_gain = False
    saw_six_decimal_budget_loss = False
    for row in report["targets"]:
        target = int(Decimal(row["target_bpp"]) * nparams / 8)
        assert row["target_bytes"] == target
        full = _validate_pick(plan, row, "full", target)
        meas = _validate_pick(plan, row, "meas", target)
        fullm = _validate_pick(plan, row, "fullm", meas["achieved_bytes"])
        rounded_bpp = f"{Decimal(meas['achieved_bytes']) * 8 / nparams:.6f}"
        lossy_budget = int(Decimal(rounded_bpp) * nparams / 8)
        saw_six_decimal_budget_loss |= lossy_budget != meas["achieved_bytes"]
        # Pin actual constrained selection, not just three plausible JSON rows.
        for arm, pick in (("full", full), ("meas", meas), ("fullm", fullm)):
            oracle = _oracle(plan, arm, pick["budget_bytes"])
            assert pick["predicted_dloss"] == pytest.approx(oracle[0])
        assert row["reach"]["numerator"] == meas["predicted_dloss"]
        assert row["reach"]["denominator"] == full["predicted_dloss"]
        ranking = row["ranking"]
        assert ranking["numerator"] == meas["predicted_dloss"]
        assert ranking["denominator"] == fullm["predicted_dloss"]
        assert ranking["ratio"] == pytest.approx(
            meas["predicted_dloss"] / fullm["predicted_dloss"])
        gap = meas["achieved_bytes"] - fullm["achieved_bytes"]
        assert ranking["byte_gap"] == gap
        assert ranking["equal_size"] is (gap == 0)
        saw_non_equal_sizes |= gap != 0
        saw_strict_ranking_gain |= ranking["ratio"] > 1
    assert saw_six_decimal_budget_loss, "fixture must catch the archived six-decimal bpp roundtrip"
    assert saw_non_equal_sizes, "fixture must exercise common-budget, NOT equal-size regret"
    assert saw_strict_ranking_gain, "fixture must not reproduce measured-only 1.0000 by construction"


def test_refusals_and_measured_wire_restrictions_apply_before_arm_filtering():
    plan = _fixture()
    names = list(plan["stats"])
    # A refused numeric price must never be rescued by interpolation or DP gain.
    plan["refused_cells"] = {(names[0], HIGH)}
    plan["measured_only_units"] = {names[0]}
    for c in plan["candidates"][names[0]]:
        if c.fmt == HIGH:
            c.predicted_dloss = -10000
    before = copy.deepcopy(plan["candidates"])
    report = _replay(**plan)
    assert plan["candidates"] == before, "arm preparation must not mutate the shared menu"
    assert report["eligible_interpolated_count"] == 1
    for row in report["targets"]:
        target = row["target_bytes"]
        meas = _validate_pick(plan, row, "meas", target)
        _validate_pick(plan, row, "full", target)
        _validate_pick(plan, row, "fullm", meas["achieved_bytes"])
        for arm in row["arms"].values():
            assert arm["candidate_formats"][names[0]] == [LOW]


@pytest.mark.parametrize("mode", ["measured_only", "all_interpolation_refused",
                                 "all_interpolation_wire_restricted"])
def test_degenerate_full_menu_is_refused_instead_of_reporting_unit_regret(mode):
    plan = _fixture()
    if mode == "measured_only":
        for n in plan["candidates"]:
            plan["candidates"][n] = [c for c in plan["candidates"][n]
                                     if c.activation_pricing != "interpolated_output_mse"]
    elif mode == "all_interpolation_refused":
        plan["refused_cells"] = {(n, MID) for n in plan["stats"]}
    else:
        plan["measured_only_units"] = set(plan["stats"])
    with pytest.raises(ValueError, match="(?i)(no eligible interpolat|nondegenerate)"):
        _replay(**plan)


def test_infeasible_targets_are_preserved_and_report_exact_all_in_floor():
    plan = _fixture()
    # Retained bytes alone now exceed each of the three immutable targets.
    plan["source_manifest"]["model.embed_tokens.weight"] = 1_000_003
    plan["source_total_bytes"] += 1_000_003
    report = _replay(**plan)
    floor = _bytes(plan, {n: LOW for n in plan["stats"]})
    assert [r["target_bpp"] for r in report["targets"]] == list(TARGETS)
    for row in report["targets"]:
        for arm in ("full", "meas"):
            pick = row["arms"][arm]
            assert pick["status"] == "infeasible"
            assert pick["floor_bytes"] == floor > row["target_bytes"]
            assert pick["floor_gap_bytes"] == floor - row["target_bytes"]
            assert pick.get("assignment") in (None, {})
        assert row["arms"]["fullm"]["status"] == "unavailable"
        assert row["ranking"] is None


class _GroupedProfile(DefaultProfile):
    def packed_expert_format_group(self, name):
        return "model.layers.0.mlp.experts::__packed_format__:gate_proj,up_proj,down_proj" if ".experts." in name else None

    def fused_sibling_group(self, name):
        return "model.layers.1.mlp.gate_up_proj" if name in (
            "model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj") else None


def test_both_production_group_kinds_are_atomic_and_expanded_before_accounting():
    names = [f"model.layers.0.mlp.experts.{e}.{proj}"
             for e in range(2) for proj in ("gate_proj", "up_proj", "down_proj")]
    fused = ["model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj"]
    dense = "model.layers.2.self_attn.o_proj"
    plan = _fixture(names + fused + [dense], _GroupedProfile())
    # Conflicting per-member desires expose per-Linear solving without grouping.
    for n in names[1:] + [fused[1]]:
        for c in plan["candidates"][n]:
            c.predicted_dloss = {LOW: 1000, MID: 500, HIGH: 0}[c.fmt]
    report = _replay(**plan)
    for row in report["targets"]:
        meas = row["arms"]["meas"]
        for label in ("full", "meas", "fullm"):
            pick = _validate_pick(plan, row, label,
                                  meas["achieved_bytes"] if label == "fullm"
                                  else row["target_bytes"])
            assignment = pick["assignment"]
            assert len({assignment[n] for n in names}) == 1
            assert len({assignment[n] for n in fused}) == 1
            assert set(assignment) == set(plan["stats"]), "synthetic group names leaked"
    # Grouping must be explicit evidence, not accidental agreement of leaf picks.
    assert sorted(sorted(g) for g in report["serving_groups"]["packed"]) == [sorted(names)]
    assert sorted(sorted(g) for g in report["serving_groups"]["fused"]) == [sorted(fused)]


def test_candidate_and_assignment_hashes_are_deterministic_and_content_bound():
    plan = _fixture()
    first = _replay(**plan)
    reordered = dict(plan)
    reordered["stats"] = dict(reversed(list(plan["stats"].items())))
    reordered["candidates"] = {n: list(reversed(cs)) for n, cs in
                               reversed(list(plan["candidates"].items()))}
    second = _replay(**reordered)
    for left, right in zip(first["targets"], second["targets"]):
        for arm in ("full", "meas", "fullm"):
            assert left["arms"][arm]["candidate_sha256"] == right["arms"][arm]["candidate_sha256"]
            assert left["arms"][arm]["assignment_sha256"] == right["arms"][arm]["assignment_sha256"]
    changed = copy.deepcopy(plan["candidates"])
    changed[next(iter(changed))][1].predicted_dloss += 0.125
    third = _replay(**dict(plan, candidates=changed))
    a, b = first["targets"][0]["arms"], third["targets"][0]["arms"]
    assert a["full"]["candidate_sha256"] != b["full"]["candidate_sha256"]
    assert a["meas"]["candidate_sha256"] == b["meas"]["candidate_sha256"]
