"""Bounded paired-trade diagnostic retention (#2286).

Menu, applicability and stdout records keep bounded summaries of priced
paired-rate trades; only the emitted assignment carries the complete paired
arrays and per-expert breakdown. These tests read stored records, never
source text: they fail on the pre-fix retention (every menu option storing
its full trade) and preserve the priced math and refusal verdicts.
"""
import json

from prismaquant import allocator_candidates as ac
from prismaquant.digests import DIRECT_UTF8_STRICT
from prismaquant.model_profiles import DefaultProfile
from test_joint_aura_assignment_diagnostics import _row

LOW = "FP8_E4M3"
HIGH = "FP8_E5M2"

_FULL_ARRAY_KEYS = (
    "difference_per_probe",
    "group_differences",
    "assignment_a",
    "assignment_b",
    "probe_ids",
)


def _trade_rows(values):
    costs, assignment, baseline = {}, {}, {}
    for name, (a, b) in values.items():
        costs[name] = {LOW: _row(name, a, fmt=LOW), HIGH: _row(name, b, fmt=HIGH)}
        assignment[name], baseline[name] = LOW, HIGH
    return costs, assignment, baseline


def _routed_values(dominant=True):
    if dominant:
        return {f"model.layers.5.mlp.experts.{expert}.{role}_proj":
                ([1, 1], [3 if expert == 0 else 1, 3 if expert == 0 else 1])
                for expert in (0, 1) for role in ("gate", "up", "down")}
    return {f"model.layers.5.mlp.experts.{expert}.{role}_proj": ([1, 1], [3, 3])
            for expert in (0, 1) for role in ("gate", "up", "down")}


def _find_full_arrays(node, path=""):
    """Yield paths where a stored record still carries a full diagnostic."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _FULL_ARRAY_KEYS:
                yield f"{path}.{key}" if path else key
            yield from _find_full_arrays(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _find_full_arrays(value, f"{path}[{i}]")


def test_menu_report_stores_summaries_not_full_trades():
    from prismaquant.allocator_solver import Candidate
    values = _routed_values(dominant=False)
    costs, assignment, baseline = _trade_rows(values)
    group = "actual-layer::__packed__"
    stats = {group: {"_packed_group_members": sorted(values)}}
    candidates = {group: [Candidate(LOW, 1, 10, 3.0), Candidate(HIGH, 2, 20, 15.0)]}
    report = {}
    repriced = ac.reprice_paired_candidates(stats, costs, candidates, baseline,
                                            profile=DefaultProfile(), ucb_z=1,
                                            report=report)
    assert [c.fmt for c in repriced[group]] == [LOW, HIGH]
    leaked = [leak for option in report[group].values()
              for leak in _find_full_arrays(option)]
    assert leaked == [], f"menu report retains full diagnostics: {leaked[:5]}"
    # The priced math survives the bound: summaries match a fresh full price.
    for fmt in (LOW, HIGH):
        full = ac.price_paired_rate_trade(
            costs, {m: fmt for m in sorted(values)}, baseline,
            profile=DefaultProfile(), ucb_z=1)
        summary = report[group][fmt]
        for key in ("mean_difference", "paired_standard_error", "hedged_difference",
                    "candidate_point_cost", "predicted_dloss", "ucb_z", "refused",
                    "probe_identity_sha256"):
            assert summary[key] == full[key]
        assert summary["full_trade_sha256"] == DIRECT_UTF8_STRICT.sha256(full)
        assert summary["n_probes"] == len(full["probe_ids"])
        assert summary["refused"] is False
    # The record stays JSON-serializable for the applicability payload.
    json.dumps(report)


def test_menu_summary_preserves_refusal_and_dominance_diagnostics():
    from prismaquant.allocator_solver import Candidate
    values = _routed_values(dominant=True)
    costs, assignment, baseline = _trade_rows(values)
    group = "actual-layer::__packed__"
    stats = {group: {"_packed_group_members": sorted(values)}}
    candidates = {group: [Candidate(LOW, 1, 10, 3.0), Candidate(HIGH, 2, 20, 15.0)]}
    report = {}
    repriced = ac.reprice_paired_candidates(stats, costs, candidates, baseline,
                                            profile=DefaultProfile(), ucb_z=0,
                                            report=report)
    assert [c.fmt for c in repriced[group]] == [HIGH]
    refused = report[group][LOW]
    assert refused["refused"] is True
    layer = refused["routed_layers"]["model.layers.5.mlp.experts"]
    assert layer["refused"] is True
    assert layer["refusal_reason"] == "expert_dominance"
    assert layer["experts"]["0"]["fraction_of_layer_change"] == 1.0
    assert layer["experts"]["0"]["mean_difference"] == -12
    assert report[group][HIGH]["refused"] is False
    leaked = [leak for option in report[group].values()
              for leak in _find_full_arrays(option)]
    assert leaked == []


def test_pricing_entry_point_still_returns_complete_evidence():
    # The bound is on retention, not on arithmetic: the priced trade itself
    # still carries the complete arrays the emitted assignment ships.
    values = _routed_values(dominant=False)
    costs, assignment, baseline = _trade_rows(values)
    trade = ac.price_paired_rate_trade(costs, assignment, baseline,
                                       profile=DefaultProfile(), ucb_z=1)
    assert trade["difference_per_probe"] != []
    assert trade["group_differences"] != {}
    assert trade["routed_layers"]["model.layers.5.mlp.experts"]["experts"]["0"][
        "difference_per_probe"] != []


def test_refusal_stdout_row_drops_arrays_keeps_verdict():
    values = _routed_values(dominant=True)
    costs, assignment, baseline = _trade_rows(values)
    trade = ac.price_paired_rate_trade(costs, assignment, baseline,
                                       profile=DefaultProfile(), ucb_z=0)
    assert trade["refused"] is True
    row = trade["routed_layers"]["model.layers.5.mlp.experts"]
    printed = ac.summarize_paired_routed_layer(row)
    assert list(_find_full_arrays(printed)) == []
    assert printed["refused"] is True
    assert printed["dominant_experts"] == ["0"]
    assert printed["refusal_reason"] == "expert_dominance"
    assert "dominant_experts" in json.dumps(printed)
