"""Tier-2 counterfactual coupling, floor, and budget contracts."""
from __future__ import annotations

from prismaquant.tier2_per_expert_counterfactual import (
    DecisionUnit,
    RowChoice,
    UnitChoice,
    apply_unmeasured_floor,
    build_decision_units,
    expand_packed_expert_rows,
    expand_solution,
    solve_lambda_bisection,
)


def test_packed_probe_and_cost_expand_to_w13_and_w2_rows():
    stats = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            "h_trace": 10.0,
            "h_trace_per_expert": [4.0, 6.0],
            "n_params": 64,
            "in_features": 4,
            "out_features": 8,
            "num_experts": 2,
            "_packed_experts_module": "model.layers.0.mlp.experts",
            "_packed_param": "gate_up_proj",
        },
        "model.layers.0.mlp.experts.down_proj": {
            "h_trace": 20.0,
            "h_trace_per_expert": [8.0, 12.0],
            "n_params": 32,
            "in_features": 2,
            "out_features": 4,
            "num_experts": 2,
            "_packed_experts_module": "model.layers.0.mlp.experts",
            "_packed_param": "down_proj",
        },
    }
    costs = {"costs": {
        name: {"FP8_CB_K28": {
            "weight_mse": 0.15,
            "weight_mse_per_expert": [0.1, 0.2],
            "output_mse": 0.3,
            "output_mse_measured": True,
        }}
        for name in stats
    }}
    expanded_stats, expanded_costs, children = expand_packed_expert_rows(
        stats, costs)

    assert len(expanded_stats) == 6
    assert len(children) == 2
    assert expanded_stats[
        "model.layers.0.mlp.experts.0.gate_proj"]["h_trace"] == 2.0
    assert expanded_stats[
        "model.layers.0.mlp.experts.0.up_proj"]["n_params"] == 16
    assert expanded_stats[
        "model.layers.0.mlp.experts.1.down_proj"]["h_trace"] == 12.0
    entry = expanded_costs[
        "model.layers.0.mlp.experts.1.down_proj"]["FP8_CB_K28"]
    assert entry["weight_mse"] == 0.2
    assert entry["output_mse_measured"] is False
    assert "output_mse" not in entry
    units = build_decision_units({
        name: (_row("FP8_CB_K28", 1.0, 10),)
        for name in expanded_stats
    })
    assert len(units) == 4  # two experts x (coupled w13 + w2)


def _row(fmt, dloss, payload_bytes):
    return RowChoice(fmt=fmt, dloss=dloss, payload_bytes=payload_bytes)


def test_w1_w3_are_one_common_format_decision():
    rows = {
        "model.layers.0.mlp.experts.7.w1": (
            _row("NVFP4_CB_K14", 3.0, 10),
            _row("FP8_CB_K36", 1.0, 20),
        ),
        "model.layers.0.mlp.experts.7.w3": (
            _row("NVFP4_CB_K14", 1.0, 10),
            _row("FP8_CB_K36", 3.0, 20),
        ),
        "model.layers.0.mlp.experts.7.w2": (
            _row("NVFP4_CB_K14", 2.0, 10),
            _row("FP8_CB_K36", 0.0, 20),
        ),
    }
    units = build_decision_units(rows)
    solution = solve_lambda_bisection(units, budget_bytes=50)
    assignment, _prices = expand_solution(units, solution)
    assert assignment["model.layers.0.mlp.experts.7.w1"] == assignment[
        "model.layers.0.mlp.experts.7.w3"]
    assert assignment["model.layers.0.mlp.experts.7.w2"] == "FP8_CB_K36"


def test_floor_removes_sub_k14_only_from_zero_evidence_experts():
    q_zero = "model.layers.0.mlp.experts.0.w2"
    q_measured = "model.layers.0.mlp.experts.1.w2"
    q_body = "model.layers.0.self_attn.o_proj"
    choices = {
        qname: (
            _row("NVFP4_CB_K12", 1.0, 8),
            _row("NVFP4_CB_K14", 0.5, 10),
        )
        for qname in (q_zero, q_measured, q_body)
    }
    stats = {
        q_zero: {"h_trace": 0.0},
        q_measured: {"h_trace": 1.0},
        q_body: {"h_trace": 0.0},
    }
    floored = apply_unmeasured_floor(choices, stats, enabled=True)
    free = apply_unmeasured_floor(choices, stats, enabled=False)
    assert [choice.fmt for choice in floored[q_zero]] == ["NVFP4_CB_K14"]
    assert {choice.fmt for choice in free[q_zero]} == {
        "NVFP4_CB_K12", "NVFP4_CB_K14"}
    assert {choice.fmt for choice in floored[q_measured]} == {
        "NVFP4_CB_K12", "NVFP4_CB_K14"}
    assert {choice.fmt for choice in floored[q_body]} == {
        "NVFP4_CB_K12", "NVFP4_CB_K14"}


def _unit(name, cheap_loss, expensive_loss):
    return DecisionUnit(
        name=name,
        members=(name,),
        choices=(
            UnitChoice(
                "NVFP4_CB_K14", cheap_loss, 10, (), ((name, cheap_loss),)),
            UnitChoice(
                "FP8_CB_K36", expensive_loss, 20, (), ((name, expensive_loss),)),
        ),
    )


def test_lambda_bisection_and_tidy_never_exceed_exact_budget():
    units = (
        _unit("a", 5.0, 0.0),
        _unit("b", 4.0, 0.0),
        _unit("c", 3.0, 0.0),
    )
    solution = solve_lambda_bisection(
        units, budget_bytes=47, fixed_bytes=7)
    assert solution.exact_bytes <= 47
    assert solution.exact_bytes == 47
    assignment, _prices = expand_solution(units, solution)
    assert sum(fmt == "FP8_CB_K36" for fmt in assignment.values()) == 1


def test_exact_budget_deduplicates_shared_sidecars():
    sidecar = (("codebook.fp8-k36", 3),)
    units = tuple(
        DecisionUnit(
            name=name,
            members=(name,),
            choices=(
                UnitChoice(
                    "NVFP4_CB_K14", 5.0, 10, (), ((name, 5.0),)),
                UnitChoice(
                    "FP8_CB_K36", 0.0, 20, sidecar, ((name, 0.0),)),
            ),
        )
        for name in ("a", "b")
    )
    solution = solve_lambda_bisection(units, budget_bytes=43)
    assert solution.exact_bytes == 43  # 2 * 20 row bytes + one 3-byte sidecar
    assignment, _prices = expand_solution(units, solution)
    assert set(assignment.values()) == {"FP8_CB_K36"}
