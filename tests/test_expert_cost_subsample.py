"""Unit tests for the dense-path stratified expert cost subsample.

``_expert_cost_sample_split`` picks a deterministic, evenly spaced sample of
expert ids per (experts-prefix, projection) group; ``_extrapolate_expert_costs``
fills the skipped experts' cost rows with their group's sampled mean so every
expert keeps a cost row (the allocator drops row-less names entirely, which
would silently shrink the DP's bit/disk accounting and serving-unit
membership).
"""
from __future__ import annotations

import pytest

from prismaquant.measure_quant_cost import (
    _expert_cost_sample_split,
    _extrapolate_expert_costs,
)


def _expert_names(layer: int, n_experts: int, proj: str) -> list[str]:
    return [
        f"model.layers.{layer}.mlp.experts.{e}.{proj}"
        for e in range(n_experts)
    ]


def test_sample_split_disabled_measures_everything(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_EXPERT_COST_SAMPLE", raising=False)
    names = set(_expert_names(0, 16, "gate_proj")) | {"model.layers.0.self_attn.q_proj"}
    measure, extrapolate = _expert_cost_sample_split(names)
    assert measure == names
    assert extrapolate == {}


def test_sample_split_is_deterministic_evenly_spaced_and_keeps_non_experts(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_EXPERT_COST_SAMPLE", "4")
    dense = "model.layers.0.self_attn.q_proj"
    experts = _expert_names(0, 16, "down_proj")
    names = set(experts) | {dense}
    measure, extrapolate = _expert_cost_sample_split(names)
    # Deterministic: same call, same answer.
    measure2, extrapolate2 = _expert_cost_sample_split(names)
    assert measure == measure2 and extrapolate == extrapolate2
    # Non-expert rows are always measured.
    assert dense in measure
    # linspace(0, 15, 4).round() -> expert ids 0, 5, 10, 15.
    sampled = sorted(n for n in measure if n != dense)
    assert sampled == sorted(
        f"model.layers.0.mlp.experts.{e}.down_proj" for e in (0, 5, 10, 15))
    # Every skipped expert maps to the sorted sampled names of its group.
    assert set(extrapolate) == set(experts) - set(sampled)
    for skipped, sources in extrapolate.items():
        assert sources == sampled
        assert skipped not in measure


def test_sample_split_leaves_small_groups_alone(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_EXPERT_COST_SAMPLE", "8")
    names = set(_expert_names(0, 8, "up_proj"))
    measure, extrapolate = _expert_cost_sample_split(names)
    assert measure == names
    assert extrapolate == {}


def test_sample_split_groups_by_projection(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_EXPERT_COST_SAMPLE", "2")
    gate = _expert_names(3, 8, "gate_proj")
    down = _expert_names(3, 8, "down_proj")
    measure, extrapolate = _expert_cost_sample_split(set(gate) | set(down))
    # Each (prefix, projection) group samples independently: 2 gate + 2 down.
    assert len(measure) == 4
    assert sum(1 for n in measure if n.endswith("gate_proj")) == 2
    assert sum(1 for n in measure if n.endswith("down_proj")) == 2
    # Skipped rows only extrapolate from their own projection's sample.
    for skipped, sources in extrapolate.items():
        proj = skipped.rsplit(".", 1)[1]
        assert all(s.endswith(proj) for s in sources)


def test_extrapolate_fills_skipped_rows_with_sampled_mean():
    sources = ["e.experts.0.down_proj", "e.experts.7.down_proj"]
    results = {
        sources[0]: {
            "Q4_K": {"weight_mse": 1.0, "output_mse": 2.0},
            "IQ2_XXS": {"weight_mse": 3.0, "output_mse": 5.0,
                        "output_mse_measured": False},
        },
        sources[1]: {
            "Q4_K": {"weight_mse": 3.0, "output_mse": 4.0},
            # IQ2_XXS errored on this expert: excluded from the mean.
            "IQ2_XXS": {"error": "quantize failed"},
        },
    }
    _extrapolate_expert_costs(results, {"e.experts.3.down_proj": sources})
    row = results["e.experts.3.down_proj"]
    assert row["Q4_K"]["weight_mse"] == pytest.approx(2.0)
    assert row["Q4_K"]["output_mse"] == pytest.approx(3.0)
    assert row["Q4_K"]["expert_cost_extrapolated"] is True
    # Only the non-error entry feeds IQ2_XXS; its unmeasured-output marker
    # propagates so downstream pricing knows the mean is weight-mse-derived.
    assert row["IQ2_XXS"]["weight_mse"] == pytest.approx(3.0)
    assert row["IQ2_XXS"]["output_mse_measured"] is False


def test_extrapolate_skips_rows_with_no_usable_sources():
    results = {"e.experts.0.down_proj": {"Q4_K": {"error": "boom"}}}
    _extrapolate_expert_costs(
        results, {"e.experts.1.down_proj": ["e.experts.0.down_proj"]})
    assert "e.experts.1.down_proj" not in results
