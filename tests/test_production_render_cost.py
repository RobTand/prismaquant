from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import build_candidates
from prismaquant.production_render_cost import (
    synthesize_production_render_cost_payload,
)
from prismaquant.production_weight_cache import ProductionWeightCache


def _cache_with_scores() -> ProductionWeightCache:
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True, "joint_scale_opt": True},
        metadata={
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "records": {
                    "layers.0.q_proj|NVFP4": {
                        "qname": "layers.0.q_proj",
                        "format": "NVFP4",
                        "metric": "output_mse",
                        "score": 0.5,
                        "score_sum": 12.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                    "layers.0.q_proj|MXFP8_E4M3": {
                        "qname": "layers.0.q_proj",
                        "format": "MXFP8_E4M3",
                        "metric": "output_mse",
                        "score": 0.25,
                        "score_sum": 6.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                },
            },
        },
    )


def _split_cache(*, identity: str | None = "f" * 64) -> ProductionWeightCache:
    records = {
        "layers.0.low|NVFP4": {
            "qname": "layers.0.low",
            "format": "NVFP4",
            "metric": "output_mse",
            "score": 1.0,
            "score_sum": 4.0,
        },
    }
    return ProductionWeightCache(
        weights={},
        levers={},
        metadata={
            "format_plan_identity_sha256": identity,
            "render_scores": {"records": records},
        },
    )


def _split_baseline() -> dict:
    formats = ["NVFP4", "BF16"]
    return {
        "formats": formats,
        "costs": {
            "layers.0.low": {fmt: {"predicted_dloss": 99.0} for fmt in formats},
        },
    }


def test_render_cost_refuses_a_cache_built_under_a_format_plan():
    """The source-class format plan was archived with the codebook lane
    (#1345). Pricing a plan-scoped cache without its plan would price rows
    the plan withheld, so a non-null plan identity refuses."""
    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane") as info:
        synthesize_production_render_cost_payload(
            _split_cache(), _split_baseline())
    assert "production cache metadata" in str(info.value)


@pytest.mark.parametrize("part", ["provenance", "meta"])
def test_render_cost_refuses_a_baseline_priced_under_a_format_plan(part):
    baseline = _split_baseline()
    baseline[part] = {"source_format_plan_identity_sha256": "e" * 64}
    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane") as info:
        synthesize_production_render_cost_payload(
            _split_cache(identity=None), baseline)
    assert f"baseline cost {part}" in str(info.value)


def test_render_cost_keeps_the_null_plan_identity_it_always_wrote():
    payload = synthesize_production_render_cost_payload(
        _split_cache(identity=None), _split_baseline())
    assert payload["meta"]["source_format_plan_identity_sha256"] is None
    assert "source_format_plan_identity_sha256" not in payload["provenance"]
    assert set(payload["costs"]["layers.0.low"]) == {"NVFP4", "BF16"}


def test_production_render_cost_uses_render_score_directly():
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "MXFP8_E4M3": {"output_mse": 88.0},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.o_proj": {
                "NVFP4": {"predicted_dloss": 4.0},
                "MXFP8_E4M3": {"predicted_dloss": 2.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )

    q = cost["costs"]["layers.0.q_proj"]
    assert q["NVFP4"]["predicted_dloss"] == 12.0
    assert q["NVFP4"]["output_mse_measured"] is False
    assert q["NVFP4"]["cost_source"] == "production_render_score"
    assert q["MXFP8_E4M3"]["predicted_dloss"] == 6.0
    assert q["BF16"]["predicted_dloss"] == 0.0

    o = cost["costs"]["layers.0.o_proj"]
    assert o["NVFP4"]["predicted_dloss"] == 4.0
    assert o["NVFP4"]["cost_source"] == "fallback_baseline"
    assert cost["meta"]["render_score_entries"] == 2
    assert cost["meta"]["fallback_entries"] == 2


def test_production_render_cost_bypasses_h_trace_proxy_in_allocator():
    baseline = {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }
    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )
    stats = {
        "layers.0.q_proj": {
            "h_trace": 1000.0,
            "out_features": 32,
            "in_features": 32,
            "n_params": 1024,
        },
    }

    candidates = build_candidates(
        stats,
        cost["costs"],
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
    )
    by_fmt = {cand.fmt: cand for cand in candidates["layers.0.q_proj"]}

    assert by_fmt["NVFP4"].predicted_dloss == 12.0
    assert by_fmt["BF16"].predicted_dloss == 0.0


def test_production_render_cost_can_reject_weight_mse_fallbacks():
    cache = _cache_with_scores()
    cache.metadata["render_scores"]["records"]["layers.0.q_proj|NVFP4"][
        "metric"
    ] = "weight_mse"
    baseline = {
        "formats": ["NVFP4"],
        "costs": {"layers.0.q_proj": {"NVFP4": {"predicted_dloss": 4.0}}},
    }

    with pytest.raises(ValueError, match="non-output metrics"):
        synthesize_production_render_cost_payload(
            cache,
            baseline,
            require_output_metric=True,
        )


# --------------------------------------------------------------------------
# R14 — calibration identity propagation
# --------------------------------------------------------------------------

def test_render_cost_inherits_calib_hash_from_the_cache():
    cache = _cache_with_scores()
    cache.metadata["calib_hash"] = "cachehash"
    payload = synthesize_production_render_cost_payload(
        cache, {"costs": {}, "formats": ["NVFP4"]})
    assert payload["meta"]["calib_hashes"] == ["cachehash"]
    assert payload["meta"]["calib_hash"] == "cachehash"


def test_render_cost_unions_cache_and_baseline_hashes():
    cache = _cache_with_scores()
    cache.metadata["calib_hash"] = "cachehash"
    payload = synthesize_production_render_cost_payload(
        cache,
        {"costs": {}, "formats": ["NVFP4"],
         "meta": {"calib_hashes": ["baselinehash"]}},
    )
    assert payload["meta"]["calib_hashes"] == ["baselinehash", "cachehash"]
    # Ambiguous single-draw identity -> None, so a downstream reader cannot
    # mistake a two-draw cost table for one draw.
    assert payload["meta"]["calib_hash"] is None


def test_render_cost_stays_inert_on_pre_r14_artifacts():
    payload = synthesize_production_render_cost_payload(
        _cache_with_scores(), {"costs": {}, "formats": ["NVFP4"]})
    assert payload["meta"]["calib_hashes"] == []
    assert payload["meta"]["calib_hash"] is None


def test_render_cost_refuses_a_retired_codebook_rung():
    # A stale cost.pkl naming a retired codebook rung (the retired codebook
    # lane, archived 2026-09-25, #1304) refuses; it is neither dropped from
    # the menu nor priced from a leftover cache tensor.
    fmt = "NVFP4_CB_K16"
    qname = "layers.0.q_proj"
    cache = ProductionWeightCache(
        weights={(qname, fmt): torch.ones(2, 256)},
        levers={},
        metadata={"render_scores": {"records": {}}},
    )
    baseline = {
        "formats": [fmt],
        "costs": {qname: {fmt: {"predicted_dloss": 9.0}}},
    }

    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
        synthesize_production_render_cost_payload(cache, baseline)
