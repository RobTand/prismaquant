"""A Tessera ship record carries the artifact's route histogram (PQ #1377).

Principle 12: every published size or quality claim carries the artifact's
route-status and activation-contract histogram, in the same table as the bpp.
The allocator writes that histogram into the recipe's metadata
(``serving_lane_provenance``); the exporter copies its counts onto the card as
``build.route_histogram``; ``verify`` replays their shape and refuses a Tessera
card that carries none.
"""
from __future__ import annotations

import json

import pytest

from prismaquant import allocator_candidates, shipcard
from prismaquant.shipcard import (
    ROUTE_HISTOGRAM_SCHEMA,
    build_shipcard,
    route_histogram_claim,
    verify,
)

#: The shape of the GLM-5.3 scoped allocation's provenance (#1377): every
#: Tessera unit backed behind a serve flag, and the plain-BF16 picks with no
#: declared lane.
PROVENANCE = {
    "schema": "prismaquant.serving_lane_route.v1",
    "target_profile": "tessera",
    "serving_runtime_version": "0.1.0",
    "units_total": 36547,
    "route_status_counts": {"backed_with_serve_flag": 36309, "no_declared_lane": 238},
    "activation_contracts": {"fp8_per_token_dynamic": 36309},
    "activation_pricing_branches": {"measured_output_mse": 36547},
    "by_format": {},
    "by_unit": {"model.layers.0.mlp.down_proj": {"format": "BF16", "route": None}},
}


def _card(tmp_path, *, lane="tessera", build=None):
    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return build_shipcard(model_dir, build=build or {}, lane=lane)


def _route_problems(card):
    return [p for p in verify(card) if "route_histogram" in p]


def test_the_claim_carries_the_counts_and_never_the_per_unit_rows():
    claim = route_histogram_claim(PROVENANCE)
    assert claim == {
        "schema": ROUTE_HISTOGRAM_SCHEMA,
        "units_total": 36547,
        "route_status_counts": {"backed_with_serve_flag": 36309, "no_declared_lane": 238},
        "activation_contracts": {"fp8_per_token_dynamic": 36309},
    }


def test_a_recipe_without_provenance_has_no_claim():
    assert route_histogram_claim(None) is None
    assert route_histogram_claim({}) is None


def test_a_tessera_card_without_the_route_histogram_does_not_verify(tmp_path):
    card = _card(tmp_path)
    assert _route_problems(card), verify(card)


def test_a_tessera_card_with_the_route_histogram_verifies_it(tmp_path):
    card = _card(tmp_path, build={"route_histogram": route_histogram_claim(PROVENANCE)})
    assert _route_problems(card) == []


def test_the_obligation_survives_erasing_the_lane(tmp_path):
    card = _card(tmp_path, lane=None, build={"export_container": "tessera"})
    assert _route_problems(card)


@pytest.mark.parametrize("damage", [
    lambda h: h.update(schema="other"),
    lambda h: h.update(units_total=1),
    lambda h: h["route_status_counts"].update(backed_with_serve_flag=0),
    lambda h: h["activation_contracts"].update(fp8_per_token_dynamic=10 ** 6),
    lambda h: h.update(route_status_counts=[]),
])
def test_a_malformed_route_histogram_is_refused(tmp_path, damage):
    histogram = json.loads(json.dumps(route_histogram_claim(PROVENANCE)))
    damage(histogram)
    card = _card(tmp_path, lane=None, build={"route_histogram": histogram})
    assert _route_problems(card)


def test_a_native_card_owes_no_histogram_until_its_allocation_writes_one(tmp_path):
    # #1387: native allocations write no serving_lane_provenance yet.
    card = _card(tmp_path, lane=None)
    assert _route_problems(card) == []


def test_the_report_answers_the_route_question_once():
    report = allocator_candidates.selection_serving_lane_provenance(
        {"a": "BF16", "b": "BF16"}, target_profile=None)
    for retired in ("units_on_backed_fused_mid_m_lane", "units_on_fallback_route",
                    "units_without_declared_lane", "route_status_attested",
                    "selected_rungs_fused_mid_m_backed",
                    "selected_rungs_on_fallback_route"):
        assert retired not in report
    assert report["route_status_counts"] == {"no_declared_lane": 2}
    assert shipcard.route_histogram_claim(report)["units_total"] == 2
