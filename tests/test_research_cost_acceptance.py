from __future__ import annotations

import json

import pytest

from prismaquant.research_cost_acceptance import (
    RESEARCH_COST_PROVENANCE,
    accepted_cost_provenance,
    enforce_research_export_acknowledgement,
    propagated_cost_provenance,
)


def test_accepted_cost_provenance_reads_back_a_stamped_table():
    # Until 2026-09-25 this was read back from an assembled codebook-ladder
    # table; the assembler was archived with that lane (#1304). The reader
    # is lane-independent.
    manifest = {
        "schema": "prismaquant.research_cost_manifest.v1",
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "assembled_row_count": 1,
    }
    payload = {
        "costs": {"model.layers.0.linear": {"NVFP4": {"output_mse": 1.0}}},
        "provenance": {
            "cost_provenance": RESEARCH_COST_PROVENANCE,
            "research_cost_manifest": manifest,
        },
    }
    assert accepted_cost_provenance(payload) == manifest
    assert accepted_cost_provenance({"costs": {}}) is None
    payload["provenance"]["research_cost_manifest"] = {
        **manifest, "assembled_row_count": 2}
    with pytest.raises(ValueError, match="row count"):
        accepted_cost_provenance(payload)


def test_stamp_propagation_and_export_gate_refusal():
    manifest = {
        "schema": "prismaquant.research_cost_manifest.v1",
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "assembled_row_count": 1,
    }
    selection = propagated_cost_provenance(manifest)
    layer_config = {
        "model.layers.0.linear": "NVFP4",
        "__prismaquant__": {"cost_provenance": selection["cost_provenance"]},
    }
    round_trip = json.loads(json.dumps(layer_config))
    with pytest.raises(ValueError, match="refusing to export"):
        enforce_research_export_acknowledgement(
            round_trip, acknowledged=False, where="test exporter"
        )
    accepted = enforce_research_export_acknowledgement(
        round_trip, acknowledged=True, where="test exporter"
    )
    assert accepted == manifest


def test_export_gate_is_inert_for_unstamped_selection():
    assert propagated_cost_provenance(None) == {}
    assert enforce_research_export_acknowledgement(
        {"model.layers.0.linear": "BF16"},
        acknowledged=False,
        where="test exporter",
    ) is None
