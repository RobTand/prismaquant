"""The GLM MTP skeleton must not load checkpoint orchestration (PQ #3215).

Profiles build this skeleton, so importing loading, expert projection or
capture provenance from it pulls the profiles into the pipeline's cycle.
The graph includes function-local imports and package initializers.
"""
from __future__ import annotations

import pytest

from tests.test_import_cycle_wave1_3215 import _reach, graph  # noqa: F401


@pytest.mark.parametrize("higher", [
    "prismaquant.layer_streaming",
    "prismaquant.routed_experts",
    "prismaquant.stage_a_chain_seed",
])
def test_mtp_skeleton_cannot_reach_loading_or_capture(graph, higher):
    assert "prismaquant.glm_mtp" in graph and higher in graph
    assert higher not in _reach(graph, "prismaquant.glm_mtp")


def test_profiles_still_build_the_original_mtp_skeleton(graph):
    assert "prismaquant.glm_mtp" in graph["prismaquant.model_profiles.glm5_next"]


def test_profile_walker_cannot_load_calibration_io(graph):
    assert "prismaquant.calibration_data" not in _reach(graph, "prismaquant.model_walk")
