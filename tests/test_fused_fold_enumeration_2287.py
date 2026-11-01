"""Qualify mixed-rate enumeration limits at real fused-group shapes (#2287).

Independent review of #2285 found paired covariance retains complete
combinations and may hit the combination cap on a real fused group. This
file derives GLM fused members and per-member menu sizes from the pinned
contracts and folds the largest real fused group on CPU.

Result: the dense gate/up group (2 members at (12288, 4096)) refuses in
complete-combination mode at 11,082,241 pairs over the 8,000,000 guard.
The cap fails closed and yields no wrong artifact. The routed expert
group fits (261,123 options, 261,121 max step pairs).

Repair before adoption: attach whole-module runtime measurements and
reduce on measured dominance in the measured solver
(``reduce_continuous_menu`` stamps this owner as
``deferred_to_measured_runtime_solver``). Member pre-reduction on
unpaired byte/loss costs is unsound for the paired objective: the paired
price comes from matched probe samples, a different cost function from
the one dominance was computed on. #2282 stays closed.
"""
from __future__ import annotations

import json

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import tessera_lane as lane
from prismaquant import tessera_legal_domain as domain
from prismaquant import tessera_menu as tm
from prismaquant import tessera_runtime_contract as trc
from prismaquant.allocator_solver import Candidate
from prismaquant.lane_eligibility import ServingContext
from test_allocator_sibling_aggregation import _installed_fused_licence

#: Largest real fused-group member shape: dense MLP gate/up
#: (``glm_linear_shapes``: rows = intermediate_size 12288, columns =
#: hidden_size 4096).
DENSE_GATE_UP_SHAPE = (12288, 4096)
#: Routed expert gate/up member shape (moe_intermediate_size 2048).
ROUTED_GATE_UP_SHAPE = (2048, 4096)

#: The 8M refusal is the adoption gate. This file pins it; any move
#: re-qualifies every count below.
GUARD_PAIRS = 8_000_000


def _scope_for(family: str, structure: str) -> ServingContext:
    """One scope read from the contract's own first matching cell."""
    cells = json.loads(trc.contract_path().read_text())[
        "lane_eligibility"]["cells"]
    cell = next(
        c for c in cells
        if c["family"] == family and c["structure"] == structure)
    return ServingContext(
        platform=cell["platform"], structure=structure, residency="resident",
        runtime_image=cell["runtime"]["image"],
        execution_mode=cell["runtime"]["execution_modes"][0])


@pytest.fixture(scope="module")
def dense_menu():
    """Full attested menu at the largest real fused-group member shape."""
    assert DENSE_GATE_UP_SHAPE in domain.GLM53_LINEAR_SHAPES
    return tm.expand_tessera_menu(
        list(DENSE_GATE_UP_SHAPE), mode=tm.MENU_ATTESTED,
        serving_context=_scope_for("TESSERA_BF16_K1", "dense"))


@pytest.fixture(scope="module")
def routed_menu():
    """Full attested menu at the routed expert gate/up member shape."""
    assert ROUTED_GATE_UP_SHAPE in domain.GLM53_LINEAR_SHAPES
    return tm.expand_tessera_menu(
        list(ROUTED_GATE_UP_SHAPE), mode=tm.MENU_ATTESTED,
        serving_context=_scope_for("TESSERA_E4M3_K1", "routed_moe"))


def _candidates(names: list[str], rungs) -> dict[str, list[Candidate]]:
    """Member menus from real byte sizes. Costs stay 0.0: pair counts in
    complete-combination mode do not read costs (sorted, never pruned)."""
    return {
        m: [Candidate(fmt=r.format_name, bits_per_param=r.bits_per_param,
                       memory_bytes=r.memory_bytes, predicted_dloss=0.0)
            for r in rungs]
        for m in names
    }


def test_combination_guard_stays_at_8m():
    assert ac._GROUP_FOLD_MAX_PAIRS == GUARD_PAIRS


def test_glm_fused_members_come_from_the_runtime():
    mapping = lane.glm_fused_sibling_leaf_mapping()
    assert {k: list(v) for k, v in mapping.items()} == {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj",
                             "b_proj", "f_a_proj", "g_a_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }
    licence = _installed_fused_licence()
    assert licence.licence_for("q256") == "per_member"
    assert sorted(licence.shared_fields()) == [
        "body", "columns", "family", "grid", "plane", "structure"]
    assert sorted(licence.per_member_fields()) == ["q256", "rows"]


def test_dense_gate_up_refuses_in_complete_combination_mode(dense_menu):
    """5,123 rungs per member; the BF16 cell alone builds 11,082,241
    pairs over the guard. The fold refuses instead of pruning."""
    by_family: dict[str, int] = {}
    for rung in dense_menu:
        by_family[rung.family] = by_family.get(rung.family, 0) + 1
    assert by_family == {"TESSERA_BF16_K1": 3329,
                         "TESSERA_E4M3_K1": 1793,
                         "TESSERA_E2M1_K2": 1}
    members = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
    cands = _candidates(members, dense_menu)
    report: dict = {}
    with pytest.raises(AssertionError, match=r"11,082,241.*8,000,000 guard"):
        ac.tessera_group_composites(
            members, cands, 2 * 12288 * 4096,
            licence=_installed_fused_licence(), report=report,
            preserve_runtime_frontier=True)


def test_routed_gate_up_fits_in_complete_combination_mode(routed_menu):
    """513 rungs per member; the E4M3 cell builds 261,121 max step
    pairs under the guard and completes with 261,123 options."""
    by_family: dict[str, int] = {}
    for rung in routed_menu:
        by_family[rung.family] = by_family.get(rung.family, 0) + 1
    assert by_family == {"TESSERA_E4M3_K1": 511,
                         "TESSERA_E2M1_K2": 1,
                         "TESSERA_BF16_K1": 1}
    members = ["model.layers.3.mlp.experts.0.gate_proj",
               "model.layers.3.mlp.experts.0.up_proj"]
    cands = _candidates(members, routed_menu)
    report: dict = {}
    options = ac.tessera_group_composites(
        members, cands, 2 * 2048 * 4096,
        licence=_installed_fused_licence(), report=report,
        preserve_runtime_frontier=True)
    assert len(options) == 261123
    e4m3 = report["TESSERA_E4M3_K1[grid=E4M3,body=WINDOW,plane=channel]"]
    assert e4m3["member_menu"] == [511, 511]
    assert e4m3["fold_frontier"] == [511, 261121]
    assert e4m3["options"] == 261121
    assert max(step for cell in report.values()
               if isinstance(cell, dict) and "fold_frontier" in cell
               for step in cell["fold_frontier"]) == 261121 < GUARD_PAIRS


def test_six_member_kda_group_carries_no_tessera_menu():
    """The largest member-count group (6 KDA projections) is pinned, so
    its stock-only menu never reaches the fold and the cap is moot."""
    from prismaquant import format_registry as fr

    assert fr.promotion_class_for("NVFP4") == "NVFP4"
    members = [f"model.layers.0.self_attn.{leaf}" for leaf in
               ("q_proj", "k_proj", "v_proj",
                "b_proj", "f_a_proj", "g_a_proj")]
    cands = {m: [Candidate(fmt="NVFP4", bits_per_param=4.0,
                            memory_bytes=4096 * 4096 // 2,
                            predicted_dloss=1.0)]
             for m in members}
    report: dict = {}
    assert ac.tessera_group_composites(
        members, cands, 6 * 4096 * 4096,
        licence=_installed_fused_licence(), report=report) == []
    assert report == {}
