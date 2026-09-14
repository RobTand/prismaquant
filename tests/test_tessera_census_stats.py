"""A packed Fisher probe lands on a Tessera census's per-expert rows, exactly."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.allocator import renormalize_probe_fisher
from prismaquant.allocator_candidates import (
    _has_measured_output_mse, _stats_indicates_packed_expert,
)
from prismaquant.model_profiles.structure import load_structure_spec
from prismaquant.schemas import validate_probe_payload
from prismaquant.tessera_census_stats import (
    DROPPED_FIELDS, META_KEY, CensusStatsError, expand_probe_onto_census,
)
from test_tessera_expert_projection import STACK, _declared, _projection

FMT = "TESSERA_E4M3_K1_R1024"
DENSE = "model.layers.0.mlp.down_proj"
FOREIGN = "model.layers.0.self_attn.q_proj"
EXPERTS = (0, 1, 2)
TOKENS = 64
GATE_UP = f"{STACK}.gate_up_proj"
DOWN = f"{STACK}.down_proj"
STRUCTURE = load_structure_spec("glm5_next")


def _packed(param: str, per: list[float], rows: int, cols: int) -> dict:
    experts = len(per)
    return {
        "h_trace": math.fsum(per), "h_trace_raw": math.fsum(per) * TOKENS,
        "h_trace_per_expert": list(per), "h_trace_per_expert_raw": [v * TOKENS for v in per],
        "h_trace_norm_tokens": TOKENS, "h_w2_sum": 1.0, "h_w2_sum_raw": float(TOKENS),
        "w_max_abs": 0.5, "w_norm_sq": 2.0, "n_params": experts * rows * cols,
        "in_features": cols, "out_features": rows, "num_experts": experts,
        "n_tokens_seen": TOKENS, "route_prob": None, "router_path": None, "expert_id": None,
        "_packed_experts_module": STACK, "_packed_param": param,
        "expert_g_sq_sum": np.ones((experts, cols)), "expert_act_sq_sum": np.ones((experts, cols)),
        "expert_act_absmax": np.ones((experts, cols)), "expert_tokens": np.array([10, 0, 54]),
    }


def _fixture():
    projection = _projection(experts=EXPERTS)
    carried = tep.carried_projection(
        projection, tep.bind_expert_projection(projection, declared=_declared(experts=EXPERTS)),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")
    _source, units, _stack_of = tep.carried_units(carried)
    cell = {"output_mse": 1e-3, "output_mse_measured": True,
            "cost_source": "tessera_campaign_measured"}
    cost = {"costs": {name: {FMT: dict(cell)} for name in [*units, DENSE]},
            "provenance": {tep.PROJECTION_KEY: carried}}
    # _projection: gate/up are 8x4 per expert, down is 4x8.
    probe = {
        "stats": {
            GATE_UP: _packed("gate_up_proj", [0.5, 1.25, 2.0], rows=16, cols=4),
            DOWN: _packed("down_proj", [0.25, 0.0, 3.0], rows=4, cols=8),
            DENSE: {"h_trace": 0.75, "h_trace_raw": 0.75 * TOKENS, "h_trace_norm_tokens": TOKENS,
                    "n_params": 32, "in_features": 8, "out_features": 4,
                    "fisher_row": np.ones(4)},
            FOREIGN: {"h_trace": 2.0, "n_params": 16, "in_features": 4, "out_features": 4},
        },
        "router_counts": {},
        "meta": {"nsamples": 8, "seqlen": 8, "fisher_norm_tokens": TOKENS},
    }
    return probe, cost, units


def test_expanded_rows_are_the_census_roster_and_keep_every_trace():
    probe, cost, units = _fixture()
    before = copy.deepcopy(probe)
    out = expand_probe_onto_census(probe, cost, structure=STRUCTURE)
    stats = out["stats"]

    assert set(stats) == set(cost["costs"])
    assert out["meta"][META_KEY]["dropped_rows"] == [FOREIGN]
    assert out["meta"][META_KEY]["marginals_dropped"] is True
    assert sum(row["n_params"] for row in stats.values()) == 3 * 2 * 32 + 3 * 32 + 32
    assert stats[DENSE]["h_trace"] == 0.75 and "fisher_row" in stats[DENSE]

    by_projection: dict[str, list[str]] = {}
    for name, unit in units.items():
        by_projection.setdefault(unit["projection"], []).append(name)
    gate_up = by_projection["gate_proj"] + by_projection["up_proj"]
    assert len(gate_up) == 6 and len(by_projection["down_proj"]) == 3
    assert math.fsum(stats[n]["h_trace"] for n in gate_up) == probe["stats"][GATE_UP]["h_trace"]
    assert math.fsum(stats[n]["h_trace"] for n in by_projection["down_proj"]) == \
        probe["stats"][DOWN]["h_trace"]
    for name in gate_up:
        expert = units[name]["expert"]
        assert stats[name]["h_trace"] == [0.5, 1.25, 2.0][expert] / 2
        assert (stats[name]["out_features"], stats[name]["in_features"]) == (8, 4)
    assert stats[f"{STACK}.1.w2"]["n_tokens_seen"] == 0

    for name in units:
        row = stats[name]
        assert not any(key.startswith("_packed") for key in row)
        assert not set(DROPPED_FIELDS) & set(row)
        assert row["num_experts"] == 0
        assert not _stats_indicates_packed_expert(row)
        assert _has_measured_output_mse(row, cost["costs"][name][FMT])
    validate_probe_payload(out)

    # The allocator recomputes h_trace from the raw sums; the children must agree.
    renormed = copy.deepcopy(stats)
    renormalize_probe_fisher(renormed, out["meta"])
    assert {n: r["h_trace"] for n, r in renormed.items()} == {n: r["h_trace"] for n, r in stats.items()}
    assert probe["stats"].keys() == before["stats"].keys()


def _unknown_field(probe, cost):
    probe["stats"][GATE_UP]["h_detail"] = 1.0


def _geometry(probe, cost):
    probe["stats"][DOWN]["n_params"] += 1


def _trace_drift(probe, cost):
    probe["stats"][GATE_UP]["h_trace"] *= 1.001


def _missing_dense(probe, cost):
    del probe["stats"][DENSE]


def _missing_stack(probe, cost):
    del probe["stats"][DOWN]


def _unpacked_routed(probe, cost):
    probe["stats"][f"{STACK}.0.w1"] = {"h_trace": 1.0, "n_params": 32}


def _census_expert_missing(probe, cost):
    probe["stats"][DOWN]["num_experts"] = 2
    probe["stats"][DOWN]["h_trace_per_expert"].pop()
    probe["stats"][DOWN]["h_trace_per_expert_raw"].pop()


@pytest.mark.parametrize("mutate,match", [
    (_unknown_field, "does not place"),
    (_geometry, "does not rebuild"),
    (_trace_drift, "per-expert traces sum"),
    (_missing_dense, "dense census rows have no probe row"),
    (_missing_stack, "no packed probe row"),
    (_unpacked_routed, "unpacked row for a projected routed unit"),
    (_census_expert_missing, "per-expert traces sum|covers experts"),
])
def test_rows_that_cannot_be_placed_refuse_by_name(mutate, match):
    probe, cost, _units = _fixture()
    mutate(probe, cost)
    with pytest.raises(CensusStatsError, match=match):
        expand_probe_onto_census(probe, cost, structure=STRUCTURE)
