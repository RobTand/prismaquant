import hashlib
import itertools
import json
import pickle
import random
import sys
from dataclasses import replace
from importlib.util import find_spec

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator, format_registry, tessera_rate_surface as surface
from prismaquant.allocator_candidates import (
    aggregate_packed_serving_groups,
    build_candidates,
    expand_packed_group_assignment,
    packed_serving_group_members,
    reduce_continuous_menu,
)
from prismaquant.allocator_solver import legal_formats_from_candidates, solve_with_promotion
from prismaquant.assignment_space import AggregationPolicy, AssignmentSpace
from prismaquant.model_profiles import DefaultProfile
from prismaquant.tessera_formats import TesseraFormatError


_FIXTURE = json.loads('''{"cost_row_fields":["predicted_dloss"],"formats":["NVFP4","FP8_E4M3"],"packed":{"dense":"model.layers.0.self_attn.o_proj","dense_losses":[20.0,1.0],"group_key":"model.layers.0.mlp.experts::packed","h_trace":1.0,"member_losses":[5.0,4.0],"members":["model.layers.0.mlp.experts.0.down_proj","model.layers.0.mlp.experts.1.down_proj"],"shape":[128,128]},"solve":{"allow_default_profile":true,"bit_precision":0.0001,"byte_budget":43016,"overshoot_tolerance":0.0,"profile":"default","tight_byte_budget":43015},"source":{"dtype":"bfloat16","immutable_shapes":{"lm_head.weight":[16,16],"model.embed_tokens.weight":[16,16]},"value":0.0},"transfer":{"budget_sensitivity":[0.5,0.75,1.0,1.25],"byte_budget":2100,"currency":"fixture_output_mse","draws":50,"experts":[0,1,2],"max_regret_pct":0.1,"projections":["down_proj"],"reference_q256":960,"sampled_experts":[0,1,2],"sampled_multipliers":{"1088":0.25,"832":4.0},"seed":7,"stacks":{"a":[1.0,2.0,4.0],"b":[2.0,4.0,8.0]},"unit_bytes":{"1088":1100,"832":700,"960":1000},"weights":null},"truths":{"base":[[7.0,1.0],[11.0,1.0]],"changed":[[20.0,1.0],[2.0,1.0]],"frontier_flip":[[1.0,7.0],[11.0,1.0]]},"units":{"model.layers.0.self_attn.o_proj":{"h_trace":1.0,"losses":{"FP8_E4M3":1.0,"NVFP4":7.0},"shape":[128,128]},"model.layers.1.self_attn.o_proj":{"h_trace":1.0,"losses":{"FP8_E4M3":1.0,"NVFP4":11.0},"shape":[256,128]}}}''')
_FORMATS = _FIXTURE["formats"]
_SPECS = [format_registry.get_format(fmt) for fmt in _FORMATS]
_RANK = {fmt: i for i, fmt in enumerate(_FORMATS)}
_NAMES = tuple(_FIXTURE["units"])
_TRANSFER = _FIXTURE["transfer"]


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((data + "\n").encode("utf-8")).hexdigest()


def _stats(shape, h_trace):
    out_features, in_features = shape
    return dict(out_features=out_features, in_features=in_features,
                n_params=out_features * in_features, h_trace=h_trace)


def _costs(names, losses):
    return {name: {fmt: {"predicted_dloss": loss} for fmt, loss in zip(_FORMATS, row)}
            for name, row in zip(names, losses)}


def _scalar_inputs(case="base"):
    return ({n: _stats(row["shape"], row["h_trace"]) for n, row in _FIXTURE["units"].items()},
            _costs(_NAMES, _FIXTURE["truths"][case]))


def test_cli_scalar_projection(monkeypatch, tmp_path):
    assert _digest(_FIXTURE) == "28360f4cee218d2ab0850d3015050729c130ec20df3db4f8827ea8960ba040b2"
    monkeypatch.chdir(tmp_path)
    source = _FIXTURE["source"]
    shapes = {f"{n}.weight": row["shape"] for n, row in _FIXTURE["units"].items()}
    shapes.update(source["immutable_shapes"])
    (tmp_path / "model").mkdir()
    save_file({n: torch.full(shape, source["value"], dtype=getattr(torch, source["dtype"]))
               for n, shape in shapes.items()}, "model/model.safetensors")
    stats, _ = _scalar_inputs()
    (tmp_path / "probe.pkl").write_bytes(pickle.dumps({"stats": stats, "meta": {"model": "model"}}))
    projection = {}
    for case, formats, payload, cap, bpp in (
        ("base", _FORMATS, 43016, 43016, 7.001302083333333),
        ("changed", _FORMATS[::-1], 35336, 43016, 5.751302083333333),
        ("frontier_flip", _FORMATS, 43016, 43016, 7.001302083333333),
        ("tight", _FORMATS[::-1], 35336, 43015, 5.751302083333333),
    ):
        _, costs = _scalar_inputs("base" if case == "tight" else case)
        (tmp_path / "cost.pkl").write_bytes(pickle.dumps({"costs": costs, "meta": {"formats": _FORMATS}}))
        monkeypatch.setattr(sys, "argv", [
            "allocator", "--probe", "probe.pkl", "--costs", "cost.pkl",
            "--formats", ",".join(_FORMATS), "--bit-precision", "0.0001",
            "--overshoot-tolerance", "0.0", "--allow-default-profile",
            "--target-bits", repr(cap * 8 / 49152), "--layer-config", f"{case}.json",
            "--pareto-csv", f"{case}.csv",
        ])
        result = allocator.main()
        layer = json.loads((tmp_path / f"{case}.json").read_text())
        meta = layer["__prismaquant__"]
        projection[case] = {
            "return": result,
            "assignment": {n: "FP8_E4M3" if layer[n]["bits"] == 8 else "NVFP4" for n in _NAMES},
            "body_payload_bytes": int(meta["body_assignment_payload_bits_total"] / 8),
            "body_params": meta["body_assignment_quantizable_params"],
            "achieved_bits": meta["achieved_bits"], "byte_budget": cap,
            "solver_contract": meta["solver_contract"],
            "global_optimality_claimed": meta["global_optimality_claimed"],
        }
        assert projection[case] == {
            "return": None, "assignment": dict(zip(_NAMES, formats)),
            "body_payload_bytes": payload, "body_params": 49152,
            "achieved_bits": bpp, "byte_budget": cap,
            "solver_contract": "additive_candidate_proposal_then_exact_assignment_filter",
            "global_optimality_claimed": False,
        }
    assert _digest(projection) == "31fae347479b214329deaf5009473eef232ee1c2d8ac0358526b4a7339b7e5ee"


def test_scalar_prices_and_greedy_oracle():
    stats, costs = _scalar_inputs()
    candidates = build_candidates(stats, costs, _SPECS).candidates
    space = AssignmentSpace(
        stats=stats, accounting_stats=stats, fixed_stats={}, fixed_formats={},
        aggregation=AggregationPolicy(False, False), profile=DefaultProfile(),
        format_rank=_RANK, legal_formats=legal_formats_from_candidates(candidates),
    )
    a = _NAMES[0]
    assert next(c.memory_bytes for c in candidates[a] if c.fmt == "NVFP4") == 9216
    assert space.bits_total({a: "NVFP4"}) / 8 == 9224
    scores = {n: {i: costs[n][fmt]["predicted_dloss"] for i, fmt in enumerate(_FORMATS)}
              for n in _NAMES}
    prices = {n: {i: int(space.bits_total({n: fmt}) / 8) for i, fmt in enumerate(_FORMATS)}
              for n in _NAMES}
    pairs = [(sum(prices[n][i] for n, i in zip(_NAMES, indices)),
              sum(scores[n][i] for n, i in zip(_NAMES, indices)))
             for indices in itertools.product(range(2), repeat=2)]
    assert pairs == [(27664, 18.0), (43016, 8.0), (35336, 12.0), (50688, 2.0)]
    cap = _FIXTURE["solve"]["byte_budget"]
    assert min(loss for size, loss in pairs if size <= cap) == 8.0
    greedy = surface._greedy_allocate(scores, prices, cap)
    assert greedy == dict(zip(_NAMES, (1, 0)))
    assert sum(scores[n][i] for n, i in greedy.items()) == 12.0


def test_packed_additive_compatibility():
    fixture = _FIXTURE["packed"]
    members, dense = fixture["members"], fixture["dense"]

    class PackedProfile(DefaultProfile):
        def packed_expert_format_group(self, name):
            return fixture["group_key"] if name in members else None

    profile = PackedProfile()
    names = [*members, dense]
    stats = {n: _stats(fixture["shape"], fixture["h_trace"]) for n in names}
    costs = _costs(names, [fixture["member_losses"]] * len(members) + [fixture["dense_losses"]])
    candidates = build_candidates(
        stats, costs, _SPECS, bit_precision=0.0001,
        defer_menu_reduction=packed_serving_group_members(stats, profile),
    ).candidates
    stats, _, candidates = aggregate_packed_serving_groups(stats, costs, _SPECS, candidates, profile)
    candidates = reduce_continuous_menu(candidates, stats, bit_precision=0.0001)
    for cap, bits, member_format in ((35328, 5.75, "NVFP4"), (50688, 8.25, "FP8_E4M3")):
        assignment, achieved = solve_with_promotion(
            stats, candidates, cap * 8 / 49152, dict(zip(_FORMATS, _SPECS)), _RANK, 0.0001,
            profile=profile, overshoot_tolerance=0.0,
        )
        assert expand_packed_group_assignment(assignment, stats) == {
            **dict.fromkeys(members, member_format), dense: "FP8_E4M3",
        }
        assert achieved == bits


def _records():
    projection, = _TRANSFER["projections"]
    return {name: surface.StackRateSample(
        stack=name, projections=(projection,), experts=tuple(_TRANSFER["experts"]),
        sampled_experts=tuple(_TRANSFER["sampled_experts"]),
        reference_q256=_TRANSFER["reference_q256"], currency=_TRANSFER["currency"],
        reference_mse={e: {projection: v} for e, v in enumerate(vector)},
        sampled_mse={int(rate): {e: {projection: v * multiplier} for e, v in enumerate(vector)}
                     for rate, multiplier in _TRANSFER["sampled_multipliers"].items()},
        weights=_TRANSFER["weights"],
    ) for name, vector in _TRANSFER["stacks"].items()}


def test_transfer_diagnostic_compatibility():
    records = _records()
    for name, other, prediction in (("a", "b", (28.0, 1.75)), ("b", "a", (56.0, 3.5))):
        law = surface.fit_stack_transfer_law(records, hold_out=name)
        assert law.pooled_stacks == (other,)
        assert law.slope == {r: {"down_proj": 1.0} for r in (832, 1088)}
        assert law.residual_sd_log2 == {r: {"down_proj": 0.0} for r in (832, 1088)}
        result = surface.predict_stack_rates(records[name], law)
        assert result["predicted"] == dict(zip((832, 1088), prediction))
        assert result["intercept"] == {832: {"down_proj": 2.0}, 1088: {"down_proj": -2.0}}
    args = dict(byte_budget=2100, draws=50, seed=7, max_regret_pct=0.1,
                budget_sensitivity=(0.5, 0.75, 1.0, 1.25),
                unit_bytes={n: {int(r): b for r, b in _TRANSFER["unit_bytes"].items()} for n in records})
    result = surface.stack_transfer_regret_gate(records, **args)
    assert result == surface.stack_transfer_regret_gate(records, **args)
    assert result["regrets_pct"] == [0.0] * 50
    assert result["passes"] is True
    assert result["predicted_assignment"] == {"a": 960, "b": 1088}
    assert result["sensitivity"]["0.5"] == {"byte_budget": 1050, "feasible": False}
    with pytest.raises(TesseraFormatError) as error:
        surface.stack_transfer_regret_gate(records, **{**args, "draws": 49})
    assert str(error.value) == "draws must be an integer of at least 50 (the study's floor), got 49"


def _problem(stats=None, costs=None, **kwargs):
    assert find_spec("prismaquant.production_scalar_solve") is not None
    from prismaquant import production_scalar_solve

    cls = getattr(production_scalar_solve, "ProductionScalarProblem", None)
    assert cls is not None
    if stats is None:
        stats, costs = _scalar_inputs()
    return cls(stats, costs, _FORMATS, currency=_TRANSFER["currency"], **kwargs)


def _gate_args(records, problem, cap=43016):
    return dict(problem=problem, byte_budget=cap, draws=50, seed=7,
                population={n: tuple((s.stack, e, "down_proj") for e in s.experts)
                            for n, s in zip(_NAMES, records.values())},
                format_rates={"NVFP4": 832, "FP8_E4M3": 960})


def _scalar_records(multipliers, missing=False):
    return {name: replace(
        record, experts=(0, 1, 2, 3) if missing else (0, 1, 2),
        reference_mse={e: {"down_proj": v / 7} for e, v in enumerate((1, 2, 4, 1))},
        sampled_mse={832: {e: {"down_proj": v * multiplier / 7}
                           for e, (v, multiplier) in enumerate(zip((1, 2, 4), row))}},
    ) for (name, record), row in zip(_records().items(), multipliers)}


def test_production_gate_matches_cli_counterexample():
    gate = getattr(surface, "production_stack_transfer_regret_gate", None)
    assert callable(gate)
    records = _scalar_records(((7, 7, 7), (11, 11, 11)))
    args = _gate_args(records, _problem())
    result = gate(records, **args)
    assert result == gate(records, **args)
    assert result == gate(dict(reversed(records.items())), **args)
    assert result["status"] == "pass"
    assert result["max_regret_pct"] == 0.1
    assert result["regrets_pct"] == [0.0] * 50
    assert result["predicted_assignment"] == dict(zip(_NAMES, _FORMATS))
    assert result["predicted_loss"] == pytest.approx(8)
    assert result["predicted_bytes"] == 43016
    assert result["solver_contract"] == "additive_candidate_proposal_then_exact_assignment_filter"
    assert result["currency"] == _TRANSFER["currency"]
    assert result["sensitivity"]["0.5"] == {"byte_budget": 21508, "feasible": False}
    tight = gate(records, **{**args, "byte_budget": 43015})
    assert tight["predicted_assignment"] == dict(zip(_NAMES, reversed(_FORMATS)))
    assert tight["predicted_bytes"] == 35336
    assert tight["predicted_loss"] == pytest.approx(12)
    assert result["budget_sha256"] != tight["budget_sha256"]
    assert result["evidence_sha256"] == tight["evidence_sha256"]
    with pytest.raises(TesseraFormatError, match="at least 50"):
        gate(records, **{**args, "draws": 49})
    assert gate(records, **{**args, "byte_budget": 27663})["status"] == "infeasible"
    assert gate({"a": records["a"]}, **args)["status"] == "insufficient_evidence"


def test_production_gate_scores_the_same_draw_and_returns_global_fallback():
    gate = getattr(surface, "production_stack_transfer_regret_gate", None)
    assert callable(gate)
    records = _scalar_records(((7, 7, 7), (1, 32, 1)), missing=True)
    args = _gate_args(records, _problem())
    result = gate(records, **args)
    assert result["status"] == "regret_failure"
    assert result["predicted_assignment"] == dict(zip(_NAMES, reversed(_FORMATS)))
    rng = random.Random(7)
    expected = []
    for _ in range(50):
        rng.choices((0, 1, 2), k=3)
        count = rng.choices((0, 1, 2), k=3).count(1)
        expected.append(max(0, ((1 + 32 ** (count / 3)) / 8 - 1) * 100))
    assert result["regrets_pct"] == pytest.approx(sorted(expected))
    assert result["fallback"] == {
        "mode": "full_missing_band", "scope": "global",
        "rows": [{"stack": n, "expert": 3, "rate_q256": 832} for n in records],
    }
    other = gate(records, **args, budget_sensitivity=(1.25,))
    assert other["status"] == result["status"]
    assert other["regrets_pct"] == result["regrets_pct"]
    base = _scalar_records(((7, 7, 7), (11, 11, 11)), missing=True)
    assert result["evidence_sha256"] != gate(base, **args)["evidence_sha256"]


def test_scalar_problem_resolves_changed_scores_at_one_cap():
    problem = _problem()
    for case, assignment, exact_bytes, loss in (
        ("base", _FORMATS, 43016, 8),
        ("changed", _FORMATS[::-1], 35336, 3),
        ("frontier_flip", _FORMATS, 43016, 2),
        ("base", _FORMATS, 43016, 8),
    ):
        _, costs = _scalar_inputs(case)
        scores = {n: {f: row["predicted_dloss"] for f, row in menu.items()}
                  for n, menu in costs.items()}
        selected = problem.solve(43016, scores)
        assert selected.assignment == dict(zip(_NAMES, assignment))
        assert selected.exact_bytes == exact_bytes
        assert selected.loss == loss


@pytest.mark.parametrize("charge", ["fixed", "shared"])
def test_scalar_problem_charges_extra_bytes_once(charge):
    problem = _problem(**({"fixed_bytes": 17} if charge == "fixed" else {
        "shared_sidecars": {"scale": (17, tuple((n, "NVFP4") for n in _NAMES))},
    }))
    exact = problem.solve(27664 + 17)
    assert exact.assignment == dict.fromkeys(_NAMES, "NVFP4")
    assert exact.exact_bytes == 27664 + 17
    assert problem.solve(27663 + 17) is None


def test_scalar_problem_expands_packed_members_with_dense_competition():
    packed = _FIXTURE["packed"]
    members, dense = packed["members"], packed["dense"]

    class PackedProfile(DefaultProfile):
        def packed_expert_format_group(self, name):
            return packed["group_key"] if name in members else None

    names = [*members, dense]
    stats = {n: _stats(packed["shape"], packed["h_trace"]) for n in names}
    costs = _costs(names, [packed["member_losses"]] * 2 + [packed["dense_losses"]])
    problem = _problem(stats, costs, profile=PackedProfile())
    exact = problem.solve(35344)
    assert exact.assignment == {**dict.fromkeys(members, "NVFP4"), dense: "FP8_E4M3"}
    assert exact.exact_bytes == 35344
    assert problem.solve(50688).assignment == dict.fromkeys(names, "FP8_E4M3")


def test_packed_gate_matches_cli_and_exhaustive_oracle(monkeypatch, tmp_path):
    packed = _FIXTURE["packed"]
    members, dense = packed["members"], packed["dense"]

    class PackedProfile(DefaultProfile):
        def packed_expert_format_group(self, name):
            return packed["group_key"] if name in members else None

    from prismaquant import model_profiles

    profile = PackedProfile()
    monkeypatch.setattr(model_profiles, "detect_profile", lambda path: profile)
    monkeypatch.chdir(tmp_path)
    names = [*members, dense]
    stats = {n: _stats(packed["shape"], 1.0) for n in names}
    costs = _costs(names, [(5, 1), (5, 1), (20, 1)])
    shapes = {f"{n}.weight": packed["shape"] for n in names}
    shapes.update(_FIXTURE["source"]["immutable_shapes"])
    (tmp_path / "model").mkdir()
    save_file({n: torch.zeros(shape, dtype=torch.bfloat16) for n, shape in shapes.items()},
              "model/model.safetensors")
    (tmp_path / "probe.pkl").write_bytes(pickle.dumps({"stats": stats, "meta": {"model": "model"}}))
    (tmp_path / "cost.pkl").write_bytes(pickle.dumps({"costs": costs, "meta": {"formats": _FORMATS}}))
    records = _scalar_records(((5, 5, 5), (5, 5, 5)))
    args = _gate_args(records, _problem(stats, costs, profile=profile))
    args["population"] = dict(zip(members, args["population"].values()))
    oracle = [(3 * 9224, 30, (0, 0)), (2 * 9224 + 16896, 11, (0, 1)),
              (2 * 16896 + 9224, 22, (1, 0)), (3 * 16896, 3, (1, 1))]
    for cap in (35343, 35344, 43016):
        size, loss, (group_fmt, dense_fmt) = min((row for row in oracle if row[0] <= cap),
                                                key=lambda row: row[1])
        monkeypatch.setattr(sys, "argv", [
            "allocator", "--probe", "probe.pkl", "--costs", "cost.pkl",
            "--formats", ",".join(_FORMATS), "--allow-default-profile",
            "--bit-precision", "0.0001", "--overshoot-tolerance", "0.0",
            "--target-bits", repr(cap * 8 / 49152), "--layer-config", "choice.json",
            "--pareto-csv", "choice.csv",
        ])
        allocator.main()
        layer = json.loads((tmp_path / "choice.json").read_text())
        cli = {n: "FP8_E4M3" if layer[n]["bits"] == 8 else "NVFP4" for n in names}
        result = surface.production_stack_transfer_regret_gate(
            records, **{**args, "byte_budget": cap}, budget_sensitivity=(1.0,))
        assert result["predicted_assignment"] == cli == {
            **dict.fromkeys(members, _FORMATS[group_fmt]), dense: _FORMATS[dense_fmt],
        }
        assert result["predicted_bytes"] == size == layer["__prismaquant__"]["body_assignment_payload_bits_total"] / 8
        assert result["predicted_loss"] == pytest.approx(loss)
        assert result["regrets_pct"] == [0.0] * 50


def test_campaign_transfer_scores_rebuild_tessera_frontiers(monkeypatch):
    from types import SimpleNamespace

    from prismaquant import allocator_candidates, production_scalar_solve
    from prismaquant.tessera_campaign_table import CURRENCY

    family = "TESSERA_E4M3_K1"
    low, high = f"{family}_R832", f"{family}_R960"
    name = _NAMES[0]
    stats = {name: _stats((256, 256), 4.0)}
    cap = 65536
    costs = {name: {fmt: {
        "output_mse": mse, "output_mse_measured": measured,
        "cost_source": ("tessera_campaign_measured" if measured else "tessera_campaign_interpolated"),
        "currency": CURRENCY, "tessera_family": family, "tessera_body_rate_q256": rate,
        "hessian_identity": {"applied": True}, "activation_quantized": True,
        "activation_contract": "fp8_per_token_dynamic",
    } for fmt, rate, mse, measured in ((low, 832, 5.0, True), (high, 960, 1.0, False))}}
    loo = {name: {family: {"max_abs_log2_error": 0.05}}}
    # Supply admission for this CPU fixture only. Keep the allocator and byte accounting real.
    monkeypatch.setattr(allocator_candidates, "candidate_rung_admission",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            requires_serving_context=False, admits=lambda _mode: True))
    cls = production_scalar_solve.ProductionScalarProblem
    try:
        problem = cls(stats, costs, (low, high), currency=CURRENCY, census_loo=loo,
                      source_manifest={name: "BF16"})
    except TypeError:
        assert False, "The scalar problem must accept census evidence."
    solve_scalar = production_scalar_solve.solve_scalar
    menus = []

    def capture(stats, candidates, *args, **kwargs):
        menus.append({c.fmt for c in candidates[name]})
        return solve_scalar(stats, candidates, *args, **kwargs)

    monkeypatch.setattr(production_scalar_solve, "solve_scalar", capture)
    for evidence, expected_menu, winner, expected_loss in (
        (None, {low, high}, high, 2.0),
        (24.0, {low}, low, 10.0),
        (19.8, {low}, low, 10.0),
        (4.0, {low, high}, high, 2.0),
    ):
        solve, score = problem.prepare(None if evidence is None else {name: {high: evidence}})
        selected = solve(cap)
        assert menus[-1] == expected_menu
        assert selected.assignment == {name: winner}
        assert score(selected.assignment) == expected_loss
        if evidence is not None:
            assert score({name: high}) == pytest.approx(evidence / 2)
    records = {stack: replace(
        record, currency=CURRENCY, reference_q256=832,
        reference_mse={e: {"down_proj": 20 * v / 7} for e, v in enumerate((1, 2, 4))},
        sampled_mse={960: {e: {"down_proj": 4 * v / 7} for e, v in enumerate((1, 2, 4))}},
    ) for stack, record in _records().items()}
    result = surface.production_stack_transfer_regret_gate(
        records, problem=problem, population={name: [("a", e, "down_proj") for e in range(3)]},
        format_rates={low: 832, high: 960}, byte_budget=cap, budget_sensitivity=(1.0,),
        draws=50, seed=7)
    assert result["predicted_assignment"] == {name: high}
    assert result["predicted_loss"] == pytest.approx(2.0)
    assert result["regrets_pct"] == [0.0] * 50
    assert problem.evidence_inputs()["costs"] == costs
    with pytest.raises(allocator_candidates.CensusLooBandError):
        cls(stats, costs, (low, high), currency=CURRENCY)
