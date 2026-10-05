"""Deterministic common-probe rate trades; no serving or quality claim."""
import copy
import math

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant.model_profiles import DefaultProfile
from test_joint_aura_assignment_diagnostics import _row, _rebuild

LOW = "FP8_E4M3"
HIGH = "FP8_E5M2"


def _trade_rows(values):
    costs, assignment, baseline = {}, {}, {}
    for name, (a, b) in values.items():
        costs[name] = {LOW: _row(name, a, fmt=LOW), HIGH: _row(name, b, fmt=HIGH)}
        assignment[name], baseline[name] = LOW, HIGH
    return costs, assignment, baseline


def _price(values, z=1.0):
    costs, assignment, baseline = _trade_rows(values)
    return ac.price_paired_rate_trade(costs, assignment, baseline,
                                     profile=DefaultProfile(), ucb_z=z)


def test_common_noise_cancels_before_hedging_and_no_second_normalizer():
    name = "model.layers.5.self_attn.q_proj"
    result = _price({name: ([5, -4, 5, -4], [3, 0, -3, 0])}, z=2)
    assert result["mean_difference"] == 8
    assert result["paired_standard_error"] == 0
    assert result["hedged_difference"] == 8
    assert result["predicted_dloss"] == 10.25
    assert result["cost_currency"] == "joint_aura_predicted_dloss"
    assert result["normalization"] == "global_kl_fisher"
    assert result["clipping"] == "nonnegative_candidate_total_only"


def test_correlation_between_member_differences_is_retained():
    names = [f"model.layers.5.self_attn.{role}_proj" for role in ("q", "k")]
    result = _price({names[0]: ([1, 3], [1, 1]), names[1]: ([3, 1], [1, 1])})
    assert result["difference_per_probe"] == [4.0, 4.0]
    assert result["mean_difference"] == 4
    assert result["paired_standard_error"] == 0
    assert result["predicted_dloss"] == 5


@pytest.mark.parametrize("mutation", ["seed", "normalization", "currency", "missing_samples", "source"])
def test_unmatched_or_foreign_evidence_refuses(mutation):
    name = "model.layers.5.self_attn.q_proj"
    costs, assignment, baseline = _trade_rows({name: ([1, 2], [2, 3])})
    row = costs[name][HIGH]
    if mutation == "seed":
        costs[name][HIGH] = _rebuild(row, probe_change=lambda p: p.update(seed_base=100))
    elif mutation == "source":
        costs[name][HIGH] = _rebuild(row, operator_change=lambda o: o["source_weight"].update(content_sha256="f"*64))
    elif mutation == "normalization":
        row["probe_identity"]["normalization"] = "mean_per_weight"
    elif mutation == "currency":
        row["cost_currency"] = "weight_mse"
    else:
        row.pop("x2_per_probe")
    with pytest.raises(ValueError):
        ac.price_paired_rate_trade(costs, assignment, baseline,
                                  profile=DefaultProfile(), ucb_z=1)


def test_negative_trade_difference_is_not_clipped_into_a_positive_hedge():
    name = "model.layers.5.self_attn.q_proj"
    result = _price({name: ([1, 1], [3, 5])}, z=0)
    assert result["difference_per_probe"] == [-4, -12]
    assert result["mean_difference"] == -8
    assert result["paired_standard_error"] == 4
    assert result["hedged_difference"] == -8
    assert result["predicted_dloss"] == 0.5


def test_zero_z_preserves_authoritative_candidate_sum_bitwise(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "7")
    values = {f"model.layers.5.self_attn.{role}_proj": ([v, v], [1, 2])
              for role, v in (("q", 0.1), ("k", 0.2), ("v", 0.3))}
    result = _price(values, z=0)
    expected = sum(0.5*(v*v) for v in (0.2, 0.1, 0.3))
    assert result["predicted_dloss"].hex() == expected.hex()


def test_expert_projections_are_aggregated_and_strict_half_is_allowed():
    values = {f"model.layers.5.mlp.experts.{expert}.{role}_proj": ([1, 1], [3, 3])
              for expert in (0, 1) for role in ("gate", "up", "down")}
    result = _price(values, z=0)
    layer = result["routed_layers"]["model.layers.5.mlp.experts"]
    assert layer["mean_difference"] == -24
    assert layer["experts"]["0"]["mean_difference"] == -12
    assert layer["experts"]["0"]["fraction_of_layer_change"] == 0.5
    assert layer["experts"]["0"]["members"] == sorted(n for n in values if n.split(".experts.")[1].startswith("0."))
    assert layer["refused"] is False


def test_single_expert_dominance_refuses_with_real_layer_and_signed_change():
    values = {f"model.layers.5.mlp.experts.{expert}.{role}_proj":
              ([1, 1], [3 if expert == 0 else 1, 3 if expert == 0 else 1])
              for expert in (0, 1) for role in ("gate", "up", "down")}
    result = _price(values, z=0)
    layer = result["routed_layers"]["model.layers.5.mlp.experts"]
    assert layer["refused"] is True
    assert layer["experts"]["0"]["mean_difference"] == -12
    assert layer["dominant_experts"] == ["0"]
    assert result["refused"] is True


def test_cancellation_and_zero_net_do_not_hide_dominance():
    values = {"model.layers.5.mlp.experts.0.gate_proj": ([1, 1], [3, 3]),
              "model.layers.5.mlp.experts.1.gate_proj": ([3, 3], [1, 1])}
    result = _price(values, z=0)
    layer = result["routed_layers"]["model.layers.5.mlp.experts"]
    assert layer["mean_difference"] == 0
    assert layer["refused"] is True
    assert layer["dominant_experts"] == ["0", "1"]
    assert layer["experts"]["0"]["fraction_of_layer_change"] is None


def test_unchanged_assignment_never_refuses_and_is_exact_zero():
    name = "model.layers.5.mlp.experts.0.down_proj"
    costs, assignment, _ = _trade_rows({name: ([1, 2], [2, 3])})
    result = ac.price_paired_rate_trade(costs, assignment, copy.deepcopy(assignment),
                                       profile=DefaultProfile(), ucb_z=2)
    assert result["refused"] is False
    assert result["difference_per_probe"] == [0, 0]
    assert result["hedged_difference"] == 0


@pytest.mark.parametrize("z", [-1, math.nan, math.inf])
def test_invalid_hedge_parameter_refuses(z):
    with pytest.raises(ValueError, match="ucb_z"):
        _price({"model.layers.5.self_attn.q_proj": ([1, 2], [2, 3])}, z=z)


def _mixed_fixture():
    from prismaquant.allocator_solver import Candidate
    low, high = "TESSERA_E4M3_K1_R832", "TESSERA_E4M3_K1_R864"
    names = [f"model.layers.5.self_attn.{role}_proj" for role in ("q", "k")]
    costs = {name: {low: _row(name, signed, fmt=low), high: _row(name, [1, 1], fmt=high)}
             for name, signed in zip(names, ([1, 3], [3, 1]))}
    candidates = {name: [Candidate(fmt, 0, 10 + 10*i,
                                   row["predicted_dloss"] + 2*row["predicted_dloss_stderr"])
                         for i, (fmt, row) in enumerate(costs[name].items())] for name in names}
    return names, costs, candidates, {name: high for name in names}


def test_mixed_rate_ucb_matches_every_complete_common_probe_combination():
    from itertools import product
    from test_allocator_sibling_aggregation import _installed_fused_licence
    names, costs, candidates, baseline = _mixed_fixture()
    report = {}
    options = ac.tessera_group_composites(
        names, candidates, 8, licence=_installed_fused_licence(), ucb_z=2,
        costs=costs, baseline_assignment=baseline, profile=DefaultProfile(), report=report)
    assert len(options) == 4
    assert any(len(set(o.member_formats.values())) == 2 for o in options)
    expected = {}
    for formats in product(*(costs[name] for name in names)):
        assignment = dict(zip(names, formats))
        paired = ac.price_paired_rate_trade(costs, assignment, baseline,
                                           profile=DefaultProfile(), ucb_z=2)
        expected[tuple(formats)] = paired["predicted_dloss"]
    for option in options:
        assert option.predicted_dloss == expected[tuple(option.member_formats[n] for n in names)]
        assert report["__paired_trades__"][option.fmt]["predicted_dloss"] == option.predicted_dloss
    low = [o for o in options if o.memory_bytes == 20][0]
    assert low.predicted_dloss == 5.0  # correlated deltas cancel, unlike marginal UCB=13


def test_fused_uniform_and_mixed_prices_share_one_paired_cost_table(monkeypatch):
    from prismaquant import format_registry as fr
    from prismaquant import tessera_runtime_contract as trc
    from test_allocator_sibling_aggregation import _FakeProfile
    monkeypatch.setenv(trc.TESSERA_DEV_PIN_ENV, trc.TESSERA_DEV_PIN_COMMIT)
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "2")
    names, costs, candidates, baseline = _mixed_fixture()
    stats = {n: {"n_params": 4, "h_trace": 10, "n_tokens_seen": 2,
                 "in_features": 2, "out_features": 2} for n in names}
    specs = [fr.get_format(fmt) for fmt in costs[names[0]]]
    grouped_stats, grouped_costs, grouped = ac.aggregate_fused_siblings(
        stats, costs, specs, candidates, _FakeProfile(), baseline_assignment=baseline)
    group = next(iter(grouped))
    for candidate in grouped[group]:
        row = grouped_costs[group][candidate.fmt]
        assert ac.cost_entry_predicted_dloss(grouped_stats[group], row) == candidate.predicted_dloss
        assert row["paired_rate_trade"]["candidate_point_cost"] == row["predicted_dloss"]


def test_packed_menu_refuses_dominated_trade_but_keeps_baseline(monkeypatch):
    from prismaquant.allocator_solver import Candidate
    values = {f"model.layers.5.mlp.experts.{expert}.{role}_proj":
              ([1, 1], [3 if expert == 0 else 1, 3 if expert == 0 else 1])
              for expert in (0, 1) for role in ("gate", "up", "down")}
    costs, assignment, baseline = _trade_rows(values)
    group = "actual-layer::__packed__"
    stats = {group: {"_packed_group_members": sorted(values)}}
    candidates = {group: [Candidate(LOW, 1, 10, 3.0), Candidate(HIGH, 2, 20, 15.0)]}
    report = {}
    repriced = ac.reprice_paired_candidates(stats, costs, candidates, baseline,
                                           profile=DefaultProfile(), ucb_z=0, report=report)
    assert [c.fmt for c in repriced[group]] == [HIGH]
    assert report[group][LOW]["refused"] is True
    assert report[group][HIGH]["refused"] is False
    assert repriced[group][0].predicted_dloss.hex() == candidates[group][1].predicted_dloss.hex()


@pytest.mark.parametrize("baseline", [None, {}])
def test_enabled_pair_missing_baseline_refuses_by_name(baseline):
    costs, assignment, _ = _trade_rows({"model.layers.5.self_attn.q_proj": ([1, 2], [2, 3])})
    with pytest.raises(ValueError, match="baseline_assignment"):
        ac.price_paired_rate_trade(costs, assignment, baseline, profile=DefaultProfile(), ucb_z=1)


def test_changed_formats_with_all_zero_expert_terms_are_not_dominant():
    result = _price({"model.layers.5.mlp.experts.0.down_proj": ([0, 0], [0, 0])}, z=1)
    assert result["refused"] is False
    assert result["predicted_dloss"] == result["paired_standard_error"] == 0


def test_strict_half_boundary_has_no_heuristic_tolerance():
    epsilon = math.ulp(1.0)
    values = {"model.layers.5.mlp.experts.0.down_proj": ([1 + epsilon, 1 + epsilon], [0, 0]),
              "model.layers.5.mlp.experts.1.down_proj": ([1, 1], [0, 0])}
    result = _price(values, z=0)
    assert result["refused"] is True
    assert result["routed_layers"]["model.layers.5.mlp.experts"]["dominant_experts"] == ["0"]


def test_changed_packed_rows_without_expert_identity_refuse():
    name = "model.layers.5.mlp.experts.down_proj"
    with pytest.raises(ValueError, match="per-expert attribution"):
        _price({name: ([1, 1], [2, 2])})


@pytest.mark.parametrize("z", [0, 2])
def test_real_cli_consumes_explicit_baseline_and_emits_paired_provenance(tmp_path, monkeypatch, z):
    import json

    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture
    name, argv = _main_fixture(tmp_path, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index("--measured-runtime-table")]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({name: HIGH}))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", str(z))
    argv += ["--cost-baseline-assignment", str(baseline)]
    allocator.main(argv[1:])
    assert load_assignment(tmp_path / "layer.json") == {name: LOW}
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    assert meta["paired_rate_trade"]["mean_difference"] == pytest.approx(-8)
    assert meta["paired_rate_trade"]["refused"] is False
    assert meta["paired_rate_trade"]["paired_standard_error"] == 0
    applicability = json.loads((tmp_path / "format_applicability.json").read_text())
    assert applicability["paired_rate_trades"][name][LOW]["predicted_dloss"] == 0.5


def test_real_cli_refuses_incomplete_baseline_before_emission(tmp_path, monkeypatch):
    import json
    from prismaquant import allocator
    from test_allocator_measured_runtime_cli import _main_fixture
    _name, argv = _main_fixture(tmp_path)
    argv = argv[:argv.index("--measured-runtime-table")]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"another.actual.unit": HIGH}))
    with pytest.raises(SystemExit, match="baseline_assignment missing members"):
        allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline)])
    assert not (tmp_path / "layer.json").exists()


def test_real_cli_refuses_dominant_packed_layer_trade_before_selection(tmp_path, monkeypatch):
    import json
    import pickle
    from prismaquant import allocator
    from test_allocator_measured_runtime_cli import _main_fixture
    names = [f"model.layers.5.mlp.experts.{e}.{p}_proj" for e in (0, 1)
             for p in ("gate", "up", "down")]
    _name, argv = _main_fixture(tmp_path, units=names, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index("--measured-runtime-table")]
    payload = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    for name in names:
        if ".experts.1." in name:
            # A different format still owns its own operator identity. Only
            # the samples are equal; no expert-1 benefit finances this move.
            row = payload["costs"][name][HIGH]
            from prismaquant.joint_aura import make_joint_aura_entry
            payload["costs"][name][HIGH] = make_joint_aura_entry(
                operator_identity=row["joint_operator_identity"], probe_identity=row["probe_identity"],
                signed_components=payload["costs"][name][LOW]["signed_components_per_probe"])
    (tmp_path / "costs.pkl").write_bytes(pickle.dumps(payload))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict.fromkeys(names, HIGH)))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    with pytest.raises(SystemExit, match="routed layer rate trade refused:.*dominant_experts"):
        allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline),
                        "--no-packed-aggregation", "--no-fused-aggregation"])
    assert not (tmp_path / "layer.json").exists()

