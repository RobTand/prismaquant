"""Deterministic common-probe rate trades; no serving or quality claim."""
import copy
import math

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant.model_profiles import DefaultProfile
from test_joint_aura_assignment_diagnostics import _row, _rebuild

LOW = "FP8_E4M3"
HIGH = "FP8_E5M2"


@pytest.mark.parametrize('value', [None, '', '1', 'true', '00'])
@pytest.mark.parametrize('field', ['producer_source_sha256', 'source_model',
                                  'source_execution', 'arithmetic', 'source'])
def test_dev_trade_metadata_drift_prices_stored_samples(monkeypatch, capsys, value, field):
    if value is None:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    else:
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', value)
    name = 'model.layers.5.self_attn.q_proj'
    costs, assignment, baseline = _trade_rows({name: ([1, 2], [2, 3])})
    expected = ac.price_paired_rate_trade(costs, assignment, baseline,
                                        profile=DefaultProfile(), ucb_z=1)
    row = costs[name][HIGH]
    def change(probe):
        if field == 'producer_source_sha256':
            probe[field] = 'e'*64
        elif field == 'source_model':
            probe[field]['source'] = 'another source label'
        elif field == 'source_execution':
            probe[field] = {'label': 'another execution producer'}
        else:
            probe[field]['measurement_dtype'] = 'torch.bfloat16'

    if field == 'source':
        row = _rebuild(row, operator_change=lambda o: o['source_weight'].update(content_sha256='f'*64))
    else:
        row = _rebuild(row, probe_change=change)
    costs[name][HIGH] = row
    result = ac.price_paired_rate_trade(costs, assignment, baseline,
                                      profile=DefaultProfile(), ucb_z=1)
    for key in ('difference_per_probe', 'mean_difference', 'paired_standard_error',
                'predicted_dloss', 'hedged_difference'):
        assert result[key] == expected[key]
    assert costs[name][HIGH] is row
    assert result['dev_uncertified'] is True
    assert '[DEV-MODE]' in capsys.readouterr().out



@pytest.mark.parametrize('mode', [None, '1', '0'])
@pytest.mark.parametrize('field,value', [
    ('seed_base', 100), ('calibration_sha256', 'b'*64), ('token_scope', 'last'),
    ('temperature', 2.0), ('normalization', 'stored-normalizer'),
    ('distribution', 'gaussian'), ('noise_layout', {'rows': 7}),
])
def test_sample_coordinate_and_unit_contracts_refuse_in_both_modes(monkeypatch, mode, field, value):
    if mode is None:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    else:
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    name = 'model.layers.5.self_attn.q_proj'
    costs, assignment, baseline = _trade_rows({name: ([1, 2], [2, 3])})
    with pytest.raises(ValueError):
        costs[name][HIGH] = _rebuild(costs[name][HIGH], probe_change=lambda p: p.update({field: value}))
        ac.price_paired_rate_trade(costs, assignment, baseline, profile=DefaultProfile(), ucb_z=1)


@pytest.mark.parametrize('mode', [None, '1', '0'])
def test_probe_coordinate_json_types_are_not_conflated(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    else:
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    name = 'model.layers.5.self_attn.q_proj'
    costs, assignment, baseline = _trade_rows({name: ([1, 2], [2, 3])})
    for fmt in (LOW, HIGH):
        costs[name][fmt] = _rebuild(costs[name][fmt], probe_change=lambda p: p.update(seed_base=0))
    costs[name][HIGH]['probe_ids'] = [False, True]
    with pytest.raises(ValueError):
        ac.price_paired_rate_trade(costs, assignment, baseline, profile=DefaultProfile(), ucb_z=1)


def test_dev_trade_still_refuses_corrupt_or_partial_samples(monkeypatch):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    name = 'model.layers.5.self_attn.q_proj'
    costs, assignment, baseline = _trade_rows({name: ([1, 2], [2, 3])})
    costs[name][HIGH]['probe_identity']['producer_source_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='identity'):
        ac.price_paired_rate_trade(costs, assignment, baseline, profile=DefaultProfile(), ucb_z=1)
    costs[name][HIGH] = _row(name, [2, 3, 4], fmt=HIGH)
    with pytest.raises(ValueError, match='align'):
        ac.price_paired_rate_trade(costs, assignment, baseline, profile=DefaultProfile(), ucb_z=1)



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


@pytest.mark.parametrize('value', [None, '', 'true', '0'])
def test_real_cli_dev_metadata_drift_or_certified_refusal(tmp_path, monkeypatch, capsys, value):
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture

    name, argv = _main_fixture(tmp_path, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index('--measured-runtime-table')]
    baseline = tmp_path/'baseline.json'
    baseline.write_text(json.dumps({name: HIGH}))
    payload = pickle.loads((tmp_path/'costs.pkl').read_bytes())
    payload['costs'][name][HIGH] = _rebuild(payload['costs'][name][HIGH],
        probe_change=lambda p: p.update(producer_source_sha256='e'*64))
    (tmp_path/'costs.pkl').write_bytes(pickle.dumps(payload))
    if value is None:
        monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    else:
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', value)
    monkeypatch.setenv('PRISMAQUANT_COST_UCB_Z', '1')
    command = [*argv[1:], '--cost-baseline-assignment', str(baseline)]
    if value == '0':
        with pytest.raises(SystemExit, match='probe/calibration identity'):
            allocator.main(command)
        assert not (tmp_path/'layer.json').exists()
        return
    allocator.main(command)
    assert load_assignment(tmp_path/'layer.json') == {name: LOW}
    trade = json.loads((tmp_path/'layer.json').read_text())['__prismaquant__']['paired_rate_trade']
    assert trade['mean_difference'] == -8
    assert trade['paired_standard_error'] == 0
    assert trade['dev_uncertified'] is True
    assert '[DEV-MODE]' in capsys.readouterr().out



@pytest.mark.parametrize("z", [0, 2])
def test_real_cli_consumes_explicit_baseline_and_emits_paired_provenance(tmp_path, monkeypatch, z):
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture
    name, argv = _main_fixture(tmp_path, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index("--measured-runtime-table")]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({name: HIGH}))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", str(z))
    allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline)])
    selected = load_assignment(tmp_path / "layer.json")
    assert selected == {name: LOW}
    costs = pickle.loads((tmp_path / "costs.pkl").read_bytes())["costs"]
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    full = _assert_complete_cli_trade(meta["paired_rate_trade"], costs, selected, {name: HIGH}, z)
    assert full["difference_per_probe"] == [-8.0, -8.0, -8.0]
    applicability = json.loads((tmp_path / "format_applicability.json").read_text())
    _assert_cli_summary(applicability["paired_rate_trades"][name][LOW], full)


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


@pytest.mark.parametrize("phase", ["exact_filter", "final_writer"])
def test_real_cli_refuses_dominant_packed_layer_trade_before_selection(tmp_path, monkeypatch, phase):
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.joint_aura import make_joint_aura_entry
    from test_allocator_measured_runtime_cli import _main_fixture
    names = [f"model.layers.5.mlp.experts.{e}.{p}_proj" for e in (0, 1)
             for p in ("gate", "up", "down")]
    _name, argv = _main_fixture(tmp_path, units=names, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index("--measured-runtime-table")]
    payload = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    def dominate(costs):
        for name in names:
            if ".experts.1." in name:
                row = costs[name][HIGH]
                costs[name][HIGH] = make_joint_aura_entry(
                    operator_identity=row["joint_operator_identity"], probe_identity=row["probe_identity"],
                    signed_components=costs[name][LOW]["signed_components_per_probe"])
    complete_calls = []
    real_price = allocator.price_paired_rate_trade
    def price_with_late_row_fault(costs, assignment, baseline, **kwargs):
        if set(assignment) == set(names) and set(assignment.values()) == {LOW}:
            if phase == "final_writer" and complete_calls:
                # Inject a valid row change after the exact solve. Keep real arithmetic.
                dominate(costs)
            trade = real_price(costs, assignment, baseline, **kwargs)
            complete_calls.append(trade)
            return trade
        return real_price(costs, assignment, baseline, **kwargs)
    if phase == "exact_filter":
        dominate(payload["costs"])
    monkeypatch.setattr(allocator, "price_paired_rate_trade", price_with_late_row_fault)
    (tmp_path / "costs.pkl").write_bytes(pickle.dumps(payload))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict.fromkeys(names, HIGH)))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    with pytest.raises(SystemExit, match="routed layer rate trade refused:.*dominant_experts") as refusal:
        allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline),
                        "--no-packed-aggregation", "--no-fused-aggregation"])
    assert not (tmp_path / "layer.json").exists()
    assert [t["refused"] for t in complete_calls] == ([True] if phase == "exact_filter" else [False, True])
    printed = json.loads(str(refusal.value).split("refused: ", 1)[1])
    layer = "model.layers.5.mlp.experts"
    row = printed[layer]
    assert set(printed) == {layer}
    assert row["refused"] is True
    assert row["refusal_reason"] == "expert_dominance"
    assert row["dominant_experts"] == ["0"]
    assert row["members"] == sorted(names)
    assert row["mean_difference"] == pytest.approx(-24)
    assert row["paired_standard_error"] == 0
    assert row["experts"]["0"] == {
        "members": sorted(n for n in names if ".experts.0." in n),
        "mean_difference": pytest.approx(-24), "paired_standard_error": 0,
        "fraction_of_layer_change": 1.0}
    assert row["experts"]["1"] == {
        "members": sorted(n for n in names if ".experts.1." in n),
        "mean_difference": 0, "paired_standard_error": 0, "fraction_of_layer_change": 0}
    from test_paired_rate_trade_retention import _find_full_arrays
    assert list(_find_full_arrays(printed)) == []


def _assert_complete_cli_trade(actual, costs, selected, baseline, z):
    from prismaquant.joint_aura import identity_sha256
    expected = ac.price_paired_rate_trade(costs, selected, baseline, profile=DefaultProfile(), ucb_z=float(z))
    assert actual == expected
    assert actual["probe_ids"] == costs[next(iter(selected))][LOW]["probe_ids"]
    for key, assignment in (("assignment_a", selected), ("assignment_b", baseline)):
        identities = {n: costs[n][f]["joint_operator_identity_sha256"] for n, f in sorted(assignment.items())}
        assert actual[key]["operator_identity_sha256_by_unit"] == identities
        assert actual[key]["assignment_identity_sha256"] == identity_sha256(identities)
    return expected


def _assert_cli_summary(summary, full):
    from prismaquant.digests import DIRECT_UTF8_STRICT
    from test_paired_rate_trade_retention import _find_full_arrays
    assert list(_find_full_arrays(summary)) == []
    assert summary["full_trade_sha256"] == DIRECT_UTF8_STRICT.sha256(full)
    assert summary["assignment_a_sha256"] == full["assignment_a"]["assignment_identity_sha256"]
    assert summary["assignment_b_sha256"] == full["assignment_b"]["assignment_identity_sha256"]
    assert summary["n_probes"] == len(full["probe_ids"])
    for key in ("mean_difference", "paired_standard_error", "hedged_difference",
                "candidate_point_cost", "predicted_dloss", "ucb_z", "refused",
                "probe_identity_sha256", "dev_mode", "dev_uncertified"):
        assert summary.get(key) == full.get(key)
    assert summary["groups"] == {g: {k: v for k, v in row.items() if k != "difference_per_probe"}
                                 for g, row in full["group_differences"].items()}
    assert ac.summarize_paired_rate_trade(summary) == summary
    for row in summary["routed_layers"].values():
        assert ac.summarize_paired_routed_layer(row) == row


def test_real_cli_emits_complete_diverse_expert_evidence(tmp_path, monkeypatch):
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture
    names = [f"model.layers.5.mlp.experts.{e}.{p}_proj" for e in range(3)
             for p in ("gate", "up", "down")]
    _name, argv = _main_fixture(tmp_path, units=names)
    argv = argv[:argv.index("--measured-runtime-table")]
    costs = {n: {f: _row(n, [i + 1, -(i + 2), i + 3, -(i + 4)], fmt=f)
                 for f, i in ((LOW, index % 3), (HIGH, index % 3 + 4))}
             for index, n in enumerate(names)}
    payload = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    payload["costs"] = costs
    (tmp_path / "costs.pkl").write_bytes(pickle.dumps(payload))
    baseline = dict.fromkeys(names, HIGH)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "1")
    allocator.main([*argv[1:], "--cost-baseline-assignment", str(path)])
    selected = load_assignment(tmp_path / "layer.json")
    assert selected == dict.fromkeys(names, LOW)
    full = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]["paired_rate_trade"]
    _assert_complete_cli_trade(full, costs, selected, baseline, 1)
    assert full["difference_per_probe"] == [-144.0, -180.0, -216.0, -252.0]
    layer = "model.layers.5.mlp.experts"
    for expert in range(3):
        row = full["routed_layers"][layer]["experts"][str(expert)]
        assert row["members"] == sorted(n for n in names if f".experts.{expert}." in n)
        assert row["difference_per_probe"] == [-48.0, -60.0, -72.0, -84.0]
        assert full["group_differences"][f"{layer}.{expert}"]["difference_per_probe"] == row["difference_per_probe"]
    assert full["group_differences"][layer]["difference_per_probe"] == full["difference_per_probe"]
    report = json.loads((tmp_path / "format_applicability.json").read_text())["paired_rate_trades"]
    for options in report.values():
        for fmt, summary in options.items():
            members = full["routed_layers"][layer]["members"]
            expected = ac.price_paired_rate_trade(costs, dict.fromkeys(members, fmt), baseline,
                                                  profile=DefaultProfile(), ucb_z=1.0)
            _assert_cli_summary(summary, expected)


def test_real_cli_exact_filter_summary_preserves_dev_stamps(tmp_path, monkeypatch, capsys):
    import json
    import pickle
    from prismaquant import allocator
    from test_allocator_measured_runtime_cli import _main_fixture, admit_synthetic_table
    admit_synthetic_table(monkeypatch)
    name, argv = _main_fixture(tmp_path, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({name: HIGH}))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "1")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    allocator.main([*argv[1:], "--cost-baseline-assignment", str(path)])
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    costs = pickle.loads((tmp_path / "costs.pkl").read_bytes())["costs"]
    full = _assert_complete_cli_trade(meta["paired_rate_trade"], costs, {name: LOW}, {name: HIGH}, 1)
    trace = meta["measured_runtime_search"]["target_diagnostics"]["exact_filter_trace"]
    assert len(trace) == 1
    assert trace[0]["feasible"] is True
    _assert_cli_summary(trace[0]["paired_rate_trade"], full)
    assert trace[0]["paired_rate_trade"]["dev_uncertified"] is True
    assert trace[0]["paired_rate_trade"]["dev_mode"] == full["dev_mode"]

def test_real_cli_measured_runtime_rejects_a_runtime_feasible_dominated_trade(tmp_path, monkeypatch):
    """The measured-runtime proposal loop ANDs the serving verdict with the paired guard.

    The only proposal on the runtime frontier (every unit at the faster, lower-loss rate)
    is byte- and runtime-feasible, but all of its benefit over the baseline sits in one
    expert. Without a baseline the run emits it, which shows it is feasible on its own;
    with a baseline the paired guard must reject it inside the measured-runtime proposal
    filter, so the run refuses and no layer config is emitted.
    """
    import hashlib
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.joint_aura import make_joint_aura_entry
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture, admit_synthetic_table
    admit_synthetic_table(monkeypatch)  # producer attestation only; parse and hash checks still run
    names = [f"model.layers.5.mlp.experts.{e}.{p}_proj" for e in (0, 1)
             for p in ("gate", "up", "down")]
    _name, argv = _main_fixture(tmp_path, units=names, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    # Room for every proposal: the paired guard, not the prefill budget, is under test.
    argv[argv.index("--slo-prefill-p95-ttft-ms") + 1] = "100"
    payload = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    for name in names:
        if ".experts.1." in name:
            # Expert 1 keeps its own operator identity for the other rate but prices the
            # same samples, so no expert-1 benefit finances the move off the baseline.
            row = payload["costs"][name][HIGH]
            payload["costs"][name][HIGH] = make_joint_aura_entry(
                operator_identity=row["joint_operator_identity"], probe_identity=row["probe_identity"],
                signed_components=payload["costs"][name][LOW]["signed_components_per_probe"])
    (tmp_path / "costs.pkl").write_bytes(pickle.dumps(payload))
    table_path = tmp_path / "runtime.json"
    table = json.loads(table_path.read_text())
    table["cost_sha256"] = hashlib.sha256((tmp_path / "costs.pkl").read_bytes()).hexdigest()
    table_path.write_text(json.dumps(table))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    flags = ["--no-packed-aggregation", "--no-fused-aggregation"]

    allocator.main([*argv[1:], *flags])
    assert load_assignment(tmp_path / "layer.json") == dict.fromkeys(names, LOW)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict.fromkeys(names, HIGH)))
    (tmp_path / "layer.json").unlink()
    with pytest.raises(SystemExit, match="no_runtime_frontier_assignment_passed_exact_checks"):
        allocator.main([*argv[1:], *flags, "--cost-baseline-assignment", str(baseline)])
    assert not (tmp_path / "layer.json").exists()


def _two_expert_layer():
    layer = "model.layers.5.mlp.experts"
    return layer, [f"{layer}.{expert}.{role}_proj"
                   for expert in (0, 1) for role in ("gate", "up", "down")]


def test_subgroup_menu_option_survives_reprice_when_complete_layer_is_valid():
    """A subgroup denominator must not prune a valid whole-layer option (#2288).

    The packed menu prices only expert 0, where expert 0 trivially dominates
    its own delta. The complete layer splits the signed change exactly in
    half, so the option stays valid and the final guard still refuses a
    genuinely dominant complete-layer move.
    """
    from prismaquant.allocator_solver import Candidate
    layer, names = _two_expert_layer()
    costs, _assignment, baseline = _trade_rows(
        {name: ([1, 1], [3, 3]) for name in names})
    subgroup = sorted(n for n in names if ".experts.0." in n)
    group = "actual-layer::__packed__"
    stats = {group: {"_packed_group_members": subgroup}}
    candidates = {group: [Candidate(LOW, 1, 10, 3.0), Candidate(HIGH, 2, 20, 15.0)]}
    report = {}
    repriced = ac.reprice_paired_candidates(stats, costs, candidates, baseline,
                                            profile=DefaultProfile(), ucb_z=0, report=report)
    assert report[group][LOW]["refused"] is True
    assert [c.fmt for c in repriced[group]] == [LOW, HIGH]
    complete = ac.price_paired_rate_trade(costs, dict.fromkeys(names, LOW), baseline,
                                          profile=DefaultProfile(), ucb_z=0)
    assert complete["refused"] is False
    assert complete["routed_layers"][layer]["experts"]["0"]["fraction_of_layer_change"] == 0.5
    invalid = ac.price_paired_rate_trade(
        costs, {n: (LOW if ".experts.0." in n else HIGH) for n in names}, baseline,
        profile=DefaultProfile(), ucb_z=0)
    assert invalid["refused"] is True
    assert invalid["routed_layers"][layer]["dominant_experts"] == ["0"]


def test_subgroup_composite_survives_fold_when_complete_layer_is_valid():
    """The mixed-group fold keeps a subgroup-refused composite (#2288).

    Members are expert 0's fused projections, priced against the complete
    two-expert baseline roster. Every mixed option refuses on the subgroup
    denominator, but the complete layer splits the change exactly in half.
    """
    from itertools import product
    from prismaquant.allocator_solver import Candidate
    from test_allocator_sibling_aggregation import _installed_fused_licence
    layer = "model.layers.5.mlp.experts"
    members = [f"{layer}.0.{role}_proj" for role in ("gate", "up")]
    rest = [f"{layer}.1.{role}_proj" for role in ("gate", "up")]
    low, high = "TESSERA_E4M3_K1_R832", "TESSERA_E4M3_K1_R864"
    costs = {name: {low: _row(name, [1, 1], fmt=low), high: _row(name, [3, 3], fmt=high)}
             for name in members + rest}
    candidates = {name: [Candidate(fmt, 0, 10 + 10 * i, 1.0 + float(i))
                         for i, fmt in enumerate((low, high))]
                  for name in members}
    baseline = dict.fromkeys(members + rest, high)
    report = {}
    options = ac.tessera_group_composites(
        members, candidates, 8, licence=_installed_fused_licence(), ucb_z=2,
        costs=costs, baseline_assignment=baseline, profile=DefaultProfile(), report=report)
    assert len(options) == 4
    assert sorted(tuple(o.member_formats[m] for m in members) for o in options) == sorted(
        product((low, high), repeat=2))
    expected = {}
    for formats in product((low, high), repeat=2):
        assignment = dict(zip(members, formats))
        paired = ac.price_paired_rate_trade(costs, assignment,
                                            {m: baseline[m] for m in members},
                                            profile=DefaultProfile(), ucb_z=2)
        expected[formats] = paired["predicted_dloss"]
    for option in options:
        key = tuple(option.member_formats[m] for m in members)
        assert option.predicted_dloss == expected[key]
        assert report["__paired_trades__"][option.fmt]["predicted_dloss"] == option.predicted_dloss
    refused = report["__paired_trades__"]
    assert sum(1 for trade in refused.values() if trade["refused"]) == 3
    complete = ac.price_paired_rate_trade(
        costs, dict.fromkeys(members + rest, low), baseline,
        profile=DefaultProfile(), ucb_z=2)
    assert complete["refused"] is False
    dominated = dict(dict.fromkeys(members, low), **dict.fromkeys(rest, high))
    invalid = ac.price_paired_rate_trade(costs, dominated, baseline,
                                         profile=DefaultProfile(), ucb_z=2)
    assert invalid["refused"] is True
    assert invalid["routed_layers"][layer]["dominant_experts"] == ["0"]


def test_real_cli_packed_menu_emits_and_final_guard_refuses_dominant_trade(tmp_path, monkeypatch):
    """Public-path smoke: packed reprice emits, the final guard still refuses (#2288)."""
    import json
    import pickle
    from prismaquant import allocator
    from prismaquant.joint_aura import make_joint_aura_entry
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture
    _layer, names = _two_expert_layer()
    _name, argv = _main_fixture(tmp_path, units=names, menu={LOW: (0.5, 1), HIGH: (8.5, 2)})
    argv = argv[:argv.index("--measured-runtime-table")]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict.fromkeys(names, HIGH)))
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "1")
    allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline)])
    assert load_assignment(tmp_path / "layer.json") == dict.fromkeys(names, LOW)
    full = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]["paired_rate_trade"]
    assert full["refused"] is False
    menu_report = json.loads((tmp_path / "format_applicability.json").read_text())["paired_rate_trades"]
    assert menu_report
    payload = pickle.loads((tmp_path / "costs.pkl").read_bytes())
    for name in names:
        if ".experts.1." in name:
            row = payload["costs"][name][HIGH]
            payload["costs"][name][HIGH] = make_joint_aura_entry(
                operator_identity=row["joint_operator_identity"], probe_identity=row["probe_identity"],
                signed_components=payload["costs"][name][LOW]["signed_components_per_probe"])
    (tmp_path / "costs.pkl").write_bytes(pickle.dumps(payload))
    (tmp_path / "layer.json").unlink()
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    with pytest.raises(SystemExit, match="routed layer rate trade refused"):
        allocator.main([*argv[1:], "--cost-baseline-assignment", str(baseline),
                        "--no-packed-aggregation", "--no-fused-aggregation"])
    assert not (tmp_path / "layer.json").exists()
