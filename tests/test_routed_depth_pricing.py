"""#2535 / #2774: receipt-bound pricing, not a fit or qualification run.

Adopted from d8b03b074cf9bc7889c2df871fb1cad174350c03 (the existing
#2535 test-only delivery); extended for #2774, not a second implementation.
No attributable numeric flat authorization was located. Positive application
of an actual interim profile remains decision-blocked, not a synthetic permit.
W1's test_paired_rate_trade.py and owned documentation are not edited here.

Proposed public intake seam: CostPricingPolicy.from_depth_profile(payload,
profile_record=..., production=..., ucb_z=...). Records are plain JSON. These
numeric fits/receipts are SYNTHETIC CPU fixtures, never CEO flat authorization
or evidence that any production price has passed GPU qualification.

Contract band_loss_multiplier.v1 multiplies each cell's already-normalized
per-probe loss (and its candidate-minus-baseline contribution) by its band's
positive multiplier. It must NOT renormalize Fisher, mutate source samples,
or multiply a post-hoc marginal UCB in lieu of paired covariance.
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import sys
from pathlib import Path

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import priced_cell
from prismaquant.joint_aura_rows import joint_identity_sha256 as digest
from prismaquant.model_profiles import DefaultProfile
from test_joint_aura_assignment_diagnostics import _row
from test_paired_rate_trade import LOW, HIGH, _trade_rows

SCHEMA = "prismaquant.routed_depth_profile.v1"
CONTRACT = "band_loss_multiplier.v1"
# down_proj is a complete standalone serving unit; an up_proj-only CLI
# fixture is legitimately BF16-pinned by the independent fused-sibling gate.
EARLY = [f"model.layers.5.mlp.experts.{e}.down_proj" for e in (0, 1)]
LATE = [f"model.layers.9.mlp.experts.{e}.down_proj" for e in (0, 1)]


def _payload(*, scope="all_rates"):
    costs, assignment, baseline = _trade_rows({
        **{n: ([1, 3], [1, 1]) for n in EARLY},
        **{n: ([3, 1], [1, 1]) for n in LATE},
    })
    return ({"costs": costs, "provenance": {
        "cost_mode": "aura", "joint_activation": True,
        "cost_currency": "joint_aura_predicted_dloss",
        "depth_price_scope": {"lane": "joint_t8_routed", "allocation": scope,
                              "model_id": "synthetic-depth-model"}}}, assignment, baseline)


def _bindings(payload, record):
    # The raw cost table has no derived price-policy fields. Parameter and
    # roster hashes are separate to make replay against changed facts fail.
    return {"source_table_sha256": digest(payload),
            "band_roster_sha256": digest(record["bands"]),
            "calibration_contract": record["calibration_contract"],
            "parameters_sha256": digest(record["parameters"])}


def _record(payload, *, multipliers=(4.0, 1.0), qualified=False):
    record = {"schema": SCHEMA, "state": "qualified" if qualified else "research_calibrated",
        "scope": copy.deepcopy(payload["provenance"]["depth_price_scope"]),
        "bands": {"early": list(EARLY), "late": list(LATE)}, "calibration_contract": CONTRACT,
        "parameters": {"early": {"loss_multiplier": multipliers[0]},
                       "late": {"loss_multiplier": multipliers[1]}}}
    record.update(_bindings(payload, record))
    record["measurement_receipt"] = {
        "schema": "prismaquant.routed_depth_measurement_receipt.v1",
        **_bindings(payload, record), "arm_id": "synthetic-fit",
        "sample_roster_sha256": "a" * 64, "result": "complete"}
    if qualified:
        record["qualification_receipts"] = {
            kind: {"schema": "prismaquant.routed_depth_qualification_receipt.v1",
                   **_bindings(payload, record), "kind": kind, "passed": True,
                   "arm_id": "synthetic-" + kind,
                   "sample_roster_sha256": ("b" if kind == "measured_arm_agreement" else "c") * 64}
            for kind in ("measured_arm_agreement", "held_out_arm")}
    return record


def _policy(payload, record, *, production=False, z=0.0):
    # Fail on the missing feature rather than raising an import/discovery
    # error on main. The existing pricing module is the shared consumer seam.
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), (
        "#2535: CostPricingPolicy has no receipt-bound routed-depth intake; "
        "raw joint prices are still the only shared price policy")
    return factory(payload, profile_record=record, production=production, ucb_z=z)


def _price(payload, policy, name, fmt=LOW, *, gain=1.0):
    return priced_cell.price_cell({"h_trace": 1e12, "n_tokens_seen": 999},
        payload["costs"][name][fmt], format_name=fmt, policy=policy, gain=gain).predicted_dloss


def test_research_intake_applies_band_prices_without_touching_raw_rows():
    payload, _, _ = _payload()
    before, before_digest = copy.deepcopy(payload), digest(payload)
    record = _record(payload)
    record_before = copy.deepcopy(record)
    policy = _policy(payload, record)
    assert _price(payload, policy, EARLY[0]) == 10.0
    assert _price(payload, policy, LATE[0]) == 2.5
    assert payload == before and digest(payload) == before_digest
    assert record == record_before
    provenance = policy.depth_profile
    assert provenance["state"] == "research_calibrated"
    assert provenance["validated"] is False
    for key in ("source_table_sha256", "band_roster_sha256", "calibration_contract", "parameters_sha256"):
        assert provenance[key] == record[key]
    assert provenance["measurement_receipt_sha256"] == digest(record["measurement_receipt"])


@pytest.mark.parametrize("record", [None, {"schema": SCHEMA, "state": "flat", "authorization": {}},
    {"schema": SCHEMA, "state": "flat", "authorization": {
        "record": "synthetic incomplete permit: no numeric decision",
        "record_sha256": "a" * 64}}])
def test_affected_flat_without_written_numeric_authorization_refuses(record):
    payload, _, _ = _payload()
    # Even research cannot invent flat values. This test deliberately contains
    # no purported CEO values. Positive flat CLI coverage awaits that decision.
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2535: missing fail-closed depth-profile admission"
    with pytest.raises(ValueError, match="(?i)(flat|authorization|numeric)"):
        factory(payload, profile_record=record, production=False, ucb_z=0)


@pytest.mark.parametrize("mutation", ["missing_band", "missing_member", "extra_member", "duplicate_member",
    "foreign_table", "foreign_scope", "changed_parameter", "changed_roster", "bad_receipt",
    "foreign_receipt", "missing_measurement", "contract", "version", "negative", "nonfinite", "scalar", "scalar_parameters"])
def test_intake_refuses_unbound_or_malformed_calibration(mutation):
    payload, _, _ = _payload()
    record = _record(payload)
    if mutation == "missing_band":
        record["parameters"].pop("late")
    elif mutation == "missing_member":
        record["bands"]["early"].pop()
    elif mutation == "extra_member":
        record["bands"]["early"].append("model.layers.99.mlp.experts.0.up_proj")
    elif mutation == "duplicate_member":
        record["bands"]["late"].append(EARLY[0])
    elif mutation == "foreign_table":
        record["source_table_sha256"] = "f" * 64
    elif mutation == "foreign_scope":
        record["scope"]["model_id"] = "unrelated-model"
    elif mutation == "changed_parameter":
        record["parameters"]["early"]["loss_multiplier"] = 8.0
    elif mutation == "changed_roster":
        record["band_roster_sha256"] = "f" * 64
    elif mutation == "bad_receipt":
        record["measurement_receipt"] = "not a receipt"
    elif mutation == "foreign_receipt":
        record["measurement_receipt"]["parameters_sha256"] = "f" * 64
    elif mutation == "missing_measurement":
        record.pop("measurement_receipt")
    elif mutation == "contract":
        record["calibration_contract"] = "unknown.v99"
    elif mutation == "version":
        record["schema"] = "prismaquant.routed_depth_profile.v99"
    elif mutation == "scalar":
        record["gain"] = 0.1
    elif mutation == "scalar_parameters":
        record["parameters"] = {band: {"loss_multiplier": 2.0} for band in ("early", "late")}
    else:
        record["parameters"]["early"]["loss_multiplier"] = -1 if mutation == "negative" else math.inf
    if mutation in {"missing_band", "missing_member", "extra_member", "duplicate_member",
                    "contract", "negative", "scalar_parameters"}:
        # Self-consistent hashes must not license an incomplete/alien roster,
        # unsupported numerical contract, or a scalar gain disguised as bands.
        record.update(_bindings(payload, record))
        record["measurement_receipt"].update(_bindings(payload, record))
    # Ensure the feature exists BEFORE the refusal assertion: absence must not
    # masquerade as a successful rejection of bad evidence.
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2535: missing versioned calibration admission"
    with pytest.raises(ValueError):
        factory(payload, profile_record=record, production=False, ucb_z=0)


@pytest.mark.parametrize("mutation", ["missing", "failed", "mismatch", "same_arm", "same_samples"])
def test_production_requires_both_matching_independent_qualification_receipts(mutation):
    payload, _, _ = _payload()
    record = _record(payload, qualified=True)
    receipts = record["qualification_receipts"]
    if mutation == "missing":
        receipts.pop("held_out_arm")
    elif mutation == "failed":
        receipts["held_out_arm"]["passed"] = False
    elif mutation == "mismatch":
        receipts["measured_arm_agreement"]["parameters_sha256"] = "f" * 64
    elif mutation == "same_arm":
        receipts["held_out_arm"]["arm_id"] = receipts["measured_arm_agreement"]["arm_id"]
    else:
        receipts["held_out_arm"]["sample_roster_sha256"] = receipts["measured_arm_agreement"]["sample_roster_sha256"]
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2535: missing production price qualification admission"
    with pytest.raises(ValueError):
        factory(payload, profile_record=record, production=True, ucb_z=0)


def test_research_is_not_production_qualification_and_qualified_prices_retain_links():
    payload, _, _ = _payload()
    research = _record(payload)
    assert _policy(payload, research).depth_profile["validated"] is False
    factory = priced_cell.CostPricingPolicy.from_depth_profile
    with pytest.raises(ValueError):
        factory(payload, profile_record=research, production=True, ucb_z=0)
    qualified = _record(payload, qualified=True)
    policy = _policy(payload, qualified, production=True)
    assert policy.depth_profile["state"] == "qualified"
    assert policy.depth_profile["validated"] is True
    assert policy.depth_profile["qualification_receipts_sha256"] == digest(qualified["qualification_receipts"])
    assert _price(payload, policy, EARLY[0]) == 10.0


def test_calibrated_paired_covariance_dominance_and_one_fisher_normalization():
    payload, assignment, baseline = _payload()
    before = digest(payload)
    policy = _policy(payload, _record(payload), z=2)
    result = ac.price_paired_rate_trade(payload["costs"], assignment, baseline,
        profile=DefaultProfile(), ucb_z=2, pricing_policy=policy)
    # Two experts per layer: early [0, 8]*4, late [8, 0]*1.
    # Raw per-cell samples are signed; square first, then apply band facts.
    assert result["difference_per_probe"] == [8.0, 32.0]
    assert result["mean_difference"] == 20.0
    assert result["paired_standard_error"] == 12.0
    assert result["candidate_point_cost"] == 25.0
    assert result["predicted_dloss"] == 49.0
    assert result["normalization"] == "global_kl_fisher"
    assert result["refused"] is False  # exactly half from each expert
    assert all(layer["dominant_experts"] == [] for layer in result["routed_layers"].values())
    assert digest(payload) == before
    # The existing scalar gain route must remain refused even after a fit.
    with pytest.raises(ValueError, match="gain|scalar"):
        _price(payload, policy, EARLY[0], gain=2)


def test_calibration_does_not_erase_complete_layer_dominance_refusal():
    payload, assignment, baseline = _payload()
    payload["costs"][EARLY[0]][LOW] = _row(EARLY[0], [-3, 3], fmt=LOW)
    before = copy.deepcopy(payload)
    policy = _policy(payload, _record(payload))
    result = ac.price_paired_rate_trade(payload["costs"], assignment, baseline,
        profile=DefaultProfile(), ucb_z=0, pricing_policy=policy)
    assert result["refused"] is True
    early_layer = next(layer for layer in result["routed_layers"].values() if EARLY[0] in layer["members"])
    assert early_layer["dominant_experts"]
    assert set(early_layer["members"]) == set(EARLY)
    assert payload == before


def test_two_band_fits_change_existing_routed_unit_selector_picks():
    from prismaquant.tessera_expert_projection import select_priced_unit_upgrades
    low, high = "TESSERA_E4M3_K1_R1024", "TESSERA_E4M3_K1_R1088"
    payload, _, _ = _payload()
    payload["costs"] = {
        name: {fmt: {**_row(name, signed, fmt=fmt), "wire_bytes": wire}
               for fmt, signed, wire in ((low, [3, -3], 100), (high, [1, -1], 110))}
        for name in EARLY + LATE}
    before = digest(payload)
    picked = []
    for multipliers in ((4.0, 1.0), (1.0, 4.0)):
        policy = _policy(payload, _record(payload, multipliers=multipliers))
        picks, receipt = select_priced_unit_upgrades(payload["costs"],
            {name: low for name in EARLY + LATE}, byte_budget=10,
            pricing_policy=policy)
        picked.append(picks)
        assert receipt["whole_artifact_bytes_claimed"] is False
        assert sum(payload["costs"][name][fmt]["wire_bytes"] - 100 for name, fmt in picks.items()) == 10
        assert digest(payload) == before
    assert picked == [{EARLY[0]: high}, {LATE[0]: high}]


def test_grouped_price_path_uses_same_band_prices_once():
    from prismaquant import format_registry as fr
    from test_allocator_packed_group_units import _PackedProfile, _mk_candidates, _LOW, _HIGH
    payload, _, _ = _payload()
    payload["costs"] = {name: {_LOW: _row(name, [3, -3], fmt=_LOW),
                               _HIGH: _row(name, [1, -1], fmt=_HIGH)}
                         for name in EARLY + LATE}
    stats = {name: {"n_params": 4, "h_trace": 1, "in_features": 2, "out_features": 2}
             for name in EARLY + LATE}
    selected_groups = []
    for factors in ((4.0, 1.0), (1.0, 4.0)):
        policy = _policy(payload, _record(payload, multipliers=factors))
        candidates = {name: _mk_candidates(4,
            _price(payload, policy, name, _LOW), _price(payload, policy, name, _HIGH))
            for name in stats}
        grouped_stats, _, grouped = ac.aggregate_packed_serving_groups(stats, payload["costs"],
            [fr.REGISTRY[_LOW], fr.REGISTRY[_HIGH]], candidates, _PackedProfile(),
            pricing_policy=policy)
        assert len(grouped) == 2
        members = {group: grouped_stats[group]["_packed_group_members"] for group in grouped}
        for group, options in grouped.items():
            prices = {option.fmt: option.predicted_dloss for option in options}
            factor = factors[0] if EARLY[0] in members[group] else factors[1]
            assert prices[_LOW] == 9 * factor
            assert prices[_HIGH] == factor
            for option in options:
                assert option.memory_bytes == sum(
                    next(c.memory_bytes for c in candidates[n] if c.fmt == option.fmt)
                    for n in members[group])
        # Exercise the existing allocator, not a parallel ranking algorithm.
        # Equal-size groups: budget buys exactly one whole-group upgrade.
        from prismaquant.allocator_solver import solve_allocation
        total_params = sum(s["n_params"] for s in grouped_stats.values())
        floor = sum(min(c.memory_bytes for c in options) for options in grouped.values())
        first = next(iter(grouped.values()))
        delta = max(c.memory_bytes for c in first) - min(c.memory_bytes for c in first)
        solved = solve_allocation(grouped_stats, grouped, 8 * (floor + delta) / total_params)
        assert solved is not None
        assignment, _ = solved
        selected_groups.append({name for group, fmt in assignment.items() if fmt == _HIGH
                                for name in members[group]})
    # Equal-byte whole-layer choices: which upgrade wins must reverse.
    assert selected_groups == [set(EARLY), set(LATE)]


def test_real_allocator_cli_research_band_fits_change_assignment(tmp_path, monkeypatch):
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    from test_allocator_measured_runtime_cli import _main_fixture
    payload, _, _ = _payload()
    selected = []
    for index, factors in enumerate(((4.0, 1.0), (1.0, 4.0))):
        root = tmp_path / str(index)
        root.mkdir()
        _, argv = _main_fixture(root, units=tuple(EARLY + LATE),
            menu={"FP8_E4M3": (4.5, 8.0), "BF16": (0.0, 2.0)}, target_bits="12.5")
        # This test is about depth-price selection, not admission of a new
        # performance table. Remove that fixture's independent SLO inputs.
        for flag in ("--measured-runtime-table", "--measured-runtime-context", "--slo-prefill-p95-ttft-ms"):
            at = argv.index(flag)
            del argv[at:at + 2]
        cost_path = root / "costs.pkl"
        cli_payload = pickle.loads(cost_path.read_bytes())
        cli_payload["provenance"]["depth_price_scope"] = payload["provenance"]["depth_price_scope"]
        cost_path.write_bytes(pickle.dumps(cli_payload))
        raw_bytes = cost_path.read_bytes()
        record = _record(cli_payload, multipliers=factors)
        record_path = root / "depth.json"
        record_path.write_text(json.dumps(record))
        argv += ["--depth-price-profile", str(record_path), "--depth-price-purpose", "research"]
        monkeypatch.setattr(sys, "argv", argv)
        allocator.main()
        result_path = root / "layer.json"
        assignment = load_assignment(result_path)
        selected.append({name for name in EARLY + LATE if assignment[name] == "BF16"})
        provenance = json.loads(result_path.read_text())["__prismaquant__"]["routed_depth_pricing"]
        assert provenance["state"] == "research_calibrated"
        assert provenance["validated"] is False
        assert provenance["parameters_sha256"] == record["parameters_sha256"]
        assert provenance["source_table_sha256"] == digest(cli_payload)
        assert provenance["selected_prices"] == {
            name: (0.0 if assignment[name] == "BF16" else
                   4.5 * (factors[0] if name in EARLY else factors[1]))
            for name in EARLY + LATE}
        assert cost_path.read_bytes() == raw_bytes
    assert selected == [set(EARLY), set(LATE)]


def test_depth_price_contract_documentation_and_independent_gates():
    root = Path(__file__).resolve().parents[1]
    design = (root / "docs/design/joint_aura_runtime_allocation.md").read_text()
    architecture = (root / "docs/ARCHITECTURE.md").read_text()
    changelog = (root / "CHANGELOG.md").read_text()
    for term in (SCHEMA, CONTRACT, "research_calibrated", "qualified", "unvalidated"):
        assert term in design, f"#2535: depth-price contract omits {term}"
    assert "routed_depth_pricing" in architecture
    assert "2535" in changelog
    for term in ("interim_flat", "authorization", "numeric", "2774"):
        assert term in design, f"#2774: depth-price decision/provenance omits {term}"
    # Wording of the independent gate descriptions is reviewed rather than
    # encoding a particular sentence here; existing gate suites still run.


def test_unaffected_policies_preserve_exact_legacy_price_and_metadata():
    payload, _, _ = _payload(scope="uniform_rate")
    # Factory must be a no-op outside the explicit affected lane, including
    # serialized provenance, not just numerically equivalent float prices.
    policy = _policy(payload, None)
    legacy = priced_cell.CostPricingPolicy(ucb_z=0.0, fisher_output_mse=False)
    assert policy.depth_profile is None
    for name in EARLY + LATE:
        row = payload["costs"][name][LOW]
        assert priced_cell.price_cell({}, row, format_name=LOW, policy=policy) == priced_cell.price_cell(
            {}, row, format_name=LOW, policy=legacy)
    stock = {"weight_mse": 0.25}
    assert priced_cell.price_cell({"h_trace": 8}, stock, format_name=LOW, policy=policy) == priced_cell.price_cell(
        {"h_trace": 8}, stock, format_name=LOW, policy=legacy)
    # Legitimate non-joint calibration is not globally disabled by this fix.
    assert priced_cell.price_cell({"h_trace": 8}, stock, format_name=LOW, policy=policy, gain=3) == priced_cell.price_cell(
        {"h_trace": 8}, stock, format_name=LOW, policy=legacy, gain=3)


@pytest.mark.parametrize("mutation", ["currency", "cost_mode", "probe_contract", "sample_count"])
def test_rehashed_but_incompatible_inputs_do_not_acquire_band_admission(mutation):
    payload, _, _ = _payload()
    if mutation == "currency":
        payload["provenance"]["cost_currency"] = "weight_mse"
    elif mutation == "cost_mode":
        payload["provenance"]["cost_mode"] = "weight_only"
    elif mutation == "probe_contract":
        payload["costs"][EARLY[0]][LOW]["probe_identity"]["normalization"] = "local_fisher"
    else:
        payload["costs"][EARLY[0]][LOW]["signed_per_probe"] = [1.0]
    # Rebind the entire source digest: receipt hash consistency alone must
    # never excuse incompatible measurement semantics.
    record = _record(payload)
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2774: missing semantic input admission"
    with pytest.raises(ValueError):
        factory(payload, profile_record=record, production=False, ucb_z=0)


@pytest.mark.parametrize("mutation", ["receipt_schema", "kind", "passed_string", "empty_arm",
    "invalid_sample_digest", "fit_samples_as_holdout", "missing_contract", "non_object"])
def test_qualification_rejects_malformed_or_nonheldout_receipts(mutation):
    payload, _, _ = _payload()
    record = _record(payload, qualified=True)
    heldout = record["qualification_receipts"]["held_out_arm"]
    if mutation == "receipt_schema":
        heldout["schema"] = "unknown.v99"
    elif mutation == "kind":
        heldout["kind"] = "measured_arm_agreement"
    elif mutation == "passed_string":
        heldout["passed"] = "false"
    elif mutation == "empty_arm":
        heldout["arm_id"] = ""
    elif mutation == "invalid_sample_digest":
        heldout["sample_roster_sha256"] = "not-a-digest"
    elif mutation == "fit_samples_as_holdout":
        heldout["sample_roster_sha256"] = record["measurement_receipt"]["sample_roster_sha256"]
    elif mutation == "missing_contract":
        heldout.pop("calibration_contract")
    else:
        record["qualification_receipts"] = [heldout]
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2774: missing strict production qualification contract"
    with pytest.raises(ValueError):
        factory(payload, profile_record=record, production=True, ucb_z=0)


def _cli_fixture(root):
    from test_allocator_measured_runtime_cli import _main_fixture
    _, argv = _main_fixture(root, units=tuple(EARLY + LATE),
        menu={"FP8_E4M3": (4.5, 8.0), "BF16": (0.0, 2.0)}, target_bits="12.5")
    argv = argv[:argv.index("--measured-runtime-table")]
    cost_path = root / "costs.pkl"
    payload = pickle.loads(cost_path.read_bytes())
    return argv, cost_path, payload


def test_cli_new_affected_all_rates_allocation_stops_without_numeric_decision(tmp_path, monkeypatch):
    from prismaquant import allocator
    argv, source, payload = _cli_fixture(tmp_path)
    payload["provenance"]["depth_price_scope"] = _payload()[0]["provenance"]["depth_price_scope"]
    source.write_bytes(pickle.dumps(payload))
    original = source.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((ValueError, SystemExit), match="(?i)(depth|authoriz|flat)"):
        allocator.main()
    assert source.read_bytes() == original
    assert not (tmp_path / "layer.json").exists()
    assert not list((tmp_path / "seeds").glob("*.json"))


@pytest.mark.parametrize("control", ["stock", "uniform_rate", "unrelated_model"])
def test_cli_controls_keep_assignments_prices_metadata_and_source_identical(control, tmp_path, monkeypatch):
    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment
    factory = getattr(priced_cell.CostPricingPolicy, "from_depth_profile", None)
    assert callable(factory), "#2774: no scoped depth-price policy to compare with stock"
    argv, source, payload = _cli_fixture(tmp_path)
    if control == "uniform_rate":
        payload["provenance"]["depth_price_scope"] = _payload(scope="uniform_rate")[0]["provenance"]["depth_price_scope"]
    elif control == "unrelated_model":
        payload["provenance"]["model_id"] = "unrelated-control-model"
    source.write_bytes(pickle.dumps(payload))
    raw = source.read_bytes()
    policy = factory(payload, profile_record=None, production=False, ucb_z=0)
    assert policy.depth_profile is None
    legacy = priced_cell.CostPricingPolicy(ucb_z=0, fisher_output_mse=False)
    for name, rows in payload["costs"].items():
        for fmt, row in rows.items():
            assert priced_cell.price_cell({}, row, format_name=fmt, policy=policy) == priced_cell.price_cell(
                {}, row, format_name=fmt, policy=legacy)
    monkeypatch.setattr(sys, "argv", list(argv))
    allocator.main()
    result = tmp_path / "layer.json"
    before_assignment = load_assignment(result)
    before_metadata = json.loads(result.read_text())["__prismaquant__"]
    assert "routed_depth_pricing" not in before_metadata
    monkeypatch.setattr(sys, "argv", argv + ["--depth-price-purpose", "research"])
    allocator.main()
    assert load_assignment(result) == before_assignment
    after_metadata = json.loads(result.read_text())["__prismaquant__"]
    # Compare stable policy output, not independently measured wall time.
    # Keep timing present and valid: production telemetry must not be removed
    # or falsified to satisfy this control. No other metadata is excluded.
    for metadata in (before_metadata, after_metadata):
        for diagnostics in metadata.get("solve_diagnostics", {}).values():
            if "solver_seconds" in diagnostics:
                elapsed = diagnostics.pop("solver_seconds")
                assert isinstance(elapsed, (int, float))
                assert math.isfinite(elapsed) and elapsed >= 0
    assert after_metadata == before_metadata
    assert source.read_bytes() == raw
