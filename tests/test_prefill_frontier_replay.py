"""Research replay exercises the real allocator writer; no GPU evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_prefill_frontier_shape_only_scope import _sweep, _v2_sweep_fixture

from prismaquant import prefill_frontier, shipcard, tessera_export_lane
from prismaquant.layer_config import LAYER_CONFIG_META_KEY, load_assignment
from prismaquant.measured_runtime_prices import identity_sha256


def _case(tmp_path, monkeypatch):
    argv = _v2_sweep_fixture(tmp_path, monkeypatch)
    code, doc = _sweep(tmp_path, argv)
    assert code == 0
    point = next(p for p in doc["points"] if p["feasible"])
    output = tmp_path / "replayed.json"
    return doc, point, output


def _replay(tmp_path, point, output):
    assert prefill_frontier.main([
        "replay", "--frontier", str(tmp_path / "frontier.json"),
        "--assignment-sha256", point["assignment_sha256"],
        "--layer-config", str(output),
    ]) == 0
    return json.loads(output.read_text())


def test_exact_assignment_and_full_metadata_through_the_only_writer(tmp_path, monkeypatch):
    doc, point, output = _case(tmp_path, monkeypatch)
    result = _replay(tmp_path, point, output)
    expected = load_assignment(point["assignment_path"])
    actual = load_assignment(output)
    # Canonical assignment bytes, not JSON whitespace or AutoRound entry spelling.
    assert json.dumps(actual, sort_keys=True).encode() == json.dumps(expected, sort_keys=True).encode()
    assert identity_sha256(actual) == point["assignment_sha256"]
    meta = result[LAYER_CONFIG_META_KEY]
    assert meta["schema"] == "prismaquant.layer_config_meta.v1"
    assert meta["research_only"] is True and meta["certifies_placement"] is False
    assert meta["fixed_resource_scope"] == doc["fixed_resource_scope"]
    assert meta["serve_constraints"]["predicted"]["device_memory_bytes"] is None
    assert meta["prefill_frontier_replay"]["probe_bound_by_sweep"] is True
    assert meta["target_profile"] == "research"
    assert "assignment_payload_bits_total" in meta
    assert not (tmp_path / "layer.json").exists()
    assert not (tmp_path / "pareto.csv").exists()
    card = shipcard.build_shipcard(tmp_path, build={"layer_config": str(output)}, lane="tessera")
    assert card["build"]["research_only"] is True
    assert card["build"]["certifies_placement"] is False
    assert "route.trace" in card["slots"]
    assert shipcard._verify_build_block(card) == []


def test_replay_metadata_matches_normal_writer_for_an_admitted_fixed_charge(tmp_path, monkeypatch):
    from test_allocator_measured_runtime_cli import admit_synthetic_table
    from test_prefill_frontier import _curve_fixture, _run

    from prismaquant import allocator

    admit_synthetic_table(monkeypatch)
    _, argv = _curve_fixture(tmp_path)
    _, doc = _run(tmp_path, ["--slo-grid", "auto"], argv)
    point = doc["points"][2]  # A mixed-format point rather than either endpoint.
    replayed = _replay(tmp_path, point, tmp_path / "replayed.json")
    ordinary = tmp_path / "ordinary.json"
    allocator.main([*argv, "--slo-prefill-p95-ttft-ms", str(point["slo_ms"]),
                    "--layer-config", str(ordinary)])
    expected = json.loads(ordinary.read_text())
    for key in ("research_only", "certifies_placement", "prefill_frontier_replay", "fixed_resource_scope"):
        replayed[LAYER_CONFIG_META_KEY].pop(key)
    # The measured solver wall clock is not part of the metadata comparison.
    for config in (replayed, expected):
        meta = config[LAYER_CONFIG_META_KEY]
        meta["measured_runtime_search"]["target_diagnostics"].pop("solver_seconds", None)
        for diagnostic in meta["solve_diagnostics"].values():
            diagnostic.pop("solver_seconds", None)
    assert replayed == expected


@pytest.mark.parametrize("key,bad", [("research_only", None), ("research_only", False),
                                     ("certifies_placement", None), ("certifies_placement", True)])
def test_missing_or_false_research_stamps_refused_at_intake_and_shipcard(tmp_path, monkeypatch, key, bad):
    _, point, output = _case(tmp_path, monkeypatch)
    result = _replay(tmp_path, point, output)
    card = shipcard.build_shipcard(tmp_path, build={"layer_config": str(output)}, lane="tessera")
    if bad is None:
        del result[LAYER_CONFIG_META_KEY][key]
        del card["build"][key]
    else:
        result[LAYER_CONFIG_META_KEY][key] = bad
        card["build"][key] = bad
    output.write_text(json.dumps(result))
    with pytest.raises(tessera_export_lane.TesseraExportLaneError, match=key):
        tessera_export_lane.preflight(tmp_path, assignment_path=output)
    with pytest.raises(ValueError, match=key):
        shipcard.build_shipcard(tmp_path, build={"layer_config": str(output)}, lane="tessera")
    assert any(key in p for p in shipcard._verify_build_block(card))


@pytest.mark.parametrize("mutation,match", [
    ("probe", "probe_sha256 mismatch"), ("cost", "cost_sha256 mismatch"),
    ("digest", "no feasible sweep point"), ("assignment", "does not hash"),
    ("scope", "fixed_resource_scope differs"), ("context", "runtime_context differs"),
    ("resolve", "re-solved assignment differs"), ("exists", "refuses to overwrite"),
])
def test_replay_refuses_drift_before_emission(tmp_path, monkeypatch, mutation, match):
    doc, point, output = _case(tmp_path, monkeypatch)
    if mutation in ("probe", "cost"):
        with Path(doc["provenance"][f"{mutation}_path"]).open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "digest":
        point = {**point, "assignment_sha256": "f" * 64}
    elif mutation == "assignment":
        path = Path(point["assignment_path"])
        payload = json.loads(path.read_text())
        name = next(k for k in payload if k != LAYER_CONFIG_META_KEY)
        payload[name] = "BF16"
        path.write_text(json.dumps(payload))
    elif mutation == "scope":
        doc["fixed_resource_scope"]["scope"] = "changed"
    elif mutation == "context":
        doc["provenance"]["runtime_context"]["prompt_tokens"] += 1
    elif mutation == "resolve":
        # Valid, self-consistent assignment file but NOT the solution at this SLO.
        path = Path(point["assignment_path"])
        payload = json.loads(path.read_text())
        expected = {k: v for k, v in payload.items() if k != LAYER_CONFIG_META_KEY}
        expected[next(iter(expected))] = "BF16"
        digest = identity_sha256(expected)
        payload = {**expected, LAYER_CONFIG_META_KEY: {**payload[LAYER_CONFIG_META_KEY],
                                                     "assignment_sha256": digest}}
        path.write_text(json.dumps(payload))
        old = point["assignment_sha256"]
        for p in doc["points"]:
            if p["assignment_sha256"] == old:
                p["assignment_sha256"] = digest
    elif mutation == "exists":
        output.write_text("do not replace")
    (tmp_path / "frontier.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        _replay(tmp_path, point, output)
    # Direct API gives the concrete diagnostic rather than argparse's exit(2).
    with pytest.raises((ValueError, OSError), match=match):
        prefill_frontier.replay(tmp_path / "frontier.json", point["assignment_sha256"], output)
    assert not output.exists() or output.read_text() == "do not replace"


def test_stub_cannot_bypass_replay_at_export_intake(tmp_path, monkeypatch):
    _, point, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(tessera_export_lane.TesseraExportLaneError, match="requires replay"):
        tessera_export_lane.preflight(tmp_path, assignment_path=point["assignment_path"])


def test_replay_allows_a_new_pb_checkout_cwd_for_bound_absolute_inputs(tmp_path, monkeypatch):
    _, point, output = _case(tmp_path, monkeypatch)
    other = tmp_path / "next-pb-checkout"
    other.mkdir()
    monkeypatch.chdir(other)
    assert _replay(tmp_path, point, output)[LAYER_CONFIG_META_KEY]["research_only"] is True
    assert identity_sha256(load_assignment(output)) == point["assignment_sha256"]


def test_legacy_sweep_replay_records_probe_was_not_bound(tmp_path, monkeypatch):
    doc, point, output = _case(tmp_path, monkeypatch)
    del doc["provenance"]["probe_sha256"]
    del doc["provenance"]["allocator_cwd"]
    (tmp_path / "frontier.json").write_text(json.dumps(doc))
    assert _replay(tmp_path, point, output)[LAYER_CONFIG_META_KEY]["prefill_frontier_replay"]["probe_bound_by_sweep"] is False


@pytest.mark.parametrize("fmt", [
    f"{family}_R{rate}" for family, rates in
    (("TESSERA_E2M1_K2", (768, 896)), ("TESSERA_E4M3_K1", (832, 1024)),
     ("TESSERA_BF16_K1", (832, 1024)))
    for rate in rates
])
def test_tessera_families_and_rates_reach_export_intake(tmp_path, monkeypatch, fmt):
    """Six real writer cases, not a fixed three-rung shortcut.

    Synthetic pricing/attestation only. Preflight infrastructure and source
    scope gates are stand-ins; priced-input intake and claim propagation are
    real. No encoding or serving is claimed by this CPU test.
    """
    import hashlib

    from conftest import down_convert_lane_table
    from test_allocator_measured_runtime_cli import _main_fixture
    from test_allocator_tessera_priced_inputs import SCALE, _campaign_outputs, _stamp_rows
    from test_prefill_frontier_shape_only_scope import (
        _promote_to_v2,
        _stand_in_for_the_gates,
    )
    from test_tessera_scope_endpoints import DENSE, IMAGE, _cli_scope

    from prismaquant import tessera_menu, tessera_runtime_contract

    # Synthetic scope table, NOT device qualification: borrow each family's
    # native dense cell grammar and explicitly declare the two test rates on
    # one fixture image. The real pin does not qualify this combined roster.
    payload = json.loads(tessera_runtime_contract.contract_path().read_text())
    templates = {}
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["structure"] == "dense":
            templates.setdefault((cell["family"], cell["regime"]), cell)
    rates = {"TESSERA_E2M1_K2": [768, 896], "TESSERA_E4M3_K1": [832, 1024],
             "TESSERA_BF16_K1": [832, 1024]}
    payload["lane_eligibility"]["cells"] = list(templates.values())
    payload["lane_eligibility"]["structures"] = ["dense"]
    for cell in payload["lane_eligibility"]["cells"]:
        cell["runtime"] = {"image": IMAGE, "execution_modes": ["eager"]}
        cell["rungs_q256"] = rates[cell["family"]]
    payload = down_convert_lane_table(payload, "tessera.lane-eligibility.v5")
    contract = tessera_runtime_contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    monkeypatch.setattr(tessera_menu, "tessera_runtime_contract", lambda: contract)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")
    capture, scales, digest = _campaign_outputs(tmp_path)
    _, argv = _main_fixture(tmp_path, units=(DENSE,), menu={fmt: (1.0, 2.0)},
                           target_bits="16", shape=(256, 256),
                           stats_extra={"router_path": None, "expert_id": None},
                           activation_max_abs={DENSE: 448.0 * 6.0 / 37.5})
    argv = argv[1:argv.index("--slo-prefill-p95-ttft-ms")]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    argv += ["--model-override", str(model), "--target-profile", "tessera_research_sm121",
             *_cli_scope()]
    _stamp_rows(argv, fmt=fmt, capture_sha256=digest, scale=SCALE)
    table_path = tmp_path / "runtime.json"
    table = json.loads(table_path.read_text())
    table["cost_sha256"] = hashlib.sha256((tmp_path / "costs.pkl").read_bytes()).hexdigest()
    table_path.write_text(json.dumps(table))
    _promote_to_v2(tmp_path)
    _stand_in_for_the_gates(monkeypatch)
    _, doc = _sweep(tmp_path, argv)
    point = next(p for p in doc["points"] if p["feasible"])
    output = tmp_path / "replayed.json"
    result = _replay(tmp_path, point, output)
    assert load_assignment(output) == {DENSE: fmt}
    meta = result[LAYER_CONFIG_META_KEY]
    assert meta["tessera_hessian"]["capture_sha256"] == digest
    assert meta["tessera_serving_scope"]
    assert meta["serving_lane_provenance"]
    for name, value in {
        "require_declared_structure": "dense", "require_serving_target": None,
        "require_executes_derived_from_contract": (), "require_producer_tools": (),
        "require_producer_repo_is_pinned": (), "require_release_pin": None,
        "require_assignment_scope": None,
    }.items():
        monkeypatch.setattr(tessera_export_lane, name, lambda *a, _value=value, **k: _value)
    report = tessera_export_lane.preflight(tmp_path, assignment_path=output,
                                          hessian_path=capture, input_scales_path=scales)
    assert report["build"]["research_only"] is True
    assert report["build"]["certifies_placement"] is False
    card = shipcard.build_shipcard(tmp_path, build=report["build"], lane="tessera")
    assert shipcard._verify_build_block(card) == []
    assert card["build"]["prefill_frontier_replay"] == meta["prefill_frontier_replay"]
