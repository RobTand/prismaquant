"""PACT mode through the real allocator (PQ #1584): hull, replay, export intake.

One dense unit, three Tessera rungs, and a shape-time table that prices the
unit's served operator at one regime M. The allocator admits the table against
the pinned scope, builds the exact hull through ``prefill_frontier``, and a
replay re-runs the ONE probe that found the interior vertex and writes it
through the allocator's only layer-config writer. Export intake reads the
replay's research standing onto the build record and the shipcard.

Synthetic pricing and attestation only: the scope is the real tracked pin
(whose live gate runs), while the installed contract's eligibility table is
swapped for the same synthetic table the attested menu reads, as
``test_prefill_frontier_replay`` does for the SLO sweep. No time here is GPU
evidence.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from prismaquant import prefill_frontier, shipcard, tessera_export_lane
from prismaquant.layer_config import (
    LAYER_CONFIG_META_KEY, load_assignment, prefill_frontier_replay_claim,
)
from prismaquant.measured_runtime_prices import identity_sha256

SLOW, MID, FAST = "TESSERA_BF16_K1_R1024", "TESSERA_E4M3_K1_R1024", "TESSERA_E2M1_K2_R896"
#: predicted Δloss per rung (the second number feeds only the measured-table
#: fixture this argv is cut from, never PACT).
MENU = {SLOW: (1.0, 1.0), MID: (2.0, 1.0), FAST: (10.0, 1.0)}
#: Operator time per rung at M. MID lies strictly below the SLOW-FAST chord
#: (at 5 ms the chord reads Δloss 8.5), so it is the interior hull vertex.
TIMES = {SLOW: 10.0, MID: 5.0, FAST: 4.0}
M = 2048


def _fixture(tmp_path, monkeypatch):
    from conftest import down_convert_lane_table
    from test_allocator_measured_runtime_cli import _main_fixture
    from test_allocator_tessera_priced_inputs import SCALE, _campaign_outputs, _stamp_rows
    from test_tessera_scope_endpoints import DENSE, IMAGE, _cli_scope

    from prismaquant import lane_eligibility, tessera_menu, tessera_render
    from prismaquant import shape_runtime_prices as srp
    from prismaquant import tessera_runtime_contract
    from prismaquant import tessera_serving_runtime_pin as pins

    payload = json.loads(tessera_runtime_contract.contract_path().read_text())
    templates = {}
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["structure"] == "dense":
            templates.setdefault((cell["family"], cell["regime"]), cell)
    rates = {"TESSERA_E2M1_K2": [896], "TESSERA_E4M3_K1": [1024], "TESSERA_BF16_K1": [1024]}
    payload["lane_eligibility"]["cells"] = list(templates.values())
    payload["lane_eligibility"]["structures"] = ["dense"]
    for cell in payload["lane_eligibility"]["cells"]:
        cell["runtime"] = {"image": IMAGE, "execution_modes": ["eager"]}
        cell["rungs_q256"] = rates[cell["family"]]
    payload = down_convert_lane_table(payload, "tessera.lane-eligibility.v5")
    contract = tessera_runtime_contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    monkeypatch.setattr(tessera_menu, "tessera_runtime_contract", lambda: contract)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")

    # The pinned scope the shape table must equal: the REAL tracked pin (its
    # live gate runs against the installed contract), and the installed
    # contract's eligibility swapped for the same synthetic table the attested
    # menu reads.
    pin = pins.load_tessera_serving_runtime_pin()
    eligibility = lane_eligibility._parse_table(
        payload["lane_eligibility"], payload["formats"], "", pin.serving_commit,
        pin.contract_sha256, native_extensions=payload["native_extensions"])
    published = {row["family"]: row for row in payload["formats"]}
    monkeypatch.setattr(tessera_render, "_pinned_serving_table", lambda: (eligibility, published))

    import hashlib

    receipt = tmp_path / "bench.json"
    receipt.write_text("Synthetic CPU test fixture, not GPU measurement evidence.\n")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    regime = srp.regime_for_m(M)
    rows = []
    for fmt, milliseconds in TIMES.items():
        family, rate = fmt.rsplit("_R", 1)
        cell = next(c for c in eligibility.cells if c.family == family and c.regime == regime)
        symbol, decoder = cell.executes[0]
        rows.append({"structure": "dense", "rank_local_shape": "256x256", "family": family,
                     "rate_q256": int(rate), "m": M,
                     "kernel_lane": {"symbol": symbol, "decoder": decoder},
                     "measurement": {"method": "cuda_events",
                                     "samples_ms": [milliseconds, milliseconds * 1.01,
                                                    milliseconds * 0.99],
                                     "warmup_iterations": 10, "receipt_path": receipt.name,
                                     "receipt_sha256": receipt_sha}})
    table = {"schema": srp.SCHEMA, "table_id": "pact-fixture", "status": "proposal_data",
             "composition": "sequential_operator_sum", "claims": dict(srp.CLAIMS),
             "context": {"runtime_image_digest": IMAGE, "tessera_commit": pin.serving_commit,
                         "contract_sha256": pin.contract_sha256, "tensor_parallel": 1, "platform": "sm_121",
                         "execution_mode": "eager", "residency": "resident", "batch_size": 1,
                         "regimes": [M]},
             "rows": rows, "rate_pools": []}
    table_path = tmp_path / "shape_table.json"
    table_path.write_text(json.dumps(table))

    capture, scales, digest = _campaign_outputs(tmp_path)
    _, argv = _main_fixture(tmp_path, units=(DENSE,), menu=MENU, target_bits="16",
                           shape=(256, 256), stats_extra={"router_path": None, "expert_id": None},
                           activation_max_abs={DENSE: 448.0 * 6.0 / 37.5})
    argv = argv[1:argv.index("--measured-runtime-table")]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    argv += ["--model-override", str(model), "--target-profile", "tessera_research_sm121",
             *_cli_scope(), "--pact-shape-table", str(table_path), "--pact-regime", str(M),
             "--pact-tensor-parallel", "1"]
    for fmt in MENU:
        _stamp_rows(argv, fmt=fmt, capture_sha256=digest, scale=SCALE)
    return SimpleNamespace(argv=argv, dense=DENSE, capture=capture, scales=scales,
                           table_path=table_path)


def _hull(tmp_path, argv, *own):
    output = tmp_path / "hull.json"
    assert prefill_frontier.main(["--output", str(output), *own, "--", *argv]) == 0
    return output, json.loads(output.read_text())


def _replay(frontier, digest, output):
    return prefill_frontier.main(["replay", "--frontier", str(frontier),
                                  "--assignment-sha256", digest, "--layer-config", str(output)])


def test_hull_replay_and_export_intake_carry_the_research_standing(tmp_path, monkeypatch):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "200")

    assert doc["schema"] == prefill_frontier.PACT_SCHEMA
    assert doc["candidate_generator"] == "lower_convex_hull_dichotomic"
    assert doc["time_claim"] == "operator_sum_proposal"
    assert doc["research_only"] is True and doc["certifies_placement"] is False
    assert doc["regime_m"] == M and doc["tensor_parallel"] == 1 and doc["fixed_prefill_ms"] == 0.0
    # Accuracy end first; the interior vertex is MID.
    got = [load_assignment(v["assignment_path"])[case.dense] for v in doc["vertices"]]
    assert got == [SLOW, MID, FAST]
    assert [v["operator_sum_ms"] for v in doc["vertices"]] == pytest.approx(
        [TIMES[SLOW], TIMES[MID], TIMES[FAST]])
    assert doc["probe_count"] == 2 * len(got) - 1
    assert doc["gap_report"]["unpriced_options"] == 0
    assert all(v["feasible"] for v in doc["vertices"])
    boot = doc["vertices"][1]["operator_sum_ms_bootstrap"]
    assert boot["p2.5"] <= TIMES[MID] * 1.01 and boot["p97.5"] >= TIMES[MID] * 0.99

    # The stub each vertex is published as is not an exportable config.
    interior = doc["vertices"][1]
    with pytest.raises(tessera_export_lane.TesseraExportLaneError, match="requires replay"):
        tessera_export_lane.preflight(tmp_path, assignment_path=interior["assignment_path"])

    output = tmp_path / "replayed.json"
    assert _replay(frontier, interior["assignment_sha256"], output) == 0
    assert load_assignment(output) == {case.dense: MID}
    assert identity_sha256(load_assignment(output)) == interior["assignment_sha256"]
    meta = json.loads(output.read_text())[LAYER_CONFIG_META_KEY]
    assert meta["research_only"] is True and meta["certifies_placement"] is False
    assert meta["candidate_generator"] == "lower_convex_hull_dichotomic"
    assert meta["time_claim"] == "operator_sum_proposal"
    replay = meta["prefill_frontier_replay"]
    assert replay["schema"] == prefill_frontier.REPLAY_SCHEMA
    assert replay["regime_m"] == M and replay["vertex"] == 1
    assert replay["probe_weights"] == interior["finding_probe"]["weights"]
    assert not (tmp_path / "layer.json").exists()

    for name, value in {
        "require_declared_structure": "dense", "require_serving_target": None,
        "require_executes_derived_from_contract": (), "require_producer_tools": (),
        "require_producer_repo_is_pinned": (), "require_release_pin": None,
        "require_assignment_scope": None,
    }.items():
        monkeypatch.setattr(tessera_export_lane, name, lambda *a, _value=value, **k: _value)
    report = tessera_export_lane.preflight(tmp_path, assignment_path=output,
                                          hessian_path=case.capture,
                                          input_scales_path=case.scales)
    assert report["build"]["research_only"] is True
    assert report["build"]["certifies_placement"] is False
    card = shipcard.build_shipcard(tmp_path, build=report["build"], lane="tessera")
    # build.research_only is what the publish gate reads (PQ #1598).
    assert card["build"]["research_only"] is True
    assert card["build"]["prefill_frontier_replay"]["candidate_generator"] == \
        "lower_convex_hull_dichotomic"
    assert shipcard._verify_build_block(card) == []


def test_replay_refuses_a_moved_shape_table(tmp_path, monkeypatch):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "50")
    table = json.loads(case.table_path.read_text())
    table["rows"][0]["measurement"]["samples_ms"][0] += 0.5
    case.table_path.write_text(json.dumps(table))
    with pytest.raises(SystemExit):
        _replay(frontier, doc["vertices"][1]["assignment_sha256"], tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("extra,diagnostic", [
    (["--serve-device-budget-bytes", "1000000"], "tessera#624"),
    (["--slo-prefill-p95-ttft-ms", "5"], "mutually exclusive"),
    (["--target-disk-gb", "1"], "not read in PACT mode"),
])
def test_pact_refuses_budgets_it_cannot_price(tmp_path, monkeypatch, capsys, extra, diagnostic):
    case = _fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        prefill_frontier.main(["--output", str(tmp_path / "h.json"), "--", *case.argv, *extra])
    assert diagnostic in capsys.readouterr().err


def test_pact_refuses_a_grid_and_a_missing_regime(tmp_path, monkeypatch, capsys):
    case = _fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        prefill_frontier.main(["--output", str(tmp_path / "h.json"), "--slo-grid", "auto",
                               "--", *case.argv])
    assert "has no grid" in capsys.readouterr().err
    cut = case.argv.index("--pact-regime")
    with pytest.raises(SystemExit):
        prefill_frontier.main(["--output", str(tmp_path / "h.json"), "--",
                               *case.argv[:cut], *case.argv[cut + 2:]])
    assert "requires a positive --pact-regime" in capsys.readouterr().err


def test_a_single_solve_refuses_pact(tmp_path, monkeypatch, capsys):
    from prismaquant import allocator

    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["allocator", *case.argv])
    with pytest.raises(SystemExit):
        allocator.main()
    assert "available only to prismaquant.prefill_frontier" in capsys.readouterr().err
    assert not (tmp_path / "layer.json").exists()


def test_the_one_claim_carries_every_research_standing():
    # A PACT stamp with no replay block is not an exportable config.
    with pytest.raises(ValueError, match="PACT hull config requires"):
        prefill_frontier_replay_claim({"candidate_generator": "lower_convex_hull_dichotomic"})
    # A measured-runtime single solve's own research stamp reaches the build.
    assert prefill_frontier_replay_claim(
        {"measured_runtime_search": {"research_only": True}}) == {"research_only": True}
    assert prefill_frontier_replay_claim({"measured_runtime_search": {}}) == {}
    assert prefill_frontier_replay_claim({}) == {}
