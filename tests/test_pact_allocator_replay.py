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
import math
import sys
from types import SimpleNamespace
from pathlib import Path

from test_shape_runtime_prices import checker_sdk_fixture  # noqa: F401

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


def _fixture(tmp_path, monkeypatch, *, units=None):
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

    from test_shape_runtime_prices import observation_fixture, _consume_observations

    regime = srp.regime_for_m(M)
    observations = []
    for fmt, milliseconds in TIMES.items():
        family, rate = fmt.rsplit("_R", 1)
        cell = next(c for c in eligibility.cells if c.family == family and c.regime == regime)
        grid = family.split("_")[1]
        route = {"BF16": "TESSERA_VALUE_W16", "E4M3": "TESSERA_FP8", "E2M1": "TESSERA_FP4"}[grid]
        binding = observation_fixture(
            tmp_path, agent=family, m=M, family=family, grid=grid, route=route,
            rate_q256=int(rate), kernel_lane=cell.executes[0], runtime_image=IMAGE,
            samples=(milliseconds, milliseconds * 1.01, milliseconds * 0.99))
        observations.append(Path(binding["path"]))
    table = _consume_observations(observations, table_id="pact-fixture")
    table_path = tmp_path / "shape_table.json"
    srp.write_shape_table(table, table_path)

    chosen_units = tuple(units or (DENSE,))
    capture, scales, digest = (_campaign_outputs(tmp_path) if units is None
                              else _campaign_outputs(tmp_path, units=chosen_units))
    _, argv = _main_fixture(tmp_path, units=chosen_units, menu=MENU, target_bits="16",
                           shape=(256, 256), stats_extra={"router_path": None, "expert_id": None},
                           activation_max_abs={unit: 448.0 * 6.0 / 37.5 for unit in chosen_units})
    argv = argv[1:argv.index("--measured-runtime-table")]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    argv += ["--model-override", str(model), "--target-profile", "tessera_research_sm121",
             *_cli_scope(), "--pact-shape-table", str(table_path), "--pact-regime", str(M),
             "--pact-tensor-parallel", "1"]
    for fmt in MENU:
        if units is None:
            _stamp_rows(argv, fmt=fmt, capture_sha256=digest, scale=SCALE)
        else:
            _stamp_rows(argv, fmt=fmt, capture_sha256=digest, scale=SCALE, units=chosen_units)
    return SimpleNamespace(argv=argv, dense=DENSE, capture=capture, scales=scales,
                           table_path=table_path, units=chosen_units)


def _hull(tmp_path, argv, *own):
    output = tmp_path / "hull.json"
    assert prefill_frontier.main(["--output", str(output), *own, "--", *argv]) == 0
    return output, json.loads(output.read_text())


def _replay(frontier, digest, output):
    return prefill_frontier.main(["replay", "--frontier", str(frontier),
                                  "--assignment-sha256", digest, "--layer-config", str(output)])


@pytest.mark.parametrize("mutation", ["digest", "invalid_json", "bare_observation", "unknown_schema"])
def test_receipt_refuses_before_frontier_publication(tmp_path, monkeypatch, mutation):
    import hashlib
    from prismaquant import shape_runtime_prices as srp

    case = _fixture(tmp_path, monkeypatch)
    receipt = Path(json.loads(case.table_path.read_text())["rows"][0]["measurement"]["receipt_path"])
    if mutation == "invalid_json":
        receipt.write_bytes(b"not JSON")
    elif mutation == "bare_observation":
        receipt.write_text(json.dumps({"schema": srp.SHAPE_TIME_OBSERVATION_SCHEMA}))
    elif mutation == "unknown_schema":
        receipt.write_text(json.dumps({"schema": "foreign.fixture.v1"}))
    else:
        receipt.write_text(json.dumps({"changed": True}))
    # A matching digest is not checker authority, nor does it make malformed
    # JSON acceptable. The first case separately exercises byte integrity.
    if mutation != "digest":
        table = json.loads(case.table_path.read_text())
        for row in table["rows"]:
            if Path(row["measurement"]["receipt_path"]).resolve() == receipt.resolve():
                row["measurement"]["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        case.table_path.write_text(json.dumps(table))
    output = tmp_path / "refused.json"
    with pytest.raises(SystemExit):
        prefill_frontier.main(["--output", str(output), "--", *case.argv])
    assert not output.exists()
    assert not output.with_suffix(".json.assignments").exists()

def test_hull_replay_and_export_intake_carry_the_research_standing(tmp_path, monkeypatch,
                                                                     capsys):
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

    # The selection record (PQ #1585) rides beside the replay stamp, bound by digest.
    record = meta["pact_selection"]
    assert record["schema"] == "prismaquant.pact_selection.v1"
    assert replay["pact_selection_sha256"] == record["identity_sha256"]
    assert record["regime_m"] == M and record["table_identity"] == replay["table_identity"]
    assert record["frontier_scope"] == "lower_convex_hull_vertices"
    assert record["materiality"]["verdict"] == "material"
    assert set(record["picks"]) == {"high_accuracy", "balanced", "high_prefill"}
    by_id = {r["point_id"]: r for r in record["roster"]}
    assert interior["assignment_sha256"] in {r["assignment_sha256"] for r in by_id.values()}
    accuracy = by_id[record["picks"]["high_accuracy"]["point_id"]]
    assert accuracy["assignment_sha256"] == doc["vertices"][0]["assignment_sha256"]
    assert accuracy["kernel_lanes"], "the roster carries the kernel-lane histogram"
    assert doc["pact_selection"]["identity_sha256"] == record["identity_sha256"]

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
    assert card["build"]["pact_selection"] == record
    # A card whose record was edited fails its own identity (PQ #1585).
    forged = json.loads(json.dumps(card))
    forged["build"]["pact_selection"]["picks"]["balanced"]["point_id"] = "forged"
    assert any("pact_selection" in problem for problem in shipcard._verify_build_block(forged))
    # A build that disagrees with the recipe's record is refused at build time.
    other = json.loads(json.dumps(report["build"]))
    other["pact_selection"]["regime_m"] = M + 1
    with pytest.raises(Exception, match="pact_selection"):
        shipcard.build_shipcard(tmp_path, build=other, lane="tessera")

    # Publication refuses the research standing (PQ #1598); every slot is
    # closed, so the research stamp is the only thing it can refuse on.
    from test_publish_artifact import _argv, _artifact, _close_all_slots
    from tools.publish_artifact import main as publish_cli

    model_dir = _artifact(tmp_path, name="pact-exported")
    card = shipcard.build_shipcard(model_dir, build={**report["build"],
                                                     "achieved_bpp": {"value": 4.75}})
    shipcard.write_shipcard(model_dir / "shipcard.json", card)
    _close_all_slots(model_dir)
    capsys.readouterr()
    assert publish_cli(_argv(model_dir)) == 1
    err = capsys.readouterr().err
    assert "build.research_only is True" in err and "prefill_frontier_replay_claim" in err
    assert "nothing was uploaded" in err


def test_replay_refuses_a_moved_shape_table(tmp_path, monkeypatch):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "50")
    table = json.loads(case.table_path.read_text())
    table["rows"][0]["measurement"]["samples_ms"][0] += 0.5
    case.table_path.write_text(json.dumps(table))
    with pytest.raises(SystemExit):
        _replay(frontier, doc["vertices"][1]["assignment_sha256"], tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("field", ["operator_sum_ms", "predicted_dloss", "candidate_bytes"])
def test_hull_replay_refuses_changed_numeric_claims(tmp_path, monkeypatch, field):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "20")
    vertex = doc["vertices"][1]
    vertex[field] += 1
    frontier.write_text(json.dumps(doc))
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit) as refused:
        _replay(frontier, vertex["assignment_sha256"], output)
    assert refused.value.code == 2
    assert not output.exists()


def test_hull_replay_refuses_boolean_zero_loss(tmp_path, monkeypatch):
    monkeypatch.setitem(MENU, SLOW, (0.0, 1.0))
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "20")
    vertex = doc["vertices"][0]
    assert vertex["predicted_dloss"] == 0.0
    vertex["predicted_dloss"] = False
    frontier.write_text(json.dumps(doc))
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit) as refused:
        _replay(frontier, vertex["assignment_sha256"], output)
    assert refused.value.code == 2
    assert not output.exists()


def test_hull_replay_accepts_integer_real_claims(tmp_path, monkeypatch):
    monkeypatch.setitem(MENU, SLOW, (0.0, 1.0))
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "20")
    vertex = doc["vertices"][0]
    vertex["predicted_dloss"] = 0
    vertex["operator_sum_ms"] = int(vertex["operator_sum_ms"])
    frontier.write_text(json.dumps(doc))
    assert _replay(frontier, vertex["assignment_sha256"], tmp_path / "out.json") == 0


@pytest.mark.parametrize("extra,diagnostic", [
    (["--serve-device-budget-bytes", "1000000"], "tessera#624"),
    (["--slo-prefill-p95-ttft-ms", "5"], "mutually exclusive"),
    # The whole-artifact cap prices the bytes outside the units from the
    # source checkpoint; this fixture's model directory holds none.
    (["--target-disk-gb", "1", "--artifact-overhead-reserve-bytes", "1"],
     "could not be priced"),
])
def test_pact_refuses_budgets_it_cannot_price(tmp_path, monkeypatch, capsys, extra, diagnostic):
    case = _fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as refused:
        prefill_frontier.main(["--output", str(tmp_path / "h.json"), "--", *case.argv, *extra])
    # argparse prints its refusal; a SystemExit raised with a message carries it.
    assert diagnostic in capsys.readouterr().err + str(refused.value.code)


def _whole_artifact_card(tmp_path, case):
    """The shared tiny header/card fixture, including fixed and member-name bytes."""
    import struct

    from prismaquant import footprint
    from prismaquant import format_registry as fr

    tensors = {f"{case.dense}.weight": ("BF16", (256, 256)), "model.norm.weight": ("BF16", (64,))}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        nbytes = footprint._ST_DTYPE_BYTES[dtype] * shape[0] * (shape[1] if len(shape) > 1 else 1)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    (tmp_path / "model" / "model-00001.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)
    floor = 64 * 2  # the norm ships verbatim
    unit = {fmt: fr.get_format(fmt).memory_bytes_for_shape((256, 256)) for fmt in MENU}
    # A Tessera unit ships as one TSRFUSE1 member, and the member frame carries
    # the projection role (the last component of the tensor name, no leaf) as
    # UTF-8.  A unit candidate's bytes leave that name to the caller (#1609),
    # so the whole-artifact accountant adds it per unit; it is neither a unit
    # candidate byte nor part of the non-unit payload (#1716).
    from prismaquant.name_projection import strip_weight_leaf
    role = strip_weight_leaf(case.dense).rsplit(".", 1)[-1]
    assert role and role != "weight"  # a projection role, never the parameter leaf
    names = len(role.encode("utf-8"))
    reserve = 1000
    # The card admits MID and FAST, never SLOW.
    disk_gb = repr((floor + names + unit[MID] + reserve + 500) / footprint.GB)
    card = math.floor(float(disk_gb) * footprint.GB)
    assert (floor + names + unit[MID] + reserve <= card
            < floor + names + unit[SLOW] + reserve)
    return SimpleNamespace(floor=floor, names=names, unit=unit, reserve=reserve,
                           disk_gb=disk_gb, card=card)


def test_a_whole_artifact_card_binds_the_hull_and_stamps_the_replay(tmp_path, monkeypatch):
    """``--target-disk-gb`` is an on-disk cap (footprint.py), not a device budget."""
    case = _fixture(tmp_path, monkeypatch)
    card_case = _whole_artifact_card(tmp_path, case)
    floor, names, unit, reserve, disk_gb, card = (
        card_case.floor, card_case.names, card_case.unit, card_case.reserve,
        card_case.disk_gb, card_case.card)
    frontier, doc = _hull(tmp_path, [*case.argv, "--target-disk-gb", disk_gb,
                                     "--artifact-overhead-reserve-bytes", str(reserve)],
                          "--bootstrap-draws", "50")

    assert doc["whole_artifact_budget"] == {
        "budget_bytes": card, "reserve_bytes": reserve, "non_unit_payload_bytes": floor,
        "unit_name_bytes": names, "unit_budget_bytes": card - reserve - floor - names}
    assert doc["max_memory_bytes"] == card - reserve - floor - names
    got = [load_assignment(v["assignment_path"])[case.dense] for v in doc["vertices"]]
    assert got == [MID, FAST]
    assert [v["whole_artifact_upper_bound_bytes"] for v in doc["vertices"]] == [
        floor + names + unit[MID] + reserve, floor + names + unit[FAST] + reserve]
    assert all(v["feasible"] for v in doc["vertices"])

    output = tmp_path / "replayed.json"
    assert _replay(frontier, doc["vertices"][0]["assignment_sha256"], output) == 0
    assert load_assignment(output) == {case.dense: MID}
    stamp = json.loads(output.read_text())[LAYER_CONFIG_META_KEY]["whole_artifact_budget"]
    assert stamp["budget_bytes"] == card
    assert stamp["selection_tensor_payload_bytes"] == floor + names + unit[MID]
    assert stamp["selection_non_tensor_reserve_bytes"] == reserve


def test_the_exact_probe_bounds_are_flags_recorded_and_replayed(tmp_path, monkeypatch, capsys):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, [*case.argv, "--pact-max-states", "250000",
                                     "--pact-max-transitions", "9000000"],
                          "--bootstrap-draws", "50")
    assert (doc["max_states"], doc["max_transitions"]) == (250000, 9000000)
    rss = doc["hull_peak_rss_kib"]
    assert 0 < rss["before_hull"] <= rss["after_hull"]
    output = tmp_path / "replayed.json"
    assert _replay(frontier, doc["vertices"][0]["assignment_sha256"], output) == 0
    # The defaults are the solver's own, stated.
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    _, plain = _hull(plain_dir, case.argv, "--bootstrap-draws", "50")
    from prismaquant import pact_hull
    assert (plain["max_states"], plain["max_transitions"]) == (
        pact_hull.DEFAULT_MAX_STATES, pact_hull.DEFAULT_MAX_TRANSITIONS)
    for flag in ("--pact-max-states", "--pact-max-transitions"):
        with pytest.raises(SystemExit):
            prefill_frontier.main(["--output", str(tmp_path / "x.json"), "--",
                                   *case.argv, flag, "0"])
        assert "must be a positive integer" in capsys.readouterr().err


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


def test_the_claim_binds_the_selection_record_to_its_replay(tmp_path, monkeypatch):
    case = _fixture(tmp_path, monkeypatch)
    frontier, doc = _hull(tmp_path, case.argv, "--bootstrap-draws", "50")
    interior = doc["vertices"][1]
    output = tmp_path / "replayed.json"
    assert _replay(frontier, interior["assignment_sha256"], output) == 0
    meta = json.loads(output.read_text())[LAYER_CONFIG_META_KEY]
    good = prefill_frontier_replay_claim(meta)
    assert good["pact_selection"] == meta["pact_selection"]

    def refused(mutate, match):
        bad = json.loads(json.dumps(meta))
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            prefill_frontier_replay_claim(bad)

    refused(lambda m: m["pact_selection"]["picks"]["balanced"].update(point_id="x"),
            "pact_selection")
    refused(lambda m: m["prefill_frontier_replay"].update(pact_selection_sha256="0" * 64),
            "pact_selection")
    refused(lambda m: m["prefill_frontier_replay"].update(assignment_sha256="1" * 64),
            "roster")
    refused(lambda m: m.pop("prefill_frontier_replay"), "pact_selection")
    # A recipe with no record (legacy) stays accepted and carries none.
    legacy = json.loads(json.dumps(meta))
    del legacy["pact_selection"]
    legacy["prefill_frontier_replay"].pop("pact_selection_sha256")
    assert "pact_selection" not in prefill_frontier_replay_claim(legacy)
