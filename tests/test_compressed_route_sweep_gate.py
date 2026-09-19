"""Principle 14's serve-side leg on the compressed-tensors lane (#631).

The fixture is a REAL sweep: `validate_native_export --route-sweep-out` wrote
it from its own eager smoke inside the pinned vLLM image on sparklina, and the
`config.json` beside it is the artifact's own. Nothing here imports vLLM or
torch — the attestation travels as data, which is the constraint the design is
built around (`AGENTS.md` forbids vendoring the serving runtime in a test).

Every refusal case below mutates the OBSERVATION or the PRICE, never the
gate: a bad input proves a check runs. The companion driver mutation (dropping
the activation contract from the comparison) is run separately and recorded on
the PR.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from prismaquant import compressed_route_sweep_gate as gate
from prismaquant import shipcard

FIXTURE = Path(__file__).parent / "fixtures" / "compressed_route_sweep_0p6b"


def _config() -> dict:
    return json.loads((FIXTURE / "config.json").read_text())


def _sweep() -> dict:
    return json.loads((FIXTURE / "sweep_rank0.json").read_text())


def _compare(sweep: dict | None, *, expected_ranks: int = 1, config=None):
    text = None if sweep is None else json.dumps(sweep)
    return gate.compare_route_sweeps(
        [("rank0:sweep_rank0.json", text)],
        expected_ranks=expected_ranks,
        config=_config() if config is None else config,
    )


def _modules(sweep: dict) -> list[dict]:
    return sweep["ranks"][0]["modules"]


def _first_quantized(sweep: dict) -> dict:
    for row in _modules(sweep):
        if row.get("scheme"):
            return row
    raise AssertionError("fixture carries no quantized module")


# --------------------------------------------------------------------------
# The real observation agrees with the real price.
# --------------------------------------------------------------------------
def test_real_sweep_agrees_with_the_artifacts_own_price():
    verdict = _compare(_sweep())
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    # 28 layers x (qkv_proj, gate_up_proj, o_proj, down_proj).
    assert verdict["histogram"] == {"A4:float/g16:dynamic": 112}
    # The kernel is recorded and NOT judged; it must still be visible.
    assert verdict["kernels"] == {
        "A4:float/g16:dynamic": {"FlashInferCutlassNvFp4LinearKernel": 112}}
    # The KV-cache method modules are priced by no config group and are not a
    # refusal; they are named rather than dropped.
    assert len(verdict["unpriced_modules"]) == 28
    assert all(name.endswith(".attn") for name in verdict["unpriced_modules"])


def test_verdict_is_deterministic_so_verify_can_replay_it():
    assert _compare(_sweep()) == _compare(_sweep())


# --------------------------------------------------------------------------
# REFUSED: the observation disagrees with the price.
# --------------------------------------------------------------------------
def test_refuses_a_served_module_that_dropped_its_activation_quantization():
    sweep = _sweep()
    _first_quantized(sweep)["scheme_attrs"]["use_a16"] = "True"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "priced" in verdict["detail"] and "served" in verdict["detail"]


def test_refuses_a_priced_module_the_runtime_left_unquantized():
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["scheme"] = None
    row["scheme_module"] = None
    row["scheme_attrs"] = {}
    row["quant_method"] = "UnquantizedLinearMethod"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert row["name"] in verdict["detail"]


def test_refuses_a_module_vllms_own_isinstance_sweep_would_skip():
    """The trap: vLLM finalizes by `isinstance(quant_method, QuantizeMethodBase)`.

    A method that is not a subclass is skipped in SILENCE and dies on the first
    forward. This sweep filters on nothing and records the predicate, so the
    skipped module is a refusal here instead of an absence.
    """
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["quantize_method_base"] = False
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "QuantizeMethodBase" in verdict["detail"]
    assert "skipped in silence" in verdict["detail"]


def test_refuses_a_priced_module_that_never_dispatched():
    sweep = _sweep()
    _first_quantized(sweep)["dispatches"] = 0
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "dispatched ZERO forwards" in verdict["detail"]


def test_refuses_when_a_priced_target_reached_no_served_module():
    sweep = _sweep()
    victim = _first_quantized(sweep)["name"]
    sweep["ranks"][0]["modules"] = [
        row for row in _modules(sweep) if row["name"] != victim]
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "reached no served module" in verdict["detail"]


def test_refuses_a_served_module_the_artifact_never_priced():
    sweep = _sweep()
    _first_quantized(sweep)["name"] = "model.layers.99.self_attn.qkv_proj"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "never priced" in verdict["detail"]


def test_refuses_an_ignored_module_the_runtime_quantized():
    sweep = _sweep()
    for row in _modules(sweep):
        if row["name"] == "lm_head":
            row["scheme"] = "CompressedTensorsW4A4Fp4"
            row["scheme_attrs"] = {"use_a16": "False", "group_size": "16"}
            break
    else:  # pragma: no cover - the fixture has lm_head
        pytest.fail("fixture has no lm_head row")
    verdict = _compare(sweep)
    assert verdict["status"] == gate.REFUSED
    assert "ignore list" in verdict["detail"]


def test_refuses_when_the_price_itself_moved_under_the_observation():
    """A different config.json is a different price; the serve did not change."""
    config = _config()
    group = config["quantization_config"]["config_groups"]["group_0"]
    group["input_activations"]["group_size"] = 32
    verdict = _compare(_sweep(), config=config)
    assert verdict["status"] == gate.REFUSED


def test_refuses_two_ranks_that_served_different_contract_maps():
    sweep = _sweep()
    other = json.loads(json.dumps(sweep))
    other["ranks"][0]["rank"] = 1
    other["ranks"][0]["world_size"] = 2
    sweep["ranks"][0]["world_size"] = 2
    _first_quantized(other)["name"] = "model.layers.0.mlp.down_proj"
    verdict = gate.compare_route_sweeps(
        [("rank0", json.dumps(sweep)), ("rank1", json.dumps(other))],
        expected_ranks=2, config=_config())
    assert verdict["status"] == gate.REFUSED


# --------------------------------------------------------------------------
# NOT VERIFIED: there is no qualifying observation. Never a pass.
# --------------------------------------------------------------------------
def test_a_missing_sweep_is_not_verified_not_verified_by_absence():
    verdict = _compare(None)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "no sweep file" in verdict["detail"]


def test_fewer_sweeps_than_ranks_is_not_verified():
    verdict = _compare(_sweep(), expected_ranks=2)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "not a rank that agreed" in verdict["detail"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.update({"schema": "something.else/1"}), "is not"),
    (lambda s: s.update({"forward_observed": False}), "generated nothing"),
    (lambda s: s["load"].update({"enforce_eager": False}), "enforce_eager"),
    (lambda s: s["load"].update({"speculative_config": True}), "speculative"),
    (lambda s: s.update({"unavailable": "no LLM.apply_model"}), "apply_model"),
    (lambda s: s["ranks"][0].update({"rank_source": None}), "torch.distributed"),
])
def test_an_unqualified_sweep_is_not_verified(mutate, needle):
    sweep = _sweep()
    mutate(sweep)
    verdict = _compare(sweep)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert needle in verdict["detail"]


def test_an_unrecognized_scheme_class_is_not_verified_never_guessed():
    sweep = _sweep()
    _first_quantized(sweep)["scheme"] = "CompressedTensorsW6A6Imaginary"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "never observed" in verdict["detail"]


def test_a_scheme_missing_the_attributes_the_descriptor_reads_is_not_verified():
    sweep = _sweep()
    _first_quantized(sweep)["scheme_attrs"] = {}
    verdict = _compare(sweep)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "do not carry the numbers" in verdict["detail"]


def test_a_moe_method_with_no_scheme_is_not_verified_rather_than_passed():
    """No packed-MoE artifact has been swept; the gap refuses, it does not pass."""
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["scheme"] = None
    row["scheme_attrs"] = {}
    row["quant_method"] = "CompressedTensorsW4A4MoeMethod"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "no packed-MoE artifact has been swept" in verdict["detail"]


def test_an_artifact_that_prices_nothing_is_refused():
    config = _config()
    config["quantization_config"]["config_groups"] = {}
    with pytest.raises(gate.CompressedRouteSweepError):
        _compare(_sweep(), config=config)


# --------------------------------------------------------------------------
# The lane owes the slot, and the shipcard replays the comparison.
# --------------------------------------------------------------------------
def test_the_compressed_tensors_lane_owes_route_sweep():
    assert shipcard.ROUTE_SWEEP_SLOT in shipcard.lane_gate_slots(
        "compressed-tensors")
    assert shipcard.ROUTE_SWEEP_SLOT in shipcard.LANE_SLOT_VERIFIERS
    assert shipcard.ROUTE_SWEEP_SLOT in shipcard.lane_scoped_slots()


def test_a_compressed_tensors_card_is_required_to_close_route_sweep():
    card = {"lane": "compressed-tensors", "slots": {}}
    assert shipcard.ROUTE_SWEEP_SLOT in shipcard.required_slots(card)


def test_a_historical_card_with_no_lane_keeps_its_base_set():
    """Cards opened before #631 carry no lane and must not retroactively owe."""
    card = {"slots": {}}
    assert shipcard.ROUTE_SWEEP_SLOT not in shipcard.required_slots(card)


def test_the_run_prints_how_to_close_the_slot_it_opens(tmp_path, capsys):
    """A slot a run opens and prints no way to close teaches --force-unverified."""
    from prismaquant.lane_shipcard import main as lane_main

    model_dir = tmp_path / "artifact"
    model_dir.mkdir()
    shutil.copy(FIXTURE / "config.json", model_dir / "config.json")
    card = tmp_path / "card.json"
    assert lane_main([
        "open", "--lane", "compressed-tensors", "--artifact", str(model_dir),
        "--shipcard", str(card)]) == 0
    capsys.readouterr()
    assert lane_main([
        "gates", "--lane", "compressed-tensors", "--shipcard", str(card)]) == 0
    printed = capsys.readouterr().out
    assert f"[OPEN] route.sweep -> slots['{shipcard.ROUTE_SWEEP_SLOT}']" in printed
    assert "--route-sweep-out" in printed
    assert "fill-route-sweep" in printed


def _model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    shutil.copy(FIXTURE / "config.json", root / "config.json")
    return root


def test_make_route_sweep_record_refuses_a_disagreeing_sweep():
    sweep = _sweep()
    _first_quantized(sweep)["scheme_attrs"]["use_a16"] = "True"
    with pytest.raises(gate.CompressedRouteSweepError):
        shipcard.make_route_sweep_record(
            tool="test", model_sha=None,
            sweeps=[("rank0", json.dumps(sweep))], expected_ranks=1,
            config_json=(FIXTURE / "config.json").read_text())


def test_make_route_sweep_record_refuses_a_missing_sweep():
    with pytest.raises(gate.RouteSweepNotVerified):
        shipcard.make_route_sweep_record(
            tool="test", model_sha=None, sweeps=[("rank0", None)],
            expected_ranks=1,
            config_json=(FIXTURE / "config.json").read_text())


def test_verify_replays_the_comparison_and_refuses_a_hand_set_pass(tmp_path):
    model_dir = _model_dir(tmp_path)
    record = shipcard.make_route_sweep_record(
        tool="test", model_sha=None,
        sweeps=[("rank0", (FIXTURE / "sweep_rank0.json").read_text())],
        expected_ranks=1, config_json=(model_dir / "config.json").read_text())
    assert shipcard._verify_route_sweep_record(
        shipcard.ROUTE_SWEEP_SLOT, record, model_dir=model_dir) == []

    forged = json.loads(json.dumps(record))
    for row in forged["route_sweeps"][0]["sweep"]["ranks"][0]["modules"]:
        if row.get("scheme"):
            row["scheme_attrs"]["use_a16"] = "True"
            break
    problems = shipcard._verify_route_sweep_record(
        shipcard.ROUTE_SWEEP_SLOT, forged, model_dir=model_dir)
    assert problems and "priced" in problems[0]

    lying = json.loads(json.dumps(record))
    lying["sweep_verdict"]["histogram"] = {"A4:float/g16:dynamic": 1}
    problems = shipcard._verify_route_sweep_record(
        shipcard.ROUTE_SWEEP_SLOT, lying, model_dir=model_dir)
    assert any("differs from the replay" in p for p in problems)


def test_verify_refuses_carried_config_text_that_is_not_the_artifacts(tmp_path):
    model_dir = _model_dir(tmp_path)
    record = shipcard.make_route_sweep_record(
        tool="test", model_sha=None,
        sweeps=[("rank0", (FIXTURE / "sweep_rank0.json").read_text())],
        expected_ranks=1, config_json=(model_dir / "config.json").read_text())
    (model_dir / "config.json").write_text(
        (model_dir / "config.json").read_text() + " ")
    problems = shipcard._verify_route_sweep_record(
        shipcard.ROUTE_SWEEP_SLOT, record, model_dir=model_dir)
    assert any("compared against another price" in p for p in problems)


def test_verify_reads_an_absent_observation_as_unfilled_not_as_clean(tmp_path):
    problems = shipcard._verify_route_sweep_record(
        shipcard.ROUTE_SWEEP_SLOT, {"passed": True},
        model_dir=_model_dir(tmp_path))
    assert any("not a clean bill" in p for p in problems)


# --------------------------------------------------------------------------
# The CLI's three exits.
# --------------------------------------------------------------------------
def _card(tmp_path: Path) -> tuple[Path, Path]:
    from prismaquant.lane_shipcard import open_lane_shipcard

    model_dir = _model_dir(tmp_path)
    card = open_lane_shipcard(
        model_dir, "compressed-tensors", shipcard_path=tmp_path / "card.json")
    return model_dir, card


def _fill(card: Path, model_dir: Path, sweep: Path) -> int:
    from prismaquant.shipcard_cli import main

    return main(["fill-route-sweep", str(card), "--sweep", str(sweep),
                 "--expected-ranks", "1", "--model-dir", str(model_dir)])


def test_cli_fills_the_slot_from_the_real_sweep(tmp_path):
    model_dir, card = _card(tmp_path)
    assert _fill(card, model_dir, FIXTURE / "sweep_rank0.json") == 0
    filled = json.loads(card.read_text())["slots"][shipcard.ROUTE_SWEEP_SLOT]
    assert filled["passed"] is True
    assert filled["metrics"]["activation_contracts"] == {
        "A4:float/g16:dynamic": 112}


def test_cli_exits_1_and_leaves_the_slot_unfilled_on_disagreement(tmp_path):
    model_dir, card = _card(tmp_path)
    sweep = _sweep()
    _first_quantized(sweep)["scheme_attrs"]["use_a16"] = "True"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(sweep))
    assert _fill(card, model_dir, path) == 1
    assert json.loads(card.read_text())["slots"][
        shipcard.ROUTE_SWEEP_SLOT] is None


def test_cli_exits_3_when_the_sweep_is_missing(tmp_path):
    model_dir, card = _card(tmp_path)
    assert _fill(card, model_dir, tmp_path / "absent.json") == 3
    assert json.loads(card.read_text())["slots"][
        shipcard.ROUTE_SWEEP_SLOT] is None


def test_cli_exits_3_on_a_graph_arm_sweep(tmp_path):
    model_dir, card = _card(tmp_path)
    sweep = _sweep()
    sweep["load"]["enforce_eager"] = False
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(sweep))
    assert _fill(card, model_dir, path) == 3


# --------------------------------------------------------------------------
# PQ #706: method classes that carry no scheme read through METHOD_ACTIVATION.
# No packed-MoE artifact has been swept, so the table is empty and the gap
# refuses by name. The mechanics below run on a synthetic test-only entry
# that is removed again, never on a method name read from vLLM's source.
# --------------------------------------------------------------------------
def test_an_unobserved_moe_method_names_its_method_class():
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["scheme"] = None
    row["scheme_attrs"] = {}
    row["quant_method"] = "CompressedTensorsW4A4MoeMethod"
    verdict = _compare(sweep)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "CompressedTensorsW4A4MoeMethod" in verdict["detail"]
    assert "METHOD_ACTIVATION" in verdict["detail"]


def test_a_swept_method_class_is_read_from_its_own_fields():
    """The registry path end to end: a swept method's own fields price it."""
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["scheme"] = None
    row["scheme_attrs"] = {}
    row["quant_method"] = "TestOnlySweptMoeMethod"
    gate.METHOD_ACTIVATION["TestOnlySweptMoeMethod"] = (
        lambda swept: {"quantized": True, "num_bits": 4, "type": "float",
                       "group_size": 16, "dynamic": True})
    try:
        verdict = _compare(sweep)
    finally:
        del gate.METHOD_ACTIVATION["TestOnlySweptMoeMethod"]
    assert verdict["status"] == gate.AGREE, verdict["detail"]


def test_a_swept_method_missing_its_numbers_is_not_verified():
    sweep = _sweep()
    row = _first_quantized(sweep)
    row["scheme"] = None
    row["scheme_attrs"] = {}
    row["quant_method"] = "TestOnlySweptMoeMethod"
    gate.METHOD_ACTIVATION["TestOnlySweptMoeMethod"] = lambda swept: None
    try:
        verdict = _compare(sweep)
    finally:
        del gate.METHOD_ACTIVATION["TestOnlySweptMoeMethod"]
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "do not carry the numbers" in verdict["detail"]
