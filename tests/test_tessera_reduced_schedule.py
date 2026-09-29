"""Offline next-work data is not a measured wire or a production DP result."""
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from prismaquant.tessera_formats import TesseraFormatError
from prismaquant.tessera_rate_surface import StackRateSample


def _inputs() -> dict[str, Any]:
    records = {}
    ledger = []
    for name in ("a", "b", "c"):
        experts = tuple(range(8))
        sampled = (0, 1, 2, 3)
        reference = {e: {"gate": 2.0 ** (e / 8)} for e in experts}
        targets = {q: {e: {"gate": reference[e]["gate"] * factor}
                       for e in sampled} for q, factor in ((832, 2.0), (1088, 0.5))}
        records[name] = StackRateSample(
            stack=name, projections=("gate",), experts=experts,
            reference_q256=960, reference_mse=reference,
            sampled_experts=sampled, sampled_mse=targets,
            currency="uniform_output_mse",
        )
        ledger.extend((name, e, 960) for e in experts)
        ledger.extend((name, e, q) for q in targets for e in sampled)
    return dict(records=records, unit_bytes={n: {832: 10, 960: 20, 1088: 30}
                for n in records}, byte_budget=60,
                winners={"a": 832, "b": 960, "c": 1088}, measured_cells=ledger,
                draws=50, seed=495)


def _plan(**kwargs):
    from prismaquant.tessera_reduced_schedule import plan_reduced_schedule
    return plan_reduced_schedule(**kwargs)


def test_passing_real_gate_requests_only_missing_current_winner_cells():
    out = _plan(**_inputs())
    assert out["gate"]["passes"]
    assert out["state"] == "awaiting_selective_measurements"
    assert out["fallback_scope"] == "none"
    assert out["pending_cells"] == [
        {"stack": n, "expert": e, "rate_q256": q}
        for n, q in (("a", 832), ("c", 1088)) for e in range(4, 8)]
    assert out["winners"] == _inputs()["winners"]
    assert out["wire_ready"] is False
    assert out["gate_allocator"] == "greedy_diagnostic_not_production_dp"


def test_global_failure_requests_full_missing_band_not_invented_failed_stacks(monkeypatch):
    from prismaquant import tessera_reduced_schedule as planner
    from prismaquant.tessera_rate_surface import stack_transfer_regret_gate
    def fail(records, **kwargs):
        return {**stack_transfer_regret_gate(records, **kwargs), "passes": False}
    monkeypatch.setattr(planner, "stack_transfer_regret_gate", fail)
    out = _plan(**_inputs())
    assert out["state"] == "awaiting_full_measurements"
    assert out["fallback_scope"] == "all_stacks"
    assert out["pending_cells"] == [
        {"stack": n, "expert": e, "rate_q256": q}
        for n in ("a", "b", "c") for e in range(4, 8) for q in (832, 1088)]


def test_reference_winners_mean_coverage_only_not_wire_readiness():
    args = _inputs()
    args["winners"] = {n: 960 for n in args["records"]}
    out = _plan(**args)
    assert out["pending_cells"] == []
    assert out["state"] == "winner_cells_covered"
    assert not out["wire_ready"]


def test_binding_and_output_are_insertion_order_independent():
    args = _inputs()
    first = _plan(**args)
    args["records"] = dict(reversed(list(args["records"].items())))
    args["unit_bytes"] = dict(reversed(list(args["unit_bytes"].items())))
    args["winners"] = dict(reversed(list(args["winners"].items())))
    args["measured_cells"] = list(reversed(args["measured_cells"]))
    assert _plan(**args) == first
    args["seed"] += 1
    assert _plan(**args)["input_sha256"] != first["input_sha256"]


@pytest.mark.parametrize("problem", ["winner_roster", "winner_rate", "unknown_stack",
                                    "unknown_expert", "unknown_rate", "missing_evidence",
                                    "stack_key", "mixed_currency", "byte_roster",
                                    "negative_bytes", "bool_budget", "short_draws",
                                    "infeasible_winners", "missing_ledger"])
def test_invalid_or_incomplete_inputs_fail_closed(problem):
    args = _inputs()
    if problem == "winner_roster":
        args["winners"].pop("a")
    elif problem == "winner_rate":
        args["winners"]["a"] = 1000
    elif problem == "unknown_stack":
        args["measured_cells"].append(("unknown", 0, 960))
    elif problem == "unknown_expert":
        args["measured_cells"].append(("a", 99, 960))
    elif problem == "unknown_rate":
        args["measured_cells"].append(("a", 0, 1000))
    elif problem == "missing_evidence":
        args["measured_cells"].pop()
    elif problem == "stack_key":
        args["records"]["a"] = replace(args["records"]["a"], stack="other")
    elif problem == "mixed_currency":
        args["records"]["a"] = replace(args["records"]["a"], currency="other")
    elif problem == "byte_roster":
        args["unit_bytes"]["extra"] = {960: 20}
    elif problem == "negative_bytes":
        args["unit_bytes"]["a"][832] = -1
    elif problem == "bool_budget":
        args["byte_budget"] = True
    elif problem == "short_draws":
        args["draws"] = 49
    elif problem == "infeasible_winners":
        args["winners"] = {n: 1088 for n in args["records"]}
    elif problem == "missing_ledger":
        args["measured_cells"] = None
    with pytest.raises(TesseraFormatError):
        _plan(**args)


def _json_input():
    args = _inputs()
    args["records"] = {n: asdict(s) for n, s in args["records"].items()}
    return json.loads(json.dumps(dict(args, schema="prismaquant.stack_reduced_schedule_input.v1")))


def test_offline_cli_emits_manifest_and_refuses_existing_output(tmp_path):
    from prismaquant.tessera_reduced_schedule import main
    source, output = tmp_path / "input.json", tmp_path / "next.json"
    source.write_text(json.dumps(_json_input()))
    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes()) == json.loads(json.dumps(_plan(**_inputs())))
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem", ["schema", "unknown_field", "integer_key"])
def test_cli_refuses_bad_input_without_publishing(tmp_path, problem):
    from prismaquant.tessera_reduced_schedule import main
    payload = _json_input()
    if problem == "schema":
        payload["schema"] = "unknown"
    elif problem == "unknown_field":
        payload["typo"] = True
    else:
        payload["unit_bytes"]["a"]["0832"] = payload["unit_bytes"]["a"].pop("832")
    source, output = tmp_path / "input.json", tmp_path / "next.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as error:
        main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
