"""Prefill-vs-accuracy frontier sweep on synthetic tables (CPU; not GPU evidence)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant import allocator, prefill_frontier
from prismaquant.layer_config import load_assignment
from prismaquant.measured_runtime_prices import identity_sha256
from test_allocator_measured_runtime_cli import _main_fixture

#: Three rungs per unit: the slow accurate one, a same-byte faster one with
#: more loss, and a smaller, fastest, lossiest one.
CURVE_MENU = {"FP8_E4M3": (1.0, 8.0), "FP8_E5M2": (2.0, 2.0), "NVFP4": (4.0, 1.0)}
UNITS = tuple(f"model.layers.{i}.self_attn.o_proj" for i in range(3))

POINT_KEYS = {"slo_ms", "target_bits", "feasible", "refusal_reason", "predicted_dloss",
              "achieved_bits", "payload_bytes", "attained_prefill_ms", "attained_decode_ms",
              "device_memory_bytes", "assignment_sha256", "assignment_path", "solver",
              "nondominated"}
DOCUMENT_KEYS = {"schema", "status", "composition", "certifies_p95", "certifies_end_to_end_slo",
                 "objective", "loss_noise_floor", "slo_axis", "saturation", "n_points",
                 "n_feasible", "n_nondominated", "distinct_assignments", "monotone_loss",
                 "monotone_loss_violations", "points", "provenance"}


def _curve_fixture(tmp_path, **overrides):
    name, argv = _main_fixture(tmp_path, units=UNITS, menu=CURVE_MENU, **overrides)
    cut = argv.index("--slo-prefill-p95-ttft-ms")
    return name, argv[1:cut]          # drop the "allocator" prog name and the SLO


def _without_wall_clock(doc):
    doc = json.loads(json.dumps(doc))
    for point in doc["points"]:
        point["solver"].pop("solver_seconds", None)
    return doc


def _run(tmp_path, own, allocator_argv):
    output = tmp_path / "frontier.json"
    code = prefill_frontier.main([*own, "--output", str(output), "--", *allocator_argv])
    return code, json.loads(output.read_text())


def test_auto_grid_produces_the_whole_curve_and_a_verified_saturation(tmp_path):
    _, allocator_argv = _curve_fixture(tmp_path)
    code, doc = _run(tmp_path, ["--slo-grid", "auto"], allocator_argv)
    assert code == 0
    assert doc["schema"] == prefill_frontier.SCHEMA
    assert DOCUMENT_KEYS <= set(doc)
    assert doc["certifies_p95"] is False and doc["status"] == "proposal_data"
    points = doc["points"]
    assert len(points) >= 3
    assert all(POINT_KEYS <= set(point) for point in points)
    assert doc["n_feasible"] == len(points)
    # The axis bounds come from the table alone: 3 units x fastest / slowest rung.
    assert doc["slo_axis"]["lower_bound_ms"] == 3.0
    assert doc["slo_axis"]["upper_bound_ms"] == 24.0
    # Saturation is the attained prefill of the unconstrained solve (all FP8_E4M3).
    saturation = doc["saturation"]
    assert saturation["slo_ms"] == 24.0 and saturation["verified"] is True
    assert saturation["predicted_dloss"] == pytest.approx(3.0)
    assert saturation["differing_slo_ms"] == []
    # Loss is nonincreasing as the budget relaxes; attained never exceeds the budget.
    slos = [point["slo_ms"] for point in points]
    assert slos == sorted(slos)
    losses = [point["predicted_dloss"] for point in points]
    assert losses == sorted(losses, reverse=True) and doc["monotone_loss"] is True
    assert all(point["attained_prefill_ms"] <= point["slo_ms"] for point in points)
    # The auto grid is the solver's own breakpoint set: every point sits on its
    # own attained prefill, so every point is a corner and all are nondominated.
    assert all(point["attained_prefill_ms"] == point["slo_ms"] for point in points)
    assert all(point["nondominated"] for point in points)
    assert doc["distinct_assignments"] == len(points) >= 3
    # The tightest corner is every unit on the fastest rung.
    fastest = load_assignment(points[0]["assignment_path"])
    assert set(fastest.values()) == {"NVFP4"} and points[0]["slo_ms"] == 3.0
    slowest = load_assignment(points[-1]["assignment_path"])
    assert set(slowest.values()) == {"FP8_E4M3"}
    for point in points:
        assignment = json.loads(Path(point["assignment_path"]).read_text())
        meta = assignment.pop("__prismaquant__")
        assert meta["schema"] == prefill_frontier.ASSIGNMENT_SCHEMA
        assert identity_sha256(assignment) == point["assignment_sha256"] == meta["assignment_sha256"]
        assert Path(point["assignment_path"]).name == point["assignment_sha256"] + ".json"
    provenance = doc["provenance"]
    assert provenance["table_identity"]["table_id"] == "synthetic-only"
    assert provenance["grid"] == {"kind": "auto", "spec": "prefill_slo_breakpoints_ms"}
    assert provenance["cost_sha256"] == provenance["table_identity"]["cost_sha256"]
    assert "p95_ttft_ms" not in provenance["serve_slos_other_axes"]


def test_linear_grid_spans_lower_bound_to_saturation(tmp_path):
    _, allocator_argv = _curve_fixture(tmp_path)
    code, doc = _run(tmp_path, ["--slo-grid", "5"], allocator_argv)
    assert code == 0
    slos = [point["slo_ms"] for point in doc["points"]]
    assert slos == pytest.approx([3.0, 8.25, 13.5, 18.75, 24.0])
    assert doc["provenance"]["grid"] == {"kind": "linear", "spec": 5}
    assert doc["monotone_loss"] is True and doc["saturation"]["verified"] is True
    # Interior linear points attain less than their budget; a repeated
    # assignment at a looser budget is dominated by its own first appearance.
    corners = [point for point in doc["points"] if point["nondominated"]]
    assert len(corners) == doc["distinct_assignments"]
    losses = [point["predicted_dloss"] for point in corners]
    assert losses == sorted(losses, reverse=True) and len(set(losses)) == len(losses)


def test_explicit_grid_records_infeasible_points_and_keeps_going(tmp_path):
    _, allocator_argv = _curve_fixture(tmp_path)
    code, doc = _run(tmp_path, ["--slo-ms", "0.5,3,30"], allocator_argv)
    assert code == 0
    by_slo = {point["slo_ms"]: point for point in doc["points"]}
    # Saturation (24) and the table upper bound (24) are always on the grid.
    assert sorted(by_slo) == [0.5, 3.0, 24.0, 30.0]
    assert by_slo[0.5]["feasible"] is False
    assert by_slo[0.5]["refusal_reason"] == "no_runtime_frontier_assignment_passed_exact_checks"
    assert by_slo[0.5]["assignment_sha256"] is None and by_slo[0.5]["nondominated"] is False
    assert by_slo[3.0]["feasible"] and by_slo[3.0]["predicted_dloss"] == pytest.approx(12.0)
    assert by_slo[30.0]["assignment_sha256"] == by_slo[24.0]["assignment_sha256"]
    assert by_slo[30.0]["nondominated"] is False and by_slo[24.0]["nondominated"] is True
    assert doc["n_feasible"] == 3 and doc["saturation"]["verified"] is True


def test_sweep_point_equals_the_single_solve_at_that_slo(tmp_path, monkeypatch):
    """A grid point is what one --slo-prefill-p95-ttft-ms run would ship."""
    name, argv = _main_fixture(tmp_path, units=UNITS, menu=CURVE_MENU)
    cut = argv.index("--slo-prefill-p95-ttft-ms")
    code, doc = _run(tmp_path, ["--slo-ms", "12"], argv[1:cut])
    assert code == 0
    # The sweep never writes the single solve's outputs (it did not run one).
    assert not (tmp_path / "pareto.csv").exists() and not (tmp_path / "layer.json").exists()
    single = argv[:cut] + ["--slo-prefill-p95-ttft-ms", "12"]
    monkeypatch.setattr(sys, "argv", single)
    allocator.main()
    single_assignment = load_assignment(tmp_path / "layer.json")
    single_meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    point = next(point for point in doc["points"] if point["slo_ms"] == 12.0)
    assert load_assignment(point["assignment_path"]) == single_assignment
    assert point["attained_prefill_ms"] == single_meta["serve_constraints"]["predicted"]["operator_sum_prefill_ms"]
    assert point["serve_constraints"]["predicted"] == single_meta["serve_constraints"]["predicted"]


def test_default_allocator_path_is_untouched_by_the_hook(tmp_path, monkeypatch):
    name, argv = _main_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    allocator.main()
    assert load_assignment(tmp_path / "layer.json")[name] == "FP8_E5M2"
    # And main(argv) is the same call without touching sys.argv.
    (tmp_path / "layer.json").unlink()
    monkeypatch.setattr(sys, "argv", ["unrelated"])
    allocator.main(argv[1:])
    assert load_assignment(tmp_path / "layer.json")[name] == "FP8_E5M2"


def test_invalid_table_rows_refuse_with_the_loader_message(tmp_path):
    _, allocator_argv = _curve_fixture(tmp_path)
    table_path = tmp_path / "runtime.json"
    payload = json.loads(table_path.read_text())
    payload["rows"][0]["resources"]["prefill_ms"] = 3.5   # no longer the sample median
    table_path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="resource times must equal medians"):
        _run(tmp_path, ["--slo-grid", "auto"], allocator_argv)
    assert not (tmp_path / "frontier.json").exists()


def test_stale_cost_payload_refuses(tmp_path):
    _, allocator_argv = _curve_fixture(tmp_path)
    (tmp_path / "costs.pkl").write_bytes((tmp_path / "costs.pkl").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="cost payload SHA-256 mismatch"):
        _run(tmp_path, ["--slo-grid", "auto"], allocator_argv)


def test_grid_owns_the_prefill_slo_flag(tmp_path, capsys):
    _, allocator_argv = _curve_fixture(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        _run(tmp_path, ["--slo-grid", "auto"],
             allocator_argv + ["--slo-prefill-p95-ttft-ms", "5"])
    assert "belongs to the prefill frontier sweep grid" in capsys.readouterr().err


@pytest.mark.parametrize("own,message", [
    ([], "exactly one of --slo-ms or --slo-grid"),
    (["--slo-ms", "1", "--slo-grid", "auto"], "exactly one of"),
    (["--slo-grid", "1"], "integer >= 2"),
    (["--slo-ms", "0,4"], "positive and finite"),
    (["--slo-grid", "auto", "--loss-noise-floor", "-1"], "nonnegative"),
])
def test_grid_arguments_refuse(tmp_path, capsys, own, message):
    with pytest.raises(SystemExit, match="2"):
        prefill_frontier.main([*own, "--output", str(tmp_path / "f.json"), "--", "--probe", "x"])
    assert message in capsys.readouterr().err


def test_sweep_without_a_table_refuses(tmp_path, capsys):
    _, argv = _main_fixture(tmp_path)
    cut = argv.index("--measured-runtime-table")
    with pytest.raises(SystemExit, match="2"):
        _run(tmp_path, ["--slo-grid", "auto"], argv[1:cut])
    assert "requires --measured-runtime-table" in capsys.readouterr().err


def test_ties_are_deterministic_and_lexical(tmp_path):
    menu = {"FP8_E4M3": (1.0, 8.0), "FP8_E5M2": (1.0, 8.0)}   # same bytes, loss, prefill
    _, argv = _main_fixture(tmp_path, units=UNITS[:2], menu=menu)
    cut = argv.index("--slo-prefill-p95-ttft-ms")
    first = _run(tmp_path, ["--slo-grid", "4"], argv[1:cut])[1]
    second = _run(tmp_path, ["--slo-grid", "4"], argv[1:cut])[1]
    assert _without_wall_clock(first) == _without_wall_clock(second)
    for point in first["points"]:
        if point["feasible"]:
            assert set(load_assignment(point["assignment_path"]).values()) == {"FP8_E4M3"}
    assert first["distinct_assignments"] == 1 and first["n_nondominated"] == 1


def test_unconstrained_infeasibility_is_reported_not_hidden(tmp_path):
    """A byte budget below every rung: no saturation point, exit 2, every point says why."""
    _, argv = _main_fixture(tmp_path, units=UNITS, menu=CURVE_MENU, target_bits="1")
    cut = argv.index("--slo-prefill-p95-ttft-ms")
    output = tmp_path / "frontier.json"
    code = prefill_frontier.main(["--slo-ms", "5,10", "--output", str(output), "--", *argv[1:cut]])
    assert code == 2
    doc = json.loads(output.read_text())
    assert doc["saturation"] is None and doc["n_feasible"] == 0
    reasons = {point["slo_ms"]: point["refusal_reason"] for point in doc["points"]}
    assert reasons[24.0] == "no_runtime_frontier_assignment_passed_exact_checks"
    assert reasons[5.0].startswith("unconstrained_point_infeasible:")


def test_build_document_flags_dominance_saturation_and_monotonicity(tmp_path):
    def point(slo, loss, prefill, sha, feasible=True):
        return {"slo_ms": slo, "target_bits": 4.0, "feasible": feasible,
                "refusal_reason": None if feasible else "x", "predicted_dloss": loss,
                "achieved_bits": 4.0, "payload_bytes": 1, "attained_prefill_ms": prefill,
                "attained_decode_ms": None, "device_memory_bytes": None,
                "assignment_sha256": sha, "assignment_path": str(tmp_path / f"{sha}.json"),
                "solver": {}}
    points = [point(1.0, 9.0, 1.0, "a"), point(2.0, 9.0, 1.0, "a"),      # repeat: dominated
              point(3.0, 5.0, 3.0, "b"), point(4.0, 6.0, 3.5, "c"),      # c: worse on both
              point(6.0, 4.0, 5.0, "d"), point(0.5, 0.0, 0.0, None, False)]
    doc = prefill_frontier.build_frontier_document(
        points, saturation={"slo_ms": 5.0, "assignment_sha256": "d", "predicted_dloss": 4.0},
        slo_axis={}, loss_noise_floor=0.0, provenance={})
    flags = {p["slo_ms"]: p["nondominated"] for p in doc["points"]}
    assert flags == {1.0: True, 2.0: False, 3.0: True, 4.0: False, 6.0: True, 0.5: False}
    assert doc["monotone_loss"] is False
    assert [v["slo_ms"] for v in doc["monotone_loss_violations"]] == [4.0]
    assert doc["saturation"]["verified"] is True and doc["saturation"]["points_at_or_above"] == 1
    # A noise floor wider than the b->d gain merges those corners, as the
    # validated frontier's --kl-noise-floor does.
    wide = prefill_frontier.build_frontier_document(
        points, saturation={"slo_ms": 7.0, "assignment_sha256": "zzz", "predicted_dloss": 0},
        slo_axis={}, loss_noise_floor=1.5, provenance={})
    assert [p["slo_ms"] for p in wide["points"] if p["nondominated"]] == [1.0, 3.0]
    # No grid point at or above the claimed saturation: nothing verifies it.
    assert wide["saturation"]["verified"] is False and wide["saturation"]["differing_slo_ms"] == []
    assert wide["saturation"]["points_at_or_above"] == 0


def test_module_help_runs_from_a_clean_interpreter():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-m", "prismaquant.prefill_frontier", "--help"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--slo-grid" in result.stdout and "--measured-runtime-table" in result.stdout
