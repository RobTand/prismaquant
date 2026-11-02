"""#3132: a budget the exact minimum-byte assignment fits must not be refused.

The byte-budget selector's menu-floor probe feeds the floor's bits-per-param
back through the *binned* solver as a target. Bin charging rounds every
positive upgrade UP to at least one bin, so at the floor target (zero excess,
a single usable bin) the DP can still buy one sub-bin upgrade whose real bytes
overshoot the target, and the rung is declared infeasible although the
unrounded, promoted, exactly priced minimum-byte assignment fits.

Solver seam: ``solve_with_promotion`` must fall back to that endpoint, but only
after the binned search has failed. CLI seam: the real ``allocator.main`` with
the real solver must ship the minimum-byte assignment, and a budget genuinely
below it must still fail closed.
"""
from __future__ import annotations

import json
import pickle
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from prismaquant.allocator_solver import Candidate, solve_with_promotion
from tests.safetensors_support import _write_safetensors

# ---------------------------------------------------------------------------
# Solver seam
# ---------------------------------------------------------------------------


def _rounding_inputs():
    """Eight equal units; the denser format is better but 0.00125 bpp dearer
    per unit. At the floor target the DP charges it one whole 0.1-bpp bin."""
    stats = {f"u{i}": {"n_params": 8000} for i in range(8)}
    candidates = {
        n: [Candidate("R1024", 4.0, 4000, 1.0),
            Candidate("R1088", 4.01, 4010, 0.0)]
        for n in stats
    }
    return stats, candidates


def test_solver_min_byte_fallback_rescues_coarse_bin_overshoot():
    stats, candidates = _rounding_inputs()
    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, candidates, 4.0, {}, {"R1024": 0, "R1088": 1}, 0.1,
        overshoot_tolerance=0.0, diagnostics=diag)
    assert assignment is not None
    assert set(assignment.values()) == {"R1024"}
    assert achieved == pytest.approx(4.0)
    assert diag["feasible"] is True
    assert diag["quality_endpoint_selected"] is False
    assert diag["min_byte_endpoint_selected"] is True


def test_solver_binned_search_still_wins_when_it_is_feasible():
    """The fallback is failure-path only: a feasible binned result is kept."""
    stats, candidates = _rounding_inputs()
    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, candidates, 4.5, {}, {"R1024": 0, "R1088": 1}, 0.1,
        overshoot_tolerance=0.0, diagnostics=diag)
    # Quality endpoint (all R1088, 4.01 bpp) fits 4.5 and is selected first.
    assert set(assignment.values()) == {"R1088"}
    assert "min_byte_endpoint_selected" not in diag


def test_solver_below_the_min_byte_floor_stays_infeasible():
    stats, candidates = _rounding_inputs()
    assignment, achieved = solve_with_promotion(
        stats, candidates, 3.9, {}, {"R1024": 0, "R1088": 1}, 0.1,
        overshoot_tolerance=0.0)
    assert assignment is None
    assert achieved != achieved  # NaN


def test_solver_min_byte_fallback_still_respects_the_budget():
    """If even the minimum-byte assignment overshoots, the rung is refused."""
    stats, candidates = _rounding_inputs()
    assignment, _ = solve_with_promotion(
        stats, candidates, 3.99, {}, {"R1024": 0, "R1088": 1}, 0.1,
        overshoot_tolerance=0.0)
    assert assignment is None


# ---------------------------------------------------------------------------
# CLI seam
# ---------------------------------------------------------------------------

_N_UNITS = 64
_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(_N_UNITS)]
_OUT = _IN = 256
_NPARAMS = _OUT * _IN
_RESERVE = 512
_FLOOR_TENSORS = {
    "model.embed_tokens.weight": ("BF16", (512, 64)),
    "lm_head.weight": ("BF16", (512, 64)),
    "model.norm.weight": ("BF16", (64,)),
}


def _fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    for n in _NAMES:
        tensors[f"{n}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)
    stats = {
        n: {"h_trace": 1.0 + 0.01 * i, "n_params": _NPARAMS,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(_NAMES)
    }
    probe = {"stats": stats, "meta": {"model": str(model_dir)}}

    def row(d):
        return {"weight_mse": d, "output_mse": d,
                "output_mse_measured": True, "predicted_dloss": d}

    costs = {
        "costs": {n: {"NVFP4": row(1e-4), "FP8_E4M3": row(1e-6)}
                  for n in _NAMES},
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return model_dir, probe_p, cost_p, stats


def _nvfp4_artifact_bytes(model_dir, stats):
    info = fp.floor_bytes_for_model(str(model_dir), _NAMES, stats)
    body = 0
    for n in _NAMES:
        body += fr.get_format("NVFP4").memory_bytes_for_shape((_OUT, _IN))
        body += fp.nvfp4_global_sidecar_bytes(n, (_OUT, _IN))
    return info["floor_bytes"] + body


def _run(monkeypatch, tmp_path, probe_p, cost_p, *, budget_bytes,
         refusal_is_failure=True):
    """Run ``allocator.main``.

    A SystemExit refusal is surfaced as an assertion failure by default, so a
    false 'below the floor' refusal shows up as a failed assertion carrying the
    refusal text rather than as a bare SystemExit.
    """
    lc = tmp_path / "layer_config.json"
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", "8.2",
        "--bit-precision", "0.1",
        "--target-disk-gb", repr(budget_bytes / fp.GB),
        "--layer-config", str(lc),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
        "--artifact-overhead-reserve-bytes", str(_RESERVE),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    if refusal_is_failure:
        try:
            alloc.main()
        except SystemExit as exc:
            raise AssertionError(
                f"budget of {budget_bytes} bytes was refused although the "
                f"exact minimum-byte assignment fits it: {exc}") from None
    else:
        alloc.main()
    selection = json.loads((tmp_path / "selection.json").read_text())
    return selection, json.loads(lc.read_text())


def test_cli_budget_fitting_min_byte_assignment_selects_it(
        monkeypatch, tmp_path):
    sizing = tmp_path / "sizing"
    sizing.mkdir()
    sized_dir, _p, _c, sized_stats = _fixture(sizing)
    nvfp4_bytes = _nvfp4_artifact_bytes(sized_dir, sized_stats)
    budget = nvfp4_bytes + _RESERVE + 2_048

    _model_dir, probe_p, cost_p, _stats = _fixture(tmp_path)
    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p, budget_bytes=budget)

    assert selection["feasible"] and not selection["below_floor"]
    assert (selection["predicted_whole_artifact_upper_bound_gb"] * fp.GB
            <= selection["budget_bytes"])
    assert selection["predicted_whole_artifact_upper_bound_gb"] * fp.GB == (
        pytest.approx(nvfp4_bytes + _RESERVE, abs=1.0))


def test_cli_selection_record_reflects_min_byte_floor_rung(
        monkeypatch, tmp_path):
    sizing = tmp_path / "sizing"
    sizing.mkdir()
    sized_dir, _p, _c, sized_stats = _fixture(sizing)
    budget = _nvfp4_artifact_bytes(sized_dir, sized_stats) + _RESERVE + 2_048

    _model_dir, probe_p, cost_p, _stats = _fixture(tmp_path)
    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p, budget_bytes=budget)

    ext = selection["pareto_grid_extension"]
    assert ext["reason"] == "swept_grid_min_target_above_menu_floor"
    assert ext["added_targets"]
    assert not ext["infeasible_targets"]
    assert min(ext["added_targets"]) == pytest.approx(
        ext["menu_floor_target_bits"])
    # The shipped rung is the floor rung, and the grid carries it.
    assert selection["chosen_target_bits"] == pytest.approx(
        ext["menu_floor_target_bits"])
    assert any(
        row["target_bits"] == pytest.approx(ext["menu_floor_target_bits"])
        for row in selection["grid"])


def test_cli_genuine_below_min_byte_floor_reports_the_exact_floor(
        monkeypatch, tmp_path):
    sizing = tmp_path / "sizing"
    sizing.mkdir()
    sized_dir, _p, _c, sized_stats = _fixture(sizing)
    nvfp4_bytes = _nvfp4_artifact_bytes(sized_dir, sized_stats)
    budget = nvfp4_bytes + _RESERVE - 5_000

    _model_dir, probe_p, cost_p, _stats = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="below the floor"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, budget_bytes=budget,
             refusal_is_failure=False)

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert not selection["feasible"] and selection["below_floor"]
    cheapest = selection["cheapest_whole_artifact_upper_bound_gb"] * fp.GB
    assert cheapest == pytest.approx(float(nvfp4_bytes + _RESERVE), abs=1.0)
