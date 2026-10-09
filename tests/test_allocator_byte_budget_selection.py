"""Byte-budget ("fit the card") ship selection in allocator.main() — issue #25.

Drives the real ``allocator.main()`` with ``--target-disk-gb`` against a
synthetic bf16 checkpoint, so what is pinned is the code that ships
selections, not a helper beside it. Three properties:

1. **One accounting path** (#23). The selector must price rungs through
   ``footprint.assignment_artifact_bytes`` — the shared function whose exact
   agreement with ``floor_bytes_for_model`` ``tests/test_footprint.py`` pins.
   It used to inline a third copy of the manifest -> resolve ->
   check-non-negative -> floor sequence, so that agreement was never
   exercised against the shipping selector. The floor the selector reports
   must equal ``floor_bytes_for_model``'s to the byte.

2. **The objective is minimum predicted Δloss subject to fitting** (#25).
   Feasibility is the card (exact footprint <= budget); the objective is
   Δloss. Ratcheting on MAX bytes ("fill the card") is only equivalent while
   more bytes implies lower Δloss, which is false — 5.5 bpp has beaten 6.0
   bpp on served PPL, and serving-unit promotion can flip a group into a
   denser-but-worse format. So a denser-fitting-but-worse allocation must be
   REJECTED, while in the well-behaved monotone case the two objectives must
   still agree (the change is a no-op there).

3. **The selection is self-describing.** The ratchet objective, the tightened
   ``search_hi`` ceiling, and the case where the bisection never runs (so a
   denser fitting allocation is forgone) must all be recoverable from
   ``selection.json``; and a footprint accounting failure must reach the
   operator as the allocator's ``SystemExit`` idiom, not a raw traceback.

The solver is stubbed per-target so the (footprint, Δloss) grid is under the
test's control: the real ``solve_with_promotion`` search is monotone by
construction on a 4-Linear dense model, and the non-monotonicity this
selector must survive comes from promotion on real MoE shapes.
"""
from __future__ import annotations

import json
import pickle
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from test_footprint import _write_safetensors

_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
_OUT = _IN = 256
_NPARAMS = _OUT * _IN
_OVERHEAD_RESERVE = 512
_FLOOR_TENSORS = {                      # never re-encoded: the floor
    "model.embed_tokens.weight": ("BF16", (512, 64)),   # 65536 B
    "lm_head.weight": ("BF16", (512, 64)),              # 65536 B
    "model.norm.weight": ("BF16", (64,)),               # 128 B
}


def _fixture(tmp_path, *, nvfp4_dloss, fp8_dloss, drop_tensor=None,
             extra_tensors=None):
    """Synthetic bf16 checkpoint + probe/cost pickles. Returns paths."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    tensors.update(extra_tensors or {})
    for n in _NAMES:
        if n == drop_tensor:
            continue
        tensors[f"{n}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)

    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": _NPARAMS,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(_NAMES)
    }
    probe = {"stats": stats, "meta": {"model": str(model_dir)}}
    costs = {
        "costs": {
            n: {
                "NVFP4": {"weight_mse": nvfp4_dloss, "output_mse": nvfp4_dloss,
                          "output_mse_measured": True,
                          "predicted_dloss": nvfp4_dloss},
                "FP8_E4M3": {"weight_mse": fp8_dloss, "output_mse": fp8_dloss,
                             "output_mse_measured": True,
                             "predicted_dloss": fp8_dloss},
            }
            for n in _NAMES
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return model_dir, probe_p, cost_p, stats


def _stub_solver(fmt_for_target):
    """A ``solve_with_promotion`` that returns a chosen format per target.

    Achieved bits are the exact candidate bytes, so the rung labels and the
    footprint stay mutually consistent; only WHICH format each target lands
    on is the test's choice.
    """
    def solve(stats, candidates, target_bits, format_specs, format_rank,
              bit_precision, **kw):
        fmt = fmt_for_target(float(target_bits))
        assign = {n: fmt for n in candidates}
        total_params = sum(stats[n]["n_params"] for n in assign)
        bits = 0.0
        for n in assign:
            cand = next(c for c in candidates[n] if c.fmt == fmt)
            bits += 8.0 * cand.memory_bytes
        achieved = bits / max(total_params, 1)
        diag = kw.get("diagnostics")
        if diag is not None:
            diag.update({"feasible": True, "achieved_bits": achieved,
                         "predicted_dloss": None, "evals": 1})
        return assign, achieved
    return solve


def _artifact_bytes(model_dir, fmt, stats):
    """Independent expectation: floor from the shared function + body bytes."""
    info = fp.floor_bytes_for_model(str(model_dir), _NAMES, stats)
    body = 0
    for n in _NAMES:
        body += fr.get_format(fmt).memory_bytes_for_shape((_OUT, _IN))
        if fmt == "NVFP4":
            body += fp.nvfp4_global_sidecar_bytes(n, (_OUT, _IN))
    return info["floor_bytes"] + body, info["floor_bytes"]


def _run(monkeypatch, tmp_path, probe_p, cost_p, *, disk_gb, fmt_for_target,
         pareto="4.6,8.2", overhead_reserve=_OVERHEAD_RESERVE,
         exclude_prefixes=(), extra_argv=()):
    monkeypatch.setattr(alloc, "solve_with_promotion",
                        _stub_solver(fmt_for_target))
    lc = tmp_path / "layer_config.json"
    csv = tmp_path / "pareto.csv"
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", pareto,
        "--target-disk-gb", repr(disk_gb),
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--allow-default-profile",
    ]
    if overhead_reserve is not None:
        argv.extend([
            "--artifact-overhead-reserve-bytes", str(overhead_reserve)
        ])
    for prefix in exclude_prefixes:
        argv.extend(["--exclude-source-prefix", prefix])
    argv.extend(extra_argv)
    monkeypatch.setattr(sys, "argv", argv)
    alloc.main()
    selection = json.loads((tmp_path / "selection.json").read_text())
    layer_cfg = json.loads(lc.read_text())
    return selection, layer_cfg


# ---------------------------------------------------------------------------
# 1. One accounting path (#23)
# ---------------------------------------------------------------------------


def test_whole_artifact_budget_requires_explicit_non_tensor_reserve(
    monkeypatch, tmp_path,
):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6
    )
    with pytest.raises(SystemExit, match="requires a positive"):
        _run(
            monkeypatch,
            tmp_path,
            probe_p,
            cost_p,
            disk_gb=1.0,
            fmt_for_target=lambda _target: "NVFP4",
            overhead_reserve=None,
        )


@pytest.mark.parametrize("disk_gb", [0.0, -1.0, float("nan")])
def test_whole_artifact_budget_must_be_positive_and_finite(
    monkeypatch, tmp_path, disk_gb,
):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6
    )
    with pytest.raises(SystemExit, match="positive finite"):
        _run(
            monkeypatch,
            tmp_path,
            probe_p,
            cost_p,
            disk_gb=disk_gb,
            fmt_for_target=lambda _target: "NVFP4",
        )


def test_selector_prices_through_the_shared_footprint_function(
        monkeypatch, tmp_path):
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    nvfp4_bytes, floor = _artifact_bytes(model_dir, "NVFP4", stats)
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)

    calls = []
    real = fp.assignment_artifact_bytes

    def spy(assignment, s, **kw):
        calls.append(kw.get("context"))
        return real(assignment, s, **kw)

    monkeypatch.setattr(fp, "assignment_artifact_bytes", spy)
    selection, cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")

    assert calls, ("the byte-budget selector must price rungs through "
                   "footprint.assignment_artifact_bytes, not an inlined copy")
    assert selection["footprint_path"] == "footprint.assignment_artifact_bytes"
    # The property test_footprint pins ("the two paths agree exactly") now
    # holds against the shipping selector: same floor, to the byte.
    assert selection["predicted_floor_gb"] * fp.GB == pytest.approx(
        float(floor), abs=1.0)
    # ...and the tensor payload is the shared function's, sidecars included;
    # the separately named whole-artifact selection upper bound adds the
    # operator-supplied non-tensor reserve.
    for row, expect in ((4.6, nvfp4_bytes), (8.2, fp8_bytes)):
        [g] = [r for r in selection["grid"] if r["target_bits"] == row]
        assert g["tensor_payload_gb"] * fp.GB == pytest.approx(
            float(expect), abs=1.0
        )
        assert g["whole_artifact_upper_bound_gb"] * fp.GB == pytest.approx(
            float(expect + _OVERHEAD_RESERVE), abs=1.0
        )
    budget_stamp = cfg["__prismaquant__"]["whole_artifact_budget"]
    assert budget_stamp["budget_bytes"] == selection["budget_bytes"]
    assert budget_stamp["selection_non_tensor_reserve_bytes"] == (
        _OVERHEAD_RESERVE
    )
    assert budget_stamp["selection_whole_artifact_upper_bound_bytes"] <= (
        budget_stamp["budget_bytes"]
    )


def test_shared_function_changes_only_the_nvfp4_global_sidecars(
        monkeypatch, tmp_path):
    """Bound the refactor's numerical delta, don't just assert it is small.

    The retired inline accounting summed ``memory_bytes_for_shape`` per name
    and nothing else; the shared function additionally counts the two fp32
    NVFP4 global scalars the export really writes per emitted 2-D Linear
    (``weight_global_scale`` + ``input_global_scale``, 8 B). So: the FLOOR is
    unchanged (it is format-independent and priced from the same manifest
    spans), and the artifact grows by exactly 8 B per NVFP4 Linear — toward
    the exported ``index.json`` total, not away from it.
    """
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    floor = fp.floor_bytes_for_model(
        str(model_dir), _NAMES, stats)["floor_bytes"]
    retired_body = sum(
        fr.get_format("NVFP4").memory_bytes_for_shape((_OUT, _IN))
        for _ in _NAMES)                     # no sidecars: the old body term
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)

    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")

    [nvfp4_rung] = [r for r in selection["grid"] if r["target_bits"] == 4.6]
    new_bytes = round(nvfp4_rung["tensor_payload_gb"] * fp.GB)
    assert round(selection["predicted_floor_gb"] * fp.GB) == floor
    assert new_bytes - (floor + retired_body) == 8 * len(_NAMES)

    # The FP8 rung has no NVFP4 sidecars, so it is byte-identical either way.
    [fp8_rung] = [r for r in selection["grid"] if r["target_bits"] == 8.2]
    fp8_retired = floor + sum(
        fr.get_format("FP8_E4M3").memory_bytes_for_shape((_OUT, _IN))
        for _ in _NAMES)
    assert round(fp8_rung["tensor_payload_gb"] * fp.GB) == fp8_retired


# ---------------------------------------------------------------------------
# 2. The objective: min predicted Δloss among the rungs that fit (#25)
# ---------------------------------------------------------------------------

def test_denser_fitting_rung_with_worse_dloss_is_rejected(
        monkeypatch, tmp_path):
    """The substantive fix: both rungs fit the card, the denser one is WORSE.

    'Fill the card' ships the denser/worse one; the objective is Δloss, so the
    sparser one must ship — and the artifact (layer_config) must show it.
    """
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-6, fp8_dloss=1e-4)   # FP8 measured WORSE
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)

    selection, layer_cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + 10_000) / fp.GB,   # roomy: everything fits
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")

    assert selection["ratchet_objective"] == (
        "min_predicted_dloss__ties_to_larger_footprint")
    assert selection["selection_feasibility_test"] == (
        "exact_tensor_payload_bytes + operator_non_tensor_reserve_bytes "
        "<= whole_artifact_budget_bytes")
    assert selection["final_feasibility_test"] == (
        "stat_all_regular_files_recursive <= whole_artifact_budget_bytes")
    # The objective change is a semantic change to what `chosen_*` means.
    assert selection["schema"] == (
        "prismaquant.allocator.byte_budget_selection.v3")
    # Every rung fits, so the choice is purely the objective's.
    assert all(r["fits"] for r in selection["grid"])
    assert selection["chosen_target_bits"] == 4.6
    assert selection["predicted_dloss"] < selection["max_bytes_pick_dloss"], (
        "the retired max-bytes objective must be recorded and be worse here")
    assert selection["max_bytes_pick_target_bits"] == 8.2
    assert not selection["max_bytes_grid_pick_agrees"]
    # The shipped artifact is the sparser, better one.
    assert {cfg["data_type"] for name, cfg in layer_cfg.items()
            if name != "__prismaquant__"} == {"nv_fp"}
    # Headroom is deliberately left on the card: filling it costs quality.
    assert selection["selection_headroom_gb"] > 0
    # The near-lossless cap was probed and REJECTED, not ignored.
    caps = [r for r in selection["ratchet_trace"] if r["stage"] == "search_hi_cap"]
    assert len(caps) == 1 and caps[0]["fits"] and not caps[0]["accepted"]


def test_monotone_case_still_fills_the_card(monkeypatch, tmp_path):
    """Non-regression: where more bytes DO mean lower Δloss (the normal case),
    the min-Δloss objective picks exactly what fill-the-card picked."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)   # denser is better
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)

    selection, layer_cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")

    assert selection["chosen_target_bits"] == 8.2
    assert selection["max_bytes_grid_pick_agrees"], (
        "in the monotone regime the objective change must be a no-op")
    assert selection["predicted_dloss"] == selection["max_bytes_pick_dloss"]
    assert {cfg["data_type"] for name, cfg in layer_cfg.items()
            if name != "__prismaquant__"} == {"fp8_e4m3"}
    assert selection["has_slack"]


def test_over_budget_rung_is_never_selected(monkeypatch, tmp_path):
    """Feasibility is unchanged: a lower-Δloss rung that does NOT fit loses."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)   # FP8 better AND bigger
    nvfp4_bytes, _floor = _artifact_bytes(model_dir, "NVFP4", stats)

    selection, layer_cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(nvfp4_bytes + 1_000) / fp.GB,   # only the NVFP4 rung fits
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")

    assert selection["chosen_target_bits"] == 4.6
    assert {cfg["data_type"] for name, cfg in layer_cfg.items()
            if name != "__prismaquant__"} == {"nv_fp"}
    assert (
        selection["predicted_whole_artifact_upper_bound_gb"] * fp.GB
        <= selection["budget_bytes"]
    )
    assert not any(r["accepted"] for r in selection["ratchet_trace"]
                   if not r["fits"])


def test_below_floor_budget_exits_with_the_cheapest_artifact(
        monkeypatch, tmp_path):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    with pytest.raises(SystemExit, match="below the floor"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, disk_gb=1e-6,
             fmt_for_target=lambda t: "NVFP4")
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert not selection["feasible"] and selection["below_floor"]
    assert selection["ratchet_objective"] == (
        "min_predicted_dloss__ties_to_larger_footprint")


# ---------------------------------------------------------------------------
# 3. Self-describing selection + operator-facing failures
# ---------------------------------------------------------------------------

def test_search_hi_tightening_and_skipped_bisection_are_recorded(
        monkeypatch, tmp_path):
    """A non-monotone disk(target) tightens search_hi BELOW the grid pick's
    target, so the bisection breaks on iteration 0 and the grid pick ships
    with no exploration. That used to leave no trace at all."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    nvfp4_bytes, _floor = _artifact_bytes(model_dir, "NVFP4", stats)

    # target 4.5 -> the BIG (FP8) artifact, target 8.0 -> the small (NVFP4)
    # one: disk(target) decreasing, the premise the tightening assumes away.
    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(nvfp4_bytes + 1_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t < 8.3 else "NVFP4",
        pareto="8.2,8.4")

    assert selection["chosen_target_bits"] == 8.4
    assert selection["search_hi_tightened"]
    assert selection["search_hi_tightened_by_rung"] == 8.2
    assert selection["search_hi_target_bits"] == 8.2
    assert selection["search_hi_cap_target_bits"] > 8.2
    assert not selection["bisection_ran"]
    assert selection["bisection_skipped_reason"] == (
        "tightened_search_hi_at_or_below_grid_pick_target")
    assert not any(r["stage"] == "bisect" for r in selection["ratchet_trace"])


def test_unresolvable_source_name_exits_like_an_allocator_failure(
        monkeypatch, tmp_path):
    """#23/#25: the message was already good, but it reached the operator as a
    raw ValueError traceback. It must exit the way main()'s other fatals do."""
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    real = fp.source_tensor_bytes_manifest

    def drop_one(model_path, **kw):
        m = real(model_path, **kw)
        spans = dict(m.spans)
        del m[_NAMES[0]]
        spans.pop(_NAMES[0], None)
        out = fp.SourceByteManifest(m, spans=spans)
        return out

    monkeypatch.setattr(fp, "source_tensor_bytes_manifest", drop_one)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, probe_p, cost_p, disk_gb=1.0,
             fmt_for_target=lambda t: "NVFP4")
    msg = str(exc.value)
    assert msg.startswith("[alloc] ERROR:")
    assert "[footprint]" in msg and _NAMES[0] in msg
    assert "byte-budget selector" in msg


def test_double_charged_source_span_exits_like_an_allocator_failure(
        monkeypatch, tmp_path):
    """The new structural guard must also surface as a SystemExit, not a
    traceback: two allocated names resolving to one source span."""
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    real = fp.source_tensor_bytes_manifest

    def alias_two(model_path, **kw):
        m = real(model_path, **kw)
        spans = dict(m.spans)
        # Make two distinct allocated Linears claim the same source span.
        spans[_NAMES[1]] = spans[_NAMES[0]]
        return fp.SourceByteManifest(m, spans=spans)

    monkeypatch.setattr(fp, "source_tensor_bytes_manifest", alias_two)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, probe_p, cost_p, disk_gb=1.0,
             fmt_for_target=lambda t: "NVFP4")
    msg = str(exc.value)
    assert msg.startswith("[alloc] ERROR:")
    assert "charged twice" in msg


# ---------------------------------------------------------------------------
# 4. "The cheapest allocation" must mean the MENU's floor, not the sweep's
# ---------------------------------------------------------------------------
# The Pareto grid is a coarse SAMPLE of the RD curve; the byte-budget selector
# reads its cheapest point as the cheapest allocation. That is only true while
# the sample reaches the bottom of the format menu, and --pareto-targets
# defaults to 4.5..8.25 — a range written for a 4-bit/8-bit menu. On the CB
# menu (NVFP4-CB is ~2.03 bpp) DeepSeek-V4-Flash's 92.0 GB budget, whose menu
# floor is 87.2 GB, was rejected against a "cheapest allocation" of 172.4 GB
# that is nothing but the 4.5 bpp rung — with a remedy ("raise the budget")
# pointing the operator in exactly the wrong direction. Reproduced on that
# run's REAL layer-0 production rows, so it is not a fixture artifact.


# FP8_E4M3's exact assignment payload on this fixture, in bpp (8 bits/param
# plus the shared-payload overhead the exact filter charges). A target below
# it cannot hold FP8, which is what makes the stub below spend-to-target the
# way a real solver does rather than ignoring the budget it was given.
_FP8_EXACT_BPP = 8.125


def _menu_floor_scenario(monkeypatch, tmp_path, *, disk_gb):
    """Grid sampled far above the menu floor; only the cheap rung can fit.

    The stub spends up to its target (densest format the target can hold),
    so ``pareto="8.2,9.0"`` samples ONLY the expensive rung while NVFP4 —
    the menu's own cheapest candidate, ~4.5 bpp — is never sampled. That is
    the DeepSeek-V4-Flash shape in miniature: a 4.5..8.25 default grid over
    a menu whose floor is less than half of it.
    """
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    return _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=disk_gb,
        fmt_for_target=(
            lambda t: "FP8_E4M3" if t >= _FP8_EXACT_BPP else "NVFP4"),
        pareto="8.2,9.0")


def test_budget_below_the_swept_grid_but_above_the_menu_floor_still_ships(
        monkeypatch, tmp_path):
    sizing = tmp_path / "sizing"
    sizing.mkdir()
    model_dir, _probe_p, _cost_p, stats = _fixture(
        sizing, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    nvfp4_bytes, _floor = _artifact_bytes(model_dir, "NVFP4", stats)
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    # A budget the MENU can meet with room to spare and no SAMPLED rung can.
    budget = nvfp4_bytes + _OVERHEAD_RESERVE + 5_000
    assert budget < fp8_bytes + _OVERHEAD_RESERVE

    selection, _cfg = _menu_floor_scenario(
        monkeypatch, tmp_path, disk_gb=budget / fp.GB)

    assert selection["feasible"] and not selection["below_floor"]
    ext = selection["pareto_grid_extension"]
    assert ext["reason"] == "swept_grid_min_target_above_menu_floor"
    assert ext["swept_min_target_bits"] == 8.2
    # The probe starts AT the menu floor, and that rung must be one the
    # solver can actually land on — an epsilon-below nominal rate is not.
    assert ext["menu_floor_target_bits"] < 8.2
    assert ext["added_targets"]
    assert not ext["infeasible_targets"]
    assert min(ext["added_targets"]) == pytest.approx(
        ext["menu_floor_target_bits"])
    # ...and the shipped allocation is one of the probed rungs, under budget.
    assert selection["chosen_target_bits"] < 8.2
    assert (selection["predicted_whole_artifact_upper_bound_gb"] * fp.GB
            <= selection["budget_bytes"])


def test_menu_floor_probe_does_not_run_when_the_swept_grid_already_fits(
        monkeypatch, tmp_path):
    """Repair path only. A run that selects today keeps its exact grid."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + _OVERHEAD_RESERVE + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")
    assert selection["feasible"]
    assert selection["pareto_grid_extension"] is None
    assert {row["target_bits"] for row in selection["grid"]} == {4.6, 8.2}


def test_genuine_below_floor_reports_the_menu_floor_not_the_swept_minimum(
        monkeypatch, tmp_path):
    """Still fails closed — but on a number that means what the label says."""
    sizing = tmp_path / "sizing"
    sizing.mkdir()
    model_dir, _probe_p, _cost_p, stats = _fixture(
        sizing, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    nvfp4_bytes, _floor = _artifact_bytes(model_dir, "NVFP4", stats)
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)

    with pytest.raises(SystemExit, match="below the floor"):
        _menu_floor_scenario(
            monkeypatch, tmp_path,
            disk_gb=(nvfp4_bytes + _OVERHEAD_RESERVE - 5_000) / fp.GB)

    selection = json.loads((tmp_path / "selection.json").read_text())
    assert not selection["feasible"] and selection["below_floor"]
    cheapest = selection["cheapest_whole_artifact_upper_bound_gb"] * fp.GB
    # The floor quoted is the MENU's cheapest allocation, not the cheapest
    # rung the sweep happened to sample.
    assert cheapest == pytest.approx(
        float(nvfp4_bytes + _OVERHEAD_RESERVE), abs=1.0)
    assert cheapest < fp8_bytes
    assert selection["pareto_grid_extension"]["added_targets"]


# ---------------------------------------------------------------------------
# 9. --exclude-source-prefix: an artifact that ships only PART of the source
# ---------------------------------------------------------------------------

# A draft head that ships as its own directory, so the target artifact holds
# none of it. Sized to dominate the reserve so its effect is unmistakable.
_DRAFT_TENSORS = {"mtp.0.attn.wo_a.weight": ("BF16", (512, 64))}   # 65536 B
_DRAFT_BYTES = 65536


def test_excluded_source_prefix_leaves_the_floor(monkeypatch, tmp_path):
    """The whole point: a tensor that is not in this artifact must not be
    priced into it."""
    _, probe_p, cost_p = _fixture(
        tmp_path, nvfp4_dloss=1.0, fp8_dloss=0.5,
        extra_tensors=_DRAFT_TENSORS)[:3]

    kw = dict(disk_gb=1.0, fmt_for_target=lambda t: "FP8_E4M3")
    full, _ = _run(monkeypatch, tmp_path, probe_p, cost_p, **kw)
    part, _ = _run(monkeypatch, tmp_path, probe_p, cost_p,
                   exclude_prefixes=("mtp.",), **kw)

    assert (full["source_total_bytes"] - part["source_total_bytes"]
            == _DRAFT_BYTES)
    assert "source_partition" not in full
    sp = part["source_partition"]
    assert sp["excluded_prefixes"] == ["mtp."]
    assert sp["excluded_source_bytes"] == _DRAFT_BYTES
    assert sp["n_excluded_source_tensors"] == 1
    assert sp["source_total_bytes_unpartitioned"] == full["source_total_bytes"]


def test_excluded_mass_is_returned_to_the_body(monkeypatch, tmp_path):
    """The failure this prevents is an UNDERSHOOT, so assert the budget buys
    MORE: at a budget that fits only the cheap rung while the draft is priced
    in, excluding the draft must let the expensive rung fit instead."""
    _, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1.0, fp8_dloss=0.5,
        extra_tensors=_DRAFT_TENSORS)

    model_dir = tmp_path / "model"
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    # A card that the FP8 rung overshoots by less than the draft's mass: it
    # cannot fit while the draft is priced in, and must fit once it is not.
    budget = fp8_bytes + _OVERHEAD_RESERVE - _DRAFT_BYTES // 2

    def fmt_for_target(t):
        return "FP8_E4M3" if t > 6.0 else "NVFP4"

    kw = dict(disk_gb=budget / 1e9, fmt_for_target=fmt_for_target)
    full, _ = _run(monkeypatch, tmp_path, probe_p, cost_p, **kw)
    part, _ = _run(monkeypatch, tmp_path, probe_p, cost_p,
                   exclude_prefixes=("mtp.",), **kw)

    # Same card, and the draft's mass comes back as shipped body: the cheap
    # rung was all that fit while the draft was priced in.
    assert part["budget_bytes"] == full["budget_bytes"]
    assert full["chosen_achieved_bits"] < part["chosen_achieved_bits"]
    assert (part["predicted_body_tensor_payload_gb"]
            > full["predicted_body_tensor_payload_gb"])
    # ...and the floor shed exactly the draft, nothing else.
    assert round((full["predicted_floor_gb"] - part["predicted_floor_gb"])
                 * 1e9) == _DRAFT_BYTES


def test_exclude_prefix_matching_nothing_exits(monkeypatch, tmp_path):
    """A typo'd prefix excludes zero bytes and reproduces the undershoot with
    every downstream number self-consistent. It must not be a silent no-op —
    and it must not be swallowed by the `except Exception` that degrades
    footprint pricing to a warning."""
    _, probe_p, cost_p = _fixture(
        tmp_path, nvfp4_dloss=1.0, fp8_dloss=0.5,
        extra_tensors=_DRAFT_TENSORS)[:3]
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, probe_p, cost_p, disk_gb=1.0,
             fmt_for_target=lambda t: "FP8_E4M3",
             exclude_prefixes=("mtp_",))
    assert "matched no tensor" in str(exc.value)
