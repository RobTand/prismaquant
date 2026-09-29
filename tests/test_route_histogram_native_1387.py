"""Native compressed-tensors ship records carry the route histogram (PQ #1387),
and a format summary names every route its units take (PQ #1289).

Principle 12: every published size or quality claim carries the artifact's
route histogram beside the bpp. Until #1387 only a Tessera allocation wrote
``serving_lane_provenance``; a native compressed-tensors allocation wrote none,
so its card could not carry ``build.route_histogram`` and ``verify`` could not
owe it. Until #1289 a format whose dense and routed units took different routes
reported ``route: null`` in ``by_format`` and hid both.
"""
from __future__ import annotations

import json
import pickle
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

import prismaquant.allocator as alloc
import prismaquant.allocator_candidates as ac
from prismaquant import footprint as fp
from prismaquant import shipcard
from prismaquant.layer_config import read_layer_config_metadata
from prismaquant.serving_profiles import ResolvedServingLane


# ---------------------------------------------------------------------------
# #1289: one format, two structures, two routes
# ---------------------------------------------------------------------------
_FORMAT = "TESSERA_E2M1_K2_R896"
_PROFILE = "tessera_research_sm121"


@dataclass(frozen=True)
class _Context:
    platform: str = "sm_121"
    structure: str = "dense"
    residency: str = "resident"
    runtime_image: str = "example/vllm@sha256:" + "a" * 64
    execution_mode: str = "eager"

    def key(self) -> tuple[str, ...]:
        return (self.platform, self.structure, self.residency,
                self.runtime_image, self.execution_mode)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _lane(structure: str) -> ResolvedServingLane:
    return ResolvedServingLane(
        lane_id=f"tessera-{structure}",
        format=_FORMAT,
        activation_contract="W4A4",
        fallback_route=f"native-{structure}",
        fused_mid_m_backed=False,
        fused_mid_m_rungs=(),
        fused_mid_m_range=None,
        runtime_version="test-runtime",
        rungs_source="test-contract",
        route_status="backed",
        route_status_source=f"test-contract:{structure}",
    )


def test_a_format_on_dense_and_routed_units_names_both_routes(monkeypatch):
    lanes = {s: _lane(s) for s in ("dense", "routed_moe")}
    monkeypatch.setattr(
        ac, "serving_lane_route",
        lambda profile, fmt, *, serving_context=None, **kw:
            lanes[serving_context.structure])
    contexts = {
        "unit.a": _Context(),
        "unit.b": _Context(),
        "unit.c": _Context(structure="routed_moe"),
    }
    report = ac.selection_serving_lane_provenance(
        dict.fromkeys(contexts, _FORMAT), candidates=None,
        target_profile=_PROFILE, context_by_unit=contexts)

    row = report["by_format"][_FORMAT]
    assert row["units"] == 3
    # The single-route field stays None when the units disagree; the
    # histogram is what names them.
    assert row["route"] is None
    routes = row["routes"]
    assert sorted(r["route"]["fallback_route"] for r in routes) == [
        "native-dense", "native-routed_moe"]
    by_route = {r["route"]["fallback_route"]: r for r in routes}
    assert by_route["native-dense"]["units"] == 2
    assert by_route["native-routed_moe"]["units"] == 1
    assert by_route["native-dense"]["structures"] == {"dense": 2}
    assert by_route["native-routed_moe"]["structures"] == {"routed_moe": 1}


def test_a_format_with_one_route_still_reports_it_once(monkeypatch):
    lane = _lane("dense")
    monkeypatch.setattr(
        ac, "serving_lane_route", lambda *a, **kw: lane)
    contexts = {"unit.a": _Context(), "unit.b": _Context()}
    report = ac.selection_serving_lane_provenance(
        dict.fromkeys(contexts, _FORMAT), candidates=None,
        target_profile=_PROFILE, context_by_unit=contexts)
    row = report["by_format"][_FORMAT]
    assert row["route"]["fallback_route"] == "native-dense"
    assert [r["units"] for r in row["routes"]] == [2]


# ---------------------------------------------------------------------------
# #1387: the native allocation writes the provenance
# ---------------------------------------------------------------------------
_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
_OUT = _IN = 256
_FLOOR_TENSORS = {
    "model.embed_tokens.weight": ("BF16", (512, 64)),
    "lm_head.weight": ("BF16", (512, 64)),
    "model.norm.weight": ("BF16", (64,)),
}


def _write_safetensors(path, tensors):
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)


def _allocator_fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    for n in _NAMES:
        tensors[f"{n}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)
    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": _OUT * _IN,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(_NAMES)
    }
    probe = {"stats": stats, "meta": {"model": str(model_dir)}}
    costs = {
        "costs": {
            n: {
                "NVFP4": {"weight_mse": 1e-4, "output_mse": 4e-4,
                          "output_mse_measured": True},
                "FP8_E4M3": {"weight_mse": 1e-6, "output_mse": 2e-6,
                             "output_mse_measured": True},
            }
            for n in _NAMES
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return probe_p, cost_p


def _stub_solver(fmt):
    def solve(stats, candidates, target_bits, format_specs, format_rank,
              bit_precision, **kw):
        assign = {n: fmt for n in candidates}
        total_params = sum(stats[n]["n_params"] for n in assign)
        bits = sum(
            8.0 * next(c for c in candidates[n] if c.fmt == fmt).memory_bytes
            for n in assign
        )
        achieved = bits / max(total_params, 1)
        diag = kw.get("diagnostics")
        if diag is not None:
            diag.update({"feasible": True, "achieved_bits": achieved,
                         "predicted_dloss": None, "evals": 1})
        return assign, achieved
    return solve


def test_a_native_allocation_writes_serving_lane_provenance(monkeypatch, tmp_path):
    probe_p, cost_p = _allocator_fixture(tmp_path)
    monkeypatch.setattr(alloc, "solve_with_promotion", _stub_solver("NVFP4"))
    lc = tmp_path / "layer_config.json"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", "4.6",
        "--target-bits", "4.6",
        "--layer-config", str(lc),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
    ])
    alloc.main()

    meta = read_layer_config_metadata(lc)
    prov = meta.get("serving_lane_provenance")
    assert prov is not None, sorted(meta)
    assert prov["units_total"] == len(_NAMES)
    assert prov["route_status_counts"] == {"no_declared_lane": len(_NAMES)}
    # The recipe's scope stays a Tessera-only claim.
    assert "tessera_serving_scope" not in meta


# ---------------------------------------------------------------------------
# #1387: a recompute from the bare assignment equals the allocator's own
# ---------------------------------------------------------------------------
def _run_allocator_capturing_provenance(monkeypatch, tmp_path, profile):
    """Run the allocator and return every (assignment, profile, report) the
    real ``selection_serving_lane_provenance`` call produced."""
    probe_p, cost_p = _allocator_fixture(tmp_path)
    monkeypatch.setattr(alloc, "solve_with_promotion", _stub_solver("NVFP4"))
    seen = []
    real = alloc.selection_serving_lane_provenance

    def spy(assignment, candidates=None, target_profile=None, **kw):
        report = real(assignment, candidates, target_profile, **kw)
        seen.append({
            "assignment": dict(assignment),
            "had_candidates": candidates is not None,
            "profile": target_profile,
            "report": report,
        })
        return report

    monkeypatch.setattr(alloc, "selection_serving_lane_provenance", spy)
    lc = tmp_path / "layer_config.json"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", "4.6",
        "--target-bits", "4.6",
        "--layer-config", str(lc),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--allow-default-profile",
        "--target-profile", profile,
    ])
    alloc.main()
    return seen, read_layer_config_metadata(lc)


def _without_branches(report):
    return {k: v for k, v in report.items()
            if k != "activation_pricing_branches"}


@pytest.mark.parametrize("profile", ["research", "vllm_packed_moe"])
def test_recompute_from_the_bare_assignment_equals_the_allocators_report(
        monkeypatch, tmp_path, profile):
    """The exporter and the frontier selector have only the assignment. What
    they derive must be the object the allocator, holding the candidates,
    would have stamped -- ``by_format.routes`` included. The one field a bare
    assignment cannot carry is the candidates' activation-pricing branch,
    which the recompute names ``unrecorded`` rather than guess."""
    seen, meta = _run_allocator_capturing_provenance(
        monkeypatch, tmp_path, profile)
    assert seen and any(s["had_candidates"] for s in seen)
    assert meta.get("serving_lane_provenance") is not None
    for call in seen:
        recomputed = ac.recompute_serving_lane_provenance(
            call["assignment"], call["profile"])
        assert recomputed is not None, call["profile"]
        assert _without_branches(recomputed) == _without_branches(
            call["report"])
        assert recomputed["by_format"] == call["report"]["by_format"]
        assert recomputed["activation_pricing_branches"] == {
            "unrecorded": recomputed["units_total"]}


def test_recompute_refuses_a_scoped_tessera_assignment(monkeypatch):
    """A scoped Tessera unit prices differently with and without its serving
    context, and a bare assignment has no context. Deriving would guess."""
    from prismaquant.tessera_menu import TESSERA_LANE_ID

    lane = ResolvedServingLane(
        lane_id=TESSERA_LANE_ID, format=_FORMAT, activation_contract="W4A4",
        fallback_route="native", fused_mid_m_backed=False,
        fused_mid_m_rungs=(), fused_mid_m_range=None,
        runtime_version="test-runtime", rungs_source="test-contract",
        route_status="backed", route_status_source="test-contract")
    monkeypatch.setattr(ac, "serving_lane_route", lambda *a, **kw: lane)
    assert ac.recompute_serving_lane_provenance(
        {"unit.a": _FORMAT}, _PROFILE) is None


@pytest.mark.parametrize("profile", [None, ""])
def test_recompute_refuses_without_a_target_profile(profile):
    assert ac.recompute_serving_lane_provenance(
        {"unit.a": "NVFP4"}, profile) is None


def test_routes_are_ordered_by_the_canonical_key_not_dict_order(monkeypatch):
    """Two routes must group and sort by the digests owner's canonical text,
    so a producer that emits the same fields in another key order still lands
    in the same group."""
    from prismaquant.digests import DIRECT_UTF8_STRICT

    lanes = {s: _lane(s) for s in ("dense", "routed_moe")}
    monkeypatch.setattr(
        ac, "serving_lane_route",
        lambda profile, fmt, *, serving_context=None, **kw:
            lanes[serving_context.structure])
    contexts = {
        "unit.a": _Context(structure="routed_moe"),
        "unit.b": _Context(),
        "unit.c": _Context(structure="routed_moe"),
    }
    report = ac.selection_serving_lane_provenance(
        dict.fromkeys(contexts, _FORMAT), candidates=None,
        target_profile=_PROFILE, context_by_unit=contexts)
    routes = report["by_format"][_FORMAT]["routes"]
    keys = [DIRECT_UTF8_STRICT.text(r["route"]) for r in routes]
    assert keys == sorted(keys)
    assert sorted(r["units"] for r in routes) == [1, 2]


# ---------------------------------------------------------------------------
# #1387: the exporter stamps it and verify owes it
# ---------------------------------------------------------------------------
def _layer_config(tmp_path, meta):
    path = tmp_path / "layer_config_for_card.json"
    path.write_text(json.dumps({"__prismaquant__": meta}))
    return path


def _write_native_card(tmp_path, monkeypatch, *, layer_config_path=None):
    from prismaquant import export_native_compressed as exp
    from prismaquant import read_traffic

    monkeypatch.setattr(read_traffic, "read_traffic_claim", lambda d: None)
    out = tmp_path / "exported"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (out / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_DYNAMIC",
        "model.layers.1.mlp.down_proj": "BF16",
    }
    exp._write_shipcard(
        out, source_model=str(out),
        layer_config_path=layer_config_path,
        assignment=assignment, config_assignment=assignment, hist={})
    return out, json.loads((out / shipcard.SHIPCARD_FILENAME).read_text())


def test_the_native_exporter_stamps_the_route_histogram(tmp_path, monkeypatch):
    lc = _layer_config(tmp_path, {"target_profile": "research"})
    _out, card = _write_native_card(
        tmp_path, monkeypatch, layer_config_path=lc)
    histogram = card["build"].get("route_histogram")
    assert histogram is not None, sorted(card["build"])
    assert histogram["schema"] == shipcard.ROUTE_HISTOGRAM_SCHEMA
    assert histogram["units_total"] == 3
    assert sum(histogram["route_status_counts"].values()) == 3
    assert card["build"]["route_histogram_owed"] is True
    assert [p for p in shipcard.verify(card) if "route_histogram" in p] == []


def test_the_exporter_stamps_nothing_without_a_target_profile(
        tmp_path, monkeypatch):
    """Deriving under profile None would report every unit against the
    research profile, a claim about a target the recipe never named."""
    for name, lc in (("no_config", None),
                     ("no_profile", _layer_config(tmp_path, {}))):
        sub = tmp_path / name
        sub.mkdir()
        _out, card = _write_native_card(sub, monkeypatch, layer_config_path=lc)
        assert "route_histogram" not in card["build"], name
        assert "route_histogram_owed" not in card["build"], name
        assert [p for p in shipcard.verify(card)
                if "route_histogram" in p] == [], name


def _native_card(tmp_path, *, build):
    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return shipcard.build_shipcard(model_dir, build=build, lane=None)


def test_a_card_marked_as_owing_the_histogram_is_refused_without_it(tmp_path):
    card = _native_card(tmp_path, build={"route_histogram_owed": True})
    problems = [p for p in shipcard.verify(card) if "route_histogram" in p]
    assert problems, shipcard.verify(card)


def test_a_marked_card_with_the_histogram_verifies(tmp_path):
    histogram = {
        "schema": shipcard.ROUTE_HISTOGRAM_SCHEMA,
        "units_total": 3,
        "route_status_counts": {"no_declared_lane": 3},
        "activation_contracts": {},
    }
    card = _native_card(tmp_path, build={
        "route_histogram_owed": True, "route_histogram": histogram})
    assert [p for p in shipcard.verify(card) if "route_histogram" in p] == []


def test_a_historical_native_card_without_the_marker_still_verifies(tmp_path):
    card = _native_card(tmp_path, build={})
    assert [p for p in shipcard.verify(card) if "route_histogram" in p] == []


# ---------------------------------------------------------------------------
# #1387: the validated-frontier selector must not refuse the new provenance
# ---------------------------------------------------------------------------
def _run_selector(tmp_path, meta):
    assignment_path = tmp_path / "cand.json"
    assignment_path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.mlp.down_proj": "BF16",
        },
    }))
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"results": [{
        "label": "alloc_4.0", "path": str(assignment_path), "bpp": 4.0,
        "last_token_kl": 0.02,
        "format_counts": {"NVFP4": 1, "BF16": 1},
    }]}))
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text(json.dumps({
        "model.layers.0.self_attn.q_proj": {"data_type": "float", "bits": 16},
        "model.layers.0.mlp.down_proj": {"data_type": "float", "bits": 16},
        "__prismaquant__": meta,
    }))
    proc = subprocess.run(
        [sys.executable, "-m", "prismaquant.select_validated_frontier",
         "--validation-json", str(validation), "--mode", "best-kl",
         "--output-layer-config", str(layer_config),
         "--output-assignment", str(tmp_path / "selected.json"),
         "--output-summary", str(tmp_path / "summary.json")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The selector writes no reserved block at all when nothing is carried.
    return json.loads(layer_config.read_text()).get("__prismaquant__", {})


_STALE_PROVENANCE = {
    "schema": "prismaquant.serving_lane_route.v1",
    "units_total": 99,
    "route_status_counts": {"no_declared_lane": 99},
}


def test_selector_recomputes_provenance_for_a_laneless_destination(tmp_path):
    """The allocator now writes provenance for every allocation, so a
    frontier pick that differs from the allocator's own assignment finds
    provenance in its destination. The selector owns the pick, so it
    recomputes the census rather than refusing or carrying a stale one."""
    meta = _run_selector(tmp_path, {
        "target_profile": "research",
        "serving_lane_provenance": dict(_STALE_PROVENANCE)})
    prov = meta["serving_lane_provenance"]
    assert prov["units_total"] == 2
    assert prov["route_status_counts"] == {"no_declared_lane": 2}


def test_selector_drops_stale_provenance_when_no_profile_is_named(tmp_path):
    """No target profile means nothing to derive against; the stale census
    must not be carried onto the pick either."""
    meta = _run_selector(tmp_path, {
        "serving_lane_provenance": dict(_STALE_PROVENANCE)})
    assert "serving_lane_provenance" not in meta
