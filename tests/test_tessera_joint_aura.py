"""Checked full-campaign intake; inert fixture wires are not GPU qualification."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, canonical_json_sha256, unit_path, write_unit,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(path), "sha256": sha(path)}


def fixture(tmp_path):
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
    fmt = "TESSERA_E4M3_K1_R1024"
    root = tmp_path / "campaign"
    rowdir = root / "rows/row-0000"
    cache = rowdir / "cache"
    cache.mkdir(parents=True)
    wire = tmp_path / "merged/cache/wire"
    wire.mkdir(parents=True)
    from prismaquant.production_weight_cache import _cache_weight_filename
    source = {"shape": [4, 4], "dtype": "bfloat16", "sha256": "a" * 64}
    identity = {"campaign_schema": "prismaquant.tessera_campaign_cost.v1",
                "currency": "output_mse_under_route_activation_contract",
                "prismaquant_source_sha256": "b" * 64, "encoder_source_sha256": "c" * 64,
                "calibration": {"fit_ids_sha256": "d" * 64},
                "units": {n: {"weight": source, "hessian": None,
                               "input_global_scale": None, "menu": [fmt],
                               "scoring_rows": {"shape": [2, 4], "sha256": "e" * 64}}
                          for n in names}}
    checkpoint = tmp_path / "merged/cost.anchors.json"
    parts = checkpoint.with_name(checkpoint.name + ".parts")
    seal = canonical_json_sha256(identity, where="fixture")
    manifest = {"schema": MANIFEST_SCHEMA, "stage": "Tessera campaign",
                "identity": identity, "identity_sha256": seal,
                "units": [{"qname": n, "file": str(unit_path(parts, n).relative_to(parts))}
                          for n in names]}
    checkpoint.write_text(json.dumps(manifest))
    costs, states = {}, {}
    for n in names:
        filename = n + ".tessera"
        blob = ("inert " + n).encode()
        (wire / filename).write_bytes(blob)
        (cache / _cache_weight_filename(n, fmt)).write_bytes(b"inert render")
        anchor = {"qname": n, "format_name": fmt, "family": "TESSERA_E4M3_K1",
                  "body_rate_q256": 1024, "dloss": 0.25, "dloss_stderr": 0.0,
                  "memory_bytes": 16, "bits_per_param": 8.0,
                  "activation_contract": "fp8_e4m3", "activation_quantized": True,
                  "wire_bytes": len(blob), "seconds": 0.1, "hessian_applied": False,
                  "input_global_scale": None}
        record = {"file": filename, "blob_bytes": len(blob),
                  "blob_sha256": hashlib.sha256(blob).hexdigest(),
                  "identity": {"unit": n, "source": source, "calibration": None,
                               "encoder_source_sha256": "c" * 64,
                               "recipe": {"grid": "E4M3", "q256": 1024}}}
        states[n] = {"anchors": [anchor], "wire_records": {fmt: record}}
        write_unit(parts, stage="Tessera campaign", qname=n,
                   identity_sha256=seal, state=states[n])
        costs[n] = {fmt: {"output_mse": 0.25, "output_mse_measured": True,
                         "cost_source": "tessera_campaign_measured", "tessera_provenance": "measured",
                         "currency": identity["currency"], "tessera_family": anchor["family"],
                         "tessera_body_rate_q256": 1024, "activation_contract": "fp8_e4m3",
                         "activation_quantized": True, "wire_bytes": len(blob),
                         "hessian_identity": {"applied": False, "supplied": False}}}
    census = {"model": "/fixture/model", "unit_shapes": {n: [4, 4] for n in names},
              "anchor_groups": {"g:qk": names}, "max_abs": {names[0]: 1.0, names[1]: 2.0}}
    census_path = root / "census.json"
    census_path.write_text(json.dumps(census))
    plan = {"schema": "prismaquant.tessera_campaign_plan.v1", "model": census["model"],
            "census": str(census_path), "rows": [{"row_id": "row-0000", "members": names,
            "groups": ["g:qk"], "dir": str(rowdir)}]}
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    receipts = root / "receipts.json"
    receipts.write_text(json.dumps({"returncode": 0, "rows": [{"key": "f" * 64,
        "rc": 0, "status": "executed", "host": "fixture"}]}))
    payload = {"schema": identity["campaign_schema"], "currency": identity["currency"],
               "costs": costs, "provenance": {"model": census["model"],
               "cost_mode": "production-render-score", "wire_dir": str(wire),
               "hessian": {"supplied": False}, "stopped_early": False,
               "campaign_fanout": {"schema": plan["schema"], "rows": {"row-0000": ["g:qk"]}},
               "activation_static_scales": {"units": {}, "policy": "fixture"}}}
    cost_path = tmp_path / "merged/cost.pkl"
    cost_path.write_bytes(pickle.dumps(payload))
    config = {"campaign_plan": bind(plan_path), "census": bind(census_path),
              "campaign_receipts": bind(receipts), "merged_cost": bind(cost_path),
              "merged_checkpoint": bind(checkpoint), "required_source_units": 2,
              "required_campaign_groups": 1}
    return config, names, fmt, payload, states


def load(config):
    from prismaquant.tessera_joint_aura import load_measured_anchor_input
    return load_measured_anchor_input(config)


def test_exact_measured_roster_excludes_interpolation(tmp_path):
    config, names, fmt, payload, _ = fixture(tmp_path)
    for rows in payload["costs"].values():
        rows["TESSERA_E4M3_K1_R1000"] = {"output_mse_measured": False,
            "cost_source": "tessera_campaign_interpolated", "output_mse": 0.3}
    path = Path(config["merged_cost"]["path"])
    path.write_bytes(pickle.dumps(payload)); config["merged_cost"] = bind(path)
    data = load(config)
    assert data.formats_by_qname == {n: (fmt, "BF16") for n in names}
    assert set(data.cells) == {(n, fmt) for n in names}
    assert data.total_render_bytes == 64


@pytest.mark.parametrize("mutation", ["missing_unit", "missing_shard", "unfinished_fleet",
    "interpolated_anchor", "unreceipted_measured", "altered_wire",
    "missing_render_undecodable_wire", "missing_render_and_wire",
    "wrong_source", "wrong_scale", "partial_fanout"])
def test_incomplete_or_conflicting_anchor_evidence_refuses(tmp_path, mutation):
    config, names, fmt, payload, states = fixture(tmp_path)
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text())
    parts = path.with_name(path.name + ".parts")
    if mutation == "missing_unit":
        payload["costs"].pop(names[1])
    elif mutation == "missing_shard":
        unit_path(parts, names[1]).unlink()
    elif mutation == "unfinished_fleet":
        receipt = Path(config["campaign_receipts"]["path"])
        receipt.write_text(json.dumps({"returncode": 1, "rows": []}))
        config["campaign_receipts"] = bind(receipt)
    elif mutation == "interpolated_anchor":
        payload["costs"][names[0]][fmt]["output_mse_measured"] = False
    elif mutation == "unreceipted_measured":
        payload["costs"][names[0]]["TESSERA_E4M3_K1_R896"] = dict(payload["costs"][names[0]][fmt])
    elif mutation == "altered_wire":
        (Path(payload["provenance"]["wire_dir"]) / states[names[0]]["wire_records"][fmt]["file"]).write_bytes(b"changed")
    elif mutation in ("missing_render_undecodable_wire", "missing_render_and_wire"):
        # Synthesis is the answer to a missing shard only when the wire behind
        # it decodes. The fixture's wires are inert bytes, so this is the
        # refusal that survives; removing the wire too is the other one.
        from prismaquant.production_weight_cache import _cache_weight_filename
        (tmp_path / "campaign/rows/row-0000/cache" / _cache_weight_filename(names[0], fmt)).unlink()
        if mutation == "missing_render_and_wire":
            (Path(payload["provenance"]["wire_dir"]) /
             states[names[0]]["wire_records"][fmt]["file"]).unlink()
    elif mutation in {"wrong_source", "wrong_scale"}:
        state = states[names[0]]
        if mutation == "wrong_source":
            state["wire_records"][fmt]["identity"]["source"] = {"shape": [8, 4], "dtype": "bfloat16", "sha256": "0" * 64}
        else:
            state["anchors"][0]["input_global_scale"] = 2.0
        write_unit(parts, stage="Tessera campaign", qname=names[0],
                   identity_sha256=manifest["identity_sha256"], state=state)
    else:
        payload["provenance"]["campaign_fanout"]["rows"] = {}
    target = Path(config["merged_cost"]["path"])
    target.write_bytes(pickle.dumps(payload)); config["merged_cost"] = bind(target)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        load(config)


def test_fused_census_maxima_are_used_and_scale_mismatch_refuses(tmp_path):
    from prismaquant.tessera_joint_aura import calibrated_maxima
    from prismaquant.tessera_campaign import _static_input_scales
    from types import SimpleNamespace

    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
    maxima = dict(zip(names, (1.0, 4.0)))
    scales, policy = _static_input_scales(maxima)
    data = SimpleNamespace(census={"max_abs": maxima}, payload={"provenance": {
        "activation_static_scales": {"units": scales, "policy": policy}}})
    unified, actual = calibrated_maxima(data, None)
    assert unified[names[0]] == unified[names[1]] == 4.0
    assert actual == scales
    data.payload["provenance"]["activation_static_scales"]["units"] = {**scales, names[0]: 999.0}
    with pytest.raises(ValueError, match="fused static scales"):
        calibrated_maxima(data, None)


def test_imported_absolute_pwc_paths_release_without_losing_donors(tmp_path):
    import torch
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.joint_aura import prefetch_joint_cache

    fmt = "TESSERA_E4M3_K1_R1024"
    paths = {}
    for index, name in enumerate(("a", "b")):
        path = tmp_path / name / "render.pt"
        path.parent.mkdir()
        torch.save(torch.full((16, 16), index, dtype=torch.bfloat16), path)
        paths[name, fmt] = str(path)
    cache = ProductionWeightCache(weights=dict(paths), levers={})
    cache.enable_lru(1 << 20)
    report = prefetch_joint_cache(cache, ["a", "b"], {"a": [fmt], "b": [fmt]}, max_resident_bytes=1 << 20)
    assert report["entries"] == 2 and report["loaded"] == 2
    assert cache.compact_for_pickle() == 2
    assert cache.weights == paths
    assert torch.equal(cache.get("b", fmt), torch.ones((16, 16), dtype=torch.bfloat16))


def test_wire_verification_rederives_source_before_accepting_render(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_campaign as tc
    from prismaquant.tessera_joint_aura import verify_anchor_render
    from tessera import unit_artifact

    source = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
    rendered = torch.ones_like(source)
    expected_inputs = {"source": source.clone(), "calibration": object(), "projection": {"fixture": True}}
    anchor = dict(qname="model.layers.0.q_proj", format_name="TESSERA_E4M3_K1_R1024",
        family="TESSERA_E4M3_K1", body_rate_q256=1024, dloss=0.25, dloss_stderr=0.0,
        memory_bytes=256, bits_per_param=8.0, activation_contract="fp8_e4m3",
        activation_quantized=True, wire_bytes=4, seconds=0.1, hessian_applied=True,
        input_global_scale=None)
    wire = tmp_path / "fixture.wire"; wire.write_bytes(b"wire")
    cell = {"anchor": anchor, "record": {"expected": "derived"}, "wire": str(wire),
            "render_file_sha256": "a" * 64, "render_origin": "encoded"}
    seen = []
    def derive(value, *, weights, menus, calibration_source, static_scales, projected_units):
        assert value.qname == anchor["qname"]
        assert torch.equal(weights[value.qname], expected_inputs["source"])
        assert calibration_source is expected_inputs["calibration"]
        assert projected_units[value.qname] is expected_inputs["projection"]
        assert menus[value.qname][0].format_name == anchor["format_name"]
        return {"expected": "derived"}
    def verify(blob, record, expected):
        seen.append((blob, expected)); assert record == expected
    monkeypatch.setattr(tc, "_checkpoint_anchor_identity", derive)
    monkeypatch.setattr(tc, "_checkpoint_identity_api", lambda: SimpleNamespace(verify_cached_unit=verify))
    monkeypatch.setattr(unit_artifact, "read_unit_artifact", lambda blob, device: rendered.clone())
    kwargs = dict(calibration_source=expected_inputs["calibration"],
                  projected_unit=expected_inputs["projection"], static_scales={})
    result = verify_anchor_render(cell, source, rendered, **kwargs)
    assert seen and result["wire_sha256"] == sha(wire)
    assert result["render_origin"] == "encoded"
    assert result["render_comparison"] == "independent_render_vs_wire"
    with pytest.raises(ValueError, match="decoded wire differs"):
        verify_anchor_render(cell, source, rendered + 1, **kwargs)
    # A synthesized render's equality leg is the same call and a different
    # claim; it can never report the independent comparison.
    synthesized = {**cell, "render_origin": "synthesized_from_wire"}
    assert (verify_anchor_render(synthesized, source, rendered, **kwargs)["render_comparison"]
            == "wire_round_trip_only")
    with pytest.raises(ValueError, match="no longer decodes from its wire"):
        verify_anchor_render(synthesized, source, rendered + 1, **kwargs)
    for origin in (None, "unknown", True):
        with pytest.raises(ValueError, match="closed-vocabulary render_origin"):
            verify_anchor_render({**cell, "render_origin": origin}, source, rendered, **kwargs)
    with pytest.raises(ValueError, match="closed-vocabulary render_origin"):
        verify_anchor_render({k: v for k, v in cell.items() if k != "render_origin"},
                             source, rendered, **kwargs)


@pytest.mark.parametrize("field,value", [("probe_microbatch", 0), ("n_probes", 1),
    ("token_scope", "last"), ("production_act_scales", "1"), ("temperature", 2.0)])
def test_plan_refuses_unqualified_probe_or_activation_policies(tmp_path, field, value):
    from prismaquant.tessera_joint_aura import _load_plan, SCHEMA
    config = {"schema": SCHEMA, "execution": {"n_calib_samples": 512,
        "calib_seqlen": 512, "probe_microbatch": 1, "n_probes": 4,
        "seed_base": 7000, "token_scope": "all", "temperature": 1.0,
        "production_act_scales": "0"}, "profile_tool": "cprofile",
        "max_render_bytes": 1024, "max_gpu_bytes": 2048, "min_free_gib": 0}
    config["source_prefetch"] = dict(max_cache_slots=24, prefetch_workers=4,
        prefetch_lookahead=4, cache_headroom_gb=4.0,
        prefetch_min_available_gb=2.0, require_prefetched_residency=True)
    config["execution"][field] = value
    path = tmp_path / "plan.json"; path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        _load_plan(path, sha(path))


@pytest.mark.parametrize("prior", [None, "1"])
def test_execute_scopes_the_activation_scale_env_to_the_call(tmp_path, monkeypatch, prior):
    """The plan's activation-scale value is live inside execute and gone after.

    execute sets PRISMAQUANT_PROD_ACT_SCALES so the render path it drives
    reads the campaign's value, and every admitted plan carries "0".  Left
    behind, that "0" is the input that turns the render scorer's activation
    clip off for the rest of the process: with it set,
    ``_local_forward_render_score`` skips the clamp and reports
    ``activation_clipped`` False, which is what
    ``tests/test_served_activation_contract.py::
    test_stock_nvfp4_cache_score_is_unchanged`` then fails on.  Both halves
    are asserted here; restoring without setting would be just as wrong.
    """
    import os
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    key = bridge.ACTIVATION_SCALE_ENV
    monkeypatch.delenv(key, raising=False)
    if prior is not None:
        monkeypatch.setenv(key, prior)

    draw = dict(fit_ids_sha256="a" * 64, text_sha256="b" * 64, nsamples=512, seqlen=512, seed=0)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    monkeypatch.setattr(bridge, "load_measured_anchor_input", lambda _inputs, **_kwargs: SimpleNamespace(
        census={"model": "fixture", "attention_implementation": "eager"}, cells={},
        payload={"provenance": {"hessian": {"calibration_identity": draw}}}))
    # Called after the write, so it is where the live value can be read.
    during = {}
    def _observe(*_args, **_kwargs):
        during["value"] = os.environ.get(key)
        return torch.zeros((1, 512), dtype=torch.int64), {"provenance": {**draw, "nsamples": 1}}
    monkeypatch.setattr(calibration_data, "load_calibration_input", _observe)
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", lambda *_a, **_k:
        pytest.fail("subset draw reached model construction"))
    config = {"model": "fixture", "inputs": {}, "output_root": str(tmp_path),
        "calibration_input": {"path": "fixture", "sha256": "a" * 64},
        "profile_tool": "cprofile",
        "execution": {"production_act_scales": "0", "n_calib_samples": 1, "calib_seqlen": 512}}

    with pytest.raises(ValueError, match="original full draw nsamples"):
        bridge.execute("prepare", config, plan_sha256="b" * 64)

    assert during["value"] == "0", "the plan's value must be live inside execute"
    assert os.environ.get(key) == prior, "and must not outlive the call"


@pytest.mark.parametrize("profile_tool", ["cprofile", "py-spy"])
def test_original_full_draw_refuses_subset_before_model_load(tmp_path, monkeypatch, profile_tool):
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    draw = dict(fit_ids_sha256="a" * 64, text_sha256="b" * 64, nsamples=512, seqlen=512, seed=0)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    monkeypatch.setattr(bridge, "load_measured_anchor_input", lambda _inputs, **_kwargs: SimpleNamespace(
        census={"model": "fixture", "attention_implementation": "eager"}, cells={},
        payload={"provenance": {"hessian": {"calibration_identity": draw}}}))
    monkeypatch.setattr(calibration_data, "load_calibration_input", lambda *_args, **_kwargs:
        (torch.zeros((1, 512), dtype=torch.int64), {"provenance": {**draw, "nsamples": 1}}))
    def forbidden(*_args, **_kwargs):
        pytest.fail("subset draw reached model construction")
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", forbidden)
    config = {"model": "fixture", "inputs": {}, "output_root": str(tmp_path),
        "calibration_input": {"path": "fixture", "sha256": "a" * 64},
        "execution": {"production_act_scales": "0", "n_calib_samples": 1, "calib_seqlen": 512}}
    config["profile_tool"] = profile_tool
    if profile_tool == "py-spy":
        import os
        session = tmp_path / "child-start.json"
        session.write_text(json.dumps({"schema": "prismaquant.profiled_command_start.v1",
            "wrapper_pid": os.getppid(),
            "command": ["python", "-m", "prismaquant.tessera_joint_aura", "prepare"]}))
        monkeypatch.setenv("PRISMAQUANT_SAMPLER_SESSION", str(session))
    with pytest.raises(ValueError, match="original full draw nsamples"):
        bridge.execute("prepare", config, plan_sha256="b" * 64)
    result = json.loads((tmp_path / "prepare/results.json").read_text())
    assert result["passed"] is False
    assert (tmp_path / "prepare/profile.pstats").is_file() is (profile_tool == "cprofile")
    if profile_tool == "py-spy":
        assert result["sampling_session"]["sha256"] == sha(session)


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_explicit_source_prefetch_reaches_streamed_builder(tmp_path, monkeypatch, command):
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: object())
    draw = dict(fit_ids_sha256="a" * 64, text_sha256="b" * 64, nsamples=512, seqlen=512, seed=0)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    def intake(_inputs, **kwargs):
        assert kwargs == {"reader": None,
                          **({"verify_payloads": False} if command == "prepare" else {})}
        return SimpleNamespace(census={"model": "fixture", "attention_implementation": "eager"},
            cells={}, payload={"provenance": {"hessian": {"calibration_identity": draw}}})
    monkeypatch.setattr(bridge, "load_measured_anchor_input", intake)
    monkeypatch.setattr(calibration_data, "load_calibration_input", lambda *_args, **_kwargs:
        (torch.zeros((512, 512), dtype=torch.int64), {"provenance": draw}))
    prefetch = dict(max_cache_slots=24, prefetch_workers=4, prefetch_lookahead=4,
        cache_headroom_gb=4.0, prefetch_min_available_gb=2.0, require_prefetched_residency=True)
    class Reached(Exception):
        pass
    def inspect(*_args, **kwargs):
        assert {key: kwargs.get(key) for key in prefetch} == prefetch
        raise Reached
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", inspect)
    config = {"model": "fixture", "inputs": {}, "output_root": str(tmp_path),
        "source_prefetch": prefetch,
        "calibration_input": {"path": "fixture", "sha256": "a" * 64},
        "execution": {"production_act_scales": "0", "n_calib_samples": 512, "calib_seqlen": 512}}
    with pytest.raises(Reached):
        bridge.execute(command, config, plan_sha256="b" * 64)


@pytest.mark.parametrize("defect", ["missing", "disabled", "auto_slots", "zero_workers",
    "oversize_lookahead", "nonfinite_headroom", "extra_field"])
def test_source_prefetch_refuses_implicit_or_nonresident_settings(defect):
    from prismaquant.tessera_joint_aura import _source_prefetch
    prefetch = dict(max_cache_slots=24, prefetch_workers=4, prefetch_lookahead=4,
        cache_headroom_gb=4.0, prefetch_min_available_gb=2.0,
        require_prefetched_residency=True)
    if defect == "missing":
        config = {}
    else:
        field, value = {"disabled": ("require_prefetched_residency", False),
            "auto_slots": ("max_cache_slots", None), "zero_workers": ("prefetch_workers", 0),
            "oversize_lookahead": ("prefetch_lookahead", 24),
            "nonfinite_headroom": ("cache_headroom_gb", float("inf")),
            "extra_field": ("unreviewed_fallback", True)}[defect]
        prefetch[field] = value
        config = {"source_prefetch": prefetch}
    with pytest.raises(ValueError, match="source_prefetch"):
        _source_prefetch(config)


def test_parallel_intake_hashes_independent_files_without_changing_cells(tmp_path, monkeypatch):
    import threading
    from prismaquant import tessera_joint_aura as bridge
    config, *_ = fixture(tmp_path)
    expected = bridge.load_measured_anchor_input(config)
    original = bridge._sha
    barrier = threading.Barrier(2)
    workers = set()
    def observed(path):
        if str(path).endswith('.tessera'):
            workers.add(threading.get_ident())
            barrier.wait(timeout=10)
        return original(path)
    monkeypatch.setattr(bridge, '_sha', observed)
    actual = bridge.load_measured_anchor_input(config, file_hash_workers=2)
    assert actual.cells == expected.cells
    assert actual.formats_by_qname == expected.formats_by_qname
    assert len(workers) == 2


@pytest.mark.parametrize("defect", ["missing", "parent", "command", "schema"])
def test_sampling_refuses_unobserved_execution(tmp_path, monkeypatch, defect):
    import os
    from prismaquant import tessera_joint_aura as bridge, gpu_guard
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    session = {"schema": "prismaquant.profiled_command_start.v1",
        "wrapper_pid": os.getppid(),
        "command": ["python", "-m", "prismaquant.tessera_joint_aura", "prepare"]}
    if defect == "parent":
        session["wrapper_pid"] = -1
    elif defect == "command":
        session["command"][-1] = "run"
    elif defect == "schema":
        session["schema"] = "unbound"
    path = tmp_path / "child-start.json"
    if defect != "missing":
        path.write_text(json.dumps(session))
    monkeypatch.setenv("PRISMAQUANT_SAMPLER_SESSION", str(path))
    config = {"output_root": str(tmp_path), "profile_tool": "py-spy",
              "execution": {"production_act_scales": "0"}}
    with pytest.raises(ValueError, match="sampl|observed"):
        bridge.execute("prepare", config, plan_sha256="b" * 64)


def test_prepare_metadata_intake_defers_heavy_files_but_keeps_strict_default(tmp_path, monkeypatch):
    from prismaquant import tessera_joint_aura as bridge
    config, names, fmt, _payload, _states = fixture(tmp_path)
    calls = []
    original = bridge._sha
    def observed(path):
        calls.append(Path(path))
        return original(path)
    monkeypatch.setattr(bridge, '_sha', observed)
    metadata = bridge.load_measured_anchor_input(config, verify_payloads=False)
    heavy = {Path(cell[k]) for cell in metadata.cells.values() for k in ('wire', 'render')}
    assert not heavy.intersection(calls)
    assert all('render_file_sha256' not in cell for cell in metadata.cells.values())
    assert metadata.formats_by_qname == {name: (fmt, 'BF16') for name in names}
    calls.clear()
    strict = bridge.load_measured_anchor_input(config)
    assert heavy <= set(calls)
    assert all('render_file_sha256' in cell for cell in strict.cells.values())


def test_prepare_refuses_oversized_later_render_before_loading_any_layer(tmp_path):
    from types import SimpleNamespace
    from prismaquant import tessera_joint_aura as bridge
    small, huge = tmp_path / 'early.pt', tmp_path / 'later.pt'
    small.write_bytes(b'early')
    with huge.open('wb') as stream:
        stream.truncate(8192)
    data = SimpleNamespace(cells={('early', 'fmt'): {'render': str(small)},
                                  ('later', 'fmt'): {'render': str(huge)}})
    with pytest.raises(ValueError, match='read buffer|shard|budget'):
        bridge._prepare_file_read_bound(data, max_render_bytes=4096)


def _unlink_render(tmp_path, name, fmt):
    from prismaquant.production_weight_cache import _cache_weight_filename

    path = tmp_path / "campaign/rows/row-0000/cache" / _cache_weight_filename(name, fmt)
    path.unlink()
    return path


def _decoder(monkeypatch, tensor):
    """Bind the module-level decode seam to a fixed answer."""
    from tessera import unit_artifact

    seen = []

    def decode(blob, device="cpu"):
        seen.append((blob, device))
        return tensor.clone() if hasattr(tensor, "clone") else tensor

    monkeypatch.setattr(unit_artifact, "read_unit_artifact", decode)
    return seen


def test_adopted_rung_render_is_synthesized_from_its_wire_and_stays_marked(tmp_path, monkeypatch):
    """An adopted rung's shard is written from its wire, and says so -- twice.

    The second load is the prepare/run boundary: the ``.pt`` now exists, and
    the origin must not quietly become ``encoded`` because of it.
    """
    import torch
    from prismaquant import tessera_joint_aura as bridge

    config, names, fmt, _payload, states = fixture(tmp_path)
    render = _unlink_render(tmp_path, names[0], fmt)
    decoded = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    seen = _decoder(monkeypatch, decoded)

    data = load(config)
    assert data.cells[names[0], fmt]["render_origin"] == "synthesized_from_wire"
    assert data.cells[names[1], fmt]["render_origin"] == "encoded"
    assert bridge.cell_render_census(data.cells) == {
        "render_origins": {"encoded": 1, "synthesized_from_wire": 1},
        "render_comparisons": {"independent_render_vs_wire": 1, "wire_round_trip_only": 1}}
    assert len(seen) == 1 and seen[0][0] == (tmp_path / "merged/cache/wire" /
        states[names[0]]["wire_records"][fmt]["file"]).read_bytes()
    assert render.is_file() and torch.equal(torch.load(render, weights_only=True), decoded)
    marker = json.loads(bridge._render_origin_marker_path(render).read_text())
    assert marker == {"schema": bridge.RENDER_ORIGIN_SCHEMA,
                      "render_origin": "synthesized_from_wire", "unit": names[0],
                      "format_name": fmt, "wire_file": names[0] + ".tessera",
                      "wire_sha256": states[names[0]]["wire_records"][fmt]["blob_sha256"]}

    again = load(config)
    assert again.cells[names[0], fmt]["render_origin"] == "synthesized_from_wire"
    assert again.cells[names[1], fmt]["render_origin"] == "encoded"
    assert len(seen) == 1, "an existing shard must not be re-decoded"


def test_synthesis_uses_the_bound_reader_when_one_is_declared(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from prismaquant.tessera_joint_aura import load_measured_anchor_input

    config, names, fmt, _payload, _states = fixture(tmp_path)
    _unlink_render(tmp_path, names[0], fmt)
    module_level = _decoder(monkeypatch, torch.zeros((4, 4), dtype=torch.bfloat16))
    bound = []

    def read(blob, device="cpu"):
        bound.append(blob)
        return torch.ones((4, 4), dtype=torch.bfloat16)

    data = load_measured_anchor_input(config, reader=SimpleNamespace(read_unit_artifact=read))
    assert bound and not module_level
    assert data.cells[names[0], fmt]["render_origin"] == "synthesized_from_wire"


@pytest.mark.parametrize("field,value", [("schema", "other"), ("unit", "elsewhere"),
    ("format_name", "TESSERA_E4M3_K1_R768"), ("wire_sha256", "0" * 64),
    ("render_origin", "encoded")])
def test_a_render_origin_marker_that_names_another_rung_refuses(tmp_path, monkeypatch, field, value):
    import torch
    from prismaquant import tessera_joint_aura as bridge

    config, names, fmt, _payload, _states = fixture(tmp_path)
    render = _unlink_render(tmp_path, names[0], fmt)
    _decoder(monkeypatch, torch.zeros((4, 4), dtype=torch.bfloat16))
    load(config)
    marker = bridge._render_origin_marker_path(render)
    stamp = json.loads(marker.read_text())
    stamp[field] = value
    marker.write_text(json.dumps(stamp))
    with pytest.raises(ValueError, match="marked render|render origin schema|names another wire"):
        load(config)


@pytest.mark.parametrize("answer", ["raises", "wrong_shape", "nonfinite", "not_a_tensor"])
def test_a_wire_that_does_not_decode_to_the_census_render_still_refuses(tmp_path, monkeypatch, answer):
    import torch
    from tessera import unit_artifact

    config, names, fmt, _payload, _states = fixture(tmp_path)
    _unlink_render(tmp_path, names[0], fmt)
    values = {"wrong_shape": torch.zeros((2, 8), dtype=torch.bfloat16),
              "nonfinite": torch.full((4, 4), float("nan"), dtype=torch.bfloat16),
              "not_a_tensor": None}

    def decode(blob, device="cpu"):
        if answer == "raises":
            raise RuntimeError("undecodable")
        return values[answer]

    monkeypatch.setattr(unit_artifact, "read_unit_artifact", decode)
    with pytest.raises(ValueError, match="does not decode|not the census BF16 render|NoneType"):
        load(config)


def test_a_corrupted_synthesized_shard_fails_its_round_trip_leg(tmp_path, monkeypatch):
    """The equality leg is vacuous about the encode and live about the bytes."""
    import torch
    from types import SimpleNamespace
    from prismaquant import tessera_campaign as tc
    from prismaquant.tessera_joint_aura import verify_anchor_render
    from tessera import unit_artifact

    config, names, fmt, _payload, _states = fixture(tmp_path)
    render = _unlink_render(tmp_path, names[0], fmt)
    decoded = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    _decoder(monkeypatch, decoded)
    data = load(config)
    cell = data.cells[names[0], fmt]
    assert cell["render_origin"] == "synthesized_from_wire"

    monkeypatch.setattr(tc, "_checkpoint_anchor_identity",
                        lambda *_args, **_kwargs: {"expected": "derived"})
    monkeypatch.setattr(tc, "_checkpoint_identity_api",
                        lambda: SimpleNamespace(verify_cached_unit=lambda *_args: None))
    monkeypatch.setattr(unit_artifact, "read_unit_artifact",
                        lambda blob, device="cpu": decoded.clone())
    source = torch.zeros((4, 4), dtype=torch.bfloat16)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={})
    record = verify_anchor_render(cell, source, torch.load(render, weights_only=True), **kwargs)
    assert record["render_comparison"] == "wire_round_trip_only"

    # Corrupt the published shard: the same call now refuses, which is the
    # only thing this leg was ever able to establish for a synthesized rung.
    torch.save(decoded + 1, render)
    with pytest.raises(ValueError, match="no longer decodes from its wire"):
        verify_anchor_render(cell, source, torch.load(render, weights_only=True), **kwargs)


def test_render_census_counts_every_vocabulary_value_and_refuses_unknown_ones():
    from prismaquant.tessera_joint_aura import cell_render_census, render_origin_census

    assert render_origin_census([]) == {
        "render_origins": {"encoded": 0, "synthesized_from_wire": 0},
        "render_comparisons": {"independent_render_vs_wire": 0, "wire_round_trip_only": 0}}
    with pytest.raises(ValueError, match="unknown render origin"):
        render_origin_census(["adopted"])
    with pytest.raises(ValueError, match="carries no render_origin"):
        cell_render_census({("unit", "fmt"): {}})
