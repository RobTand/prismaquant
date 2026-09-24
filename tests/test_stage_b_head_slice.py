"""The sealed Stage B head slice (PQ #1010).

The head intake runs once, in the metadata producer; each layer quantum's
``head`` phase declares its slice and the head files the slice binds, and the
quantum reads nothing else from the campaign inputs. These tests cover the
producer/consumer round trip on the Tessera campaign fixture, the refusals
(another layer's slice, another preparation's slice, tampered bytes), the
opened-path audit, the progress report, and the executable record diff.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant.joint_stage_b_head import (  # noqa: E402
    HEAD_SLICE_SCHEMA,
    HeadSliceRefused,
    build_head_slices,
    head_slice_bytes,
    head_slice_path,
    load_quantum_head,
    read_prepared_head,
    read_quantum_head_slice,
)

PLAN_SHA = "c" * 64
PROJECTION = {"name": "torch"}


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind(path):
    return {"path": str(path), "sha256": _sha(path)}


def _calibration(root):
    from safetensors.torch import save_file

    ids = torch.arange(2 * 4).reshape(2, 4)
    provenance = {"source": "fixture", "fit_tokens": ids.numel(), "nsamples": 2,
                  "seqlen": 4, "seed": 11, "text_sha256": "1" * 64,
                  "fit_ids_sha256": hashlib.sha256(
                      ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    path = root / "calibration" / "tokens.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    return path


def _campaign(tmp_path):
    """The fixture campaign, prepared: plan, completion, PWC and calibration."""
    from test_tessera_joint_aura import fixture

    from prismaquant.aura_cost import _aura_source_sha256
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.tessera_joint_aura import load_measured_anchor_input

    inputs, names, fmt, payload, _ = fixture(tmp_path)
    calib_path = _calibration(tmp_path)
    _, calibration = load_calibration_input(
        calib_path, expected_sha256=_sha(calib_path), n_samples=2, seqlen=4)
    payload["provenance"]["hessian"]["calibration_identity"] = {
        key: calibration["provenance"].get(key)
        for key in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed")}
    cost_path = Path(inputs["merged_cost"]["path"])
    cost_path.write_bytes(pickle.dumps(payload))
    inputs["merged_cost"] = _bind(cost_path)
    # The quantum admits only the qualified eager source attention.
    census_path = Path(inputs["census"]["path"])
    census = json.loads(census_path.read_text())
    census["attention_implementation"] = "eager"
    census_path.write_text(json.dumps(census))
    inputs["census"] = _bind(census_path)
    data = load_measured_anchor_input(
        inputs, require_existing_renders=True, verify_payloads=False,
        progress_phase=None)
    renders = {pair: cell["render"] for pair, cell in data.cells.items()}
    cache = ProductionWeightCache(
        weights=dict(renders), levers={},
        metadata={"inputs": dict(inputs), "verified_cells": {
            pair: {"render_file_sha256": _sha(path)} for pair, path in renders.items()}})
    cache.enable_lru(1 << 20)
    prepare = tmp_path / "run" / "prepare"
    prepare.mkdir(parents=True)
    pkl = prepare / "production.pkl"
    pkl.write_bytes(pickle.dumps(cache))
    completion = {
        "schema": "prismaquant.tessera_joint_aura.prepared.v3", "status": "complete",
        "plan_sha256": PLAN_SHA, "implementation_sha256": _aura_source_sha256(),
        "reader_identity": None, "projection_backend": PROJECTION,
        "formats_by_qname": {n: list(v) for n, v in data.formats_by_qname.items()},
        "calibration_input": calibration, "production_cache": _bind(pkl),
        "source_model_identity": {"fixture": True},
    }
    prepared_path = prepare / "prepared.json"
    prepared_path.write_text(json.dumps(completion, sort_keys=True))
    config = {"model": data.census["model"], "inputs": dict(inputs),
              "calibration_input": _bind(calib_path),
              "execution": {"n_calib_samples": 2, "calib_seqlen": 4},
              "max_render_bytes": 1 << 20, "max_gpu_bytes": 1 << 30,
              "stage_b_resource_policy": None, "served_activation_policy": None,
              "source_identity_cache": None, "historical_encoder_reuse": None}
    return {"config": config, "data": data, "cache": cache, "names": names,
            "completion": completion, "prepared": _bind(prepared_path),
            "output_root": str(tmp_path / "run")}


def _produce(campaign, *, layers=(0,)):
    from prismaquant.aura_cost import _aura_source_sha256
    from prismaquant.joint_layer_quanta import qname_layer

    slices = build_head_slices(
        config=campaign["config"], plan_sha256=PLAN_SHA,
        prepared=campaign["prepared"], completion=campaign["completion"],
        production_cache=campaign["cache"], data=campaign["data"],
        layers=layers, layer_of=qname_layer,
        implementation_sha256=_aura_source_sha256())
    bindings = {}
    for layer, head_slice in slices.items():
        path = Path(head_slice_path(campaign["output_root"], layer))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = head_slice_bytes(head_slice)
        path.write_bytes(raw)
        bindings[layer] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                           "bytes": len(raw), "schema": HEAD_SLICE_SCHEMA}
    return slices, bindings


def _record(layer, binding):
    return {"quantum_id": f"layer-{layer:03d}", "layer": layer,
            "executable_readset": {"head_slice": dict(binding)}}


def _consume(campaign, record, *, progress_phase=None):
    from prismaquant.aura_cost import _aura_source_sha256

    head_slice, _, files = read_quantum_head_slice(
        campaign["config"], record=record, prepared=campaign["prepared"],
        plan_sha256=PLAN_SHA)
    return load_quantum_head(
        campaign["config"], record=record, head_slice=head_slice, files=files,
        completion=read_prepared_head(files), plan_sha256=PLAN_SHA,
        implementation_sha256=_aura_source_sha256(), reader_identity=None,
        projection_backend=PROJECTION, progress_phase=progress_phase)


def test_round_trip_reproduces_the_walk_intake(tmp_path):
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    head = _consume(campaign, _record(0, bindings[0]))
    data = campaign["data"]
    assert head.formats_by_qname == dict(data.formats_by_qname)
    assert head.units == len(data.formats_by_qname)
    assert head.measured_cells == len(data.cells)
    assert head.progress_units == len(data.formats_by_qname)
    assert head.calibration == campaign["completion"]["calibration_input"]
    assert set(head.cache.weights) == set(data.cells)
    assert head.cache._expected_file_sha256 == {
        pair: row["render_file_sha256"]
        for pair, row in campaign["cache"].metadata["verified_cells"].items()}
    assert head.identity_cache_bytes is None
    intake = slices[0]["intake"]
    assert intake["layer_formats"] == {n: list(data.formats_by_qname[n])
                                       for n in campaign["names"]}
    assert [row["role"] for row in slices[0]["head_files"]] == [
        "prepared", "production_cache"]


def test_quantum_opens_only_declared_head_entries(tmp_path):
    """Path audit: nothing under the campaign input roots but declared entries."""
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    record = _record(0, bindings[0])
    declared = {str(Path(p).resolve()) for p in (
        campaign["config"]["calibration_input"]["path"], bindings[0]["path"],
        *[row["path"] for row in slices[0]["head_files"]])}
    roots = [str((tmp_path / name).resolve())
             for name in ("campaign", "merged", "run", "calibration")]
    opened, armed = [], [True]

    def _audit(event, args):
        # An audit hook cannot be removed; it goes quiet after this test.
        if armed[0] and event == "open" and args and \
                isinstance(args[0], (str, bytes, Path)):
            raw = args[0]
            opened.append(str(Path(raw.decode() if isinstance(raw, bytes)
                                   else raw).resolve()))

    sys.addaudithook(_audit)
    try:
        _consume(campaign, record)
    finally:
        armed[0] = False
    seen = list(opened)
    under_roots = {path for path in seen
                   if any(path == root or path.startswith(root + "/") for root in roots)}
    assert under_roots, "the audit hook observed no campaign read"
    assert under_roots <= declared, sorted(under_roots - declared)
    # The walk's inputs are never opened: no checkpoint, journal, cost or wire.
    assert not any("/merged/" in path or "/campaign/" in path for path in under_roots)


def test_refuses_another_layers_slice(tmp_path):
    campaign = _campaign(tmp_path)
    _, bindings = _produce(campaign)
    with pytest.raises(HeadSliceRefused, match="for layer 0, not layer 1"):
        _consume(campaign, _record(1, bindings[0]))


def test_refuses_a_slice_bound_to_another_preparation(tmp_path):
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    foreign = copy.deepcopy(slices[0])
    foreign["campaign"]["prepared_sha256"] = "e" * 64
    path = Path(bindings[0]["path"])
    raw = head_slice_bytes(foreign)
    path.write_bytes(raw)
    binding = dict(bindings[0], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    with pytest.raises(HeadSliceRefused, match="not the quantum's prepared"):
        _consume(campaign, _record(0, binding))


def test_refuses_a_slice_bound_to_another_plan(tmp_path):
    campaign = _campaign(tmp_path)
    _, bindings = _produce(campaign)
    with pytest.raises(HeadSliceRefused, match="not the quantum's plan"):
        read_quantum_head_slice(campaign["config"], record=_record(0, bindings[0]),
                                prepared=campaign["prepared"], plan_sha256="d" * 64)


def test_refuses_a_slice_produced_by_another_implementation(tmp_path, monkeypatch):
    """The producer re-derived the policies under its own package."""
    from prismaquant.aura_cost import _aura_source_sha256

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    foreign = copy.deepcopy(slices[0])
    foreign["campaign"]["producer_implementation_sha256"] = "e" * 64
    path = Path(bindings[0]["path"])
    raw = head_slice_bytes(foreign)
    path.write_bytes(raw)
    record = _record(0, dict(bindings[0], sha256=hashlib.sha256(raw).hexdigest(),
                             bytes=len(raw)))
    head_slice, _, files = read_quantum_head_slice(
        campaign["config"], record=record, prepared=campaign["prepared"],
        plan_sha256=PLAN_SHA)
    with pytest.raises(HeadSliceRefused, match="head slice producer implementation"):
        load_quantum_head(
            campaign["config"], record=record, head_slice=head_slice, files=files,
            completion=read_prepared_head(files), plan_sha256=PLAN_SHA,
            implementation_sha256=_aura_source_sha256(), reader_identity=None,
            projection_backend=PROJECTION)


def test_refuses_a_tampered_slice(tmp_path):
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    tampered = copy.deepcopy(slices[0])
    tampered["intake"]["max_render_file_bytes"] += 1
    Path(bindings[0]["path"]).write_bytes(head_slice_bytes(tampered))
    with pytest.raises(HeadSliceRefused, match="does not hash to its bound digest"):
        _consume(campaign, _record(0, bindings[0]))


def test_refuses_a_tampered_head_file(tmp_path):
    campaign = _campaign(tmp_path)
    _, bindings = _produce(campaign)
    pkl = Path(campaign["completion"]["production_cache"]["path"])
    pkl.write_bytes(pkl.read_bytes() + b"\0")
    with pytest.raises(HeadSliceRefused, match="production-cache .* does not hash"):
        _consume(campaign, _record(0, bindings[0]))


def test_producer_refuses_a_layer_without_units(tmp_path):
    campaign = _campaign(tmp_path)
    with pytest.raises(HeadSliceRefused, match="no unit for layer 7"):
        _produce(campaign, layers=(0, 7))


def test_head_reports_the_walks_cumulative_units_once(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _, bindings = _produce(campaign)
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "t")
    _consume(campaign, _record(0, bindings[0]), progress_phase="head")
    record = json.loads(progress.read_text())
    assert record["phase"] == "head"
    assert record["units_completed"] == len(campaign["data"].formats_by_qname)


# --------------------------------------------------------------------------
# The executable manifest and record: only the fields that name the slice
# --------------------------------------------------------------------------


def _head_binding(record, *, layer=2, prepared_sha256=None):
    from test_quantum_executable_readset import RENDER_PREREQ

    root = "/mnt/shared/run/layer-quanta/head-slices"
    return {"path": f"{root}/layer-{layer:03d}.json", "sha256": "a" * 64,
            "bytes": 900, "schema": HEAD_SLICE_SCHEMA,
            "head_files": [
                {"role": "prepared", "path": "/mnt/shared/run/prepare/prepared.json",
                 "sha256": prepared_sha256 or record["campaign"]["prepared_sha256"],
                 "bytes": 1000},
                {"role": "production_cache",
                 "path": "/mnt/shared/run/prepare/production.pkl",
                 "sha256": RENDER_PREREQ["production_pkl_sha256"], "bytes": 5000}]}


def _bound_pair(tmp_path):
    from test_quantum_executable_readset import (
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bound_inputs)
    from prismaquant.joint_layer_quanta import (
        bind_quantum_executable, build_quantum_executable_manifest,
        seal_manifest_bytes)

    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    plain = bind_quantum_executable(record, receipt, parent, **kwargs)
    head = _head_binding(record)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ), head_slice=head)
    sliced = bind_quantum_executable(
        record, receipt, parent, **{**kwargs, "manifest": manifest,
        "manifest_sha256": hashlib.sha256(seal_manifest_bytes(manifest)).hexdigest(),
        "head_slice": head})
    return plain, sliced, kwargs["manifest"], manifest, head


def test_record_differs_only_in_fields_that_name_the_slice(tmp_path):
    plain, sliced, _, _, head = _bound_pair(tmp_path)
    changed = {key for key in set(plain) | set(sliced) if plain.get(key) != sliced.get(key)}
    assert changed == {"executable_readset", "identity_sha256"}
    old, new = plain["executable_readset"], sliced["executable_readset"]
    moved = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    assert moved == {"head_slice", "manifest_sha256", "entry_count", "total_bytes",
                     "read_bytes"}
    assert new["head_slice"] == {key: head[key]
                                 for key in ("path", "sha256", "bytes", "schema")}
    assert new["entry_count"] == old["entry_count"] + 3
    added = head["bytes"] + sum(row["bytes"] for row in head["head_files"])
    assert new["total_bytes"] == old["total_bytes"] + added
    assert new["read_bytes"] == old["read_bytes"] + added
    assert new["phases"] == old["phases"]


def test_head_phase_declares_the_slice_and_its_files(tmp_path):
    _, _, plain_manifest, manifest, head = _bound_pair(tmp_path)
    phase = manifest["read_plan"]["phases"][0]
    assert phase["name"] == "head"
    paths = [manifest["entries"][i]["path"] for i in phase["entry_indices"]]
    assert paths[1:] == [head["path"], *[row["path"] for row in head["head_files"]]]
    assert manifest["annotations"]["head_slice"]["path"] == head["path"]
    assert "head_slice" not in plain_manifest["annotations"]
    # Every other phase reads the same bytes it did without the slice.
    for old, new in zip(plain_manifest["read_plan"]["phases"][1:],
                        manifest["read_plan"]["phases"][1:]):
        assert old["name"] == new["name"] and old["bytes"] == new["bytes"]
        assert [plain_manifest["entries"][i] for i in old["entry_indices"]] == \
            [manifest["entries"][i] for i in new["entry_indices"]]


@pytest.mark.parametrize("mutation, message", [
    (lambda head: head.update(path=head["path"].replace("layer-002", "layer-003")),
     "not layer 2's"),
    (lambda head: head["head_files"][0].update(sha256="e" * 64),
     "another prepared completion"),
    (lambda head: head["head_files"][1].update(sha256="e" * 64),
     "another production pickle"),
    (lambda head: head["head_files"].append(dict(head["head_files"][0])),
     "unknown or repeated"),
    (lambda head: head.update(schema="other"), "foreign schema"),
])
def test_manifest_refuses_a_foreign_head_binding(tmp_path, mutation, message):
    from test_quantum_executable_readset import (
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bound_inputs)
    from prismaquant.joint_layer_quanta import build_quantum_executable_manifest

    record, receipt, parent, _ = _bound_inputs(tmp_path)
    head = _head_binding(record)
    mutation(head)
    with pytest.raises(ValueError, match=message):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ), head_slice=head)


# --------------------------------------------------------------------------
# The producer tool
# --------------------------------------------------------------------------


def test_regenerate_builds_one_bound_slice_per_layer(tmp_path):
    from regenerate_joint_quanta import _build_head_slices

    campaign = _campaign(tmp_path)
    metadata_root = str(tmp_path / "meta")
    sealed = _build_head_slices(
        campaign["config"], plan_sha256=PLAN_SHA,
        prepared=campaign["completion"], prepared_binding=campaign["prepared"],
        production_cache=campaign["cache"], layers=[0],
        output_root=campaign["output_root"], metadata_root=metadata_root)
    assert set(sealed) == {0}
    binding = sealed[0]["binding"]
    assert binding["path"] == metadata_root + "/head-slices/layer-000.json"
    assert binding["sha256"] == hashlib.sha256(sealed[0]["bytes"]).hexdigest()
    assert binding["bytes"] == len(sealed[0]["bytes"])
    head_slice = json.loads(sealed[0]["bytes"])
    assert head_slice_bytes(head_slice) == sealed[0]["bytes"]
    assert binding["head_files"] == head_slice["head_files"]
    # The same slice the library producer cuts from the same intake.
    library, _ = _produce(campaign)
    assert head_slice["intake"] == library[0]["intake"]
    assert head_slice["campaign"] == library[0]["campaign"]


def test_regenerate_refuses_head_slices_without_executable_readsets(tmp_path, capsys):
    from regenerate_joint_quanta import main

    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    digest = _sha(plan)
    argv = ["--plan", str(plan), "--plan-sha256", digest,
            "--prepared", str(plan), "--prepared-sha256", digest,
            "--parent-manifest", str(plan), "--parent-manifest-sha256", digest,
            "--derivation", str(plan), "--head-slices"]
    assert main(argv) != 0
    assert "--head-slices needs --executable-readsets" in capsys.readouterr().err


def test_dev_mode_runs_a_quantum_whose_resource_policy_differs(tmp_path, monkeypatch, capsys):
    """PQ #1147: a policy re-declared from a measurement stamps by default.

    The slice and the preparation were sealed under one Stage B resource
    policy; the quantum's plan names another that differs only in
    ``workspace_reserve_bytes``. Dev mode prints both and runs under the
    plan's policy; certified mode refuses as before.
    """
    campaign = _campaign(tmp_path)
    slices, bindings = _produce(campaign)
    policies = {}
    for name, reserve in (("sealed", 1 << 30), ("measured", 3 << 29)):
        path = tmp_path / f"{name}-policy.json"
        path.write_text(json.dumps({"budget": {"workspace_reserve_bytes": reserve}}))
        policies[name] = {"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sealed = copy.deepcopy(slices[0])
    sealed["resource_policy"] = {"binding": policies["sealed"],
                                 "limits": {"gpu_bytes": 1 << 30}}
    path = Path(bindings[0]["path"])
    raw = head_slice_bytes(sealed)
    path.write_bytes(raw)
    binding = dict(bindings[0], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    campaign["config"]["stage_b_resource_policy"] = policies["measured"]

    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    head = _consume(campaign, _record(0, binding))
    out = capsys.readouterr().out
    assert "[DEV-MODE] seal Stage B head slice resource policy differs" in out
    assert "[DEV-MODE] seal prepared Stage B resource policy differs" in out
    assert head.cache._joint_stage_b_resource_policy == policies["measured"]

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(HeadSliceRefused, match="resource policy"):
        _consume(campaign, _record(0, binding))


class _EnvelopeReached(Exception):
    pass


def _policy_binding(tmp_path, name, reserve):
    path = tmp_path / f"{name}-policy.json"
    path.write_text(json.dumps({"budget": {"workspace_reserve_bytes": reserve}}))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _slice_sealed_under(campaign, policy, gpu_bytes):
    """A layer-0 head slice whose resource policy and device limit are these."""
    slices, bindings = _produce(campaign)
    sealed = copy.deepcopy(slices[0])
    sealed["resource_policy"] = {"binding": policy, "limits": {"gpu_bytes": gpu_bytes}}
    path = Path(bindings[0]["path"])
    raw = head_slice_bytes(sealed)
    path.write_bytes(raw)
    return dict(bindings[0], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _run_to_the_device_envelope(campaign, binding, monkeypatch):
    """Drive ``run_layer_quantum`` through the head slice to its device envelope.

    Returns the GPU ceiling the quantum applies. The envelope stops the run
    there; nothing after it is under test.
    """
    import prismaquant.gpu_guard as gpu_guard
    from prismaquant import memory_management, residency_map
    from prismaquant.joint_cost_quantum import run_layer_quantum

    applied = []

    def envelope(device, limit, **_kwargs):
        applied.append((device, limit))
        raise _EnvelopeReached()

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(memory_management, "enforce_device_envelope", envelope)
    monkeypatch.delenv(residency_map.ENV_VAR, raising=False)
    residency_map.reset_residency_resolver_for_tests()
    try:
        with pytest.raises(_EnvelopeReached):
            run_layer_quantum(
                campaign["config"], record=_record(0, binding), adjoint_slice={},
                plan_sha256=PLAN_SHA, prepared=campaign["prepared"],
                output_root=Path(campaign["output_root"]), data_manifest_sha256="d" * 64)
    finally:
        residency_map.reset_residency_resolver_for_tests()
    return applied


def test_dev_mode_applies_the_plan_device_limit_over_the_slice(tmp_path, monkeypatch, capsys):
    """PQ #1147 (v7): a slice sealed under a 1 GiB ceiling, a plan re-declared at 1.5 GiB.

    The quantum's plan names a measured policy with another device ceiling
    than the one the head slice was prepared under. Dev mode prints both
    and applies the plan's ceiling.
    """
    campaign = _campaign(tmp_path)
    sealed = _policy_binding(tmp_path, "sealed", 1 << 30)
    binding = _slice_sealed_under(campaign, sealed, 1 << 30)
    campaign["config"]["stage_b_resource_policy"] = _policy_binding(
        tmp_path, "measured", 3 << 29)
    campaign["config"]["max_gpu_bytes"] = 3 << 29

    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    applied = _run_to_the_device_envelope(campaign, binding, monkeypatch)
    out = capsys.readouterr().out
    assert "[DEV-MODE] seal Stage B head slice resource policy differs" in out
    assert "[DEV-MODE] seal Stage B device limit differs" in out
    assert applied == [("cuda", 3 << 29)]


def test_certified_mode_refuses_a_slice_device_limit_the_plan_does_not_name(
        tmp_path, monkeypatch):
    """The certified refusal at the device limit is unchanged (PQ #1147)."""
    campaign = _campaign(tmp_path)
    policy = _policy_binding(tmp_path, "sealed", 1 << 30)
    binding = _slice_sealed_under(campaign, policy, 1 << 30)
    campaign["config"]["stage_b_resource_policy"] = policy
    campaign["config"]["max_gpu_bytes"] = 3 << 29

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError) as refused:
        _run_to_the_device_envelope(campaign, binding, monkeypatch)
    assert str(refused.value) == "Stage B resources: device limit differs from policy"

    # At the sealed ceiling the same slice reaches the envelope.
    campaign["config"]["max_gpu_bytes"] = 1 << 30
    assert _run_to_the_device_envelope(campaign, binding, monkeypatch) == [
        ("cuda", 1 << 30)]
