"""Stage A takes its head from the prepared completion (PQ #1051, part of #997).

These tests drive ``run_adjoint_capture``, the action's entry point, over the
Tessera campaign fixture: a real plan ``inputs`` block, census, anchor
catalog and prepared completion, and a calibration file holding the chain
tests' draw. Every device seam is stubbed; the capture itself is the real
``run_adjoint_capture_core`` over the five-layer dense fixture model at
stride 2. The claims:

* With ``load_measured_anchor_input`` made to raise, a fresh root, a fresh
  root bound to a forward-recovery capsule, and a seed all complete, and a
  fresh run opens nothing under the walk's input roots.
* A fresh root run with and without the walk seals the same roster digest
  and the same checkpoint bytes; the receipts differ only in ``head.walked``.
* A completion under another digest, a completion prepared under another
  plan (dev mode included), and a capsule whose campaign record changed all
  refuse.
* The adjoint data manifest's head declares none of the walk's reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_receipt_path,
    adjoint_space,
    checkpoint_cotangent_plane,
    checkpoint_directory,
    checkpoint_entry_session,
    read_exact_entry_tensors,
)
from prismaquant.stage_a_chain_seed import (
    SEED_SPEC_SCHEMA,
    seed_receipt_path,
    tensor_payload_sha256,
)
from prismaquant.stage_a_head import STAGE_A_HEAD_SCHEMA

from test_joint_cost_quantum_runtime import _boundary_policy, _execution
from test_layer_major_boundary_capture import draw
from test_stage_a_chain_resume import CAP, ONE, THREE, TWO, _dense_runner, _interrupted, \
    _without
from test_streamed_cost_checkpoints import _model_identity

PLAN = "b" * 64
PROJECTION = {"schema": "fixture.projection_backend", "name": "fixture"}
IMAGE = "sha256:" + "9" * 64
GENERATION = 1001
#: A seed binds its own generation, never its source run's.
SEED_GENERATION = 2002


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
    """The chain tests' draw as a pinned calibration file."""
    from safetensors.torch import save_file

    ids = draw().contiguous()
    provenance = {"source": "fixture", "fit_tokens": ids.numel(), "nsamples": ids.shape[0],
                  "seqlen": ids.shape[1], "seed": 11, "text_sha256": "1" * 64,
                  "fit_ids_sha256": hashlib.sha256(
                      ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    path = root / "calibration" / "tokens.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    return path


def _campaign(tmp_path, *, implementation):
    """The Tessera fixture campaign, prepared under ``PLAN`` by ``implementation``."""
    from test_tessera_joint_aura import fixture

    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.tessera_joint_aura import load_measured_anchor_input

    inputs, names, _, payload, _ = fixture(tmp_path)
    calib_path = _calibration(tmp_path)
    _, calibration = load_calibration_input(
        calib_path, expected_sha256=_sha(calib_path), n_samples=5, seqlen=4)
    payload["provenance"]["hessian"]["calibration_identity"] = {
        key: calibration["provenance"].get(key)
        for key in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed")}
    cost_path = Path(inputs["merged_cost"]["path"])
    cost_path.write_bytes(pickle.dumps(payload))
    inputs["merged_cost"] = _bind(cost_path)
    census_path = Path(inputs["census"]["path"])
    census = json.loads(census_path.read_text())
    census["attention_implementation"] = "eager"
    census_path.write_text(json.dumps(census))
    inputs["census"] = _bind(census_path)
    # The prepare's own walk, over the same inputs.
    data = load_measured_anchor_input(
        inputs, require_existing_renders=True, verify_payloads=False,
        progress_phase=None)
    prepare = tmp_path / "prepare"
    prepare.mkdir()
    completion = {
        "schema": "prismaquant.tessera_joint_aura.prepared.v3", "status": "complete",
        "plan_sha256": PLAN, "implementation_sha256": implementation,
        "reader_identity": None, "projection_backend": PROJECTION,
        "source_model_identity": _model_identity("joint-source"),
        "calibration_input": calibration,
        "formats_by_qname": {n: list(v) for n, v in data.formats_by_qname.items()},
        "measured_cells": len(data.cells), "encoder_source_reuse": None,
    }
    prepared_path = prepare / "prepared.json"
    prepared_path.write_text(json.dumps(completion, sort_keys=True))
    return SimpleNamespace(inputs=dict(inputs), names=sorted(names), data=data,
                           model=census["model"], calibration=_bind(calib_path),
                           completion=completion, prepared=_bind(prepared_path))


def _config(campaign, root, *, plan_root=None):
    """The plan: ``plan_root`` is its ``output_root`` (a seed's source run),
    and the boundary storage is the run's own."""
    execution = _execution(root)
    policy = _boundary_policy(root / "boundaries")
    policy["max_resident_bytes"] = CAP
    execution.update(boundary_storage=policy, n_calib_samples=5, calib_seqlen=4)
    return {
        "model": campaign.model, "inputs": dict(campaign.inputs),
        "calibration_input": dict(campaign.calibration), "execution": execution,
        "output_root": str(plan_root or root), "max_gpu_bytes": 1 << 40,
        "canonical_capture": None, "source_capture_compatibility": None,
        "source_identity_cache": None, "historical_encoder_reuse": None,
        "source_prefetch": {"max_cache_slots": 2, "prefetch_workers": 1,
                            "prefetch_lookahead": 1, "cache_headroom_gb": 1.0,
                            "prefetch_min_available_gb": 1.0,
                            "require_prefetched_residency": True},
    }


class _WalkCalled(AssertionError):
    """The head walk ran."""


class _Sampler:
    def start(self):
        return self

    def stop(self):
        return {}


def _entry(monkeypatch, *, implementation, walk, generation):
    """Stub every device seam of the entry point; the capture core stays real."""
    from prismaquant import (
        aura_cost,
        autoscale,
        cost_streaming,
        glm_capture_compatibility,
        gpu_guard,
        joint_projection_backend,
        matmul_arithmetic,
        model_profiles,
        tessera_joint_aura,
    )
    from prismaquant.joint_cost_stage_a import KernelTimeProfiler

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(matmul_arithmetic, "pin_matmul_arithmetic", lambda *a, **k: None)
    monkeypatch.setattr(autoscale, "require_bounded_capture_environment",
                        lambda *a, **k: None)
    monkeypatch.setenv(tessera_joint_aura.ACTIVATION_SCALE_ENV, "0")
    monkeypatch.delenv("PRISMAQUANT_MAX_GPU_MEM_GB", raising=False)
    monkeypatch.setattr(stage_a, "GpuPowerSampler", _Sampler)
    monkeypatch.setattr(stage_a, "_stage_a_kernel_profiler",
                        lambda: KernelTimeProfiler(not_measured="not measured: fixture"))
    monkeypatch.setattr(joint_projection_backend, "executing_image", lambda: IMAGE)
    monkeypatch.setattr(joint_projection_backend, "prewarm_projection_backend",
                        lambda *a, **k: SimpleNamespace(identity=PROJECTION))
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: implementation)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda *a, **k: None)
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm",
                        lambda *a, **k: _dense_runner())
    monkeypatch.setattr(cost_streaming, "build_streamed_model_identity",
                        lambda *a, **k: _model_identity("joint-source"))
    monkeypatch.setattr(glm_capture_compatibility, "require_capture_compatibility",
                        lambda *a, **k: None)
    monkeypatch.setattr(stage_a, "_run_artifact_preflight",
                        lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(stage_a, "bind_stage_a_produced_output", lambda **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 0)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=generation))
    if not walk:
        def walked(*_args, **_kwargs):
            raise _WalkCalled("the head walk ran")
        monkeypatch.setattr(tessera_joint_aura, "load_measured_anchor_input", walked)


def _capture(campaign, root, monkeypatch, *, implementation=ONE, walk=False,
             plan_sha256=PLAN, prepared=None, config_root=None, generation=GENERATION,
             **kw):
    """One Stage A action through ``run_adjoint_capture``; returns its result.

    The generation id is pinned, so two runs into one path write equal bytes.
    """
    with monkeypatch.context() as patch:
        _entry(patch, implementation=implementation, walk=walk, generation=generation)
        return stage_a.run_adjoint_capture(
            _config(campaign, root, plan_root=config_root), plan_sha256=plan_sha256,
            prepared=prepared or campaign.prepared, output_root=root, stride=2,
            head_walk=walk, **kw)


def _receipt(root):
    return json.loads(adjoint_receipt_path(adjoint_space(root)).read_text())


def _plane(record):
    """``{(probe, batch): payload sha256}`` of a sealed checkpoint's cotangents."""
    plane = {}
    for key, row in checkpoint_cotangent_plane(record).items():
        tensors = read_exact_entry_tensors(
            [row], expected_session=checkpoint_entry_session(record))
        plane[key] = tensor_payload_sha256(tensors.pop(row["name"]))
    return plane


def _checkpoint_files(root):
    """Each sealed checkpoint's own files and its plane's files, by digest.

    A referenced checkpoint (PQ #1036) keeps its plane in the owner's
    entries, so those files are hashed too.
    """
    space = adjoint_space(root).resolve()
    base = space / "checkpoints"
    files = {str(path.relative_to(space)): _sha(path)
             for path in sorted(base.rglob("*")) if path.is_file()}
    for record_path in sorted(base.glob("boundary-*/checkpoint.json")):
        record = json.loads(record_path.read_text())
        for row in checkpoint_cotangent_plane(record).values():
            path = Path(row["path"]).resolve()
            files[str(path.relative_to(space))] = _sha(path)
    return files


def _newline_roster(names):
    return hashlib.sha256("".join(f"{name}\n" for name in sorted(names)).encode()).hexdigest()


# -- the skip completes, and opens nothing the walk read ----------------------

def test_a_fresh_root_completes_without_the_walk(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    roots = [str((tmp_path / name).resolve()) for name in ("campaign", "merged")]
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
        result = _capture(campaign, tmp_path / "run", monkeypatch)
    finally:
        armed[0] = False
    assert result["passed"] is True
    walk_reads = [path for path in opened
                  if any(path == root or path.startswith(root + "/") for root in roots)]
    assert walk_reads == []
    assert any(path.endswith("prepared.json") for path in opened)

    receipt = _receipt(tmp_path / "run")
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]
    assert receipt["run_identity"]["unit_roster_sha256"] == _newline_roster(campaign.names)
    assert receipt["head"] == {
        "schema": STAGE_A_HEAD_SCHEMA, "walked": False,
        "prepared_completion": campaign.prepared,
        "units": len(campaign.names), "measured_cells": len(campaign.data.cells)}
    assert result["head"] == receipt["head"]
    assert (result["units"], result["measured_cells"]) == (
        len(campaign.names), len(campaign.data.cells))
    assert "head" not in receipt["run_identity"]


def test_the_skip_reports_the_roster_once_under_the_head_phase(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "t")
    seen = []
    from prismaquant import tessera_joint_aura
    commit = tessera_joint_aura._pb_commit

    def recorded(units, phase, unit=None):
        seen.append((units, phase))
        return commit(units, phase, unit=unit)

    monkeypatch.setattr(tessera_joint_aura, "_pb_commit", recorded)
    _capture(campaign, tmp_path / "run", monkeypatch)
    assert seen[0] == (len(campaign.names), "head")


# -- with and without the walk: one roster, one set of checkpoint bytes -------

def _shape(value):
    """Every key and list length of ``value``, without its scalars."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return None


def test_the_walk_and_the_skip_seal_the_same_checkpoints(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    root = tmp_path / "run"
    skipped = _capture(campaign, root, monkeypatch)
    skipped_receipt, skipped_files = _receipt(root), _checkpoint_files(root)
    skipped_planes = {c["boundary"]: _plane(c) for c in skipped_receipt["checkpoints"]}
    root.rename(tmp_path / "run-skipped")

    walked = _capture(campaign, root, monkeypatch, walk=True)
    walked_receipt = _receipt(root)
    assert skipped["passed"] is walked["passed"] is True
    assert (adjoint_space(root) / "head-walk").is_dir()

    # The roster digest and every sealed plane are the same bytes.
    assert (walked_receipt["run_identity"]["unit_roster_sha256"]
            == skipped_receipt["run_identity"]["unit_roster_sha256"]
            == _newline_roster(campaign.names))
    assert _checkpoint_files(root) == skipped_files
    assert {c["boundary"]: _plane(c) for c in walked_receipt["checkpoints"]} == skipped_planes

    # The receipts differ only in the head record's walked flag. The two
    # telemetry blocks hold wall clocks, so only their shape is compared; the
    # dev-mode stamp names one tree for both arms and is compared whole.
    assert walked_receipt["head"] == {**skipped_receipt["head"], "walked": True}
    clocks = ("telemetry", "retention.telemetry")
    assert _shape(walked_receipt["telemetry"]) == _shape(skipped_receipt["telemetry"])
    assert _shape(walked_receipt["retention"].get("telemetry")) == \
        _shape(skipped_receipt["retention"].get("telemetry"))
    assert _without(walked_receipt, *clocks, "head.walked") == \
        _without(skipped_receipt, *clocks, "head.walked")
    assert walked_receipt["head"]["walked"] is True
    assert skipped_receipt["head"]["walked"] is False


def test_the_walk_arm_refuses_a_completion_the_walk_disagrees_with(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    completion = dict(campaign.completion, measured_cells=campaign.completion[
        "measured_cells"] + 1)
    path = Path(campaign.prepared["path"])
    path.write_text(json.dumps(completion, sort_keys=True))
    with pytest.raises(ValueError, match="prepared measured cells"):
        _capture(campaign, tmp_path / "run", monkeypatch, walk=True, prepared=_bind(path))


# -- the refusals the walk used to sit in front of ----------------------------

def test_a_completion_under_another_digest_refuses(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    with pytest.raises(ValueError, match="checksum changed"):
        _capture(campaign, tmp_path / "run", monkeypatch,
                 prepared={**campaign.prepared, "sha256": "0" * 64})
    assert not adjoint_receipt_path(adjoint_space(tmp_path / "run")).exists()


@pytest.mark.parametrize("dev", [False, True], ids=["certified", "dev"])
def test_a_completion_prepared_under_another_plan_refuses(tmp_path, monkeypatch, dev):
    campaign = _campaign(tmp_path, implementation=ONE)
    if dev:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="prepared plan"):
        _capture(campaign, tmp_path / "run", monkeypatch, plan_sha256="9" * 64)
    assert not (adjoint_space(tmp_path / "run") / "checkpoints").exists()


def test_a_completion_with_another_calibration_refuses(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=ONE)
    completion = json.loads(Path(campaign.prepared["path"]).read_text())
    completion["calibration_input"]["provenance"]["seed"] = 12
    path = Path(campaign.prepared["path"])
    path.write_text(json.dumps(completion, sort_keys=True))
    with pytest.raises(ValueError, match="prepared calibration"):
        _capture(campaign, tmp_path / "run", monkeypatch, prepared=_bind(path))


# -- a published capsule: fresh recovery root and seed ------------------------

def _published(campaign, tmp_path):
    """The campaign a capsule publishes for this roster, and its sealed record."""
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.joint_layer_quanta import (
        LAYER_QUANTUM_SCHEMA, canonical_sha256, roster_digest)

    names = campaign.names
    scope = {"kind": "complete_campaign", "source_unit_count": len(names),
             "source_roster_sha256": canonical_json_sha256(names, where="fixture"),
             "window_count": 5, "campaign_window_count": 5, "calib_seqlen": 4}
    identity = {"plan_sha256": PLAN, "prepared_sha256": campaign.prepared["sha256"],
                "read_manifest_sha256": "0" * 64,
                "unit_roster_sha256": roster_digest(names), "campaign_scope": scope}
    record = {"schema": LAYER_QUANTUM_SCHEMA, "quantum_id": "layer-000",
              "campaign": {**identity, "prepared_path": campaign.prepared["path"]},
              "adjoint": {}}
    record["identity_sha256"] = canonical_sha256(record, where="fixture")
    path = tmp_path / "campaign-record.json"
    path.write_text(json.dumps(record))
    return identity, _bind(path)


def _capsule(base, monkeypatch, *, frontier, identity, record):
    """R9's forward pass, contained after ``frontier``, frozen into a capsule
    that publishes ``identity`` and names its campaign ``record``."""
    from prismaquant import joint_forward_resume as recovery_mod
    from test_joint_forward_resume import _bound, _chain_sdk, _owner_spool

    monkeypatch.setattr(recovery_mod, "_sdk", _chain_sdk)
    base.mkdir(parents=True, exist_ok=True)
    refs, identities = {}, []
    last = len(draw()) - 1
    _interrupted(base / "r9", monkeypatch, generation=9, forward_refs=refs,
                 identities=identities,
                 interrupt=lambda kw: kw.get("probe_index") is None and (
                     kw["boundary_index"], kw["batch_index"]) == (frontier, last))
    spool, instance, template = _owner_spool(base, "e" * 64, refs)
    ref = refs[0, 0]
    capsule = base / "r9-capsule.json"
    recovery_mod.build_forward_recovery(specification={
        "schema": recovery_mod.SCHEMA, "queue_root": str(base / "queue"),
        "instance": instance, "template": template,
        "session": json.loads(ref.metadata_json)["identity"]["session"],
        "original_bind_identity": identities[-1],
        "campaign_identity": dict(identity),
        "published_campaign_identity": dict(identity),
        "source_campaign_record": dict(record),
        "implementation_compatibility": {"original": ONE, "recovery": TWO,
                                         "scope": "forward-identical-memory-only"},
        "n_batches": len(draw()), "entry_shape": list(ref.shape), "entry_dtype": ref.dtype},
        spool_directory=spool, output=capsule, frontier=frontier)
    return _bound(capsule)


def _source(tmp_path, monkeypatch):
    """Source run A through the entry: fresh root, capsule-bound, implementation 2."""
    campaign = _campaign(tmp_path, implementation=TWO)
    identity, record = _published(campaign, tmp_path)
    capsule = _capsule(tmp_path / "capsule", monkeypatch, frontier=3,
                       identity=identity, record=record)
    root = tmp_path / "A"
    result = _capture(campaign, root, monkeypatch, implementation=TWO,
                      forward_recovery=capsule)
    return SimpleNamespace(campaign=campaign, identity=identity, record=record,
                           capsule=capsule, root=root, result=result)


def test_a_capsule_bound_fresh_root_takes_its_campaign_from_the_completion(
        tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    assert source.result["passed"] is True
    receipt = _receipt(source.root)
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]
    run = receipt["run_identity"]
    assert run["unit_roster_sha256"] == source.identity["unit_roster_sha256"]
    assert run["campaign_scope"] == source.identity["campaign_scope"]
    assert receipt["head"]["walked"] is False


def test_a_capsule_whose_campaign_record_changed_refuses(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, implementation=TWO)
    identity, record = _published(campaign, tmp_path)
    capsule = _capsule(tmp_path / "capsule", monkeypatch, frontier=3,
                       identity=identity, record=record)
    Path(record["path"]).write_text("{}")
    with pytest.raises(ValueError, match="forward recovery source campaign record"):
        _capture(campaign, tmp_path / "A", monkeypatch, implementation=TWO,
                 forward_recovery=capsule)
    assert not (adjoint_space(tmp_path / "A") / "checkpoints").exists()


def _pin(path):
    return {"path": str(path), "sha256": _sha(path)}


def _manifest(root, boundary):
    return checkpoint_directory(adjoint_space(root), boundary) / "checkpoint.json"


def test_a_seed_completes_without_the_walk(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    spec = {"schema": SEED_SPEC_SCHEMA, "checkpoint": _pin(_manifest(source.root, 4)),
            "capsule": dict(source.capsule), "through": 2,
            "implementation_compatibility": {"from": TWO, "to": THREE},
            "compare": _pin(_manifest(source.root, 2))}
    scratch = tmp_path / "seed"
    result = _capture(source.campaign, scratch, monkeypatch, implementation=THREE,
                      config_root=source.root, generation=SEED_GENERATION,
                      chain_seed=spec)
    assert result["passed"] is True
    assert result["plane_comparison"]["bitwise_equal"] is True
    receipt = json.loads(seed_receipt_path(adjoint_space(scratch)).read_text())
    assert receipt["head"]["walked"] is False
    assert receipt["head"]["prepared_completion"] == source.campaign.prepared
    assert (receipt["run_identity"]["unit_roster_sha256"]
            == source.identity["unit_roster_sha256"])
    assert not adjoint_receipt_path(adjoint_space(scratch)).exists()


# -- the data manifest's head declares none of the walk's reads ---------------

def test_the_adjoint_manifest_head_drops_the_walk_reads(tmp_path):
    from prismaquant.joint_layer_quanta import build_adjoint_manifest
    from prismaquant.tessera_joint_aura import merged_checkpoint_parts

    campaign = _campaign(tmp_path, implementation=ONE)
    inputs = campaign.inputs
    parts = sorted(path for path in merged_checkpoint_parts(
        inputs["merged_checkpoint"]["path"]).rglob("*") if path.is_file())
    assert parts
    walk = [inputs[key]["path"] for key in (
        "campaign_plan", "census", "campaign_receipts", "merged_cost",
        "merged_checkpoint")] + [str(path) for path in parts]
    kept = [campaign.prepared["path"], campaign.calibration["path"]]
    head = [{"path": path, "offset": 0, "bytes": 10, "sha256": None}
            for path in [kept[0], *walk, kept[1]]]
    model = "/mnt/shared/model"
    layers = [{"path": f"{model}/layer{layer}.pt", "offset": 0, "bytes": 100,
               "sha256": None} for layer in range(2)]
    cumulative, phases = 0, []
    for name, size in (("head", 10 * len(head)), ("layer-0", 100), ("layer-1", 100)):
        cumulative += size
        phases.append({"name": name, "bytes": size, "cumulative_bytes": cumulative})
    parent = {"schema": "prismaquant.prismabuild.data_manifest.v1",
              "produced_by": {"tool": "fixture"}, "mount_prefix": "/mnt/shared",
              "entries": head + layers, "entry_count": len(head) + 2,
              "total_bytes": cumulative,
              "annotations": {"campaign_scope": {"campaign": "fixture"},
                              "phases": phases}}
    kwargs = dict(plan_path="/fixture/plan.json", plan_sha256=PLAN,
                  prepared_path=campaign.prepared["path"],
                  prepared_sha256=campaign.prepared["sha256"],
                  parent_manifest_sha256="c" * 64, output_root="/fixture/out")
    manifest = build_adjoint_manifest({"model": model, "inputs": inputs}, parent, **kwargs)
    first = manifest["read_plan"]["phases"][0]
    assert first["name"] == "head"
    assert [manifest["entries"][i]["path"] for i in first["entry_indices"]] == kept
    assert first["bytes"] == 20
    assert manifest["annotations"]["head_walk_reads_dropped"] == {
        "entries": len(walk), "bytes": 10 * len(walk)}
    assert not set(walk) & {entry["path"] for entry in manifest["entries"]}

    # A plan with no inputs block keeps the head verbatim, as before #1051.
    verbatim = build_adjoint_manifest({"model": model}, parent, **kwargs)
    first = verbatim["read_plan"]["phases"][0]
    assert [verbatim["entries"][i]["path"] for i in first["entry_indices"]] == \
        [entry["path"] for entry in head]
    assert "head_walk_reads_dropped" not in verbatim["annotations"]


def test_the_walk_read_set_names_only_the_inputs_a_plan_binds(tmp_path):
    # A catalog extension's plan binds other inputs than the walk's: the
    # read set names what it binds and nothing else, and no parts directory
    # without a merged checkpoint.
    from prismaquant.tessera_joint_aura import head_walk_read_set, is_head_walk_read
    census, overlay = str(tmp_path / "census.json"), str(tmp_path / "overlay.json")
    read_set = head_walk_read_set({"census": {"path": census, "sha256": "a" * 64},
                                   "candidate_overlay": {"path": overlay, "sha256": "b" * 64}})
    assert read_set == (frozenset({os.path.normpath(census)}), ())
    assert is_head_walk_read(census, read_set)
    assert not is_head_walk_read(overlay, read_set)
