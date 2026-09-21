"""Stage A durable artifact budget declaration + early preflight (#882).

The sealed plan's ``boundary_storage.max_artifact_bytes`` (416 GiB on the
GLM-5.3-Flash complete-512 panel) cannot hold Stage A's retained peak
(~584 GiB before headers/shared/manifest: 45 retained INPUT boundary
groups + one live cotangent plane + six strided checkpoints). The core
already owns an optional ``boundary_artifact_bytes`` override and a
reserve-before-write guard; the defect is that no invocation ever derived
or passed a budget, and no CLI forwarded one. These regressions exercise
the actual seam (resolve + estimate + preflight + core stamp + dispatcher
threading) on tiny CPU fixtures -- never a stand-in writer, never a whole
model.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest
import torch

PLAN_SEALED = 446676598784
RUN_640_GIB = 640 * 1024 ** 3


def _config(sealed: int = PLAN_SEALED) -> dict:
    return {"execution": {"boundary_storage": {
        "schema": "prismaquant.aura.boundary_storage.v2",
        "directory": "/tmp/stagea-budget-fixture",
        "capture_order": "layer_major",
        "max_artifact_bytes": sealed,
        "max_auxiliary_bytes": 1 << 21,
        "max_resident_bytes": 1 << 24,
        "prefetch_batches": 4,
    }}}


def test_resolve_defaults_to_plan_verbatim(tmp_path):
    from prismaquant.joint_cost_stage_a import resolve_artifact_budget_override

    resolved = resolve_artifact_budget_override(_config(), None, environ={})
    assert resolved["run_used"] == PLAN_SEALED
    assert resolved["override"] is None


def test_resolve_strict_positive_and_bool_rejection():
    from prismaquant.joint_cost_stage_a import (
        ARTIFACT_BUDGET_ENV, AdjointIdentityRefused,
        resolve_artifact_budget_override)

    for bad in (True, False, 0, -8, "0", "-4", "640.0", "1e9", "8 GiB",
                "", "  ", 640.0, None.__class__):
        with pytest.raises(AdjointIdentityRefused):
            resolve_artifact_budget_override(
                _config(), bad, environ={})
    with pytest.raises(AdjointIdentityRefused):
        resolve_artifact_budget_override(
            _config(), None, environ={ARTIFACT_BUDGET_ENV: "True"})
    with pytest.raises(AdjointIdentityRefused):
        resolve_artifact_budget_override(_config(), None, environ={
            ARTIFACT_BUDGET_ENV: "687194767360.0"})


def test_resolve_cli_env_sources_and_disagreement(tmp_path):
    from prismaquant.joint_cost_stage_a import (
        ARTIFACT_BUDGET_ENV, ARTIFACT_BUDGET_STAMP_SCHEMA,
        AdjointIdentityRefused, resolve_artifact_budget_override)

    cli = resolve_artifact_budget_override(
        _config(), RUN_640_GIB, environ={})
    assert cli["run_used"] == RUN_640_GIB
    assert cli["override"]["schema"] == ARTIFACT_BUDGET_STAMP_SCHEMA
    assert cli["override"]["source"] == "cli"
    assert cli["override"]["unit"] == "bytes"
    assert cli["override"]["plan_sealed_bytes"] == PLAN_SEALED
    assert cli["override"]["run_used_bytes"] == RUN_640_GIB

    env = resolve_artifact_budget_override(
        _config(), None,
        environ={ARTIFACT_BUDGET_ENV: str(RUN_640_GIB)})
    assert env["override"]["source"] == "env"
    assert env["run_used"] == RUN_640_GIB

    same = resolve_artifact_budget_override(
        _config(), str(RUN_640_GIB),
        environ={ARTIFACT_BUDGET_ENV: str(RUN_640_GIB)})
    assert same["override"]["source"] == "cli"

    with pytest.raises(AdjointIdentityRefused, match="disagrees"):
        resolve_artifact_budget_override(
            _config(), RUN_640_GIB,
            environ={ARTIFACT_BUDGET_ENV: str(RUN_640_GIB + 1)})


def test_estimate_matches_actual_stage_a_retention_math():
    """The production geometry's file counts, from the existing derivation.

    GLM-5.3-Flash panel: 45 layers, stride 8 -> {45,40,32,24,16,8};
    512 batches (probe_microbatch 1 over 512 rows); 4 probes; 16 MiB raw
    per tensor (1x512x4x4096 BF16 via hc_mult 4). 45 retained INPUT groups
    (46 written, tail retires the final boundary), one live cotangent plane,
    six checkpoint copies. The 416 GiB plan is below even the raw floor.
    """
    from prismaquant.joint_cost_stage_a import estimate_stage_a_artifact_demand

    demand = estimate_stage_a_artifact_demand(
        n_probes=4, n_batches=512, per_tensor_nbytes=16777216,
        num_layers=45, stride=8)
    assert demand["boundaries"] == [45, 40, 32, 24, 16, 8]
    assert demand["n_checkpoints"] == 6
    assert demand["n_retained_boundary_groups"] == 45
    assert demand["per_file_envelope_bytes"] == 16777216 + 65536
    # Hard floor: (23040 + 2048 + 12288) files x 16 MiB raw.
    assert demand["lower_bound_bytes"] == 37376 * 16777216
    assert demand["lower_bound_bytes"] == 627114830336
    assert PLAN_SEALED < demand["lower_bound_bytes"]
    # File envelopes alone (no shared/manifest) already exceed the plan.
    assert demand["conservative_bytes"] > PLAN_SEALED
    assert demand["conservative_bytes"] == (
        37376 * (16777216 + 65536) + (16777216 + 65536))
    # The proposed 640 GiB invocation covers the file-envelope peak with
    # headroom for shared/manifest/temp (bounded separately below).
    assert RUN_640_GIB > demand["conservative_bytes"]


def test_preflight_refuses_plan_budget_and_accepts_explicit_override():
    """Old too-small config fails early with required/declared/remedy; the
    adequate override proceeds with n_probes/seed/calibration/guard
    unchanged -- only the declared ceiling moves."""
    from prismaquant.joint_cost_stage_a import (
        AdjointIdentityRefused, estimate_stage_a_artifact_demand,
        preflight_stage_a_artifact_budget)

    demand = estimate_stage_a_artifact_demand(
        n_probes=4, n_batches=512, per_tensor_nbytes=16777216,
        num_layers=45, stride=8,
        shared_per_checkpoint_bytes=1 << 31,
        manifest_per_checkpoint_bytes=4 << 20)
    with pytest.raises(AdjointIdentityRefused) as excinfo:
        preflight_stage_a_artifact_budget(
            declared_bytes=PLAN_SEALED, demand=demand,
            plan_sealed_bytes=PLAN_SEALED)
    message = str(excinfo.value)
    assert str(demand["conservative_bytes"]) in message
    assert str(PLAN_SEALED) in message
    assert "--artifact-budget-bytes" in message
    assert "sealed plan is unchanged" in message

    ok = preflight_stage_a_artifact_budget(
        declared_bytes=int(demand["conservative_bytes"]), demand=demand,
        plan_sealed_bytes=PLAN_SEALED)
    assert ok["declared_bytes"] == demand["conservative_bytes"]
    # The demand inputs are the sealed invocation geometry, untouched.
    assert demand["n_probes"] == 4
    assert demand["n_batches"] == 512
    assert demand["boundaries"] == [45, 40, 32, 24, 16, 8]


def test_scaled_writer_late_failure_then_early_refusal_then_override_proceeds(
        tmp_path):
    """Scaled actual-writer exhibit on the real checkpoint path.

    A tiny owner + the real ``write_adjoint_checkpoint`` with tiny CPU
    tensors: a 416-equivalent too-small ceiling lets ordinary writes land
    and then refuses late at the checkpoint reservation (the runtime guard
    doing its job); the new preflight refuses the same geometry early
    before any file lands; the explicit adequate override then proceeds.
    No missing import is claimed RED: every name here exists.
    """
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, write_adjoint_checkpoint)
    from prismaquant.joint_cost_stage_a import (
        estimate_stage_a_artifact_demand, preflight_stage_a_artifact_budget)
    from test_streamed_boundary_artifacts import _policy
    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused

    def _owner(root, disk):
        owner = StreamedBoundaryArtifacts(
            _policy(root, cap=1 << 24, aux=1 << 22, disk=disk))
        owner.bind({"fixture": "stagea-budget-scaled"}, n_probes=1,
                   published=True)
        return owner

    def _plane():
        return {(0, 0): torch.zeros(32, 32, dtype=torch.float32)}

    def _shared():
        return {(0, 0): {"w": torch.zeros(4)}}

    # Scaled demand: 2 retained groups x 2 batches + 1 live + 1 checkpoint,
    # 4 KiB tensors. Conservative ~= 5 files x (4096+65536) + temp.
    demand = estimate_stage_a_artifact_demand(
        n_probes=1, n_batches=2, per_tensor_nbytes=4096,
        num_layers=2, stride=2)
    assert demand["boundaries"] == [2, 1]
    tiny_conservative = demand["conservative_bytes"]
    tiny_lower = demand["lower_bound_bytes"]
    assert tiny_lower == 5 * 4096

    # 1. Late failure: a ceiling between the lower floor and the envelope
    # lets the ordinary entry land, then refuses the checkpoint reservation.
    late_disk = tiny_lower + 70000
    assert tiny_lower < late_disk < tiny_conservative
    owner = _owner(tmp_path / "late", late_disk)
    space = adjoint_space(tmp_path / "late-out")
    with owner:
        owner.write(torch.zeros(32, 32), batch_index=0, boundary_index=0)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            write_adjoint_checkpoint(
                space, boundary=2,
                session={"generation": "ab" * 16,
                         "kind": "adjoint_checkpoint",
                         "run_identity_sha256": "cd" * 32},
                cotangents=_plane(), shared_adjoint=_shared(),
                shared_pass={0: {"tag": "a"}}, owner=owner)

    # 2. Early refusal: the same ceiling fails the preflight before any file.
    with pytest.raises(AdjointIdentityRefused, match="too small"):
        preflight_stage_a_artifact_budget(
            declared_bytes=late_disk, demand=demand,
            plan_sealed_bytes=late_disk)

    # 3. Adequate override: the conservative ceiling proceeds on the real
    # writer, same tensors, same guard.
    owner2 = _owner(tmp_path / "adequate", tiny_conservative)
    space2 = adjoint_space(tmp_path / "adequate-out")
    with owner2:
        owner2.write(torch.zeros(32, 32), batch_index=0, boundary_index=0)
        record = write_adjoint_checkpoint(
            space2, boundary=2,
            session={"generation": "ab" * 16,
                     "kind": "adjoint_checkpoint",
                     "run_identity_sha256": "cd" * 32},
            cotangents=_plane(), shared_adjoint=_shared(),
            shared_pass={0: {"tag": "a"}}, owner=owner2)
    assert record["boundary"] == 2
    assert record["cotangent_sha256"]
    preflight_stage_a_artifact_budget(
        declared_bytes=tiny_conservative, demand=demand,
        plan_sealed_bytes=late_disk)


def test_core_carries_override_stamp_and_ceiling(tmp_path, monkeypatch):
    """The scaled actual Stage A core threads the override into storage
    policy + receipt provenance without changing probes/seed/guard."""
    from test_layer_major_boundary_capture import fixture, draw
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from test_streamed_boundary_artifacts import _policy

    _, _, runner, _ = fixture()
    runner.context.settle_prefetch_layers = lambda layers: None
    execution = {
        "n_probes": 1,
        "seed_base": 7000,
        "probe_microbatch": 0,
        "boundary_storage": {
            **_policy(tmp_path / "core-boundaries", cap=1 << 24,
                      aux=1 << 22, disk=PLAN_SEALED),
            "schema": "prismaquant.aura.boundary_storage.v2",
            "capture_order": "layer_major",
        },
    }
    stamp = {"schema": "prismaquant.joint_adjoint_capture.artifact_budget.v1",
             "source": "cli", "unit": "bytes",
             "plan_sealed_bytes": PLAN_SEALED,
             "run_used_bytes": PLAN_SEALED + 1024}
    receipt = run_adjoint_capture_core(
        runner, draw(), execution=execution,
        output_root=tmp_path / "core-out", stride=8,
        source_model_identity={"identity": "stub"},
        unit_roster_sha256="a" * 64, plan_sha256="d" * 64,
        prepared_sha256="e" * 64, read_manifest_sha256="f" * 64,
        implementation_sha256="b" * 64,
        boundary_artifact_bytes=PLAN_SEALED + 1024,
        artifact_budget_stamp=stamp)
    assert receipt["artifact_budget_override"] == stamp
    assert (receipt["boundary_storage"]["policy"]["max_artifact_bytes"]
            == PLAN_SEALED + 1024)
    assert receipt["run_identity"]["n_probes"] == 1
    assert receipt["run_identity"]["seed_base"] == 7000


def test_core_rejects_nonpositive_artifact_bytes(tmp_path, monkeypatch):
    from test_layer_major_boundary_capture import fixture, draw
    from prismaquant.joint_cost_stage_a import (
        AdjointIdentityRefused, run_adjoint_capture_core)
    from test_streamed_boundary_artifacts import _policy

    _, _, runner, _ = fixture()
    runner.context.settle_prefetch_layers = lambda layers: None
    execution = {
        "n_probes": 1,
        "seed_base": 7000,
        "probe_microbatch": 0,
        "boundary_storage": {
            **_policy(tmp_path / "core-bad", cap=1 << 24,
                      aux=1 << 22, disk=PLAN_SEALED),
            "schema": "prismaquant.aura.boundary_storage.v2",
            "capture_order": "layer_major",
        },
    }
    for bad in (True, 0, -1):
        with pytest.raises(AdjointIdentityRefused):
            run_adjoint_capture_core(
                runner, draw(), execution=execution,
                output_root=tmp_path / f"core-bad-{bad}", stride=8,
                source_model_identity={"identity": "stub"},
                unit_roster_sha256="a" * 64, plan_sha256="d" * 64,
                prepared_sha256="e" * 64, read_manifest_sha256="f" * 64,
                implementation_sha256="b" * 64,
                boundary_artifact_bytes=bad)


def test_stage_a_cli_threads_artifact_flag(tmp_path, monkeypatch):
    """main() hands --artifact-budget-bytes to the capture call; dropping
    the kwarg would revert to the plan ceiling with no stamp."""
    import prismaquant.joint_cost_stage_a as stage_a
    import prismaquant.tessera_joint_aura as aura

    captured = {}

    def fake_capture(config, **kwargs):
        captured.update(kwargs)
        return {"command": "adjoint-capture", "passed": True,
                "stride": {"value": 8, "source": "plan"}, "checkpoints": []}

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(stage_a, "require_dev_mode", lambda *a, **k: None)
    monkeypatch.setattr(aura, "_load_plan", lambda *a, **k: {})
    monkeypatch.setattr(stage_a, "run_adjoint_capture", fake_capture)

    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    out = tmp_path / "out"
    assert stage_a.main(["--plan", str(plan), "--plan-sha256", "d" * 64,
                         "--prepared", str(prepared),
                         "--prepared-sha256", "e" * 64,
                         "--output-root", str(out),
                         "--artifact-budget-bytes", str(RUN_640_GIB)]) == 0
    assert captured["artifact_budget_bytes"] == str(RUN_640_GIB)

    captured.clear()
    assert stage_a.main(["--plan", str(plan), "--plan-sha256", "d" * 64,
                         "--prepared", str(prepared),
                         "--prepared-sha256", "e" * 64,
                         "--output-root", str(out)]) == 0
    assert captured["artifact_budget_bytes"] is None


def test_dispatcher_threads_artifact_flag(tmp_path):
    """The dispatcher forwards the NEW field as a payload flag (the PB
    channel), validates strict bytes, and leaves argv unchanged when
    absent -- mirroring the prefetch-override threading."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from dispatch_joint_quanta import DispatchRefused, stage_a_argv
    import gzip

    manifest = tmp_path / "adjoint.data-manifest.json.gz"
    manifest.write_bytes(gzip.compress(json.dumps({
        "schema": "prismaquant.data_manifest.v1",
        "entries": [],
    }).encode()))
    campaign = {"plan_path": "plan.json", "plan_sha256": "d" * 64,
                "prepared_path": "prepared.json",
                "prepared_sha256": "e" * 64}

    plain = stage_a_argv(manifest, campaign)
    assert "--artifact-budget-bytes" not in plain

    threaded = stage_a_argv(manifest, campaign,
                            artifact_budget_bytes=RUN_640_GIB)
    assert "--artifact-budget-bytes" in threaded
    assert str(RUN_640_GIB) in threaded

    for bad in (True, 0, -5, "8 GiB"):
        with pytest.raises(DispatchRefused):
            stage_a_argv(manifest, campaign, artifact_budget_bytes=bad)
