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


@pytest.fixture(autouse=True)
def _release_process_state_this_file_installs():
    """Give back the process-global state the CLI entrypoints install (#889).

    ``joint_cost_stage_a.main`` activates the staged-tier policy from its
    sealed ``--allowed-tiers`` flag, always and process-globally: that is
    the production contract, and a campaign run owns the process it ends.
    A test that calls ``main`` in-process does not, so the policy it turned
    on stayed on for every file pytest ran afterwards in the same process,
    and every bulk read there refused with no residency map to serve from.
    Two files were seen failing behind it, in the order pbtest happened to
    group them.

    The same applies to the ``tools/`` entry this file puts on ``sys.path``
    and the module it imports from there: both outlive the test that added
    them. Teardown only -- a setup-time reset would hide the leak this
    file's last test asserts against.
    """

    import sys

    yield
    from prismaquant.staged_tier_policy import (
        deactivate_staged_tier_policy_for_tests)
    deactivate_staged_tier_policy_for_tests()
    sys.modules.pop("dispatch_joint_quanta", None)


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

    GLM-5.3-Flash panel (plan cited in the design doc): 45 layers, stride
    8 -> {45,40,32,24,16,8}; 512 full batches (probe_microbatch 1 over 512
    rows, no remainder); 4 probes; 16 MiB raw per tensor (1 row x 512
    tokens (plan calib_seqlen) x hc_mult 4 streams x 4096 BF16). 45
    retained INPUT groups (46 written, tail retires the final boundary),
    one live cotangent plane set (4 probe planes x 512 batches), six
    checkpoint copies. The 416 GiB plan is below even the raw floor.
    """
    from prismaquant.joint_cost_stage_a import estimate_stage_a_artifact_demand

    demand = estimate_stage_a_artifact_demand(
        n_probes=4, n_full_batches=512, remainder_rows=0,
        per_full_tensor_nbytes=16777216,
        num_layers=45, stride=8)
    assert demand["boundaries"] == [45, 40, 32, 24, 16, 8]
    assert demand["n_checkpoints"] == 6
    assert demand["n_retained_boundary_groups"] == 45
    assert demand["n_batches_total"] == 512
    assert demand["per_full_file_envelope_bytes"] == 16777216 + 65536
    # Hard floor: (23040 + 2048 + 12288) files x 16 MiB raw = 584 GiB exact.
    assert demand["lower_bound_bytes"] == 37376 * 16777216
    assert demand["lower_bound_bytes"] == 627065225216
    assert PLAN_SEALED < demand["lower_bound_bytes"]
    # File envelopes alone (no shared/manifest) already exceed the plan.
    assert demand["planning_estimate_bytes"] > PLAN_SEALED
    assert demand["planning_estimate_bytes"] == (
        37376 * (16777216 + 65536) + (16777216 + 65536))

    # With the stated planning allowances (2 GiB shared per checkpoint from
    # the plan auxiliary bound -- an assumption, not a proven ceiling -- and
    # 4 MiB manifest per checkpoint), the 640 GiB proposal still covers the
    # planning estimate with ~44 GiB headroom.
    allowed = estimate_stage_a_artifact_demand(
        n_probes=4, n_full_batches=512, remainder_rows=0,
        per_full_tensor_nbytes=16777216,
        num_layers=45, stride=8,
        shared_per_checkpoint_bytes=1 << 31,
        manifest_per_checkpoint_bytes=4 << 20)
    assert allowed["planning_estimate_bytes"] == 642441609216
    assert RUN_640_GIB > allowed["planning_estimate_bytes"]


def test_estimate_counts_remainder_at_true_size_not_ceiling():
    """An uneven last batch is smaller than a full one: the hard floor sums
    full groups plus the true remainder geometry, strictly below the naive
    ``ceil`` full-size estimate (which is an UPPER estimate, not a floor).

    Uses the actual tiny fixture runner: the remainder tensor is derived
    through the runner's own profile expansion on ``meta`` tensors, the
    same path the run preflight uses -- not a hand constant.
    """
    from test_layer_major_boundary_capture import fixture
    from prismaquant.joint_cost_stage_a import (
        _stage_a_per_tensor_nbytes, estimate_stage_a_artifact_demand)

    _, _, runner, _ = fixture()
    seqlen = 4
    per_full = _stage_a_per_tensor_nbytes(runner, batch_rows=2, seqlen=seqlen)
    per_rem = _stage_a_per_tensor_nbytes(runner, batch_rows=1, seqlen=seqlen)
    assert per_rem < per_full
    assert per_rem * 2 == per_full  # row-linear hidden geometry
    n_layers = int(runner.num_layers)
    demand = estimate_stage_a_artifact_demand(
        n_probes=1, n_full_batches=2, remainder_rows=1,
        per_full_tensor_nbytes=per_full,
        per_remainder_tensor_nbytes=per_rem,
        num_layers=n_layers, stride=2)
    assert demand["n_batches_total"] == 3
    unit_sets = n_layers + 1 + len(demand["boundaries"]) * 1
    assert demand["lower_bound_bytes"] == unit_sets * (2 * per_full + per_rem)
    naive_ceil = unit_sets * 3 * per_full
    assert demand["lower_bound_bytes"] < naive_ceil


def test_preflight_refuses_plan_budget_and_accepts_explicit_override():
    """Old too-small config fails early on the hard floor with
    required/declared/remedy; the adequate override proceeds with
    n_probes/seed/calibration/guard unchanged -- only the declared
    ceiling moves."""
    from prismaquant.joint_cost_stage_a import (
        AdjointIdentityRefused, estimate_stage_a_artifact_demand,
        preflight_stage_a_artifact_budget)

    demand = estimate_stage_a_artifact_demand(
        n_probes=4, n_full_batches=512, remainder_rows=0,
        per_full_tensor_nbytes=16777216,
        num_layers=45, stride=8,
        shared_per_checkpoint_bytes=1 << 31,
        manifest_per_checkpoint_bytes=4 << 20)
    with pytest.raises(AdjointIdentityRefused) as excinfo:
        preflight_stage_a_artifact_budget(
            declared_bytes=PLAN_SEALED, demand=demand,
            plan_sealed_bytes=PLAN_SEALED)
    message = str(excinfo.value)
    assert str(demand["lower_bound_bytes"]) in message
    assert str(demand["planning_estimate_bytes"]) in message
    assert str(PLAN_SEALED) in message
    assert "--artifact-budget-bytes" in message
    assert "sealed plan is unchanged" in message

    ok = preflight_stage_a_artifact_budget(
        declared_bytes=int(demand["lower_bound_bytes"]), demand=demand,
        plan_sealed_bytes=PLAN_SEALED)
    assert ok["declared_bytes"] == demand["lower_bound_bytes"]
    assert ok["required_floor_bytes"] == demand["lower_bound_bytes"]
    assert ok["planning_estimate_bytes"] == demand["planning_estimate_bytes"]
    # The demand inputs are the sealed invocation geometry, untouched.
    assert demand["n_probes"] == 4
    assert demand["n_batches_total"] == 512
    assert demand["boundaries"] == [45, 40, 32, 24, 16, 8]


def test_scaled_writer_floor_refusal_mid_guard_and_adequate_override(
        tmp_path):
    """Scaled actual-writer exhibit on the real checkpoint path.

    A tiny owner + the real ``write_adjoint_checkpoint`` with tiny CPU
    tensors, in three bands: below the hard floor the new preflight
    refuses early before any file lands; between the floor and the
    planning estimate the preflight passes and the runtime guard still
    refuses late at the checkpoint reservation (the guard stays
    authoritative for serialized bytes); at the planning estimate the
    explicit adequate override proceeds on the same writer, same guard.
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

    def _session():
        return {"generation": "ab" * 16, "kind": "adjoint_checkpoint",
                "run_identity_sha256": "cd" * 32}

    # Scaled demand: 3 retained groups x 2 full batches + 1 live plane set
    # (1 probe plane x 2 batches) + 2 checkpoint copies, 4 KiB tensors.
    demand = estimate_stage_a_artifact_demand(
        n_probes=1, n_full_batches=2, remainder_rows=0,
        per_full_tensor_nbytes=4096,
        num_layers=3, stride=2)
    assert demand["boundaries"] == [3, 2]
    tiny_floor = demand["lower_bound_bytes"]
    tiny_planning = demand["planning_estimate_bytes"]
    assert tiny_floor == 12 * 4096
    assert tiny_planning == 12 * (4096 + 65536) + (4096 + 65536)
    assert tiny_floor < tiny_planning

    # 1. Below the floor: early refusal, and no checkpoint file lands.
    below = tiny_floor - 1
    with pytest.raises(AdjointIdentityRefused, match="hard geometry floor"):
        preflight_stage_a_artifact_budget(
            declared_bytes=below, demand=demand, plan_sealed_bytes=below)
    assert not list(Path(adjoint_space(tmp_path / "below-out")).rglob("*"))

    # 2. Between floor and planning estimate: the preflight passes (the
    # floor is the mandatory gate) and the runtime guard refuses late.
    mid_disk = tiny_floor + 70000
    assert tiny_floor <= mid_disk < tiny_planning
    preflight_stage_a_artifact_budget(
        declared_bytes=mid_disk, demand=demand, plan_sealed_bytes=mid_disk)
    owner = _owner(tmp_path / "late", mid_disk)
    space = adjoint_space(tmp_path / "late-out")
    with owner:
        owner.write(torch.zeros(32, 32), batch_index=0, boundary_index=0)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            write_adjoint_checkpoint(
                space, boundary=2, session=_session(),
                cotangents=_plane(), shared_adjoint=_shared(),
                shared_pass={0: {"tag": "a"}}, owner=owner)

    # 3. Adequate override: the planning estimate proceeds on the real
    # writer, same tensors, same guard.
    owner2 = _owner(tmp_path / "adequate", tiny_planning)
    space2 = adjoint_space(tmp_path / "adequate-out")
    with owner2:
        owner2.write(torch.zeros(32, 32), batch_index=0, boundary_index=0)
        record = write_adjoint_checkpoint(
            space2, boundary=2, session=_session(),
            cotangents=_plane(), shared_adjoint=_shared(),
            shared_pass={0: {"tag": "a"}}, owner=owner2)
    assert record["boundary"] == 2
    assert record["cotangent_sha256"]
    preflight_stage_a_artifact_budget(
        declared_bytes=tiny_planning, demand=demand,
        plan_sealed_bytes=mid_disk)


def test_core_carries_override_stamp_and_ceiling(tmp_path, monkeypatch):
    """The scaled actual Stage A core threads the override into storage
    policy + receipt provenance without changing probes/seed/guard."""
    from test_layer_major_boundary_capture import fixture, draw
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from test_streamed_boundary_artifacts import _policy

    _, _, runner, _ = fixture()
    runner.context.settle_prefetch_layers = lambda layers: None
    from test_streamed_cost_checkpoints import _model_identity
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
        source_model_identity=_model_identity("joint-source"),
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
    from test_streamed_cost_checkpoints import _model_identity
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
                source_model_identity=_model_identity("joint-source"),
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


def _dispatcher_argv_fixture(tmp_path, monkeypatch):
    """Minimal valid manifest + campaign for stage_a_argv budget tests."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from dispatch_joint_quanta import stage_a_argv  # noqa: F401
    import dispatch_joint_quanta
    import gzip

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    monkeypatch.setattr(dispatch_joint_quanta, "SPEC_PATH", spec)

    parent = "d" * 64
    manifest_doc = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "mount_prefix": "/mnt/shared",
        "entries": [],
        "entry_count": 0,
        "total_bytes": 0,
        "annotations": {
            "parent_manifest_sha256": parent,
            "plan_sha256": "d" * 64,
            "prepared_sha256": "e" * 64,
            "phases": [{"name": "head", "bytes": 0, "cumulative_bytes": 0}],
        },
    }
    manifest = tmp_path / "adjoint.data-manifest.json.gz"
    manifest.write_bytes(gzip.compress(json.dumps(manifest_doc).encode()))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(
        {"output_root": str(tmp_path / "campaign-root")}))
    campaign = {"plan_path": str(plan_path), "plan_sha256": "d" * 64,
                "prepared_path": "prepared.json",
                "prepared_sha256": "e" * 64,
                "read_manifest_sha256": parent}
    return manifest, campaign


def test_dispatcher_threads_artifact_flag(tmp_path, monkeypatch):
    """The dispatcher forwards the NEW field as a payload flag (the PB
    channel), validates strict bytes, and leaves argv unchanged when
    absent -- mirroring the prefetch-override threading."""
    # `pytest.ini` puts the repo root on the path, not `tools/`, and this
    # import runs BEFORE the fixture helper below that would have added
    # it -- so it only ever resolved when something else had already run.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from dispatch_joint_quanta import DispatchRefused, stage_a_argv

    manifest, campaign = _dispatcher_argv_fixture(tmp_path, monkeypatch)

    plain = stage_a_argv(manifest, campaign)
    assert "--artifact-budget-bytes" not in plain

    threaded = stage_a_argv(manifest, campaign,
                            artifact_budget_bytes=RUN_640_GIB)
    assert "--artifact-budget-bytes" in threaded
    assert str(RUN_640_GIB) in threaded

    for bad in (True, 0, -5, "8 GiB"):
        with pytest.raises(DispatchRefused):
            stage_a_argv(manifest, campaign, artifact_budget_bytes=bad)


def test_dispatcher_refuses_non_integer_numerics(tmp_path, monkeypatch):
    """The dispatcher boundary permits only int (not bool) or ASCII-decimal
    strings: a float like 640.9 must refuse, never truncate to 640 the way
    a bare int() coercion would -- the capture resolver refuses floats, so
    the dispatcher must not launder one."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from decimal import Decimal
    from dispatch_joint_quanta import DispatchRefused, stage_a_argv

    manifest, campaign = _dispatcher_argv_fixture(tmp_path, monkeypatch)

    for bad in (687194767360.9, 640.0, Decimal("640"), "640.9", "0x10"):
        with pytest.raises(DispatchRefused):
            stage_a_argv(manifest, campaign, artifact_budget_bytes=bad)

    threaded = stage_a_argv(manifest, campaign,
                            artifact_budget_bytes=f" {RUN_640_GIB} ")
    assert threaded[threaded.index("--artifact-budget-bytes") + 1] == str(
        RUN_640_GIB)


def test_parse_refuses_unicode_digits_named():
    """`str.isdigit` accepts non-ASCII digits (superscripts, fullwidth)
    that `int()` then refuses with a raw ValueError. The parser requires
    ASCII decimal up front so every invalid input carries the named
    AdjointIdentityRefused, never an untyped error."""
    from prismaquant.joint_cost_stage_a import (
        AdjointIdentityRefused, _parse_artifact_budget_bytes)

    for bad in ("²", "⁶⁴⁰", "640²"):
        assert bad.isdigit()
        with pytest.raises(AdjointIdentityRefused):
            _parse_artifact_budget_bytes(bad, where="test")
    # Not even isdigit, still a named refusal (never untyped).
    with pytest.raises(AdjointIdentityRefused):
        _parse_artifact_budget_bytes("½", where="test")


def test_this_file_leaves_no_process_global_state_behind():
    """FAILING-BEFORE (PQ #889): the leak, asserted where it is made.

    Deliberately the last test in the file and deliberately order
    dependent: pytest runs a module's tests in file order, so by the time
    this runs, the two ``joint_cost_stage_a.main`` calls above have already
    activated the staged-tier policy process-wide. Before the teardown
    fixture at the top of this file, this assertion failed here instead of
    failing -- as it did on main -- in whichever unrelated file pbtest
    happened to group next in the same process.

    ``sys.path`` is covered by ``monkeypatch.syspath_prepend`` at the three
    sites that add ``tools/``; this checks the two pieces monkeypatch
    cannot see.
    """

    import sys
    from prismaquant.staged_tier_policy import active_policy

    assert active_policy() is None, (
        "the staged-tier policy an in-process CLI run installed is still "
        "active: every bulk read in the next file refuses", active_policy())
    assert "dispatch_joint_quanta" not in sys.modules
