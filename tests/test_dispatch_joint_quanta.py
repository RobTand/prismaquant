"""Dispatcher gates for the distributed joint-AURA cost campaign (§5 of
``docs/design/distributed_campaign_2026-09-19.md``).

RED-first: these tests fixture the producer (§3) record shapes and the
stage-A receipt (§3.3) per the contract's schemas and pin the dispatcher's
behavior before ``tools/dispatch_joint_quanta.py`` exists. The dispatcher
is a receipt-driven submitter, never a scheduler: no test here lets it
choose a box, read capacity, or steer at runtime. Real dispatch is the
coordinator's call; these tests use fixtures and a fake submission
gateway, and ``--dry-run`` submits nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prismaquant.cost_stage_checkpoint import canonical_json_sha256

from dispatch_joint_quanta import (  # noqa: E402
    ADJOINT_SCHEMA,
    CHUNK_PROGRESS_GRACE_S,
    CONSUMER_TAGS,
    SUBMISSION_PRIORITY,
    FakeGateway,
    main,
    quantum_argv,
    stage_a_argv,
)

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
N_LAYERS = 3


@pytest.fixture(autouse=True)
def _portable_spec(tmp_path, monkeypatch):
    """A mount-independent spec: the default SPEC_PATH lives on Rob's shared
    mount, which CI runners cannot see.  Every argv built in these tests
    inlines the spec content, so the content -- not the path -- is the
    contract under test."""
    spec = tmp_path / "spec-hostcap32-ram-dev.json"
    spec.write_text(json.dumps(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    import dispatch_joint_quanta
    monkeypatch.setattr(dispatch_joint_quanta, "SPEC_PATH", spec)


@pytest.fixture
def campaign(tmp_path):
    scope = {"campaign": "dispatch-fixture", "layers": list(range(N_LAYERS))}
    # The plan file must exist: stage A's payload --output-root and the
    # receipt default are both read from the plan's sealed output_root.
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(
        {"output_root": str(tmp_path / "campaign-root")}))
    return {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
            "manifest_sha256": "c" * 64, "scope": scope,
            "plan_path": str(plan_path),
            "prepared_path": "/fixture/prepare/prepared.json",
            "roster_sha256": hashlib.sha256(b"roster\n").hexdigest()}


def _record(campaign, layer, receipts_root="adjoint-receipt.json"):
    quantum_id = f"layer-{layer:03d}"
    size = 1024 * (layer + 1)
    record = {
        "schema": RECORD_SCHEMA, "quantum_id": quantum_id, "layer": layer,
        "campaign": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "plan_path": campaign["plan_path"],
            "prepared_path": campaign["prepared_path"],
            "read_manifest_sha256": campaign["manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": campaign["roster_sha256"],
        },
        "read_set": {
            "manifest_path": f"manifests/{quantum_id}.data-manifest.json.gz",
            "manifest_sha256": hashlib.sha256(quantum_id.encode()).hexdigest(),
            "source_phase": {"name": f"layer-{layer}",
                             "start_bytes": 0, "end_bytes": size},
        },
        "chunks": [{"name": f"{quantum_id}-chunk-000", "start_bytes": 0,
                    "end_bytes": size}],
        "windows": [{"window_index": 0, "names": []}],
        "adjoint": {"checkpoint_boundary": N_LAYERS, "chain_layers": [],
                    "receipt_sha256": None},
        "output_space": {"root": f"layer-quanta/{quantum_id}"},
    }
    record["identity_sha256"] = canonical_json_sha256(
        record, where="fixture layer-quantum record")
    return record


@pytest.fixture
def records_dir(tmp_path, campaign):
    directory = tmp_path / "records"
    directory.mkdir()
    for layer in range(N_LAYERS):
        record = _record(campaign, layer)
        (directory / f"layer-{layer:03d}.json").write_text(json.dumps(record))
    return directory


def _receipt(campaign, *, plan_sha256=None):
    return {"schema": ADJOINT_SCHEMA,
            "plan_sha256": campaign["plan_sha256"] if plan_sha256 is None else plan_sha256,
            "prepared_sha256": campaign["prepared_sha256"],
            "checkpoints": [{"boundary": N_LAYERS, "cotangent_sha256": "9" * 64}],
            "generation": "gen-fixture"}


def _write_receipt(path, receipt):
    path.write_text(json.dumps(receipt))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp_receipt(records_dir, digest):
    for record_path in sorted(records_dir.glob("layer-*.json")):
        record = json.loads(record_path.read_text())
        record["adjoint"]["receipt_sha256"] = digest
        record_path.write_text(json.dumps(record))


def _argv(records_dir, output_root, receipt=None, extra=()):
    argv = ["--records", str(records_dir), "--output-root", str(output_root)]
    if receipt is not None:
        argv += ["--adjoint-receipt", str(receipt)]
    return argv + list(extra)


# -- the exact pbrun argv shape (§5.2, pinned: drift breaks placement) ----------


def test_quantum_argv_matches_the_pinned_submission_shape(tmp_path, campaign):
    """Every quantum submits exactly the contract's argv: the shared GB10
    class tag (PB requires *every* tag a row lists, so a host pair would
    admit neither Spark), the slice manifest, stage residency, per-chunk
    progress phases, dev-mode env, detached. PB owns placement past that."""
    record = _record(campaign, 1)
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    argv = quantum_argv(record, record_path=record_path,
                        output_root=Path("/out/root"))
    tags = [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "--tag"]
    assert tags == ["gb10"]
    assert CONSUMER_TAGS == ("gb10",)
    # The regression, at the argv layer: naming both Sparks is a conjunction
    # PB can satisfy on neither box.
    assert not {"sparky", "sparklina"} <= set(tags)
    manifest = argv[argv.index("--data-manifest") + 1]
    assert manifest.endswith("manifests/layer-001.data-manifest.json.gz")
    assert argv[argv.index("--residency") + 1] == "stage"
    assert argv[argv.index("--residency-ram") + 1] == "auto"
    phases = [argv[i + 1] for i, word in enumerate(argv[:-1])
              if word == "--progress-phase"]
    assert phases[0].startswith("head=")
    assert f"layer-001-chunk-000={CHUNK_PROGRESS_GRACE_S}" in phases
    assert CHUNK_PROGRESS_GRACE_S == 900
    assert argv[argv.index("--priority") + 1] == str(SUBMISSION_PRIORITY)
    # The GPU envelope: the capture and the quanta are GPU-or-bust
    # (require_cuda_hot_path refused the c94602e9d63c run whose rows
    # demanded no device).
    assert argv[argv.index("--demand") + 1] == "gpu=1,mem_gb=104"
    assert argv[argv.index("--env") + 1] == "PRISMAQUANT_DEV_MODE=1"
    assert "--detach" in argv
    assert "--" in argv
    tail = argv[argv.index("--") + 1:]
    # The payload runs inside the qualified campaign container: the
    # projection backend's runtime identity (and the workload's own
    # torch/CUDA) qualify one image, and a bare python3 -m refuses as
    # unidentified (the 31bab41cc812 failure).
    assert tail[:3] == ["python3", "-m", "tools.tessera_campaign_container"]
    spec = json.loads(tail[tail.index("--spec") + 1])
    assert spec["container"]["image"].startswith("sha256:")
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    assert inner[inner.index("--quantum") + 1] == str(record_path)
    assert inner[inner.index("--quantum-sha256") + 1] == record["identity_sha256"]
    assert inner[inner.index("--output-root") + 1] == "/out/root"


def test_plan_consumer_tags_override_reaches_every_quantum_row(
        tmp_path, campaign, records_dir):
    """§5.1: the plan block is the declared placement policy, and every row
    carries it. Before the fix the block was read for the dry-run print only
    while the rows kept the module default; the override is now the argv.

    The two tags are a conjunction (PB matches every one), so this test also
    pins that the plan's list is not silently reordered or truncated."""
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, digest)
    gateway = FakeGateway(terminal=True)
    gateway.mark_terminal("stage-a-action-key")
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    (out / "layer-quanta" / "campaign-state.json").write_text(
        json.dumps({"event": "stage-a-submitted",
                    "action_key": "stage-a-action-key"}) + "\n")
    plan_path = tmp_path / "plan-with-block.json"
    plan_path.write_text(json.dumps(
        {"output_root": str(tmp_path / "campaign-root"),
         "distributed_campaign": {"consumer_tags": ["gb10", "progress-v1"]}}))
    assert main(_argv(records_dir, out, receipt_path,
                      ("--plan", str(plan_path))), _gateway=gateway) == 0
    quantum_rows = [row for row in gateway.submitted
                    if row["kind"] == "quantum"]
    assert len(quantum_rows) == N_LAYERS
    for row in quantum_rows:
        argv = row["argv"]
        tags = [argv[i + 1] for i, word in enumerate(argv[:-1])
                if word == "--tag"]
        assert tags == ["gb10", "progress-v1"]


@pytest.mark.parametrize("blocked_tags", [[], "gb10", ["gb10", 7], ["gb10", ""]])
def test_ill_typed_plan_consumer_tags_refuse_before_publishing(
        tmp_path, campaign, records_dir, blocked_tags):
    """The placement policy is a non-empty list of tag strings or the
    module default. An empty list would publish unconstrained rows, a bare
    string is not a policy, and a non-string entry is a typo: all refuse at
    dispatch time (exit 3) with nothing submitted."""
    plan_path = tmp_path / "bad-plan.json"
    plan_path.write_text(json.dumps(
        {"output_root": str(tmp_path / "campaign-root"),
         "distributed_campaign": {"consumer_tags": blocked_tags}}))
    gateway = FakeGateway()
    out = tmp_path / "out"
    assert main(_argv(records_dir, out, None,
                      ("--plan", str(plan_path))), _gateway=gateway) == 3
    assert gateway.submitted == []


def test_stage_a_argv_prefetch_override_is_payload_flagged(tmp_path, campaign):
    """The #819 seam's dispatcher half: without an override the stage-A argv
    is unchanged (the plan's sealed budget, no flag); with one, the payload
    carries ``--prefetch-override`` -- the channel that crosses the container
    boundary, since the launcher forwards no ambient action environment into
    the payload -- and nothing else in the argv moves."""
    manifest = tmp_path / "adjoint.data-manifest.json.gz"
    plain = stage_a_argv(manifest, campaign)
    assert "--prefetch-override" not in plain
    assert plain[-1] == "--resume"

    override = Path("/mnt/shared/joint-panel/prefetch-override-v10.json")
    widened = stage_a_argv(manifest, campaign, prefetch_override=override)
    # The pbrun envelope (everything before the payload ``--``) is unchanged;
    # the override rides inside the container payload only.
    split = widened.index("--")
    assert widened[:split] == plain[:plain.index("--")]
    payload = widened[split + 1:]
    inner = payload[payload.index("--", payload.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_adjoint_capture"]
    assert inner[inner.index("--prefetch-override"):][:2] == [
        "--prefetch-override", str(override)]
    assert inner[inner.index("--prefetch-override") + 1] == str(override)


def test_main_threads_stage_a_prefetch_override(tmp_path, campaign, records_dir,
                                                capsys):
    """``--stage-a-prefetch-override`` reaches the published stage-A row and
    is recorded in the campaign state's submission event (the deviation is
    the dispatcher's provenance too, not only the run's)."""
    override = tmp_path / "prefetch-override-v10.json"
    override.write_text("{}")
    gateway = FakeGateway()
    out = tmp_path / "out"
    code = main(["--records", str(records_dir), "--output-root", str(out),
                 "--state", str(tmp_path / "state.json"),
                 "--stage-a-prefetch-override", str(override)],
                _gateway=gateway)
    assert code == 0
    stage_a_rows = [row for row in gateway.submitted if row["kind"] == "stage-a"]
    assert len(stage_a_rows) == 1
    argv = stage_a_rows[0]["argv"]
    assert argv[argv.index("--prefetch-override") + 1] == str(override)
    events = [json.loads(line) for line
              in (tmp_path / "state.json").read_text().splitlines() if line]
    assert events[0]["event"] == "stage-a-submitted"
    assert events[0]["prefetch_override"] == str(override)


def test_publication_order_is_descending_layer_id(tmp_path, campaign, records_dir):
    """Deterministic order, fronts the cutover gate: high layers first."""
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, digest)
    gateway = FakeGateway(terminal=True)
    out = tmp_path / "out"
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    published = [call["quantum_id"] for call in gateway.submitted
                 if call["kind"] == "quantum"]
    assert published == ["layer-002", "layer-001", "layer-000"]


# -- stage-A-receipt gating (no quanta before the receipt) ----------------------


def test_no_quanta_before_the_stage_a_receipt(tmp_path, campaign, records_dir):
    """Stage A first: quanta publish only once the receipt lands and
    validates. Without it the tool submits stage A and stops."""
    gateway = FakeGateway(terminal=False)
    out = tmp_path / "out"
    assert main(_argv(records_dir, out), _gateway=gateway) == 0
    kinds = [call["kind"] for call in gateway.submitted]
    assert kinds == ["stage-a"]
    assert gateway.submitted[0]["argv"][gateway.submitted[0]["argv"].index("--tag") + 1] == "sparky"


def test_stale_stage_a_receipt_refuses(tmp_path, campaign, records_dir):
    """A receipt from another plan revision fails closed, exit 3, and
    publishes nothing."""
    gateway = FakeGateway(terminal=True)
    gateway.mark_terminal("stage-a-action-key")
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign, plan_sha256="f" * 64))
    _stamp_receipt(records_dir, digest)
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    state = out / "layer-quanta" / "campaign-state.json"
    state.write_text(json.dumps({"event": "stage-a-submitted",
                                 "action_key": "stage-a-action-key"}) + "\n")
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 3
    assert gateway.submitted == []
    assert not (out / "layer-quanta" / "layer-002").exists()


def test_quanta_publish_once_the_receipt_lands(tmp_path, campaign, records_dir):
    gateway = FakeGateway(terminal=True)
    gateway.mark_terminal("stage-a-action-key")
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, digest)
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    (out / "layer-quanta" / "campaign-state.json").write_text(
        json.dumps({"event": "stage-a-submitted",
                    "action_key": "stage-a-action-key"}) + "\n")
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    kinds = [call["kind"] for call in gateway.submitted]
    assert kinds == ["quantum", "quantum", "quantum"]


# -- dry-run: plan with digests, submits nothing ---------------------------------


def test_dry_run_prints_plan_with_digests_and_submits_nothing(
        tmp_path, campaign, records_dir, capsys):
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, digest)
    gateway = FakeGateway(terminal=True)
    out = tmp_path / "out"
    assert main(_argv(records_dir, out, receipt_path, ("--dry-run",)),
                _gateway=gateway) == 0
    assert gateway.submitted == []
    plan = capsys.readouterr().out
    for layer in range(N_LAYERS):
        assert f"layer-{layer:03d}" in plan
    assert digest in plan
    assert "PRISMAQUANT_DEV_MODE=1" in plan


# -- idempotency: re-running publishes nothing already terminal -----------------


def test_rerun_publishes_nothing_already_terminal(tmp_path, campaign, records_dir):
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, digest)
    gateway = FakeGateway(terminal=True)
    gateway.mark_terminal("stage-a-action-key")
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    (out / "layer-quanta" / "campaign-state.json").write_text(
        json.dumps({"event": "stage-a-submitted",
                    "action_key": "stage-a-action-key"}) + "\n")
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    first = list(gateway.submitted)
    assert len(first) == N_LAYERS
    for call in first:
        gateway.mark_terminal(call["action_key"])
    gateway.submitted.clear()
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    assert gateway.submitted == []
