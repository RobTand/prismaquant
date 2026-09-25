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
from prismaquant.joint_adjoint_slices import stage_a_slice, write_adjoint_slice
from prismaquant.joint_layer_quanta import adjoint_binding_fields

from test_stage_b_band_binding import band_from_receipt, synthetic_receipt
from stage_a_spool_spec import (SPOOL_ENV, SPOOL_MOUNT, SPOOL_ROOT, SPOOL_WINDOW_BYTES,
                                 STAGE_A_SPOOL_ENV, stage_a_plan, with_spool)

from dispatch_joint_quanta import (  # noqa: E402
    ADJOINT_SCHEMA,
    CHUNK_PROGRESS_GRACE_S,
    CONSUMER_TAGS,
    HEAD_PROGRESS_GRACE_S,
    SUBMISSION_PRIORITY,
    DispatchRefused,
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
    spec.write_text(json.dumps(with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}})))
    import dispatch_joint_quanta
    monkeypatch.setattr(dispatch_joint_quanta, "SPEC_PATH", spec)


@pytest.fixture
def campaign(tmp_path):
    scope = {"campaign": "dispatch-fixture", "layers": list(range(N_LAYERS))}
    # The plan file must exist: stage A's payload --output-root and the
    # receipt default are both read from the plan's sealed output_root.
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(stage_a_plan(
        tmp_path, output_root=str(tmp_path / "campaign-root"))))
    return {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
            "manifest_sha256": "c" * 64, "scope": scope,
            "read_manifest_sha256": "d" * 64,
            "plan_path": str(plan_path),
            "prepared_path": "/fixture/prepare/prepared.json",
            "roster_sha256": hashlib.sha256(b"roster\n").hexdigest()}


def _record(campaign, layer, receipts_root="adjoint-receipt.json",
            slice_dir=None):
    quantum_id = f"layer-{layer:03d}"
    size = 1024 * (layer + 1)
    if slice_dir is None:
        manifest_path = f"manifests/{quantum_id}.data-manifest.json.gz"
        manifest_sha256 = hashlib.sha256(quantum_id.encode()).hexdigest()
    else:
        # A real slice manifest the dispatcher reads back: the sealed
        # digest covers these exact wire bytes, so a tampered slice is a
        # dispatch-time refusal, not a launched row with an inert tier.
        manifest_path = str(Path(slice_dir) / f"{quantum_id}.data-manifest.json")
        Path(manifest_path).write_bytes(
            json.dumps({"slice": quantum_id}).encode("utf-8"))
        manifest_sha256 = hashlib.sha256(
            Path(manifest_path).read_bytes()).hexdigest()
    record = {
        "schema": RECORD_SCHEMA, "quantum_id": quantum_id, "layer": layer,
        "campaign": {
            "plan_sha256": campaign["plan_sha256"],
            "prepared_sha256": campaign["prepared_sha256"],
            "plan_path": campaign["plan_path"],
            "prepared_path": campaign["prepared_path"],
            "read_manifest_sha256": campaign["read_manifest_sha256"],
            "campaign_scope": campaign["scope"],
            "unit_roster_sha256": campaign["roster_sha256"],
        },
        "read_set": {
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
            "source_phase": {"name": f"layer-{layer}",
                             "start_bytes": 0, "end_bytes": size},
        },
        "chunks": [{"name": f"{quantum_id}-chunk-000", "start_bytes": 0,
                    "end_bytes": size}],
        "windows": [{"window_index": 0, "names": []}],
        # Stride-1 geometry: every layer reads checkpoint ``layer + 1`` and
        # carries no chain. Unbound (pre-A) until ``_bind`` seals its slice.
        "adjoint": {"checkpoint_boundary": layer + 1, "chain_layers": [],
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
    slices = tmp_path / "slices"
    slices.mkdir()
    for layer in range(N_LAYERS):
        record = _record(campaign, layer, slice_dir=slices)
        (directory / f"layer-{layer:03d}.json").write_text(json.dumps(record))
    return directory


def _receipt(campaign, *, plan_sha256=None, **overrides):
    """A completed stride-1 stage-A receipt for the fixture campaign."""
    assert ADJOINT_SCHEMA == "prismaquant.joint_adjoint_capture.v1"
    return synthetic_receipt(
        plan_sha256=campaign["plan_sha256"] if plan_sha256 is None else plan_sha256,
        prepared_sha256=campaign["prepared_sha256"], scope=campaign["scope"],
        num_layers=N_LAYERS, stride=1, **overrides)


def _write_receipt(path, receipt):
    path.write_text(json.dumps(receipt))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind(record, proof, slice_root):
    """Bind one record to the slice ``proof`` gives its layer, the way the
    producer does (PQ #993): the slice file first, then the record's adjoint
    block names that file and its digest, and the identity is resealed."""
    layer = record["layer"]
    adjoint_slice = stage_a_slice(proof, layer)
    path = Path(slice_root) / f"{record['quantum_id']}.json"
    write_adjoint_slice(path, adjoint_slice, layer=layer)
    record["adjoint"] = {
        "checkpoint_boundary": record["adjoint"]["checkpoint_boundary"],
        "chain_layers": record["adjoint"]["chain_layers"],
        **adjoint_binding_fields(adjoint_slice, slice_path=str(path))}
    record.pop("identity_sha256", None)
    record["identity_sha256"] = canonical_json_sha256(
        record, where="fixture layer-quantum record")
    return record


def _stamp_receipt(records_dir, receipt_path):
    """Seal every record against the slice the receipt gives its layer."""
    proof = json.loads(Path(receipt_path).read_text())
    for record_path in sorted(records_dir.glob("layer-*.json")):
        record = json.loads(record_path.read_text())
        _bind(record, proof, records_dir.parent / "adjoint-slices")
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
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
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
    assert manifest == str(Path(slices) / "layer-001.data-manifest.json")
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
    envelope = argv[:argv.index("--")]
    envs = [envelope[i + 1] for i, word in enumerate(envelope) if word == "--env"]
    assert envs[-1] == "PRISMAQUANT_DEV_MODE=1"
    # The fixture spec declares the produced spool, so the quantum row seals
    # it too; the container refuses a declared spool the action lacks (#1012).
    assert envs[:-1] == [f"{name}={value}" for name, value in SPOOL_ENV.items()]
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
    assert inner[inner.index("--quantum-sha256") + 1] == hashlib.sha256(
        record_path.read_bytes()).hexdigest()
    assert inner[inner.index("--plan") + 1] == campaign["plan_path"]
    assert inner[inner.index("--plan-sha256") + 1] == campaign["plan_sha256"]
    assert inner[inner.index("--prepared") + 1] == campaign["prepared_path"]
    assert inner[inner.index("--prepared-sha256") + 1] == campaign["prepared_sha256"]
    # PQ #993: the quantum reads its own stage-A slice, never a receipt.
    slice_path = Path(record["adjoint"]["slice_path"])
    assert inner[inner.index("--adjoint-slice") + 1] == str(slice_path)
    assert inner[inner.index("--adjoint-slice-sha256") + 1] == hashlib.sha256(
        slice_path.read_bytes()).hexdigest() == record["adjoint"]["slice_sha256"]
    assert "--adjoint" not in inner and "--adjoint-sha256" not in inner
    assert "--resume" in inner
    assert inner[inner.index("--data-manifest-sha256") + 1] == record["read_set"]["manifest_sha256"]
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
    _stamp_receipt(records_dir, receipt_path)
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
    manifest = _adjoint_manifest(tmp_path, campaign,
                                   name="adjoint.data-manifest.json.gz")
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
                 "--adjoint-manifest", str(_adjoint_manifest(tmp_path, campaign)),
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
    _stamp_receipt(records_dir, receipt_path)
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
    manifest = _adjoint_manifest(tmp_path, campaign)
    gateway = FakeGateway(terminal=False)
    out = tmp_path / "out"
    assert main(_argv(records_dir, out) + ["--adjoint-manifest", str(manifest)],
                _gateway=gateway) == 0
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
    _stamp_receipt(records_dir, receipt_path)
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
    _stamp_receipt(records_dir, receipt_path)
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    (out / "layer-quanta" / "campaign-state.json").write_text(
        json.dumps({"event": "stage-a-submitted",
                    "action_key": "stage-a-action-key"}) + "\n")
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    kinds = [call["kind"] for call in gateway.submitted]
    assert kinds == ["quantum", "quantum", "quantum"]


# -- checkpoint bands (PQ #993): publish a band's layers before the receipt ------


def _write_band(tmp_path, receipt, boundary):
    path = tmp_path / "bands" / f"band-{boundary:03d}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(band_from_receipt(receipt, boundary)))
    return path


def test_a_band_publishes_its_own_layers_while_stage_a_runs(
        tmp_path, campaign, records_dir, capsys):
    """Stage A seals the tail band hours before its receipt. The dispatcher
    publishes exactly the quanta that band serves; the rest stay pending, and
    the running stage A is neither refused nor republished."""
    receipt = _receipt(campaign)
    receipt_path = tmp_path / "adjoint-capture.json"
    _write_receipt(receipt_path, receipt)
    _stamp_receipt(records_dir, receipt_path)
    receipt_path.unlink()
    band = _write_band(tmp_path, receipt, N_LAYERS)
    gateway = FakeGateway(terminal=False)
    out = tmp_path / "out"
    (out / "layer-quanta").mkdir(parents=True)
    (out / "layer-quanta" / "campaign-state.json").write_text(
        json.dumps({"event": "stage-a-submitted",
                    "action_key": "stage-a-action-key"}) + "\n")
    argv = _argv(records_dir, out, None, ("--adjoint-band", str(band)))
    assert main(argv + ["--dry-run"], _gateway=gateway) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [row["quantum_id"] for row in printed["rows"]] == ["layer-002"]
    assert printed["stage_a_pending"] == ["layer-001", "layer-000"]
    assert main(argv, _gateway=gateway) == 0
    assert [call.get("quantum_id") for call in gateway.submitted] == ["layer-002"]
    # A second band adds its layer and republishes nothing.
    gateway.mark_terminal(gateway.submitted[0]["action_key"])
    gateway.submitted.clear()
    lower = _write_band(tmp_path, receipt, N_LAYERS - 1)
    assert main(argv + ["--adjoint-band", str(lower)], _gateway=gateway) == 0
    assert [call.get("quantum_id") for call in gateway.submitted] == ["layer-001"]


def test_bands_of_another_run_or_slice_refuse(tmp_path, campaign, records_dir):
    """A named band that does not prove the sealed records refuses at once
    (exit 3, nothing published), whether stage A is terminal or not: a band
    from another stage-A generation mixed with this run's band, or a band
    whose slice differs from the one each record binds."""
    receipt = _receipt(campaign)
    other = _receipt(campaign, generation="another-generation")
    receipt_path = tmp_path / "adjoint-capture.json"
    _write_receipt(receipt_path, receipt)
    _stamp_receipt(records_dir, receipt_path)
    receipt_path.unlink()
    out = tmp_path / "out"
    (tmp_path / "other").mkdir()
    mixed = ("--adjoint-band", str(_write_band(tmp_path, receipt, N_LAYERS)),
             "--adjoint-band", str(_write_band(tmp_path / "other", other, N_LAYERS - 1)))
    gateway = FakeGateway(terminal=False)
    assert main(_argv(records_dir, out, None, mixed), _gateway=gateway) == 3
    assert gateway.submitted == []
    foreign = ("--adjoint-band", str(_write_band(tmp_path / "other", other, N_LAYERS)))
    assert main(_argv(records_dir, out, None, foreign), _gateway=gateway) == 3
    assert gateway.submitted == []


def test_a_proof_whose_stride_places_a_record_elsewhere_refuses(
        tmp_path, campaign, records_dir):
    """The dispatcher derives no stride. The slice gate holds the property:
    records sealed at stride 1 (layer 0 reads checkpoint 1) bound to the
    slices of a stride-2 run (layer 0 reads checkpoint 2) refuse, because
    ``load_adjoint_slice`` checks where the header's stride places each
    record's layer."""
    receipt = synthetic_receipt(
        plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"], scope=campaign["scope"],
        num_layers=N_LAYERS, stride=2)
    assert receipt["stride"]["boundaries"] == [3, 2]
    receipt_path = tmp_path / "adjoint-capture.json"
    _write_receipt(receipt_path, receipt)
    _stamp_receipt(records_dir, receipt_path)
    gateway = FakeGateway(terminal=True)
    assert main(_argv(records_dir, tmp_path / "out", receipt_path),
                _gateway=gateway) == 3
    assert gateway.submitted == []


# -- dry-run: plan with digests, submits nothing ---------------------------------


def test_dry_run_prints_plan_with_digests_and_submits_nothing(
        tmp_path, campaign, records_dir, capsys):
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, receipt_path)
    gateway = FakeGateway(terminal=True)
    out = tmp_path / "out"
    assert main(_argv(records_dir, out, receipt_path, ("--dry-run",)),
                _gateway=gateway) == 0
    assert gateway.submitted == []
    printed = capsys.readouterr().out
    for layer in range(N_LAYERS):
        assert f"layer-{layer:03d}" in printed
    assert "PRISMAQUANT_DEV_MODE=1" in printed
    # Each row names the slice it binds and the chain it reads, never the
    # receipt's digest.
    rows = {row["quantum_id"]: row for row in json.loads(printed)["rows"]}
    for record_path in records_dir.glob("layer-*.json"):
        record = json.loads(record_path.read_text())
        row = rows[record["quantum_id"]]
        assert row["slice_sha256"] == record["adjoint"]["slice_sha256"]
        assert row["checkpoint_boundary"] == record["layer"] + 1
        assert row["chain_layers"] == []
    assert digest not in printed


# -- idempotency: re-running publishes nothing already terminal -----------------


def test_rerun_publishes_nothing_already_terminal(tmp_path, campaign, records_dir):
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, receipt_path)
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


# -- stage-A manifest binding: payload digests + progress phases (#835) -------


def _adjoint_manifest(tmp_path, campaign, *, phases=("head", "chain-000", "chain-001"),
                      parent=None, plan_sha256=None, prepared_sha256=None,
                      name="adjoint.data-manifest.json", gzip_bytes=False,
                      entry_count=0, schema="prismaquant.prismabuild.data_manifest.v1",
                      read_plan=None):
    """A minimal stage-A data manifest: schema, entries, and the annotations
    the dispatcher derives the submission binding from (phase names in read
    order, the parent read-set digest, the sealed plan/prepared digests).

    ``read_plan`` (a list of phase dicts) selects the v2 shape: phases move
    to ``read_plan`` and ``annotations.phases`` must be absent.
    """
    if parent is None:
        parent = campaign["read_manifest_sha256"]
    annotations = {
        "parent_manifest_sha256": parent,
        "plan_sha256": campaign["plan_sha256"] if plan_sha256 is None else plan_sha256,
        "prepared_sha256": campaign["prepared_sha256"] if prepared_sha256 is None else prepared_sha256,
    }
    manifest = {
        "schema": schema,
        "mount_prefix": "/mnt/shared",
        "entries": [],
        "entry_count": entry_count,
        "total_bytes": 0,
        "annotations": annotations,
    }
    if read_plan is not None:
        manifest["read_plan"] = {"phases": read_plan, "read_bytes": 0}
    else:
        annotations["phases"] = [{"name": name, "bytes": 0,
                                  "cumulative_bytes": 0} for name in phases]
    path = tmp_path / name
    raw = json.dumps(manifest).encode("utf-8")
    if gzip_bytes:
        import gzip
        raw = gzip.compress(raw)
    path.write_bytes(raw)
    return path


def _payload_inner(argv):
    tail = argv[argv.index("--") + 1:]
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_adjoint_capture"]
    return inner


def _envelope_envs(argv):
    envelope = argv[:argv.index("--")]
    return [envelope[i + 1] for i, word in enumerate(envelope) if word == "--env"]


def _write_fixture_spec(spec):
    import dispatch_joint_quanta
    dispatch_joint_quanta.SPEC_PATH.write_text(json.dumps(spec))


def test_stage_a_seals_the_paced_spool_and_the_ram_tier(tmp_path, campaign):
    """PQ #1012: the Stage A request carries the spool root, its byte bound
    and the paced-export opt-in as sealed environment, which PrismaBuild
    reads from the producer's request, and it seals the storage box's RAM
    tier as the quantum row does. Both are pbrun envelope options."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = stage_a_argv(manifest, campaign)
    envelope = argv[:argv.index("--")]
    assert envelope[envelope.index("--residency") + 1] == "stage"
    assert envelope[envelope.index("--residency-ram") + 1] == "auto"
    assert _envelope_envs(argv) == [
        *(f"{name}={value}" for name, value in STAGE_A_SPOOL_ENV.items()),
        "PRISMAQUANT_DEV_MODE=1"]
    # The spec the container launches from declares the same spool, so the
    # launcher's own check (the sealed launch equals the spec) holds.
    tail = argv[argv.index("--") + 1:]
    sealed = json.loads(tail[tail.index("--spec") + 1])
    assert {name: sealed["env"][name] for name in STAGE_A_SPOOL_ENV} == STAGE_A_SPOOL_ENV
    assert SPOOL_MOUNT in sealed["container"]["mounts"]


def test_stage_a_seals_the_plans_two_plane_window_not_the_spec_bound(tmp_path, campaign):
    """PQ #1110: the chain reads its own cotangent planes back from the
    spool, so the row seals the plan's two-plane window in place of the
    spec's bound, in the request environment and in the launched spec alike.
    The spec on disk is unchanged."""
    import dispatch_joint_quanta
    before = dispatch_joint_quanta.SPEC_PATH.read_text()
    argv = stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)
    assert f"PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES={SPOOL_WINDOW_BYTES}" in _envelope_envs(argv)
    assert SPOOL_WINDOW_BYTES != int(SPOOL_ENV["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"])
    tail = argv[argv.index("--") + 1:]
    sealed = json.loads(tail[tail.index("--spec") + 1])
    assert sealed["env"]["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"] == str(SPOOL_WINDOW_BYTES)
    assert dispatch_joint_quanta.SPEC_PATH.read_text() == before


def test_the_two_plane_window_at_the_glm_shape(tmp_path):
    """R12's shape: 4 probes x 512 one-row entries, each 512 tokens x 4096
    hidden x 4 mHC streams of bf16 plus the 64 KiB envelope, two planes."""
    from dispatch_joint_quanta import stage_a_spool_window_bytes
    model = tmp_path / "glm"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {
        "hidden_size": 4096, "hc_mult": 4, "dtype": "bfloat16"}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"model": str(model), "execution": {
        "n_probes": 4, "n_calib_samples": 512, "calib_seqlen": 512,
        "probe_microbatch": 1, "boundary_storage": {"prefetch_batches": 64}}}))
    window = stage_a_spool_window_bytes({"plan_path": str(plan)})
    assert window == 2 * 4 * 512 * (512 * 4096 * 4 * 2 + 65536) == 68_987_912_192
    # PQ #738: a split quantum of R13's round 1 owns 64 of the 512 batches.
    quantum = stage_a_spool_window_bytes({"plan_path": str(plan)}, (64, 128))
    assert quantum == 2 * 4 * 64 * (512 * 4096 * 4 * 2 + 65536) == window // 8


def test_a_split_quantum_seals_two_planes_of_its_own_batches(tmp_path, campaign):
    """PQ #738: a split quantum's spool holds its own range's planes, and the
    capture derives its need from the same range, so the row seals that
    window rather than the whole run's."""
    argv = stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign, batch_range=(4, 8))
    envs = dict(env.split("=", 1) for env in _envelope_envs(argv))
    assert int(envs["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"]) == SPOOL_WINDOW_BYTES // 2
    tail = argv[argv.index("--") + 1:]
    sealed = json.loads(tail[tail.index("--spec") + 1])
    assert sealed["env"]["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"] == str(
        SPOOL_WINDOW_BYTES // 2)
    with pytest.raises(DispatchRefused, match="not inside the plan's 8 batches"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign, batch_range=(4, 9))


def test_stage_a_seals_the_host_window_beside_its_two_plane_bound(tmp_path, campaign):
    """PQ #1120: the Stage A row opts into PrismaBuild's host spool window
    (PB #910), so placement charges its two-plane window to the executing
    box. The opt-in is sealed in the request environment, where PrismaBuild
    reads it, and in the launched spec, which the container checks against
    it. The spec on disk is unchanged."""
    import dispatch_joint_quanta
    before = dispatch_joint_quanta.SPEC_PATH.read_text()
    argv = stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)
    envs = dict(env.split("=", 1) for env in _envelope_envs(argv))
    assert envs["PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"] == "1"
    assert envs["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"] == str(SPOOL_WINDOW_BYTES)
    tail = argv[argv.index("--") + 1:]
    sealed = json.loads(tail[tail.index("--spec") + 1])
    assert sealed["env"]["PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"] == "1"
    assert dispatch_joint_quanta.SPEC_PATH.read_text() == before


def test_stage_a_refuses_a_spec_that_opts_out_of_the_host_window(tmp_path, campaign):
    """A spec that declares the host window off would leave the row's
    two-plane window uncharged at placement; the row refuses it rather than
    overriding what the spec says."""
    import dispatch_joint_quanta
    spec = json.loads(dispatch_joint_quanta.SPEC_PATH.read_text())
    spec["env"]["PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"] = "0"
    dispatch_joint_quanta.SPEC_PATH.write_text(json.dumps(spec))
    with pytest.raises(DispatchRefused, match="HOST_WINDOW"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)


#: Run in a child process: it imports the published PrismaBuild tree, which
#: this process may already have imported from another generation.
_PLACEMENT_PROBE = r"""
import importlib.util, json, sys
from pathlib import Path
published, variables, queue_root = Path(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("published_pbrun", published / "tools" / "pbrun.py")
pbrun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbrun)
from prismabuild import pool
terms = pbrun.local_disk_terms(variables, transport="pool")
queue = pool.PoolQueue(queue_root)
queue.ensure_layout()
resources = {"cpu": 1, "mem_gb": 1, **terms}
for key in ("a" * 64, "b" * 64):
    queue.publish(action_key=key, cas_root="/cas",
                  worker_script=str(published / "tools" / "prismabuild_worker.py"),
                  checkout_root=str(queue_root.parent / "checkout"), resources=resources)
capacity = {"cpu": 4, "mem_gb": 8, "spool_gb": max(2 * terms.get("spool_gb", 0) - 1, 0)}
first = queue.claim(owner="first", capacity=capacity)
second = queue.claim(owner="second", capacity=capacity)
print(json.dumps({"pbrun": pbrun.__file__, "pool": pool.__file__, "terms": terms,
                  "first": None if first is None else first["resources"],
                  "second": None if second is None else second["resources"],
                  "available": queue.ledger().available()}))
"""


def test_two_stage_a_rows_whose_windows_exceed_a_box_are_not_both_placed(
        tmp_path, campaign):
    """PQ #1120 acceptance, on PrismaBuild's own published code: the sealed
    environment of a Stage A row derives a ``spool_gb`` demand of its window
    in whole GiB (pbrun ``local_disk_terms``), and a box whose spool budget
    holds one such window but not two claims one row and refuses the other.
    """
    import subprocess
    published = Path("/mnt/shared/prismabuild-fleet/repo")
    if not (published / "tools" / "pbrun.py").is_file():
        pytest.skip(f"published PrismaBuild not visible at {published}")
    argv = stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)
    variables = dict(env.split("=", 1) for env in _envelope_envs(argv))
    run = subprocess.run(
        [sys.executable, "-c", _PLACEMENT_PROBE, str(published),
         json.dumps(variables), str(tmp_path / "pb-queue")],
        capture_output=True, text=True, timeout=300,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(published / "src"),
             "HOME": str(tmp_path)})
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    for module in ("pbrun", "pool"):
        assert Path(result[module]).resolve().is_relative_to(
            published.resolve()), result
    need = -(-SPOOL_WINDOW_BYTES // (1 << 30))
    assert result["terms"] == {"spool_gb": need}
    assert result["first"]["spool_gb"] == need
    assert result["second"] is None, result


@pytest.mark.parametrize("defect, reason", [
    ("no execution", "plane geometry"),
    ("no dtype", "known dtype"),
    ("no hidden size", "hidden size"),
])
def test_stage_a_refuses_a_plan_it_cannot_derive_the_window_from(
        tmp_path, campaign, defect, reason):
    plan_path = Path(campaign["plan_path"])
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan["model"]) / "config.json"
    config = json.loads(config_path.read_text())
    if defect == "no execution":
        plan.pop("execution")
    elif defect == "no dtype":
        config.pop("dtype")
    else:
        config.pop("hidden_size")
    plan_path.write_text(json.dumps(plan))
    config_path.write_text(json.dumps(config))
    with pytest.raises(DispatchRefused, match=reason):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)


@pytest.mark.parametrize("aggregate, expected", [
    (None, "104"),                       # a plan stating no bound: the old reservation
    (108447924224, "101"),               # the GLM plan's bound, exactly 101 GiB
    (101 * 1024 ** 3 + 1, "102"),        # one byte over rounds up to a whole GiB
])
def test_stage_a_reserves_the_plans_memory_bound(tmp_path, campaign, aggregate, expected):
    """PQ #997: the Stage A row reserves the plan's combined physical bound
    (``aggregate_memory_bytes``, rounded up to whole GiB), not a constant 104
    that a GB10 offering 102 GiB live could never place."""
    plan_path = Path(campaign["plan_path"])
    plan = json.loads(plan_path.read_text())
    if aggregate is not None:
        plan["aggregate_memory_bytes"] = aggregate
    plan_path.write_text(json.dumps(plan))
    argv = stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)
    envelope = argv[:argv.index("--")]
    assert envelope[envelope.index("--demand") + 1] == f"gpu=1,mem_gb={expected}"


@pytest.mark.parametrize("aggregate", [0, -1, 1.5, True, "108447924224"])
def test_stage_a_refuses_a_malformed_memory_bound(tmp_path, campaign, aggregate):
    plan_path = Path(campaign["plan_path"])
    plan = json.loads(plan_path.read_text())
    plan["aggregate_memory_bytes"] = aggregate
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(DispatchRefused, match="aggregate_memory_bytes"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)


def test_stage_a_refuses_a_spec_without_the_spool(tmp_path, campaign):
    """A spec with no spool root would let the owner write every boundary
    entry synchronously into the pool: the Stage A dispatch refuses it."""
    _write_fixture_spec({"container": {"image": "sha256:" + "0" * 64}, "env": {}})
    manifest = _adjoint_manifest(tmp_path, campaign)
    with pytest.raises(DispatchRefused,
                       match="declares no PRISMABUILD_PRODUCED_SPOOL_ROOT"):
        stage_a_argv(manifest, campaign)


def _spool_variant(**env):
    spec = with_spool({"container": {"image": "sha256:" + "0" * 64}, "env": {}})
    for name, value in env.items():
        if value is None:
            spec["env"].pop(name, None)
        else:
            spec["env"][name] = value
    return spec


def _shared_root_spec():
    root = "/mnt/shared/pb-spool/fixture"
    spec = _spool_variant(PRISMABUILD_PRODUCED_SPOOL_ROOT=root)
    spec["container"]["mounts"] = [{"source": root, "target": root, "readonly": False}]
    return spec


def _readonly_bind_spec():
    spec = _spool_variant()
    spec["container"]["mounts"] = [{**SPOOL_MOUNT, "readonly": True}]
    return spec


def _unbound_spec():
    spec = _spool_variant()
    spec["container"]["mounts"] = []
    return spec


SPOOL_REFUSALS = {
    "root under /mnt/shared": (_shared_root_spec, "under /mnt/shared"),
    "no byte bound": (lambda: _spool_variant(
        PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES=None), "positive byte ceiling"),
    "zero byte bound": (lambda: _spool_variant(
        PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES="0"), "positive byte ceiling"),
    "relative root": (lambda: _spool_variant(
        PRISMABUILD_PRODUCED_SPOOL_ROOT="pb-spool"), "canonical absolute root"),
    "read-only bind": (_readonly_bind_spec, "writable identity bind"),
    "no bind": (_unbound_spec, "writable identity bind"),
    "malformed paced opt-in": (lambda: _spool_variant(
        PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT="yes"), "must be \"0\" or \"1\""),
    "opt-in without a root": (lambda: _spool_variant(
        PRISMABUILD_PRODUCED_SPOOL_ROOT=None,
        PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES=None),
        "PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT without"),
}


@pytest.mark.parametrize("case", sorted(SPOOL_REFUSALS))
def test_stage_a_refuses_a_malformed_spool(tmp_path, campaign, case):
    """The container's spool check runs at dispatch, and the root must be on
    the executing box's own disk; each defect refuses before anything is
    published, for its own reason."""
    build, reason = SPOOL_REFUSALS[case]
    _write_fixture_spec(build())
    manifest = _adjoint_manifest(tmp_path, campaign)
    with pytest.raises(DispatchRefused, match=reason):
        stage_a_argv(manifest, campaign)


def test_a_quantum_row_seals_the_spool_its_spec_declares(tmp_path, campaign):
    """The quantum row shares the spec. When the spec declares the spool the
    row seals it (the container refuses a declared spool the action lacks);
    when it declares none the row carries nothing new, and a quantum never
    refuses for it."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    argv = quantum_argv(record, record_path=record_path, output_root=Path("/out/root"))
    assert _envelope_envs(argv) == [
        *(f"{name}={value}" for name, value in SPOOL_ENV.items()),
        "PRISMAQUANT_DEV_MODE=1"]
    _write_fixture_spec({"container": {"image": "sha256:" + "0" * 64}, "env": {}})
    argv = quantum_argv(record, record_path=record_path, output_root=Path("/out/root"))
    assert _envelope_envs(argv) == ["PRISMAQUANT_DEV_MODE=1"]
    _write_fixture_spec(_shared_root_spec())
    with pytest.raises(DispatchRefused, match="under /mnt/shared"):
        quantum_argv(record, record_path=record_path, output_root=Path("/out/root"))


def test_the_default_spec_declares_the_paced_spool():
    """The campaign's default spec (PQ #1012) carries the spool root on the
    executing box's disk, a 32 GiB bound and the paced export, and no host
    window: the Stage A row seals that opt-in itself (PQ #1120), and a
    quantum row seals the spec's spool as it is. It lives on the shared
    mount."""
    import dispatch_joint_quanta
    path = dispatch_joint_quanta.DEFAULT_SPEC_PATH
    if not path.is_file():
        pytest.skip(f"the campaign spec is not mounted here: {path}")
    spec = json.loads(path.read_text())
    forwarded = dispatch_joint_quanta.produced_spool_row_environment(spec)
    assert forwarded == {
        "PRISMABUILD_PRODUCED_SPOOL_ROOT": "/home/rob/pb-spool/glm-campaign",
        "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": str(32 << 30),
        "PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT": "1"}


def test_stage_a_argv_binds_manifest_digests_into_payload(tmp_path, campaign):
    """The tier redirect needs the manifest digest *inside* the payload:
    ``bind_residency_manifest`` refuses a run that bound nothing, and the
    wrapper forwards no digest flag the dispatcher does not thread. The
    payload therefore carries ``--data-manifest-sha256`` (the submitted
    manifest bytes) and ``--read-manifest-sha256`` (the annotated parent
    read-set digest); the pbrun envelope keeps naming the file."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = stage_a_argv(manifest, campaign)
    inner = _payload_inner(argv)
    assert inner[inner.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    assert inner[inner.index("--read-manifest-sha256") + 1] == "d" * 64
    assert argv[argv.index("--data-manifest") + 1] == str(manifest)


def test_stage_a_argv_declares_manifest_phases_as_progress(tmp_path, campaign):
    """Progress declarations come from the validated manifest annotations, in
    manifest order -- never a hardcoded list. The worker refuses undeclared
    names, so a stale or invented list would silence the run's window."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = stage_a_argv(manifest, campaign)
    phases = [argv[i + 1] for i, word in enumerate(argv[:-1])
              if word == "--progress-phase"]
    assert phases == ["head=1800", "chain-000=900", "chain-001=900"]
    assert HEAD_PROGRESS_GRACE_S == 1800
    assert CHUNK_PROGRESS_GRACE_S == 900


def test_stage_a_argv_reads_gzip_manifest_bytes(tmp_path, campaign):
    """The default manifest filename is ``.json.gz``; the bound digest covers
    the wire bytes pbrun ingests either way."""
    manifest = _adjoint_manifest(tmp_path, campaign, gzip_bytes=True,
                                 name="adjoint.data-manifest.json.gz")
    argv = stage_a_argv(manifest, campaign)
    inner = _payload_inner(argv)
    assert inner[inner.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    phases = [argv[i + 1] for i, word in enumerate(argv[:-1])
              if word == "--progress-phase"]
    assert phases[0] == "head=1800"


def test_stage_a_argv_refuses_manifest_plan_mismatch(tmp_path, campaign):
    """A manifest built against another plan/prepared pair is a mixed
    campaign: refuse before publishing, exit 3 downstream."""
    manifest = _adjoint_manifest(tmp_path, campaign, plan_sha256="f" * 64)
    with pytest.raises(DispatchRefused, match="plan"):
        stage_a_argv(manifest, campaign)


@pytest.mark.parametrize("defect", [
    "missing-annotations", "empty-phases", "dup-phase", "blank-phase",
    "bad-parent", "bad-schema", "count-drift", "unreadable",
])
def test_stage_a_argv_refuses_malformed_manifest(tmp_path, campaign, defect):
    """Every malformed manifest shape refuses at dispatch time, never as a
    launched action with an inert tier path."""
    manifest = tmp_path / "adjoint.data-manifest.json"
    if defect == "unreadable":
        pass
    else:
        phases = ("head", "chain-000")
        parent = "d" * 64
        schema = "prismaquant.prismabuild.data_manifest.v1"
        count = 0
        annotations = True
        if defect == "empty-phases":
            phases = ()
        elif defect == "dup-phase":
            phases = ("head", "head")
        elif defect == "blank-phase":
            phases = ("head", "")
        elif defect == "bad-parent":
            parent = "not-a-digest"
        elif defect == "bad-schema":
            schema = "prismaquant.prismabuild.data_manifest.v9"
        elif defect == "count-drift":
            count = 7
        manifest.write_text(json.dumps({
            "schema": schema,
            "mount_prefix": "/mnt/shared",
            "entries": [],
            "entry_count": count,
            "total_bytes": 0,
            **({"annotations": {
                "phases": [{"name": name} for name in phases],
                "parent_manifest_sha256": parent,
                "plan_sha256": campaign["plan_sha256"],
                "prepared_sha256": campaign["prepared_sha256"],
            }} if annotations else {}),
        }))
        if defect == "missing-annotations":
            payload = json.loads(manifest.read_text())
            del payload["annotations"]
            manifest.write_text(json.dumps(payload))
    with pytest.raises(DispatchRefused):
        stage_a_argv(manifest, campaign)


def test_main_publishes_bound_stage_a_row(tmp_path, campaign, records_dir):
    """The published stage-A row carries the manifest binding end to end:
    pbrun envelope (file + residency + phases), payload digests, and the
    state event recording both digests for the launch record."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    gateway = FakeGateway()
    out = tmp_path / "out"
    assert main(_argv(records_dir, out) + ["--adjoint-manifest", str(manifest)],
                _gateway=gateway) == 0
    rows = [row for row in gateway.submitted if row["kind"] == "stage-a"]
    assert len(rows) == 1
    argv = rows[0]["argv"]
    inner = _payload_inner(argv)
    data_id = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert inner[inner.index("--data-manifest-sha256") + 1] == data_id
    assert inner[inner.index("--read-manifest-sha256") + 1] == "d" * 64
    phases = [argv[i + 1] for i, word in enumerate(argv[:-1])
              if word == "--progress-phase"]
    assert phases == ["head=1800", "chain-000=900", "chain-001=900"]
    events = [json.loads(line) for line
              in (out / "layer-quanta" / "campaign-state.json").read_text().splitlines()
              if line]
    assert events[0]["event"] == "stage-a-submitted"
    assert events[0]["data_manifest_sha256"] == data_id
    assert events[0]["read_manifest_sha256"] == "d" * 64


# -- quantum slice binding + v2 read plans (#835, root extension) --------------


def _quantum_payload(argv):
    tail = argv[argv.index("--") + 1:]
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    return inner


def test_quantum_argv_binds_slice_digest_not_campaign_parent(tmp_path, campaign):
    """The resolver binding is the slice manifest pbrun stages for this row,
    not the campaign parent the record also carries. The fixture seals a
    parent digest that differs from the slice file's, exactly the live shape
    (parent ``71fd…`` vs per-slice digests): the payload must carry the
    slice's actual bytes digest, or the map can never match."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
    assert record["campaign"]["read_manifest_sha256"] != record["read_set"]["manifest_sha256"]
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    inner = _quantum_payload(quantum_argv(
        record, record_path=record_path, output_root=Path("/out/root")))
    bound = inner[inner.index("--data-manifest-sha256") + 1]
    assert bound == record["read_set"]["manifest_sha256"]
    assert bound == hashlib.sha256(
        Path(record["read_set"]["manifest_path"]).read_bytes()).hexdigest()


def test_quantum_argv_refuses_drifted_slice(tmp_path, campaign):
    """A slice whose bytes no longer hash to the sealed digest refuses at
    dispatch time -- it would otherwise launch a row whose map matches
    nothing."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
    Path(record["read_set"]["manifest_path"]).write_bytes(b"tampered")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    with pytest.raises(DispatchRefused, match="do not hash to the sealed"):
        quantum_argv(record, record_path=record_path,
                     output_root=Path("/out/root"))


def test_quantum_argv_refuses_absent_slice(tmp_path, campaign):
    """A sealed digest over bytes the dispatcher cannot read is a producer
    defect, refused before publishing -- never a row that binds blind."""
    record = _bind(_record(campaign, 1), _receipt(campaign), tmp_path / "adjoint-slices")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    with pytest.raises(DispatchRefused, match="unreadable"):
        quantum_argv(record, record_path=record_path,
                     output_root=Path("/out/root"))


def test_quantum_argv_refuses_an_unbound_or_drifted_stage_a_slice(tmp_path, campaign):
    """PQ #993: a row reads the slice its record binds. An unbound (pre-A)
    record, or a slice file whose bytes no longer hash to the sealed digest,
    refuses before anything publishes."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _record(campaign, 1, slice_dir=slices)
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    with pytest.raises(DispatchRefused, match="unbound"):
        quantum_argv(record, record_path=record_path, output_root=Path("/out/root"))
    record = _bind(record, _receipt(campaign), tmp_path / "adjoint-slices")
    record_path.write_text(json.dumps(record))
    Path(record["adjoint"]["slice_path"]).write_bytes(b"{}")
    with pytest.raises(DispatchRefused, match="does not hash to the sealed digest"):
        quantum_argv(record, record_path=record_path, output_root=Path("/out/root"))


def test_stage_a_argv_accepts_v2_read_plan(tmp_path, campaign):
    """The v2 shape carries phases in ``read_plan`` (``annotations.phases``
    is forbidden there): the dispatcher derives declarations and the parent
    binding from the v2 table, with the digest still covering wire bytes."""
    read_plan = [
        {"name": "head", "entry_indices": [], "bytes": 0, "cumulative_bytes": 0},
        {"name": "layer-0", "entry_indices": [], "bytes": 0, "cumulative_bytes": 0},
        {"name": "chain_000", "entry_indices": [], "bytes": 0, "cumulative_bytes": 0},
    ]
    manifest = _adjoint_manifest(
        tmp_path, campaign, name="adjoint.data-manifest.json.gz", gzip_bytes=True,
        schema="prismaquant.prismabuild.data_manifest.v2", read_plan=read_plan)
    argv = stage_a_argv(manifest, campaign)
    inner = _payload_inner(argv)
    assert inner[inner.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    assert inner[inner.index("--read-manifest-sha256") + 1] == "d" * 64
    phases = [argv[i + 1] for i, word in enumerate(argv[:-1])
              if word == "--progress-phase"]
    assert phases == ["head=1800", "layer-0=900", "chain_000=900"]


def test_stage_a_argv_refuses_v2_with_annotations_phases(tmp_path, campaign):
    """v2 forbids ``annotations.phases`` (PB's ``core`` holds that rule);
    a manifest carrying both tables refuses here, not at the worker."""
    manifest = _adjoint_manifest(tmp_path, campaign,
                                 schema="prismaquant.prismabuild.data_manifest.v2",
                                 read_plan=[{"name": "head"}])
    payload = json.loads(manifest.read_text())
    payload["annotations"]["phases"] = [{"name": "head"}]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(DispatchRefused, match="read_plan, not"):
        stage_a_argv(manifest, campaign)


def test_stage_a_argv_refuses_foreign_read_parent(tmp_path, campaign):
    """A manifest from another lineage -- parent digest well-formed but not
    the sealed campaign read parent -- would launch a run whose receipt
    binds an incompatible identity. Refuse as a mixed campaign."""
    manifest = _adjoint_manifest(tmp_path, campaign, parent="e" * 64)
    with pytest.raises(DispatchRefused, match="mixed campaign"):
        stage_a_argv(manifest, campaign)


def test_resource_policy_controls_real_container_and_pb_envelopes(tmp_path, campaign, monkeypatch):
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    policy = {"limits": {"physical_bytes": 100 << 30, "host_bytes": 28 << 30, "gpu_bytes": 72 << 30}}
    monkeypatch.setattr(resources, 'verify_policy', lambda _: policy)
    plan = {"stage_b_resource_policy": {"path": "/resource", "sha256": "0"*64},
        "source_prefetch": {"prefetch_workers": 1}, "execution": {"operator_windows": {"prefetch_workers": 4}}}
    raw = json.dumps(plan).encode(); Path(campaign['plan_path']).write_bytes(raw)
    campaign['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    spec = {"container": {"image": "sha256:" + "0"*64, "content_sha256": "b"*64},
        "container_admission_reference": "content:sha256:" + "c"*64, "cpu_memory_gb": 28,
        "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72", "PRISMAQUANT_LAYER_READ_THREADS": "10"}}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path/'record.json'; path.write_text(json.dumps(record))
    args = dict(record_path=path, output_root=tmp_path/'out')
    argv = quantum_argv(record, **args)
    assert argv[argv.index('--demand')+1] == 'gpu=1,mem_gb=100'
    assert argv[argv.index('--gpu-memory-gb')+1] == '72'
    assert argv[argv.index('--cpus')+1] == '10'
    assert argv[argv.index('--container-image')+1] == 'content:sha256:' + 'c'*64
    spec['cpu_memory_gb'] = 32; dispatch.SPEC_PATH.write_text(json.dumps(spec))
    with pytest.raises(DispatchRefused, match='envelope differs'):
        quantum_argv(record, **args)



def test_dev_mode_dispatches_a_re_declared_resource_plan(tmp_path, campaign, monkeypatch, capsys):
    """PQ #1147: a plan re-declared after the quantum was sealed stamps by default."""
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    policy = {"limits": {"physical_bytes": 100 << 30, "host_bytes": 28 << 30, "gpu_bytes": 72 << 30}}
    monkeypatch.setattr(resources, 'verify_policy', lambda _: policy)
    plan = {"stage_b_resource_policy": {"path": "/resource", "sha256": "0"*64},
        "source_prefetch": {"prefetch_workers": 1}, "execution": {"operator_windows": {"prefetch_workers": 4}}}
    raw = json.dumps(plan).encode(); Path(campaign['plan_path']).write_bytes(raw)
    campaign['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    spec = {"container": {"image": "sha256:" + "0"*64, "content_sha256": "b"*64},
        "container_admission_reference": "content:sha256:" + "c"*64, "cpu_memory_gb": 28,
        "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72", "PRISMAQUANT_LAYER_READ_THREADS": "10"}}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path/'record.json'; path.write_text(json.dumps(record))
    # The plan is re-declared after the record sealed its digest.
    plan["stage_b_resource_policy"]["sha256"] = "1"*64
    Path(campaign['plan_path']).write_bytes(json.dumps(plan).encode())
    args = dict(record_path=path, output_root=tmp_path/'out')
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    argv = quantum_argv(record, **args)
    assert argv[argv.index('--demand')+1] == 'gpu=1,mem_gb=100'
    assert "[DEV-MODE] seal resource-bound plan differs" in capsys.readouterr().out
    # The row runs under the plan on disk, by the digest of its bytes.
    on_disk = hashlib.sha256(Path(campaign['plan_path']).read_bytes()).hexdigest()
    assert on_disk in " ".join(argv) and campaign['plan_sha256'] not in " ".join(argv)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(DispatchRefused, match='resource-bound plan differs'):
        quantum_argv(record, **args)

def test_portable_admission_does_not_skip_container_spec_validation(tmp_path):
    from dispatch_joint_quanta import _container_wrap
    path = tmp_path/'spec.json'
    path.write_text(json.dumps({'container_admission_reference': 'content:sha256:' + 'c'*64,
        'container': {'image': 'image', 'content_sha256': 'b'*64, 'unknown_field': True}}))
    with pytest.raises(RuntimeError, match='container must declare'):
        _container_wrap(path, ['python3'], progress=[('head', 1800)])


def test_extended_catalog_cannot_launch_historical_bare_parent_readset(tmp_path, campaign):
    record = _record(campaign, 1, slice_dir=tmp_path)
    record['catalog_extension'] = {'path': '/proof', 'sha256': 'e'*64}
    with pytest.raises(DispatchRefused, match='requires executable prepared-input'):
        quantum_argv(record, record_path=tmp_path/'record', output_root=tmp_path/'out')


@pytest.mark.parametrize('readonly', [False, True])
def test_cotangent_scratch_is_validated_and_sealed_in_outer_request(
        tmp_path, campaign, readonly):
    import dispatch_joint_quanta as dispatch
    from tools.tessera_campaign_container import cotangent_scratch_environment
    root = '/home/rob/pb-scratch/glm-stageb'
    env = {'PRISMAQUANT_STAGE_B_COTANGENT_ROOT': root,
           'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES': str(36 << 30)}
    spec = {'container': {'image': 'sha256:' + '0' * 64,
             'mounts': [{'source': root, 'target': root, 'readonly': readonly}]}, 'env': env}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    args = dict(record_path=path, output_root=tmp_path / 'out')
    if readonly:
        with pytest.raises(dispatch.DispatchRefused, match='writable identity bind'):
            quantum_argv(record, **args)
        return
    argv = quantum_argv(record, **args)
    outer = argv[:argv.index('--')]
    sealed = dict(outer[i + 1].split('=', 1) for i, value in enumerate(outer[:-1])
                  if value == '--env' and '=' in outer[i + 1])
    assert all(sealed.get(name) == value for name, value in env.items())
    assert sealed['PRISMABUILD_LOCAL_SCRATCH_PAIRS'] == (
        'PRISMAQUANT_STAGE_B_COTANGENT_ROOT:PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES')
    actual = json.loads(argv[argv.index('--spec') + 1])
    assert cotangent_scratch_environment(actual, sealed) == env


@pytest.mark.parametrize('readonly', [False, True])
def test_stage_b_spill_is_validated_and_sealed_in_outer_request(
        tmp_path, campaign, readonly):
    """The #994 replay spill is declared and sealed like the cotangent scratch."""
    import dispatch_joint_quanta as dispatch
    from tools.tessera_campaign_container import stage_b_spill_environment
    root = '/home/rob/pb-scratch/glm-stageb-spill'
    env = {'PRISMAQUANT_STAGE_B_SPILL_ROOT': root,
           'PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES': str(200 << 30)}
    spec = {'container': {'image': 'sha256:' + '0' * 64,
             'mounts': [{'source': root, 'target': root, 'readonly': readonly}]}, 'env': env}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    args = dict(record_path=path, output_root=tmp_path / 'out')
    if readonly:
        with pytest.raises(dispatch.DispatchRefused, match='writable identity bind'):
            quantum_argv(record, **args)
        return
    argv = quantum_argv(record, **args)
    outer = argv[:argv.index('--')]
    sealed = dict(outer[i + 1].split('=', 1) for i, value in enumerate(outer[:-1])
                  if value == '--env' and '=' in outer[i + 1])
    assert all(sealed.get(name) == value for name, value in env.items())
    assert sealed['PRISMABUILD_LOCAL_SCRATCH_PAIRS'] == (
        'PRISMAQUANT_STAGE_B_SPILL_ROOT:PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES')
    actual = json.loads(argv[argv.index('--spec') + 1])
    assert stage_b_spill_environment(actual, sealed) == env


_COTANGENT = ('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', 'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES')
_SPILL = ('PRISMAQUANT_STAGE_B_SPILL_ROOT', 'PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES')


def _scratch_quantum_argv(tmp_path, campaign, env, roots):
    """A Stage B quantum's argv under a spec declaring ``env`` over ``roots``."""
    import dispatch_joint_quanta as dispatch
    spec = {'container': {'image': 'sha256:' + '0' * 64,
             'mounts': [{'source': root, 'target': root} for root in roots]},
            'env': env}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    return lambda: quantum_argv(record, record_path=path, output_root=tmp_path / 'out')


def _outer_env(argv):
    """The ``--env`` entries pbrun seals, in order, before the payload."""
    outer = argv[:argv.index('--')]
    return [outer[i + 1] for i, value in enumerate(outer[:-1]) if value == '--env']


def test_every_declared_scratch_pair_is_listed_for_pb(tmp_path, campaign):
    """PQ #1019: the sealed request lists each pair, cotangent first, so PB
    #911 charges both ceilings to the executing box's ``spool_gb``."""
    cotangent, spill = '/home/rob/pb-scratch/cot', '/home/rob/pb-scratch/spill'
    env = {_COTANGENT[0]: cotangent, _COTANGENT[1]: str(36 << 30),
           _SPILL[0]: spill, _SPILL[1]: str(178_000_000_000)}
    argv = _scratch_quantum_argv(tmp_path, campaign, env, (cotangent, spill))()
    sealed = dict(item.split('=', 1) for item in _outer_env(argv))
    assert all(sealed.get(name) == value for name, value in env.items())
    assert sealed['PRISMABUILD_LOCAL_SCRATCH_PAIRS'] == ','.join(
        ':'.join(pair) for pair in (_COTANGENT, _SPILL))
    # PB's grammar: ROOT_ENV:MAX_ENV items, each variable sealed beside it,
    # each ceiling a positive decimal byte count.
    for item in sealed['PRISMABUILD_LOCAL_SCRATCH_PAIRS'].split(','):
        root_env, max_env = item.split(':')
        assert sealed[root_env].startswith('/')
        assert sealed[max_env].isdecimal() and int(sealed[max_env]) > 0
    # The list is a request option, never forwarded into the container.
    wrapped = argv[argv.index('--') + 1:]
    assert not any('PRISMABUILD_LOCAL_SCRATCH_PAIRS' in part for part in wrapped)


def test_a_row_without_scratch_seals_todays_request(tmp_path, campaign, monkeypatch):
    """No scratch declared: no pair list, and the argv is what it was."""
    import dispatch_joint_quanta as dispatch
    build = _scratch_quantum_argv(tmp_path, campaign, {}, ())
    argv = build()
    assert _outer_env(argv) == [dispatch.DEV_MODE_ENV]
    container = sys.modules[dispatch.local_scratch_environment.__module__]
    monkeypatch.setattr(container, 'LOCAL_SCRATCH_KINDS', ())
    assert build() == argv


@pytest.mark.parametrize('names', [_COTANGENT, _SPILL], ids=['cotangent', 'spill'])
@pytest.mark.parametrize('bound', [None, '0', '-5', '1.5'], ids=['unset', 'zero', 'negative', 'fraction'])
def test_a_scratch_root_without_a_positive_bound_is_refused(
        tmp_path, campaign, names, bound):
    import dispatch_joint_quanta as dispatch
    root = '/home/rob/pb-scratch/unbounded'
    env = {names[0]: root}
    if bound is not None:
        env[names[1]] = bound
    build = _scratch_quantum_argv(tmp_path, campaign, env, (root,))
    with pytest.raises(dispatch.DispatchRefused, match='positive byte ceiling'):
        build()


def test_a_spec_cannot_declare_the_pair_list(tmp_path, campaign):
    import dispatch_joint_quanta as dispatch
    root = '/home/rob/pb-scratch/spill'
    env = {_SPILL[0]: root, _SPILL[1]: str(1 << 30),
           'PRISMABUILD_LOCAL_SCRATCH_PAIRS': ''}
    build = _scratch_quantum_argv(tmp_path, campaign, env, (root,))
    with pytest.raises(dispatch.DispatchRefused, match='derived from the declared'):
        build()


def test_two_scratch_kinds_cannot_share_a_root(tmp_path, campaign):
    import dispatch_joint_quanta as dispatch
    root = '/home/rob/pb-scratch/shared'
    env = {_COTANGENT[0]: root, _COTANGENT[1]: str(1 << 30),
           _SPILL[0]: root, _SPILL[1]: str(1 << 30)}
    build = _scratch_quantum_argv(tmp_path, campaign, env, (root,))
    with pytest.raises(dispatch.DispatchRefused, match='same root'):
        build()


@pytest.mark.parametrize('regime,spill,message', [
    ('capture_batch=2', True, None),
    ('capture_batch=2,accumulation=operator_gemm,chunk_rows=65536', True, None),
    ('capture_batch=2', False, 'declare the spill'),
    ('capture_batch=1', True, 'spells the default'),
    ('capture_batch=two', True, 'canonical integer'),
])
def test_stage_b_replay_regime_is_validated_and_sealed_in_the_spec(
        tmp_path, campaign, regime, spill, message):
    """The #994 replay regime rides the one spec every quantum shares."""
    import dispatch_joint_quanta as dispatch
    from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV
    root = '/home/rob/pb-scratch/glm-stageb-spill'
    env = {REPLAY_REGIME_ENV: regime}
    if spill:
        env.update({'PRISMAQUANT_STAGE_B_SPILL_ROOT': root,
                    'PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES': str(200 << 30)})
    spec = {'container': {'image': 'sha256:' + '0' * 64,
             'mounts': [{'source': root, 'target': root, 'readonly': False}]}, 'env': env}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    args = dict(record_path=path, output_root=tmp_path / 'out')
    if message is not None:
        with pytest.raises(dispatch.DispatchRefused, match=message):
            quantum_argv(record, **args)
        return
    argv = quantum_argv(record, **args)
    actual = json.loads(argv[argv.index('--spec') + 1])
    assert actual['env'][REPLAY_REGIME_ENV] == regime


@pytest.mark.parametrize('value,message', [
    (None, None), ('off', None), ('on', 'spells the default'), ('0', 'the only value')])
def test_the_bf16_reduction_setting_is_validated_and_sealed_in_the_spec(
        tmp_path, value, message):
    """The PQ #1028 flag rides the one spec every row shares, like the regime."""
    import dispatch_joint_quanta as dispatch
    from prismaquant.matmul_arithmetic import BF16_REDUCTION_ENV
    spec = {'container': {'image': 'sha256:' + '0' * 64, 'mounts': []},
            'env': {} if value is None else {BF16_REDUCTION_ENV: value}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    payload = ['python3', '-m', 'prismaquant.joint_cost_quantum', '--quantum', 'q.json']
    head_only = [('head', dispatch.HEAD_PROGRESS_GRACE_S)]
    if message is not None:
        with pytest.raises(dispatch.DispatchRefused, match=message):
            dispatch._container_wrap(path, payload, progress=head_only)
        return
    argv, _image = dispatch._container_wrap(path, payload, progress=head_only)
    sealed = json.loads(argv[argv.index('--spec') + 1])
    assert sealed['env'].get(BF16_REDUCTION_ENV) == value


@pytest.mark.parametrize('value,message', [
    (None, None), ('kda_gram_v1', None), ('fallback', None),
    ('kda_gram_fp32_v1', 'names no known capture kernel'), ('', 'names no known capture kernel')])
def test_the_kda_capture_kernel_setting_is_validated_and_sealed_in_the_spec(
        tmp_path, value, message):
    """The PQ #1199 launch setting rides the one spec every row shares, like the flag."""
    import dispatch_joint_quanta as dispatch
    from prismaquant.glm_kda_capture_kernel import KDA_KERNEL_ENV
    spec = {'container': {'image': 'sha256:' + '0' * 64, 'mounts': []},
            'env': {} if value is None else {KDA_KERNEL_ENV: value}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    payload = ['python3', '-m', 'prismaquant.joint_cost_quantum', '--quantum', 'q.json']
    head_only = [('head', dispatch.HEAD_PROGRESS_GRACE_S)]
    if message is not None:
        with pytest.raises(dispatch.DispatchRefused, match=message):
            dispatch._container_wrap(path, payload, progress=head_only)
        return
    argv, _image = dispatch._container_wrap(path, payload, progress=head_only)
    sealed = json.loads(argv[argv.index('--spec') + 1])
    assert sealed['env'].get(KDA_KERNEL_ENV) == value


@pytest.mark.parametrize('regime,chain_batch_size,refused', [
    ('capture_batch=2', 1, True),
    ('capture_batch=2,accumulation=operator_gemm,chunk_rows=65536', 1, True),
    ('accumulation=operator_gemm,chunk_rows=65536', 1, False),
    # The campaign regime under R13's batch-4 chain regime, and off it.
    ('capture_batch=4,accumulation=operator_gemm,chunk_rows=65536', 4, False),
    ('capture_batch=4,accumulation=operator_gemm,chunk_rows=65536', 1, True),
    ('accumulation=operator_gemm,chunk_rows=65536', 4, True),
])
def test_a_band_serial_producer_captures_at_the_chains_batch_size_at_dispatch(
        tmp_path, regime, chain_batch_size, refused):
    """A #996 producer row captures at its slice's chain batch size (#994, #997).

    The row's payload carries ``--emit-adjoint-handoff``; the one spec every
    row shares carries the regime, and the row names the batch size its
    slice rolls the chain at. The same spec wraps a row that emits no
    handoff, at any chain batch size.
    """
    import dispatch_joint_quanta as dispatch
    from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV
    root = '/home/rob/pb-scratch/glm-stageb-spill'
    spec = {'container': {'image': 'sha256:' + '0' * 64,
                          'mounts': [{'source': root, 'target': root, 'readonly': False}]},
            'env': {REPLAY_REGIME_ENV: regime, 'PRISMAQUANT_STAGE_B_SPILL_ROOT': root,
                    'PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES': str(200 << 30)}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    payload = ['python3', '-m', 'prismaquant.joint_cost_quantum', '--quantum', 'q.json']
    head_only = [('head', dispatch.HEAD_PROGRESS_GRACE_S)]
    argv, _image = dispatch._container_wrap(path, payload, progress=head_only)
    assert argv[-len(payload):] == payload
    producer = [*payload, '--emit-adjoint-handoff']
    # A producer row that names no chain batch size refuses outright.
    with pytest.raises(dispatch.DispatchRefused, match='names the chain batch size'):
        dispatch._container_wrap(path, producer, progress=head_only)
    if refused:
        with pytest.raises(dispatch.DispatchRefused,
                           match=f'batch size {chain_batch_size}; a band-serial '
                                 'handoff must equal the plane that rebuild ends on'):
            dispatch._container_wrap(path, producer, progress=head_only,
                                     handoff_chain_batch_size=chain_batch_size)
    else:
        argv, _image = dispatch._container_wrap(
            path, producer, progress=head_only,
            handoff_chain_batch_size=chain_batch_size)
        assert argv[-len(producer):] == producer
