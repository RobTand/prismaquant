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
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_adjoint_slices import stage_a_slice, write_adjoint_slice
from prismaquant.joint_layer_quanta import adjoint_binding_fields

from test_stage_b_band_binding import band_from_receipt, synthetic_receipt
from stage_a_spool_spec import (SPOOL_MOUNT, SPOOL_ROOT, SPOOL_WINDOW_BYTES, stage_a_plan,
                                with_spool, write_declaration)

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
    stage_a_argv as _dispatch_stage_a_argv,
)


def stage_a_argv(manifest, campaign, **kwargs):
    """The shipped Stage A row builder, given the launch-bound produced declaration it requires."""
    kwargs.setdefault("produced_declaration", write_declaration(Path(campaign["plan_path"]).parent))
    return _dispatch_stage_a_argv(manifest, campaign, timeout_s=3600, **kwargs)

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


def _zero_budget():
    from prismaquant.joint_retained_window_plan import HOST_RESIDENT_BUDGET_FIELDS
    return {name: 0 for name in (*HOST_RESIDENT_BUDGET_FIELDS, "retained_render_cap_bytes")}


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
    # These fixtures exercise unrelated dispatcher contracts without a GPU
    # measurement. Pilot admission itself uses the production defaults in
    # test_joint_dispatch_pilot.py; never invent a passing pilot here.
    from stage_a_spool_spec import write_declaration
    argv = ["--timeout-s", "3600", "--stage-a-produced-declaration", str(write_declaration(records_dir.parent)), "--records", str(records_dir), "--output-root", str(output_root),
            "--force-unverified-pilot"]
    if receipt is not None:
        argv += ["--adjoint-receipt", str(receipt)]
    return argv + list(extra)


# -- neutral application row intent (§5.2, preserving placement requirements) --


@pytest.mark.parametrize("mode", ["0", "1"])
def test_dispatch_refuses_edited_record_before_publishing(tmp_path, records_dir, monkeypatch, mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    path = records_dir / "layer-000.json"
    record = json.loads(path.read_text())
    record["campaign"]["plan_path"] = "/edited-plan"
    path.write_text(json.dumps(record))
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out"), _gateway=gateway) == 3
    assert gateway.submitted == []


@pytest.mark.parametrize("defect", ["edited", "schema", "campaign", "receipt"])
def test_load_records_names_the_offending_file_among_valid_records(records_dir, defect):
    from dispatch_joint_quanta import load_records
    from prismaquant.joint_campaign_ledger import seal_quantum_record

    path = records_dir / "layer-001.json"
    record = json.loads(path.read_text())
    if defect in {"edited", "campaign"}:
        record["campaign"]["plan_path"] = "/other-plan"
    elif defect == "schema":
        record["schema"] = "foreign"
    else:
        record["adjoint"]["receipt_sha256"] = "f" * 64
    if defect != "edited":
        record = seal_quantum_record(record)
    path.write_text(json.dumps(record))
    with pytest.raises(DispatchRefused) as refused:
        load_records(records_dir)
    assert str(refused.value).startswith(f"{path}: ")


def test_quantum_argv_matches_the_pinned_submission_shape(tmp_path, campaign):
    """Each quantum keeps its GB10 placement, resource demand and container command."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    argv = quantum_argv(record, timeout_s=3600, record_path=record_path,
                        output_root=Path("/out/root"))
    assert argv["tags"] == ["gb10"] and CONSUMER_TAGS == ("gb10",)
    assert CHUNK_PROGRESS_GRACE_S == 900
    assert argv["priority"] == SUBMISSION_PRIORITY
    assert argv["demand"] == {"gpu": 1, "mem_gb": 104, "cpu": 10}
    assert argv["timeout_s"] == 3600
    assert argv["env"] == {}
    tail = argv["argv"]
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


def _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway, *, terminal=True):
    manifest = _adjoint_manifest(records_dir, campaign, name="adjoint.data-manifest.json.gz", gzip_bytes=True)
    assert main(_argv(records_dir, out, None, ("--adjoint-manifest", str(manifest))), _gateway=gateway) == 0
    if terminal:
        gateway.mark_terminal(gateway.submitted[-1]["job_id"])
    gateway.submitted.clear()


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
    out = tmp_path / "out"
    _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway)
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
        argv = row["row"]
        tags = argv["tags"]
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
    assert "--prefetch-override" not in plain["argv"]
    assert "--resume" in plain["argv"]
    assert "--produced-declaration" in plain["argv"]

    override = Path("/mnt/shared/joint-panel/prefetch-override-v10.json")
    widened = stage_a_argv(manifest, campaign, prefetch_override=override)
    # The application row retains all execution requirements;
    # the override rides inside the container payload only.
    assert {k: v for k, v in widened.items() if k != "argv"} == {k: v for k, v in plain.items() if k != "argv"}
    payload = widened["argv"]
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
    code = main(_argv(records_dir, out) + [
                 "--state", str(tmp_path / "state.json"),
                 "--adjoint-manifest", str(_adjoint_manifest(tmp_path, campaign)),
                 "--stage-a-prefetch-override", str(override)],
                _gateway=gateway)
    assert code == 0
    stage_a_rows = [row for row in gateway.submitted if row["kind"] == "stage-a"]
    assert len(stage_a_rows) == 1
    argv = stage_a_rows[0]["row"]["argv"]
    assert argv[argv.index("--prefetch-override") + 1] == str(override)
    events = [json.loads(line) for line
              in (tmp_path / "state.json").read_text().splitlines() if line]
    events = [event for event in events if event["event"] == "stage-a-submitted"]
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
    assert gateway.submitted[0]["row"]["tags"][0] == "sparky"


def test_stale_stage_a_receipt_refuses(tmp_path, campaign, records_dir):
    """A receipt from another plan revision fails closed, exit 3, and
    publishes nothing."""
    gateway = FakeGateway(terminal=True)
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign, plan_sha256="f" * 64))
    _stamp_receipt(records_dir, receipt_path)
    out = tmp_path / "out"
    _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway)
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 3
    assert gateway.submitted == []
    assert not (out / "layer-quanta" / "layer-002").exists()


def test_real_gateway_refuses_receipt_without_native_producer(tmp_path, campaign, records_dir, capsys):
    from unittest.mock import patch
    from dispatch_joint_quanta import Gateway
    from prismaquant import tessera_campaign_rows
    receipt_path = tmp_path / 'adjoint-capture.json'
    _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, receipt_path)
    with patch.object(tessera_campaign_rows, 'submit_rows') as submit:
        assert main(_argv(records_dir, tmp_path/'out', receipt_path), _gateway=Gateway()) == 3
        submit.assert_not_called()
    assert 'no saved native producer' in capsys.readouterr().err


def test_quanta_publish_once_the_receipt_lands(tmp_path, campaign, records_dir):
    gateway = FakeGateway(terminal=True)
    receipt_path = tmp_path / "adjoint-capture.json"
    digest = _write_receipt(receipt_path, _receipt(campaign))
    _stamp_receipt(records_dir, receipt_path)
    out = tmp_path / "out"
    _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway)
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    kinds = [call["kind"] for call in gateway.submitted]
    assert kinds == ["quantum", "quantum", "quantum"]


# -- checkpoint bands (PQ #993): publish a band's layers before the receipt ------


def _write_band(tmp_path, receipt, boundary):
    path = tmp_path / "bands" / f"band-{boundary:03d}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(band_from_receipt(receipt, boundary)))
    return path


def test_a_band_waits_for_native_success_even_if_its_bytes_arrived_early(
        tmp_path, campaign, records_dir, capsys):
    receipt = _receipt(campaign)
    receipt_path = tmp_path / "adjoint-capture.json"
    _write_receipt(receipt_path, receipt)
    _stamp_receipt(records_dir, receipt_path)
    receipt_path.unlink()
    band = _write_band(tmp_path, receipt, N_LAYERS)
    gateway = FakeGateway(terminal=False)
    out = tmp_path / "out"
    _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway, terminal=False)
    argv = _argv(records_dir, out, None, ("--adjoint-band", str(band)))
    assert main(argv, _gateway=gateway) == 0
    resumed, = gateway.submitted
    assert resumed["kind"] == "stage-a"
    assert resumed["job_id"] == next(iter(gateway._saved.values()))["job_id"]
    gateway.submitted.clear()
    gateway.mark_terminal(next(iter(gateway._saved.values()))["job_id"])
    assert main(argv, _gateway=gateway) == 0
    assert [call.get("quantum_id") for call in gateway.submitted] == ["layer-002"]


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
    assert '"timeout_s": 3600' in printed
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
    out = tmp_path / "out"
    _submit_stage_a_fixture(tmp_path, campaign, records_dir, out, gateway)
    assert main(_argv(records_dir, out, receipt_path), _gateway=gateway) == 0
    first = list(gateway.submitted)
    assert len(first) == N_LAYERS
    for call in first:
        gateway.mark_terminal(call["job_id"])
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
    tail = argv["argv"]
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_adjoint_capture"]
    return inner


def _envelope_envs(argv):
    return [f"{k}={v}" for k, v in argv["env"].items()]


def _write_fixture_spec(spec):
    import dispatch_joint_quanta
    dispatch_joint_quanta.SPEC_PATH.write_text(json.dumps(spec))


def test_stage_a_binds_the_produced_declaration(tmp_path, campaign):
    """The Stage A row binds its produced declaration without a spool environment."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = stage_a_argv(manifest, campaign)
    assert _envelope_envs(argv) == []
    flag = argv["argv"].index("--produced-declaration")
    assert argv["argv"][flag + 1].endswith("stage-a-declaration.json")
    tail = argv["argv"]
    sealed = json.loads(tail[tail.index("--spec") + 1])
    assert not [name for name in sealed["env"] if name.startswith("PRISMABUILD_")]
    assert SPOOL_MOUNT in sealed["container"]["mounts"]


def test_stage_a_requires_the_launch_bound_declaration(tmp_path, campaign):
    manifest = _adjoint_manifest(tmp_path, campaign)
    with pytest.raises(DispatchRefused, match="explicit launch-bound produced declaration"):
        _dispatch_stage_a_argv(manifest, campaign, timeout_s=3600)


def test_the_plans_two_plane_window_is_the_fixture_spool_bound(tmp_path, campaign):
    """Ported equivalent of the sealed two-plane window: the derivation, not its PB envelope."""
    from dispatch_joint_quanta import stage_a_spool_window_bytes
    assert stage_a_spool_window_bytes(campaign) == SPOOL_WINDOW_BYTES


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


def test_a_split_quantum_owns_two_planes_of_its_own_batches(tmp_path, campaign):
    """PQ #738: a split quantum's window covers its own range's planes."""
    from dispatch_joint_quanta import stage_a_spool_window_bytes
    assert stage_a_spool_window_bytes(campaign, (4, 8)) == SPOOL_WINDOW_BYTES // 2
    with pytest.raises(DispatchRefused, match="not inside the plan's 8 batches"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign, batch_range=(4, 9))


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
    envelope = argv
    assert envelope["demand"]["mem_gb"] == int(expected)


@pytest.mark.parametrize("aggregate", [0, -1, 1.5, True, "108447924224"])
def test_stage_a_refuses_a_malformed_memory_bound(tmp_path, campaign, aggregate):
    plan_path = Path(campaign["plan_path"])
    plan = json.loads(plan_path.read_text())
    plan["aggregate_memory_bytes"] = aggregate
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(DispatchRefused, match="aggregate_memory_bytes"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)


def test_stage_a_refuses_a_spec_naming_a_retired_spool_environment(tmp_path, campaign):
    """The retired PB spool names refuse at dispatch; the bound is the declaration's."""
    _write_fixture_spec(with_spool({"container": {"image": "sha256:" + "0" * 64},
                                     "env": {"PRISMABUILD_PRODUCED_SPOOL_ROOT": SPOOL_ROOT}}))
    with pytest.raises(DispatchRefused, match="retired service environment"):
        stage_a_argv(_adjoint_manifest(tmp_path, campaign), campaign)


def test_a_quantum_row_seals_no_spool_environment(tmp_path, campaign):
    """A quantum row has no spool environment. Retired spool settings refuse."""
    slices = tmp_path / "slices"
    slices.mkdir()
    record = _bind(_record(campaign, 1, slice_dir=slices), _receipt(campaign),
                   tmp_path / "adjoint-slices")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    argv = quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root"))
    assert _envelope_envs(argv) == []
    _write_fixture_spec({"container": {"image": "sha256:" + "0" * 64}, "env": {}})
    argv = quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root"))
    assert _envelope_envs(argv) == []
    _write_fixture_spec(with_spool({"container": {"image": "sha256:" + "0" * 64},
                                     "env": {"PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": "1"}}))
    with pytest.raises(DispatchRefused, match="retired service environment"):
        quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root"))


def test_stage_a_argv_binds_manifest_digests_into_payload(tmp_path, campaign):
    """The payload binds the submitted manifest and its parent read-set digest."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    argv = stage_a_argv(manifest, campaign)
    inner = _payload_inner(argv)
    assert inner[inner.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    assert inner[inner.index("--read-manifest-sha256") + 1] == "d" * 64


def test_dispatch_requires_an_explicit_walltime(tmp_path, records_dir):
    argv = _argv(records_dir, tmp_path / "out")
    start = argv.index("--timeout-s")
    del argv[start:start + 2]
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_stage_a_argv_reads_gzip_manifest_bytes(tmp_path, campaign):
    """The default manifest filename is ``.json.gz``; the bound digest covers
    the stored wire bytes either way."""
    manifest = _adjoint_manifest(tmp_path, campaign, gzip_bytes=True,
                                 name="adjoint.data-manifest.json.gz")
    argv = stage_a_argv(manifest, campaign)
    inner = _payload_inner(argv)
    assert inner[inner.index("--data-manifest-sha256") + 1] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()


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
    payload digests and the
    state event recording both digests for the launch record."""
    manifest = _adjoint_manifest(tmp_path, campaign)
    gateway = FakeGateway()
    out = tmp_path / "out"
    assert main(_argv(records_dir, out) + ["--adjoint-manifest", str(manifest)],
                _gateway=gateway) == 0
    rows = [row for row in gateway.submitted if row["kind"] == "stage-a"]
    assert len(rows) == 1
    argv = rows[0]["row"]
    inner = _payload_inner(argv)
    data_id = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert inner[inner.index("--data-manifest-sha256") + 1] == data_id
    assert inner[inner.index("--read-manifest-sha256") + 1] == "d" * 64
    events = [json.loads(line) for line
              in (out / "layer-quanta" / "campaign-state.json").read_text().splitlines()
              if line]
    intent, submitted = events
    assert intent["event"] == "native-intent"
    assert intent["row"] == argv
    assert submitted["event"] == "stage-a-submitted"
    assert submitted["data_manifest_sha256"] == data_id
    assert submitted["read_manifest_sha256"] == "d" * 64


# -- quantum slice binding + v2 read plans (#835, root extension) --------------


def _quantum_payload(argv):
    tail = argv["argv"]
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    return inner


def test_quantum_argv_binds_slice_digest_not_campaign_parent(tmp_path, campaign):
    """The resolver binding is the slice manifest declared by this row,
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
    inner = _quantum_payload(quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root")))
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
        quantum_argv(record, timeout_s=3600, record_path=record_path,
                     output_root=Path("/out/root"))


def test_quantum_argv_refuses_absent_slice(tmp_path, campaign):
    """A sealed digest over bytes the dispatcher cannot read is a producer
    defect, refused before publishing -- never a row that binds blind."""
    record = _bind(_record(campaign, 1), _receipt(campaign), tmp_path / "adjoint-slices")
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps(record))
    with pytest.raises(DispatchRefused, match="unreadable"):
        quantum_argv(record, timeout_s=3600, record_path=record_path,
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
        quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root"))
    record = _bind(record, _receipt(campaign), tmp_path / "adjoint-slices")
    record_path.write_text(json.dumps(record))
    Path(record["adjoint"]["slice_path"]).write_bytes(b"{}")
    with pytest.raises(DispatchRefused, match="does not hash to the sealed digest"):
        quantum_argv(record, timeout_s=3600, record_path=record_path, output_root=Path("/out/root"))


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


def test_resource_policy_controls_container_and_native_row_intent(tmp_path, campaign, monkeypatch):
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    policy = {"limits": {"physical_bytes": 100 << 30, "host_bytes": 28 << 30, "gpu_bytes": 72 << 30},
              "budget": _zero_budget()}
    monkeypatch.setattr(resources, 'verify_policy', lambda _: policy)
    plan = {"stage_b_resource_policy": {"path": "/resource", "sha256": "0"*64},
        "source_prefetch": {"prefetch_workers": 1}, "execution": {"operator_windows": {"prefetch_workers": 4}}}
    raw = json.dumps(plan).encode(); Path(campaign['plan_path']).write_bytes(raw)
    campaign['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    spec = {"container": {"image": "sha256:" + "0"*64, "content_sha256": "b"*64},
        "container_admission_reference": "content:sha256:" + "c"*64, "cpu_memory_gb": 28,
        "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72", "PRISMAQUANT_LAYER_READ_THREADS": "10"}}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path),
                   _huge_plane_receipt(campaign, row_bytes=1 << 20),
                   tmp_path / 'adjoint-slices')
    path = tmp_path/'record.json'; path.write_text(json.dumps(record))
    args = dict(record_path=path, output_root=tmp_path/'out')
    argv = quantum_argv(record, timeout_s=3600, **args)
    assert argv["demand"]["mem_gb"] == 100
    assert argv["demand"]["cpu"] == 10
    spec['cpu_memory_gb'] = 32; dispatch.SPEC_PATH.write_text(json.dumps(spec))
    with pytest.raises(DispatchRefused, match='envelope differs'):
        quantum_argv(record, timeout_s=3600, **args)


def test_dev_mode_dispatches_a_re_declared_resource_plan(tmp_path, campaign, monkeypatch, capsys):
    """PQ #1147: a plan re-declared after the quantum was sealed stamps by default."""
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    policy = {"limits": {"physical_bytes": 100 << 30, "host_bytes": 28 << 30, "gpu_bytes": 72 << 30},
              "budget": _zero_budget()}
    monkeypatch.setattr(resources, 'verify_policy', lambda _: policy)
    plan = {"stage_b_resource_policy": {"path": "/resource", "sha256": "0"*64},
        "source_prefetch": {"prefetch_workers": 1}, "execution": {"operator_windows": {"prefetch_workers": 4}}}
    raw = json.dumps(plan).encode(); Path(campaign['plan_path']).write_bytes(raw)
    campaign['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    spec = {"container": {"image": "sha256:" + "0"*64, "content_sha256": "b"*64},
        "container_admission_reference": "content:sha256:" + "c"*64, "cpu_memory_gb": 28,
        "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72", "PRISMAQUANT_LAYER_READ_THREADS": "10"}}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path),
                   _huge_plane_receipt(campaign, row_bytes=1 << 20),
                   tmp_path / 'adjoint-slices')
    path = tmp_path/'record.json'; path.write_text(json.dumps(record))
    # The plan is re-declared after the record sealed its digest.
    plan["stage_b_resource_policy"]["sha256"] = "1"*64
    Path(campaign['plan_path']).write_bytes(json.dumps(plan).encode())
    args = dict(record_path=path, output_root=tmp_path/'out')
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    argv = quantum_argv(record, timeout_s=3600, **args)
    assert argv["demand"]["mem_gb"] == 100
    assert "[DEV-MODE] seal resource-bound plan differs" in capsys.readouterr().out
    # The row runs under the plan on disk, by the digest of its bytes.
    on_disk = hashlib.sha256(Path(campaign['plan_path']).read_bytes()).hexdigest()
    assert on_disk in " ".join(argv["argv"]) and campaign['plan_sha256'] not in " ".join(argv["argv"])
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(DispatchRefused, match='resource-bound plan differs'):
        quantum_argv(record, timeout_s=3600, **args)

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
        quantum_argv(record, timeout_s=3600, record_path=tmp_path/'record', output_root=tmp_path/'out')


def _huge_plane_receipt(campaign, row_bytes=8 << 30):
    """The fixture receipt with every cotangent row sized ``row_bytes``.

    The fixture plane is 2 probes x 2 batches, so 8 GiB rows make the 32 GiB
    plane the layer-044 v4 gate row held on a 28 GiB host (PQ #1141).
    """
    receipt = _receipt(campaign)
    for checkpoint in receipt["checkpoints"]:
        for row in checkpoint["activation_entries"]:
            row["tensor_bytes"] = row_bytes
            row["file_bytes"] = row_bytes + 65536
        checkpoint["cotangent_sha256"] = canonical_json_sha256(
            {key: checkpoint[key] for key in ("schema", "boundary", "session",
                                              "activation_entries", "shared_state_entries")},
            where="adjoint checkpoint")
    return receipt


def _plane_dispatch(tmp_path, campaign, monkeypatch, *, env=None, host=28 << 30, retained=0):
    """Dispatch one quantum under a verified 28 GiB-host policy; return the argv call."""
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    budget = _zero_budget()
    budget["retained_render_cap_bytes"] = retained
    policy = {"limits": {"physical_bytes": 100 << 30, "host_bytes": host, "gpu_bytes": 72 << 30},
              "budget": budget}
    monkeypatch.setattr(resources, 'verify_policy', lambda _: policy)
    plan = {"stage_b_resource_policy": {"path": "/resource", "sha256": "0"*64},
        "source_prefetch": {"prefetch_workers": 1}, "execution": {"operator_windows": {"prefetch_workers": 4}}}
    raw = json.dumps(plan).encode(); Path(campaign['plan_path']).write_bytes(raw)
    campaign['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    spec = {"container": {"image": "sha256:" + "0"*64, "content_sha256": "b"*64,
                          "mounts": [{"source": "/home/rob/pb-scratch/glm-stageb",
                                      "target": "/home/rob/pb-scratch/glm-stageb",
                                      "readonly": False}]},
        "container_admission_reference": "content:sha256:" + "c"*64, "cpu_memory_gb": host >> 30,
        "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72", "PRISMAQUANT_LAYER_READ_THREADS": "10",
                **(env or {})}}
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = _bind(_record(campaign, 1, slice_dir=tmp_path), _huge_plane_receipt(campaign),
                   tmp_path / 'adjoint-slices')
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    return lambda: quantum_argv(record, timeout_s=3600, record_path=path, output_root=tmp_path / 'out')


def test_a_host_held_cotangent_plane_over_the_cap_is_refused_at_dispatch(
        tmp_path, campaign, monkeypatch):
    """PQ #1141: no scratch pair means the plane sits on the host; 32 GiB > 28 GiB."""
    with pytest.raises(DispatchRefused, match='cotangent plane'):
        _plane_dispatch(tmp_path, campaign, monkeypatch)()


def test_a_cotangent_scratch_ceiling_below_the_plane_is_refused_at_dispatch(
        tmp_path, campaign, monkeypatch):
    env = {'PRISMAQUANT_STAGE_B_COTANGENT_ROOT': '/home/rob/pb-scratch/glm-stageb',
           'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES': str(16 << 30)}
    with pytest.raises(DispatchRefused, match='cotangent plane'):
        _plane_dispatch(tmp_path, campaign, monkeypatch, env=env)()


def test_a_cotangent_scratch_that_covers_the_plane_dispatches(tmp_path, campaign, monkeypatch):
    env = {'PRISMAQUANT_STAGE_B_COTANGENT_ROOT': '/home/rob/pb-scratch/glm-stageb',
           'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES': str(36 << 30)}
    assert "--" in _plane_dispatch(tmp_path, campaign, monkeypatch, env=env)()["argv"]


def test_a_host_held_plane_that_fits_beside_the_render_cap_dispatches(
        tmp_path, campaign, monkeypatch):
    assert '--' in _plane_dispatch(tmp_path, campaign, monkeypatch, host=64 << 30, retained=8 << 30)()["argv"]
    with pytest.raises(DispatchRefused, match='cotangent plane'):
        _plane_dispatch(tmp_path, campaign, monkeypatch, host=36 << 30, retained=8 << 30)()


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
            quantum_argv(record, timeout_s=3600, **args)
        return
    argv = quantum_argv(record, timeout_s=3600, **args)
    sealed = argv["env"]
    assert all(sealed.get(name) == value for name, value in env.items())
    actual = json.loads(argv["argv"][argv["argv"].index("--spec") + 1])
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
            quantum_argv(record, timeout_s=3600, **args)
        return
    argv = quantum_argv(record, timeout_s=3600, **args)
    sealed = argv["env"]
    assert all(sealed.get(name) == value for name, value in env.items())
    actual = json.loads(argv["argv"][argv["argv"].index("--spec") + 1])
    assert stage_b_spill_environment(actual, sealed) == env


_COTANGENT = ('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', 'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES')
_SPILL = ('PRISMAQUANT_STAGE_B_SPILL_ROOT', 'PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES')
_CACHE_2798 = ('PRISMAQUANT_CONTAINER_CACHE_ROOT', 'PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES')


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
    return lambda: quantum_argv(record, timeout_s=3600, record_path=path, output_root=tmp_path / 'out')


def _outer_env(argv):
    """The explicit environment assignments retained by the neutral row."""
    return [f"{k}={v}" for k, v in argv["env"].items()]


def test_every_declared_scratch_kind_is_sealed_with_its_ceiling(tmp_path, campaign):
    """PQ #1019: the sealed request carries each declared kind's root and a positive decimal ceiling."""
    cotangent, spill = '/home/rob/pb-scratch/cot', '/home/rob/pb-scratch/spill'
    env = {_COTANGENT[0]: cotangent, _COTANGENT[1]: str(36 << 30),
           _SPILL[0]: spill, _SPILL[1]: str(178_000_000_000)}
    argv = _scratch_quantum_argv(tmp_path, campaign, env, (cotangent, spill))()
    sealed = dict(item.split('=', 1) for item in _outer_env(argv))
    assert all(sealed.get(name) == value for name, value in env.items())
    for root_env, max_env in (_COTANGENT, _SPILL):
        assert sealed[root_env].startswith('/')
        assert sealed[max_env].isdecimal() and int(sealed[max_env]) > 0
    assert not [name for name in sealed if name.startswith('PRISMABUILD_')]


def test_a_stage_b_row_retains_spill_and_cotangent_ceilings(tmp_path, campaign):
    """The planned container keeps the independent cotangent and spill ceilings."""
    cotangent, spill = '/home/rob/pb-scratch/cot', '/home/rob/pb-scratch/spill'
    ceilings = {_COTANGENT[1]: 32 << 30, _SPILL[1]: 199_051_640_832}
    env = {_COTANGENT[0]: cotangent, _SPILL[0]: spill,
           **{name: str(value) for name, value in ceilings.items()}}
    argv = _scratch_quantum_argv(tmp_path, campaign, env, (cotangent, spill))()
    variables = dict(item.split('=', 1) for item in _outer_env(argv))
    assert {key: int(variables[key]) for key in ceilings} == ceilings


def test_a_row_without_scratch_seals_todays_request(tmp_path, campaign, monkeypatch):
    """No scratch declared: no pair list, and the argv is what it was."""
    import dispatch_joint_quanta as dispatch
    build = _scratch_quantum_argv(tmp_path, campaign, {}, ())
    argv = build()
    assert _outer_env(argv) == []
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
            quantum_argv(record, timeout_s=3600, **args)
        return
    argv = quantum_argv(record, timeout_s=3600, **args)
    actual = json.loads(argv["argv"][argv["argv"].index("--spec") + 1])
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
    argv = dispatch._container_wrap(path, payload, progress=head_only)
    sealed = json.loads(argv[argv.index("--spec") + 1])
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
    argv = dispatch._container_wrap(path, payload, progress=head_only)
    sealed = json.loads(argv[argv.index("--spec") + 1])
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
    argv = dispatch._container_wrap(path, payload, progress=head_only)
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
        argv = dispatch._container_wrap(
            path, producer, progress=head_only,
            handoff_chain_batch_size=chain_batch_size)
        assert argv[-len(producer):] == producer


@pytest.mark.parametrize('name', ['PRISMABUILD_LOCAL_SCRATCH_PAIRS', 'PRISMABUILD_LOCAL_SCRATCH_IO'])
def test_a_spec_naming_a_retired_scratch_setting_refuses(tmp_path, campaign, name):
    import dispatch_joint_quanta as dispatch
    root = '/home/rob/pb-scratch/spill'
    env = {_SPILL[0]: root, _SPILL[1]: str(1 << 30), name: ''}
    build = _scratch_quantum_argv(tmp_path, campaign, env, (root,))
    with pytest.raises(dispatch.DispatchRefused, match='retired service environment'):
        build()


@pytest.mark.parametrize('bad_pair', [_COTANGENT, _SPILL, _CACHE_2798], ids=['cotangent', 'spill', 'cache'])
def test_every_scratch_kind_needs_a_positive_ceiling(tmp_path, campaign, bad_pair):
    pairs = (_COTANGENT, _SPILL, _CACHE_2798)
    roots = ['/scratch/fixture-2798/' + name for name in ('cot', 'spill', 'cache')]
    env = {name: value for pair, root in zip(pairs, roots)
           for name, value in ((pair[0], root), (pair[1], '1073741825'))}
    env.update({bad_pair[1]: '0', 'PRISMAQUANT_TMPDIR': roots[1] + '/tmp'})
    build = _scratch_quantum_argv(tmp_path, campaign, env, roots)
    with pytest.raises(DispatchRefused, match='positive byte ceiling'):
        build()


def test_the_spill_ceiling_follows_the_rows_sealed_bound_from_one_spec_read(tmp_path, campaign, monkeypatch):
    """Changing the row reservation rewrites only the spill ceiling, and the row reads its spec once."""
    import dispatch_joint_quanta as dispatch
    root = str(tmp_path / 'spill')
    env = {_SPILL[0]: root, _SPILL[1]: '1', 'PRISMAQUANT_STAGE_B_REPLAY_REGIME': 'capture_batch=2'}
    build = _scratch_quantum_argv(tmp_path, campaign, env, (root,))
    bound = {'capture_batch': 2, 'reservation_bytes': 1073741825}
    monkeypatch.setattr(dispatch, '_sealed_spill_bound', lambda record: bound)
    read_text = Path.read_text
    spec_reads = []

    def once_per_row(path, *args, **kwargs):
        if path == dispatch.SPEC_PATH:
            spec_reads.append(path)
            assert len(spec_reads) == 1, 'ceiling and root must use the same spec parse'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', once_per_row)
    for reservation in (1073741825, 8_589_934_593):
        bound['reservation_bytes'] = reservation
        spec_reads.clear()
        argv = build()
        assert len(spec_reads) == 1
        outer = dict(item.split('=', 1) for item in _outer_env(argv))
        assert outer[_SPILL[0]] == root and outer[_SPILL[1]] == str(reservation)
        sealed = json.loads(argv["argv"][argv["argv"].index("--spec") + 1])
        assert sealed['env'][_SPILL[1]] == str(reservation)
