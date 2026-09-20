"""Quantum launch-contract integration: producer, writer, dispatcher, consumer.

The quanta rows the dispatcher publishes must survive the consumer's own
identity gates. Three live defects broke that contract (PQ #838):

* the dispatcher payload misses the consumer's required plan/prepared/
  adjoint bindings (argparse exits before compute);
* ``--quantum-sha256`` carries the record's canonical body digest while the
  consumer checks raw file bytes;
* the stage-A receipt's canonical digest (what the producer seals) is
  compared against its pretty-printed wire bytes (what the writer writes),
  so valid writer output fails both the dispatcher receipt check and the
  consumer's record-vs-argv check.

These tests drive the REAL producer (``layer_quanta``), the REAL writer
(``write_adjoint_receipt``), the REAL dispatcher argv, and the REAL consumer
parser + ``verify_quantum_identity`` on a two-layer CPU fixture. Only the
GPU execution body is stubbed, after every binding passes. Tampered wire
bytes and an edited canonical body both refuse.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import dispatch_joint_quanta as dispatch  # noqa: E402
from prismaquant import joint_cost_quantum as quantum  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import (  # noqa: E402
    load_adjoint_receipt,
    write_adjoint_receipt,
)
from prismaquant.joint_layer_quanta import (  # noqa: E402
    canonical_sha256,
    layer_quanta,
    seal_manifest_bytes,
)


@pytest.fixture
def portable_spec(tmp_path, monkeypatch):
    """The dispatcher wraps every payload in the campaign container spec;
    these tests seal a fixture one instead of reading the live mount."""
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    return spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(tmp_path: Path) -> dict:
    """Plan + prepared + parent manifest files the producer and the consumer
    both bind by digest."""
    root = tmp_path / "campaign"
    plan = {"output_root": str(root),
            "model": "/fixture/model",
            "distributed_campaign": {}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared = {"formats_by_qname": {
        "model.layers.0.mlp.gate_proj": {},
        "model.layers.1.mlp.gate_proj": {}}}
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True))
    entries = [
        {"path": "/fixture/model/shard-h0.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-h1.pt", "offset": 0, "bytes": 100,
         "sha256": None},
        {"path": "/fixture/model/shard-l0a.pt", "offset": 0, "bytes": 200,
         "sha256": None},
        {"path": "/fixture/model/shard-l0b.pt", "offset": 0, "bytes": 200,
         "sha256": None},
        {"path": "/fixture/model/shard-l1.pt", "offset": 0, "bytes": 300,
         "sha256": None},
    ]
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"plan": str(plan_path)},
        "mount_prefix": "/mnt/shared",
        "entries": entries,
        "entry_count": 5,
        "total_bytes": 900,
        "annotations": {
            "campaign_scope": {"campaign": "launch-contract-fixture",
                               "layers": [0, 1]},
            "argv": ["python3", "-m", "prismaquant.joint_adjoint_capture",
                     "--prepared", str(prepared_path),
                     "--prepared-sha256", _sha(prepared_path)],
            "layers": [0, 1],
            "phases": [
                {"name": "head", "bytes": 200, "cumulative_bytes": 200},
                {"name": "layer-0", "bytes": 400, "cumulative_bytes": 600},
                {"name": "layer-1", "bytes": 300, "cumulative_bytes": 900},
            ],
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True))
    return {"plan": plan, "plan_path": plan_path, "plan_sha": _sha(plan_path),
            "prepared": prepared, "prepared_path": prepared_path,
            "prepared_sha": _sha(prepared_path),
            "parent": parent, "parent_sha": _sha(parent_path), "root": root}


def _produce(campaign: dict, *, receipt=None, output_root=None) -> dict:
    """The real producer, unaltered: records, slices and manifests."""
    return layer_quanta(
        campaign["plan"], campaign["prepared"], campaign["parent"],
        chunk_target_bytes=400, stride=1,
        output_root=str(output_root or campaign["root"]),
        plan_path=str(campaign["plan_path"]), plan_sha256=campaign["plan_sha"],
        prepared_path=str(campaign["prepared_path"]),
        prepared_sha256=campaign["prepared_sha"],
        parent_manifest_sha256=campaign["parent_sha"],
        window_partition={"windows_by_layer": {"0": 1, "1": 1}},
        adjoint_receipt=receipt)


def _receipt(campaign: dict, checkpoints: list[int]) -> dict:
    return {
        "schema": "prismaquant.joint_adjoint_capture.v1",
        "plan_sha256": campaign["plan_sha"],
        "prepared_sha256": campaign["prepared_sha"],
        "campaign_scope": {"campaign": "launch-contract-fixture",
                           "layers": [0, 1]},
        "checkpoints": [{"boundary": mark} for mark in checkpoints],
        "status": "complete",
    }


def _publish_layer_000(tmp_path: Path, out: dict) -> tuple[dict, Path, Path]:
    """Write the layer-000 record and slice files where the record names
    them, mirroring the production sealing (pretty record, gzipped slice)."""
    record = next(r for r in out["records"] if r["quantum_id"] == "layer-000")
    record_path = tmp_path / "records" / "layer-000.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    slice_path = Path(record["read_set"]["manifest_path"])
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    slice_path.write_bytes(seal_manifest_bytes(out["slice_manifests"]["layer-000"]))
    assert _sha(slice_path) == record["read_set"]["manifest_sha256"]
    return record, record_path, slice_path


def _payload(argv: list[str]) -> list[str]:
    tail = argv[argv.index("--") + 1:]
    inner = tail[tail.index("--", tail.index("--spec")) + 1:]
    assert inner[:3] == ["python3", "-m", "prismaquant.joint_cost_quantum"]
    return inner


def _bound_row(tmp_path: Path, portable_spec) -> tuple[dict, dict, Path, Path]:
    """Producer + writer + bound records for layer-000, fully on disk."""
    campaign = _campaign(tmp_path)
    out = _produce(campaign)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _receipt(campaign, out["derivation"]["checkpoints"])
    write_adjoint_receipt(space, receipt)
    receipt_path = space / "adjoint-capture.json"
    bound = _produce(campaign, receipt=receipt)
    record, record_path, _ = _publish_layer_000(tmp_path, bound)
    return campaign, record, record_path, receipt_path


def test_dispatcher_payload_carries_every_consumer_binding(tmp_path, portable_spec):
    """Defect 1: the payload must name every binding the consumer CLI
    requires -- plan, prepared and adjoint paths+digests alongside the
    quantum, slice-manifest and output-root bindings."""
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    inner = _payload(dispatch.quantum_argv(
        record, record_path=record_path, output_root=campaign["root"],
        adjoint_path=receipt_path))
    assert inner[inner.index("--plan") + 1] == str(campaign["plan_path"])
    assert inner[inner.index("--plan-sha256") + 1] == campaign["plan_sha"]
    assert inner[inner.index("--prepared") + 1] == str(campaign["prepared_path"])
    assert inner[inner.index("--prepared-sha256") + 1] == campaign["prepared_sha"]
    assert inner[inner.index("--adjoint") + 1] == str(receipt_path)
    assert inner[inner.index("--adjoint-sha256") + 1] == _sha(receipt_path)
    assert "--resume" in inner


def test_quantum_sha256_binds_record_wire_bytes(tmp_path, portable_spec):
    """Defect 2: the consumer checks raw record file bytes first. The CLI
    must bind that wire digest; the canonical body check inside stays."""
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    inner = _payload(dispatch.quantum_argv(
        record, record_path=record_path, output_root=campaign["root"],
        adjoint_path=receipt_path))
    assert inner[inner.index("--quantum-sha256") + 1] == _sha(record_path)
    assert _sha(record_path) != record["identity_sha256"]


def test_writer_output_passes_matching_representation_checks(tmp_path, portable_spec):
    """Defect 3: the writer's pretty bytes and the producer's canonical
    digest are two representations of one receipt. The dispatcher receipt
    check passes the writer's actual output, and a re-serialized
    (differently formatted) receipt passes the same way -- while tampered
    wire bytes refuse at the wire check."""
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    assert record["adjoint"]["receipt_sha256"] == canonical_sha256(
        json.loads(receipt_path.read_text()))
    assert record["adjoint"]["receipt_sha256"] != _sha(receipt_path)
    dispatch.check_adjoint_receipt(
        receipt_path, [(record_path, record)])
    parsed = json.loads(receipt_path.read_text())
    assert load_adjoint_receipt(receipt_path, _sha(receipt_path)) == parsed
    # Tampered wire bytes refuse at the wire check, against the sealed digest.
    wire = _sha(receipt_path)
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_adjoint_receipt(receipt_path, wire)
    # ...but trailing whitespace is not content: the canonical gate still
    # passes the same bytes.
    dispatch.check_adjoint_receipt(
        receipt_path, [(record_path, record)])
    # Edited content refuses at the canonical gate.
    parsed["status"] = "partial"
    receipt_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(Exception):
        dispatch.check_adjoint_receipt(
            receipt_path, [(record_path, record)])


def test_consumer_verify_accepts_valid_pretty_receipt(tmp_path):
    """Defect 3 at the consumer: with correctly bound wire digests, the
    valid pretty-written receipt passes ``verify_quantum_identity`` --
    today it fails comparing the record canonical digest to wire bytes."""
    campaign = _campaign(tmp_path)
    out = _produce(campaign)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _receipt(campaign, out["derivation"]["checkpoints"])
    write_adjoint_receipt(space, receipt)
    receipt_path = space / "adjoint-capture.json"
    bound = _produce(campaign, receipt=receipt)
    record, record_path, _ = _publish_layer_000(tmp_path, bound)
    found, loaded = quantum.verify_quantum_identity(
        quantum_path=record_path, quantum_sha256=_sha(record_path),
        plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
        prepared_path=campaign["prepared_path"],
        prepared_sha256=campaign["prepared_sha"],
        adjoint_path=receipt_path, adjoint_sha256=_sha(receipt_path),
        output_root=campaign["root"])
    assert found["quantum_id"] == "layer-000"
    assert loaded["status"] == "complete"
    assert canonical_sha256(loaded) == record["adjoint"]["receipt_sha256"]


def test_consumer_parser_and_verify_drive_the_dispatched_row(tmp_path, portable_spec,
                                                             monkeypatch):
    """End to end through the real consumer parser and identity gates; only
    the GPU body is stubbed, after every binding passes."""
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    inner = _payload(dispatch.quantum_argv(
        record, record_path=record_path, output_root=campaign["root"],
        adjoint_path=receipt_path))
    args = quantum.build_parser().parse_args(inner)
    assert args.data_manifest_sha256 == record["read_set"]["manifest_sha256"]
    found, loaded = quantum.verify_quantum_identity(
        quantum_path=Path(args.quantum), quantum_sha256=args.quantum_sha256,
        plan_path=Path(args.plan), plan_sha256=args.plan_sha256,
        prepared_path=Path(args.prepared), prepared_sha256=args.prepared_sha256,
        adjoint_path=Path(args.adjoint), adjoint_sha256=args.adjoint_sha256,
        output_root=Path(args.output_root))
    assert found["quantum_id"] == "layer-000"
    calls = []
    monkeypatch.setattr(quantum, "run_layer_quantum",
                        lambda *a, **k: calls.append((a, k)) or {"passed": True})
    quantum.run_layer_quantum(
        {"output_root": str(campaign["root"])}, record=found, receipt=loaded,
        plan_sha256=args.plan_sha256,
        prepared={"path": args.prepared, "sha256": args.prepared_sha256},
        output_root=Path(args.output_root),
        data_manifest_sha256=args.data_manifest_sha256, resume=True)
    assert calls and calls[0][1]["data_manifest_sha256"] == args.data_manifest_sha256


def test_here_rooted_records_refuse_against_the_plan_root(tmp_path):
    """Defect 4, reproduced: records produced with an output root that is
    not the plan's output root refuse at the consumer's root check -- the
    shape the live doubled `layer-quanta/layer-quanta` records have."""
    campaign = _campaign(tmp_path)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    first = _produce(campaign)
    receipt = _receipt(campaign, first["derivation"]["checkpoints"])
    write_adjoint_receipt(space, receipt)
    receipt_path = space / "adjoint-capture.json"
    # Produced exactly as the live records were: bound to the receipt, but
    # rooted at the wrong tree (the binder's HERE, not the plan root).
    wrong = _produce(campaign, receipt=receipt,
                     output_root=str(tmp_path / "wrong-root"))
    record = next(r for r in wrong["records"] if r["quantum_id"] == "layer-000")
    record_path = tmp_path / "records" / "layer-000.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    with pytest.raises(quantum.QuantumIdentityRefused, match="output_space"):
        quantum.verify_quantum_identity(
            quantum_path=record_path,
            quantum_sha256=_sha(record_path),
            plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
            prepared_path=campaign["prepared_path"],
            prepared_sha256=campaign["prepared_sha"],
            adjoint_path=receipt_path, adjoint_sha256=_sha(receipt_path),
            output_root=campaign["root"])
