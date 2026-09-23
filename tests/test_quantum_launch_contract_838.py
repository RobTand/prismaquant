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

Since PQ #993 a record binds its own stage-A slice, never the receipt: the
slice file is canonical JSON, so its wire digest is the digest the producer
seals, and the receipt's formatting is not part of any quantum identity.

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
from prismaquant.joint_adjoint_checkpoints import write_adjoint_receipt  # noqa: E402
from prismaquant.joint_adjoint_slices import (  # noqa: E402
    AdjointSliceRefused,
    adjoint_slice_sha256,
    load_adjoint_slice,
    stage_a_slice,
    write_adjoint_slice,
)
from prismaquant.joint_layer_quanta import (  # noqa: E402
    layer_quanta,
    seal_manifest_bytes,
)

from test_stage_b_band_binding import synthetic_receipt  # noqa: E402


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
    receipt = synthetic_receipt(
        plan_sha256=campaign["plan_sha"], prepared_sha256=campaign["prepared_sha"],
        scope={"campaign": "launch-contract-fixture", "layers": [0, 1]},
        num_layers=2, stride=1)
    assert sorted(receipt["stride"]["boundaries"]) == sorted(checkpoints)
    return receipt


def _publish_layer_000(tmp_path: Path, out: dict) -> tuple[dict, Path, Path]:
    """Write the layer-000 record, read-set manifest and stage-A slice where
    the record names them, mirroring the production sealing (pretty record,
    gzipped read-set manifest, canonical slice)."""
    record = next(r for r in out["records"] if r["quantum_id"] == "layer-000")
    record_path = tmp_path / "records" / "layer-000.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    slice_path = Path(record["read_set"]["manifest_path"])
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    slice_path.write_bytes(seal_manifest_bytes(out["slice_manifests"]["layer-000"]))
    assert _sha(slice_path) == record["read_set"]["manifest_sha256"]
    if record["adjoint"].get("slice_path") is not None:
        write_adjoint_slice(record["adjoint"]["slice_path"],
                            out["adjoint_slices"]["layer-000"], layer=0)
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
        record, record_path=record_path, output_root=campaign["root"]))
    assert inner[inner.index("--plan") + 1] == str(campaign["plan_path"])
    assert inner[inner.index("--plan-sha256") + 1] == campaign["plan_sha"]
    assert inner[inner.index("--prepared") + 1] == str(campaign["prepared_path"])
    assert inner[inner.index("--prepared-sha256") + 1] == campaign["prepared_sha"]
    # PQ #993: the row names the quantum's slice, never the receipt.
    slice_path = Path(record["adjoint"]["slice_path"])
    assert inner[inner.index("--adjoint-slice") + 1] == str(slice_path)
    assert inner[inner.index("--adjoint-slice-sha256") + 1] == _sha(slice_path)
    assert str(receipt_path) not in inner
    assert "--resume" in inner


def test_quantum_sha256_binds_record_wire_bytes(tmp_path, portable_spec):
    """Defect 2: the consumer checks raw record file bytes first. The CLI
    must bind that wire digest; the canonical body check inside stays."""
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    inner = _payload(dispatch.quantum_argv(
        record, record_path=record_path, output_root=campaign["root"]))
    assert inner[inner.index("--quantum-sha256") + 1] == _sha(record_path)
    assert _sha(record_path) != record["identity_sha256"]


def test_writer_output_passes_matching_representation_checks(tmp_path, portable_spec):
    """Defect 3, under slice binding (PQ #993): the writer's pretty receipt
    is decoded, never hashed into a record, so its formatting cannot fail a
    gate. The record binds its slice, whose file is canonical JSON: the wire
    digest and the canonical digest are one value. Tampered slice bytes
    refuse; edited receipt content refuses at the dispatcher's proof gate."""
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    slice_path = Path(record["adjoint"]["slice_path"])
    parsed = json.loads(receipt_path.read_text())
    expected = adjoint_slice_sha256(stage_a_slice(parsed, 0))
    assert record["adjoint"]["slice_sha256"] == expected == _sha(slice_path)
    assert "receipt_sha256" not in record["adjoint"]
    proofs = dispatch.load_stage_a_proofs(receipt_path, [])
    assert dispatch.check_stage_a_proofs(proofs, [(record_path, record)]) == {
        "layer-000": expected}
    # Trailing whitespace on the receipt is not content: the gate passes.
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    assert dispatch.check_stage_a_proofs(
        dispatch.load_stage_a_proofs(receipt_path, []),
        [(record_path, record)]) == {"layer-000": expected}
    # Tampered slice wire bytes refuse against the sealed digest.
    good = slice_path.read_bytes()
    slice_path.write_bytes(good + b" ")
    with pytest.raises(AdjointSliceRefused, match="digest mismatch"):
        load_adjoint_slice(slice_path, expected, layer=0)
    with pytest.raises(dispatch.DispatchRefused):
        dispatch.check_stage_a_proofs(proofs, [(record_path, record)])
    slice_path.write_bytes(good)
    # Edited receipt content refuses at the proof gate.
    parsed["status"] = "partial"
    receipt_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(dispatch.DispatchRefused):
        dispatch.load_stage_a_proofs(receipt_path, [])


def test_consumer_verify_accepts_valid_pretty_receipt(tmp_path):
    """Defect 3 at the consumer: a record bound against the valid
    pretty-written receipt passes ``verify_quantum_identity``, which reads
    the record's slice file, never the receipt (PQ #993)."""
    campaign = _campaign(tmp_path)
    out = _produce(campaign)
    space = tmp_path / "adjoint-space"
    space.mkdir()
    receipt = _receipt(campaign, out["derivation"]["checkpoints"])
    write_adjoint_receipt(space, receipt)
    receipt_path = space / "adjoint-capture.json"
    bound = _produce(campaign, receipt=receipt)
    record, record_path, _ = _publish_layer_000(tmp_path, bound)
    slice_path = Path(record["adjoint"]["slice_path"])
    found, loaded = quantum.verify_quantum_identity(
        quantum_path=record_path, quantum_sha256=_sha(record_path),
        plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
        prepared_path=campaign["prepared_path"],
        prepared_sha256=campaign["prepared_sha"],
        adjoint_path=slice_path, adjoint_sha256=_sha(slice_path),
        output_root=campaign["root"])
    assert found["quantum_id"] == "layer-000"
    assert loaded == stage_a_slice(json.loads(receipt_path.read_text()), 0)
    assert adjoint_slice_sha256(loaded) == record["adjoint"]["slice_sha256"]
    # The whole receipt is not a slice: naming it refuses.
    with pytest.raises(quantum.QuantumIdentityRefused):
        quantum.verify_quantum_identity(
            quantum_path=record_path, quantum_sha256=_sha(record_path),
            plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
            prepared_path=campaign["prepared_path"],
            prepared_sha256=campaign["prepared_sha"],
            adjoint_path=receipt_path, adjoint_sha256=_sha(receipt_path),
            output_root=campaign["root"])


def test_consumer_parser_and_verify_drive_the_dispatched_row(tmp_path, portable_spec,
                                                             monkeypatch):
    """End to end through the real consumer parser and identity gates; only
    the GPU body is stubbed, after every binding passes."""
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    campaign, record, record_path, receipt_path = _bound_row(tmp_path, portable_spec)
    inner = _payload(dispatch.quantum_argv(
        record, record_path=record_path, output_root=campaign["root"]))
    args = quantum.build_parser().parse_args(inner[3:])
    assert args.data_manifest_sha256 == record["read_set"]["manifest_sha256"]
    found, loaded = quantum.verify_quantum_identity(
        quantum_path=Path(args.quantum), quantum_sha256=args.quantum_sha256,
        plan_path=Path(args.plan), plan_sha256=args.plan_sha256,
        prepared_path=Path(args.prepared), prepared_sha256=args.prepared_sha256,
        adjoint_path=args.adjoint_slice, adjoint_sha256=args.adjoint_slice_sha256,
        output_root=Path(args.output_root))
    assert found["quantum_id"] == "layer-000"
    calls = []
    monkeypatch.setattr(quantum, "run_layer_quantum",
                        lambda *a, **k: calls.append((a, k)) or {"passed": True})
    quantum.run_layer_quantum(
        {"output_root": str(campaign["root"])}, record=found, adjoint_slice=loaded,
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
    slice_path = Path(record["adjoint"]["slice_path"])
    write_adjoint_slice(slice_path, wrong["adjoint_slices"]["layer-000"], layer=0)
    with pytest.raises(quantum.QuantumIdentityRefused, match="output_space"):
        quantum.verify_quantum_identity(
            quantum_path=record_path,
            quantum_sha256=_sha(record_path),
            plan_path=campaign["plan_path"], plan_sha256=campaign["plan_sha"],
            prepared_path=campaign["prepared_path"],
            prepared_sha256=campaign["prepared_sha"],
            adjoint_path=slice_path, adjoint_sha256=_sha(slice_path),
            output_root=campaign["root"])


def _regen_argv(tmp_path: Path, campaign: dict) -> list[str]:
    derivation = tmp_path / "derivation.json"
    derivation.write_text(json.dumps({
        "chunk_target_bytes": 400, "stride": 1,
        "ram_window_gib": 160, "max_resident_consumers": 2}))
    partition = tmp_path / "window-partition.json"
    partition.write_text(json.dumps({"windows_by_layer": {"0": 1, "1": 1}}))
    return ["--plan", str(campaign["plan_path"]),
            "--plan-sha256", campaign["plan_sha"],
            "--prepared", str(campaign["prepared_path"]),
            "--prepared-sha256", campaign["prepared_sha"],
            "--parent-manifest", str(tmp_path / "parent.json"),
            "--parent-manifest-sha256", campaign["parent_sha"],
            "--derivation", str(derivation),
            "--partition", str(partition)]


def test_regenerate_writes_new_roots_and_binds_receipt(tmp_path):
    """Defect 4 path: the repo CLI regenerates records with the authoritative
    output root (never its own directory), writes slices at the
    producer-named paths the records bind (verified to resolve), and
    re-seals against the receipt -- old files untouched, nothing copied,
    no adjoint manifest written (the phase worker owns that file)."""
    import regenerate_joint_quanta as regen
    campaign = _campaign(tmp_path)
    base = _regen_argv(tmp_path, campaign)
    out = tmp_path / "reviewed"
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(out)]) == 0
    records = [json.loads(path.read_text())
               for path in sorted(out.glob("layer-*.json"))]
    assert len(records) == 2
    for record in records:
        assert record["output_space"]["root"].startswith(str(campaign["root"]))
        assert record["adjoint"]["receipt_sha256"] is None
        slice_path = Path(record["read_set"]["manifest_path"])
        assert slice_path.is_file(), f"bound slice missing at {slice_path}"
        assert hashlib.sha256(slice_path.read_bytes()).hexdigest() == \
            record["read_set"]["manifest_sha256"]
    assert (out / "records.json").is_file()
    assert not (out / "adjoint-manifest.json").exists()
    assert not (out / "manifests").exists()
    # Gate 1 without the original root refuses (reproducing against
    # known-bad originals can never match by definition).
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(out),
                              "--expect-existing", str(out),
                              "--check-only"]) == 3
    # Gate 1 with the original root: reproduce, then authorized move.
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(out),
                              "--expect-existing", str(out),
                              "--original-root", str(campaign["root"]),
                              "--check-only"]) == 0
    # Gate 2: binding re-seals every record against the receipt.
    space = tmp_path / "adjoint-space"
    space.mkdir()
    first = _produce(campaign)
    receipt = _receipt(campaign, first["derivation"]["checkpoints"])
    write_adjoint_receipt(space, receipt)
    bound_out = tmp_path / "bound"
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(bound_out),
                              "--adjoint-receipt",
                              str(space / "adjoint-capture.json")]) == 0
    bound = [json.loads(path.read_text())
             for path in sorted(bound_out.glob("layer-*.json"))]
    # Each record binds its own slice, written before the record, whose
    # file bytes hash to the sealed digest (PQ #993).
    for r in bound:
        assert "receipt_sha256" not in r["adjoint"]
        assert r["adjoint"]["slice_sha256"] == adjoint_slice_sha256(
            stage_a_slice(receipt, r["layer"]))
        assert _sha(Path(r["adjoint"]["slice_path"])) == r["adjoint"]["slice_sha256"]
    assert any(r["identity_sha256"] != f["identity_sha256"]
               for r, f in zip(bound, records))
    # Refusal, not overwrite: differing bytes at the destination fail closed.
    (bound_out / "layer-000.json").write_text("{}")
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(bound_out),
                              "--adjoint-receipt",
                              str(space / "adjoint-capture.json")]) == 3


def test_regenerate_gate1_reproduces_old_root_then_moves(tmp_path):
    """The live shape: originals rooted at the binder's nested directory
    reproduce exactly at that root (input fidelity), then move to the
    authoritative root with only the authorized path fields changed --
    against a realistic gzip parent, with records-out split from the
    output root."""
    import gzip as _gzip
    import regenerate_joint_quanta as regen
    campaign = _campaign(tmp_path)
    # Realistic parent: gzipped wire bytes, digest over the compressed form.
    parent_path = tmp_path / "parent.json"
    wire = _gzip.compress(parent_path.read_bytes(), mtime=0)
    parent_gz = tmp_path / "parent.json.gz"
    parent_gz.write_bytes(wire)
    campaign = dict(campaign, parent_sha=hashlib.sha256(wire).hexdigest())
    base = _regen_argv(tmp_path, campaign)
    base[base.index("--parent-manifest") + 1] = str(parent_gz)
    base[base.index("--parent-manifest-sha256") + 1] = campaign["parent_sha"]
    old_root = tmp_path / "layer-quanta"
    old_records = tmp_path / "old-records"
    old_records.mkdir()
    assert regen.main(base + ["--output-root", str(old_root),
                              "--records-out", str(old_records)]) == 0
    assert json.loads((old_records / "layer-000.json").read_text())[
        "output_space"]["root"].startswith(str(old_root))
    new_out = tmp_path / "new-records"
    # Gate 1 dry run: reproduce at the old root, validate the authorized
    # move to the new root, write nothing.
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(new_out),
                              "--expect-existing", str(old_records),
                              "--original-root", str(old_root),
                              "--check-only"]) == 0
    assert not new_out.exists()
    # Edited originals refuse at the fidelity step with the reason named.
    tampered = json.loads((old_records / "layer-001.json").read_text())
    tampered["layer"] = 99
    (old_records / "layer-001.json").write_text(json.dumps(tampered))
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(new_out),
                              "--expect-existing", str(old_records),
                              "--original-root", str(old_root),
                              "--check-only"]) == 3
    # The real move: new roots, slices resolving to exact sealed bytes.
    assert regen.main(base + ["--output-root", str(campaign["root"]),
                              "--records-out", str(new_out)]) == 0
    for record in [json.loads(path.read_text())
                   for path in sorted(new_out.glob("layer-*.json"))]:
        assert record["output_space"]["root"].startswith(str(campaign["root"]))
        slice_path = Path(record["read_set"]["manifest_path"])
        assert slice_path.is_file()
        assert hashlib.sha256(slice_path.read_bytes()).hexdigest() == \
            record["read_set"]["manifest_sha256"]
