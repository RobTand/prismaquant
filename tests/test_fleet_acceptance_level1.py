"""Tiny CPU producer, native staged-reader and scientific join contracts.

Known fixture bytes and original hash/nonfinite/gap checks remain. PB fleet,
broker, mover and SDK scenarios are retired, not reimplemented here.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import pytest
import torch
from test_fullstack_real_chain import _entry, _tensors, _write_shard
from staged_native_fixtures import bind, native_staged_state, publish_inputs
from prismaquant.staged_tier_policy import staged_tier_policy_test_context


@pytest.fixture(autouse=True)
def _native_policy(native_staged_state):
    with staged_tier_policy_test_context('ssd'):
        yield


@pytest.fixture
def staged(campaign, tmp_path):
    files = list(dict.fromkeys(Path(e['path']) for e in campaign['parent']['entries']))
    return publish_inputs(tmp_path, files)


@pytest.fixture(scope="module")
def probe():
    from tests.test_joint_quanta_join import (  # noqa: E402
        probe as join_probe_fixture,
    )
    return join_probe_fixture.__wrapped__()



@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Harness fixture: two shards, one nonzero-offset range, real digests."""
    tmp = tmp_path_factory.mktemp("accept")
    files = tmp / "pool" / "model"
    files.mkdir(parents=True)
    head = files / "head.bin"
    head_raw = hashlib.sha256(b"fleet-acceptance-head").digest() * 8
    head.write_bytes(head_raw)
    _write_shard(files / "shard-0.safetensors", _tensors())
    _write_shard(files / "shard-1.safetensors", _tensors())
    l0_raw = (files / "shard-0.safetensors").read_bytes()
    l1_raw = (files / "shard-1.safetensors").read_bytes()
    half = len(l1_raw) // 2
    entries = [
        _entry(head, 0, head_raw),
        _entry(files / "shard-0.safetensors", 0, l0_raw),
        _entry(files / "shard-1.safetensors", 0, l1_raw[:half]),
        _entry(files / "shard-1.safetensors", half, l1_raw[half:]),
    ]
    total = sum(e["bytes"] for e in entries)
    running, phases = 0, []
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"]),
                       ("layer-1", entries[2]["bytes"] + entries[3]["bytes"])):
        running += size
        phases.append({"name": name, "bytes": size,
                       "cumulative_bytes": running})
    assert running == total
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fleet-acceptance-harness"},
        "mount_prefix": str(files),
        "entries": entries, "entry_count": 4, "total_bytes": total,
        "annotations": {
            "campaign_scope": {"campaign": "fleet-acceptance", "layers": [0, 1]},
            "layers": [0, 1], "phases": phases,
        },
    }
    units = [f"model.layers.{layer}.mlp.gate_proj" for layer in (0, 1)]
    prepared = {"formats_by_qname": {
        name: ["TESSERA_BF16_K1_R1792", "TESSERA_E4M3_K1_R896"]
        for name in units}}
    plan = {"output_root": str(tmp / "campaign"),
            "model": str(tmp / "pool" / "model"), "distributed_campaign": {}}
    sealed = {}
    for name, payload in (("plan.json", plan), ("prepared.json", prepared),
                          ("parent.json", parent)):
        raw = json.dumps(payload, sort_keys=True).encode()
        (tmp / name).write_bytes(raw)
        sealed[name] = hashlib.sha256(raw).hexdigest()
    from tests.test_stage_b_band_binding import synthetic_receipt
    receipt = synthetic_receipt(
        plan_sha256=sealed["plan.json"], prepared_sha256=sealed["prepared.json"],
        scope=parent["annotations"]["campaign_scope"], num_layers=2, stride=1)
    return {"tmp": tmp, "plan": plan, "prepared": prepared, "parent": parent,
            "units": units, "plan_sha": sealed["plan.json"],
            "prepared_sha": sealed["prepared.json"],
            "parent_sha": sealed["parent.json"], "receipt": receipt}



@pytest.fixture(scope="module")
def produced(campaign):
    """The real PQ producer on the harness fixture, persisted by digest."""
    from prismaquant.joint_layer_quanta import (  # noqa: E402
        layer_quanta,
        seal_manifest_bytes,
        verify_quanta_coverage,
    )
    built = layer_quanta(
        campaign["plan"], campaign["prepared"], campaign["parent"],
        chunk_target_bytes=1 << 20, stride=1,
        output_root=str(campaign["tmp"] / "campaign"),
        plan_path=str(campaign["tmp"] / "plan.json"),
        plan_sha256=campaign["plan_sha"],
        prepared_path=str(campaign["tmp"] / "prepared.json"),
        prepared_sha256=campaign["prepared_sha"],
        parent_manifest_sha256=campaign["parent_sha"],
        window_partition={"windows_by_layer": {"0": 1, "1": 1}},
        adjoint_receipt=campaign["receipt"])
    proof = verify_quanta_coverage(built["records"], campaign["parent"])
    assert proof["layers"] == [0, 1]
    assert len(built["records"]) == 2
    root = campaign["tmp"] / "campaign" / "layer-quanta"
    for record in built["records"]:
        quantum_id = record["quantum_id"]
        (root / "records").mkdir(parents=True, exist_ok=True)
        (root / "records" / f"{quantum_id}.json").write_text(
            json.dumps(record))
        manifest_path = root / record["read_set"]["manifest_path"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(seal_manifest_bytes(
            built["slice_manifests"][quantum_id]))
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == \
            record["read_set"]["manifest_sha256"]
    return built



def _slice_body(campaign, record):
    raw = (campaign["tmp"] / "campaign" / "layer-quanta"
           / record["read_set"]["manifest_path"]).read_bytes()
    assert record["read_set"]["manifest_sha256"] == \
        hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw).decode())



def test_slice_wire_digests_cover_sealed_bytes(produced, campaign):
    """ID-03: every slice digest is SHA-256 over the sealed wire bytes."""
    for record in produced["records"]:
        raw = (campaign["tmp"] / "campaign" / "layer-quanta"
               / record["read_set"]["manifest_path"]).read_bytes()
        assert record["read_set"]["manifest_sha256"] == \
            hashlib.sha256(raw).hexdigest()
        assert len(raw) > 0



def test_canonical_record_digest_is_stable_and_rejects_nonfinite():
    """ID-04: canonical digest is deterministic; non-finite floats refuse."""
    from prismaquant.cost_stage_checkpoint import (  # noqa: E402
        canonical_json_sha256,
    )
    from tests.test_joint_quanta_join import (  # noqa: E402
        FORMATS,
        _row,
    )
    import tests.test_joint_quanta_join as joint_fix  # noqa: E402
    campaign_binding = {
        "plan_sha256": "0" * 64, "prepared_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "scope": {"campaign": "fleet-acceptance", "layers": [0, 1]},
        "roster": ["model.layers.0.mlp.gate_proj"],
        "formats_by_qname": {"model.layers.0.mlp.gate_proj": list(FORMATS)},
    }
    probe = joint_fix.probe.__wrapped__()
    row = _row("model.layers.0.mlp.gate_proj", FORMATS[0], probe)
    first = canonical_json_sha256(row, where="acceptance row")
    assert canonical_json_sha256(row, where="acceptance row") == first
    assert len(first) == 64
    broken = json.loads(json.dumps(row))
    broken["x2_per_probe"] = [float("nan")]
    with pytest.raises(Exception):
        canonical_json_sha256(broken, where="nonfinite row")



def test_gzip_member_parses_and_tamper_refuses(produced, campaign):
    """ACC-01: valid gzip wire parses; flipped bytes refuse."""
    import gzip as gzip_mod
    for record in produced["records"]:
        raw = (campaign["tmp"] / "campaign" / "layer-quanta"
               / record["read_set"]["manifest_path"]).read_bytes()
        body = json.loads(gzip_mod.decompress(raw).decode())
        assert body["entry_count"] == len(body["entries"])
        tampered = bytearray(raw)
        tampered[len(tampered) // 2] ^= 0xFF
        assert hashlib.sha256(bytes(tampered)).hexdigest() != \
            record["read_set"]["manifest_sha256"]
        with pytest.raises(Exception):
            json.loads(gzip_mod.decompress(bytes(tampered)).decode())



def test_produced_records_carry_boundary_derivation_inputs(produced):
    """Accepted PQ853 metadata: records hold the boundary-readset inputs.

    The derivation itself (``build_quantum_boundary_readset``) needs the
    completed capture receipt plus the new lane's combined plan, so its
    execution stays nonqualified until that runtime exists. What this
    harness CAN assert now: every produced record carries the sealed
    fields the derivation reads (chain, boundary, windows, campaign
    binding, adjoint receipt digest).
    """
    for record in produced["records"]:
        assert isinstance(record["layer"], int)
        adjoint = record["adjoint"]
        # Stride-1 single-layer chains seal empty lists; the derivation
        # reads the fields, and emptiness here is the producer's real
        # answer for this camera-ready tiny fixture, not a gap.
        assert isinstance(adjoint["chain_layers"], list)
        assert isinstance(adjoint["checkpoint_boundary"], int)
        # PQ #993: each record binds its own stage-A slice, never a receipt.
        assert isinstance(adjoint["slice_sha256"], str)
        assert "receipt_sha256" not in adjoint
        assert isinstance(record["windows"], list), record["quantum_id"]
        campaign = record["campaign"]
        for key in ("plan_sha256", "prepared_sha256",
                    "read_manifest_sha256", "campaign_scope"):
            assert campaign.get(key), (record["quantum_id"], key)



def test_staged_reads_are_bit_identical_with_stage_bytes_counted(
        staged, campaign, monkeypatch):
    """Actual immutable staged tensors equal independently known fixture values."""
    from prismaquant import layer_streaming
    from test_streamed_metadata_staged_reads import _deny_pool_opens

    resolver = bind(monkeypatch, staged)
    sources = {Path(e['path']) for e in campaign['parent']['entries']}
    _deny_pool_opens(monkeypatch, campaign['tmp'] / 'pool' / 'model',
                     names=tuple(source.name for source in sources))
    expected = _tensors()
    for source in sorted(sources):
        if source.suffix != '.safetensors':
            continue
        with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
            assert set(reader.keys()) == set(expected)
            for name, want in expected.items():
                got = reader.get_tensor(name)
                assert got.dtype == want.dtype and got.shape == want.shape
                assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name
    report = resolver.report()
    assert report['fallback_count'] == 0 and report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] > 0 and resolver._windows == 0


def test_join_accepts_and_gapped_refuses_downstream(
        produced, campaign, probe, tmp_path):
    """INV-10: coverage accepted + allocation accepts; gapped refused."""
    from prismaquant.joint_quanta_join import (  # noqa: E402
        GappedPayloadRefused,
        load_joint_cost_for_allocation,
        main as join_main,
    )
    from tests.test_joint_quanta_join import (  # noqa: E402
        FORMATS,
        _payload_provenance,
        _row,
    )
    import pickle as pickle_mod

    def seal(root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "plan.json").write_bytes(
            json.dumps(campaign["plan"], sort_keys=True).encode())
        (root / "prepared.json").write_bytes(
            json.dumps(campaign["prepared"], sort_keys=True).encode())
        (root / "manifest.json").write_bytes(
            json.dumps(campaign["parent"], sort_keys=True).encode())
        (root / "scope.json").write_text(json.dumps(
            campaign["parent"]["annotations"]["campaign_scope"]))
        (root / "roster.txt").write_text(
            "\n".join(sorted(campaign["prepared"]["formats_by_qname"])) + "\n")
        (root / "formats.json").write_text(json.dumps(
            campaign["prepared"]["formats_by_qname"]))
        return {
            "plan_sha256": campaign["plan_sha"],
            "prepared_sha256": campaign["prepared_sha"],
            "manifest_sha256": campaign["parent_sha"],
            "scope": campaign["parent"]["annotations"]["campaign_scope"],
            "roster": sorted(campaign["prepared"]["formats_by_qname"]),
            "formats_by_qname": campaign["prepared"]["formats_by_qname"],
        }

    def write_payloads(root, binding):
        for record in produced["records"]:
            layer = record["layer"]
            quantum_id = record["quantum_id"]
            units = sorted(q for q in binding["roster"]
                           if f".layers.{layer}." in q)
            costs = {qname: {fmt: _row(qname, fmt, probe)
                             for fmt in FORMATS} for qname in units}
            records = root / "layer-quanta" / "records"
            records.mkdir(parents=True, exist_ok=True)
            (records / f"{quantum_id}.json").write_text(json.dumps(record))
            space = Path(record["output_space"]["root"])
            space.mkdir(parents=True, exist_ok=True)
            payload = {"costs": costs,
                       "provenance": _payload_provenance(
                           {**binding,
                            "manifest_sha256": binding["manifest_sha256"]},
                           record)}
            (space / "cost.pkl").write_bytes(pickle_mod.dumps(
                payload, protocol=pickle_mod.HIGHEST_PROTOCOL))
            (space / "status.json").write_text(json.dumps({
                "schema": "prismaquant.joint_layer_quantum.status.v1",
                "quantum_id": quantum_id,
                "identity_sha256": record["identity_sha256"],
                "status": "complete", "units": [len(costs), len(costs)],
                "unix": 1750000000}))
            (space / "results.json").write_text(
                json.dumps({"quantum_id": quantum_id}))

    def argv(root, out, binding):
        receipt_path = root / "adjoint-capture.json"
        receipt_path.write_text(json.dumps(campaign["receipt"]))
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        return ["--input-root", str(root), "--output-dir", str(out),
                "--plan", str(root / "plan.json"),
                "--plan-sha256", binding["plan_sha256"],
                "--prepared", str(root / "prepared.json"),
                "--prepared-sha256", binding["prepared_sha256"],
                "--manifest", str(root / "manifest.json"),
                "--manifest-sha256", binding["manifest_sha256"],
                "--scope", str(root / "scope.json"),
                "--roster", str(root / "roster.txt"),
                "--formats-by-qname", str(root / "formats.json"),
                "--adjoint-receipt", str(receipt_path),
                "--adjoint-receipt-sha256", receipt_sha]

    root = tmp_path / "join-in"
    binding = seal(root)
    write_payloads(root, binding)
    out = tmp_path / "join-out"
    assert join_main(argv(root, out, binding)) == 0
    allocated = load_joint_cost_for_allocation(out / "joint-cost.pkl")
    assert sorted(allocated["costs"]) == sorted(binding["roster"])
    (root / "layer-quanta" / "records" / "layer-001.json").unlink()
    out2 = tmp_path / "join-out-2"
    assert join_main(argv(root, out2, binding)) == 0
    results = json.loads((out2 / "results.json").read_text())
    assert results["status"] == "gapped"
    with pytest.raises(GappedPayloadRefused):
        load_joint_cost_for_allocation(out2 / "joint-cost.pkl")
