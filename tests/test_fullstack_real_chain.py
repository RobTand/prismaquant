"""Real producer slices, native staged readers and scientific result joining.

The bounded CPU chain uses exact producer readsets, immutable materialized
extents and real safetensors, wire and PWC readers. Known tensor bytes seed
the sealed cost/activation rows consumed by the actual join and allocation
gates. PB mover, queue, RAM-epoch and fragment protocols are retired.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from staged_native_fixtures import bind, native_staged_state, publish_inputs
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_launch import read_contract_document
from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
from test_streamed_metadata_staged_reads import _deny_pool_opens
from prismaquant.joint_layer_quanta import layer_quanta, seal_manifest_bytes, verify_quanta_coverage
from prismaquant.joint_quanta_join import (
    GappedPayloadRefused, load_joint_cost_for_allocation, main as join_main)
from tests.test_joint_quanta_join import probe as join_probe_fixture


@pytest.fixture(autouse=True)
def _native_policy(native_staged_state):
    with staged_tier_policy_test_context('ssd'):
        yield


@pytest.fixture(scope='module')
def probe():
    return join_probe_fixture.__wrapped__()


def _tensors():
    return {
        "w0": torch.linspace(-2, 2, 64, dtype=torch.float32).reshape(8, 8),
        "w1": torch.arange(64, dtype=torch.float32).reshape(8, 8),
    }


def _write_shard(path: Path, tensors: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def _safetensors_tensor_span(path: Path, name: str) -> tuple[int, int]:
    """Absolute ``(offset, bytes)`` of one tensor payload inside a shard."""
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen].decode())
    offs = header[name]["data_offsets"]
    start = 8 + hlen + offs[0]
    assert start > 0, "payload range must sit at a nonzero source offset"
    return start, offs[1] - offs[0]


def _entry(path: Path, offset: int, data: bytes) -> dict:
    """One manifest entry with a real digest over the exact staged span."""
    return {"path": str(path), "offset": offset, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One manifest end to end: entries point at real tmp files with digests.

    Layer phases carry the whole storage corpus: layer-0 holds its shard
    plus the wire blob, layer-1 its shard plus the render plus a
    nonzero-offset payload range. The producer's per-layer slice manifests
    therefore cover every staged object, and the storage legs move real
    slices -- no substitute manifest anywhere in this chain.
    """
    tmp = tmp_path_factory.mktemp("chain")
    files = tmp / "pool" / "model"
    files.mkdir(parents=True)
    head = files / "head.bin"
    head_raw = hashlib.sha256(b"chain-head").digest() * 8
    head.write_bytes(head_raw)
    _write_shard(files / "shard-l0.safetensors", _tensors())
    _write_shard(files / "shard-l1.safetensors", _tensors())
    l0_raw = (files / "shard-l0.safetensors").read_bytes()
    l1_raw = (files / "shard-l1.safetensors").read_bytes()
    wire = files / "wire-0.bin"
    wire_raw = hashlib.sha256(b"chain-wire").digest() * 64
    wire.write_bytes(wire_raw)
    pt_path = files / "pwc-0.pt"
    pt_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.save(pt_tensor, str(pt_path))
    pt_raw = pt_path.read_bytes()
    range_path = files / "shard-range.safetensors"
    _write_shard(range_path, _tensors())
    range_off, range_len = _safetensors_tensor_span(range_path, "w0")
    range_source = range_path.read_bytes()
    range_raw = range_source[range_off:range_off + range_len]
    entries = [
        _entry(head, 0, head_raw),
        _entry(files / "shard-l0.safetensors", 0, l0_raw),
        _entry(wire, 0, wire_raw),
        _entry(files / "shard-l1.safetensors", 0, l1_raw),
        _entry(pt_path, 0, pt_raw),
        _entry(range_path, 0, range_source[:range_off]),
        _entry(range_path, range_off, range_raw),
    ]
    total = sum(e["bytes"] for e in entries)
    bounds, phases, running = [], [], 0
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"] + entries[2]["bytes"]),
                       ("layer-1", sum(entry['bytes'] for entry in entries[3:]))):
        running += size
        bounds.append(running)
        phases.append({"name": name, "bytes": size,
                       "cumulative_bytes": running})
    assert running == total
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fullstack-real-chain"},
        "mount_prefix": str(files),
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": total,
        "annotations": {
            "campaign_scope": {"campaign": "fullstack-real-chain",
                               "layers": [0, 1]},
            "layers": [0, 1],
            "phases": phases,
        },
    }
    units = [f"model.layers.{layer}.mlp.gate_proj" for layer in (0, 1)]
    prepared = {"formats_by_qname": {
        name: ["TESSERA_BF16_K1_R1792", "TESSERA_E4M3_K1_R896"] for name in units}}
    plan = {"output_root": str(tmp / "campaign"),
            "model": str(tmp / "pool" / "model"),
            "distributed_campaign": {}}
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
    return {"tmp": tmp, "plan": plan, "prepared": prepared,
            "parent": parent, "units": units,
            "plan_sha": sealed["plan.json"],
            "prepared_sha": sealed["prepared.json"],
            "parent_sha": sealed["parent.json"], "receipt": receipt,
            "wire": {"path": str(wire), "bytes": len(wire_raw),
                     "sha256": hashlib.sha256(wire_raw).hexdigest()},
            "pt": {"path": str(pt_path), "bytes": len(pt_raw),
                   "sha256": hashlib.sha256(pt_raw).hexdigest(),
                   "tensor": pt_tensor},
            "range": {"path": str(range_path), "offset": range_off,
                      "bytes": range_len,
                      "sha256": hashlib.sha256(range_raw).hexdigest(),
                      "tensor": "w0"}}


@pytest.fixture(scope="module")
def produced(campaign):
    """The real producer on the real manifest, persisted by digest."""
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
    assert proof["parent_total_bytes"] == campaign["parent"]["total_bytes"]
    assert len(built["records"]) == 2
    root = campaign["tmp"] / "campaign" / "layer-quanta"
    for record in built["records"]:
        quantum_id = record["quantum_id"]
        (root / "records").mkdir(parents=True, exist_ok=True)
        (root / "records" / f"{quantum_id}.json").write_text(json.dumps(record))
        manifest_path = root / record["read_set"]["manifest_path"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(seal_manifest_bytes(
            built["slice_manifests"][quantum_id]))
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == \
            record["read_set"]["manifest_sha256"]
    return built


def _slice_file(campaign, record) -> tuple[Path, dict]:
    """The persisted slice file and its parsed body for one record."""
    rel = campaign["tmp"] / "campaign" / "layer-quanta" \
        / record["read_set"]["manifest_path"]
    raw = rel.read_bytes()
    assert record["read_set"]["manifest_sha256"] == \
        hashlib.sha256(raw).hexdigest(), record["quantum_id"]
    return rel, json.loads(gzip.decompress(raw).decode())


@pytest.fixture
def staged(produced, campaign, tmp_path):
    """Materialize each persisted producer slice without a substitute readset."""
    bindings = {}
    for record in produced['records']:
        directory = tmp_path / record['quantum_id']
        directory.mkdir()
        path, manifest = _slice_file(campaign, record)
        files, ranges = {}, {}
        for entry in manifest['entries']:
            source = Path(entry['path'])
            files[source] = None
            ranges.setdefault(source, []).append((entry['offset'], entry['offset'] + entry['bytes']))
        binding = publish_inputs(directory, files, ranges=ranges, read_contract=path.read_bytes())
        assert binding['read_contract_sha256'] == record['read_set']['manifest_sha256']
        bindings[record['quantum_id']] = binding
    return bindings


def _bind_slice(monkeypatch, staged, campaign, quantum_id):
    resolver = bind(monkeypatch, staged[quantum_id])
    _deny_pool_opens(monkeypatch, campaign['tmp'] / 'pool' / 'model',
        names=[Path(e.source_path).name for e in resolver.manifest.extents])
    return resolver


def test_native_binding_authenticates_producer_slices(staged, produced, campaign):
    for record in produced['records']:
        path, manifest = _slice_file(campaign, record)
        assert read_contract_document(path.read_bytes()) == manifest
        assert manifest['entry_count'] == len(manifest['entries'])
        assert all(p['bytes'] > 0 for p in manifest['annotations']['phases'])
        assert sum(e['bytes'] for e in manifest['entries']) == manifest['total_bytes']
        binding = staged[record['quantum_id']]
        native = load_staged_manifest(binding['path'], binding['sha256'])
        assert native.read_contract_sha256 == record['read_set']['manifest_sha256']
        assert {(e.source_path, e.source_offset, e.length, e.sha256) for e in native.extents} == {
            (e['path'], e['offset'], e['bytes'], e['sha256']) for e in manifest['entries']}


def test_native_materializes_both_slices_byte_identical(staged, produced, campaign):
    for record in produced['records']:
        binding = staged[record['quantum_id']]
        native = load_staged_manifest(binding['path'], binding['sha256'])
        for extent in native.extents:
            raw = Path(extent.object_path).read_bytes()
            source = Path(extent.source_path).read_bytes()[extent.source_offset:extent.source_offset + extent.length]
            assert raw[extent.object_offset:extent.object_offset + extent.length] == source
            assert hashlib.sha256(source).hexdigest() == extent.sha256
            assert len(raw) == extent.object_bytes
            assert hashlib.sha256(raw).hexdigest() == extent.object_sha256


def test_real_reader_serves_staged_tensors_bit_identical(staged, campaign, monkeypatch):
    from prismaquant import layer_streaming
    for quantum_id in ('layer-000', 'layer-001'):
        resolver = _bind_slice(monkeypatch, staged, campaign, quantum_id)
        rows = {e.source_path for e in resolver.manifest.extents
                if e.source_path.endswith('.safetensors') and e.source_offset == 0
                and e.length == e.source_bytes}
        assert rows
        for declared in rows:
            with layer_streaming._source_safe_open(declared, framework='pt') as reader:
                for name, expected in _tensors().items():
                    got = reader.get_tensor(name)
                    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
                assert resolver._windows > 0
                with pytest.raises(RuntimeError, match='live reader windows'):
                    resolver.close()
            assert resolver._windows == 0
        assert resolver.report()['bytes_from_pool'] == 0


def test_nonzero_offset_range_serves_its_tensor_from_stage(staged, campaign, monkeypatch):
    from prismaquant import layer_streaming
    resolver = _bind_slice(monkeypatch, staged, campaign, 'layer-001')
    span = campaign['range']
    assert span['offset'] > 0
    entry = resolver.staged_range(span['path'], span['offset'], span['offset'] + span['bytes'])
    assert entry['offset'] == span['offset'] and entry['object_offset'] == 0
    with layer_streaming._source_safe_open(span['path'], framework='pt') as reader:
        got = reader.get_tensor(span['tensor'])
        assert torch.equal(got.view(torch.uint8), _tensors()[span['tensor']].view(torch.uint8))
    assert resolver.report()['range_hits'] >= 1
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_wire_blob_is_served_from_native_stage(staged, campaign, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    resolver = _bind_slice(monkeypatch, staged, campaign, 'layer-000')
    wire = campaign['wire']
    cell = {'record': {'blob_bytes': wire['bytes'], 'blob_sha256': wire['sha256']}, 'wire': wire['path']}
    blob, digest = _read_verified_wire_blob(cell)
    assert blob == hashlib.sha256(b'chain-wire').digest() * 64
    assert digest == wire['sha256']
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_pwc_render_is_served_from_stage(staged, campaign, monkeypatch):
    from prismaquant.production_weight_cache import ProductionWeightCache
    resolver = _bind_slice(monkeypatch, staged, campaign, 'layer-001')
    pt = campaign['pt']
    key = ('chain-pwc', 'pt')
    cache = ProductionWeightCache(weights={key: pt['path']}, levers={})
    cache.enable_lru(1 << 20)
    cache.require_file_load_sha256({key: pt['sha256']}, max_file_bytes=pt['bytes'])
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    assert torch.equal(tensor, torch.arange(16, dtype=torch.float32).reshape(4, 4))
    assert cache.file_load_receipt(key, tensor)['path'] == pt['path']
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_uncovered_span_refuses_without_pool_reads(staged, campaign, tmp_path, monkeypatch):
    from prismaquant import layer_streaming
    resolver = _bind_slice(monkeypatch, staged, campaign, 'layer-000')
    missing = tmp_path / 'never-staged.safetensors'
    _write_shard(missing, _tensors())
    assert not resolver.stages(str(missing))
    _deny_pool_opens(monkeypatch, tmp_path, names=[missing.name])
    with pytest.raises(TierPolicyRefused):
        with layer_streaming._source_safe_open(str(missing), framework='pt') as reader:
            reader.get_tensor('w0')
            pytest.fail('undeclared source granted tensor bytes')
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def _seal_join_inputs(root: Path, campaign) -> dict:
    """Write join CLI inputs from THIS producer's real digests."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_bytes(
        json.dumps(campaign["plan"], sort_keys=True).encode())
    (root / "prepared.json").write_bytes(
        json.dumps(campaign["prepared"], sort_keys=True).encode())
    manifest_bytes = json.dumps(campaign["parent"], sort_keys=True).encode()
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "scope.json").write_text(json.dumps(
        campaign["parent"]["annotations"]["campaign_scope"]))
    (root / "roster.txt").write_text(
        "\n".join(sorted(campaign["prepared"]["formats_by_qname"])) + "\n")
    (root / "formats.json").write_text(json.dumps(
        campaign["prepared"]["formats_by_qname"]))
    (root / "adjoint-capture.json").write_text(json.dumps(campaign["receipt"]))
    binding = {
        "plan_sha256": campaign["plan_sha"],
        "prepared_sha256": campaign["prepared_sha"],
        "manifest_sha256": campaign["parent_sha"],
        "scope": campaign["parent"]["annotations"]["campaign_scope"],
        "roster": sorted(campaign["prepared"]["formats_by_qname"]),
        "formats_by_qname": campaign["prepared"]["formats_by_qname"],
    }
    return binding


def _derive_sign_from_read(tensor) -> float:
    """Bounded deterministic sign seeded by actual staged-read bytes."""
    digest = hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    return 0.05 + (int(digest[:4], 16) % 150) / 1000.0


def _measure_activation_max_abs(tensor) -> float:
    """The activation statistic read off the staged tensor (CPU plumbing)."""
    value = float(tensor.detach().cpu().float().abs().max())
    assert value > 0 and value < 1e6, "bounded fixture activation"
    return value


def _measured_row(qname, fmt, probe, sign, activation_max_abs):
    """A joint-AURA row sealed by the real constructor, carrying measured IO.

    The cost sign seeds from staged-read bytes and the activation block
    carries the staged tensor's measured max-abs -- both bounded,
    deterministic fixture outputs of actual reads, sealed by the same
    production entry constructor the runtime publishes.
    """
    from prismaquant.joint_aura_rows import joint_identity_sha256, make_joint_aura_entry
    from prismaquant.production_weight_cache import (  # noqa: E402
        _cb_cache_tensor_identity,
    )
    weight = _cb_cache_tensor_identity(
        torch.arange(16, dtype=torch.bfloat16).reshape(4, 4))
    activation = {"schema": "prismaquant.joint_aura.activation.v1",
                  "quantizes_input": False, "activation_max_abs": activation_max_abs,
                  "input_global_scale": None, "clip_enabled": False}
    operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": qname,
                "format": fmt, "source_weight": weight,
                "rendered_weight": weight, "activation": activation,
                "arithmetic": probe["arithmetic"],
                "probe_identity_sha256": joint_identity_sha256(probe)}
    steps = [sign * (i + 1) * (1 if i % 2 == 0 else -1) for i in range(3)]
    return make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[{"weight": v, "activation": 0., "mixed": 0.,
                            "total": v} for v in steps])


def _write_payloads(root: Path, binding, produced, probe, fmt_list,
                    *, sign: float = 0.1,
                    activation_max_abs: float | None = None) -> None:
    from tests.test_joint_quanta_join import (  # noqa: E402
        _payload_provenance, _row, _units_of_layer,
    )
    for record in produced["records"]:
        layer = record["layer"]
        quantum_id = record["quantum_id"]
        units = sorted(q for q in binding["roster"]
                       if f".layers.{layer}." in q)
        assert units, f"layer {layer} has units"
        if activation_max_abs is None:
            costs = {qname: {fmt: _row(qname, fmt, probe, sign=sign)
                             for fmt in fmt_list} for qname in units}
        else:
            costs = {qname: {fmt: _measured_row(
                qname, fmt, probe, sign, activation_max_abs)
                for fmt in fmt_list} for qname in units}
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
        cost_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        (space / "cost.pkl").write_bytes(cost_bytes)
        (space / "status.json").write_text(json.dumps({
            "schema": "prismaquant.joint_layer_quantum.status.v1",
            "quantum_id": quantum_id,
            "identity_sha256": record["identity_sha256"],
            "status": "complete", "units": [len(costs), len(costs)],
            "unix": 1750000000}))
        (space / "results.json").write_text(json.dumps({"quantum_id": quantum_id}))
    return {r["quantum_id"]: Path(r["output_space"]["root"])
            for r in produced["records"]}


def _join_argv(root: Path, out: Path, binding) -> list[str]:
    """Join CLI argv: mode (a), the completed stage-A receipt (PQ #993)."""
    receipt_sha = hashlib.sha256((root / "adjoint-capture.json").read_bytes()).hexdigest()
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
            "--adjoint-receipt", str(root / "adjoint-capture.json"),
            "--adjoint-receipt-sha256", receipt_sha]


def test_join_accepts_producer_records_with_coverage(
        staged, produced, campaign, probe, tmp_path, monkeypatch) -> None:
    """REAL join CLI over this producer's records; costs seed from staged reads."""
    from prismaquant import layer_streaming  # noqa: E402
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    tmp = campaign["tmp"]
    resolver = _bind_slice(monkeypatch, staged, campaign, 'layer-000')
    shard_l0 = str(tmp / "pool" / "model" / "shard-l0.safetensors")
    with layer_streaming._source_safe_open(
            shard_l0, framework="pt") as reader:
        got = reader.get_tensor("w0")
        assert torch.equal(got, _tensors()['w0'])
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0
    sign = _derive_sign_from_read(got)
    measured_max_abs = _measure_activation_max_abs(got)
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS, sign=sign,
                    activation_max_abs=measured_max_abs)
    out = tmp_path / "join-out"
    assert join_main(_join_argv(root, out, binding)) == 0
    joined = pickle.loads((out / "joint-cost.pkl").read_bytes())
    assert sorted(joined["costs"]) == sorted(binding["roster"])
    allocated = load_joint_cost_for_allocation(out / "joint-cost.pkl")
    assert sorted(allocated["costs"]) == sorted(binding["roster"])


def test_join_missing_quantum_is_named_gap(
        produced, campaign, probe, tmp_path) -> None:
    """Dropping a layer record gaps the join instead of shrinking the set."""
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS)
    (root / "layer-quanta" / "records" / "layer-001.json").unlink()
    out = tmp_path / "join-out"
    assert join_main(_join_argv(root, out, binding)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["status"] == "gapped"
    assert [g["quantum_id"] for g in
            results["distributed"]["gaps"]] == ["layer-001"]
    # The complete gate is a refusal, not a silent shrink: the allocation
    # stage rejects the gapped payload while the join itself exits 0.
    with pytest.raises(GappedPayloadRefused):
        load_joint_cost_for_allocation(out / "joint-cost.pkl")


def test_join_duplicate_and_mismatched_payloads_refuse(
        produced, campaign, probe, tmp_path) -> None:
    """Tampered bytes refuse (exit 1); a qname answered twice refuses."""
    from tests.test_joint_quanta_join import FORMATS, _row  # noqa: E402
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS)
    out = tmp_path / "join-out"
    dup_space = Path(produced["records"][0]["output_space"]["root"])
    dup_space.joinpath("cost.pkl").write_bytes(
        pickle.dumps({"costs": {}, "provenance": {}}, protocol=2))
    assert join_main(_join_argv(root, out, binding)) == 1
    assert not (out / "joint-cost.pkl").exists()
    _write_payloads(root, binding, produced, probe, FORMATS)
    other = [q for q in binding["roster"] if ".layers.0." in q][0]
    victim = Path(produced["records"][1]["output_space"]["root"])
    payload = pickle.loads(victim.joinpath("cost.pkl").read_bytes())
    payload["costs"][other] = {fmt: _row(other, fmt, probe)
                               for fmt in FORMATS}
    victim.joinpath("cost.pkl").write_bytes(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    out2 = tmp_path / "join-out-2"
    assert join_main(_join_argv(root, out2, binding)) == 1
    assert not (out2 / "joint-cost.pkl").exists()
