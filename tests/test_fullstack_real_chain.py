"""Full-stack real chain: producer -> PB validation -> movers -> readers -> join.

Bounded actual chain on tiny fixtures, every leg through real production
code. New tests only; no production file is edited here.

- Real PQ producer: ``joint_layer_quanta.layer_quanta`` on a two-layer
  fixture whose manifest entries point at real tmp safetensors/plain
  files with real digests (one manifest end to end, no remap).
- Real PB parsing: published PB ``core`` manifest validation and
  ``storage_tiers`` phase ranges from the immutable published generation
  (see fullstack_pb_generation). Slice phase tables carry chunk phases
  only (producer #852); the storage legs move the real per-layer slices
  under their own wire digests -- no substitute manifest anywhere.
- Real movers: published ``stage_move`` (each slice in two entry-aligned
  legs under distinct keys, including a nonzero source-offset
  ``.pbrange`` range) and ``ram_promote`` with epoch into tmp tiers (the
  wire blob on layer-000, the payload range on layer-001); each slice's
  PQ map is the real PB ``compose`` overlaid with its real RAM fragments
  plus the current epoch announcement.
- Real PQ readers: source shards via ``staged_shard_opener`` (whole and
  nonzero-offset range, bit-identical, range hits counted), a wire blob
  via the real wire reader served from RAM, and a ``.pt`` render via the
  real production weight cache served from stage.
- Real receipts/join: ``joint_quanta_join`` CLI on payloads built from
  this producer's own records with cost signs and activation maxima
  measured off the actual staged reads and sealed by the real row
  constructor; coverage accepted, absent-record named gap (with the
  downstream allocation gate refusing the gapped payload),
  tampered/duplicate refusals (exit 1).
- Gap posture: an uncovered span reads the pool silently (demonstrated,
  named gap — not conformance). Strict SDK enforcement belongs to the
  owning workers (PR #846 glue with real context/proof).

Only CPU fixtures (torch CPU, safetensors); no model launch, no GPU work.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import struct
from pathlib import Path
import sys

import pytest

from fleet_sdk import require_prismabuild_sdk
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import fullstack_pb_generation as pbgen  # noqa: E402
from prismaquant.joint_layer_quanta import (  # noqa: E402
    layer_quanta,
    seal_manifest_bytes,
    verify_quanta_coverage,
)
from prismaquant.joint_quanta_join import (  # noqa: E402
    GappedPayloadRefused,
    load_joint_cost_for_allocation,
    main as join_main,
)

from tests.test_joint_quanta_join import (  # noqa: E402
    FORMATS,
    _payload_provenance,
    _roster_digest_of,
    _row,
    _units_of_layer,
    probe as join_probe_fixture,
)


STAGE_TIER = "prismabuild-stage:dl380g10"
RAM_TIER = "ram:dl380g10"


@pytest.fixture(scope="module")
def pb():
    """Published PB stdlib + fleet tools from the immutable generation."""
    require_prismabuild_sdk()
    info = pbgen.require_paths()
    import prismabuild.core as core  # noqa: E402
    import prismabuild.storage_tiers as tiers  # noqa: E402
    import prismabuild.residency_plan as plans  # noqa: E402
    import prismabuild.pool as pool  # noqa: E402
    import prismabuild.residency_map as pb_map  # noqa: E402
    import stage_move  # noqa: E402
    import ram_promote  # noqa: E402
    root = info["root"]
    for module in (core, tiers, plans, pool, pb_map, stage_move,
                   ram_promote):
        location = Path(getattr(module, "__file__", "")).resolve()
        assert location.is_relative_to(Path(root)), (
            f"{module.__name__} loaded from {location}, not {root}")
    assert info["generation"], "generation recorded once per admitted action"
    return {"generation": info["generation"], "root": root, "core": core,
            "tiers": tiers, "plans": plans, "pool": pool,
            "pb_map": pb_map, "stage_move": stage_move,
            "ram_promote": ram_promote}


@pytest.fixture(scope="module")
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
    range_raw = range_path.read_bytes()[range_off:range_off + range_len]
    entries = [
        _entry(head, 0, head_raw),
        _entry(files / "shard-l0.safetensors", 0, l0_raw),
        _entry(wire, 0, wire_raw),
        _entry(files / "shard-l1.safetensors", 0, l1_raw),
        _entry(pt_path, 0, pt_raw),
        _entry(range_path, range_off, range_raw),
    ]
    total = sum(e["bytes"] for e in entries)
    bounds, phases, running = [], [], 0
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"] + entries[2]["bytes"]),
                       ("layer-1", entries[3]["bytes"] + entries[4]["bytes"]
                        + entries[5]["bytes"])):
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
    receipt = {
        "schema": "prismaquant.joint_adjoint_capture.v1",
        "plan_sha256": sealed["plan.json"],
        "prepared_sha256": sealed["prepared.json"],
        "campaign_scope": parent["annotations"]["campaign_scope"],
        "checkpoints": [{"boundary": mark} for mark in (1, 2)],
        "status": "complete",
    }
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


def test_pb_validates_producer_slices(pb, produced, campaign) -> None:
    """Published PB parses the real slice tables into stageable ranges."""
    core, tiers = pb["core"], pb["tiers"]
    assert pb["generation"], "generation recorded once per admitted action"
    assert pb["root"], "dependency identity recorded once per admitted action"
    for record in produced["records"]:
        path, slice_manifest = _slice_file(campaign, record)
        validated = core.validate_data_manifest(slice_manifest)
        assert validated["entry_count"] == len(validated["entries"])
        assert all(p["bytes"] > 0
                   for p in validated["annotations"]["phases"]), path
        ranges = tiers.manifest_phase_ranges(validated)
        assert ranges, f"slice stages nothing: {record['quantum_id']}"
        assert [r["name"] for r in ranges] == [
            p["name"] for p in validated["annotations"]["phases"]]
        assert ranges[0]["start_bytes"] == 0
        assert ranges[-1]["end_bytes"] == validated["total_bytes"]


def _slice_spans(slice_manifest) -> list[tuple[int, int]]:
    """Manifest-byte spans of a slice's entries, in entry order."""
    spans, cursor = [], 0
    for entry in slice_manifest["entries"]:
        spans.append((cursor, cursor + entry["bytes"]))
        cursor += entry["bytes"]
    assert cursor == slice_manifest["total_bytes"]
    return spans


def _run_mover(stage_move, queue, pool, tmp, stage, *, consumer, mover, digest,
               manifest_file, start, end, receipts) -> None:
    args = stage_move.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--cas-root", str(tmp / "pb-queue" / "cas"),
        "--action-key", mover,
        "--consumer-action-key", consumer,
        "--tier-id", STAGE_TIER,
        "--stage-root", str(stage),
        "--manifest-sha256", digest,
        "--range-start-bytes", str(start),
        "--range-end-bytes", str(end),
        "--manifest", str(manifest_file),
        "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
        "--block", str(1 << 16),
        "--readers", "1", "--max-readers", "1", "--unpaced",
    ])
    receipt = stage_move.move(args)
    assert receipt["complete"] is True, (mover, receipt.get("refusal"))
    assert receipt.get("refusal") is None
    queue.record_move(mover, receipt)
    receipts[mover] = receipt


def _run_promotion(ram_promote, queue, pool, tmp, stage, ram, *, consumer, mover,
                   digest, manifest_file, start, end, promotions) -> None:
    args = ram_promote.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--action-key", mover,
        "--consumer-action-key", consumer,
        "--tier-id", RAM_TIER,
        "--ram-root", str(ram),
        "--source-stage-root", str(stage),
        "--manifest-sha256", digest,
        "--range-start-bytes", str(start),
        "--range-end-bytes", str(end),
        "--manifest", str(manifest_file),
        "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
    ])
    promotion = ram_promote.promote(args)
    assert promotion["complete"] is True, (mover, promotion.get("refusal"))
    assert promotion.get("refusal") is None
    queue.record_move(mover, promotion)
    promotions[mover] = promotion


@pytest.fixture(scope="module")
def staged(pb, produced, campaign):
    """Real movers stage both real slices; RAM promotions ride the same roots.

    Each slice moves in two entry-aligned legs under distinct mover keys,
    and one entry span per slice promotes to RAM (the wire blob on
    layer-000, the nonzero-offset payload range on layer-001).
    """
    require_prismabuild_sdk()
    stage_move, ram_promote = pb["stage_move"], pb["ram_promote"]
    pool = pb["pool"]
    tmp = campaign["tmp"]
    queue = pool.PoolQueue(tmp / "pb-queue")
    queue.ensure_layout()
    stage = tmp / "stage"
    stage.mkdir()
    ram = tmp / "ram"
    ram.mkdir()
    from prismabuild import storage_tiers as tiers  # noqa: E402
    epoch = tiers.ensure_ram_epoch(ram, host="dl380g10")
    assert epoch is not None
    ran, promotions, slices = {}, {}, {}
    stage_keys = iter(["aa" * 32, "ab" * 32, "ac" * 32, "ad" * 32])
    ram_keys = iter(["bb" * 32, "bc" * 32])
    for record in produced["records"]:
        quantum_id = record["quantum_id"]
        manifest_file, slice_manifest = _slice_file(campaign, record)
        digest = record["read_set"]["manifest_sha256"]
        # Separate quantum actions must have separate material namespaces.
        # Reuse the fixture record's sealed identity as its stable key.
        consumer = record["identity_sha256"]
        spans = _slice_spans(slice_manifest)
        total = slice_manifest["total_bytes"]
        first, second = next(stage_keys), next(stage_keys)
        _run_mover(stage_move, queue, pool, tmp, stage, consumer=consumer, mover=first,
                   digest=digest, manifest_file=manifest_file,
                   start=0, end=spans[0][1], receipts=ran)
        _run_mover(stage_move, queue, pool, tmp, stage, consumer=consumer, mover=second,
                   digest=digest, manifest_file=manifest_file,
                   start=spans[0][1], end=total, receipts=ran)
        if quantum_id == "layer-000":
            entry = next(e for e in slice_manifest["entries"]
                         if e["path"].endswith("wire-0.bin"))
        else:
            entry = next(e for e in slice_manifest["entries"]
                         if e["path"].endswith("shard-range.safetensors"))
        index = slice_manifest["entries"].index(entry)
        _run_promotion(ram_promote, queue, pool, tmp, stage, ram,
                       consumer=consumer,
                       mover=next(ram_keys), digest=digest,
                       manifest_file=manifest_file,
                       start=spans[index][0], end=spans[index][1],
                       promotions=promotions)
        slices[quantum_id] = {"consumer": consumer,
                              "manifest_file": manifest_file,
                              "manifest_sha256": digest,
                              "manifest": slice_manifest, "spans": spans}
    assert set(slices) == {"layer-000", "layer-001"}
    assert len({view["consumer"] for view in slices.values()}) == 2
    return {"queue": queue, "stage": stage, "ram": ram,
            "epoch": str(epoch["epoch"]), "receipts": ran,
            "promotions": promotions, "slices": slices}


def test_movers_stage_both_slices_byte_identical(staged, campaign) -> None:
    """Every slice entry's staged bytes equal the source span by digest."""
    assert len(staged["receipts"]) == 4, "two entry-aligned legs per slice"
    for mover, receipt in staged["receipts"].items():
        assert receipt["complete"] is True
        assert receipt.get("refusal") is None
    assert len(staged["promotions"]) == 2, "one RAM promotion per slice"
    for mover, promotion in staged["promotions"].items():
        assert promotion["complete"] is True
        assert promotion.get("refusal") is None
    queue = staged["queue"]
    frags = [fragment for view in staged["slices"].values()
             for fragment in _pq_read_fragments(queue, view["consumer"])]
    assert len(frags) == 6, "four stage movers plus two ram promotions filed"
    for fragment in frags:
        for key, entry in fragment["entries"].items():
            data = Path(entry["stage_path"]).read_bytes()
            assert len(data) == entry["bytes"], key
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], key
    for quantum_id, view in staged["slices"].items():
        for entry in view["manifest"]["entries"]:
            data = Path(entry["path"]).read_bytes()[
                entry["offset"]:entry["offset"] + entry["bytes"]]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], (
                quantum_id, entry["path"], entry["offset"])


def _pq_read_fragments(queue, consumer):
    root = Path(queue.root) / "residency" / consumer
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def _announce_ram_tier(tmp: Path, epoch: str) -> Path:
    """File the tier record the PQ reader compares the map's ram epoch against."""
    tiers = tmp / "tiers"
    tiers.mkdir(parents=True, exist_ok=True)
    record = {"schema": "prismabuild.storage_tier.v1", "tier": "ram",
              "tier_id": RAM_TIER, "mountpoint": "/ram/prewarm",
              "epoch": epoch}
    (tiers / f"{RAM_TIER}.json").write_text(json.dumps(record))
    return tiers


def _slice_fragments(pb, staged, quantum_id) -> tuple[list, list]:
    """A slice's own stage + RAM fragments, selected by its wire digest."""
    pool, pb_map = pb["pool"], pb["pb_map"]
    queue = staged["queue"]
    view = staged["slices"][quantum_id]
    digest = view["manifest_sha256"]
    frags = [pb_map.validate_fragment(f) for f in
             pb_map.read_fragments(queue.root / pool.RESIDENCY, view["consumer"])]
    assert all(f["manifest_sha256"] == digest for f in frags)
    stage_frags = [f for f in frags if f["tier_id"] == STAGE_TIER]
    ram_frags = [f for f in frags if f["tier_id"] == RAM_TIER]
    assert len(stage_frags) == 2, (quantum_id, "both legs filed")
    assert len(ram_frags) == 1, (quantum_id, "the promotion filed")
    return stage_frags, ram_frags


def _write_pq_map(pb, staged, campaign, tmp, quantum_id):
    """PQ map for one slice: real compose overlaid with its real RAM."""
    _, pb_map = pb["pool"], pb["pb_map"]
    stage_frags, ram_frags = _slice_fragments(pb, staged, quantum_id)
    digest = staged["slices"][quantum_id]["manifest_sha256"]
    composed = pb_map.compose(stage_frags)
    assert composed["manifest_sha256"] == digest
    overlaid = pb_map.overlay_ram(
        composed, ram_frags, ram_tier_id=RAM_TIER,
        ram_root=str(staged["ram"]), ram_epoch=staged["epoch"])
    assert overlaid.get("ram_tier_id") == RAM_TIER
    assert overlaid.get("ram_epoch") == staged["epoch"]
    assert any("ram_path" in entry for entry in overlaid["entries"].values())
    for key, entry in overlaid["entries"].items():
        data = Path(entry["stage_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], key
        if "ram_path" in entry:
            ram_data = Path(entry["ram_path"]).read_bytes()
            assert hashlib.sha256(ram_data).hexdigest() == entry["sha256"], key
    _announce_ram_tier(tmp, staged["epoch"])
    residency = tmp / "residency"
    residency.mkdir(parents=True, exist_ok=True)
    map_path = residency / f"residency-{quantum_id}.json"
    map_path.write_text(json.dumps(overlaid))
    return map_path


def test_real_reader_serves_staged_tensors_bit_identical(
        pb, staged, campaign, monkeypatch) -> None:
    """REAL PQ source open over mover-staged safetensors: bit-identical."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    tmp = campaign["tmp"]
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    for quantum_id in ("layer-000", "layer-001"):
        digest = staged["slices"][quantum_id]["manifest_sha256"]
        map_path = _write_pq_map(pb, staged, campaign, tmp, quantum_id)
        monkeypatch.setenv(ENV_VAR, str(map_path))
        reset_residency_resolver_for_tests()
        bind_residency_manifest(digest)
        resolver = residency_resolver()
        assert resolver is not None
        rows = [Path(e["path"])
                for e in staged["slices"][quantum_id]["manifest"]["entries"]
                if e["path"].endswith(".safetensors") and e["offset"] == 0]
        assert rows, f"whole-file shard present: {quantum_id}"
        for declared in rows:
            with layer_streaming._source_safe_open(
                    str(declared), framework="pt") as reader:
                with safe_open(str(declared), framework="pt") as reference:
                    for name in reference.keys():
                        got = reader.get_tensor(name)
                        want = reference.get_tensor(name)
                        if got.numel():
                            assert torch.equal(
                                got.view(torch.uint8),
                                want.view(torch.uint8)), (quantum_id, name)
        report = resolver.report()
        assert report["fallback_count"] == 0, quantum_id


def test_nonzero_offset_range_serves_its_tensor_from_stage(
        pb, staged, campaign, monkeypatch) -> None:
    """A ``.pbrange`` payload range stages, promotes, and serves its tensor."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    tmp = campaign["tmp"]
    map_path = _write_pq_map(pb, staged, campaign, tmp, "layer-001")
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(staged["slices"]["layer-001"]["manifest_sha256"])
    resolver = residency_resolver()
    assert resolver is not None
    span = campaign["range"]
    assert span["offset"] > 0, "nonzero source offset, not a logical split"
    key = f"{span['offset']}:{span['path']}"
    assert key in json.loads(map_path.read_text())["entries"]
    with layer_streaming._source_safe_open(
            span["path"], framework="pt") as reader:
        with safe_open(span["path"], framework="pt") as reference:
            assert torch.equal(reader.get_tensor(span["tensor"]),
                               reference.get_tensor(span["tensor"]))
    report = resolver.report()
    assert report["range_hits"] >= 1
    assert report["fallback_count"] == 0


def test_wire_blob_is_served_from_ram(pb, staged, campaign, monkeypatch) -> None:
    """The real wire reader prefers the live RAM copy under the staged one."""
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from prismaquant.tessera_joint_aura import (  # noqa: E402
        _read_verified_wire_blob,
    )
    tmp = campaign["tmp"]
    map_path = _write_pq_map(pb, staged, campaign, tmp, "layer-000")
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(staged["slices"]["layer-000"]["manifest_sha256"])
    resolver = residency_resolver()
    assert resolver is not None
    wire = campaign["wire"]
    cell = {"record": {"blob_bytes": wire["bytes"],
                       "blob_sha256": wire["sha256"]},
            "wire": wire["path"]}
    blob, digest = _read_verified_wire_blob(cell)
    assert blob == Path(wire["path"]).read_bytes()
    assert digest == wire["sha256"]
    report = resolver.report()
    assert report["ram_hits"] >= 1
    assert report["fallback_count"] == 0


def test_pwc_render_is_served_from_stage(pb, staged, campaign, monkeypatch) -> None:
    """The real production weight cache reads its render off the stage."""
    from prismaquant.production_weight_cache import (  # noqa: E402
        ProductionWeightCache,
    )
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_report, residency_resolver,
        reset_residency_resolver_for_tests,
    )
    tmp = campaign["tmp"]
    map_path = _write_pq_map(pb, staged, campaign, tmp, "layer-001")
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(staged["slices"]["layer-001"]["manifest_sha256"])
    assert residency_resolver() is not None
    pt = campaign["pt"]
    key = ("chain-pwc", "pt")
    cache = ProductionWeightCache(weights={key: pt["path"]}, levers={})
    cache.enable_lru(1 << 20)
    cache.require_file_load_sha256({key: pt["sha256"]},
                                   max_file_bytes=pt["bytes"])
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    assert torch.equal(tensor, pt["tensor"])
    assert cache.file_load_receipt(key, tensor)["path"] == pt["path"]
    report = residency_report()
    assert report is not None
    assert report["hits"] >= 1
    assert report["fallbacks"] == []


def test_uncovered_span_reads_pool_silently_gap_not_conformance(
        pb, staged, campaign, monkeypatch) -> None:
    """Unmapped span reads the pool with no stage service: the demonstrated gap."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    tmp = campaign["tmp"]
    map_path = _write_pq_map(pb, staged, campaign, tmp, "layer-000")
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(staged["slices"]["layer-000"]["manifest_sha256"])
    resolver = residency_resolver()
    missing = tmp / "pool" / "model" / "never-staged.safetensors"
    _write_shard(missing, _tensors())
    assert resolver.stages(str(missing)) is False
    with layer_streaming._source_safe_open(
            str(missing), framework="pt") as reader:
        with safe_open(str(missing), framework="pt") as reference:
            assert torch.equal(reader.get_tensor("w0"),
                               reference.get_tensor("w0"))
    report = resolver.report()
    assert report["fallback_count"] == 0
    assert report["hits"] == 0



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
    from prismaquant.joint_aura import (  # noqa: E402
        identity_sha256,
        make_joint_aura_entry,
    )
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
                "probe_identity_sha256": identity_sha256(probe)}
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


def _join_argv(root: Path, out: Path, binding, receipt_sha: str) -> list[str]:
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
            "--adjoint-receipt-sha256", receipt_sha]


def test_join_accepts_producer_records_with_coverage(
        pb, staged, produced, campaign, probe, tmp_path, monkeypatch) -> None:
    """REAL join CLI over this producer's records; costs seed from staged reads."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, TIERS_DIR_ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    tmp = campaign["tmp"]
    map_path = _write_pq_map(pb, staged, campaign, tmp, "layer-000")
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(staged["slices"]["layer-000"]["manifest_sha256"])
    assert residency_resolver() is not None
    shard_l0 = str(tmp / "pool" / "model" / "shard-l0.safetensors")
    with layer_streaming._source_safe_open(
            shard_l0, framework="pt") as reader:
        with safe_open(shard_l0, framework="pt") as reference:
            got = reader.get_tensor("w0")
            assert torch.equal(got, reference.get_tensor("w0"))
    sign = _derive_sign_from_read(got)
    measured_max_abs = _measure_activation_max_abs(got)
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS, sign=sign,
                    activation_max_abs=measured_max_abs)
    out = tmp_path / "join-out"
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    assert join_main(_join_argv(root, out, binding, receipt_sha)) == 0
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
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    assert join_main(_join_argv(root, out, binding, receipt_sha)) == 0
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
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    dup_space = Path(produced["records"][0]["output_space"]["root"])
    dup_space.joinpath("cost.pkl").write_bytes(
        pickle.dumps({"costs": {}, "provenance": {}}, protocol=2))
    assert join_main(_join_argv(root, out, binding, receipt_sha)) == 1
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
    assert join_main(_join_argv(root, out2, binding, receipt_sha)) == 1
    assert not (out2 / "joint-cost.pkl").exists()
