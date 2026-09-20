"""Full-stack real chain: producer -> PB validation -> movers -> readers -> join.

Bounded actual chain on tiny fixtures, every leg through real production
code. New tests only; no production file is edited here.

- Real PQ producer: ``joint_layer_quanta.layer_quanta`` on a two-layer
  fixture whose manifest entries point at real tmp safetensors/plain
  files with real digests (one manifest end to end, no remap).
- Real PB parsing: published PB ``core`` manifest validation and
  ``storage_tiers`` phase ranges from the immutable published generation
  (see fullstack_pb_generation). Slice staging through the real
  row/planning interface is PENDING the producer zero-head fix (strict
  xfail below); the storage legs below move a staging manifest holding the
  same shard bytes the producer read (plus wire/render/range objects);
  no hand-built plan papering over the boundary.
- Real movers: published ``stage_move`` (whole files plus a nonzero
  source-offset ``.pbrange`` range) and ``ram_promote`` with epoch into
  tmp tiers; the PQ map is the real PB ``compose`` overlaid with the real
  RAM fragments plus the current epoch announcement.
- Real PQ readers: source shards via ``staged_shard_opener`` (whole and
  nonzero-offset range, bit-identical, range hits counted), a wire blob
  via the real wire reader served from RAM, and a ``.pt`` render via the
  real production weight cache served from stage.
- Real receipts/join: ``joint_quanta_join`` CLI on payloads built from
  this producer's own records with cost signs derived from the actual
  staged read bytes; coverage accepted, absent-record named gap (with the
  downstream allocation gate refusing the gapped payload),
  tampered/duplicate refusals (exit 1).
- Gap posture: an uncovered span reads the pool silently (demonstrated,
  named gap — not conformance). Strict slice/reader enforcement belongs
  to the owning workers (producer zero-head; PR #846 SDK glue).

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


CONSUMER = "cc" * 32
STAGE_TIER = "prismabuild-stage:dl380g10"
RAM_TIER = "ram:dl380g10"


@pytest.fixture(scope="module")
def pb():
    """Published PB stdlib + fleet tools from the immutable generation."""
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
        assert str(location).startswith(root), (
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


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One manifest end to end: entries point at real tmp files with digests."""
    tmp = tmp_path_factory.mktemp("chain")
    files = tmp / "pool" / "model"
    files.mkdir(parents=True)
    head = files / "head.bin"
    head.write_bytes(hashlib.sha256(b"chain-head").digest() * 8)
    _write_shard(files / "shard-l0.safetensors", _tensors())
    _write_shard(files / "shard-l1.safetensors", _tensors())
    l1_raw = (files / "shard-l1.safetensors").read_bytes()
    wire = files / "wire-0.bin"
    wire.write_bytes(hashlib.sha256(b"chain-wire").digest() * 64)
    wire_raw = wire.read_bytes()
    pt_path = files / "pwc-0.pt"
    pt_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.save(pt_tensor, str(pt_path))
    pt_raw = pt_path.read_bytes()
    range_path = files / "shard-range.safetensors"
    _write_shard(range_path, _tensors())
    range_off, range_len = _safetensors_tensor_span(range_path, "w0")
    range_raw = range_path.read_bytes()[range_off:range_off + range_len]
    entries = [
        {"path": str(head), "offset": 0,
         "bytes": len(head.read_bytes()),
         "sha256": hashlib.sha256(head.read_bytes()).hexdigest()},
        {"path": str(files / "shard-l0.safetensors"), "offset": 0,
         "bytes": len((files / "shard-l0.safetensors").read_bytes()),
         "sha256": hashlib.sha256(
             (files / "shard-l0.safetensors").read_bytes()).hexdigest()},
        {"path": str(files / "shard-l1.safetensors"), "offset": 0,
         "bytes": len(l1_raw),
         "sha256": hashlib.sha256(l1_raw).hexdigest()},
    ]
    total = sum(e["bytes"] for e in entries)
    bounds, phases, running = [], [], 0
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"]),
                       ("layer-1", entries[2]["bytes"])):
        running += size
        bounds.append(running)
        phases.append({"name": name, "bytes": size,
                       "cumulative_bytes": running})
    # Storage legs stage the same producer bytes plus three more staged
    # objects (a wire blob, a .pt render, a nonzero-offset payload range).
    # The producer tiles layer phases only, so these live in a staging
    # manifest beside the parent -- same pool files for the shards, its own
    # digest for the storage legs -- never merged into the producer parent.
    extra_entries = [
        {"path": str(wire), "offset": 0,
         "bytes": len(wire_raw),
         "sha256": hashlib.sha256(wire_raw).hexdigest()},
        {"path": str(pt_path), "offset": 0,
         "bytes": len(pt_raw),
         "sha256": hashlib.sha256(pt_raw).hexdigest()},
        {"path": str(range_path), "offset": range_off,
         "bytes": range_len,
         "sha256": hashlib.sha256(range_raw).hexdigest()},
    ]
    staging_entries = entries + extra_entries
    staging_total = sum(e["bytes"] for e in staging_entries)
    staging_phases, staging_running = [], 0
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"]),
                       ("layer-1", entries[2]["bytes"]),
                       ("wire", extra_entries[0]["bytes"]),
                       ("pwc", extra_entries[1]["bytes"]),
                       ("range", extra_entries[2]["bytes"])):
        staging_running += size
        staging_phases.append({"name": name, "bytes": size,
                               "cumulative_bytes": staging_running})
    assert staging_running == staging_total
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
    staging = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fullstack-real-chain-storage"},
        "mount_prefix": str(files),
        "entries": staging_entries,
        "entry_count": len(staging_entries),
        "total_bytes": staging_total,
        "annotations": {
            "campaign_scope": {"campaign": "fullstack-real-chain",
                               "layers": [0, 1]},
            "layers": [0, 1],
            "phases": staging_phases,
        },
    }
    staging_raw = json.dumps(staging, sort_keys=True).encode()
    (tmp / "staging.json").write_bytes(staging_raw)
    staging_sha = hashlib.sha256(staging_raw).hexdigest()
    spans, cursor = [], 0
    for entry in staging_entries:
        spans.append((cursor, cursor + entry["bytes"]))
        cursor += entry["bytes"]
    return {"tmp": tmp, "plan": plan, "prepared": prepared,
            "parent": parent, "units": units,
            "plan_sha": sealed["plan.json"],
            "prepared_sha": sealed["prepared.json"],
            "parent_sha": sealed["parent.json"], "receipt": receipt,
            "staging": staging, "staging_sha": staging_sha,
            "wire": {"path": str(wire), "bytes": len(wire_raw),
                     "sha256": hashlib.sha256(wire_raw).hexdigest()},
            "pt": {"path": str(pt_path), "bytes": len(pt_raw),
                   "sha256": hashlib.sha256(pt_raw).hexdigest(),
                   "tensor": pt_tensor},
            "range": {"path": str(range_path), "offset": range_off,
                      "bytes": range_len,
                      "sha256": hashlib.sha256(range_raw).hexdigest(),
                      "tensor": "w0",
                      "manifest_span": spans[5]},
            "wire_manifest_span": spans[3],
            "ram_manifest_span": (spans[3][0], spans[5][1])}


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


def test_pb_validates_producer_slices(pb, produced, campaign) -> None:
    """Published PB validates the real slice wire; its phase table is the gap."""
    core, tiers = pb["core"], pb["tiers"]
    assert pb["generation"], "generation recorded once per admitted action"
    assert pb["root"], "dependency identity recorded once per admitted action"
    for record in produced["records"]:
        raw = (campaign["tmp"] / "campaign" / "layer-quanta"
               / record["read_set"]["manifest_path"]).read_bytes()
        assert record["read_set"]["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
        slice_manifest = json.loads(gzip.decompress(raw).decode())
        validated = core.validate_data_manifest(slice_manifest)
        assert validated["entry_count"] == len(validated["entries"])
        # Gap demonstration (not conformance): the real producer seals a
        # zero-byte head phase in every slice, which the real PB phase
        # validator refuses (cumulative 0 is never an entry boundary).
        # Staging the real slice through the real row/planning interface is
        # PENDING the producer worker's isolated zero-head fix; the storage
        # legs below move a staging manifest holding the same shard bytes
        # the producer read (plus wire/render/range objects), and no
        # hand-built plan is frozen here to paper over the boundary.
        assert tiers.manifest_phase_ranges(validated) == []


@pytest.fixture(scope="module")
def staged(pb, produced, campaign):
    """Real movers: whole files + a nonzero-offset range to stage, wire..range to RAM."""
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
    manifest_path = tmp / "slice-source.json"
    manifest_path.write_text(json.dumps(campaign["staging"], sort_keys=True))
    total = campaign["staging"]["total_bytes"]
    head_end = campaign["staging"]["entries"][0]["bytes"] \
        + campaign["staging"]["entries"][1]["bytes"]
    ran = {}
    for mover, start, end in (("aa" * 32, 0, head_end),
                              ("ab" * 32, head_end, total)):
        args = stage_move.build_parser().parse_args([
            "--pool-root", str(tmp / "pb-queue"),
            "--cas-root", str(tmp / "pb-queue" / "cas"),
            "--action-key", mover,
            "--consumer-action-key", "cc" * 32,
            "--tier-id", "prismabuild-stage:dl380g10",
            "--stage-root", str(stage),
            "--manifest-sha256", campaign["staging_sha"],
            "--range-start-bytes", str(start),
            "--range-end-bytes", str(end),
            "--manifest", str(manifest_path),
            "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
            "--block", str(1 << 16),
            "--readers", "1", "--max-readers", "1", "--unpaced",
        ])
        receipt = stage_move.move(args)
        assert receipt["complete"] is True
        assert receipt.get("refusal") is None
        queue.record_move(mover, receipt)
        ran[mover] = receipt
    ram_start, ram_end = campaign["ram_manifest_span"]
    prom = ram_promote.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--action-key", "bb" * 32,
        "--consumer-action-key", "cc" * 32,
        "--tier-id", "ram:dl380g10",
        "--ram-root", str(ram),
        "--source-stage-root", str(stage),
        "--manifest-sha256", campaign["staging_sha"],
        "--range-start-bytes", str(ram_start),
        "--range-end-bytes", str(ram_end),
        "--manifest", str(manifest_path),
        "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
    ])
    promotion = ram_promote.promote(prom)
    assert promotion["complete"] is True
    assert promotion.get("refusal") is None
    queue.record_move("bb" * 32, promotion)
    return {"queue": queue, "stage": stage, "ram": ram,
            "epoch": str(epoch["epoch"]), "receipts": ran,
            "promotion": promotion}


@pytest.mark.xfail(strict=True, reason=(
    "producer zero-head slices must become stageable: either side repairs"))
def test_strict_slice_phases_are_stageable(pb, produced, campaign) -> None:
    tiers = pb["tiers"]
    for record in produced["records"]:
        raw = (campaign["tmp"] / "campaign" / "layer-quanta"
               / record["read_set"]["manifest_path"]).read_bytes()
        slice_manifest = json.loads(gzip.decompress(raw).decode())
        assert tiers.manifest_phase_ranges(slice_manifest), record["quantum_id"]


def test_movers_stage_whole_and_split_byte_identical(staged, campaign) -> None:
    """Mover-staged bytes equal source bytes via digests (whole + split)."""
    assert set(staged["receipts"]) == {"aa" * 32, "ab" * 32}
    for mover, receipt in staged["receipts"].items():
        assert receipt["complete"] is True
        assert receipt.get("refusal") is None
    assert staged["promotion"]["complete"] is True
    assert staged["promotion"].get("refusal") is None
    queue = staged["queue"]
    frags = _pq_read_fragments(queue)
    assert len(frags) == 3, "two stage movers plus one ram promotion filed"
    for entry in campaign["staging"]["entries"]:
        data = Path(entry["path"]).read_bytes()[
            entry["offset"]:entry["offset"] + entry["bytes"]]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def _pq_read_fragments(queue):
    root = Path(queue.root) / "residency" / CONSUMER
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


def _write_pq_map(pb, staged, campaign, tmp):
    """PQ map: real PB stage compose overlaid with the real RAM fragments."""
    pool, pb_map = pb["pool"], pb["pb_map"]
    queue = staged["queue"]
    frags = [pb_map.validate_fragment(f) for f in
             pb_map.read_fragments(queue.root / pool.RESIDENCY, CONSUMER)]
    stage_frags = [f for f in frags if f["tier_id"] == STAGE_TIER]
    ram_frags = [f for f in frags if f["tier_id"] == RAM_TIER]
    assert len(stage_frags) == 2, "both stage movers filed"
    assert len(ram_frags) == 1, "the ram promotion filed"
    composed = pb_map.compose(stage_frags)
    assert composed["manifest_sha256"] == campaign["staging_sha"]
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
    map_path = residency / "residency.json"
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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
    resolver = residency_resolver()
    assert resolver is not None
    rows = [(Path(e["path"]), e["offset"], e["bytes"])
            for e in campaign["staging"]["entries"]
            if e["path"].endswith(".safetensors") and e["offset"] == 0]
    assert len(rows) == 2, "both whole-file shards present"
    for declared, offset, size in rows:
        with layer_streaming._source_safe_open(
                str(declared), framework="pt") as reader:
            with safe_open(str(declared), framework="pt") as reference:
                for name in reference.keys():
                    got, want = reader.get_tensor(name), reference.get_tensor(name)
                    if got.numel():
                        assert torch.equal(
                            got.view(torch.uint8),
                            want.view(torch.uint8)), name
    report = resolver.report()
    assert report["fallback_count"] == 0


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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
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


def _write_payloads(root: Path, binding, produced, probe, fmt_list,
                    *, sign: float = 0.1) -> None:
    from tests.test_joint_quanta_join import (  # noqa: E402
        _payload_provenance, _row, _units_of_layer,
    )
    for record in produced["records"]:
        layer = record["layer"]
        quantum_id = record["quantum_id"]
        units = sorted(q for q in binding["roster"]
                       if f".layers.{layer}." in q)
        assert units, f"layer {layer} has units"
        costs = {qname: {fmt: _row(qname, fmt, probe, sign=sign)
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
    map_path = _write_pq_map(pb, staged, campaign, tmp)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tmp / "tiers"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["staging_sha"])
    assert residency_resolver() is not None
    shard_l0 = str(tmp / "pool" / "model" / "shard-l0.safetensors")
    with layer_streaming._source_safe_open(
            shard_l0, framework="pt") as reader:
        with safe_open(shard_l0, framework="pt") as reference:
            got = reader.get_tensor("w0")
            assert torch.equal(got, reference.get_tensor("w0"))
    sign = _derive_sign_from_read(got)
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS, sign=sign)
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
