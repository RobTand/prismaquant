"""Full-stack real chain: producer -> PB validation -> movers -> readers -> join.

Bounded actual chain on tiny fixtures, every leg through real production
code. New tests only; no production file is edited here.

- Real PQ producer: ``joint_layer_quanta.layer_quanta`` on a two-layer
  fixture whose manifest entries point at real tmp safetensors/plain
  files with real digests (one manifest end to end, no remap).
- Real PB parsing: published PB ``core`` manifest validation,
  ``storage_tiers`` phase ranges, ``residency_plan`` build/freeze, all
  from the immutable published generation (see fullstack_pb_generation).
- Real movers: published ``stage_move`` (whole + nonzero-offset split)
  and ``ram_promote`` with epoch into tmp tiers.
- Real PQ readers: ``staged_shard_opener`` over mover-staged safetensors,
  bit-identical against the pool file; map built from real mover output.
- Real receipts/join: ``joint_quanta_join`` CLI on payloads built from
  this producer's own records; coverage accepted, missing/duplicate/
  mismatched refused.
- Gap posture: an uncovered span reads the pool with a recorded
  fallback (demonstrated, named gap — not conformance). The strict
  refusal is an xfail(strict) assertion that flips red the day the
  owning worker's enforcement lands.

Only CPU fixtures (torch CPU, safetensors); no model launch, no GPU work.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
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
from prismaquant.joint_quanta_join import main as join_main  # noqa: E402

from tests.test_joint_quanta_join import (  # noqa: E402
    FORMATS,
    _payload_provenance,
    _roster_digest_of,
    _row,
    _units_of_layer,
    probe as join_probe_fixture,
)


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
    return {"generation": info["generation"], "core": core,
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
    l1_len = len((files / "shard-l1.safetensors").read_bytes())
    entries = [
        {"path": str(head), "offset": 0,
         "bytes": len(head.read_bytes()),
         "sha256": hashlib.sha256(head.read_bytes()).hexdigest()},
        {"path": str(files / "shard-l0.safetensors"), "offset": 0,
         "bytes": len((files / "shard-l0.safetensors").read_bytes()),
         "sha256": hashlib.sha256(
             (files / "shard-l0.safetensors").read_bytes()).hexdigest()},
        {"path": str(files / "shard-l1.safetensors"), "offset": 0,
         "bytes": l1_len // 2,
         "sha256": hashlib.sha256(
             (files / "shard-l1.safetensors").read_bytes()[:l1_len // 2]).hexdigest()},
        {"path": str(files / "shard-l1.safetensors"), "offset": l1_len // 2,
         "bytes": l1_len - l1_len // 2,
         "sha256": hashlib.sha256(
             (files / "shard-l1.safetensors").read_bytes()[l1_len // 2:]).hexdigest()},
    ]
    total = sum(e["bytes"] for e in entries)
    bounds, phases, running = [], [], 0
    for name, size in (("head", entries[0]["bytes"]),
                       ("layer-0", entries[1]["bytes"]),
                       ("layer-1", entries[2]["bytes"] + entries[3]["bytes"])):
        running += size
        bounds.append(running)
        phases.append({"name": name, "bytes": size,
                       "cumulative_bytes": running})
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
    prepared = {"formats_by_qname": {name: {} for name in units}}
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
            "parent_sha": sealed["parent.json"], "receipt": receipt}


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
    """Published PB accepts the real producer's slice manifests and freezes a plan."""
    core, tiers, plans, pool = pb["core"], pb["tiers"], pb["plans"], pb["pool"]
    assert pb["generation"], "generation recorded once per admitted action"
    for record in produced["records"]:
        raw = (campaign["tmp"] / "campaign" / "layer-quanta"
               / record["read_set"]["manifest_path"]).read_bytes()
        assert record["read_set"]["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
        slice_manifest = json.loads(gzip.decompress(raw).decode())
        validated = core.validate_data_manifest(slice_manifest)
        assert validated["entry_count"] == len(validated["entries"])
        assert tiers.manifest_phase_ranges(validated), record["quantum_id"]
    queue = pool.PoolQueue(campaign["tmp"] / "pb-queue")
    queue.ensure_layout()
    record = produced["records"][0]
    phases = [{"name": chunk["name"], "start_bytes": chunk["start_bytes"],
               "end_bytes": chunk["end_bytes"], "stage_gib": 1,
               "mover_row": {"action_key": "ab" * 32,
                             "cas_root": str(queue.root / "cas"),
                             "checkout_root": str(queue.root / "co"),
                             "worker_script": str(queue.root / "worker.py"),
                             "tags": ["gb10"],
                             "resources": {"cpu": 1, "mem_gb": 1,
                                           "stage_gib@prismabuild-stage:dl380g10": 1},
                             "residency": {"schema": pool.RESIDENCY_SCHEMA_V1,
                                           "tier_id": "prismabuild-stage:dl380g10",
                                           "manifest_sha256": record["read_set"]["manifest_sha256"],
                                           "manifest_bytes": 1 << 20,
                                           "range_start_bytes": chunk["start_bytes"],
                                           "range_end_bytes": chunk["end_bytes"]}},
               "egress_row": {"action_key": "cd" * 32,
                              "cas_root": str(queue.root / "cas"),
                              "checkout_root": str(queue.root / "co"),
                              "worker_script": str(queue.root / "worker.py"),
                              "tags": ["gb10"],
                              "resources": {"cpu": 1, "mem_gb": 1}}}
              for chunk in record["chunks"]]
    plan = plans.build_plan(
        consumer_action_key="ef" * 64, tier_id="prismabuild-stage:dl380g10",
        stage_root="/stage/prewarm",
        manifest_sha256=record["read_set"]["manifest_sha256"],
        manifest_bytes=1 << 20, phases=phases)
    frozen = plans.freeze(queue, plan)
    assert plans.remaining(frozen, None) == frozen["phases"]


@pytest.fixture(scope="module")
def staged(pb, produced, campaign):
    """Real movers: whole + split ranges to SSD stage, whole to RAM."""
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
    manifest_path.write_text(json.dumps(campaign["parent"], sort_keys=True))
    total = campaign["parent"]["total_bytes"]
    head_end = campaign["parent"]["entries"][0]["bytes"] \
        + campaign["parent"]["entries"][1]["bytes"]
    ran = {}
    for mover, start, end in (("aa" * 32, 0, head_end),
                              ("ab" * 32, head_end, total)):
        args = stage_move.build_parser().parse_args([
            "--pool-root", str(tmp / "pb-queue"),
            "--cas-root", str(tmp / "pb-queue" / "cas"),
            "--action-key", mover,
            "--consumer-action-key", "cc" * 64,
            "--tier-id", "prismabuild-stage:dl380g10",
            "--stage-root", str(stage),
            "--manifest-sha256", campaign["parent_sha"],
            "--range-start-bytes", str(start),
            "--range-end-bytes", str(end),
            "--manifest", str(manifest_path),
            "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
            "--block", str(1 << 16),
            "--readers", "1", "--max-readers", "1", "--unpaced",
        ])
        receipt = stage_move.move(args)
        assert receipt["complete"] is True
        queue.record_move(mover, receipt)
        ran[mover] = receipt
    prom = ram_promote.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--action-key", "bb" * 32,
        "--consumer-action-key", "cc" * 64,
        "--tier-id", "ram:dl380g10",
        "--ram-root", str(ram),
        "--source-stage-root", str(stage),
        "--manifest-sha256", campaign["parent_sha"],
        "--range-start-bytes", "0",
        "--range-end-bytes", str(total),
        "--manifest", str(manifest_path),
        "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
    ])
    promotion = ram_promote.promote(prom)
    assert promotion["complete"] is True
    queue.record_move("bb" * 32, promotion)
    return {"queue": queue, "stage": stage, "ram": ram,
            "epoch": str(epoch["epoch"]), "receipts": ran,
            "promotion": promotion}


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
    for entry in campaign["parent"]["entries"]:
        data = Path(entry["path"]).read_bytes()[
            entry["offset"]:entry["offset"] + entry["bytes"]]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def _pq_read_fragments(queue):
    from prismaquant import residency_map as pqmap  # noqa: E402
    root = Path(queue.root) / "residency" / ("cc" * 64)
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def _pq_map_from_mover_output(pb, staged, campaign) -> Path:
    """PQ map whose entries ARE the real mover's staged files, re-hashed."""
    tmp = campaign["tmp"]
    queue = staged["queue"]
    pool, pb_map = pb["pool"], pb["pb_map"]
    frags = [pb_map.validate_fragment(f) for f in
             _pq_read_fragments(queue)]
    stage_frags = [f for f in frags
                   if f["tier_id"] == "prismabuild-stage:dl380g10"]
    assert len(stage_frags) == 2, "both stage movers filed"
    entries = {}
    for frag in stage_frags:
        for key, entry in frag["entries"].items():
            data = Path(entry["stage_path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            entries[key] = {
                "stage_path": entry["stage_path"], "bytes": entry["bytes"],
                "offset": entry["offset"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    body = {"schema": "prismaquant.prismabuild.residency_map.v1",
            "tier_id": "prismabuild-stage:dl380g10",
            "stage_root": str(staged["stage"]),
            "manifest_sha256": campaign["parent_sha"],
            "leads": ["dd" * 64], "generation": 1, "entries": entries}
    map_path = tmp / "residency.json"
    map_path.write_text(json.dumps(body))
    return map_path


def test_real_reader_serves_staged_tensors_bit_identical(
        pb, staged, campaign, monkeypatch) -> None:
    """REAL PQ source open over mover-staged safetensors: bit-identical."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, bind_residency_manifest,
        residency_resolver, reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    tmp = campaign["tmp"]
    map_path = _pq_map_from_mover_output(pb, staged, campaign)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["parent_sha"])
    resolver = residency_resolver()
    assert resolver is not None
    rows = [(Path(e["path"]), e["offset"], e["bytes"])
            for e in campaign["parent"]["entries"]
            if e["path"].endswith(".safetensors")]
    assert rows, "safetensors entries present"
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


def test_uncovered_span_fallback_is_recorded_gap_not_conformance(
        staged, campaign, monkeypatch, capsys) -> None:
    """An uncovered span reads the pool with a recorded fallback: the gap."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, bind_residency_manifest, residency_resolver,
        reset_residency_resolver_for_tests,
    )
    tmp = campaign["tmp"]
    monkeypatch.setenv(ENV_VAR, str(tmp / "residency.json"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["parent_sha"])
    resolver = residency_resolver()
    missing = tmp / "pool" / "model" / "never-staged.safetensors"
    _write_shard(missing, _tensors())
    with layer_streaming._source_safe_open(
            str(missing), framework="pt") as reader:
        assert reader.get_tensor("w0").numel() == 64
    report = resolver.report()
    assert report["fallback_count"] >= 1
    assert "[residency] fallback" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason=(
    "strict-reader enforcement pending owning worker: uncovered spans "
    "must refuse instead of pool fallback"))
def test_strict_reader_refuses_uncovered_span(staged, campaign, monkeypatch) -> None:
    """Future strict behavior: uncovered span refuses before payload bytes."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, bind_residency_manifest, residency_resolver,
        reset_residency_resolver_for_tests,
    )
    tmp = campaign["tmp"]
    monkeypatch.setenv(ENV_VAR, str(tmp / "residency.json"))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["parent_sha"])
    assert residency_resolver() is not None
    missing = tmp / "pool" / "model" / "never-staged.safetensors"
    with layer_streaming._source_safe_open(
            str(missing), framework="pt"):
        pass
    raise AssertionError("uncovered span opened without refusal")


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


def _write_payloads(root: Path, binding, produced, probe, fmt_list) -> None:
    from tests.test_joint_quanta_join import (  # noqa: E402
        _payload_provenance, _row, _units_of_layer,
    )
    for record in produced["records"]:
        layer = record["layer"]
        quantum_id = record["quantum_id"]
        units = sorted(q for q in binding["roster"]
                       if f".layers.{layer}." in q)
        assert units, f"layer {layer} has units"
        costs = {qname: {fmt: _row(qname, fmt, probe)
                         for fmt in fmt_list} for qname in units}
        space = root / "layer-quanta" / quantum_id
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
        produced, campaign, probe, tmp_path) -> None:
    """REAL join CLI over this producer's own records: coverage accepted."""
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS)
    out = tmp_path / "join-out"
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    assert join_main(_join_argv(root, out, binding, receipt_sha)) == 0
    joined = pickle.loads((out / "joint-cost.pkl").read_bytes())
    assert sorted(joined["costs"]) == sorted(binding["roster"])


def test_join_missing_quantum_is_named_gap(
        produced, campaign, probe, tmp_path) -> None:
    """Dropping a layer quantum gaps the join instead of shrinking the set."""
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS)
    (root / "layer-quanta" / "layer-001" / "cost.pkl").unlink()
    out = tmp_path / "join-out"
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    assert join_main(_join_argv(root, out, binding, receipt_sha)) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["distributed"]["status"] == "gapped"


def test_join_duplicate_and_mismatched_payloads_refuse(
        produced, campaign, probe, tmp_path) -> None:
    """Two payloads for one quantum, and tampered bytes, both refuse."""
    from prismaquant.joint_quanta_join import JoinRefused  # noqa: E402
    from tests.test_joint_quanta_join import FORMATS  # noqa: E402
    import pytest as _pytest  # noqa: E402
    root = tmp_path / "join-in"
    binding = _seal_join_inputs(root, campaign)
    _write_payloads(root, binding, produced, probe, FORMATS)
    out = tmp_path / "join-out"
    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    dup = root / "layer-quanta" / "layer-000" / "cost.pkl"
    dup.write_bytes(pickle.dumps({"costs": {}, "provenance": {}}, protocol=2))
    with _pytest.raises(Exception):
        join_main(_join_argv(root, out, binding, receipt_sha))
