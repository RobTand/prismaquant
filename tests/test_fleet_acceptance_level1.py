"""Level-1 connected acceptance: tiny CPU fixture through actual APIs.

Two halves, explicitly distinguished:

- Published-PB + PQ-main legs run GREEN here: real PQ writers, real
  published validation/stage/promotion/map flows, real PoolQueue
  lifecycle, real protocol-client refusals, real PQ reads and join
  gates over the harness's own fixture.
- Candidate-gated scenarios (broker Authority, SDK acquire/open/release,
  proof/cleanup/egress flows) execute in
  ``fleet_acceptance_runner.py`` subprocesses. A ``nonqualified`` result
  document becomes SKIP with its reason -- machine output, never a green
  conformance row. ``failed`` fails loudly. Only ``qualified`` asserts
  evidence.

No production file is edited or imported beyond the two real trees
(published fleet mount; verified candidate extraction).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fleet_acceptance_pins as pins  # noqa: E402
import fullstack_pb_generation as published  # noqa: E402
from test_fullstack_real_chain import (  # noqa: E402
    _entry,
    _safetensors_tensor_span,
    _tensors,
    _write_shard,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "fleet_acceptance_runner.py"
CONSUMER = "cc" * 32
STAGE_TIER = "prismabuild-stage:dl380g10"
RAM_TIER = "ram:dl380g10"


def _pb():
    """Published PB modules from the immutable generation (verified)."""
    info = published.require_paths()
    import prismabuild.core as core  # noqa: E402
    import prismabuild.pool as pool  # noqa: E402
    import prismabuild.residency_map as pmap  # noqa: E402
    import prismabuild.resource_scope as scopes  # noqa: E402
    import prismabuild.storage_tiers as tiers  # noqa: E402
    import stage_move  # noqa: E402
    import ram_promote  # noqa: E402
    root = info.get("root") or str(
        Path(core.__file__).resolve().parents[3])
    for module in (core, pool, pmap, scopes, tiers, stage_move,
                   ram_promote):
        location = Path(module.__file__).resolve()
        assert str(location).startswith(root), (
            f"{module.__name__} loaded from {location}, not {root}")
    return {"generation": info["generation"], "root": root, "core": core,
            "pool": pool, "pmap": pmap, "scopes": scopes, "tiers": tiers,
            "stage_move": stage_move, "ram_promote": ram_promote}


@pytest.fixture(scope="module")
def pb():
    return _pb()


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
    receipt = {
        "schema": "prismaquant.joint_adjoint_capture.v1",
        "plan_sha256": sealed["plan.json"],
        "prepared_sha256": sealed["prepared.json"],
        "campaign_scope": parent["annotations"]["campaign_scope"],
        "checkpoints": [{"boundary": mark} for mark in (1, 2)],
        "status": "complete",
    }
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


# -- ID-03/ID-04/ACC-01: real-producer parser fixtures ------------------------


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
        assert isinstance(adjoint["receipt_sha256"], str)
        assert isinstance(record["windows"], list), record["quantum_id"]
        campaign = record["campaign"]
        for key in ("plan_sha256", "prepared_sha256",
                    "read_manifest_sha256", "campaign_scope"):
            assert campaign.get(key), (record["quantum_id"], key)


# -- published PB: validate, stage, promote, map -----------------------------


def test_published_pb_parses_slice_ranges(pb, produced, campaign):
    """Slice tables parse into stageable ranges with exact tiling."""
    core, tiers = pb["core"], pb["tiers"]
    for record in produced["records"]:
        validated = core.validate_data_manifest(
            _slice_body(campaign, record))
        assert validated["entry_count"] == len(validated["entries"])
        ranges = tiers.manifest_phase_ranges(validated)
        assert ranges, record["quantum_id"]
        assert ranges[0]["start_bytes"] == 0
        assert ranges[-1]["end_bytes"] == validated["total_bytes"]


@pytest.fixture(scope="module")
def staged(pb, produced, campaign):
    """Real published movers over the parent manifest (whole + split)."""
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
    manifest_path = tmp / "staging.json"
    manifest_path.write_text(json.dumps(campaign["parent"], sort_keys=True))
    total = campaign["parent"]["total_bytes"]
    split = campaign["parent"]["entries"][0]["bytes"]
    receipts = {}
    for mover, start, end in (("aa" * 32, 0, split),
                              ("ab" * 32, split, total)):
        args = stage_move.build_parser().parse_args([
            "--pool-root", str(tmp / "pb-queue"),
            "--cas-root", str(tmp / "pb-queue" / "cas"),
            "--action-key", mover,
            "--consumer-action-key", CONSUMER,
            "--tier-id", STAGE_TIER,
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
        receipts[mover] = receipt
    prom = ram_promote.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--action-key", "bb" * 32,
        "--consumer-action-key", CONSUMER,
        "--tier-id", RAM_TIER,
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
            "epoch": str(epoch["epoch"]), "receipts": receipts,
            "promotion": promotion}


def test_movers_publish_fragments_with_real_digests(pb, staged, campaign):
    """SM-02 legs: published map names staged bytes under epoch digests."""
    pb_map = pb["pmap"]
    pool = pb["pool"]
    queue = staged["queue"]
    frags = [pb_map.validate_fragment(f) for f in
             pb_map.read_fragments(queue.root / pool.RESIDENCY, CONSUMER)]
    assert len(frags) == 3
    composed = pb_map.compose(
        [f for f in frags if f["tier_id"] == STAGE_TIER])
    assert composed["manifest_sha256"] == campaign["parent_sha"]
    for key, entry in composed["entries"].items():
        data = Path(entry["stage_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], key


def test_mover_overrun_refuses_before_copy(pb, staged, campaign):
    """SM-02 guard: a range smaller than its covering entries refuses."""
    stage_move, pool = pb["stage_move"], pb["pool"]
    tmp = campaign["tmp"]
    total = campaign["parent"]["total_bytes"]
    args = stage_move.build_parser().parse_args([
        "--pool-root", str(tmp / "pb-queue"),
        "--cas-root", str(tmp / "pb-queue" / "cas"),
        "--action-key", "cc" * 32,
        "--consumer-action-key", CONSUMER,
        "--tier-id", STAGE_TIER,
        "--stage-root", str(tmp / "stage-overrun"),
        "--manifest-sha256", campaign["parent_sha"],
        "--range-start-bytes", "1",
        "--range-end-bytes", str(total),
        "--manifest", str(tmp / "staging.json"),
        "--residency-root", str(tmp / "pb-queue" / pool.RESIDENCY),
        "--block", str(1 << 16),
        "--readers", "1", "--max-readers", "1", "--unpaced",
    ])
    with pytest.raises(SystemExit):
        stage_move.move(args)


def test_ram_overlay_refuses_stale_epoch(pb, staged, campaign):
    """SM-02 readiness: a prior epoch's fragment never overlays as current."""
    pb_map = pb["pmap"]
    pool = pb["pool"]
    queue = staged["queue"]
    frags = [pb_map.validate_fragment(f) for f in
             pb_map.read_fragments(queue.root / pool.RESIDENCY, CONSUMER)]
    stage_frags = [f for f in frags if f["tier_id"] == STAGE_TIER]
    ram_frags = [f for f in frags if f["tier_id"] == RAM_TIER]
    assert ram_frags, "a ram promotion filed"
    composed = pb_map.compose(stage_frags)
    with pytest.raises(pb_map.ResidencyMapError):
        pb_map.overlay_ram(composed, ram_frags, ram_tier_id=RAM_TIER,
                           ram_root=str(staged["ram"]),
                           ram_epoch="stale-" + staged["epoch"])


# -- published PoolQueue: lifecycle, ledger, finish --------------------------


def test_publish_claim_finish_lifecycle(pb, tmp_path):
    """SM-01/SAFE-03 legs: a real row is published, claimed, finished."""
    pool = pb["pool"]
    queue = pool.PoolQueue(tmp_path / "queue")
    queue.ensure_layout()
    key = "d" * 64
    (tmp_path / "cas").mkdir()
    (tmp_path / "co").mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("import sys\n")
    queue.publish(action_key=key, cas_root=tmp_path / "cas",
                  checkout_root=tmp_path / "co", worker_script=worker,
                  resources={"cpu": 1, "mem_gb": 1},
                  max_attempts=1, retry_safe=True)
    assert (queue.root / "ready" / f"{key}.json").is_file()
    record = queue.claim(owner="fleet-harness:1:lifecycle",
                         capacity={"cpu": 1, "mem_gb": 1})
    assert record is not None, "real claim refused a published cpu row"
    assert record["action_key"] == key
    assert queue.ledger().held() != {}, "claim committed no local tokens"
    terminal = queue.finish(key, status="executed",
                            detail={"acceptance": "lifecycle"},
                            claim_snapshot=record)
    assert Path(terminal).is_file()
    assert queue.ledger().held() == {}, "finish returned no capacity"


def test_tier_ledger_concludes_both_sides_once(pb, tmp_path):
    """INV-01 legs: one release helper concludes holder and free tokens."""
    pool = pb["pool"]
    queue = pool.PoolQueue(tmp_path / "queue")
    queue.ensure_layout()
    queue.mint_tier_capacity(STAGE_TIER, {"stage_gib": 4})
    ledger = queue.tier_ledger(STAGE_TIER)
    assert ledger.acquire("ee" * 32, {"stage_gib": 1}) is True
    assert ledger.holder_tokens("ee" * 32) == {"stage_gib": 1}
    queue.release_tier_reservations("ee" * 32)
    assert ledger.holder_tokens("ee" * 32) == {}
    assert ledger.available() == {"stage_gib": 4}


# -- published protocol client: refusals, never invented replies ------------


def test_scope_identity_validates_action_key_and_nonce(pb):
    """Real constructor refuses malformed identity (no broker needed)."""
    scopes = pb["scopes"]
    with pytest.raises(ValueError):
        scopes.ResourceScope("short", "b" * 32, 64 << 20,
                             Path("/tmp/x.json"))
    with pytest.raises(ValueError):
        scopes.ResourceScope("b" * 64, "short", 64 << 20,
                             Path("/tmp/x.json"))


def test_scope_adoption_refuses_forged_replies(pb, tmp_path):
    """The client-side acceptance predicate refuses malformed verdicts."""
    scopes = pb["scopes"]
    scope = scopes.ResourceScope("b" * 64, "b" * 32, 64 << 20,
                                 tmp_path / "telemetry.json")
    good_unit = "prismabuild-job" + hashlib.sha256(
        ("b" * 64 + "b" * 32).encode()).hexdigest()[:32] + ".slice"
    good = {"scope_id": good_unit, "token": "c" * 64,
            "cgroup_path": f"/sys/fs/cgroup/prismabuild.slice/{good_unit}"}
    scope._adopt_created_scope(dict(good))
    assert scope.unit == good_unit
    bad_unit = dict(good, scope_id="prismabuild-job" + "0" * 32 + ".slice")
    with pytest.raises(OSError):
        scopes.ResourceScope("b" * 64, "b" * 32, 64 << 20,
                             tmp_path / "t2.json")._adopt_created_scope(bad_unit)
    bad_token = dict(good, token="not-hex")
    with pytest.raises(OSError):
        scopes.ResourceScope("b" * 64, "b" * 32, 64 << 20,
                             tmp_path / "t3.json")._adopt_created_scope(bad_token)
    bad_path = dict(good, cgroup_path="/elsewhere/x.slice")
    with pytest.raises(OSError):
        scopes.ResourceScope("b" * 64, "b" * 32, 64 << 20,
                             tmp_path / "t4.json")._adopt_created_scope(bad_path)


def test_broker_request_without_server_is_a_refusal(pb, tmp_path):
    """No peer, no verdict: the real call fails instead of inventing one."""
    scopes = pb["scopes"]
    with pytest.raises(OSError):
        scopes.broker_request({"op": "status"},
                              socket_path=tmp_path / "absent.sock")


# -- PQ reads + join gates over the harness staged fixture -------------------


def _harness_map(pb, staged, campaign):
    pool, pmap = pb["pool"], pb["pmap"]
    queue = staged["queue"]
    frags = [pmap.validate_fragment(f) for f in
             pmap.read_fragments(queue.root / pool.RESIDENCY, CONSUMER)]
    stage_frags = [f for f in frags if f["tier_id"] == STAGE_TIER]
    assert len(stage_frags) == 2
    composed = pmap.compose(stage_frags)
    assert composed["manifest_sha256"] == campaign["parent_sha"]
    maps = campaign["tmp"] / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    path = maps / "residency.json"
    path.write_text(json.dumps(composed))
    return path


def test_staged_reads_are_bit_identical_with_stage_bytes_counted(
        pb, staged, campaign, monkeypatch):
    """Real PQ source reads over published-staged bytes; stage serves."""
    from prismaquant import layer_streaming  # noqa: E402
    from prismaquant.residency_map import (  # noqa: E402
        ENV_VAR, bind_residency_manifest, residency_resolver,
        reset_residency_resolver_for_tests,
    )
    from safetensors import safe_open  # noqa: E402
    tmp = campaign["tmp"]
    map_path = _harness_map(pb, staged, campaign)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(campaign["parent_sha"])
    resolver = residency_resolver()
    assert resolver is not None
    for entry in campaign["parent"]["entries"]:
        if not entry["path"].endswith(".safetensors"):
            continue
        with layer_streaming._source_safe_open(
                entry["path"], framework="pt") as reader:
            with safe_open(entry["path"], framework="pt") as reference:
                for name in reference.keys():
                    got = reader.get_tensor(name)
                    want = reference.get_tensor(name)
                    if got.numel():
                        assert torch.equal(
                            got.view(torch.uint8),
                            want.view(torch.uint8)), name
    report = resolver.report()
    assert report["fallback_count"] == 0
    assert report["bytes_from_stage"] > 0


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

    def argv(root, out, binding, receipt_sha):
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

    receipt_sha = produced["records"][0]["adjoint"]["receipt_sha256"]
    root = tmp_path / "join-in"
    binding = seal(root)
    write_payloads(root, binding)
    out = tmp_path / "join-out"
    assert join_main(argv(root, out, binding, receipt_sha)) == 0
    allocated = load_joint_cost_for_allocation(out / "joint-cost.pkl")
    assert sorted(allocated["costs"]) == sorted(binding["roster"])
    (root / "layer-quanta" / "records" / "layer-001.json").unlink()
    out2 = tmp_path / "join-out-2"
    assert join_main(argv(root, out2, binding, receipt_sha)) == 0
    results = json.loads((out2 / "results.json").read_text())
    assert results["status"] == "gapped"
    with pytest.raises(GappedPayloadRefused):
        load_joint_cost_for_allocation(out2 / "joint-cost.pkl")


# -- candidate-gated scenarios: machine nonqualified output, never green ----


@pytest.fixture(scope="module")
def candidate_work(tmp_path_factory):
    """One shared resolution dir; scenarios run in fresh subdirs below it."""
    return tmp_path_factory.mktemp("candidate")


def _run_scenario(name: str, tmp_path: Path, scenario_dir: str,
                  candidate_work=None) -> dict:
    """Run one runner scenario; skip on named nonqualification."""
    base = candidate_work if candidate_work is not None else tmp_path
    work = base / scenario_dir
    result = base / (scenario_dir + ".json")
    # The runner nests one level by scenario name under --work.
    notes = work / name / "notes.json"
    done = subprocess.run(
        [sys.executable, str(RUNNER), name, "--work", str(work),
         "--result", str(result), "--checkout", str(ROOT)],
        capture_output=True, text=True, timeout=600)
    assert result.is_file(), (
        f"runner left no result document: rc={done.returncode}\n"
        f"{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    doc = json.loads(result.read_text())
    assert doc["schema"] == "prismaquant.fleet_acceptance.result.v1", doc
    assert doc["scenario"] == name, doc
    if doc["status"] == "nonqualified":
        print(f"nonqualified {name}: {doc['reason']}")
        pytest.skip(f"nonqualified: {doc['reason']}")
    if doc["status"] != "qualified":
        print("EVIDENCE " + json.dumps(doc.get("evidence", {}),
                                       sort_keys=True)[:15000])
        if notes.is_file():
            print("NOTES " + notes.read_text()[:8000])
    assert doc["status"] == "qualified", doc["reason"]
    return doc


def test_resolve_names_the_verified_tree(tmp_path, candidate_work):
    """The pin resolves to a verified tree or skips with the reason."""
    doc = _run_scenario("resolve", tmp_path, "resolve", candidate_work)
    assert doc["evidence"]["rev"] == pins.PB_CANDIDATE_REV
    assert len(doc["evidence"]["tree_sha256"]) == 64


def test_broker_roundtrip_is_real_client_and_authority(tmp_path, candidate_work):
    """Broker ops go through the real client, socket, and Authority."""
    doc = _run_scenario("broker-roundtrip", tmp_path, "broker", candidate_work)
    assert doc["evidence"]["export"]["stopped"] is True


def test_sdk_first_release_reclaims_once(tmp_path, candidate_work):
    """R8 core: export proof, reclaim-once, finish, egress (RED on R7)."""
    doc = _run_scenario("sdk-first-release", tmp_path, "first-release", candidate_work)
    assert doc["evidence"]["reclaimed_once"] is True
    assert doc["evidence"]["export"]["stopped"] is True
    assert isinstance(doc["evidence"]["export"]["empty"], bool)


def test_sdk_pending_ticket_flows_or_names_its_gate(tmp_path, candidate_work):
    """Ticket settle flow; cgroup-gated begin is a named nonqualification."""
    doc = _run_scenario("sdk-pending-ticket", tmp_path, "pending-ticket", candidate_work)
    assert doc["evidence"]["retired"] is True


def test_sdk_namespace_separation_holds(tmp_path, candidate_work):
    """Two owners pin separately; one release keeps the other's pin."""
    doc = _run_scenario("sdk-namespace-separation", tmp_path, "namespaces", candidate_work)
    assert len(doc["evidence"]["owner_dirs"]) == 2


def test_sdk_failure_unwind_releases(tmp_path, candidate_work):
    """Mid-hold errors unwind exactly; a fresh acquire still works."""
    doc = _run_scenario("sdk-failure-unwind", tmp_path, "unwind", candidate_work)
    assert doc["evidence"]["unwound_once"] is True


def test_sdk_alias_host_derives_on_real_records(tmp_path, candidate_work):
    """Holder derivation runs on the live claim; evidence names both."""
    doc = _run_scenario("sdk-alias-host", tmp_path, "alias", candidate_work)
    assert doc["evidence"]["derived_host"]
