"""Tests for prismaquant.joint_layer_quanta (distributed campaign §3–§5, §7).

The producer is pure, stdlib-only and torch-free: same inputs byte-identical
records. The real-plan tests read the sealed hostcap32 campaign read-only and
assert the 45 quanta, the coverage tiling, pairwise disjointness (read sets
and output spaces) and chunk-table sums. Nothing here submits anything.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os

import pytest

from prismaquant import joint_layer_quanta as jl


JOINT_PANEL = "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel"
OUTPUT_ROOT = os.path.join(
    JOINT_PANEL, "complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02")
PLAN_PATH = os.path.join(
    JOINT_PANEL, "complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json")
PREPARED_PATH = os.path.join(OUTPUT_ROOT, "prepare", "prepared.json")
MANIFEST_PATH = os.path.join(
    OUTPUT_ROOT, "data-manifests", "prismaquant.tessera_joint_aura.run.json.gz")

# The plan pin tracks the accepted takeover (manifest produced_by.plan_sha256,
# issue #839). The retired hostcap32 plan below carries only the sealed
# 360-window partition, read explicitly as producer input -- never as the
# plan under test.
PLAN_SHA256 = "0b2cc0066bb612e32af6d0c8c809912d325b2975583297eedeee97851ee545da"
PREPARED_SHA256 = "962207a3385e9531adaf951b823871a2fb7ff4684320e7a8e19a1d0aa85d8f16"

PARTITION_SOURCE_PATH = os.path.join(
    JOINT_PANEL, "complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.hostcap32.plan.json")
PARTITION_SOURCE_SHA256 = "eff2f7fb421aeadf8cde591b515ef6db4b68b1d4d9062c5f3a3a0c0a2c048cc5"


def _live_manifest_sha256():
    # The single-consumer campaign reseals its run manifest on resubmission
    # (234591 entries at the contract's 4100221caa… → 234590 live), so the
    # tests bind whatever bytes are sealed now instead of pinning stale ones.
    return _sha_file(MANIFEST_PATH)

GIB = 1024 ** 3

# The sealed hostcap32 campaign lives on the fleet's shared mount; skip --
# never fail -- where it is absent (GitHub CI has no /mnt/shared, the
# convention tests/test_glm_campaign_streaming.py:261 established).
real_plan = pytest.mark.skipif(
    not os.path.exists(PLAN_PATH),
    reason=f"the sealed campaign {PLAN_PATH} is absent (GitHub CI has no "
           f"/mnt/shared)")


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan():
    with open(PLAN_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_prepared():
    with open(PREPARED_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest():
    with gzip.open(MANIFEST_PATH, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _build_real():
    return jl.layer_quanta(
        _load_plan(), _load_prepared(), _load_manifest(),
        parent_manifest_sha256=_live_manifest_sha256(),
        window_partition=_sealed_window_partition())


def _sealed_window_partition():
    """The sealed 360-window partition as an explicit producer input.

    The accepted takeover plan carries no window derivation (it is the raw
    single-run plan under test), so the producer takes the partition
    explicitly. The bytes are the retired hostcap32 plan's sealed derivation
    block -- the same tiling the accepted banked output reflects -- read here
    as data (issue #839).
    """
    with open(PARTITION_SOURCE_PATH, "rb") as handle:
        raw = handle.read()
    assert hashlib.sha256(raw).hexdigest() == PARTITION_SOURCE_SHA256
    return json.loads(raw)["retained_window_budget_derivation"]


def _synthetic_inputs():
    """Two layers, a handful of entries: gap/overlap/misalignment fixtures."""
    model = "/mnt/shared/models/TEST"
    head_entries = [
        {"path": "/mnt/shared/inputs/plan.json", "offset": 0, "bytes": 100, "sha256": None},
        {"path": "/mnt/shared/inputs/calib.bin", "offset": 0, "bytes": 200, "sha256": None},
    ]
    layer0 = [
        {"path": f"{model}/shard-000.safetensors", "offset": 0, "bytes": 1000, "sha256": None},
        {"path": "/mnt/shared/rows/row-0/cache/a__F.pt", "offset": 0, "bytes": 100, "sha256": None},
        {"path": "/mnt/shared/rows/row-0/cache/b__F.pt", "offset": 0, "bytes": 100, "sha256": None},
        {"path": "/mnt/shared/rows/row-0/cache/c__F.pt", "offset": 0, "bytes": 100, "sha256": None},
    ]
    layer1 = [
        {"path": f"{model}/shard-001.safetensors", "offset": 0, "bytes": 2000, "sha256": None},
        {"path": "/mnt/shared/rows/row-1/cache/d__F.pt", "offset": 0, "bytes": 400, "sha256": None},
        {"path": "/mnt/shared/rows/row-1/cache/e__F.pt", "offset": 0, "bytes": 400, "sha256": None},
    ]
    entries = head_entries + layer0 + layer1
    total = 0
    phases = []
    for name, group in (("head", head_entries), ("layer-0", layer0), ("layer-1", layer1)):
        size = sum(e["bytes"] for e in group)
        total += size
        phases.append({"name": name, "bytes": size, "cumulative_bytes": total})
    parent = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fixture", "plan": "/mnt/shared/inputs/plan.json",
                        "plan_sha256": "0" * 64},
        "mount_prefix": "/mnt/shared",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": total,
        "annotations": {
            "entry_point": "prismaquant.tessera_joint_aura:run",
            "plan_sha256": "0" * 64,
            "layers": [0, 1],
            "campaign_scope": {"schema": "s", "kind": "fixture"},
            "argv": ["python3", "-m", "prismaquant.tessera_joint_aura", "run",
                     "--prepared", "/mnt/shared/inputs/prepared.json",
                     "--prepared-sha256", "1" * 64],
            "phases": phases,
        },
    }
    plan = {"output_root": "/mnt/shared/out",
            "model": model,
            "retained_window_budget_derivation": {"windows_by_layer": {"0": 1, "1": 2}}}
    prepared = {"formats_by_qname": {
        "model.language_model.layers.0.mlp.gate_proj": {"F": {}},
        "model.language_model.layers.1.mlp.gate_proj": {"F": {}},
        "model.language_model.layers.1.mlp.up_proj": {"F": {}, "G": {}},
    }}
    return plan, prepared, parent


def _campaign_of(records):
    first = records[0]["campaign"]
    return {
        "plan_sha256": first["plan_sha256"],
        "prepared_sha256": first["prepared_sha256"],
        "read_manifest_sha256": first["read_manifest_sha256"],
        "campaign_scope": first["campaign_scope"],
        "unit_roster_sha256": first["unit_roster_sha256"],
    }


# The sealed file digests are the campaign binding: if these move, every
# record's campaign block must move with them (checked again inside T2/T8).

@real_plan
def test_sealed_inputs_are_the_contract_campaign():
    assert _sha_file(PLAN_PATH) == PLAN_SHA256
    assert _sha_file(PREPARED_PATH) == PREPARED_SHA256
    manifest = _load_manifest()
    # The live run manifest reseals on resubmission (the contract cites
    # 4100221caa… at 234591 entries); continuity is the plan binding, the
    # layer set and the scope, not the file digest.
    assert manifest["annotations"]["plan_sha256"] == PLAN_SHA256
    assert manifest["annotations"]["layers"] == list(range(45))
    assert manifest["annotations"]["campaign_scope"]["kind"] == "complete_campaign"
    assert manifest["annotations"]["counts"]["renders"] == 197990


@real_plan
def test_determinism_real_plan_byte_identical():
    first = _build_real()
    second = _build_real()
    assert jl.canonical_bytes(first["records"]) == jl.canonical_bytes(second["records"])
    for record_a, record_b in zip(first["records"], second["records"]):
        assert record_a["identity_sha256"] == record_b["identity_sha256"]
    for key in first["slice_manifests"]:
        assert (jl.seal_manifest_bytes(first["slice_manifests"][key])
                == jl.seal_manifest_bytes(second["slice_manifests"][key]))
    assert (jl.seal_manifest_bytes(first["adjoint_manifest"])
            == jl.seal_manifest_bytes(second["adjoint_manifest"]))


@real_plan
def test_real_plan_45_quanta_shape():
    built = _build_real()
    records = built["records"]
    assert len(records) == 45
    assert [r["quantum_id"] for r in records] == [f"layer-{i:03d}" for i in range(45)]
    assert [r["layer"] for r in records] == list(range(45))
    assert set(built["slice_manifests"]) == {f"layer-{i:03d}" for i in range(45)}
    manifest = _load_manifest()
    # The campaign binding is one shared block across all 45 records.
    campaigns = {jl.canonical_sha256(r["campaign"]) for r in records}
    assert len(campaigns) == 1
    record = records[13]
    assert record["campaign"]["plan_sha256"] == PLAN_SHA256
    assert record["campaign"]["prepared_sha256"] == PREPARED_SHA256
    assert record["campaign"]["read_manifest_sha256"] == _live_manifest_sha256()
    assert record["campaign"]["campaign_scope"] == manifest["annotations"]["campaign_scope"]
    prepared = _load_prepared()
    roster = sorted(prepared["formats_by_qname"])
    assert len(roster) == 36423
    assert record["campaign"]["unit_roster_sha256"] == hashlib.sha256(
        "\n".join(roster).encode("utf-8")).hexdigest()
    assert record["campaign"]["plan_path"] == PLAN_PATH
    assert record["campaign"]["prepared_path"] == PREPARED_PATH
    # The sealed window partition: per-layer counts and the 360 total.
    plan = _load_plan()
    counts = _sealed_window_partition()["windows_by_layer"]
    assert sum(counts.values()) == 360
    assert sum(len(r["windows"]) for r in records) == 360
    for r in records:
        assert len(r["windows"]) == counts[str(r["layer"])]
        assert [w["window_index"] for w in r["windows"]] == list(range(len(r["windows"])))
    # Coverage proof is green on the real plan.
    coverage = jl.verify_quanta_coverage(
        records, manifest, plan=plan,
        window_partition=_sealed_window_partition())
    assert coverage["layers"] == list(range(45))
    assert coverage["window_total"] == 360
    assert built["coverage"]["coverage_sha256"] == coverage["coverage_sha256"]


@real_plan
def test_real_plan_chunk_derivation_pinned():
    built = _build_real()
    assert built["derivation"]["chunk_target_bytes"] == 40 * GIB
    assert built["derivation"]["ram_window_gib"] == 160
    assert built["derivation"]["max_resident_consumers"] == 2
    by_id = {r["quantum_id"]: r for r in built["records"]}
    assert [c["name"] for c in by_id["layer-003"]["chunks"]] == [
        f"layer-003-chunk-{i:03d}" for i in range(4)]
    assert len(by_id["layer-013"]["chunks"]) == 3  # 81.8 GiB at a 40 GiB target
    for layer in (0, 1, 2):  # 3.07 GiB layers are single-chunk
        assert len(by_id[f"layer-{layer:03d}"]["chunks"]) == 1
    for record in built["records"]:
        chunks = record["chunks"]
        total = record["read_set"]["total_bytes"]
        assert chunks[0]["start_bytes"] == 0
        assert chunks[-1]["end_bytes"] == total
        assert sum(c["end_bytes"] - c["start_bytes"] for c in chunks) == total
        for prev, cur in zip(chunks, chunks[1:]):
            assert prev["end_bytes"] == cur["start_bytes"]
        manifest = built["slice_manifests"][record["quantum_id"]]
        assert manifest["entry_count"] == record["read_set"]["entry_count"]
        assert manifest["total_bytes"] == total


@real_plan
def test_real_plan_stride_derivation_pinned():
    built = _build_real()
    assert built["derivation"]["stride"] == 8
    assert built["derivation"]["checkpoints"] == [8, 16, 24, 32, 40, 45]
    assert built["derivation"]["num_checkpoints"] == 6
    by_id = {r["quantum_id"]: r for r in built["records"]}
    assert by_id["layer-044"]["adjoint"] == {
        "checkpoint_boundary": 45, "chain_layers": [],
        "boundary_artifacts": os.path.join(OUTPUT_ROOT, "layer-quanta", "adjoint"),
        "receipt_sha256": None}
    assert by_id["layer-013"]["adjoint"]["checkpoint_boundary"] == 16
    assert by_id["layer-013"]["adjoint"]["chain_layers"] == [15, 14]
    assert by_id["layer-000"]["adjoint"]["checkpoint_boundary"] == 8
    assert by_id["layer-000"]["adjoint"]["chain_layers"] == [7, 6, 5, 4, 3, 2, 1]
    for record in built["records"]:
        assert len(record["adjoint"]["chain_layers"]) <= 7


@real_plan
def test_real_plan_read_sets_pairwise_disjoint():
    built = _build_real()
    seen = {}
    for quantum_id, manifest in built["slice_manifests"].items():
        for entry in manifest["entries"]:
            key = (entry["path"], entry["offset"])
            assert key not in seen, f"{key} in {quantum_id} and {seen.get(key)}"
            seen[key] = quantum_id
    parent = _load_manifest()
    head_end = parent["annotations"]["phases"][0]["cumulative_bytes"]
    head_count = 0
    running = 0
    for entry in parent["entries"]:
        running += entry["bytes"]
        head_count += 1
        if running == head_end:
            break
    assert running == head_end
    assert len(seen) == parent["entry_count"] - head_count
    # Output spaces are pairwise disjoint by key (§2.3: qnames embed the layer).
    roots = [r["output_space"]["root"] for r in built["records"]]
    assert len(set(roots)) == 45
    payloads = [r["output_space"]["cost_payload"] for r in built["records"]]
    assert len(set(payloads)) == 45
    for record in built["records"]:
        assert record["output_space"]["root"].endswith(
            os.path.join("layer-quanta", record["quantum_id"]))


@real_plan
def test_real_plan_output_space_layout():
    built = _build_real()
    record = built["records"][13]
    root = os.path.join(OUTPUT_ROOT, "layer-quanta", "layer-013")
    assert record["output_space"] == {
        "root": root,
        "cost_payload": os.path.join(root, "cost.pkl"),
        "results": os.path.join(root, "results.json"),
        "counters": os.path.join(root, "counters.json"),
        "checkpoint_dir": os.path.join(root, "checkpoints")}
    assert record["read_set"]["manifest_path"] == os.path.join(
        OUTPUT_ROOT, "layer-quanta", "manifests", "layer-013.data-manifest.json.gz")
    assert record["read_set"]["source_phase"]["name"] == "layer-13"


@real_plan
def test_real_plan_slice_manifests_are_standalone_v1():
    built = _build_real()
    for quantum_id, manifest in built["slice_manifests"].items():
        assert set(manifest) == {"schema", "produced_by", "mount_prefix", "entries",
                                 "entry_count", "total_bytes", "annotations"}
        assert manifest["schema"] == "prismaquant.prismabuild.data_manifest.v1"
        assert manifest["mount_prefix"] == "/mnt/shared"
        assert manifest["entry_count"] == len(manifest["entries"])
        assert manifest["total_bytes"] == sum(e["bytes"] for e in manifest["entries"])
        assert manifest["entry_count"] > 0
        for entry in manifest["entries"]:
            assert set(entry) == {"path", "offset", "bytes", "sha256"}
            assert entry["path"].startswith("/mnt/shared/")
            assert type(entry["offset"]) is int and entry["offset"] >= 0
            assert type(entry["bytes"]) is int and entry["bytes"] > 0
        assert len({(e["path"], e["offset"]) for e in manifest["entries"]}) == len(
            manifest["entries"])
        ann = manifest["annotations"]
        assert ann["entry_point"] == "prismaquant.joint_cost_quantum"
        assert ann["quantum_id"] == quantum_id
        assert ann["parent_manifest_sha256"] == _live_manifest_sha256()
        assert ann["plan_sha256"] == PLAN_SHA256
        assert ann["prepared_sha256"] == PREPARED_SHA256
        phases = ann["phases"]
        # No zero-byte head phase (PQ #849): PB voids a table whose
        # cumulative falls outside the entries' own prefix sums, so the
        # table starts at the first chunk. Startup/head progress is
        # declared separately by the dispatch lane, not here.
        assert len(phases) > 0
        assert phases[0]["bytes"] > 0 and phases[0]["cumulative_bytes"] > 0
        cumulative = 0
        names = set()
        for row in phases:
            assert set(row) == {"name", "bytes", "cumulative_bytes"}
            assert row["name"] not in names
            names.add(row["name"])
            cumulative += row["bytes"]
            assert row["cumulative_bytes"] == cumulative
        assert cumulative == manifest["total_bytes"]
        record = next(r for r in built["records"] if r["quantum_id"] == quantum_id)
        assert names == {c["name"] for c in record["chunks"]}
        blob = jl.seal_manifest_bytes(manifest)
        assert hashlib.sha256(blob).hexdigest() == [
            r for r in built["records"] if r["quantum_id"] == quantum_id][0][
            "read_set"]["manifest_sha256"]
        assert len(blob) <= 64 * 1024 * 1024


@real_plan
def test_real_plan_adjoint_manifest():
    from prismaquant.joint_layer_quanta import adjoint_read_plan_phase_names
    built = _build_real()
    manifest = built["adjoint_manifest"]
    assert manifest["schema"] == "prismaquant.prismabuild.data_manifest.v2"
    assert "phases" not in manifest["annotations"]
    assert manifest["annotations"]["entry_point"] == "prismaquant.joint_adjoint_capture"
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    assert names == list(adjoint_read_plan_phase_names(45))
    parent = _load_manifest()
    head_end = parent["annotations"]["phases"][0]["cumulative_bytes"]
    head_count = 0
    running = 0
    for entry in parent["entries"]:
        running += entry["bytes"]
        head_count += 1
        if running == head_end:
            break
    assert running == head_end
    # Stage A takes its head from the prepared completion (PQ #1051), so
    # the head keeps the parent's entries less the head walk's reads.
    from prismaquant.tessera_joint_aura import head_walk_read_set, is_head_walk_read
    read_set = head_walk_read_set(_load_plan()["inputs"])
    parent_head = parent["entries"][:head_count]
    kept = [e for e in parent_head if not is_head_walk_read(e["path"], read_set)]
    walk = [e for e in parent_head if is_head_walk_read(e["path"], read_set)]
    assert walk, "the real plan's parent head holds the walk's reads"
    assert manifest["annotations"]["head_walk_reads_dropped"] == {
        "entries": len(walk), "bytes": sum(e["bytes"] for e in walk)}
    head_count, head_end = len(kept), sum(e["bytes"] for e in kept)
    assert [{key: e[key] for key in ("path", "offset", "bytes")}
            for e in manifest["entries"][:head_count]] == [
        {key: e[key] for key in ("path", "offset", "bytes")} for e in kept]
    chain_entries = manifest["entries"][head_count:]
    assert manifest["read_plan"]["phases"][0]["bytes"] == head_end
    model_prefix = _load_plan()["model"].rstrip("/") + "/"
    for entry in chain_entries:
        assert entry["path"].startswith(model_prefix), entry["path"]
    assert len(chain_entries) == parent["annotations"]["counts"]["source_extents"]
    assert sum(e["bytes"] for e in chain_entries) == parent["annotations"]["bytes"][
        "source_extents"]
    # The repeated reverse reads reference the same entries, not copies.
    by_name = {p["name"]: p["entry_indices"]
               for p in manifest["read_plan"]["phases"]}
    head_size = len(manifest["entries"][:head_count])
    for layer in range(45):
        forward = by_name[f"forward-{layer:03d}"]
        assert by_name[f"chain-{layer:03d}"] == forward
        assert all(index >= head_size for index in forward)
    assert manifest["read_plan"]["read_bytes"] == sum(
        p["bytes"] for p in manifest["read_plan"]["phases"])
    assert manifest["total_bytes"] == head_end + parent["annotations"]["bytes"][
        "source_extents"]


@real_plan
def test_schema_every_record_real_plan():
    built = _build_real()
    for record in built["records"]:
        assert set(record) == {"schema", "quantum_id", "layer", "campaign", "read_set",
                               "chunks", "windows", "adjoint", "output_space",
                               "identity_sha256"}
        assert record["schema"] == "prismaquant.joint_layer_quanta.v1"
        assert record["quantum_id"] == f"layer-{record['layer']:03d}"
        assert type(record["layer"]) is int and 0 <= record["layer"] <= 44
        assert set(record["campaign"]) == {"plan_path", "plan_sha256", "prepared_path",
                                           "prepared_sha256", "read_manifest_sha256",
                                           "campaign_scope", "unit_roster_sha256"}
        assert set(record["read_set"]) == {"manifest_path", "manifest_sha256",
                                           "entry_count", "total_bytes", "source_phase"}
        assert set(record["read_set"]["source_phase"]) == {"name", "start_bytes",
                                                           "end_bytes"}
        for chunk in record["chunks"]:
            assert set(chunk) == {"name", "start_bytes", "end_bytes"}
            assert type(chunk["start_bytes"]) is int and type(chunk["end_bytes"]) is int
            assert 0 <= chunk["start_bytes"] < chunk["end_bytes"]
        for window in record["windows"]:
            assert set(window) == {"window_index"}
            assert type(window["window_index"]) is int
        assert set(record["adjoint"]) == {"checkpoint_boundary", "chain_layers",
                                          "boundary_artifacts", "receipt_sha256"}
        assert set(record["output_space"]) == {"root", "cost_payload", "results",
                                               "counters", "checkpoint_dir"}
        for key in ("plan_sha256", "prepared_sha256", "read_manifest_sha256",
                    "unit_roster_sha256", "manifest_sha256", "identity_sha256"):
            holder = record["campaign"] if key in record["campaign"] else record[
                "read_set"] if key in record["read_set"] else record
            value = holder[key]
            assert type(value) is str and len(value) == 64
            int(value, 16)
        headless = {k: v for k, v in record.items() if k != "identity_sha256"}
        assert jl.canonical_sha256(headless) == record["identity_sha256"]


@real_plan
def test_check_quantum_for_campaign_real_plan():
    built = _build_real()
    campaign = _campaign_of(built["records"])
    jl.check_quantum_for_campaign(built["records"][13], campaign)
    for field in ("plan_sha256", "prepared_sha256", "read_manifest_sha256",
                  "unit_roster_sha256"):
        bad = dict(campaign)
        bad[field] = "0" * 64
        with pytest.raises(ValueError):
            jl.check_quantum_for_campaign(built["records"][13], bad)
    bad = dict(campaign)
    bad["campaign_scope"] = {"schema": "s", "kind": "foreign"}
    with pytest.raises(ValueError):
        jl.check_quantum_for_campaign(built["records"][13], bad)
    # An unbound (pre-A) record refuses a bound campaign: stage A must be
    # sealed and the record rebound before any quantum publishes.
    bound = dict(campaign, adjoint_slice_sha256="a" * 64)
    with pytest.raises(ValueError, match="unbound"):
        jl.check_quantum_for_campaign(built["records"][13], bound)
    # A campaign still naming a whole receipt is refused, never ignored
    # (PQ #993: records bind their slice).
    legacy = dict(campaign, adjoint_receipt_sha256="a" * 64)
    with pytest.raises(ValueError, match="whole stage-A receipt"):
        jl.check_quantum_for_campaign(built["records"][13], legacy)
    edited = copy.deepcopy(built["records"][13])
    edited["chunks"] = edited["chunks"][:1]
    with pytest.raises(ValueError):
        jl.check_quantum_for_campaign(edited, campaign)


def test_chunk_target_derivation_pinned():
    assert jl.derive_chunk_target_bytes(160, 2) == 40 * GIB
    assert jl.derive_chunk_target_bytes(512, 2) == 64 * GIB  # clamped
    assert jl.derive_chunk_target_bytes(32, 2) == 8 * GIB  # exact floor, no refuse
    with pytest.raises(ValueError):  # a window under 32 GiB cannot serve 8 GiB chunks
        jl.derive_chunk_target_bytes(31, 2)
    with pytest.raises(ValueError):
        jl.derive_chunk_target_bytes(160, 0)


def test_chunk_cuts_are_entry_aligned_ceiling_cuts():
    ends = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    chunks = jl.derive_chunks("layer-000", 1000, ends, 400)
    assert [(c["start_bytes"], c["end_bytes"]) for c in chunks] == [
        (0, 400), (400, 700), (700, 1000)]
    assert [c["name"] for c in chunks] == [
        "layer-000-chunk-000", "layer-000-chunk-001", "layer-000-chunk-002"]
    single = jl.derive_chunks("layer-001", 300, [100, 200, 300], 40 * GIB)
    assert [(c["start_bytes"], c["end_bytes"]) for c in single] == [(0, 300)]
    with pytest.raises(ValueError):  # one entry larger than its share: refuse
        jl.derive_chunks("layer-002", 1000, [1000], 400)


def test_stride_derivation_pinned():
    derived = jl.derive_stride(45, 8)
    assert derived["checkpoints"] == [8, 16, 24, 32, 40, 45]
    assert derived["num_checkpoints"] == 6
    assert derived["max_chain_layers"] == 7
    assert jl.adjoint_binding(44, derived["checkpoints"]) == (45, [])
    assert jl.adjoint_binding(13, derived["checkpoints"]) == (16, [15, 14])
    assert jl.adjoint_binding(0, derived["checkpoints"]) == (8, [7, 6, 5, 4, 3, 2, 1])
    assert jl.adjoint_binding(39, derived["checkpoints"]) == (40, [])


def test_coverage_refuses_gap_overlap_misalignment_mixed_campaign():
    plan, prepared, parent = _synthetic_inputs()
    built = jl.layer_quanta(plan, prepared, parent, parent_manifest_sha256="f" * 64,
                            chunk_target_bytes=2000, stride=8)
    assert len(built["records"]) == 2
    jl.verify_quanta_coverage(built["records"], parent, plan=plan)
    missing = [r for r in built["records"] if r["layer"] != 1]
    with pytest.raises(ValueError, match="[Gg]ap|missing|coverage"):
        jl.verify_quanta_coverage(missing, parent, plan=plan)
    overlap = [copy.deepcopy(built["records"][0])]
    claimed_twice = copy.deepcopy(built["records"][0])
    claimed_twice["quantum_id"] = "layer-001"
    claimed_twice["layer"] = 1
    overlap.append(claimed_twice)
    with pytest.raises(ValueError, match="[Oo]verlap|tiling|coverage"):
        jl.verify_quanta_coverage(overlap, parent, plan=plan)
    shifted = copy.deepcopy(built["records"])
    shifted[0]["read_set"]["source_phase"]["end_bytes"] += 1
    with pytest.raises(ValueError, match="[Aa]lign|tiling|coverage|[Gg]ap"):
        jl.verify_quanta_coverage(shifted, parent, plan=plan)
    foreign = copy.deepcopy(built["records"])
    foreign[1]["campaign"]["plan_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="[Cc]ampaign|foreign|mixed"):
        jl.verify_quanta_coverage(foreign, parent, plan=plan)
    duplicated = built["records"] + [copy.deepcopy(built["records"][0])]
    with pytest.raises(ValueError, match="[Dd]uplicate|unique|coverage"):
        jl.verify_quanta_coverage(duplicated, parent, plan=plan)


def test_synthetic_windows_union_checked_against_plan():
    plan, prepared, parent = _synthetic_inputs()
    built = jl.layer_quanta(plan, prepared, parent, parent_manifest_sha256="f" * 64,
                            chunk_target_bytes=2000, stride=1)
    assert len(built["records"][0]["windows"]) == 1
    assert len(built["records"][1]["windows"]) == 2
    assert built["records"][1]["adjoint"]["chain_layers"] == []
    assert built["records"][0]["adjoint"] == {
        "checkpoint_boundary": 1, "chain_layers": [],
        "boundary_artifacts": "/mnt/shared/out/layer-quanta/adjoint",
        "receipt_sha256": None}


@real_plan
def test_canonical_identity_matches_cost_stage_checkpoint():
    from prismaquant import cost_stage_checkpoint
    built = _build_real()
    for record in built["records"][:3]:
        headless = {k: v for k, v in record.items() if k != "identity_sha256"}
        assert cost_stage_checkpoint.canonical_json_sha256(
            headless, where="quantum record") == record["identity_sha256"]


def test_canonical_helpers_delegate_to_the_shared_implementation():
    """#787 B5: the producer no longer carries a second canonical-JSON
    spelling; the wrappers are the shared cost_stage_checkpoint bytes."""
    from prismaquant import cost_stage_checkpoint
    value = {"b": [1, 2.5, None], "a": "ü", "nested": {"k": [True]}}
    assert jl.canonical_bytes(value) == cost_stage_checkpoint.canonical_json_bytes(
        value, where="delegation check")
    assert jl.canonical_sha256(value) == \
        cost_stage_checkpoint.canonical_json_sha256(value, where="delegation")
    with pytest.raises(ValueError):
        jl.canonical_bytes({"bad": float("nan")})


def test_qname_layer_reads_the_layer_grammar():
    """#787 D5: one spelling of the roster's layer grammar, shared with the
    joiner (gap unit naming)."""
    assert jl.qname_layer(
        "model.language_model.layers.13.mlp.gate_proj") == 13
    assert jl.qname_layer("model.layers.0.self_attn.q_proj") == 0
    assert jl.qname_layer("model.layers.notanumber.q_proj") is None
    assert jl.qname_layer("no.layers.here") is None
    assert jl.qname_layer(None) is None


def test_the_joiner_is_not_colocated():
    """#787 survival decision (D5): one joiner -- joint_quanta_join. The
    producer module is the data plane only; the parallel joiner that refused
    the merged joiner's bytes is gone, not adapted."""
    import prismaquant.joint_quanta_join

    assert not hasattr(jl, "join_layer_quanta")
    assert not hasattr(jl, "refuse_gapped_for_allocation")
    assert hasattr(prismaquant.joint_quanta_join, "join_joint_quanta")
    assert hasattr(prismaquant.joint_quanta_join, "load_joint_cost_for_allocation")
