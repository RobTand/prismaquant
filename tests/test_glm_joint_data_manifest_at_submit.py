"""The post-campaign GPU passes declare the bytes they read, in read order.

PR #524 gave every campaign row a PrismaBuild data manifest. The passes that
run *after* the rows merge -- ``prismaquant.tessera_joint_aura prepare`` and
``run``, the allocation handoff and the serving export -- are submitted
outside the row dispatcher and carried none, so PrismaBuild's prewarm loop
could not see their read set at all.

The joint pass is where that costs the most: 5.20 TB measured at 07:57Z on
2026-09-13, and still growing, against the 240 GiB ARC on dl380g10. No warm of the whole set is possible, so a manifest that only
listed the files would be useless -- the loop has to know *when* each byte is
read. These tests hold the consumption order, the per-layer phase boundaries
and the exclusions that make the order true:

* the head set first, then one ``layer-<L>`` phase per transformer layer;
* a rung the campaign never measured contributes no wire, even though its blob
  sits in the same directory as the measured ones;
* a rung the campaign adopted has a wire and no decoded shard, so the head
  decodes one from that wire: those wire bytes belong to the head phase, not
  to the layer that later verifies them;
* ``run`` hashes every cell up front, so its wires and renders are declared in
  that order and the streaming re-read is accounted for rather than declared
  twice, which the contract refuses;
* a joint pass submitted before the campaign merge published the checkpoint's
  unit shards is refused with the directory named, not submitted blind;
* the submit command puts ``--data-manifest`` before ``--detach`` and mirrors
  the chain's demand, because a ``pbrun`` option after the separator is an
  argument of the action instead.

The tests are torch-free on purpose: they run on the fleet's CPU boxes, which
have no torch, and the producer they exercise imports nothing from the
``prismaquant`` package for the same reason.
"""

import gzip
import hashlib
import json
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from experiments import glm_data_manifests  # noqa: E402

PREFIX = "model.language_model.layers."
UNITS = {
    0: [PREFIX + "0.mlp.gate_proj", PREFIX + "0.mlp.up_proj"],
    1: [PREFIX + "1.mlp.gate_proj", PREFIX + "1.mlp.up_proj"],
}
#: Two rungs the campaign measured and one it did not. The unmeasured rung's
#: wire blob is written into the same directory, because that is exactly how
#: the frozen campaign looks: the row caches hold the whole menu and the joint
#: pass opens only the anchors.
MEASURED = ("TESSERA_BF16_K1_R832", "TESSERA_E4M3_K1_R1024")
UNMEASURED = "TESSERA_E2M1_K2_R896"

TENSOR_BYTES = 2 * (1 << 20)
PRODUCED_BY = {"tool": "test", "commit": "0" * 40}


def _wire_filename(qname: str, fmt: str) -> str:
    return qname.replace(".", "__") + "__" + fmt + ".tessera"


def _write_unit_shard(parts: Path, qname: str, wire_bytes: dict) -> str:
    state = {
        "anchors": [{"qname": qname, "format_name": fmt} for fmt in MEASURED],
        "wire_records": {
            fmt: {"file": _wire_filename(qname, fmt),
                  "blob_bytes": wire_bytes[qname, fmt],
                  "blob_sha256": "0" * 64}
            for fmt in MEASURED},
    }
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": glm_data_manifests.CHECKPOINT_UNIT_SCHEMA,
        "stage": "Tessera campaign",
        "qname": qname,
        "identity_sha256": "1" * 64,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
    }
    relative = "units/" + hashlib.sha256(qname.encode()).hexdigest() + ".pkl"
    path = parts / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    return relative


def _workspace(scratch: Path) -> dict:
    """A two-layer campaign and a joint plan over it, on disk, nothing mocked."""
    names = [name for layer in sorted(UNITS) for name in UNITS[layer]]

    model = scratch / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    header = {}
    for index, name in enumerate(names):
        header[name + ".weight"] = {
            "dtype": "BF16", "shape": [TENSOR_BYTES // 2],
            "data_offsets": [index * TENSOR_BYTES, (index + 1) * TENSOR_BYTES]}
    blob = json.dumps(header).encode()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob
                      + b"\0" * (TENSOR_BYTES * len(names)))
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name + ".weight": shard.name for name in names}}))

    captures = scratch / "calibration-cache"
    (captures / "inputs").mkdir(parents=True)
    entries = {}
    for index, name in enumerate(names):
        relative = f"inputs/{index}.pt"
        (captures / relative).write_bytes(b"c" * (4096 + index))
        entries[name] = {"path": relative}
    capture_manifest = captures / "capture_manifest.json"
    capture_manifest.write_text(json.dumps({"entries": entries}))

    workspace = scratch / "workspace"
    rows, units_dir = [], workspace / "units"
    units_dir.mkdir(parents=True)
    for layer in sorted(UNITS):
        row_id = f"row-{layer:04d}"
        row_dir = workspace / "rows" / row_id
        (row_dir / "cache").mkdir(parents=True)
        units_file = units_dir / f"{row_id}.json"
        units_file.write_text(json.dumps(
            {"groups": [{"members": UNITS[layer]}]}))
        rows.append({"row_id": row_id, "dir": str(row_dir),
                     "units": str(units_file), "members": UNITS[layer],
                     "groups": [f"g:{layer}"]})
    campaign_plan = workspace / "plan.json"
    campaign_plan.write_text(json.dumps({
        "model": str(model),
        "calibration_cache": {"path": str(capture_manifest)},
        "rows": rows,
    }))
    receipts = workspace / "receipts.json"
    receipts.write_text(json.dumps({"rows": len(rows)}))
    census = workspace / "census.json"
    census.write_text(json.dumps({"unit_shapes": {name: [16, 16] for name in names}}))

    wire_dir = workspace / "merged" / "cache" / "wire"
    wire_dir.mkdir(parents=True)
    wire_bytes = {}
    for index, name in enumerate(names):
        for offset, fmt in enumerate(MEASURED):
            size = 8192 + 100 * index + offset
            (wire_dir / _wire_filename(name, fmt)).write_bytes(b"w" * size)
            wire_bytes[name, fmt] = size
        # The rung no anchor priced. Its blob is real and must stay out.
        (wire_dir / _wire_filename(name, UNMEASURED)).write_bytes(b"x" * 7777)

    for layer in sorted(UNITS):
        cache = workspace / "rows" / f"row-{layer:04d}" / "cache"
        for index, name in enumerate(UNITS[layer]):
            for offset, fmt in enumerate(MEASURED):
                filename = glm_data_manifests._cache_weight_filename(name, fmt)
                (cache / filename).write_bytes(b"r" * (2048 + 10 * index + offset))

    merged = workspace / "merged"
    cost = merged / "cost.pkl"
    cost.write_bytes(pickle.dumps({"provenance": {"wire_dir": str(wire_dir)}},
                                  protocol=pickle.HIGHEST_PROTOCOL))
    parts = merged / "cost.anchors.json.parts"
    checkpoint_units = [
        {"qname": name, "file": _write_unit_shard(parts, name, wire_bytes)}
        for name in names]
    checkpoint = merged / "cost.anchors.json"
    checkpoint.write_text(json.dumps({
        "schema": glm_data_manifests.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "Tessera campaign",
        "units": checkpoint_units,
    }))

    calibration = scratch / "calibration_tokens.safetensors"
    calibration.write_bytes(b"t" * 4096)
    compatibility = scratch / "capture-compatibility.json"
    compatibility.write_text(json.dumps({"schema": "x"}))
    backend = scratch / "projection.so"
    backend.write_bytes(b"e" * 2048)

    output_root = scratch / "joint-01"
    plan = scratch / "plan.inputs-resolved.json"
    plan.write_text(json.dumps({
        "schema": glm_data_manifests.JOINT_PLAN_SCHEMA,
        "model": str(model),
        "output_root": str(output_root),
        "calibration_input": {"path": str(calibration), "sha256": None},
        "canonical_capture": {"path": str(capture_manifest), "sha256": None},
        "source_capture_compatibility": {"path": str(compatibility), "sha256": None},
        "execution": {"projection_backend": {
            "binary": {"path": str(backend), "sha256": None}, "name": "f"}},
        "qualification_window": {"max_load_buffer_bytes": 1},
        "inputs": {
            "campaign_plan": {"path": str(campaign_plan), "sha256": None},
            "census": {"path": str(census), "sha256": None},
            "campaign_receipts": {"path": str(receipts), "sha256": None},
            "merged_cost": {"path": str(cost), "sha256": None},
            "merged_checkpoint": {"path": str(checkpoint), "sha256": None},
        },
    }))
    return {"plan": plan, "workspace": workspace, "wire_dir": wire_dir,
            "checkpoint": checkpoint, "parts": parts, "names": names,
            "shard": shard, "captures": captures, "model": model,
            "output_root": output_root}


@pytest.fixture()
def scratch(request):
    """A per-test directory that is never under ``/tmp``.

    ``pytest``'s own ``tmp_path`` lands wherever ``TMPDIR`` points, and
    ``pbtest`` refuses a ``--basetemp`` override, so the one place a fleet
    shard can be told to write is a path this module chooses. /tmp was cleared
    by an OOM in 2026-04 and took a set of artifacts with it; nothing here
    writes there.
    """
    root = ROOT / ".data-manifest-scratch" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def shared_mount(scratch, monkeypatch):
    """The fixture's files are not on /mnt/shared, so the prefix moves to it."""
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(scratch))
    return scratch


def _phase(manifest, name):
    return next(phase for phase in manifest["annotations"]["phases"]
                if phase["name"] == name)


def _paths(manifest, phase_name):
    """The entries belonging to one phase, in manifest order."""
    phases = manifest["annotations"]["phases"]
    index = next(i for i, phase in enumerate(phases) if phase["name"] == phase_name)
    start = 0 if index == 0 else phases[index - 1]["cumulative_bytes"]
    end = phases[index]["cumulative_bytes"]
    seen, out = 0, []
    for entry in manifest["entries"]:
        if start <= seen < end:
            out.append(entry["path"])
        seen += entry["bytes"]
    return out


def test_the_prepare_manifest_is_the_head_then_one_phase_per_layer(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    names = [phase["name"] for phase in manifest["annotations"]["phases"]]
    assert names == ["head", "layer-0", "layer-1"]
    assert manifest["annotations"]["layers"] == [0, 1]
    assert manifest["annotations"]["entry_point"] == (
        "prismaquant.tessera_joint_aura:prepare")

    # The phase sums are a running total over ``entries``; the prewarm loop
    # stops at a byte budget and needs the boundary to be a property of the
    # read, not a guess.
    running = 0
    for phase in manifest["annotations"]["phases"]:
        running += phase["bytes"]
        assert phase["cumulative_bytes"] == running
        assert set(phase) == {"name", "bytes", "cumulative_bytes"}
    assert running == manifest["total_bytes"]
    assert sum(entry["bytes"] for entry in manifest["entries"]) == running

    head = _paths(manifest, "head")
    assert head[0] == str(fixture["plan"])
    assert str(fixture["checkpoint"]) in head
    assert all(str(fixture["parts"]) not in path for path in _paths(manifest, "layer-0"))
    # Every unit shard of the merged checkpoint is read before the first layer.
    assert manifest["annotations"]["counts"]["checkpoint_units"] == len(fixture["names"])


def test_a_layer_phase_reads_source_then_capture_then_render_then_wire(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    layer0 = _paths(manifest, "layer-0")
    kinds = []
    for path in layer0:
        if path.endswith(".safetensors"):
            kinds.append("source")
        elif "/inputs/" in path:
            kinds.append("capture")
        elif path.endswith(".pt"):
            kinds.append("render")
        elif path.endswith(".tessera"):
            kinds.append("wire")
        else:  # pragma: no cover - a kind the fixture does not produce
            kinds.append(path)
    # The runner installs the layer's source weights, then walks its units in
    # sorted order: that unit's capture, then each measured rung's render and
    # then its wire.
    assert kinds == ["source",
                     "capture", "render", "wire", "render", "wire",
                     "capture", "render", "wire", "render", "wire"]
    assert manifest["annotations"]["capture_window"] == "per_unit"


def test_prepare_manifest_traces_full_source_sha_at_first_use_and_completion(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    model = fixture["model"]
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())

    def add_shard(filename, tensor):
        header = json.dumps({tensor: {"dtype": "BF16", "shape": [1],
                                      "data_offsets": [0, 2]}}).encode()
        path = model / filename
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
        index["weight_map"][tensor] = filename
        return path

    head = add_shard("head.safetensors", "lm_head.weight")
    untouched = add_shard("mtp.safetensors", "model.mtp.weight")
    index_path.write_text(json.dumps(index))

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)
    phases = manifest["annotations"]["phases"]
    assert [phase["name"] for phase in phases] == [
        "head", "layer-0", "layer-1", "source-complete"]

    def entries(phase):
        start = next((phases[i - 1]["cumulative_bytes"]
                      for i, item in enumerate(phases) if item["name"] == phase and i), 0)
        end = next(item["cumulative_bytes"] for item in phases if item["name"] == phase)
        seen = 0
        selected = []
        for entry in manifest["entries"]:
            if start <= seen < end:
                selected.append(entry)
            seen += entry["bytes"]
        return selected

    def whole_shard(phase, path):
        return any(row["path"] == str(path) and row["offset"] == 0
                   and row["bytes"] == path.stat().st_size for row in entries(phase))

    assert whole_shard("head", head)
    assert whole_shard("layer-0", fixture["shard"])
    assert not whole_shard("layer-1", fixture["shard"])
    assert whole_shard("source-complete", untouched)
    assert str(index_path) in _paths(manifest, "head")
    assert manifest["annotations"]["counts"]["source_authentication"] == 3


def test_mtp_index_after_backbone_is_completion_auth_only(scratch, shared_mount):
    fixture = _workspace(scratch)
    model = fixture["model"]
    (model / "config.json").write_text(json.dumps({
        "num_hidden_layers": 3, "text_config": {"num_hidden_layers": 2}}))
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    tensor = "model.language_model.layers.2.mlp.down_proj.weight"
    header = json.dumps({tensor: {"dtype": "BF16", "shape": [1],
                                  "data_offsets": [0, 2]}}).encode()
    shard = model / "mtp-passthrough.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    index["weight_map"][tensor] = shard.name
    index_path.write_text(json.dumps(index))

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)
    assert str(shard) in _paths(manifest, "source-complete")
    assert str(shard) not in _paths(manifest, "layer-0")
    assert str(shard) not in _paths(manifest, "layer-1")


def test_prepare_manifest_uses_exact_cached_source_sha_and_refuses_mutation(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    source = fixture["shard"]
    stat = source.stat()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    capture_manifest = fixture["captures"] / "capture_manifest.json"
    capture = json.loads(capture_manifest.read_text())
    capture["identity"] = {"source_files": {source.name: digest}}
    capture_manifest.write_text(json.dumps(capture))
    cache = scratch / "source-identity.json"
    cache.write_text(json.dumps({
        "schema": "prismaquant.streamed_model.identity_cache.v1",
        "source": str(fixture["model"]),
        "identity": {"shards": [{"path": str(source), "sha256": digest}]},
        "fingerprints": [{"path": str(source), "device": stat.st_dev,
                          "inode": stat.st_ino, "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns,
                          "ctime_ns": stat.st_ctime_ns}],
    }))
    plan = json.loads(fixture["plan"].read_text())
    plan["source_identity_cache"] = {"path": str(cache),
                                      "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    fixture["plan"].write_text(json.dumps(plan))
    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)
    assert manifest["annotations"]["source_authentication_mode"] == (
        "verified_streamed_identity_cache")
    assert manifest["annotations"]["source_identity_cache_host"] == socket.gethostname()
    assert manifest["annotations"]["counts"]["source_identity_cache"] == 1
    assert manifest["annotations"]["counts"].get("source_authentication", 0) == 0
    assert str(cache) in _paths(manifest, "head")

    before = source.stat()
    with source.open("r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 1]))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(SystemExit, match="no longer proves"):
        glm_data_manifests.build_joint_pass_manifest(
            str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)


def test_a_rung_the_campaign_never_measured_contributes_no_wire(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    declared = {entry["path"] for entry in manifest["entries"]}
    # The blob is really there, and the pass really does not open it.
    for name in fixture["names"]:
        unmeasured = fixture["wire_dir"] / _wire_filename(name, UNMEASURED)
        assert unmeasured.is_file()
        assert str(unmeasured) not in declared
        for fmt in MEASURED:
            assert str(fixture["wire_dir"] / _wire_filename(name, fmt)) in declared
    assert manifest["annotations"]["counts"]["wires"] == len(fixture["names"]) * len(MEASURED)


def test_the_wire_of_a_render_the_campaign_never_wrote_is_read_in_the_head(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    name, fmt = fixture["names"][0], MEASURED[0]
    render = (fixture["workspace"] / "rows" / "row-0000" / "cache"
              / glm_data_manifests._cache_weight_filename(name, fmt))
    assert render.is_file()
    # A rung this campaign adopted rather than encoded: wire, no shard.
    render.unlink()
    wire = fixture["wire_dir"] / _wire_filename(name, fmt)

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    # ``_resolve_render_origin`` decodes the missing shard from this wire
    # before any layer installs, so the warm has to reach it in the head. A
    # declaration in ``layer-0`` would leave the pass to read it cold.
    assert str(wire) in _paths(manifest, "head")
    assert str(wire) not in _paths(manifest, "layer-0")
    assert str(render) not in {entry["path"] for entry in manifest["entries"]}

    annotations = manifest["annotations"]
    assert annotations["renders_absent"] == 1
    assert annotations["renders_absent_first"] == str(render)
    assert annotations["synthesized_render_wire_bytes"] == wire.stat().st_size
    # The layer verifies that wire again. The contract carries a byte range
    # once, so the second read is recorded rather than declared twice.
    assert annotations["reread_bytes_by_phase"]["layer-0"] >= wire.stat().st_size


def test_the_run_manifest_hashes_every_cell_before_the_first_layer(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    prepared = scratch / "prepared.json"
    cache = scratch / "production.pkl"
    cache.write_bytes(b"p" * 5000)
    prepared.write_text(json.dumps({
        "schema": glm_data_manifests.JOINT_PREPARED_SCHEMA,
        "status": "complete",
        "production_cache": {"path": str(cache), "sha256": None},
    }))

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="run", produced_by=PRODUCED_BY,
        prepared=str(prepared))

    names = [phase["name"] for phase in manifest["annotations"]["phases"]]
    assert names == ["head", "hash", "layer-0", "layer-1"]
    assert str(prepared) in _paths(manifest, "head")
    assert str(cache) in _paths(manifest, "head")

    # ``verify_files`` hashes the wire and then the render of each cell, over
    # cells in sorted unit then sorted format order.
    expected = []
    for name in sorted(fixture["names"]):
        for fmt in sorted(MEASURED):
            expected.append(str(fixture["wire_dir"] / _wire_filename(name, fmt)))
            layer = int(name[len(PREFIX)].split(".")[0])
            expected.append(str(
                fixture["workspace"] / "rows" / f"row-{layer:04d}" / "cache"
                / glm_data_manifests._cache_weight_filename(name, fmt)))
    assert _paths(manifest, "hash") == expected

    # The streaming pass reads those renders again. A repeated (path, offset)
    # is refused by the contract, so the re-read is accounted for instead.
    assert _paths(manifest, "layer-0") == [str(fixture["shard"])]
    reread = manifest["annotations"]["reread_bytes_by_phase"]
    assert set(reread) == {"layer-0", "layer-1"}
    assert all(value > 0 for value in reread.values())


def test_a_joint_pass_before_the_merge_is_refused_with_the_path_named(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    for shard in (fixture["parts"] / "units").iterdir():
        shard.unlink()

    with pytest.raises(SystemExit) as refusal:
        glm_data_manifests.build_joint_pass_manifest(
            str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    message = str(refusal.value)
    assert str(fixture["parts"]) in message
    assert "merge" in message


def test_every_post_campaign_manifest_satisfies_the_prismabuild_contract(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=PRODUCED_BY)

    # ``check_manifest`` already ran inside the builder; this states what it
    # accepted, because the validator that matters lives in PrismaBuild.
    assert set(manifest) == set(glm_data_manifests.MANIFEST_KEYS)
    assert manifest["schema"] == glm_data_manifests.SCHEMA
    assert manifest["annotations"]["sha256_present"] is False
    assert manifest["annotations"]["sha256_absent_reason"]
    seen = set()
    for entry in manifest["entries"]:
        assert set(entry) == set(glm_data_manifests.ENTRY_KEYS)
        assert entry["sha256"] is None
        assert entry["bytes"] > 0
        assert entry["path"].startswith(manifest["mount_prefix"] + "/")
        assert (entry["path"], entry["offset"]) not in seen
        seen.add((entry["path"], entry["offset"]))
    assert manifest["entry_count"] == len(manifest["entries"])
    glm_data_manifests.check_manifest(manifest, where="joint")
    # And it is JSON, because it is written to a file another program reads.
    assert json.loads(json.dumps(manifest)) == manifest


def test_the_allocation_manifest_is_the_handoff_files_then_the_head(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    prepared = scratch / "prepared.json"
    cache = scratch / "production.pkl"
    cache.write_bytes(b"p" * 5000)
    prepared.write_text(json.dumps({
        "schema": glm_data_manifests.JOINT_PREPARED_SCHEMA,
        "status": "complete",
        "production_cache": {"path": str(cache), "sha256": None},
    }))
    joint_cost = scratch / "joint-cost.pkl"
    joint_cost.write_bytes(pickle.dumps(
        {"provenance": {"tessera_joint_anchors": {
            "prepared": {"path": str(prepared), "sha256": None}}}},
        protocol=pickle.HIGHEST_PROTOCOL))

    manifest = glm_data_manifests.build_allocation_manifest(
        str(joint_cost), str(fixture["plan"]), produced_by=PRODUCED_BY)

    assert [phase["name"] for phase in manifest["annotations"]["phases"]] == [
        "handoff", "head"]
    handoff = _paths(manifest, "handoff")
    assert handoff[:4] == [str(joint_cost), str(fixture["plan"]), str(prepared),
                           str(cache)]
    # The handoff joins recorded identities; it opens no capture, render, wire
    # or source byte.
    for kind in ("captures", "renders", "wires", "source_extents"):
        assert kind not in manifest["annotations"]["counts"]


def test_the_export_manifest_declares_selected_wires_and_passthrough_source(
    scratch, shared_mount,
):
    fixture = _workspace(scratch)
    allocation = scratch / "shared-allocation-cost.pkl"
    allocation.write_bytes(b"a" * 1024)
    selected = {name: MEASURED[0] for name in fixture["names"]}
    # One unit stays on the source precision, so the export reads its weights
    # instead of a wire.
    selected[fixture["names"][-1]] = "BF16"
    assignment = scratch / "layer_config.json"
    assignment.write_text(json.dumps({**selected, "__prismaquant__": {"x": 1}}))

    manifest = glm_data_manifests.build_export_manifest(
        str(fixture["plan"]), assignment=str(assignment),
        allocation_cost=str(allocation), produced_by=PRODUCED_BY)

    assert [phase["name"] for phase in manifest["annotations"]["phases"]] == [
        "head", "layer-0", "layer-1"]
    declared = {entry["path"] for entry in manifest["entries"]}
    assert str(fixture["wire_dir"] / _wire_filename(fixture["names"][0], MEASURED[0])) in declared
    assert str(fixture["wire_dir"] / _wire_filename(fixture["names"][-1], MEASURED[0])) not in declared
    assert str(fixture["shard"]) in declared
    assert manifest["annotations"]["passthrough_units"] == 1
    # The exporter is Tessera's, so the order is declared as the artifact's
    # layer order and says so rather than claiming to be the read order.
    assert manifest["annotations"]["read_order_attested"] is False


def test_the_submit_command_puts_the_manifest_before_the_detach(
    scratch, shared_mount, capsys, monkeypatch,
):
    import dispatch_tessera_campaign as dispatch

    fixture = _workspace(scratch)
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    manifest_dir = scratch / "manifests"
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)

    code = dispatch.main([
        "submit-joint", "prepare",
        "--plan", str(fixture["plan"]),
        "--spec", str(spec),
        "--demand", "gpu=1,mem_gb=104",
        "--cpus", "6",
        "--tag", "gb10",
        "--priority", "-10",
        "--timeout-s", "86400",
        "--manifest-dir", str(manifest_dir),
        "--dry-run",
    ])

    assert code == 0
    printed = capsys.readouterr().out
    command = next(line for line in printed.splitlines()
                   if line.startswith("[dry-run] "))[len("[dry-run] "):]
    argv = command.split()
    # ``--data-manifest`` is an option of pbrun. After the separator it would
    # be an argument of the action, and the fleet would warm nothing.
    assert argv.index("--data-manifest") < argv.index("--detach")
    assert argv.index("--detach") < argv.index("--")
    # The demand mirrors the chain's own PBRUN_DEMAND for the joint pass.
    for expected in ("--demand", "gpu=1,mem_gb=104", "--cpus", "6", "--tag",
                     "gb10", "--priority", "-10", "--timeout-s", "86400"):
        assert expected in argv
    assert argv[argv.index("--data-manifest") + 1].endswith(
        "prismaquant.tessera_joint_aura.prepare.json.gz")
    assert "tools.tessera_campaign_container" in argv
    assert "prismaquant.tessera_joint_aura" in argv
    # A dry run submits nothing and writes nothing.
    assert not manifest_dir.exists()
    assert "--detach" not in "".join(
        line for line in printed.splitlines() if line.startswith("[submit] "))


def test_cached_source_proof_refuses_a_broad_gpu_tag_before_submission(
    scratch, shared_mount, monkeypatch,
):
    import dispatch_tessera_campaign as dispatch

    fixture = _workspace(scratch)
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    original = glm_data_manifests.build_joint_pass_manifest

    def cached(*args, **kwargs):
        manifest = original(*args, **kwargs)
        manifest["annotations"]["source_identity_cache_host"] = socket.gethostname()
        return manifest

    monkeypatch.setattr(glm_data_manifests, "build_joint_pass_manifest", cached)
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    with pytest.raises(RuntimeError, match="proof is local"):
        dispatch.main([
            "submit-joint", "prepare", "--plan", str(fixture["plan"]),
            "--spec", str(spec), "--demand", "gpu=1,mem_gb=104",
            "--cpus", "6", "--tag", "gb10", "--dry-run",
        ])


def test_joint_submit_writes_one_deterministic_gzip_manifest(
    scratch, shared_mount, monkeypatch,
):
    import dispatch_tessera_campaign as dispatch

    fixture = _workspace(scratch)
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    directory = scratch / "manifests"
    submitted = []
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return real_run(argv, **kwargs)
        submitted.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    args = ["submit-joint", "prepare", "--plan", str(fixture["plan"]),
            "--spec", str(spec), "--demand", "gpu=1,mem_gb=104",
            "--cpus", "6", "--tag", "gb10", "--manifest-dir", str(directory)]
    assert dispatch.main(args) == 0
    path = directory / "prismaquant.tessera_joint_aura.prepare.json.gz"
    first = path.read_bytes()
    payload = json.loads(gzip.decompress(first))
    glm_data_manifests.check_manifest(payload)
    assert submitted[0][submitted[0].index("--data-manifest") + 1] == str(path)
    assert dispatch.main(args) == 0
    assert path.read_bytes() == first


def test_a_dry_run_reports_the_phase_boundaries_it_would_submit(
    scratch, shared_mount, capsys, monkeypatch,
):
    import dispatch_tessera_campaign as dispatch

    fixture = _workspace(scratch)
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)

    dispatch.main([
        "submit-joint", "prepare", "--plan", str(fixture["plan"]),
        "--spec", str(spec), "--demand", "gpu=1,mem_gb=104", "--cpus", "6",
        "--tag", "gb10", "--timeout-s", "86400",
        "--manifest-dir", str(scratch / "manifests"), "--dry-run",
    ])

    printed = capsys.readouterr().out
    # The command line is printed first and contains the spec's own JSON, so
    # the summary starts at the first line that is a bare opening brace.
    lines = printed.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "{")
    summary = json.loads("\n".join(lines[start:]))
    assert summary["entry_point"] == "prismaquant.tessera_joint_aura:prepare"
    assert [phase["name"] for phase in summary["phases"]] == [
        "head", "layer-0", "layer-1"]
    assert summary["total_bytes"] == summary["phases"][-1]["cumulative_bytes"]
    assert summary["manifest_bytes"] <= glm_data_manifests.MAX_MANIFEST_BYTES
    assert summary["manifest_bytes"] < summary["decoded_manifest_bytes"]


def test_the_manifest_of_a_pass_is_byte_identical_across_two_builds(
    scratch, shared_mount,
):
    """``pbrun`` seals the manifest's digest into the action key.

    A field that drifted between two submissions of the same pass would give
    the finished pass a new key and re-run it, which is the opposite of what
    re-running submit is for.
    """
    fixture = _workspace(scratch)
    provenance = glm_data_manifests.deterministic_entry_provenance(
        "prismaquant.tessera_joint_aura:prepare", plan=str(fixture["plan"]),
        plan_sha256="0" * 64, workspace=str(fixture["workspace"]))

    first = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=provenance)
    second = glm_data_manifests.build_joint_pass_manifest(
        str(fixture["plan"]), command="prepare", produced_by=provenance)

    assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)
    assert "unix" not in json.dumps(first["produced_by"])
    assert os.uname().nodename not in json.dumps(first["produced_by"])


def test_a_read_set_over_the_prismabuild_ceiling_is_refused_at_submit(
    scratch, shared_mount, capsys, monkeypatch,
):
    """The real joint pass is over the ceiling, so the refusal has to bite.

    PrismaBuild's ``load_data_manifest`` stats the file before it parses it and
    refuses anything over ``DATA_MANIFEST_MAX_BYTES``. The joint pass's own
    read set is 104 MB of compact JSON against a 64 MiB ceiling, so this is not
    a theoretical path: the producer has to refuse it here, where the number is
    readable, instead of submitting a truncated read set that would warm the
    wrong bytes.
    """
    import dispatch_tessera_campaign as dispatch

    fixture = _workspace(scratch)
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    monkeypatch.setattr(glm_data_manifests, "MAX_MANIFEST_BYTES", 128)

    with pytest.raises(SystemExit) as refusal:
        dispatch.main([
            "submit-joint", "prepare", "--plan", str(fixture["plan"]),
            "--spec", str(spec), "--demand", "gpu=1,mem_gb=104", "--cpus", "6",
            "--tag", "gb10", "--timeout-s", "86400",
            "--manifest-dir", str(scratch / "manifests"), "--dry-run",
        ])

    assert "over the 128-byte limit" in str(refusal.value)
    # The summary is printed first, so the refusal is readable rather than bare.
    printed = capsys.readouterr().out
    assert '"phases"' in printed
    assert not (scratch / "manifests").exists()


#: The frozen census the measured numbers in the issue and in
#: ``docs/ARCHITECTURE.md`` came from. Absent on a fleet CPU box that has no
#: campaign mount, and absent before the campaign merge publishes the
#: checkpoint, so the smoke names its reason rather than failing.
BASE = Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908")
REAL_PLAN = BASE / "first-proof-joint-preparation-03" / "plan.inputs-resolved.json"
REAL_PLAN_TEMPLATE = BASE / "first-proof-joint-preparation-03" / "dryrun" / "plan.template.json"
#: The joint pass's whole read set, measured from the dl380g10 local pool at
#: 07:57Z on 2026-09-13: 5.20 TB over 371,734 entries. The band is wide on
#: purpose, because the tree is being written while it is read. A joint
#: ``prepare`` has been running against this plan since 04:59Z, and it writes
#: a decoded shard for every rung the campaign adopted rather than encoded --
#: each one marked by a ``.render_origin.json`` record beside it in the row
#: cache. Those records are written from one place only -- the head's
#: ``_resolve_render_origin`` -- and the running pass was still writing them
#: at 08:12Z, three hours after it started, so the head placement below is
#: measured rather than inferred. The read set gained 3,216 renders in
#: eighteen minutes (5.14 TB at
#: 07:39Z, 5.20 TB at 07:57Z), and the brief's 4.75 TB estimate was taken
#: earlier the same morning against fewer shards. Projected settled size once
#: the remaining 97,302 shards exist: 5.20 TB + 97,302 x the 16.8 MB mean
#: render = about 6.8 TB, which is where the upper bound below comes from.
#: What the test pins is the shape -- one head phase and one phase per layer,
#: a read set in the terabytes, dominated by captures, renders and wires --
#: not a byte count of a tree that is still being written.
TOTAL_BYTES_BAND = (4.5e12, 7.5e12)
EXPECTED_PHASES = 46


def _real_plan() -> Path | None:
    for candidate in (REAL_PLAN, REAL_PLAN_TEMPLATE):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def test_the_real_joint_pass_read_set_is_terabytes_in_46_phases(scratch):
    plan = _real_plan()
    if plan is None:
        pytest.skip(f"the frozen joint plan is absent: {REAL_PLAN}")
    payload = json.loads(plan.read_text())
    checkpoint = Path(payload["inputs"]["merged_checkpoint"]["path"])
    parts = checkpoint.with_name(checkpoint.name + ".parts") / "units"
    # The merge publishes the shards first and the manifest that names them
    # last, so both halves are checked: a joint pass is not submittable until
    # the manifest is there.
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        pytest.skip(
            "the campaign merge has not published the merged checkpoint yet: "
            f"{checkpoint}")
    if not parts.is_dir():
        pytest.skip(
            "the campaign merge has not published the checkpoint unit shards "
            f"yet: {parts}")
    source_cache = Path(payload["output_root"]) / "prepare/source-identity.json"
    if source_cache.is_file():
        cached = json.loads(source_cache.read_text())
        first = cached["fingerprints"][0]
        if first["device"] != Path(first["path"]).stat().st_dev:
            pytest.skip("the full-source SHA cache is local to another mount "
                        "device; this host cannot submit its reuse request")

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(plan), command="prepare",
        produced_by={"tool": "test", "commit": "0" * 40})

    phases = manifest["annotations"]["phases"]
    assert len(phases) == EXPECTED_PHASES, [phase["name"] for phase in phases]
    assert phases[0]["name"] == "head"
    assert [phase["name"] for phase in phases[1:]] == [
        f"layer-{index}" for index in range(len(phases) - 1)]
    low, high = TOTAL_BYTES_BAND
    assert low <= manifest["total_bytes"] <= high, (
        f"{manifest['total_bytes']} bytes is outside the measured band "
        f"{low:.3g}-{high:.3g}")
    annotations = manifest["annotations"]
    kinds = annotations["bytes"]
    assert kinds["captures"] > 2e12 and kinds["renders"] > 1e12
    assert kinds["wires"] > 7e11 and kinds["source_extents"] > 5e11

    # Rungs this campaign adopted have a wire and no decoded shard. The head
    # decodes one per cell, so every such wire is declared in the head phase
    # and its shard is declared nowhere.
    assert annotations["renders_absent"] == (
        annotations["measured_cells"] - annotations["counts"]["renders"])
    assert phases[0]["bytes"] >= annotations["synthesized_render_wire_bytes"]

    # The full read set exceeds the 64 MiB plain-file ceiling, but current PB
    # accepts a gzip member up to 64 MiB stored / 512 MiB expanded. The submit
    # path seals that member rather than truncating the read set.
    encoded = json.dumps(manifest, separators=(",", ":")).encode() + b"\n"
    assert len(encoded) > glm_data_manifests.MAX_MANIFEST_BYTES, (
        "the joint pass's read set now fits the plain manifest ceiling")
    assert len(encoded) <= 512 * 1024 * 1024
    assert len(gzip.compress(encoded, mtime=0)) <= glm_data_manifests.MAX_MANIFEST_BYTES

    # Written under the test's own scratch directory, never into the frozen
    # campaign tree.
    (scratch / "joint.prepare.json").write_bytes(encoded)
