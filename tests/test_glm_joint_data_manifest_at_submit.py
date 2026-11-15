"""Build joint campaign fixtures for kept runtime tests."""

from prismaquant.render_cache_paths import cache_weight_filename
from prismaquant import tessera_campaign_table as campaign_table

import hashlib
import importlib.util
import json
import pickle
import shutil
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


#: The sealed replay frontier's own definitions, loaded the way the producer
#: loads them: no ``prismaquant`` package, no GPU image.
_replay_spec = importlib.util.spec_from_file_location(
    "joint_replay_frontier",
    ROOT / "prismaquant" / "joint_replay_frontier.py")
replay_frontier = importlib.util.module_from_spec(_replay_spec)
_replay_spec.loader.exec_module(replay_frontier)

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
        "schema": 'prismaquant.cost_stage_checkpoint.unit.v1',
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
    census = workspace / "census.json"
    census.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_census.v1",
        "model": str(model),
        # The draw, not just the window count: the campaign scope binds which
        # corpus revision and tokenizer ids the calibration windows came from.
        "text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
        "seed": 0, "layer_stride": 1,
        "nsamples": 512, "seqlen": 512,
        "unit_shapes": {name: [16, 16] for name in names},
        # The scope a joint pass evaluates is its roster and its window set;
        # this fixture's census is the authority for both.
        "anchor_groups": {f"g:{layer}": UNITS[layer] for layer in sorted(UNITS)},
    }))
    campaign_plan = workspace / "plan.json"
    campaign_plan.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_plan.v1",
        "model": str(model),
        "census": str(census),
        "calibration_cache": {"path": str(capture_manifest)},
        "rows": rows,
    }))
    receipts = workspace / "receipts.json"
    receipts.write_text(json.dumps({"rows": len(rows)}))

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
                filename = cache_weight_filename(name, fmt)
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
        "schema": 'prismaquant.cost_stage_checkpoint.manifest.v1',
        "stage": "Tessera campaign",
        "units": checkpoint_units,
    }))

    calibration = scratch / "calibration_tokens.safetensors"
    calibration.write_bytes(b"t" * 4096)
    compatibility = scratch / "capture-compatibility.json"
    compatibility.write_text(json.dumps({"schema": "x"}))
    backend = scratch / "projection.so"
    backend.write_bytes(b"e" * 2048)

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    output_root = scratch / "joint-01"
    plan = scratch / "plan.inputs-resolved.json"
    plan.write_text(json.dumps({
        "schema": 'prismaquant.tessera_joint_aura.plan.v1',
        "model": str(model),
        "output_root": str(output_root),
        "calibration_input": {"path": str(calibration), "sha256": digest(calibration)},
        "canonical_capture": {"path": str(capture_manifest),
                              "sha256": digest(capture_manifest)},
        "source_capture_compatibility": {"path": str(compatibility),
                                         "sha256": digest(compatibility)},
        "execution": {"calib_seqlen": 512, "n_calib_samples": 512,
                      "projection_backend": {
                          "binary": {"path": str(backend), "sha256": digest(backend)},
                          "name": "f"}},
        "qualification_window": {"max_load_buffer_bytes": 1},
        "source_prefetch": {"max_cache_slots": 2, "prefetch_lookahead": 1},
        "inputs": {
            "campaign_plan": {"path": str(campaign_plan),
                              "sha256": digest(campaign_plan)},
            "census": {"path": str(census), "sha256": digest(census)},
            "campaign_receipts": {"path": str(receipts),
                                  "sha256": digest(receipts)},
            "merged_cost": {"path": str(cost), "sha256": digest(cost)},
            "merged_checkpoint": {"path": str(checkpoint),
                                  "sha256": digest(checkpoint)},
            "required_source_units": len(names),
            "required_campaign_groups": len(UNITS),
        },
    }))
    # The campaign identity is fixed outside the plan: a joint submission names
    # the campaign it belongs to, and the plan's own bound census cannot show
    # that by itself.
    scope = campaign_table.joint_campaign_scope(json.loads(plan.read_text()))
    campaign_identity = scratch / "campaign-identity.json"
    campaign_identity.write_text(json.dumps(campaign_table.campaign_identity(scope),
                                            sort_keys=True))
    return {"plan": plan, "workspace": workspace, "wire_dir": wire_dir,
            "checkpoint": checkpoint, "parts": parts, "names": names,
            "shard": shard, "captures": captures, "model": model,
            "output_root": output_root, "scope": scope,
            "campaign_identity": campaign_identity}


def _scope_args(fixture, require_scope="complete_campaign"):
    """The two arguments every joint submission now states explicitly."""
    return ["--require-scope", require_scope,
            "--campaign-identity", str(fixture["campaign_identity"])]


@pytest.fixture()
def scratch(request):
    """A per-test directory that is never under ``/tmp``.

    ``pytest``'s own ``tmp_path`` lands wherever ``TMPDIR`` points, and
    the sharded runner refuses a ``--basetemp`` override, so the one place a fleet
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


def canonical_sha256(value) -> str:
    """``cost_stage_checkpoint.canonical_json_sha256``, restated."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workspace_with_sealed_checkpoint(scratch):
    """The campaign fixture, with the merged checkpoint's own seal.

    The journal identity names the campaign checkpoint it was qualified
    against, and the checkpoint names its own seal, so a journal that was
    written against another merge is recognisable as one.

    Sealing the checkpoint changes its bytes, so the plan is re-bound to the
    sealed file and the campaign identity re-derived from that plan: the plan
    has to bind the artifact it reads, and a submission that carried the
    pre-seal digest would declare a roster the pass does not hold.
    """

    fixture = _workspace(scratch)
    checkpoint = Path(fixture["checkpoint"])
    document = json.loads(checkpoint.read_text())
    identity = {
        "campaign_schema": "prismaquant.tessera_joint_cost.v1",
        "currency": "fixture",
        "units": sorted(fixture["names"]),
        "prismaquant_source_sha256": "a" * 64,
        "encoder_source_sha256": "b" * 64,
    }
    document["identity"] = identity
    document["identity_sha256"] = canonical_sha256(identity)
    checkpoint.write_text(json.dumps(document))
    plan_path = Path(fixture["plan"])
    plan = json.loads(plan_path.read_text())
    plan["inputs"]["merged_checkpoint"]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    plan_path.write_text(json.dumps(plan))
    fixture["scope"] = campaign_table.joint_campaign_scope(
        json.loads(plan_path.read_text()))
    Path(fixture["campaign_identity"]).write_text(json.dumps(
        campaign_table.campaign_identity(fixture["scope"]), sort_keys=True))
    return fixture, document["identity_sha256"]


def plant_journal(fixture, completed, *, checkpoint_sha256, roster=None):
    """A qualification journal as ``cost_stage_checkpoint`` writes one."""
    journal = Path(fixture["output_root"]) / "prepare" / "qualification"
    (journal / "units").mkdir(parents=True, exist_ok=True)
    names = sorted(fixture["names"] if roster is None else roster)
    identity = {
        "schema": "prismaquant.joint_qualification_journal.v1",
        "inputs": {"fixture": "inputs"},
        "campaign_checkpoint_sha256": checkpoint_sha256,
        "cells_sha256": "c" * 64,
    }
    identity_sha256 = canonical_sha256(identity)
    units = []
    for name in names:
        relative = f"units/{hashlib.sha256(name.encode()).hexdigest()}.pkl"
        units.append({"qname": name, "file": relative})
        if name not in completed:
            continue
        payload = pickle.dumps({"fixture": name}, protocol=pickle.HIGHEST_PROTOCOL)
        envelope = {
            "schema": replay_frontier.JOURNAL_UNIT_SCHEMA,
            "stage": replay_frontier.QUALIFICATION_STAGE,
            "qname": name,
            "identity_sha256": identity_sha256,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload": payload,
        }
        (journal / relative).write_bytes(
            pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    (journal / "manifest.json").write_text(json.dumps({
        "schema": replay_frontier.JOURNAL_MANIFEST_SCHEMA,
        "stage": replay_frontier.QUALIFICATION_STAGE,
        "identity_sha256": identity_sha256,
        "identity": identity,
        "units": units,
    }))
    return identity_sha256


def _phase(manifest, name):
    return next(phase for phase in manifest["annotations"]["phases"]
                if phase["name"] == name)


def _prepared_fixture(scratch):
    prepared = scratch / "prepared.json"
    cache = scratch / "production.pkl"
    cache.write_bytes(b"p" * 5000)
    prepared.write_text(json.dumps({
        "schema": 'prismaquant.tessera_joint_aura.prepared.v3',
        "status": "complete",
        "production_cache": {"path": str(cache), "sha256": None},
    }))
    return prepared, cache
