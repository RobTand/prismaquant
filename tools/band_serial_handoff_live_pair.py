"""One executed band-serial handoff pair on the live fleet (PQ #1015).

Quantum ``L`` (the producer) writes its handoff through PrismaBuild's
produced-output lifecycle, and quantum ``L - 1`` (the consumer) reads it as
declared, staged inputs. The unit tests drive both halves on a private queue;
this driver runs them as two real admitted actions, so the consumer reads the
handoff from the stage tier that PrismaBuild filled, not from the pool.

The campaign is the tiny one the unit tests use
(``tests/test_quantum_executable_readset.py``: layers 3 and 2 of band 4,
bound to a real Stage A receipt), written once under ``--root`` by
``prepare``. Every call site below is the production one:

* ``prepare`` (coordinator side, like the dispatcher): the producer's
  handoff template from ``dispatch_joint_quanta.handoff_template_path``, and
  a one-phase data manifest of the producer's checkpoint plane, so the
  producer action has a residency map to bind its owner on.
* ``producer`` (inside the producer action): reads its checkpoint plane
  staged, binds ``bind_handoff_publication`` and emits through
  ``HandoffEmitter``. Prints the published handoff.
* ``derive`` (coordinator side): ``dispatch_joint_quanta.bind_consumer_handoff``,
  which runs the consumer's handoff checks and writes the band-serial
  readset the consumer row stages.
* ``consumer`` (inside the consumer action): ``require_band_serial_readset``,
  ``load_quantum_handoff`` and ``load_handoff_inputs`` under the strict tier
  policy, then checks where every handoff byte was served from.

What is synthetic: the consumer's sealed chain-mode readset. The tiny
campaign's own readsets name ``/fixture`` paths that do not exist, so
``prepare`` seals one whose ``head`` phase is a real 4 KiB file and whose
checkpoint and chain phases name real Stage A entries. The band-serial
derivation drops those phases, so the consumer stages ``head`` and
``handoff-load`` only.

``producer`` and ``consumer`` print one JSON object on their last line and
exit nonzero unless every check in it holds.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

PAIR = "pair.json"
PRODUCER_PHASE = "checkpoint-load"
HEAD_BYTES = 4096
ARTIFACT_MAX = 1 << 20
PREFETCH = 2


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _seal_record(record: dict) -> dict:
    from prismaquant.joint_layer_quanta import canonical_sha256

    body = {key: value for key, value in record.items() if key != "identity_sha256"}
    return {**body, "identity_sha256": canonical_sha256(body, where="live pair")}


def _manifest(entries: list[dict], phases: list[tuple[str, list[int]]],
              annotations: dict) -> dict:
    from prismaquant.joint_layer_quanta import MANIFEST_SCHEMA_V2

    read_phases, cumulative = [], 0
    for name, indices in phases:
        size = sum(entries[index]["bytes"] for index in indices)
        cumulative += size
        read_phases.append({"name": name, "entry_indices": indices,
                            "bytes": size, "cumulative_bytes": cumulative})
    return {"schema": MANIFEST_SCHEMA_V2,
            "produced_by": {"tool": "tools/band_serial_handoff_live_pair.py",
                            "entry_point": "prepare"},
            "mount_prefix": "/mnt/shared",
            "entries": entries, "entry_count": len(entries),
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "annotations": annotations,
            "read_plan": {"phases": read_phases, "read_bytes": cumulative}}


def _row(entry: dict) -> dict:
    return {"path": entry["path"], "offset": 0,
            "bytes": int(entry["file_bytes"]), "sha256": entry["sha256"]}


def prepare(root: Path, tier: str) -> dict:
    """Write the tiny campaign, the producer's template and both readsets."""
    import dispatch_joint_quanta as dispatch
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA
    from prismaquant.joint_layer_quanta import (
        CHECKPOINT_LOAD_PHASE, executable_bound_phase_name,
        executable_source_phase_name, seal_manifest_bytes)
    from test_quantum_executable_readset import (
        _bind_slice, _tiny_receipt, _tiny_records)

    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty; prepare writes a fresh campaign")
    root.mkdir(parents=True, exist_ok=True)
    records, _parent = _tiny_records(root)
    receipt = _tiny_receipt(root, records[0]["campaign"])
    bound = {record["layer"]: _bind_slice(record, receipt, root / "adjoint-slices")[0]
             for record in records if record["layer"] in (2, 3)}
    producer, consumer = bound[3], bound[2]
    storage = {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(root / "exact"),
               "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
               "max_artifact_bytes": ARTIFACT_MAX, "prefetch_batches": PREFETCH}

    def slice_of(record):
        return json.loads(Path(record["adjoint"]["slice_path"]).read_text())

    # The dispatcher's default id: derived from the template's body, which
    # names this root's handoff directory, so it is this root's own (PQ #1054).
    # The template names the handoff directory under the output root it is
    # given (PQ #1200), so it is given the root the records were cut under,
    # where the producer writes, as the campaign's dispatch is.
    template = dispatch.handoff_template_path(
        producer, plan={"execution": {"boundary_storage": storage}},
        adjoint_slice=slice_of(producer), tier=tier, output_root=root / "run")

    # The producer's staged input: its checkpoint plane (boundary 4).
    plane = slice_of(producer)["checkpoint"]["activation_entries"]
    producer_wire = seal_manifest_bytes(_manifest(
        [_row(entry) for entry in plane],
        [(PRODUCER_PHASE, list(range(len(plane))))],
        {"live_pair": "producer", "quantum_layer": 3}))
    producer_manifest = root / "producer.data-manifest.json.gz"
    _write_new(producer_manifest, producer_wire)

    # The consumer's sealed chain-mode readset: head, checkpoint load, and
    # layer 3's source and bound. Only head survives the derivation.
    head = root / "head.bin"
    head_raw = hashlib.sha256(b"band-serial live pair head").digest() * (HEAD_BYTES // 32)
    _write_new(head, head_raw)
    checkpoint = slice_of(consumer)["checkpoint"]["activation_entries"]
    boundary3 = receipt["boundary_entries"]["3"]
    entries = [{"path": str(head), "offset": 0, "bytes": len(head_raw),
                "sha256": _sha(head_raw)}]
    entries += [_row(entry) for entry in checkpoint]
    entries += [_row(entry) for entry in boundary3]
    n_ck, n_b3 = len(checkpoint), len(boundary3)
    chain_manifest = _manifest(entries, [
        ("head", [0]),
        (CHECKPOINT_LOAD_PHASE, list(range(1, 1 + n_ck))),
        (executable_source_phase_name(3), list(range(1 + n_ck, 1 + n_ck + n_b3))),
        (executable_bound_phase_name(3), list(range(1 + n_ck, 1 + n_ck + n_b3))),
    ], {"live_pair": "consumer", "quantum_layer": 2, "chain_layers": [3]})
    chain_wire = seal_manifest_bytes(chain_manifest)
    chain_path = root / "consumer-chain.executable.json.gz"
    _write_new(chain_path, chain_wire)
    consumer = _seal_record({**consumer, "executable_readset": {
        "manifest_path": str(chain_path), "manifest_sha256": _sha(chain_wire),
        "phases": [phase["name"] for phase in chain_manifest["read_plan"]["phases"]]}})

    paths = {}
    for name, record in (("producer", producer), ("consumer", consumer)):
        raw = (json.dumps(record, sort_keys=True, indent=1) + "\n").encode()
        paths[name] = root / f"{name}.record.json"
        _write_new(paths[name], raw)
    pair = {"schema": "prismaquant.band_serial_handoff_live_pair.v1",
            "root": str(root), "tier": tier, "storage": storage,
            "producer_record": str(paths["producer"]),
            "consumer_record": str(paths["consumer"]),
            "template": str(template),
            "producer_manifest": str(producer_manifest),
            "producer_manifest_sha256": _sha(producer_wire),
            "producer_phase": PRODUCER_PHASE,
            "head": {"path": str(head), "sha256": _sha(head_raw)},
            "n_probes": len({_probe_batch(e)[0] for e in plane}),
            "n_batches": len({_probe_batch(e)[1] for e in plane})}
    _write_new(root / PAIR, (json.dumps(pair, sort_keys=True, indent=1) + "\n").encode())
    return pair


def _probe_batch(entry):
    from prismaquant.joint_quantum_handoff import _checkpoint_coordinates
    return _checkpoint_coordinates(entry)


def _load(root: Path):
    pair = json.loads((root / PAIR).read_text())
    records = {name: json.loads(Path(pair[f"{name}_record"]).read_text())
               for name in ("producer", "consumer")}
    return pair, records


def _slice(record):
    return json.loads(Path(record["adjoint"]["slice_path"]).read_text())


class _Owner:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)


def _bind_staged(data_manifest_sha256: str) -> None:
    from prismaquant.residency_map import bind_residency_manifest
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    activate_staged_tier_policy("ram,ssd")
    bind_residency_manifest(data_manifest_sha256)


def _serving(paths) -> dict:
    """Where each path was served from, in the resolver's own record."""
    from prismaquant.residency_map import residency_resolver

    resolver = residency_resolver()
    report = resolver.report() if resolver is not None else {}
    rows = report.get("serving_tiers", [])
    served = {}
    for path in paths:
        want = os.path.normpath(str(path))
        served[str(path)] = sorted({str(row.get("serving_tier")) for row in rows
                                    if os.path.normpath(str(row.get("path", ""))) == want})
    counters = {key: report.get(key) for key in (
        "tier_id", "bytes_from_stage", "bytes_from_ram", "bytes_from_pool",
        "fallback_count", "ram_fallback_count", "serving_tier_count",
        "declared_readset")}
    # PQ #1026: the tiers' byte counters must sum to every byte read, so
    # the per-tier split is a measurement, not a partial sample.
    tier_bytes = sum(int(counters.get(key) or 0) for key in (
        "bytes_from_stage", "bytes_from_ram", "bytes_from_pool"))
    read_bytes = sum(Path(path).stat().st_size for path in paths)
    return {"served": served, "resolver": counters,
            "tier_bytes": tier_bytes, "read_bytes": read_bytes}


def producer_role(root: Path, data_manifest_sha256: str) -> int:
    import torch

    from prismaquant.joint_adjoint_checkpoints import (
        checkpoint_entry_session, read_exact_entry_tensors)
    from prismaquant.joint_quantum_handoff import (
        HANDOFF_ORIGIN_LIFETIME, HANDOFF_OWNER_STATES_NAME, HANDOFF_RECORD_BATCH_KIND,
        HANDOFF_RECORD_NAME,
        HandoffEmitter, bind_handoff_publication)
    from prismaquant.prismabuild_progress import report as progress
    from prismaquant.produced_output_spool import MAX_ENV, ROOT_ENV
    from prismaquant.staged_lease import sdk_submodule

    pair, records = _load(root)
    producer = records["producer"]
    adjoint_slice = _slice(producer)
    checkpoint = adjoint_slice["checkpoint"]
    out: dict = {"role": "producer",
                 "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
                 "host": os.uname().nodename}
    checks: dict = {}
    _bind_staged(data_manifest_sha256)

    progress(PRODUCER_PHASE, 0, unit="entries")
    plane = {}
    for entry in checkpoint["activation_entries"]:
        probe, batch = _probe_batch(entry)
        tensor = read_exact_entry_tensors(
            [entry], expected_session=checkpoint_entry_session(checkpoint))[entry["name"]]
        plane[(probe, batch)] = tensor + (10.0 * probe + batch)
    progress(PRODUCER_PHASE, len(plane), unit="entries")
    out["checkpoint_read"] = _serving([e["path"] for e in checkpoint["activation_entries"]])
    checks["checkpoint_plane_staged"] = all(
        tiers and set(tiers) <= {"stage", "ram"}
        for tiers in out["checkpoint_read"]["served"].values())
    checks["checkpoint_counts_complete"] = (
        out["checkpoint_read"]["tier_bytes"] == out["checkpoint_read"]["read_bytes"])

    publication = bind_handoff_publication(boundary_storage=pair["storage"])
    env = getattr(publication, "env", {}) or {}
    out["spool"] = {"root": env.get(ROOT_ENV), "max_bytes": env.get(MAX_ENV)}
    checks["spool_sealed"] = bool(env.get(ROOT_ENV))
    n_probes, n_batches = pair["n_probes"], pair["n_batches"]
    owners = [[_Owner({"scale": float(p + b)}) for b in range(n_batches)]
              for p in range(n_probes)]
    # The plane here is synthetic, but the handoff's contract is the real
    # one: it is captured at the launch regime's batch, which must be the
    # slice's chain batch size (PQ #994, #997).
    from prismaquant.joint_replay_regime import (
        normalize_replay_regime, replay_regime_from_environment)
    capture_batch = normalize_replay_regime(
        replay_regime_from_environment(os.environ))["capture_batch"]
    emitter = HandoffEmitter(record=producer, adjoint_slice=adjoint_slice,
                             boundary_storage=pair["storage"],
                             capture_batch=capture_batch, publication=publication)
    published = emitter.emit(grad_plane=plane, cotangent_owners=owners,
                             n_probes=n_probes, n_batches=n_batches)
    out["published"] = published

    # The record group: its files, and its commit at the origin (PQ #1075).
    po = sdk_submodule("produced_output")
    directory = Path(published["path"]).parent
    batch_id = publication.batch_id_for(
        kind=HANDOFF_RECORD_BATCH_KIND, boundary_index=int(producer["layer"]),
        group_index=0)
    files = [directory / HANDOFF_OWNER_STATES_NAME, directory / HANDOFF_RECORD_NAME]
    planned = sorted(p for f in files for p in (str(f), str(f) + ".tmp"))
    commitments = json.loads((Path(po.instance_dir(
        publication.queue.root, publication.instance)) / "commitments.json"
        ).read_text())["batches"]
    entry = commitments.get(batch_id) or {}
    refs = published.get("origin_batches") or []
    out["record_group"] = {
        "batch_id": batch_id,
        "commitment": entry,
        "file_bytes": {f.name: f.stat().st_size for f in files},
    }
    checks["record_committed_at_origin"] = (
        entry.get("origin_only") is True
        and entry.get("lifetime") == HANDOFF_ORIGIN_LIFETIME
        and sorted(entry.get("paths") or []) == sorted(str(f) for f in files))
    checks["record_prewrite_consumed"] = po._read_prewrite(
        po._prewrites_dir(publication.queue.root, publication.instance)
        / f"{batch_id}.prewrite.json") is None
    checks["record_ref_last"] = bool(refs) and refs[-1]["batch_id"] == batch_id
    checks["every_group_committed"] = all(
        (commitments.get(ref["batch_id"]) or {}).get("origin_only") is True
        and commitments[ref["batch_id"]].get("manifest_digest")
        == ref["manifest_digest"] for ref in refs)
    checks["record_digest"] = (
        _sha(files[1].read_bytes()) == published["sha256"])
    checks["no_temporaries"] = not any(Path(p).exists() for p in planned
                                       if p.endswith(".tmp"))
    out["checks"] = checks
    ok = all(checks.values())
    out["ok"] = ok
    print(json.dumps(out, sort_keys=True, default=str))
    return 0 if ok else 1


def derive(root: Path, handoff: str, handoff_sha256: str) -> dict:
    import dispatch_joint_quanta as dispatch

    pair, records = _load(root)
    bound = dispatch.bind_consumer_handoff(
        records["consumer"], path=handoff, sha256=handoff_sha256,
        producer=records["producer"]["quantum_id"], output_root=root / "out")
    return {**bound, "consumer_record": pair["consumer_record"]}


def consumer_role(root: Path, data_manifest_sha256: str, handoff: str,
                  handoff_sha256: str) -> int:
    import torch

    from prismaquant.calibration_data import _read_calibration_payload
    from prismaquant.joint_quantum_handoff import (
        HANDOFF_LOAD_PHASE, handoff_read_entries, load_handoff_inputs,
        load_quantum_handoff, require_band_serial_readset)
    from prismaquant.prismabuild_progress import report as progress
    from prismaquant.staged_tier_policy import TierPolicyRefused

    pair, records = _load(root)
    consumer = records["consumer"]
    adjoint_slice = _slice(consumer)
    checkpoint = adjoint_slice["checkpoint"]
    out: dict = {"role": "consumer",
                 "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
                 "host": os.uname().nodename}
    checks: dict = {}
    _bind_staged(data_manifest_sha256)

    handoff_doc = load_quantum_handoff(handoff, handoff_sha256, record=consumer,
                                       adjoint_slice=adjoint_slice)
    require_band_serial_readset(consumer, handoff_doc, checkpoint,
                                output_root=root / "out",
                                data_manifest_sha256=data_manifest_sha256)
    checks["readset_is_the_derivation"] = True

    progress("head", 0, unit="entries")
    head = _read_calibration_payload(Path(pair["head"]["path"]),
                                     expected_sha256=pair["head"]["sha256"])
    checks["head_read"] = _sha(head) == pair["head"]["sha256"]
    progress(HANDOFF_LOAD_PHASE, 1, unit="entries")

    # Negative control: an undeclared file of the same handoff generation
    # is refused, so a pass below cannot be a silent pool read.
    undeclared = Path(handoff).parent / "generation.json"
    try:
        _read_calibration_payload(undeclared,
                                  expected_sha256=_sha(undeclared.read_bytes()))
    except TierPolicyRefused as refusal:
        out["undeclared_refusal"] = str(refusal)
        checks["undeclared_refused"] = "readset-not-staged" in str(refusal)
    except Exception as other:  # noqa: BLE001 - recorded, and fails the check
        out["undeclared_refusal"] = f"NOT A REFUSAL: {type(other).__name__}: {other}"
        checks["undeclared_refused"] = False
    else:
        checks["undeclared_refused"] = False

    n_probes, n_batches = pair["n_probes"], pair["n_batches"]
    plane, shared_adjoint, shared_pass = load_handoff_inputs(
        handoff_doc, checkpoint, n_probes=n_probes, n_batches=n_batches)
    rows = handoff_read_entries(handoff_doc, checkpoint)
    progress(HANDOFF_LOAD_PHASE, 1 + len(rows), unit="entries")
    checks["plane_values"] = all(
        torch.equal(plane[(p, b)], torch.full((2, 4), 10.0 * p + b))
        for p in range(n_probes) for b in range(n_batches))
    checks["owner_states"] = all(
        shared_adjoint[(p, b)] == {"scale": float(p + b)}
        for p in range(n_probes) for b in range(n_batches))
    checks["shared_pass"] = all(shared_pass[b] == {"mask": [0, 1]}
                                for b in range(n_batches))
    out["handoff_load"] = _serving([pair["head"]["path"], *[row["path"] for row in rows]])
    served = out["handoff_load"]["served"]
    checks["every_read_staged"] = all(
        tiers and set(tiers) <= {"stage", "ram"} for tiers in served.values())
    checks["owner_states_staged"] = bool(served.get(handoff_doc["owner_states"]["path"]))
    checks["no_pool_bytes"] = not out["handoff_load"]["resolver"].get("bytes_from_pool")
    checks["counts_complete"] = (
        out["handoff_load"]["tier_bytes"] == out["handoff_load"]["read_bytes"])
    out["rows"] = len(rows)
    out["checks"] = checks
    ok = all(checks.values())
    out["ok"] = ok
    print(json.dumps(out, sort_keys=True, default=str))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tier", required=True)
    p = sub.add_parser("producer")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p = sub.add_parser("derive")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--handoff", required=True)
    p.add_argument("--handoff-sha256", required=True)
    p = sub.add_parser("consumer")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--handoff", required=True)
    p.add_argument("--handoff-sha256", required=True)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.tier), sort_keys=True))
        return 0
    if args.command == "derive":
        print(json.dumps(derive(args.root, args.handoff, args.handoff_sha256),
                         sort_keys=True))
        return 0
    if args.command == "producer":
        return producer_role(args.root, args.data_manifest_sha256)
    return consumer_role(args.root, args.data_manifest_sha256, args.handoff,
                         args.handoff_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
