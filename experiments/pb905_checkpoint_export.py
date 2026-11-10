"""Retained offline checkpoint byte controls for the retired PB905 experiment.

Fleet pacing and execution entrypoints refuse. The original exact tensor reader
and finite packet checks remain available as CPU scientific/IO controls.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "prismaquant.pb905.checkpoint_export_packet.v1"
RESULT_SCHEMA = "prismaquant.pb905.checkpoint_export_component.v1"
KIND = "pb905-checkpoint-export"
GROUPS = 12
GROUP_SIZE = 64
MAX_CONTROL_BYTES = 4 << 20


def sha(raw):
    from prismaquant.digests import bytes_sha256hex
    return bytes_sha256hex(raw)


def json_bytes(value):
    from prismaquant.digests import indent2_json_file_bytes
    return indent2_json_file_bytes(value)


def _new_json(path, value):
    from prismaquant.cost_stage_checkpoint import publish_new_bytes
    if not publish_new_bytes(path, json_bytes(value), nofollow=True):
        raise FileExistsError(f"PB905 evidence already exists: {path}")


def check_packet(packet):
    """Check the historical component's finite offline roster."""
    required = {"schema", "scope", "checkpoint", "checkpoint_record", "groups",
                "output_prefix", "tier", "group_bytes", "total_bytes",
                "storage", "spool_max_bytes", "source_proposal_sha256"}
    if not isinstance(packet, dict) or set(packet) != required or packet["schema"] != SCHEMA:
        raise ValueError("PB905 packet schema/fields differ")
    if packet["scope"] != "offline-checkpoint-export-component":
        raise ValueError("PB905 scope differs")
    groups = packet["groups"]
    if not isinstance(groups, list) or len(groups) != GROUPS:
        raise ValueError("PB905 requires twelve frozen groups")
    record = packet["checkpoint_record"]
    from prismaquant.joint_adjoint_checkpoints import checkpoint_manifest_bytes
    checkpoint_raw = checkpoint_manifest_bytes(record)
    if (sha(checkpoint_raw) != packet["checkpoint"]["sha256"]
            or type(packet["checkpoint"]["bytes"]) is not int
            or len(checkpoint_raw) != packet["checkpoint"]["bytes"]):
        raise ValueError("PB905 checkpoint record differs from its pinned bytes")
    anchor = {row["path"]: row for row in record["activation_entries"]}
    frozen = sorted(record["activation_entries"], key=lambda row: row["name"])[:GROUPS * GROUP_SIZE]
    if [entry for group in groups for entry in group["entries"]] != frozen:
        raise ValueError("PB905 selection differs from the fixed checkpoint name order")
    seen = set()
    expected_bytes = packet["group_bytes"]
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("PB905 group bytes must be a positive integer")
    for index, group in enumerate(groups):
        if (set(group) != {"index", "paced", "entries"}
                or type(group["index"]) is not int or group["index"] != index):
            raise ValueError("PB905 group order differs")
        if type(group["paced"]) is not bool or group["paced"] != (index % 2 == 0):
            raise ValueError("PB905 paced/unpaced pair order differs")
        entries = group["entries"]
        if not isinstance(entries, list) or len(entries) != GROUP_SIZE:
            raise ValueError("PB905 requires 64 fixed entries per group")
        total = 0
        names = set()
        for entry in entries:
            path = entry.get("path")
            if path in seen or anchor.get(path) != entry:
                raise ValueError("PB905 repeats or changes a checkpoint entry")
            seen.add(path)
            name = Path(path).name
            if name in names or name.endswith(".tmp"):
                raise ValueError("PB905 repeats or changes a destination name")
            names.add(name)
            size = entry["file_bytes"]
            if type(size) is not int or size <= 0:
                raise ValueError("PB905 entry byte bound differs")
            total += size
        if total != expected_bytes:
            raise ValueError("PB905 groups have unequal byte counts")
    if packet["total_bytes"] != GROUPS * expected_bytes:
        raise ValueError("PB905 total byte bound differs")
    from prismaquant.streamed_boundary_artifacts import check_boundary_storage
    storage = check_boundary_storage(packet["storage"])
    if storage["max_artifact_bytes"] != packet["total_bytes"]:
        raise ValueError("PB905 artifact bound differs")
    largest = max(entry["file_bytes"] for group in groups for entry in group["entries"])
    largest_tensor = max(entry["tensor_bytes"] for group in groups for entry in group["entries"])
    if storage["max_resident_bytes"] < expected_bytes + 2 * largest + largest_tensor:
        raise ValueError("PB905 input residency cannot fit one group plus reader scratch")
    if type(packet["spool_max_bytes"]) is not int or packet["spool_max_bytes"] < 2 * expected_bytes:
        raise ValueError("PB905 spool bound cannot fit two groups")
    declared_prefix = Path(packet["output_prefix"])
    prefix = declared_prefix.resolve()
    if not declared_prefix.is_absolute() or Path(storage["directory"]).resolve() != prefix / "exact":
        raise ValueError("PB905 writer directory differs from its output prefix")
    for path in seen | {packet["checkpoint"]["path"]}:
        source = Path(path).resolve()
        if source.is_relative_to(prefix) or prefix.is_relative_to(source.parent):
            raise ValueError("PB905 output prefix overlaps an original input")
    return packet


def selected_groups(packet, group_index=None):
    if group_index is None:
        return packet["groups"]
    if type(group_index) is not int or not 0 <= group_index < GROUPS:
        raise ValueError("PB905 authentication cohort is outside the frozen roster")
    return [packet["groups"][group_index]]


def data_manifest(packet, packet_path, packet_raw, *, group_index=None):
    from prismaquant.staged_launch import read_contract_document
    entries = [{"path": str(packet_path), "offset": 0, "bytes": len(packet_raw),
                "sha256": sha(packet_raw)},
               {"path": packet["checkpoint"]["path"], "offset": 0,
                "bytes": packet["checkpoint"]["bytes"],
                "sha256": packet["checkpoint"]["sha256"]}]
    for group in selected_groups(packet, group_index):
        entries.extend({"path": row["path"], "offset": 0, "bytes": row["file_bytes"],
                        "sha256": row["sha256"]} for row in group["entries"])
    return read_contract_document(json_bytes({
        "schema": "prismaquant.read_contract.v1", "entries": entries}))


def prepare(*args, **kwargs):
    raise ValueError("PB905 fleet pacing experiment is retired; native publication has no pacing arms")


def verified_bytes(entry, *, expected_session, scratch, owner):
    """Copy the serial scratch only inside its existing verified exact-entry window."""
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    from prismaquant.perturbed_x_cache import exact_read_threads, prefetch_exact_activation_cache_entries
    if exact_read_threads() != 1:
        raise ValueError("PB905 original-byte custody requires one serial exact-entry reader")
    ref = reference_from_record(entry)
    with prefetch_exact_activation_cache_entries(
            [ref], max_tensor_bytes=ref.tensor_bytes, expected_session=expected_session,
            scratch=scratch, residency_check=owner.reserve_resident) as window:
        window.get(ref)  # the existing owner has verified body, archive, tensor and metadata
        view = memoryview(scratch.buffer(ref.file_bytes))[:ref.file_bytes]
        try:
            return bytes(view)
        finally:
            view.release()


@contextmanager
def verified_group(group, *, expected_session, scratch, owner, keep_bytes, resource_check=None):
    """One bounded cohort; reader leases release before writer-side export waits."""
    largest = max(row["file_bytes"] for row in group["entries"])
    charge = (sum(row["file_bytes"] for row in group["entries"]) if keep_bytes else largest) + 2 * largest
    if charge > owner.config["max_resident_bytes"]:
        raise RuntimeError("PB905 serialized input exceeds its residency budget")
    from prismaquant.memory_management import reserve_allocation
    reserve_allocation(resource_check, "pb905-serialized-input", cpu_bytes=charge)
    files, observations = [], []
    try:
        from prismaquant.perturbed_x_cache import file_stat_signature
        for entry in group["entries"]:
            path = Path(entry["path"])
            before = file_stat_signature(path.lstat())
            raw = verified_bytes(entry, expected_session=expected_session, scratch=scratch, owner=owner)
            if before != file_stat_signature(path.lstat()):
                raise ValueError("PB905 original changed around its verified window")
            observations.append({"path": str(path), "sha256": entry["sha256"],
                                 "bytes": len(raw), "file_signature": before})
            if keep_bytes:
                files.append((path.name, raw))
            del raw
        yield files, observations
    finally:
        files.clear()


def load_runtime_packet(*args, **kwargs):
    raise ValueError("PB905 fleet pacing experiment is retired; historical packets cannot authorize native execution")


def run_component(*args, **kwargs):
    raise ValueError("PB905 fleet pacing experiment is retired; native publication has no pacing arms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--proposal", required=True)
    build.add_argument("--proposal-sha256", required=True)
    build.add_argument("--directory", required=True)
    build.add_argument("--output-prefix", required=True)
    build.add_argument("--tier", required=True)
    build.add_argument("--helper-root", required=True)
    for mode in ("authenticate", "export"):
        action = sub.add_parser(mode)
        action.add_argument("--packet", required=True)
        action.add_argument("--packet-sha256", required=True)
        action.add_argument("--packet-bytes", type=int, required=True)
        action.add_argument("--readset-sha256", required=True)
        action.add_argument("--evidence-directory", required=True)
        if mode == "authenticate":
            action.add_argument("--group-index", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        result = prepare(args.proposal, args.proposal_sha256, args.directory,
                         output_prefix=args.output_prefix, tier=args.tier, helper_root=args.helper_root)
    else:
        group_index = getattr(args, "group_index", None)
        packet = load_runtime_packet(args.packet, args.packet_sha256, args.packet_bytes,
                                     args.readset_sha256, group_index=group_index)
        result = run_component(packet, mode=args.mode, evidence_directory=args.evidence_directory,
                               group_index=group_index)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
