"""Opt-in, CPU-only checkpoint export component for PB #905 (PQ #2111).

This driver freezes inputs and arms, then delegates reads, writes, admission,
export, origin commit and release to their existing owners. It schedules no
worker. A completed component is not the organic reader-contention A/B gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType

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


def _json_control(path, digest):
    from prismaquant.staged_lease import sdk_submodule
    core = sdk_submodule("core")
    raw = core._read_regular_file_nofollow(
        Path(path), where="PB905 packet control", max_bytes=MAX_CONTROL_BYTES,
        replaced_leaf=True)
    if sha(raw) != digest:
        raise ValueError("PB905 control digest differs")
    return core._decode_strict_json(raw, where="PB905 packet control")


def check_packet(packet):
    """Check this component's finite roster; PB still owns every resource gate."""
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
    from prismaquant.cost_streaming import check_boundary_storage
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
    from prismaquant.staged_lease import sdk_submodule
    core = sdk_submodule("core")
    entries = [{"path": str(packet_path), "offset": 0, "bytes": len(packet_raw),
                "sha256": sha(packet_raw)},
               {"path": packet["checkpoint"]["path"], "offset": 0,
                "bytes": packet["checkpoint"]["bytes"],
                "sha256": packet["checkpoint"]["sha256"]}]
    indices = [("input-check", [0, 1])]
    for group in selected_groups(packet, group_index):
        start = len(entries)
        entries.extend({"path": row["path"], "offset": 0, "bytes": row["file_bytes"],
                        "sha256": row["sha256"]} for row in group["entries"])
        indices.append((f"group-{group['index']:02d}", list(range(start, len(entries)))))
    phases, cumulative = [], 0
    for name, members in indices:
        size = sum(entries[index]["bytes"] for index in members)
        cumulative += size
        phases.append({"name": name, "entry_indices": members,
                       "bytes": size, "cumulative_bytes": cumulative})
    return core.validate_data_manifest({
        "schema": core.DATA_MANIFEST_SCHEMA_V2,
        "produced_by": {"tool": "experiments/pb905_checkpoint_export.py", "entry_point": "prepare"},
        "annotations": {"scope": packet["scope"], "pb905_gate_qualified": False},
        "mount_prefix": "/mnt/shared", "entries": entries,
        "entry_count": len(entries), "total_bytes": cumulative,
        "read_plan": {"phases": phases, "read_bytes": cumulative}})


def prepare(proposal_path, proposal_sha256, directory, *, output_prefix, tier, helper_root):
    """Metadata-only preparation; run through PB, never qualify source bodies here."""
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(Path(helper_root).resolve(strict=True)))
    proposal = _json_control(proposal_path, proposal_sha256)
    from prismaquant.stage_a_chain_seed import load_pinned_checkpoint
    checkpoint = proposal["checkpoint"]
    record = load_pinned_checkpoint({"path": checkpoint["path"], "sha256": checkpoint["sha256"]})
    entries = proposal["entries"]
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA
    largest = max(row["file_bytes"] for row in entries)
    tensor = max(row["tensor_bytes"] for row in entries)
    group_bytes = proposal["group_bytes"]
    packet = check_packet({
        "schema": SCHEMA, "scope": "offline-checkpoint-export-component",
        "checkpoint": {k: checkpoint[k] for k in ("path", "bytes", "sha256")},
        "checkpoint_record": record,
        "groups": [{"index": index, "paced": index % 2 == 0,
                    "entries": entries[index * GROUP_SIZE:(index + 1) * GROUP_SIZE]}
                   for index in range(GROUPS)],
        "output_prefix": str(Path(output_prefix).resolve()), "tier": tier,
        "group_bytes": group_bytes, "total_bytes": proposal["total_bytes"],
        "storage": {"schema": BOUNDARY_STORAGE_SCHEMA,
                    "directory": str(Path(output_prefix).resolve() / "exact"),
                    "max_resident_bytes": group_bytes + 4 * largest + tensor,
                    "max_auxiliary_bytes": MAX_CONTROL_BYTES,
                    "max_artifact_bytes": proposal["total_bytes"], "prefetch_batches": GROUP_SIZE},
        "spool_max_bytes": proposal["proposed_admission"]["spool_max_bytes"],
        "source_proposal_sha256": proposal_sha256})
    directory = Path(directory).resolve()
    prefix = Path(output_prefix).resolve()
    if prefix.is_relative_to(directory) or directory.is_relative_to(prefix):
        raise ValueError("PB905 packet controls cannot live inside the producer output prefix")
    directory.mkdir(parents=True, exist_ok=False)
    raw = json_bytes(packet)
    packet_path = directory / "packet.json"
    _new_json(packet_path, packet)
    manifest = data_manifest(packet, packet_path, raw)
    from prismaquant.stage_a_produced_output import build_boundary_template
    template = build_boundary_template(
        output_prefix=packet["output_prefix"], tier=tier,
        artifact_max_bytes=packet["total_bytes"], checkpoint_max_bytes=1,
        group_size=GROUP_SIZE, max_entry_tensor_bytes=tensor,
        template_id="pq-pb905-checkpoint-export-v1", write_only=True,
        export_rate_family="pq-pb905-checkpoint-export-v1")
    _new_json(directory / "readset.json", manifest)
    _new_json(directory / "produced-template.json", template)
    cohorts = []
    for group in packet["groups"]:
        cohort = data_manifest(packet, packet_path, raw, group_index=group["index"])
        name = f"authenticate-{group['index']:02d}-readset.json"
        _new_json(directory / name, cohort)
        cohorts.append({"index": group["index"], "path": str(directory / name),
                        "sha256": sha(json_bytes(cohort))})
    summary = {"schema": "prismaquant.pb905.packet_files.v1", "packet": {
        "path": str(packet_path), "bytes": len(raw), "sha256": sha(raw)},
        "readset": {"path": str(directory / "readset.json"), "sha256": sha(json_bytes(manifest))},
        "template": {"path": str(directory / "produced-template.json"), "sha256": sha(json_bytes(template))},
        "phases": [name + "=" + ("120" if name == "input-check" else "180" if name == "finalize" else "300")
                   for name in ["input-check", *(f"group-{index:02d}" for index in range(GROUPS)), "finalize"]],
        "authentication_cohorts": cohorts, "spool_max_bytes": packet["spool_max_bytes"],
        "body_authentication": "not performed", "benchmark_go": False}
    _new_json(directory / "packet-files.json", summary)
    return summary


class FrozenArmBackend:
    """Delegate one frozen per-group arm to the existing PB spool owner."""
    def __init__(self, backend, arms):
        self.backend = backend
        if any(not isinstance(key, str) or type(value) is not bool for key, value in arms.items()):
            raise ValueError("PB905 arm map needs exact batch IDs and bool modes")
        self.arms = MappingProxyType(dict(arms))
        self.observations = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def submit_group(self, batch_id, *, entries):
        if batch_id not in self.arms:
            raise ValueError("PB905 submission is outside its frozen arm roster")
        started = time.monotonic()
        observation = {"batch_id": batch_id, "paced": self.arms[batch_id],
                       "submitted_unix": time.time(), "status": "failed"}
        try:
            answer = self.backend.submit_group(batch_id, entries=entries, paced=self.arms[batch_id])
            observation.update(status="submitted", export_key=answer.get("export_key"),
                               ok=answer.get("ok"))
            return answer
        finally:
            observation["submit_seconds"] = time.monotonic() - started
            self.observations.append(observation)


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


def load_runtime_packet(path, packet_sha256, packet_bytes, readset_sha256, *, group_index=None):
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.residency_map import bind_residency_manifest
    from prismaquant.joint_adjoint_checkpoints import _read_shared_state_payload, _verified_checkpoint_manifest
    from prismaquant.staged_lease import load_sealed_read_order
    activate_staged_tier_policy("ram,ssd")
    bind_residency_manifest(readset_sha256)
    if type(packet_bytes) is not int or not 0 < packet_bytes <= MAX_CONTROL_BYTES:
        raise ValueError("PB905 packet exceeds control budget")
    raw = _read_shared_state_payload(Path(path), {
        "name": "pb905-packet", "file_bytes": packet_bytes, "sha256": packet_sha256})
    if sha(raw) != packet_sha256:
        raise ValueError("PB905 staged packet digest differs")
    from prismaquant.staged_lease import sdk_submodule
    packet = check_packet(sdk_submodule("core")._decode_strict_json(raw, where="PB905 staged packet"))
    expected = data_manifest(packet, Path(path), raw, group_index=group_index)
    if sha(json_bytes(expected)) != readset_sha256:
        raise ValueError("PB905 complete readset differs from its sealed roster")
    actual_order = load_sealed_read_order(readset_sha256)
    wanted_order = [(row["path"], row["offset"], row["bytes"]) for row in expected["entries"]]
    if actual_order != wanted_order:
        raise ValueError("PB905 admitted read order differs")
    source = Path(packet["checkpoint"]["path"])
    _verified_checkpoint_manifest(source.parents[2], packet["checkpoint_record"],
                                  deadline=time.monotonic() + 120)
    return packet


def run_component(packet, *, mode, evidence_directory, group_index=None):
    import torch
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.memory_management import CaptureMemoryGuard
    from prismaquant.perturbed_x_cache import EntryReadScratch
    from prismaquant.prismabuild_progress import report
    from prismaquant.joint_quantum_handoff import bind_handoff_publication
    from prismaquant.joint_adjoint_checkpoints import checkpoint_entry_session
    if mode not in ("authenticate", "export"):
        raise ValueError("PB905 component mode differs")
    if mode == "export" and group_index is not None:
        raise ValueError("PB905 A/B exports stay in one paired action")
    groups = selected_groups(packet, group_index)
    if not os.environ.get("PRISMABUILD_ACTION_KEY"):
        raise ValueError("PB905 execution requires an admitted PB action")
    if mode == "export" and Path(packet["output_prefix"]).exists():
        raise ValueError("PB905 export requires a fresh evidence output prefix; retained attempts are not overwritten")
    evidence_directory = Path(evidence_directory)
    evidence_directory.mkdir(parents=True, exist_ok=False)
    guard = CaptureMemoryGuard(torch.device("cpu"))
    guard.check("pb905-startup")
    # Authentication writes no produced payload and must not consume the future export prefix.
    storage = dict(packet["storage"])
    if mode == "authenticate":
        storage["directory"] = str(evidence_directory / "reader-owner")
    owner = StreamedBoundaryArtifacts(storage, driven_by="writer")
    owner.bind({"pb905_scope": packet["scope"], "proposal_sha256": packet["source_proposal_sha256"]},
               n_probes=1, published=True, check_memory=guard.check)
    publication = None
    arm_backend = None
    scratch = EntryReadScratch()
    result = {"schema": RESULT_SCHEMA, "mode": mode, "status": "failed", "groups": [],
              "checkpoint_sha256": packet["checkpoint"]["sha256"],
              "action_key": os.environ["PRISMABUILD_ACTION_KEY"], "pb905_gate_qualified": False,
              "gate_disposition": "INCONCLUSIVE: component evidence alone has no organic mover-contention witness."}
    _new_json(evidence_directory / "input-check.json", {
        "checkpoint_sha256": packet["checkpoint"]["sha256"], "mode": mode})
    report("input-check", 1, unit="durable-component-records")
    expected_session = checkpoint_entry_session(packet["checkpoint_record"])
    try:
        with owner:
            if mode == "export":
                publication = bind_handoff_publication(boundary_storage=storage)
                if publication is None:
                    raise ValueError("PB905 export has no bound produced-output publication")
                owner.bind_produced_output(publication, group_size=GROUP_SIZE, n_batches=GROUPS * GROUP_SIZE,
                                          max_entry_tensor_bytes=max(row["tensor_bytes"] for group in groups for row in group["entries"]),
                                          origin_lifetime="retain", dispose_on_failure=True)
                adapter = owner._local_output_spool
                if adapter is None or adapter.max_bytes != packet["spool_max_bytes"]:
                    raise ValueError("PB905 local spool differs from the sealed bound")
                arms = {publication.batch_id_for(kind=KIND, boundary_index=group["index"], group_index=0): group["paced"]
                        for group in groups}
                arm_backend = FrozenArmBackend(adapter.backend, arms)
                adapter.backend = arm_backend
            for group in groups:
                phase = f"group-{group['index']:02d}"
                report(phase, 1 + len(result["groups"]), unit="durable-component-records")
                started = time.monotonic()
                future = ((packet["group_bytes"] if mode == "export" else max(row["file_bytes"] for row in group["entries"]))
                          + 2 * max(row["file_bytes"] for row in group["entries"]))
                guard.check("pb905-group-input", reserve_bytes=future)
                with torch.profiler.record_function("pb905." + phase):
                    with verified_group(group, expected_session=expected_session, scratch=scratch,
                                        owner=owner, keep_bytes=mode == "export",
                                        resource_check=guard.check) as (files, checked):
                        refs = (owner.write_produced_files(files, kind=KIND, boundary_index=group["index"])
                                if mode == "export" else [])
                row = {"index": group["index"], "paced": group["paced"], "sources": checked,
                       "bytes": sum(entry["bytes"] for entry in checked),
                       "end_to_end_seconds": time.monotonic() - started,
                       "origins": [vars(ref) for ref in refs]}
                result["groups"].append(row)
                _new_json(evidence_directory / (phase + ".json"), row)
                report(phase, 1 + len(result["groups"]), unit="durable-component-records")
        result["status"] = "complete"
    finally:
        primary = sys.exception()
        try:
            scratch.release()
            result["writer"] = owner.produced_output_report()
            result["memory_guard"] = guard.snapshot()
            if arm_backend is not None:
                result["submissions"] = arm_backend.observations
            _new_json(evidence_directory / "result.json", result)
        except BaseException as failure:
            if primary is None:
                raise
            primary.add_note(f"PB905 final evidence/cleanup also failed: {failure!r}")
    report("finalize", len(groups) + 2, unit="durable-component-records")
    return result


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
