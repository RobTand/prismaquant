"""Replay existing unit states into fresh host-only checkpoint namespaces.

This qualifies byte equivalence and execution ownership, not GPU overlap or a
representative row speedup. It never modifies its published input namespace.
The historical record stores window indices, not resolved unit membership;
therefore the replay uses explicitly reported balanced host partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import threading

from prismaquant import aura_cost
from prismaquant.joint_checkpoint_publication import CheckpointPublicationLedger


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(source: Path, destination: Path, *, expected_units: int,
           host_windows: int, budget_bytes: int, max_jobs: int) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("replay destination must not extend the published source")
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = source / "manifest.json"
    manifest_digest = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["units"]
    if len(rows) != expected_units or expected_units <= 0 or host_windows <= 0:
        raise ValueError("replay census does not match its explicit contract")
    names = [row["qname"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate replay unit")
    identity = manifest["identity_sha256"]
    paths = {}
    for row in rows:
        path = (source / row["file"]).resolve(strict=True)
        if source not in path.parents or path != aura_cost._aura_unit_checkpoint_path(source, row["qname"]).resolve():
            raise ValueError("unit path is not the existing canonical destination")
        paths[row["qname"]] = path
    windows = [{"names": names[start:stop]} for start, stop in
               ((len(names) * i // host_windows, len(names) * (i + 1) // host_windows)
                for i in range(host_windows))]
    sync, asynchronous = destination / "synchronous", destination / "shared-engine"
    consumer = threading.get_ident()
    calls = {"synchronous": [], "shared-engine": []}
    arm = "synchronous"
    encoder = aura_cost._encode_aura_unit_checkpoint

    def observed_encode(**kwargs):
        calls[arm].append((threading.get_ident(), threading.current_thread().name))
        return encoder(**kwargs)

    def load(name):
        # These are trusted existing campaign pickles, never arbitrary CLI URLs.
        envelope = pickle.loads(paths[name].read_bytes())
        if envelope.get("identity_sha256") != identity:
            raise ValueError("replay lineage differs from the existing manifest")
        return aura_cost._load_aura_unit_checkpoint(
            paths[name], qname=name, identity_sha256=identity)

    source_digests, expected, input_bytes = {}, {}, 0
    historical_matches = 0
    aura_cost._encode_aura_unit_checkpoint = observed_encode
    try:
        for name in names:
            source_digests[name] = digest(paths[name])
            input_bytes += paths[name].stat().st_size
            aura_cost._write_aura_unit_checkpoint(
                sync, qname=name, identity_sha256=identity, state=load(name))
            target = aura_cost._aura_unit_checkpoint_path(sync, name)
            expected[name] = digest(target)
            historical_matches += expected[name] == source_digests[name]
        arm = "shared-engine"
        durable, checked, done = set(), set(), []

        def acknowledge():
            for name in durable - checked:
                target = aura_cost._aura_unit_checkpoint_path(asynchronous, name)
                if digest(target) != expected[name]:
                    raise AssertionError("durable acknowledgement precedes matching bytes")
                checked.add(name)

        ledger = CheckpointPublicationLedger(
            checkpoint_root=asynchronous, identity_sha256=identity,
            windows=windows, completed=durable, acknowledge=acknowledge,
            window_done=done.append, budget_bytes=budget_bytes, max_jobs=max_jobs)
        try:
            for index, window in enumerate(windows):
                ledger.start_window(index, window["names"])
                for name in window["names"]:
                    if not ledger.submit(name, lambda limit, name=name: load(name)):
                        raise AssertionError("fresh replay skipped a unit")
            ledger.flush()
            if durable != set(names) or checked != durable or done != list(range(host_windows)):
                raise AssertionError("replay did not reach its durable frontier")
            for name in names:
                if ledger.submit(name, lambda limit: (_ for _ in ()).throw(
                        AssertionError("durable replay resumed by constructing state"))):
                    raise AssertionError("durable unit was resubmitted")
            ledger.close()
        except BaseException:
            ledger.cancel()
            raise
        stats = ledger.stats()
        if stats["charged_bytes"] or stats["pending_units"] or stats["acknowledged_units"] != expected_units:
            raise AssertionError("replay retained outstanding publication ownership")
        if len(calls["synchronous"]) != expected_units or any(t != consumer for t, _ in calls["synchronous"]):
            raise AssertionError("synchronous control did not use the consumer encoder")
        if len(calls["shared-engine"]) != expected_units or any(
                t == consumer or not label.startswith("pq-io") for t, label in calls["shared-engine"]):
            raise AssertionError("candidate serialization did not use the shared IO engine")
        if digest(manifest_path) != manifest_digest or any(
                digest(paths[name]) != source_digests[name] for name in names):
            raise AssertionError("published input changed during replay")
        file_record = {name: {"source_sha256": source_digests[name],
                              "baseline_candidate_sha256": expected[name]}
                       for name in names}
        file_bytes = (json.dumps(file_record, indent=2, sort_keys=True) + "\n").encode()
        (destination / "files.json").write_bytes(file_bytes)
        result = {
            "schema": "prismaquant.stage_b_host_checkpoint_replay.v1",
            "qualification": "cpu_equivalence_and_ownership_only",
            "source": str(source), "destination": str(destination),
            "source_manifest_sha256": manifest_digest, "identity_sha256": identity,
            "file_digest_record_sha256": hashlib.sha256(file_bytes).hexdigest(),
            "units": expected_units, "source_bytes": input_bytes,
            "host_partitions": [len(w["names"]) for w in windows],
            "original_resolved_window_membership_replayed": False,
            "baseline_candidate_digests_equal": True,
            "historical_file_digest_matches": historical_matches,
            "source_unchanged": True, "durable_resume_skipped": expected_units,
            "encoder_calls_consumer": len(calls["synchronous"]),
            "encoder_calls_shared_io": len(calls["shared-engine"]),
            "publication": stats,
            "representative_profile_obligation": "next real Stage B production run; PQ #1253",
            "gpu_overlap_or_speedup_claim": False,
        }
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        aura_cost._encode_aura_unit_checkpoint = encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, required=True)
    parser.add_argument("--host-windows", type=int, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    args = parser.parse_args()
    replay(args.source, args.destination, expected_units=args.expected_units,
           host_windows=args.host_windows, budget_bytes=args.budget_bytes,
           max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
