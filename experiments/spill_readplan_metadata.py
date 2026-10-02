"""CPU-only read-plan metadata evidence, not routed/model or IO throughput.

The supplied source runs a real StageBReplaySpill planner and IO-engine
descriptor constructor over the same synthetic firing/input roster. Probe
readiness remains false, so no scratch payload is allocated or read. Existing
capture entries/firing storage are built before plan tracing and held equally.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import threading
import tracemalloc
from types import SimpleNamespace

from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant.io_spans import PeriodicSampler
from prismaquant import joint_replay_spill as spill


def metadata_session(targets, batches, *, probes=4, packed=False, read_bytes=1 << 20):
    names = tuple(f"target-{index}" for index in range(targets))
    maximum = targets * batches * (probes + 1)
    window = spill._Window(names, records=spill._PackedRecords(names, max_records=maximum))
    g_logical = dict.fromkeys(names, 0)
    for batch in range(batches):
        rows, residue = 4 + batch % 8, batch % 4 * 128
        for index, name in enumerate(names):
            owner = names[index - index % 2]
            x_size, g_size = rows * 1024 * 2, rows * 2048 * 2
            if index % 2 == 0:
                entries = window.entries.setdefault(owner, [])
                logical = window.x_logical.get(owner, 0)
                entries.append(spill._Entry(logical, x_size, ((rows, 1024), (1024, 1), residue)))
                window.x_logical[owner] = logical + x_size
            window.x_source[name] = owner
            window.records.append((name, owner, batch, g_logical[name], g_size,
                                   ((rows, 2048), (2048, 1), residue)))
            g_logical[name] += g_size
    session = object.__new__(spill.StageBReplaySpill)
    session.geometry = SimpleNamespace(max_parts=maximum)
    session._packed_read_plan = packed
    session._windows = [window]
    session._block = 4096
    session.read_bytes = read_bytes
    session.n_probes = probes
    session._cuda = False
    session._captured = -1  # Real IO-engine gate must defer all reads.
    session._replay_stream = session._replay_budget = session.wait_sink = None
    session.telemetry = {}
    return session, window


def plan_digest(window):
    names = {name: index for index, name in enumerate(window.names)}
    digest = hashlib.sha256()
    for owner, (records, inputs, gradients, used) in window.plan:
        digest.update(struct.pack("<5Q", names[owner], len(records), len(inputs), len(gradients), used))
        for position in records:
            digest.update(struct.pack("<Q", position))
        for pairs in (inputs, gradients):
            for position, offset in pairs:
                digest.update(struct.pack("<2Q", position, offset))
    for owner, entries in window.entries.items():
        for entry in range(len(entries)):
            digest.update(struct.pack("<3Q", names[owner], entry, window.last_ref[owner, entry]))
    return digest.hexdigest()


def arm(args):
    session, window = metadata_session(args.targets, args.batches, probes=args.probes,
                                       packed=args.mode == "packed")
    gc.collect()
    profile = cProfile.Profile()
    tracemalloc.start()
    profile.enable()
    window.plan = session._plan(window)
    plan_retained, plan_peak = tracemalloc.get_traced_memory()
    stream = session._open_replay_stream()
    profile.disable()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profile.dump_stats(str(args.out / f"{args.label}.pstats"))
    report = dict(mode=args.mode, targets=args.targets, batches=args.batches,
                  probes=args.probes, firing_records=len(window.records),
                  declared_max_parts=session.geometry.max_parts, chunks=len(window.plan),
                  descriptors=len(stream._entries), plan_retained_bytes=plan_retained,
                  plan_peak_bytes=plan_peak, plan_and_engine_retained_bytes=retained,
                  plan_and_engine_peak_bytes=peak,
                  process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  retained_reader_chunk_objects=sum(
                      isinstance(entry.reader.args[-1], tuple) for entry in stream._entries),
                  ordered_plan_last_use_sha256=plan_digest(window))
    session.close_replay_stream()
    if session.telemetry["replay_stream"]["entries_read"]:
        raise RuntimeError("metadata experiment unexpectedly read spill payloads")
    (args.out / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=867)
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--mode", choices=("legacy", "packed"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.targets <= 867 and 1 <= args.batches <= 512 and 1 <= args.probes <= 4):
        parser.error("bounded corpus: 867 targets, 512 batches, 4 probes maximum")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode:
        arm(args)
        return
    modes = ("legacy", "packed", "legacy", "packed") if args.paired else ("legacy",)
    reports = []
    with (args.out / "netdata.jsonl").open("x") as handle:
        writer = NetdataWriter(handle)
        def sample():
            for host in ("sparky.lan", "sparklina.lan"):
                writer.write(sample_netdata(host))
        sample()
        with PeriodicSampler(sample, interval_s=2, name="readplan-metadata"):
            for index, mode in enumerate(modes):
                label = f"{index}-{mode}"
                subprocess.run([sys.executable, "-m", "experiments.spill_readplan_metadata",
                                "--out", str(args.out), "--targets", str(args.targets),
                                "--batches", str(args.batches), "--probes", str(args.probes),
                                "--mode", mode, "--label", label], check=True, timeout=300)
                reports.append(json.loads((args.out / f"{label}.json").read_text()))
        sample()
    if len({row["ordered_plan_last_use_sha256"] for row in reports}) != 1:
        raise RuntimeError("candidate changed ordered chunk/offset/last-use metadata")
    artifacts = [dict(path=str(path), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in sorted(args.out.iterdir()) if path.is_file()]
    print(json.dumps(dict(scope="synthetic_metadata_only_no_payload_reads", reports=reports,
                          artifacts=artifacts, affinity=sorted(os.sched_getaffinity(0)))))


if __name__ == "__main__":
    main()
