"""CPU metadata stress for PQ #1086, without loading weights or operands.

867 targets x 512 invocations retains 443,904 probe-0 gradient records.
Four probes plus probe-0 inputs yield the conservative 2,219,520 part bound.
These synthetic metadata shapes are not measured GLM routing or its proxy
arithmetic digest. Fresh child processes separate allocator high-water marks.
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
import tracemalloc

from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant.io_spans import PeriodicSampler
from prismaquant.joint_replay_spill import _PackedRecords


def corpus(names, batches):
    for batch in range(batches):
        rows = 1 + batch % 64
        for index, name in enumerate(names):
            width = 2048 if index % 3 else 4096
            # Construct fresh tuples as _layout does for each invocation.
            layout = (tuple([rows, width]), tuple([width, 1]), batch % 4 * 128)
            yield (name, name, batch, batch * 64 * width * 2, rows * width * 2, layout)


def arm(args):
    names = tuple(f"target-{index}" for index in range(args.targets))
    count = args.targets * args.batches
    parts = count * (args.probes + 1)
    gc.collect()
    tracemalloc.start()
    profile = cProfile.Profile()
    profile.enable()
    records = ([] if args.mode == "list" else
               _PackedRecords(names, max_records=parts))
    for record in corpus(names, args.batches):
        records.append(record)
    del record
    profile.disable()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profile.dump_stats(str(args.out / f"{args.mode}.pstats"))
    # Include every field, including alignment residue, in the order oracle.
    name_ids = {name: index for index, name in enumerate(names)}
    digest = hashlib.sha256()
    for name, owner, entry, logical, nbytes, (shape, stride, residue) in records:
        digest.update(struct.pack("<10Q", name_ids[name], name_ids[owner], entry,
                                  logical, nbytes, *shape, *stride, residue))
    report = dict(mode=args.mode, records=len(records), geometry_max_parts=parts,
                  traced_retained_bytes=retained, traced_peak_bytes=peak,
                  process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  ordered_record_sha256=digest.hexdigest(),
                  layouts=(None if args.mode == "list" else len(records._layouts)))
    (args.out / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=867)
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--mode", choices=("list", "packed"))
    args = parser.parse_args()
    if not (1 <= args.targets <= 867 and 1 <= args.batches <= 512
            and 1 <= args.probes <= 4):
        parser.error("the bounded corpus permits at most 867 targets, 512 batches and 4 probes")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode:
        arm(args)
        return
    with (args.out / "netdata.jsonl").open("x") as handle:
        writer = NetdataWriter(handle)
        def sample():
            for host in ("sparky.lan", "sparklina.lan"):
                try:
                    writer.write(sample_netdata(host))
                except Exception as error:
                    writer.write(dict(host=host, error=repr(error)))
        sample()
        with PeriodicSampler(sample, interval_s=2, name="spill-record-memory"):
            for mode in ("list", "packed"):
                command = [sys.executable, "-m", "experiments.stageb_spill_record_memory",
                           "--out", str(args.out), "--targets", str(args.targets),
                           "--batches", str(args.batches), "--probes", str(args.probes),
                           "--mode", mode]
                subprocess.run(command, check=True, timeout=240)
        sample()
    reports = [json.loads((args.out / f"{mode}.json").read_text())
               for mode in ("list", "packed")]
    if reports[0]["ordered_record_sha256"] != reports[1]["ordered_record_sha256"]:
        raise RuntimeError("packed metadata changed the ordered record fields")
    print(json.dumps(dict(scope="synthetic_cpu_metadata_only", reports=reports,
                          cpu_affinity=sorted(os.sched_getaffinity(0)))))


if __name__ == "__main__":
    main()
