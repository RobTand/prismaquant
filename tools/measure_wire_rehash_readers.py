#!/usr/bin/env python3
"""Measure census wire re-hash throughput at several concurrent reader counts.

Each arm hashes its own cold slice of routed-expert wire blobs (no file is
read by two arms), so an arm is not flattered by the page cache of an earlier
one.  Names come from the census naming scheme, never from a directory
listing: a READDIR over a 266k-entry wire directory is itself an NFS storm.

Per arm it records bytes, seconds, MB/s, the process's ``rchar`` and
``read_bytes`` deltas, and the NFS client's READ queue and round-trip
milliseconds per op from ``/proc/self/mountstats``.  The queue-versus-RTT
split is the diagnostic: a long queue with a short RTT is a client transport
bound that more readers cannot relieve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_BLOCK = 16 << 20


def _proc_io() -> dict[str, int]:
    out = {}
    with open("/proc/self/io") as handle:
        for line in handle:
            key, value = line.split(":")
            out[key.strip()] = int(value)
    return out


def _nfs_read(mount: str) -> list[int] | None:
    current = False
    with open("/proc/self/mountstats") as handle:
        for line in handle:
            if line.startswith("device "):
                current = f" {mount} " in line and " nfs" in line
            elif current:
                fields = line.split()
                if fields and fields[0] == "READ:":
                    return [int(x) for x in fields[1:9]]
    return None


def _hash(path: Path) -> int:
    digest = hashlib.sha256()
    buffer = bytearray(_BLOCK)
    view = memoryview(buffer)
    size = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(view)
            if not count:
                break
            digest.update(view[:count])
            size += count
    return size


def _candidates(wire_dir: Path, fmt: str, layers: range, experts: int):
    for layer in layers:
        for expert in range(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield wire_dir / (f"model__language_model__layers__{layer}__mlp__experts__"
                                  f"{expert}__{proj}__{fmt}.tessera")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire-dir", required=True)
    parser.add_argument("--format", default="TESSERA_E4M3_K1_R1024")
    parser.add_argument("--first-layer", type=int, required=True)
    parser.add_argument("--last-layer", type=int, default=44)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--readers", default="1,8,16")
    parser.add_argument("--bytes-per-arm", type=int, default=2 << 30)
    parser.add_argument("--mount", default="/mnt/shared")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    source = _candidates(Path(args.wire_dir), args.format,
                         range(args.first_layer, args.last_layer + 1), args.experts)
    arms = []
    for readers in (int(x) for x in args.readers.split(",")):
        files, planned = [], 0
        for path in source:
            try:
                planned += path.stat().st_size
            except FileNotFoundError:
                continue
            files.append(path)
            if planned >= args.bytes_per_arm:
                break
        if not files:
            raise SystemExit(f"no wire files left for the {readers}-reader arm")
        io0, nfs0, t0 = _proc_io(), _nfs_read(args.mount), time.monotonic()
        with ThreadPoolExecutor(max_workers=readers) as pool:
            hashed = sum(pool.map(_hash, files))
        seconds = time.monotonic() - t0
        io1, nfs1 = _proc_io(), _nfs_read(args.mount)
        arm = {"readers": readers, "files": len(files), "bytes": hashed,
               "seconds": round(seconds, 2), "mb_per_s": round(hashed / seconds / 1e6, 1),
               "rchar_mb_per_s": round((io1["rchar"] - io0["rchar"]) / seconds / 1e6, 1),
               "read_bytes_mb_per_s": round(
                   (io1["read_bytes"] - io0["read_bytes"]) / seconds / 1e6, 1),
               "first_file": files[0].name, "last_file": files[-1].name}
        if nfs0 and nfs1:
            delta = [b - a for a, b in zip(nfs0, nfs1)]
            ops = delta[0] or 1
            arm["nfs_read"] = {"ops": delta[0], "queue_ms_per_op": round(delta[5] / ops, 2),
                               "rtt_ms_per_op": round(delta[6] / ops, 2),
                               "execute_ms_per_op": round(delta[7] / ops, 2)}
        print(json.dumps(arm, sort_keys=True), flush=True)
        arms.append(arm)
    report = {"schema": "prismaquant.wire_rehash_reader_sweep.v1", "host": os.uname().nodename,
              "format": args.format, "block_bytes": _BLOCK, "arms": arms}
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
