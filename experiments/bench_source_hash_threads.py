"""Source-shard hashing throughput vs thread count, through cost_streaming._hash_source_shards (PQ #1363, #1379).

Run under PB: pbrun --tag gb10 --demand cpu=8,mem_gb=4 --env PYTHONPATH=. -- <venv python> experiments/bench_source_hash_threads.py

Each thread count reads its own disjoint set of GLM-5.3 shards from the NFS
pool. Page-cache residency (fincore) is recorded before each read so a warm
set is visible; /proc/self/io read_bytes and wall clock are recorded per run.
"""
import json, os, subprocess, time
from pathlib import Path
from prismaquant import cost_streaming as cs

root = Path("/mnt/shared/models/GLM-5.3-Flash-BF16")
PER_SET = 8
sweep = [1, 2, 4, 8]
first = 40

def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io").read().splitlines())}

def fincore(paths):
    try:
        out = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES,SIZE", *map(str, paths)],
                             capture_output=True, text=True, check=True).stdout.split()
        return round(sum(int(x) for x in out[0::2]) / sum(int(x) for x in out[1::2]), 4)
    except Exception as e:
        return f"unavailable: {e}"

runs = []
for index, threads in enumerate(sweep):
    paths = [root / f"model-{first + index * PER_SET + k:05d}-of-00120.safetensors" for k in range(PER_SET)]
    os.environ["PRISMAQUANT_SOURCE_HASH_THREADS"] = str(threads)
    cached = fincore(paths)
    work = [(p, cs._streamed_identity_stat_fingerprint(p)) for p in paths]
    before, start, t0 = io(), time.time(), time.monotonic()
    digests = cs._hash_source_shards(work)
    wall, after = time.monotonic() - t0, io()
    size = sum(p.stat().st_size for p in paths)
    row = dict(threads=threads, shards=[p.name[6:11] for p in paths], bytes=size,
               page_cache_resident_before=cached, read_bytes=after["read_bytes"] - before["read_bytes"],
               wall_s=round(wall, 2), mb_per_s=round(size / wall / 1e6, 1),
               start_unix=round(start, 1), end_unix=round(start + wall, 1),
               digests=[d[:12] for d in digests])
    runs.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(15)  # separate the Netdata windows
print(json.dumps({"host": os.uname().nodename, "affinity": len(os.sched_getaffinity(0)), "runs": runs}), flush=True)
