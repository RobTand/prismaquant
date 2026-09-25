"""How fast one staged tensor arrives, on one stream and on the mount's own.

The claim PQ #746 rests on is a delta, so this tool measures both arms of it on
the same bytes, in the same action, alternating so a drift in the tier cannot
land on one arm. It is the measurement half of the change; the byte half stays
where it already is, in ``tools/layer1_stage_fed.py``, and is repeated here per
arm so a fast read that is wrong cannot be reported as a fast read.

Each arm:

* drops the client's page cache for every staged range it is about to read
  (``posix_fadvise(DONTNEED)``), because a warm client page cache measures
  memcpy and not the link, and reports the mount counters so a warm arm is
  visible rather than silently fast;
* reads the tensor through the production seam
  (``layer_streaming._source_safe_open`` -> ``get_tensor``), never around it;
* records wall time, the process's own ``/proc/self/io`` deltas, and the NFS
  client and server byte counters for the stage mount.

Read-only. Writes nothing anywhere.
"""
import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import time


def mount_bytes():
    """{mountpoint: (client_read_bytes, server_read_bytes)} from mountstats."""
    out, cur = {}, None
    for line in open("/proc/self/mountstats"):
        if line.startswith("device "):
            parts = line.split()
            cur = parts[parts.index("on") + 1] if " on " in line else None
        elif cur and line.strip().startswith("bytes:"):
            f = [int(x) for x in line.split()[1:]]
            out[cur] = (f[0], f[4])
            cur = None
    return out


def drop_client_cache(paths):
    """Forget what this client cached, so the next read is a read."""
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def digest_tensor(tensor, torch):
    """A frame of its own, so a profile does not charge it to the read.

    Hashing 1.2 GB costs real time and it is this tool's cost, not the
    reader's; unnamed it lands inside the read's frame and reads as if the
    reader spent it.
    """
    return hashlib.sha256(
        memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,4,8,16,32",
                    help="comma-separated stream counts; 0 means the mount's own")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tensors", type=int, default=2,
                    help="largest fully staged tensors per shard to read")
    ap.add_argument("--warmup", action="store_true",
                    help="read every target once, untimed, before the arms, so "
                         "the server's cache is warm for all of them and the "
                         "only cold thing left is this client's page cache")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    named = os.environ.get("PRISMABUILD_RESIDENCY_MAP")
    if not named:
        print("REFUSED: no residency map in the environment.", flush=True)
        return 2
    if not os.path.isdir("/stage/prewarm"):
        print("REFUSED: /stage/prewarm is not mounted.", flush=True)
        return 2
    body = json.load(open(named))

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from safetensors import safe_open
    from prismaquant import residency_shard_reader
    from prismaquant.io_spans import read_proc_io
    from prismaquant.layer_streaming import _source_safe_open
    from prismaquant.residency_map import bind_residency_manifest, residency_resolver

    bind_residency_manifest(body["manifest_sha256"])
    resolver = residency_resolver()
    if resolver is None:
        print("REFUSED: the map did not bind.", flush=True)
        return 2

    mount = residency_shard_reader._mount_options("/stage/prewarm")
    print(f"[mount] /stage/prewarm {mount[1] if mount else '?'} "
          f"{mount[2] if mount else '?'}", flush=True)
    print(f"[mount] read shape from the mount: "
          f"{residency_shard_reader._read_shape('/stage/prewarm')}", flush=True)

    # The tensors to read: the largest ones a staged range covers outright.
    targets = []
    for shard in sorted({k.split(":", 1)[1] for k in body["entries"]}):
        ranges = [(int(v["offset"]), int(v["offset"]) + int(v["bytes"]),
                   v["stage_path"])
                  for k, v in body["entries"].items()
                  if k.split(":", 1)[1] == shard]
        header, base, _ = residency_shard_reader._read_shard_header(shard)
        rows = []
        for name, row in header.items():
            if name == "__metadata__" or not isinstance(row, dict):
                continue
            begin, end = row["data_offsets"]
            begin, end = base + begin, base + end
            covering = [r for r in ranges if r[0] <= begin and end <= r[1]]
            if covering and end > begin:
                rows.append((end - begin, name, covering[0][2]))
        rows.sort(reverse=True)
        for size, name, stage_path in rows[:args.tensors]:
            targets.append({"shard": shard, "name": name, "bytes": size,
                            "stage_path": stage_path})
    if not targets:
        print("REFUSED: no tensor is covered outright by a staged range.", flush=True)
        return 2
    for t in targets:
        print(f"[plan] {os.path.basename(t['shard'])} {t['name'][:50]} "
              f"{t['bytes'] / 2**20:.1f} MiB", flush=True)

    arms = [int(x) for x in args.streams.split(",") if x.strip()]
    samples = {n: [] for n in arms}
    records = []

    def read_once(target, streams):
        if streams:
            os.environ[residency_shard_reader.STREAMS_ENV] = str(streams)
        else:
            os.environ.pop(residency_shard_reader.STREAMS_ENV, None)
        residency_shard_reader.reset_read_shape_cache_for_tests()
        shape = residency_shard_reader._read_shape(target["stage_path"])
        drop_client_cache([target["stage_path"]])
        m0, i0 = mount_bytes(), read_proc_io()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.time()
        with _source_safe_open(target["shard"], framework="pt") as handle:
            tensor = handle.get_tensor(target["name"])
        wall = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        m1, i1 = mount_bytes(), read_proc_io()
        digest = digest_tensor(tensor, torch)
        stage_client = m1["/stage/prewarm"][0] - m0["/stage/prewarm"][0]
        stage_server = m1["/stage/prewarm"][1] - m0["/stage/prewarm"][1]
        pool_server = (m1.get("/mnt/shared", (0, 0))[1]
                       - m0.get("/mnt/shared", (0, 0))[1])
        row = {
            "shard": os.path.basename(target["shard"]), "tensor": target["name"],
            "streams": streams, "bytes": target["bytes"], "wall_s": wall,
            "mb_s": target["bytes"] / wall / 1e6,
            "stage_client_bytes": stage_client, "stage_server_bytes": stage_server,
            "pool_server_bytes": pool_server,
            "rchar": i1["rchar"] - i0["rchar"],
            "read_bytes": i1["read_bytes"] - i0["read_bytes"],
            "cold": stage_server >= target["bytes"] * 0.95,
            "read_shape": shape,
            "cpu_s": ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)),
            "sha256": digest,
        }
        del tensor
        return row

    if args.warmup:
        # Rob's 645 -> 10045 MB/s dd pair was ARC-warm on both sides, so the
        # question this tool answers is the client's, not the pool's. Touch
        # every target once first and the arms differ in stream count alone.
        for target in targets:
            # The arm's own stream count, so a single-stream arm's profile is
            # not haunted by a pool the warm-up built and then left idle.
            row = read_once(target, arms[0])
            print(f"[warm] {target['name'][:38]:38s} "
                  f"{row['mb_s']:8.1f} MB/s (untimed, discarded)", flush=True)

    print("", flush=True)
    for repeat in range(args.repeats):
        order = arms if repeat % 2 == 0 else list(reversed(arms))
        for streams in order:
            for target in targets:
                row = read_once(target, streams)
                row["repeat"] = repeat
                records.append(row)
                if row["cold"]:
                    samples[streams].append(row["mb_s"])
                print(f"[arm] streams={streams if streams else 'mount'}"
                      f" rep={repeat} {row['tensor'][:38]:38s} "
                      f"{row['bytes'] / 2**20:8.1f} MiB in {row['wall_s']:6.2f}s "
                      f"= {row['mb_s']:8.1f} MB/s | stage server "
                      f"{row['stage_server_bytes'] / 2**20:8.1f} MiB | "
                      f"{'COLD' if row['cold'] else 'warm - discarded'}", flush=True)

    # Byte identity, once per tensor, outside every timed read.
    mismatches = []
    for target in targets:
        with safe_open(target["shard"], framework="pt") as handle:
            reference = handle.get_tensor(target["name"])
        want = digest_tensor(reference, torch)
        got = {r["sha256"] for r in records if r["tensor"] == target["name"]}
        if got != {want}:
            mismatches.append(target["name"])
        del reference

    print("", flush=True)
    print("=" * 78, flush=True)
    summary = {}
    for streams in arms:
        values = samples[streams]
        if not values:
            print(f"streams={streams if streams else 'mount':>6}  "
                  f"no cold sample", flush=True)
            continue
        cold = [r for r in records if r["streams"] == streams and r["cold"]]
        cpu_share = statistics.median([r["cpu_s"] / r["wall_s"] for r in cold])
        rchar = statistics.median([r["rchar"] / r["bytes"] for r in cold])
        summary[streams] = {
            "cold_samples": len(values),
            "median_mb_s": statistics.median(values),
            "min_mb_s": min(values), "max_mb_s": max(values),
            "median_cpu_cores": cpu_share,
            "median_rchar_over_payload": rchar,
        }
        print(f"streams={streams if streams else 'mount':>6}  "
              f"median {statistics.median(values):8.1f} MB/s  "
              f"(min {min(values):8.1f}, max {max(values):8.1f}, "
              f"n={len(values)})  cpu {cpu_share:5.2f} cores  "
              f"rchar/payload {rchar:5.3f}", flush=True)
    print("=" * 78, flush=True)
    print(f"byte mismatches: {len(mismatches)}", flush=True)

    out = {"schema": "prismaquant.staged_read_stream_ab.v1",
           "mount": {"fstype": mount[1] if mount else None,
                     "options": mount[2] if mount else None},
           "targets": targets, "records": records, "summary": summary,
           "mismatches": mismatches}
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(out, handle, indent=1)
    print(json.dumps({"summary": summary, "mismatches": mismatches}), flush=True)

    if mismatches:
        print("FAIL: a staged read returned different bytes than the pool.", flush=True)
        return 1
    if not summary:
        print("FAIL: every arm read a warm client cache; nothing was measured.",
              flush=True)
        return 1
    # An arm that asked for several streams and got no read shape read on one
    # stream and would be reported as if the split had been measured. That is
    # what the first sweep of PQ #746 did, so it is a refusal and not a note.
    inert = sorted({r["streams"] for r in records
                    if r["streams"] != 1 and r["read_shape"] is None})
    if inert:
        print(f"FAIL: arms {inert} found no read shape on the staged mount, so "
              f"they read on one stream and measured the arm they were "
              f"comparing against.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
