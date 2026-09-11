"""Read one PrismaBuild data manifest off the local pool and say where the bytes came from.

The point is the three-way split of a row's bytes: what the ARC already held,
what the L2ARC NVMe served, and what the spindles had to fetch. Counters are
read on this box, around the read, so the bracket is the read itself.

Run it on the storage box, pinned, through PrismaBuild.
"""

from __future__ import annotations

import argparse
import json
import os
import stat as statmod
import sys
import threading
import time

ARCSTATS = "/proc/spl/kstat/zfs/arcstats"
BLOCK = 1 << 20
HDDS = ("sdb", "sdc", "sdd", "sde")
NVMES = ("nvme0n1",)

ARC_KEYS = (
    "size", "c", "c_min", "c_max", "hits", "misses",
    "demand_data_hits", "demand_data_misses",
    "l2_hits", "l2_misses", "l2_size", "l2_asize",
    "l2_read_bytes", "l2_write_bytes", "l2_feeds",
    "mru_size", "mfu_size", "evict_l2_cached",
)


def arcstats(path: str = ARCSTATS) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            field = line.split()
            if len(field) == 3 and field[0] in ARC_KEYS:
                out[field[0]] = int(field[2])
    return out


def diskstats() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    with open("/proc/diskstats") as handle:
        for line in handle:
            field = line.split()
            if len(field) > 9 and field[2] in HDDS + NVMES:
                out[field[2]] = (int(field[5]) * 512, int(field[9]) * 512)
    return out


def nfsd_io() -> tuple[int, int]:
    try:
        with open("/proc/net/rpc/nfsd") as handle:
            for line in handle:
                if line.startswith("io "):
                    field = line.split()
                    return int(field[1]), int(field[2])
    except OSError:
        pass
    return (0, 0)


def sample() -> dict:
    return {
        "unix": time.time(),
        "arc": arcstats(),
        "disk": diskstats(),
        "nfsd": nfsd_io(),
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }


class Local:
    """Rewrites a manifest path onto this box's own pool mount."""

    def __init__(self, text: str) -> None:
        self.shared, _, self.local = text.partition("=")
        if not self.shared or not self.local:
            raise SystemExit(f"a mount map is SHARED=LOCAL, not {text!r}")

    def __call__(self, path: str) -> str:
        if path == self.shared:
            return self.local
        if not path.startswith(self.shared + "/"):
            raise SystemExit(f"{path!r} is not under {self.shared!r}")
        return self.local + path[len(self.shared):]


def read_entries(entries, to_local, readers: int) -> dict:
    """Reads every entry, in manifest order, across ``readers`` threads."""
    lock = threading.Lock()
    state = {"next": 0, "bytes": 0, "files": 0, "errors": []}
    buffer_size = BLOCK

    def worker() -> None:
        buf = bytearray(buffer_size)
        view = memoryview(buf)
        while True:
            with lock:
                index = state["next"]
                if index >= len(entries):
                    return
                state["next"] = index + 1
            entry = entries[index]
            path = to_local(entry["path"])
            want = int(entry["bytes"])
            offset = int(entry.get("offset", 0))
            got = 0
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except OSError as error:
                with lock:
                    state["errors"].append(f"{path}: {error}")
                continue
            try:
                if not statmod.S_ISREG(os.fstat(fd).st_mode):
                    with lock:
                        state["errors"].append(f"{path}: not a regular file")
                    continue
                if offset:
                    os.lseek(fd, offset, os.SEEK_SET)
                while got < want:
                    chunk = min(buffer_size, want - got)
                    n = os.readv(fd, [view[:chunk]])
                    if n <= 0:
                        break
                    got += n
            finally:
                os.close(fd)
            with lock:
                state["bytes"] += got
                state["files"] += 1
                if got != want:
                    state["errors"].append(
                        f"{path}: read {got} of {want} at offset {offset}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return state


def delta(before: dict, after: dict, logical: int) -> dict:
    seconds = after["unix"] - before["unix"]
    hdd = sum(after["disk"][d][0] - before["disk"][d][0] for d in HDDS)
    nvme_read = sum(after["disk"][d][0] - before["disk"][d][0] for d in NVMES)
    nvme_write = sum(after["disk"][d][1] - before["disk"][d][1] for d in NVMES)
    arc = {k: after["arc"].get(k, 0) - before["arc"].get(k, 0) for k in ARC_KEYS}
    return {
        "seconds": round(seconds, 3),
        "logical_bytes": logical,
        "logical_mb_per_s": round(logical / 1e6 / seconds, 1) if seconds else None,
        "hdd_read_bytes": hdd,
        "hdd_read_mb_per_s": round(hdd / 1e6 / seconds, 1) if seconds else None,
        "nvme_read_bytes": nvme_read,
        "nvme_read_mb_per_s": round(nvme_read / 1e6 / seconds, 1) if seconds else None,
        "nvme_write_bytes": nvme_write,
        "nvme_write_mb_per_s": round(nvme_write / 1e6 / seconds, 1) if seconds else None,
        "supply_bytes": hdd + nvme_read,
        "supply_over_logical": round((hdd + nvme_read) / logical, 3) if logical else None,
        "nfsd_read_bytes": after["nfsd"][0] - before["nfsd"][0],
        "arc_delta": arc,
        "arc_after": {k: after["arc"].get(k) for k in ("size", "c", "c_min", "c_max", "l2_size", "l2_asize")},
        "loadavg_before": before["loadavg"],
        "loadavg_after": after["loadavg"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[],
                        help="a data manifest to read; repeat for several arms")
    parser.add_argument("--mount-map", required=True,
                        help="SHARED=LOCAL, so the read never leaves this box")
    parser.add_argument("--readers", type=int, default=8,
                        help="parallel readers inside one arm")
    parser.add_argument("--idle-s", type=float, default=0.0,
                        help="measure the box's own background traffic first, for this long")
    parser.add_argument("--settle-s", type=float, default=5.0,
                        help="pause between arms so their counters do not overlap")
    parser.add_argument("--label", default="",
                        help="name for this run in the output")
    parser.add_argument("--out", help="write the JSON here as well as to stdout")
    args = parser.parse_args(argv)

    if not args.manifest:
        raise SystemExit("name at least one --manifest")
    to_local = Local(args.mount_map)

    run = {
        "label": args.label,
        "host": os.uname().nodename,
        "readers": args.readers,
        "mount_map": args.mount_map,
        "started_unix": time.time(),
        "arms": [],
    }

    if args.idle_s > 0:
        before = sample()
        time.sleep(args.idle_s)
        after = sample()
        run["idle_baseline"] = delta(before, after, 1)
        run["idle_baseline"].pop("logical_mb_per_s", None)

    for path in args.manifest:
        with open(path) as handle:
            manifest = json.load(handle)
        entries = manifest["entries"]
        time.sleep(args.settle_s)
        before = sample()
        state = read_entries(entries, to_local, args.readers)
        after = sample()
        arm = {
            "manifest": path,
            "row_id": (manifest.get("annotations") or {}).get("row_id"),
            "entry_count": manifest["entry_count"],
            "declared_bytes": manifest["total_bytes"],
            "bytes_read": state["bytes"],
            "files_read": state["files"],
            "errors": state["errors"][:20],
            "error_count": len(state["errors"]),
        }
        arm.update(delta(before, after, state["bytes"]))
        run["arms"].append(arm)
        print(f"{arm['row_id']}: {arm['bytes_read']} B in {arm['seconds']} s "
              f"= {arm['logical_mb_per_s']} MB/s | HDD {arm['hdd_read_bytes']/1e9:.3f} GB "
              f"NVMe {arm['nvme_read_bytes']/1e9:.3f} GB "
              f"l2_hits {arm['arc_delta']['l2_hits']} l2_misses {arm['arc_delta']['l2_misses']}",
              flush=True)

    run["finished_unix"] = time.time()
    text = json.dumps(run, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
