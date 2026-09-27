"""Time one overlay fence re-hash over N stat-drifted wires (PQ #1531).

Builds ``--n`` wire files of ``--mib`` MiB, records each file's stat fence and
SHA-256, moves only ctime (a hard link added and dropped, the PQ #1495 case)
and then times ``joint_catalog_extension.streamed_fences`` proving every
wire by its recorded digest: the path the rooted selected-cache export runs,
with the same ``_rehash_drifted`` job ``attach_candidate_overlay`` runs.

It reports wall time, bytes/s, the threads that hashed and their peak
concurrency, and two single-thread references on the same files: sha256 over
bytes already in memory, and a plain read with no hash. Compare the fence's
bytes/s with those two to tell whether sha256 or the read bounds it.

``--cold`` drops each file's page cache before every repeat
(``POSIX_FADV_DONTNEED``), so the reads come from the device.

``--busy-consumer`` times a plain ``io_engine.read_stream`` over the same
wires whose consumer does CPU work of its own after each take: a sha256 over
an in-memory buffer as large as one wire, so it is busy about as long as one
read. It reports the stream's ``peak_workers`` and its consumer busy and wait
seconds, which show how many reads the engine ran beside a working consumer
(PQ #1533).

Run it through PrismaBuild (``pbrun --profile sample``) on each code version;
it builds no threads of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fadvise_drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _build(root, n, mib):
    root.mkdir(parents=True, exist_ok=True)
    block = os.urandom(1 << 20)
    wires = []
    for index in range(n):
        path = root / f"wire-{index:05d}.bin"
        digest = hashlib.sha256()
        with open(path, "wb") as handle:
            for chunk in range(mib):
                piece = index.to_bytes(8, "little") + chunk.to_bytes(8, "little") + block[16:]
                handle.write(piece)
                digest.update(piece)
        s = path.stat()
        wires.append((path, {"inode": s.st_ino, "bytes": s.st_size,
                             "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns},
                      digest.hexdigest()))
    time.sleep(0.01)
    for path, _, _ in wires:
        link = Path(str(path) + ".hl")
        os.link(path, link)
        os.unlink(link)
    return wires


class _DigestSpy:
    """Wrap ``hashlib.file_digest`` as the fence sees it: threads and peak concurrency."""

    def __init__(self, module):
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0
        self.threads = set()
        self._module, self._real = module, module.hashlib.file_digest

        def spy(handle, digest):
            with self.lock:
                self.active += 1
                self.calls += 1
                self.peak = max(self.peak, self.active)
                self.threads.add(threading.current_thread().name)
            try:
                return self._real(handle, digest)
            finally:
                with self.lock:
                    self.active -= 1

        module.hashlib.file_digest = spy

    def restore(self):
        self._module.hashlib.file_digest = self._real


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="scratch directory for the wires (removed after)")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--mib", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--busy-consumer", action="store_true")
    args = parser.parse_args(argv)

    from prismaquant import joint_catalog_extension as jce
    root = Path(args.dir) / f"bench-{os.getpid()}"
    wires = _build(root, args.n, args.mib)
    total = sum(recorded["bytes"] for _, recorded, _ in wires)
    report = {"n": args.n, "mib": args.mib, "total_bytes": total, "cold": args.cold,
              "affinity": len(os.sched_getaffinity(0)), "host": os.uname().nodename,
              "runs": []}
    try:
        # References, single thread, on the same files.
        blob = wires[0][0].read_bytes()
        started = time.perf_counter()
        for _ in range(4):
            hashlib.sha256(blob).hexdigest()
        report["sha256_in_memory_bytes_per_s"] = 4 * len(blob) / (time.perf_counter() - started)
        del blob
        if args.cold:
            for path, _, _ in wires:
                _fadvise_drop(path)
        started = time.perf_counter()
        for path, _, _ in wires:
            with open(path, "rb", buffering=0) as handle:
                while handle.read(8 << 20):
                    pass
        report["plain_read_bytes_per_s"] = total / (time.perf_counter() - started)

        engine_counters = []
        try:
            from prismaquant import io_engine
            real_stream = io_engine.read_stream

            def spy_stream(*a, **k):
                stream = real_stream(*a, **k)
                engine_counters.append(stream.counters)
                return stream
            io_engine.read_stream = spy_stream
        except ImportError:
            io_engine = None

        if args.busy_consumer:
            report["mode"] = "busy_consumer"
            _busy_consumer(report, wires, total, args)
            return _finish(report, root, total)
        report["mode"] = "fence"
        for repeat in range(args.repeats):
            if args.cold:
                for path, _, _ in wires:
                    _fadvise_drop(path)
            jce.FENCE_REHASHED.clear()
            engine_counters.clear()
            spy = _DigestSpy(jce)
            try:
                started = time.perf_counter()
                with jce.streamed_fences() as fences:
                    for path, recorded, digest in wires:
                        fences.check(path, path.stat(), recorded, digest, "bench wire fence")
                wall = time.perf_counter() - started
            finally:
                spy.restore()
            assert jce.FENCE_REHASHED == {"bench wire fence": len(wires)}, jce.FENCE_REHASHED
            assert spy.calls == len(wires), spy.calls
            run = {"repeat": repeat, "wall_s": wall, "bytes_per_s": total / wall,
                   "hash_threads": sorted({name.rsplit("_", 1)[0] for name in spy.threads}),
                   "distinct_hash_threads": len(spy.threads), "peak_concurrent_hashes": spy.peak}
            if engine_counters:
                c = engine_counters[-1]
                run["engine"] = {k: c.get(k) for k in ("pool_width", "peak_workers", "peak_workers_consumer_busy", "entries_read",
                                                         "consumer_wait_s", "per_stream_bytes_per_s")}
            report["runs"].append(run)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return _finish(report, root, total)


def _finish(report, root, total):
    shutil.rmtree(root, ignore_errors=True)
    walls = sorted(r["wall_s"] for r in report["runs"])
    report["median_wall_s"] = walls[len(walls) // 2]
    report["median_bytes_per_s"] = total / report["median_wall_s"]
    print("BENCH " + json.dumps(report, sort_keys=True))
    return 0


def _read_digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest(), None


def _busy_consumer(report, wires, total, args):
    """A stream whose consumer works: one wire-sized sha256 per take."""
    from functools import partial

    from prismaquant import io_engine
    work = bytes(wires[0][1]["bytes"])
    for repeat in range(args.repeats):
        if args.cold:
            for path, _, _ in wires:
                _fadvise_drop(path)
        entries = [io_engine.ReadEntry(key=i, path=None, size=recorded["bytes"],
                                       limit=recorded["bytes"], held_bytes=0,
                                       expected_sha256=None, decoder=None, group=i,
                                       reader=partial(_read_digest, path))
                   for i, (path, recorded, _) in enumerate(wires)]
        started = time.perf_counter()
        with io_engine.read_stream(entries, budget=io_engine.FixedBudget(buffer_bytes=1)) as stream:
            for i, (_, _, digest) in enumerate(wires):
                (delivered,) = stream.take(i)
                assert delivered.value == digest
                hashlib.sha256(work).digest()
            stream.release()
        wall = time.perf_counter() - started
        c = stream.counters
        report["runs"].append({"repeat": repeat, "wall_s": wall, "bytes_per_s": total / wall,
                               "engine": {k: c.get(k) for k in (
                                   "pool_width", "peak_workers", "peak_workers_consumer_busy", "consumer_busy_s",
                                   "consumer_wait_s", "per_stream_bytes_per_s")}})


if __name__ == "__main__":
    raise SystemExit(main())
