"""Staged exact-entry read bench: Stage B checkpoint-load and Stage A windows.

PQ #1142 measured the Stage B checkpoint-load at about 186 ms per 16 MiB
entry, most of it lease, cover and metadata work. The claim of any change to
that path is a delta, so this tool measures both sides on the same staged
bytes, in one admitted action, on one box, alternating the arms so a drift in
the tier cannot land on one of them.

Run it as a PrismaBuild action whose data manifest names the entries it reads
(``pbrun --data-manifest ... --residency stage``); the action's injected
reader context is what the strict tier policy leases through. The driver:

1. waits until every declared entry has landed on the tier, and lists the
   staged copies (a ``paths`` child);
2. measures the raw read ceiling of those copies (a ``ceiling`` child:
   plain threaded reads, no hashing, no lease);
3. runs each ``--arm NAME=TREE[:SDK_ROOT]`` once per repeat and workload.
   An arm imports ``prismaquant`` from ``TREE``; ``SDK_ROOT`` overrides the
   PrismaBuild reader SDK generation it leases through. Each child runs
   under ``py-spy record --threads``.

Before each timed child the driver drops this client's page cache for every
staged copy (``posix_fadvise(DONTNEED)``), so an arm measures the tier and
the link, not a warm client cache. It records the NFS per-operation counters
of the mounts before and after, and the wall-clock window for Netdata.

Workloads:

* ``stage-b``: ``load_adjoint_checkpoint`` of a whole checkpoint plane, the
  production call, into a sink (``--sink scratch`` is the production local
  cotangent scratch; ``discard`` counts and drops).
* ``stage-b-discard``: the same load into the discarding sink whatever
  ``--sink`` says, so the read rate is measured apart from the sink's writes.
* ``stage-a``: R13-shaped fused windows (``--stage-a-batches`` boundary
  entries plus each probe's cotangent of the same batches) through
  ``prefetch_exact_activation_cache_entries``, the Stage A seam.

Reads only. Writes its results, profiles and (``--sink scratch``) one
unlinked scratch file under ``--out`` and ``--scratch-root``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

MARKER = "STAGED-EXACT-READ-BENCH "
MOUNTS = ("/mnt/shared", "/stage/prewarm", "/ram/prewarm")
OPS = ("READ", "LOOKUP", "GETATTR", "ACCESS", "OPEN", "CLOSE", "WRITE",
       "COMMIT", "READDIR", "READDIRPLUS", "LOCK", "LOCKU", "SETATTR", "RENAME",
       "REMOVE", "CREATE", "OPEN_NOATTR", "DELEGRETURN", "TEST_STATEID",
       "LAYOUTGET", "SEQUENCE")


# -- host instruments --------------------------------------------------------

def mountstats() -> dict:
    """``{mount: {"bytes": [...], "ops": {OP: [ops, trans, ..., execute_ms]}}}``."""
    out, cur = {}, None
    try:
        lines = open("/proc/self/mountstats").read().splitlines()
    except OSError:
        return out
    for line in lines:
        if line.startswith("device "):
            parts = line.split()
            where = parts[parts.index("on") + 1] if "on" in parts else None
            cur = where if where in MOUNTS else None
            if cur is not None:
                out.setdefault(cur, {"bytes": None, "ops": {}})
            continue
        if cur is None:
            continue
        text = line.strip()
        if text.startswith("bytes:"):
            out[cur]["bytes"] = [int(x) for x in text.split()[1:]]
        name, sep, rest = text.partition(":")
        if sep and name in OPS:
            fields = rest.split()
            if len(fields) >= 8 and all(f.isdigit() for f in fields[:8]):
                out[cur]["ops"][name] = [int(f) for f in fields[:9]]
    return out


def mountstats_delta(before: dict, after: dict) -> dict:
    """Per mount: bytes read by the client and, per op, count and times."""
    delta = {}
    for mount, row in after.items():
        old = before.get(mount, {"bytes": None, "ops": {}})
        entry = {}
        if row["bytes"] and old["bytes"]:
            entry["client_read_bytes"] = row["bytes"][0] - old["bytes"][0]
            entry["server_read_bytes"] = row["bytes"][4] - old["bytes"][4]
        ops = {}
        for op, fields in row["ops"].items():
            prior = old["ops"].get(op, [0] * len(fields))
            count = fields[0] - prior[0]
            if count:
                # ops, trans, timeouts, bytes_sent, bytes_recv,
                # queue_ms, rtt_ms, execute_ms[, errors]
                ops[op] = {"ops": count,
                           "queue_ms": fields[5] - prior[5],
                           "rtt_ms": fields[6] - prior[6],
                           "execute_ms": fields[7] - prior[7]}
        entry["ops"] = ops
        delta[mount] = entry
    return delta


def proc_io() -> dict:
    out = {}
    try:
        for line in open("/proc/self/io"):
            key, _, value = line.partition(":")
            out[key.strip()] = int(value)
    except OSError:
        pass
    return out


def drop_client_cache(paths) -> int:
    """Forget what this client cached for ``paths``; returns files advised."""
    done = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            done += 1
        finally:
            os.close(fd)
    return done


# -- inputs -------------------------------------------------------------------

def load_slice(path: str, sha256: str) -> dict:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise SystemExit(f"slice digest {digest} is not --slice-sha256 {sha256}")
    return json.loads(raw)


def stage_a_windows(slice_doc: dict, *, boundary: str, batches: int,
                    windows: int) -> list[list[dict]]:
    """R13 fused windows: boundary entries, then each probe's, per batch group."""
    bounds = {int(r["metadata"]["identity"]["coordinates"]["batch"]): r
              for r in slice_doc["boundary_entries"][boundary]}
    plane = {}
    for row in slice_doc["checkpoint"]["activation_entries"]:
        coords = row["metadata"]["identity"]["coordinates"]
        plane[int(coords["probe"]), int(coords["batch"])] = row
    probes = sorted({probe for probe, _ in plane})
    out = []
    for index in range(windows):
        group = range(index * batches, (index + 1) * batches)
        rows = [bounds[b] for b in group]
        rows += [plane[p, b] for p in probes for b in group]
        out.append(rows)
    return out


# -- child side ---------------------------------------------------------------

def _bind(manifest_sha256: str, tiers: str):
    from prismaquant.residency_map import bind_residency_manifest, residency_resolver
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    activate_staged_tier_policy(tiers)
    bind_residency_manifest(manifest_sha256)
    resolver = residency_resolver()
    if resolver is None:
        raise SystemExit("no residency map in this action's environment")
    return resolver


def _all_records(slice_doc: dict, space: str) -> list[dict]:
    """Every file the two workloads read, as ``{path, file_bytes}`` rows."""
    checkpoint = slice_doc["checkpoint"]
    rows = [{"path": str(Path(space) / "checkpoints"
                         / f"boundary-{int(checkpoint['boundary']):03d}"
                         / "checkpoint.json"), "file_bytes": None}]
    rows += checkpoint["activation_entries"] + checkpoint["shared_state_entries"]
    for entries in slice_doc["boundary_entries"].values():
        rows += entries
    return rows


def child_paths(args, slice_doc) -> dict:
    """Wait for every declared entry to land; list the staged copies."""
    from prismaquant.residency_shard_reader import await_staged_spans
    from prismaquant.staged_lease import stage_covers_are_published
    resolver = _bind(args.manifest_sha256, args.allowed_tiers)
    rows = _all_records(slice_doc, args.space)
    wanted = []
    for row in rows:
        size = row["file_bytes"] if row["file_bytes"] is not None else \
            os.lstat(row["path"]).st_size
        wanted.append((row["path"], 0, size, size))
    started = time.monotonic()
    verdict = await_staged_spans(resolver, wanted, deadline=started + args.land_wait_s,
                                 published_batch=stage_covers_are_published)
    waited = time.monotonic() - started
    # The ram leg promotes each landed stage range afterwards; wait for it
    # too, so the first arm does not read a tier the later arms do not.
    ram_deadline = time.monotonic() + args.ram_wait_s
    while True:
        copies, ram = [], 0
        for row in rows:
            staged = resolver.staged_read(Path(row["path"]))
            if staged is None:
                continue
            copies.append({"declared": row["path"],
                           "stage_path": staged.get("stage_path"),
                           "ram_path": staged.get("ram_path"),
                           "bytes": staged["bytes"]})
            ram += staged.get("ram_path") is not None
        if ram == len(copies) or time.monotonic() >= ram_deadline:
            break
        time.sleep(10.0)
    return {"verdict": verdict, "land_wait_s": round(waited, 3),
            "ram_wait_s": round(time.monotonic() - started - waited, 3),
            "files": len(rows), "staged": len(copies), "ram_offered": ram,
            "copies": copies}


def child_ceiling(args, _slice_doc) -> dict:
    """Raw threaded reads of the staged copies: no hash, no lease, no load.

    Two modes per thread count. ``buffered`` reads through the page cache in
    4 MiB calls, as the reader does, so the mount's readahead window bounds
    how many READs one stream keeps in flight. ``direct`` opens with
    ``O_DIRECT`` and reads a whole 16 MiB block per call, so the client issues
    every READ of the block at once: the rate the link and the tier give when
    no readahead window stands between them and the reader.
    """
    import mmap
    copies = json.loads(Path(args.copies).read_text())
    results = []
    for tier in ("ram", "stage"):
        paths = [c["ram_path"] if tier == "ram" else c["stage_path"] for c in copies
                 if (c["ram_path"] if tier == "ram" else c["stage_path"])
                 and c["bytes"] > (1 << 20)][args.ceiling_skip:args.ceiling_skip + args.ceiling_files]
        if not paths:
            continue
        for mode in args.ceiling_modes:
            for threads in args.ceiling_threads:
                drop_client_cache(paths)
                before = mountstats()
                lock = threading.Lock()
                cursor = [0]
                total = [0]
                errors = []

                def work():
                    block = (1 << 24) if mode == "direct" else (1 << 22)
                    buf = mmap.mmap(-1, block)
                    view = memoryview(buf)
                    flags = os.O_RDONLY | os.O_CLOEXEC
                    if mode == "direct":
                        flags |= os.O_DIRECT
                    try:
                        while True:
                            with lock:
                                if cursor[0] >= len(paths) or errors:
                                    return
                                path = paths[cursor[0]]
                                cursor[0] += 1
                            fd = os.open(path, flags)
                            try:
                                while True:
                                    got = os.readv(fd, [view])
                                    if not got:
                                        break
                                    with lock:
                                        total[0] += got
                            finally:
                                os.close(fd)
                    except OSError as exc:
                        with lock:
                            errors.append(f"{type(exc).__name__}: {exc}")
                    finally:
                        view.release()
                        buf.close()

                started = time.monotonic()
                pool = [threading.Thread(target=work) for _ in range(threads)]
                for thread in pool:
                    thread.start()
                for thread in pool:
                    thread.join()
                wall = time.monotonic() - started
                results.append({"tier": tier, "mode": mode, "threads": threads,
                                "files": len(paths), "bytes": total[0],
                                "wall_s": round(wall, 4),
                                "gb_s": round(total[0] / wall / 1e9, 3),
                                "errors": errors[:3],
                                "nfs": mountstats_delta(before, mountstats())})
    return {"ceiling": results}


def child_sink_ceiling(args, _slice_doc) -> dict:
    """Sequential writes to the Stage B scratch file system: no reads, no torch.

    ``direct`` writes 16 MiB blocks with ``O_DIRECT`` from one aligned buffer
    per thread, so no page cache stands between the writer and the device:
    the rate a scratch writer can reach on this disk. ``sync-each`` writes the
    same blocks buffered and runs ``fdatasync`` and a whole-file
    ``DONTNEED`` after each one, which is what the scratch did per entry.
    """
    import mmap
    import tempfile
    block = 1 << 24
    total = args.sink_ceiling_bytes // block * block
    pattern = os.urandom(block)
    results = []
    for mode, threads in (("direct", 1), ("direct", 4), ("sync-each", 1)):
        with tempfile.TemporaryFile(dir=args.scratch_root) as handle:
            os.posix_fallocate(handle.fileno(), 0, total)
            os.fdatasync(handle.fileno())
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            lock = threading.Lock()
            cursor = [0]
            errors = []

            def work():
                buf = mmap.mmap(-1, block)
                buf[:] = pattern
                view = memoryview(buf)
                flags = os.O_WRONLY | os.O_CLOEXEC | (os.O_DIRECT if mode == "direct" else 0)
                fd = os.open(f"/proc/self/fd/{handle.fileno()}", flags)
                try:
                    while True:
                        with lock:
                            if cursor[0] >= total or errors:
                                return
                            offset = cursor[0]
                            cursor[0] += block
                        done = 0
                        while done < block:
                            done += os.pwritev(fd, [view[done:]], offset + done)
                        if mode == "sync-each":
                            os.fdatasync(fd)
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError as exc:
                    with lock:
                        errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    os.close(fd)
                    view.release()
                    buf.close()

            memcg = MemcgPeaks()
            started = time.monotonic()
            with memcg:
                pool = [threading.Thread(target=work) for _ in range(threads)]
                for thread in pool:
                    thread.start()
                for thread in pool:
                    thread.join()
                os.fdatasync(handle.fileno())
            wall = time.monotonic() - started
            results.append({"mode": mode, "threads": threads, "bytes": total,
                            "block_bytes": block, "wall_s": round(wall, 4),
                            "gb_s": round(total / wall / 1e9, 3),
                            "errors": errors[:3], "memcg": memcg.report()})
    return {"sink_ceiling": results, "scratch_root": args.scratch_root,
            "fs": _statfs_type(args.scratch_root)}


def _statfs_type(path):
    try:
        best = ("", "")
        for line in Path("/proc/self/mounts").read_text().splitlines():
            parts = line.split()
            mount = parts[1]
            if (str(Path(path).resolve()) + "/").startswith(mount.rstrip("/") + "/") \
                    and len(mount) > len(best[0]):
                best = (mount, parts[2] + " " + parts[0])
        return {"mount": best[0], "type_device": best[1]}
    except OSError:
        return None


class _DiscardSink(dict):
    """A cotangent factory result that keeps nothing but a count."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.bytes = 0

    def __setitem__(self, key, tensor):
        self.count += 1
        self.bytes += tensor.numel() * tensor.element_size()


def _call_supported(function, *args, **kwargs):
    """Pass only the keyword arguments this tree's ``function`` takes."""
    accepted = inspect.signature(function).parameters
    return function(*args, **{k: v for k, v in kwargs.items() if k in accepted})


class MemcgPeaks:
    """Sample this process's cgroup ``memory.stat`` and keep each field's peak.

    The file charge (``file``, ``file_dirty``, ``file_writeback``) is what a
    buffered scratch writer adds to the cgroup on top of its anon tensors.
    """

    FIELDS = ("anon", "file", "file_dirty", "file_writeback", "active_file",
              "inactive_file")

    def __init__(self, period_s=0.02):
        self.period_s = period_s
        self.path = None
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    self.path = Path("/sys/fs/cgroup") / line[3:].lstrip("/") / "memory.stat"
        except OSError:
            self.path = None
        self.peaks = {field: 0 for field in self.FIELDS}
        self.first = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None

    def _read(self):
        values = {}
        for line in self.path.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in self.FIELDS:
                values[key] = int(value)
        return values

    def __enter__(self):
        if self.path is None or not self.path.exists():
            return self
        self.first = self._read()

        def run():
            while not self._stop.wait(self.period_s):
                try:
                    values = self._read()
                except OSError:
                    continue
                self.samples += 1
                for key, value in values.items():
                    self.peaks[key] = max(self.peaks[key], value)

        self._thread = threading.Thread(target=run, name="memcg-peaks", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False

    def report(self):
        return {"cgroup_memory_stat": str(self.path) if self.path else None,
                "samples": self.samples, "period_s": self.period_s,
                "start": self.first, "peak": self.peaks}


def _sink_variant(base, variant, *, window_bytes):
    """A candidate cotangent scratch writer (PQ #1152), measured on the same slots.

    ``current`` is the tree's own class. The others keep its slots, file and
    checks and change only how written bytes leave the page cache:

    * ``window``: buffered writes; one ``fdatasync`` and one ranged
      ``DONTNEED`` once the unsynced range reaches ``window_bytes``.
    * ``writebehind``: buffered writes; ``sync_file_range(WRITE)`` starts each
      entry's writeback at once, and entries more than ``window_bytes`` behind
      are waited on and dropped by range.
    * ``direct``: ``O_DIRECT`` through one aligned bounce buffer, so no page
      cache holds scratch bytes at all.
    """
    if variant == "current":
        return base
    import ctypes
    import mmap
    import torch
    libc = ctypes.CDLL(None, use_errno=True)
    sync_file_range = libc.sync_file_range
    sync_file_range.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint]
    wait_before, start_write, wait_after = 1, 2, 4

    def sfr(fd, offset, size, flags):
        if sync_file_range(fd, offset, size, flags) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def drop(fd, offset, size):
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)

    def checked_view(self, key, tensor):
        if self._file is None:
            raise RuntimeError("cotangent scratch is closed")
        offset, size, shape, dtype = self._slots[key]
        if (tensor.device.type != 'cpu' or tensor.dtype != dtype
                or tuple(tensor.shape) != shape):
            raise ValueError("cotangent scratch rollover changed shape/dtype")
        self._written.discard(key)
        compact = tensor.detach().contiguous()
        return offset, size, memoryview(compact.view(torch.uint8).reshape(-1).numpy())

    def write_all(fd, view, offset, size):
        done = 0
        while done < size:
            put = os.pwritev(fd, [view[done:]], offset + done)
            if put <= 0:
                raise RuntimeError("cotangent scratch short write")
            done += put

    def read_slot(self, key, fd, *, drop_range):
        if self._file is None or key not in self._written:
            raise RuntimeError("cotangent scratch slot is not ready")
        offset, size, shape, dtype = self._slots[key]
        tensor = torch.empty(shape, dtype=dtype, device='cpu')
        view = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
        try:
            done = 0
            while done < size:
                got = os.preadv(fd, [view[done:]], offset + done)
                if got <= 0:
                    raise RuntimeError("cotangent scratch slot is truncated")
                done += got
            if drop_range:
                drop(fd, offset, size)
        finally:
            view.release()
        return tensor

    if variant == "window":
        class Window(base):
            _unsynced = None

            def _flush(self):
                if self._unsynced:
                    lo, hi, keys = self._unsynced
                    self._unsynced = None
                    fd = self._file.fileno()
                    try:
                        os.fdatasync(fd)
                        drop(fd, lo, hi - lo)
                    except BaseException:
                        self._written.difference_update(keys)
                        raise

            def __setitem__(self, key, tensor):
                offset, size, view = checked_view(self, key, tensor)
                try:
                    write_all(self._file.fileno(), view, offset, size)
                finally:
                    view.release()
                lo, hi, keys = self._unsynced or (offset, offset + size, [])
                self._unsynced = (min(lo, offset), max(hi, offset + size), keys + [key])
                self._written.add(key)
                if self._unsynced[1] - self._unsynced[0] >= window_bytes:
                    self._flush()

            def __getitem__(self, key):
                # Dirty pages read back from the cache; only synced slots are dropped.
                offset, size = self._slots[key][:2]
                pending = self._unsynced
                inside = pending and pending[0] < offset + size and offset < pending[1]
                return read_slot(self, key, self._file.fileno(), drop_range=not inside)
        return Window

    if variant == "writebehind":
        class WriteBehind(base):
            def _pending(self):
                if not hasattr(self, "_behind"):
                    self._behind = []
                return self._behind

            def _retire(self, keep_bytes):
                pending, fd = self._pending(), self._file.fileno()
                while pending and sum(size for _o, size, _k in pending) > keep_bytes:
                    offset, size, key = pending.pop(0)
                    try:
                        sfr(fd, offset, size, wait_before | start_write | wait_after)
                        drop(fd, offset, size)
                    except BaseException:
                        self._written.discard(key)
                        raise

            def __setitem__(self, key, tensor):
                offset, size, view = checked_view(self, key, tensor)
                fd = self._file.fileno()
                try:
                    write_all(fd, view, offset, size)
                finally:
                    view.release()
                sfr(fd, offset, size, start_write)
                self._pending().append((offset, size, key))
                self._written.add(key)
                self._retire(window_bytes)

            def __getitem__(self, key):
                offset, size = self._slots[key][:2]
                inside = any(o == offset for o, _s, _k in self._pending())
                return read_slot(self, key, self._file.fileno(), drop_range=not inside)
        return WriteBehind

    if variant in ("direct", "direct-zc"):
        zero_copy = variant == "direct-zc"

        class Direct(base):
            copies = 0
            direct_calls = 0

            def _direct(self):
                if getattr(self, "_dfd", None) is None:
                    from prismaquant.perturbed_x_cache import _direct_io_block
                    self._dfd = os.open(f"/proc/self/fd/{self._file.fileno()}",
                                        os.O_RDWR | os.O_DIRECT | os.O_CLOEXEC)
                    self._grid = _direct_io_block(self._dfd)
                    self._bounce = mmap.mmap(-1, self.max_slot_bytes)
                    self._bounce_view = memoryview(self._bounce)
                return self._dfd

            def _on_grid(self, view):
                import ctypes
                address = ctypes.addressof(ctypes.c_char.from_buffer(view)) \
                    if not view.readonly else None
                return address is not None and address % self._grid == 0

            def __setitem__(self, key, tensor):
                offset, size, view = checked_view(self, key, tensor)
                fd = self._direct()
                if offset % self._grid or size % self._grid:
                    raise RuntimeError("direct scratch bench needs grid-sized slots")
                try:
                    if zero_copy and self._on_grid(view):
                        write_all(fd, view, offset, size)
                    else:
                        self._bounce_view[:size] = view
                        type(self).copies += 1
                        write_all(fd, self._bounce_view, offset, size)
                finally:
                    view.release()
                type(self).direct_calls += 1
                self._written.add(key)

            def __getitem__(self, key):
                if self._file is None or key not in self._written:
                    raise RuntimeError("cotangent scratch slot is not ready")
                offset, size, shape, dtype = self._slots[key]
                fd = self._direct()
                tensor = torch.empty(shape, dtype=dtype, device='cpu')
                out = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
                try:
                    target = out if zero_copy and self._on_grid(out) else self._bounce_view
                    done = 0
                    while done < size:
                        got = os.preadv(fd, [target[done:size]], offset + done)
                        if got <= 0:
                            raise RuntimeError("cotangent scratch slot is truncated")
                        done += got
                    if target is not out:
                        out[:] = self._bounce_view[:size]
                        type(self).copies += 1
                finally:
                    out.release()
                return tensor

            def close(self):
                if getattr(self, "_dfd", None) is not None:
                    self._bounce_view.release()
                    self._bounce.close()
                    os.close(self._dfd)
                    self._dfd = None
                super().close()
        return Direct
    raise SystemExit(f"unknown sink variant {variant!r}")


def child_stage_b(args, slice_doc) -> dict:
    from prismaquant.joint_adjoint_checkpoints import load_adjoint_checkpoint
    resolver = _bind(args.manifest_sha256, args.allowed_tiers)
    record = slice_doc["checkpoint"]
    policy = slice_doc["boundary_storage"]["policy"]
    arena = None
    sink = None

    discard = args.sink == "discard" or args.child == "stage-b-discard"
    variant = os.environ.get("BENCH_SINK_VARIANT", "current")
    readback = {}

    def factory(rows):
        nonlocal arena, sink
        if discard:
            sink = _DiscardSink()
            return sink
        from prismaquant.joint_adjoint_checkpoints import exact_entry_windows
        from prismaquant.perturbed_x_cache import ExactCotangentScratch
        windows, _ahead = exact_entry_windows(
            rows, max_resident_bytes=policy["max_resident_bytes"])
        window_bytes = max(sum(int(r["tensor_bytes"]) for r in w) for w in windows)
        scratch_class = _sink_variant(ExactCotangentScratch, variant,
                                      window_bytes=window_bytes)
        arena = scratch_class(
            rows, directory=args.scratch_root, max_bytes=args.scratch_max_bytes,
            max_tensor_bytes=policy["max_resident_bytes"])
        return arena

    resident = {"now": 0, "peak": 0}

    def residency_check(delta):
        resident["now"] += delta
        resident["peak"] = max(resident["peak"], resident["now"])
        if resident["now"] > policy["max_resident_bytes"] or resident["now"] < 0:
            raise RuntimeError("bench residency budget exceeded")

    io0 = proc_io()
    memcg = MemcgPeaks()
    started = time.monotonic()
    try:
      with memcg:
        cotangents, _adjoint, _pass = _call_supported(
            load_adjoint_checkpoint, args.space, record, cotangent_factory=factory,
            shared_state_max_bytes=policy["max_auxiliary_bytes"],
            max_resident_bytes=policy["max_resident_bytes"],
            residency_check=residency_check)
        wall = time.monotonic() - started
        entries = len(record["activation_entries"])
        tensor_bytes = sum(int(r["tensor_bytes"]) for r in record["activation_entries"])
        file_bytes = sum(int(r["file_bytes"]) for r in record["activation_entries"])
        got = len(cotangents) if sink is None else sink.count
        if got != entries:
            raise RuntimeError(f"stage-b read {got} of {entries} entries")
        if arena is not None:
            readback_mode = _scratch_mode(arena)
            # Every slot read back once, in plane order, as Stage B's renders do.
            back = time.monotonic()
            nbytes = 0
            for key in list(arena):
                tensor = arena[key]
                nbytes += tensor.numel() * tensor.element_size()
                del tensor
            readback = {"slots": len(arena), "bytes": nbytes,
                        "wall_s": round(time.monotonic() - back, 4),
                        "scratch_mode": readback_mode}
            readback["mb_s"] = round(nbytes / max(readback["wall_s"], 1e-9) / 1e6, 1)
    finally:
        if arena is not None:
            arena.close()
    return {"entries": entries, "file_bytes": file_bytes, "tensor_bytes": tensor_bytes,
            "wall_s": round(wall, 4), "entries_per_s": round(entries / wall, 3),
            "mb_s": round(file_bytes / wall / 1e6, 1),
            "resident_peak_bytes": resident["peak"], "io": _io_delta(io0, proc_io()),
            "counters": _counters(), "residency": _residency(resolver),
            "sink": "discard" if discard else variant, "readback": readback,
            "memcg": memcg.report()}


def _scratch_mode(arena):
    """Which path a scratch took: the tree's ``_direct`` grid when it has one."""
    direct = getattr(arena, "_direct", "absent")
    return {"direct": list(direct) if isinstance(direct, tuple) else direct,
            "class": type(arena).__name__}


def child_sink_feed(args, slice_doc) -> dict:
    """The cotangent scratch alone, fed from memory: no reads, no link.

    The Stage B plane's slots (names, shapes, dtypes and order from the
    slice) are written from a small pool of distinct in-memory tensors, as
    the checkpoint load writes them; then one chain roll reads and rewrites
    every slot, as ``render_free_layer_roll`` does per chain layer; then
    every slot is read back and compared with the tensor it was written
    from. ``BENCH_SINK_VARIANT`` picks the writer (see ``_sink_variant``);
    ``memcg`` holds the cgroup's peak file, dirty and writeback bytes.
    """
    import torch
    from prismaquant.joint_adjoint_checkpoints import exact_entry_windows
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    variant = os.environ.get("BENCH_SINK_VARIANT", "current")
    policy = slice_doc["boundary_storage"]["policy"]
    rows = []
    for entry in slice_doc["checkpoint"]["activation_entries"]:
        name = re.sub(r"-at-[0-9]+$", "", entry["name"])
        rows.append({"name": name, "shape": list(entry["shape"]),
                     "dtype": entry["dtype"], "tensor_bytes": int(entry["tensor_bytes"])})
    windows, _ahead = exact_entry_windows(rows, max_resident_bytes=policy["max_resident_bytes"])
    window_bytes = max(sum(r["tensor_bytes"] for r in w) for w in windows)
    scratch_class = _sink_variant(ExactCotangentScratch, variant, window_bytes=window_bytes)
    shape = tuple(rows[0]["shape"])
    dtype = getattr(torch, rows[0]["dtype"].removeprefix("torch."))
    generator = torch.Generator().manual_seed(1152)
    pool = [torch.randn(shape, generator=generator).to(dtype) for _ in range(8)]
    keys = [tuple(int(x) for x in r["name"].split("-")[1:]) for r in rows]
    total = sum(r["tensor_bytes"] for r in rows)
    phases = {}
    arena = scratch_class(rows, directory=args.scratch_root,
                          max_bytes=args.scratch_max_bytes,
                          max_tensor_bytes=policy["max_resident_bytes"])
    mode = _scratch_mode(arena)
    memcg_all = MemcgPeaks()
    try:
        with memcg_all:
            for phase in ("load", "roll", "readback"):
                memcg = MemcgPeaks()
                started = time.monotonic()
                mismatches = 0
                with memcg:
                    for index, key in enumerate(keys):
                        if phase == "load":
                            arena[key] = pool[index % len(pool)]
                        elif phase == "roll":
                            tensor = arena[key]
                            del tensor
                            arena[key] = pool[(index + 1) % len(pool)]
                        else:
                            tensor = arena[key]
                            if not torch.equal(tensor, pool[(index + 1) % len(pool)]):
                                mismatches += 1
                            del tensor
                wall = time.monotonic() - started
                moved = total * (2 if phase == "roll" else 1)
                phases[phase] = {"wall_s": round(wall, 4),
                                 "entries_per_s": round(len(keys) / wall, 2),
                                 "mb_s": round(moved / wall / 1e6, 1),
                                 "mismatches": mismatches, "memcg": memcg.report()}
    finally:
        arena.close()
    extra = {k: getattr(scratch_class, k) for k in ("copies", "direct_calls")
             if hasattr(scratch_class, k)}
    load = phases["load"]
    return {"sink": variant, "entries": len(keys), "tensor_bytes": total,
            "window_bytes": window_bytes, "wall_s": load["wall_s"],
            "entries_per_s": load["entries_per_s"], "mb_s": load["mb_s"],
            "phases": phases, "memcg": memcg_all.report(), "variant_counters": extra,
            "scratch_mode": mode,
            "scratch_fs": _statfs_type(args.scratch_root)}


def child_stage_a(args, slice_doc) -> dict:
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    from prismaquant.perturbed_x_cache import (
        EntryReadScratch, prefetch_exact_activation_cache_entries)
    resolver = _bind(args.manifest_sha256, args.allowed_tiers)
    policy = slice_doc["boundary_storage"]["policy"]
    windows = stage_a_windows(slice_doc, boundary=args.stage_a_boundary,
                              batches=args.stage_a_batches, windows=args.stage_a_windows)
    session = windows[0][0]["metadata"]["identity"]["session"]
    scratch = EntryReadScratch()
    io0 = proc_io()
    started = time.monotonic()
    entries = file_bytes = 0
    per_window = []
    for rows in windows:
        refs = [reference_from_record(row) for row in rows]
        opened = time.monotonic()
        with prefetch_exact_activation_cache_entries(
                refs, max_tensor_bytes=policy["max_resident_bytes"],
                expected_session=session, scratch=scratch) as window:
            for ref in refs:
                if window.get(ref).numel() == 0:
                    raise RuntimeError("empty exact tensor")
        per_window.append(round(time.monotonic() - opened, 4))
        entries += len(refs)
        file_bytes += sum(ref.file_bytes for ref in refs)
    wall = time.monotonic() - started
    scratch.release()
    return {"entries": entries, "file_bytes": file_bytes, "windows": len(windows),
            "window_entries": len(windows[0]), "wall_s": round(wall, 4),
            "entries_per_s": round(entries / wall, 3),
            "mb_s": round(file_bytes / wall / 1e6, 1), "window_s": per_window,
            "io": _io_delta(io0, proc_io()), "counters": _counters(),
            "residency": _residency(resolver)}


def _io_delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def _counters():
    try:
        from prismaquant.perturbed_x_cache import exact_lease_counters
    except ImportError:
        return None
    return exact_lease_counters()


def _residency(resolver):
    report = resolver.report()
    keep = ("bytes_from_pool", "bytes_from_stage", "bytes_from_ram", "hits", "misses",
            "fallbacks", "ram_fallbacks")
    out = {key: report.get(key) for key in keep if key in report}
    tiers = {}
    for row in report.get("serving_tiers") or ():
        tiers[row.get("tier")] = tiers.get(row.get("tier"), 0) + 1
    out["serving_tier_rows"] = tiers
    return out


LOCAL_WORKLOADS = frozenset({"sink-feed"})

CHILDREN = {"paths": child_paths, "ceiling": child_ceiling,
            "sink-ceiling": child_sink_ceiling, "sink-feed": child_sink_feed,
            "stage-b": child_stage_b, "stage-b-discard": child_stage_b,
            "stage-a": child_stage_a}


def child_main(args) -> int:
    slice_doc = load_slice(args.slice, args.slice_sha256)
    result = CHILDREN[args.child](args, slice_doc)
    result["tree"] = str(Path(__import__("prismaquant").__file__).resolve().parents[1])
    result["pid"] = os.getpid()
    print(MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


# -- driver side --------------------------------------------------------------

def _git_head(tree: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_child(args, mode: str, *, tree: str, sdk_root: str | None, out: Path,
              label: str, profile: bool, extra=(), arm_env=None) -> dict:
    env = dict(os.environ)
    env.update(arm_env or {})
    env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"]
                                if env.get("PYTHONPATH") else "")
    if sdk_root:
        env["PRISMABUILD_READER_HELPER_ROOT"] = sdk_root
    command = [sys.executable, str(Path(__file__).resolve()), "--child", mode,
               *args.child_argv, *extra]
    if profile:
        command = [args.py_spy, "record", "--threads", "--idle", "--rate",
                   str(args.py_spy_rate), "--format", "raw", "--output",
                   str(out / f"{label}.pyspy.raw"), "--", *command]
    before = mountstats()
    started_unix = time.time()
    started = time.monotonic()
    done = subprocess.run(command, env=env, capture_output=True, text=True,
                          cwd=tree)
    wall = time.monotonic() - started
    after = mountstats()
    (out / f"{label}.stdout").write_text(done.stdout)
    (out / f"{label}.stderr").write_text(done.stderr)
    result = None
    for line in done.stdout.splitlines():
        if line.startswith(MARKER):
            result = json.loads(line[len(MARKER):])
    if done.returncode != 0 or result is None:
        raise SystemExit(f"{label}: child exited {done.returncode}; see {out}/{label}.stderr")
    result.update({"label": label, "mode": mode, "sdk_root": sdk_root,
                   "arm_env": dict(arm_env or {}),
                   "started_unix": round(started_unix, 3),
                   "ended_unix": round(time.time(), 3),
                   "child_wall_s": round(wall, 3),
                   "nfs": mountstats_delta(before, after)})
    return result


def driver_main(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        tree, _, sdk = rest.partition(":")
        arms.append({"name": name, "tree": str(Path(tree).resolve()),
                     "sdk_root": sdk or None, "head": _git_head(tree), "env": {}})
    for spec in args.arm_env:
        name, _, setting = spec.partition("=")
        key, _, value = setting.partition("=")
        matched = [arm for arm in arms if arm["name"] == name]
        if not matched or not key:
            raise SystemExit(f"--arm-env {spec!r} names no arm or no variable")
        matched[0]["env"][key] = value
    own_tree = str(Path(__file__).resolve().parents[1])
    report = {"schema": "prismaquant.staged_exact_read_bench.v1",
              "host": os.uname().nodename, "action_key": os.environ.get(
                  "PRISMABUILD_ACTION_KEY"), "arms": arms, "argv": sys.argv,
              "runs": []}
    # Local workloads read nothing staged, so they wait for no landing.
    staged = [w for w in args.workloads if w not in LOCAL_WORKLOADS]
    staged_files = []
    if staged or args.ceiling_threads:
        paths = run_child(args, "paths", tree=own_tree, sdk_root=None, out=out,
                          label="paths", profile=False)
        copies = paths.pop("copies")
        (out / "copies.json").write_text(json.dumps(copies))
        report["paths"] = paths
        if paths["verdict"] != "hit":
            print(f"warning: landing verdict {paths['verdict']}", flush=True)
        staged_files = [p for c in copies for p in (c["stage_path"], c["ram_path"]) if p]
    if args.ceiling_threads:
        report["ceiling"] = run_child(
            args, "ceiling", tree=own_tree, sdk_root=None, out=out, label="ceiling",
            profile=False, extra=["--copies", str(out / "copies.json")])["ceiling"]
        (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if args.sink_ceiling_bytes:
        started_unix = time.time()
        report["sink_ceiling"] = run_child(
            args, "sink-ceiling", tree=own_tree, sdk_root=None, out=out,
            label="sink-ceiling", profile=False)
        report["sink_ceiling"]["started_unix"] = round(started_unix, 3)
        (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    for repeat in range(args.repeats):
        for workload in args.workloads:
            order = arms if repeat % 2 == 0 else list(reversed(arms))
            for arm in order:
                advised = drop_client_cache(staged_files) if staged_files else 0
                label = f"r{repeat}-{workload}-{arm['name']}"
                result = run_child(args, workload, tree=arm["tree"],
                                   sdk_root=arm["sdk_root"], out=out, label=label,
                                   profile=args.py_spy is not None,
                                   arm_env=arm["env"])
                result.update({"arm": arm["name"], "repeat": repeat,
                               "workload": workload, "advised_files": advised})
                report["runs"].append(result)
                print(json.dumps({k: result.get(k) for k in (
                    "label", "entries", "wall_s", "entries_per_s", "mb_s")}), flush=True)
                (out / "report.json").write_text(
                    json.dumps(report, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--child", choices=sorted(CHILDREN))
    parser.add_argument("--slice", required=True)
    parser.add_argument("--slice-sha256", required=True)
    parser.add_argument("--space", required=True,
                        help="the adjoint space the checkpoint lives under")
    parser.add_argument("--manifest-sha256", required=True,
                        help="the data manifest this action was submitted with")
    parser.add_argument("--allowed-tiers", default="ram,ssd")
    parser.add_argument("--land-wait-s", type=float, default=3600.0)
    parser.add_argument("--ram-wait-s", type=float, default=1800.0)
    parser.add_argument("--sink", choices=("scratch", "discard"), default="scratch")
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--scratch-max-bytes", type=int, default=64 << 30)
    parser.add_argument("--sink-ceiling-bytes", type=int, default=0,
                        help="write this many bytes per mode to the scratch root "
                             "to measure its sequential write rate; 0 skips it")
    parser.add_argument("--stage-a-boundary", default="44")
    parser.add_argument("--stage-a-batches", type=int, default=16)
    parser.add_argument("--stage-a-windows", type=int, default=8)
    parser.add_argument("--copies", default=None)
    parser.add_argument("--ceiling-files", type=int, default=256)
    parser.add_argument("--ceiling-skip", type=int, default=0,
                        help="start the ceiling's files this many copies in, to read "
                             "copies the file server's cache is less likely to hold")
    parser.add_argument("--ceiling-threads", type=lambda s: [int(x) for x in s.split(",") if x],
                        default=[])
    parser.add_argument("--ceiling-modes", type=lambda s: [x for x in s.split(",") if x],
                        default=["buffered", "direct"])
    # driver only
    parser.add_argument("--arm", action="append", default=[],
                        help="NAME=TREE[:SDK_ROOT]; repeatable")
    parser.add_argument("--arm-env", action="append", default=[],
                        help="NAME=VARIABLE=VALUE: set one environment variable "
                             "for one arm's children; repeatable")
    parser.add_argument("--workloads", type=lambda s: s.split(","),
                        default=["stage-b", "stage-a"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--py-spy", default=None,
                        help="py-spy executable; a comma-separated list is "
                             "searched in order, so one command runs on boxes "
                             "that install it in different places")
    parser.add_argument("--py-spy-rate", type=int, default=100)
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args)
    if not args.arm or not args.out:
        parser.error("the driver needs --arm and --out")
    if args.py_spy:
        found = [c for c in args.py_spy.split(",") if c and os.access(c, os.X_OK)
                 and os.path.isfile(c)]
        if not found:
            parser.error(f"no py-spy executable among {args.py_spy!r}")
        args.py_spy = found[0]
    # What every child is told, minus the driver-only options.
    args.child_argv = [
        "--slice", args.slice, "--slice-sha256", args.slice_sha256,
        "--space", args.space, "--manifest-sha256", args.manifest_sha256,
        "--allowed-tiers", args.allowed_tiers, "--land-wait-s", str(args.land_wait_s),
        "--ram-wait-s", str(args.ram_wait_s),
        "--sink", args.sink, "--scratch-max-bytes", str(args.scratch_max_bytes),
        "--sink-ceiling-bytes", str(args.sink_ceiling_bytes),
        "--stage-a-boundary", args.stage_a_boundary,
        "--stage-a-batches", str(args.stage_a_batches),
        "--stage-a-windows", str(args.stage_a_windows),
        "--ceiling-files", str(args.ceiling_files),
        "--ceiling-skip", str(args.ceiling_skip),
        "--ceiling-threads", ",".join(str(t) for t in args.ceiling_threads),
        "--ceiling-modes", ",".join(args.ceiling_modes)]
    if args.scratch_root:
        args.child_argv += ["--scratch-root", args.scratch_root]
    return driver_main(args)


if __name__ == "__main__":
    sys.exit(main())
