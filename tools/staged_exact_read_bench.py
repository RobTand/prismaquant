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
                 and c["bytes"] > (1 << 20)][:args.ceiling_files]
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


def child_stage_b(args, slice_doc) -> dict:
    from prismaquant.joint_adjoint_checkpoints import load_adjoint_checkpoint
    resolver = _bind(args.manifest_sha256, args.allowed_tiers)
    record = slice_doc["checkpoint"]
    policy = slice_doc["boundary_storage"]["policy"]
    arena = None
    sink = None

    discard = args.sink == "discard" or args.child == "stage-b-discard"

    def factory(rows):
        nonlocal arena, sink
        if discard:
            sink = _DiscardSink()
            return sink
        from prismaquant.perturbed_x_cache import ExactCotangentScratch
        arena = ExactCotangentScratch(
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
    started = time.monotonic()
    try:
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
    finally:
        if arena is not None:
            arena.close()
    return {"entries": entries, "file_bytes": file_bytes, "tensor_bytes": tensor_bytes,
            "wall_s": round(wall, 4), "entries_per_s": round(entries / wall, 3),
            "mb_s": round(file_bytes / wall / 1e6, 1),
            "resident_peak_bytes": resident["peak"], "io": _io_delta(io0, proc_io()),
            "counters": _counters(), "residency": _residency(resolver)}


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


CHILDREN = {"paths": child_paths, "ceiling": child_ceiling,
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
    for repeat in range(args.repeats):
        for workload in args.workloads:
            order = arms if repeat % 2 == 0 else list(reversed(arms))
            for arm in order:
                advised = drop_client_cache(staged_files)
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
    parser.add_argument("--stage-a-boundary", default="44")
    parser.add_argument("--stage-a-batches", type=int, default=16)
    parser.add_argument("--stage-a-windows", type=int, default=8)
    parser.add_argument("--copies", default=None)
    parser.add_argument("--ceiling-files", type=int, default=256)
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
        "--stage-a-boundary", args.stage_a_boundary,
        "--stage-a-batches", str(args.stage_a_batches),
        "--stage-a-windows", str(args.stage_a_windows),
        "--ceiling-files", str(args.ceiling_files),
        "--ceiling-threads", ",".join(str(t) for t in args.ceiling_threads),
        "--ceiling-modes", ",".join(args.ceiling_modes)]
    if args.scratch_root:
        args.child_argv += ["--scratch-root", args.scratch_root]
    return driver_main(args)


if __name__ == "__main__":
    sys.exit(main())
