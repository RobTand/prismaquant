"""Render window bench: the main thread's load work in a Stage B window.

PQ #1192 takes a Stage B render window's load work off the main thread: the
rendered-weight hash, once per probe on main, and the window preflight's
archive-directory scans. The claim is a delta in the main thread's seconds
per window, so this tool runs both trees on the same bytes, in one admitted
action, on one box, alternating the arms, each arm a child process under
py-spy.

The workload is the production retained-window replay on a stub layer, not
a model of it:

* ``--units`` bf16 ``Linear(--dim, --dim)`` sources on the GPU, and two
  disk-backed renders per unit (``FP8_E4M3`` and ``NVFP4A16``, each the
  source plus a small offset, saved as ordinary Torch files);
* a ``ProductionWeightCache`` over those files, with an LRU at the window's
  render cap and every render bound to its file SHA-256
  (``require_file_load_sha256``), as the Stage B run leg binds them;
* ``observe_and_project_retained_windows`` with real statistics leases and
  projections, ``--probes`` probes and ``--per-window`` units per window;
  the backward runs every source Linear on a random ``--tokens`` batch;
* a ``record_operator`` that performs the Stage B quantum's render identity
  check: a tree without ``resident_render_identity`` hashes the render on
  every probe, as ``_record_joint_operator`` did on main; a tree with it
  reads the load's hash and asks the loaders to compute it.

What differs from a production window: the renders are read from local
disk, not through PrismaBuild's staged tiers or from the pool over NFS, so
the archive scans here are local reads and much cheaper than row 43's cold
pool reads; there is no capture guard, no spill and no source install.

Subcommands:

* ``host`` (the action's command) extracts each earlier arm's
  ``prismaquant`` and ``tools`` from the action's own Git snapshot
  (``--arm NAME=REF``, whose REF pbrun's ``--snapshot-ref`` must advertise),
  writes the render files once, runs each repeat's arms, this tree last on
  the first repeat and in a rotated order after it, as ``child`` processes
  under ``py-spy record``, and ends with ``analyze``.
* ``child`` runs one arm and writes one result record.
* ``analyze`` reports, per arm, the main thread's seconds per window by
  instrumented timers and by py-spy samples, GPU power, and whether every
  arm computed the same projections.

Writes only under ``--scratch``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "RENDER-WINDOW-BENCH "
FORMATS = ("FP8_E4M3", "NVFP4A16")


# -- host ----------------------------------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", "-c", "safe.directory=*", *args], check=True,
                          capture_output=True).stdout


def _renders(args, root: Path):
    """Write the render files once; the same bytes serve every arm."""
    import torch

    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(1192)
    manifest = {}
    for unit in range(args.units):
        name = f"unit{unit:02d}"
        source = (torch.rand((args.dim, args.dim), generator=generator) * 2 - 1) / args.dim ** 0.5
        torch.save(source.to(torch.bfloat16), root / f"{name}.source.pt")
        for index, fmt in enumerate(FORMATS):
            path = root / f"{name}.{fmt}.pt"
            torch.save((source + (index + 1) / 128).to(torch.bfloat16), path)
            manifest[f"{name}|{fmt}"] = {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def _py_spy(explicit):
    for candidate in (explicit, shutil.which("py-spy"), "/usr/local/bin/py-spy",
                      str(Path.home() / ".local/bin/py-spy")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def cmd_host(args) -> int:
    scratch = Path(args.scratch) / args.label
    if scratch.exists():
        raise SystemExit(f"{scratch} exists: use a fresh --label")
    head_tree = Path.cwd()
    arms, commits = {}, {}
    for spec in args.arm:
        name, _, ref = spec.partition("=")
        if not name or not ref or name == "head" or name in arms:
            raise SystemExit(f"--arm wants a new NAME=REF, not {spec!r}")
        tree = scratch / f"arm-{name}"
        tree.mkdir(parents=True)
        archive = _git("archive", ref, "prismaquant", "tools")
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
        arms[name] = tree
        commits[name] = _git("rev-parse", ref).decode().strip()
    arms["head"] = head_tree
    commits["head"] = _git("rev-parse", "HEAD").decode().strip()
    identity = {"arms": commits, "host": os.uname().nodename,
                "affinity": len(os.sched_getaffinity(0))}
    print(MARKER + json.dumps(identity), flush=True)
    renders = scratch / "renders"
    _renders(args, renders)
    spy = _py_spy(args.py_spy)
    print(MARKER + json.dumps({"py_spy": spy}), flush=True)
    out = scratch / "out"
    out.mkdir()
    runs = []
    names = list(arms)
    for repeat in range(args.repeats):
        shift = repeat % len(names)
        order = names[shift:] + names[:shift]
        for arm in order:
            label = f"{arm}-r{repeat}"
            env = {**os.environ, "PYTHONPATH": str(arms[arm])}
            command = [sys.executable, str(head_tree / "tools" / "render_window_bench.py"),
                       "child", "--arm", arm, "--label", label,
                       "--renders", str(renders), "--result", str(out / f"{label}.json"),
                       *_shape_args(args)]
            if spy is not None:
                command = [spy, "record", "--rate", str(args.py_spy_rate), "--format", "raw",
                           "--idle", "--threads", "--nonblocking",
                           "--output", str(out / f"{label}.raw"), "--", *command]
            started = time.time()
            code = subprocess.run(command, env=env, cwd=str(arms[arm])).returncode
            runs.append({"label": label, "arm": arm, "repeat": repeat, "returncode": code,
                         "started_unix": started, "ended_unix": time.time()})
            print(f"{MARKER}end {label}: rc={code} in {time.time() - started:.1f}s", flush=True)
            if code != 0:
                shutil.rmtree(renders)
                for name, tree in arms.items():
                    if name != "head":
                        shutil.rmtree(tree)
                return code
    (out / "host.json").write_text(json.dumps({**identity, "runs": runs}, indent=1))
    shutil.rmtree(renders)
    for name, tree in arms.items():
        if name != "head":
            shutil.rmtree(tree)
    return cmd_analyze(argparse.Namespace(out=str(out), py_spy_rate=args.py_spy_rate))


def _shape_args(args):
    return ["--units", str(args.units), "--dim", str(args.dim),
            "--per-window", str(args.per_window), "--probes", str(args.probes),
            "--tokens", str(args.tokens), "--workers", str(args.workers)]


# -- child ---------------------------------------------------------------------------------

class _Timers:
    """Seconds per (window, thread kind, category), under one lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.window = None
        self.seconds = defaultdict(float)
        self.calls = Counter()

    def add(self, category, elapsed):
        kind = "main" if threading.current_thread() is threading.main_thread() else "loader"
        with self.lock:
            self.seconds[(self.window, kind, category)] += elapsed
            self.calls[(self.window, kind, category)] += 1

    def wrap(self, function, category, *, when=None):
        def timed(*args, **kwargs):
            if when is not None and not when(*args, **kwargs):
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(category, time.perf_counter() - started)
        return timed


def _power_summary(values):
    return ({"samples": len(values), "mean_w": statistics.fmean(values),
             "max_w": max(values)} if values else {"samples": 0})


def _drop_pages(paths):
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def cmd_child(args) -> int:
    import torch

    from prismaquant import format_registry as fr
    from prismaquant import production_weight_cache as pwc
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    from prismaquant.joint_statistics_replay import (
        SCHEMA, observe_and_project_retained_windows)

    device = torch.device("cuda")
    renders = Path(args.renders)
    manifest = json.loads((renders / "manifest.json").read_text())
    torch.manual_seed(1192)
    modules, specs = {}, {}
    for unit in range(args.units):
        name = f"unit{unit:02d}"
        module = torch.nn.Linear(args.dim, args.dim, bias=False, dtype=torch.bfloat16,
                                 device=device).eval()
        module.weight.requires_grad_(False)
        module.weight.copy_(torch.load(renders / f"{name}.source.pt").to(device))
        modules[name] = module
        specs[name] = {fmt: fr.get_format(fmt) for fmt in FORMATS}
    keys = {(entry.split("|")[0], entry.split("|")[1]): value
            for entry, value in manifest.items()}
    file_bytes = max(Path(value["path"]).stat().st_size for value in keys.values())
    maxima = {name: 1.0 for name in modules}
    prepared = {key: pwc._cb_cache_tensor_identity(torch.load(value["path"]))
                for key, value in keys.items()}

    full = plan_joint_statistics_target_windows(
        modules, specs, max_statistics_bytes=1 << 40, activation_max_abs=maxima)
    per_target = max(target.statistics_bytes for target in full.targets)
    statistics_cap = args.per_window * per_target
    render_cap = args.per_window * len(FORMATS) * file_bytes
    workers = min(args.workers, len(os.sched_getaffinity(0)))
    budget = RetainedWindowBudget(
        physical_limit_bytes=1 << 40, safety_margin_bytes=1 << 30,
        metadata_reserve_bytes=1 << 20, runtime_reserve_bytes=1 << 20,
        workspace_reserve_bytes=1 << 20, boundary_reserve_bytes=0,
        auxiliary_reserve_bytes=0, load_buffer_bytes=workers * file_bytes,
        read_page_reserve_bytes=1 << 20, candidate_delta_bytes=4 * args.dim * args.dim,
        statistics_cap_bytes=statistics_cap, retained_render_cap_bytes=render_cap,
        max_windows_per_layer=-(-args.units // args.per_window))
    policy = dict(schema=SCHEMA, max_statistics_bytes=statistics_cap,
                  max_candidate_bytes=4 * args.dim * args.dim,
                  max_render_resident_bytes=render_cap,
                  max_load_buffer_bytes=workers * file_bytes,
                  workspace_reserve_bytes=1 << 20, max_replay_cotangent_bytes=1 << 20,
                  prefetch_workers=workers)
    cache = pwc.ProductionWeightCache(
        weights={key: value["path"] for key, value in keys.items()}, levers={},
        activation_max_abs=maxima)
    cache.enable_lru(render_cap)
    cache.require_file_load_sha256({key: value["sha256"] for key, value in keys.items()},
                                   max_file_bytes=file_bytes)

    memo = hasattr(pwc.ProductionWeightCache, "resident_render_identity")
    options = {}
    if "render_identities" in inspect.signature(observe_and_project_retained_windows).parameters:
        options["render_identities"] = True

    timers = _Timers()
    pwc._cb_cache_tensor_identity = timers.wrap(pwc._cb_cache_tensor_identity, "hash")
    scan = pwc.ProductionWeightCache._window_archive_storage_bytes
    pwc.ProductionWeightCache._window_archive_storage_bytes = staticmethod(timers.wrap(
        scan, "archive_scan", when=lambda source: isinstance(source, (str, Path))))
    for method, category in (("prefetch", "prefetch_wait"),
                             ("_retained_window_preflight", "preflight"),
                             ("_prefill_window_archive_memo", "archive_scan_pool_wait")):
        if hasattr(pwc.ProductionWeightCache, method):
            setattr(pwc.ProductionWeightCache, method,
                    timers.wrap(getattr(pwc.ProductionWeightCache, method), category))

    def record(name, fmt, source, rendered):
        # The Stage B quantum's render identity check (``_record_joint_operator``).
        identity = (cache.resident_render_identity(name, fmt, rendered) if memo
                    else pwc._cb_cache_tensor_identity(rendered))
        if identity != prepared[(name, fmt)]:
            raise RuntimeError(f"render identity changed for {name}@{fmt}")

    inputs = [((torch.rand((args.tokens, args.dim), generator=torch.Generator().manual_seed(
        7000 + probe)) * 2 - 1).to(device=device, dtype=torch.bfloat16))
        for probe in range(args.probes)]

    def backward(*, probe_index, final, lease):
        started = time.perf_counter()
        x = inputs[probe_index].detach().requires_grad_(True)
        loss = sum(module(x).float().square().sum() * (probe_index + 1)
                   for module in modules.values())
        loss.backward()
        torch.cuda.synchronize(device)
        timers.add("backward_gpu", time.perf_counter() - started)

    digest = hashlib.sha256()

    def consume(probe_index, terms, diagnostics, receipt):
        for key in sorted(terms):
            components = terms[key]
            digest.update(repr((probe_index, key, sorted(components.items()))).encode())

    windows = {}

    def before_window(index, names):
        torch.cuda.synchronize(device)
        timers.window = int(index)
        windows[int(index)] = {"names": list(names), "start_unix": time.time(),
                               "start": time.perf_counter()}

    def after_window(index, names):
        torch.cuda.synchronize(device)
        window = windows[int(index)]
        window["wall_s"] = time.perf_counter() - window.pop("start")
        window["end_unix"] = time.time()
        timers.window = None

    _drop_pages([value["path"] for value in keys.values()])
    from prismaquant.io_spans import GpuPowerSampler

    power = GpuPowerSampler(0.1).start()
    time.sleep(0.5)
    started = time.time()
    observe_and_project_retained_windows(
        modules, specs, cache, policy, retained_budget=budget, n_probes=args.probes,
        source_bytes=1 << 20, backward=backward, record_operator=record,
        consume_probe=consume, collect_col_energy=False, backend=None,
        before_window=before_window, after_window=after_window, **options)
    torch.cuda.synchronize(device)
    ended = time.time()
    time.sleep(0.3)
    power.stop()
    per_window = {}
    for (window, kind, category), seconds in timers.seconds.items():
        per_window.setdefault(str(window), {})[f"{kind}.{category}"] = {
            "seconds": seconds, "calls": timers.calls[(window, kind, category)]}
    for index, window in windows.items():
        window["gpu_power"] = _power_summary(
            power.watts_between(window["start_unix"], window["end_unix"]))
        window["timers"] = per_window.get(str(index), {})
    result = {"arm": args.arm, "label": args.label, "memo": memo,
              "render_identities": bool(options), "workers": workers,
              "units": args.units, "dim": args.dim, "per_window": args.per_window,
              "probes": args.probes, "tokens": args.tokens, "file_bytes": file_bytes,
              "window_count": len(windows), "replay_s": ended - started,
              "start_unix": started, "end_unix": ended,
              "gpu_power": _power_summary(power.watts_between(started, ended)),
              "projection_digest": digest.hexdigest(), "windows": windows,
              "outside_windows": per_window.get("None", {})}
    Path(args.result).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(MARKER + json.dumps({"label": args.label, "replay_s": result["replay_s"],
                               "projection_digest": result["projection_digest"]}), flush=True)
    return 0


# -- analyze -------------------------------------------------------------------------------

CATEGORIES = ("hash", "archive_scan", "archive_scan_pool_wait", "prefetch_wait",
              "preflight", "backward_gpu")


def _spy_main_shares(path: Path, rate: int):
    """Main-thread seconds inside the replay, by the innermost matching frame."""
    buckets = Counter()
    for line in path.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        if "MainThread" not in stack.split(";", 1)[0]:
            continue
        if "observe_and_project_retained_windows" not in stack:
            continue
        frames = stack.split(";")
        if any("_cb_cache_tensor_identity" in frame for frame in frames):
            bucket = "hash"
        elif any("torch_archive_storage_bytes" in frame for frame in frames):
            bucket = "archive_scan"
        elif any("_prefill_window_archive_memo" in frame for frame in frames):
            bucket = "archive_scan_pool_wait"
        elif any(frame.startswith("prefetch ") for frame in frames):
            bucket = "prefetch_wait"
        elif any(frame.startswith("_retained_window_preflight ") for frame in frames):
            bucket = "preflight_other"
        elif any(frame.startswith("backward ") for frame in frames):
            bucket = "backward_gpu"
        else:
            bucket = "other"
        buckets[bucket] += int(count)
    return {bucket: samples / rate for bucket, samples in buckets.items()}


def cmd_analyze(args) -> int:
    out = Path(args.out)
    results = [json.loads(path.read_text()) for path in sorted(out.glob("*-r*.json"))]
    by_arm = defaultdict(list)
    for result in results:
        by_arm[result["arm"]].append(result)
    summary = {}
    for arm, runs in sorted(by_arm.items()):
        rows = defaultdict(list)
        spy = defaultdict(list)
        for run in runs:
            windows = run["window_count"]
            for category in CATEGORIES:
                for kind in ("main", "loader"):
                    total = sum(window["timers"].get(f"{kind}.{category}", {}).get("seconds", 0.0)
                                for window in run["windows"].values())
                    rows[f"{kind}.{category}"].append(total / windows)
            rows["window_wall"].append(statistics.fmean(
                window["wall_s"] for window in run["windows"].values()))
            rows["gpu_power_mean_w"].append(run["gpu_power"].get("mean_w", float("nan")))
            raw = out / f"{run['label']}.raw"
            if raw.exists():
                for bucket, seconds in _spy_main_shares(raw, args.py_spy_rate).items():
                    spy[bucket].append(seconds / windows)
        summary[arm] = {
            "runs": len(runs),
            "projection_digests": sorted({run["projection_digest"] for run in runs}),
            "timers_s_per_window": {key: {"mean": statistics.fmean(values),
                                          "values": values}
                                    for key, values in rows.items()},
            "py_spy_main_s_per_window": {key: {"mean": statistics.fmean(values),
                                               "values": values}
                                         for key, values in spy.items()},
        }
    digests = {digest for arm in summary.values() for digest in arm["projection_digests"]}
    report = {"arms": summary, "identical_projections": len(digests) == 1}
    (out / "analysis.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(MARKER + "analysis " + json.dumps(report, sort_keys=True), flush=True)
    return 0 if len(digests) == 1 else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def shape(p):
        p.add_argument("--units", type=int, default=16)
        p.add_argument("--dim", type=int, default=4096)
        p.add_argument("--per-window", type=int, default=8)
        p.add_argument("--probes", type=int, default=4)
        p.add_argument("--tokens", type=int, default=512)
        p.add_argument("--workers", type=int, default=8)

    host = commands.add_parser("host")
    shape(host)
    host.add_argument("--arm", action="append", required=True,
                      help="NAME=REF, an earlier tree to compare; repeatable")
    host.add_argument("--scratch", required=True)
    host.add_argument("--label", required=True)
    host.add_argument("--repeats", type=int, default=3)
    host.add_argument("--py-spy", default=None)
    host.add_argument("--py-spy-rate", type=int, default=100)
    child = commands.add_parser("child")
    shape(child)
    child.add_argument("--arm", required=True)
    child.add_argument("--label", required=True)
    child.add_argument("--renders", required=True)
    child.add_argument("--result", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--py-spy-rate", type=int, default=100)
    args = parser.parse_args(argv)
    return {"host": cmd_host, "child": cmd_child, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
