"""Unit journal bench: what a Stage B window's unit commit costs (PQ #1207).

PQ #1207 reports that a Stage B render window spends 25-30 s of its 45-50 s
checkpointing its units on the main thread. The row's own counters disagree
(``span.wall_s - windows[i].wall_s``, staging wait plus commit, is about 2 s
on rows 38, 39 and 43), so this tool measures the commit itself, on the
checkpoint bytes a production row writes, under the conditions the report
was taken in.

The workload is the body of ``commit_streamed_units`` in
``joint_cost_quantum.py``, called unit by unit on the real functions:
``make_joint_aura_entry`` for each format, ``_aura_unit_state`` and
``_write_aura_unit_checkpoint``. Its inputs come from one production unit
checkpoint (``--sample-checkpoint``): the probe identity (6.5 MB, almost all
of it ``source_model``), one object shared by every unit as in the quantum,
and the unit's rows, formats and probe count. Each unit gets its own name, so
each writes its own file.

Arms (``--arms``), each a ``child`` process, interleaved over ``--repeats``:

* ``clean``: the commit loop alone.
* ``pyspy-block``: the same under ``py-spy record --idle`` in its default
  blocking mode, at the report's 50 Hz, which stops the process for every
  sample. The report's py-spy ran this way.
* ``pyspy-nonblock``: the same with ``--nonblocking``.
* ``pressure``: the child first reads ``--ballast-gb`` of file data so the
  action's memory cgroup sits at ``memory.max`` on page cache, as row 42's did
  (28 GiB, 16-19 GiB of it file pages), then runs the loop.

Every child starts ``--idle-threads`` threads parked on a queue, as a
quantum's staged-read, exact-read and layer-read pools are during a commit,
because a blocking sampler's cost grows with the thread count.

Per window the child records the wall time, the main thread's CPU time
(``time.thread_time``), the cgroup's memory stall time and reclaim counters,
MemAvailable, and a per-unit phase breakdown. The host samples GPU power at
1 Hz per arm, and ``perf`` on the ``clean`` arm when it can.

Writes only under ``--scratch``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MARKER = "[unit-journal-bench] "
ARMS = ("clean", "pyspy-block", "pyspy-nonblock", "pressure")


def _cgroup_dir() -> Path | None:
    try:
        line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    except OSError:
        return None
    path = Path("/sys/fs/cgroup") / line.split("::", 1)[-1].lstrip("/")
    return path if path.is_dir() else None


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return None if text == "max" else int(text)


def _memory_sample(group: Path | None) -> dict:
    sample = {"unix": time.time()}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            sample["mem_available_bytes"] = int(line.split()[1]) * 1024
    if group is None:
        return sample
    sample["memory_current"] = _read_int(group / "memory.current")
    sample["memory_max"] = _read_int(group / "memory.max")
    try:
        stat = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
        for key in ("file", "anon", "pgscan", "pgsteal", "pgscan_direct",
                    "pgsteal_direct", "workingset_refault_file"):
            if key in stat:
                sample[key] = int(stat[key])
    except (OSError, ValueError):
        pass
    try:
        for line in (group / "memory.pressure").read_text().splitlines():
            kind, *fields = line.split()
            sample[f"memory_pressure_{kind}_total_us"] = int(
                dict(field.split("=") for field in fields)["total"])
    except (OSError, ValueError, KeyError):
        pass
    return sample


def _delta(after: dict, before: dict) -> dict:
    return {key: after[key] - before[key] for key in after
            if key != "unix" and isinstance(after.get(key), int)
            and isinstance(before.get(key), int)
            and key not in ("memory_max", "memory_current", "mem_available_bytes")}


def _drop_cache(path: Path) -> None:
    """Drop a clean file's page cache, so a later arm does not start charged."""
    targets = [path] if path.is_file() else sorted(path.rglob("*")) if path.exists() else []
    for target in targets:
        if not target.is_file():
            continue
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------
# child: one arm
# --------------------------------------------------------------------------

def _load_sample(path: Path) -> dict:
    with path.open("rb") as handle:
        envelope = pickle.load(handle)
    return pickle.loads(envelope["payload"])


def _unit_inputs(sample: dict, units: int):
    """Per-unit inputs shaped like the quantum's, from one production unit."""
    from prismaquant.joint_aura import validated_probe_identity

    rows = sample["joint_aura_rows"]
    first = next(iter(rows.values()))
    probe_identity = first["probe_identity"]
    joint_probe = validated_probe_identity(probe_identity)
    formats = list(rows)
    render_formats = [fmt for fmt in sample["rows"]]
    names = [f"bench.layers.0.mlp.experts.{index}.down_proj" for index in range(units)]
    operators, components, s2, s4, x2_probe, dw_src, g_trace = {}, {}, {}, {}, {}, {}, {}
    for name in names:
        g_trace[name] = float(sample["g_trace"])
        for fmt in formats:
            operator = dict(rows[fmt]["joint_operator_identity"])
            operator["qname"] = name
            operators[(name, fmt)] = operator
            components[(name, fmt)] = [dict(value) for value in
                                       rows[fmt]["signed_components_per_probe"]]
        for fmt in render_formats:
            row = sample["rows"][fmt]
            s2[(name, fmt)] = float(row["s2"])
            s4[(name, fmt)] = float(row["s4"])
            x2_probe[(name, fmt)] = list(row["x2_probe"])
            dw_src[(name, fmt)] = str(row["dw_src"])
    return SimpleInputs(names=names, formats=formats, render_formats=render_formats,
                        probe_identity=probe_identity, joint_probe=joint_probe,
                        operators=operators, components=components, s2=s2, s4=s4,
                        x2_probe=x2_probe, dw_src=dw_src, g_trace=g_trace)


class SimpleInputs:
    def __init__(self, **values):
        self.__dict__.update(values)


def _commit_unit(inputs, name, root: Path, identity_sha256: str, timings: dict | None):
    """``commit_streamed_units``' body for one unit, on the same functions."""
    from prismaquant.aura_cost import _aura_unit_state, _write_aura_unit_checkpoint
    from prismaquant.joint_aura import make_joint_aura_entry

    started = time.perf_counter()
    rows = {}
    for fmt in inputs.formats:
        row = make_joint_aura_entry(
            operator_identity=inputs.operators[(name, fmt)],
            probe_identity=inputs.joint_probe,
            signed_components=inputs.components[(name, fmt)])
        row["probe_identity"] = inputs.probe_identity
        rows[fmt] = row
    entries = time.perf_counter()
    state = {**_aura_unit_state(
        name, inputs.render_formats, s2=inputs.s2, s4=inputs.s4,
        x2_probe=inputs.x2_probe, dw_src=inputs.dw_src, g_trace=inputs.g_trace,
        col_energy={}, weight_mse_diagnostic={}, source_weight_identity={},
        observation_counts=None), "joint_aura_rows": rows}
    built = time.perf_counter()
    if timings is None:
        _write_aura_unit_checkpoint(root, qname=name, identity_sha256=identity_sha256,
                                    state=state)
        return
    # The writer's steps, timed one by one (the same calls, in its order).
    from prismaquant.aura_cost import (AURA_CHECKPOINT_UNIT_SCHEMA, _atomic_write_bytes,
                                       _aura_unit_checkpoint_path)
    state_bytes = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
    pickled = time.perf_counter()
    digest = hashlib.sha256(state_bytes).hexdigest()
    hashed = time.perf_counter()
    encoded = pickle.dumps({"schema": AURA_CHECKPOINT_UNIT_SCHEMA, "qname": str(name),
                            "identity_sha256": str(identity_sha256),
                            "payload_sha256": digest, "payload": state_bytes},
                           protocol=pickle.HIGHEST_PROTOCOL)
    enveloped = time.perf_counter()
    _atomic_write_bytes(_aura_unit_checkpoint_path(root, name), encoded)
    written = time.perf_counter()
    for key, seconds in (("entries_s", entries - started), ("state_s", built - entries),
                         ("pickle_s", pickled - built), ("sha256_s", hashed - pickled),
                         ("envelope_s", enveloped - hashed), ("write_s", written - enveloped)):
        timings[key] = timings.get(key, 0.0) + seconds
    timings["bytes"] = timings.get("bytes", 0) + len(encoded)


def _read_ballast(path: Path, chunk: int = 64 << 20) -> int:
    total = 0
    buffer = bytearray(chunk)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(view)
            if not count:
                return total
            total += count


def cmd_child(args) -> int:
    group = _cgroup_dir()
    parked = queue.Queue()
    threads = [threading.Thread(target=parked.get, daemon=True, name=f"parked_{index}")
               for index in range(args.idle_threads)]
    for thread in threads:
        thread.start()
    sample = _load_sample(Path(args.sample_checkpoint))
    inputs = _unit_inputs(sample, args.units * args.windows)
    root = Path(args.scratch) / f"checkpoints-{args.arm}-{args.repeat}"
    shutil.rmtree(root, ignore_errors=True)
    (root / "units").mkdir(parents=True)
    identity_sha256 = hashlib.sha256(b"unit-journal-bench").hexdigest()
    result = {"arm": args.arm, "repeat": args.repeat, "pid": os.getpid(),
              "cgroup": str(group) if group else None,
              "formats": len(inputs.formats), "render_formats": len(inputs.render_formats),
              "n_probes": len(next(iter(inputs.components.values()))),
              "idle_threads": args.idle_threads, "units_per_window": args.units}
    if args.arm == "pressure":
        before = _memory_sample(group)
        result["ballast_read_bytes"] = _read_ballast(Path(args.ballast))
        result["after_ballast"] = _memory_sample(group)
        result["before_ballast"] = before
    windows = []
    for window in range(args.windows):
        names = inputs.names[window * args.units:(window + 1) * args.units]
        timings = {} if window == args.windows - 1 else None
        before = _memory_sample(group)
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        for name in names:
            _commit_unit(inputs, name, root, identity_sha256, timings)
        wall, cpu = time.perf_counter() - wall0, time.thread_time() - cpu0
        after = _memory_sample(group)
        windows.append({"window": window, "units": len(names), "wall_s": wall,
                        "main_thread_cpu_s": cpu, "phase_breakdown": timings,
                        "memory_before": before, "memory_after": after,
                        "memory_delta": _delta(after, before)})
    result["windows"] = windows
    sizes = sorted({path.stat().st_size for path in (root / "units").glob("*.pkl")})
    result["checkpoint_bytes"] = sizes
    result["checkpoint_sha256_first"] = hashlib.sha256(
        sorted((root / "units").glob("*.pkl"))[0].read_bytes()).hexdigest()
    Path(args.result).write_text(json.dumps(result, indent=1, sort_keys=True))
    _drop_cache(root)
    shutil.rmtree(root, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------
# host: all arms, interleaved
# --------------------------------------------------------------------------

def _cpu_parts() -> dict:
    """The CPUs this action may run on, by core part (GB10: X925 0xd85, A725 0xd87)."""
    parts, cpu = {}, None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("processor"):
            cpu = int(line.split(":")[1])
        elif line.startswith("CPU part") and cpu is not None:
            parts[cpu] = line.split(":")[1].strip()
    allowed = sorted(os.sched_getaffinity(0))
    return {"allowed": allowed, "parts": [parts.get(cpu) for cpu in allowed]}


def _py_spy(explicit):
    for candidate in (explicit, shutil.which("py-spy"), "/usr/local/bin/py-spy",
                      str(Path.home() / ".local/bin/py-spy")):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


def _power(seconds_fn):
    from prismaquant.io_spans import GpuPowerSampler

    sampler = GpuPowerSampler().start()
    try:
        seconds_fn()
    finally:
        block = sampler.stop()
    watts = sampler.samples
    block["mean_w"] = statistics.fmean(watts) if watts else None
    block["envelope_fraction"] = (block["mean_w"] / 140.0) if watts else None
    return block


def cmd_host(args) -> int:
    scratch = Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "results"
    out.mkdir(exist_ok=True)
    spy = _py_spy(args.py_spy)
    perf = shutil.which("perf")
    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    if any(arm.startswith("pyspy") for arm in arms) and spy is None:
        raise SystemExit("py-spy arms requested but py-spy is not installed")
    ballast = scratch / "ballast.bin"
    if "pressure" in arms:
        size = int(args.ballast_gb * (1 << 30))
        if not ballast.exists() or ballast.stat().st_size != size:
            block = os.urandom(64 << 20)
            with ballast.open("wb") as handle:
                for _ in range(size // len(block)):
                    handle.write(block)
            _drop_cache(ballast)
    print(MARKER + json.dumps({"py_spy": spy, "perf": perf, "arms": arms,
                               "cgroup": str(_cgroup_dir()), "host": os.uname().nodename,
                               "cpus": _cpu_parts(), "loadavg": os.getloadavg()}),
          flush=True)
    records = []
    for repeat in range(args.repeats):
        order = arms[repeat % len(arms):] + arms[:repeat % len(arms)]
        for arm in order:
            result = out / f"{arm}-{repeat}.json"
            child = [sys.executable, os.path.abspath(__file__), "child", "--arm", arm,
                     "--repeat", str(repeat), "--sample-checkpoint", args.sample_checkpoint,
                     "--units", str(args.units), "--windows", str(args.windows),
                     "--idle-threads", str(args.idle_threads), "--scratch", str(scratch),
                     "--ballast", str(ballast), "--result", str(result)]
            command = child
            profile = None
            if arm.startswith("pyspy"):
                profile = out / f"{arm}-{repeat}.pyspy.raw"
                command = [spy, "record", "--rate", str(args.py_spy_rate), "--format", "raw",
                           "--idle", "--threads", "-o", str(profile)]
                if arm == "pyspy-nonblock":
                    command.append("--nonblocking")
                command += ["--", *child]
            perf_data = None
            if arm == "clean" and perf and repeat == 0:
                perf_data = out / "clean-0.perf.data"
                command = [perf, "record", "-F", "199", "-g", "-e", "cpu-clock:u",
                           "-o", str(perf_data), "--", *child]
            env = dict(os.environ, PYTHONPATH=str(REPO))
            wall0 = time.perf_counter()
            state = {}

            def run():
                state["proc"] = subprocess.run(command, env=env, capture_output=True, text=True)

            power = _power(run)
            proc = state["proc"]
            record = {"arm": arm, "repeat": repeat, "returncode": proc.returncode,
                      "wall_s": time.perf_counter() - wall0, "gpu_power": power,
                      "stderr_tail": proc.stderr[-2000:]}
            if proc.returncode == 0:
                record["child"] = json.loads(result.read_text())
            if perf_data is not None and perf_data.exists():
                report = subprocess.run(
                    [perf, "report", "-i", str(perf_data), "--stdio", "--no-children",
                     "--sort", "dso,sym", "-g", "none"], capture_output=True, text=True)
                record["perf_top"] = [line for line in report.stdout.splitlines()
                                      if line.strip() and not line.startswith("#")][:40]
            if profile is not None and profile.exists():
                record["pyspy_main_leaf"] = _spy_main(profile)
            if arm == "pressure":
                _drop_cache(ballast)
            records.append(record)
            print(MARKER + json.dumps(_summary(record)), flush=True)
            for key in ("perf_top", "pyspy_main_leaf"):
                if key in record:
                    print(MARKER + json.dumps({"arm": arm, "repeat": repeat,
                                               key: record[key]}), flush=True)
            if proc.returncode:
                print(MARKER + json.dumps({"arm": arm, "repeat": repeat,
                                           "stderr_tail": record["stderr_tail"]}), flush=True)
    (out / "records.json").write_text(json.dumps(records, indent=1, sort_keys=True))
    print(MARKER + json.dumps({"analysis": _analyze(records)}, indent=1), flush=True)
    return 0 if all(record["returncode"] == 0 for record in records) else 1


def _spy_main(path: Path) -> dict:
    counts: dict[str, int] = {}
    total = 0
    for line in path.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        if "MainThread" not in stack.split(";", 1)[0]:
            continue
        frames = [frame.split(" (")[0] for frame in stack.split(";")[1:]]
        leaf = next((frame for frame in reversed(frames) if frame in (
            "_write_aura_unit_checkpoint", "_atomic_write_bytes", "make_joint_aura_entry",
            "_aura_unit_state", "_read_ballast", "_load_sample", "_unit_inputs")), "other")
        counts[leaf] = counts.get(leaf, 0) + int(count)
        total += int(count)
    return {"samples": total, "by_function": counts}


def _summary(record: dict) -> dict:
    child = record.get("child") or {}
    windows = child.get("windows") or []
    return {"arm": record["arm"], "repeat": record["repeat"],
            "returncode": record["returncode"],
            "window_wall_s": [round(w["wall_s"], 3) for w in windows],
            "window_main_cpu_s": [round(w["main_thread_cpu_s"], 3) for w in windows],
            "gpu_mean_w": record["gpu_power"].get("mean_w")}


def _analyze(records: list[dict]) -> dict:
    by_arm: dict[str, list] = {}
    for record in records:
        child = record.get("child")
        if not child:
            continue
        for window in child["windows"]:
            by_arm.setdefault(record["arm"], []).append(window)
    analysis = {}
    for arm, windows in by_arm.items():
        walls = [w["wall_s"] for w in windows]
        cpus = [w["main_thread_cpu_s"] for w in windows]
        units = windows[0]["units"]
        phases = [w["phase_breakdown"] for w in windows if w["phase_breakdown"]]
        stall = [w["memory_delta"].get("memory_pressure_some_total_us", 0) / 1e6
                 for w in windows]
        analysis[arm] = {
            "windows": len(windows), "units_per_window": units,
            "window_wall_s_mean": statistics.fmean(walls),
            "window_wall_s_max": max(walls),
            "ms_per_unit": 1000 * statistics.fmean(walls) / units,
            "main_thread_on_cpu_fraction": sum(cpus) / sum(walls),
            "memory_stall_some_s_per_window": statistics.fmean(stall),
            "phase_ms_per_unit": {key: 1000 * statistics.fmean(p[key] for p in phases) / units
                                  for key in (phases[0] if phases else {}) if key != "bytes"},
            "stage_b_row_s_at_15_windows": 15 * statistics.fmean(walls),
        }
    return analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    host = sub.add_parser("host")
    host.add_argument("--sample-checkpoint", required=True)
    host.add_argument("--scratch", required=True)
    host.add_argument("--arms", default=",".join(ARMS))
    host.add_argument("--repeats", type=int, default=2)
    host.add_argument("--units", type=int, default=59)
    host.add_argument("--windows", type=int, default=3)
    host.add_argument("--idle-threads", type=int, default=40)
    host.add_argument("--ballast-gb", type=float, default=8.0)
    host.add_argument("--py-spy", default=None)
    host.add_argument("--py-spy-rate", type=int, default=50)
    child = sub.add_parser("child")
    for name in ("--arm", "--sample-checkpoint", "--scratch", "--ballast", "--result"):
        child.add_argument(name, required=True)
    for name in ("--repeat", "--units", "--windows", "--idle-threads"):
        child.add_argument(name, type=int, required=True)
    args = parser.parse_args(argv)
    return cmd_host(args) if args.command == "host" else cmd_child(args)


if __name__ == "__main__":
    sys.exit(main())
