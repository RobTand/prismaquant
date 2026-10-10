"""Paired before/after measure for PQ #2039: the device check's host copy.

One admitted action, one GB10 worker, one process. The check under test is the
production serial one: ``_start_projected_unit_check`` for every unit, then one
``_settle_projected_unit_checks``. Every unit's bytes come through the real
read, ``source_unit_weight`` over a bound residency map whose staged ranges hold
each unit, which is the reader PrismaBuild's stage tier feeds. A read the stage
did not serve shows in ``bytes_from_pool`` and fails the run.

The arms differ in one request. ``before`` withholds the page-locked request:
the staged reader hands back the pageable buffer it always did, and the check
makes its one private ``pinned.copy_(weight)``. ``after`` is the shipped code:
the reader fills a pinned buffer and the check adopts it. Order is ABBA, so a
linear drift between the arms cancels. Both arms retire source pages and use the
same units, bytes, live CUDA tensors and settle.

Timing and profiling never share a pass. The timed passes run with no profiler,
so its overhead is not in the wall time or the CPU seconds. The profiled passes
that follow carry Python stacks and attribute ``aten::copy_`` to its call site.
Profile microseconds include the profiler's own cost: they locate the work, the
timed passes measure it.

Both Sparks' Netdata series come from ``tools/collect_row_netdata.py`` over the
whole window. Local power is a descriptive sample against the envelope. Clock
alignment and energy qualification stay on HOLD, and no work-per-joule claim is
made here.

``--cpu-dry-run`` is the CPU entry point of the same code (CEO D38): imports,
arguments, fixture, map binding and one pageable staged read of every unit, with
no CUDA and no pinned memory. It proves the harness, not the saving.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "prismaquant.pq2039_paired_copy_measure.v2"
ORDER = ("before", "after", "after", "before")
#: One BF16 expert projection of GLM-5.3-Flash: [I=2048, H=4096], 16 MiB.
EXPERT_ROWS, EXPERT_COLS = 2048, 4096
POWER_ENVELOPE_W = 140.0
#: Netdata's slowest required chart (GPU power) updates every ten seconds.
NETDATA_PAD_S = 12.0
SHARD = "model-00001-of-00001.safetensors"


# -- fixture ------------------------------------------------------------------

class Fixture:
    """One declared shard, one staged range per unit, one bound residency map."""

    def __init__(self, root, model, source, units, tensors, map_path, nbytes):
        self.root, self.model, self.source = root, model, source
        self.units, self.tensors = units, tensors
        self.map_path, self.unit_bytes = map_path, nbytes
        self.names = sorted(units)


def build_fixture(root, *, units, rows, cols, seed=0):
    """Write the declared shard, stage every tensor's range and write the map.

    The staged bytes equal the declared bytes, as PrismaBuild's mover leaves
    them. The tensors differ from one another, so a read that served the wrong
    unit's range would fail the equality check, not pass it.
    """
    import torch
    from safetensors.torch import save_file
    from tests.test_residency_shard_reader import (
        _header, _stage_range, _stage_root, _write_map)

    root = Path(root)
    model = root / "model"
    model.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {f"w{i}": torch.randn(rows, cols, generator=generator).to(torch.bfloat16)
               for i in range(units)}
    declared = model / SHARD
    save_file(tensors, str(declared))
    spans = _header(declared)
    stage = _stage_root(root)
    rows_ = []
    for name, tensor in tensors.items():
        start, end = spans[name]
        blob = tensor.view(torch.uint8).numpy().tobytes()
        assert len(blob) == end - start
        rows_.append((declared, start, end - start,
                      _stage_range(stage, declared, start, end - start, blob=blob)))
    map_path = _write_map(root, stage, rows_)
    return Fixture(
        root, model, {"tensors": {name: SHARD for name in tensors}},
        {name: {"source_tensor": name, "rows": rows, "cols": cols} for name in tensors},
        tensors, map_path, rows * cols * 2)


def bind_fixture(fixture, *, sdk_root=None):
    """Bind PQ's one SDK owner and the fixture's map; return what was bound.

    ``sdk_root`` names a sealed PrismaBuild source tree (the reviewed SDK
    pin); with none, whatever SDK the process already bound serves. The map is
    validated by that SDK, so an unbound or wrong-version SDK refuses here
    rather than reading the pool silently.
    """
    from prismaquant import staged_lease
    from prismaquant.residency_map import (
        ENV_VAR, bind_residency_manifest, reset_residency_resolver_for_tests,
        residency_resolver)
    from tests.test_residency_shard_reader import MANIFEST

    if sdk_root is not None:
        staged_lease.set_lease_helper_root(sdk_root)
    sdk = staged_lease.client_sdk()
    os.environ[ENV_VAR] = str(fixture.map_path)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    resolver = residency_resolver()
    declared = str(fixture.model / SHARD)
    if resolver is None or not resolver.stages(declared):
        raise RuntimeError("the fixture's residency map does not stage its shard")
    return {"sdk_root": sdk_root, "sdk_version": sdk.SDK_VERSION,
            "sdk_file": str(sdk.__file__), "map_path": str(fixture.map_path)}


def stage_counters():
    """The resolver's byte and fallback counters, for per-arm deltas."""
    from prismaquant.residency_map import residency_report
    report = residency_report() or {}
    return {key: report.get(key, 0) for key in
            ("bytes_from_stage", "bytes_from_pool", "range_hits", "fallback_count")}


def counter_delta(before, after):
    return {key: after[key] - before[key] for key in after}


# -- profile attribution ------------------------------------------------------
#
# torch 2.11 keeps no Python stack on a ``FunctionEvent`` (``event.stack`` is
# empty for eager code). With ``with_stack=True`` the Python frames live in the
# exported Chrome trace instead, as ``python_function`` events beside the
# ``cpu_op`` events. Each op is charged to the innermost frame that contains it
# on its own thread, then up the frame tree to the first frame outside torch.

_FRAME = re.compile(r"^(?P<file>.*)\((?P<line>\d+)\): (?P<func>.*)$")
#: Trace categories of host work an op's own time does not include.
_HOST_CALLS = ("cpu_op", "cuda_runtime", "cuda_driver")


def _foreign(file):
    return file.startswith("torch/") or "/torch/" in file or "site-packages" in file


def _user_site(frame, by_id):
    """``file:line function`` of ``frame`` or its nearest caller outside torch."""
    while frame is not None:
        found = _FRAME.match(frame["name"])
        if found is not None and not _foreign(found["file"]):
            return f"{Path(found['file']).name}:{found['line']} {found['func']}"
        frame = by_id.get(frame.get("args", {}).get("Python parent id"))
    return None


def _union_us(intervals):
    """Length of the union of ``(start, end)`` intervals."""
    total, reach = 0.0, None
    for start, end in sorted(intervals):
        if reach is None or start > reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


def trace_call_sites(events, op="aten::copy_"):
    """``op`` per Python call site, from a Chrome trace's event list.

    Every ``op`` event is charged to the innermost non-torch Python frame that
    contains it on its thread. ``self_cpu_us`` is the op's duration less the
    union of the host calls nested inside it, so a host ``memcpy`` into pinned
    memory shows as self time and the ``cudaMemcpyAsync`` enqueue of a
    host-to-device copy shows as ``runtime_us``. Rows come back sorted by self
    CPU, largest first. ``python_frames`` is how many frames the trace held, so
    an interpreter whose profiler records none says so instead of attributing
    everything to ``<no python frame>`` silently.
    """
    frames = {}
    host = {}
    ops = []
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category == "python_function":
            frames.setdefault(event["tid"], []).append(event)
        elif category in _HOST_CALLS:
            host.setdefault(event["tid"], []).append(event)
        if category == "cpu_op" and event["name"] == op:
            ops.append(event)
    by_id = {frame["args"]["Python id"]: frame
             for rows in frames.values() for frame in rows
             if "Python id" in frame.get("args", {})}
    for rows in frames.values():
        rows.sort(key=lambda frame: frame["ts"])
    starts = {tid: [frame["ts"] for frame in rows] for tid, rows in frames.items()}

    sites = {}
    for event in ops:
        begin, finish = event["ts"], event["ts"] + event["dur"]
        rows = frames.get(event["tid"], [])
        # The frames that start no later than the op, latest start first: the
        # first one still running when the op ends is the innermost.
        index = bisect.bisect_right(starts.get(event["tid"], []), begin) - 1
        while index >= 0 and rows[index]["ts"] + rows[index]["dur"] < finish:
            index -= 1
        owner = rows[index] if index >= 0 else None
        site = _user_site(owner, by_id) or "<no python frame>"
        nested = [(max(child["ts"], begin), min(child["ts"] + child["dur"], finish))
                  for child in host.get(event["tid"], [])
                  if child is not event and child["ts"] >= begin
                  and child["ts"] + child["dur"] <= finish]
        runtime = [(max(child["ts"], begin), min(child["ts"] + child["dur"], finish))
                   for child in host.get(event["tid"], [])
                   if child["cat"] != "cpu_op" and child["ts"] >= begin
                   and child["ts"] + child["dur"] <= finish]
        row = sites.setdefault(site, {"site": site, "count": 0, "total_cpu_us": 0.0,
                                      "self_cpu_us": 0.0, "runtime_us": 0.0})
        row["count"] += 1
        row["total_cpu_us"] += event["dur"]
        row["self_cpu_us"] += event["dur"] - _union_us(nested)
        row["runtime_us"] += _union_us(runtime)
    return {"python_frames": len(by_id),
            "rows": sorted(sites.values(), key=lambda row: -row["self_cpu_us"])}


def trace_events(profile, scratch, keep=None):
    """The finished profile's Chrome-trace events, optionally keeping the file."""
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(path))
        if keep is not None:
            Path(keep).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, keep)
        return json.loads(path.read_text())["traceEvents"]


def profile_totals(profile, ops):
    """Count and self CPU seconds of each named op over a finished profile.

    The same instrument as the aggregate the issue cites: ``FunctionEvent``
    self CPU time, summed per op name.
    """
    totals = {}
    for event in profile.events():
        if event.name in ops:
            row = totals.setdefault(event.name, {"count": 0, "self_cpu_s": 0.0})
            row["count"] += 1
            row["self_cpu_s"] += float(event.self_cpu_time_total) / 1e6
    return totals


# -- host instruments ---------------------------------------------------------

class PowerSampler:
    """GPU board power by ``nvidia-smi``, one reading a second, on the one
    sampler thread (``io_spans.PeriodicSampler``). Descriptive only."""

    def __init__(self, interval_s=1.0):
        from prismaquant.io_spans import PeriodicSampler
        self.samples, self.errors = [], []
        self._sampler = PeriodicSampler(self._tick, interval_s=interval_s,
                                        name="pq2039-power")

    def _tick(self):
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True)
            self.samples.append((time.time(), float(done.stdout.split()[0])))
        except Exception as error:  # noqa: BLE001 - counted, never raised
            self.errors.append(str(error))

    def __enter__(self):
        self._sampler.start()
        return self

    def __exit__(self, *_):
        self._sampler.stop(15)

    def window(self, start, end):
        values = [watts for stamp, watts in self.samples if start <= stamp <= end]
        return {"samples": len(values),
                "mean_w": statistics.fmean(values) if values else None,
                "max_w": max(values) if values else None,
                "envelope_fraction_mean": (statistics.fmean(values) / POWER_ENVELOPE_W
                                           if values else None)}


def clock_probe(host, *, rounds=3):
    """Peer clock minus ours, with the round trip that bounds it. HOLD input."""
    if host == socket.gethostname().split(".")[0]:
        return {"host": host, "local": True}
    rows = []
    for _ in range(rounds):
        sent = time.time()
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 "date +%s.%N"], check=True, capture_output=True, text=True,
                timeout=15).stdout.strip()
        except Exception as error:  # noqa: BLE001 - recorded, never raised
            return {"host": host, "error": str(error)}
        received = time.time()
        rows.append({"offset_s": float(out) - (sent + received) / 2,
                     "bound_s": (received - sent) / 2})
    return {"host": host, "rounds": rows}


def both_host_netdata(after, before):
    """Both Sparks' validated Netdata windows, or the reason one is missing."""
    try:
        from tools.collect_row_netdata import collect_window
        document = collect_window(after, before)
        return {"complete": True, "document": document,
                "clock_probes": [clock_probe(host) for host in document["hosts"]]}
    except BaseException as error:  # noqa: BLE001 - SystemExit included, recorded
        return {"complete": False, "error": f"{type(error).__name__}: {error}"}


def head_binding():
    """The git head this tree is, and whether any edit sits on top of it."""
    repo = Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], check=True,
                                  capture_output=True, text=True, timeout=30).stdout
        except Exception as error:  # noqa: BLE001 - recorded, never raised
            return f"unavailable: {error}"

    closures = {path.name: json.loads(path.read_text())
                for path in sorted(repo.glob(".pbrun-closure*.json"))}
    return {"head": git("rev-parse", "HEAD").strip(),
            "parent": git("rev-parse", "HEAD^").strip(),
            "dirty_sha256": hashlib.sha256(git("status", "--porcelain").encode()).hexdigest(),
            "pbrun_closure": closures}


# -- the arms -----------------------------------------------------------------

class Phases:
    """Accumulated seconds of the check's stages, one object per arm."""

    def __init__(self):
        self.seconds = {"read": 0.0, "prepare": 0.0, "launch": 0.0, "settle": 0.0}
        self.pinned_reads = self.pageable_reads = 0

    def reset(self):
        self.__init__()


def instrument(campaign, kind, phases):
    """Wrap the three seams of one arm; return the function that restores them.

    ``before`` withholds ``pinned_host`` from the real read, which is exactly
    the call the check made before #2039. Everything else is the shipped code,
    with a clock around each stage and a count of what the read returned.
    """
    names = ("_read_projected_unit", "_prepare_device_projected_check",
             "_launch_prepared_projected_check")
    originals = {name: getattr(campaign, name) for name in names}

    def read(name, unit, **kwargs):
        if kind == "before":
            kwargs["pinned_host"] = False
        start = time.perf_counter()
        weight, release = originals["_read_projected_unit"](name, unit, **kwargs)
        phases.seconds["read"] += time.perf_counter() - start
        if weight.is_pinned():
            phases.pinned_reads += 1
        else:
            phases.pageable_reads += 1
        return weight, release

    def prepare(*args, **kwargs):
        start = time.perf_counter()
        try:
            return originals["_prepare_device_projected_check"](*args, **kwargs)
        finally:
            phases.seconds["prepare"] += time.perf_counter() - start

    def launch(*args, **kwargs):
        start = time.perf_counter()
        try:
            return originals["_launch_prepared_projected_check"](*args, **kwargs)
        finally:
            phases.seconds["launch"] += time.perf_counter() - start

    campaign._read_projected_unit = read
    campaign._prepare_device_projected_check = prepare
    campaign._launch_prepared_projected_check = launch

    def restore():
        for name, original in originals.items():
            setattr(campaign, name, original)
    return restore


def run_pass(campaign, fixture, live, phases):
    """One serial check pass over every unit; returns wall seconds and verdicts."""
    import torch
    start = time.perf_counter()
    checks = [campaign._start_projected_unit_check(
        name, fixture.units[name], live=live[name], model_path=str(fixture.model),
        source=fixture.source, release_source_pages=True, source_authentication=None)
        for name in fixture.names]
    settling = time.perf_counter()
    outcomes = campaign._settle_projected_unit_checks(checks)
    torch.cuda.synchronize()
    end = time.perf_counter()
    phases.seconds["settle"] += end - settling
    return end - start, outcomes


def mismatch_pass(campaign, fixture, live, phases, picks):
    """Flip one element of each picked live tensor, check, then put it back.

    The check must name exactly the flipped units, in order. Both arms must
    name them with the same text.
    """
    flipped = {}
    for index in picks:
        tensor = live[fixture.names[index]].view(-1)
        flipped[index] = tensor[0].clone()
        tensor[0] = flipped[index] + 1
    try:
        _, outcomes = run_pass(campaign, fixture, live, phases)
    finally:
        for index, value in flipped.items():
            live[fixture.names[index]].view(-1)[0] = value
    named = [text for text in outcomes if text is not None]
    expected = [fixture.names[index] for index in sorted(picks)]
    ok = (len(named) == len(picks)
          and all(text.startswith(f"{name} (") for name, text in zip(expected, named)))
    return {"picks": [fixture.names[i] for i in picks], "verdicts": named, "ok": ok}


def timed_arm(campaign, fixture, live, kind, *, passes, profile_passes, power, trace_dir):
    """One arm: timed passes, then profiled passes, then the mismatch pass."""
    import resource
    import torch

    phases = Phases()
    restore = instrument(campaign, kind, phases)
    try:
        counters_before = stage_counters()
        torch.cuda.synchronize()
        usage_before, cpu_before = resource.getrusage(resource.RUSAGE_SELF), time.process_time()
        started = time.time()
        walls = []
        for _ in range(passes):
            wall, outcomes = run_pass(campaign, fixture, live, phases)
            if any(text is not None for text in outcomes):
                raise RuntimeError(f"{kind}: equal bytes reported a mismatch: {outcomes}")
            walls.append(wall)
        ended = time.time()
        cpu_s = time.process_time() - cpu_before
        usage = resource.getrusage(resource.RUSAGE_SELF)
        timed_phases = dict(phases.seconds)
        timed_reads = (phases.pinned_reads, phases.pageable_reads)
        counters_timed = counter_delta(counters_before, stage_counters())

        phases.reset()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, with_stack=True) as profile:
            for _ in range(profile_passes):
                run_pass(campaign, fixture, live, phases)
            torch.cuda.synchronize()
        keep = Path(trace_dir) / f"{kind}-{started:.0f}.json" if trace_dir else None
        sites = trace_call_sites(trace_events(profile, fixture.root, keep))
        totals = profile_totals(profile, {
            "aten::copy_", "aten::_to_copy", "aten::empty_strided", "aten::ne",
            "aten::any", "cudaMemcpyAsync", "cudaLaunchKernel"})

        phases.reset()
        mismatch = mismatch_pass(campaign, fixture, live, phases,
                                 picks=(1, len(fixture.names) - 1))
        counters_total = counter_delta(counters_before, stage_counters())
    finally:
        restore()
    reads = (passes + profile_passes + 1) * len(fixture.names)
    return {
        "kind": kind, "passes": passes, "units_per_pass": len(fixture.names),
        "epoch_start": started, "epoch_end": ended,
        "wall_s": walls, "wall_median_s": statistics.median(walls),
        "wall_mean_s": statistics.fmean(walls),
        "cpu_s_per_pass": cpu_s / passes,
        "minor_faults_per_pass": (usage.ru_minflt - usage_before.ru_minflt) / passes,
        "phase_s_per_pass": {key: value / passes for key, value in timed_phases.items()},
        "pinned_reads": timed_reads[0], "pageable_reads": timed_reads[1],
        "stage_timed": counters_timed, "stage_total": counters_total,
        "stage_expected_bytes": reads * fixture.unit_bytes,
        "power": power.window(started, ended),
        "profile": {"passes": profile_passes, "python_frames": sites["python_frames"],
                    "aten_copy_sites": sites["rows"], "ops": totals},
        "mismatch": mismatch,
    }


def paired_summary(arms):
    """Pool the two arms of each kind and compare the pools."""
    pools = {kind: [arm for arm in arms if arm["kind"] == kind] for kind in ("before", "after")}

    def pooled(kind, key):
        return statistics.fmean(arm[key] for arm in pools[kind])

    before, after = pooled("before", "wall_median_s"), pooled("after", "wall_median_s")
    summary = {
        "wall_median_s": {"before": before, "after": after,
                          "reduction_fraction": 1 - after / before},
        "cpu_s_per_pass": {"before": pooled("before", "cpu_s_per_pass"),
                           "after": pooled("after", "cpu_s_per_pass")},
        "minor_faults_per_pass": {"before": pooled("before", "minor_faults_per_pass"),
                                  "after": pooled("after", "minor_faults_per_pass")},
        "within_kind_spread_fraction": {
            kind: abs(pools[kind][0]["wall_median_s"] - pools[kind][1]["wall_median_s"])
            / pooled(kind, "wall_median_s") for kind in pools},
        "private_staging_copies": {kind: sum(arm["pageable_reads"] for arm in pools[kind])
                                   for kind in pools},
    }
    summary["cpu_s_per_pass"]["reduction_fraction"] = (
        1 - summary["cpu_s_per_pass"]["after"] / summary["cpu_s_per_pass"]["before"])
    return summary


def stage_gate(arms):
    """Every arm's every unit read came off the stage and none off the pool."""
    problems = []
    for arm in arms:
        total = arm["stage_total"]
        if (total["bytes_from_stage"] != arm["stage_expected_bytes"]
                or total["bytes_from_pool"] or total["fallback_count"]):
            problems.append({"arm": arm["kind"], **total,
                             "expected_bytes": arm["stage_expected_bytes"]})
        if not arm["mismatch"]["ok"]:
            problems.append({"arm": arm["kind"], "mismatch": arm["mismatch"]})
    verdicts = {json.dumps(arm["mismatch"]["verdicts"]) for arm in arms}
    if len(verdicts) != 1:
        problems.append({"verdicts_differ": sorted(verdicts)})
    return problems


# -- driver -------------------------------------------------------------------

def calibrate(campaign, fixture, live, *, arm_seconds, min_passes):
    """Warm both paths, then size every arm so the slower one lasts ``arm_seconds``."""
    medians = {}
    for kind in ("before", "after"):
        phases = Phases()
        restore = instrument(campaign, kind, phases)
        try:
            walls = [run_pass(campaign, fixture, live, phases)[0] for _ in range(4)]
        finally:
            restore()
        medians[kind] = statistics.median(walls[1:])
    passes = max(min_passes, math.ceil(arm_seconds / max(medians.values())))
    return passes, medians


def run_cuda(args, fixture, binding):
    import torch
    from prismaquant import tessera_campaign as campaign

    torch.cuda.set_device(0)
    live = {name: tensor.to("cuda") for name, tensor in fixture.tensors.items()}
    torch.cuda.synchronize()
    passes, medians = calibrate(
        campaign, fixture, live, arm_seconds=args.arm_seconds, min_passes=args.min_passes)
    passes = args.passes or passes
    arms = []
    with PowerSampler() as power:
        for kind in ORDER:
            arms.append(timed_arm(campaign, fixture, live, kind, passes=passes,
                                  profile_passes=args.profile_passes, power=power,
                                  trace_dir=args.trace_dir))
            print(f"[pq2039] {kind}: median {arms[-1]['wall_median_s'] * 1e3:.2f} ms/pass "
                  f"({passes} passes)", file=sys.stderr, flush=True)
    problems = stage_gate(arms)
    netdata = None
    if not args.no_netdata:
        wait = arms[-1]["epoch_end"] + NETDATA_PAD_S - time.time()
        if wait > 0:
            time.sleep(wait)
        netdata = both_host_netdata(arms[0]["epoch_start"] - NETDATA_PAD_S, time.time())
    return {"calibration_median_s": medians, "passes": passes, "order": list(ORDER),
            "arms": arms, "paired": paired_summary(arms), "stage_gate_problems": problems,
            "netdata": netdata, "power_errors": power.errors[:5],
            "power_samples_total": len(power.samples)}


def run_cpu_dry(fixture):
    """One pageable staged read of every unit, bytes compared, stage counted."""
    import torch
    from prismaquant.tessera_expert_projection import source_unit_weight

    before = stage_counters()
    for name in fixture.names:
        weight = source_unit_weight(fixture.model, fixture.source, fixture.units[name])
        if not torch.equal(weight, fixture.tensors[name]):
            raise RuntimeError(f"{name}: staged bytes differ from the source tensor")
        if weight.is_pinned():
            raise RuntimeError(f"{name}: a pageable read came back pinned")
    delta = counter_delta(before, stage_counters())
    problems = []
    if delta["bytes_from_stage"] != len(fixture.names) * fixture.unit_bytes:
        problems.append(f"stage served {delta['bytes_from_stage']} bytes")
    if delta["bytes_from_pool"] or delta["fallback_count"]:
        problems.append(f"pool or fallback reads: {delta}")
    return {"dry_run": True, "reads": len(fixture.names), "stage": delta,
            "stage_gate_problems": problems}


def environment():
    import torch
    out = {"python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "host": socket.gethostname().split(".")[0],
           "cpu_affinity": sorted(os.sched_getaffinity(0)),
           "torch_threads": torch.get_num_threads(),
           "env": {key: os.environ.get(key) for key in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "PRISMABUILD_ACTION_KEY")}}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
    return out


def observed_peaks():
    """What this process peaked at; PrismaBuild's receipt holds the cgroup's own."""
    import resource
    import torch
    out = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.cuda.is_available():
        out["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        out["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return out


def summary_text(result):
    """The record without the raw series: what the CAS-captured stdout carries."""
    slim = json.loads(json.dumps(result))
    netdata = slim.get("netdata") or {}
    document = netdata.pop("document", None)
    if document is not None:
        netdata["hosts"] = {host: {"charts": row["charts"],
                                   "points": {chart: s["points"] for chart, s in row["series"].items()}}
                            for host, row in document["hosts"].items()}
    return json.dumps(slim, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdk-root", help="sealed PrismaBuild SDK source tree that validates the map")
    parser.add_argument("--fixture-dir", default=os.environ.get("TMPDIR", "."),
                        help="scratch root for the fixture; a private subdirectory is made and removed")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--rows", type=int, default=EXPERT_ROWS)
    parser.add_argument("--cols", type=int, default=EXPERT_COLS)
    parser.add_argument("--passes", type=int, default=0,
                        help="passes per arm; 0 sizes them from --arm-seconds")
    parser.add_argument("--arm-seconds", type=float, default=30.0)
    parser.add_argument("--min-passes", type=int, default=20)
    parser.add_argument("--profile-passes", type=int, default=3)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--no-netdata", action="store_true")
    for name in ("cpus", "mem-gb", "gpu-memory-gb"):
        parser.add_argument(f"--declared-{name}", type=int, default=None,
                            help="the reservation the submitting pbrun command declared")
    parser.add_argument("--cpu-dry-run", action="store_true")
    args = parser.parse_args(argv)

    import torch
    if args.cpu_dry_run:
        args.units, args.rows, args.cols = min(args.units, 4), min(args.rows, 64), min(args.cols, 128)
    elif not torch.cuda.is_available():
        print(json.dumps({"skipped": True, "reason": "no CUDA device on this worker"}))
        return 2
    scratch = Path(args.fixture_dir) / f"pq2039-{os.getpid()}"
    try:
        fixture = build_fixture(scratch, units=args.units, rows=args.rows, cols=args.cols)
        binding = bind_fixture(fixture, sdk_root=args.sdk_root)
        body = run_cpu_dry(fixture) if args.cpu_dry_run else run_cuda(args, fixture, binding)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result = {"schema": SCHEMA, "issue": 2039, "binding": head_binding(),
              "environment": environment(), "sdk": binding,
              "fixture": {"units": args.units, "shape": [args.rows, args.cols],
                          "dtype": "bfloat16", "unit_bytes": fixture.unit_bytes,
                          "release_source_pages": True, "stage": "local-disk range files"},
              "reservations": {"declared": {"cpus": args.declared_cpus,
                                            "mem_gb": args.declared_mem_gb,
                                            "gpu_memory_gb": args.declared_gpu_memory_gb},
                               "cpu_affinity": sorted(os.sched_getaffinity(0)),
                               "observed_in_process": observed_peaks()},
              "hold": ["clock alignment across hosts stays on HOLD",
                       "energy integration and work per joule stay on HOLD; watts are "
                       f"descriptive samples against the {POWER_ENVELOPE_W:.0f} W envelope",
                       "no full-campaign, export, serving, KL or bpp claim"],
              **body}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_text(json.dumps(result))
        os.replace(temporary, out)
        result["result_file"] = {"path": str(out), "bytes": out.stat().st_size,
                                 "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    print(summary_text(result))
    return 1 if body["stage_gate_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
