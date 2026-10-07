#!/usr/bin/env python3
"""Measure one Stage B quantum's head intake without a GPU or the output root.

PQ #1010, principle 15. Runs the head intake a layer quantum performs before
its first GPU allocation -- the prepared completion, the campaign metadata
intake, the calibration draw, the PWC and the source-identity seed -- and
reports wall time, ``/proc/self/io`` and the ``/mnt/shared`` NFS client
operation counts around it. Nothing is written under the record's output
space: every path that the quantum would write there is redirected to
``--scratch``, a host-local directory the caller names.

``--mode walk`` is the historical intake (``load_measured_anchor_input`` over
the whole campaign). ``--mode slice`` reads the record's sealed head slice
instead (``joint_stage_b_head.load_quantum_head``).

``--mode scoped-walk`` walks only ``sorted(names)[lo:hi]`` of the census
roster (``--unit-scope lo:hi``, at most ``SCOPED_WALK_MAX_UNITS`` units) with
one explicit I/O worker count (``--head-walk-workers``). It is read-only: no
head checkpoint, no payload verification, no render synthesis. It exists to
measure the walk's I/O concurrency on a bounded sample (#1492, #1247). A guard
thread stops the run when the NFS READ round trip rises past
``--stop-read-rtt-ms``. The census-wide gates still run over the whole
campaign, so a one-unit run gives the constant cost to subtract.

The NFS counts come from ``/proc/self/mountstats``, which is per mount, not
per process: concurrent NFS traffic from other processes on the same box
lands in the same counters. The report records the box load beside them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

#: A scoped walk reads at most this many units. The dl380g10 pool crashed on
#: 2026-10-06, so no measurement may walk the complete campaign.
SCOPED_WALK_MAX_UNITS = 2000


def mount_ops(mount_point: str = "/mnt/shared") -> dict:
    """Per-operation NFS client counts for one mount (``ops`` column)."""
    from prismaquant.io_spans import read_mountstats

    row = read_mountstats().get(mount_point, {"bytes": None, "ops": {}})
    ops = {"_bytes": row["bytes"]} if row["bytes"] is not None else {}
    ops.update({name: fields[0] for name, fields in row["ops"].items()})
    return ops


def delta(after: dict, before: dict) -> dict:
    out = {}
    for key, value in after.items():
        if isinstance(value, list):
            prior = before.get(key, [0] * len(value))
            out[key] = [a - b for a, b in zip(value, prior)]
        else:
            out[key] = value - before.get(key, 0)
    return {k: v for k, v in out.items() if v not in (0, [0] * 8)}


def read_rtt(mount_point: str = "/mnt/shared") -> tuple:
    """Cumulative ``(READ ops, READ round-trip ms)`` for one NFS mount."""
    from prismaquant.io_spans import read_mountstats

    fields = read_mountstats().get(mount_point, {"ops": {}})["ops"].get("READ")
    # Columns: ops, transmissions, timeouts, bytes sent, bytes received,
    # queue ms, round-trip ms, execute ms.
    return (0, 0) if not fields else (fields[0], fields[6])


def interval_read_rtt_ms(before: tuple, after: tuple, *, min_ops: int = 20):
    """Mean READ round trip in ms between two samples, or ``None`` when the
    interval holds too few READ operations to mean anything."""
    ops = after[0] - before[0]
    if ops < min_ops:
        return None
    return (after[1] - before[1]) / ops


def install_stop_signal(report):
    """Turn ``SIGTERM`` into one stop report and an immediate exit.

    The pool watcher stops a run from outside with ``SIGTERM``. The handler
    prints what the run knew, then leaves without waiting for worker threads
    that block on the pool. Returns the handler so a test can call it.
    """
    def handler(signum, frame):
        report("SIGTERM")
        os._exit(75)

    signal.signal(signal.SIGTERM, handler)
    return handler


class GuardStop(Exception):
    """The READ round trip rose past the limit; the run is stopped."""


def start_read_guard(limit_ms: float, *, interval_s: float = 5.0, min_ops: int = 20,
                     sample=read_rtt, on_stop=None):
    """Watch the NFS READ round trip in a thread; call ``on_stop`` past the limit.

    Returns ``(stop_event, trace)``. ``trace`` lists every interval mean the
    guard saw, so the report shows what the guard watched.
    """
    stop, trace = threading.Event(), []

    def watch():
        before = sample()
        while not stop.wait(interval_s):
            after = sample()
            mean = interval_read_rtt_ms(before, after, min_ops=min_ops)
            trace.append(None if mean is None else round(mean, 3))
            before = after
            if mean is not None and mean > limit_ms:
                trace.append(f"STOP above {limit_ms} ms")
                if on_stop is not None:
                    on_stop(mean)
                return

    threading.Thread(target=watch, name="read-rtt-guard", daemon=True).start()
    return stop, trace


def scoped_walk_intake(config, *, scope, workers, metadata_memo=None):
    """Walk ``sorted(names)[lo:hi]`` with one explicit I/O worker count.

    Read-only by construction: no head checkpoint, no payload verification,
    existing renders only. The loader still runs every census-wide gate. The
    candidate overlay is left out of the inputs: its fence can hash wire
    payloads after stat drift whatever ``verify_payloads`` says (#1519), and
    this mode reads no payload. ``metadata_memo`` lets a sweep read the
    census-wide metadata once.
    """
    from prismaquant.tessera_joint_aura import load_measured_anchor_input
    from prismaquant.tessera_reader import load_declared_reader

    low, high = scope
    if not 0 <= low < high:
        raise SystemExit(f"empty or reversed unit scope {low}:{high}")
    if high - low > SCOPED_WALK_MAX_UNITS:
        raise SystemExit(
            f"unit scope {low}:{high} holds {high - low} units; the limit is "
            f"{SCOPED_WALK_MAX_UNITS} (the pool crashed on 2026-10-06)")
    reader = load_declared_reader(config.get("reader"))
    started = time.monotonic()
    inputs = {key: value for key, value in config["inputs"].items()
              if key != "candidate_overlay"}
    data = load_measured_anchor_input(
        inputs, reader=reader, synthesis_device="cpu",
        progress_phase=None, head_checkpoint=None, head_resume=False,
        require_existing_renders=True, verify_payloads=False,
        unit_scope=(low, high), head_walk_workers=workers,
        historical_encoder_reuse=config.get("historical_encoder_reuse"),
        metadata_memo=metadata_memo)
    return {"marks": {"anchor_input_s": time.monotonic() - started},
            "scope": [low, high], "scope_units": high - low,
            "candidate_overlay_left_out": "candidate_overlay" in config["inputs"],
            "head_walk_workers": data.head_walk_workers,
            "measured_cells": len(data.cells),
            "units_with_cells": len({name for name, _ in data.cells})}


def sweep_plan(start: int, slice_units: int, workers_order) -> list:
    """The disjoint scopes of one sweep, baseline first.

    The first scope holds one unit and gives the cost of the census-wide
    gates, which every load repeats. Each later scope is a fresh slice of
    ``slice_units`` units, so no run reads files an earlier run left in the
    client page cache. The whole sweep is one budget of
    ``SCOPED_WALK_MAX_UNITS`` units (#1492).
    """
    order = list(workers_order)
    if not order or slice_units <= 0 or start < 0:
        raise SystemExit("a sweep needs a start, a slice size and at least one worker count")
    total = 1 + slice_units * len(order)
    if total > SCOPED_WALK_MAX_UNITS:
        raise SystemExit(
            f"the sweep holds {total} units (1 baseline plus {len(order)} slices of "
            f"{slice_units}); the limit is {SCOPED_WALK_MAX_UNITS} for the whole sweep")
    scopes = [{"label": "baseline", "scope": (start, start + 1), "workers": order[0]}]
    low = start + 1
    for index, workers in enumerate(order):
        scopes.append({"label": f"run{index}", "scope": (low, low + slice_units),
                       "workers": workers})
        low += slice_units
    return scopes


def scoped_walk_sweep(config, *, start, slice_units, workers_order, walk=None):
    """Walk every scope of the sweep in one process, loading the metadata once.

    The baseline scope reads, hashes and validates the large metadata files
    from the pool. Later scopes reuse that state through one memo while the
    stat fences of the files hold. Each run reports its wall time and the NFS
    READ operations and round trip it caused.
    """
    import gc

    walk = scoped_walk_intake if walk is None else walk
    memo = {}
    runs = []
    for item in sweep_plan(start, slice_units, workers_order):
        before_ops, before_rtt = mount_ops(), read_rtt()
        began = time.monotonic()
        result = walk(config, scope=item["scope"], workers=item["workers"],
                      metadata_memo=memo)
        wall = time.monotonic() - began
        after_ops, after_rtt = mount_ops(), read_rtt()
        reads = after_rtt[0] - before_rtt[0]
        runs.append({"label": item["label"], "scope": list(item["scope"]),
                     "workers": item["workers"], "wall_s": round(wall, 3),
                     "read_ops": reads,
                     "read_rtt_ms_mean": None if reads == 0 else round(
                         (after_rtt[1] - before_rtt[1]) / reads, 3),
                     "nfs_ops": delta(after_ops, before_ops),
                     "units_with_cells": result["units_with_cells"],
                     "measured_cells": result["measured_cells"]})
        del result
        gc.collect()
    return {"sweep": {"start": start, "slice_units": slice_units,
                      "workers_order": list(workers_order),
                      "units_total": 1 + slice_units * len(list(workers_order))},
            "metadata_loads": memo.get("loads", 0),
            "candidate_overlay_left_out": "candidate_overlay" in config["inputs"],
            "runs": runs}


def load_record(path, sha256):
    import hashlib
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SystemExit(f"record digest mismatch at {path}")
    return json.loads(raw)


def walk_intake(config, *, prepared, plan_sha256, scratch):
    """The historical head: ``run_layer_quantum`` from the device policy to
    the source-identity seed, minus the GPU-only steps."""
    import pickle

    from prismaquant.aura_cost import _aura_source_sha256
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.stage_inputs import bound as _bound, same as _same
    from prismaquant.tessera_joint_aura import (
        _prepare_file_read_bound, seed_source_identity_cache,
        load_measured_anchor_input)
    from prismaquant.tessera_reader import load_declared_reader

    marks = {}
    started = time.monotonic()
    if config.get("stage_b_resource_policy") is not None:
        from prismaquant.joint_stageb_resources import verify_policy
        verify_policy(config["stage_b_resource_policy"])
    marks["device_policy_s"] = time.monotonic() - started
    header = json.loads(_bound(prepared, "prepared anchors").read_text())
    reader = load_declared_reader(config.get("reader"))
    implementation = _aura_source_sha256()
    _same(header.get("status"), "complete", "prepared completion")
    marks["prepared_s"] = time.monotonic() - started
    execution = config["execution"]
    data = load_measured_anchor_input(
        config["inputs"], reader=reader, synthesis_device="cpu",
        progress_phase=None,
        head_checkpoint=Path(scratch) / "checkpoints" / "head-walk",
        head_resume=False, require_existing_renders=True,
        verify_payloads=False,
        historical_encoder_reuse=config.get("historical_encoder_reuse"))
    marks["anchor_input_s"] = time.monotonic() - started
    _same(config["model"], data.census["model"], "requested source model")
    ids, calibration = load_calibration_input(
        config["calibration_input"]["path"],
        expected_sha256=config["calibration_input"]["sha256"],
        n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
    marks["calibration_s"] = time.monotonic() - started
    completion = json.loads(_bound(prepared, "prepared anchors").read_text())
    _same(completion["formats_by_qname"],
          {n: list(v) for n, v in data.formats_by_qname.items()},
          "prepared exact candidate roster")
    cache = pickle.loads(
        _bound(completion["production_cache"], "qualified PWC").read_bytes())
    if not isinstance(cache, ProductionWeightCache):
        raise RuntimeError("prepared cache is not ProductionWeightCache")
    _same(cache.metadata["inputs"], data.inputs, "prepared source bindings")
    marks["pwc_s"] = time.monotonic() - started
    if config.get("served_activation_policy") is not None:
        from prismaquant.joint_served_activation import activate_policy
        activate_policy(cache, config["served_activation_policy"])
    expected = {pair: cache.metadata["verified_cells"][pair]["render_file_sha256"]
                for pair in data.cells}
    bound = _prepare_file_read_bound(data, max_render_bytes=config["max_render_bytes"])
    cache.require_file_load_sha256(expected, max_file_bytes=bound)
    marks["render_bound_s"] = time.monotonic() - started
    seed_source_identity_cache(config, Path(scratch) / "run")
    marks["identity_seed_s"] = time.monotonic() - started
    return {"marks": marks, "units": len(data.formats_by_qname),
            "measured_cells": len(data.cells),
            "progress_units": len(data.formats_by_qname),
            "max_render_file_bytes": bound,
            "implementation_sha256": implementation}


def slice_intake(config, *, record, prepared, plan_sha256, scratch):
    """The slice-mode head (PQ #1010), as ``run_layer_quantum`` runs it.

    The projection backend identity is the one the completion records: the
    harness has no device to prewarm one, and the head only compares it.
    """
    from prismaquant.aura_cost import _aura_source_sha256
    from prismaquant.joint_stage_b_head import (
        load_quantum_head, read_prepared_head, read_quantum_head_slice)
    from prismaquant.tessera_reader import load_declared_reader

    marks = {}
    started = time.monotonic()
    head_slice, _, files = read_quantum_head_slice(
        config, record=record, prepared=prepared, plan_sha256=plan_sha256)
    marks["slice_s"] = time.monotonic() - started
    completion = read_prepared_head(files)
    marks["prepared_s"] = time.monotonic() - started
    reader = load_declared_reader(config.get("reader"))
    head = load_quantum_head(
        config, record=record, head_slice=head_slice, files=files,
        completion=completion, plan_sha256=plan_sha256,
        implementation_sha256=_aura_source_sha256(),
        reader_identity=None if reader is None else reader.identity,
        projection_backend=completion.get("projection_backend"))
    marks["head_s"] = time.monotonic() - started
    return {"marks": marks, "units": head.units,
            "measured_cells": head.measured_cells,
            "progress_units": head.progress_units,
            "max_render_file_bytes": head.max_render_file_bytes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("walk", "slice", "scoped-walk"), required=True)
    parser.add_argument("--quantum", type=Path, required=True)
    parser.add_argument("--quantum-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--scratch", type=Path, required=True,
                        help="host-local directory for everything the quantum "
                             "would have written under its output space")
    parser.add_argument("--allowed-tiers", default=None,
                        help="activate the staged-tier policy (slice mode)")
    parser.add_argument("--unit-scope", default=None, metavar="LO:HI",
                        help="scoped-walk: half-open slice of the sorted census "
                             f"roster, at most {SCOPED_WALK_MAX_UNITS} units")
    parser.add_argument("--head-walk-workers", type=int, default=None,
                        help="scoped-walk: explicit I/O worker count (1-16)")
    parser.add_argument("--sweep-start", type=int, default=None,
                        help="scoped-walk sweep: first roster index (replaces --unit-scope)")
    parser.add_argument("--slice-units", type=int, default=None,
                        help="scoped-walk sweep: units in each measured slice")
    parser.add_argument("--sweep-workers", default=None,
                        help="scoped-walk sweep: worker counts in run order, e.g. 4,1,8,2")
    parser.add_argument("--stop-read-rtt-ms", type=float, default=40.0,
                        help="scoped-walk: stop when the NFS READ round trip "
                             "exceeds this mean over a guard interval")
    parser.add_argument("--guard-interval-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    if str(args.scratch).startswith("/mnt/shared"):
        parser.error("--scratch must be host-local, never the pool")
    scope, sweep = None, None
    sweep_args = (args.sweep_start, args.slice_units, args.sweep_workers)
    if args.mode == "scoped-walk":
        from prismaquant.tessera_joint_aura import (
            HEAD_WALK_MAX_WORKERS, parse_unit_scope)
        single = args.unit_scope is not None or args.head_walk_workers is not None
        swept = any(value is not None for value in sweep_args)
        if single == swept:
            parser.error("scoped-walk needs either --unit-scope with --head-walk-workers, "
                         "or --sweep-start with --slice-units and --sweep-workers")
        if not all(math.isfinite(value) and value > 0
                   for value in (args.stop_read_rtt_ms, args.guard_interval_s)):
            parser.error("guard limits must be positive and finite")
        if swept:
            if any(value is None for value in sweep_args):
                parser.error("a sweep needs --sweep-start, --slice-units and --sweep-workers")
            try:
                order = [int(item) for item in args.sweep_workers.split(",")]
            except ValueError:
                parser.error("--sweep-workers must be integers separated by commas")
            if not order or any(not 1 <= item <= HEAD_WALK_MAX_WORKERS for item in order):
                parser.error(f"--sweep-workers must be counts in 1:{HEAD_WALK_MAX_WORKERS}")
            try:
                sweep_plan(args.sweep_start, args.slice_units, order)
            except SystemExit as exc:
                parser.error(str(exc.code))
            sweep = {"start": args.sweep_start, "slice_units": args.slice_units,
                     "order": order}
        else:
            if args.unit_scope is None or args.unit_scope.endswith(":"):
                parser.error("scoped-walk needs --unit-scope LO:HI with an explicit end")
            try:
                scope = parse_unit_scope(args.unit_scope)
            except (ValueError, TypeError) as exc:
                parser.error(f"--unit-scope: {exc}")
            if scope is None or scope[1] - scope[0] > SCOPED_WALK_MAX_UNITS:
                parser.error(f"--unit-scope must hold at most {SCOPED_WALK_MAX_UNITS} units")
            if args.head_walk_workers is None or not 1 <= args.head_walk_workers <= HEAD_WALK_MAX_WORKERS:
                parser.error(f"scoped-walk needs --head-walk-workers in 1:{HEAD_WALK_MAX_WORKERS}")
    elif (args.unit_scope is not None or args.head_walk_workers is not None
          or any(value is not None for value in sweep_args)):
        parser.error("the scope, worker and sweep options belong to --mode scoped-walk")
    args.scratch.mkdir(parents=True, exist_ok=True)

    from prismaquant.io_spans import read_proc_io, read_proc_status
    from prismaquant.tessera_joint_aura import load_joint_anchor_plan as _load_plan

    record = load_record(args.quantum, args.quantum_sha256)
    # The projection-runtime qualification gates GPU arithmetic, which
    # the head intake never runs; the plan's inputs are checked the same.
    config = _load_plan(args.plan, args.plan_sha256, projection_runtime=False)
    prepared = {"path": record["campaign"]["prepared_path"],
                "sha256": record["campaign"]["prepared_sha256"]}
    if args.allowed_tiers:
        from prismaquant.staged_tier_policy import activate_staged_tier_policy
        activate_staged_tier_policy(args.allowed_tiers)
    load_before = os.getloadavg()
    io_before, nfs_before = read_proc_io(), mount_ops()
    wall_before, cpu_before = time.monotonic(), time.process_time()
    guard_trace = None
    stopped = {"by_guard": False, "mean_ms": None}
    if args.mode == "scoped-walk":
        def on_stop(mean):
            # Report what was seen, then leave at once: a worker thread blocked
            # on the pool must not hold the process while latency is high.
            stopped.update(by_guard=True, mean_ms=round(mean, 3))
            print("STAGE_B_HEAD_PROFILE_STOPPED " + json.dumps(
                {"limit_ms": args.stop_read_rtt_ms, "mean_ms": stopped["mean_ms"],
                 "trace": guard_trace,
                 "scope": None if scope is None else list(scope),
                 "workers": args.head_walk_workers, "sweep": sweep}, sort_keys=True), flush=True)
            os._exit(75)
        guard_stop, guard_trace = start_read_guard(
            args.stop_read_rtt_ms, interval_s=args.guard_interval_s, on_stop=on_stop)

        def sigterm_report(reason):
            print("STAGE_B_HEAD_PROFILE_STOPPED " + json.dumps(
                {"reason": reason, "trace": guard_trace,
                 "scope": None if scope is None else list(scope),
                 "workers": args.head_walk_workers, "sweep": sweep}, sort_keys=True),
                flush=True)

        install_stop_signal(sigterm_report)
    if args.mode == "walk":
        result = walk_intake(config, prepared=prepared,
                             plan_sha256=args.plan_sha256, scratch=args.scratch)
    elif args.mode == "scoped-walk":
        if sweep is not None:
            result = scoped_walk_sweep(config, start=sweep["start"],
                                       slice_units=sweep["slice_units"],
                                       workers_order=sweep["order"])
        else:
            result = scoped_walk_intake(config, scope=scope, workers=args.head_walk_workers)
        guard_stop.set()
        result["guard"] = {"limit_ms": args.stop_read_rtt_ms,
                           "interval_s": args.guard_interval_s, "trace": guard_trace}
    else:
        result = slice_intake(config, record=record, prepared=prepared,
                              plan_sha256=args.plan_sha256, scratch=args.scratch)
    wall = time.monotonic() - wall_before
    cpu = time.process_time() - cpu_before
    io_after, nfs_after = read_proc_io(), mount_ops()
    journal = args.scratch / "checkpoints" / "head-walk"
    written = {"files": 0, "bytes": 0}
    if journal.exists():
        for path in journal.rglob("*"):
            if path.is_file():
                written["files"] += 1
                written["bytes"] += path.stat().st_size
    peak = read_proc_status().get("VmHWM")
    report = {
        "schema": "prismaquant.stage_b_head_profile.v1",
        "mode": args.mode, "layer": record["layer"],
        "quantum_sha256": args.quantum_sha256, "host": os.uname().nodename,
        "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
        "peak_rss_kib": None if peak is None else peak // 1024,
        "loadavg_before": load_before, "loadavg_after": os.getloadavg(),
        "proc_io": delta(io_after, io_before),
        "nfs_ops_mnt_shared": delta(nfs_after, nfs_before),
        "nfs_ops_total": sum(v for k, v in delta(nfs_after, nfs_before).items()
                             if not k.startswith("_")),
        "local_journal_written": written,
        **result,
    }
    print("STAGE_B_HEAD_PROFILE " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
