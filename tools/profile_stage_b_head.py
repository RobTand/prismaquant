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

The NFS counts come from ``/proc/self/mountstats``, which is per mount, not
per process: concurrent NFS traffic from other processes on the same box
lands in the same counters. The report records the box load beside them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def mount_ops(mount_point: str = "/mnt/shared") -> dict:
    """Per-operation NFS client counts for one mount (``ops`` column)."""
    ops: dict = {}
    current = None
    with open("/proc/self/mountstats") as handle:
        for line in handle:
            if line.startswith("device "):
                parts = line.split()
                current = parts[4] if len(parts) > 4 else None
                continue
            if current != mount_point:
                continue
            stripped = line.strip()
            if ":" not in stripped or stripped.startswith(("device", "opts", "age", "caps", "sec", "events", "bytes", "RPC", "xprt", "nfsv", "per-op")):
                if stripped.startswith("bytes:"):
                    ops["_bytes"] = [int(v) for v in stripped.split()[1:]]
                continue
            name, _, rest = stripped.partition(":")
            fields = rest.split()
            if fields and fields[0].isdigit():
                ops[name] = int(fields[0])
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
    from prismaquant.tessera_joint_aura import (
        _bound, _prepare_file_read_bound, _same, _seed_source_identity_cache,
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
    _seed_source_identity_cache(config, Path(scratch) / "run")
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
    parser.add_argument("--mode", choices=("walk", "slice"), required=True)
    parser.add_argument("--quantum", type=Path, required=True)
    parser.add_argument("--quantum-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--scratch", type=Path, required=True,
                        help="host-local directory for everything the quantum "
                             "would have written under its output space")
    parser.add_argument("--allowed-tiers", default=None,
                        help="activate the staged-tier policy (slice mode)")
    args = parser.parse_args(argv)
    if str(args.scratch).startswith("/mnt/shared"):
        parser.error("--scratch must be host-local, never the pool")
    args.scratch.mkdir(parents=True, exist_ok=True)
    from prismaquant.io_spans import read_proc_io
    from prismaquant.tessera_joint_aura import _load_plan

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
    if args.mode == "walk":
        result = walk_intake(config, prepared=prepared,
                             plan_sha256=args.plan_sha256, scratch=args.scratch)
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
    with open("/proc/self/status") as handle:
        peak = next((line.split()[1] for line in handle
                     if line.startswith("VmHWM:")), None)
    report = {
        "schema": "prismaquant.stage_b_head_profile.v1",
        "mode": args.mode, "layer": record["layer"],
        "quantum_sha256": args.quantum_sha256, "host": os.uname().nodename,
        "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
        "peak_rss_kib": int(peak) if peak else None,
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
