#!/usr/bin/env python3
"""One-shot GPU profile of the production CB encode, budget-conscious.

Runs ONE real layer-0 Linear per format and reports:
  * bit-identity of the replica vs the production shard row,
  * phase attribution (wall + CUDA-event time) for every hot function,
  * call counts and tensor shapes per phase,
  * torch.profiler kernel table (self CUDA time, launch counts),
  * host-gap estimate: wall - sum(kernel self time).

Designed to be run under a lock while the production container timeshares.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import torch

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "cb_encode_replica",
    str(Path(__file__).resolve().parent / "cb_encode_replica.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)

STATS: dict[str, dict] = collections.defaultdict(
    lambda: {"n": 0, "wall": 0.0, "shapes": collections.Counter()})
_DEPTH = [0]


def _wrap(mod, name, label=None, shape_fn=None):
    fn = getattr(mod, name)
    label = label or name

    def inner(*a, **kw):
        # Only time at the outermost nesting level per label to avoid
        # double counting; every label here is a leaf or measured alone.
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        s = STATS[label]
        s["n"] += 1
        s["wall"] += dt
        if shape_fn is not None:
            try:
                s["shapes"][shape_fn(*a, **kw)] += 1
            except Exception:
                pass
        return out

    setattr(mod, name, inner)
    return fn


def install_probes():
    from prismaquant import nvfp4_cb_formats as F

    orig = {}
    orig["_stream_moments"] = _wrap(
        F, "_stream_moments", "moments/_stream_moments",
        shape_fn=lambda x, ws, t: (tuple(x.shape), tuple(t.shape)))
    orig["_score_min_batched"] = _wrap(
        F, "_score_min_batched", "score/min_batched",
        shape_fn=lambda A, B, s: (tuple(A.shape), tuple(s.shape)))
    orig["_score_minargmin_batched"] = _wrap(
        F, "_score_minargmin_batched", "score/minargmin_batched",
        shape_fn=lambda A, B, s: (tuple(A.shape), tuple(s.shape)))
    orig["_score_argmin"] = _wrap(
        F, "_score_argmin", "score/argmin",
        shape_fn=lambda A, B, s: (tuple(A.shape), tuple(s.shape)))
    orig["_score_min"] = _wrap(
        F, "_score_min", "score/min",
        shape_fn=lambda A, B, s: (tuple(A.shape), tuple(s.shape)))
    orig["_vq_assign"] = _wrap(
        F, "_vq_assign", "exact/_vq_assign",
        shape_fn=lambda x, cb, wq: (tuple(x.shape), tuple(cb.shape)))
    orig["_two_tier_window"] = _wrap(F, "_two_tier_window", "host/_two_tier_window")
    orig["_calibrate_m2_used"] = _wrap(
        F, "_calibrate_m2_used", "pilot/_calibrate_m2_used")
    return orig


def phase_probes():
    """Coarser phase timers that wrap whole encoders (installed separately)."""
    from prismaquant import nvfp4_cb_formats as F
    _wrap(F, "_eval_candidate", "exact/_eval_candidate",
          shape_fn=lambda w2d, wq, s, grid, mode, cb: (tuple(w2d.shape), grid))
    _wrap(F, "_moment_err_groups_batched", "score/err_groups_batched",
          shape_fn=lambda moms, s_g, v: tuple(s_g.shape))
    _wrap(F, "_argmin_from_moments", "score/argmin_from_moments")
    _wrap(F, "_scan_and_assign", "score/scan_and_assign")
    _wrap(F, "_chunk_moments", "moments/_chunk_moments")


def report_stats(tag, total_wall):
    rows = sorted(STATS.items(), key=lambda kv: -kv[1]["wall"])
    print(f"\n--- phase attribution [{tag}]  total={total_wall:.3f}s ---")
    print(f"{'phase':38s} {'n':>7s} {'wall_s':>9s} {'%':>6s} {'us/call':>9s}")
    for k, v in rows:
        print(f"{k:38s} {v['n']:7d} {v['wall']:9.3f} "
              f"{100 * v['wall'] / max(total_wall, 1e-9):6.1f} "
              f"{1e6 * v['wall'] / max(v['n'], 1):9.1f}")
    for k, v in rows:
        if v["shapes"]:
            top = v["shapes"].most_common(4)
            print(f"  {k} shapes: {top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--outdir", default="/home/rob/dq-runs/dsv4-flash-0731/encoder-profile")
    ap.add_argument("--torch-profiler", action="store_true")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    R.apply_prod_env()

    w = R.load_expert_weight(args.layer, args.expert, args.proj)
    cw = R.load_col_weights(args.layer, args.expert, args.proj)
    row = R.shard_row(args.layer, args.expert, args.proj)
    print(f"device={torch.cuda.get_device_name(0)} weight={tuple(w.shape)}")

    # CLEAN pass first: the probes add a sync per call, so the unprobed
    # wall is the only honest per-format cost.
    results = {}
    clean = {}
    for fmt in args.formats.split(","):
        R.encode_one(w, cw, fmt, device="cuda")           # warm
        t0 = time.perf_counter()
        for _ in range(args.reps):
            _, mse_c, _ = R.encode_one(w, cw, fmt, device="cuda")
        clean[fmt] = (time.perf_counter() - t0) / args.reps
        print(f"[clean] {fmt:14s} {clean[fmt]:.3f}s  mse={mse_c!r}")
    results["__clean_wall_s__"] = clean

    install_probes()
    if args.phases:
        phase_probes()

    for fmt in args.formats.split(","):
        STATS.clear()
        # warm: one pass to trigger torch.compile / lattice load, untimed.
        _, mse0, dt0 = R.encode_one(w, cw, fmt, device="cuda")
        STATS.clear()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            w_hat, mse, dt = R.encode_one(w, cw, fmt, device="cuda")
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.reps
        ref = row[fmt]["weight_mse"]
        exact = (mse == ref)
        print(f"\n=== {fmt} ===")
        print(f"warm_pass={dt0:.3f}s probed={wall:.3f}s "
              f"clean={clean.get(fmt, float('nan')):.3f}s")
        print(f"weight_mse={mse!r} shard={ref!r} EXACT={exact} "
              f"rel={abs(mse - ref) / max(ref, 1e-30):.3e}")
        report_stats(fmt, wall)
        results[fmt] = {
            "weight_mse": mse, "shard_weight_mse": ref, "exact": exact,
            "wall_s": wall,
            "phases": {k: {"n": v["n"], "wall": v["wall"],
                           "shapes": {str(s): c for s, c in v["shapes"].items()}}
                       for k, v in STATS.items()},
        }

    if args.torch_profiler:
        fmt = args.formats.split(",")[0]
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            R.encode_one(w, cw, fmt, device="cuda")
        key = prof.key_averages()
        tbl = key.table(sort_by="self_device_time_total", row_limit=30)
        print(f"\n=== torch.profiler kernels [{fmt}] ===\n{tbl}")
        (outdir / f"torchprof_{fmt}.txt").write_text(tbl)
        prof.export_chrome_trace(str(outdir / f"trace_{fmt}.json"))
        tot_dev = sum(getattr(e, "self_device_time_total", 0) for e in key)
        results[f"{fmt}__profiler"] = {"total_self_device_us": tot_dev}
        print(f"sum self device time = {tot_dev / 1e6:.3f} s")

    (outdir / "phase_profile.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {outdir / 'phase_profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
