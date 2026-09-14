#!/usr/bin/env python3
"""Sweep the work per apply of prepared Tessera native dense cells until the GB10 is loaded.

PQ #563 priced one prefill point per cell (M=512) and found fp8 x fp8 ahead of
fp4 x fp4 by about 1.04x. That measurement drew 8.15 W of a 140 W envelope and
fp4's prefill/decode ratio was 1.01x across a 512-fold change in M, so it priced
per-apply dispatch overhead, not arithmetic. This driver keeps the arms, the
wires and the session identical and moves the only axis that was never moved:
the number of rows in one apply.

Scope, stated plainly because it is the cost of reaching large M:

* The panel's prefill row count is capped by ``prepare_native_inputs`` at the
  retained calibration activations, so a priced reference output exists only at
  M=512. Every M above 512 is a **timing-only** point whose input is the frozen
  512-row prefill input tiled ``M/512`` times.
* Numerics are anchored, never widened: at M=512 the output and the
  activation-side QDQ are compared against the frozen references at the panel's
  own ``atol=rtol=0.015625``; at every M the replica block ``y[i*512:(i+1)*512]``
  is compared against ``y[:512]`` and against the frozen reference, which is how
  a shape-dependent silent ``_scaled_mm`` miscompute would show.
* The route is re-read at every M and must still say ``served`` with the panel's
  own contract and the ``M{m}:N{n}:K{k}`` shape this apply actually ran.

Two instruments, because they answer different questions: CUDA events time one
complete apply, and a sustained apply loop with an NVML sampler reads power
against the 140 W envelope so the table can say whether the device was ever
actually loaded. An optional Torch profiler pass runs last, after every timed
and powered point, and reports per-kernel self device time and kernel names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

SCHEMA = "prismaquant.native_prefill_load_sweep.v1"
ENVELOPE_W = 140.0
BASE_ROWS = 512
RUNTIME_INVARIANT = ("image", "image_declaration", "execution", "arithmetic",
                     "versions", "gpu", "source")


def utc(when=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bootstrap_median(samples, *, resamples=10000, confidence=0.95, seed=20260913):
    """Percentile bootstrap of the median over this row's own samples."""
    values = sorted(float(v) for v in samples)
    n = len(values)
    if n < 3:
        raise ValueError("bootstrap needs at least three samples")
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        draw = sorted(values[int(rng.random() * n)] for _ in range(n))
        medians.append(draw[n // 2] if n % 2 else 0.5 * (draw[n // 2 - 1] + draw[n // 2]))
    medians.sort()
    low = medians[int((1.0 - confidence) / 2.0 * resamples)]
    high = medians[min(resamples - 1, int((1.0 + confidence) / 2.0 * resamples))]
    return {"median_ms": statistics.median(values), "low_ms": low, "high_ms": high,
            "min_ms": values[0], "max_ms": values[-1], "samples": n,
            "resamples": resamples, "confidence": confidence}


class PowerSeries:
    """One long-lived NVML reader. No per-sample fork on the pinned cores."""

    def __init__(self, interval_s=0.1):
        self.interval_s = float(interval_s)
        self.samples = []
        self.backend = None
        self.error = None
        self.uuid = None
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None
        self._process = None

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetUUID(self._handle)
            self.uuid = raw.decode() if isinstance(raw, bytes) else str(raw)
            self._nvml = pynvml
            self.backend = "pynvml"
        except Exception as exc:                                  # pragma: no cover - environment
            self.error = f"pynvml unavailable: {exc}"
            self.backend = "nvidia_smi_stream"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        if self.backend == "pynvml":
            nvml, handle = self._nvml, self._handle
            while not self._stop.is_set():
                try:
                    watts = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    self.samples.append({"t": time.time(), "watts": watts,
                                         "sm_mhz": int(clock), "temperature_c": int(temperature)})
                except Exception as exc:                          # pragma: no cover - environment
                    self.error = f"pynvml sample failed: {exc}"
                    break
                self._stop.wait(self.interval_s)
            return
        command = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu",
                   "--format=csv,noheader,nounits", "-lms", str(int(self.interval_s * 1000))]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
        except Exception as exc:                                  # pragma: no cover - environment
            self.error = f"{self.error}; nvidia-smi unavailable: {exc}"
            return
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                self.samples.append({"t": time.time(), "watts": float(parts[0]),
                                     "sm_mhz": int(float(parts[1])), "temperature_c": int(float(parts[2]))})
            except ValueError:
                continue

    def stop(self):
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.backend == "pynvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:                                     # pragma: no cover - environment
                pass

    def window(self, start, end, *, settle_s):
        """Summarise one sustained window, discarding the NVML meter's lag."""
        usable = [s for s in self.samples if start + settle_s <= s["t"] <= end]
        if not usable:
            return {"status": "no_samples", "backend": self.backend, "error": self.error,
                    "window_utc": [utc(start), utc(end)], "settle_s": settle_s}
        watts = [s["watts"] for s in usable]
        mean = sum(watts) / len(watts)
        return {"status": "observed", "backend": self.backend, "error": self.error,
                "window_utc": [utc(start), utc(end)], "settle_s": settle_s,
                "samples": len(usable), "mean_w": mean, "min_w": min(watts), "max_w": max(watts),
                "median_w": statistics.median(watts), "envelope_w": ENVELOPE_W,
                "envelope_fraction": mean / ENVELOPE_W,
                "sm_mhz_mean": sum(s["sm_mhz"] for s in usable) / len(usable),
                "temperature_c_max": max(s["temperature_c"] for s in usable)}


def render_table(report):
    """One markdown row per (M, arm): time, power against the envelope, work per joule.

    Rows are grouped by M so the reader compares arms measured back to back, and
    every interval is a percentile bootstrap over that row's own CUDA-event
    samples -- never a pooled or shared interval.
    """
    header = ("| M | unit | format | median ms | 95% CI ms | wall ms/apply | W | frac of 140 W "
              "| TFLOP/s | GFLOP/J |")
    lines = [header, "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in report["rows"]:
        for name, cell in report["cells"].items():
            point = cell["points"].get(str(m))
            if point is None:
                continue
            timing, power = point["timing"], point["sustained"]
            watts = power.get("mean_w")
            lines.append("| {m} | {unit} | {fmt} | {median:.6f} | [{low:.6f}, {high:.6f}] "
                         "| {wall} | {watts} | {frac} | {tflops} | {joule} |".format(
                m=m, unit=cell["unit"].rsplit(".", 1)[-1], fmt=cell["format"],
                median=timing["median_ms"], low=timing["low_ms"], high=timing["high_ms"],
                wall=f"{power['wall_ms_per_apply']:.6f}" if "wall_ms_per_apply" in power else "-",
                watts=f"{watts:.2f}" if watts else "-",
                frac=f"{power['envelope_fraction']:.3f}" if watts else "-",
                tflops=f"{power['achieved_tflop_s']:.2f}" if "achieved_tflop_s" in power else "-",
                joule=f"{power['gflop_per_joule']:.2f}" if "gflop_per_joule" in power else "-"))
    return "\n".join(lines)


def load_cell(directory, torch, load_file):
    cell = Path(directory)
    request = json.loads((cell / "request.json").read_text())
    inputs = json.loads((cell / "inputs.json").read_text())
    record = json.loads((cell / request["wire_record_path"]).read_text())
    blob = (cell / request["wire_path"]).read_bytes()
    tensors = load_file(str(cell / request["tensors_path"]), device="cuda")
    return {"name": cell.name, "path": str(cell), "request": request, "inputs": inputs,
            "record": record, "blob": blob, "tensors": tensors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--cells", required=True, help="comma-separated prepared cell directory names")
    parser.add_argument("--rows", required=True, help="comma-separated M values; each a multiple of 512")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--power-seconds", type=float, default=6.0)
    parser.add_argument("--power-settle-seconds", type=float, default=1.5)
    parser.add_argument("--power-interval-ms", type=int, default=100)
    parser.add_argument("--profile", action="store_true", help="final Torch profiler pass, after all timing")
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args(argv)

    # Argument shape is checked before anything is imported, so a bad row count
    # is refused on any box rather than after a CUDA context exists.
    names = [name for name in args.cells.split(",") if name]
    rows = [int(value) for value in args.rows.split(",") if value]
    for m in rows:
        if m % BASE_ROWS or m < BASE_ROWS:
            raise SystemExit(f"M={m} is not a positive multiple of {BASE_ROWS}")
    if not names or not rows:
        raise SystemExit("the sweep needs at least one cell and one row count")

    import torch
    from safetensors.torch import load_file
    from experiments.bench_native_operator import (compare_tensors, identity_sha256,
                                                   native_runtime_context, observe_runtime,
                                                   prepare_native_operator, represented_native_input,
                                                   tensor_identity, time_apply)

    report = {"schema": SCHEMA, "started_utc": utc(),
              "scope": ("timing_only_above_%d_rows_tiled_from_the_frozen_prefill_input" % BASE_ROWS),
              "base_rows": BASE_ROWS, "rows": rows, "cells": {}, "runtime": None,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "warmup_iterations": args.warmup_iterations, "iterations": args.iterations,
              "power": {"seconds": args.power_seconds, "settle_seconds": args.power_settle_seconds,
                        "interval_ms": args.power_interval_ms, "envelope_w": ENVELOPE_W}}

    power = PowerSeries(interval_s=args.power_interval_ms / 1000.0).start()
    try:
        with native_runtime_context():
            from tessera.serving.telemetry import read_route
            loaded = []
            for name in names:
                cell = load_cell(args.cells_root / name, torch, load_file)
                request, inputs = cell["request"], cell["inputs"]
                prepared = prepare_native_operator(
                    cell["blob"], cell["record"], cell["tensors"]["source_weight"],
                    cell["tensors"]["rendered_weight"], unit=request["unit"],
                    format_name=request["format"], runtime_image=request["runtime_image"],
                    input_global_scale=request.get("input_global_scale"),
                    execution=request["execution"])
                cell["prepared"] = prepared
                # One process prepares every arm, so the mapped-library roster
                # legitimately grows as the first fp4/fp8/bf16 owner loads its
                # dependencies. Everything that identifies the run -- image,
                # GPU, versions, arithmetic settings, Tessera source bytes --
                # must be identical across arms, and is checked as such.
                invariant = {key: prepared["runtime"][key] for key in RUNTIME_INVARIANT}
                if report["runtime"] is None:
                    report["runtime"] = prepared["runtime"]
                    report["runtime_invariant"] = invariant
                    report["runtime_invariant_sha256"] = identity_sha256(invariant)
                elif invariant != report["runtime_invariant"]:
                    raise SystemExit(f"{name}: runtime identity differs inside one session")
                report["cells"][name] = {
                    "unit": request["unit"], "format": request["format"],
                    "shape": list(inputs["shape"]),
                    "wire_bytes": inputs["wire"]["blob_bytes"],
                    "wire_sha256": inputs["wire"]["blob_sha256"],
                    "activation_contract": prepared["operator"]["activation_contract"],
                    "declared_route": prepared["operator"]["declared_route"],
                    "runtime_sha256": identity_sha256(prepared["runtime"]),
                    "native_libraries": len(prepared["runtime"]["native_libraries"]),
                    "numerics": inputs["numerics"], "anchor": None, "points": {}}
                loaded.append(cell)

            # Anchor: the only M with an independently priced reference output.
            with torch.inference_mode():
                for cell in loaded:
                    name, prepared = cell["name"], cell["prepared"]
                    layer, method = prepared["layer"], prepared["method"]
                    numerics = cell["inputs"]["numerics"]
                    x = cell["tensors"]["prefill.input"]
                    qdq = represented_native_input(layer, x)
                    output = method.apply(layer, x)
                    torch.cuda.synchronize()
                    route = read_route(layer)
                    anchor = {
                        "m": BASE_ROWS,
                        "qdq_numerics": compare_tensors(qdq, cell["tensors"]["prefill.reference_qdq"], **numerics),
                        "numerics": compare_tensors(output, cell["tensors"]["prefill.reference_output"], **numerics),
                        "route": route,
                        "reference_output": tensor_identity(cell["tensors"]["prefill.reference_output"]),
                        "input": tensor_identity(x)}
                    del qdq, output
                    report["cells"][name]["anchor"] = anchor
                    if anchor["numerics"]["status"] != "passed" or anchor["qdq_numerics"]["status"] != "passed":
                        raise SystemExit(f"{name}: anchor numerics refused; tolerances are not widened")
                    if route.get("state") != "served" or route.get("reason") is not None:
                        raise SystemExit(f"{name}: anchor route is not served")

            # Sweep: M outer, arms inner, so clock and thermal drift is shared.
            for m in rows:
                reps = m // BASE_ROWS
                for cell in loaded:
                    name, prepared = cell["name"], cell["prepared"]
                    layer, method = prepared["layer"], prepared["method"]
                    numerics = cell["inputs"]["numerics"]
                    n, k = cell["inputs"]["shape"]
                    base = cell["tensors"]["prefill.input"]
                    reference = cell["tensors"]["prefill.reference_output"]
                    with torch.inference_mode():
                        x = base if reps == 1 else base.repeat(reps, 1).contiguous()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = torch.cuda.memory_allocated()
                        y = method.apply(layer, x)
                        torch.cuda.synchronize()
                        peak = max(0, torch.cuda.max_memory_allocated() - before)
                        route = read_route(layer)
                        blocks = y.view(reps, BASE_ROWS, n)
                        spread = float((blocks - blocks[0:1]).abs().max()) if reps > 1 else 0.0
                        head = compare_tensors(y[:BASE_ROWS], reference, **numerics)
                        del blocks, y
                        point = {
                            "m": m, "replicas": reps,
                            "route": route,
                            "route_ok": bool(isinstance(route, dict)
                                             and route.get("state") == "served"
                                             and route.get("reason") is None
                                             and all(route.get(key) == value for key, value
                                                     in prepared["operator"]["declared_route"].items())
                                             and route.get("contract") == prepared["operator"]["activation_contract"]
                                             and route.get("shape") == f"M{m}:N{n}:K{k}"),
                            "replica_max_abs_deviation": spread,
                            "head_numerics": head,
                            "torch_peak_increment_bytes": peak,
                            "flop_per_apply": 2 * m * n * k,
                            "input_bytes": x.untyped_storage().nbytes(),
                            "output_bytes": m * n * 2}
                        if not point["route_ok"]:
                            raise SystemExit(f"{name} M={m}: route is not the panel's served native route")
                        if head["status"] != "passed":
                            raise SystemExit(f"{name} M={m}: tiled head differs from the frozen reference")
                        if reps > 1 and spread != 0.0:
                            raise SystemExit(f"{name} M={m}: identical replicas disagree by {spread}")

                        point["timing"] = time_apply(
                            lambda: method.apply(layer, x),
                            warmup_iterations=args.warmup_iterations, iterations=args.iterations)
                        point["timing"].update(bootstrap_median(point["timing"]["samples_ms"]))

                        torch.cuda.synchronize()
                        start = time.time()
                        applies = 0
                        while time.time() - start < args.power_seconds:
                            for _ in range(8):
                                out = method.apply(layer, x)
                                del out
                            torch.cuda.synchronize()
                            applies += 8
                        end = time.time()
                        elapsed = end - start
                        window = power.window(start, end, settle_s=args.power_settle_seconds)
                        window.update(applies=applies, elapsed_s=elapsed,
                                      wall_ms_per_apply=1000.0 * elapsed / applies)
                        if window.get("status") == "observed" and window["mean_w"] > 0:
                            flops = applies * point["flop_per_apply"] / elapsed
                            window["achieved_tflop_s"] = flops / 1e12
                            window["gflop_per_joule"] = flops / window["mean_w"] / 1e9
                        point["sustained"] = window
                        if read_route(layer) != route:
                            raise SystemExit(f"{name} M={m}: route changed during measurement")
                        del x
                    report["cells"][name]["points"][str(m)] = point
                    torch.cuda.empty_cache()
                    print(json.dumps({"cell": name, "m": m,
                                      "median_ms": point["timing"]["median_ms"],
                                      "mean_w": point["sustained"].get("mean_w"),
                                      "envelope_fraction": point["sustained"].get("envelope_fraction")},
                                     sort_keys=True), flush=True)

            # Profiler last: nothing measured runs after this instrumentation.
            if args.profile:
                from torch.profiler import profile, ProfilerActivity
                directory = args.profile_dir or args.out.parent
                directory.mkdir(parents=True, exist_ok=True)
                for m in rows:
                    reps = m // BASE_ROWS
                    for cell in loaded:
                        name, prepared = cell["name"], cell["prepared"]
                        layer, method = prepared["layer"], prepared["method"]
                        base = cell["tensors"]["prefill.input"]
                        with torch.inference_mode():
                            x = base if reps == 1 else base.repeat(reps, 1).contiguous()
                            for _ in range(3):
                                method.apply(layer, x)
                            torch.cuda.synchronize()
                            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                         record_shapes=True) as prof:
                                for _ in range(5):
                                    out = method.apply(layer, x)
                                    del out
                                torch.cuda.synchronize()
                            del x
                        events = prof.key_averages()
                        kernels = [{"name": e.key, "calls": e.count,
                                    "self_device_us": e.self_device_time_total}
                                   for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
                        if not kernels or sum(row["self_device_us"] for row in kernels) <= 0:
                            raise SystemExit(f"{name} M={m}: profiler recorded no CUDA kernel work")
                        trace = directory / f"{name}.M{m}.trace.json"
                        prof.export_chrome_trace(str(trace))
                        report["cells"][name]["points"][str(m)]["profile"] = {
                            "replays": 5,
                            "kernels": sorted(kernels, key=lambda row: -row["self_device_us"]),
                            "total_self_device_us": sum(row["self_device_us"] for row in kernels),
                            "trace_file": trace.name, "trace_sha256": digest(trace)}
                        torch.cuda.empty_cache()
    finally:
        power.stop()

    report["session_runtime"] = {
        "scope": "final_observation_after_every_arm_was_prepared",
        "native_libraries": len(report["runtime"]["native_libraries"]) if report["runtime"] else 0}
    report["power"]["backend"] = power.backend
    report["power"]["backend_error"] = power.error
    report["power"]["gpu_uuid"] = power.uuid
    report["power"]["series_samples"] = len(power.samples)
    report["finished_utc"] = utc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    series = args.out.parent / (args.out.stem + "-power-series.json")
    series.write_text(json.dumps({"schema": "prismaquant.native_sweep_power_series.v1",
                                 "backend": power.backend, "gpu_uuid": power.uuid,
                                 "envelope_w": ENVELOPE_W, "samples": power.samples},
                                sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(args.out), "sha256": digest(args.out),
                      "power_series": str(series), "power_samples": len(power.samples)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
