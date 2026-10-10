"""QDQ-only before/after bench for the fused per-token FP8 QDQ (PQ #1398).

Method follows docs/measurements/fp8-native-activation-parity-2026-09-07.md:
call only ``fp8_dynamic_activation_qdq_vllm(...).dequant`` on GPU-resident
inputs, time with CUDA events, count kernels from an exported profiler trace,
and sample device power over a sustained phase. The before arm forces the
torch reference through ``PRISMAQUANT_DISABLE_FP8_FUSED_QDQ=1``; the after
arm uses the fused default path. Kernel counts come from one fresh process
per arm because later profiler sessions in a shared process can miss CUDA
launches and key_averages double-counts each kernel (PQ #1398).

Usage (through PrismaBuild on a GB10 worker):
    python3 experiments/fp8_per_token_qdq_bench.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

from prismaquant.fp8_dynamic import DISABLE_FUSED_ENV, fp8_dynamic_activation_qdq_vllm

WIDTHS = (512, 1536, 2048, 4096, 16384)
ROWS = 256
WARMUP = 20
ROUNDS = 11
CALLS_PER_ROUND = 32
POWER_PHASE_S = 8.0


def _real_activations(rows: int, cols: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1398 + rows + cols)
    values = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
    outliers = torch.randint(0, cols, (rows, 4), generator=generator)
    values.scatter_(1, outliers, values.gather(1, outliers) * 40.0)
    return values.to(torch.bfloat16).to("cuda")


def _timed_calls(inputs: torch.Tensor, calls: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(calls):
        fp8_dynamic_activation_qdq_vllm(inputs).dequant
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / calls


_TRACE_CHILD = """\
import json, os, sys, tempfile, torch
from prismaquant.fp8_dynamic import DISABLE_FUSED_ENV, fp8_dynamic_activation_qdq_vllm
rows, width, calls, arm, trace_path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
os.environ[DISABLE_FUSED_ENV] = arm
generator = torch.Generator(device="cpu").manual_seed(1398 + rows + width)
values = torch.randn((rows, width), generator=generator, dtype=torch.float32)
outliers = torch.randint(0, width, (rows, 4), generator=generator)
values.scatter_(1, outliers, values.gather(1, outliers) * 40.0)
inputs = values.to(torch.bfloat16).to("cuda")
for _ in range(20):
    fp8_dynamic_activation_qdq_vllm(inputs).dequant
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(calls):
        fp8_dynamic_activation_qdq_vllm(inputs).dequant
    torch.cuda.synchronize()
prof.export_chrome_trace(trace_path)
kernels = [e for e in json.load(open(trace_path))["traceEvents"] if e.get("cat") == "kernel"]
print(json.dumps({"kernels_per_call": len(kernels) / calls, "kernel_us_per_call": sum(e.get("dur", 0) for e in kernels) / calls, "kernel_names": sorted({e.get("name", "?") for e in kernels})}), flush=True)
"""


def _profile_arm_trace(width: int, arm: str) -> dict:
    # Count kernels from the exported trace in a fresh process. The first
    # profiler session in a process sees CUDA launches reliably; later
    # sessions after a CPU-only one can miss them, and key_averages counts
    # each kernel twice (aten parent plus device kernel), so neither the
    # shared session nor the averages give the true per-call count (PQ #1398).
    with tempfile.TemporaryDirectory(prefix="fp8-bench-trace-") as tmp:
        trace_path = os.path.join(tmp, "trace.json")
        out = subprocess.run(
            [sys.executable, "-c", _TRACE_CHILD,
             str(ROWS), str(width), str(CALLS_PER_ROUND), arm, trace_path],
            capture_output=True, text=True, timeout=1200, check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"trace child failed for {arm} K={width}:\n{out.stderr[-3000:]}")
        return json.loads(out.stdout.strip().splitlines()[-1])


class _PowerSampler:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                        "-i",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.samples.append(float(out.stdout.strip().split()[0]))
            except Exception:
                pass
            self._stop.wait(0.2)

    def __enter__(self) -> "_PowerSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None


def _power_phase(inputs: torch.Tensor, seconds: float) -> dict:
    # Sustained load: the timed rounds finish in milliseconds, far under the
    # power sampling interval, so power needs its own wall-clock phase.
    calls = 0
    start = time.monotonic()
    with _PowerSampler() as sampler:
        while time.monotonic() - start < seconds:
            for _ in range(32):
                fp8_dynamic_activation_qdq_vllm(inputs).dequant
                calls += 1
            torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    watts = sampler.mean()
    return {
        "power_w_mean": watts,
        "power_samples": len(sampler.samples),
        "power_phase_s": elapsed,
        "power_phase_calls": calls,
        "calls_per_joule": calls / (watts * elapsed) if watts else None,
    }


def _bench_case(width: int, out: Path) -> dict:
    inputs = _real_activations(ROWS, width)
    case: dict = {
        "rows": ROWS,
        "width": width,
        "dtype": "bfloat16",
        "element": "e4m3",
        "rounds": ROUNDS,
        "calls_per_round": CALLS_PER_ROUND,
        "power_phase_s": POWER_PHASE_S,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
    }
    for arm in ("reference", "fused"):
        os.environ[DISABLE_FUSED_ENV] = "1" if arm == "reference" else "0"
        for _ in range(WARMUP):
            fp8_dynamic_activation_qdq_vllm(inputs).dequant
        torch.cuda.synchronize()
        timed_start = time.time()
        per_call = [
            _timed_calls(inputs, CALLS_PER_ROUND) for _ in range(ROUNDS)
        ]
        timed_end = time.time()
        case[arm] = {
            "ms_per_call_median": statistics.median(per_call),
            "ms_per_call_min": min(per_call),
            "timed_epoch_start": timed_start,
            "timed_epoch_end": timed_end,
        }
        power_start = time.time()
        case[arm].update(_power_phase(inputs, POWER_PHASE_S))
        case[arm]["power_epoch_start"] = power_start
        case[arm]["power_epoch_end"] = time.time()
    for arm, flag in (("reference", "1"), ("fused", "0")):
        traced = _profile_arm_trace(width, flag)
        case[arm]["cuda_kernels_per_call"] = traced["kernels_per_call"]
        case[arm]["kernel_us_per_call"] = traced["kernel_us_per_call"]
        case[arm]["kernel_names"] = traced["kernel_names"]
    ref = case["reference"]["ms_per_call_median"]
    fused = case["fused"]["ms_per_call_median"]
    case["speedup"] = ref / fused if fused else None
    (out / f"case-r{ROWS}-k{width}.json").write_text(json.dumps(case, indent=2))
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the QDQ bench needs CUDA")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = [_bench_case(int(w), out) for w in args.widths.split(",")]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for case in summary:
        ref, fused = case["reference"], case["fused"]
        print(
            f'K={case["width"]}: ref {ref["ms_per_call_median"]:.4f} ms/call '
            f'({ref["cuda_kernels_per_call"]} kernels, {ref["kernel_us_per_call"]:.1f} us, '
            f'{ref["power_w_mean"]} W) vs fused {fused["ms_per_call_median"]:.4f} ms/call '
            f'({fused["cuda_kernels_per_call"]} kernels, {fused["kernel_us_per_call"]:.1f} us, '
            f'{fused["power_w_mean"]} W) -> {case["speedup"]:.2f}x',
            flush=True,
        )
    print("SUMMARY_JSON:" + json.dumps(summary), flush=True)
    print("EPOCHS_JSON:" + json.dumps({
        str(case["width"]): {
            arm: {
                "timed": [case[arm]["timed_epoch_start"], case[arm]["timed_epoch_end"]],
                "power": [case[arm]["power_epoch_start"], case[arm]["power_epoch_end"]],
            }
            for arm in ("reference", "fused")
        }
        for case in summary
    }), flush=True)
    del os.environ[DISABLE_FUSED_ENV]


if __name__ == "__main__":
    main()
