"""QDQ-only before/after bench for the fused per-token FP8 QDQ (PQ #1398).

Method follows docs/measurements/fp8-native-activation-parity-2026-09-07.md:
call only ``fp8_dynamic_activation_qdq_vllm(...).dequant`` on GPU-resident
inputs, time with CUDA events, count kernels under torch.profiler, and sample
device power. The before arm forces the torch reference through
``PRISMAQUANT_DISABLE_FP8_FUSED_QDQ=1`` (read per call, so one process runs
both arms interleaved); the after arm uses the fused default path.

Usage (through PrismaBuild measurement admission on a GB10 worker):
    python3 experiments/fp8_per_token_qdq_bench.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
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


def _is_fused_key(key: str) -> bool:
    return "fp8_per_token" in key or key.startswith("triton")


def _profile_both_arms(inputs: torch.Tensor) -> dict:
    # One profiler session for both arms: a CPU-only session earlier in the
    # process stops later sessions from seeing CUDA launches (PQ #1398), so
    # the arms share the first session and split by kernel key afterwards.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        os.environ[DISABLE_FUSED_ENV] = "1"
        for _ in range(CALLS_PER_ROUND):
            fp8_dynamic_activation_qdq_vllm(inputs).dequant
        torch.cuda.synchronize()
        os.environ[DISABLE_FUSED_ENV] = "0"
        for _ in range(CALLS_PER_ROUND):
            fp8_dynamic_activation_qdq_vllm(inputs).dequant
        torch.cuda.synchronize()
    arms = {"reference": ([], 0.0), "fused": ([], 0.0)}
    for event in prof.key_averages():
        # Self device time only: parent CPU ops carry their children's device
        # time and would double-count every kernel (PQ #1398).
        if event.self_device_time_total <= 0:
            continue
        arm = "fused" if _is_fused_key(event.key) else "reference"
        keys, _ = arms[arm]
        keys.append(f"{event.key}x{event.count}")
        arms[arm] = (keys, arms[arm][1] + event.self_device_time_total)
    counts = {
        arm: sum(
            event.count for event in prof.key_averages()
            if event.self_device_time_total > 0
            and (_is_fused_key(event.key) == (arm == "fused"))
        ) // CALLS_PER_ROUND
        for arm in ("reference", "fused")
    }
    return {
        arm: {
            "cuda_kernels_per_call": counts[arm],
            "kernel_us_per_call": arms[arm][1] / CALLS_PER_ROUND,
            "kernel_keys": sorted(arms[arm][0]),
        }
        for arm in ("reference", "fused")
    }


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
        per_call = [
            _timed_calls(inputs, CALLS_PER_ROUND) for _ in range(ROUNDS)
        ]
        case[arm] = {
            "ms_per_call_median": statistics.median(per_call),
            "ms_per_call_min": min(per_call),
        }
        case[arm].update(_power_phase(inputs, POWER_PHASE_S))
    profiled = _profile_both_arms(inputs)
    for arm in ("reference", "fused"):
        case[arm].update(profiled[arm])
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
    del os.environ[DISABLE_FUSED_ENV]


if __name__ == "__main__":
    main()
