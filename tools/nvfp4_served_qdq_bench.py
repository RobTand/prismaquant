"""Interleaved A/B of the served NVFP4 activation leg (RobTand/prismaquant#1211).

Arm ``unfused`` is the pre-#1211 Torch composition
(``_nvfp4_activation_qdq_registered_op_unfused``); arm ``fused`` is the leg the
contract dispatches (``_nvfp4_activation_qdq_registered_op``).  Both call the
registered ``torch.ops._C.scaled_fp4_quant``, so this runs only where the vLLM
extension is installed -- the campaign image, inside an admitted PB action.

For every shape it reports, per arm and interleaved over ``--reps`` rounds
whose arm order alternates:

* ``leg``: the leg alone, ``--calls`` back-to-back calls -- host time until the
  last call returned (what the Python thread spends, sync waits included) and
  wall time once the device drained;
* ``replay``: a Stage B ``operator_gemm`` chunk as ``joint_aura._observe_rows``
  runs it -- ``g.T @ x.float()``, the QDQ, ``dx = q.float() - x2``,
  ``g.T @ dx`` -- so the cost of a host sync shows up as lost overlap, not as
  time inside the leg;
* mean GPU power over each arm's windows (NVML), against the GB10's 140 W
  envelope, and the energy per call;
* the SHA-256 of both arms' outputs, which must be equal.

``--profile-dir`` also runs ``torch.profiler`` over a few calls of each arm at
the first shape and writes both Chrome traces plus a summary of the host-side
CUDA API time (synchronising copies, stream syncs, launches) and the kernel
time.

Prints one JSON document on the line after ``BENCH-JSON``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prismaquant import nvfp4_activation_contract as owner  # noqa: E402

ARMS = {
    "unfused": owner._nvfp4_activation_qdq_registered_op_unfused,
    "fused": owner._nvfp4_activation_qdq_registered_op,
}
ENVELOPE_W = 140.0


class PowerSampler:
    """NVML power at ``hz``; ``window(t0, t1)`` is the mean over a span."""

    def __init__(self, hz: float = 50.0):
        self.samples: list[tuple[float, float]] = []
        self.source = None
        self._stop = threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._read = lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            self._read()
            self.source = "nvml"
        except Exception as exc:  # noqa: BLE001 -- reported, never guessed
            self._read = None
            self.source = f"unavailable: {type(exc).__name__}: {exc}"
        self._period = 1.0 / hz
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), float(self._read())))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self._period)

    def start(self):
        if self._read is not None:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def window(self, t0: float, t1: float):
        values = [w for t, w in self.samples if t0 <= t <= t1]
        return (statistics.fmean(values), len(values)) if values else (None, 0)


def activation(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, width, device="cuda", generator=generator)
    x = x * torch.exp(torch.randn(rows, 1, device="cuda", generator=generator))
    spikes = torch.rand(rows, width, device="cuda", generator=generator) < 0.002
    x = torch.where(spikes, x * 40.0, x)
    x[:, 16:32] = 0.0
    return x.to(torch.bfloat16)


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def time_leg(leg, x, g, calls):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        leg(x, g)
    host = time.perf_counter() - t0
    torch.cuda.synchronize()
    return t0, host, time.perf_counter() - t0


def time_replay(leg, x, grad, g, chunks):
    """``_observe_rows`` for one Linear and one QDQ group, ``chunks`` times."""
    acc0 = torch.zeros(grad.shape[1], x.shape[1], device="cuda")
    acc1 = torch.zeros_like(acc0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(chunks):
        x2 = x.float()
        acc0 += grad.T @ x2
        quantized = leg(x, g)
        dx = quantized.reshape_as(x2).float() - x2
        acc1 += grad.T @ dx
    host = time.perf_counter() - t0
    torch.cuda.synchronize()
    return t0, host, time.perf_counter() - t0


def profile_arms(x, g, out_dir: Path, calls: int = 5):
    from torch.profiler import ProfilerActivity, profile

    summary = {}
    for name, leg in ARMS.items():
        leg(x, g)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                leg(x, g)
            torch.cuda.synchronize()
        trace = out_dir / f"torch-trace-{name}.json"
        prof.export_chrome_trace(str(trace))
        api = {}
        kernels = {}
        kernel_total = 0.0
        kernel_count = 0
        for event in prof.key_averages():
            key = event.key
            if key.startswith("cuda") or key.startswith("cu"):
                api[key] = {"calls": int(event.count),
                            "self_cpu_us": float(event.self_cpu_time_total)}
            device_us = float(getattr(event, "self_device_time_total",
                                      getattr(event, "self_cuda_time_total", 0.0)))
            if device_us > 0 and not key.startswith("aten::") and event.count:
                kernels[key] = {"calls": int(event.count), "device_us": device_us}
                kernel_total += device_us
                kernel_count += int(event.count)
        top = sorted(kernels.items(), key=lambda kv: -kv[1]["device_us"])[:12]
        summary[name] = {
            "calls": calls,
            "trace": str(trace),
            "host_cuda_api": api,
            "kernel_launches_per_call": kernel_count / calls,
            "kernel_device_us_per_call": kernel_total / calls,
            "top_kernels": dict(top),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shapes", default="65536x4096,65536x2048,8192x4096,"
                                            "2048x16384,1024x2048,256x1536,37x512")
    parser.add_argument("--grad-width", type=int, default=2048)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()

    if not owner._register_served_quantizer_op():
        raise SystemExit("torch.ops._C.scaled_fp4_quant is not registered here")
    owner._served_dequant_kernels()
    power = PowerSampler()
    power.start()
    result = {
        "schema": "prismaquant.nvfp4_served_qdq_bench.v1",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "dequant_kernel": owner.SERVED_QUANTIZER_DEQUANT_KERNEL,
        "power_source": power.source,
        "envelope_w": ENVELOPE_W,
        "reps": args.reps, "calls": args.calls, "chunks": args.chunks,
        "shapes": {},
    }
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    for index, (rows, width) in enumerate(shapes):
        x = activation(rows, width, seed=rows + width)
        g = 448.0 * 6.0 / float(x.float().abs().amax())
        grad = torch.randn(rows, args.grad_width, device="cuda")
        outputs = {name: leg(x, g) for name, leg in ARMS.items()}   # warm + digest
        torch.cuda.synchronize()
        digests = {name: digest(out) for name, out in outputs.items()}
        equal = torch.equal(outputs["fused"], outputs["unfused"])
        del outputs
        # Small shapes are overhead-bound; give each window comparable work.
        scale = 65536 * 4096 / (rows * width)
        calls = max(args.calls, min(2000, int(args.calls * scale)))
        chunks = max(args.chunks, min(500, int(args.chunks * scale)))
        record = {"rows": rows, "width": width, "g": g, "digests": digests,
                  "torch_equal": bool(equal), "calls_per_window": calls,
                  "chunks_per_window": chunks,
                  "arms": {name: {"leg_host_ms": [], "leg_wall_ms": [], "leg_power_w": [],
                                  "replay_host_ms": [], "replay_wall_ms": [],
                                  "replay_power_w": []} for name in ARMS}}
        for rep in range(args.reps):
            order = list(ARMS) if rep % 2 == 0 else list(reversed(list(ARMS)))
            for name in order:
                arm = record["arms"][name]
                t0, host, wall = time_leg(ARMS[name], x, g, calls)
                arm["leg_host_ms"].append(1e3 * host / calls)
                arm["leg_wall_ms"].append(1e3 * wall / calls)
                arm["leg_power_w"].append(power.window(t0, t0 + wall)[0])
                t0, host, wall = time_replay(ARMS[name], x, grad, g, chunks)
                arm["replay_host_ms"].append(1e3 * host / chunks)
                arm["replay_wall_ms"].append(1e3 * wall / chunks)
                arm["replay_power_w"].append(power.window(t0, t0 + wall)[0])
        for name, arm in record["arms"].items():
            arm["summary"] = {
                key: {"median": statistics.median(values), "min": min(values),
                      "max": max(values)}
                for key, values in arm.items()
                if values and all(v is not None for v in values)
            }
            leg_w = arm["summary"].get("leg_power_w", {}).get("median")
            if leg_w is not None:
                arm["summary"]["leg_joules_per_call"] = (
                    leg_w * arm["summary"]["leg_wall_ms"]["median"] / 1e3)
                arm["summary"]["leg_envelope_fraction"] = leg_w / ENVELOPE_W
        a, b = (record["arms"][n]["summary"] for n in ("unfused", "fused"))
        record["delta"] = {
            "leg_wall_speedup": a["leg_wall_ms"]["median"] / b["leg_wall_ms"]["median"],
            "leg_host_speedup": a["leg_host_ms"]["median"] / b["leg_host_ms"]["median"],
            "replay_wall_speedup": a["replay_wall_ms"]["median"] / b["replay_wall_ms"]["median"],
            "replay_wall_saved_ms_per_chunk": (a["replay_wall_ms"]["median"]
                                               - b["replay_wall_ms"]["median"]),
        }
        if index == 0 and args.profile_dir:
            out_dir = Path(args.profile_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            record["profile"] = profile_arms(x, g, out_dir)
        record["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["shapes"][f"{rows}x{width}"] = record
        del x, grad
        torch.cuda.empty_cache()
        print(f"shape {rows}x{width}: equal={equal} digests_equal="
              f"{digests['fused'] == digests['unfused']} delta={record['delta']}",
              flush=True)
    power.stop()
    result["all_digests_equal"] = all(
        r["digests"]["fused"] == r["digests"]["unfused"] for r in result["shapes"].values())
    print("BENCH-JSON")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_digests_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
