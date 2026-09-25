"""Time and profile the KDA attention call alone: kernel and fallback (PQ #1199).

One timed step is the capture's arithmetic for one KDA call at the
production shape (capture batch 4, sequence 512, 64 heads, head dim 128):
forward, then backward with a bf16 output cotangent. Each implementation is
timed over ``--iters`` steps and profiled over one more with
``torch.profiler`` (kernel time by name, launches, GPU-idle gap). A 10 Hz
``nvidia-smi`` sampler gives the mean power over the timed window, and so the
energy per step.

``stage_b_kda_capture_bench.py`` times the same call inside the decoder
layer; this bench isolates the attention function, needs no model config and
fits beside a running quantum when ``--no-fallback`` drops the fallback,
whose FP32 intermediates take about 24 GB. Run it inside the GLM derivative
image, through PrismaBuild.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import torch

from experiments.stage_b_kda_capture_bench import _kernel_summary, _mean_power, _op_summary, _power_sampler


def draw(batch, length, heads, dim, device, seed=1199):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, length, heads, dim)

    def normal(*size):
        return torch.randn(size, device=device, generator=generator)

    return {"q": normal(*shape).to(torch.bfloat16), "k": normal(*shape).to(torch.bfloat16),
            "v": normal(*shape).to(torch.bfloat16),
            "g": (-5.0 * torch.sigmoid(normal(*shape) * 3.0)).contiguous(),
            "beta": torch.sigmoid(normal(batch, length, heads)).to(torch.bfloat16),
            "stimulus": normal(*shape).to(torch.bfloat16)}


def step(function, inputs):
    leaves = [inputs[name].detach().clone().requires_grad_(True) for name in ("q", "k", "v", "g", "beta")]
    out, _ = function(leaves[0], leaves[1], leaves[2], g=leaves[3], beta=leaves[4],
                      use_qk_l2norm_in_kernel=True)
    torch.autograd.backward([out], [inputs["stimulus"]])
    return out.detach(), [leaf.grad for leaf in leaves]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--min-seconds", type=float, default=0.0)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args(argv)

    from prismaquant.kernels import kda_chunk
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic
    pin_matmul_arithmetic()
    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = draw(args.batch, args.seqlen, args.heads, 128, device)
    impls = [("kernel", kda_chunk.chunk_kimi_delta_attention)]
    if not args.no_fallback:
        from transformers.models.glm5_next import modeling_glm5_next as modeling
        impls.append(("fallback", modeling.chunk_kimi_delta_attention))
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=_power_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    summary = {"schema": "prismaquant.kda_kernel_microbench.v1", "torch": torch.__version__,
               "device": torch.cuda.get_device_name(0),
               "shape": {"batch": args.batch, "seqlen": args.seqlen, "heads": args.heads, "head_dim": 128},
               "kernel": {"name": kda_chunk.NAME, "source_sha256": kda_chunk.source_sha256()},
               "impls": {}}
    grads = {}
    try:
        for name, function in impls:
            step(function, inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            walls = []
            t0 = time.time()
            while len(walls) < args.iters or sum(walls) < args.min_seconds:
                torch.cuda.synchronize()
                start = time.time()
                step(function, inputs)
                torch.cuda.synchronize()
                walls.append(time.time() - start)
            t1 = time.time()
            power, count = _mean_power(samples, t0, t1)
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                out, grad = step(function, inputs)
                torch.cuda.synchronize()
            trace = out_dir / f"trace-{name}.json.gz"
            prof.export_chrome_trace(str(trace))
            mean = sum(walls) / len(walls)
            row = {"wall_s": walls, "wall_mean_s": mean, "window_unix": [t0, t1],
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "resident_before_bytes": base,
                   "power_w_mean": power, "power_samples": count,
                   "energy_j_per_step": power * mean if power is not None else None,
                   "profile": {**_kernel_summary(prof), "top_ops": _op_summary(prof),
                               "trace": str(trace)}}
            grads[name] = [out, *grad]
            summary["impls"][name] = row
            print(json.dumps({"impl": name, "wall_mean_s": mean, "power_w": power,
                              "peak_gb": row["peak_allocated_bytes"] / 1e9,
                              "launches": row["profile"]["kernel_launches"],
                              "kernel_time_s": row["profile"]["kernel_time_s"]}), flush=True)
            del prof
            torch.cuda.empty_cache()
    finally:
        stop.set()
        sampler.join(timeout=5)
    if "fallback" in grads:
        summary["kernel_vs_fallback"] = {}
        for label, k, f in zip(("o", "dq", "dk", "dv", "dg", "dbeta"), grads["kernel"], grads["fallback"]):
            diff = (k.double() - f.double()).abs()
            summary["kernel_vs_fallback"][label] = {
                "max_abs": float(diff.max()),
                "rel_fro": float(diff.norm() / f.double().norm())}
    summary["kernel"]["compiled"] = kda_chunk.compiled_kernels()
    summary["power_samples_total"] = len(samples)
    target = out_dir / "microbench.json"
    target.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"microbench {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
