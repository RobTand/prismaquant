"""Oracle: is ``acc.addmm_(g.T, x)`` bitwise ``acc.add_(g.T @ x)`` on GB10?

The joint-statistics hook (``JointOperatorStatisticsLease._observe_rows``)
accumulates one FP32 ``g2.T @ x2`` per observed invocation into a dense
out x in matrix. On the GLM-5.3 MTP profile (PB fe9730f7) the separate
``add_`` is 50% of GPU time and the SIMT sgemm producing its operand 16%.
Fusing the accumulate into the GEMM epilogue (beta=1) removes a full-matrix
write and read. Whether it is bitwise depends on the kernel cuBLAS picks for
beta=1, so this measures it on the real shapes rather than assuming it.

Three accumulations run over the same draws, under the matmul settings the
joint entry points pin (``pin_matmul_arithmetic``, bf16 reduced-precision
reduction off, FP32 precision "highest", TF32 off):

* ``ref``: ``acc = g.T @ x`` on the first step, then ``acc.add_(g.T @ x)``;
  exactly the hook's arithmetic.
* ``addmm``: the same first step, then ``acc.addmm_(g.T, x)``.
* ``batched_S``: rows of S consecutive steps concatenated and accumulated as
  one GEMM (a numerics change by construction; measured for drift and speed).

Row counts per step follow the routed-expert distribution (1..64, mean
about 14) and, for the shared expert, the full 511-row sequence. Operands are
BF16-representable values upcast to FP32, as the hook's ``.float()`` makes
them. Reports bitwise equality after every step, the first differing step,
max abs/rel drift against ``ref``, and CUDA-event time per step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = [(512, 2048), (1792, 2048), (2048, 1792), (2048, 2048), (2048, 7168), (7168, 2048)]


def draws(out_dim, in_dim, steps, rows, generator, device):
    for _ in range(steps):
        k = int(rows()) if callable(rows) else int(rows)
        g = torch.randn(k, out_dim, generator=generator, device=device).to(torch.bfloat16).float()
        x = torch.randn(k, in_dim, generator=generator, device=device).to(torch.bfloat16).float()
        # Scale spread like real cotangents/activations: small g, O(1) x.
        yield g * 1e-3, x


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def drift(reference, value):
    diff = (value.double() - reference.double()).abs()
    scale = reference.double().abs().max().item()
    return {"max_abs": diff.max().item(), "max_rel_to_max": diff.max().item() / scale if scale else 0.0,
            "mismatched": int((value != reference).sum().item()), "elements": reference.numel()}


def run_case(out_dim, in_dim, steps, rows, seed, batch_sizes, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    data = list(draws(out_dim, in_dim, steps, rows, generator, device))
    row_counts = [g.shape[0] for g, _ in data]
    ref = addmm = None
    ms_ref = ms_addmm = 0.0
    first_diff = None
    for index, (g, x) in enumerate(data):
        if ref is None:
            ref = g.T @ x
            addmm = g.T @ x
            continue

        def step_ref(g=g, x=x):
            ref.add_(g.T @ x)

        def step_addmm(g=g, x=x):
            addmm.addmm_(g.T, x)

        ms_ref += timed(step_ref)
        ms_addmm += timed(step_addmm)
        if first_diff is None and not torch.equal(ref, addmm):
            first_diff = {"step": index, "rows": g.shape[0], **drift(ref, addmm)}
    result = {"shape": [out_dim, in_dim], "steps": steps, "rows_mean": sum(row_counts) / steps,
              "rows_max": max(row_counts), "bitwise_final": bool(torch.equal(ref, addmm)),
              "first_difference": first_diff, "addmm_drift_final": drift(ref, addmm),
              "ms_per_step_ref": ms_ref / (steps - 1), "ms_per_step_addmm": ms_addmm / (steps - 1),
              "batched": {}}
    for size in batch_sizes:
        acc = None
        total_ms = 0.0
        for start in range(0, steps, size):
            g = torch.cat([d[0] for d in data[start:start + size]])
            x = torch.cat([d[1] for d in data[start:start + size]])
            if acc is None:
                acc = g.T @ x
                continue

            def step(g=g, x=x):
                acc.addmm_(g.T, x)

            total_ms += timed(step)
        chunks = (steps + size - 1) // size
        result["batched"][str(size)] = {
            "ms_per_original_step": total_ms / max(steps - size, 1),
            "chunks": chunks, **drift(ref, acc)}
    del data, ref, addmm
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--batch-sizes", default="8,32,128")
    args = parser.parse_args()
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic

    pin_matmul_arithmetic({"PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION": "off"})
    device = torch.device("cuda")
    sizes = [int(s) for s in args.batch_sizes.split(",")]
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "started": time.time(),
              "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "float32_matmul_precision": torch.get_float32_matmul_precision(), "cases": []}
    for seed in range(args.seeds):
        for out_dim, in_dim in SHAPES:
            rng = torch.Generator().manual_seed(1000 + seed)

            def routed(rng=rng):
                return min(64, 1 + int(torch.poisson(torch.tensor(13.0), generator=rng).item()))

            for label, rows in (("routed", routed), ("full_sequence", 511)):
                case = run_case(out_dim, in_dim, args.steps, rows, seed * 7919 + out_dim + in_dim,
                                sizes, device)
                case.update(label=label, seed=seed)
                report["cases"].append(case)
                print(json.dumps({k: case[k] for k in ("label", "seed", "shape", "bitwise_final",
                                                       "ms_per_step_ref", "ms_per_step_addmm")}),
                      flush=True)
    report["all_bitwise"] = all(case["bitwise_final"] and case["first_difference"] is None
                                for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"all_bitwise={report['all_bitwise']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
