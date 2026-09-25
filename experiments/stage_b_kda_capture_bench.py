"""Profile one Stage B capture group on a GLM-5.3 KDA layer and a DSA layer (PQ #1199).

The Stage B spill capture runs, per capture group, one forward and one
backward of a single decoder layer over ``capture_batch`` stored batches
(``joint_cost_quantum.capture_group``). On GLM-5.3-Flash that pass takes about
4x longer on a KDA (``linear_attention``) layer than on a DSA
(``deepseek_sparse_attention``) layer. This bench runs the same pass on
shape-faithful synthetic weights, outside the campaign, so it needs no
checkpoint, chain or spill:

* the decoder layer is the image's own ``Glm5NextTextDecoderLayer``, built from
  the model's ``config.json`` with the campaign's recorded dispatch
  (``attention=eager``, ``experts=grouped_mm``) in bf16;
* the input is the hc-stream shape ``[capture_batch, seqlen, hc_mult, hidden]``;
  the mask and the layer call are PrismaQuant's own
  (``layer_streaming._compute_attention_mask`` and ``_call_layer``);
* each timed step is ``capture_group``'s arithmetic: forward, ``backward`` with
  an incoming cotangent, and the input cotangent copied to the host.

The spill hooks are not installed: they write about the same bytes on both
layer types (PQ #1199), so the difference this bench reports is the layer's
own forward and backward.

``--kda-kernel`` adds, for a KDA layer, the same arms with the Stage B
capture kernel (``prismaquant.kernels.kda_chunk``) in place of the image's
fallback, and compares each arm's input cotangent with its fallback arm's.
The bench swaps the module global for the length of each kernel arm's
forward. It does not use the quantum's ``CaptureKernelDispatch``, which needs
a whole bound model.

Outputs, under ``--out``: ``summary.json`` (wall, peak memory, kernel time by
name, kernel launches, GPU-idle gaps, power) and one Chrome trace per arm.
Run it inside the GLM derivative image, through PrismaBuild.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch


def _power_sampler(stop, samples, period_s=0.1):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            watts = float(out.stdout.strip().splitlines()[0])
            samples.append((time.time(), watts))
        except Exception:  # noqa: BLE001 - a missing sampler reads as no samples
            return
        stop.wait(period_s)


def _mean_power(samples, start, end):
    window = [w for t, w in samples if start <= t <= end]
    return (sum(window) / len(window), len(window)) if window else (None, 0)


def _init_parameters(layer, generator):
    for name, parameter in layer.named_parameters():
        with torch.no_grad():
            if name.endswith("A_log"):
                parameter.copy_(torch.log(torch.empty_like(parameter, dtype=torch.float32)
                                          .uniform_(1, 16, generator=generator)))
            elif name.endswith(("dt_bias", ".base", "e_score_correction_bias")):
                parameter.zero_()
            elif name.endswith(".scale") or "norm" in name.split(".")[-2:][0]:
                parameter.fill_(1.0)
            elif parameter.ndim >= 2:
                parameter.normal_(0.0, 0.02, generator=generator)
            else:
                parameter.fill_(1.0)
    for name, buffer in layer.named_buffers():
        if name.endswith("e_score_correction_bias"):
            buffer.zero_()


def _kernel_summary(prof, top=25):
    kernels = [e for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA]
    by_name = {}
    intervals = []
    for event in kernels:
        start, end = event.time_range.start, event.time_range.end
        intervals.append((start, end))
        row = by_name.setdefault(event.name, [0, 0.0])
        row[0] += 1
        row[1] += (end - start)
    intervals.sort()
    busy, current = 0.0, None
    for start, end in intervals:
        if current is None or start > current[1]:
            if current is not None:
                busy += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        busy += current[1] - current[0]
    span = (intervals[-1][1] - intervals[0][0]) if intervals else 0.0
    total = sum(v[1] for v in by_name.values())
    ranked = sorted(by_name.items(), key=lambda item: -item[1][1])
    return {
        "kernel_launches": len(kernels),
        "kernel_time_s": total / 1e6,
        "kernel_busy_union_s": busy / 1e6,
        "kernel_span_s": span / 1e6,
        "gpu_idle_gap_s": (span - busy) / 1e6,
        "top_kernels": [{"name": name[:160], "count": count, "time_s": t / 1e6,
                         "share": (t / total if total else None)}
                        for name, (count, t) in ranked[:top]],
    }


def _op_summary(prof, top=40):
    table = []
    for avg in prof.key_averages(group_by_input_shape=True):
        device = getattr(avg, "self_device_time_total", None)
        if device is None:
            device = getattr(avg, "self_cuda_time_total", 0)
        if device <= 0:
            continue
        table.append({"op": avg.key, "count": avg.count, "self_device_s": device / 1e6,
                      "shapes": str(avg.input_shapes)[:200]})
    table.sort(key=lambda row: -row["self_device_s"])
    return table[:top]


def build_layer(config, layer_index, device, dtype, seed):
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.device(device):
        layer = modeling.Glm5NextTextDecoderLayer(config, layer_index).to(dtype)
    _init_parameters(layer, generator)
    # As the streamed runner holds a layer: eval mode, frozen weights, so the
    # backward computes activation cotangents only (streaming_model.py).
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return layer


@contextlib.contextmanager
def kda_kernel_dispatch():
    """The capture kernel in place of the image's fallback, for one block."""
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    from prismaquant.kernels import kda_chunk
    original = modeling.chunk_kimi_delta_attention
    modeling.chunk_kimi_delta_attention = kda_chunk.chunk_kimi_delta_attention
    try:
        yield
    finally:
        modeling.chunk_kimi_delta_attention = original


def with_kda_kernel(call):
    def wrapped(layer, x_in):
        with kda_kernel_dispatch():
            return call(layer, x_in)
    return wrapped


def run_group(layer, call, x_host, grad_host, device, dtype):
    x_in = x_host.to(device, dtype).detach().requires_grad_(True)
    incoming = grad_host.to(device, dtype)
    out = call(layer, x_in)
    torch.autograd.backward([out], [incoming])
    gradient = x_in.grad.detach().to("cpu")
    return gradient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", default="38,39")
    parser.add_argument("--capture-batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=0.0,
                        help="repeat each timed arm until it has run this long, so a "
                             "1 s Netdata series sees several samples of it")
    parser.add_argument("--kda-chunk-sizes", default="",
                        help="comma list: also time the KDA self_attn with these "
                             "chunk_size kwargs (prices a non-bit-exact option)")
    parser.add_argument("--attention-only", action="store_true",
                        help="also time the layer's self_attn sublayer alone")
    parser.add_argument("--kda-kernel", action="store_true",
                        help="also time each KDA arm with the Stage B capture kernel")
    parser.add_argument("--skip-fallback", action="store_true",
                        help="with --kda-kernel, time only the kernel arms")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from prismaquant.layer_streaming import _call_layer, _compute_attention_mask

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16
    raw = json.loads(Path(args.config).read_text())
    full = Glm5NextConfig(**raw)
    config = full.text_config
    config._attn_implementation = "eager"
    config._experts_implementation = "grouped_mm"
    hc = int(config.hc_mult)
    batch, seq, hidden = args.capture_batch, args.seqlen, int(config.hidden_size)

    class _Base(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

    position_ids = torch.arange(seq, device=device).unsqueeze(0)
    probe_hidden = torch.zeros(batch, seq, hidden, device=device, dtype=dtype)
    masks = _compute_attention_mask(_Base(config), probe_hidden, position_ids)
    del probe_hidden

    def call(layer, x_in):
        return _call_layer(layer, x_in, position_embeddings=None, attention_mask=masks,
                           position_ids=position_ids, pass_state={})

    def attention_call(layer, x_in, **extra):
        normed = layer.input_layernorm(x_in)
        if layer.block_type == "linear_attention":
            return layer.self_attn(hidden_states=normed, cache_params=None,
                                   attention_mask=(masks.get("linear_attention")
                                                   if isinstance(masks, dict) else masks),
                                   **extra)
        result = layer.self_attn(hidden_states=normed,
                                 attention_mask=(masks.get("deepseek_sparse_attention")
                                                 if isinstance(masks, dict) else masks),
                                 position_ids=position_ids, past_key_values=None,
                                 use_cache=False, position_embeddings=None,
                                 prev_topk_indices=None)
        return result[0]

    generator = torch.Generator().manual_seed(1199)
    x_host = torch.randn(batch, seq, hc, hidden, generator=generator).to(dtype)
    grad_host = (torch.randn(batch, seq, hc, hidden, generator=generator) * 1e-3).to(dtype)
    x_attn = torch.randn(batch, seq, hidden, generator=generator).to(dtype)
    g_attn = (torch.randn(batch, seq, hidden, generator=generator) * 1e-3).to(dtype)

    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=_power_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    summary = {"schema": "prismaquant.stage_b_kda_capture_bench.v1",
               "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
               "capture_batch": batch, "seqlen": seq, "hc_mult": hc, "hidden": hidden,
               "dtype": str(dtype), "attention": "eager", "experts": "grouped_mm",
               "layers": {}}
    try:
        for layer_index in [int(v) for v in args.layers.split(",")]:
            layer_type = config.layer_types[layer_index]
            layer = build_layer(config, layer_index, device, dtype, seed=layer_index)
            record = {"layer_type": layer_type}
            arms = [] if args.skip_fallback else [("layer", call, x_host, grad_host)]
            if args.attention_only and not args.skip_fallback:
                arms.append(("self_attn", attention_call, x_attn, g_attn))
            if layer_type == "linear_attention" and args.kda_kernel:
                arms.append(("layer_kernel", with_kda_kernel(call), x_host, grad_host))
                if args.attention_only:
                    arms.append(("self_attn_kernel", with_kda_kernel(attention_call),
                                 x_attn, g_attn))
            if layer_type == "linear_attention" and args.kda_chunk_sizes:
                for size in [int(v) for v in args.kda_chunk_sizes.split(",")]:
                    arms.append((f"self_attn_chunk{size}",
                                 lambda lyr, x, size=size: attention_call(lyr, x, chunk_size=size),
                                 x_attn, g_attn))
            for arm, fn, xh, gh in arms:
                for _ in range(args.warmup):
                    run_group(layer, fn, xh, gh, device, dtype)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_alloc = torch.cuda.memory_allocated()
                walls = []
                t_arm0 = time.time()
                while len(walls) < args.iters or sum(walls) < args.min_seconds:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    run_group(layer, fn, xh, gh, device, dtype)
                    torch.cuda.synchronize()
                    walls.append(time.time() - t0)
                t_arm1 = time.time()
                power, n_power = _mean_power(samples, t_arm0, t_arm1)
                row = {"wall_s": walls, "wall_mean_s": sum(walls) / len(walls),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "resident_before_bytes": base_alloc,
                       "window_unix": [t_arm0, t_arm1],
                       "power_w_mean": power, "power_samples": n_power,
                       "energy_j_per_group": (power * sum(walls) / len(walls)
                                              if power is not None else None)}
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True) as prof:
                    for _ in range(args.profile_iters):
                        run_group(layer, fn, xh, gh, device, dtype)
                    torch.cuda.synchronize()
                trace = out_dir / f"trace-layer{layer_index:03d}-{arm}.json.gz"
                prof.export_chrome_trace(str(trace))
                row["profile"] = {"iters": args.profile_iters,
                                  **_kernel_summary(prof),
                                  "top_ops": _op_summary(prof),
                                  "trace": str(trace)}
                reference = record.get("_reference_grad")
                gradient = run_group(layer, fn, xh, gh, device, dtype)
                if arm in ("self_attn", "layer"):
                    record.setdefault("_reference_grads", {})[arm] = gradient
                if arm == "self_attn":
                    record["_reference_grad"] = gradient
                elif reference is not None and arm.startswith("self_attn_chunk"):
                    diff = (gradient.float() - reference.float()).abs()
                    row["input_grad_vs_chunk64"] = {
                        "bit_equal": bool(torch.equal(gradient, reference)),
                        "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean()),
                        "reference_mean_abs": float(reference.float().abs().mean())}
                fallback_grad = record.get("_reference_grads", {}).get(
                    arm[:-len("_kernel")] if arm.endswith("_kernel") else None)
                if fallback_grad is not None:
                    reference64 = fallback_grad.double()
                    diff = (gradient.double() - reference64).abs()
                    row["input_grad_vs_fallback"] = {
                        "bit_equal": bool(torch.equal(gradient, fallback_grad)),
                        "differing_elements": int((gradient != fallback_grad).sum()),
                        "elements": int(gradient.numel()),
                        "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean()),
                        "rel_fro": float(diff.norm() / reference64.norm()),
                        "reference_mean_abs": float(reference64.abs().mean())}
                if arm in ("self_attn", "self_attn_kernel", "layer_kernel"):
                    again = run_group(layer, fn, xh, gh, device, dtype)
                    row["repeat_bit_equal"] = bool(torch.equal(again, gradient))
                record[arm] = row
                print(json.dumps({"layer": layer_index, "type": layer_type, "arm": arm,
                                  "wall_mean_s": row["wall_mean_s"],
                                  "peak_gb": row["peak_allocated_bytes"] / 1e9,
                                  "power_w": power,
                                  "kernel_time_s": row["profile"]["kernel_time_s"],
                                  "launches": row["profile"]["kernel_launches"],
                                  "idle_gap_s": row["profile"]["gpu_idle_gap_s"]}),
                      flush=True)
                del prof
            record.pop("_reference_grad", None)
            record.pop("_reference_grads", None)
            summary["layers"][str(layer_index)] = record
            del layer
            torch.cuda.empty_cache()
    finally:
        stop.set()
        sampler.join(timeout=5)
    summary["power_samples_total"] = len(samples)
    if args.kda_kernel:
        from prismaquant.kernels import kda_chunk
        summary["kda_kernel"] = {"name": kda_chunk.NAME,
                                 "source_sha256": kda_chunk.source_sha256(),
                                 "compiled": kda_chunk.compiled_kernels(),
                                 "counts": kda_chunk.counts()}
    target = out_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=1))
    print(f"summary {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
