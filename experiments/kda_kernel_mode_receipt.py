"""GPU receipt for Stage B KDA kernel mode (PQ #1214): the capture is the fused chain roll.

A band-serial consumer equals chain mode byte for byte (PQ #996) only if the
producer's capture pass and the consumer's chain roll compute the same input
cotangent, bit for bit, for the producer's layer. In kernel mode both run the
KDA capture kernel. The CPU fixture (``tests/test_kda_kernel_mode.py``) runs
the whole band with a fixture kernel; this receipt runs the real Triton
kernel on the image's own GLM KDA decoder layer, which the fixture cannot:

* the decoder layer is the image's ``Glm5NextTextDecoderLayer``, with
  shape-faithful synthetic weights (``stage_b_kda_capture_bench.build_layer``)
  and the campaign's dispatch (eager attention, grouped_mm experts), in bf16;
* the regime is R13's: capture batch 4, chain batch 4 with probe fusion;
* the capture is ``joint_cost_quantum``'s batched spill capture
  (``capture_group``), without its spill hooks: per probe and per group of
  four stored batches, one forward and one backward, the cotangent split back
  per batch;
* the chain is the real ``joint_adjoint_checkpoints.render_free_layer_roll``:
  per group one forward, then one backward per probe on the retained graph.

In kernel mode the capture passes run inside
``AdmittedKdaKernel.layer_pass(site="target")`` and the roll gets
``layer_pass(site="chain")`` through its hook, exactly as the quantum core
passes them, so every pass's kernel counts are checked (one call and two Gram
forwards per forward, two Gram backwards per backward).

The receipt passes when, in each mode: the capture's and the roll's planes are
byte-equal; a second run of each is byte-equal to the first; and, across
modes, the kernel's planes differ from the fallback's (so the kernel ran and
the equality is not vacuous). The kernel record must count every pass.

Every plane comparison also records its scale (PQ #1214 E4): the reference
plane's mean and largest magnitude and the count of elements whose bits
differ, next to the largest absolute difference. A pre-hook on the layer's
packed experts records the top-k experts each forward sent its tokens to, and
the receipt compares them per group across arms: the tokens whose selected
experts differ and the expert slots that changed. These are recorded, not
checked; the weights are synthetic, so the counts show the mechanism, not
GLM's rate.

Not covered here: ``CaptureKernelDispatch`` (it needs a whole bound model;
the receipt swaps the module global for each block, as the #1199 scope smoke
did), the spill hooks, and the quantum core around the passes. The CPU
fixture covers those. Run it inside the GLM derivative image, through
PrismaBuild.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch


def _power_sampler(stop, samples, period_s=0.5):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            samples.append((time.time(), float(out.stdout.strip().splitlines()[0])))
        except Exception:  # noqa: BLE001 - a missing sampler reads as no samples
            return
        stop.wait(period_s)


def _plane_digest(plane) -> str:
    digest = hashlib.sha256()
    for key in sorted(plane):
        tensor = plane[key].contiguous()
        digest.update(json.dumps([list(key), list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bits(tensor):
    """The elements as same-width integers: -0.0 differs from 0.0, a NaN equals itself."""
    if not tensor.is_floating_point():
        return tensor
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return tensor.contiguous().view(width[tensor.element_size()])


def _compare(first, second) -> dict:
    """Compare ``first`` with the reference plane ``second``.

    Besides equality and the largest absolute difference, records what gives
    that difference a scale (PQ #1214 E4): the reference's mean and largest
    magnitude, and how many elements differ bit for bit.
    """
    if sorted(first) != sorted(second):
        return {"equal": False, "keys": "differ"}
    if any(first[key].shape != second[key].shape or first[key].dtype != second[key].dtype
           for key in first):
        return {"equal": False, "keys": "shapes or dtypes differ"}
    unequal = [list(key) for key in sorted(first) if not torch.equal(first[key], second[key])]
    worst = max((float((first[key].float() - second[key].float()).abs().max())
                 for key in first if first[key].numel()), default=0.0)
    elements = sum(int(second[key].numel()) for key in second)
    differing = sum(int((_bits(first[key]) != _bits(second[key])).sum()) for key in first)
    mean_abs = (math.fsum(float(second[key].double().abs().sum()) for key in second) / elements
                if elements else None)
    max_abs = max((float(second[key].float().abs().max()) for key in second
                   if second[key].numel()), default=0.0)
    return {"equal": not unequal, "unequal_keys": unequal[:8],
            "unequal_count": len(unequal), "max_abs_diff": worst,
            "elements": elements, "differing_elements": differing,
            "differing_fraction": differing / elements if elements else None,
            "reference_mean_abs": mean_abs, "reference_max_abs": max_abs,
            "max_abs_diff_over_reference_mean_abs": (worst / mean_abs) if mean_abs else None}


def _route_sets(index):
    """Each token's selected experts, sorted: an int64 ``[tokens, k]`` tensor."""
    return index.reshape(-1, index.shape[-1]).to(torch.int64).sort(dim=-1).values


def _compare_routes(first, second) -> dict:
    """Compare two arms' top-k selections, group by group (PQ #1214 E4).

    ``first`` and ``second`` hold one ``[tokens, k]`` index tensor per group,
    in the same group order. A token is changed when its set of selected
    experts differs; ``changed_expert_slots`` counts the experts that the
    token selects in ``first`` and not in ``second``.
    """
    if first is None or second is None or len(first) != len(second) or any(
            a.shape != b.shape for a, b in zip(first, second)):
        return {"comparable": False}
    tokens = changed_tokens = changed_slots = 0
    for a, b in zip(first, second):
        sa, sb = _route_sets(a), _route_sets(b)
        tokens += int(sa.shape[0])
        changed_tokens += int((sa != sb).any(dim=-1).sum())
        changed_slots += int((~(sa.unsqueeze(-1) == sb.unsqueeze(-2)).any(dim=-1)).sum())
    return {"comparable": True, "tokens": tokens,
            "tokens_with_a_changed_expert": changed_tokens,
            "changed_expert_slots": changed_slots}


def _group_routes(forwards, groups: int, passes_per_group: int):
    """One route tensor per group, and whether every pass over a group chose the same.

    ``forwards`` holds each forward's top-k index in the order the arm ran
    them: the capture runs every group once per probe, probe-major; the fused
    roll runs each group once. Returns ``(None, False)`` when the arm ran
    another number of forwards.
    """
    if len(forwards) != groups * passes_per_group:
        return None, False
    first = list(forwards[:groups])
    stable = all(torch.equal(forwards[p * groups + g], first[g])
                 for p in range(passes_per_group) for g in range(groups))
    return first, stable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--batches", type=int, default=8,
                        help="stored batches, a multiple of the capture batch")
    parser.add_argument("--capture-batch", type=int, default=4)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.batches % args.capture_batch or args.probes < 2:
        parser.error("--batches must be a multiple of --capture-batch; --probes at least 2")

    import transformers
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

    from experiments.stage_b_kda_capture_bench import build_layer
    from prismaquant import glm_kda_capture_kernel as capture
    from prismaquant.cost_streaming import StreamedForwardBoundaries
    from prismaquant.joint_adjoint_checkpoints import (
        _chain_group_batch, _stack_to_device, render_free_layer_roll)
    from prismaquant.kernels import kda_chunk
    from prismaquant.layer_streaming import _call_layer, _compute_attention_mask
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16
    config = Glm5NextConfig(**json.loads(Path(args.config).read_text())).text_config
    config._attn_implementation = "eager"
    config._experts_implementation = "grouped_mm"
    layer_index, seq = int(args.layer), int(args.seqlen)
    hc, hidden = int(config.hc_mult), int(config.hidden_size)
    group_size, n_probes = int(args.capture_batch), int(args.probes)
    if config.layer_types[layer_index] != "linear_attention":
        parser.error(f"layer {layer_index} is {config.layer_types[layer_index]}, not KDA")

    class _Base(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

    profile = Glm5NextProfile()
    base = _Base(config)

    class Runner:
        """What the capture and the roll read off ``StreamedCausalLM``, for one layer."""

        def __init__(self, layer):
            self.device, self.dtype, self.profile = torch.device(device), dtype, profile
            self.layers = {layer_index: layer}

        def _prepare(self, ids):
            # ``StreamedCausalLM._prepare``'s positions and mask; the embedding
            # is not read: every pass starts from a stored boundary.
            ids = ids.to(self.device)
            position_ids = torch.arange(ids.size(-1), device=self.device).unsqueeze(0)
            probe_hidden = torch.zeros(ids.shape[0], ids.shape[1], hidden,
                                       device=self.device, dtype=dtype)
            mask = _compute_attention_mask(base, probe_hidden, position_ids)
            return ids, position_ids, None, None, mask

        def isolated_layer(self, batch, layer, x_in, *, pass_state):
            # ``StreamedCausalLM._call``.
            return _call_layer(self.layers[layer], x_in,
                               position_embeddings=batch.position_embeddings,
                               attention_mask=batch.attention_mask,
                               position_ids=batch.position_ids,
                               **self.profile.extra_layer_kwargs(input_ids=batch.input_ids),
                               pass_state=pass_state)

    class SwapDispatch:
        """The kernel in place of the modeling global for one block (#1199 scope smoke)."""

        def __enter__(self):
            self.original = modeling.chunk_kimi_delta_attention
            modeling.chunk_kimi_delta_attention = kda_chunk.chunk_kimi_delta_attention
            return self

        def __exit__(self, *exc_info):
            current = modeling.chunk_kimi_delta_attention
            modeling.chunk_kimi_delta_attention = self.original
            if current is not kda_chunk.chunk_kimi_delta_attention:
                raise RuntimeError("KDA dispatch changed inside a kernel block")
            return False

    layer = build_layer(config, layer_index, device, dtype, seed=layer_index)
    # build_layer constructs the layer in FP32 before casting it to BF16. Give
    # the freed FP32 blocks back so that the process holds what the arms use.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    layer_resident_bytes = torch.cuda.memory_allocated()
    runner = Runner(layer)
    generator = torch.Generator().manual_seed(1214)
    count = int(args.batches)
    batches, boundaries = [], []
    for index in range(count):
        ids = torch.randint(0, int(config.vocab_size), (1, seq), generator=generator)
        boundary = torch.randn(1, seq, hc, hidden, generator=generator).to(dtype)
        boundaries.append(boundary)
        batches.append(StreamedForwardBoundaries(
            input_ids=ids, position_ids=None, position_embeddings=None,
            attention_mask=None, activations_cpu={layer_index: boundary},
            shared_pass_state=None))
    plane_in = {(probe, index): (torch.randn(1, seq, hc, hidden, generator=generator)
                                 * 1e-3).to(dtype)
                for probe in range(n_probes) for index in range(count)}
    owners = [[SimpleNamespace(is_empty=lambda: True) for _ in range(count)]
              for _ in range(n_probes)]
    groups = [list(range(start, start + group_size))
              for start in range(0, count, group_size)]

    def capture_plane(target_pass):
        """``capture_group``: per probe and group, one forward and one backward."""
        plane, cache = {}, {}
        for probe in range(n_probes):
            for indices in groups:
                incoming_grad = _stack_to_device(
                    [plane_in[(probe, index)] for index in indices], device=runner.device)
                x_in = _stack_to_device([boundaries[index] for index in indices],
                                        device=runner.device,
                                        dtype=runner.dtype).detach().requires_grad_(True)
                batch = _chain_group_batch(runner, batches, indices, cache)
                with target_pass():
                    out = runner.isolated_layer(batch, layer_index, x_in, pass_state={})
                    torch.autograd.backward([out], [incoming_grad])
                gradient = x_in.grad.detach().to("cpu")
                start = 0
                for index in indices:
                    rows = int(boundaries[index].shape[0])
                    plane[(probe, index)] = gradient[start:start + rows].clone()
                    start += rows
                out = x_in = incoming_grad = gradient = None
        return plane

    def roll_plane(layer_pass):
        """The chain's fused roll of the layer, batch ``group_size``."""
        plane = {}
        backwards = render_free_layer_roll(
            runner, storage=None, batches=batches, layer=layer_index,
            cotangents=owners, n_probes=n_probes, incoming_entries=None,
            incoming_tensor=lambda probe, batch: plane_in[(probe, batch)],
            roll=lambda tensor, batch, probe: plane.__setitem__((probe, batch), tensor),
            batch_size=group_size, probe_fusion=True, layer_pass=layer_pass)
        if backwards != n_probes * count:
            raise RuntimeError(f"the roll ran {backwards} backwards, not {n_probes * count}")
        return plane

    # The experts' own arguments, by the class's signature: the top-k each
    # forward routed its tokens to. Kept on the device until the arm's clock
    # stops, so recording them adds no synchronization inside a timed arm.
    route_log = []
    experts_signature = inspect.signature(type(layer.mlp.experts).forward)

    def record_route(module, args, kwargs):
        bound = experts_signature.bind(module, *args, **kwargs)
        route_log.append(bound.arguments["top_k_index"].detach().clone())

    route_hook = layer.mlp.experts.register_forward_pre_hook(record_route, with_kwargs=True)
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=_power_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    arms, planes, routes = {}, {}, {}

    def arm(name, run):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident_before = torch.cuda.memory_allocated()
        before = kda_chunk.counts()
        route_log.clear()
        start = time.time()
        plane = run()
        torch.cuda.synchronize()
        end = time.time()
        after = kda_chunk.counts()
        forwards = [index.to("cpu") for index in route_log]
        route_log.clear()
        routes[name] = forwards
        watts = [w for t, w in samples if start <= t <= end]
        arms[name] = {"wall_s": end - start, "window_unix": [start, end],
                      "kernel_counts": {key: after[key] - before[key] for key in after},
                      "power_w_mean": (sum(watts) / len(watts)) if watts else None,
                      "power_samples": len(watts), "plane_sha256": _plane_digest(plane),
                      "resident_before_bytes": resident_before,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "route_forwards": len(forwards),
                      "routes_sha256": _plane_digest(
                          {(position,): index for position, index in enumerate(forwards)})}
        planes[name] = plane

    from contextlib import nullcontext
    kernel = capture.AdmittedKdaKernel(SwapDispatch(), {"name": "kda_gram_v1"}, None)
    target_pass = partial(kernel.layer_pass, layer, site="target", layer=layer_index)
    chain_pass = partial(kernel.layer_pass, layer, site="chain", layer=layer_index)
    for run in (1, 2):
        arm(f"fallback_capture_{run}", lambda: capture_plane(nullcontext))
        arm(f"fallback_roll_{run}", lambda: roll_plane(None))
        arm(f"kernel_capture_{run}", lambda: capture_plane(target_pass))
        arm(f"kernel_roll_{run}", lambda: roll_plane(chain_pass))
    stop.set()
    sampler.join(timeout=5)
    route_hook.remove()
    per_group = {}
    for name, forwards in routes.items():
        passes = n_probes if "_capture_" in name else 1
        per_group[name] = _group_routes(forwards, len(groups), passes)
    route_comparisons = {
        f"kernel_vs_fallback_{site}": _compare_routes(per_group[f"kernel_{site}_1"][0],
                                                      per_group[f"fallback_{site}_1"][0])
        for site in ("capture", "roll")}
    route_comparisons.update({
        f"{mode}_capture_vs_roll_run{run}": _compare_routes(per_group[f"{mode}_capture_{run}"][0],
                                                            per_group[f"{mode}_roll_{run}"][0])
        for mode in ("fallback", "kernel") for run in (1, 2)})
    route_comparisons.update({
        f"{mode}_{site}_run1_vs_run2": _compare_routes(per_group[f"{mode}_{site}_1"][0],
                                                       per_group[f"{mode}_{site}_2"][0])
        for mode in ("fallback", "kernel") for site in ("capture", "roll")})

    comparisons = {
        f"{mode}_capture_vs_roll_run{run}": _compare(planes[f"{mode}_capture_{run}"],
                                                     planes[f"{mode}_roll_{run}"])
        for mode in ("fallback", "kernel") for run in (1, 2)}
    comparisons.update({
        f"{mode}_{site}_run1_vs_run2": _compare(planes[f"{mode}_{site}_1"],
                                                planes[f"{mode}_{site}_2"])
        for mode in ("fallback", "kernel") for site in ("capture", "roll")})
    comparisons["kernel_vs_fallback_capture"] = _compare(planes["kernel_capture_1"],
                                                         planes["fallback_capture_1"])
    record = kernel.record()
    group_passes = n_probes * len(groups)
    expected_record = {
        "passes": 2 * group_passes, "calls": 2 * group_passes,
        "chain": {"layers": [layer_index], "passes": 2 * len(groups),
                  "calls": 2 * len(groups),
                  "gram_backward": 2 * 2 * n_probes * len(groups)},
        "scoped_passes": 2 * group_passes + 2 * len(groups)}
    fallback_silent = all(arms[name]["kernel_counts"] == {
        "calls": 0, "gram_forward": 0, "gram_backward": 0}
        for name in arms if name.startswith("fallback"))
    checks = {
        "capture_equals_roll_within_each_mode": all(
            comparisons[f"{mode}_capture_vs_roll_run{run}"]["equal"]
            for mode in ("fallback", "kernel") for run in (1, 2)),
        "each_arm_repeats_bitwise": all(
            comparisons[f"{mode}_{site}_run1_vs_run2"]["equal"]
            for mode in ("fallback", "kernel") for site in ("capture", "roll")),
        "kernel_differs_from_fallback": not comparisons["kernel_vs_fallback_capture"]["equal"],
        "kernel_record_counts_every_pass": all(
            record[key] == value for key, value in expected_record.items()),
        "fallback_arms_ran_no_kernel": fallback_silent,
        "global_restored": modeling.chunk_kimi_delta_attention
        is not kda_chunk.chunk_kimi_delta_attention,
    }
    result = {
        "schema": "prismaquant.kda_kernel_mode_receipt.v1",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0),
        "kernel_source_sha256": kda_chunk.source_sha256(),
        "layer": layer_index, "layer_type": config.layer_types[layer_index],
        "geometry": {"batches": count, "capture_batch": group_size,
                     "chain_batch": group_size, "probe_fusion": True,
                     "probes": n_probes, "seqlen": seq, "hc_mult": hc, "hidden": hidden},
        "layer_resident_bytes": layer_resident_bytes,
        "peak_reserved_bytes": max(a["peak_reserved_bytes"] for a in arms.values()),
        "peak_allocated_bytes": max(a["peak_allocated_bytes"] for a in arms.values()),
        "arms": arms, "comparisons": comparisons,
        "routes": {"comparisons": route_comparisons,
                   "stable_within_each_arm": {name: per_group[name][1] for name in per_group},
                   "note": ("recorded, not checked: synthetic weights, so the counts show "
                            "the route-flip mechanism, not GLM's rate")},
        "kernel_record": record, "expected_record": expected_record,
        "checks": checks, "pass": all(checks.values()),
    }
    (out_dir / "receipt.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "pass": result["pass"]}, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
