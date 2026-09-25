"""Compare the KDA capture kernel and the executed fallback with a float64 oracle (PQ #1199).

Run inside the GLM derivative image, through PrismaBuild. Three
implementations see the same inputs:

* ``fallback``: the image's own ``chunk_kimi_delta_attention``, reached
  through its dispatch wrapper exactly as the layer reaches it, after
  checking that the wrapper dispatches to the Torch fallback and that the
  modeling file is the corrected one (``glm_source_derivative``).
* ``kernel``: ``prismaquant.kernels.kda_chunk.chunk_kimi_delta_attention``.
* ``oracle``: the executed fallback's source text, run in float64. The text
  is the pinned original with the reviewed premask edit, checked to occur
  verbatim in the image's modeling file, with ``torch.float32`` and
  ``.float()`` replaced by their float64 forms. Heads are independent, so it
  runs a few heads at a time.

Each case draws bf16 ``q``, ``k``, ``v``, ``beta``, an FP32 gate ``g`` and a
bf16 output cotangent at the production shape (batch 4, sequence 512, 64
heads, head dim 128), then compares the output and the ``q``, ``k``, ``v``,
``g`` and ``beta`` gradients in two paths:

* ``fp32``: the bf16 values upcast to FP32. No bf16 rounding enters, so the
  errors are each implementation's own FP32 arithmetic.
* ``bf16``: the production dtypes. The output and the ``q``, ``k``, ``v`` and
  ``beta`` gradients are rounded to bf16 at the end.

The kernel passes a tensor's metric when its error is no larger than the
fallback's, or larger by no more than the two implementations' final
roundings can explain. Both return a result rounded to its dtype, and round
to nearest moves each element by at most ``u |x|`` (``u`` is 2^-24 for FP32
and 2^-8 for bf16). If the kernel's result is at least as accurate as the
fallback's before that rounding, its largest element error can still exceed
the fallback's by up to ``2 u max|ref|`` and its relative Frobenius error by
up to ``2 u``. An excess above that bound shows the kernel is less accurate
than the fallback. A kernel result that is not finite where the fallback's
is finite fails outright.

Every kernel run is repeated once and compared bit for bit with the first.
The output also carries the kernel's qualification candidate: the identity
fields ``kernels/kda_chunk_qualification.json`` binds, read from this runtime
(``glm_kda_capture_kernel.qualification_candidate``).

Outputs ``--out/numerics.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PINNED = (Path(__file__).parent
          / "measurements/glm-kda-backward-source-repro-20260908/pinned-functions.json")
OLD = "(g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()"
NEW = ("(g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(mask.triu(diagonal=1).unsqueeze(-1), 0)"
       ".exp().float()")
TENSORS = ("o", "dq", "dk", "dv", "dg", "dbeta")
U32 = 2.0 ** -24
#: The unit roundoff of each result dtype under round to nearest.
UNIT_ROUNDOFF = {torch.float32: 2.0 ** -24, torch.bfloat16: 2.0 ** -8}


def executed_fallback():
    """The image's dispatch-wrapped fallback, and its modeling source text."""
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    from prismaquant.glm_source_derivative import CORRECTED_MODELING_SHA256
    raw = Path(modeling.__file__).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CORRECTED_MODELING_SHA256:
        raise SystemExit(f"modeling source {digest} is not the corrected {CORRECTED_MODELING_SHA256}")
    function = modeling.chunk_kimi_delta_attention
    closure = inspect.getclosurevars(function).nonlocals
    if closure.get("implementation") is not closure.get("torch_function") \
            or closure.get("is_new_implementation") is not False:
        raise SystemExit("the image's KDA dispatch is not the decorated Torch fallback")
    return function, raw.decode(), {"modeling_path": modeling.__file__, "modeling_sha256": digest}


def pinned_texts():
    """``l2norm`` and the executed fallback: the pinned original with the premask edit."""
    pinned = json.loads(PINNED.read_text())
    sources = pinned["functions"]
    for row in sources.values():
        if hashlib.sha256(row["source"].encode()).hexdigest() != row["sha256"]:
            raise SystemExit("pinned function text differs from its digest")
    kda = sources["chunk_kimi_delta_attention"]["source"]
    if kda.count(OLD) != 1:
        raise SystemExit("the pinned fallback does not hold the reviewed expression once")
    return sources["l2norm"]["source"], kda.replace(OLD, NEW)


def compile_pinned(norm, kda, dtype):
    """The text as a function; float64 replaces ``torch.float32`` and ``.float()``."""
    code = norm + "\n" + kda
    if dtype == torch.float64:
        code = code.replace("torch.float32", "torch.float64").replace(".float()", ".double()")
    elif dtype != torch.float32:
        raise SystemExit(f"no pinned form in {dtype}")
    scope = {"torch": torch, "F": F}
    exec(compile(code, f"<executed_glm_kda_fallback_{dtype}>", "exec"), scope)  # noqa: S102
    return scope["chunk_kimi_delta_attention"], hashlib.sha256(code.encode()).hexdigest()


def oracle_function(live_source):
    """The executed fallback's text, run in float64."""
    norm, kda = pinned_texts()
    if kda not in live_source or norm not in live_source:
        raise SystemExit("the oracle text does not occur verbatim in the executed modeling file")
    function, code_sha256 = compile_pinned(norm, kda, torch.float64)
    return function, {
        "pinned_sha256": hashlib.sha256(PINNED.read_bytes()).hexdigest(),
        "oracle_code_sha256": code_sha256}


def gate(name, shape, generator, device):
    """The FP32 gate ``g`` for one case, ``[B, T, H, K]``."""
    def uniform(low, high):
        return torch.empty(shape, device=device).uniform_(low, high, generator=generator)
    batch, length, heads, dim = shape
    if name == "glm_gate_random":
        # GLM's own form, -5 * sigmoid(x), over a wide spread of x.
        z = torch.randn(shape, device=device, generator=generator) * 3.0
        return -5.0 * torch.sigmoid(z)
    if name == "bound_exact":
        return torch.full(shape, -5.0, device=device)
    if name == "bound_inexact":
        return -5.0 + 1e-3 * uniform(0.0, 1.0)
    if name == "weak_decay":
        return -1e-4 * uniform(0.0, 1.0)
    if name == "zero_decay":
        return torch.zeros(shape, device=device)
    if name == "mixed_channels":
        strong = -5.0 * uniform(0.9, 1.0)
        weak = -1e-4 * uniform(0.0, 1.0)
        channel = (torch.arange(dim, device=device) % 2 == 0).view(1, 1, 1, dim)
        return torch.where(channel, strong, weak)
    if name == "alternating_steps":
        strong = -5.0 * uniform(0.9, 1.0)
        weak = -1e-4 * uniform(0.0, 1.0)
        step = (torch.arange(length, device=device) % 2 == 0).view(1, length, 1, 1)
        return torch.where(step, strong, weak)
    if name == "correlated_keys_beta1":
        return -1e-3 * uniform(0.0, 1.0)
    if name == "beyond_bound":
        return torch.full(shape, -20.0, device=device) + uniform(-1.0, 0.0)
    raise SystemExit(f"unknown case {name}")


CASES = ("glm_gate_random", "bound_exact", "bound_inexact", "weak_decay", "zero_decay",
         "mixed_channels", "alternating_steps", "correlated_keys_beta1", "beyond_bound")


def draw_case(name, *, batch, length, heads, dim, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, length, heads, dim)

    def normal(*size):
        return torch.randn(size, device=device, generator=generator)

    q, k, v = normal(*shape), normal(*shape), normal(*shape)
    beta = torch.sigmoid(normal(batch, length, heads))
    if name == "correlated_keys_beta1":
        # Nearly parallel keys and beta near 1: the worst-conditioned
        # (I + A_kk) the gate range allows.
        k = normal(batch, 1, heads, dim).expand(shape) + 0.05 * normal(*shape)
        beta = 1.0 - 1e-3 * torch.rand((batch, length, heads), device=device, generator=generator)
    g = gate(name, shape, generator, device).contiguous()
    stimulus = normal(*shape)
    bf16 = [x.contiguous().to(torch.bfloat16) for x in (q, k, v, beta, stimulus)]
    return {"q": bf16[0], "k": bf16[1], "v": bf16[2], "g": g, "beta": bf16[3], "stimulus": bf16[4]}


def run(function, inputs, dtype, stimulus):
    leaves = [inputs[name].detach().to(dtype if name != "g" else
                                        (torch.float64 if dtype == torch.float64 else torch.float32))
              .clone().requires_grad_(True) for name in ("q", "k", "v", "g", "beta")]
    output, _ = function(leaves[0], leaves[1], leaves[2], g=leaves[3], beta=leaves[4],
                         use_qk_l2norm_in_kernel=True)
    torch.autograd.backward([output], [stimulus.to(output.dtype)])
    return [output.detach()] + [leaf.grad.detach() for leaf in leaves]


def run_sliced(function, inputs, heads_per_slice):
    heads = inputs["q"].shape[2]
    parts = []
    for start in range(0, heads, heads_per_slice):
        stop = min(heads, start + heads_per_slice)
        # Dim 2 is the head dim of every input: [B, T, H, D] and beta's [B, T, H].
        sliced = {name: value[:, :, start:stop] for name, value in inputs.items()}
        parts.append(run(function, sliced, torch.float64, sliced["stimulus"].to(torch.float64)))
        torch.cuda.synchronize()
    return [torch.cat([part[index] for part in parts], dim=2) for index in range(len(TENSORS))]


def compare(actual, reference):
    actual64 = actual.to(torch.float64)
    nonfinite = int((~torch.isfinite(actual64)).sum())
    difference = (actual64 - reference).abs()
    reference_fro = float(torch.linalg.vector_norm(reference))
    reference_max = float(reference.abs().max())
    max_abs = float(difference.max()) if nonfinite == 0 else math.inf
    fro = float(torch.linalg.vector_norm(difference)) if nonfinite == 0 else math.inf
    return {"max_abs": max_abs,
            "rel_fro": fro / reference_fro if reference_fro > 0 else None,
            "rel_max": max_abs / reference_max if reference_max > 0 else None,
            "rel_fro_in_fp32_u": (fro / reference_fro / U32) if reference_fro > 0 else None,
            "reference_fro": reference_fro, "reference_max_abs": reference_max,
            "nonfinite": nonfinite}


def pairwise(left, right):
    left64, right64 = left.to(torch.float64), right.to(torch.float64)
    difference = (left64 - right64).abs()
    scale = float(torch.linalg.vector_norm(right64))
    return {"bit_equal": bool(torch.equal(left, right)),
            "differing_elements": int((left != right).sum()),
            "elements": int(left.numel()),
            "max_abs": float(difference.max()),
            "rel_fro": float(torch.linalg.vector_norm(difference)) / scale if scale > 0 else None}


def verdict(kernel, fallback, dtype):
    """The kernel's excess error over the fallback's, against the rounding bound."""
    unit = UNIT_ROUNDOFF[dtype]
    rows = {}
    for metric, scale in (("max_abs", fallback["reference_max_abs"]), ("rel_fro", 1.0)):
        excess = kernel[metric] - fallback[metric]
        bound = 2.0 * unit * scale
        rows[metric] = {"no_worse": kernel[metric] <= fallback[metric],
                        "excess": excess, "bound": bound,
                        "excess_in_u": excess / (unit * scale) if scale > 0 else None,
                        "within_bound": excess <= bound}
    rows["dtype"] = str(dtype).removeprefix("torch.")
    rows["finite_where_fallback_finite"] = not (kernel["nonfinite"] > fallback["nonfinite"])
    rows["pass"] = rows["finite_where_fallback_finite"] and all(
        rows[metric]["within_bound"] for metric in ("max_abs", "rel_fro"))
    return rows


def table(result):
    """One Markdown row per case, path and tensor."""
    lines = ["| Case | Path | Tensor | Fallback max abs | Kernel max abs | "
             "Fallback rel Fro | Kernel rel Fro | Excess (u) | Pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, case in result["cases"].items():
        for path in ("fp32", "bf16"):
            if path not in case:
                continue
            rows = case[path]
            for tensor in TENSORS:
                f, k, v = rows["fallback"][tensor], rows["kernel"][tensor], rows["verdict"][tensor]
                excess = max(v["max_abs"]["excess_in_u"] or 0.0, v["rel_fro"]["excess_in_u"] or 0.0)
                lines.append(
                    f"| {name} | {path} | {tensor} | {f['max_abs']:.3e} | {k['max_abs']:.3e} | "
                    f"{f['rel_fro']:.3e} | {k['rel_fro']:.3e} | {max(excess, 0.0):.2f} | "
                    f"{'yes' if v['pass'] else 'NO'} |")
    return "\n".join(lines)


def timed(label, fn):
    torch.cuda.synchronize()
    start = time.time()
    result = fn()
    torch.cuda.synchronize()
    return result, {label: time.time() - start}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--heads-per-slice", type=int, default=8)
    parser.add_argument("--no-bf16-path", action="store_true")
    args = parser.parse_args(argv)

    from prismaquant.kernels import kda_chunk
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic

    # The capture's own matmul settings: FP32 matmuls at "highest" precision.
    pin_matmul_arithmetic()
    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback, live_source, executed = executed_fallback()
    oracle, oracle_identity = oracle_function(live_source)
    import triton
    result = {
        "schema": "prismaquant.kda_kernel_numerics.v1",
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda, "triton": triton.__version__,
        "machine": platform.machine(), "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "matmul": {"float32_precision": torch.get_float32_matmul_precision(),
                   "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                   "bf16_reduced_precision_reduction":
                       bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)},
        "executed": executed, "oracle": oracle_identity,
        "kernel": {"name": kda_chunk.NAME, "source_sha256": kda_chunk.source_sha256()},
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": {"batch": args.batch, "seqlen": args.seqlen, "heads": args.heads, "head_dim": 128},
        "cases": {},
    }
    from prismaquant.glm_kda_capture_kernel import qualification_candidate
    result["qualification_candidate"] = qualification_candidate(device)
    for index, name in enumerate(args.cases.split(",")):
        inputs = draw_case(name, batch=args.batch, length=args.seqlen, heads=args.heads, dim=128,
                           device=device, seed=1199 + index)
        stimulus32 = inputs["stimulus"].to(torch.float32)
        g = inputs["g"]
        case = {"g": {"min": float(g.min()), "max": float(g.max()), "mean": float(g.mean())},
                "wall_s": {}}
        reference, wall = timed("oracle", lambda: run_sliced(oracle, inputs, args.heads_per_slice))
        case["wall_s"].update(wall)
        paths = [("fp32", torch.float32, stimulus32)]
        if not args.no_bf16_path:
            paths.append(("bf16", torch.bfloat16, inputs["stimulus"]))
        for path, dtype, stimulus in paths:
            rows = {}
            got = {}
            for impl, function in (("fallback", fallback), ("kernel", kda_chunk.chunk_kimi_delta_attention)):
                values, wall = timed(f"{path}_{impl}", lambda: run(function, inputs, dtype, stimulus))
                case["wall_s"].update(wall)
                got[impl] = values
                rows[impl] = {tensor: compare(value, ref)
                              for tensor, value, ref in zip(TENSORS, values, reference)}
                if impl == "kernel":
                    again = run(function, inputs, dtype, stimulus)
                    rows["kernel_repeat_bit_equal"] = {
                        tensor: bool(torch.equal(first, second))
                        for tensor, first, second in zip(TENSORS, values, again)}
                    del again
                torch.cuda.empty_cache()
            rows["kernel_vs_fallback"] = {tensor: pairwise(k, f) for tensor, k, f in
                                          zip(TENSORS, got["kernel"], got["fallback"])}
            rows["kernel_no_worse"] = {
                tensor: {metric: (rows["kernel"][tensor][metric] <= rows["fallback"][tensor][metric])
                         for metric in ("max_abs", "rel_fro")}
                for tensor in TENSORS}
            rows["verdict"] = {tensor: verdict(rows["kernel"][tensor], rows["fallback"][tensor],
                                               value.dtype)
                               for tensor, value in zip(TENSORS, got["kernel"])}
            case[path] = rows
            del got
        result["cases"][name] = case
        del reference, inputs
        torch.cuda.empty_cache()
        # Each finished case is on disk before the next one starts.
        (out_dir / "numerics.partial.json").write_text(json.dumps(result, allow_nan=True) + "\n")
        print(json.dumps({"case": name, "wall_s": case["wall_s"],
                          "fp32_rel_fro": {impl: {t: case["fp32"][impl][t]["rel_fro"] for t in TENSORS}
                                           for impl in ("fallback", "kernel")}}), flush=True)
    result["kernel"]["compiled"] = kda_chunk.compiled_kernels()
    result["kernel"]["counts"] = kda_chunk.counts()
    failures = [(name, path, tensor) for name, case in result["cases"].items()
                for path in ("fp32", "bf16") if path in case
                for tensor in TENSORS if not case[path]["verdict"][tensor]["pass"]]
    unequal = [(name, path, tensor) for name, case in result["cases"].items()
               for path in ("fp32", "bf16") if path in case
               for tensor in TENSORS if not case[path]["kernel_repeat_bit_equal"][tensor]]
    result["summary"] = {"pass": not failures and not unequal, "failures": failures,
                         "repeat_unequal": unequal}
    result["table"] = table(result)
    print(result["table"], flush=True)
    print(json.dumps(result["summary"]), flush=True)
    target = out_dir / "numerics.json"
    target.write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    print(f"numerics {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
