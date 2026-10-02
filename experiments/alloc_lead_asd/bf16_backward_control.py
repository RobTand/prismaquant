"""Same-graph, same-probe backward API and fixed-cotangent contraction control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.alloc_lead_asd import a_side_diag as diag
from experiments.alloc_lead_asd.run_pair import load_source_model, load_staged_inputs, runtime_identity
from prismaquant.residency_map import residency_report
from prismaquant.prismabuild_progress import commit


def relative_max(a, b):
    a, b = a.double(), b.double()
    rms = float(a.square().mean().sqrt())
    difference = float((a - b).abs().max())
    return {"max_abs": difference, "rms_reference": rms,
            "max_rel_to_rms": difference / rms if rms else (0.0 if not difference else None),
            "bitwise_equal": torch.equal(a, b)}


def control(model, ids):
    units = {name: module for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.")}
    names = list(units)
    with diag.Capture(units) as capture:
        logits = model(input_ids=ids[:1].to("cuda"), use_cache=False).logits
    operands, ys = {}, []
    a_spec, w_spec = diag.fr.get_format(diag.A8), diag.fr.get_format(diag.W4)
    with torch.no_grad():
        for name in names:
            x, y = capture.store[name]
            weight = units[name].weight.float()
            x2 = x.reshape(-1, x.shape[-1]).float()
            dx = diag._activation_qdq(x, a_spec, {}, name).reshape_as(x2).float() - x2
            dw = w_spec.quantize_dequantize(weight) - weight
            operands[name] = (weight, x2, dx, dw, dx @ weight.T, x2 @ dw.T)
            ys.append(y)
    capture.store.clear()
    hook_values, hook_counts, handles = {}, {}, []
    for name, y in zip(names, ys):
        def record(g, name=name):
            hook_values[name] = g.detach().clone()
            hook_counts[name] = hook_counts.get(name, 0) + 1
            return g
        handles.append(y.register_hook(record))
    results, scalars = [], []
    try:
        for seed in (7000, 7001):
            # The same scalar and primal graph serve every traversal below.
            probe = diag.fisher_probe_scalar(logits, seed=seed, token_scope="all",
                temperature=1.0, distribution="rademacher", token_count_override=2048,
                global_row_offset=0)
            hook_values.clear(); hook_counts.clear()
            returned = torch.autograd.grad(probe, ys, retain_graph=True)
            g_grad = {name: value.detach().clone() for name, value in zip(names, returned)}
            grad_counts = dict(hook_counts)
            if any(not torch.equal(g_grad[name], hook_values[name]) for name in names):
                raise RuntimeError("autograd.grad result differs from its observed hook cotangent")
            del returned
            hook_values.clear(); hook_counts.clear()
            probe.backward(retain_graph=True)
            g_backward, backward_counts = dict(hook_values), dict(hook_counts)
            hook_values.clear(); hook_counts.clear()
            probe.backward(retain_graph=True)
            g_repeat, repeat_counts = dict(hook_values), dict(hook_counts)
            if any(counts != {name: 1 for name in names}
                   for counts in (grad_counts, backward_counts, repeat_counts)):
                raise RuntimeError("a backward API repeated or skipped a Linear output hook")
            cohort, per_unit = {}, []
            with torch.no_grad():
                for api, grads in (("grad", g_grad), ("backward", g_backward)):
                    operator, dot = [], []
                    for name in names:
                        weight, x2, dx, dw, dy_a, dy_w = operands[name]
                        g2 = grads[name].reshape(-1, grads[name].shape[-1]).float()
                        operator.append(torch.stack((((g2.T @ dx) * weight).sum(),
                                                     ((g2.T @ x2) * dw).sum())))
                        dot.append(torch.stack(((g2 * dy_a).sum(), (g2 * dy_w).sum())))
                    cohort[api] = {"operator": torch.stack(operator).cpu().double(),
                                   "dot": torch.stack(dot).cpu().double()}
                for name in names:
                    per_unit.append({"unit": name,
                        "grad_vs_backward": relative_max(g_backward[name], g_grad[name]),
                        "backward_vs_repeat": relative_max(g_backward[name], g_repeat[name])})
                checks = {api: {component: relative_max(values["operator"][:, index],
                                                        values["dot"][:, index])
                                for index, component in enumerate(("A", "W"))}
                          for api, values in cohort.items()}
                effect = {component: relative_max(cohort["backward"]["operator"][:, index],
                                                  cohort["grad"]["operator"][:, index])
                          for index, component in enumerate(("A", "W"))}
                for api, values in checks.items():
                    if any(v["max_rel_to_rms"] is None or v["max_rel_to_rms"] > 1e-3
                           for v in values.values()):
                        raise RuntimeError(f"fixed-g FP32 contraction mismatch: {api}: {values}")
            results.append({"seed": seed, "fixed_g_contractions": checks,
                            "backward_api_projection_effect": effect, "units": per_unit})
            scalars.append(cohort)
            del g_grad, g_backward, g_repeat, probe
    finally:
        for handle in handles:
            handle.remove()
    return names, results, scalars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(0)
    config, generation, state, ids = load_staged_inputs(args.data_manifest_sha256)
    identity = {"manifest_sha256": args.data_manifest_sha256,
                "input_prefix_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "row": 0, "seeds": [7000, 7001], "normalization": 2048,
                "same_primal_graph": True, "same_probe_scalar": True,
                "source_commit": os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"),
                "input_residency": residency_report()}
    diag.atomic_json_dump(identity, str(output / "inputs.ready.json")); commit(1, "startup")
    model = load_source_model(config, state, dtype=torch.bfloat16, device="cuda")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.get_input_embeddings().register_forward_hook(lambda m, i, y: y.requires_grad_(True))
    identity["runtime"] = runtime_identity(model)
    commit(1, "control")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        names, results, scalars = control(model, ids)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / "control.trace.json.gz"))
    diag.atomic_torch_save({"units": names, "scalars": scalars}, str(output / "control.pt"))
    diag.atomic_json_dump({"identity": identity, "results": results, "complete": True},
                          str(output / "control.json")); commit(2, "control")
    print(json.dumps({"complete": True, "results": [{"seed": r["seed"],
        "fixed_g_contractions": r["fixed_g_contractions"],
        "backward_api_projection_effect": r["backward_api_projection_effect"],
        "units_with_api_difference": sum(not u["grad_vs_backward"]["bitwise_equal"] for u in r["units"]),
        "units_with_repeat_difference": sum(not u["backward_vs_repeat"]["bitwise_equal"] for u in r["units"])}
        for r in results]}, sort_keys=True), flush=True); commit(3, "publish")


if __name__ == "__main__":
    main()
