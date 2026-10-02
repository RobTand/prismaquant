"""Small-scale validation of the joint-AURA A-side estimator (research diagnostic).

For a dense causal LM, price per-token dynamic FP8 E4M3 activation quantization
of every decoder Linear with the production estimator's arithmetic -- the
signed per-probe projections of ``prismaquant.joint_aura.SignedJointProjectionLease``
driven by ``kl_fisher.fisher_probe_scalar`` global-row Rademacher probes with the
global token normalizer, token scope "all", temperature 1 (as Stage A/B do) --
and measure, on the same tokens:

  P_add    sum_u 0.5 mean_p a_{p,u}^2         the production additive price
  P_joint  0.5 mean_p (sum_u a_{p,u})^2        keeps cross-unit terms
  S_real   0.5 mean_p (r_p . dz_real)^2        same probes, realized logit change
  Q_real   0.5 dz^T F dz / N (exact)           quadratic of the realized change
  KL_true  KL(p0 || p_arm), float64, full vocab, mean over the N tokens

The ratio chain isolates one assumption at a time:
  P_add/P_joint   cross-unit terms
  P_joint/S_real  linearization through the network (corr(s_lin, s_real) per probe)
  S_real/Q_real   probe sampling of the realized change
  Q_real/KL_true  second order in logit space

Execution (v2, sequence-major).  The lease contracts G^T X per probe: for K probes on one
sequence that is K [out x in] GEMMs per unit.  The same scalars are
sum_t g_{p,t} . dY_t, where dY is the unit's output perturbation (dX W^T for the A side,
X dW^T for the W side), so pass 1 forms dY once per sequence and takes one dot product per
probe; the K probe cotangents come from K backward passes over one retained forward graph.
Pass 2 (the arms) computes the clean logits, the probe vectors and every arm's realized
logit change once per sequence instead of once per (arm, sequence).  ``--profile`` runs the
v1 (lease) path and the v2 path on sequence 0 under torch.profiler, and checks that their
components agree (the cross-check that keeps this tied to the production arithmetic); the
closed-form KL is checked against the v1 two-log-softmax KL the same way.

Rows are independent probe draws (global-row seeds), so each probe projection decomposes
per sequence, which gives a per-sequence predicted-vs-measured scatter.  A W-only control
(NVFP4A16 RTN weights, A16) is priced and measured the same way, plus the W+A arm GLM's G3
measures (A increment on top of quantized W).  Nothing here is production code.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from safetensors import safe_open

from prismaquant import format_registry as fr
from prismaquant.joint_aura import SignedJointProjectionLease
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.perturbed_x_cache import _activation_qdq

A8 = "FP8_E4M3"
A8N = "FP8_E4M3_NOPOS0"
W4 = "NVFP4A16"
SPECS = (A8, A8N, W4)
COMPONENT = {A8: "activation", A8N: "activation", W4: "weight"}
IMPL = "a_side_diag.v2.seqmajor"
DEVICE = "cuda"          # tests set "cpu"


@contextmanager
def backward_policy(deterministic=False):
    """Opt-in research backward policy; preserve the primal and prior flags."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        if deterministic:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_torch_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(obj, handle)
    os.replace(tmp, path)


def load_tokens(path, key, n):
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibration token artifact does not exist: {path}; "
                                "provide the prepared immutable input artifact")
    with safe_open(path, "pt") as handle:
        ids = handle.get_tensor(key)
    if n > ids.shape[0]:
        raise SystemExit(f"text {key} has {ids.shape[0]} rows < {n}")
    return ids[:n].contiguous()


def nopos0_spec(base):
    """A8 with position 0 of every sequence left unquantized (diagnostic)."""
    inner = base.activation_quantize_dequantize

    def qdq(x):
        if x.dim() != 3:
            raise RuntimeError("nopos0 QDQ expects [batch, seq, features]")
        out = inner(x).clone()
        out[:, 0, :] = x[:, 0, :]
        return out

    return dataclasses.replace(base, name=A8N, activation_quantize_dequantize=qdq)


class Perturb:
    """Realize a perturbation: A-side QDQ pre-hooks and/or W4 RTN weight swap."""

    def __init__(self, units, a_names=(), a_spec=None, w_names=(), w_spec=None):
        self.units, self.a_names, self.a_spec = units, list(a_names), a_spec
        self.w_names, self.w_spec = list(w_names), w_spec
        self.handles, self.saved = [], {}

    def __enter__(self):
        with torch.no_grad():
            for name in self.w_names:
                weight = self.units[name].weight
                self.saved[name] = weight.detach().clone()
                weight.copy_(self.w_spec.quantize_dequantize(weight.float()).to(weight.dtype))
        for name in self.a_names:
            def pre(module, args, _name=name):
                return (_activation_qdq(args[0], self.a_spec, {}, _name),) + tuple(args[1:])
            self.handles.append(self.units[name].register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        with torch.no_grad():
            for name, saved in self.saved.items():
                self.units[name].weight.copy_(saved)
        self.saved.clear()


def probe_grad(z0, seed, row, n_global, scope, temperature):
    """r_p = d probe / d logits for one global row, exactly as the pricing seeds it."""
    leaf = z0.detach().requires_grad_(True)
    probe = fisher_probe_scalar(leaf, seed=seed, token_scope=scope, temperature=temperature,
                                distribution="rademacher", token_count_override=n_global,
                                global_row_offset=row)
    (grad,) = torch.autograd.grad(probe, leaf)
    return grad


# ----------------------------------------------------------------------------- v1 (reference)
def measure_v1(model, ids, n_global, seeds, scope, temperature, context, with_probes=True):
    """v1 arm measurement (per arm, per sequence): kept as the reference for --profile."""
    n = ids.shape[0]
    kl = torch.zeros(n, dtype=torch.float64)
    q = torch.zeros(n, dtype=torch.float64)
    s_real = torch.zeros(len(seeds), n, dtype=torch.float64)
    for i in range(n):
        x = ids[i:i + 1].to(DEVICE)
        with torch.no_grad():
            z0 = model(input_ids=x, use_cache=False).logits.float()
        with torch.no_grad(), context():
            za = model(input_ids=x, use_cache=False).logits.float()
        dz = za - z0
        with torch.no_grad():
            lp0 = torch.log_softmax(z0.double() / temperature, dim=-1)
            lpa = torch.log_softmax(za.double() / temperature, dim=-1)
            kl_tok = (lp0.exp() * (lp0 - lpa)).sum(-1)[0]
            kl[i] = float(kl_tok.sum()) / n_global
            dzd = dz.double() / temperature
            p0 = lp0.exp()
            mean = (p0 * dzd).sum(-1, keepdim=True)
            q[i] = float(0.5 * (p0 * (dzd - mean) ** 2).sum()) / n_global
        for k, seed in enumerate(seeds if with_probes else ()):
            r = probe_grad(z0, seed, i, n_global, scope, temperature)
            s_real[k, i] = float((r.double() * dz.double()).sum())
    return {"kl": kl, "q": q, "s_real": s_real}


def price_v1(model, units, names, ids, seeds, specs_obj, n_global, scope, temperature,
             *, deterministic_backward=False):
    """The production lease over every (probe, sequence): comps[spec, unit, probe, seq]."""
    n_seqs = ids.shape[0]
    deltas = {}
    with torch.no_grad():
        for name, module in units.items():
            weight = module.weight
            zero = torch.zeros((), device=weight.device, dtype=torch.float32).expand(weight.shape)
            deltas[(name, A8)] = zero
            deltas[(name, A8N)] = zero
            deltas[(name, W4)] = specs_obj[W4].quantize_dequantize(weight.float()) - weight.float()
    specs = {name: dict(specs_obj) for name in names}
    comps = torch.zeros(len(SPECS), len(names), len(seeds), n_seqs, dtype=torch.float64)
    with SignedJointProjectionLease(units, specs, deltas, activation_max_abs={}) as lease:
        for k, seed in enumerate(seeds):
            for i in range(n_seqs):
                lease.begin_probe()
                logits = model(input_ids=ids[i:i + 1].to(DEVICE), use_cache=False).logits
                probe = fisher_probe_scalar(logits, seed=seed, token_scope=scope,
                                            temperature=temperature, distribution="rademacher",
                                            token_count_override=n_global, global_row_offset=i)
                with backward_policy(deterministic_backward):
                    probe.backward()
                del logits, probe
                result = lease.finish_probe()
                for si, spec in enumerate(SPECS):
                    field = COMPONENT[spec]
                    for ui, name in enumerate(names):
                        value = result[(name, spec)]
                        if spec != W4 and value["weight"] != 0.0:
                            raise RuntimeError("zero-delta A row returned a W component")
                        comps[si, ui, k, i] = value[field]
    del deltas, lease
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return comps


# ----------------------------------------------------------------------------- v2 pass 1
class Capture:
    """Forward hooks that keep each unit's input (detached) and output (graph) for one pass."""

    def __init__(self, units):
        self.units, self.store, self.handles = units, {}, []

    def __enter__(self):
        for name, module in self.units.items():
            def hook(mod, inputs, output, _name=name):
                self.store[_name] = (inputs[0].detach(), output)
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def price_sequence(model, units, names, ids_row, row, seeds, specs_obj, n_global, scope,
                   temperature, comps, dens, *, deterministic_backward=False):
    """Pass 1 for one sequence: comps[spec, unit, probe, row] and the per-position
    diagonal densities dens[spec, unit, t] += c_{p,u,t}^2 (summed over probes)."""
    with Capture(units) as cap:
        logits = model(input_ids=ids_row, use_cache=False).logits
    ys = [cap.store[name][1] for name in names]
    dy_a, dy_w = [], []
    with torch.no_grad():
        for name in names:
            xin = cap.store[name][0]
            weight = units[name].weight.float()
            x2 = xin.reshape(-1, xin.shape[-1]).float()
            quantized = _activation_qdq(xin, specs_obj[A8], {}, name)
            dx = quantized.reshape_as(x2).float() - x2
            dy_a.append(dx @ weight.T)
            dw = specs_obj[W4].quantize_dequantize(weight) - weight
            dy_w.append(x2 @ dw.T)
            del quantized, dx, dw, weight, x2
    cap.store.clear()
    n_units, seqlen = len(names), ids_row.shape[1]
    buf = torch.empty(2, n_units, seqlen, device=ids_row.device, dtype=torch.float32)
    out = torch.empty(3, n_units, len(seeds), device=ids_row.device, dtype=torch.float64)
    dens_gpu = torch.zeros(3, n_units, seqlen, device=ids_row.device, dtype=torch.float64)
    for k, seed in enumerate(seeds):
        probe = fisher_probe_scalar(logits, seed=seed, token_scope=scope, temperature=temperature,
                                    distribution="rademacher", token_count_override=n_global,
                                    global_row_offset=row)
        with backward_policy(deterministic_backward):
            grads = torch.autograd.grad(probe, ys, retain_graph=k < len(seeds) - 1)
        with torch.no_grad():
            for ui, g in enumerate(grads):
                g2 = g.reshape(-1, g.shape[-1]).float()
                torch.linalg.vecdot(g2, dy_a[ui], dim=-1, out=buf[0, ui])
                torch.linalg.vecdot(g2, dy_w[ui], dim=-1, out=buf[1, ui])
            b = buf.double()
            out[0, :, k] = b[0].sum(-1)
            out[1, :, k] = b[0, :, 1:].sum(-1)
            out[2, :, k] = b[1].sum(-1)
            sq = b.pow(2)
            dens_gpu[0] += sq[0]
            dens_gpu[1, :, 1:] += sq[0, :, 1:]
            dens_gpu[2] += sq[1]
        del grads, probe
    comps[:, :, :, row] = out.cpu()
    dens += dens_gpu.cpu()
    del logits, ys, dy_a, dy_w, buf, out, dens_gpu


# ----------------------------------------------------------------------------- v2 pass 2
def kl_q_closed_form(p0, s0, dzd, temperature):
    """Per-position KL(p0 || softmax(z0 + dz)) and 0.5 dz^T F dz, float64.

    KL_t = log(sum_v p0 e^{dz}) - sum_v p0 dz (p0 normalized by its own sum s0), formed as
    log1p(sum p0 expm1(dz) / s0) - mu: one exp pass, no log-softmax difference.  ``dzd`` is
    the float64 difference of the two float32 logit tensors (exact)."""
    dzd = dzd / temperature
    mu = (p0 * dzd).sum(-1, keepdim=True) / s0
    kl_t = torch.log1p((p0 * torch.expm1(dzd)).sum(-1, keepdim=True) / s0) - mu
    q_t = 0.5 * (p0 * (dzd - mu) ** 2).sum(-1, keepdim=True) / s0
    return kl_t[0, :, 0], q_t[0, :, 0]


def arms_sequence(model, ids_row, row, plan, contexts, probe_arms, seeds, n_global, scope,
                  temperature, dz_stack, res):
    """Pass 2 for one sequence: every arm's KL/Q/kl_pos, and s_real for the probe arms."""
    with torch.no_grad():
        z0 = model(input_ids=ids_row, use_cache=False).logits.float()
        p0 = torch.softmax(z0.double() / temperature, dim=-1)
        s0 = p0.sum(-1, keepdim=True)
        for label, _, _ in plan:
            with contexts[label]():
                za = model(input_ids=ids_row, use_cache=False).logits.float()
            dz = za.double() - z0.double()
            del za
            kl_t, q_t = kl_q_closed_form(p0, s0, dz, temperature)
            res[label]["kl"][row] = float(kl_t.sum()) / n_global
            res[label]["q"][row] = float(q_t.sum()) / n_global
            res[label]["kl_pos"] += kl_t.cpu()
            if label in probe_arms:
                dz_stack[probe_arms[label]].copy_(dz.reshape(-1))
            del dz, kl_t, q_t
        del p0, s0
    if probe_arms:
        labels = sorted(probe_arms, key=probe_arms.get)
        for k, seed in enumerate(seeds):
            r = probe_grad(z0, seed, row, n_global, scope, temperature)
            with torch.no_grad():
                values = (dz_stack @ r.reshape(-1).to(dz_stack.dtype)).double().cpu()
            for j, label in enumerate(labels):
                res[label]["s_real"][k, row] = values[j]
            del r
    del z0


def build_plan(names, unit_price, ui, n_single, layer_arms):
    attn = [n for n in names if ".self_attn." in n]
    mlp = [n for n in names if ".mlp." in n]
    plan = [("A_all", names, (A8,)), ("A_attn", attn, (A8,)), ("A_mlp", mlp, (A8,)),
            ("A_all_nopos0", names, (A8N,)),
            ("W4_all", names, (W4,)), ("W4_attn", attn, (W4,)), ("W4_mlp", mlp, (W4,)),
            ("W4A8_all", names, (W4, A8))]
    if layer_arms:
        for layer in sorted({int(n.split(".")[2]) for n in names}):
            group = [n for n in names if n.split(".")[2] == str(layer)]
            plan.append((f"A_layer{layer:02d}", group, (A8,)))
    ranked = sorted(names, key=lambda n: float(unit_price[A8][ui[n]]), reverse=True)
    rng = random.Random(1234)
    picks = []
    for n in ranked[:6] + ranked[len(ranked) // 4:len(ranked) // 4 + 3] + \
            ranked[len(ranked) // 2:len(ranked) // 2 + 3] + rng.sample(ranked, k=6):
        if n not in picks:
            picks.append(n)
    for n in picks[:n_single]:
        plan.append((f"A_unit:{n}", [n], (A8,)))
        plan.append((f"W4_unit:{n}", [n], (W4,)))
    return plan


def make_contexts(plan, units, specs_obj):
    contexts = {}
    for label, group, parts in plan:
        a_spec = specs_obj[A8N] if A8N in parts else (specs_obj[A8] if A8 in parts else None)
        contexts[label] = (lambda g=group, p=parts, a=a_spec: Perturb(
            units, a_names=g if a is not None else (), a_spec=a,
            w_names=g if W4 in p else (), w_spec=specs_obj[W4]))
    return contexts


def load_partial(path, identity):
    if not os.path.exists(path):
        return None
    saved = torch.load(path)
    if saved.get("identity") != identity:
        log(f"ignoring {path}: identity differs")
        return None
    log(f"resuming from {path} at row {saved['done']}")
    return saved


def refuse_unbound_pricing_reuse(path):
    if path:
        raise SystemExit("cached pricing reuse is unsupported: this artifact has no immutable "
                         "loaded-model/run binding; use the existing identity-bound partial resume")


# ----------------------------------------------------------------------------- profile
def profile_and_crosscheck(args, model, units, names, ids, seeds, specs_obj, n_global, scope,
                           temperature):
    """v1 vs v2 on sequence 0 under torch.profiler, plus the component and KL cross-checks."""
    from torch.profiler import ProfilerActivity, profile
    row = ids[:1]
    k2 = seeds[:2]
    report = {}
    tables = []

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    sync = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
    sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"

    def timed(label, fn):
        sync()
        t = time.time()
        with profile(activities=activities) as prof:
            value = fn()
            sync()
        wall = time.time() - t
        tables.append(f"===== {label}: wall {wall:.3f}s\n" + prof.key_averages().table(
            sort_by=sort_key, row_limit=18))
        report[label] = {"wall_s": wall}
        if label.startswith("v2 pricing"):
            trace = args.output + ".pricing.trace.json.gz"
            prof.export_chrome_trace(trace)
            report[label]["trace"] = trace
        log(f"profile {label}: {wall:.3f}s")
        return value

    v1 = timed(f"v1 lease pricing, 1 seq x {len(k2)} probes",
               lambda: price_v1(model, units, names, row, k2, specs_obj, n_global, scope,
                                temperature, deterministic_backward=getattr(args, "deterministic_backward", False)))
    comps_v2 = torch.zeros(len(SPECS), len(names), len(seeds), 1, dtype=torch.float64)
    dens = torch.zeros(len(SPECS), len(names), ids.shape[1], dtype=torch.float64)
    timed(f"v2 pricing, 1 seq x {len(seeds)} probes",
          lambda: price_sequence(model, units, names, row.to(DEVICE), 0, seeds, specs_obj, n_global,
                                 scope, temperature, comps_v2, dens,
                                 deterministic_backward=getattr(args, "deterministic_backward", False)))
    report["v1_s_per_probe_seq"] = report[f"v1 lease pricing, 1 seq x {len(k2)} probes"]["wall_s"] / len(k2)
    report["v2_s_per_probe_seq"] = report[f"v2 pricing, 1 seq x {len(seeds)} probes"]["wall_s"] / len(seeds)
    diffs = {}
    for si, spec in enumerate(SPECS):
        a, b = v1[si, :, :, 0], comps_v2[si, :, :2, 0]
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise SystemExit(f"nonfinite component observation in {spec} crosscheck")
        rms = float(a.pow(2).mean().sqrt())
        if not math.isfinite(rms) or rms <= 0:
            raise SystemExit(f"invalid reference RMS in {spec} crosscheck: {rms}")
        maximum = float((a - b).abs().max())
        relative = maximum / rms
        if not math.isfinite(maximum) or not math.isfinite(relative):
            raise SystemExit(f"nonfinite component error in {spec} crosscheck")
        diffs[spec] = {"max_abs": maximum, "rms_v1": rms, "max_rel_to_rms": relative}
    report["component_crosscheck"] = diffs
    log(f"component cross-check v1 vs v2: {json.dumps(diffs)}")

    # one arm, v1 measurement vs v2 closed form
    plan = [("A_all", names, (A8,))]
    contexts = make_contexts(plan, units, specs_obj)
    m1 = timed("v1 arm A_all, 1 seq x 2 probes",
               lambda: measure_v1(model, row, n_global, k2, scope, temperature,
                                  contexts["A_all"]))
    res = {"A_all": {"kl": torch.zeros(1, dtype=torch.float64), "q": torch.zeros(1, dtype=torch.float64),
                     "kl_pos": torch.zeros(ids.shape[1], dtype=torch.float64),
                     "s_real": torch.zeros(len(k2), 1, dtype=torch.float64)}}
    dz_stack = torch.empty(1, ids.shape[1] * model.config.vocab_size, device=DEVICE,
                           dtype=torch.float32)
    timed("v2 arm A_all, 1 seq x 2 probes",
          lambda: arms_sequence(model, row.to(DEVICE), 0, plan, contexts, {"A_all": 0}, k2, n_global,
                                scope, temperature, dz_stack, res))
    del dz_stack
    for label, observation in (("reference", m1), ("candidate", res["A_all"])):
        for field in ("kl", "q", "s_real"):
            if not bool(torch.isfinite(observation[field]).all()):
                raise SystemExit(f"nonfinite {label} arm {field} observation in crosscheck")
    kl_check = {"kl_v1": float(m1["kl"][0]), "kl_v2": float(res["A_all"]["kl"][0]),
                "q_v1": float(m1["q"][0]), "q_v2": float(res["A_all"]["q"][0]),
                "s_real_v1": m1["s_real"][:, 0].tolist(),
                "s_real_v2": res["A_all"]["s_real"][:, 0].tolist()}
    if abs(kl_check["kl_v1"]) <= 0:
        raise SystemExit("invalid reference KL bound in crosscheck")
    kl_check["kl_rel_diff"] = abs(kl_check["kl_v1"] - kl_check["kl_v2"]) / abs(kl_check["kl_v1"])
    if not math.isfinite(kl_check["kl_rel_diff"]):
        raise SystemExit("nonfinite KL error in crosscheck")
    report["arm_crosscheck"] = kl_check
    log(f"arm cross-check v1 vs v2: {json.dumps(kl_check)}")
    with open(args.output + ".profile.txt", "w") as handle:
        handle.write("\n\n".join(tables))
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    bad = [spec for spec, d in diffs.items() if d["max_rel_to_rms"] is None or d["max_rel_to_rms"] > 1e-3]
    if bad or kl_check["kl_rel_diff"] > 1e-6:
        raise SystemExit(f"v2 disagrees with v1: components {bad}, kl_rel {kl_check['kl_rel_diff']:.3g}")
    return report


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--n-probes", type=int, default=64)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--n-single", type=int, default=16)
    ap.add_argument("--layer-arms", action="store_true")
    ap.add_argument("--dz-dtype", default="float32", choices=["float32", "bfloat16"],
                    help="storage of the probe arms' logit changes for the s_real GEMV")
    ap.add_argument("--pricing-from", default=None,
                    help="unsupported cached-pricing reuse; refuses before model loading")
    ap.add_argument("--profile", action="store_true",
                    help="v1 vs v2 on sequence 0 under torch.profiler, with cross-checks")
    ap.add_argument("--deterministic-backward", action="store_true",
                    help="research only: strict deterministic algorithms scoped to pricing backward")
    ap.add_argument("--smoke-first", action="store_true",
                    help="run a 2-sequence, 2-probe pass end to end before the real one")
    ap.add_argument("--output", required=True, help="stem; writes STEM.json and STEM.pt")
    args = ap.parse_args()
    refuse_unbound_pricing_reuse(args.pricing_from)

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForCausalLM
    log(f"loading {args.model} as {args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 attn_implementation="sdpa").to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.get_input_embeddings().register_forward_hook(
        lambda module, inputs, output: output.requires_grad_(True))
    if args.smoke_first:
        smoke = argparse.Namespace(**{**vars(args), "n_seqs": 2, "n_probes": 2, "n_single": 1,
                                      "layer_arms": False, "pricing_from": None, "profile": False,
                                      "output": args.output + ".smoke"})
        run(smoke, model)
        log("smoke pass ok")
    run(args, model)


def run(args, model, *, ids=None, progress_callback=None):
    refuse_unbound_pricing_reuse(args.pricing_from)
    scope, temperature = "all", 1.0
    if ids is None:
        ids = load_tokens(args.inputs, args.text, args.n_seqs)
    n_seqs, seqlen = ids.shape
    n_global = n_seqs * seqlen
    units = {name: module for name, module in model.named_modules()
             if isinstance(module, nn.Linear) and name.startswith("model.layers.")}
    names = list(units)
    log(f"{len(units)} decoder Linears; text={args.text} {n_seqs}x{seqlen} N={n_global} "
        f"probes={args.n_probes} impl={IMPL}")

    spec_a8 = fr.get_format(A8)
    specs_obj = {A8: spec_a8, A8N: nopos0_spec(spec_a8), W4: fr.get_format(W4)}
    assert specs_obj[A8].act_quant_changes_input and specs_obj[A8N].act_quant_changes_input
    assert not specs_obj[W4].act_quant_changes_input

    seeds = [args.seed_base + k for k in range(args.n_probes)]
    ids_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    identity = {"impl": IMPL, "model": args.model, "dtype": args.dtype, "text": args.text,
                "ids_sha256": ids_sha, "seeds": seeds, "units": names}
    deterministic = bool(getattr(args, "deterministic_backward", False))
    if deterministic:
        identity["backward_policy"] = "torch_strict_deterministic_scoped"
    t0 = time.time()
    profile_report = None
    if args.profile:
        profile_report = profile_and_crosscheck(args, model, units, names, ids, seeds, specs_obj,
                                                n_global, scope, temperature)
    if not args.pricing_from:
        part = load_partial(args.output + ".pricing.partial.pt", identity)
        comps = (part["comps"] if part else
                 torch.zeros(len(SPECS), len(names), len(seeds), n_seqs, dtype=torch.float64))
        dens = (part["dens"] if part else
                torch.zeros(len(SPECS), len(names), seqlen, dtype=torch.float64))
        start = part["done"] if part else 0
        if progress_callback is not None:
            progress_callback("pricing", start)
        for i in range(start, n_seqs):
            price_sequence(model, units, names, ids[i:i + 1].to(DEVICE), i, seeds, specs_obj,
                           n_global, scope, temperature, comps, dens,
                           deterministic_backward=deterministic)
            if (i + 1) % 8 == 0 or i == n_seqs - 1:
                atomic_torch_save({"identity": identity, "done": i + 1, "comps": comps,
                                   "dens": dens}, args.output + ".pricing.partial.pt")
                if progress_callback is not None:
                    progress_callback("pricing", i + 1)
                log(f"priced row {i + 1}/{n_seqs} ({time.time() - t0:.0f}s)")
        atomic_torch_save({"comps": comps, "dens": dens, "units": names, "seeds": seeds,
                           "ids_sha256": ids_sha, "specs": list(SPECS), "n_global": n_global,
                           "impl": IMPL, "args": vars(args)}, args.output + ".pricing.pt")
        log(f"wrote {args.output}.pricing.pt ({time.time() - t0:.0f}s)")

    si = {spec: SPECS.index(spec) for spec in SPECS}
    ui = {name: j for j, name in enumerate(names)}
    # per-unit price over the whole draw: 0.5 mean_p (sum_seq c)^2
    unit_price = {spec: 0.5 * comps[si[spec]].sum(-1).pow(2).mean(-1) for spec in SPECS}

    def s_lin(group, spec):                    # [probe, seq]
        return comps[si[spec], [ui[n] for n in group]].sum(0)

    plan = build_plan(names, unit_price, ui, args.n_single, args.layer_arms)
    contexts = make_contexts(plan, units, specs_obj)
    probe_labels = [label for label, _, _ in plan if not label.split(":")[0].endswith("_unit")]
    probe_arms = {label: j for j, label in enumerate(probe_labels)}
    plan_labels = [label for label, _, _ in plan]
    arm_identity = {**identity, "plan": plan_labels}
    part = load_partial(args.output + ".arms.partial.pt", arm_identity)
    if part:
        res, start = part["res"], part["done"]
    else:
        res = {label: {"kl": torch.zeros(n_seqs, dtype=torch.float64),
                       "q": torch.zeros(n_seqs, dtype=torch.float64),
                       "kl_pos": torch.zeros(seqlen, dtype=torch.float64),
                       "s_real": torch.zeros(len(seeds), n_seqs, dtype=torch.float64)}
               for label in plan_labels}
        start = 0
    dz_stack = torch.empty(len(probe_arms), seqlen * model.config.vocab_size, device=DEVICE,
                           dtype=getattr(torch, args.dz_dtype))
    t1 = time.time()
    if progress_callback is not None:
        progress_callback("arms", start)
    for i in range(start, n_seqs):
        arms_sequence(model, ids[i:i + 1].to(DEVICE), i, plan, contexts, probe_arms, seeds, n_global,
                      scope, temperature, dz_stack, res)
        if (i + 1) % 4 == 0 or i == n_seqs - 1:
            atomic_torch_save({"identity": arm_identity, "done": i + 1, "res": res},
                              args.output + ".arms.partial.pt")
            if progress_callback is not None:
                progress_callback("arms", i + 1)
            log(f"arms row {i + 1}/{n_seqs} ({time.time() - t1:.0f}s)")
    del dz_stack
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    arms, tensors = {}, {"comps": comps.float()}
    if dens is not None:
        tensors["dens_unit_pos"] = (0.5 * dens / len(seeds)).float()
    for label, group, parts in plan:
        m = res[label]
        with_probes = label in probe_arms
        lin = sum(s_lin(group, spec) for spec in parts)                  # [probe, seq]
        lin_tot, real_tot = lin.sum(-1), m["s_real"].sum(-1)              # [probe]
        p_add = sum(float(unit_price[spec][[ui[n] for n in group]].sum()) for spec in parts)
        p_joint = float(0.5 * lin_tot.pow(2).mean())
        s_price = float(0.5 * real_tot.pow(2).mean()) if with_probes else float("nan")
        corr = (float(torch.corrcoef(torch.stack((lin_tot, real_tot)))[0, 1])
                if with_probes else float("nan"))
        slope = (float((lin_tot * real_tot).sum() / real_tot.pow(2).sum())
                 if with_probes else float("nan"))
        rec = {"n_units": len(group), "parts": list(parts), "P_add": p_add, "P_joint": p_joint,
               "S_real": s_price, "Q_real": float(m["q"].sum()), "KL_true": float(m["kl"].sum()),
               "corr_slin_sreal": corr, "slope_slin_on_sreal": slope,
               "P_joint_seq": (0.5 * lin.pow(2).mean(0)).tolist(),
               "S_real_seq": (0.5 * m["s_real"].pow(2).mean(0)).tolist(),
               "Q_seq": m["q"].tolist(), "KL_seq": m["kl"].tolist()}
        arms[label] = rec
        tensors[f"kl_pos:{label}"] = (m["kl_pos"] / n_seqs).float()
        if with_probes:
            tensors[f"s_real:{label}"] = m["s_real"].float()
        log(f"{label}: P_add={p_add:.4g} P_joint={p_joint:.4g} S_real={s_price:.4g} "
            f"Q={rec['Q_real']:.4g} KL={rec['KL_true']:.4g} corr={corr:.3f} slope={slope:.3f}")
    out = {"args": vars(args), "impl": IMPL, "n_global": n_global, "units": names, "seeds": seeds,
           "unit_price": {spec: unit_price[spec].tolist() for spec in SPECS},
           "arms": arms, "elapsed_s": time.time() - t0, "complete": True,
           "pass2_s": time.time() - t1, "profile": profile_report,
           "token_scope": scope, "temperature": temperature, "ids_sha256": ids_sha}
    atomic_torch_save(tensors, args.output + ".pt")
    atomic_json_dump(out, args.output + ".json")
    log(f"wrote {args.output}.json/.pt (complete, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
