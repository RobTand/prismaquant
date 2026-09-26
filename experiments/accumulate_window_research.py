"""PQ #1406: non-bitwise joint-statistics accumulates over a FULL real window.

Research instrument. It runs ``tessera_joint_aura run`` on a copy of an M5 run
plan, with its own ``output_root``. ``joint_statistics_replay.observe_and_project_windows``
is replaced by a loop that prices ONE statistics window (``--window``) over
all calibration sequences of probe 0, once per arm, projecting every priced
render exactly as the production loop does:

* ``A``: the checkout's arithmetic, ``acc.add_(g2.T @ x2)`` per invocation.
* ``C``: ``acc.addmm_(g2.T, x2)`` per invocation, the fused epilogue.
* ``D<S>``: each Linear's rows (input, FP32 rows, FP32 gradient rows) buffered
  over S sequences, then one ``_observe_rows`` on the concatenation. The
  activation QDQ is per-token, so it is row-local and the concatenation
  quantizes every row as the per-invocation path does. The operator GEMM
  then runs with K = rows of S sequences.

Per arm it reports the backward's wall time over the window, the priced
signed terms of every (unit, rung) in the window against arm A, and for a
fixed sample of statistics keys the max abs/rel drift of the matrices against
arm A. Arm A's signed totals are also compared with the published M5 payload
(``--reference-cost``, probe 0), which checks the harness reproduces the
production pass. Drift in a priced term is set against that term's
across-probe spread in the reference payload, the scale that decides whether
it matters. GPU power is sampled by ``nvidia-smi`` next to epoch-stamped
phases, so the Netdata series can be read afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

STATE = {"phases": [], "arms": {}}
BUFFER = {"arm": None, "size": None, "rows": {}}


def _phase(label):
    STATE["phases"].append({"label": label, "epoch": time.time()})
    print(f"[acc-research] {label} at {time.time():.3f}", flush=True)


def install(args):
    from prismaquant import glm_mtp, glm_mtp_quantum, joint_aura, joint_statistics_replay as jsr
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    from prismaquant.joint_served_activation import joint_activation_maxima
    from prismaquant.perturbed_x_cache import _activation_qdq

    lease_cls = joint_aura.JointOperatorStatisticsLease
    original_rows = lease_cls._observe_rows

    def fused_rows(self, name, x, x2, g2, *, calls):
        # ``_observe_rows`` with each ``_accumulate(key, a @ b)`` fused into
        # ``acc.addmm_(a, b)``; every other step is the checkout's.
        def accumulate(key, a, b):
            if key in self._operators:
                self._operators[key].addmm_(a, b)
            else:
                self._accumulate(key, a @ b)
                return
            self.telemetry['peak_statistics_bytes'] = max(
                self.telemetry['peak_statistics_bytes'], self.resident_statistics_bytes)

        self._observed_tokens[name] += int(x2.shape[0])
        self._observed_calls[name] += calls
        accumulate((name, None), g2.T, x2)
        self.telemetry['operator_gemms'] += 1
        for index, (spec, _) in enumerate(self.groups[name]):
            if not spec.act_quant_changes_input:
                continue
            quantized = _activation_qdq(x, spec, self.activation_max_abs, name)
            if (not isinstance(quantized, torch.Tensor) or quantized.shape != x.shape
                    or quantized.device != x.device or quantized.dtype != x.dtype):
                raise RuntimeError(f"joint statistics QDQ changed residency/dtype/shape for {name}")
            dx = quantized.reshape_as(x2).float() - x2
            accumulate((name, index), g2.T, dx)
            self.telemetry['qdq_calls'] += 1
            self.telemetry['operator_gemms'] += 1

    def buffered_rows(self, name, x, x2, g2, *, calls):
        rows = BUFFER["rows"].setdefault((id(self), name), [self, [], [], [], 0])
        rows[1].append(x.reshape(-1, x.shape[-1]))
        rows[2].append(x2)
        rows[3].append(g2)
        rows[4] += calls

    def flush():
        pending, BUFFER["rows"] = BUFFER["rows"], {}
        for (_, name), (lease, xs, x2s, g2s, calls) in pending.items():
            original_rows(lease, name, torch.cat(xs), torch.cat(x2s), torch.cat(g2s), calls=calls)

    def one_window_backward(model, embed_tokens, lm_head, calibration_ids, hidden_states, *,
                            seed, device, min_free_gib=None, free_gib=None, report=None):
        n = int(calibration_ids.shape[0])
        dtype = model.layer.eh_proj.weight.dtype
        size = BUFFER["size"]
        profiled = max(8, size or 0)
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index in range(n):
            if index == profiled:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                row = STATE["arms"][BUFFER["arm"]]
                row["profiled_sequences"] = profiled
                row["profiled_s"] = time.perf_counter() - started
                averages = profiler.key_averages()
                kernels = sorted((e for e in averages if e.device_type == torch.autograd.DeviceType.CUDA),
                                 key=lambda e: -e.self_device_time_total)
                total = sum(e.self_device_time_total for e in kernels)
                row["kernel_ms_per_sequence"] = total / 1e3 / profiled
                row["top_kernels"] = [{"name": e.key[:110], "count": e.count,
                                       "ms_per_sequence": e.self_device_time_total / 1e3 / profiled,
                                       "share": e.self_device_time_total / total if total else 0.0}
                                      for e in kernels[:12]]
                del averages, kernels, profiler
                started = time.perf_counter()
            ids = calibration_ids[index:index + 1].to(device)
            hidden = hidden_states(index).to(device=device, dtype=dtype).detach().requires_grad_(True)
            logits = glm_mtp.mtp_logits(model.layer, embed_tokens, lm_head, ids, hidden)
            scalar = glm_mtp.mtp_probe_scalar(logits, seed=int(seed), global_row_offset=index,
                                              n_sequences=n)
            scalar.backward()
            del ids, hidden, logits, scalar
            if size and (index + 1) % size == 0:
                flush()
            if (index + 1) % 64 == 0:
                print(f"[acc-research] arm {BUFFER['arm']} {index + 1}/{n} "
                      f"{time.perf_counter() - started:.0f} s", flush=True)
        if size:
            flush()
        torch.cuda.synchronize()
        row = STATE["arms"][BUFFER["arm"]]
        row["unprofiled_backward_s"] = time.perf_counter() - started
        row["unprofiled_sequences"] = n - row["profiled_sequences"]
        row["s_per_sequence"] = row["unprofiled_backward_s"] / row["unprofiled_sequences"]

    glm_mtp_quantum.mtp_backward = one_window_backward

    def research_windows(modules, specs, cache, policy, *, backward, record_operator,
                         collect_col_energy, backend, guard=None, source_fingerprints=None):
        del collect_col_energy, source_fingerprints
        plan = plan_joint_statistics_target_windows(
            modules, specs, max_statistics_bytes=policy['max_statistics_bytes'],
            activation_max_abs=joint_activation_maxima(cache), projection_backend=backend)
        names = plan.windows[args.window]
        STATE["window"] = {"index": args.window, "units": len(names),
                           "windows": [len(w) for w in plan.windows]}
        sample = sorted({(name, None) for name in names[::max(1, len(names) // 48)]}
                        | {(name, None) for name in names if ".experts." not in name},
                        key=repr)
        reference_stats, reference_terms = {}, None
        arms = ["A", "C", *[f"D{s}" for s in args.buffer_sizes]]
        for arm in arms:
            BUFFER.update(arm=arm, size=int(arm[1:]) if arm.startswith("D") else None, rows={})
            lease_cls._observe_rows = {"A": original_rows, "C": fused_rows}.get(
                arm, buffered_rows)
            STATE["arms"][arm] = {}
            _phase(f"arm {arm} backward start")
            with lease_cls({name: modules[name] for name in names},
                           {name: specs[name] for name in names},
                           max_statistics_bytes=policy['max_statistics_bytes'],
                           max_candidate_bytes=policy['max_candidate_bytes'],
                           activation_max_abs=joint_activation_maxima(cache),
                           projection_backend=backend) as lease:
                lease.begin_probe()
                backward(final=True, lease=lease)
                lease_cls._observe_rows = original_rows
                _phase(f"arm {arm} backward end")
                lease.finish_observations()
                drift = {}
                for key in sample:
                    value = lease._operators.get(key)
                    if value is None:
                        continue
                    if arm == "A":
                        reference_stats[key] = value.detach().clone()
                        continue
                    ref = reference_stats[key]
                    diff = (value.double() - ref.double()).abs()
                    scale = ref.double().abs().max().item()
                    drift[repr(key)] = {"max_abs": diff.max().item(),
                                        "max_rel_to_max": diff.max().item() / scale if scale else 0.0,
                                        "mismatched_fraction": (value != ref).float().mean().item()}
                STATE["arms"][arm]["statistics_drift_sample"] = drift
                _phase(f"arm {arm} projection start")
                keys = [(name, fmt) for name in names for fmt in specs[name]]
                with jsr.resident_candidates(cache, keys, policy, guard=guard) as windows:
                    for quantum, _receipt in windows:
                        for name, fmt in quantum:
                            source = modules[name].weight.detach()
                            rendered = cache.get_resident(name, fmt)
                            record_operator(name, fmt, source, rendered)
                            delta = rendered.to(device=source.device, dtype=torch.float32, copy=True)
                            delta.sub_(source)
                            lease.project({(name, fmt): delta})
                            rendered = source = delta = None
                terms = lease.finish_projections()
                _phase(f"arm {arm} projection end")
            if arm == "A":
                reference_terms = terms
            else:
                STATE["arms"][arm]["terms_drift"] = _terms_drift(reference_terms, terms)
            STATE["arms"][arm]["terms_sha256"] = hashlib.sha256(json.dumps(
                sorted((repr(k), {f: float(v).hex() for f, v in t.items()}) for k, t in terms.items())
            ).encode()).hexdigest()
            if arm == "A":
                STATE["reference_check"] = _against_payload(args.reference_cost, terms)
                STATE["probe_spread"] = _probe_spread(args.reference_cost, terms)
            STATE["_terms"] = STATE.get("_terms", {})
            STATE["_terms"][arm] = {repr(k): t["total"] for k, t in terms.items()}
            _flush_report(args.out)
        for arm in arms[1:]:
            STATE["arms"][arm]["total_drift_vs_probe_spread"] = _against_spread(
                STATE["_terms"]["A"], STATE["_terms"][arm], STATE["probe_spread"])
        STATE["finished_epoch"] = time.time()
        _flush_report(args.out)
        sampler = STATE.get("_sampler")
        if sampler is not None:
            sampler.terminate()
        sys.stdout.flush()
        os._exit(0)

    jsr.observe_and_project_windows = research_windows


def _terms_drift(reference, terms):
    rows = {}
    for field in ("weight", "activation", "mixed", "total"):
        diffs = [abs(terms[k][field] - reference[k][field]) for k in reference]
        rels = [abs(terms[k][field] - reference[k][field]) / abs(reference[k][field])
                for k in reference if reference[k][field] != 0.0]
        rows[field] = {"max_abs": max(diffs), "max_rel": max(rels) if rels else 0.0,
                       "median_rel": statistics.median(rels) if rels else 0.0,
                       "identical": sum(terms[k][field] == reference[k][field] for k in reference),
                       "count": len(reference)}
    return rows


def _load_payload(path):
    return pickle.loads(Path(path).read_bytes())


def _against_payload(path, terms):
    payload = _load_payload(path)
    same, worst = 0, 0.0
    for (name, fmt), row in terms.items():
        published = payload["costs"][name][fmt]["signed_components_per_probe"][0]["total"]
        same += published == row["total"]
        if published != 0.0:
            worst = max(worst, abs(published - row["total"]) / abs(published))
    return {"compared": len(terms), "bitwise_equal": same, "max_rel": worst}


def _probe_spread(path, terms):
    payload = _load_payload(path)
    spread = {}
    for name, fmt in terms:
        values = payload["costs"][name][fmt]["signed_per_probe"]
        spread[repr((name, fmt))] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return spread


def _against_spread(reference, totals, spread):
    ratios = [abs(totals[k] - reference[k]) / spread[k] for k in reference if spread[k] > 0]
    return {"max": max(ratios), "median": statistics.median(ratios),
            "p99": sorted(ratios)[int(0.99 * (len(ratios) - 1))], "count": len(ratios)}


def _flush_report(out):
    public = {k: v for k, v in STATE.items() if not k.startswith("_")}
    (Path(out) / "report.json").write_text(json.dumps(public, indent=1, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--reference-cost", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--buffer-sizes", default="8,32")
    args = parser.parse_args()
    args.buffer_sizes = [int(s) for s in args.buffer_sizes.split(",") if s]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    source_root = Path(plan["output_root"])
    if source_root.resolve() == out.resolve():
        raise RuntimeError("the research row must not write into the run's output root")
    plan["output_root"] = str(out)
    (out / "prepare").mkdir(exist_ok=True)
    shutil.copyfile(source_root / "prepare/source-identity.json", out / "prepare/source-identity.json")
    plan_bytes = json.dumps(plan, indent=1, sort_keys=True).encode()
    plan_path = out / "plan-research.json"
    plan_path.write_bytes(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    STATE.update(plan={"source": str(args.plan), "research": str(plan_path), "sha256": plan_sha},
                 host=os.uname().nodename, reference_cost=str(args.reference_cost))
    STATE["_sampler"] = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm",
         "--format=csv,noheader", "-lms", "500"],
        stdout=open(out / "power.csv", "w"), stderr=subprocess.DEVNULL)
    _phase("start")
    install(args)
    from prismaquant import tessera_joint_aura
    sys.argv = ["tessera_joint_aura", "run", "--plan", str(plan_path), "--plan-sha256", plan_sha,
                "--prepared", str(args.prepared), "--prepared-sha256", args.prepared_sha256]
    try:
        tessera_joint_aura.main()
    finally:
        STATE["_sampler"].terminate()
    raise RuntimeError("the research row returned without reaching the statistics windows")


if __name__ == "__main__":
    main()
