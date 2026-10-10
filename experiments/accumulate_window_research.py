"""PQ #1406 research: fused and row-buffered joint-statistics accumulate.

Research instrument, not production. It runs ``tessera_joint_aura run`` on a
copy of an M5 run plan with its own ``output_root``. The replacement
``joint_statistics_replay.observe_and_project_windows`` prices ONE statistics
window (``--window``) over the calibration sequences of probe 0, once per
arm, and projects every priced render as the production loop does.

Arms:

* ``A``: the checkout arithmetic, ``acc.add_(g2.T @ x2)`` per invocation.
* ``C``: ``acc.addmm_(g2.T, x2)`` per invocation (fused beta=1 epilogue).
* ``D<S>``: each Linear's rows buffer over S sequences, then one fused
  ``addmm_`` with K = S sequences of rows. The activation QDQ is row-local
  (checked by ``require_row_local_activation_qdq`` before the D arms run),
  so the concatenation quantizes every row as the per-invocation path does.

Per arm the report holds backward wall time per sequence, CUDA kernel time
per sequence with a kernel-family breakdown, nvidia-smi power against the
140 W envelope, max abs/rel drift of a fixed statistics-matrix sample
against arm A, and drift of the priced signed terms and of the priced
``predicted_dloss`` against arm A. Term drift is set against the reference
payload's across-probe spread, and ``predicted_dloss`` drift against its
``predicted_dloss_stderr``. That scale decides whether the drift matters.

Arm A's signed totals are also checked against the published M5 payload
(``--reference-cost``, probe 0). That check proves the harness reproduces
the production pass. No arm is adopted here; adoption needs a coordinator
re-baseline. The row buffer refuses past ``--max-buffer-bytes`` (default:
the statistics window budget), so an S choice cannot grow past the budget
the window was planned under.
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
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

STATE = {"phases": [], "arms": {}}

ENVELOPE_W = 140.0
WARMUP_SEQUENCES = 64


def _commit_progress(phase: str, units: int = 1) -> None:
    """Report committed units to PrismaBuild's stall watchdog (never raises)."""
    try:
        from prismaquant.prismabuild_progress import report
        report(phase, units)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable; no lease needed except where noted).
# ---------------------------------------------------------------------------


def kernel_family(name: str) -> str:
    """Map a profiler kernel name to one coarse family."""
    text = name.lower()
    if "vectorized_elementwise" in text or "elementwise_kernel" in text:
        return "elementwise"
    if "simt_sgemm" in text or "simt_" in text:
        return "simt_gemm"
    if "mma" in text or "cutlass" in text or "gemm" in text or "matmul" in text:
        return "tensor_gemm"
    if "reduce" in text:
        return "reduction"
    if "memcpy" in text or "memset" in text or "copy" in text:
        return "copy"
    if "norm" in text or "softmax" in text or "activation" in text:
        return "norm_activation"
    return "other"


def matrix_drift(ref: torch.Tensor, value: torch.Tensor) -> dict:
    """Max abs/rel drift of ``value`` against ``ref`` (both FP32)."""
    diff = (value.double() - ref.double()).abs()
    scale = ref.double().abs().max().item()
    peak = diff.max().item()
    return {"max_abs": peak,
            "max_rel_to_max": peak / scale if scale else 0.0,
            "mismatched_fraction": float((value != ref).float().mean().item())}


def accumulate_fused(lease, key, a: torch.Tensor, b: torch.Tensor) -> None:
    """Fuse one ``acc.add_(a @ b)`` into ``acc.addmm_(a, b)`` on ``lease``.

    The first touch of a key keeps the checkout's ``a @ b`` insert. Later
    touches run beta=1 in the GEMM epilogue instead of a separate add pass.
    """
    if key in lease._operators:
        lease._operators[key].addmm_(a, b)
    else:
        lease._accumulate(key, a @ b)
        return
    lease.telemetry["peak_statistics_bytes"] = max(
        lease.telemetry["peak_statistics_bytes"], lease.resident_statistics_bytes)


def fused_observe_rows(lease, name, x, x2, g2, *, calls) -> None:
    """``_observe_rows`` with every ``_accumulate(key, a @ b)`` fused."""
    from prismaquant.perturbed_x_cache import _activation_qdq

    lease._observed_tokens[name] += int(x2.shape[0])
    lease._observed_calls[name] += calls
    accumulate_fused(lease, (name, None), g2.T, x2)
    lease.telemetry["operator_gemms"] += 1
    for index, (spec, _) in enumerate(lease.groups[name]):
        if not spec.act_quant_changes_input:
            continue
        quantized = _activation_qdq(x, spec, lease.activation_max_abs, name)
        if (not isinstance(quantized, torch.Tensor) or quantized.shape != x.shape
                or quantized.device != x.device or quantized.dtype != x.dtype):
            raise RuntimeError(f"joint statistics QDQ changed residency/dtype/shape for {name}")
        dx = quantized.reshape_as(x2).float() - x2
        accumulate_fused(lease, (name, index), g2.T, dx)
        lease.telemetry["qdq_calls"] += 1
        lease.telemetry["operator_gemms"] += 1


class RowBuffer:
    """Buffer one Linear's rows across S sequences, then one fused GEMM.

    ``append`` keeps the caller's row tensors alive (no copy). ``flush``
    concatenates each key's rows and runs one ``observe`` call with the
    summed call count, so the GEMM sees K = S sequences of rows. ``cap``
    bounds the buffered bytes; ``append`` refuses past it.
    """

    def __init__(self, size: int, cap_bytes: int):
        if type(size) is not int or size < 1:
            raise ValueError("row buffer size must be a positive integer")
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError("row buffer cap must be a nonnegative integer")
        self.size = size
        self.cap_bytes = cap_bytes
        self.pending: dict = {}
        self.peak_bytes = 0
        self.live_bytes = 0

    def append(self, name, lease, x, x2, g2, *, calls) -> None:
        held = x.numel() * x.element_size() + x2.numel() * x2.element_size()
        held += g2.numel() * g2.element_size()
        if self.live_bytes + held > self.cap_bytes:
            raise RuntimeError(
                f"row buffer for S={self.size} crossed its {self.cap_bytes} B cap "
                f"at {name}: bound S by the statistics window budget")
        row = self.pending.setdefault((id(lease), name), [lease, [], [], [], 0])
        row[1].append(x.reshape(-1, x.shape[-1]))
        row[2].append(x2)
        row[3].append(g2)
        row[4] += calls
        self.live_bytes += held
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)

    def flush(self, observe) -> None:
        pending, self.pending = self.pending, {}
        self.live_bytes = 0
        for (_, name), (lease, xs, x2s, g2s, calls) in pending.items():
            observe(lease, name, torch.cat(xs), torch.cat(x2s), torch.cat(g2s),
                    calls=calls)


CURRENT_BUFFER: RowBuffer | None = None


def buffered_observe_rows(lease, name, x, x2, g2, *, calls) -> None:
    """``_observe_rows`` deferred to the arm's ``RowBuffer``."""
    assert CURRENT_BUFFER is not None, "row-buffered arm runs without a buffer"
    CURRENT_BUFFER.append(name, lease, x, x2, g2, calls=calls)


def field_drift(reference: dict, terms: dict) -> dict:
    """Per-field drift of priced signed terms against the reference arm."""
    rows = {}
    for field in ("weight", "activation", "mixed", "total"):
        diffs = [abs(terms[k][field] - reference[k][field]) for k in reference]
        rels = [abs(terms[k][field] - reference[k][field]) / abs(reference[k][field])
                for k in reference if reference[k][field] != 0.0]
        rows[field] = {"max_abs": max(diffs), "max_rel": max(rels) if rels else 0.0,
                       "median_rel": statistics.median(rels) if rels else 0.0,
                       "identical": sum(terms[k][field] == reference[k][field]
                                        for k in reference),
                       "count": len(reference)}
    return rows


def priced_dloss(signed_total: float) -> float:
    """One probe's priced ``predicted_dloss``: 0.5 times the squared total."""
    return 0.5 * signed_total * signed_total


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------

def _phase(label: str) -> None:
    STATE["phases"].append({"label": label, "epoch": time.time()})
    print(f"[acc-research] {label} at {time.time():.3f}", flush=True)


def _sync() -> None:
    torch.cuda.synchronize()


def _cuda_mem() -> dict:
    return {"allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
            "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30)}


def _power_rows(out: Path) -> list:
    """Parse the nvidia-smi sampler CSV into (epoch, watts) pairs."""
    rows = []
    try:
        lines = (out / "power.csv").read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            watts = float(parts[1].split()[0])
        except ValueError:
            continue
        rows.append((stamp.timestamp(), watts))
    return rows


def _arm_power(out: Path, start: float, end: float) -> dict:
    rows = [(epoch, watts) for epoch, watts in _power_rows(out)
            if start <= epoch <= end]
    if not rows:
        return {"samples": 0}
    watts = [watts for _, watts in rows]
    return {"samples": len(rows), "mean_w": sum(watts) / len(watts),
            "max_w": max(watts), "envelope_w": ENVELOPE_W,
            "mean_fraction_of_envelope": (sum(watts) / len(watts)) / ENVELOPE_W,
            "max_fraction_of_envelope": max(watts) / ENVELOPE_W}


def _netdata_power(start: float, end: float) -> dict:
    """Best-effort Netdata power read over one arm window (guarded)."""
    try:
        charts = json.loads(subprocess.run(
            ["curl", "-s", "--max-time", "10", "http://localhost:19999/api/v1/charts"],
            capture_output=True, check=True, timeout=15).stdout)
        names = [name for name, chart in charts.get("charts", {}).items()
                 if "power" in name.lower() or "watt" in name.lower()]
        return {"attempted": True, "power_charts": names[:8]}
    except Exception as error:  # Netdata is corroboration, never the gate
        return {"attempted": True, "unavailable": repr(error)}


def _summarize_profile(prof, sequences: int) -> dict:
    averages = prof.key_averages()
    kernels = [e for e in averages
               if e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.self_device_time_total for e in kernels)
    families: dict = {}
    for e in kernels:
        family = kernel_family(e.key)
        row = families.setdefault(family, {"count": 0, "total_ms": 0.0})
        row["count"] += e.count
        row["total_ms"] += e.self_device_time_total / 1e3
    for row in families.values():
        row["ms_per_sequence"] = row["total_ms"] / sequences
        row["share"] = row["total_ms"] / (total / 1e3) if total else 0.0
    top = sorted(kernels, key=lambda e: -e.self_device_time_total)[:12]
    summary = {
        "profiled_sequences": sequences,
        "kernel_ms_per_sequence": total / 1e3 / sequences,
        "families": dict(sorted(families.items(), key=lambda kv: -kv[1]["total_ms"])),
        "top_kernels": [{"name": e.key[:110], "count": e.count,
                         "ms_per_sequence": e.self_device_time_total / 1e3 / sequences,
                         "share": e.self_device_time_total / total if total else 0.0}
                        for e in top],
    }
    del averages, kernels, top
    return summary


def install(args) -> None:
    from prismaquant import glm_mtp, glm_mtp_quantum, joint_aura, joint_statistics_replay as jsr
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    from prismaquant.joint_served_activation import joint_activation_maxima

    lease_cls = joint_aura.JointOperatorStatisticsLease
    original_rows = lease_cls._observe_rows
    out = Path(args.out)

    def one_window_backward(model, embed_tokens, lm_head, calibration_ids, hidden_states,
                            *, seed, device, min_free_gib=None, free_gib=None,
                            report=None):
        del min_free_gib, free_gib
        n = int(calibration_ids.shape[0])
        if args.max_sequences is not None:
            n = min(n, args.max_sequences)
        dtype = model.layer.eh_proj.weight.dtype
        size = BUFFER_SIZE["size"]
        profiled = max(8, size or 0)
        arm = BUFFER_SIZE["arm"]
        row = STATE["arms"][arm]
        started_all = time.perf_counter()
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        in_profile = False
        timed_s, timed_n, profiled_wall = 0.0, 0, 0.0
        for index in range(n):
            if index == WARMUP_SEQUENCES:
                _sync()
                profiler.__enter__()
                in_profile = True
                tick = time.perf_counter()
            ids = calibration_ids[index:index + 1].to(device)
            hidden = hidden_states(index).to(device=device, dtype=dtype).detach()
            hidden.requires_grad_(True)
            logits = glm_mtp.mtp_logits(model.layer, embed_tokens, lm_head, ids, hidden)
            scalar = glm_mtp.mtp_probe_scalar(logits, seed=int(seed),
                                              global_row_offset=index,
                                              n_sequences=int(calibration_ids.shape[0]))
            scalar.backward()
            del ids, hidden, logits, scalar
            if size and (index + 1) % size == 0:
                assert CURRENT_BUFFER is not None
                CURRENT_BUFFER.flush(original_rows)
            step = time.perf_counter()
            if in_profile and index == WARMUP_SEQUENCES + profiled - 1:
                _sync()
                profiler.__exit__(None, None, None)
                in_profile = False
                profiled_wall = step - tick
                row.update(_summarize_profile(profiler, profiled))
                row["profiled_wall_s"] = profiled_wall
                del profiler
                tick = time.perf_counter()
            elif not in_profile and index >= WARMUP_SEQUENCES:
                timed_s += step - tick
                timed_n += 1
                tick = step
            elif not in_profile:
                tick = step
            if (index + 1) % 64 == 0:
                print(f"[acc-research] arm {arm} {index + 1}/{n} "
                      f"cuda={_cuda_mem()['allocated_gib']:.1f} GiB", flush=True)
                _commit_progress(BUFFER_SIZE["phase"], BUFFER_SIZE["base"] + index + 1)
            if report is not None:
                report(index + 1)
        if in_profile:  # short slices end inside the profiled region
            _sync()
            profiler.__exit__(None, None, None)
            profiled_wall = time.perf_counter() - tick
            row.update(_summarize_profile(profiler, n - WARMUP_SEQUENCES))
            row["profiled_wall_s"] = profiled_wall
            del profiler
        if size:
            assert CURRENT_BUFFER is not None
            CURRENT_BUFFER.flush(original_rows)
        _sync()
        row["timed_s_per_sequence"] = timed_s / timed_n if timed_n else 0.0
        row["timed_sequences"] = timed_n
        row["window_s"] = time.perf_counter() - started_all
        row["window_sequences"] = n
        row["cuda_mem"] = _cuda_mem()

    BUFFER_SIZE = {"arm": None, "size": None}

    def research_windows(modules, specs, cache, policy, *, backward, record_operator,
                         collect_col_energy, backend, guard=None,
                         source_fingerprints=None, attribution=None):
        del collect_col_energy, source_fingerprints, attribution
        global CURRENT_BUFFER
        _record_environment()
        plan = plan_joint_statistics_target_windows(
            modules, specs, max_statistics_bytes=policy["max_statistics_bytes"],
            activation_max_abs=joint_activation_maxima(cache),
            projection_backend=backend)
        names = plan.windows[args.window]
        STATE["window"] = {"index": args.window, "units": len(names),
                           "windows": [len(w) for w in plan.windows]}
        sample = sorted({(name, None) for name in names[::max(1, len(names) // 48)]}
                        | {(name, None) for name in names if ".experts." not in name},
                        key=repr)
        cap = args.max_buffer_bytes or policy["max_statistics_bytes"]
        reference_stats, reference_terms = {}, None
        arms = list(args.arms)
        done = 0
        _commit_progress("startup", 1)
        for arm in arms:
            size = int(arm[1:]) if arm.startswith("D") else None
            BUFFER_SIZE.update(arm=arm, size=size, base=done, phase=f"arm-{arm}")
            torch.cuda.reset_peak_memory_stats()
            if arm == "C":
                lease_cls._observe_rows = fused_observe_rows
            elif arm.startswith("D"):
                if CURRENT_BUFFER is None or CURRENT_BUFFER.size != size:
                    CURRENT_BUFFER = RowBuffer(size, cap)
                else:
                    CURRENT_BUFFER.pending.clear()
                lease_cls._observe_rows = buffered_observe_rows
                if "row_local_qdq" not in STATE:
                    from prismaquant.joint_replay_spill import (
                        require_row_local_activation_qdq)
                    checked = require_row_local_activation_qdq(
                        {name: modules[name] for name in names},
                        {name: specs[name] for name in names},
                        joint_activation_maxima(cache),
                        device=modules[names[0]].weight.device,
                        dtype=modules[names[0]].weight.dtype)
                    STATE.setdefault("row_local_qdq", checked)
            else:
                lease_cls._observe_rows = original_rows
            STATE["arms"][arm] = {}
            start_epoch = time.time()
            _phase(f"arm {arm} backward start")
            with lease_cls({name: modules[name] for name in names},
                           {name: specs[name] for name in names},
                           max_statistics_bytes=policy["max_statistics_bytes"],
                           max_candidate_bytes=policy["max_candidate_bytes"],
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
                    drift[repr(key)] = matrix_drift(reference_stats[key], value)
                STATE["arms"][arm]["statistics_drift_sample"] = drift
                if size:
                    STATE["arms"][arm]["buffer_peak_bytes"] = CURRENT_BUFFER.peak_bytes
                    STATE["arms"][arm]["buffer_cap_bytes"] = cap
                STATE["arms"][arm]["observed_tokens"] = dict(lease._observed_tokens)
                STATE["arms"][arm]["observed_calls"] = dict(lease._observed_calls)
                _phase(f"arm {arm} projection start")
                keys = [(name, fmt) for name in names for fmt in specs[name]]
                with jsr.resident_candidates(cache, keys, policy, guard=guard) as windows:
                    for quantum, _receipt in windows:
                        for name, fmt in quantum:
                            source = modules[name].weight.detach()
                            rendered = cache.get_resident(name, fmt)
                            record_operator(name, fmt, source, rendered)
                            delta = rendered.to(device=source.device,
                                                dtype=torch.float32, copy=True)
                            delta.sub_(source)
                            lease.project({(name, fmt): delta})
                            rendered = source = delta = None
                terms = lease.finish_projections()
                _phase(f"arm {arm} projection end")
            end_epoch = time.time()
            STATE["arms"][arm]["power"] = _arm_power(out, start_epoch, end_epoch)
            STATE["arms"][arm]["netdata"] = _netdata_power(start_epoch, end_epoch)
            if arm == "A":
                reference_terms = terms
            else:
                STATE["arms"][arm]["terms_drift"] = field_drift(reference_terms, terms)
            STATE["arms"][arm]["terms_sha256"] = hashlib.sha256(json.dumps(
                sorted((repr(k), {f: float(v).hex() for f, v in t.items()})
                       for k, t in terms.items())).encode()).hexdigest()
            if arm == "A":
                STATE["reference_check"] = _against_payload(args.reference_cost, terms)
                STATE["probe_spread"] = _probe_spread(args.reference_cost, terms)
            STATE["_terms"] = STATE.get("_terms", {})
            STATE["_terms"][arm] = {repr(k): t["total"] for k, t in terms.items()}
            STATE["_keys"] = {repr(k): list(k) for k in terms}
            STATE["arms"][arm]["term_totals"] = dict(STATE["_terms"][arm])
            done += STATE["arms"][arm]["window_sequences"]
            _commit_progress(f"arm-{arm}", 1 + done)
            _flush_report(out)
        for arm in arms[1:]:
            STATE["arms"][arm]["total_drift_vs_probe_spread"] = _against_spread(
                STATE["_terms"]["A"], STATE["_terms"][arm], STATE["probe_spread"])
            STATE["arms"][arm]["dloss_vs_stderr"] = _dloss_vs_stderr(
                args.reference_cost, STATE["_terms"]["A"], STATE["_terms"][arm],
                STATE["_keys"])
        STATE["finished_epoch"] = time.time()
        _flush_report(out)
        sampler = STATE.get("_sampler")
        if sampler is not None:
            sampler.terminate()
        sys.stdout.flush()
        os._exit(0)

    jsr.observe_and_project_windows = research_windows


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
    ratios = [abs(totals[k] - reference[k]) / spread[k] for k in reference
              if spread[k] > 0]
    ordered = sorted(ratios)
    return {"max": max(ratios), "median": statistics.median(ratios),
            "p99": ordered[int(0.99 * (len(ordered) - 1))], "count": len(ratios)}


def _dloss_vs_stderr(path, reference, totals, keys):
    """Priced ``predicted_dloss`` drift of one arm against sampling stderr."""
    payload = _load_payload(path)
    ratios = {}
    for key in reference:
        name, fmt = keys[key]
        row = payload["costs"][name][fmt]
        stderr = float(row.get("predicted_dloss_stderr") or 0.0)
        if not stderr:
            values = [float(v) for v in row["signed_per_probe"]]
            stderr = (statistics.pstdev(values) / math.sqrt(len(values))
                      if len(values) > 1 else 0.0)
        drift = abs(priced_dloss(totals[key]) - priced_dloss(reference[key]))
        ratios[key] = drift / stderr if stderr else 0.0
    ordered = sorted(ratios.values())
    worst = sorted(ratios.items(), key=lambda kv: -kv[1])[:10]
    return {"max": max(ratios.values()), "median": statistics.median(ratios.values()),
            "p99": ordered[int(0.99 * (len(ordered) - 1))], "count": len(ratios),
            "worst": [[key, ratio] for key, ratio in worst]}


def _flush_report(out: Path) -> None:
    public = {k: v for k, v in STATE.items() if not k.startswith("_")}
    (out / "report.json").write_text(json.dumps(public, indent=1, default=str))


def _record_environment() -> None:
    STATE["environment"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def run_preflight(args) -> int:
    """CPU check: args, plan, prepared, inputs, and the arm kernels."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text())
    prepared = json.loads(Path(args.prepared).read_text())
    calib = Path(plan["calibration_input"]["path"])
    manifest = Path(plan["canonical_capture"]["path"])
    capture = json.loads(manifest.read_text())
    with open(calib, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    tensors = header if isinstance(header, dict) else {}
    first = next(iter(tensors.values())) if tensors else {}
    record = {
        "plan_keys": sorted(plan.keys()),
        "n_calib_samples": plan["execution"]["n_calib_samples"],
        "n_probes": plan["execution"]["n_probes"],
        "prepared_bytes": Path(args.prepared).stat().st_size,
        "calibration_bytes": calib.stat().st_size,
        "calibration_first_tensor": {k: first.get(k) for k in ("dtype", "shape")},
        "capture_keys": sorted(capture.keys())[:8],
        "arms": args.arms,
        "buffer_sizes": args.buffer_sizes,
        "window": args.window,
    }
    generator = torch.Generator().manual_seed(1406)
    ref = torch.randn(16, 8, generator=generator).float()
    acc = ref.clone()
    g = torch.randn(6, 16, generator=generator).to(torch.bfloat16).float()
    x = torch.randn(6, 8, generator=generator).to(torch.bfloat16).float()
    acc.add_(g.T @ x)
    fused = ref.clone()
    fused.addmm_(g.T, x)
    record["synthetic_addmm_drift"] = matrix_drift(acc, fused)
    acc_b = ref.clone()
    acc_b.addmm_(torch.cat([g[:3], g[3:]]).T, torch.cat([x[:3], x[3:]]))
    record["synthetic_buffered_drift"] = matrix_drift(acc, acc_b)
    record["family_probe"] = kernel_family("void cutlass::Kernel2<cutlass_80_simt_sgemm_foo>")
    (out / "preflight.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str), flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--reference-cost", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--buffer-sizes", default="8,32,128")
    parser.add_argument("--arms", default="A,C,D8,D32,D128")
    parser.add_argument("--max-buffer-bytes", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.buffer_sizes = [int(s) for s in args.buffer_sizes.split(",") if s]
    args.arms = [arm for arm in args.arms.split(",") if arm]
    for arm in args.arms:
        if arm not in {"A", "C"} and (
                not arm.startswith("D") or int(arm[1:]) not in args.buffer_sizes):
            parser.error(f"arm {arm} is not one of A, C, or D<S> in --buffer-sizes")
    if args.preflight:
        raise SystemExit(run_preflight(args))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    source_root = Path(plan["output_root"])
    if source_root.resolve() == out.resolve():
        raise RuntimeError("the research row must not write into the run's output root")
    plan["output_root"] = str(out)
    (out / "prepare").mkdir(exist_ok=True)
    shutil.copyfile(source_root / "prepare/source-identity.json",
                    out / "prepare/source-identity.json")
    plan_bytes = json.dumps(plan, indent=1, sort_keys=True).encode()
    plan_path = out / "plan-research.json"
    plan_path.write_bytes(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    STATE.update(plan={"source": str(args.plan), "research": str(plan_path),
                       "sha256": plan_sha},
                 host=os.uname().nodename, reference_cost=str(args.reference_cost),
                 arms_requested=args.arms, buffer_sizes=args.buffer_sizes,
                 max_sequences=args.max_sequences)
    STATE["_sampler"] = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm",
         "--format=csv,noheader", "-lms", "500"],
        stdout=open(out / "power.csv", "w"), stderr=subprocess.DEVNULL)
    _phase("start")
    install(args)
    from prismaquant import tessera_joint_aura
    sys.argv = ["tessera_joint_aura", "run", "--plan", str(plan_path),
                "--plan-sha256", plan_sha, "--prepared", str(args.prepared),
                "--prepared-sha256", args.prepared_sha256]
    try:
        tessera_joint_aura.main()
    finally:
        STATE["_sampler"].terminate()
    raise RuntimeError("the research row returned without reaching the statistics windows")


if __name__ == "__main__":
    main()
