"""Profile the GLM-5.3 MTP statistics backward on the real M5 inputs.

Runs ``tessera_joint_aura run`` against a copy of an M5 run plan whose
``output_root`` is this profile's directory, with two substitutions:

* ``joint_statistics_replay.observe_and_project_windows`` observes each
  statistics window over the first ``--sequences`` calibration sequences and
  does not project; after the last arm it writes ``report.json`` and exits.
* ``glm_mtp_quantum.mtp_backward`` runs those sequences with the same per-sequence
  arithmetic (the probe normalization still uses the full draw's
  ``n_sequences``) and times or profiles them.

Two arms run in one process on the same box and the same sequences:

* ``A``: the checkout's ``JointOperatorStatisticsLease._accumulate``.
* ``B``: an ``_accumulate`` that keeps a running resident-byte count instead of
  summing every held matrix on each call.

The statistics each arm accumulates are hashed per key (SHA-256 of the raw
bytes) so that bit identity between the arms is checked directly, not assumed.

Per window and arm, sequences ``[0, warmup)`` warm up, ``[warmup, warmup+timed)``
are timed with no profiler, and the rest run under ``torch.profiler`` with CUDA
activity. The report ranks GPU idle against host work from the trace: kernel
busy time, launches, median kernel duration, memcpy bytes by kind, syncs and
``.item()`` calls, plus the Python hook time. GPU power is sampled by
``nvidia-smi`` next to epoch-stamped phases, so the matching Netdata series
can be read afterwards.

This is a research instrument (PQ #1271, M5). It never writes to the run's
own output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

STATE = {"arm": None, "window": None, "phases": [], "arms": {}}


def _phase(label):
    STATE["phases"].append({"label": label, "epoch": time.time()})
    print(f"[mtp-profile] {label} at {time.time():.3f}", flush=True)


def _sync():
    torch.cuda.synchronize()


def _counter_accumulate(self, key, matrix):
    """Arm B: identical accumulation; the peak telemetry reads a running count."""
    held = getattr(self, "_profile_resident_bytes", 0)
    if key in self._operators:
        self._operators[key].add_(matrix)
    else:
        self._operators[key] = matrix
        held += matrix.numel() * matrix.element_size()
        self._profile_resident_bytes = held
    self.telemetry['peak_statistics_bytes'] = max(
        self.telemetry['peak_statistics_bytes'], held)


def _trace_metrics(trace_path, n_sequences):
    trace = json.loads(Path(trace_path).read_text())
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    memcpy = [e for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    cpu_ops = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
    region = [e for e in cpu_ops if e.get("name") == "mtp_profile.region"]
    if not region:
        raise RuntimeError("profile trace has no region annotation")
    start = min(e["ts"] for e in region)
    end = max(e["ts"] + e["dur"] for e in region)
    wall_us = end - start
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels + memcpy)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        s, e = max(s, start), min(e, end)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    durations = [e["dur"] for e in kernels]
    by_kernel = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        row = by_kernel[e["name"][:120]]
        row[0] += 1
        row[1] += e["dur"]
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[:15]
    copies = defaultdict(lambda: [0, 0])
    for e in memcpy:
        kind = e["name"].split("(")[0].strip()
        copies[kind][0] += 1
        copies[kind][1] += int(e.get("args", {}).get("bytes", 0) or 0)
    runtime_counts = Counter(e["name"] for e in runtime)
    launches = sum(n for name, n in runtime_counts.items() if "Launch" in name)
    syncs = {name: n for name, n in runtime_counts.items()
             if "Synchronize" in name or name in ("cudaMemcpy", "cuMemcpyDtoH_v2")}
    op_counts = Counter(e["name"] for e in cpu_ops)
    op_time = defaultdict(float)
    for e in cpu_ops:
        op_time[e["name"]] += e["dur"]
    annotations = {name: {"count": op_counts[name], "cpu_us": op_time[name]}
                   for name in op_counts if name.startswith("mtp_profile.")}
    autograd_us = sum(t for name, t in op_time.items()
                      if name.startswith("autograd::engine::evaluate_function"))
    median = statistics.median(durations) if durations else 0.0
    return {
        "sequences": n_sequences,
        "wall_ms": wall_us / 1e3,
        "wall_ms_per_sequence": wall_us / 1e3 / n_sequences,
        "gpu_busy_ms": busy / 1e3,
        "gpu_busy_fraction": busy / wall_us if wall_us else None,
        "gpu_idle_ms_per_sequence": (wall_us - busy) / 1e3 / n_sequences,
        "kernels": len(kernels),
        "kernels_per_sequence": len(kernels) / n_sequences,
        "launch_calls_per_sequence": launches / n_sequences,
        "kernel_us_median": median,
        "kernel_us_p90": (statistics.quantiles(durations, n=10)[-1]
                          if len(durations) >= 10 else None),
        "kernel_us_total": sum(durations),
        "top_kernels": [{"name": name, "count": c, "total_ms": t / 1e3, "mean_us": t / c}
                        for name, (c, t) in top],
        "memcpy": {kind: {"count": c, "bytes": b} for kind, (c, b) in copies.items()},
        "sync_calls": syncs,
        "item_calls": op_counts.get("aten::item", 0),
        "local_scalar_dense_calls": op_counts.get("aten::_local_scalar_dense", 0),
        "nonzero_calls": op_counts.get("aten::nonzero", 0),
        "autograd_evaluate_function_cpu_ms": autograd_us / 1e3,
        "annotations": annotations,
        "top_cpu_ops_by_count": op_counts.most_common(25),
    }


def install(args):
    from prismaquant import glm_mtp, glm_mtp_quantum, joint_aura, joint_statistics_replay as jsr
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    from prismaquant.joint_served_activation import joint_activation_maxima

    out = Path(args.out)
    warmup, timed, profiled = args.warmup, args.timed, args.profiled
    k = warmup + timed + profiled
    original_accumulate = joint_aura.JointOperatorStatisticsLease._accumulate
    original_observe = joint_aura.JointOperatorStatisticsLease._observe_invocation

    def annotated_observe(self, *a, **kw):
        with torch.profiler.record_function("mtp_profile.observe_invocation"):
            return original_observe(self, *a, **kw)

    joint_aura.JointOperatorStatisticsLease._observe_invocation = annotated_observe

    def profiled_backward(model, embed_tokens, lm_head, calibration_ids, hidden_states, *,
                          seed, device, min_free_gib=None, free_gib=None, report=None):
        n = int(calibration_ids.shape[0])
        if n < k:
            raise RuntimeError(f"the draw has {n} sequences, fewer than {k}")
        dtype = model.layer.eh_proj.weight.dtype
        arm, window = STATE["arm"], STATE["window"]
        row = STATE["arms"].setdefault(arm, {}).setdefault(str(window), {})

        def one(index):
            ids = calibration_ids[index:index + 1].to(device)
            hidden = hidden_states(index).to(device=device, dtype=dtype).detach().requires_grad_(True)
            with torch.profiler.record_function("mtp_profile.forward"):
                logits = glm_mtp.mtp_logits(model.layer, embed_tokens, lm_head, ids, hidden)
                scalar = glm_mtp.mtp_probe_scalar(logits, seed=int(seed), global_row_offset=index,
                                                  n_sequences=n)
            with torch.profiler.record_function("mtp_profile.backward"):
                scalar.backward()

        _sync()
        t0 = time.perf_counter()
        for index in range(warmup):
            one(index)
        _sync()
        t1 = time.perf_counter()
        _phase(f"arm {arm} window {window} timed start")
        for index in range(warmup, warmup + timed):
            one(index)
        _sync()
        t2 = time.perf_counter()
        _phase(f"arm {arm} window {window} timed end")
        row["warmup_s_per_sequence"] = (t1 - t0) / max(warmup, 1)
        row["timed_s_per_sequence"] = (t2 - t1) / max(timed, 1)
        trace = out / f"trace-{arm}-w{window}.json"
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        _phase(f"arm {arm} window {window} profiled start")
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    with_stack=False) as prof:
            with torch.profiler.record_function("mtp_profile.region"):
                for index in range(warmup + timed, k):
                    one(index)
                _sync()
        _phase(f"arm {arm} window {window} profiled end")
        prof.export_chrome_trace(str(trace))
        (out / f"ops-{arm}-w{window}.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40) + "\n\n"
            + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
        try:
            row["profiled"] = _trace_metrics(trace, profiled)
        except Exception as error:  # the trace is kept; parse it offline
            row["profiled"] = {"error": repr(error)}
        subprocess.run(["gzip", "-f", str(trace)], check=False)

    glm_mtp_quantum.mtp_backward = profiled_backward

    def profiled_windows(modules, specs, cache, policy, *, backward, record_operator,
                         collect_col_energy, backend, guard=None, source_fingerprints=None):
        del record_operator, collect_col_energy, source_fingerprints
        plan = plan_joint_statistics_target_windows(
            modules, specs, max_statistics_bytes=policy['max_statistics_bytes'],
            activation_max_abs=joint_activation_maxima(cache), projection_backend=backend)
        STATE["windows"] = [len(names) for names in plan.windows]
        hashes = {}
        for arm, accumulate in (("A", original_accumulate), ("B", _counter_accumulate)):
            joint_aura.JointOperatorStatisticsLease._accumulate = accumulate
            STATE["arm"] = arm
            for index, names in enumerate(plan.windows):
                STATE["window"] = index
                with joint_aura.JointOperatorStatisticsLease(
                        {name: modules[name] for name in names},
                        {name: specs[name] for name in names},
                        max_statistics_bytes=policy['max_statistics_bytes'],
                        max_candidate_bytes=policy['max_candidate_bytes'],
                        activation_max_abs=joint_activation_maxima(cache),
                        projection_backend=backend) as lease:
                    lease.begin_probe()
                    backward(final=index == len(plan.windows) - 1, lease=lease)
                    lease.finish_observations()
                    _phase(f"arm {arm} window {index} hash start")
                    digest = {}
                    for key in sorted(lease._operators, key=repr):
                        value = lease._operators[key].detach().contiguous().cpu()
                        digest[repr(key)] = hashlib.sha256(
                            value.view(torch.uint8).numpy().tobytes()).hexdigest()
                    digest["activation_terms"] = repr(sorted(
                        (repr(key), value.hex()) for key, value in lease._activation_terms.items()))
                    hashes.setdefault(arm, {})[index] = digest
                    STATE["arms"][arm][str(index)]["statistics_keys"] = len(digest)
                    STATE["arms"][arm][str(index)]["peak_statistics_bytes"] = \
                        lease.telemetry["peak_statistics_bytes"]
                    _phase(f"arm {arm} window {index} hash end")
        joint_aura.JointOperatorStatisticsLease._accumulate = original_accumulate
        STATE["bit_identical"] = hashes["A"] == hashes["B"]
        STATE["statistics_sha256"] = hashlib.sha256(
            json.dumps(hashes["A"], sort_keys=True).encode()).hexdigest()
        STATE["finished_epoch"] = time.time()
        (out / "report.json").write_text(json.dumps(
            {k: v for k, v in STATE.items() if not k.startswith("_")}, indent=1, default=str))
        print(f"[mtp-profile] report {out / 'report.json'} bit_identical={STATE['bit_identical']}",
              flush=True)
        sys.stdout.flush()
        sampler = STATE.get("_sampler")
        if sampler is not None:
            sampler.terminate()
        os._exit(0)

    jsr.observe_and_project_windows = profiled_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timed", type=int, default=4)
    parser.add_argument("--profiled", type=int, default=3)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    source_root = Path(plan["output_root"])
    if source_root.resolve() == out.resolve():
        raise RuntimeError("the profile must not write into the run's output root")
    plan["output_root"] = str(out)
    (out / "prepare").mkdir(exist_ok=True)
    # The run's source owner adopts <output_root>/prepare/source-identity.json.
    shutil.copyfile(source_root / "prepare/source-identity.json",
                    out / "prepare/source-identity.json")
    plan_path = out / "plan-profile.json"
    plan_bytes = json.dumps(plan, indent=1, sort_keys=True).encode()
    plan_path.write_bytes(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    STATE["plan"] = {"source": str(args.plan), "profile": str(plan_path), "sha256": plan_sha}
    STATE["host"] = os.uname().nodename
    STATE["counts"] = {"warmup": args.warmup, "timed": args.timed, "profiled": args.profiled}
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm",
         "--format=csv,noheader", "-lms", "250"],
        stdout=open(out / "power.csv", "w"), stderr=subprocess.DEVNULL)
    STATE["_sampler"] = smi
    _phase("start")
    install(args)
    from prismaquant import tessera_joint_aura
    sys.argv = ["tessera_joint_aura", "run", "--plan", str(plan_path), "--plan-sha256", plan_sha,
                "--prepared", str(args.prepared), "--prepared-sha256", args.prepared_sha256]
    try:
        tessera_joint_aura.main()
    finally:
        smi.terminate()
    raise RuntimeError("the profile returned without reaching the statistics windows")


if __name__ == "__main__":
    main()
