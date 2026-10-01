"""Before/after profiles on one GLM-5.3 MoE layer: PQ #1931 and PQ #1935.

One process, one box, one input: the per-expert loop (verbatim, from the
equality test) against the vectorized routing. The layer is the real
``Glm5NextTextMoE`` with layer ``--layer``'s router and expert weights read
from the BF16 source. The input is one census batch, ``[1, 512, 4096]``
(``tessera_campaign`` splits the calibration IDs one sample at a time), drawn
as ``randn * post_attention_layernorm.weight``.

Records, per variant:
  * outputs equal to the loop (``torch.equal`` on every tensor, plus dtype and
    shape) for ``capture_down`` on and off, with and without subsampling;
  * host syncs per call (``torch.cuda.set_sync_debug_mode("warn")``);
  * one ``torch.profiler`` trace per call: CPU wall, summed CUDA kernel time,
    sync and copy runtime-call counts;
  * per-call wall time over a sustained phase, with in-process GPU power
    samples and the phase's UTC bounds for the Netdata cross-check.
A second pair of phases adds the census consumer's per-expert work (the
``tessera_campaign.accumulate`` amax and Hessian gram) to show what the hook
as a whole gains.

PQ #1935: the same layer's 864 projected units (288 experts x gate, up, down)
through ``_checked_projected_units`` -- the old per-unit host copy (verbatim,
from that issue's test) against the device comparison -- with the live views
cut from the loaded packed parameters as the census's are. Each pass checks
the whole layer; the source pages stay cached (``release_source_pages`` off),
so the passes time the comparison, not the disk.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import statistics
import subprocess
import threading
import time
import warnings
from pathlib import Path

import torch

from prismaquant import measure_quant_cost as mqc


def _utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _test_module(filename, name):
    path = Path(__file__).resolve().parents[1] / "tests" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference():
    return _test_module("test_derive_per_expert_routing_1931.py",
                        "routing_1931_reference")._reference_derive


def _build_layer(model_dir: Path, layer: int, device: str):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as glm

    text = AutoConfig.from_pretrained(model_dir).text_config
    with torch.device("meta"):
        moe = glm.Glm5NextTextMoE(text)
    # Cast on meta, then allocate: bf16 storage without a float32 detour.
    moe = moe.to(torch.bfloat16).to_empty(device=device).eval()
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    # The routing reads the router and gate_up_proj; the #1935 check reads
    # every expert projection. The shared experts stay allocated but unread.
    wanted = {k: v for k, v in index.items()
              if k.startswith(prefix + "mlp.gate.") or k.startswith(prefix + "mlp.experts.")
              or k == prefix + "post_attention_layernorm.weight"}
    by_file: dict[str, list[str]] = {}
    for key, fname in wanted.items():
        by_file.setdefault(fname, []).append(key)
    tensors = {}
    for fname, keys in sorted(by_file.items()):
        with safe_open(model_dir / fname, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    E, inter = text.num_local_experts, text.moe_intermediate_size
    with torch.no_grad():
        moe.gate.weight.copy_(tensors[prefix + "mlp.gate.weight"])
        # The bias keeps its checkpoint dtype, as the model loader leaves it.
        moe.gate.register_buffer("e_score_correction_bias", tensors[
            prefix + "mlp.gate.e_score_correction_bias"].to(device))
        for e in range(E):
            ep = f"{prefix}mlp.experts.{e}."
            moe.experts.gate_up_proj[e, :inter].copy_(tensors[ep + "gate_proj.weight"])
            moe.experts.gate_up_proj[e, inter:].copy_(tensors[ep + "up_proj.weight"])
            moe.experts.down_proj[e].copy_(tensors[ep + "down_proj.weight"])
    ln = tensors[prefix + "post_attention_layernorm.weight"].to(device=device, dtype=torch.bfloat16)
    tensors.clear()
    files = {k: v for k, v in wanted.items() if k.startswith(prefix + "mlp.experts.")}
    return moe, ln, text, files


def _projected_units(moe, files, layer, inter):
    """The layer's projected units, as the producer binds them, and their live views."""
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    bound, weights = {}, {}
    for e in range(moe.experts.num_experts):
        views = {"gate_proj": moe.experts.gate_up_proj[e, :inter],
                 "up_proj": moe.experts.gate_up_proj[e, inter:],
                 "down_proj": moe.experts.down_proj[e]}
        for proj, view in views.items():
            key = f"{prefix}{e}.{proj}.weight"
            bound.setdefault(f"layer{layer}.{proj}", {})[key] = {
                "source_tensor": key, "rows": int(view.shape[0]), "cols": int(view.shape[1])}
            weights[key] = view
    return bound, weights, {"tensors": files}


def _same(a, b):
    if a["row_counts"] != b["row_counts"]:
        return False
    for key in ("gate_up", "down", "gate_weights"):
        if len(a[key]) != len(b[key]):
            return False
        for x, y in zip(a[key], b[key]):
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                return False
    return True


def _sync_count(fn):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return sum("synchroniz" in str(w.message) for w in caught)


def _profile(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    averages = prof.key_averages()
    kernel_us = sum(getattr(ev, "self_device_time_total", 0.0) for ev in averages)
    runtime = {ev.key: ev.count for ev in averages if ev.key.startswith("cuda") and (
        "Synchronize" in ev.key or "Memcpy" in ev.key or "Launch" in ev.key)}
    top = sorted(averages, key=lambda ev: getattr(ev, "self_cpu_time_total", 0.0), reverse=True)[:8]
    return {"wall_s_profiled": wall, "cuda_kernel_s": kernel_us / 1e6,
            "gpu_busy_fraction": (kernel_us / 1e6) / wall if wall else None,
            "runtime_calls": runtime,
            "top_self_cpu": [{"op": ev.key, "count": ev.count,
                              "self_cpu_s": ev.self_cpu_time_total / 1e6} for ev in top]}


class _Power:
    """nvidia-smi power samples every 500 ms while a phase runs."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                self.samples.append((time.time(), float(out[0])))
            except Exception:  # noqa: BLE001 - a missing sample is recorded as absence
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        watts = [w for _, w in self.samples]
        if not watts:
            return {"samples": 0}
        return {"samples": len(watts), "mean_w": statistics.fmean(watts),
                "median_w": statistics.median(watts), "min_w": min(watts), "max_w": max(watts)}


def _phase(name, fn, seconds, *, warmup=3, min_calls=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = _utc()
    deadline = time.perf_counter() + seconds
    with _Power() as power:
        while time.perf_counter() < deadline or len(times) < min_calls:
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    end = _utc()
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]  # noqa: E731
    row = {"phase": name, "utc_start": start, "utc_end": end, "calls": len(times),
           "wall_s_median": statistics.median(times), "wall_s_p10": q(0.10), "wall_s_p90": q(0.90),
           "wall_s_mean": statistics.fmean(times), "power": power.summary()}
    print(json.dumps(row), flush=True)
    return row


_ACC: dict = {}


def _consumer(derived, capture_down):
    """The census accumulate's per-expert device work: amax and the Hessian gram.

    Mirrors ``tessera_campaign.accumulate`` (``fmax`` of ``|x|.amax()``, then
    ``hess += f32.t() @ f32``) into one accumulator per input kind, so the
    bench keeps 80 MB of state instead of the census's 288 per-expert Hessians.
    """
    kinds = ("gate_up", "down") if capture_down else ("gate_up",)
    for kind in kinds:
        for x in derived[kind]:
            flat = x.detach().reshape(-1, x.shape[-1])
            if not flat.shape[0]:
                continue
            batch_max = flat.abs().amax().float()
            amax, hess = _ACC.get(kind, (None, None))
            if amax is None:
                amax = torch.zeros((), dtype=torch.float32, device=batch_max.device)
            f32 = flat.to(dtype=torch.float32)
            gram = f32.t() @ f32
            hess = gram if hess is None else hess.add_(gram)
            _ACC[kind] = (torch.fmax(amax, batch_max), hess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda"
    reference = _load_reference()
    record = {"schema": "prismaquant.pq1931_routing_profile.v1", "utc_start": _utc(),
              "host": os.uname().nodename, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "layer": args.layer, "tokens": args.tokens,
              "git_head": os.environ.get("PQ1931_GIT_HEAD")}
    t0 = time.perf_counter()
    moe, ln, text, files = _build_layer(args.model, args.layer, device)
    record["load_s"] = time.perf_counter() - t0
    gen = torch.Generator(device=device).manual_seed(1931)
    X = (torch.randn(1, args.tokens, text.hidden_size, device=device, generator=gen,
                     dtype=torch.float32).to(torch.bfloat16) * ln)
    experts, parent = moe.experts, moe
    variants = {"loop": reference, "vectorized": mqc.derive_per_expert_activations}

    equality = []
    for capture_down in (True, False):
        for max_rows in (None, 16):
            kwargs = dict(capture_down=capture_down, max_rows_per_expert=max_rows)
            a = reference(experts, X, parent, **kwargs)
            b = mqc.derive_per_expert_activations(experts, X, parent, **kwargs)
            equality.append({"capture_down": capture_down, "max_rows_per_expert": max_rows,
                             "equal": _same(a, b), "zero_row_experts": a["row_counts"].count(0),
                             "max_rows_routed": max(a["row_counts"])})
            del a, b
    record["equality"] = equality
    print(json.dumps({"equality": equality}), flush=True)

    census_kwargs = dict(capture_down=True, max_rows_per_expert=None)
    calls = {name: (lambda f=f: f(experts, X, parent, **census_kwargs)) for name, f in variants.items()}
    hooks = {name: (lambda f=f: _consumer(f(experts, X, parent, **census_kwargs), True))
             for name, f in variants.items()}
    for fn in list(calls.values()) + list(hooks.values()):
        fn()
    record["syncs_per_call"] = {name: _sync_count(fn) for name, fn in calls.items()}
    record["profile_per_call"] = {name: _profile(fn) for name, fn in calls.items()}
    record["profile_per_hook"] = {name: _profile(fn) for name, fn in hooks.items()}
    print(json.dumps({k: record[k] for k in ("syncs_per_call", "profile_per_call")}), flush=True)

    phases = []
    idle_start = _utc()
    with _Power() as idle:
        time.sleep(15)
    phases.append({"phase": "idle", "utc_start": idle_start, "utc_end": _utc(), "power": idle.summary()})
    for name in ("loop", "vectorized"):
        phases.append(_phase(f"derive_{name}", calls[name], args.seconds))
    for name in ("loop", "vectorized"):
        phases.append(_phase(f"hook_{name}", hooks[name], args.seconds))
    # PQ #1935: the layer's projected-unit byte check.
    from prismaquant import tessera_campaign as campaign
    old_check = _test_module("test_projected_unit_check_1935.py",
                             "check_1935_reference")._reference_checked_units
    bound, weights, source = _projected_units(moe, files, args.layer, text.moe_intermediate_size)
    check_kwargs = dict(weights=weights, model_path=args.model, source=source,
                        release_source_pages=False)
    checks = {"check_host": lambda: old_check(bound, **check_kwargs),
              "check_device": lambda: campaign._checked_projected_units(bound, **check_kwargs)}
    for fn in checks.values():
        assert len(fn()) == len(weights)  # warm the page cache; both accept every unit
    record["projected_units"] = len(weights)
    record["check_syncs_per_layer"] = {name: _sync_count(fn) for name, fn in checks.items()}
    record["check_profile_per_layer"] = {name: _profile(fn) for name, fn in checks.items()}
    print(json.dumps({k: record[k] for k in ("check_syncs_per_layer",)}), flush=True)
    for name, fn in checks.items():
        phases.append(_phase(name, fn, args.seconds, warmup=1))

    record["phases"] = phases
    by = {p["phase"]: p for p in phases}
    record["speedup_median"] = {
        "derive": by["derive_loop"]["wall_s_median"] / by["derive_vectorized"]["wall_s_median"],
        "hook": by["hook_loop"]["wall_s_median"] / by["hook_vectorized"]["wall_s_median"],
        "check": by["check_host"]["wall_s_median"] / by["check_device"]["wall_s_median"],
    }
    record["utc_end"] = _utc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps({"speedup_median": record["speedup_median"]}), flush=True)
    if not all(row["equal"] for row in equality):
        raise SystemExit("vectorized routing differs from the per-expert loop")


if __name__ == "__main__":
    main()
