"""Hardware-adaptive memory sizing for the PrismaQuant pipeline.

Two knobs the probe/cost passes care about:

  1. `layers_per_shard` — how many decoder layers get their Fisher stats
     accumulated in a single reverse sweep. Bigger shards = fewer sweeps
     through the full model = faster probe, but each shard needs more
     gradient + retained-activation memory.

  2. `cache_headroom_gb` — safety margin subtracted from available RAM
     when sizing the streaming layer cache. Lower headroom = bigger
     cache = fewer evictions = fewer `torch.cuda.empty_cache()` stalls
     on UMA hosts, but less slack for autograd spikes.

Both defaults were historically tuned for a 35B-A3B MoE on a 128 GB
Spark. Dense-27B / 122B-A10B / etc. want different values. This module
derives them from the actual checkpoint + host at runtime.

The heuristic is deliberately simple:

    per_layer_bytes(shard) ≈ weight + activations + gradients
    available = free_RAM - safety
    reserved_for_cache = num_layers * per_layer_weight   # hold all layers ⇒ no evictions
    layers_per_shard = (available - reserved_for_cache) / per_layer_bytes

and clamped to [1, num_layers]. Explicit env overrides always win.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_SAFETY_GB = 40.0     # slack for CUDA allocator spikes, HF transformers
                             # overhead, tokenizer caches, OS, prefetch buffers.
                             # 20 GB triggered OOM on Qwen3.6-27B dense (hit 9 GB
                             # swap before SIGKILL); 40 GB has proven absorbent.
                             # NEVER rely on swap — kernel OOM-kills BEFORE swap
                             # fills on many Linux configs.
DEFAULT_ACT_MULT = 12        # multiplier in (N*T*hidden*dtype*K) per layer.
                             # 8 models steady-state activation retention; backward
                             # passes transiently allocate ~50% more scratch tensors
                             # that the caching allocator holds. 12 covers the peak.
DEFAULT_DTYPE_BYTES = 2      # bf16


def _num_layers(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("num_hidden_layers")
               or tc.get("n_layer")
               or cfg.get("num_hidden_layers", 0))


def _hidden_size(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("hidden_size")
               or tc.get("n_embd")
               or cfg.get("hidden_size", 0))


def _model_weight_bytes_on_disk(model_path: str) -> int:
    """Sum of all *.safetensors blob sizes. Works on both HF snapshots
    and staged copies. Falls back to 0 if the dir doesn't exist yet."""
    p = Path(model_path)
    if not p.exists():
        return 0
    total = 0
    for f in p.glob("*.safetensors"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _available_ram_bytes() -> int:
    """Free RAM right now. On UMA (Grace-Blackwell) this is the shared
    LPDDR5X pool that both CPU and GPU draw from — same number matters
    for CUDA and host work."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 64 * 1024 ** 3  # conservative fallback


def estimate_per_layer_bytes(
    model_path: str,
    num_layers: int,
    hidden_size: int,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
) -> tuple[int, int]:
    """Return `(per_layer_weight_bytes, per_layer_active_shard_bytes)`.

    - weight bytes: disk size / num_layers, minus head/embed approximation
    - active_shard bytes: gradients (~weight) + retained activations
      (N·T·hidden·dtype·act_mult)
    """
    total_disk = _model_weight_bytes_on_disk(model_path)
    if total_disk > 0 and num_layers > 0:
        # subtract a conservative 10% for non-layer weights (embed, lm_head, norms)
        body_bytes = int(total_disk * 0.90)
        per_layer_weight = body_bytes // num_layers
    else:
        per_layer_weight = 1 * 1024 ** 3  # 1 GB fallback

    grad_bytes = per_layer_weight  # same shape, same dtype
    act_bytes = nsamples * seqlen * hidden_size * dtype_bytes * act_mult
    per_layer_active = grad_bytes + act_bytes
    return per_layer_weight, per_layer_active


def pick_layers_per_shard(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
    safety_gb: float = DEFAULT_SAFETY_GB,
    available_ram_bytes: int | None = None,
    hold_all_layers_in_cache: bool = True,
    default: int = 2,
) -> tuple[int, dict]:
    """Pick LAYERS_PER_SHARD from host memory + model size.

    Returns `(lps, diagnostics)` so callers can log the derivation.

    `hold_all_layers_in_cache=True` reserves enough RAM for the layer
    cache to fit every decoder layer (zero evictions → no empty_cache
    stalls). Falls back to holding half the layers if that leaves
    too little for shard work.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "lps": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden in config", "lps": default}

    per_layer_weight, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen,
        dtype_bytes=dtype_bytes, act_mult=act_mult,
    )
    avail = available_ram_bytes if available_ram_bytes is not None else _available_ram_bytes()
    safety = int(safety_gb * 1024 ** 3)

    if hold_all_layers_in_cache:
        cache_reserve = n_layers * per_layer_weight
    else:
        cache_reserve = (n_layers // 2) * per_layer_weight

    shard_budget = avail - safety - cache_reserve
    # If reserving the full cache leaves too little, fall back to half-cache
    if shard_budget < per_layer_active and hold_all_layers_in_cache:
        return pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen,
            dtype_bytes=dtype_bytes, act_mult=act_mult,
            safety_gb=safety_gb, available_ram_bytes=avail,
            hold_all_layers_in_cache=False, default=default,
        )
    shard_budget = max(shard_budget, per_layer_active)  # never below 1 layer

    lps = max(1, int(shard_budget // per_layer_active))
    lps = min(lps, n_layers)

    return lps, {
        "lps": lps,
        "n_layers": n_layers,
        "hidden": hidden,
        "per_layer_weight_gb": per_layer_weight / 1024 ** 3,
        "per_layer_active_gb": per_layer_active / 1024 ** 3,
        "available_gb": avail / 1024 ** 3,
        "safety_gb": safety_gb,
        "cache_reserve_gb": cache_reserve / 1024 ** 3,
        "shard_budget_gb": shard_budget / 1024 ** 3,
        "hold_all_layers": hold_all_layers_in_cache,
    }


def pick_cache_headroom_gb(
    model_path: str,
    *,
    safety_gb: float = DEFAULT_SAFETY_GB,
    layers_per_shard: int = 1,
    nsamples: int = 32,
    seqlen: int = 1024,
    default: float = 75.0,
) -> tuple[float, dict]:
    """Pick `cache_headroom_gb` so the layer cache gets (available - headroom)
    bytes for fitting decoder layers. Returns `(headroom_gb, diagnostics)`.

    The probe's active working set dominates the headroom: safety margin
    + gradients/activations for `layers_per_shard` layers. Anything
    leftover goes to the streaming cache.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "headroom_gb": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden", "headroom_gb": default}

    _, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen,
    )
    shard_working_bytes = layers_per_shard * per_layer_active
    headroom_bytes = shard_working_bytes + int(safety_gb * 1024 ** 3)
    headroom_gb = headroom_bytes / 1024 ** 3
    return headroom_gb, {
        "headroom_gb": headroom_gb,
        "shard_working_gb": shard_working_bytes / 1024 ** 3,
        "safety_gb": safety_gb,
        "layers_per_shard": layers_per_shard,
    }


def autoscale(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    layers_per_shard_env: str | int | None = None,
    cache_headroom_gb_env: str | float | None = None,
    safety_gb: float = DEFAULT_SAFETY_GB,
) -> tuple[int, float, dict]:
    """Compute `(layers_per_shard, cache_headroom_gb)` from model + host.

    Explicit env overrides win:
      - `layers_per_shard_env` (int or stringified int) skips LPS autoscale
      - `cache_headroom_gb_env` (float or stringified float) skips headroom autoscale

    Use `"auto"` or `None` to request autoscale.
    """
    diag: dict = {}

    # Parse LPS override
    lps: int
    if layers_per_shard_env in (None, "", "auto", "AUTO"):
        lps, lps_diag = pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen, safety_gb=safety_gb,
        )
        diag["lps_autoscaled"] = lps_diag
    else:
        lps = int(layers_per_shard_env)
        diag["lps_source"] = f"explicit={lps}"

    # Parse headroom override
    headroom: float
    if cache_headroom_gb_env in (None, "", "auto", "AUTO"):
        headroom, hr_diag = pick_cache_headroom_gb(
            model_path, safety_gb=safety_gb,
            layers_per_shard=lps, nsamples=nsamples, seqlen=seqlen,
        )
        diag["headroom_autoscaled"] = hr_diag
    else:
        headroom = float(cache_headroom_gb_env)
        diag["headroom_source"] = f"explicit={headroom}"

    return lps, headroom, diag


if __name__ == "__main__":
    # CLI usage: python -m prismaquant.autoscale <model_path> [--nsamples N --seqlen T]
    import argparse
    ap = argparse.ArgumentParser(description="Print autoscaled memory knobs.")
    ap.add_argument("model_path")
    ap.add_argument("--nsamples", type=int, default=int(os.environ.get("NSAMPLES", 32)))
    ap.add_argument("--seqlen", type=int, default=int(os.environ.get("SEQLEN", 1024)))
    ap.add_argument("--safety-gb", type=float, default=DEFAULT_SAFETY_GB)
    ap.add_argument("--lps", default=os.environ.get("LAYERS_PER_SHARD"))
    ap.add_argument("--headroom", default=os.environ.get("CACHE_HEADROOM_GB"))
    args = ap.parse_args()

    lps, hr, diag = autoscale(
        args.model_path,
        nsamples=args.nsamples, seqlen=args.seqlen,
        layers_per_shard_env=args.lps,
        cache_headroom_gb_env=args.headroom,
        safety_gb=args.safety_gb,
    )
    print(f"LAYERS_PER_SHARD={lps}")
    print(f"CACHE_HEADROOM_GB={hr:.1f}")
    print(json.dumps(diag, indent=2))
