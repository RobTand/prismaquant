#!/usr/bin/env python3
"""streaming_model.py — shared streaming-skeleton infrastructure.

Factored out of `incremental_probe.py` so the cost-measurement side
(`incremental_measure_quant_cost.py`) can reuse the exact same
"skeleton-on-meta, head-resident, decoder-layers-swap" plumbing without
copy-pasting.

What lives here:

  - `StreamingContext`: holds the model, per-layer install resolvers,
    weight map, LayerCache, and a single-worker prefetch pool. Built once,
    reused across every shard.
  - `_build_streaming_context`: one-time setup (AutoConfig, empty
    skeleton, `from_pretrained` with explicit device_map pinning head
    resident and decoder layers to disk, strip accelerate hooks, unload
    layers back to meta).
  - `_classify_shard`: maps a shard-include regex to one of
    {"body", "mtp", "visual", "lm_head"}.

What stays in `incremental_probe`:
  - `build_layer_shard_regexes` / `build_extended_shard_regexes`
  - `load_num_hidden_layers`
  - Body/MTP shard runners (those are Fisher-semantics-specific).

The cost side will import from both this module and
`incremental_probe` (for the regex builders) — the regex helpers are
stable public API that both sides share.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from safetensors import safe_open

from .layer_streaming import (
    _build_fp8_scale_inv_map,
    LayerCache,
    _build_install_resolver,
    _build_weight_map,
    _fast_install,
    _get_layer_list,
    _get_rotary,
    _head_prefixes,
    _materialize,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
)


def _minimax_native_fp8_checkpoint(model_path: str) -> bool:
    """True for MiniMax native-FP8 checkpoints with block scales.

    MiniMax-M2/M2.7 exposes 256 experts as a ModuleList. Transformers
    5.x's FP8 pre-load rewrite currently replaces that ModuleList with
    FP8Experts, then tries to set `experts.0.w1`, which fails because
    FP8Experts is not integer-indexable. The streaming path does not
    need HF's module rewrite: `_read_layer_to_device` reads the source
    fp8 bytes and applies `.weight_scale_inv` inline.
    """
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:
        return False
    model_type = str(cfg.get("model_type", "")).replace("-", "_").lower()
    archs = [str(a) for a in cfg.get("architectures", [])]
    qc = cfg.get("quantization_config") or {}
    return (
        model_type.startswith("minimax_m2")
        or any(a.startswith("MiniMaxM2") for a in archs)
    ) and qc.get("quant_method") == "fp8" and "weight_block_size" in qc


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """Populate deterministic rotary buffers on a meta-built skeleton."""
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(
        dtype=torch.float32, device=device), persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False,
        )
    rotary.attention_scaling = attention_scaling


def _safetensors_cache_dtype_bytes(dtype_name: str,
                                   target_dtype: torch.dtype) -> int:
    """Bytes a safetensors tensor will occupy in the layer cache."""
    dtype_name = str(dtype_name).upper()
    # Floating checkpoint tensors are cast to the requested execution
    # dtype by `_read_layer_to_device` before caching. Native FP8 source
    # weights therefore cache as bf16/fp16/fp32 after block dequant.
    if dtype_name.startswith("F") or dtype_name == "BF16":
        return torch.empty((), dtype=target_dtype).element_size()
    return {
        "BOOL": 1,
        "U8": 1, "I8": 1,
        "U16": 2, "I16": 2,
        "U32": 4, "I32": 4,
        "U64": 8, "I64": 8,
    }.get(dtype_name, 1)


def _estimate_layer_cache_bytes(
    *,
    weight_shard: dict[str, str],
    weight_ckpt: dict[str, str],
    layers_prefix: str,
    num_layers: int,
    target_dtype: torch.dtype,
) -> tuple[int, list[int]]:
    """Estimate dequanted cache bytes per decoder layer without loading data."""
    pat = re.compile(rf"^{re.escape(layers_prefix)}(?P<idx>\d+)\.")
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for model_name, shard in weight_shard.items():
        m = pat.match(model_name)
        if m is None:
            continue
        idx = int(m.group("idx"))
        if idx < 0 or idx >= num_layers:
            continue
        by_shard.setdefault(shard, []).append((idx, weight_ckpt[model_name]))

    sizes = [0 for _ in range(num_layers)]
    try:
        for shard, pairs in by_shard.items():
            with safe_open(shard, framework="pt") as f:
                for idx, ckpt_name in pairs:
                    sl = f.get_slice(ckpt_name)
                    n = 1
                    for dim in sl.get_shape():
                        n *= int(dim)
                    sizes[idx] += n * _safetensors_cache_dtype_bytes(
                        sl.get_dtype(), target_dtype)
    except Exception:
        return 0, sizes
    nonzero = [s for s in sizes if s > 0]
    return (max(nonzero) if nonzero else 0), sizes


def _auto_prefetch_workers(cache_bytes: int, layer_bytes: int,
                           requested: Any = None) -> tuple[int, str]:
    raw = requested
    if raw is None:
        raw = os.environ.get("PREFETCH_WORKERS", "auto")
    if str(raw).strip().lower() not in ("", "auto"):
        return max(1, int(raw)), "explicit"
    if layer_bytes <= 0:
        return 3, "auto-fallback"
    cache_slots = max(1, int(cache_bytes // layer_bytes))
    # Each active worker can hold one not-yet-cached layer in addition to
    # the cache itself. Bound concurrency by cache slots so prefetch does
    # not double memory pressure on small-memory runs.
    workers = min(4, max(1, cache_slots))
    return workers, "auto"


def _auto_prefetch_min_available_bytes(layer_bytes: int,
                                       requested: Any = None) -> tuple[int, str]:
    raw = requested
    if raw is None:
        raw = os.environ.get("PREFETCH_MIN_AVAILABLE_GB", "auto")
    if str(raw).strip().lower() not in ("", "auto"):
        return int(float(raw) * 1024 ** 3), "explicit"
    # Keep enough slack for at least two full dequanted layers plus a
    # fixed floor. On UMA systems this guards both CPU RAM and CUDA
    # allocations, since they share the same physical memory.
    floor = 8 * 1024 ** 3
    if layer_bytes <= 0:
        return floor, "auto-fallback"
    return max(floor, int(2 * layer_bytes)), "auto"


# ---------------------------------------------------------------------------
# Shard classification. Each shard regex falls into exactly one of these
# kinds and is orchestrated by the matching runner in the probe / cost
# script. "body" and "mtp" are the active paths; "visual" is acknowledged
# but skipped in the text-only streaming pipeline.
# ---------------------------------------------------------------------------
_BODY_SHARD_RE = re.compile(r"^model\\\.layers\\\.")
_MTP_SHARD_RE = re.compile(r"^mtp\\\.layers\\\.")
_VISUAL_SHARD_RE = re.compile(r"^model\\\.visual\\\.")
_LM_HEAD_SHARD_RE = re.compile(r"^\^lm_head\$?$")


def _classify_shard(regex: str) -> str:
    if _BODY_SHARD_RE.match(regex):
        return "body"
    if _MTP_SHARD_RE.match(regex):
        return "mtp"
    if _VISUAL_SHARD_RE.match(regex):
        return "visual"
    if _LM_HEAD_SHARD_RE.match(regex):
        return "lm_head"
    return "body"  # conservative fallback: treat as a body pattern


# ---------------------------------------------------------------------------
# Streaming context: skeleton + head resident + per-layer resolvers + cache.
# Built once for the whole run and reused across every shard. Holding this
# object idle between shards costs the head weights + cache RAM only;
# decoder layers live on meta or on disk and get installed transiently.
# ---------------------------------------------------------------------------
class StreamingContext:
    def __init__(self, *, model, base_model, layers, layers_prefix: str,
                 num_layers: int, install_resolvers: list[dict],
                 weight_shard: dict[str, str], weight_ckpt: dict[str, str],
                 layer_cache: LayerCache, prefetch_pool: ThreadPoolExecutor,
                 device: torch.device, dtype: torch.dtype, offload_folder: str,
                 visual_module: Any | None = None,
                 visual_prefix: str | None = None,
                 multimodal: bool = False,
                 fp8_scale_inv_map: dict[str, tuple[str, str]] | None = None,
                 estimated_layer_bytes: int = 0,
                 prefetch_workers: int = 3,
                 prefetch_min_available_bytes: int = 0):
        self.model = model
        self.base_model = base_model
        self.layers = layers
        self.layers_prefix = layers_prefix
        self.num_layers = num_layers
        self.install_resolvers = install_resolvers
        self.weight_shard = weight_shard
        self.weight_ckpt = weight_ckpt
        self.layer_cache = layer_cache
        self.prefetch_pool = prefetch_pool
        self.device = device
        self.dtype = dtype
        self.offload_folder = offload_folder
        # Populated when `_build_streaming_context(..., multimodal=True)`:
        # full visual tower resident on `device`, requires_grad=True on
        # Linear params so Fisher hooks fire in run_multimodal_visual_probe_pass.
        # Also exposes `visual_prefix` so cost / probe code can iterate
        # over visual Linears under `model.visual.*` (or whatever the
        # declared multimodal arch calls it).
        self.visual_module = visual_module
        self.visual_prefix = visual_prefix
        self.multimodal = multimodal
        self.estimated_layer_bytes = int(estimated_layer_bytes or 0)
        self.prefetch_workers = int(prefetch_workers)
        self.prefetch_min_available_bytes = int(prefetch_min_available_bytes or 0)
        self.prefetch_memory_skips = 0
        # Native-FP8 checkpoint dequant map: `{live_weight_key:
        # (shard_path, scale_inv_ckpt_key)}`. When non-empty, every
        # per-layer reload via `_read_layer_to_device` applies the
        # 128x128 block dequant inline so `mod.weight` holds true
        # dequanted weights, not raw fp8 codes cast to bf16. Empty dict
        # for BF16-native checkpoints — loader path is unchanged.
        self.fp8_scale_inv_map = fp8_scale_inv_map or {}
        self._inflight: dict[int, Any] = {}
        self._inflight_lock = threading.Lock()

    def _prefetch_worker(self, L: int):
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype,
            self.device, fp8_scale_inv_map=self.fp8_scale_inv_map)
        self.layer_cache.put(L, tensors)
        with self._inflight_lock:
            self._inflight.pop(L, None)
        return tensors

    def schedule_prefetch(self, L: int):
        if L < 0 or L >= self.num_layers:
            return None
        if self.layer_cache.peek(L):
            return None
        if self.prefetch_min_available_bytes > 0:
            try:
                import psutil
                if psutil.virtual_memory().available < self.prefetch_min_available_bytes:
                    self.prefetch_memory_skips += 1
                    return None
            except Exception:
                pass
        with self._inflight_lock:
            if L in self._inflight:
                return self._inflight[L]
            fut = self.prefetch_pool.submit(self._prefetch_worker, L)
            self._inflight[L] = fut
            return fut

    def ensure_loaded(self, L: int) -> tuple[dict[str, torch.Tensor], str]:
        cached = self.layer_cache.get(L)
        if cached is not None:
            return cached, "hot"
        with self._inflight_lock:
            fut = self._inflight.get(L)
        if fut is not None:
            fut.result()
            cached = self.layer_cache.get(L)
            if cached is not None:
                return cached, "wait"
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype,
            self.device, fp8_scale_inv_map=self.fp8_scale_inv_map)
        self.layer_cache.put(L, tensors)
        return tensors, "cold"

    def install(self, L: int):
        tensors, src = self.ensure_loaded(L)
        _fast_install(self.install_resolvers[L], tensors, self.device, model=self.model)
        # This run is a one-way layer stream. Once the tensors are
        # installed, the model owns them until `unload`; keeping a cache
        # reference would make the current layer MRU and cause prefetch
        # churn to evict the next layer we are about to need.
        self.layer_cache.discard(L)
        return src

    def unload(self, L: int):
        _unload(self.model, [f"{self.layers_prefix}{L}."])

    def shutdown(self):
        self.prefetch_pool.shutdown(wait=True)

    def suggest_prefetch_lookahead(self) -> int:
        if self.estimated_layer_bytes <= 0:
            return 3
        cache_slots = max(
            1, int(self.layer_cache.max_bytes // self.estimated_layer_bytes))
        # Queue at most what the cache can plausibly retain. More than
        # this tends to turn prefetch into churn on memory-constrained
        # runs, especially when backward has become fast.
        # Leave one cache slot for the currently installed layer's live
        # tensors. `install()` drops cache ownership, but the model still
        # owns that layer until the caller unloads it after forward/bwd.
        return max(1, min(12, cache_slots - 1))

    def prefetch_summary(self) -> str:
        with self._inflight_lock:
            inflight = len(self._inflight)
        est_gb = self.estimated_layer_bytes / (1024 ** 3)
        min_gb = self.prefetch_min_available_bytes / (1024 ** 3)
        return (f"Prefetch: workers={self.prefetch_workers} "
                f"inflight={inflight} est_layer={est_gb:.1f}GB "
                f"min_avail={min_gb:.1f}GB "
                f"mem_skips={self.prefetch_memory_skips}")


def _resolve_declared_model_cls(config, default_cls):
    """Return the transformers class named by `config.architectures[0]`
    if importable, else `default_cls`. Used to bypass
    `AutoModelForCausalLM`'s silent text-only downgrade for multimodal
    umbrella configs (e.g. Qwen3_5MoeConfig → Qwen3_5MoeForCausalLM
    text-only, which drops `model.visual.*`)."""
    try:
        import transformers
        arch_names = getattr(config, "architectures", None) or []
        if arch_names and hasattr(transformers, arch_names[0]):
            return getattr(transformers, arch_names[0])
    except Exception:
        pass
    return default_cls


def _find_visual_module(model) -> tuple[Any | None, str]:
    """Return (visual_module, dotted_prefix) if the model has a visual
    tower; (None, '') otherwise. Handles the v5 multimodal umbrella
    layout (`model.model.visual`) and a few common variants."""
    import torch.nn as nn
    # Most common: `model.model.visual` (Qwen3_5MoeModel.visual)
    cand = getattr(model, "model", None)
    if cand is not None:
        vis = getattr(cand, "visual", None)
        if isinstance(vis, nn.Module):
            return vis, "model.visual"
    # Fallback: top-level `model.visual` (some arch variants)
    vis = getattr(model, "visual", None)
    if isinstance(vis, nn.Module):
        return vis, "visual"
    return None, ""


def _build_streaming_context(model_path: str, *,
                             device: torch.device, dtype: torch.dtype,
                             offload_folder: str,
                             cache_headroom_gb: float | None = None,
                             prefetch_workers: int | str | None = None,
                             prefetch_min_available_gb: float | str | None = None,
                             log_prefix: str = "[streaming]",
                             multimodal: bool = False,
                             visual_requires_grad: bool = False,
                             ) -> StreamingContext:
    """One-time setup: AutoConfig + empty skeleton, then manually
    materialize only the always-resident head pieces. Decoder layers
    stay on meta until PrismaQuant streams them from safetensors.

    When `multimodal=True`:
      - Stages via `stage_multimodal` (preserves vision_config).
      - Instantiates via `config.architectures[0]` (declared arch) so the
        visual tower actually materializes — bypasses
        AutoModelForCausalLM's silent text-only downgrade.
      - After the skeleton is built, materializes the head and visual
        tower onto `device` (small — 2-3 GB even at 122B scale). Body
        still streams.
      - If `visual_requires_grad=True`, flips `.requires_grad_(True)` on
        every visual Linear's weight so Fisher backward hooks fire when
        `run_multimodal_visual_probe_pass` drives the combined forward
        (pixel_values → visual_tower → merged inputs_embeds → streamed
        body → lm_head → CE)."""
    import psutil
    from accelerate import init_empty_weights
    from accelerate.hooks import remove_hook_from_module
    from transformers import AutoConfig, AutoModelForCausalLM

    from .sensitivity_probe import stage_multimodal, stage_text_only

    bypass_hf_fp8_rewrite = False
    if multimodal:
        staged = stage_multimodal(model_path)
    else:
        bypass_hf_fp8_rewrite = _minimax_native_fp8_checkpoint(model_path)
        staged = stage_text_only(model_path)
        if bypass_hf_fp8_rewrite:
            print(f"{log_prefix} manual meta streaming load avoids HF fp8 "
                  "module rewrite; PrismaQuant will apply weight_scale_inv "
                  "during layer loads", flush=True)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)

    if multimodal:
        model_cls = _resolve_declared_model_cls(config, AutoModelForCausalLM)
    else:
        model_cls = AutoModelForCausalLM

    with init_empty_weights():
        if model_cls is AutoModelForCausalLM:
            skeleton = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True)
        else:
            skeleton = model_cls._from_config(config)
    skel_base, skel_layers = _get_layer_list(skeleton)
    base_prefix = _resolve_base_prefix(skeleton, skel_base)
    num_layers = len(skel_layers)

    # Find the visual module on the skeleton so we know which names to
    # keep resident in device_map. We rebuild these after `from_pretrained`
    # on the real model anyway — skeleton lookup only tells us the path.
    _skel_visual, skel_visual_prefix = _find_visual_module(skeleton)

    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    resident_device = 0 if device.type == "cuda" else "cpu"

    os.makedirs(offload_folder, exist_ok=True)
    t0 = time.time()
    print(f"{log_prefix} base_prefix={base_prefix!r}  layers={num_layers}  "
          f"head_resident_on={resident_device}  offload={offload_folder}  "
          f"multimodal={multimodal}  visual_prefix={skel_visual_prefix or 'n/a'}",
          flush=True)

    model = skeleton
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)

    weight_shard, weight_ckpt = _build_weight_map(model_path, multimodal=multimodal)
    # Native-FP8 source dequant map. Populated only for checkpoints that
    # ship `.weight_scale_inv` siblings (MiniMax-M2/M2.7, DeepSeek-V3).
    # Empty dict for plain BF16 checkpoints — `_read_layer_to_device`
    # then skips the dequant pass entirely. This map is THE fix for the
    # probe/cost/export mismatch where the streaming loader previously
    # cast fp8 codes to bf16 without applying the 128x128 block scale,
    # leaving every downstream pass operating on raw codes (range ±448)
    # instead of true weights (range ±0.2).
    fp8_scale_inv_map = _build_fp8_scale_inv_map(
        model_path, multimodal=multimodal)
    if fp8_scale_inv_map:
        print(f"{log_prefix} fp8 scale_inv map: {len(fp8_scale_inv_map)} "
              f"weights will be dequanted inline at layer-load",
              flush=True)

    head_pfxs = _head_prefixes(model, base_prefix)
    loaded_head = _materialize(
        model,
        head_pfxs,
        weight_shard,
        weight_ckpt,
        device,
        dtype,
        fp8_scale_inv_map=fp8_scale_inv_map,
    )
    _init_rotary_inplace(base_model, device, dtype)
    print(f"{log_prefix} head materialized ({loaded_head} tensors, "
          f"rotary re-init) in {time.time()-t0:.1f}s", flush=True)

    # Locate the visual module on the meta skeleton. When multimodal is
    # set, fully materialize the visual tower onto `device`; body
    # layers remain meta and stream per shard.
    visual_module = None
    visual_prefix: str | None = None
    if multimodal:
        visual_module, visual_prefix = _find_visual_module(model)
        if visual_module is not None and visual_prefix:
            remove_hook_from_module(visual_module, recurse=True)
            vis_keys = [k for k in weight_shard if k.startswith(visual_prefix + ".")]
            # Load all visual tensors from safetensors onto device.
            tensors = _read_layer_to_device(
                visual_prefix + ".",
                weight_shard, weight_ckpt, dtype, device,
                fp8_scale_inv_map=fp8_scale_inv_map)
            print(f"{log_prefix} materializing visual tower: "
                  f"{len(tensors)}/{len(vis_keys)} tensors -> {device}", flush=True)
            from accelerate.utils.modeling import set_module_tensor_to_device
            for model_name, t in tensors.items():
                set_module_tensor_to_device(model, model_name, device, value=t)
            if visual_requires_grad:
                # Enable grad on every Linear's weight + bias so backward
                # hooks fire on the reverse sweep. Embeddings and norms
                # stay frozen (no Fisher tracked for those).
                import torch.nn as nn
                n_grad = 0
                for n, m in visual_module.named_modules():
                    if isinstance(m, nn.Linear):
                        for p in m.parameters(recurse=False):
                            p.requires_grad_(True)
                            n_grad += 1
                print(f"{log_prefix} visual: enabled grad on "
                      f"{n_grad} Linear params", flush=True)
    print(f"{log_prefix} model ready in {time.time()-t0:.1f}s", flush=True)

    print(f"{log_prefix} building install resolvers for {num_layers} layers ...",
          flush=True)
    t_res = time.time()
    install_resolvers = [
        _build_install_resolver(model, f"{layers_prefix}{L}".rstrip("."))
        for L in range(num_layers)
    ]
    print(f"{log_prefix} resolvers built: "
          f"{sum(len(r) for r in install_resolvers)} tensors across "
          f"{num_layers} layers in {time.time()-t_res:.1f}s", flush=True)

    free_bytes = psutil.virtual_memory().available
    # Resolve headroom: env override > explicit arg > autoscale > legacy 75 GB default.
    resolved_headroom_gb = cache_headroom_gb
    autoscale_diag = None
    if resolved_headroom_gb is None:
        env_val = os.environ.get("CACHE_HEADROOM_GB")
        if env_val not in (None, "", "auto", "AUTO"):
            resolved_headroom_gb = float(env_val)
        else:
            try:
                from .autoscale import pick_cache_headroom_gb
                resolved_headroom_gb, autoscale_diag = pick_cache_headroom_gb(
                    model_path,
                    layers_per_shard=int(os.environ.get("LAYERS_PER_SHARD", "1") or 1)
                        if str(os.environ.get("LAYERS_PER_SHARD", "")).isdigit() else 1,
                    nsamples=int(os.environ.get("NSAMPLES", "32")),
                    seqlen=int(os.environ.get("SEQLEN", "1024")),
                )
            except Exception as e:
                print(f"{log_prefix} autoscale failed ({e!r}); falling back to 75 GB headroom",
                      flush=True)
                resolved_headroom_gb = 75.0
    cache_bytes = max(int(free_bytes) - int(resolved_headroom_gb * 1024 ** 3),
                      8 * 1024 ** 3)
    layer_cache = LayerCache(max_bytes=cache_bytes)
    src = "explicit" if autoscale_diag is None else "autoscaled"
    print(f"{log_prefix} layer cache budget={cache_bytes/(1024**3):.1f} GB "
          f"(free={free_bytes/(1024**3):.1f} GB, headroom={resolved_headroom_gb:.1f} GB, {src})",
          flush=True)
    if autoscale_diag is not None:
        print(f"{log_prefix}   autoscale: shard_working={autoscale_diag['shard_working_gb']:.1f} GB "
              f"+ safety={autoscale_diag['safety_gb']:.1f} GB "
              f"(lps={autoscale_diag['layers_per_shard']})", flush=True)

    estimated_layer_bytes, layer_bytes = _estimate_layer_cache_bytes(
        weight_shard=weight_shard,
        weight_ckpt=weight_ckpt,
        layers_prefix=layers_prefix,
        num_layers=num_layers,
        target_dtype=dtype,
    )
    worker_count, worker_src = _auto_prefetch_workers(
        cache_bytes, estimated_layer_bytes, requested=prefetch_workers)
    min_available_bytes, min_available_src = _auto_prefetch_min_available_bytes(
        estimated_layer_bytes, requested=prefetch_min_available_gb)
    cache_slots = (
        int(cache_bytes // estimated_layer_bytes)
        if estimated_layer_bytes > 0 else 0
    )
    memory_slots = 0
    if estimated_layer_bytes > 0:
        memory_slots = max(
            0, int((free_bytes - min_available_bytes) // estimated_layer_bytes))
    print(f"{log_prefix} prefetch auto: workers={worker_count} "
          f"({worker_src}), cache_slots={cache_slots}, "
          f"memory_slots={memory_slots}, "
          f"est_layer={estimated_layer_bytes/(1024**3):.1f} GB, "
          f"min_avail={min_available_bytes/(1024**3):.1f} GB "
          f"({min_available_src})", flush=True)

    prefetch_pool = ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="prefetch")

    return StreamingContext(
        model=model, base_model=base_model, layers=layers,
        layers_prefix=layers_prefix, num_layers=num_layers,
        install_resolvers=install_resolvers,
        weight_shard=weight_shard, weight_ckpt=weight_ckpt,
        layer_cache=layer_cache, prefetch_pool=prefetch_pool,
        device=device, dtype=dtype, offload_folder=offload_folder,
        visual_module=visual_module,
        visual_prefix=visual_prefix,
        multimodal=multimodal,
        fp8_scale_inv_map=fp8_scale_inv_map,
        estimated_layer_bytes=estimated_layer_bytes,
        prefetch_workers=worker_count,
        prefetch_min_available_bytes=min_available_bytes,
    )
