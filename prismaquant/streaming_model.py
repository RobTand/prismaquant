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

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .layer_streaming import (
    LayerCache,
    _build_install_resolver,
    _build_weight_map,
    _fast_install,
    _get_layer_list,
    _read_layer_to_device,
    _resolve_base_prefix,
    _unload,
)


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
                 multimodal: bool = False):
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
        self._inflight: dict[int, Any] = {}
        self._inflight_lock = threading.Lock()

    def _prefetch_worker(self, L: int):
        prefix = f"{self.layers_prefix}{L}."
        tensors = _read_layer_to_device(
            prefix, self.weight_shard, self.weight_ckpt, self.dtype, self.device)
        self.layer_cache.put(L, tensors)
        with self._inflight_lock:
            self._inflight.pop(L, None)
        return tensors

    def schedule_prefetch(self, L: int):
        if L < 0 or L >= self.num_layers:
            return None
        if self.layer_cache.peek(L):
            return None
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
            prefix, self.weight_shard, self.weight_ckpt, self.dtype, self.device)
        self.layer_cache.put(L, tensors)
        return tensors, "cold"

    def install(self, L: int):
        tensors, src = self.ensure_loaded(L)
        _fast_install(self.install_resolvers[L], tensors, self.device, model=self.model)
        return src

    def unload(self, L: int):
        _unload(self.model, [f"{self.layers_prefix}{L}."])

    def shutdown(self):
        self.prefetch_pool.shutdown(wait=True)


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
                             log_prefix: str = "[streaming]",
                             multimodal: bool = False,
                             visual_requires_grad: bool = False,
                             ) -> StreamingContext:
    """One-time setup: AutoConfig + empty skeleton, from_pretrained with an
    explicit device_map that pins the head resident and every decoder layer
    to disk, then strip accelerate's auto-load hooks and unload each layer
    back to meta so WE own materialization from this point on.

    When `multimodal=True`:
      - Stages via `stage_multimodal` (preserves vision_config).
      - Instantiates via `config.architectures[0]` (declared arch) so the
        visual tower actually materializes — bypasses
        AutoModelForCausalLM's silent text-only downgrade.
      - Maps the decoder layers to `"disk"` as usual; the visual tower
        and head pieces stay resident.
      - After the skeleton is built, FULLY materializes the visual tower
        onto `device` (small — 2-3 GB even at 122B scale). Body still
        streams.
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

    if multimodal:
        staged = stage_multimodal(model_path)
    else:
        staged = stage_text_only(model_path)
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
    del skeleton, skel_base, skel_layers

    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    base = base_prefix if base_prefix else ""
    device_map: dict[str, object] = {}
    resident_device = 0 if device.type == "cuda" else "cpu"
    for pfx in (f"{base}.embed_tokens" if base else "embed_tokens",
                f"{base}.norm" if base else "norm",
                f"{base}.rotary_emb" if base else "rotary_emb",
                "lm_head"):
        device_map[pfx] = resident_device
    for L in range(num_layers):
        device_map[f"{base}.layers.{L}" if base else f"layers.{L}"] = "disk"

    # In multimodal mode, keep the entire visual tower resident so the
    # Fisher probe's backward sweep can flow gradients into visual
    # Linears. Visual weights are small (2-3 GB even at 122B scale).
    if multimodal and skel_visual_prefix:
        device_map[skel_visual_prefix] = resident_device

    os.makedirs(offload_folder, exist_ok=True)
    t0 = time.time()
    print(f"{log_prefix} base_prefix={base_prefix!r}  layers={num_layers}  "
          f"head_resident_on={resident_device}  offload={offload_folder}  "
          f"multimodal={multimodal}  visual_prefix={skel_visual_prefix or 'n/a'}",
          flush=True)

    model = model_cls.from_pretrained(
        staged,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
        offload_buffers=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)

    for L in range(num_layers):
        remove_hook_from_module(layers[L], recurse=True)
    for L in range(num_layers):
        _unload(model, [f"{layers_prefix}{L}."])

    weight_shard, weight_ckpt = _build_weight_map(model_path, multimodal=multimodal)

    # Locate the actual visual module on the real (post-from_pretrained)
    # model. When multimodal is set, fully materialize the visual tower
    # onto `device`: accelerate pinned it to resident_device in the
    # device_map, but via the `offload_folder` path it may still be
    # on meta for tensors the loader didn't find in the index. Our
    # own `_read_layer_to_device` + `_fast_install`-style load catches
    # any strays reliably.
    visual_module = None
    visual_prefix: str | None = None
    if multimodal:
        visual_module, visual_prefix = _find_visual_module(model)
        if visual_module is not None and visual_prefix:
            # Strip accelerate hooks off the visual tower so we own its
            # materialization (same hygiene as the decoder layers).
            remove_hook_from_module(visual_module, recurse=True)
            # Count how many visual tensors already ended up on meta
            # (they shouldn't if device_map pinned them resident, but
            # HF/accelerate can still miss offload_buffers entries).
            # Install any that are on meta using our streaming primitive.
            vis_keys = [k for k in weight_shard if k.startswith(visual_prefix + ".")]
            # Load all visual tensors from safetensors onto device.
            tensors = _read_layer_to_device(
                visual_prefix + ".",
                weight_shard, weight_ckpt, dtype, device)
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

    # Modern NVMe handles concurrent queue depth well; 3 workers saturate
    # the drive and eliminate prefetcher-stall windows where main's compute
    # drains the ahead-queue faster than a single reader can refill it.
    # Going higher (>3) sees diminishing returns and risks cache thrash.
    prefetch_pool = ThreadPoolExecutor(
        max_workers=3, thread_name_prefix="prefetch")

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
    )
