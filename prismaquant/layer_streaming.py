"""Shared primitives for layer-by-layer streaming of HF model weights
from safetensors.

Used by the unified incremental probe, cost, and export paths. Each
primitive is a pure move from the original monolithic streaming probe —
signatures and behavior are byte-identical. The goal is to share one
install/unload/cache implementation across every stage of the pipeline
so the allocator's view of layer materialization is the same regardless
of which script is driving it.

Notes:
  - The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` hack must be
    set by the *entrypoint* module before torch.cuda initializes; it is
    deliberately NOT set here so this module can be imported lazily
    after cuda is already up.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open


def _build_weight_map(model_path: str, *,
                      multimodal: bool = False
                      ) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({model_key: shard_path}, {model_key: checkpoint_key}).

    Multimodal umbrella checkpoints store tensors under
    `model.language_model.layers.X.*` (and similar visual/audio paths),
    but the staged text-only model has layers at `model.layers.X.*`.
    HF's `from_pretrained` applies a `WeightsMapper` to bridge the two;
    our streaming loader reads safetensors directly, so we apply the
    same rename up front and expose the model-side key to callers
    (while remembering the checkpoint-side key for the safetensors open).

    Also drops keys the text-only probe never needs (visual encoder,
    audio encoder, MTP — those follow their own code paths and would
    shadow real body tensors if they share suffixes).

    When `multimodal=True` the multimodal umbrella arch is used in the
    streaming skeleton (body at `model.language_model.layers.X.*`,
    visual at `model.visual.*`); no rename is applied and visual/audio
    keys are preserved so `_materialize` can load them onto the visual
    tower. MTP stays dropped — MTP has its own synthesis path."""
    def _rename_text_only(k: str) -> str | None:
        # Prefix renames mirroring vLLM's `hf_to_vllm_mapper` for the
        # multimodal → text-only body remap. Order matters: more
        # specific rules first.
        # `.weight_scale_inv` is the native-FP8 per-128x128-block scale
        # that pairs with a `.weight` fp8_e4m3fn tensor (MiniMax-M2/M2.7,
        # DeepSeek-V3, any natively-FP8 checkpoint). Drop it from the
        # primary weight map so callers that install by model-qname
        # don't try to install it onto the redeclared Linear; fp8-aware
        # loaders pick it up from `_build_fp8_scale_inv_map` instead
        # and apply the scale during load.
        if k.endswith(".weight_scale_inv"):
            return None
        if k.startswith("model.visual.") or k.startswith("model.audio_tower.") \
                or k.startswith("model.vision_tower.") \
                or k.startswith("model.embed_vision.") \
                or k.startswith("model.embed_audio.") \
                or k.startswith("mtp."):
            return None
        if k.startswith("model.language_model."):
            return "model." + k[len("model.language_model."):]
        return k

    def _rename_multimodal(k: str) -> str | None:
        # Multimodal umbrella keeps `model.language_model.layers.*` and
        # `model.visual.*` verbatim — they already match the declared
        # multimodal model's named modules. MTP weights follow a
        # separate synthesis path; drop them so they don't shadow
        # anything.
        if k.endswith(".weight_scale_inv"):
            return None
        if k.startswith("mtp."):
            return None
        return k

    _rename = _rename_multimodal if multimodal else _rename_text_only

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            raw = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"no safetensors under {model_path}")
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}
    model_to_shard: dict[str, str] = {}
    model_to_ckpt: dict[str, str] = {}
    for ck, shard in raw.items():
        mk = _rename(ck)
        if mk is None:
            continue
        model_to_shard[mk] = os.path.join(model_path, shard)
        model_to_ckpt[mk] = ck
    return model_to_shard, model_to_ckpt


def _build_fp8_scale_inv_map(model_path: str, *,
                             multimodal: bool = False
                             ) -> dict[str, tuple[str, str]]:
    """Return `{model_weight_key: (scale_shard_path, scale_ckpt_key)}`
    for every native-FP8 weight tensor (fp8_e4m3fn + paired
    `.weight_scale_inv` fp32 block scale).

    The key space matches what `_build_weight_map` returns — i.e., the
    live model qname for the weight (`...something.weight`). Callers
    pair it with `_read_layer_to_device(..., fp8_scale_inv_map=...)`
    to apply the 128x128 block dequant inline at load.

    Returns `{}` for checkpoints that have no `.weight_scale_inv`
    tensors — load-time dequant is then a no-op and callers behave
    exactly as they did before this function existed.
    """
    # Same rename table as `_build_weight_map`, but we KEEP
    # `.weight_scale_inv` and rewrite the same body-vs-multimodal
    # remap so lookup keys line up with the weight map.
    def _rename_text_only(k: str) -> str | None:
        if (k.startswith("model.visual.")
                or k.startswith("model.audio_tower.")
                or k.startswith("model.vision_tower.")
                or k.startswith("model.embed_vision.")
                or k.startswith("model.embed_audio.")
                or k.startswith("mtp.")):
            return None
        if k.startswith("model.language_model."):
            return "model." + k[len("model.language_model."):]
        return k

    def _rename_multimodal(k: str) -> str | None:
        if k.startswith("mtp."):
            return None
        return k

    _rename = _rename_multimodal if multimodal else _rename_text_only

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            raw = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            return {}
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}

    out: dict[str, tuple[str, str]] = {}
    for ck_key, shard in raw.items():
        if not ck_key.endswith(".weight_scale_inv"):
            continue
        renamed = _rename(ck_key)
        if renamed is None:
            continue
        # Key the map by the WEIGHT name (drop `_scale_inv`, leaving
        # `...something.weight`) so callers can look up by the same
        # model_key they use in `_build_weight_map`.
        assert renamed.endswith(".weight_scale_inv"), renamed
        weight_key = renamed[: -len("_scale_inv")]
        out[weight_key] = (os.path.join(model_path, shard), ck_key)
    return out


def _dequant_fp8_block_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Apply the (ceil(out/128), ceil(in/128)) block scale to a 2D
    fp8-sourced weight and return bf16. Used by the fp8-aware streaming
    loader for native-FP8 checkpoints (MiniMax-M2/M2.7, DeepSeek-V3).

    `weight` is the fp8-sourced tensor as returned by the streaming
    loader — typically already cast to bf16 (each fp8 code maps
    losslessly to bf16). `scale_inv` is the fp32 `weight_scale_inv`
    block tensor; we cast it to bf16 here, since the fp8 code's 3-bit
    mantissa (≈12.5% precision) dominates the overall error budget —
    bf16's ~0.4% scale precision is well below that.

    Implementation: reshape to 4-D block tiles `(out_blocks, block_r,
    in_blocks, block_c)` and multiply by a broadcasted scale of shape
    `(out_blocks, 1, in_blocks, 1)`. Avoids materializing the full
    (out_dim, in_dim) expanded-scale intermediate — on MiniMax-M2.7's
    772 fp8 weights per layer this cuts allocation pressure ~25× vs a
    `repeat_interleave(128)`-pair expansion, and keeps everything in
    bf16 so no fp32 intermediate is ever allocated.
    """
    out_dim, in_dim = weight.shape
    block_r, block_c = block
    target_dtype = torch.bfloat16
    if out_dim % block_r != 0 or in_dim % block_c != 0:
        # Unaligned tail: fall back to the expanded-scale path. Shouldn't
        # hit in practice on MiniMax/DeepSeek — both ship weights that
        # are exact multiples of 128 along both dims.
        scale = scale_inv.to(device=weight.device, dtype=target_dtype)
        expanded = scale.repeat_interleave(block_r, dim=0)[:out_dim]
        expanded = expanded.repeat_interleave(block_c, dim=1)[:, :in_dim]
        return (weight.to(target_dtype) * expanded)
    out_blocks = out_dim // block_r
    in_blocks = in_dim // block_c
    scale_bf16 = scale_inv.to(device=weight.device, dtype=target_dtype)
    scale_view = scale_bf16.reshape(out_blocks, 1, in_blocks, 1)
    w_bf16 = weight.to(target_dtype)
    w4 = w_bf16.reshape(out_blocks, block_r, in_blocks, block_c)
    return (w4 * scale_view).reshape(out_dim, in_dim)


def _apply_fp8_dequant_inplace(
    out: dict[str, torch.Tensor],
    fp8_scale_inv_map: dict[str, tuple[str, str]],
    device: torch.device,
) -> int:
    """For each tensor in `out` whose key matches a `fp8_scale_inv_map`
    entry, read the scale_inv, apply the 128x128 block dequant, and
    replace the loaded tensor with the dequanted bf16 weight.

    Tensors and scales are both grouped by shape and multiplied in a
    single batched 5-D broadcast op per shape-group. On MiniMax-M2.7
    that collapses 772 per-tensor kernel launches per layer-load
    (256 experts × 3 projs + 4 attention projs) into ~4-5 shape
    buckets — which is what the prefetch thread's CPU cost scales on,
    since each kernel launch has fixed Python + CUDA dispatch overhead.
    """
    if not fp8_scale_inv_map:
        return 0
    # Collect (name, scale_key) per shard so we can open each shard
    # once and pre-read every scale we need before batching.
    scale_reads: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name in list(out.keys()):
        entry = fp8_scale_inv_map.get(model_name)
        if entry is None:
            continue
        shard, scale_key = entry
        scale_reads[shard].append((model_name, scale_key))
    if not scale_reads:
        return 0

    # Step 1: Read all scales from source safetensors once per shard.
    loaded_scales: dict[str, torch.Tensor] = {}  # name -> fp32 scale (cpu)
    for shard, reads in scale_reads.items():
        with safe_open(shard, framework="pt") as f:
            for model_name, scale_key in reads:
                loaded_scales[model_name] = f.get_tensor(scale_key)

    # Step 2: Group matched weights by (out_dim, in_dim) shape. We only
    # batch along exact-128-multiple shapes; odd-shaped tensors (rare)
    # fall back to the per-tensor path.
    block_r, block_c = 128, 128
    by_shape: dict[tuple[int, int], list[str]] = defaultdict(list)
    fallback: list[str] = []
    for name in loaded_scales:
        w = out[name]
        if w.dim() != 2:
            fallback.append(name)
            continue
        out_dim, in_dim = w.shape
        if out_dim % block_r != 0 or in_dim % block_c != 0:
            fallback.append(name)
            continue
        by_shape[(out_dim, in_dim)].append(name)

    dequanted = 0

    # Step 3: Batched multiply per shape-group. Stack all weights of
    # the same shape along a new outer dim, stack their scales the
    # same way, reshape both into block-tile form, one bf16 multiply,
    # split back.
    for (out_dim, in_dim), names in by_shape.items():
        out_blocks = out_dim // block_r
        in_blocks = in_dim // block_c
        E = len(names)
        # Stack weights: (E, out, in) bf16
        w_stack = torch.stack([out[n] for n in names], dim=0).to(
            torch.bfloat16)
        # Stack scales: (E, out_blocks, in_blocks) bf16 on device
        s_stack = torch.stack(
            [loaded_scales[n] for n in names], dim=0
        ).to(device=device, dtype=torch.bfloat16)
        # Reshape to block-tile form:
        #   w: (E, out_blocks, block_r, in_blocks, block_c)
        #   s: (E, out_blocks, 1, in_blocks, 1)
        w4 = w_stack.reshape(E, out_blocks, block_r, in_blocks, block_c)
        s4 = s_stack.reshape(E, out_blocks, 1, in_blocks, 1)
        dequanted_stack = (w4 * s4).reshape(E, out_dim, in_dim)
        # Split back — zero-copy via unbind.
        for i, n in enumerate(names):
            out[n] = dequanted_stack[i].contiguous()
        dequanted += E
        del w_stack, s_stack, w4, s4, dequanted_stack

    # Step 4: Fallback path for any shapes we didn't batch.
    for name in fallback:
        w = out[name]
        scale_fp = loaded_scales[name].to(device=device)
        out[name] = _dequant_fp8_block_weight(w, scale_fp)
        dequanted += 1

    return dequanted


def _materialize(model: nn.Module, prefixes: list[str],
                 model_to_shard: dict[str, str],
                 model_to_ckpt: dict[str, str],
                 device: torch.device, dtype: torch.dtype,
                 fp8_scale_inv_map: dict[str, tuple[str, str]] | None = None,
                 ) -> int:
    """Load all tensors whose model-side name starts with any prefix in
    `prefixes` onto `device` as `dtype`. Uses the checkpoint-side key to
    read from safetensors but assigns to the model-side name.

    When `fp8_scale_inv_map` is provided, fp8-sourced weights get their
    128x128 block `weight_scale_inv` applied inline so the installed
    parameter holds the true dequanted weight, not the raw fp8 codes
    cast to bf16. See `_dequant_fp8_block_weight`.

    Returns count of tensors loaded."""
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        if any(model_name.startswith(p) for p in prefixes):
            by_shard[shard].append((model_name, model_to_ckpt[model_name]))
    # Collect loaded tensors first so we can batch the scale-read pass.
    out: dict[str, torch.Tensor] = {}
    for shard, pairs in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for model_name, ckpt_name in pairs:
                t = f.get_tensor(ckpt_name)
                if t.is_floating_point():
                    t = t.to(dtype)
                out[model_name] = t
    if fp8_scale_inv_map:
        _apply_fp8_dequant_inplace(out, fp8_scale_inv_map, device)
    loaded = 0
    for model_name, t in out.items():
        set_module_tensor_to_device(model, model_name, device, value=t)
        loaded += 1
    return loaded


def _read_layer_to_device(prefix: str,
                          model_to_shard: dict[str, str],
                          model_to_ckpt: dict[str, str],
                          dtype: torch.dtype,
                          device: torch.device,
                          fp8_scale_inv_map: dict[str, tuple[str, str]]
                              | None = None,
                          ) -> dict[str, torch.Tensor]:
    """Read all tensors under `prefix` from safetensors and place them
    on `device`. Returns {model_name: device_tensor}.

    When `fp8_scale_inv_map` is provided, every fp8-sourced weight gets
    its 128x128 block `weight_scale_inv` applied inline so `mod.weight`
    holds the TRUE dequanted weight on install. Without this, the
    plain `.to(bfloat16)` cast leaves raw fp8 codes (range ±448)
    interpreted as bf16, which silently breaks every downstream pass
    that reads `mod.weight` — Fisher probe, RTN cost measurement, and
    non-FP8_SOURCE export paths all treated codes as weights.

    On a UMA system like Spark's GB10 the `.to(device)` call here is
    still a torch-level memcpy (CPU ↔ CUDA allocators are logically
    distinct even when the underlying memory is one pool), but we pay
    it in the prefetch worker thread — overlapped with GPU compute —
    rather than on the main thread's critical path. The resulting
    cache holds device-resident tensors, so `_fast_install`'s hot path
    does a pure `.data =` swap with zero copies."""
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        if model_name.startswith(prefix):
            by_shard[shard].append((model_name, model_to_ckpt[model_name]))
    out: dict[str, torch.Tensor] = {}
    for shard, pairs in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for model_name, ckpt_name in pairs:
                t = f.get_tensor(ckpt_name)
                if t.is_floating_point():
                    t = t.to(dtype)
                t = t.to(device, non_blocking=True)
                # Safetensors tensors are usually already contiguous; only
                # pay for a second cuda allocation when they aren't.
                if not t.is_contiguous():
                    t = t.contiguous()
                out[model_name] = t
    if fp8_scale_inv_map:
        _apply_fp8_dequant_inplace(out, fp8_scale_inv_map, device)
    return out


# Back-compat alias. Historically callers treated this as a "CPU cache"
# reader, but the implementation has always returned tensors resident on
# the requested execution device.
_read_layer_to_cpu = _read_layer_to_device


def _install_cached_tensors(model: nn.Module,
                            cached_tensors: dict[str, torch.Tensor],
                            device: torch.device):
    """Install cached layer tensors into the model on `device`."""
    for model_name, t in cached_tensors.items():
        set_module_tensor_to_device(model, model_name, device, value=t)


def _build_install_resolver(model: nn.Module,
                            layer_qname: str) -> dict[str, tuple]:
    """Pre-compute `(parent_module, attr, is_buffer)` for every
    Parameter / buffer under `layer_qname`. Letting us bypass
    accelerate's `set_module_tensor_to_device` at install time — 10×
    fewer Python frames per tensor, 90 s saved per phase at batch=32.

    The resolver maps full dotted names (e.g.
    `model.layers.3.linear_attn.in_proj_qkv.weight`) to the direct
    `nn.Module` + attribute that owns the storage slot."""
    layer_mod = model.get_submodule(layer_qname)
    resolver: dict[str, tuple] = {}
    for sub_name, param in layer_mod.named_parameters():
        full = f"{layer_qname}.{sub_name}"
        if "." in sub_name:
            parent_path, attr = sub_name.rsplit(".", 1)
            parent = layer_mod.get_submodule(parent_path)
        else:
            parent, attr = layer_mod, sub_name
        resolver[full] = (parent, attr, False)
    for sub_name, buf in layer_mod.named_buffers():
        # Skip non-persistent buffers (rotary inv_freq caches, attention
        # masks) — those aren't in our safetensors weight map anyway.
        if "." in sub_name:
            parent_path, attr = sub_name.rsplit(".", 1)
            parent = layer_mod.get_submodule(parent_path)
        else:
            parent, attr = layer_mod, sub_name
        if attr in getattr(parent, "_non_persistent_buffers_set", set()):
            continue
        full = f"{layer_qname}.{sub_name}"
        resolver[full] = (parent, attr, True)
    return resolver


def _fast_install(resolver: dict[str, tuple],
                  cached_tensors: dict[str, torch.Tensor],
                  device: torch.device,
                  model: nn.Module | None = None):
    """Direct install via `resolver` (built by `_build_install_resolver`).
    Swaps `Parameter.data` in place when the existing storage matches
    shape and isn't meta — otherwise allocates a fresh Parameter. On
    unified-memory systems `t.to(device)` for a same-device tensor is
    essentially a pointer rebind, so the hot path is a single attribute
    write per tensor."""
    import torch.nn as _nn
    for model_name, t in cached_tensors.items():
        slot = resolver.get(model_name)
        if slot is None:
            # Unknown key — fall back to the safe-but-slow path. Shouldn't
            # happen in practice; if we see it, the resolver-build logic
            # missed a branch of the module tree.
            if model is not None:
                set_module_tensor_to_device(model, model_name, device, value=t)
            continue
        parent, attr, is_buffer = slot
        target = t if t.device == device else t.to(device, non_blocking=True)
        if is_buffer:
            parent._buffers[attr] = target
            continue
        existing = parent._parameters.get(attr)
        if (existing is not None
                and not existing.is_meta
                and existing.shape == target.shape
                and existing.dtype == target.dtype):
            existing.data = target
        else:
            parent._parameters[attr] = _nn.Parameter(
                target, requires_grad=False)


def _unload(model: nn.Module, prefixes: list[str]) -> int:
    """Move all params/buffers under `prefixes` back to meta."""
    n = 0
    for name, _ in list(model.named_parameters()):
        if any(name.startswith(p) for p in prefixes):
            set_module_tensor_to_device(model, name, "meta")
            n += 1
    for name, _ in list(model.named_buffers()):
        if any(name.startswith(p) for p in prefixes):
            set_module_tensor_to_device(model, name, "meta")
            n += 1
    return n


class LayerCache:
    """LRU cache of decoded layer tensors keyed by layer index.

    Values are dicts `{model_name: tensor}` returned by the layer-read
    helper. In the current streaming path those tensors live on the
    execution device, not on a detached CPU-only cache. Cache size is
    bounded by bytes, not entries, so the same path degenerates to
    "keep everything resident" when enough memory is available.
    Eviction is LRU, which matches the forward-then-reverse access
    pattern used by the streaming probe.
    """

    def __init__(self, max_bytes: int):
        from collections import OrderedDict as _OD
        self._cache: "_OD[int, dict[str, torch.Tensor]]" = _OD()
        self._bytes: dict[int, int] = {}
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0

    def _sizeof(self, tensors: dict[str, torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors.values())

    def _residency(self, tensors: dict[str, torch.Tensor]) -> str:
        devices = {str(t.device) for t in tensors.values()}
        if not devices:
            return "empty"
        if len(devices) == 1:
            return next(iter(devices))
        return "mixed"

    def get(self, layer_idx: int):
        if layer_idx in self._cache:
            self._cache.move_to_end(layer_idx)
            self.hits += 1
            return self._cache[layer_idx]
        self.misses += 1
        return None

    def peek(self, layer_idx: int) -> bool:
        """Non-LRU-touching existence check — used by the prefetch
        scheduler so checking doesn't reshuffle eviction order."""
        return layer_idx in self._cache

    def put(self, layer_idx: int, tensors: dict[str, torch.Tensor]):
        if layer_idx in self._cache:
            return
        size = self._sizeof(tensors)
        evicted = False
        while (self.total_bytes + size > self.max_bytes
               and len(self._cache) > 0):
            evict_idx, _ = self._cache.popitem(last=False)  # evict LRU
            self.total_bytes -= self._bytes.pop(evict_idx, 0)
            evicted = True
        self._cache[layer_idx] = tensors
        self._bytes[layer_idx] = size
        self.total_bytes += size
        # On UMA the cuda caching allocator won't return freed blocks to
        # the OS on its own, so every eviction would otherwise leak into
        # the shared LPDDR5X pool. Force a release after each eviction.
        if evicted and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        self._cache.clear()
        self._bytes.clear()
        self.total_bytes = 0

    def residency_summary(self) -> str:
        counts: dict[str, int] = defaultdict(int)
        for tensors in self._cache.values():
            counts[self._residency(tensors)] += 1
        if not counts:
            return "empty"
        return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))

    def summary(self) -> str:
        tot = self.hits + self.misses
        return (f"LayerCache: {len(self._cache)} layers, "
                f"{self.total_bytes / (1024**3):.1f} GB / "
                f"{self.max_bytes / (1024**3):.1f} GB, "
                f"residency={self.residency_summary()} "
                f"hits={self.hits} misses={self.misses} "
                f"hit_rate={(self.hits/tot*100 if tot else 0):.0f}%")


def _get_layer_list(model: nn.Module):
    """Return (base_model, layer_list) for a causal-LM model. Walks
    past any `ForConditionalGeneration` / `ForCausalLM` wrapper to
    find the decoder layers."""
    # Typical layouts:
    #   model.model.layers                              — text-only CausalLM
    #   model.language_model.model.layers               — pre-v5 multimodal
    #   model.model.language_model.layers               — v5 multimodal umbrella
    #                                                     (Qwen3_5MoeForConditionalGeneration)
    cand = getattr(model, "model", None)
    if cand is not None and hasattr(cand, "layers"):
        return cand, cand.layers
    # v5 multimodal: model.model wraps .visual + .language_model
    if cand is not None:
        lm = getattr(cand, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm, lm.layers
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", lm)
        if hasattr(inner, "layers"):
            return inner, inner.layers
    raise RuntimeError("could not locate model.layers in model tree")


def _get_rotary(base_model: nn.Module) -> nn.Module | None:
    """Find the rotary embedding module so we can compute
    position_embeddings once per sample."""
    for attr in ("rotary_emb", "rope", "rotary_embedding"):
        r = getattr(base_model, attr, None)
        if r is not None:
            return r
    return None


def _embed_prefix(base_model: nn.Module, full_path: str) -> str:
    """Return the full-dotted prefix to the embed_tokens param."""
    return f"{full_path}.embed_tokens." if full_path else "embed_tokens."


def _call_layer(layer: nn.Module, hidden: torch.Tensor, *,
                position_embeddings, attention_mask, position_ids,
                past_key_values=None) -> torch.Tensor:
    """Call a decoder layer with the common transformers v5 signature.
    Returns hidden output tensor."""
    out = layer(
        hidden_states=hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def _compute_position_embeddings(base_model: nn.Module,
                                 hidden: torch.Tensor,
                                 position_ids: torch.Tensor):
    """Call the rotary module to get (cos, sin). Returns None if
    the model doesn't expose a standalone rotary (unusual)."""
    rotary = _get_rotary(base_model)
    if rotary is None:
        return None
    with torch.no_grad():
        cos, sin = rotary(hidden, position_ids)
    return (cos, sin)


def _make_causal_mask(seqlen: int, device: torch.device, dtype: torch.dtype):
    """Build an additive causal mask [1, 1, T, T]. Standard
    upper-triangle -inf convention."""
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _resolve_base_prefix(root: nn.Module, base: nn.Module) -> str:
    """Return the dotted name of `base` within `root`, or '' if it is root."""
    for name, mod in root.named_modules():
        if mod is base:
            return name
    return ""


def _head_prefixes(root: nn.Module, base_prefix: str) -> list[str]:
    """Prefixes for the always-resident pieces: embed + norm + lm_head +
    any rotary/position buffers under the base model."""
    p = f"{base_prefix}." if base_prefix else ""
    prefixes = [
        f"{p}embed_tokens.",
        f"{p}norm.",
        "lm_head.",
        f"{p}rotary_emb.",
    ]
    # Some models put per-layer embeddings inputs (layer_scalar on Gemma 4,
    # per_layer_embeddings on multimodal umbrellas) at top level too —
    # not relevant to causal LM text-only path; skip.
    return prefixes
