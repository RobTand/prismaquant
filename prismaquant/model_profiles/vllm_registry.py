"""vLLM model-registry introspection.

vLLM already encodes most of the architecture-specific knowledge
PrismaQuant needs — fused-sibling structure (`packed_modules_mapping`)
and HF→vLLM name remaps (`hf_to_vllm_mapper`) — as class attributes
on its model classes. A PrismaQuant profile that points at the right
vLLM class therefore gets correct `fused_sibling_group()` and
`to_vllm_internal_name()` for free, with no hand-coded patterns to
drift when vLLM evolves.

This module is the adapter. `vllm_class_for_architecture(arch_name)`
looks up the class, and the helpers below convert its class-level
metadata into PrismaQuant's expected return shapes.

If the vLLM package is absent, profiles use their declarative structure specs.
An installed registry that fails to load a declared model propagates its error.
"""
from __future__ import annotations

import re
from typing import Callable


def vllm_class_for_architecture(arch_name: str):
    """Resolve declared registry metadata; failed installed readers propagate.

    A missing vLLM package or an explicitly unregistered architecture is an
    absence. A registered model that cannot load is an environment failure.
    """
    try:
        from vllm.model_executor.models import registry
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            return None
        raise

    def model_class(value):
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, type):
            return value
        for attribute in ("model_cls", "cls"):
            resolved = getattr(value, attribute, None)
            if isinstance(resolved, type):
                return resolved
        return None

    declared = None
    reg = getattr(registry, "ModelRegistry", None)
    if reg is not None:
        supported = getattr(reg, "get_supported_archs", None)
        if callable(supported):
            declared = arch_name in supported()
            if not declared:
                return None
        # Registered entries expose metadata loading without the runtime's
        # ModelConfig-dependent architecture selection in newer vLLM.
        entries = getattr(reg, "models", None)
        if isinstance(entries, dict) and arch_name in entries:
            load = getattr(entries[arch_name], "load_model_cls", None)
            if callable(load):
                resolved = model_class(load())
                if resolved is None:
                    raise RuntimeError(
                        f"registered vLLM architecture {arch_name!r} loaded no model class")
                return resolved
        for method, argument in (("resolve_model_cls", [arch_name]),
                                 ("inspect_model_cls", arch_name),
                                 ("load_model_cls", arch_name)):
            resolve = getattr(reg, method, None)
            if callable(resolve):
                resolved = model_class(resolve(argument))
                if resolved is not None:
                    return resolved

    import importlib
    for table_name in ("_TEXT_GENERATION_MODELS", "_MULTIMODAL_MODELS",
                       "_SPECULATIVE_DECODING_MODELS", "_VLLM_MODELS",
                       "_SUPPORTED_MODELS"):
        table = getattr(registry, table_name, None)
        if not isinstance(table, dict) or arch_name not in table:
            continue
        declared = True
        entry = table[arch_name]
        if isinstance(entry, tuple) and len(entry) == 2:
            module_path, class_name = entry
        else:
            module_path = (getattr(entry, "module_name", None)
                           or getattr(entry, "mod_name", None))
            class_name = (getattr(entry, "class_name", None)
                          or getattr(entry, "cls_name", None))
        if module_path and class_name:
            module = importlib.import_module(f"vllm.model_executor.models.{module_path}")
            return getattr(module, class_name)
    if declared:
        raise RuntimeError(f"registered vLLM architecture {arch_name!r} resolved no model class")
    return None

def packed_modules_mapping_from_class(cls) -> dict[str, list[str]]:
    """Return `{fused: [siblings, ...]}` from a vLLM model class's
    `packed_modules_mapping` attribute. Falls back to {} if missing."""
    if cls is None:
        return {}
    pm = getattr(cls, "packed_modules_mapping", None)
    if not isinstance(pm, dict):
        return {}
    out: dict[str, list[str]] = {}
    for fused, siblings in pm.items():
        if isinstance(siblings, list) and siblings:
            out[fused] = list(siblings)
    return out


def hf_to_vllm_prefix_map_from_class(cls) -> dict[str, str]:
    """Return `{hf_prefix: vllm_prefix}` from a vLLM model class's
    `hf_to_vllm_mapper` attribute. Falls back to {} if missing or if
    the mapper doesn't use prefix-substitution (we skip regex/substr
    mappers because they need full regex engine support)."""
    if cls is None:
        return {}
    mapper = getattr(cls, "hf_to_vllm_mapper", None)
    if mapper is None:
        return {}
    prefix = getattr(mapper, "orig_to_new_prefix", None)
    if not isinstance(prefix, dict):
        return {}
    return dict(prefix)


def fused_sibling_matcher_from_packed_mapping(
        packed_mapping: dict[str, list[str]],
) -> Callable[[str], str | None]:
    """Given `packed_modules_mapping` (the vLLM class attribute),
    return a function `fn(linear_qname) -> canonical_group_key | None`
    suitable for use as `ModelProfile.fused_sibling_group()`.

    The canonical group key collapses all siblings that would be
    fused in vLLM to the same string, so the allocator treats them as
    a single promotion bucket. Example: for Qwen3.5
    `packed_modules_mapping['qkv_proj'] = ['q_proj','k_proj','v_proj']`,
    any of `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`
    map to `self_attn.qkv_proj`.

    Sibling list is compiled into a single regex with the parent
    captured so we can substitute the fused name and return a stable
    group key. Patterns are evaluated in registration order so a more
    specific match (e.g. `in_proj_qkv` before `in_proj_q`) wins.
    """
    # Sort siblings by descending length so longer matches bind first
    # (prevents `in_proj_b` accidentally matching inside `in_proj_ba`
    # if both were registered without anchors — vLLM's conventions
    # don't do this today, but defense in depth is cheap).
    compiled: list[tuple[re.Pattern, str]] = []
    for fused, siblings in packed_mapping.items():
        siblings_sorted = sorted(siblings, key=len, reverse=True)
        alt = "|".join(re.escape(s) for s in siblings_sorted)
        # `(.*\.)` captures the parent module qname up to the Linear.
        # `(?:{alt})$` matches one of the sibling leaf names.
        pat = re.compile(rf"^(.*\.)?(?:{alt})$")
        compiled.append((pat, fused))

    def match(qname: str) -> str | None:
        for pat, fused in compiled:
            m = pat.match(qname)
            if m:
                parent = m.group(1) or ""
                return f"{parent}{fused}"
        return None

    return match


def name_remapper_from_prefix_map(
        prefix_map: dict[str, str],
) -> Callable[[str], str]:
    """Given an HF→vLLM prefix map (vLLM's `WeightsMapper.orig_to_new_prefix`),
    return a function that applies the first matching prefix rewrite.

    Longest-prefix-wins so `model.language_model.` rebinds before
    `model.` would."""
    items = sorted(prefix_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    def remap(name: str) -> str:
        for src, dst in items:
            if name.startswith(src):
                return dst + name[len(src):]
        return name

    return remap
