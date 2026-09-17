"""Principle 14's serve-side leg on the compressed-tensors lane (#631).

The producer leg holds on this lane already: the exporter's own scheme table
implies the lane spec's ``served_activation_quantization.executes`` set, and
``export_native_compressed.require_compressed_executes_derived_from_scheme``
refuses drift before any render.  That leg is about the LANE.  This module is
the other leg, and it is about one ARTIFACT on one box: what did a live vLLM
actually resolve and dispatch for the modules this checkpoint prices?

Why a sweep and not telemetry
=============================

Stock vLLM emits nothing like Tessera's ``TESSERA_ROUTE_TRACE``: there is no
``emit_route``, no counter file, no packaged ``runtime_contract.json``.  So the
served side has to be READ off the running engine rather than received from
it.  vLLM publishes one supported way to do that -- ``LLM.apply_model(func)``,
which runs ``func(model)`` on every worker and returns one result per rank --
and this module is a function written for that hook.  Nothing here is vendored
or monkeypatched: the values recorded are vLLM's own attributes on vLLM's own
objects, and they cross into the gate as a JSON file.

What is recorded, and why each field
====================================

Per module that the runtime gave a quantization method:

``quant_method`` / ``scheme``
    The classes the compressed-tensors dispatcher resolved from the
    CHECKPOINT'S OWN ``config_groups``.  This is the comparison's substance:
    the price is the ``input_activations`` we wrote, and the scheme is what
    vLLM built from it.
``scheme_attrs``
    The scheme object's own public attributes (``use_a16``, ``group_size``,
    ``strategy`` ...).  The activation descriptor is read from these, so the
    gate reads observed numbers rather than inferring them from a class name.
``kernel``
    The kernel object a scheme selected, where it holds one (NVFP4 does:
    ``FlashInferCutlassNvFp4LinearKernel``).  RECORDED, NOT JUDGED -- vLLM
    publishes no machine-readable table of which kernel is native on which
    target, so a gate that refused on a kernel name would be asserting exactly
    what principle 14 forbids.  It is in the record so the 2026-08-17 shape (a
    body riding an older-architecture schedule) is visible at all, and judging
    it needs the attested table this lane does not have yet.
``quantize_method_base``
    ``isinstance(quant_method, QuantizeMethodBase)`` -- the predicate vLLM's
    OWN post-load sweep uses to decide which modules to finalize
    (``model_loader/utils.process_weights_after_loading``).  That sweep is
    nominal, not structural: a method that implements the whole surface but is
    not a subclass is skipped in silence and dies on the first forward.  A
    sweep of ours that filtered on the same predicate would inherit the same
    blindness and under-report, so this sweep filters on NOTHING and records
    the predicate as a field instead.  A false here is a refusal in the gate,
    not an absence in the record.
``dispatches``
    A forward-hook count taken across a real generate.  A module can be priced,
    resolved and still never run; a zero here says so.  Forward hooks fire in
    eager execution, so a sweep whose load was not ``enforce_eager`` is NOT
    VERIFIED rather than trusted.

``packed_modules_mapping`` is read off the live model class because the
checkpoint prices unfused leaves (``q_proj``, ``gate_proj``) that the runtime
never instantiates: the fused parent is the served module.  Taking the mapping
from the runtime keeps the reconciliation an observation instead of a second
roster to maintain here.

This module imports torch and vLLM ONLY inside its functions; the gate that
reads its output (:mod:`prismaquant.compressed_route_sweep_gate`) imports
neither, so publication replays the comparison without a serving runtime.
"""
from __future__ import annotations

import json
from typing import Any

#: The record schema this module writes and
#: :mod:`prismaquant.compressed_route_sweep_gate` reads.
SWEEP_SCHEMA = "prismaquant.compressed_route_sweep/1"

#: The attribute the forward-hook counter keeps its tally under.
_COUNTER_ATTR = "_pq_route_sweep_dispatches"
_HOOK_ATTR = "_pq_route_sweep_hook"


def _safe_repr(value: Any, limit: int = 120) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # a __repr__ that raises must not kill the sweep
        text = f"<unreprable {type(value).__name__}: {type(exc).__name__}>"
    return text[:limit]


def _class_name(obj: Any) -> str | None:
    return None if obj is None else type(obj).__qualname__


def _class_module(obj: Any) -> str | None:
    return None if obj is None else type(obj).__module__


def _is_quantize_method_base(quant_method: Any) -> bool | None:
    """vLLM's own post-load-sweep predicate, recorded rather than applied.

    ``None`` when the class cannot be imported from the running vLLM -- an
    unknown answer, which the gate reads as NOT VERIFIED and never as true.
    """
    if quant_method is None:
        return None
    try:
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
    except Exception:
        return None
    return isinstance(quant_method, QuantizeMethodBase)


def install_route_sweep_hooks(model) -> int:
    """Attach a per-module forward counter. Returns the number attached.

    Attached to EVERY module that carries a ``quant_method``, including one
    whose method is not a ``QuantizeMethodBase`` subclass, because the point is
    to see the module the runtime's own sweep would have skipped.
    """
    attached = 0
    for _name, mod in model.named_modules():
        if getattr(mod, "quant_method", None) is None:
            continue
        if getattr(mod, _HOOK_ATTR, None) is not None:
            continue
        setattr(mod, _COUNTER_ATTR, 0)

        def _count(module, _args, _output):
            setattr(module, _COUNTER_ATTR,
                    int(getattr(module, _COUNTER_ATTR, 0)) + 1)

        handle = mod.register_forward_hook(_count)
        setattr(mod, _HOOK_ATTR, handle)
        attached += 1
    return attached


def collect_route_sweep(model) -> dict:
    """Read one rank's resolved route state off the live model.

    Written for ``vllm.LLM.apply_model``; the return value is plain JSON data
    so it survives the worker boundary at TP>1.
    """
    import torch

    rank, world_size, rank_source = _rank_identity()
    modules: list[dict] = []
    walked = 0
    for name, mod in model.named_modules():
        walked += 1
        quant_method = getattr(mod, "quant_method", None)
        scheme = getattr(mod, "scheme", None)
        if quant_method is None and scheme is None:
            continue
        attrs: dict[str, str] = {}
        if scheme is not None:
            for key, value in list(vars(scheme).items()):
                if key.startswith("_"):
                    continue
                attrs[key] = _safe_repr(value, 80)
        kernel = getattr(scheme, "kernel", None) if scheme is not None else None
        hooked = hasattr(mod, _HOOK_ATTR)
        modules.append({
            "name": name,
            "module_class": type(mod).__qualname__,
            "quant_method": _class_name(quant_method),
            "quant_method_module": _class_module(quant_method),
            "quantize_method_base": _is_quantize_method_base(quant_method),
            "scheme": _class_name(scheme),
            "scheme_module": _class_module(scheme),
            "scheme_attrs": attrs,
            "kernel": _class_name(kernel),
            "kernel_module": _class_module(kernel),
            "dispatches": (int(getattr(mod, _COUNTER_ATTR, 0)) if hooked
                           else None),
        })
    return {
        "schema": SWEEP_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "rank_source": rank_source,
        "model_class": type(model).__qualname__,
        "packed_modules_mapping": _packed_modules_mapping(model),
        "modules_walked": walked,
        "modules": modules,
        "torch_version": str(getattr(torch, "__version__", "")),
        "vllm_version": _vllm_version(),
        "device": _device_name(),
    }


def _packed_modules_mapping(model) -> dict[str, list[str]] | None:
    mapping = getattr(type(model), "packed_modules_mapping", None)
    if mapping is None:
        mapping = getattr(model, "packed_modules_mapping", None)
    if not isinstance(mapping, dict):
        return None
    out: dict[str, list[str]] = {}
    for parent, leaves in mapping.items():
        if isinstance(leaves, (list, tuple)):
            out[str(parent)] = [str(leaf) for leaf in leaves]
    return out


def _rank_identity() -> tuple[int | None, int | None, str | None]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size(), "torch.distributed"
    except Exception:
        pass
    return None, None, None


def _vllm_version() -> str | None:
    try:
        import vllm

        return str(getattr(vllm, "__version__", "")) or None
    except Exception:
        return None


def _device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} sm_{props.major}{props.minor}"
    except Exception:
        pass
    return None


def sweep_via_apply_model(llm, *, load: dict, prompt_ran: bool) -> dict:
    """Assemble the full sweep record from a loaded ``vllm.LLM``.

    ``LLM.apply_model`` is vLLM's own supported hook and returns one result per
    worker, so this is also the TP>1 shape.  A build without it yields a record
    that names the reason, which the gate reads as NOT VERIFIED.
    """
    apply_model = getattr(llm, "apply_model", None)
    if apply_model is None:
        return {
            "schema": SWEEP_SCHEMA,
            "load": dict(load),
            "forward_observed": bool(prompt_ran),
            "ranks": [],
            "unavailable": (
                "this vLLM build has no LLM.apply_model, so the served route "
                "state could not be read from the engine"),
        }
    ranks = apply_model(collect_route_sweep)
    return {
        "schema": SWEEP_SCHEMA,
        "load": dict(load),
        "forward_observed": bool(prompt_ran),
        "ranks": list(ranks),
    }


def write_sweep(path: str, record: dict) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=False)
        handle.write("\n")
    return path
