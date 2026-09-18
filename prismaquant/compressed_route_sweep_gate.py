"""Principle 14's serve-side leg on the compressed-tensors lane (#631).

The Tessera lane closed this leg with the plugin's own ``route.trace``
telemetry (#575, #630).  Stock vLLM publishes no such counters and no
``runtime_contract.json``, so this lane's served side is READ off the running
engine instead of received from it: ``prismaquant.compressed_route_sweep``
runs inside the smoke's own vLLM process through ``LLM.apply_model`` and
writes a ``prismaquant.compressed_route_sweep/1`` file per serve.  This module
is the consumer, and it imports neither torch nor vLLM -- publication replays
the comparison from the carried bytes, on a box with no serving runtime.

What is compared
================

The priced side is the ARTIFACT'S OWN ``config.json``.  That is not an
incidental choice: vLLM's compressed-tensors dispatcher builds each module's
scheme from ``quantization_config.config_groups[*]``, so the checkpoint we
wrote is literally the input the runtime resolved against.  A disagreement
therefore means the runtime did not build what our price implies -- the module
was ignored, fell back, or resolved elsewhere on this platform -- which is the
one thing no producer-side derivation can establish.

The served side is, per module the runtime gave a quantization method:

* the resolved ``quant_method`` and ``scheme`` classes,
* the scheme object's own attributes (``use_a16``, ``group_size``,
  ``is_static_input_scheme`` ...), which is where the served activation
  descriptor's numbers come from,
* ``quantize_method_base`` -- ``isinstance(quant_method, QuantizeMethodBase)``,
  the predicate vLLM's own post-load sweep filters on.  That sweep is nominal,
  not structural: a method that is not a subclass is finalized by nothing and
  dies on the first forward.  A ``false`` here is a REFUSAL, because a sweep
  that merely filtered it out would have reported a shorter, cleaner and wrong
  histogram,
* ``dispatches`` -- a forward-hook count over a real generate.  Zero on a
  priced module is a refusal: priced, resolved, never run.

Every priced ``config_groups`` target must be accounted for, and unfused
leaves (``q_proj``, ``gate_proj``) are reconciled through the
``packed_modules_mapping`` the sweep READ OFF THE LIVE MODEL CLASS, not
through a roster kept here.

The residual assertion, named
=============================

vLLM publishes no machine-readable contract table, so something has to say
what ``CompressedTensorsW4A4Fp4`` means.  :data:`SCHEME_ACTIVATION` is that
something and it is the residual assertion in this gate.  It is kept as thin
as it can be: the table says WHICH OBSERVED ATTRIBUTES to read, and the
numbers themselves come from the scheme object (``use_a16``, ``group_size``).
A scheme class the table does not name is NOT VERIFIED -- never a pass and
never a guess -- and the refusal prints the class and its module so the next
sweep can add it.  The roster was read from the pinned image
(``vllm/vllm-openai@sha256:61fc8a89...``, vLLM 0.28.0), and only
``CompressedTensorsW4A4Fp4`` has been observed end to end on a real serve.

What this leg does NOT see
==========================

* **The activation representation.** A scheme class is a name, not the
  quantizer rule the kernel applied, so #567's shape (right route, wrong
  representation) still passes.  Same hole the Tessera trace has.
* **Kernel correctness.** The record carries the kernel class each scheme
  selected (``FlashInferCutlassNvFp4LinearKernel``), because the 2026-08-17
  incident was a body riding an older-architecture schedule.  It is RECORDED
  AND NOT JUDGED: vLLM publishes no table of which kernel is native on which
  target, and a gate that refused on a kernel name would assert precisely what
  principle 14 forbids.  The verdict prints the kernel histogram so the
  question is visible; answering it needs an attested table this lane does not
  have.
* **MoE method classes.** No small packed-MoE compressed-tensors artifact was
  available to sweep, so no MoE method class is in the table and an MoE
  artifact reads NOT VERIFIED until one is observed.  That is the recorded
  gap, deliberately, rather than a table written from reading source.
* **A compiled or graph-captured forward.** Forward hooks do not run under
  CUDA-graph replay, so a sweep whose load was not ``enforce_eager`` is NOT
  VERIFIED rather than trusted with zeros.

Three outcomes, never two.  ``agree`` is every module accounted for and every
contract equal.  ``refused`` is a conflict between the observation and the
price, or within the observation itself.  ``not_verified`` is no qualifying
observation.  Only ``agree`` closes the shipcard slot.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

SWEEP_SCHEMA = "prismaquant.compressed_route_sweep/1"
VERDICT_SCHEMA = "prismaquant.compressed_route_sweep_verdict/1"

AGREE = "agree"
REFUSED = "refused"
NOT_VERIFIED = "not_verified"

GRANULARITY = (
    "resolved quantization method, scheme, scheme attributes and forward "
    "dispatch count per served module, per rank; unfused leaves reconciled "
    "through the runtime's own packed_modules_mapping"
)


class CompressedRouteSweepError(RuntimeError):
    """The observation conflicts with the price, its header, or itself."""


class RouteSweepNotVerified(RuntimeError):
    """No qualifying observation. Never a pass, never a quiet downgrade."""


# ---------------------------------------------------------------------------
# The residual assertion: scheme class -> which observed attributes to read.
# ---------------------------------------------------------------------------
def _as_bool(text: Any) -> bool | None:
    if isinstance(text, bool):
        return text
    if text is None:
        return None
    lowered = str(text).strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0"):
        return False
    return None


def _as_int(text: Any) -> int | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return int(text)
    if text is None:
        return None
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def _descriptor(quantized: bool, *, num_bits: int | None = None,
                kind: str | None = None, group_size: int | None = None,
                dynamic: bool | None = None) -> dict:
    return {"quantized": bool(quantized), "num_bits": num_bits,
            "type": kind, "group_size": group_size, "dynamic": dynamic}


def _nvfp4(attrs: Mapping[str, Any]) -> dict | None:
    use_a16 = _as_bool(attrs.get("use_a16"))
    if use_a16 is None:
        return None
    if use_a16:
        return _descriptor(False)
    group = _as_int(attrs.get("group_size"))
    if group is None:
        return None
    return _descriptor(True, num_bits=4, kind="float", group_size=group,
                       dynamic=True)


def _mxfp4(attrs: Mapping[str, Any]) -> dict | None:
    group = _as_int(attrs.get("group_size"))
    if group is None:
        return None
    return _descriptor(True, num_bits=4, kind="float", group_size=group,
                       dynamic=True)


def _mxfp8(attrs: Mapping[str, Any]) -> dict | None:
    # The scheme carries only its kernel; MXFP8 is E4M3 over blocks of 32 by
    # the format's definition, and the checkpoint prices group_size 32.
    return _descriptor(True, num_bits=8, kind="float", group_size=32,
                       dynamic=True)


def _fp8_a8(attrs: Mapping[str, Any]) -> dict | None:
    static = _as_bool(attrs.get("is_static_input_scheme"))
    if static is None:
        return None
    return _descriptor(True, num_bits=8, kind="float", group_size=None,
                       dynamic=not static)


def _int8_a8(attrs: Mapping[str, Any]) -> dict | None:
    static = _as_bool(attrs.get("is_static_input_scheme"))
    if static is None:
        return None
    return _descriptor(True, num_bits=8, kind="int", group_size=None,
                       dynamic=not static)


def _weight_only(attrs: Mapping[str, Any]) -> dict | None:
    return _descriptor(False)


#: Scheme class -> reader for its OWN attributes.  Adding a class means having
#: swept it; an absent class is NOT VERIFIED and says so by name.
SCHEME_ACTIVATION = {
    "CompressedTensorsW4A4Fp4": _nvfp4,
    "CompressedTensorsW4A4Mxfp4": _mxfp4,
    "CompressedTensorsW8A8Mxfp8": _mxfp8,
    "CompressedTensorsW8A8Fp8": _fp8_a8,
    "CompressedTensorsW8A8Int8": _int8_a8,
    "CompressedTensorsW8A16Fp8": _weight_only,
    "CompressedTensorsWNA16": _weight_only,
}

#: Quantization-method classes that carry NO weight contract from
#: ``config_groups`` and are therefore neither priced nor a refusal.  The KV
#: cache method is the live one: vLLM attaches it to every ``Attention``
#: module, and the checkpoint prices no Linear there.
UNPRICED_METHODS = frozenset({
    "CompressedTensorsKVCacheMethod",
    "UnquantizedEmbeddingMethod",
})

#: Methods that mean "the runtime resolved NO compressed-tensors scheme here".
#: On a priced module that is a refusal, not an absence.
UNQUANTIZED_METHODS = frozenset({
    "UnquantizedLinearMethod",
    "UnquantizedEmbeddingMethod",
    "UnquantizedFusedMoEMethod",
})


# ---------------------------------------------------------------------------
# The priced side: the artifact's own config.json.
# ---------------------------------------------------------------------------
def _quantization_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for holder in (config, config.get("text_config")):
        if isinstance(holder, Mapping):
            qc = holder.get("quantization_config")
            if isinstance(qc, Mapping):
                return qc
    raise CompressedRouteSweepError(
        "the artifact's config.json declares no quantization_config, so it "
        "prices no activation contract to compare a serve against")


def _matcher(target: str):
    """A ``config_groups`` target or ``ignore`` entry, as a predicate."""
    text = str(target)
    if text.startswith("re:"):
        pattern = re.compile(text[3:])
        return lambda name, cls: bool(pattern.match(name))
    return lambda name, cls, t=text: name == t or cls == t


def priced_activation(group: Mapping[str, Any]) -> dict:
    """The activation contract a ``config_groups`` entry prices."""
    args = group.get("input_activations")
    if not isinstance(args, Mapping):
        return _descriptor(False)
    dynamic = args.get("dynamic")
    if isinstance(dynamic, str):
        is_dynamic = dynamic.strip().lower() not in ("false", "static", "")
    else:
        is_dynamic = bool(dynamic)
    return _descriptor(
        True,
        num_bits=_as_int(args.get("num_bits")),
        kind=None if args.get("type") is None else str(args.get("type")),
        group_size=_as_int(args.get("group_size")),
        dynamic=is_dynamic,
    )


def price_table(config: Mapping[str, Any]) -> tuple[list[dict], list]:
    """``(groups, ignore_matchers)`` from the artifact's config.json."""
    qc = _quantization_config(config)
    raw_groups = qc.get("config_groups")
    if not isinstance(raw_groups, Mapping) or not raw_groups:
        raise CompressedRouteSweepError(
            "quantization_config carries no config_groups; there is no priced "
            "contract for a serve to disagree with")
    groups = []
    for name, group in sorted(raw_groups.items()):
        if not isinstance(group, Mapping):
            raise CompressedRouteSweepError(
                f"config group {name!r} is not an object")
        targets = group.get("targets")
        if not isinstance(targets, Sequence) or isinstance(targets, str):
            raise CompressedRouteSweepError(
                f"config group {name!r} declares no targets list")
        groups.append({
            "name": str(name),
            "targets": [str(t) for t in targets],
            "matchers": [_matcher(t) for t in targets],
            "activation": priced_activation(group),
        })
    ignore = qc.get("ignore") or []
    if not isinstance(ignore, Sequence) or isinstance(ignore, str):
        raise CompressedRouteSweepError("quantization_config ignore is not a list")
    return groups, [(str(t), _matcher(t)) for t in ignore]


# ---------------------------------------------------------------------------
# Reading one rank's sweep.
# ---------------------------------------------------------------------------
def _parse_rank(label: str, text: str | None) -> dict:
    if text is None:
        raise RouteSweepNotVerified(f"{label}: no sweep file")
    stripped = text.strip()
    if not stripped:
        raise RouteSweepNotVerified(f"{label}: sweep file is empty")
    try:
        payload = json.loads(stripped)
    except ValueError as exc:
        raise RouteSweepNotVerified(f"{label}: sweep is not JSON: {exc}") from None
    if not isinstance(payload, Mapping):
        raise RouteSweepNotVerified(f"{label}: sweep is not an object")
    if payload.get("schema") != SWEEP_SCHEMA:
        raise RouteSweepNotVerified(
            f"{label}: schema {payload.get('schema')!r} is not {SWEEP_SCHEMA!r}")
    if payload.get("unavailable"):
        raise RouteSweepNotVerified(f"{label}: {payload['unavailable']}")
    if payload.get("forward_observed") is not True:
        raise RouteSweepNotVerified(
            f"{label}: the smoke generated nothing, so no module was observed "
            "dispatching")
    load = payload.get("load")
    if not isinstance(load, Mapping):
        raise RouteSweepNotVerified(f"{label}: sweep carries no load block")
    if load.get("enforce_eager") is not True:
        raise RouteSweepNotVerified(
            f"{label}: the load was not enforce_eager; forward hooks do not "
            "run under CUDA-graph replay, so the dispatch counts would be "
            "zeros that mean 'not observed'")
    if load.get("speculative_config"):
        raise RouteSweepNotVerified(
            f"{label}: the load carried a speculative config; a draft model "
            "adds modules this artifact does not price")
    ranks = payload.get("ranks")
    if not isinstance(ranks, Sequence) or not ranks:
        raise RouteSweepNotVerified(f"{label}: sweep records no rank")
    return payload


def _rank_body(label: str, body: Mapping[str, Any]) -> dict:
    if not isinstance(body, Mapping):
        raise RouteSweepNotVerified(f"{label}: rank entry is not an object")
    if body.get("rank_source") != "torch.distributed":
        raise RouteSweepNotVerified(
            f"{label}: rank_source is {body.get('rank_source')!r}, not "
            "torch.distributed; the process never joined a group, so which "
            "shard this is was never established")
    if not isinstance(body.get("rank"), int) or not isinstance(
            body.get("world_size"), int):
        raise RouteSweepNotVerified(f"{label}: rank/world_size is not an int")
    mapping = body.get("packed_modules_mapping")
    if mapping is not None and not isinstance(mapping, Mapping):
        raise RouteSweepNotVerified(
            f"{label}: packed_modules_mapping is not an object")
    modules = body.get("modules")
    if not isinstance(modules, Sequence) or not modules:
        raise RouteSweepNotVerified(f"{label}: rank records no module")
    return dict(body)


def _leaf_names(name: str, mapping: Mapping[str, Any] | None) -> list[str]:
    """The checkpoint leaves a served fused module absorbed.

    ``mapping`` is the runtime's own ``packed_modules_mapping``; the leaves are
    derived by replacing the module's last path element.
    """
    if not mapping:
        return []
    head, _, tail = name.rpartition(".")
    leaves = mapping.get(tail)
    if not isinstance(leaves, Sequence) or isinstance(leaves, str):
        return []
    prefix = f"{head}." if head else ""
    return [f"{prefix}{leaf}" for leaf in leaves]


def _match_group(groups, names: Sequence[str], cls: str):
    hits = []
    for group in groups:
        if any(matcher(name, cls) for name in names
               for matcher in group["matchers"]):
            hits.append(group)
    return hits


def _matched_ignore(ignore, names: Sequence[str], cls: str) -> list[str]:
    return [target for target, matcher in ignore
            if any(matcher(name, cls) for name in names)]


def _served_activation(row: Mapping[str, Any]) -> tuple[dict | None, str]:
    """``(descriptor, reason)``; a ``None`` descriptor is NOT VERIFIED."""
    scheme = row.get("scheme")
    if scheme is None:
        method = row.get("quant_method")
        if method in UNQUANTIZED_METHODS:
            return _descriptor(False), "unquantized"
        return None, (
            f"module {row.get('name')!r} resolved quant_method "
            f"{method!r} with no scheme, and this gate has observed no "
            "activation contract for that method class (no packed-MoE "
            "artifact has been swept)")
    reader = SCHEME_ACTIVATION.get(str(scheme))
    if reader is None:
        return None, (
            f"module {row.get('name')!r} resolved scheme {scheme!r} "
            f"({row.get('scheme_module')}), which this gate has never "
            "observed; add it from a sweep, never from reading source")
    attrs = row.get("scheme_attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    descriptor = reader(attrs)
    if descriptor is None:
        return None, (
            f"module {row.get('name')!r} resolved scheme {scheme!r} but its "
            f"own attributes {sorted(attrs)} do not carry the numbers the "
            "activation descriptor is read from")
    return descriptor, "scheme"


def compare_rank(body: Mapping[str, Any], *, groups, ignore) -> dict:
    """One rank's per-module comparison. Raises on a conflict."""
    mapping = body.get("packed_modules_mapping")
    conflicts: list[str] = []
    contracts: dict[str, dict] = {}
    kernels: dict[str, dict[str, int]] = {}
    covered_targets: set[str] = set()
    unpriced: list[str] = []
    for row in body["modules"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise RouteSweepNotVerified("a module entry carries no name")
        name = row["name"]
        cls = str(row.get("module_class") or "")
        method = row.get("quant_method")
        if method is not None and row.get("quantize_method_base") is not True:
            conflicts.append(
                f"{name}: quant_method {method!r} is not a QuantizeMethodBase "
                f"(quantize_method_base={row.get('quantize_method_base')!r}); "
                "vLLM's own post-load sweep finalizes by isinstance, so this "
                "module was skipped in silence and dies on a forward")
            continue
        names = [name, *_leaf_names(name, mapping)]
        hits = _match_group(groups, names, cls)
        ignored = _matched_ignore(ignore, names, cls)
        if len(hits) > 1:
            conflicts.append(
                f"{name}: matches {len(hits)} config groups "
                f"({', '.join(g['name'] for g in hits)}); the price is "
                "ambiguous")
            continue
        served, reason = _served_activation(row)
        if hits and ignored:
            conflicts.append(
                f"{name}: is priced by config group {hits[0]['name']!r} AND "
                f"matched ignore entry {ignored[0]!r}")
            continue
        if ignored:
            if served is not None and served["quantized"]:
                conflicts.append(
                    f"{name}: is on the checkpoint's ignore list but the "
                    f"runtime resolved scheme {row.get('scheme')!r}")
            covered_targets.update(ignored)
            continue
        if not hits:
            if str(method) in UNPRICED_METHODS or (
                    served is not None and not served["quantized"]):
                unpriced.append(name)
                continue
            conflicts.append(
                f"{name}: resolved scheme {row.get('scheme')!r} but matches no "
                "config group and no ignore entry; the serve quantized a "
                "module this artifact never priced")
            continue
        group = hits[0]
        if served is None:
            raise RouteSweepNotVerified(reason)
        priced = group["activation"]
        if served != priced:
            conflicts.append(
                f"{name}: priced {priced} by config group {group['name']!r}, "
                f"served {served} (scheme {row.get('scheme')!r})")
            continue
        dispatches = row.get("dispatches")
        if dispatches is None:
            raise RouteSweepNotVerified(
                f"{name}: no dispatch count was taken, so whether the module "
                "ran was never observed")
        if int(dispatches) <= 0:
            conflicts.append(
                f"{name}: priced by config group {group['name']!r} and "
                "resolved, but dispatched ZERO forwards")
            continue
        contracts[name] = priced
        bucket = kernels.setdefault(_contract_key(priced), {})
        kernel = str(row.get("kernel"))
        bucket[kernel] = bucket.get(kernel, 0) + 1
        covered_targets.update(
            target for target in group["targets"]
            if any(_matcher(target)(candidate, cls) for candidate in names))
    if conflicts:
        raise CompressedRouteSweepError("; ".join(sorted(conflicts)))
    missing = []
    for group in groups:
        for target in group["targets"]:
            if target not in covered_targets:
                missing.append(f"{group['name']}:{target}")
    if missing:
        raise CompressedRouteSweepError(
            f"{len(missing)} priced target(s) reached no served module, so "
            "the serve did not quantize what the artifact prices: "
            + ", ".join(sorted(missing)[:8])
            + ("" if len(missing) <= 8 else " ..."))
    return {
        "rank": body["rank"],
        "world_size": body["world_size"],
        "model_class": body.get("model_class"),
        "contracts": {name: _contract_key(c) for name, c in sorted(contracts.items())},
        "histogram": _histogram(contracts),
        "kernels": {key: dict(sorted(v.items())) for key, v in sorted(kernels.items())},
        "unpriced_modules": sorted(unpriced),
    }


def _contract_key(descriptor: Mapping[str, Any]) -> str:
    if not descriptor.get("quantized"):
        return "A16"
    bits = descriptor.get("num_bits")
    kind = descriptor.get("type")
    group = descriptor.get("group_size")
    dyn = "dynamic" if descriptor.get("dynamic") else "static"
    tail = "" if group is None else f"/g{group}"
    return f"A{bits}:{kind}{tail}:{dyn}"


def _histogram(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for descriptor in contracts.values():
        key = _contract_key(descriptor)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def compare_route_sweeps(
    sweeps: Sequence[tuple[str, str | None]],
    *,
    expected_ranks: int,
    config: Mapping[str, Any],
) -> dict:
    """Compare every rank's served route sweep with the artifact's price.

    Returns a verdict whose ``status`` is :data:`AGREE`, :data:`REFUSED` or
    :data:`NOT_VERIFIED`.  Raises nothing for a disagreement -- the caller
    reads the status -- but the record writer turns each into its own exit.
    """
    if not isinstance(expected_ranks, int) or isinstance(expected_ranks, bool) \
            or expected_ranks < 1:
        raise ValueError("expected_ranks must be a positive int")
    groups, ignore = price_table(config)
    priced_activations = {g["name"]: _contract_key(g["activation"]) for g in groups}
    base = {
        "schema": VERDICT_SCHEMA,
        "granularity": GRANULARITY,
        "expected_ranks": expected_ranks,
        "priced_groups": dict(sorted(priced_activations.items())),
    }
    if len(sweeps) < expected_ranks:
        return dict(base, status=NOT_VERIFIED, detail=(
            f"{len(sweeps)} sweep(s) for a world size of {expected_ranks}; a "
            "rank that reported nothing is not a rank that agreed"))
    per_rank = []
    try:
        for label, text in sweeps:
            payload = _parse_rank(label, text)
            for body in payload["ranks"]:
                per_rank.append((label, _rank_body(label, body)))
    except RouteSweepNotVerified as exc:
        return dict(base, status=NOT_VERIFIED, detail=str(exc))
    seen_ranks = sorted({body["rank"] for _label, body in per_rank})
    if len(per_rank) != expected_ranks or seen_ranks != list(range(expected_ranks)):
        return dict(base, status=NOT_VERIFIED, detail=(
            f"observed ranks {seen_ranks} over {len(per_rank)} record(s) do "
            f"not cover 0..{expected_ranks - 1} exactly once"))
    for label, body in per_rank:
        if body["world_size"] != expected_ranks:
            return dict(base, status=REFUSED, detail=(
                f"{label}: rank {body['rank']} reports world_size "
                f"{body['world_size']}, not {expected_ranks}"))
    results = []
    try:
        for label, body in per_rank:
            results.append(compare_rank(body, groups=groups, ignore=ignore))
    except RouteSweepNotVerified as exc:
        return dict(base, status=NOT_VERIFIED, detail=str(exc))
    except CompressedRouteSweepError as exc:
        return dict(base, status=REFUSED, detail=str(exc))
    first = results[0]
    for other in results[1:]:
        if other["contracts"] != first["contracts"]:
            return dict(base, status=REFUSED, detail=(
                f"rank {other['rank']} served a different per-module contract "
                f"map than rank {first['rank']}; tensor parallelism shards a "
                "module and never splits it, so the maps must be equal"))
    return dict(
        base,
        status=AGREE,
        detail=(
            f"{sum(first['histogram'].values())} priced module(s) on "
            f"{expected_ranks} rank(s) served the contract they were priced "
            f"on: {first['histogram']}"),
        histogram=first["histogram"],
        kernels=first["kernels"],
        unpriced_modules=first["unpriced_modules"],
        model_class=first["model_class"],
        per_rank=[{"rank": r["rank"], "histogram": r["histogram"]} for r in results],
    )
