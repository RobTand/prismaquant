"""Principle 14's serve-side leg on the Tessera lane (RobTand/prismaquant#575).

The producer leg holds already: every lane spec's ``executes`` map is derived
from the pinned runtime's packaged contract and refused on drift
(``tessera_export_lane.require_platform_executes_derived_from_contract``).
This module is the other leg. It compares the activation contracts an artifact
was PRICED on with the contracts a live serve actually DISPATCHED, and refuses
when they differ.

The served side is Tessera's own telemetry: a serve started with
``TESSERA_ROUTE_TRACE=<path>`` writes one ``tessera.route_trace/1`` JSON file
per process, so one per rank. Each entry is one counter keyed by
``(policy, shape, symbol, decoder, contract, kind)`` and carries ``launches``
and ``modules``, where ``modules`` is the number of distinct module prefixes
that dispatched under that key.

Comparison granularity
======================

The trace names no modules. ``modules`` is a count of a set whose members the
trace does not emit, so a per-module comparison (this owner priced on X,
served on Y) is not something the trace can support. What it does support,
exactly, is a **module count per (route family, kind, activation contract),
per token count M, per rank**:

* ``shape`` is ``M{tokens}:N{rows}:K{cols}``. Within one M, entries with
  different N:K are different modules, and a dense forward (and a routed-MoE
  layer's one module) sees every token of that forward, so summing ``modules``
  over the entries of one M counts each served module once.
* Every M the trace recorded must give the same histogram. A module missing
  at one M, or counted twice at one M because it dispatched under two symbol
  or decoder keys, makes the histograms differ and the gate refuses. That is
  strict on purpose: relaxing it needs a measured serve, not an argument.
* Tensor parallelism shards a module across ranks and never splits it, so
  every rank must report the same histogram. Pipeline parallelism places
  different modules on different ranks and is refused by the same rule; the
  lane admits neither today.

The priced side is the artifact's own ``config.json``: each
``quantization_config.config_groups[*]`` target is one module, with a route
``scheme.family``, a ``structure`` (``dense`` or ``routed_moe``) and a
``grid``. The payload family is the pinned contract's ``formats[]`` row whose
``grid`` equals the scheme's, and the priced activation contract is what the
contract's ``lane_eligibility.platforms[<platform>].executes`` says that
payload family executes on the artifact's declared platform. Nothing here is
a local table.

What this does not see
======================

* Which module rode which contract. Two modules that swapped contracts leave
  the histogram unchanged. Closing that needs the trace to emit its module
  prefixes (filed against Tessera).
* The activation REPRESENTATION. ``contract`` is a name, not the quantizer
  rule the kernel applied, so a right name over a wrong representation passes
  (RobTand/prismaquant#567 is that shape).
* Symbol and decoder correctness. They are recorded in the verdict; judging
  them per cell is the scoped ``route.census`` replay's job.
* Compiled forwards. The trace declines to count while torch.compile traces
  (a ``M*`` shape), so a compiled serve yields ``not_verified``, never a pass.
* Fallback and error dispatches. Only served dispatches are counted, so a
  module that fell back is ABSENT from the trace; the count comparison is
  what makes that absence a refusal.

Three outcomes, never two: ``agree``, ``refused`` (the observation disagrees
with the price) and ``not_verified`` (no usable observation: a rank's trace
is missing, unreadable, empty, of another schema, or compiled). A
``not_verified`` verdict never closes the shipcard slot.

Stdlib only, no torch and no ``tessera`` import at module scope: the
shipcard replays this at publication.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

ROUTE_TRACE_SCHEMA = "tessera.route_trace/1"
VERDICT_SCHEMA = "prismaquant.tessera_route_trace_verdict/1"

AGREE = "agree"
REFUSED = "refused"
NOT_VERIFIED = "not_verified"

GRANULARITY = (
    "module count per (route family, kind, activation contract), "
    "per token count M, per rank; the trace names no modules"
)

#: The trace's ``kind`` against the artifact's ``scheme.structure``.
TRACE_KIND_FOR_STRUCTURE = {"dense": "dense", "routed_moe": "moe"}

_SHAPE = re.compile(r"^M(\*|[0-9]+):N([0-9]+):K([0-9]+)$")
_ENTRY_STR_FIELDS = ("policy", "shape", "symbol", "decoder", "contract", "kind")


class TesseraRouteTraceError(ValueError):
    """The observation or the price disagrees, or an input is malformed."""


class RouteTraceNotVerified(TesseraRouteTraceError):
    """No usable observation exists. This is not a pass."""


def _key(family: str, kind: str, contract: str) -> str:
    return f"{family}/{kind}/{contract}"


def parse_route_trace(payload: Any, *, where: str) -> list[dict[str, Any]]:
    """Validate one rank's trace and return its entries in canonical form.

    Raises :class:`RouteTraceNotVerified` when the file cannot serve as an
    observation (wrong schema, no entries, compiled counts) and
    :class:`TesseraRouteTraceError` when it claims to be one and is malformed.
    """
    if isinstance(payload, (bytes, str)):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise RouteTraceNotVerified(
                f"{where}: not readable JSON ({exc}); a partial or truncated "
                "trace is not an observation") from exc
    if not isinstance(payload, Mapping):
        raise RouteTraceNotVerified(f"{where}: trace is not a JSON object")
    if payload.get("schema") != ROUTE_TRACE_SCHEMA:
        raise RouteTraceNotVerified(
            f"{where}: schema {payload.get('schema')!r} is not "
            f"{ROUTE_TRACE_SCHEMA!r}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RouteTraceNotVerified(
            f"{where}: trace records no served dispatch; an empty trace is "
            "not a clean bill")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, entry in enumerate(entries):
        at = f"{where}.entries[{index}]"
        if not isinstance(entry, Mapping):
            raise TesseraRouteTraceError(f"{at}: entry is not an object")
        for field in _ENTRY_STR_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise TesseraRouteTraceError(f"{at}: {field} must be a non-empty string")
        for field in ("launches", "modules"):
            value = entry.get(field)
            if type(value) is not int or value <= 0:
                raise TesseraRouteTraceError(f"{at}: {field} must be a positive integer")
        family, sep, residency = entry["policy"].partition(":")
        if not sep or not family or not residency:
            raise TesseraRouteTraceError(
                f"{at}: policy {entry['policy']!r} is not '<family>:<residency>'")
        match = _SHAPE.match(entry["shape"])
        if match is None:
            raise TesseraRouteTraceError(
                f"{at}: shape {entry['shape']!r} is not 'M<tokens>:N<rows>:K<cols>'")
        if match.group(1) == "*":
            raise RouteTraceNotVerified(
                f"{at}: shape {entry['shape']!r} was written while torch.compile "
                "traced, where one graph serves every M and the count is not a "
                "launch count; serve eager to observe routes")
        if entry["kind"] not in TRACE_KIND_FOR_STRUCTURE.values():
            raise TesseraRouteTraceError(f"{at}: unknown kind {entry['kind']!r}")
        identity = tuple(entry[field] for field in _ENTRY_STR_FIELDS)
        if identity in seen:
            raise TesseraRouteTraceError(f"{at}: duplicate counter key {identity}")
        seen.add(identity)
        parsed.append({
            "family": family,
            "residency": residency,
            "m": int(match.group(1)),
            "n": int(match.group(2)),
            "k": int(match.group(3)),
            "symbol": entry["symbol"],
            "decoder": entry["decoder"],
            "contract": entry["contract"],
            "kind": entry["kind"],
            "launches": entry["launches"],
            "modules": entry["modules"],
        })
    return parsed


def served_histogram(entries: Sequence[Mapping[str, Any]], *, where: str) -> dict[str, Any]:
    """One rank's served module histogram, required equal at every M."""
    by_m: dict[int, Counter] = {}
    for entry in entries:
        by_m.setdefault(entry["m"], Counter())[
            _key(entry["family"], entry["kind"], entry["contract"])] += entry["modules"]
    token_counts = sorted(by_m)
    reference_m = token_counts[0]
    reference = dict(sorted(by_m[reference_m].items()))
    for m in token_counts[1:]:
        other = dict(sorted(by_m[m].items()))
        if other != reference:
            raise TesseraRouteTraceError(
                f"{where}: the served module histogram differs between token "
                f"counts: M={reference_m} {reference} but M={m} {other}; every "
                "forward dispatches every quantized module once, so a module "
                "absent at one M or counted twice at one M is a route change "
                "inside the serve")
    return {
        "histogram": reference,
        "token_counts": token_counts,
        "symbols": sorted({entry["symbol"] for entry in entries}),
        "decoders": sorted({entry["decoder"] for entry in entries}),
        "residencies": sorted({entry["residency"] for entry in entries}),
    }


def payload_family_for_grid(grid: str, formats: Mapping[str, Mapping[str, Any]], *, where: str) -> str:
    """The contract's one ``formats[]`` family whose grid is ``grid``."""
    matches = sorted(
        family for family, row in formats.items()
        if isinstance(row, Mapping) and row.get("grid") == grid)
    if len(matches) != 1:
        raise TesseraRouteTraceError(
            f"{where}: grid {grid!r} matches {matches or 'no'} formats[] "
            "families in the pinned contract; exactly one is required to "
            "derive its activation contract")
    return matches[0]


def priced_histogram(
    config: Mapping[str, Any],
    *,
    platform: str,
    executes_by_platform: Mapping[str, Mapping[str, "str | None"]],
    formats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Count the artifact's Tessera modules per priced activation contract."""
    quant = config.get("quantization_config") if isinstance(config, Mapping) else None
    if not isinstance(quant, Mapping) or quant.get("quant_method") != "tessera":
        raise TesseraRouteTraceError(
            "config.json quantization_config.quant_method is not 'tessera'")
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise TesseraRouteTraceError(
            "config.json declares no Tessera config_groups; an artifact with "
            "nothing priced has nothing to compare")
    if platform not in executes_by_platform:
        raise TesseraRouteTraceError(
            f"the pinned contract publishes no platform {platform!r} "
            f"(published: {sorted(executes_by_platform)})")
    executes = executes_by_platform[platform]
    histogram: Counter = Counter()
    owners: dict[str, str] = {}
    for name, group in sorted(groups.items()):
        where = f"config_groups.{name}"
        if not isinstance(group, Mapping):
            raise TesseraRouteTraceError(f"{where}: group is not an object")
        scheme = group.get("scheme")
        targets = group.get("targets")
        if not isinstance(scheme, Mapping):
            raise TesseraRouteTraceError(f"{where}: scheme is not an object")
        if not isinstance(targets, list) or not targets:
            raise TesseraRouteTraceError(f"{where}: targets must be a non-empty list")
        family = scheme.get("family")
        grid = scheme.get("grid")
        structure = scheme.get("structure", "dense")
        if not isinstance(family, str) or not family or not isinstance(grid, str) or not grid:
            raise TesseraRouteTraceError(f"{where}: scheme needs a family and a grid")
        if structure not in TRACE_KIND_FOR_STRUCTURE:
            raise TesseraRouteTraceError(f"{where}: unsupported structure {structure!r}")
        payload_family = payload_family_for_grid(grid, formats, where=where)
        if payload_family not in executes:
            raise TesseraRouteTraceError(
                f"{where}: platform {platform!r} states nothing for "
                f"{payload_family}; an unstated family is not a backed one")
        contract = executes[payload_family]
        if contract is None:
            raise TesseraRouteTraceError(
                f"{where}: priced {payload_family} ({family}, grid {grid}) but "
                f"the pinned runtime has no native route for it on {platform!r}")
        key = _key(family, TRACE_KIND_FOR_STRUCTURE[structure], contract)
        for owner in targets:
            if not isinstance(owner, str) or not owner or owner in owners:
                raise TesseraRouteTraceError(f"{where}: malformed or duplicate target {owner!r}")
            owners[owner] = key
            histogram[key] += 1
    return {"histogram": dict(sorted(histogram.items())), "owners": dict(sorted(owners.items()))}


def _histogram_difference(priced: Mapping[str, int], served: Mapping[str, int]) -> list[str]:
    lines = []
    for key in sorted(set(priced) | set(served)):
        want, got = priced.get(key, 0), served.get(key, 0)
        if want != got:
            lines.append(f"{key}: priced {want} module(s), served {got}")
    return lines


def compare_route_traces(
    traces: Sequence[tuple[str, Any]],
    *,
    expected_ranks: int,
    config: Mapping[str, Any],
    platform: str,
    executes_by_platform: Mapping[str, Mapping[str, "str | None"]],
    formats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The whole gate. Returns a verdict whose ``status`` is one of three.

    ``traces`` is ``[(rank_label, payload_or_None), ...]`` in rank order; a
    ``None`` payload is a rank whose trace file does not exist. The priced side
    is computed first, so a malformed or unbacked price raises
    :class:`TesseraRouteTraceError` whatever the traces say.
    """
    if type(expected_ranks) is not int or expected_ranks < 1:
        raise TesseraRouteTraceError("expected_ranks must be a positive integer")
    priced = priced_histogram(
        config, platform=platform,
        executes_by_platform=executes_by_platform, formats=formats)
    verdict: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "granularity": GRANULARITY,
        "platform": platform,
        "expected_ranks": expected_ranks,
        "ranks": [label for label, _payload in traces],
        "priced": priced["histogram"],
        "served": None,
        "served_by_rank": {},
    }

    def _finish(status: str, detail: str) -> dict[str, Any]:
        verdict["status"] = status
        verdict["detail"] = detail
        return verdict

    labels = [label for label, _payload in traces]
    if len(set(labels)) != len(labels):
        raise TesseraRouteTraceError(f"rank labels repeat: {labels}")
    if len(traces) != expected_ranks:
        return _finish(NOT_VERIFIED, (
            f"NOT VERIFIED: {len(traces)} rank trace(s) supplied for "
            f"{expected_ranks} expected rank(s); every rank's trace is required"))
    missing = [label for label, payload in traces if payload is None]
    if missing:
        return _finish(NOT_VERIFIED, (
            f"NOT VERIFIED: no route trace for rank(s) {missing}; a missing "
            "observation is not a pass"))
    served_by_rank: dict[str, Any] = {}
    try:
        for label, payload in traces:
            entries = parse_route_trace(payload, where=f"trace[{label}]")
            served_by_rank[label] = served_histogram(entries, where=f"trace[{label}]")
    except RouteTraceNotVerified as exc:
        return _finish(NOT_VERIFIED, f"NOT VERIFIED: {exc}")
    except TesseraRouteTraceError as exc:
        verdict["served_by_rank"] = served_by_rank
        return _finish(REFUSED, f"REFUSED: {exc}")
    verdict["served_by_rank"] = served_by_rank
    first_label = labels[0]
    served = served_by_rank[first_label]["histogram"]
    verdict["served"] = served
    disagreeing = [label for label in labels[1:]
                   if served_by_rank[label]["histogram"] != served]
    if disagreeing:
        parts = [f"rank {first_label} {served}"] + [
            f"rank {label} {served_by_rank[label]['histogram']}" for label in disagreeing]
        return _finish(REFUSED, (
            "REFUSED: ranks served different module histograms; tensor "
            "parallelism shards a module and never splits it: " + "; ".join(parts)))
    difference = _histogram_difference(priced["histogram"], served)
    if difference:
        return _finish(REFUSED, (
            "REFUSED: the priced and served activation-contract histograms "
            "differ on platform " + repr(platform) + ": " + "; ".join(difference)))
    total = sum(served.values())
    return _finish(AGREE, (
        f"{total} module(s) on every rank served the activation contract they "
        f"were priced on, across {len(labels)} rank(s): "
        + ", ".join(f"{key}={count}" for key, count in served.items())))


def load_trace_contract() -> tuple[dict[str, dict[str, "str | None"]], dict[str, dict[str, Any]]]:
    """``(executes_by_platform, formats)`` from the pinned packaged contract.

    Read through the same dependency-free home the scoped census uses. An
    absent package or table raises: with no contract there is no priced
    contract to compare against, and that is not a pass.
    """
    from .tessera_route_receipt import _current_scoped_contract

    try:
        table, formats = _current_scoped_contract()
    except ModuleNotFoundError as exc:
        raise TesseraRouteTraceError(
            f"no packaged Tessera runtime contract is importable ({exc}); the "
            "priced activation contracts cannot be derived") from exc
    if not getattr(table, "present", False):
        raise TesseraRouteTraceError(
            "the packaged Tessera contract publishes no lane_eligibility "
            "table; the priced activation contracts cannot be derived")
    executes = {key: dict(entry.executes) for key, entry in sorted(table.platform_entries.items())}
    if not executes:
        raise TesseraRouteTraceError(
            "the packaged Tessera contract publishes no platform axis; what a "
            "platform executes cannot be derived")
    return executes, dict(formats)


def resolve_platform(build: Mapping[str, Any] | None, requested: str | None) -> str:
    """The platform to price on: the card's scoped target, or the caller's.

    Both, when present, must agree. Neither is a refusal: the trace carries no
    platform, and what a family executes is a per-platform fact.
    """
    scope = (build or {}).get("tessera_serving_scope")
    scoped = None
    if isinstance(scope, Mapping) and isinstance(scope.get("target"), Mapping):
        scoped = scope["target"].get("platform")
    if requested and scoped and requested != scoped:
        raise TesseraRouteTraceError(
            f"requested platform {requested!r} differs from the card's scoped "
            f"target platform {scoped!r}")
    platform = requested or scoped
    if not isinstance(platform, str) or not platform:
        raise TesseraRouteTraceError(
            "no serving platform: the card carries no tessera_serving_scope "
            "target and none was given; the trace does not say where it ran")
    return platform
