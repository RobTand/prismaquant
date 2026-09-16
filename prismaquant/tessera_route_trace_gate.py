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

Qualification
=============

One observation grade qualifies, and the verdict says two separate things:
``granularity`` is what the trace IS, and ``exact_module_qualified`` is true
only when the whole gate passed. **Only the exact grade can agree.**

**Exact** (Tessera #509). The header stamps ``identity_version: 1``, ``rank``,
``world_size``, ``rank_source == "torch.distributed"``, a ``rank_conflict``
field and the platform the process latched; every entry carries
``module_names`` (the sorted real module prefixes it counted),
``unnamed_modules`` and ``dispatches_without_prefix``. The gate then compares
the price with the serve **per module**: every target ``config.json`` prices
must appear, under exactly the contract it was priced on, on every rank. Two
modules that swapped contracts are refused even though their histogram is
unchanged, and one module counted under two keys in one forward is refused.

Missing or unknown identity is NOT VERIFIED -- never a pass, and never a
quiet downgrade to the counts. That covers a legacy file that names no
modules, an absent ``identity_version``, a version this consumer does not read
(1 is read exactly; a future one is not "at least v1"), entries that name some
modules and not others, a null ``rank``/``world_size`` (a process that never
joined a group), a ``rank_source`` that is absent or is not
``torch.distributed``, an absent ``rank_conflict`` field, a ``""`` platform
(the producer's "never latched a token"), and ``unnamed_modules`` above zero.
A count is not a name, so none of these can be carried by the histogram
instead.

Conflicting identity is REFUSED: a non-null ``rank_conflict`` (the producer
records a later disagreeing rank observation rather than adopting it), a
stamped platform other than the one the price was derived for, a rank/world
pair that contradicts the traces supplied or the ``rankN`` label, a
``modules`` count that disagrees with ``len(module_names) + unnamed_modules``,
a repeated name in one entry, and any per-module disagreement with the price.

**Histogram.** A legacy trace names no modules. ``modules`` is then a count of
a set whose members the file does not emit, so the most it supports, exactly,
is a **module count per (route family, kind, activation contract), per token
count M, per rank**:

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

The grade is one fact about the serve, not one per rank: a set in which some
ranks name their modules and others do not cannot be read as either. It is
compared as a histogram and, whatever it says, it cannot agree: the histogram
is a diagnosis here, and a disagreement in it is still REFUSED loudly. Ranks
that both name their modules are compared per module, so the exact grade is
strictly stronger than the histogram it subsumes.

What this does not see
======================

* Which module rode which contract. Two modules that swapped contracts leave
  the histogram unchanged. Tessera #509 closes this where the serve names its
  modules; a legacy trace is a histogram diagnosis and is NOT VERIFIED, never
  per-module qualified (``exact_module_qualified`` stays false).
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

Three outcomes, never two. ``agree`` needs the exact grade AND every module
match. ``refused`` is a conflict: the observation disagrees with the price,
with its own header, or with itself. ``not_verified`` is no qualifying
observation: a rank's trace is missing, unreadable, empty, of another schema,
compiled, legacy-only, or missing any part of the identity. Only ``agree``
closes the shipcard slot; both other outcomes leave it unfilled, and
``shipcard`` reads the status rather than a flag.

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

#: The granularity a trace that names its modules supports (Tessera #509).
EXACT_GRANULARITY = (
    "module name per (route family, kind, activation contract), "
    "per token count M, per rank; the trace names its modules"
)

#: The two observation grades. ``EXACT`` is a trace that names its modules.
EXACT = "exact"
HISTOGRAM = "histogram"

#: The only identity schema version this consumer reads (#509). A future
#: version is refused rather than read as v1: its fields are not these fields.
IDENTITY_VERSION = 1

#: The only ``rank_source`` that binds a trace to a rank. The producer reports
#: ``"unavailable"`` -- with a null rank -- for a process that never joined a
#: group, which is an unknown rank rather than rank 0.
RANK_SOURCE = "torch.distributed"

#: The trace's ``kind`` against the artifact's ``scheme.structure``.
TRACE_KIND_FOR_STRUCTURE = {"dense": "dense", "routed_moe": "moe"}

_SHAPE = re.compile(r"^M(\*|[0-9]+):N([0-9]+):K([0-9]+)$")
_LABEL_RANK = re.compile(r"^rank([0-9]+)(?::|$)")
_ENTRY_STR_FIELDS = ("policy", "shape", "symbol", "decoder", "contract", "kind")

#: ``rank`` and ``world_size`` are one fact, so they are stamped together.
_HEADER_FIELDS = ("rank", "world_size", "rank_source", "rank_conflict",
                  "platform", "identity_version")

#: The producer's token for "this process never latched a platform". It is
#: what a census already reads as "this record does not say", and this gate
#: cannot bind a serve to a platform it never named: NOT VERIFIED, not a
#: comparison against the priced one.
UNKNOWN_PLATFORM = ""

#: The per-entry identity fields the #509 schema adds. All three travel
#: together: the names, the objects that have no stable name, and the
#: dispatches that arrived without one.
_IDENTITY_ENTRY_FIELDS = ("module_names", "unnamed_modules", "dispatches_without_prefix")


class TesseraRouteTraceError(ValueError):
    """The observation or the price disagrees, or an input is malformed."""


class RouteTraceNotVerified(TesseraRouteTraceError):
    """No usable observation exists. This is not a pass."""


def _key(family: str, kind: str, contract: str) -> str:
    return f"{family}/{kind}/{contract}"


def _decode(payload: Any, *, where: str) -> Mapping[str, Any]:
    """JSON-decode and schema-check one trace file."""
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
    return payload


def parse_trace_header(payload: Any, *, where: str) -> dict[str, Any]:
    """The file header's additive identity stamps (#509).

    ``rank``/``world_size``/``rank_source``/``rank_conflict``/``platform``,
    plus the ``identity_version`` that says which entry schema the file
    speaks. A legacy file stamps none of them and the consumer then knows the
    trace is a histogram diagnosis: nothing but the path binds it to a rank.

    What is here only classifies a malformed or self-contradictory header
    (REFUSED); whether the identity is COMPLETE -- present, readable, bound to
    a real rank -- is decided by :func:`trace_identity`, which owns the
    not-verified cases. ``present`` records which fields the file actually
    carries, because an absent ``rank_conflict`` is not a null one.

    The version is read exactly: :data:`IDENTITY_VERSION` or nothing. A
    different value is not "at least a version" -- its entry schema is not
    this one -- so it can never qualify as v1.
    """
    document = _decode(payload, where=where)
    header: dict[str, Any] = {field: document.get(field) for field in _HEADER_FIELDS}
    header["present"] = sorted(field for field in _HEADER_FIELDS if field in document)
    rank, world_size, platform = header["rank"], header["world_size"], header["platform"]
    version = header["identity_version"]
    for field in ("rank", "world_size"):
        value = header[field]
        if value is not None and (type(value) is not int or value < 0):
            raise TesseraRouteTraceError(
                f"{where}: header {field} must be a non-negative integer")
    if rank is not None and world_size is not None and world_size < 1:
        raise TesseraRouteTraceError(
            f"{where}: header world_size must be a positive integer")
    if rank is not None and world_size is not None and rank >= world_size:
        raise TesseraRouteTraceError(
            f"{where}: header rank {rank} is not below world_size {world_size}")
    if platform is not None and not isinstance(platform, str):
        raise TesseraRouteTraceError(
            f"{where}: header platform must be a string")
    if version is not None and type(version) is not int:
        raise TesseraRouteTraceError(
            f"{where}: header identity_version must be an integer")
    return header


def _identity_fields(
    entry: Mapping[str, Any], *, at: str,
) -> tuple[tuple[str, ...] | None, int | None, int | None]:
    """One entry's ``(module_names, unnamed_modules, dispatches_without_prefix)``.

    A legacy entry carries none of them and returns ``(None, None, None)``.
    They travel together, and the count stays a fact about objects:
    ``modules == len(module_names) + unnamed_modules``, which is what stops two
    prefix-less objects from collapsing into the placeholder "1".

    A field that is missing while its siblings are present is MISSING
    METADATA -- ``RouteTraceNotVerified``, never a quiet fall back to the
    counts. A field that is present and contradicts the others, or itself, is
    REFUSED.
    """
    present = [field for field in _IDENTITY_ENTRY_FIELDS if entry.get(field) is not None]
    if not present:
        return None, None, None
    if len(present) != len(_IDENTITY_ENTRY_FIELDS):
        missing = [field for field in _IDENTITY_ENTRY_FIELDS if field not in present]
        raise RouteTraceNotVerified(
            f"{at}: stamps {', '.join(present)} without {', '.join(missing)}; the "
            "#509 identity fields travel together, and a module count cannot "
            "stand in for the names (#509)")
    names = entry["module_names"]
    if not isinstance(names, list):
        raise TesseraRouteTraceError(
            f"{at}: module_names must be a list of module prefixes")
    for name in names:
        if not isinstance(name, str) or not name:
            raise TesseraRouteTraceError(
                f"{at}: module_names entries must be non-empty strings")
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise TesseraRouteTraceError(
            f"{at}: module_names repeats {repeated}; one entry names each of its "
            "modules once")
    counts: list[int] = []
    for field in ("unnamed_modules", "dispatches_without_prefix"):
        value = entry[field]
        if type(value) is not int or value < 0:
            raise TesseraRouteTraceError(
                f"{at}: {field} must be a non-negative integer")
        counts.append(value)
    unnamed, prefixless = counts
    if len(names) + unnamed != entry["modules"]:
        raise TesseraRouteTraceError(
            f"{at}: module_names names {len(names)} module(s) and unnamed_modules "
            f"is {unnamed} but modules={entry['modules']}; the count is the number "
            "of unique served objects (#509)")
    return tuple(sorted(names)), unnamed, prefixless


def parse_route_trace_document(
    payload: Any, *, where: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One rank's ``(entries, header)`` in canonical form.

    Raises :class:`RouteTraceNotVerified` when the file cannot serve as an
    observation (wrong schema, no entries, compiled counts) and
    :class:`TesseraRouteTraceError` when it claims to be one and is malformed.
    """
    document = _decode(payload, where=where)
    header = parse_trace_header(document, where=where)
    entries = document.get("entries")
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
        module_names, unnamed_modules, dispatches_without_prefix = _identity_fields(
            entry, at=at)
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
            "module_names": module_names,
            "unnamed_modules": unnamed_modules,
            "dispatches_without_prefix": dispatches_without_prefix,
        })
    return parsed, header


def parse_route_trace(payload: Any, *, where: str) -> list[dict[str, Any]]:
    """Validate one rank's trace and return its entries in canonical form.

    Raises :class:`RouteTraceNotVerified` when the file cannot serve as an
    observation (wrong schema, no entries, compiled counts) and
    :class:`TesseraRouteTraceError` when it claims to be one and is malformed.
    """
    entries, _header = parse_route_trace_document(payload, where=where)
    return entries


def trace_identity(
    entries: Sequence[Mapping[str, Any]], header: Mapping[str, Any], *, where: str,
) -> str:
    """``EXACT`` when the identity is complete, ``HISTOGRAM`` when there is none.

    Everything in between is MISSING OR UNKNOWN identity, which is
    ``RouteTraceNotVerified``: a version without names, names without a
    version, an entry that names some of its modules and not others, a version
    this consumer does not read, a rank that was never observed, a
    ``rank_source`` that is not ``torch.distributed``, an absent
    ``rank_conflict`` field, a platform the process never latched, or a module
    the serve could not name. None of those can qualify, and none of them may
    fall back to the legacy counts -- that is also why a legacy file
    (``HISTOGRAM``) is a diagnosis and never an agreement.

    A non-null ``rank_conflict`` is the one identity fact that is a CONFLICT
    rather than a gap: the producer observed a later, disagreeing rank and
    recorded it instead of adopting it, so the serve is not one identity.
    """
    declared = header["identity_version"]
    named = sum(1 for entry in entries if entry["module_names"] is not None)
    if declared is None and named == 0:
        return HISTOGRAM
    if declared is None:
        raise RouteTraceNotVerified(
            f"{where}: {named} of {len(entries)} entries name their modules but "
            "the header stamps no identity_version; this consumer will not guess "
            "which entry schema the file speaks (#509)")
    if declared != IDENTITY_VERSION:
        raise RouteTraceNotVerified(
            f"{where}: header identity_version is {declared!r}; this consumer "
            f"reads exactly {IDENTITY_VERSION}, and a later version's fields are "
            "not these fields (#509)")
    if named != len(entries):
        raise RouteTraceNotVerified(
            f"{where}: identity_version {declared} but only {named} of "
            f"{len(entries)} entries name their modules; a file that names some "
            "modules and not others is not one observation (#509)")
    if header["rank"] is None or header["world_size"] is None:
        raise RouteTraceNotVerified(
            f"{where}: the header stamps rank={header['rank']!r} "
            f"world_size={header['world_size']!r} (rank_source="
            f"{header['rank_source']!r}); a process that never joined a group "
            "cannot bind a trace to a rank (#509)")
    if header["rank_source"] != RANK_SOURCE:
        raise RouteTraceNotVerified(
            f"{where}: the header stamps rank_source={header['rank_source']!r}, "
            f"not {RANK_SOURCE!r}; the rank it carries is not one this gate can "
            "bind the trace to (#509)")
    if "rank_conflict" not in header["present"]:
        raise RouteTraceNotVerified(
            f"{where}: the header carries no rank_conflict field, so the file "
            "does not say whether a later rank observation disagreed (#509)")
    if header["rank_conflict"] is not None:
        raise TesseraRouteTraceError(
            f"{where}: the serve recorded a later disagreeing rank identity "
            f"({header['rank_conflict']}); its counts and its header are not one "
            "identity, which is a conflict and not a missing field (#509)")
    if header["platform"] in (None, UNKNOWN_PLATFORM):
        raise RouteTraceNotVerified(
            f"{where}: the header stamps platform={header['platform']!r}; the "
            "process never latched a platform, so which platform executed these "
            "routes is unknown (#509)")
    unnamed = sum(entry["unnamed_modules"] for entry in entries)
    prefixless = sum(entry["dispatches_without_prefix"] for entry in entries)
    if unnamed or prefixless:
        raise RouteTraceNotVerified(
            f"{where}: the serve reports {unnamed} module(s) with no stable "
            f"prefix and {prefixless} dispatch(es) without one; this gate compares "
            "per module and a count is not a name, so the legacy counts do not "
            "stand in for them (#509)")
    return EXACT


def exact_served_modules(
    entries: Sequence[Mapping[str, Any]], *, where: str,
) -> dict[str, str]:
    """One rank's ``{module prefix: '<family>/<kind>/<contract>'}``.

    Every token count must name the same modules under the same contracts, and
    no module may ride two contracts in one forward: that is what makes the
    comparison per module rather than per count.
    """
    by_m: dict[int, dict[str, str]] = {}
    for entry in entries:
        names = entry["module_names"]
        if names is None:
            raise TesseraRouteTraceError(
                f"{where}: entry at M={entry['m']} names no modules")
        key = _key(entry["family"], entry["kind"], entry["contract"])
        named = by_m.setdefault(entry["m"], {})
        for name in names:
            if name in named:
                raise TesseraRouteTraceError(
                    f"{where}: module {name!r} dispatched under {named[name]} and "
                    f"{key} in one forward (M={entry['m']}); one module holds one "
                    "activation contract")
            named[name] = key
    reference_m = min(by_m)
    reference = dict(sorted(by_m[reference_m].items()))
    for m in sorted(by_m):
        other = dict(sorted(by_m[m].items()))
        if other == reference:
            continue
        differing = [name for name in sorted(set(reference) | set(other))
                     if reference.get(name) != other.get(name)]
        detail = ", ".join(
            f"{name}: M={reference_m} {reference.get(name)}, M={m} {other.get(name)}"
            for name in _sample(differing))
        raise TesseraRouteTraceError(
            f"{where}: the served modules differ between token counts: {detail}; "
            "every forward dispatches every quantized module once, so a module "
            "absent at one M or riding another contract at one M is a route "
            "change inside the serve")
    return reference


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


def _sample(names: Sequence[str], *, limit: int = 8) -> list[str]:
    """At most ``limit`` names, so one bad shard cannot fill a receipt."""
    if len(names) <= limit:
        return list(names)
    return [*names[:limit], f"({len(names) - limit} further module(s))"]


def _module_difference(priced: Mapping[str, str], served: Mapping[str, str]) -> list[str]:
    """``priced`` and ``served`` are ``{module prefix: contract key}``.

    The whole point of the exact grade: a contract that moved from one module
    to another leaves the counts alone and shows up here.
    """
    differing = [name for name in sorted(set(priced) | set(served))
                 if priced.get(name) != served.get(name)]
    lines = []
    for name in _sample(differing):
        want, got = priced.get(name), served.get(name)
        if want is None:
            lines.append(f"{name}: served {got} but the price names no such module")
        elif got is None:
            lines.append(f"{name}: priced {want}, served by no module")
        else:
            lines.append(f"{name}: priced {want} but served {got}")
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
        "exact_module_qualified": False,
        "platform": platform,
        "expected_ranks": expected_ranks,
        "ranks": [label for label, _payload in traces],
        "priced": priced["histogram"],
        "priced_owners": priced["owners"],
        "served": None,
        "served_by_rank": {},
        "served_modules": None,
        "header": {},
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
    headers: dict[str, Any] = {}
    grades: dict[str, str] = {}
    served_modules: dict[str, Any] = {}
    try:
        for label, payload in traces:
            where = f"trace[{label}]"
            entries, header = parse_route_trace_document(payload, where=where)
            headers[label] = header
            grades[label] = trace_identity(entries, header, where=where)
            served_by_rank[label] = served_histogram(entries, where=where)
            if grades[label] == EXACT:
                served_modules[label] = exact_served_modules(entries, where=where)
    except RouteTraceNotVerified as exc:
        return _finish(NOT_VERIFIED, f"NOT VERIFIED: {exc}")
    except TesseraRouteTraceError as exc:
        verdict["served_by_rank"] = served_by_rank
        return _finish(REFUSED, f"REFUSED: {exc}")
    verdict["header"] = headers
    verdict["served_by_rank"] = served_by_rank

    grade = set(grades.values())
    if len(grade) > 1:
        # A set in which some ranks carry an identity and others do not is
        # incomplete, not in conflict: no per-module claim can be read from it,
        # and the counts cannot be read in its place either.
        return _finish(NOT_VERIFIED, (
            "NOT VERIFIED: ranks do not carry the same identity: " + ", ".join(
                f"{label} {'names its modules' if grades[label] == EXACT else 'names no modules'}"
                for label in labels)
            + "; a per-module comparison needs every rank to name its modules, "
            "and the counts are not a substitute (#509)"))
    grade = grade.pop()
    verdict["granularity"] = EXACT_GRANULARITY if grade == EXACT else GRANULARITY
    if grade == EXACT:
        verdict["served_modules"] = served_modules

    # The serve's own rank/world binding (#509). Every part of it has already
    # been read per file; what is left is the set the ranks make together, and
    # a set that contradicts the claim is a conflict rather than a gap.
    if grade == EXACT:
        stamped_ranks = sorted(headers[label]["rank"] for label in labels)
        world_sizes = sorted({headers[label]["world_size"] for label in labels})
        if len(world_sizes) != 1:
            return _finish(REFUSED, (
                "REFUSED: rank traces disagree on world_size: "
                + repr(world_sizes)))
        world_size = world_sizes[0]
        if world_size != expected_ranks or world_size != len(traces):
            return _finish(REFUSED, (
                f"REFUSED: the traces stamp world_size={world_size} but "
                f"{len(traces)} trace(s) were supplied for {expected_ranks} "
                "expected rank(s); the observation and the claim are different "
                "serves"))
        if stamped_ranks != list(range(world_size)):
            return _finish(REFUSED, (
                f"REFUSED: the traces stamp ranks {stamped_ranks}, which is not "
                f"every rank of world_size={world_size}"))
        for label in labels:
            embedded = _LABEL_RANK.match(label)
            if embedded is None:
                continue
            if int(embedded.group(1)) != headers[label]["rank"]:
                return _finish(REFUSED, (
                    f"REFUSED: trace {label!r} is bound to rank "
                    f"{embedded.group(1)} but its own header stamps rank "
                    f"{headers[label]['rank']}"))
        platforms = sorted({headers[label]["platform"] for label in labels})
        if len(platforms) > 1:
            return _finish(REFUSED, (
                "REFUSED: rank traces disagree on platform: " + repr(platforms)))
        if platforms[0] != platform:
            return _finish(REFUSED, (
                f"REFUSED: the serve recorded platform {platforms[0]!r} but the "
                f"price is for {platform!r}; what a family executes is a "
                "per-platform fact"))

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
    if grade == EXACT:
        for label in labels:
            difference = _module_difference(
                priced["owners"], served_modules[label])
            if difference:
                return _finish(REFUSED, (
                    "REFUSED: the priced and served activation contracts differ "
                    "per module on platform " + repr(platform) + " (rank "
                    + label + "): " + "; ".join(difference)))
        # Only now: every named module on every rank matched its price, so the
        # per-module claim is what actually passed.
        verdict["exact_module_qualified"] = True
        total = len(priced["owners"])
        return _finish(AGREE, (
            f"exact per-module: {total} named module(s) on every rank served "
            f"the activation contract they were priced on, across "
            f"{len(labels)} rank(s): "
            + ", ".join(f"{key}={count}" for key, count in served.items())))
    difference = _histogram_difference(priced["histogram"], served)
    if difference:
        return _finish(REFUSED, (
            "REFUSED: the priced and served activation-contract histograms "
            "differ on platform " + repr(platform) + ": " + "; ".join(difference)))
    total = sum(served.values())
    # The histogram agrees, and it still cannot qualify: the trace names no
    # modules, so nothing here says which module rode which contract.
    return _finish(NOT_VERIFIED, (
        f"NOT VERIFIED: the trace is histogram grade, not per-module: "
        f"{total} module(s) on every rank match the priced activation-contract "
        f"counts across {len(labels)} rank(s) ("
        + ", ".join(f"{key}={count}" for key, count in served.items())
        + "), but the file names no modules, so which module rode which "
        "contract is not something it can support (#509)"))


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
    platform on a legacy file, and what a family executes is a per-platform
    fact. A #509 file does stamp its platform, and
    :func:`compare_route_traces` refuses one that differs from this answer.
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
