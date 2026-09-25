"""Explicit runtime intake shared by Tessera campaign, allocation and export.

The target describes the requested runtime, not the model. Structure is read
separately for every unit from probe/discovery facts, checked against the model
profile's declarations. Neither a model name nor the calibration device is a
serving target.

A stats row carries its unit's topology in one of three forms, each naming its
source (PQ #1278):

* ``_packed_experts_module`` + ``num_experts`` -- the producer recorded a
  packed expert stack (or one expert's view of it);
* ``router_path`` + ``expert_id`` -- the producer walked the module tree
  (``sensitivity_probe.discover_moe_structure``); both ``None`` means dense;
* ``unit_structure`` + ``unit_topology_source="profile_grammar"`` -- a table
  built before its producer wrote either fact, re-stamped by
  :func:`restamp_unit_topology` from the model profile's declared grammar.

A row with none of the three is refused; nothing here guesses a structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping

from .lane_eligibility import ServingContext, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE

#: Row key of a structure stamped from the profile grammar, never by a producer.
UNIT_STRUCTURE_KEY = "unit_structure"
#: Row key naming where a stamped structure came from.
UNIT_TOPOLOGY_SOURCE_KEY = "unit_topology_source"
#: The one stamp source this reader accepts.
PROFILE_GRAMMAR_SOURCE = "profile_grammar"
RESTAMP_SCHEMA = "prismaquant.unit_topology_restamp.v1"


@dataclass(frozen=True)
class ServingTarget:
    platform: str
    runtime_image: str
    execution_mode: str
    residency: str

    def __post_init__(self):
        # Reuse the owner's validation; this does not classify any model unit.
        self.context(STRUCTURE_DENSE)

    def context(self, structure: str) -> ServingContext:
        return ServingContext(structure=structure, **self.as_dict())

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def add_serving_scope_arguments(parser) -> None:
    """No defaults: a legacy context-free call remains context-free."""
    parser.add_argument("--tessera-platform", default=None,
                        help="Exact serving platform; otherwise the selected serving profile's target_platform")
    parser.add_argument("--tessera-runtime-image", default=None,
                        help="Exact serving repository@sha256 digest, never a mutable tag")
    parser.add_argument("--tessera-execution-mode", default=None,
                        help="Explicit serving execution mode: eager or compiled")
    parser.add_argument("--tessera-residency", default=None,
                        help="Explicit serving residency: resident or streamed")


def serving_target_from_args(args, *, target_platform: str | None = None) -> ServingTarget | None:
    fields = ("platform", "runtime_image", "execution_mode", "residency")
    values = {field: getattr(args, "tessera_" + field, None) for field in fields}
    if all(value is None for value in values.values()):
        return None
    explicit_platform = values["platform"]
    if explicit_platform is not None and target_platform and explicit_platform != target_platform:
        raise ValueError(
            f"tessera platform conflict: --tessera-platform={explicit_platform!r} "
            f"but the selected serving profile declares {target_platform!r}")
    if explicit_platform is None:
        values["platform"] = target_platform
    for field, value in values.items():
        if value is None:
            raise ValueError(f"tessera serving target requires explicit {field}; "
                             f"supply --tessera-{field.replace('_', '-')}")
    return ServingTarget(**values)


def unit_structure_from_profile(qname: str, profile) -> str:
    """Classify a known checkpoint unit with the profile's declared grammar.

    Export first proves the unit exists in the source headers. This function
    does not make a missing source tensor or unknown profile into a dense one.
    """
    if profile is None or profile.structure_spec() is None:
        raise ValueError(f"{qname}: explicit Tessera scope needs a declared model profile")
    if profile.packed_expert_format_group(qname) is not None:
        return STRUCTURE_ROUTED_MOE
    for rule in (profile.per_expert_moe_regex(), profile.per_expert_mtp_regex()):
        if rule and re.fullmatch(rule.removeprefix("re:"), profile.to_vllm_internal_name(qname)):
            return STRUCTURE_ROUTED_MOE
    return STRUCTURE_DENSE


def _has_producer_topology(row: Mapping) -> bool:
    return (row.get("_packed_experts_module") is not None or row.get("num_experts") is not None
            or "router_path" in row or "expert_id" in row)


def unit_structure_from_stats(qname: str, row: Mapping, profile) -> str:
    """Use owned probe facts, never tensor shape or a model-wide MoE guess."""
    declared_structure = unit_structure_from_profile(qname, profile)
    if UNIT_STRUCTURE_KEY in row or UNIT_TOPOLOGY_SOURCE_KEY in row:
        stamped = row.get(UNIT_STRUCTURE_KEY)
        if row.get(UNIT_TOPOLOGY_SOURCE_KEY) != PROFILE_GRAMMAR_SOURCE \
                or stamped not in (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE):
            stamp = {key: row.get(key) for key in (UNIT_STRUCTURE_KEY, UNIT_TOPOLOGY_SOURCE_KEY)}
            raise ValueError(f"{qname}: unrecognised unit topology stamp {stamp!r}")
        if _has_producer_topology(row):
            raise ValueError(f"{qname}: row carries both producer topology and a "
                             "profile-grammar stamp; one source per unit")
        if stamped != declared_structure:
            raise ValueError(f"{qname}: profile-grammar stamp {stamped!r} differs from the live "
                             f"profile's {declared_structure!r}; re-stamp against this profile")
        return stamped
    packed_module = row.get("_packed_experts_module")
    count = row.get("num_experts")
    if packed_module is not None or count is not None:
        if not isinstance(packed_module, str) or not packed_module or isinstance(count, bool) \
                or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{qname}: ambiguous packed expert topology; "
                             "need _packed_experts_module and positive num_experts")
        structure = STRUCTURE_ROUTED_MOE
    else:
        if "router_path" not in row or "expert_id" not in row:
            raise ValueError(f"{qname}: missing per-unit router_path/expert_id topology")
        router, expert = row["router_path"], row["expert_id"]
        if router is None and expert is None:
            structure = STRUCTURE_DENSE
        elif isinstance(router, str) and router and expert is not None \
                and not isinstance(expert, bool) and str(expert).isdigit():
            structure = STRUCTURE_ROUTED_MOE
        else:
            raise ValueError(f"{qname}: conflicting router_path/expert_id topology")
    if declared_structure == STRUCTURE_ROUTED_MOE and structure != STRUCTURE_ROUTED_MOE:
        raise ValueError(f"{qname}: probe topology conflicts with the profile's routed expert declaration")
    return structure


def context_by_unit_from_stats(target: ServingTarget | None, stats: Mapping[str, Mapping], profile
                               ) -> dict[str, ServingContext] | None:
    if target is None:
        return None
    return {name: target.context(unit_structure_from_stats(name, row, profile))
            for name, row in stats.items()}


def restamp_unit_topology(payload: Mapping, profile, *, input_sha256: str | None = None
                          ) -> tuple[dict, dict]:
    """Stamp every row that lacks producer topology from the profile grammar.

    For a cost table built before its producer wrote per-unit topology (the
    AURA payload before PQ #1278). A row whose producer recorded topology is
    left exactly as it is. Every other row gains ``unit_structure`` from
    :func:`unit_structure_from_profile` and ``unit_topology_source`` naming
    that source; no probe fact (router, expert id, packed module, expert count)
    is written, because none was observed. The payload's provenance records
    the counts per source and per structure. The input mapping is not
    modified: rows, the stats mapping and the provenance mapping are copied,
    and every other value is shared with the input.
    """
    stats = payload.get("stats")
    if not isinstance(stats, Mapping) or not stats:
        raise ValueError("unit topology restamp needs a non-empty stats mapping")
    new_stats, sources, structures = {}, {"producer": 0, PROFILE_GRAMMAR_SOURCE: 0}, {}
    for name, row in stats.items():
        if _has_producer_topology(row):
            structure = unit_structure_from_stats(name, row, profile)
            new_stats[name] = row
            sources["producer"] += 1
        else:
            if UNIT_STRUCTURE_KEY in row or UNIT_TOPOLOGY_SOURCE_KEY in row:
                raise ValueError(f"{name}: already stamped; restamp its unstamped source table")
            structure = unit_structure_from_profile(name, profile)
            new_stats[name] = {**row, UNIT_STRUCTURE_KEY: structure,
                               UNIT_TOPOLOGY_SOURCE_KEY: PROFILE_GRAMMAR_SOURCE}
            sources[PROFILE_GRAMMAR_SOURCE] += 1
        structures[structure] = structures.get(structure, 0) + 1
    summary = {"schema": RESTAMP_SCHEMA, "profile": getattr(profile, "name", type(profile).__name__),
               "sources": sources, "structures": dict(sorted(structures.items())),
               **({"input_sha256": input_sha256} if input_sha256 is not None else {})}
    result = dict(payload)
    provenance = dict(payload.get("provenance") or {})
    if "unit_topology_restamp" in provenance:
        raise ValueError("table already carries a unit topology restamp")
    provenance["unit_topology_restamp"] = summary
    result["stats"], result["provenance"] = new_stats, provenance
    return result, summary


def scope_provenance(target: ServingTarget | None,
                     contexts: Mapping[str, ServingContext] | None) -> dict:
    if target is None:
        return {}
    return {"target": target.as_dict(),
            "by_unit": {name: context.as_dict() for name, context in sorted((contexts or {}).items())}}
