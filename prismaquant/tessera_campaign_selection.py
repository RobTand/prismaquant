"""Torch-free campaign selection metadata shared by runtime and readset owners."""
from collections.abc import Mapping

UNITS_SCHEMA = "prismaquant.tessera_campaign_units.v1"
UNITS_SCHEMA_V2 = "prismaquant.tessera_campaign_units.v2"
UNITS_SCHEMA_V3 = "prismaquant.tessera_campaign_units.v3"
EXPERT_PARTITION_SCHEMA = "prismaquant.tessera_campaign_expert_partition.v1"


def validate_unit_selection(selection, *, path="in-memory") -> dict:
    """Validate the same selection at file, plan and merged-row boundaries."""
    schema = None if not isinstance(selection, dict) else selection.get("schema")
    if schema not in (UNITS_SCHEMA, UNITS_SCHEMA_V2, UNITS_SCHEMA_V3):
        raise RuntimeError(
            f"--units {path}: not a {UNITS_SCHEMA} or {UNITS_SCHEMA_V2} or {UNITS_SCHEMA_V3} "
            "selection")
    groups = selection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"--units {path}: names no anchor group")
    for entry in groups:
        if isinstance(entry, dict) and "partition" in entry and schema != UNITS_SCHEMA_V3:
            raise RuntimeError(f"--units {path}: expert partition requires {UNITS_SCHEMA_V3}")
        if schema == UNITS_SCHEMA_V3 and (not isinstance(entry, dict) or "partition" not in entry):
            raise RuntimeError(f"--units {path}: v3 requires explicit expert partition metadata")
        if (not isinstance(entry, dict) or not isinstance(entry.get("key"), str)
                or not isinstance(entry.get("members"), list)
                or not entry["members"]
                or not all(isinstance(m, str) for m in entry["members"])):
            raise RuntimeError(f"--units {path}: a group entry is not {{key, members[]}}")
        if schema == UNITS_SCHEMA and any(
                field in entry for field in ("sampled", "audit", "inclusion_probability", "stack_samples")):
            raise RuntimeError(
                f"--units {path}: group {entry['key']!r} carries a sample, "
                f"which is a {UNITS_SCHEMA_V2} field; a file that samples "
                "must say so in its schema")
        members = set(entry["members"])
        if schema == UNITS_SCHEMA_V3:
            part = entry["partition"]
            fields = {"schema", "experts_per_row", "index", "count", "rate_q256", "members"}
            if (not entry["key"].startswith("s:") or len(members) != len(entry["members"])
                    or any(field in entry for field in ("sampled", "audit", "inclusion_probability", "stack_samples"))
                    or not isinstance(part, dict) or set(part) != fields
                    or part.get("schema") != EXPERT_PARTITION_SCHEMA
                    or any(type(part[name]) is not int for name in ("experts_per_row", "index", "count", "rate_q256"))
                    or part["experts_per_row"] <= 0 or part["count"] <= 0
                    or not 0 <= part["index"] < part["count"] or part["rate_q256"] <= 0
                    or not isinstance(part["members"], list) or not part["members"]
                    or not all(isinstance(name, str) for name in part["members"])
                    or part["members"] != sorted(set(part["members"]))
                    or not set(part["members"]) <= members):
                raise RuntimeError(f"--units {path}: invalid expert partition descriptor")
        sampled = entry.get("sampled")
        if sampled is not None:
            if (not isinstance(sampled, list) or not sampled
                    or not all(isinstance(m, str) for m in sampled)
                    or not set(sampled) <= members):
                raise RuntimeError(
                    f"--units {path}: group {entry['key']!r} samples units that are not its members")
            audit = entry.get("audit") or []
            if not isinstance(audit, list) or not set(audit) <= set(sampled):
                raise RuntimeError(f"--units {path}: group {entry['key']!r} audits units it did not sample")
            pi = entry.get("inclusion_probability")
            if not isinstance(pi, dict) or not set(sampled) <= set(pi):
                raise RuntimeError(
                    f"--units {path}: group {entry['key']!r} samples without "
                    "an inclusion probability for every sampled unit; an "
                    "unbiased estimate downstream is impossible without it")
        elif entry.get("audit"):
            raise RuntimeError(f"--units {path}: group {entry['key']!r} audits without sampling")
    return selection


def selection_priced_units(selection: Mapping) -> tuple[set, set, dict]:
    """Return priced units, audited units and inclusion probabilities.

    Full group identity is separate from the sampled or partitioned readset.
    Producer geometry and complete expert chunks are checked by the runtime.
    """
    priced: set = set()
    audit: set = set()
    pi: dict = {}
    for entry in selection["groups"]:
        sampled = entry.get("sampled")
        partition = entry.get("partition")
        priced.update(partition["members"] if partition is not None else
                      (sampled if sampled else entry["members"]))
        audit.update(entry.get("audit") or ())
        for name, value in (entry.get("inclusion_probability") or {}).items():
            pi[str(name)] = float(value)
    return priced, audit, pi
