"""D41 index reader; the pure Tessera producer owns measurement admission.

Allocation intersects that admission with the existing v11 allowable-rung rule.
Missing evidence waits; this input is not an identity seal or a serving claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .lane_eligibility import _allowable_rung_tables
from .schemas import strict_json_loads

SCHEMA = "fleet.rung_allowability.v1"
INDEX_SCHEMA = "fleet.rung_allowability.index.v1"


class RungAllowabilityError(ValueError):
    """The current publication cannot supply admitted measurement evidence."""


def read_allowability_json(path: str | Path) -> dict:
    try:
        payload = strict_json_loads(Path(path).read_text(encoding="utf-8"),
            duplicate=lambda key: RungAllowabilityError(f"{path}: duplicate JSON key {key!r}"),
            constant=lambda token: RungAllowabilityError(f"{path}: nonfinite JSON {token}"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise RungAllowabilityError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RungAllowabilityError(f"{path}: expected an object")
    return payload


def _producer_api():
    try:
        from tessera import rung_allowability
    except ImportError as exc:
        raise RungAllowabilityError("canonical tessera.rung_allowability producer admission required") from exc
    if any(not callable(getattr(rung_allowability, name, None))
           for name in ("validate_index", "validate_table", "admit_rung")):
        raise RungAllowabilityError("canonical rung allowability producer API is incomplete")
    return rung_allowability


@dataclass(frozen=True)
class RungAllowability:
    format: str
    kernel_build: Mapping
    table_version: int
    source_path: str
    dispositions: Mapping[int, str]

    def refusal(self, rung: int) -> str:
        return self.dispositions.get(rung, "unlisted rung; measurement waits")

    def allows(self, rung: int) -> bool:
        return self.refusal(rung) == ""

    def provenance(self) -> dict:
        return {"schema": SCHEMA, "format": self.format, "kernel_build": dict(self.kernel_build),
                "table_version": self.table_version, "source_path": self.source_path}


def load_rung_allowability(root: str | Path, *, format_entry: Mapping,
                           expected_kernel_build: Mapping) -> RungAllowability:
    """Validate through the producer, then select the current version only.

    The CEO's index contract selects formats[format].kernel_builds[build_id].
    current_version, not the highest filename or a caller-selected stale version.
    """
    root = Path(root).resolve()
    family = format_entry["family"]
    if not isinstance(expected_kernel_build, Mapping) or not expected_kernel_build.get("id"):
        raise RungAllowabilityError("independently observed kernel_build is required")
    index = read_allowability_json(root / "index.json")
    producer = _producer_api()
    producer.validate_index(index)
    if index.get("schema") != INDEX_SCHEMA:
        raise RungAllowabilityError("index.json: versioned allowability index required")
    try:
        build_entry = index["formats"][family]["kernel_builds"][expected_kernel_build["id"]]
        version = build_entry["current_version"]
        selected = build_entry["versions"][str(version)]
    except (KeyError, TypeError) as exc:
        raise RungAllowabilityError(f"current table for {family}/{expected_kernel_build['id']} is missing; measurement waits") from exc
    if type(version) is not int or version < 1:
        raise RungAllowabilityError("index.json: positive current_version required")
    relative = selected["path"]
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or "\\" in relative):
        raise RungAllowabilityError("index.json: table path must be a safe relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RungAllowabilityError("index.json: table path escapes publication root")
    table = read_allowability_json(path)
    producer.validate_table(table)
    if (selected["table_schema"] != SCHEMA or table["schema"] != SCHEMA
            or table["table_version"] != version
            or table["table_status"] != selected["table_status"]):
        raise RungAllowabilityError("selected table version/schema/status differs from index.json")
    if table["format"] != family or table["kernel_build"] != dict(expected_kernel_build):
        raise RungAllowabilityError("selected table format/kernel_build is stale for this allocation")
    rule = _allowable_rung_tables(format_entry, f"formats[{family}]")
    if not rule:
        raise RungAllowabilityError(f"{family}: existing v11 allowable_rungs rule required")
    if table["scope"]["grid_step_q256"] != format_entry["allowable_rungs"]["step_q256"]:
        raise RungAllowabilityError("table step differs from the published true q256 grid step")
    dispositions = {}
    for row in table["rungs"]:
        rung = row["rung"]
        decision = producer.admit_rung(table, format=family,
                                       kernel_build_id=expected_kernel_build["id"], rung=rung)
        status = decision["status"]
        if status not in {"allow", "wait", "hold", "excluded", "unsupported", "failed"}:
            raise RungAllowabilityError("unrecognized canonical producer admission decision")
        dispositions[rung] = decision["reason"] if status != "allow" else (
            "" if rung in rule else "excluded by the published allowable_rungs rule")
    if read_allowability_json(root / "index.json") != index:
        raise RungAllowabilityError("index.json changed during admission; selected version is stale")
    return RungAllowability(family, MappingProxyType(dict(expected_kernel_build)), version,
                           str(path), MappingProxyType(dispositions))
