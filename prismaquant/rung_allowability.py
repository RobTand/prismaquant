"""D41 index reader; the pure Tessera producer owns measurement admission.

Allocation intersects that admission with the existing v11 allowable-rung rule.
Missing evidence waits; this input is not an identity seal or a serving claim.

The v3 candidate join (PQ #2364) scopes admission to the unit priced: actual
dense or routed structure, declared shape, token-row regime M, activation
build and wire recipe travel from the allocator through the shared lane seam
into the owning producer calls. Rates, geometry and the performant menu are
never re-derived here; unresolvable scope waits.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .lane_eligibility import _allowable_rung_tables
from .schemas import strict_json_loads
from .dev_mode import seal_check


class RungAllowabilityError(ValueError):
    """The current publication cannot supply admitted measurement evidence."""


#: Producer calls the consumer joins through, at their shared boundary.
PRODUCER_API = ("validate_index", "validate_table", "admit_rung",
                "scope_cell_ids", "geometry_class_identity",
                "rung_speed", "rung_quality")

#: Consumer serving-structure vocabulary to producer kernel-kind vocabulary.
#: Routed experts serve under ``routed_moe`` here and ``routed`` there; the
#: map is the whole translation, stated once beside the owner that uses it.
PRODUCER_KERNEL_KINDS = MappingProxyType(
    {"dense": "dense", "routed_moe": "routed", "routed": "routed"})

#: Schema carrying per-cell performance scope (PQ #2364).
V3_SCHEMA = "fleet.rung_allowability.v3"

#: Scope axes the seam carries. Cell resolution reads kernel_kind, rows,
#: columns, m and routing; activation_contract and recipe ride through to
#: the owning admission call unchanged.
SCOPE_FIELDS = ("kernel_kind", "rows", "columns", "m", "routing",
                "activation_contract", "recipe")


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
    missing = [name for name in PRODUCER_API
               if not callable(getattr(rung_allowability, name, None))]
    if missing:
        raise RungAllowabilityError(
            "canonical tessera.rung_allowability producer admission required "
            f"(missing: {', '.join(missing)})")
    return rung_allowability


def _producer_kind(structure: str) -> str:
    """The producer kernel kind for one consumer serving structure."""
    try:
        return PRODUCER_KERNEL_KINDS[structure]
    except KeyError as exc:
        raise RungAllowabilityError(
            f"unknown serving structure {structure!r}; expected one of "
            f"{sorted(PRODUCER_KERNEL_KINDS)}") from exc


def _scope_key(scope: Mapping | None) -> tuple | None:
    """The cache key for one admission scope, or ``None`` when unscoped.

    A scope that names no axis is unscoped. Unknown axes, wrong types and
    non-positive shape integers refuse: a misspelled scope must never read
    as a wider one.
    """
    if scope is None:
        return None
    if not isinstance(scope, Mapping):
        raise RungAllowabilityError("allowability scope must be a mapping")
    unknown = [key for key in scope if key not in SCOPE_FIELDS]
    if unknown:
        raise RungAllowabilityError(
            f"unknown allowability scope axis {unknown!r}; expected subset of "
            f"{list(SCOPE_FIELDS)}")
    kind = scope.get("kernel_kind")
    if kind is not None:
        if not isinstance(kind, str):
            raise RungAllowabilityError("allowability scope kernel_kind must be a string")
        _producer_kind(kind)
    for axis in ("rows", "columns", "m"):
        value = scope.get(axis)
        if value is not None and (type(value) is not int or value <= 0):
            raise RungAllowabilityError(
                f"allowability scope {axis} must be a positive integer")
    routing = scope.get("routing")
    if routing is not None and not isinstance(routing, str):
        raise RungAllowabilityError("allowability scope routing must be a string")
    activation = scope.get("activation_contract")
    if activation is not None and (not isinstance(activation, str) or not activation):
        raise RungAllowabilityError("allowability scope activation_contract must be a non-empty string")
    recipe = scope.get("recipe")
    recipe_json = None
    if recipe is not None:
        if not isinstance(recipe, Mapping):
            raise RungAllowabilityError("allowability scope recipe must be a mapping")
        try:
            recipe_json = json.dumps(recipe, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise RungAllowabilityError(
                f"allowability scope recipe is not JSON-serialisable: {exc}") from exc
    key = (kind,
           scope.get("rows"), scope.get("columns"), scope.get("m"), routing,
           activation, recipe_json)
    if all(value is None for value in key):
        return None
    return key


@dataclass(frozen=True)
class RungAllowability:
    format: str
    kernel_build: Mapping
    table_version: int
    source_path: str
    _table: dict = field(repr=False)
    _producer: object = field(repr=False)
    _rule: Mapping = field(repr=False)
    _rungs: frozenset[int] = field(repr=False)
    _refusals: dict = field(default_factory=dict, repr=False, compare=False)
    _cells: dict = field(default_factory=dict, repr=False, compare=False)

    def refusal(self, rung: int, *, scope: Mapping | None = None) -> str:
        """The owning admission verdict for one rung, optionally scoped.

        Without scope the call is the historical whole-table verdict. With
        scope the verdict covers the unit's actual structure, declared shape
        and regime M; an unresolvable scope waits, it never widens. Cached
        per (rung, scope) within one loaded table.
        """
        key = (rung, _scope_key(scope))
        if key not in self._refusals:
            self._refusals[key] = self._decide(rung, key[1])
        return self._refusals[key]

    def allows(self, rung: int, *, scope: Mapping | None = None) -> bool:
        return self.refusal(rung, scope=scope) == ""

    def _decide(self, rung: int, scope_key: tuple | None) -> str:
        cells = None
        call: dict = {}
        if scope_key is not None and self._table.get("schema") == V3_SCHEMA:
            cells = self._resolve_cells(scope_key)
            if cells is None:
                return self._legacy_verdict(rung)
            call = {"cell_ids": list(cells)}
            if scope_key[5] is not None:
                call["activation_contract"] = scope_key[5]
            if scope_key[6] is not None:
                call["recipe"] = json.loads(scope_key[6])
        elif scope_key is not None:
            # Scopes name per-cell evidence, which pre-v3 tables have no
            # vocabulary for; the whole-row verdict stands, documented.
            return self._legacy_verdict(rung)
        decision = self._producer.admit_rung(self._table, format=self.format,
            kernel_build_id=self.kernel_build["id"], rung=rung, **call)
        return self._read_verdict(rung, decision)

    def _legacy_verdict(self, rung: int) -> str:
        decision = self._producer.admit_rung(self._table, format=self.format,
            kernel_build_id=self.kernel_build["id"], rung=rung)
        return self._read_verdict(rung, decision)

    def _read_verdict(self, rung: int, decision: Mapping) -> str:
        status = decision["status"]
        if status not in {"allow", "wait", "hold", "excluded", "unsupported", "failed"}:
            raise RungAllowabilityError("unrecognized canonical producer admission decision")
        if status != "allow":
            reason = decision["reason"]
            if rung not in self._rungs:
                reason = "unlisted rung; " + reason
        else:
            reason = "" if rung in self._rule else "excluded by the published allowable_rungs rule"
        return reason

    def _resolve_cells(self, scope_key: tuple) -> tuple | None:
        """Required cells covering one scope, ``None`` when scope is inapplicable.

        ``None`` means the scope carries no structure evidence, so the
        whole-table verdict stands. An empty tuple means the scope resolved
        and nothing covers it, so admission waits. Exact declared-shape
        resolution goes through the owning ``scope_cell_ids``; kind-only
        scopes filter the table's own required cells without re-deriving
        geometry.
        """
        if scope_key in self._cells:
            return self._cells[scope_key]
        kind, rows, columns, m, routing, _, _ = scope_key
        resolved = self._resolve_cells_uncached(kind, rows, columns, m, routing)
        self._cells[scope_key] = resolved
        return resolved

    def _resolve_cells_uncached(self, kind, rows, columns, m, routing):
        if kind is None:
            return None
        producer_kind = _producer_kind(kind)
        required = self._table["scope"]["required_cells"]
        if rows is not None and columns is not None and m is not None:
            return tuple(self._producer.scope_cell_ids(
                self._table, kernel_kind=producer_kind, rows=rows,
                columns=columns, M=m,
                **({} if routing is None else {"routing": routing})))
        if rows is not None and columns is not None:
            if routing is not None:
                # A routing without a regime has no declared-cell meaning;
                # the whole-table verdict stands rather than a narrowed one.
                return None
            regimes = sorted({cell["M"] for cell in required
                              if cell["kernel_kind"] == producer_kind})
            found: list[str] = []
        for regime in regimes:
            for cell_id in self._producer.scope_cell_ids(
                    self._table, kernel_kind=producer_kind, rows=rows,
                    columns=columns, M=regime):
                if cell_id not in found:
                    found.append(cell_id)
        return tuple(found)
        if rows is not None or columns is not None or routing is not None:
            # A partial shape identity resolves nothing; the whole-table
            # verdict stands rather than a guessed one.
            return None
        selected = [cell["cell_id"] for cell in required
                    if cell["kernel_kind"] == producer_kind
                    and (m is None or cell["M"] == m)]
        return tuple(selected)

    def cells_for(self, *, kernel_kind: str | None = None, rows: int | None = None,
                  columns: int | None = None, m: int | None = None,
                  routing: str | None = None) -> tuple[str, ...]:
        """Required cells covering one scope; empty when nothing covers it.

        Timing joins read this: a withhold here means no canonical time
        exists for the scope. Pre-v3 tables carry no cells and answer empty.
        """
        if self._table.get("schema") != V3_SCHEMA:
            return ()
        resolved = self._resolve_cells(_scope_key({
            "kernel_kind": kernel_kind, "rows": rows, "columns": columns,
            "m": m, "routing": routing}))
        return () if resolved is None else resolved

    def class_identity(self, rung: int, measurement: Mapping) -> dict:
        """The owning geometry-class identity of one actual measurement."""
        try:
            return self._producer.geometry_class_identity(self._table, rung, measurement)
        except ValueError as exc:
            raise RungAllowabilityError(str(exc)) from exc

    def canonical_time(self, rung: int, *, cell_id: str | None = None,
                       class_identity: Mapping | None = None) -> dict:
        """Actual or safe class-derived time for one rung, with provenance.

        Measured cells report their own time; a class identity without a
        measurement reports the owning safe derivation or waits. Timing
        evidence never carries numerical or serving qualification: both
        inherit flags read ``False`` from the producer. Pre-v3 tables wait.
        """
        if self._table.get("schema") != V3_SCHEMA:
            return {"status": "wait", "reason": "canonical_timing_requires_v3_table"}
        if class_identity is not None and not isinstance(class_identity, Mapping):
            raise RungAllowabilityError("canonical class identity must be a mapping")
        result = dict(self._producer.rung_speed(
            self._table, rung=rung, cell_id=cell_id,
            class_identity=None if class_identity is None else dict(class_identity)))
        if result.get("status") not in {"measured", "inherited", "wait", "hold",
                                        "unsupported", "failed"}:
            raise RungAllowabilityError("unrecognized canonical producer timing decision")
        result["provenance"] = {"schema": self._table["schema"], "format": self.format,
            "table_version": self.table_version,
            "kernel_build_id": self.kernel_build["id"]}
        return result

    def chord_quality(self, rung: int, *, lower_rung: int, upper_rung: int,
                      lower_value: float, upper_value: float,
                      lower_scope: Mapping, upper_scope: Mapping) -> dict:
        """The qualified neighbour chord between two comparable anchors.

        Both anchors must name the same unit, family and validated currency;
        a calibration named on either must match on both. Anything else
        withholds: sampled weight error is never relabelled as validated KL
        here, and the arithmetic itself stays the producer's.
        """
        unit = self._anchor_scope(lower_scope, "lower")
        if self._anchor_scope(upper_scope, "upper") != unit:
            raise RungAllowabilityError("quality anchors name different units")
        if lower_scope["currency"] != upper_scope["currency"]:
            raise RungAllowabilityError("quality anchors name different currencies")
        if (self._anchor_calibration(lower_scope) is not None
                or self._anchor_calibration(upper_scope) is not None):
            if self._anchor_calibration(lower_scope) != self._anchor_calibration(upper_scope):
                raise RungAllowabilityError("quality anchors name different calibrations")
        for value, tag in ((lower_value, "lower"), (upper_value, "upper")):
            if type(value) is bool or not isinstance(value, (int, float)):
                raise RungAllowabilityError(f"{tag} quality anchor value must be a number")
        try:
            result = dict(self._producer.rung_quality(
                rung, lower_rung=lower_rung, upper_rung=upper_rung,
                lower_value=lower_value, upper_value=upper_value))
        except ValueError as exc:
            raise RungAllowabilityError(str(exc)) from exc
        result["provenance"] = {"format": self.format, "unit": unit,
            "currency": lower_scope["currency"]}
        return result

    def _anchor_scope(self, scope: Mapping, tag: str) -> str:
        if not isinstance(scope, Mapping):
            raise RungAllowabilityError(f"{tag} quality anchor scope must be a mapping")
        unit = scope.get("unit")
        currency = scope.get("currency")
        if not isinstance(unit, str) or not unit:
            raise RungAllowabilityError(f"{tag} quality anchor scope needs a unit")
        if not isinstance(currency, str) or not currency:
            raise RungAllowabilityError(f"{tag} quality anchor scope needs a currency")
        if scope.get("validated") is not True:
            raise RungAllowabilityError(
                f"{tag} quality anchor is not validated; sampled error waits")
        family = scope.get("family", self.format)
        if family != self.format:
            raise RungAllowabilityError(f"{tag} quality anchor belongs to another format")
        return unit

    def _anchor_calibration(self, scope: Mapping) -> object:
        return scope.get("calibration", None)

    def provenance(self) -> dict:
        return {"schema": self._table["schema"], "format": self.format, "kernel_build": dict(self.kernel_build),
                "table_version": self.table_version, "source_path": self.source_path}


def _selected_table(root, index, family, expected_kernel_build, producer):
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
    if (selected["table_schema"] != table["schema"]
            or table["table_version"] != version
            or table["table_status"] != selected["table_status"]):
        raise RungAllowabilityError("selected table version/schema/status differs from index.json")
    # Source commits and diagnostic metadata are provenance, not new D32 seals.
    context_fields = ("id", "library_variant", "architecture", "activation_contract")
    if table["format"] != family or any(
            not isinstance(expected_kernel_build.get(key), str)
            or table["kernel_build"][key] != expected_kernel_build[key] for key in context_fields):
        raise RungAllowabilityError("selected table format/kernel_build is stale for this allocation")
    return version, path, table


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
    version, path, table = _selected_table(root, index, family, expected_kernel_build, producer)
    rule = _allowable_rung_tables(format_entry, f"formats[{family}]")
    if not rule:
        raise RungAllowabilityError(f"{family}: existing v11 allowable_rungs rule required")
    if table["scope"]["grid_step_q256"] != format_entry["allowable_rungs"]["step_q256"]:
        raise RungAllowabilityError("table step differs from the published true q256 grid step")
    current_index = read_allowability_json(root / "index.json")
    producer.validate_index(current_index)
    if current_index != index:
        _, _, current_table = _selected_table(
            root, current_index, family, expected_kernel_build, producer)
        scoped = producer.admit_rung(table, format=family,
            kernel_build_id=expected_kernel_build["id"], rung=table["scope"]["rung_min"],
            scope=current_table["scope"])
        if scoped["reason"] == "unmeasured_scope":
            raise RungAllowabilityError("stored measurement scope differs from the current selected scope")
        seal_check("D41 publication identity", index, current_index,
                   where=str(root / "index.json"), refusal=RungAllowabilityError)
    return RungAllowability(family, MappingProxyType(dict(expected_kernel_build)), version,
                           str(path), table, producer, MappingProxyType(rule),
                           frozenset(row["rung"] for row in table["rungs"]))
