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

from .digests import DIRECT_ASCII_SPACED_LAX, DIRECT_ASCII_SPACED_STRICT
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


def owner_for_format(owners: Mapping | None, name: str):
    if owners is None:
        return None
    from .tessera_formats import parse_tessera_format_name
    parsed = parse_tessera_format_name(name)
    return None if parsed is None else owners.get(parsed[0].name)


QUALITY_COORDINATES = ("unit", "family", "currency", "calibration", "teacher", "window",
                       "objective", "shape", "source_weight", "activation_contract")
CANONICAL_CHORD_SOURCE = "canonical_qualified_chord"


def _quality_anchor_scope(scope: Mapping, tag: str, family: str) -> str:
    """Validate the complete scientific coordinates, not a provenance label."""
    from .schemas import Contract
    from .joint_aura import JOINT_CURRENCY, identity_sha256
    contract = Contract(RungAllowabilityError, f"{tag} quality anchor: ")
    contract.mapping(scope, where="quality_scope")
    required = (*QUALITY_COORDINATES, "format", "validated", "probe_ids",
                "probe_identity", "probe_identity_sha256")
    missing = [key for key in required if key not in scope or scope[key] is None]
    contract.require(not missing, f"incomplete quality_scope coordinates: {missing}")
    unit = contract.string(scope["unit"], where="unit")
    contract.require(scope["family"] == family, "anchor belongs to another format family")
    contract.require(scope["currency"] in ("served_kl", JOINT_CURRENCY),
                     "currency is a screen, not a qualified loss")
    contract.require(scope["validated"] is True, "anchor is not validated; sampled error waits")
    contract.sha256(scope["calibration"], where="calibration")
    contract.sha256(scope["teacher"], where="teacher")
    contract.string(scope["format"], where="format")
    shape = scope["shape"]
    contract.require(isinstance(shape, (list, tuple)) and len(shape) == 2,
                     "shape requires two actual dimensions")
    for dim in shape:
        contract.integer(dim, where="shape dimension", minimum=1)
    source = contract.mapping(scope["source_weight"], where="source_weight")
    contract.require(source.get("shape") == list(shape), "source_weight differs from shape")
    contract.sha256(source.get("content_sha256"), where="source_weight content")
    contract.mapping(scope["activation_contract"], where="activation_contract")
    window = contract.mapping(scope["window"], where="window")
    probe = contract.mapping(scope["probe_identity"], where="probe_identity")
    for axis in ("calibration_shape", "token_scope", "temperature"):
        contract.require(axis in window and axis in probe and window[axis] == probe[axis],
                         f"window {axis} differs from actual probe coordinates")
    dimensions = window["calibration_shape"]
    contract.require(isinstance(dimensions, (list, tuple)) and len(dimensions) == 2,
                     "window calibration_shape requires two dimensions")
    for dim in dimensions:
        contract.integer(dim, where="window dimension", minimum=1)
    contract.string(window["token_scope"], where="token_scope")
    import math
    contract.require(type(window["temperature"]) in (int, float)
                     and math.isfinite(window["temperature"]) and window["temperature"] > 0,
                     "temperature must be finite and positive")
    contract.require(probe.get("calibration_sha256") == scope["calibration"],
                     "calibration differs from actual probe coordinates")
    teacher = contract.mapping(probe.get("source_model"), where="probe teacher")
    contract.require(teacher.get("content_sha256") == scope["teacher"],
                     "teacher differs from actual probe coordinates")
    objective = contract.mapping(scope["objective"], where="objective")
    actual_objective = probe.get("objective")
    contract.require(isinstance(actual_objective, (str, Mapping)) and bool(actual_objective),
                     "actual objective coordinate is missing")
    expected_objective = {"currency": scope["currency"],
                          "normalization": probe.get("normalization"),
                          "objective": probe.get("objective")}
    contract.require(objective == expected_objective and isinstance(probe.get("normalization"), str),
                     "objective differs from actual probe coordinates")
    for axis in ("seed_base", "n_probes", "distribution", "normalization"):
        contract.require(axis in probe and probe[axis] is not None,
                         f"probe coordinate {axis} is missing")
    seed = contract.integer(probe["seed_base"], where="seed_base", minimum=0)
    count = contract.integer(probe["n_probes"], where="n_probes", minimum=1)
    contract.require(scope["probe_ids"] == list(range(seed, seed + count)),
                     "sample coordinates differ from actual probe ids")
    contract.sha256(scope["probe_identity_sha256"], where="probe identity digest")
    contract.require(identity_sha256(probe) == scope["probe_identity_sha256"],
                     "probe identity digest differs from its own actual data")
    return unit


def require_quality_scope_alignment(left: Mapping, right: Mapping, *, where: str,
                                    same_unit: bool = True) -> None:
    """Keep current scientific facts strict and use the existing D32 split."""
    from .joint_aura import _require_probe_alignment
    for axis in QUALITY_COORDINATES:
        if not same_unit and axis in ("unit", "shape", "source_weight", "activation_contract"):
            continue
        if left[axis] != right[axis]:
            raise RungAllowabilityError(f"quality anchors name different {axis}")
    _require_probe_alignment(left, right, where=where,
        message="quality anchors name different probe sample coordinates")


def qualified_cost_scope(row: Mapping, *, family: str, unit: str | None = None,
                         format_name: str | None = None) -> Mapping | None:
    """Validate the measured row before any optional scope can be read."""
    from .joint_aura import validate_joint_aura_entry, _require_probe_alignment
    is_joint = validate_joint_aura_entry(row)
    scope = row.get("quality_scope")
    if scope is None:
        if is_joint:
            raise RungAllowabilityError("joint quality anchor requires complete quality_scope")
        return None
    actual_unit = _quality_anchor_scope(scope, "cost", family)
    if unit is not None and actual_unit != unit:
        raise RungAllowabilityError("quality anchor differs from the actual unit")
    if format_name is not None and scope["format"] != format_name:
        raise RungAllowabilityError("quality anchor differs from its actual format key")
    if is_joint:
        from .allocator_candidates import joint_row_binds_cell
        joint_row_binds_cell(row, actual_unit, scope["format"], where="canonical quality anchor")
        operator, probe = row["joint_operator_identity"], row["probe_identity"]
        expected = {"shape": operator["source_weight"]["shape"],
                    "source_weight": operator["source_weight"],
                    "activation_contract": operator["activation"],
                    "currency": row["cost_currency"],
                    "calibration": probe["calibration_sha256"],
                    "teacher": probe["source_model"]["content_sha256"]}
        for axis, value in expected.items():
            if scope[axis] != value:
                raise RungAllowabilityError(f"joint quality_scope {axis} differs from actual row")
        _require_joint_anchor(scope, row["predicted_dloss"])
        evidence = scope["joint_anchor"]
        for axis in ("probe_ids", "signed_per_probe", "x2_per_probe", "signed_components_per_probe"):
            if evidence[axis] != row[axis]:
                raise RungAllowabilityError(f"joint quality_scope {axis} differs from its actual samples")
        if evidence["joint_operator_identity"]["rendered_weight"] != operator["rendered_weight"]:
            raise RungAllowabilityError("joint quality_scope differs from actual rendered bytes")
        _require_probe_alignment(scope, row, where="joint quality_scope binding",
            message="joint quality_scope differs from actual probe sample coordinates")
    return scope


def _require_joint_anchor(scope: Mapping, value) -> None:
    """Bind a joint scope and scalar to actual validated joint samples."""
    from .joint_aura import JOINT_CURRENCY, validate_joint_aura_entry, _require_probe_alignment
    if scope["currency"] != JOINT_CURRENCY:
        return
    original = scope.get("joint_anchor")
    if not isinstance(original, Mapping) or not validate_joint_aura_entry(original):
        raise RungAllowabilityError("joint quality_scope requires its actual validated joint_anchor")
    from .allocator_candidates import joint_row_binds_cell
    joint_row_binds_cell(original, scope["unit"], scope["format"], where="joint quality_scope evidence")
    operator, probe = original["joint_operator_identity"], original["probe_identity"]
    expected = {"shape": operator["source_weight"]["shape"],
                "source_weight": operator["source_weight"], "activation_contract": operator["activation"],
                "calibration": probe["calibration_sha256"],
                "teacher": probe["source_model"]["content_sha256"]}
    for axis, actual in expected.items():
        if scope[axis] != actual:
            raise RungAllowabilityError(f"joint quality_scope {axis} differs from actual joint_anchor")
    _require_probe_alignment(scope, original, where="joint anchor evidence",
        message="joint anchor differs from actual probe sample coordinates")
    if value != original["predicted_dloss"]:
        raise RungAllowabilityError("joint anchor scalar differs from its actual aligned samples")


def qualified_rung_quality(producer, family: str, rung: int, *, lower_rung: int,
                           upper_rung: int, lower_value: float, upper_value: float,
                           lower_scope: Mapping, upper_scope: Mapping) -> dict:
    """Use the producer chord in the anchors' unchanged scientific quantity."""
    unit = _quality_anchor_scope(lower_scope, "lower", family)
    _quality_anchor_scope(upper_scope, "upper", family)
    from .tessera_formats import parse_tessera_format_name
    for rate, scope in ((lower_rung, lower_scope), (upper_rung, upper_scope)):
        parsed = parse_tessera_format_name(scope["format"])
        if parsed is None or parsed[0].name != family or parsed[1] != rate:
            raise RungAllowabilityError("quality anchor differs from its actual format and rate")
    require_quality_scope_alignment(lower_scope, upper_scope, where="canonical quality chord")
    for value, tag in ((lower_value, "lower"), (upper_value, "upper")):
        if type(value) is bool or not isinstance(value, (int, float)):
            raise RungAllowabilityError(f"{tag} quality anchor value must be a number")
    _require_joint_anchor(lower_scope, lower_value)
    _require_joint_anchor(upper_scope, upper_value)
    if not callable(getattr(producer, "rung_quality", None)):
        raise RungAllowabilityError("canonical qualified quality capability required")
    try:
        result = dict(producer.rung_quality(rung, lower_rung=lower_rung, upper_rung=upper_rung,
            lower_value=lower_value, upper_value=upper_value))
    except ValueError as exc:
        raise RungAllowabilityError(str(exc)) from exc
    target_scope = {**lower_scope, "format": f"{family}_R{rung}"}
    result["provenance"] = {"format": family, "unit": unit, "currency": lower_scope["currency"],
                            "quality_scope": target_scope}
    return result


def require_quality_result_matches(actual: Mapping, expected: Mapping, *, where: str) -> None:
    """Compare current numeric claims without a provenance-only digest wall."""
    for key in ("status", "value", "anchors", "fraction", "numerical_qualification_inherited"):
        if type(actual.get(key)) is not type(expected.get(key)) or actual.get(key) != expected.get(key):
            raise RungAllowabilityError(f"{where}: canonical quality differs from its actual bound anchors")
    left, right = actual["provenance"], expected.get("provenance", {})
    for key in ("format", "unit", "currency"):
        if left[key] != right.get(key):
            raise RungAllowabilityError(f"{where}: canonical quantity differs from actual {key}")
    scope = right.get("quality_scope")
    _quality_anchor_scope(scope, "recorded", left["format"])
    if scope["format"] != left["quality_scope"]["format"]:
        raise RungAllowabilityError(f"{where}: recorded quality differs from actual format")
    require_quality_scope_alignment(left["quality_scope"], scope, where=where)


CANONICAL_SUM_SOURCE = "canonical_qualified_sum"


def make_scientific_price_sum(members: list, *, format_name: str) -> dict | None:
    """Sum complete objective quantities without another scalar transfer."""
    prices = [complete_scientific_price(item["row"], format_name=item["format"])
              if isinstance(item["row"], Mapping) else None for item in members]
    if not any(value is not None for value in prices):
        return None
    if any(value is None for value in prices):
        raise RungAllowabilityError("scientific sum cannot mix complete and unqualified quantities")
    reference = members[0]["row"]["quality_scope"]
    for item in members:
        scope = item["row"]["quality_scope"]
        if scope["unit"] != item["unit"] or scope["format"] != item["format"]:
            raise RungAllowabilityError("scientific sum member differs from its actual unit or format")
        require_quality_scope_alignment(reference, scope, where="scientific sum", same_unit=False)
    return {"predicted_dloss": sum(prices), "cost_source": CANONICAL_SUM_SOURCE,
            "cost_currency": reference["currency"], "format": format_name,
            "canonical_members": members}


def complete_scientific_price(row: Mapping, *, format_name: str | None = None) -> float | None:
    """Read a complete objective price without gain or activation transfer."""
    from .tessera_formats import parse_tessera_format_name
    if row.get("cost_source") == CANONICAL_SUM_SOURCE or "canonical_members" in row:
        if row.get("cost_source") != CANONICAL_SUM_SOURCE or not row.get("canonical_members"):
            raise RungAllowabilityError("scientific sum requires its actual member quantities")
        actual = make_scientific_price_sum(row["canonical_members"], format_name=row["format"])
        if (actual is None or row.get("predicted_dloss") != actual["predicted_dloss"]
                or row.get("cost_currency") != actual["cost_currency"]
                or (format_name is not None and row["format"] != format_name)):
            raise RungAllowabilityError("scientific sum differs from its actual quantities or format")
        return float(actual["predicted_dloss"])
    claims = row.get("cost_source") == CANONICAL_CHORD_SOURCE or "canonical_quality" in row
    from .joint_aura import validate_joint_aura_entry
    measured_joint = not claims and validate_joint_aura_entry(row)
    scope = row.get("quality_scope")
    if not claims and (not isinstance(scope, Mapping)
                       or (not measured_joint and scope.get("currency") != "served_kl")):
        return None
    if not isinstance(scope, Mapping):
        raise RungAllowabilityError("complete scientific price requires quality_scope")
    parsed = parse_tessera_format_name(scope.get("format"))
    if parsed is None:
        raise RungAllowabilityError("complete scientific price requires an actual format")
    family, rung = parsed
    if format_name is not None and scope["format"] != format_name:
        raise RungAllowabilityError("complete scientific price cannot price a different format")
    _quality_anchor_scope(scope, "price", family.name)
    if claims:
        if row.get("cost_source") != CANONICAL_CHORD_SOURCE:
            raise RungAllowabilityError("canonical price has a foreign cost_source")
        anchors = row.get("canonical_anchors")
        if not isinstance(anchors, (list, tuple)) or len(anchors) != 2:
            raise RungAllowabilityError("canonical price requires its two bound actual anchors")
        prepared = []
        for anchor in anchors:
            name, original = anchor["format"], anchor["row"]
            parsed_anchor = parse_tessera_format_name(name)
            if parsed_anchor is None or parsed_anchor[0].name != family.name:
                raise RungAllowabilityError("canonical price anchor belongs to another family")
            prepared.append((parsed_anchor[1], original,
                qualified_cost_scope(original, family=family.name, unit=scope["unit"], format_name=name)))
        left, right = prepared
        actual = qualified_rung_quality(_producer_api(), family.name, rung,
            lower_rung=left[0], upper_rung=right[0],
            lower_value=left[1]["predicted_dloss"], upper_value=right[1]["predicted_dloss"],
            lower_scope=left[2], upper_scope=right[2])
        require_quality_result_matches(actual, row.get("canonical_quality", {}), where="canonical price")
        require_quality_scope_alignment(actual["provenance"]["quality_scope"], scope,
                                        where="canonical price quantity")
        if row.get("cost_currency") != scope["currency"] or row.get("predicted_dloss") != actual["value"]:
            raise RungAllowabilityError("canonical price differs from its actual scientific quantity")
    else:
        qualified_cost_scope(row, family=family.name, unit=scope["unit"], format_name=scope["format"])
    value = row.get("predicted_dloss")
    import math
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RungAllowabilityError("complete scientific price must be finite and nonnegative")
    return float(value)



def _producer_api():
    try:
        from tessera import rung_allowability
    except ImportError as exc:
        raise RungAllowabilityError("canonical tessera.rung_allowability producer admission required") from exc
    missing = [name for name in PRODUCER_API[:3]
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
            recipe_json = DIRECT_ASCII_SPACED_STRICT.text(recipe)
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
    _times: dict = field(default_factory=dict, repr=False, compare=False)
    index_source_path: str = ""

    def scope_for_unit(self, format_name: str, *, unit: str, shape,
                       structure: str | None, m: int | None = None,
                       tensor_parallel: int = 1, routing: str | None = None) -> dict | None:
        """Retain actual geometry and the independently observed activation build."""
        if self._table["schema"] != V3_SCHEMA:
            return None
        from .measured_runtime_prices import rank_local_member_shapes
        from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe
        family, rung = parse_tessera_format_name(format_name)
        if family.name != self.format:
            raise RungAllowabilityError("unit scope belongs to another format")
        shape = tuple(shape)
        if len(shape) == 3 and structure == "routed_moe":
            shape = shape[1:]
        if len(shape) != 2:
            return {"activation_contract": self.kernel_build["activation_contract"]}
        if tensor_parallel > 1:
            from .shape_runtime_prices import ShapeRuntimeError, served_operator
            try:
                operator = served_operator({unit: shape}, structure="dense",
                    tensor_parallel=tensor_parallel, where=unit)
            except ShapeRuntimeError as exc:
                if exc.__cause__ is not None:
                    raise
                return {"activation_contract": self.kernel_build["activation_contract"]}
            local = tuple(int(value) for value in operator.split("x"))
        else:
            local = rank_local_member_shapes({unit: shape}, tensor_parallel=tensor_parallel)[unit]
        wire = tessera_served_wire_recipe(family, rung, structure=structure,
                                         refuse_unattested=False)
        recipe = wire.to_config()
        return {"kernel_kind": structure, "rows": local[0], "columns": local[1],
                "m": m, "routing": routing,
                "activation_contract": self.kernel_build["activation_contract"], "recipe": recipe}

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
        call = {}
        if scope_key is not None:
            if self._table.get("schema") != V3_SCHEMA:
                return "scoped_policy_requires_v3_table"
            call["cell_ids"] = list(self._resolve_cells(scope_key))
            if scope_key[5] is not None:
                call["activation_contract"] = scope_key[5]
            if scope_key[6] is not None:
                call["recipe"] = json.loads(scope_key[6])
        decision = self._producer.admit_rung(self._table, format=self.format,
            kernel_build_id=self.kernel_build["id"], rung=rung, **call)
        return self._read_verdict(rung, decision)


    def _read_verdict(self, rung: int, decision: Mapping) -> str:
        status = decision["status"]
        if status not in {"allow", "wait", "hold", "excluded", "unsupported", "failed"}:
            raise RungAllowabilityError("unrecognized canonical producer admission decision")
        if status != "allow":
            reasons = [cell["reason"] for cell in decision.get("cells", ())
                       if cell["status"] != "allow"]
            reason = "; ".join(dict.fromkeys(reasons)) if reasons else decision["reason"]
            if rung not in self._rungs:
                reason = "unlisted rung; " + reason
        else:
            reason = "" if rung in self._rule else "excluded by the published allowable_rungs rule"
        return reason

    def _resolve_cells(self, scope_key: tuple) -> tuple:
        """Resolve explicit scope; an empty result waits and never broadens."""
        if scope_key in self._cells:
            return self._cells[scope_key]
        kind, rows, columns, m, routing, _, _ = scope_key
        resolved = self._resolve_cells_uncached(kind, rows, columns, m, routing)
        self._cells[scope_key] = resolved
        return resolved

    def _resolve_cells_uncached(self, kind, rows, columns, m, routing):
        if kind is None:
            return ()
        producer_kind = _producer_kind(kind)
        required = self._table["scope"]["required_cells"]
        if (rows is None) != (columns is None):
            return ()
        if rows is not None:
            if m is None and routing is not None:
                return ()
            regimes = ([m] if m is not None else sorted({cell["M"] for cell in required
                        if cell["kernel_kind"] == producer_kind}))
            found = []
            for regime in regimes:
                for cell_id in self._producer.scope_cell_ids(
                        self._table, kernel_kind=producer_kind, rows=rows,
                        columns=columns, M=regime,
                        **({} if routing is None else {"routing": routing})):
                    if cell_id not in found:
                        found.append(cell_id)
            return tuple(found)
        if routing is not None:
            return ()
        return tuple(cell["cell_id"] for cell in required
                     if cell["kernel_kind"] == producer_kind
                     and (m is None or cell["M"] == m))

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
        key = (rung, cell_id, None if class_identity is None else
               DIRECT_ASCII_SPACED_LAX.text(dict(class_identity)))
        if key in self._times:
            return self._times[key]
        result = dict(self._producer.rung_speed(
            self._table, rung=rung, cell_id=cell_id,
            class_identity=None if class_identity is None else dict(class_identity)))
        if result.get("status") not in {"measured", "inherited", "wait", "hold",
                                        "unsupported", "failed"}:
            raise RungAllowabilityError("unrecognized canonical producer timing decision")
        result["provenance"] = self.provenance()
        self._times[key] = result
        return result

    @property
    def scoped(self) -> bool:
        return self._table["schema"] == V3_SCHEMA

    def chord_cost(self, format_name: str, *, unit: str, costs: Mapping, fallback=None) -> dict | None:
        """Price mixed rates only from qualified neighbouring whole anchors."""
        from .tessera_formats import parse_tessera_format_name
        family, rung = parse_tessera_format_name(format_name)
        if not self.scoped or family.root_rate(rung).denominator == 1:
            return costs.get(format_name, fallback)
        anchors = []
        for name, row in costs.items():
            parsed = parse_tessera_format_name(name)
            if parsed is None or parsed[0].name != self.format:
                continue
            rate = parsed[1]
            if family.root_rate(rate).denominator == 1:
                anchor_scope = qualified_cost_scope(row, family=self.format,
                                                    unit=unit, format_name=name)
                if anchor_scope is not None:
                    anchors.append((rate, row, anchor_scope))
        lower = max((item for item in anchors if item[0] < rung), default=None, key=lambda x: x[0])
        upper = min((item for item in anchors if item[0] > rung), default=None, key=lambda x: x[0])
        if lower is None or upper is None:
            return None
        for _rate, row, anchor_scope in (lower, upper):
            if anchor_scope.get("unit") != unit:
                raise RungAllowabilityError("quality anchor differs from the actual unit")
        result = self.chord_quality(rung, lower_rung=lower[0], upper_rung=upper[0],
            lower_value=lower[1]["predicted_dloss"], upper_value=upper[1]["predicted_dloss"],
            lower_scope=lower[2], upper_scope=upper[2])
        return {"predicted_dloss": result["value"], "canonical_quality": result,
                "quality_scope": result["provenance"]["quality_scope"],
                "canonical_anchors": [{"format": anchor[2]["format"], "row": anchor[1]}
                                      for anchor in (lower, upper)],
                "cost_currency": lower[2]["currency"],
                "cost_source": CANONICAL_CHORD_SOURCE, "output_mse_measured": False}

    def chord_quality(self, rung: int, *, lower_rung: int, upper_rung: int,
                      lower_value: float, upper_value: float,
                      lower_scope: Mapping, upper_scope: Mapping) -> dict:
        return qualified_rung_quality(self._producer, self.format, rung,
            lower_rung=lower_rung, upper_rung=upper_rung,
            lower_value=lower_value, upper_value=upper_value,
            lower_scope=lower_scope, upper_scope=upper_scope)


    def provenance(self) -> dict:
        return {"schema": self._table["schema"], "format": self.format, "kernel_build": dict(self.kernel_build),
                "table_version": self.table_version, "source_path": self.source_path,
                "index_source_path": self.index_source_path}


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
    index_path = root if root.is_file() else root / "index.json"
    root = index_path.parent
    family = format_entry["family"]
    if not isinstance(expected_kernel_build, Mapping) or not expected_kernel_build.get("id"):
        raise RungAllowabilityError("independently observed kernel_build is required")
    index = read_allowability_json(index_path)
    producer = _producer_api()
    producer.validate_index(index)
    version, path, table = _selected_table(root, index, family, expected_kernel_build, producer)
    if table["schema"] == V3_SCHEMA:
        missing = [name for name in PRODUCER_API[3:]
                   if not callable(getattr(producer, name, None))]
        if missing:
            raise RungAllowabilityError(f"canonical v3 capabilities required: {', '.join(missing)}")
    rule = _allowable_rung_tables(format_entry, f"formats[{family}]")
    if not rule:
        raise RungAllowabilityError(f"{family}: existing v11 allowable_rungs rule required")
    if table["scope"]["grid_step_q256"] != format_entry["allowable_rungs"]["step_q256"]:
        raise RungAllowabilityError("table step differs from the published true q256 grid step")
    current_index = read_allowability_json(index_path)
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
                   where=str(index_path), refusal=RungAllowabilityError)
    return RungAllowability(family, MappingProxyType(dict(expected_kernel_build)), version,
                           str(path), table, producer, MappingProxyType(rule),
                           frozenset(row["rung"] for row in table["rungs"]),
                           index_source_path=str(index_path))
