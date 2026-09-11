"""Seal a complete measured rate curve from one campaign journal.

The campaign journal is already the durable receipt boundary: its manifest
binds the run identity, and each checksummed unit shard contains an anchor only
after its wire was written and read back. This collector verifies that
existing boundary, checks every row against the frozen measurement plan, and
then writes one small JSON receipt per measured rung for the adaptive replay.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA,
    _load_unit,
    canonical_json_sha256,
    unit_path,
)
from prismaquant.tessera_campaign import CampaignAnchor, CURRENCY as CAMPAIGN_CURRENCY


SCHEMA = "prismaquant.complete_rate_curve.v1"
PLAN_SCHEMA = "prismaquant.complete_rate_measurement_plan.v1"
POINT_SCHEMA = "prismaquant.complete_rate_point_receipt.v1"
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
AUDIT_REGION_CORRECTION_SCHEMA = "prismaquant.complete_rate_audit_region_correction.v1"
AUDIT_REGION_CORRECTION_KEYS = frozenset({
    "schema",
    "original_plan_path",
    "original_plan_sha256",
    "corrected_audit_regions",
})

# CampaignAnchor records the serving route's dtype name. The study plan uses a
# more explicit semantic spelling. Keep this translation closed so a new route
# cannot silently inherit one of the study's meanings.
SEMANTIC_ACTIVATION_BY_ROUTE = {
    "w16a16-bf16-channel": "bf16_unquantized",
    "w4a4-nvfp4-e2m1-group16-ue4m3": "e2m1_group16_ue4m3_static",
    "w8a8-dynamic-e4m3-channel": "fp8_per_token_dynamic",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _fail(where: object, message: str) -> None:
    raise RuntimeError(f"{where}: {message}")


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _campaign_source_sha256() -> str:
    import prismaquant.tessera_campaign as campaign

    return sha256(Path(campaign.__file__).resolve(strict=True))


def _bound_json_reference(owner: Mapping, path_key: str, sha_key: str, where: Path) -> dict:
    raw_path, expected_sha256 = owner.get(path_key), owner.get(sha_key)
    if not isinstance(raw_path, str) or not raw_path or not _sha(expected_sha256):
        _fail(where, f"{path_key} reference is incomplete")
    path = Path(raw_path).resolve(strict=True)
    if sha256(path) != expected_sha256:
        _fail(path, f"bytes differ from {sha_key}")
    return _json(path)


def _validate_audit_regions(
    regions: object,
    legal_rates: list[int],
    where: object,
    *,
    allow_outside: bool,
) -> dict[str, list[int]]:
    if not isinstance(regions, dict):
        _fail(where, "measurement plan lacks audit_regions")
    legal = set(legal_rates)
    for label, region in regions.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(region, list)
            or any(type(rate) is not int for rate in region)
            or len(region) != len(set(region))
            or (not allow_outside and not set(region) <= legal)
        ):
            _fail(where, "audit_regions must be named subsets of legal_rates")
    return {label: list(region) for label, region in regions.items()}


def effective_audit_regions(
    plan: Mapping,
    plan_path: Path,
    correction_path: "Path | None" = None,
) -> tuple[dict[str, list[int]], "dict[str, str] | None"]:
    """Return the plan's regions or one exact, receipt-bound intersection.

    A correction can narrow audit metadata to the frozen plan's legal roster.
    It cannot alter the plan, add rates, rename regions, or carry another
    acquisition field. Empty intersections retain their original labels.
    """
    plan_path = plan_path.resolve(strict=True)
    legal_rates = plan.get("legal_rates")
    if (
        not isinstance(legal_rates, list)
        or not legal_rates
        or any(type(rate) is not int for rate in legal_rates)
        or legal_rates != sorted(set(legal_rates))
    ):
        _fail(plan_path, "legal_rates must be sorted unique integers")
    original = _validate_audit_regions(
        plan.get("audit_regions"), legal_rates, plan_path,
        allow_outside=correction_path is not None,
    )
    if correction_path is None:
        return original, None

    correction_path = correction_path.resolve(strict=True)
    correction = _json(correction_path)
    if correction.get("schema") != AUDIT_REGION_CORRECTION_SCHEMA:
        _fail(correction_path, "unsupported audit-region correction")
    if set(correction) != AUDIT_REGION_CORRECTION_KEYS:
        _fail(
            correction_path,
            f"audit-region correction must contain exactly {sorted(AUDIT_REGION_CORRECTION_KEYS)}",
        )
    original_path = correction.get("original_plan_path")
    if not isinstance(original_path, str) or not Path(original_path).is_absolute():
        _fail(correction_path, "original_plan_path must be an absolute path")
    try:
        bound_path = Path(original_path).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{correction_path}: original plan path cannot be resolved") from exc
    if bound_path != plan_path:
        _fail(correction_path, "original plan path differs from the collected plan")
    original_sha256 = correction.get("original_plan_sha256")
    if not _sha(original_sha256) or sha256(plan_path) != original_sha256:
        _fail(correction_path, "original plan bytes differ from original_plan_sha256")

    legal = set(legal_rates)
    expected = {
        label: [rate for rate in region if rate in legal]
        for label, region in original.items()
    }
    if correction.get("corrected_audit_regions") != expected:
        _fail(
            correction_path,
            "corrected_audit_regions must be the exact intersection of each original region with legal_rates",
        )
    _validate_audit_regions(expected, legal_rates, correction_path, allow_outside=False)
    return expected, {"path": str(correction_path), "sha256": sha256(correction_path)}


def _validate_plan(
    plan: dict,
    path: Path,
    *,
    allow_outside_audit_regions: bool = False,
) -> tuple[str, str, list[int]]:
    if plan.get("schema") != PLAN_SCHEMA:
        _fail(path, "unsupported measurement plan")
    if plan.get("status") != "frozen_before_measurement":
        _fail(path, "measurement plan was not frozen before measurement")
    if plan.get("measurement_kind") != "measured":
        _fail(path, "measurement plan does not request measured values")
    if plan.get("currency") != CAMPAIGN_CURRENCY:
        _fail(path, "measurement plan has the wrong cost currency")
    for name in ("curve_id", "qname", "family", "activation_contract", "menu_mode"):
        if not isinstance(plan.get(name), str) or not plan[name]:
            _fail(path, f"measurement plan lacks {name}")
    for name in ("source_identity", "calibration_identity", "recipe_identity", "family_restriction"):
        if not isinstance(plan.get(name), dict) or not plan[name]:
            _fail(path, f"measurement plan lacks {name}")
    qshape = plan.get("qshape")
    if (
        not isinstance(qshape, list)
        or len(qshape) != 2
        or any(type(value) is not int or value <= 0 for value in qshape)
    ):
        _fail(path, "measurement plan has an invalid qshape")
    legal_rates = plan.get("legal_rates")
    if (
        not isinstance(legal_rates, list)
        or not legal_rates
        or any(type(rate) is not int for rate in legal_rates)
        or legal_rates != sorted(set(legal_rates))
    ):
        _fail(path, "legal_rates must be sorted unique integers")
    _validate_audit_regions(
        plan.get("audit_regions"), legal_rates, path,
        allow_outside=allow_outside_audit_regions,
    )
    source = plan["source_identity"]
    producer = source.get("producer") if isinstance(source, Mapping) else None
    campaign_source = source.get("campaign_source") if isinstance(source, Mapping) else None
    weight = source.get("weight_identity") if isinstance(source, Mapping) else None
    if (
        not isinstance(source.get("model"), str)
        or not source["model"]
        or not isinstance(producer, Mapping)
        or not _sha(producer.get("source_sha256"))
        or not isinstance(campaign_source, Mapping)
        or not _sha(campaign_source.get("campaign_module_sha256"))
        or not isinstance(weight, Mapping)
        or not _sha(weight.get("source_map_sha256"))
        or not _sha(weight.get("state_sha256"))
    ):
        _fail(path, "source_identity is incomplete")
    if campaign_source["campaign_module_sha256"] != _campaign_source_sha256():
        _fail(path, "campaign source bytes differ from the frozen plan")
    pin = _bound_json_reference(producer, "pin", "pin_sha256", path)
    pin_encoder = pin.get("sources", {}).get("encoder")
    if not isinstance(pin_encoder, Mapping) or pin_encoder.get("commit") != producer.get("commit"):
        _fail(path, "producer pin does not bind the planned encoder commit")

    calibration = plan["calibration_identity"]
    capture = _bound_json_reference(
        calibration, "capture_manifest", "capture_manifest_sha256", path
    )
    census = _bound_json_reference(calibration, "census", "census_sha256", path)
    capture_identity = capture.get("identity")
    if not isinstance(capture_identity, Mapping):
        _fail(path, "capture manifest lacks an identity")
    load_contract = capture_identity.get("model_load_contract")
    census_load_contract = census.get("model_load_contract")
    if (
        not isinstance(load_contract, Mapping)
        or not isinstance(census_load_contract, Mapping)
        or any(load_contract.get(key) != value for key, value in weight.items())
        or any(census_load_contract.get(key) != value for key, value in weight.items())
        or capture_identity.get("census_sha256") != calibration["census_sha256"]
        or census.get("model") != source["model"]
        or census.get("attention_implementation") != calibration.get("attention_implementation")
        or census.get("nsamples") != calibration.get("nsamples")
        or census.get("seed") != calibration.get("seed")
        or census.get("seqlen") != calibration.get("seqlen")
    ):
        _fail(path, "source/calibration references differ from the frozen plan")
    unit_shapes = census.get("unit_shapes")
    if not isinstance(unit_shapes, Mapping) or list(unit_shapes.get(plan["qname"], ())) != qshape:
        _fail(path, "census unit shape differs from the frozen plan")
    return plan["qname"], plan["family"], list(legal_rates)


def _validate_campaign_menu(
    menu: object, *, settings: Mapping, plan: Mapping, where: object
) -> None:
    """Bind an exhaustive campaign band's typed roster to the frozen plan.

    Campaign identities record the full family menu even when pricing is
    restricted to a rate band.  Every menu entry still has to be a valid,
    unique member of the planned family; only its intersection with the
    recorded band is the curve roster.
    """
    from prismaquant.tessera_campaign import parse_rate_band
    from prismaquant.tessera_formats import TesseraFormatError, parse_tessera_format_name

    legal_rates = plan["legal_rates"]
    expected_band = (legal_rates[0], legal_rates[-1])
    try:
        rate_band = parse_rate_band(settings.get("rate_band"))
    except RuntimeError as exc:
        raise RuntimeError(f"{where}: campaign rate band is malformed") from exc
    if rate_band != expected_band:
        _fail(where, "campaign rate band differs from the frozen legal-rate bounds")
    if settings.get("exhaustive_rate_grid") is not True:
        _fail(where, "complete rate curve campaign was not exhaustive")
    if type(settings.get("anchor_budget")) is not int or settings["anchor_budget"] != len(legal_rates):
        _fail(where, "campaign anchor budget differs from the frozen legal-rate roster")
    if type(settings.get("max_rounds")) is not int or settings["max_rounds"] != 1:
        _fail(where, "complete rate curve campaign did not use exactly one round")
    if not isinstance(menu, list):
        _fail(where, "campaign unit menu is not a list")

    seen: set[tuple[str, int]] = set()
    in_band = []
    for index, entry in enumerate(menu):
        try:
            parsed = parse_tessera_format_name(entry)
        except (TesseraFormatError, ValueError) as exc:
            raise RuntimeError(f"{where}: campaign menu entry {index} is invalid") from exc
        if parsed is None:
            _fail(where, f"campaign menu entry {index} is not a Tessera format")
        family, rate = parsed
        key = (family.name, rate)
        if key in seen:
            _fail(where, f"campaign unit menu repeats {family.name}_R{rate}")
        seen.add(key)
        if family.name != plan["family"]:
            _fail(where, f"campaign unit menu contains family {family.name} outside the plan")
        if rate_band[0] <= rate <= rate_band[1]:
            in_band.append(rate)

    if sorted(in_band) != legal_rates:
        missing = sorted(set(legal_rates) - set(in_band))
        extra = sorted(set(in_band) - set(legal_rates))
        _fail(
            where,
            f"campaign in-band menu differs from the frozen legal-rate roster; "
            f"missing={missing[:8]} extra={extra[:8]}",
        )


def _load_campaign_unit(checkpoint: Path, plan: dict, qname: str):
    manifest = _json(checkpoint)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("stage") != "Tessera campaign":
        _fail(checkpoint, "unsupported campaign checkpoint")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        _fail(checkpoint, "campaign checkpoint lacks an identity")
    identity_sha256 = canonical_json_sha256(identity, where="complete rate curve campaign identity")
    if manifest.get("identity_sha256") != identity_sha256:
        _fail(checkpoint, "campaign checkpoint identity digest differs")
    if (
        identity.get("campaign_schema") != CAMPAIGN_SCHEMA
        or identity.get("currency") != plan["currency"]
    ):
        _fail(checkpoint, "campaign schema or currency differs from the measurement plan")

    roster = manifest.get("units")
    if not isinstance(roster, list):
        _fail(checkpoint, "campaign checkpoint lacks a unit roster")
    roster_names = []
    for row in roster:
        if not isinstance(row, Mapping) or not isinstance(row.get("qname"), str):
            _fail(checkpoint, "campaign checkpoint unit roster is malformed")
        roster_names.append(row["qname"])
    units = identity.get("units")
    if (
        len(roster_names) != len(set(roster_names))
        or not isinstance(units, Mapping)
        or set(roster_names) != set(units)
        or qname not in units
    ):
        _fail(checkpoint, "campaign manifest and identity unit rosters differ")

    source = plan["source_identity"]
    settings = identity.get("settings")
    family_binding = identity.get("family_restriction")
    if not isinstance(settings, Mapping):
        _fail(checkpoint, "campaign identity lacks settings")
    if settings.get("model") != source["model"]:
        _fail(checkpoint, "campaign model differs from the measurement plan")
    if identity.get("encoder_source_sha256") != source["producer"]["source_sha256"]:
        _fail(checkpoint, "campaign producer source differs from the measurement plan")
    if (
        settings.get("family_restriction") != plan["family_restriction"]
        or not isinstance(family_binding, Mapping)
        or family_binding.get("policy") != plan["family_restriction"]
        or family_binding.get("structure_by_unit", {}).get(qname) != "dense"
    ):
        _fail(checkpoint, "campaign family restriction differs from the measurement plan")
    if settings.get("menu_mode") != plan["menu_mode"]:
        _fail(checkpoint, "campaign menu mode differs from the measurement plan")
    calibration = plan["calibration_identity"]
    for plan_name, setting_name in (
        ("attention_implementation", "attention_implementation"),
        ("hessian", "hessian"),
        ("max_act_rows", "max_act_rows"),
        ("nsamples", "nsamples"),
        ("seed", "seed"),
        ("seqlen", "seqlen"),
        ("tp_degree", "tp_degree"),
    ):
        if settings.get(setting_name) != calibration.get(plan_name):
            _fail(checkpoint, f"campaign {setting_name} differs from the measurement plan")
    if settings.get("streaming") is not True:
        _fail(checkpoint, "complete rate curve campaign was not streamed")
    capture = _json(Path(calibration["capture_manifest"]).resolve(strict=True))
    capture_identity = capture.get("identity")
    if (
        not isinstance(capture_identity, Mapping)
        or identity.get("calibration") != capture_identity.get("calibration")
    ):
        _fail(checkpoint, "campaign calibration identity differs from the frozen capture")

    unit = units[qname]
    if not isinstance(unit, Mapping) or not isinstance(unit.get("weight"), Mapping):
        _fail(checkpoint, "campaign unit lacks a weight identity")
    if list(unit["weight"].get("shape", ())) != plan["qshape"]:
        _fail(checkpoint, "campaign unit shape differs from the measurement plan")
    menu = unit.get("menu")
    _validate_campaign_menu(menu, settings=settings, plan=plan, where=checkpoint)

    shard = unit_path(checkpoint.with_name(checkpoint.name + ".parts"), qname)
    state = _load_unit(
        shard,
        stage="Tessera campaign",
        qname=qname,
        identity_sha256=identity_sha256,
    )
    return manifest, identity, dict(unit), shard, state


def _expected_row(plan: dict, rate: int):
    from prismaquant.tessera_formats import (
        parse_tessera_format_name,
        route_static_activation_contract,
        tessera_serving_route,
        tessera_wire_recipe,
    )
    from prismaquant.tessera_render import rung_accepts_hessian

    format_name = f"{plan['family']}_R{rate}"
    parsed = parse_tessera_format_name(format_name)
    if parsed is None:
        _fail(plan["curve_id"], f"unsupported Tessera format {format_name}")
    family, parsed_rate = parsed
    if family.name != plan["family"] or parsed_rate != rate:
        _fail(plan["curve_id"], f"format grammar differs at rate {rate}")
    wire = tessera_wire_recipe(family, parsed_rate)
    route = tessera_serving_route(family, wire, parsed_rate)
    semantic_activation = SEMANTIC_ACTIVATION_BY_ROUTE.get(route.contract)
    if semantic_activation is None or plan["activation_contract"] != semantic_activation:
        _fail(plan["curve_id"], f"activation contract differs at rate {rate}")
    # ``WireRecipe`` is the invariant body/plane grammar. The cached producer
    # receipt adds the requested root rate beside that grammar.
    expected_recipe = {
        "grid": family.payload_grid().name,
        "q256": parsed_rate,
        **wire.to_config(),
    }
    recipe_identity = plan["recipe_identity"]
    if (
        recipe_identity.get("grid") != family.base
        or recipe_identity.get("body") != expected_recipe["body"]
        or recipe_identity.get("scale_plane") != expected_recipe["plane"]
        or recipe_identity.get("activation_contract") != semantic_activation
        or recipe_identity.get("terminal", False) != (expected_recipe["body"] == "tcq")
    ):
        _fail(plan["curve_id"], f"wire recipe differs from the frozen plan at rate {rate}")
    static_contract = route_static_activation_contract(route)
    return {
        "format_name": format_name,
        "family": family.name,
        "rate": parsed_rate,
        "producer_activation": route.act_dtype_name,
        "activation_quantized": route.act_bits is not None and route.act_bits < 16,
        "hessian_applied": (
            plan["calibration_identity"].get("hessian") == "require"
            and rung_accepts_hessian(format_name, wire)
        ),
        "uses_static_scale": static_contract is not None,
        "wire_recipe": expected_recipe,
    }


def _anchor_rows(state: dict, plan: dict, unit: dict) -> dict[int, tuple[CampaignAnchor, dict]]:
    anchors = state.get("anchors")
    wires = state.get("wire_records")
    if set(state) != {"anchors", "wire_records"} or not isinstance(anchors, list) or not isinstance(wires, dict):
        _fail(plan["curve_id"], "journal unit has an invalid anchor/record envelope")
    if len(wires) != len(anchors):
        _fail(plan["curve_id"], "journal anchor and wire-record counts differ")
    all_formats = set()
    selected = {}
    for row in anchors:
        if not isinstance(row, dict):
            _fail(plan["curve_id"], "journal anchor is not an object")
        try:
            anchor = CampaignAnchor(**row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{plan['curve_id']}: malformed CampaignAnchor") from exc
        if anchor.qname != plan["qname"] or anchor.format_name in all_formats:
            _fail(plan["curve_id"], "journal anchor has a wrong or duplicate unit/format")
        all_formats.add(anchor.format_name)
        if anchor.format_name not in wires or not isinstance(wires[anchor.format_name], dict):
            _fail(plan["curve_id"], f"anchor {anchor.format_name} lacks a wire record")
        if anchor.family != plan["family"]:
            _fail(plan["curve_id"], f"journal contains family {anchor.family} outside the plan")
        expected = _expected_row(plan, anchor.body_rate_q256)
        if anchor.format_name != expected["format_name"]:
            _fail(plan["curve_id"], f"anchor format disagrees at rate {anchor.body_rate_q256}")
        value = anchor.dloss
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            _fail(plan["curve_id"], f"anchor {anchor.format_name} has an invalid measured value")
        if (
            anchor.activation_contract != expected["producer_activation"]
            or anchor.activation_quantized is not expected["activation_quantized"]
            or anchor.hessian_applied is not expected["hessian_applied"]
        ):
            _fail(plan["curve_id"], f"anchor {anchor.format_name} has the wrong activation/Hessian contract")
        wanted_scale = unit.get("input_global_scale") if expected["uses_static_scale"] else None
        if anchor.input_global_scale != wanted_scale:
            _fail(plan["curve_id"], f"anchor {anchor.format_name} has the wrong static activation scale")
        if type(anchor.wire_bytes) is not int or anchor.wire_bytes <= 0:
            _fail(plan["curve_id"], f"anchor {anchor.format_name} has invalid wire bytes")
        if anchor.body_rate_q256 in selected:
            _fail(plan["curve_id"], f"duplicate measured rate {anchor.body_rate_q256}")
        selected[anchor.body_rate_q256] = (anchor, wires[anchor.format_name])
    if all_formats != set(wires):
        _fail(plan["curve_id"], "journal has wire records outside its measured anchors")
    if sorted(selected) != plan["legal_rates"]:
        missing = sorted(set(plan["legal_rates"]) - set(selected))
        extra = sorted(set(selected) - set(plan["legal_rates"]))
        _fail(plan["curve_id"], f"rate coverage differs; missing={missing[:8]} extra={extra[:8]}")
    return selected


def _verify_wire(
    *,
    anchor: CampaignAnchor,
    record: dict,
    expected: dict,
    qname: str,
    unit: dict,
    identity: dict,
    wire_dir: Path,
) -> Path:
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        _fail(anchor.format_name, "wire filename escapes its directory")
    expected_filename = f"{qname.replace('.', '__')}__{anchor.format_name}.tessera"
    if filename != expected_filename:
        _fail(anchor.format_name, "wire filename differs from the unit/format")
    wire_path = wire_dir / filename
    try:
        if wire_path.is_symlink() or wire_path.resolve(strict=True).parent != wire_dir.resolve(strict=True):
            _fail(anchor.format_name, "wire path escapes its directory")
        stat = wire_path.stat()
    except OSError as exc:
        raise RuntimeError(f"{wire_path}: cannot read measured wire") from exc
    blob_hash = record.get("blob_sha256")
    if (
        not _sha(blob_hash)
        or type(record.get("blob_bytes")) is not int
        or record["blob_bytes"] != anchor.wire_bytes
        or stat.st_size != anchor.wire_bytes
        or sha256(wire_path) != blob_hash
    ):
        _fail(wire_path, "wire bytes differ from the journal record/anchor")
    recorded = record.get("identity")
    if not isinstance(recorded, Mapping):
        _fail(anchor.format_name, "wire record lacks producer identity")
    from tessera.cached_unit import ENCODING_INPUT_SCHEMA
    from tessera.encoder_identity import encoder_fixture_id

    if (
        recorded.get("schema") != ENCODING_INPUT_SCHEMA
        or recorded.get("unit") != qname
        or recorded.get("source") != unit["weight"]
        or recorded.get("encoder_source_sha256") != identity.get("encoder_source_sha256")
        or recorded.get("encoder_fixture_id") != encoder_fixture_id().hex()
        or recorded.get("recipe") != expected["wire_recipe"]
    ):
        _fail(anchor.format_name, "wire producer identity differs from the campaign/plan")
    calibration = recorded.get("calibration")
    if expected["hessian_applied"]:
        if not isinstance(calibration, Mapping) or calibration.get("hessian") != unit.get("hessian"):
            _fail(anchor.format_name, "wire Hessian identity differs from the campaign")
    elif calibration is not None:
        _fail(anchor.format_name, "Hessian-free wire carries a calibration receipt")
    return wire_path


def collect(
    plan_path: Path,
    checkpoint: Path,
    cache_dir: Path,
    out: Path,
    *,
    audit_region_correction: "Path | None" = None,
) -> dict:
    plan_path = plan_path.resolve(strict=True)
    plan = _json(plan_path)
    qname, family, legal_rates = _validate_plan(
        plan,
        plan_path,
        allow_outside_audit_regions=audit_region_correction is not None,
    )
    audit_regions, correction_reference = effective_audit_regions(
        plan, plan_path, audit_region_correction
    )
    checkpoint = checkpoint.resolve(strict=True)
    _manifest, identity, unit, shard, state = _load_campaign_unit(checkpoint, plan, qname)
    rows = _anchor_rows(state, plan, unit)
    wire_dir = cache_dir.resolve(strict=True) / "wire"
    plan_sha256 = sha256(plan_path)
    source_receipt = {
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "checkpoint_identity_sha256": canonical_json_sha256(
            identity, where="complete rate curve campaign identity"
        ),
        "unit_shard": {"path": str(shard), "sha256": sha256(shard)},
    }
    receipts = []
    values = []
    for rate in legal_rates:
        anchor, record = rows[rate]
        expected = _expected_row(plan, rate)
        wire_path = _verify_wire(
            anchor=anchor,
            record=record,
            expected=expected,
            qname=qname,
            unit=unit,
            identity=identity,
            wire_dir=wire_dir,
        )
        value = float(anchor.dloss)
        wrapper = {
            "schema": POINT_SCHEMA,
            "curve_id": plan["curve_id"],
            "rate": rate,
            "value": value,
            "kind": "measured",
            "measurement_plan_sha256": plan_sha256,
            "source_receipt": source_receipt,
            "wire": {"path": str(wire_path), "sha256": record["blob_sha256"]},
        }
        wrapper_path = out.parent / f"{out.stem}.rate-{rate}.json"
        _write_json(wrapper_path, wrapper)
        receipts.append({
            "rate": rate,
            "value": value,
            "kind": "measured",
            "path": str(wrapper_path),
            "sha256": sha256(wrapper_path),
        })
        values.append(value)
    curve = {
        "schema": SCHEMA,
        "curve_id": plan["curve_id"],
        "qname": qname,
        "family": family,
        "currency": plan["currency"],
        "activation_contract": plan["activation_contract"],
        "source_identity": plan["source_identity"],
        "calibration_identity": plan["calibration_identity"],
        "recipe_identity": plan["recipe_identity"],
        "legal_rates": legal_rates,
        "audit_regions": audit_regions,
        "measurement_plan": {"path": str(plan_path), "sha256": plan_sha256},
        "rates": legal_rates,
        "values": values,
        "measurement_kind": "measured",
        "receipts": receipts,
    }
    if correction_reference is not None:
        curve["audit_region_correction"] = correction_reference
    return curve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--audit-region-correction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        _fail(out, "refuse to replace a sealed curve")
    curve = collect(
        Path(args.plan),
        Path(args.checkpoint),
        Path(args.cache_dir),
        out,
        audit_region_correction=(
            None if args.audit_region_correction is None else Path(args.audit_region_correction)
        ),
    )
    _write_json(out, curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
