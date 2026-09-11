#!/usr/bin/env python3
"""Prospective, receipt-bound paired-family transfer predictions.

This is a research evaluator.  It freezes curve-piece plans before any
measurement, seals predictions using only their declared endpoint inputs, and
audits a later complementary interior curve.  It neither changes production
interpolation nor makes a qualification or measurement-saving claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.sparse_rate_adaptive import (
    CURRENCY, MEASUREMENT_PLAN_SCHEMA, error_metrics, validate_curve,
)


PROTOCOL_SCHEMA = "prismaquant.prospective_paired_family_transfer_protocol.v1"
SEAL_SCHEMA = "prismaquant.prospective_paired_family_transfer_seal.v1"
AUDIT_SCHEMA = "prismaquant.prospective_paired_family_transfer_audit.v1"
ROLES = ("bf_source", "e4_left", "e4_right", "e4_interior", "e2_left", "e2_right",
         "e2_interior", "e2_terminal")
ENDPOINT_ROLES = {
    "e4": ("bf_source", "e4_left", "e4_right"),
    "e2": ("e2_left", "e2_right"),
}


class FamilyTransferError(ValueError):
    pass


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha_bytes(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FamilyTransferError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FamilyTransferError(f"JSON object required: {path}")
    return value


def _write_exclusive(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _require(condition, message):
    if not condition:
        raise FamilyTransferError(message)


def _plan_ref(path):
    value = _load(path)
    _require(value.get("schema") == MEASUREMENT_PLAN_SCHEMA,
             "unsupported measurement plan schema")
    _require(value.get("status") == "frozen_before_measurement"
             and value.get("measurement_kind") == "measured",
             "measurement plan was not frozen for measured acquisition")
    _require(value.get("currency") == CURRENCY, "measurement plan currency differs")
    for key in ("qname", "family", "activation_contract"):
        _require(isinstance(value.get(key), str) and value[key], f"measurement plan lacks {key}")
    for key in ("source_identity", "calibration_identity", "recipe_identity"):
        _require(isinstance(value.get(key), dict) and value[key], f"measurement plan lacks {key}")
    rates = value.get("legal_rates")
    _require(isinstance(rates, list) and rates == sorted(set(rates))
             and all(type(rate) is int for rate in rates), "measurement plan roster is invalid")
    return {"sha256": digest(path), "plan": value}


def _same_identity(left, right, keys):
    return all(left["plan"].get(key) == right["plan"].get(key) for key in keys)


def freeze_protocol(plan_paths, out, *, preselection_path=None, preexposed_rates=None,
                    preexposed_metadata=None):
    """Freeze the eight independent curve pieces before GPU measurement.

    ``plan_paths`` is a role-to-path mapping.  The E4/E2 endpoints are
    deliberately singleton plans, so campaigns never have to repeat endpoint
    measurements to make a later complete curve artifact.
    """
    _require(isinstance(plan_paths, dict) and set(plan_paths) == set(ROLES),
             f"plan roles must be exactly {ROLES}")
    refs = {role: _plan_ref(path) for role, path in plan_paths.items()}
    preselection = None
    if preselection_path is not None:
        preselection = {"sha256": digest(preselection_path), "payload": _load(preselection_path)}
    bf, e4l, e4r, e4i = (refs[key] for key in ("bf_source", "e4_left", "e4_right", "e4_interior"))
    e2l, e2r, e2i, e2t = (refs[key] for key in ("e2_left", "e2_right", "e2_interior", "e2_terminal"))
    _require(bf["plan"]["family"] == "TESSERA_BF16_K1", "bf_source must be BF16")
    _require(e4l["plan"]["family"] == e4r["plan"]["family"] == e4i["plan"]["family"]
             == "TESSERA_E4M3_K1", "E4 plans must be E4M3")
    _require(e2l["plan"]["family"] == e2r["plan"]["family"] == e2i["plan"]["family"]
             == e2t["plan"]["family"] == "TESSERA_E2M1_K2", "E2 plans must be E2M1")
    common = ("qname", "source_identity", "calibration_identity")
    _require(all(_same_identity(bf, ref, common) for ref in refs.values()),
             "plans differ in qname, source or calibration identity")
    _require(_same_identity(e4l, e4r, ("activation_contract", "recipe_identity"))
             and _same_identity(e4l, e4i, ("activation_contract", "recipe_identity")),
             "E4 plans differ in recipe or activation contract")
    _require(_same_identity(e2l, e2r, ("activation_contract", "recipe_identity"))
             and _same_identity(e2l, e2i, ("activation_contract", "recipe_identity")),
             "E2 interpolated-window plans differ in recipe or activation contract")
    bf_rates = bf["plan"]["legal_rates"]
    _require(len(bf_rates) >= 3, "BF source must be a complete multi-rate curve")
    _require(e4l["plan"]["legal_rates"] == [bf_rates[0]]
             and e4r["plan"]["legal_rates"] == [bf_rates[-1]]
             and e4i["plan"]["legal_rates"] == bf_rates[1:-1],
             "E4 endpoint/interior rosters are not the BF roster partition")
    e2_rates = [e2l["plan"]["legal_rates"][0], *e2i["plan"]["legal_rates"],
                e2r["plan"]["legal_rates"][0]]
    _require(len(e2l["plan"]["legal_rates"]) == len(e2r["plan"]["legal_rates"]) == 1
             and e2_rates == sorted(set(e2_rates)) and len(e2_rates) >= 3,
             "E2 endpoint/interior rosters are invalid")
    _require(len(e2t["plan"]["legal_rates"]) == 1
             and e2t["plan"]["legal_rates"][0] not in e2_rates,
             "E2 terminal must be one separate, non-interpolated rate")
    if preselection is not None:
        selected = preselection["payload"]
        _require(selected.get("schema") == "prismaquant.family_transfer_preselection.v1"
                 and selected.get("status") == "frozen_before_new_dense_measurements"
                 and selected.get("research_only") is True
                 and selected.get("qname") == bf["plan"]["qname"],
                 "preselection is not the frozen research selection for these plans")
        bindings = selected.get("measurement_plans")
        _require(isinstance(bindings, dict) and set(bindings) == set(ROLES)
                 and all(isinstance(bindings[role], dict)
                         and bindings[role].get("path") == str(Path(plan_paths[role]))
                         and bindings[role].get("sha256") == refs[role]["sha256"]
                         for role in ROLES),
                 "preselection plan paths or hashes differ from the frozen protocol inputs")
        rules = selected.get("rules", {})
        _require(rules.get("e4_primary", {}).get("name") == "affine_bf_from_paired_endpoints"
                 and rules.get("e4_secondary", {}).get("name") == "rate_linear_delta_from_bf"
                 and rules.get("e2_primary", {}).get("name") == "endpoint_value_linear"
                 and rules.get("exact_endpoints") is True
                 and rules.get("finite_positive_required") is True
                 and rules.get("no_refit_from_interiors") is True,
                 "preselection mathematical rules differ from this protocol")
    if preexposed_rates is None:
        historical = preselection["payload"].get("historical_exposure", {}) if preselection else {}
        preexposed_rates = {"e4": historical.get("E4M3", []), "e2": historical.get("E2M1", [])}
    _require(isinstance(preexposed_rates, dict) and set(preexposed_rates) == {"e4", "e2"}
             and all(isinstance(preexposed_rates[key], list)
                     and preexposed_rates[key] == sorted(set(preexposed_rates[key]))
                     and all(type(rate) is int for rate in preexposed_rates[key])
                     for key in preexposed_rates),
             "preexposed rates must be sorted unique integer rosters by family")
    _require(set(preexposed_rates["e4"]) <= set(bf_rates)
             and set(preexposed_rates["e2"]) <= set(e2_rates) | set(e2t["plan"]["legal_rates"]),
             "preexposed rates lie outside their frozen family roster")
    if preexposed_metadata is None:
        preexposed_metadata = {}
    _require(isinstance(preexposed_metadata, dict), "preexposed metadata must be an object")
    payload = {
        "schema": PROTOCOL_SCHEMA, "research_only": True,
        "currency": CURRENCY, "qname": bf["plan"]["qname"],
        "chronology": ("preselection rules and plans were frozen before initial GPU acquisition; "
                       "this generalized implementation/protocol is frozen before reading numerical curve inputs"
                       if preselection else
                       "freeze all listed plans and this protocol before reading numerical curve inputs"),
        "plans": {role: {"sha256": refs[role]["sha256"], "payload": refs[role]["plan"]}
                  for role in ROLES},
        "preselection": preselection,
        "implementation": {"family_transfer_sha256": digest(__file__),
                             "curve_validation_sha256": digest(Path(__file__).with_name("sparse_rate_adaptive.py"))},
        "prediction_families": {
            "e4": {"forms": ["affine_bf_from_paired_endpoints", "rate_linear_delta_from_bf"],
                   "target_roster": bf_rates, "audit_role": "e4_interior"},
            "e2": {"forms": ["endpoint_value_linear"], "target_roster": e2_rates,
                   "audit_role": "e2_interior", "terminal_role": "e2_terminal",
                   "terminal": "measured separately; excluded from interpolation"},
        },
        "charge_counts": {"bf_source": len(bf_rates), "e4_endpoints": 2,
                          "e2_endpoints": 2, "e2_terminal": 1,
                          "total": len(bf_rates) + 5},
        "preexposed_rates": preexposed_rates,
        "preexposed_metadata": preexposed_metadata,
        "qualification": "research audit only; no broad qualification or time-savings claim",
    }
    payload["protocol_sha256"] = _sha_bytes(payload)
    _write_exclusive(out, payload)
    return payload


def load_protocol(path):
    value = _load(path)
    claimed = value.pop("protocol_sha256", None)
    _require(value.get("schema") == PROTOCOL_SCHEMA and claimed == _sha_bytes(value),
             "protocol digest differs from its bytes")
    _require(value.get("research_only") is True and value.get("currency") == CURRENCY,
             "protocol is not this research currency")
    _require(set(value.get("plans", {})) == set(ROLES), "protocol plan roster differs")
    preselection = value.get("preselection")
    _require(preselection is None or (isinstance(preselection, dict)
             and isinstance(preselection.get("payload"), dict)
             and isinstance(preselection.get("sha256"), str)
             and len(preselection["sha256"]) == 64), "protocol preselection binding is invalid")
    for role, ref in value["plans"].items():
        _require(isinstance(ref, dict) and isinstance(ref.get("payload"), dict)
                 and isinstance(ref.get("sha256"), str) and len(ref["sha256"]) == 64,
                 f"protocol plan {role} hash differs")
    impl = value.get("implementation", {})
    _require(impl == {"family_transfer_sha256": digest(__file__),
                      "curve_validation_sha256": digest(Path(__file__).with_name("sparse_rate_adaptive.py"))},
             "protocol implementation bytes differ")
    _require(value["plans"]["bf_source"]["payload"].get("family") == "TESSERA_BF16_K1"
             and all(value["plans"][role]["payload"].get("family") == "TESSERA_E4M3_K1"
                     for role in ("e4_left", "e4_right", "e4_interior"))
             and all(value["plans"][role]["payload"].get("family") == "TESSERA_E2M1_K2"
                     for role in ("e2_left", "e2_right", "e2_interior", "e2_terminal")),
             "protocol family roles are invalid")
    value["protocol_sha256"] = claimed
    return value


def _curve_for_role(protocol, role, path):
    curve = _load(path)
    rates, values = validate_curve(curve, min_points=1)
    ref = protocol["plans"][role]
    plan_ref = curve.get("measurement_plan", {})
    _require(plan_ref.get("sha256") == ref["sha256"], f"{role} curve plan differs from protocol")
    plan = ref["payload"]
    for key in ("qname", "family", "activation_contract", "source_identity",
                "calibration_identity", "recipe_identity", "legal_rates"):
        _require(curve.get(key) == plan.get(key), f"{role} curve identity differs from protocol plan: {key}")
    return curve, rates, values


def _single(curve_data, role):
    curve, rates, values = curve_data
    _require(len(rates) == 1, f"{role} must be a singleton endpoint curve")
    return rates[0], values[0]


def seal_predictions(protocol_path, family, input_curves, out):
    """Write an immutable prediction seal without accepting any target interior path."""
    protocol = load_protocol(protocol_path)
    _require(family in ENDPOINT_ROLES, "family must be e4 or e2")
    expected = set(ENDPOINT_ROLES[family])
    _require(isinstance(input_curves, dict) and set(input_curves) == expected,
             f"{family} sealer accepts exactly its endpoint inputs, never an interior curve")
    loaded = {role: _curve_for_role(protocol, role, path) for role, path in input_curves.items()}
    if family == "e4":
        bf_curve, bf_rates, bf_values = loaded["bf_source"]
        left_rate, left_value = _single(loaded["e4_left"], "e4_left")
        right_rate, right_value = _single(loaded["e4_right"], "e4_right")
        _require((left_rate, right_rate) == (bf_rates[0], bf_rates[-1]), "E4 endpoints differ from BF source")
        _require(bf_values[-1] != bf_values[0], "BF endpoint values cannot fit affine E4 transfer")
        beta = (right_value - left_value) / (bf_values[-1] - bf_values[0])
        alpha = left_value - beta * bf_values[0]
        records = []
        for rate, bf_value in zip(bf_rates, bf_values):
            fraction = (rate - left_rate) / (right_rate - left_rate)
            affine = alpha + beta * bf_value
            delta = bf_value + (left_value - bf_values[0]) + fraction * (
                (right_value - bf_values[-1]) - (left_value - bf_values[0]))
            if rate == left_rate:
                affine = delta = left_value
            elif rate == right_rate:
                affine = delta = right_value
            records.append({"rate": rate, "affine_bf_from_paired_endpoints": affine,
                            "rate_linear_delta_from_bf": delta})
        fit = {"alpha": alpha, "beta": beta, "left_rate": left_rate, "right_rate": right_rate}
    else:
        left_rate, left_value = _single(loaded["e2_left"], "e2_left")
        right_rate, right_value = _single(loaded["e2_right"], "e2_right")
        _require(left_rate < right_rate, "E2 endpoint roster is unordered")
        records = []
        for rate in protocol["prediction_families"]["e2"]["target_roster"]:
            value = left_value + (rate - left_rate) / (right_rate - left_rate) * (right_value - left_value)
            records.append({"rate": rate, "endpoint_value_linear":
                            left_value if rate == left_rate else right_value if rate == right_rate else value})
        fit = {"left_rate": left_rate, "right_rate": right_rate,
               "terminal": "excluded; separately measured"}
    inputs = {role: {"curve_sha256": digest(path), "plan_sha256": protocol["plans"][role]["sha256"]}
              for role, path in input_curves.items()}
    payload = {"schema": SEAL_SCHEMA, "research_only": True, "currency": CURRENCY,
               "protocol_sha256": protocol["protocol_sha256"], "family": family,
               "inputs": inputs, "fit": fit, "predictions": records,
               "audit_role": protocol["prediction_families"][family]["audit_role"],
               "measured_prediction_distinction": "all values in predictions are pre-measurement predictions",
               "qualification": "research audit only; no broad qualification or time-savings claim"}
    _validate_predictions(protocol, payload)
    payload["seal_sha256"] = _sha_bytes(payload)
    _write_exclusive(out, payload)
    return payload


def load_seal(path):
    value = _load(path)
    claimed = value.pop("seal_sha256", None)
    _require(value.get("schema") == SEAL_SCHEMA and claimed == _sha_bytes(value),
             "prediction seal digest differs from its bytes")
    _require(value.get("family") in ENDPOINT_ROLES and value.get("currency") == CURRENCY,
             "prediction seal family or currency is invalid")
    value["seal_sha256"] = claimed
    return value


def _validate_predictions(protocol, seal):
    family = seal["family"]
    rows = seal.get("predictions")
    forms = protocol["prediction_families"][family]["forms"]
    roster = protocol["prediction_families"][family]["target_roster"]
    _require(isinstance(rows, list) and [row.get("rate") for row in rows] == roster,
             "prediction roster is missing, duplicated, reordered or extra")
    for row in rows:
        _require(set(row) == {"rate", *forms} and all(isinstance(row[form], (int, float))
                 and not isinstance(row[form], bool) and math.isfinite(row[form]) and row[form] > 0
                 for form in forms), "prediction value is nonfinite, nonpositive or malformed")


def audit_predictions(protocol_path, seal_path, source_curves, target_curves, out):
    """Audit only unseen complementary interiors, after rechecking all bindings."""
    protocol, seal = load_protocol(protocol_path), load_seal(seal_path)
    _require(seal.get("protocol_sha256") == protocol["protocol_sha256"], "seal belongs to another protocol")
    family = seal.get("family")
    _validate_predictions(protocol, seal)
    expected_source = set(ENDPOINT_ROLES.get(family, ()))
    _require(set(source_curves) == expected_source, "audit source roster differs from sealed endpoint inputs")
    source = {role: _curve_for_role(protocol, role, path) for role, path in source_curves.items()}
    for role, path in source_curves.items():
        _require(digest(path) == seal["inputs"].get(role, {}).get("curve_sha256"),
                 f"sealed source curve bytes changed: {role}")
    forms = protocol["prediction_families"][family]["forms"]
    prediction = {row["rate"]: row for row in seal.get("predictions", [])}
    endpoint_roles = ("e4_left", "e4_right") if family == "e4" else ("e2_left", "e2_right")
    for role in endpoint_roles:
        rate, value = _single(source[role], role)
        _require(all(prediction[rate][form] == value for form in forms),
                 f"sealed prediction does not preserve measured endpoint: {role}")
    required_target = {seal["audit_role"]}
    if family == "e2":
        required_target.add("e2_terminal")
    _require(set(target_curves) == required_target, "audit target roster differs from complementary protocol pieces")
    interior, interior_rates, interior_values = _curve_for_role(
        protocol, seal["audit_role"], target_curves[seal["audit_role"]])
    _require(set(interior_rates) <= set(prediction), "audit has no sealed prediction for an interior rate")
    endpoint_roles = ("e4_left", "e4_right") if family == "e4" else ("e2_left", "e2_right")
    endpoint_rates = {rate for role in endpoint_roles for rate in source[role][1]}
    _require(not set(interior_rates) & endpoint_rates, "audit curve repeats a fitted endpoint")
    _require(set(interior_rates) == set(protocol["prediction_families"][family]["target_roster"])
             - endpoint_rates, "audit curve is not the exact unseen endpoint complement")
    metrics = {}
    fresh_rates = [rate for rate in interior_rates
                   if rate not in set(protocol["preexposed_rates"][family])]
    for form in forms:
        metrics[form] = {
            "all_interior": error_metrics([prediction[rate][form] for rate in interior_rates],
                                            interior_values, .005),
            "fresh_only": error_metrics([prediction[rate][form] for rate in fresh_rates],
                                          [interior_values[index] for index, rate in enumerate(interior_rates)
                                           if rate in fresh_rates], .005),
        }
    terminal = None
    if family == "e2":
        terminal_curve, terminal_rates, terminal_values = _curve_for_role(
            protocol, "e2_terminal", target_curves["e2_terminal"])
        _require(len(terminal_rates) == 1 and terminal_rates[0] not in prediction,
                 "E2 terminal must remain outside the interpolated prediction roster")
        terminal = {"rate": terminal_rates[0], "measured_value": terminal_values[0],
                    "prediction": None, "status": "separate measured terminal; not audited as interpolation"}
    payload = {"schema": AUDIT_SCHEMA, "research_only": True, "currency": CURRENCY,
               "protocol_sha256": protocol["protocol_sha256"], "seal_sha256": seal["seal_sha256"],
               "family": family, "unseen_only": True,
               "interior_curve_sha256": digest(target_curves[seal["audit_role"]]),
               "interior_rates": list(interior_rates), "fresh_interior_rates": fresh_rates,
               "metrics": metrics, "terminal": terminal,
               "qualification": "research audit only; no broad qualification or time-savings claim"}
    _write_exclusive(out, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--plans", required=True); freeze.add_argument("--out", required=True); freeze.add_argument("--preselection"); freeze.add_argument("--preexposed-rates"); freeze.add_argument("--preexposed-metadata")
    seal = sub.add_parser("seal"); seal.add_argument("--protocol", required=True); seal.add_argument("--family", choices=("e4", "e2"), required=True); seal.add_argument("--inputs", required=True); seal.add_argument("--out", required=True)
    audit = sub.add_parser("audit"); audit.add_argument("--protocol", required=True); audit.add_argument("--seal", required=True); audit.add_argument("--sources", required=True); audit.add_argument("--targets", required=True); audit.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "freeze": freeze_protocol(_load(args.plans), args.out,
                                                    preselection_path=args.preselection,
                                                    preexposed_rates=(_load(args.preexposed_rates)
                                                                      if args.preexposed_rates else None),
                                                    preexposed_metadata=(_load(args.preexposed_metadata)
                                                                        if args.preexposed_metadata else None))
    elif args.command == "seal": seal_predictions(args.protocol, args.family, _load(args.inputs), args.out)
    else: audit_predictions(args.protocol, args.seal, _load(args.sources), _load(args.targets), args.out)


if __name__ == "__main__":
    main()
