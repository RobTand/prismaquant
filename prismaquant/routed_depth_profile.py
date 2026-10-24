"""Receipt-bound routed-depth policy admission; no fitting or authorization.

Hashes check recorded consistency, not receipt authenticity or measurement authority.
No numeric interim-flat decision is available to this reader, so flat refuses.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

from .joint_aura_rows import joint_identity_sha256 as digest, validate_joint_aura_entry

SCHEMA = "prismaquant.routed_depth_profile.v1"
CONTRACT = "band_loss_multiplier.v1"
_BINDINGS = ("source_table_sha256", "band_roster_sha256", "calibration_contract", "parameters_sha256")


def _require_depth_profile_mapping(value, where):
    if not isinstance(value, Mapping):
        raise ValueError(f"depth profile {where} must be an object")
    return value


def _sample_receipt(receipt, bindings, schema):
    receipt = _require_depth_profile_mapping(receipt, "receipt")
    if receipt.get("schema") != schema:
        raise ValueError("depth profile receipt schema mismatch")
    if any(receipt.get(key) != bindings[key] for key in _BINDINGS):
        raise ValueError("depth profile receipt binding mismatch")
    arm = receipt.get("arm_id")
    samples = receipt.get("sample_roster_sha256")
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("depth profile receipt requires an arm identity")
    if not isinstance(samples, str) or re.fullmatch(r"[a-f0-9]{64}", samples) is None:
        raise ValueError("depth profile receipt requires a sample roster digest")
    return arm, samples


def admit_depth_profile(payload, record, *, production):
    """Return owned metadata and immutable (unit, multiplier) pairs, or no-op.

    Only the explicit joint_t8_routed/all_rates scope opts into this policy.
    Model identity is carried by that scope, never inferred from local paths.
    """
    payload = _require_depth_profile_mapping(payload, "source table")
    provenance = _require_depth_profile_mapping(payload.get("provenance", {}), "source provenance")
    scope = provenance.get("depth_price_scope")
    if scope is None:
        return None, ()
    scope = _require_depth_profile_mapping(scope, "scope")
    if scope.get("lane") != "joint_t8_routed" or scope.get("allocation") != "all_rates":
        return None, ()
    if not isinstance(scope.get("model_id"), str) or not scope["model_id"].strip():
        raise ValueError("depth profile affected scope requires model_id")
    if record is None:
        raise ValueError("depth profile flat pricing requires authentic complete numeric authorization")
    record = _require_depth_profile_mapping(record, "record")
    if record.get("state") in {"flat", "interim_flat"}:
        raise ValueError("depth profile interim flat has no authentic numeric authorization")
    if record.get("schema") != SCHEMA or record.get("calibration_contract") != CONTRACT:
        raise ValueError("depth profile unsupported schema or calibration contract")
    state = record.get("state")
    if state not in {"research_calibrated", "qualified"}:
        raise ValueError("depth profile unsupported state")
    if production and state != "qualified":
        raise ValueError("depth profile production requires qualified prices")
    if record.get("scope") != scope:
        raise ValueError("depth profile foreign scope")
    if any(key in record for key in ("gain", "scalar", "loss_multiplier")):
        raise ValueError("depth profile refuses scalar gain")
    if (provenance.get("cost_mode") != "aura" or
            provenance.get("joint_activation") is not True or
            provenance.get("cost_currency") != "joint_aura_predicted_dloss"):
        raise ValueError("depth profile requires joint AURA currency and cost mode")

    costs = _require_depth_profile_mapping(payload.get("costs"), "costs")
    # These are explicit per-expert rows, not a model-specific depth cutoff.
    roster = {name for name in costs if isinstance(name, str) and ".experts." in name}
    if not roster:
        raise ValueError("depth profile has no routed member roster")
    bands = _require_depth_profile_mapping(record.get("bands"), "bands")
    parameters = _require_depth_profile_mapping(record.get("parameters"), "parameters")
    if len(bands) < 2 or set(bands) != set(parameters):
        raise ValueError("depth profile requires complete band parameters")
    members = []
    factors = []
    prices = []
    for band, names in bands.items():
        if not isinstance(band, str) or not band or not isinstance(names, list) or not names:
            raise ValueError("depth profile requires named nonempty bands")
        if any(not isinstance(name, str) or name not in roster for name in names):
            raise ValueError("depth profile contains foreign roster members")
        parameter = _require_depth_profile_mapping(parameters[band], "band parameter")
        if set(parameter) != {"loss_multiplier"}:
            raise ValueError("depth profile unsupported numerical parameter")
        factor = parameter["loss_multiplier"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("depth profile loss multiplier must be positive and finite")
        factors.append(factor)
        members.extend(names)
        prices.extend((name, float(factor)) for name in names)
    if len(members) != len(set(members)) or set(members) != roster:
        raise ValueError("depth profile incomplete or duplicate band roster")
    if len(set(factors)) == 1:
        raise ValueError("depth profile refuses a scalar gain disguised as bands")

    probe_identity = None
    for name in roster:
        rows = _require_depth_profile_mapping(costs[name], "member rows")
        if not rows:
            raise ValueError("depth profile empty member rows")
        for fmt, row in rows.items():
            row = _require_depth_profile_mapping(row, "cost row")
            if not validate_joint_aura_entry(row):
                raise ValueError("depth profile requires joint AURA measurement rows")
            operator = row["joint_operator_identity"]
            if operator["qname"] != name or operator["format"] != fmt:
                raise ValueError("depth profile row coordinate mismatch")
            if probe_identity is None:
                probe_identity = row["probe_identity"]
            elif row["probe_identity"] != probe_identity:
                raise ValueError("depth profile incompatible probe populations")

    bindings = {"source_table_sha256": digest(payload), "band_roster_sha256": digest(bands),
                "calibration_contract": CONTRACT, "parameters_sha256": digest(parameters)}
    if any(record.get(key) != value for key, value in bindings.items()):
        raise ValueError("depth profile source, roster or parameter binding mismatch")
    measurement = record.get("measurement_receipt")
    fit_arm, fit_samples = _sample_receipt(measurement, bindings,
        "prismaquant.routed_depth_measurement_receipt.v1")
    if measurement.get("result") != "complete":
        raise ValueError("depth profile measurement is incomplete")
    qualified = state == "qualified"
    if qualified:
        receipts = _require_depth_profile_mapping(record.get("qualification_receipts"), "qualification receipts")
        kinds = {"measured_arm_agreement", "held_out_arm"}
        if set(receipts) != kinds:
            raise ValueError("depth profile requires both qualification receipts")
        arms, samples = {fit_arm}, {fit_samples}
        for kind in sorted(kinds):
            receipt = receipts[kind]
            arm, sample = _sample_receipt(receipt, bindings,
                "prismaquant.routed_depth_qualification_receipt.v1")
            if receipt.get("kind") != kind or receipt.get("passed") is not True:
                raise ValueError("depth profile qualification did not pass")
            if arm in arms or sample in samples:
                raise ValueError("depth profile qualification requires independent arms and samples")
            arms.add(arm)
            samples.add(sample)
    metadata = deepcopy(dict(record))
    metadata["validated"] = qualified
    metadata["measurement_receipt_sha256"] = digest(measurement)
    if qualified:
        metadata["qualification_receipts_sha256"] = digest(record["qualification_receipts"])
    return metadata, tuple(prices)
