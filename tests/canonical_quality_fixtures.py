"""Complete synthetic scientific coordinates; no measured qualification."""
from __future__ import annotations

import copy

FAMILY = "TESSERA_E4M3_K1"


def complete_probe(*, producer="d", seed=7000):
    from test_glm_mtp_selection import _probe
    probe = _probe(objective=False, seed_base=seed)
    probe.update(calibration_shape=[2, 8], token_scope="all",
                 producer_source_sha256=producer * 64, objective="additive")
    return probe


def scope_for(row, *, family=FAMILY):
    operator, probe = row["joint_operator_identity"], row["probe_identity"]
    return {"unit": operator["qname"], "family": family, "format": operator["format"],
            "currency": row["cost_currency"], "validated": True,
            "calibration": probe["calibration_sha256"],
            "teacher": probe["source_model"]["content_sha256"],
            "window": {"calibration_shape": probe["calibration_shape"],
                       "token_scope": probe["token_scope"], "temperature": probe["temperature"]},
            "objective": {"currency": row["cost_currency"], "normalization": probe["normalization"],
                          "objective": probe.get("objective")},
            "shape": operator["source_weight"]["shape"],
            "source_weight": operator["source_weight"], "activation_contract": operator["activation"],
            "probe_ids": row["probe_ids"], "probe_identity": probe,
            "probe_identity_sha256": row["probe_identity_sha256"],
            "joint_anchor": {key: value for key, value in row.items() if key != "quality_scope"}}


def served_scope(*, unit="u", rate=768, shape=(4, 8), currency="served_kl", validated=True, **extra):
    from prismaquant import joint_aura as joint
    from test_glm_mtp_selection import _row
    row = _row(unit, f"{FAMILY}_R{rate}", [1.0] * 4, complete_probe())
    scope = copy.deepcopy(scope_for(row))
    scope.update(unit=unit, currency=currency, validated=validated, shape=list(shape))
    scope["source_weight"].update(shape=list(shape), logical_bytes=2 * shape[0] * shape[1])
    scope["objective"]["currency"] = currency
    scope.update(extra)
    scope["probe_identity"]["calibration_sha256"] = scope["calibration"]
    scope["probe_identity_sha256"] = joint.identity_sha256(scope["probe_identity"])
    return scope
