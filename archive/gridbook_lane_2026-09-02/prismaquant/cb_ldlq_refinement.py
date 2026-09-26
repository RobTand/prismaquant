"""Post-allocation LDLQ refinement contract.

The burn's cost table was measured WITHOUT LDLQ but stamped with an LDLQ
serialization context (poolb + incumbent_noldlq, both ldlq=false, but the
context claims ldlq=true for the eventual export).  Rather than rewrite the
cost history, the honest path is an explicit refinement record:

* the allocator's cost provenance remains RAW (the number it optimized);
* the exported artifact's CB serialization context is LDLQ=true (the bytes it
  ships);
* a separate ``post_allocation_refinement`` record bridges the two with a
  do-no-harm gate and byte-neutral proof.

This module owns the schema and the truthful-stamp helpers.  The exporter
writes the record into ``quant_config.json/provenance`` and the derived
``layer_config.json/__prismaquant__`` carries it for audit.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

REFINEMENT_SCHEMA = "prismaquant.cb_ldlq_refinement.v1"
REFINEMENT_KIND = "fixed_codebook_ldlq_gated"


# ``holdout_activation_output_mse`` is the 2026-08-08 default: LDLQ is certified
# on rows its Hessian never saw. ``activation_output_mse`` is the legacy
# in-sample scoring, retained only to reproduce pre-2026-08-08 artifacts — it
# scores on the rows that fitted the Hessian, so it cannot fail, and its error
# was measured to be ANTI-correlated with the true benefit (20x overstatement at
# 64 rows, 48.5x at 1-3 rows). Do not select it for new artifacts.
ALLOWED_GATES = frozenset({
    "holdout_activation_output_mse",
    "activation_output_mse",
    "col_weighted_mse",
})


def build_refinement_provenance(
    *,
    cost_ldlq: bool,
    export_ldlq: bool,
    gate: str,
    gate_enabled: bool,
    byte_neutral: bool = True,
    note: str | None = None,
) -> dict[str, Any]:
    """Return a truthful refinement record for a derived artifact.

    ``cost_ldlq`` is what the cost table measured (False for the A-FAST burn);
    ``export_ldlq`` is what the exporter rendered (True for the LDLQ re-export).
    ``gate`` names the local metric (``activation_output_mse``).  The record is
    stamped with a monotonic timestamp so a re-export of the same assignment
    is distinguishable.
    """
    if not isinstance(cost_ldlq, bool) or not isinstance(export_ldlq, bool):
        raise TypeError("cost_ldlq and export_ldlq must be bool")
    if cost_ldlq and not export_ldlq:
        raise ValueError("refinement cannot downgrade LDLQ: cost true but export false")
    gate = str(gate).strip()
    if gate not in ALLOWED_GATES:
        raise ValueError(f"gate must be one of {sorted(ALLOWED_GATES)}, got {gate!r}")
    payload: dict[str, Any] = {
        "schema": REFINEMENT_SCHEMA,
        "kind": REFINEMENT_KIND,
        "cost_ldlq": bool(cost_ldlq),
        "export_ldlq": bool(export_ldlq),
        "gate": gate,
        "gate_enabled": bool(gate_enabled),
        "byte_neutral": bool(byte_neutral),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if note is not None:
        payload["note"] = str(note)
    return payload


def validate_refinement_provenance(
    value: Mapping[str, Any] | None,
    *,
    where: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: refinement provenance is not an object")
    if value.get("schema") != REFINEMENT_SCHEMA:
        raise ValueError(f"{where}: unsupported refinement schema {value.get('schema')!r}")
    if value.get("kind") != REFINEMENT_KIND:
        raise ValueError(f"{where}: unexpected refinement kind {value.get('kind')!r}")
    for key in ("cost_ldlq", "export_ldlq", "gate_enabled", "byte_neutral", "gate"):
        if key not in value:
            raise ValueError(f"{where}: refinement record missing {key!r}")
    if str(value.get("gate")) not in ALLOWED_GATES:
        raise ValueError(f"{where}: refinement gate must be one of {sorted(ALLOWED_GATES)}, got {value.get('gate')!r}")
    return dict(value)


def attach_refinement_to_layer_config(
    layer_config_path: str,
    out_path: str,
    *,
    cost_ldlq: bool = False,
    export_ldlq: bool = True,
    # Must name the gate the RENDER actually used. The renderer's default is
    # the held-out certificate since 2026-08-08; a record still claiming the
    # legacy in-sample gate would be a provenance lie of exactly the kind this
    # module exists to prevent.
    gate: str = "holdout_activation_output_mse",
    gate_enabled: bool = True,
    note: str | None = None,
) -> dict[str, Any]:
    """Copy a layer_config.json to a new path with the refinement record.

    The assignment itself is untouched; only ``__prismaquant__`` gains the
    new key ``post_allocation_refinement``.  Returns the written JSON dict.
    """
    import pathlib

    src = pathlib.Path(layer_config_path)
    dst = pathlib.Path(out_path)
    payload = json.loads(src.read_text())
    meta = payload.get("__prismaquant__")
    if not isinstance(meta, dict):
        meta = {}
        payload["__prismaquant__"] = meta
    # Preserve existing keys, add refinement.
    meta["post_allocation_refinement"] = build_refinement_provenance(
        cost_ldlq=cost_ldlq,
        export_ldlq=export_ldlq,
        gate=gate,
        gate_enabled=gate_enabled,
        note=note,
    )
    # Also stamp the CB serialized payload's truth: cost was raw, export is LDLQ.
    # The existing per-tensor identities already claim ldlq:true (the export
    # bytes); we do not rewrite them, but we record that the cost side was raw
    # so a reader does not infer cost_ldlq=true from the identity.
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
