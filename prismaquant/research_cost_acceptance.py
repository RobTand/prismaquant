"""Explicit acceptance boundary for study-grade assembled cost tables.

This module owns the one sanctioned exception to production cost provenance:
a user-acknowledged, content-inventoried assembly of complete per-layer study
segments over a production base table. The assembler itself belonged to the
retired codebook lane (archived 2026-09-25, #1304); the readers below still
recognise and gate a table that carries its stamp.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping


RESEARCH_COST_PROVENANCE = (
    "research_assembled_segments_user_accepted_2026-08-03"
)
RESEARCH_COST_MANIFEST_SCHEMA = "prismaquant.research_cost_manifest.v1"


def accepted_cost_provenance(payload: Mapping) -> dict | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    if provenance.get("cost_provenance") != RESEARCH_COST_PROVENANCE:
        return None
    manifest = provenance.get("research_cost_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("research-stamped cost table has no assembly manifest")
    if manifest.get("schema") != RESEARCH_COST_MANIFEST_SCHEMA:
        raise ValueError("research-stamped cost table has an unknown manifest schema")
    if int(manifest.get("assembled_row_count", -1)) != len(payload.get("costs", {})):
        raise ValueError("research cost manifest row count does not match the table")
    return copy.deepcopy(dict(manifest))


def propagated_cost_provenance(manifest: Mapping | None) -> dict:
    """JSON-safe fragment used by both selection and layer-config emission."""
    if manifest is None:
        return {}
    return {"cost_provenance": copy.deepcopy(dict(manifest))}


def enforce_research_export_acknowledgement(
    layer_config_payload: Mapping,
    *,
    acknowledged: bool,
    where: str,
) -> dict | None:
    meta = layer_config_payload.get("__prismaquant__")
    stamp = meta.get("cost_provenance") if isinstance(meta, Mapping) else None
    if stamp is None:
        return None
    if not isinstance(stamp, Mapping) or stamp.get("cost_provenance") != RESEARCH_COST_PROVENANCE:
        raise ValueError(f"{where}: malformed or unknown research cost provenance")
    if not acknowledged:
        raise ValueError(
            f"{where}: refusing to export a research-stamped cost selection; "
            "pass --allow-research-cost-selection only after separately "
            "acknowledging that study-grade assembled costs are not production provenance"
        )
    return copy.deepcopy(dict(stamp))
