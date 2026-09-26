#!/usr/bin/env python3
"""Derive a no-LDLQ allocator cost table from a gated LDLQ cost pickle.

An LDLQ-gated cost run measures every CB cell on the gated render AND — since
the raw-sidecar change — records ``*_raw_render`` sidecar metrics measured on
the pre-gate raw assignment, which is the identical-env no-LDLQ render (same
encode, codebook, scale sweep/coding, col_weights; see
``cb_ldlq_raw_render_sidecar`` in the payload provenance).  This tool swaps
each LDLQ-covered CB row's allocator-facing metrics for those sidecar values
and re-stamps the payload's CB serialization identity as ``ldlq=false /
ldlq_scope=none``, producing an allocator-consumable cost pickle for a
no-LDLQ allocation WITHOUT a second multi-hour cost burn — the exact isolate
needed for the LDLQ-contribution A/B.

Refusal contract (fail-closed):
  * the input payload must stamp an LDLQ scope other than ``none``;
  * every CB row that LDLQ covered under that scope must carry the sidecar
    (``weight_mse_raw_render``; ``predicted_dloss_raw_render`` and the
    per-expert vector wherever the primary row has those) — rows priced by
    the RD-ladder interpolator or stamped ``error`` have no honest raw
    measurement and abort the extraction;
  * rows LDLQ never touched (non-CB formats; the fp8 family under
    ``ldlq_scope=nvfp4``) are copied unchanged — they already ARE the raw
    render.

Output-side metrics are not re-measured for the raw arm, so extracted rows
carry ``output_mse=0.0`` / ``output_mse_measured=false`` with
``cost_source="ldlq_raw_render_sidecar"`` — the same legal shape as packed
rows whose routed forward could not be reconstructed; the allocator prices
``predicted_dloss``/``weight_mse``.

Usage:
    python3 tools/extract_raw_cost_table.py <gated_cost.pkl> <raw_cost.pkl>

Exit codes: 0 written · 1 refused · 2 usage error.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path


class RawSidecarExtractionError(ValueError):
    """A payload that cannot honestly yield a raw (no-LDLQ) cost table."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawSidecarExtractionError(message)


def extract_raw_cost_payload(payload: dict, *, where: str = "extract_raw_cost_table") -> dict:
    """Return a new cost payload whose metrics are the raw (no-LDLQ) sidecar.

    Pure function over the loaded pickle payload; see the module docstring
    for the refusal contract.
    """
    from prismaquant.measure_quant_cost import LDLQ_RAW_SIDECAR_COST_SOURCE
    from prismaquant.nvfp4_cb_footprint import (
        _ldlq_for_format,
        cb_serialization_context_from_stamp,
        cb_serialization_context_stamp,
        is_cb_format,
        validate_cb_cost_provenance,
    )

    _require(isinstance(payload, dict), f"{where}: payload is not a dict")
    costs = payload.get("costs")
    _require(isinstance(costs, dict), f"{where}: payload has no 'costs' table")
    formats = payload.get("formats")
    _require(
        isinstance(formats, (list, tuple)) and bool(formats),
        f"{where}: payload has no 'formats' list",
    )
    provenance = payload.get("provenance")
    _require(
        isinstance(provenance, dict),
        f"{where}: payload has no provenance; cannot establish the gated "
        "render identity the sidecar was measured under",
    )
    stamp = provenance.get("cb_serialized_payload")
    context = cb_serialization_context_from_stamp(stamp, where=where)
    scope = str(getattr(context, "ldlq_scope", "none")).strip().lower()
    _require(
        scope != "none",
        f"{where}: payload stamps ldlq_scope=none — it already IS a raw "
        "(no-LDLQ) cost table; nothing to extract",
    )
    raw_context = dataclasses.replace(context, ldlq=False, ldlq_scope="none")

    new_costs: dict = {}
    swapped = 0
    for qname, per_format in costs.items():
        _require(
            isinstance(per_format, dict),
            f"{where}: malformed cost row for {qname!r}",
        )
        new_row: dict = {}
        for fmt, entry in per_format.items():
            _require(
                isinstance(entry, dict),
                f"{where}: malformed cost entry {qname!r}/{fmt!r}",
            )
            _require(
                "error" not in entry,
                f"{where}: {qname!r}/{fmt!r} is an error row; a raw table "
                "cannot be derived from a failed measurement",
            )
            if not (is_cb_format(fmt) and _ldlq_for_format(fmt, context)):
                # LDLQ never touched this row: it already is the raw render.
                new_row[fmt] = dict(entry)
                continue
            _require(
                "weight_mse_raw_render" in entry,
                f"{where}: {qname!r}/{fmt!r} lacks the raw-render sidecar "
                "(pre-sidecar producer, or a ladder-interpolated row); "
                "refusing to emit a mixed raw/gated table",
            )
            new_entry: dict = {
                "weight_mse": float(entry["weight_mse_raw_render"]),
                # Output-side metrics are not re-measured for the raw arm.
                "output_mse": 0.0,
                "rel_output_mse": 0.0,
                "output_mse_measured": False,
                "cost_source": LDLQ_RAW_SIDECAR_COST_SOURCE,
            }
            if "predicted_dloss" in entry:
                _require(
                    "predicted_dloss_raw_render" in entry,
                    f"{where}: {qname!r}/{fmt!r} has predicted_dloss but no "
                    "predicted_dloss_raw_render sidecar",
                )
                new_entry["predicted_dloss"] = float(
                    entry["predicted_dloss_raw_render"]
                )
            if "weight_mse_per_expert" in entry:
                _require(
                    "weight_mse_per_expert_raw_render" in entry,
                    f"{where}: {qname!r}/{fmt!r} has weight_mse_per_expert "
                    "but no per-expert raw-render sidecar",
                )
                raw_vector = entry["weight_mse_per_expert_raw_render"]
                _require(
                    len(raw_vector) == len(entry["weight_mse_per_expert"]),
                    f"{where}: {qname!r}/{fmt!r} per-expert sidecar length "
                    "mismatch",
                )
                new_entry["weight_mse_per_expert"] = [
                    float(value) for value in raw_vector
                ]
            if entry.get("expert_cost_extrapolated"):
                new_entry["expert_cost_extrapolated"] = True
            swapped += 1
            new_row[fmt] = new_entry
        new_costs[qname] = new_row
    _require(
        swapped > 0,
        f"{where}: no LDLQ-covered CB rows found; refusing to emit a "
        "trivially relabeled table",
    )

    new_provenance = dict(provenance)
    raw_stamp = cb_serialization_context_stamp(
        raw_context,
        formats=[fmt for fmt in formats if is_cb_format(str(fmt))],
    )
    new_provenance["cb_serialized_payload"] = raw_stamp
    # The LDLQ-only provenance notes do not describe the extracted table.
    new_provenance.pop("cb_ldlq_cold_expert_prior", None)
    new_provenance.pop("cb_ldlq_raw_render_sidecar", None)
    identity = provenance.get("cb_render_identity")
    if isinstance(identity, dict):
        new_identity = dict(identity)
        cb_formats = sorted({
            str(fmt)
            for fmts in identity.get("cb_formats_by_qname", {}).values()
            for fmt in (fmts if isinstance(fmts, (list, tuple)) else [fmts])
        }) or [fmt for fmt in formats if is_cb_format(str(fmt))]
        new_identity["cb_serialized_payload"] = cb_serialization_context_stamp(
            raw_context, formats=cb_formats,
        )
        new_provenance["cb_render_identity"] = new_identity
    new_provenance["derived_from_ldlq_gated_cost"] = {
        "schema": "prismaquant.cb_raw_cost_extraction.v1",
        "source_cb_serialized_payload": stamp,
        "note": (
            "metrics on LDLQ-covered CB rows are the *_raw_render sidecar "
            "of an LDLQ-gated cost run: the identical-env no-LDLQ render "
            "(same encode, codebook, scale sweep/coding, col_weights), "
            "captured pre-gate in the same pass. Output-side metrics were "
            "not re-measured for the raw arm."
        ),
    }

    new_payload = dict(payload)
    new_payload["costs"] = new_costs
    new_payload["provenance"] = new_provenance
    meta = payload.get("meta")
    if isinstance(meta, dict):
        new_meta = dict(meta)
        new_meta["ldlq_raw_render_extraction"] = {
            "rows_swapped_to_raw_sidecar": swapped,
        }
        new_payload["meta"] = new_meta
    # The extracted table must satisfy the same fail-closed identity gate any
    # CB cost consumer applies, under the no-LDLQ context it now claims.
    validate_cb_cost_provenance(
        new_payload,
        [str(fmt) for fmt in formats],
        context=raw_context,
        where=where,
    )
    return new_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="gated LDLQ cost pickle")
    parser.add_argument("output", type=Path, help="raw (no-LDLQ) cost pickle")
    args = parser.parse_args(argv)
    with open(args.input, "rb") as fh:
        payload = pickle.load(fh)
    try:
        extracted = extract_raw_cost_payload(
            payload, where=f"extract_raw_cost_table({args.input})"
        )
    except (RawSidecarExtractionError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(extracted, fh)
    rows = extracted["meta"]["ldlq_raw_render_extraction"][
        "rows_swapped_to_raw_sidecar"
    ] if isinstance(extracted.get("meta"), dict) else "?"
    print(
        f"wrote {args.output} ({len(extracted['costs'])} Linears, "
        f"{rows} rows swapped to the raw sidecar, identity ldlq=0/scope=none)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
