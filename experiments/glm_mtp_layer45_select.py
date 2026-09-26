"""Select GLM-5.3's MTP layer 45 under its sub-budget (M6 of PQ #1271).

Research driver, not a pipeline stage. It reuses the owners:
``glm_mtp_selection.merge_mtp_costs`` joins the M5 parts (one Tessera rate
each), ``allocator._mtp_rung_attestation`` is the pinned runtime's
eligibility under the declared serving scope, and
``glm_mtp_selection.select_mtp_rungs`` chooses.

The sub-budget is the rival-matched MTP reservation from the whole-artifact
budget card: the routed experts' real bytes at the card's rung, plus BF16 for
the priced dense units (the card carries the shared expert in its BF16
non-expert group). No GLM-5.3 MTP serve constants have been measured, so the
constants declared here are inert placeholders and the run asserts that the
selector took its degenerate branch (lowest E within the budget).

Writes ``merged-cost.pkl``, ``selection.json`` and ``layer45-layer-config.json``
to ``--out``. Refuses on any failed check; nothing is written until every check
passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path


def _sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_part(path, payload, *, layer, units) -> dict:
    """The M5 verification of one part: layer, units, finite positive rows."""
    rungs = sorted({rung for rows in payload["costs"].values() for rung in rows})
    bad = [(unit, rung) for unit, rows in payload["costs"].items() for rung, row in rows.items()
           if not (math.isfinite(float(row["predicted_dloss"])) and float(row["predicted_dloss"]) > 0)]
    if payload["mtp_layer"] != layer:
        raise SystemExit(f"{path}: layer {payload['mtp_layer']}, expected {layer}")
    if units is not None and len(payload["costs"]) != units:
        raise SystemExit(f"{path}: {len(payload['costs'])} units, expected {units}")
    if bad:
        raise SystemExit(f"{path}: {len(bad)} non-finite or non-positive rows, e.g. {bad[:3]}")
    return {"path": str(path), "sha256": _sha(path), "rungs": rungs,
            "rows": sum(len(rows) for rows in payload["costs"].values()),
            "probe_identity_sha256": payload["provenance"]["probe_identity_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", action="append", required=True, help="M5 joint-cost.pkl (repeat)")
    ap.add_argument("--model", required=True, help="source checkpoint, for the model profile")
    ap.add_argument("--budget-card", required=True)
    ap.add_argument("--budget-group", default="mtp_experts")
    ap.add_argument("--layer", type=int, default=45)
    ap.add_argument("--units", type=int, default=867)
    ap.add_argument("--tessera-platform", required=True)
    ap.add_argument("--tessera-runtime-image", required=True)
    ap.add_argument("--tessera-execution-mode", required=True)
    ap.add_argument("--tessera-residency", required=True)
    ap.add_argument("--expect-unattested", default=None,
                    help="JSON {rung: count} the attestation must leave out, exactly")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from prismaquant import allocator
    from prismaquant import format_registry as fr
    from prismaquant.glm_mtp_selection import load_mtp_cost, merge_mtp_costs, select_mtp_rungs
    from prismaquant.lane_eligibility import STRUCTURE_ROUTED_MOE
    from prismaquant.model_profiles import detect_profile
    from prismaquant.tessera_campaign import contract_source_label
    from prismaquant.tessera_menu import MENU_ATTESTED, menu_mode
    from prismaquant.tessera_serving_scope import serving_target_from_args, unit_structure_from_profile

    if menu_mode() != MENU_ATTESTED:
        raise SystemExit(f"menu mode is {menu_mode()!r}; M6 selects attested rungs only")
    payloads = [load_mtp_cost(path) for path in args.part]
    parts = [_check_part(path, payload, layer=args.layer, units=args.units)
             for path, payload in zip(args.part, payloads)]
    merged = merge_mtp_costs(payloads, sources=[{"path": p["path"], "sha256": p["sha256"]}
                                                for p in parts])

    profile = detect_profile(args.model)
    target = serving_target_from_args(args)
    structure = {unit: unit_structure_from_profile(unit, profile) for unit in merged["costs"]}
    routed = sorted(unit for unit, kind in structure.items() if kind == STRUCTURE_ROUTED_MOE)
    dense = sorted(set(structure) - set(routed))

    card_path = Path(args.budget_card)
    card = json.loads(card_path.read_text())[args.budget_group]
    card_rung, card_real = str(card["rung"]), int(card["real_bytes"])
    if int(card["modules"]) != len(routed):
        raise SystemExit(f"card prices {card['modules']} routed units, the payload {len(routed)}")
    routed_at_card = (sum(int(merged["wire_bytes"][u][card_rung]) for u in routed)
                      if all(card_rung in merged["wire_bytes"][u] for u in routed) else None)
    dense_bf16 = sum(2 * int(merged["params"][u]) for u in dense)
    budget = card_real + dense_bf16

    constants = {"t_ms": 1.0, "d0_ms": 0.0, "c_ms_per_bit": 0.0,
                 "source": ("placeholder: no GLM-5.3 MTP serve constants measured; inert "
                            "without acceptance points (degenerate branch asserted)")}
    eligible = allocator._mtp_rung_attestation(target, profile)
    record = select_mtp_rungs(merged, byte_budget=budget, constants=constants,
                              acceptance_points=(), eligible=eligible)
    if record["selection"]["regime"] != "degenerate":
        raise SystemExit(f"selector regime {record['selection']['regime']!r}, expected degenerate")
    if args.expect_unattested is not None:
        expected = json.loads(args.expect_unattested)
        if record["unattested_rungs"] != expected:
            raise SystemExit(f"unattested {record['unattested_rungs']} != expected {expected}")

    assignment = record.pop("assignment")
    by_group = {}
    for name, members in merged["groups"].items():
        by_group[name] = {"units": len(members), "rung": record["rung_by_group"][name],
                          "structure": sorted({structure[u] for u in members}),
                          "bytes": sum(2 * int(merged["params"][u]) if assignment[u] == "BF16"
                                       else int(merged["wire_bytes"][u][assignment[u]])
                                       for u in members)}
    if record["resident_bytes"] != sum(group["bytes"] for group in by_group.values()):
        raise SystemExit(f"record resident bytes {record['resident_bytes']} != per-group sum")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged_path = out / "merged-cost.pkl"
    merged_path.write_bytes(pickle.dumps(merged))
    layer_cfg = {name: fr.get_format(fmt).autoround_config() for name, fmt in sorted(assignment.items())}
    (out / "layer45-layer-config.json").write_text(json.dumps(layer_cfg, indent=1, sort_keys=True) + "\n")
    import importlib.metadata as md
    try:
        tessera_version = md.version("tessera")
    except md.PackageNotFoundError:
        tessera_version = None
    result = {
        "schema": "prismaquant.experiments.glm_mtp_layer45_select.v1",
        "research_only": True,
        "parts": parts,
        "merged_cost": {"path": str(merged_path), "sha256": _sha(merged_path)},
        "serving_target": target.__dict__ if hasattr(target, "__dict__") else str(target),
        "menu_mode": menu_mode(),
        "contract": contract_source_label(),
        "tessera_package_version": tessera_version,
        "python": sys.executable,
        "profile": getattr(profile, "name", type(profile).__name__),
        "structure_counts": {"routed": len(routed), "dense": len(dense)},
        "budget": {"byte_budget": budget, "card": {"path": str(card_path), "sha256": _sha(card_path),
                                                   "group": args.budget_group, "rung": card_rung,
                                                   "real_bytes": card_real,
                                                   "closed_form_bytes": card.get("closed_form_bytes"),
                                                   "wire_sidecar_bytes": card.get("wire_sidecar_bytes")},
                   "payload_routed_bytes_at_card_rung": routed_at_card,
                   "dense_bf16_bytes": dense_bf16},
        "by_group": by_group,
        "record": record,
        "argv": sys.argv,
        "env": {k: os.environ.get(k) for k in ("PRISMAQUANT_TESSERA_MENU",) if k in os.environ},
    }
    (out / "selection.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"rung": record["rung"], "resident_bytes": record["resident_bytes"],
                      "byte_budget": budget, "E": record["E"], "bits": record["bits"],
                      "unattested": record["unattested_rungs"], "by_group": by_group}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
