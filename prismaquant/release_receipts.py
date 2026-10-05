"""Offline, dry-by-default ingestion of release-window evidence.

Consume existing producer-authored shipcard records, not hand-set pass flags.
U4's raw full-config traces use the existing trace constructor and verifier.
Unsupported measurements remain refusals; this command never runs a serve,
measures, uploads, copies a manifest, or fabricates measurement provenance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from prismaquant.digests import DIRECT_ASCII_INDENT2_LAX
from prismaquant.shipcard import (
    _strict_json_object,
    fill_slot,
    load_shipcard,
    verify,
)
from prismaquant.tessera_shipcard import make_route_trace_record

_MAX_RECEIPT_BYTES = 128 * 1024 * 1024


def _read_release_receipt(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_RECEIPT_BYTES + 1)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ValueError(f"receipt exceeds {_MAX_RECEIPT_BYTES} bytes: {path}")
    return dict(_strict_json_object(raw, where=str(path)))


def _window_trace(window: dict, card: dict, model: Path) -> dict | None:
    # The collector names the full-config phase explicitly. Never fall back to
    # body_scope, a summarized AGREE flag, or an arbitrary successful phase.
    full = window.get("route_trace_full_config") or {}
    phase = full.get("source_phase")
    if phase is None:
        return None
    if not isinstance(phase, str):
        raise ValueError("full-config trace source_phase must be a string")
    phases = window.get("phases") or {}
    verdict = ((phases.get(phase) or {}).get("route_trace") or {}).get("verdict")
    if not isinstance(verdict, dict) or full.get("verdict") != verdict:
        raise ValueError("full-config trace must match its source phase verdict")
    if verdict.get("schema") != "prismaquant.pact_u4.route_trace/1":
        raise ValueError("unsupported U4 route-trace wrapper")
    paths = verdict.get("traces")
    if not isinstance(paths, dict) or set(paths) != {"rank0", "rank1"}:
        raise ValueError("full-config TP2 trace requires rank0 and rank1 paths")
    if not all(isinstance(p, str) and Path(p).is_absolute() for p in paths.values()):
        raise ValueError("U4 trace paths must be explicit absolute paths")
    if Path(paths["rank0"]).resolve() == Path(paths["rank1"]).resolve():
        raise ValueError("TP2 trace paths must be distinct")
    return make_route_trace_record(
        tool="prismaquant.release_receipts:U4-full-config-trace",
        model_sha=card["model_sha"],
        traces=[(rank, _read_release_receipt(Path(paths[rank]))) for rank in ("rank0", "rank1")],
        expected_ranks=2, config_json=(model / "config.json").read_text(),
        build=card.get("build"), platform=verdict.get("platform"),
    )


def ingest_receipts(
    shipcard: str | Path, *, window: str | Path | None = None,
    records_dir: str | Path | None = None, apply: bool = False,
) -> dict:
    """Replay all proposed records before writing any; retain final refusals."""
    path = Path(shipcard).resolve(strict=True)
    if path.name != "shipcard.json":
        raise ValueError("use the artifact's canonical shipcard.json")
    model = path.parent
    card = load_shipcard(path)
    if Path(card.get("model_dir", "")).resolve() != model:
        raise ValueError("shipcard model_dir must name its current artifact directory")
    candidate = copy.deepcopy(card)
    records: dict[str, dict] = {}
    unsupported: list[str] = []
    input_problems: list[str] = []
    if records_dir is not None:
        root = Path(records_dir).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("records_dir must be a directory")
        for entry in sorted(root.glob("*.json")):
            slot = entry.stem
            if slot not in card["slots"]:
                raise ValueError(f"unknown/unopened slot record: {entry.name}")
            record = _read_release_receipt(entry)
            if record.get("slot") != slot or record.get("model_sha") != card["model_sha"]:
                input_problems.append(f"{slot}: producer slot/model_sha does not match the card")
            records[slot] = record
    if window is not None:
        payload = _read_release_receipt(Path(window))
        if payload.get("schema") != "prismaquant.pact_u4.arm/1":
            raise ValueError("unsupported release-window schema")
        artifact = payload.get("artifact")
        if not isinstance(artifact, str) or Path(artifact).resolve() != model:
            raise ValueError("window artifact does not match the shipcard")
        if (payload.get("kl") or {}).get("full") and "gold.kl" not in records:
            unsupported.append(
                "gold.kl: raw TR3 output is not a shipcard record; producer must carry "
                "the shipcard model_sha, observed spec_decode_detected, gold metrics "
                "and canonical serving-manifest binding. No identity or observation was invented."
            )
        if "route.trace" not in records:
            try:
                record = _window_trace(payload, card, model)
                if record is not None:
                    if "route.trace" not in card["slots"]:
                        raise ValueError("card has no route.trace slot")
                    records["route.trace"] = record
            except (ValueError, OSError, KeyError, TypeError) as exc:
                input_problems.append(f"route.trace: {exc}")
    candidate["slots"].update(records)
    # Do not parse verifier error strings or suppress global build/identity
    # errors. This is the exact verifier, scoped only for partial-ingest replay.
    preflight = verify(candidate, model_dir=model, required=sorted(records))
    applied: list[str] = []
    if apply and not input_problems and not preflight:
        for slot, record in sorted(records.items()):
            fill_slot(path, slot, record)  # Existing stat fence and atomic writer.
            applied.append(slot)
        candidate = load_shipcard(path)
    problems = verify(candidate, model_dir=model)
    return {
        "status": "ready" if not (problems or input_problems or preflight or unsupported) else "refused",
        "mode": "apply" if apply else "dry_run",
        "prepared_slots": sorted(records), "applied_slots": applied,
        "input_problems": input_problems, "preflight_problems": preflight,
        "unsupported_outputs": unsupported, "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shipcard")
    parser.add_argument("--window", help="U4 arm receipt; raw unsupported outputs remain refusals")
    parser.add_argument("--records-dir", help="producer slot records named <slot>.json")
    parser.add_argument("--apply", action="store_true", help="fill replayed records; default writes nothing")
    args = parser.parse_args(argv)
    try:
        report = ingest_receipts(args.shipcard, window=args.window,
                                 records_dir=args.records_dir, apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(DIRECT_ASCII_INDENT2_LAX.text(report))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
