#!/usr/bin/env python3
"""List every source read a staged package does not declare (PQ #1095).

A strictly staged run refuses a source tensor its read manifest does not
stage whole, at the moment it reads it (``readset-not-staged``). This tool
finds every such read before submission. It reads the checkpoint index and
safetensors headers only, needs no GPU, and reports every gap in one JSON
document (``prismaquant.readset_coverage``). It exits 0 when the package is
covered and 3 when it is not.

``stage-b --records DIR``
    Every bound Stage B quantum record in ``DIR`` that carries an executable
    readset, against the manifest it binds: the resident head in ``head``,
    each chain layer in ``chain-LLL-source``, the own layer in
    ``own-LLL-source``. ``dispatch_joint_quanta`` runs the same check before
    it publishes or prints a row.

``stage-a --manifest M --plan P --prepared Q --top T --bottom B``
    A Stage A chain walk from layer ``T`` down to ``B`` against manifest
    ``M``: the resident head in ``head`` and each layer in the phase
    ``--phase-format`` names (``chain-{layer:03d}`` by default). The prefetch
    lookahead is the plan's, or ``--prefetch-override``'s. ``--schedule``
    picks the prefetch schedule: ``stage-a`` is the one ``joint_cost_stage_a``
    runs (``readset_coverage.stage_a_prefetch_targets``); ``walk`` is a Stage
    B quantum's (``source_read_plan.chain_prefetch_window``). Both stop at the
    walk's last layer since PQ #1100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.readset_coverage import (  # noqa: E402
    gaps_report, load_manifest, load_source_plan, quantum_prefetch_targets,
    quantum_rows_gaps, source_read_gaps, stage_a_prefetch_targets,
)
from prismaquant.source_read_plan import roster_layers_prefix  # noqa: E402

EXIT_GAPS = 3


def _load_json(path: str, sha256: str | None = None):
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{path} does not hash to {sha256}")
    return json.loads(raw)


def _stage_b(args) -> dict:
    rows = []
    for path in sorted(Path(args.records).glob("layer-*.json")):
        record = _load_json(str(path))
        executable = record.get("executable_readset")
        if executable is None:
            continue
        if args.quantum_id and record.get("quantum_id") not in args.quantum_id:
            continue
        rows.append({"record": record,
                     "manifest_path": executable["manifest_path"],
                     "manifest_sha256": executable["manifest_sha256"],
                     "order": None})
    if not rows:
        raise ValueError(f"no bound executable quantum record in {args.records}")
    return gaps_report(quantum_rows_gaps(rows), mode="stage-b",
                       records=str(Path(args.records).resolve()),
                       quanta=[row["record"]["quantum_id"] for row in rows])


def _stage_a(args) -> dict:
    plan = _load_json(args.plan, args.plan_sha256)
    prepared = _load_json(args.prepared, args.prepared_sha256)
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    layers_prefix = roster_layers_prefix(prepared.get("formats_by_qname") or {})
    prefetch = plan.get("source_prefetch") or {}
    if args.prefetch_override is not None:
        # A joint_adjoint_capture prefetch override input: its
        # ``source_prefetch`` block replaces the plan's.
        override = _load_json(args.prefetch_override)
        prefetch = {**prefetch, **override.get("source_prefetch", override)}
    lookahead = prefetch.get("prefetch_lookahead")
    if type(lookahead) is not int or lookahead < 1:
        raise ValueError(f"no positive prefetch lookahead ({lookahead!r})")
    if not args.top >= args.bottom >= 0:
        raise ValueError("a chain walk runs from --top down to --bottom")
    order = list(range(args.top, args.bottom - 1, -1))
    targets = (stage_a_prefetch_targets(order, lookahead)
               if args.schedule == "stage-a"
               else quantum_prefetch_targets(order, lookahead))
    source_plan = load_source_plan(plan["model"], layers_prefix,
                                   sorted(set(order) | targets))
    gaps = source_read_gaps(
        manifest, source_plan, order=order, prefetch_targets=targets,
        phase_of=lambda layer: args.phase_format.format(layer=layer))
    return gaps_report(gaps, mode="stage-a", manifest=str(Path(args.manifest).resolve()),
                       order=order, lookahead=lookahead, schedule=args.schedule,
                       layers_prefix=layers_prefix)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("stage-b", help="bound Stage B quantum records")
    b.add_argument("--records", required=True)
    b.add_argument("--quantum-id", action="append", default=[],
                   help="check only these quanta (repeatable)")
    a = sub.add_parser("stage-a", help="a Stage A chain walk")
    a.add_argument("--manifest", required=True)
    a.add_argument("--manifest-sha256", default=None)
    a.add_argument("--plan", required=True)
    a.add_argument("--plan-sha256", default=None)
    a.add_argument("--prepared", required=True)
    a.add_argument("--prepared-sha256", default=None)
    a.add_argument("--top", type=int, required=True)
    a.add_argument("--bottom", type=int, required=True)
    a.add_argument("--prefetch-override", default=None)
    a.add_argument("--phase-format", default="chain-{layer:03d}")
    a.add_argument("--schedule", choices=("stage-a", "walk"), default="stage-a")
    parser.add_argument("--out", default=None, help="also write the report here")
    args = parser.parse_args(argv)
    try:
        report = _stage_b(args) if args.mode == "stage-b" else _stage_a(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"check_readset_coverage: refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out is not None:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["covered"] else EXIT_GAPS


if __name__ == "__main__":
    sys.exit(main())
