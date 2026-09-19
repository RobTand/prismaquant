#!/usr/bin/env python3
"""The §9.3 cutover-gate comparison: single run vs layer quanta, bitwise.

Reads the single run's ``cost_stage_checkpoint`` journal and a layer
quantum's journal for the same layers, canonicalizes each shared unit's
measurement envelope (identity extras aside) and compares SHA-256 digests.
Bitwise equality is the requirement, not a tolerance: both sides replay the
same kernels in the same order from the same digest-checked bytes, so any
difference is a defect to find, never noise to threshold away.

Re-runnable and stateless: it owns nothing, writes nothing (unless --json is
given), and reads both journals fresh each run. A layer's gate arm resolves
when BOTH campaigns hold that layer's units; a layer present on one side
only is reported pending, not failed.

Usage:
  compare_joint_layer_gate.py --single-run DIR --quantum DIR \
      [--layers 44 43 42] [--qname-filter REGEX] [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import canonical_json_sha256  # noqa: E402

DEFAULT_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _journal_identity(checkpoint_dir: Path) -> dict:
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    identity_sha256 = manifest.get("identity_sha256")
    if not isinstance(identity_sha256, str):
        raise SystemExit(f"journal at {checkpoint_dir} has no identity digest")
    return {"identity_sha256": identity_sha256,
            "units": {row["qname"] for row in manifest.get("units", [])}}


def _load_unit(checkpoint_dir: Path, qname: str, identity_sha256: str) -> dict:
    from prismaquant.aura_cost import _aura_unit_checkpoint_path, _load_aura_unit_checkpoint

    return _load_aura_unit_checkpoint(
        _aura_unit_checkpoint_path(checkpoint_dir, qname), qname=qname,
        identity_sha256=identity_sha256)


#: State fields that bind the journal's identity, not the measurement: the
#: envelopes compared are the measured rows themselves (§9.3: "identity
#: extras aside").
IDENTITY_STATE_FIELDS = frozenset()


def measurement_envelope(state: Mapping) -> dict:
    """The canonicalized per-unit measurement both journals must share."""
    envelope = {key: value for key, value in state.items()
                if key not in IDENTITY_STATE_FIELDS}
    return envelope


def envelope_sha256(state: Mapping) -> str:
    return canonical_json_sha256(measurement_envelope(state),
                                 where="unit envelope")


def select_qnames(units: set[str], *, layer: int | None, qname_filter) -> set[str]:
    selected = set(units)
    if qname_filter is not None:
        selected = {name for name in selected if qname_filter.search(name)}
    if layer is not None:
        selected = {name for name in selected
                    if (m := DEFAULT_LAYER_PATTERN.search(name))
                    and int(m.group(1)) == layer}
    return selected


def compare_layer(single_dir: Path, quantum_dir: Path, *, layer: int | None,
                  qname_filter) -> dict:
    single = _journal_identity(single_dir)
    quantum = _journal_identity(quantum_dir)
    single_names = select_qnames(single["units"], layer=layer, qname_filter=qname_filter)
    quantum_names = select_qnames(quantum["units"], layer=layer, qname_filter=qname_filter)
    shared = sorted(single_names & quantum_names)
    verdict = {
        "layer": layer,
        "units_single": len(single_names),
        "units_quantum": len(quantum_names),
        "units_shared": len(shared),
        "pending": sorted(single_names ^ quantum_names),
        "matched": 0,
        "differed": [],
    }
    for name in shared:
        left = _load_unit(single_dir, name, single["identity_sha256"])
        right = _load_unit(quantum_dir, name, quantum["identity_sha256"])
        if envelope_sha256(left) == envelope_sha256(right):
            verdict["matched"] += 1
        else:
            verdict["differed"].append(name)
    verdict["verdict"] = ("match" if shared and not verdict["differed"]
                          and not verdict["pending"]
                          else ("differ" if verdict["differed"] else "pending"))
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the single run's and the layer quanta's unit "
                    "envelopes for the cutover gate (§9.3). Bitwise or it "
                    "fails.")
    parser.add_argument("--single-run", type=Path, required=True,
                        help="the single run's checkpoint journal directory")
    parser.add_argument("--quantum", type=Path, required=True,
                        help="a layer quantum's checkpoint journal directory")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="layer indexes to compare (default: every layer "
                             "both journals hold units for)")
    parser.add_argument("--qname-filter", default=None,
                        help="regex applied to qnames before layer selection")
    parser.add_argument("--json", type=Path, default=None,
                        help="optional verdict record to write")
    args = parser.parse_args(argv)

    qname_filter = (re.compile(args.qname_filter)
                    if args.qname_filter is not None else None)
    single = _journal_identity(args.single_run)
    quantum = _journal_identity(args.quantum)
    if args.layers is None:
        def layer_of(name):
            match = DEFAULT_LAYER_PATTERN.search(name)
            return int(match.group(1)) if match else None
        layers = sorted({layer for layer in
                         (layer_of(n) for n in single["units"] & quantum["units"])
                         if layer is not None})
        if not layers:
            layers = [None]
    else:
        layers = sorted(args.layers)

    verdicts = [compare_layer(args.single_run, args.quantum, layer=layer,
                              qname_filter=qname_filter) for layer in layers]
    all_match = all(v["verdict"] == "match" for v in verdicts) and verdicts
    print(f"single-run journal : {args.single_run} "
          f"({len(single['units'])} units)")
    print(f"quantum journal    : {args.quantum} ({len(quantum['units'])} units)")
    for verdict in verdicts:
        line = (f"layer {verdict['layer']:>3} : {verdict['verdict'].upper():7} "
                f"matched {verdict['matched']}/{verdict['units_shared']} shared "
                f"(single {verdict['units_single']}, quantum "
                f"{verdict['units_quantum']})")
        if verdict["pending"]:
            line += f" pending {len(verdict['pending'])}"
        if verdict["differed"]:
            line += f" differed {verdict['differed'][:8]}"
        print(line)
    record = {"schema": "prismaquant.joint_layer_gate_comparison.v1",
              "single_run": str(args.single_run),
              "quantum": str(args.quantum),
              "all_match": bool(all_match),
              "layers": verdicts}
    if args.json is not None:
        args.json.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
