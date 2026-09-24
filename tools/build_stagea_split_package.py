"""Seal the data manifests of one Stage A chain split round (PQ #738).

A split round (``prismaquant.stage_a_chain_split``) is one prep row and one
quantum row per sample range, all under one run's chain state. Under the
campaign's ``ram,ssd`` tier policy every bulk byte a row reads must be a
declared, staged manifest entry: the capture reads the run's own forward
boundary rows and the sealed checkpoint's cotangents through the process
input map, never through the row's produced output. This builder derives
each row's manifest from the source run's submitted manifest, the way
``tools/build_stagea_seed_package.py`` derives a seed's:

* **The prep** (``prep-manifest.json.gz``): the source run's ``head``
  reads, plus ``chain-state.json`` and the ``checkpoint.json`` of every
  sealed checkpoint, which the prep reads to plan the round. It rolls
  nothing, so it stages no chain phase.
* **Each quantum** (``quantum-samples-SSSSSS-EEEEEE-manifest.json.gz``): the
  prep's head plus the resume checkpoint's shared-state pack, then the
  source run's reads of ``chain-(from-1)`` down to ``chain-(through)`` (each
  layer's weights), each followed by that layer's forward boundary rows for
  the quantum's samples only. ``chain-(from-1)`` also stages the resume
  checkpoint's cotangent rows for those samples, which the first roll reads.

The resume checkpoint is the run's lowest sealed one, as
``plan_chain_resume`` finds it: the tail checkpoint in round 1, a joined
checkpoint afterwards. Every ``forward-*`` phase is dropped, and so are the
head walk's reads of a source manifest built before PQ #1051
(``stage_a_head.drop_source_head_walk_reads``). Every added entry is
stat-checked against its recorded size, and the PrismaBuild core validator
checks every manifest.

The package directory holds the manifests and ``split-package.json`` (their
digests, phases, bytes and peak consecutive phase bytes, and the source
bytes each quantum reads, which a round reads once per quantum).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_adjoint_slices import checkpoint_cotangent_plane  # noqa: E402
from prismaquant.joint_layer_quanta import adjoint_chain_phase_name  # noqa: E402
from prismaquant.stage_a_head import drop_source_head_walk_reads  # noqa: E402
from tools.build_stagea_seed_package import (  # noqa: E402
    _checkpoint_files,
    _entry,
    _pb_validate,
    assemble_phases,
    manifest_wire,
    peak_consecutive_bytes,
)

PACKAGE_NAME = "split-package.json"
PACKAGE_SCHEMA = "prismaquant.stage_a.split_package.v1"
PREP_MANIFEST_NAME = "prep-manifest.json.gz"
_SOURCE = re.compile(r"\.safetensors$")


class SplitPackageRefused(ValueError):
    """The round cannot be staged from these inputs."""


def quantum_manifest_name(start: int, stop: int) -> str:
    from prismaquant.stage_a_chain_split import range_name
    return f"quantum-{range_name(start, stop)}-manifest.json.gz"


def load_round(output_root, chain_state_sha256, checkpoint_sha256) -> dict:
    """The run's chain state and sealed checkpoints, checked the way a relaunch checks them.

    Returns ``{space, state, state_entry, sealed, boundary}``: ``sealed``
    maps each sealed checkpoint's boundary to its record, and ``boundary``
    is the lowest, which must be the checkpoint ``checkpoint_sha256`` pins.
    """
    from prismaquant.joint_adjoint_checkpoints import adjoint_space
    from prismaquant.stage_a_chain_resume import (
        ChainResumeRefused, _sealed_checkpoints, chain_state_path, load_chain_state)

    space = adjoint_space(output_root)
    try:
        state = load_chain_state(space, chain_state_sha256)
        session = state["boundary_storage"]["session"]
        marker = {"generation": session["generation"], "kind": "adjoint_checkpoint",
                  "run_identity_sha256": session["run_identity_sha256"]}
        sealed, _partial = _sealed_checkpoints(
            space, marker, [int(mark) for mark in state["stride"]["boundaries"]])
    except ChainResumeRefused as exc:
        raise SplitPackageRefused(str(exc)) from exc
    if not sealed:
        raise SplitPackageRefused("the run has no sealed checkpoint to split from")
    boundary = min(sealed)
    path = Path(sealed[boundary][1]) / "checkpoint.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_sha256:
        raise SplitPackageRefused(
            f"the lowest sealed checkpoint {path} does not have the pinned digest "
            f"{checkpoint_sha256}")
    raw = chain_state_path(space).read_bytes()
    return {"space": space, "state": state, "boundary": boundary,
            "sealed": {mark: record for mark, (record, _directory) in sealed.items()},
            "state_entry": _entry(chain_state_path(space), len(raw), chain_state_sha256)}


def _boundary_rows(state, layer, start, stop) -> list[dict]:
    """Layer ``layer``'s forward boundary rows for samples ``start:stop``, checked."""
    session = state["boundary_storage"]["session"]
    rows = state["boundary_entries"][str(layer)]
    if len(rows) != int(state["n_batches"]):
        raise SplitPackageRefused(f"boundary {layer} records {len(rows)} rows")
    picked = []
    for batch in range(start, stop):
        row = rows[batch]
        identity = row["metadata"]["identity"]
        if (identity["kind"] != "boundary" or identity["session"] != session
                or identity["coordinates"] != {"batch": batch, "boundary": layer,
                                               "probe": None}):
            raise SplitPackageRefused(
                f"row {row['name']} is not batch {batch} of boundary {layer} of the session")
        picked.append(_entry(row["path"], row["file_bytes"], row["sha256"]))
    return picked


def _plane_rows(record, start, stop) -> list[dict]:
    """The checkpoint's cotangent rows for samples ``start:stop``, in record order."""
    try:
        plane = checkpoint_cotangent_plane(record)
    except ValueError as exc:
        raise SplitPackageRefused(f"checkpoint {record['boundary']}: {exc}") from exc
    names = {row["name"] for (_probe, batch), row in plane.items() if start <= batch < stop}
    return [_entry(row["path"], row["file_bytes"], row["sha256"])
            for row in record["activation_entries"] if row["name"] in names]


def _derive(original, names, added_rows, annotation) -> dict:
    """``original`` with ``names``' phases, each extended by ``added_rows[name]``."""
    by_name = {phase["name"]: phase for phase in original["read_plan"]["phases"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SplitPackageRefused(f"the source manifest has no phase {missing}")
    manifest = copy.deepcopy(original)
    entries = manifest["entries"]
    added = {name: [] for name in names}
    for name in names:
        for row in added_rows.get(name, ()):
            added[name].append(len(entries))
            entries.append(row)
    try:
        assemble_phases(manifest, by_name, names, added)
    except ValueError as exc:
        raise SplitPackageRefused(f"a split input {exc}") from exc
    manifest["annotations"]["chain_split"] = annotation
    return manifest


def round_manifests(original, loaded, *, through, ranges):
    """``(prep manifest, {(start, stop): quantum manifest})`` for one round."""
    from prismaquant.stage_a_chain_split import (
        ChainSplitRefused, check_ranges, split_boundaries)

    state, boundary = loaded["state"], loaded["boundary"]
    try:
        marks = split_boundaries(state["stride"]["boundaries"], boundary, through)
        check_ranges(ranges, n_batches=int(state["n_batches"]),
                     group_size=int(state["boundary_storage"]["policy"]["prefetch_batches"]),
                     where="the round's")
    except ChainSplitRefused as exc:
        raise SplitPackageRefused(str(exc)) from exc
    record = loaded["sealed"][boundary]
    # Highest first, as the relaunch reads them; the resume checkpoint last.
    records = [_checkpoint_files(loaded["sealed"][mark])[0][0]
               for mark in sorted(loaded["sealed"], reverse=True)]
    head = [loaded["state_entry"], *records]
    stamp = {"chain_state": {"path": loaded["state_entry"]["path"],
                             "sha256": loaded["state_entry"]["sha256"]},
             "from": boundary, "through": int(through), "boundaries": marks}
    prep = _derive(original, ["head"], {"head": head},
                   {**stamp, "role": "prep", "ranges": [list(pair) for pair in ranges]})
    _manifest, shared, _plane = _checkpoint_files(record)
    layers = range(boundary - 1, int(through) - 1, -1)
    names = ["head", *(adjoint_chain_phase_name(layer) for layer in layers)]
    quanta = {}
    for start, stop in sorted(tuple(pair) for pair in ranges):
        added = {"head": [*head, *shared]}
        added[names[1]] = _plane_rows(record, start, stop)
        for layer in layers:
            added.setdefault(adjoint_chain_phase_name(layer), []).extend(
                _boundary_rows(state, layer, start, stop))
        quanta[start, stop] = _derive(original, names, added,
                                      {**stamp, "role": "quantum", "samples": [start, stop]})
    return prep, quanta


def stat_check(manifest, original_paths) -> dict:
    """Every entry the builder added is on disk at its recorded size."""
    checked, missing, other = 0, [], []
    for entry in manifest["entries"]:
        if (entry["path"], entry["offset"]) in original_paths:
            continue
        checked += 1
        try:
            size = os.stat(entry["path"]).st_size
        except FileNotFoundError:
            missing.append(entry["path"])
            continue
        if size != entry["bytes"]:
            other.append([entry["path"], size, entry["bytes"]])
    if missing or other:
        raise SplitPackageRefused(
            f"declared inputs are not on disk as recorded: {len(missing)} missing, "
            f"{len(other)} of another size; first {(missing + other)[:3]}")
    return {"checked": checked, "missing": 0, "size_mismatch": 0}


def source_bytes(manifest) -> int:
    """The model source bytes the manifest's chain phases read."""
    entries = manifest["entries"]
    return sum(int(entries[index]["bytes"]) for phase in manifest["read_plan"]["phases"]
               if phase["name"].startswith("chain-") for index in phase["entry_indices"]
               if _SOURCE.search(entries[index]["path"]))


def _describe(path, wire, manifest, stat) -> dict:
    phases = manifest["read_plan"]["phases"]
    return {"data_manifest": {"path": str(path), "sha256": hashlib.sha256(wire).hexdigest()},
            "phases": [{"name": phase["name"], "bytes": phase["bytes"],
                        "entries": len(phase["entry_indices"])} for phase in phases],
            "entry_count": manifest["entry_count"],
            "read_bytes": manifest["read_plan"]["read_bytes"],
            "source_bytes": source_bytes(manifest),
            "peak_consecutive_phase_bytes": peak_consecutive_bytes(phases),
            "stat_check": stat}


def build(*, original_manifest, original_manifest_sha256, plan, output_root,
          chain_state_sha256, checkpoint_sha256, through, ranges, output,
          validate=None) -> dict:
    """Write the round's package into ``output``, a directory that must not exist."""
    wire = Path(original_manifest).read_bytes()
    if hashlib.sha256(wire).hexdigest() != original_manifest_sha256:
        raise SplitPackageRefused("the source manifest does not have the pinned digest")
    original = json.loads(gzip.decompress(wire) if wire[:2] == b"\x1f\x8b" else wire)
    try:
        original, dropped = drop_source_head_walk_reads(original, plan)
    except ValueError as error:
        raise SplitPackageRefused(str(error)) from error
    loaded = load_round(output_root, chain_state_sha256, checkpoint_sha256)
    prep, quanta = round_manifests(original, loaded, through=through, ranges=ranges)
    original_paths = {(entry["path"], entry["offset"]) for entry in original["entries"]}
    root = Path(output)
    written = [(root / PREP_MANIFEST_NAME, prep)]
    written += [(root / quantum_manifest_name(*pair), manifest)
                for pair, manifest in quanta.items()]
    described = []
    for path, manifest in written:
        stat = stat_check(manifest, original_paths)
        if validate is not None:
            validate(manifest)
        described.append((path, manifest_wire(manifest), manifest, stat))
    record = loaded["sealed"][loaded["boundary"]]
    package = {
        "schema": PACKAGE_SCHEMA,
        "source_manifest": {"path": str(original_manifest),
                            "sha256": original_manifest_sha256},
        "chain_state": {"path": loaded["state_entry"]["path"],
                        "sha256": chain_state_sha256,
                        "chain_state_sha256": loaded["state"]["chain_state_sha256"]},
        "checkpoint": {"boundary": loaded["boundary"], "sha256": checkpoint_sha256,
                       "cotangent_sha256": record["cotangent_sha256"]},
        "from": loaded["boundary"], "through": int(through),
        "boundaries": prep["annotations"]["chain_split"]["boundaries"],
        "ranges": [list(pair) for pair in sorted(tuple(pair) for pair in ranges)],
        "n_batches": int(loaded["state"]["n_batches"]),
        "head_walk_reads_dropped": dropped,
        "prep": _describe(*described[0]),
        "quanta": [{"samples": list(pair), **_describe(*item)}
                   for pair, item in zip(quanta, described[1:])],
    }
    package["round_source_bytes"] = sum(item["source_bytes"] for item in package["quanta"])
    root.mkdir(parents=True, exist_ok=False)
    for path, data, _manifest, _stat in described:
        path.write_bytes(data)
    (root / PACKAGE_NAME).write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def main(argv=None) -> int:
    from prismaquant.stage_a_chain_split import ChainSplitRefused, parse_ranges

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original-manifest", required=True,
                        help="the source run's submitted Stage A data manifest")
    parser.add_argument("--original-manifest-sha256", required=True)
    parser.add_argument("--plan", required=True,
                        help="the plan the source manifest names; its inputs say which "
                             "head entries are the head walk's")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", required=True, help="the run's output root")
    parser.add_argument("--chain-state-sha256", required=True,
                        help="the file digest of the run's chain-state.json")
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="the file digest of the lowest sealed checkpoint.json")
    parser.add_argument("--through", type=int, required=True)
    parser.add_argument("--ranges", required=True, metavar="S:E,...")
    parser.add_argument("--output", required=True, help="package directory (must not exist)")
    args = parser.parse_args(argv)
    try:
        ranges = parse_ranges(args.ranges)
    except ChainSplitRefused as exc:
        parser.error(str(exc))
    package = build(
        original_manifest=args.original_manifest,
        original_manifest_sha256=args.original_manifest_sha256,
        plan={"path": args.plan, "sha256": args.plan_sha256},
        output_root=args.output_root, chain_state_sha256=args.chain_state_sha256,
        checkpoint_sha256=args.checkpoint_sha256, through=args.through, ranges=ranges,
        output=args.output, validate=_pb_validate())
    print(json.dumps({key: value for key, value in package.items()
                      if key not in ("prep", "quanta")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
