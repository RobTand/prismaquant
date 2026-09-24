"""Seal a Stage A seed spec and the data manifest its PrismaBuild action stages.

A seed (``prismaquant.stage_a_chain_seed``, PQ #1016) continues a source
run's sealed checkpoint ``b`` down to ``through`` in a scratch output root.
Under the campaign's ``ram,ssd`` tier policy every bulk byte it reads must be
staged, so its action is submitted with a data manifest that declares them
(PQ #1043). This builder derives that manifest from the source run's own
submitted manifest, the way ``build_stagea_forward_recovery_package`` derives
a recovery manifest: it keeps the source run's reads at the phases the seed
enters, and adds the checkpoint files the seed borrows at the phases that
read them.

* ``head``: the source run's head reads, less the head walk's, plus the
  seed spec, and the ``checkpoint.json`` and shared-state files of
  checkpoint ``b``, which the seed reads while it restores the chain. Stage A
  takes its head from the prepared completion (PQ #1051), so the walk's
  reads leave a source manifest built before #1051
  (``stage_a_head.drop_source_head_walk_reads``, from the plan's
  ``inputs``). The plan must be the one the source manifest names.
* ``chain-(b-1)`` down to ``chain-(through)``: the source run's reads of
  those phases (each layer's weights and its forward boundary rows).
* ``chain-(b-1)`` also stages checkpoint ``b``'s cotangent plane, which the
  first roll reads.
* ``chain-(through)`` also stages the compare checkpoint's
  ``checkpoint.json`` and plane, which the seed reads after its last roll.

The seed runs no forward pass and no tail, so every ``forward-*`` phase is
dropped. The source manifest must name a forward-recovery capsule, and it
must be the seed's: its chain phases hold that capsule's boundary rows, the
rows the seed borrows.

The package directory holds ``seed-spec.json``, ``seed-manifest.json.gz``
and ``package.json`` (digests, phases, the head-walk entries and bytes left
out, and the peak bytes of any two consecutive phases, the lead and next
windows PrismaBuild admits together).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_layer_quanta import adjoint_chain_phase_name  # noqa: E402
from prismaquant.joint_adjoint_slices import checkpoint_manifest_entry  # noqa: E402
from prismaquant.stage_a_head import drop_source_head_walk_reads  # noqa: E402
from prismaquant.stage_a_chain_seed import (  # noqa: E402
    SEED_SPEC_SCHEMA,
    load_pinned_checkpoint,
    normalize_seed_spec,
)

SPEC_NAME = "seed-spec.json"
MANIFEST_NAME = "seed-manifest.json.gz"
PACKAGE_NAME = "package.json"
PACKAGE_SCHEMA = "prismaquant.stage_a.seed_package.v1"


class SeedPackageRefused(ValueError):
    """The seed cannot be staged from this source manifest."""


def _entry(path, size, sha256) -> dict:
    return {"path": str(path), "offset": 0, "bytes": int(size), "sha256": sha256}


def _checkpoint_files(record) -> tuple[list[dict], list[dict], list[dict]]:
    """``(manifest, shared states, cotangents)`` of a checkpoint as entries."""
    manifest = checkpoint_manifest_entry(record)
    return ([_entry(manifest["path"], manifest["file_bytes"], manifest["sha256"])],
            [_entry(row["path"], row["file_bytes"], row["sha256"])
             for row in record["shared_state_entries"]],
            [_entry(row["path"], row["file_bytes"], row["sha256"])
             for row in record["activation_entries"]])


def peak_consecutive_bytes(phases) -> int:
    """The largest bytes of one phase plus the phase after it."""
    sizes = [int(phase["bytes"]) for phase in phases]
    return max([a + b for a, b in zip(sizes, sizes[1:])] + sizes)


def seed_manifest(original, *, spec, spec_entry, checkpoint, compare=None) -> dict:
    """The seed's data manifest, derived from the source run's ``original``.

    ``spec`` is the normalized seed spec, ``spec_entry`` its own manifest
    entry, and ``checkpoint``/``compare`` the pinned checkpoint records.
    """
    boundary, through = int(checkpoint["boundary"]), int(spec["through"])
    if not through < boundary:
        raise SeedPackageRefused(
            f"a seed from checkpoint {boundary} rolls to a boundary below it, not {through}")
    if compare is not None and int(compare["boundary"]) != through:
        raise SeedPackageRefused(
            f"the compare checkpoint is boundary {compare['boundary']}; the seed stops "
            f"at {through}")
    annotations = original.get("annotations")
    if not isinstance(annotations, dict):
        raise SeedPackageRefused("the source manifest has no annotations")
    recovery = annotations.get("forward_recovery")
    if recovery is None:
        raise SeedPackageRefused(
            "the source manifest names no forward-recovery capsule: its chain phases "
            "hold the source run's own boundary rows, not the capsule rows the seed reads")
    if recovery != spec["capsule"]:
        raise SeedPackageRefused(
            "the source manifest staged another forward-recovery capsule than the "
            "seed's: its chain phases hold that capsule's boundary rows")
    by_name = {phase["name"]: phase for phase in original["read_plan"]["phases"]}
    names = ["head", *(adjoint_chain_phase_name(layer)
                       for layer in range(boundary - 1, through - 1, -1))]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SeedPackageRefused(f"the source manifest has no phase {missing}")

    manifest = copy.deepcopy(original)
    entries = manifest["entries"]
    added: dict[str, list[int]] = {name: [] for name in names}

    def add(name, rows):
        for row in rows:
            added[name].append(len(entries))
            entries.append(row)

    ckpt_manifest, ckpt_shared, ckpt_plane = _checkpoint_files(checkpoint)
    add("head", [spec_entry, *ckpt_manifest, *ckpt_shared])
    add(names[1], ckpt_plane)
    if compare is not None:
        cmp_manifest, _, cmp_plane = _checkpoint_files(compare)
        add(names[-1], [*cmp_manifest, *cmp_plane])

    try:
        assemble_phases(manifest, by_name, names, added)
    except ValueError as exc:
        raise SeedPackageRefused(f"a seed input {exc}") from exc
    manifest["annotations"]["chain_seed"] = {"path": spec_entry["path"],
                                             "sha256": spec_entry["sha256"]}
    return manifest


def assemble_phases(manifest, by_name, names, added) -> dict:
    """Keep ``names``' source phases, each extended by its ``added`` entries.

    ``manifest`` is a copy of the source manifest whose ``entries`` already
    end with the added entries; ``added`` maps a phase name to their
    indices. Entries no kept phase reads are dropped and the rest
    renumbered, and each phase's bytes and running total are recomputed.
    Raises ``ValueError`` when an added entry repeats a kept one.
    """
    entries = manifest["entries"]
    phases = []
    for name in names:
        phase = copy.deepcopy(by_name[name])
        phase["entry_indices"] = [*phase["entry_indices"], *added[name]]
        phases.append(phase)
    used = sorted({index for phase in phases for index in phase["entry_indices"]})
    paths = [(entries[index]["path"], entries[index]["offset"]) for index in used]
    if len(set(paths)) != len(paths):
        raise ValueError("is already a source-manifest entry")
    remap = {old: new for new, old in enumerate(used)}
    entries = [entries[index] for index in used]
    cumulative = 0
    for phase in phases:
        phase["entry_indices"] = [remap[index] for index in phase["entry_indices"]]
        phase["bytes"] = sum(entries[index]["bytes"] for index in phase["entry_indices"])
        cumulative += phase["bytes"]
        phase["cumulative_bytes"] = cumulative
    manifest.update(entries=entries, entry_count=len(entries),
                    total_bytes=sum(entry["bytes"] for entry in entries))
    manifest["read_plan"] = {"phases": phases, "read_bytes": cumulative}
    return manifest


def manifest_wire(manifest) -> bytes:
    """The manifest's sealed bytes: sorted compact JSON, gzip without a timestamp."""
    return gzip.compress((json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                          + "\n").encode(), mtime=0)


def _pin(path, sha256, what) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SeedPackageRefused(f"the {what} {path} does not have the pinned digest {sha256}")
    return {"path": str(path), "sha256": sha256}


def build(*, original_manifest, original_manifest_sha256, plan, checkpoint, capsule,
          through, implementation_from, implementation_to, compare, output,
          validate=None) -> dict:
    """Write the package into ``output``, a directory that must not exist.

    ``plan`` is the pinned ``{path, sha256}`` of the plan the source manifest
    names; its ``inputs`` say which head entries are the head walk's.
    """
    wire = Path(original_manifest).read_bytes()
    if hashlib.sha256(wire).hexdigest() != original_manifest_sha256:
        raise SeedPackageRefused("the source manifest does not have the pinned digest")
    original = json.loads(gzip.decompress(wire) if wire[:2] == b"\x1f\x8b" else wire)
    try:
        original, dropped = drop_source_head_walk_reads(original, plan)
    except ValueError as error:
        raise SeedPackageRefused(str(error)) from error
    declared = (None if implementation_from is None and implementation_to is None
                else {"from": implementation_from, "to": implementation_to})
    spec = normalize_seed_spec({
        "schema": SEED_SPEC_SCHEMA, "checkpoint": checkpoint, "capsule": capsule,
        "through": through, "implementation_compatibility": declared,
        "compare": compare})
    record = load_pinned_checkpoint(spec["checkpoint"], "checkpoint")
    reference = (None if spec["compare"] is None
                 else load_pinned_checkpoint(spec["compare"], "compare checkpoint"))
    spec_bytes = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    root = Path(output)
    spec_entry = _entry(root / SPEC_NAME, len(spec_bytes),
                        hashlib.sha256(spec_bytes).hexdigest())
    manifest = seed_manifest(original, spec=spec, spec_entry=spec_entry,
                             checkpoint=record, compare=reference)
    if validate is not None:
        validate(manifest)
    manifest_bytes = manifest_wire(manifest)
    phases = manifest["read_plan"]["phases"]
    package = {
        "schema": PACKAGE_SCHEMA,
        "seed_spec": {"path": spec_entry["path"], "sha256": spec_entry["sha256"]},
        "data_manifest": {"path": str(root / MANIFEST_NAME),
                          "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "source_manifest": {"path": str(original_manifest),
                            "sha256": original_manifest_sha256},
        "phases": [{"name": phase["name"], "bytes": phase["bytes"],
                    "entries": len(phase["entry_indices"])} for phase in phases],
        "read_bytes": manifest["read_plan"]["read_bytes"],
        "head_walk_reads_dropped": dropped,
        "peak_consecutive_phase_bytes": peak_consecutive_bytes(phases),
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / SPEC_NAME).write_bytes(spec_bytes)
    (root / MANIFEST_NAME).write_bytes(manifest_bytes)
    (root / PACKAGE_NAME).write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def _pb_validate():
    from prismaquant.staged_lease import sdk_submodule
    return sdk_submodule("core").validate_data_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original-manifest", required=True,
                        help="the source run's submitted Stage A data manifest")
    parser.add_argument("--original-manifest-sha256", required=True)
    parser.add_argument("--plan", required=True,
                        help="the plan the source manifest names; its inputs "
                             "say which head entries are the head walk's")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="the source run's sealed checkpoint.json the seed continues")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--capsule-sha256", required=True)
    parser.add_argument("--through", type=int, required=True)
    parser.add_argument("--implementation-from", default=None)
    parser.add_argument("--implementation-to", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--compare-sha256", default=None)
    parser.add_argument("--output", required=True, help="package directory (must not exist)")
    args = parser.parse_args(argv)
    if (args.compare is None) != (args.compare_sha256 is None):
        parser.error("--compare and --compare-sha256 must be paired")
    package = build(
        original_manifest=args.original_manifest,
        original_manifest_sha256=args.original_manifest_sha256,
        plan={"path": args.plan, "sha256": args.plan_sha256},
        checkpoint=_pin(args.checkpoint, args.checkpoint_sha256, "checkpoint"),
        capsule=_pin(args.capsule, args.capsule_sha256, "capsule"),
        through=args.through, implementation_from=args.implementation_from,
        implementation_to=args.implementation_to,
        compare=(None if args.compare is None
                 else _pin(args.compare, args.compare_sha256, "compare checkpoint")),
        output=args.output, validate=_pb_validate())
    print(json.dumps(package, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
