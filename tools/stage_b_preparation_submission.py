#!/usr/bin/env python3
"""The PrismaBuild submission inputs for a Stage B preparation (PQ #1070).

Writes, for one run of ``tools/prepare_extended_joint_quanta.py`` (``prepare``)
or of ``tools/regenerate_joint_quanta.py`` alone (``regenerate``):

* ``read-manifest.json.gz``: the data manifest of what the run reads, for
  ``pbrun --data-manifest``;
* ``template.json``: the write-only produced-output template over the
  metadata root, for ``pbrun --produced-output-template``;
* ``submission.json``: both digests, the read totals and the ``pbrun``
  options that attach them.

It submits nothing. It reads the small control documents its arguments bind
and 8 bytes of each selected safetensors shard, and nothing else.

The read set, by source:

* the parent manifest's head phase: the head walk's inputs, which the
  generator's head intake reads once (PQ #1010), and the control files the
  extended parent adds;
* for ``prepare``, the control files ``prepare_extended_joint_quanta.
  control_paths`` names (the catalog pair, the policies, the overlay and its
  proof), which are the extended parent's additions;
* the files the arguments name: the catalog pair inputs or the plan and
  preparation, the parent manifest, the derivation, the partition, the spec,
  the Stage A proofs and a catalog extension that already exists;
* the production pickle the preparation binds;
* the source checkpoint's index and each selected shard's header
  (``joint_layer_quanta.layer_source_header_reads``).

Not declared: the files the run itself writes and then reads again. The
preparation writes ``parent.json.gz``, ``partition.json`` and
``derivation-input.json`` and the generator reads them back, and the
generator hash-checks each file it has just published. Those bytes do not
exist when the manifest is built.
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

from prismaquant.joint_layer_quanta import layer_source_header_reads, seal_manifest_bytes  # noqa: E402
from prismaquant.stage_b_prep_io import (  # noqa: E402
    build_preparation_template, head_phase_entries, preparation_payload_ceiling,
    preparation_read_entries, preparation_read_manifest)

SUBMISSION_SCHEMA = "prismaquant.stage_b_preparation_submission.v1"


def _load(path, sha256, where):
    from tools.regenerate_joint_quanta import _load_json
    return _load_json(Path(path), digest=sha256, where=where)


def _proof_files(args) -> tuple[list[Path], int]:
    """The Stage A proof files and the number of records indexes they produce."""
    files = []
    if args.adjoint_receipt is not None:
        files.append(Path(args.adjoint_receipt))
    files += [Path(path) for path in args.adjoint_band]
    if not files:
        raise ValueError("Stage B preparation needs sealed Stage A proof: the "
                         "completed receipt or a checkpoint band")
    indexes = 1 if args.adjoint_receipt is not None else len(args.adjoint_band)
    return files, indexes


def _source_reads(plan, parent, prefix):
    if prefix is None:
        return []
    layers = (parent.get("annotations") or {}).get("layers") or []
    return layer_source_header_reads(str(plan["model"]), len(layers),
                                     checkpoint_layers_prefix=prefix)


def prepare_reads(args):
    """``(entries, annotations, quanta, indexes)`` for a preparation run."""
    from tools.prepare_extended_joint_quanta import SOURCE_LAYERS_PREFIX, control_paths

    inputs = _load(args.pair_inputs, args.pair_inputs_sha256, "catalog pair")
    plan = _load(inputs["extended_plan"]["path"], inputs["extended_plan"]["sha256"],
                 "extended plan")
    prepared = _load(inputs["extended_prepared"]["path"],
                     inputs["extended_prepared"]["sha256"], "extended preparation")
    parent = _load(args.parent_manifest, args.parent_manifest_sha256,
                   "original parent manifest")
    _load(args.derivation, args.derivation_sha256, "original quantum derivation")
    _load(args.spec, args.spec_sha256, "reviewed Stage B container spec")
    proofs, indexes = _proof_files(args)
    extension_path = Path(args.metadata_root) / "catalog-extension.json"
    extension = None
    files = [args.pair_inputs, args.parent_manifest, args.derivation, args.spec, *proofs]
    if extension_path.exists():
        # A later band set binds the extension the first one wrote.
        extension = {"path": str(extension_path.resolve()),
                     "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest()}
        files.append(extension_path)
    files += sorted(control_paths(inputs, plan, prepared, extension))
    files.append(prepared["production_cache"]["path"])
    entries = preparation_read_entries(
        head=head_phase_entries(parent), files=files,
        ranges=_source_reads(plan, parent, SOURCE_LAYERS_PREFIX))
    layers = len((parent.get("annotations") or {}).get("layers") or [])
    annotations = {"tool": "tools.prepare_extended_joint_quanta",
                   "plan_sha256": inputs["extended_plan"]["sha256"],
                   "prepared_sha256": inputs["extended_prepared"]["sha256"],
                   "parent_manifest_sha256": args.parent_manifest_sha256,
                   "metadata_root": str(Path(args.metadata_root).resolve())}
    return entries, annotations, layers, indexes


def regenerate_reads(args):
    """``(entries, annotations, quanta, indexes)`` for a generator run alone."""
    plan = _load(args.plan, args.plan_sha256, "plan")
    prepared = _load(args.prepared, args.prepared_sha256, "prepared")
    parent = _load(args.parent_manifest, args.parent_manifest_sha256, "parent manifest")
    proofs, indexes = _proof_files(args)
    files = [args.plan, args.prepared, args.parent_manifest, args.derivation, *proofs]
    if args.partition is not None:
        files.append(args.partition)
    if args.catalog_extension is not None:
        files.append(args.catalog_extension)
    if args.executable_readsets:
        files.append(prepared["production_cache"]["path"])
    entries = preparation_read_entries(
        head=head_phase_entries(parent), files=files,
        ranges=_source_reads(plan, parent, args.source_layers_prefix))
    layers = len((parent.get("annotations") or {}).get("layers") or [])
    annotations = {"tool": "tools.regenerate_joint_quanta",
                   "plan_sha256": args.plan_sha256,
                   "prepared_sha256": args.prepared_sha256,
                   "parent_manifest_sha256": args.parent_manifest_sha256,
                   "metadata_root": str(Path(args.metadata_root).resolve())}
    return entries, annotations, layers, indexes


def write_submission(args, entries, annotations, quanta, indexes) -> dict:
    out = Path(args.out).resolve()
    root = Path(args.metadata_root).resolve()
    if out == root or root in out.parents:
        raise ValueError("the submission directory must sit outside the metadata "
                         "root, which the produced-output template owns")
    manifest = preparation_read_manifest(
        entries, produced_by={"tool": "tools/stage_b_preparation_submission.py",
                              "entry_point": args.tool},
        annotations=annotations, mount_prefix=args.mount_prefix)
    template = build_preparation_template(
        metadata_root=root, tier=args.tier,
        payload_max_bytes=preparation_payload_ceiling(quanta, indexes=indexes))
    out.mkdir(parents=True, exist_ok=True)
    manifest_bytes = seal_manifest_bytes(manifest)
    template_bytes = (json.dumps(template, sort_keys=True, indent=1) + "\n").encode()
    manifest_path = out / "read-manifest.json.gz"
    template_path = out / "template.json"
    for path, payload in ((manifest_path, manifest_bytes), (template_path, template_bytes)):
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"{path} already holds other bytes; use a new directory")
        path.write_bytes(payload)
    submission = {
        "schema": SUBMISSION_SCHEMA, "tool": args.tool,
        "metadata_root": str(root),
        "data_manifest": {"path": str(manifest_path),
                          "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "produced_output_template": {
            "path": str(template_path), "template_id": template["template_id"],
            "sha256": hashlib.sha256(template_bytes).hexdigest()},
        "reads": {"entries": manifest["entry_count"], "bytes": manifest["total_bytes"]},
        # --residency stage also gives the action PRISMABUILD_RESIDENCY_MAP,
        # which the produced-output binding needs for its queue root.
        "pbrun_options": ["--data-manifest", str(manifest_path),
                          "--produced-output-template", str(template_path),
                          "--residency", "stage", "--residency-ram", "auto"],
    }
    (out / "submission.json").write_bytes(
        (json.dumps(submission, sort_keys=True, indent=1) + "\n").encode())
    return submission


def _common(parser):
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--adjoint-receipt", type=Path, default=None)
    parser.add_argument("--adjoint-receipt-sha256", default=None)
    parser.add_argument("--adjoint-band", type=Path, action="append", default=[])
    parser.add_argument("--adjoint-band-sha256", action="append", default=[])
    parser.add_argument("--tier", required=True,
                        help="the stage tier the template permits (a fleet fact)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the manifest, template and submission.json")
    parser.add_argument("--mount-prefix", default="/mnt/shared")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tools = parser.add_subparsers(dest="tool", required=True)
    prep = tools.add_parser("prepare")
    for name in ("pair-inputs", "parent-manifest", "derivation", "spec"):
        prep.add_argument("--" + name, type=Path, required=True)
        prep.add_argument("--" + name + "-sha256", required=True)
    _common(prep)
    regen = tools.add_parser("regenerate")
    for name in ("plan", "prepared", "parent-manifest"):
        regen.add_argument("--" + name, type=Path, required=True)
        regen.add_argument("--" + name + "-sha256", required=True)
    regen.add_argument("--derivation", type=Path, required=True)
    regen.add_argument("--partition", type=Path, default=None)
    regen.add_argument("--catalog-extension", type=Path, default=None)
    regen.add_argument("--executable-readsets", action="store_true")
    regen.add_argument("--source-layers-prefix", default=None)
    _common(regen)
    args = parser.parse_args(argv)
    try:
        reads = prepare_reads(args) if args.tool == "prepare" else regenerate_reads(args)
        submission = write_submission(args, *reads)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Stage B preparation submission refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(submission, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
