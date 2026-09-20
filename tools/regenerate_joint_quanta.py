#!/usr/bin/env python3
"""Regenerate the layer-quantum records, slice manifests and adjoint manifest
from sealed inputs with an authoritative output root (PQ #838 defect 4).

Producer D3: binding (or rebinding) records is a NEW identity set, never an
edit. This tool replays the producer call the external binder made -- plan,
prepared, parent manifest, derivation, partition -- with the output root
taken from the plan (never the tool's own directory, the doubling the live
``layer-quanta/layer-quanta`` records carry), and optionally re-seals
against a stage-A receipt. Old records and history stay where they are;
outputs land in the reviewed directory the caller names, written if absent
or byte-identical, refused if different.

Two gates, both fail closed with exit 3:

* Gate 1 (``--expect-existing``): regenerate receipt-less and require every
  on-disk record to match canonical-JSON; drifted inputs refuse before
  anything is written. ``--check-only`` runs Gate 1 alone and writes
  nothing, mirroring the external binder's dry run.
* Gate 2: ``--adjoint-receipt`` must exist and load; the bound set is
  regenerated with the receipt mapping (every ``adjoint.receipt_sha256``
  binds, every ``identity_sha256`` moves).

This is a producer, never a scheduler: it publishes no PB rows, claims
nothing, and holds no state. Run it on a checkout inside the PB code
closure (unlike the external binder script, which lives beside the data
and cannot be captured), then submit from its outputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

if __package__:
    from prismaquant.joint_layer_quanta import layer_quanta, seal_manifest_bytes
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prismaquant.joint_layer_quanta import layer_quanta, seal_manifest_bytes

EXIT_REFUSED = 3


def _fail(message: str) -> int:
    print(f"regenerate_joint_quanta: refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _load_json(path: Path, *, digest: str | None, where: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{where} unreadable at {path}: {exc}") from exc
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{where} digest mismatch at {path}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{where} is not JSON at {path}: {exc}") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(
            f"refusing to overwrite differing bytes at {path}: a re-seal "
            f"is a new reviewed directory, never an edit")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-regen")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _pretty(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--prepared-sha256", required=True)
    ap.add_argument("--parent-manifest", type=Path, required=True)
    ap.add_argument("--parent-manifest-sha256", required=True)
    ap.add_argument("--derivation", type=Path, required=True,
                    help="derivation.json carrying chunk_target_bytes, "
                         "stride, ram_window_gib, max_resident_consumers")
    ap.add_argument("--partition", type=Path, default=None,
                    help="optional window-partition.json")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="authoritative run root sealed into output_space "
                         "and manifest paths (default: the plan's "
                         "output_root)")
    ap.add_argument("--records-out", type=Path, required=True,
                    help="reviewed directory receiving the new identity set")
    ap.add_argument("--expect-existing", type=Path, default=None,
                    help="Gate 1: directory of on-disk records the "
                         "receipt-less regeneration must match")
    ap.add_argument("--adjoint-receipt", type=Path, default=None,
                    help="Gate 2: stage-A adjoint-capture.json to bind")
    ap.add_argument("--check-only", action="store_true",
                    help="Gate 1 alone; write nothing")
    args = ap.parse_args(argv)
    try:
        plan = _load_json(args.plan, digest=args.plan_sha256, where="plan")
        prepared = _load_json(args.prepared, digest=args.prepared_sha256,
                              where="prepared")
        parent = _load_json(args.parent_manifest, digest=args.parent_manifest_sha256,
                            where="parent manifest")
        derivation = _load_json(args.derivation, digest=None, where="derivation")
        partition = (_load_json(args.partition, digest=None, where="partition")
                     if args.partition is not None else None)
        output_root = (str(args.output_root) if args.output_root is not None
                       else plan.get("output_root"))
        if not isinstance(output_root, str) or not output_root.startswith("/"):
            raise ValueError("no authoritative absolute output_root: pass "
                             "--output-root or seal plan.output_root")
        produced = layer_quanta(
            plan, prepared, parent,
            chunk_target_bytes=derivation.get("chunk_target_bytes"),
            stride=derivation.get("stride"),
            output_root=output_root,
            plan_path=str(args.plan), plan_sha256=args.plan_sha256,
            prepared_path=str(args.prepared), prepared_sha256=args.prepared_sha256,
            parent_manifest_sha256=args.parent_manifest_sha256,
            ram_window_gib=derivation.get("ram_window_gib"),
            max_resident_consumers=derivation.get("max_resident_consumers"),
            window_partition=partition,
            adjoint_receipt=None)
    except (ValueError, OSError) as exc:
        return _fail(str(exc))
    if args.expect_existing is not None:
        for record in produced["records"]:
            on_disk = args.expect_existing / f"{record['quantum_id']}.json"
            try:
                stored = json.loads(on_disk.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"Gate 1 cannot read {on_disk}: {exc}")
            if json.dumps(stored, sort_keys=True) != json.dumps(
                    record, sort_keys=True):
                return _fail(f"Gate 1: {record['quantum_id']} differs from "
                              f"the sealed inputs under receipt-less "
                              f"regeneration; re-derive by review")
        print(f"Gate 1: {len(produced['records'])}/{len(produced['records'])} "
              f"records reproduce from the sealed inputs")
    if args.check_only:
        return 0
    receipt = None
    if args.adjoint_receipt is not None:
        try:
            receipt = _load_json(args.adjoint_receipt, digest=None,
                                 where="adjoint receipt")
        except ValueError as exc:
            return _fail(str(exc))
        try:
            produced = layer_quanta(
                plan, prepared, parent,
                chunk_target_bytes=derivation.get("chunk_target_bytes"),
                stride=derivation.get("stride"),
                output_root=output_root,
                plan_path=str(args.plan), plan_sha256=args.plan_sha256,
                prepared_path=str(args.prepared),
                prepared_sha256=args.prepared_sha256,
                parent_manifest_sha256=args.parent_manifest_sha256,
                ram_window_gib=derivation.get("ram_window_gib"),
                max_resident_consumers=derivation.get("max_resident_consumers"),
                window_partition=partition,
                adjoint_receipt=receipt)
        except (ValueError, OSError) as exc:
            return _fail(str(exc))
    out = args.records_out
    try:
        for record in produced["records"]:
            _atomic_write(out / f"{record['quantum_id']}.json",
                          _pretty(record))
        _atomic_write(out / "records.json", _pretty(produced["records"]))
        _atomic_write(out / "derivation.json", _pretty(produced["derivation"]))
        for qid, slice_manifest in produced["slice_manifests"].items():
            _atomic_write(out / "manifests" / f"{qid}.data-manifest.json.gz",
                          seal_manifest_bytes(slice_manifest))
        _atomic_write(out / "adjoint-manifest.json",
                      _pretty(produced["adjoint_manifest"]))
    except (ValueError, OSError) as exc:
        return _fail(str(exc))
    bound = ("unbound (pre-stage-A)" if receipt is None else
             f"bound to receipt {produced['records'][0]['adjoint']['receipt_sha256'][:16]}…")
    print(f"regenerate_joint_quanta: wrote {len(produced['records'])} records "
          f"{bound} under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
