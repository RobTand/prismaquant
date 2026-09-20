#!/usr/bin/env python3
"""Regenerate the layer-quantum records, slice manifests and adjoint manifest
from sealed inputs with an authoritative output root (PQ #838 defect 4).

Producer D3: binding (or rebinding) records is a NEW identity set, never an
edit. This tool replays the producer call the external binder made -- plan,
prepared, parent manifest, derivation, partition -- with the output root
taken from the plan (never the tool's own directory, the doubling the live
``layer-quanta/layer-quanta`` records carry), and optionally re-seals
against a stage-A receipt. Old records and history stay where they are.
Record files land in the reviewed directory the caller names; slice
manifests land at the producer-named absolute paths the records bind
(verified to resolve after writing). The adjoint manifest is the phase
worker's file and is never written here.

Inputs may be plain JSON or gzip (the parent manifest is a ``.json.gz``):
the digest always covers wire bytes, gunzip output is bounded, and trailing
bytes after a gzip member refuse.

Three gates, all fail closed with exit 3:

* Gate 1a (``--expect-existing`` + ``--original-root``): regenerate
  receipt-less at the original root and require every on-disk record to
  match canonical JSON -- input fidelity, proved before anything moves.
* Gate 1b: regenerate at the new root and require only the authorized path
  fields (plus the receipt seal when binding) to move; everything else is
  strictly equal and the new identity must recompute. A regeneration
  against known-bad originals can never "reproduce" by definition, so the
  old root is explicit rather than inferred.
* Gate 2: ``--adjoint-receipt`` must exist and load; the bound set is
  regenerated with the receipt mapping (every ``adjoint.receipt_sha256``
  binds, every ``identity_sha256`` moves). ``--check-only`` runs the gates
  and writes nothing, mirroring the external binder's dry run.

This is a producer, never a scheduler: it publishes no PB rows, claims
nothing, and holds no state. Run it on a checkout inside the PB code
closure (unlike the external binder script, which lives beside the data
and cannot be captured), then submit from its outputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

if __package__:
    from prismaquant.joint_layer_quanta import layer_quanta, seal_manifest_bytes
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prismaquant.joint_layer_quanta import layer_quanta, seal_manifest_bytes

EXIT_REFUSED = 3

#: Bound the gunzip the manifest read may materialize: the fleet's own
#: manifest ingestion bounds stored and decoded bytes independently
#: (``prismabuild.core`` ``DATA_MANIFEST_MAX_*``); this reader caps the
#: decoded document the same way rather than trusting the member size.
MAX_DECODED_MANIFEST_BYTES = 512 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"


def _fail(message: str) -> int:
    print(f"regenerate_joint_quanta: refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _load_json(path: Path, *, digest: str | None, where: str):
    """Read, digest-verify, and parse a JSON document, gzip-transparent.

    The digest always covers the wire bytes. Detection is by gzip magic,
    never by suffix; the gunzip output is bounded and trailing garbage
    after a gzip member refuses rather than being silently ignored."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{where} unreadable at {path}: {exc}") from exc
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{where} digest mismatch at {path}")
    if raw[:2] == _GZIP_MAGIC:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as member:
                raw = member.read(MAX_DECODED_MANIFEST_BYTES + 1)
                trailing = member.unused_data
        except (OSError, EOFError) as exc:
            raise ValueError(f"{where} is not valid gzip at {path}: "
                             f"{exc}") from exc
        if trailing:
            raise ValueError(f"{where} has trailing bytes after its gzip "
                             f"member at {path}: refusing")
        if len(raw) > MAX_DECODED_MANIFEST_BYTES:
            raise ValueError(f"{where} decoded document exceeds "
                             f"{MAX_DECODED_MANIFEST_BYTES} bytes at {path}")
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


#: Record fields allowed to move between the original root and the new
#: root, with the old prefix swapped for the new one exactly. Everything
#: else must be byte-stable canonical JSON; ``identity_sha256`` and the
#: slice digest are consequences (recomputed and re-verified, never
#: compared across), and ``adjoint.receipt_sha256`` may move only when a
#: receipt is bound in this run.
_MOVED_PATH_FIELDS = (
    ("output_space", "root"),
    ("output_space", "cost_payload"),
    ("output_space", "results"),
    ("output_space", "counters"),
    ("output_space", "checkpoint_dir"),
    ("read_set", "manifest_path"),
    ("adjoint", "boundary_artifacts"),
)


def _moved(record: dict, field: tuple[str, str]) -> str:
    value = record.get(field[0], {})
    path = value.get(field[1]) if isinstance(value, dict) else None
    if not isinstance(path, str) or not path:
        raise ValueError(f"record {record.get('quantum_id')!r} has no "
                         f"{field[0]}.{field[1]}")
    return path


def _check_authorized_diff(old: dict, new: dict, *, old_root: str, bound: bool,
                           where: str) -> None:
    """The new record may differ from the old one only where the root move
    (and, when binding, the receipt seal) requires. Anything else refuses.

    Each moved path must be the old path with exactly the old-root prefix
    swapped for the new one. ``identity_sha256`` is not compared across --
    it is recomputed over the new record and required to match, which
    proves the seal is well-formed rather than blind. The slice digest is
    re-verified against the written slice file by the caller, for the same
    reason. ``adjoint.receipt_sha256`` may differ only when this run binds
    a receipt; otherwise it must be equal.
    """
    from prismaquant.joint_layer_quanta import canonical_sha256
    qid = new.get("quantum_id", "?")
    old_root = old_root.rstrip("/")
    # The new tree root is whatever the first moved path swapped to; every
    # other moved path must swap to that same root, so a record cannot half
    # move between trees.
    new_prefix: str | None = None
    for field in _MOVED_PATH_FIELDS:
        old_path = _moved(old, field)
        new_path = _moved(new, field)
        if not old_path.startswith(old_root + "/"):
            raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                             f"{old_path!r} is not under the original root "
                             f"{old_root!r}")
        suffix = old_path[len(old_root):]
        if new_prefix is None:
            if not suffix or not new_path.endswith(suffix):
                raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                                 f"{new_path!r} is not the moved "
                                 f"{old_path!r}")
            new_prefix = new_path[:len(new_path) - len(suffix)]
        elif new_path != new_prefix + suffix:
            raise ValueError(f"Gate 1 {where}: {qid} {field[0]}.{field[1]} "
                             f"{new_path!r} leaves the moved tree")
    old_body = {k: v for k, v in old.items() if k != "identity_sha256"}
    new_body = {k: v for k, v in new.items() if k != "identity_sha256"}
    old_body.pop("read_set", None)
    read_set = dict(new_body.pop("read_set", {}))
    read_set.pop("manifest_sha256", None)
    old_read = dict(old.get("read_set", {}))
    old_read.pop("manifest_sha256", None)
    if old_read != read_set:
        raise ValueError(f"Gate 1 {where}: {qid} read_set differs beyond "
                         f"the moved manifest path")
    old_adjoint = dict(old_body.pop("adjoint", {}))
    new_adjoint = dict(new_body.pop("adjoint", {}))
    old_receipt = old_adjoint.pop("receipt_sha256", None)
    new_receipt = new_adjoint.pop("receipt_sha256", None)
    if old_adjoint != new_adjoint:
        raise ValueError(f"Gate 1 {where}: {qid} adjoint block differs "
                         f"beyond the receipt seal")
    if not bound and old_receipt != new_receipt:
        raise ValueError(f"Gate 1 {where}: {qid} receipt seal moved with "
                         f"no receipt bound")
    if json.dumps(old_body, sort_keys=True) != json.dumps(new_body, sort_keys=True):
        raise ValueError(f"Gate 1 {where}: {qid} differs outside the moved "
                         f"paths")
    if canonical_sha256(new, where=f"Gate 1 {where} {qid}") != new.get(
            "identity_sha256"):
        raise ValueError(f"Gate 1 {where}: {qid} identity does not recompute")


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
                    help="reviewed directory receiving the record files "
                         "(slice manifests land at the producer-named "
                         "absolute paths the records bind, verified after "
                         "writing)")
    ap.add_argument("--expect-existing", type=Path, default=None,
                    help="Gate 1: directory of on-disk records from the "
                         "original root; requires --original-root")
    ap.add_argument("--original-root", type=Path, default=None,
                    help="the root the --expect-existing records were "
                         "produced with: Gate 1 first reproduces them "
                         "exactly (input fidelity), then validates that the "
                         "new root moves only the authorized path fields")
    ap.add_argument("--adjoint-receipt", type=Path, default=None,
                    help="Gate 2: stage-A adjoint-capture.json to bind")
    ap.add_argument("--check-only", action="store_true",
                    help="Gate 1 alone; write nothing")
    args = ap.parse_args(argv)
    if args.expect_existing is not None and args.original_root is None:
        return _fail("Gate 1 needs --original-root beside --expect-existing")
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

        def _produce(root: str, receipt=None):
            return layer_quanta(
                plan, prepared, parent,
                chunk_target_bytes=derivation.get("chunk_target_bytes"),
                stride=derivation.get("stride"),
                output_root=root,
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
    receipt = None
    if args.adjoint_receipt is not None:
        try:
            receipt = _load_json(args.adjoint_receipt, digest=None,
                                 where="adjoint receipt")
        except ValueError as exc:
            return _fail(str(exc))
    if args.expect_existing is not None:
        # Step A (input fidelity): the original root must reproduce the
        # on-disk records exactly -- canonical JSON, not raw bytes, so a
        # re-serialization the producer never promised does not fail it.
        try:
            original = _produce(str(args.original_root))
        except (ValueError, OSError) as exc:
            return _fail(f"Gate 1 reproduction: {exc}")
        for record in original["records"]:
            on_disk = args.expect_existing / f"{record['quantum_id']}.json"
            try:
                stored = json.loads(on_disk.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"Gate 1 cannot read {on_disk}: {exc}")
            if json.dumps(stored, sort_keys=True) != json.dumps(
                    record, sort_keys=True):
                return _fail(f"Gate 1: {record['quantum_id']} differs from "
                              f"the sealed inputs under receipt-less "
                              f"regeneration at the original root; "
                              f"re-derive by review")
        print(f"Gate 1a: {len(original['records'])}/{len(original['records'])} "
              f"records reproduce from the sealed inputs at the original root")
        # Step B (authorized move): the new root may move only the path
        # fields (and the receipt seal when binding); everything else is
        # strictly equal, and the new identity must recompute.
        try:
            moved = _produce(output_root, receipt=receipt)
        except (ValueError, OSError) as exc:
            return _fail(f"Gate 1 move: {exc}")
        old_by_id = {}
        for record in original["records"]:
            on_disk = args.expect_existing / f"{record['quantum_id']}.json"
            try:
                old_by_id[record["quantum_id"]] = json.loads(
                    on_disk.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"Gate 1 cannot read {on_disk}: {exc}")
        try:
            for record in moved["records"]:
                _check_authorized_diff(
                    old_by_id[record["quantum_id"]], record,
                    old_root=str(args.original_root),
                    bound=receipt is not None,
                    where=f"move to {output_root}")
        except (ValueError, KeyError) as exc:
            return _fail(str(exc))
        print(f"Gate 1b: {len(moved['records'])}/{len(moved['records'])} "
              f"records move only the authorized fields to the new root")
        produced = moved
    else:
        try:
            produced = _produce(output_root, receipt=receipt)
        except (ValueError, OSError) as exc:
            return _fail(str(exc))
    if args.check_only:
        return 0
    out = args.records_out
    try:
        for record in produced["records"]:
            _atomic_write(out / f"{record['quantum_id']}.json",
                          _pretty(record))
        _atomic_write(out / "records.json", _pretty(produced["records"]))
        _atomic_write(out / "derivation.json", _pretty(produced["derivation"]))
        # Slice manifests land at the producer-named absolute paths the
        # records bind -- never beside the record files -- so the bound
        # paths resolve to the exact bytes sealed. The adjoint manifest is
        # the phase worker's file and is never written here.
        for record in produced["records"]:
            qid = record["quantum_id"]
            _atomic_write(Path(record["read_set"]["manifest_path"]),
                          seal_manifest_bytes(produced["slice_manifests"][qid]))
        # Resolution verification: every bound path must name the exact
        # file just written, whose bytes hash to the sealed digest.
        for record in produced["records"]:
            manifest_path = Path(record["read_set"]["manifest_path"])
            try:
                sealed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            except OSError as exc:
                return _fail(f"slice manifest unreadable at {manifest_path}: "
                              f"{exc}")
            if sealed != record["read_set"]["manifest_sha256"]:
                return _fail(f"slice manifest at {manifest_path} does not "
                              f"hash to the sealed digest")
    except (ValueError, OSError) as exc:
        return _fail(str(exc))
    bound = ("unbound (pre-stage-A)" if receipt is None else
             f"bound to receipt {produced['records'][0]['adjoint']['receipt_sha256'][:16]}…")
    print(f"regenerate_joint_quanta: wrote {len(produced['records'])} records "
          f"{bound} under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
