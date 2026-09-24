"""Stage A chain split: one run's reverse chain as sample-range quanta (PQ #738).

A Stage A run seals a checkpoint every ``stride`` layers on its reverse
chain. Below a sealed checkpoint the chain carries no state between
calibration samples: every ``(probe, batch)`` cotangent and its shared-state
owner roll through a layer on their own. So the chain between two stride
checkpoints splits by sample, and each part is a PrismaBuild row that any
free GPU can run:

* **A prep row, run once per round.** It is the relaunch's own resume
  (``stage_a_chain_resume.plan_chain_resume`` and ``apply_chain_resume``):
  it seals the round's resume record, which stamps the running
  implementation and the split, and it removes the interrupted attempt's
  rolling entries. It rolls nothing. A retry prep names only the ranges it
  relaunches: it removes only their rolling entries and sets aside only
  their partial checkpoints and owner status files, so the entries of every
  range that finished stay.
* **Quanta.** Each rebinds the run's own session and rolls one contiguous
  range of global sample indices, all probes, from the lowest sealed
  checkpoint down to ``through``. At every stride checkpoint on the way it
  seals a **partial** checkpoint of its range (``split/boundary-NNN/
  samples-SSSSSS-EEEEEE``), in the layout of a whole one. A range is whole
  read windows (``prefetch_batches``), so a quantum groups batches exactly
  as the single owner does, and its bytes are the single owner's.
* **A join per boundary.** It verifies every partial's bytes, requires the
  partials to tile the whole probe x batch plane, and publishes the one
  checkpoint a single owner would have sealed: the union of the cotangent
  rows, one shared-state pack with every member in name order, and the
  manifest over both. The band tool reads it unchanged.

Nothing here compares a run's source, implementation, plan or session with
a recorded one. What it checks is that the bytes it reads are the bytes that
were written, and that the plane it publishes is whole.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from .cost_stage_checkpoint import atomic_write_bytes

SPLIT_SCHEMA = "prismaquant.stage_a.chain_split.v1"
PREP_RECEIPT_SCHEMA = "prismaquant.stage_a.chain_split_prep.v1"
QUANTUM_RECEIPT_SCHEMA = "prismaquant.stage_a.chain_split_quantum.v1"
JOIN_RECEIPT_SCHEMA = "prismaquant.stage_a.chain_split_join.v1"
SPLIT_DIGESTS_SCHEMA = "prismaquant.stage_a.chain_split_digests.v1"
PREP, QUANTUM = "prep", "quantum"
_RANGE = re.compile(r"samples-(\d{6,})-(\d{6,})")
_ROLLING = re.compile(r"cotangent-(\d+)-(\d+)-at-(\d+)\.pt(\.tmp)?")
_OWNER = re.compile(r"from-(\d{3,})-through-(\d{3,})-samples-(\d{6,})-(\d{6,})\.json")


class ChainSplitRefused(RuntimeError):
    """The split cannot run or join without losing or corrupting bytes."""


# -- layout --------------------------------------------------------------------

def split_root(space) -> Path:
    return Path(space) / "split"


def range_name(start: int, stop: int) -> str:
    return f"samples-{int(start):06d}-{int(stop):06d}"


def partial_directory(space, boundary: int, start: int, stop: int) -> Path:
    """Where one range's partial checkpoint at ``boundary`` is sealed."""
    return split_root(space) / f"boundary-{int(boundary):03d}" / range_name(start, stop)


def quantum_label(from_boundary: int, through: int, start: int, stop: int) -> str:
    """One quantum's name: its owner status file and its receipt files."""
    return (f"from-{int(from_boundary):03d}-through-{int(through):03d}-"
            f"{range_name(start, stop)}")


def quantum_directory(space) -> Path:
    """Each quantum's receipt, results and counters (never the run's files)."""
    return split_root(space) / "quanta"


def owner_directory(generation_directory) -> Path:
    return Path(generation_directory) / "owners"


# -- the split spec ------------------------------------------------------------

def _range(value, where) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ChainSplitRefused(f"{where} is a [start, stop) pair of sample indices")
    return int(value[0]), int(value[1])


def normalize_chain_split(spec) -> dict:
    """``{role: prep, through, ranges}`` or ``{role: quantum, through, samples,
    digest_layer}``, validated in shape; the run's own numbers are checked by
    :func:`check_ranges` once the run is known."""
    if not isinstance(spec, dict) or spec.get("role") not in (PREP, QUANTUM):
        raise ChainSplitRefused("a chain split is a prep or a quantum")
    through = spec.get("through")
    if type(through) is not int or through < 1:
        raise ChainSplitRefused("a chain split's through boundary is a positive integer")
    if spec["role"] == PREP:
        if set(spec) != {"role", "through", "ranges"}:
            raise ChainSplitRefused("a split prep names its through boundary and ranges")
        ranges = [_range(value, "a prep range") for value in spec["ranges"] or ()]
        if not ranges:
            raise ChainSplitRefused("a split prep names at least one range")
        return {"role": PREP, "through": through,
                "ranges": [list(pair) for pair in sorted(ranges)]}
    if not set(spec) <= {"role", "through", "samples", "digest_layer"}:
        raise ChainSplitRefused("a split quantum names its through boundary and samples")
    digest_layer = spec.get("digest_layer")
    if digest_layer is not None and (type(digest_layer) is not int or digest_layer < 0):
        raise ChainSplitRefused("a quantum's digest layer is a nonnegative integer")
    return {"role": QUANTUM, "through": through,
            "samples": list(_range(spec.get("samples"), "a quantum's samples")),
            "digest_layer": digest_layer}


def parse_ranges(text) -> list[list[int]]:
    """``S:E,S:E,...`` from the command line."""
    ranges = []
    for part in str(text).split(","):
        pieces = part.split(":")
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            raise ChainSplitRefused(f"a sample range is START:STOP, not {part!r}")
        ranges.append([int(pieces[0]), int(pieces[1])])
    return ranges


def even_ranges(n_batches: int, group_size: int, quanta: int) -> list[list[int]]:
    """``quanta`` contiguous ranges of whole read windows, as even as they go."""
    groups = -(-int(n_batches) // int(group_size))
    if not 1 <= int(quanta) <= groups:
        raise ChainSplitRefused(
            f"{quanta} quanta cannot each hold whole windows of {group_size} among "
            f"{n_batches} samples ({groups} windows)")
    base, extra = divmod(groups, int(quanta))
    ranges, group = [], 0
    for index in range(int(quanta)):
        count = base + (1 if index < extra else 0)
        start = group * int(group_size)
        group += count
        ranges.append([start, min(group * int(group_size), int(n_batches))])
    return ranges


def check_ranges(ranges, *, n_batches: int, group_size: int, where: str) -> None:
    """Ranges are disjoint and whole read windows of the run.

    A range starts on a window boundary and ends on one, or at the last
    sample: a quantum then groups batches (``chain_batch_size``) and fused
    windows exactly as the single owner does, and every produced-output
    group it writes is its own.
    """
    previous = 0
    for start, stop in sorted(tuple(pair) for pair in ranges):
        if not 0 <= start < stop <= n_batches:
            raise ChainSplitRefused(
                f"{where} range {start}:{stop} is not inside the run's {n_batches} samples")
        if start % group_size or (stop % group_size and stop != n_batches):
            raise ChainSplitRefused(
                f"{where} range {start}:{stop} is not whole read windows of {group_size} "
                "samples")
        if start < previous:
            raise ChainSplitRefused(f"{where} ranges overlap at {start}")
        previous = stop


def split_boundaries(stride_boundaries, from_boundary: int, through: int) -> list[int]:
    """The stride checkpoints a round seals, highest first: ``through <= b < from``."""
    marks = sorted({int(mark) for mark in stride_boundaries}, reverse=True)
    if int(through) not in marks:
        raise ChainSplitRefused(
            f"the split stops at {through}, which is not a stride checkpoint {marks}")
    if not int(through) < int(from_boundary):
        raise ChainSplitRefused(
            f"the split rolls from checkpoint {from_boundary}; it cannot stop at {through}")
    return [mark for mark in marks if int(through) <= mark < int(from_boundary)]


def split_stamp(split: dict, *, from_boundary: int, boundaries) -> dict:
    """What a prep's resume record stamps about the round (a record, not a gate)."""
    return {"schema": SPLIT_SCHEMA, "from": int(from_boundary),
            "through": int(split["through"]), "boundaries": list(boundaries),
            "ranges": [list(pair) for pair in split["ranges"]]}


# -- the prep ------------------------------------------------------------------

def _partial_ranges(space, boundary):
    """``[(start, stop, directory)]`` of every partial at ``boundary``."""
    root = split_root(space) / f"boundary-{int(boundary):03d}"
    if not root.is_dir():
        return []
    found = []
    for directory in sorted(root.iterdir()):
        match = _RANGE.fullmatch(directory.name)
        if match is not None and directory.is_dir():
            found.append((int(match[1]), int(match[2]), directory))
    return found


def prepare_split_prep(space, plan, split, *, boundaries, generation_directory,
                       n_batches: int, group_size: int) -> dict:
    """Scope a round's resume to the ranges this prep launches. Writes nothing.

    Returns what :func:`apply_split_prep` does after every check: the
    rolling entries to remove (the plan's leftovers whose sample is in a
    named range and that no sealed partial of another range names), the
    partial directories and owner status files of the named ranges to set
    aside, and the stamp for the resume record. The PrismaBuild owner of
    every range this prep relaunches must be contained first: its rolling
    entries are about to be removed.
    """
    from .stage_a_chain_resume import require_producer_contained

    check_ranges(split["ranges"], n_batches=n_batches, group_size=group_size,
                 where="the prep's")
    marks = split_boundaries(boundaries, plan.boundary, split["through"])
    named = {tuple(pair) for pair in split["ranges"]}

    def launched(batch):
        return any(start <= batch < stop for start, stop in named)

    kept, set_aside = set(), []
    for mark in marks:
        for start, stop, directory in _partial_ranges(space, mark):
            if (start, stop) in named:
                set_aside.append(directory)
                continue
            if any(start < b_stop and b_start < stop for b_start, b_stop in named):
                raise ChainSplitRefused(
                    f"partial {directory.name} at boundary {mark} overlaps a range this "
                    "prep relaunches; a retry names the failed ranges exactly")
            manifest = directory / "checkpoint.json"
            if manifest.is_file():
                record = read_partial(directory, boundary=mark)
                kept.update(Path(row["path"]) for row in
                            _partial_plane(record, space=space).values())
    owners = []
    owner_root = owner_directory(generation_directory)
    if owner_root.is_dir():
        for path in sorted(owner_root.iterdir()):
            match = _OWNER.fullmatch(path.name)
            if match is None:
                continue
            start, stop = int(match[3]), int(match[4])
            if not any(start < b_stop and b_start < stop for b_start, b_stop in named):
                continue
            status = json.loads(path.read_bytes())
            producer = (status.get("owner") or {}).get("producer")
            if status.get("status") == "running" and producer is not None:
                require_producer_contained(producer)
            if (int(match[1]), int(match[2])) == (plan.boundary, split["through"]):
                owners.append(path)
    removed = []
    for path in plan.leftovers:
        match = _ROLLING.fullmatch(path.name)
        if match is not None and launched(int(match[2])) and path not in kept:
            removed.append(path)
    return {"leftovers": removed, "partials": set_aside, "owners": owners,
            "boundaries": marks,
            "stamp": split_stamp(split, from_boundary=plan.boundary, boundaries=marks)}


def apply_split_prep(prep: dict, *, index: int) -> dict:
    """Set the relaunched ranges' partials and owner files aside, never delete them."""
    moved = []
    for path in [*prep["partials"], *prep["owners"]]:
        suffix = f".resume-{int(index):03d}"
        target = path.with_name(
            path.name.removesuffix(".json") + suffix + (".json" if path.suffix == ".json"
                                                      else ""))
        os.rename(path, target)
        moved.append(str(target))
    return {"set_aside": moved}


# -- partial checkpoints --------------------------------------------------------

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_partial(directory, *, boundary: int) -> dict:
    """One range's sealed partial checkpoint, its bytes checked.

    The manifest must be the writer's own serialization of a record that
    seals its own fields; the pack must have the size and digest its row
    names, and every member the digest the pack's index names; every
    cotangent row's file must have its recorded size. The cotangent bytes
    stay the exact reader's to verify at read, as for any checkpoint.
    """
    from .joint_adjoint_checkpoints import (
        ADJOINT_CHECKPOINT_PACKED_SCHEMA, SHARED_STATE_PACK_FILENAME,
        SHARED_STATE_PACK_NAME)
    from .joint_adjoint_slices import checkpoint_manifest_bytes, checkpoint_seal_sha256

    directory = Path(directory)
    path = directory / "checkpoint.json"
    raw = path.read_bytes()
    record = json.loads(raw)
    try:
        serialized = checkpoint_manifest_bytes(record)
    except ValueError as exc:
        raise ChainSplitRefused(f"{path} is not a checkpoint record: {exc}") from exc
    if serialized != raw or record["cotangent_sha256"] != checkpoint_seal_sha256(record):
        raise ChainSplitRefused(f"{path} does not seal its own bytes")
    if record["schema"] != ADJOINT_CHECKPOINT_PACKED_SCHEMA or record["boundary"] != int(
            boundary):
        raise ChainSplitRefused(f"{path} is not a packed partial at boundary {boundary}")
    rows = record["shared_state_entries"]
    if (len(rows) != 1 or rows[0].get("name") != SHARED_STATE_PACK_NAME
            or Path(rows[0]["path"]) != directory / "entries" / SHARED_STATE_PACK_FILENAME):
        raise ChainSplitRefused(f"{path} names no shared-state pack of its own")
    pack = Path(rows[0]["path"])
    if pack.stat().st_size != rows[0]["file_bytes"] or _file_sha256(pack) != rows[0][
            "sha256"]:
        raise ChainSplitRefused(f"{pack} is not the pack its manifest names")
    for row in record["activation_entries"]:
        if Path(row["path"]).stat().st_size != row["file_bytes"]:
            raise ChainSplitRefused(f"{row['path']} is not the size its partial names")
    return record


def _partial_plane(record, *, space) -> dict:
    """A partial's cotangent rows by ``(probe, batch)``, held to the whole
    checkpoint's row rules: a partial's rows are what the join publishes."""
    from .joint_adjoint_checkpoints import checkpoint_directory
    from .joint_adjoint_slices import checkpoint_cotangent_plane

    placed = dict(record)
    placed["shared_state_entries"] = [{
        **record["shared_state_entries"][0],
        "path": str(checkpoint_directory(space, record["boundary"]) / "entries"
                    / Path(record["shared_state_entries"][0]["path"]).name)}]
    try:
        return checkpoint_cotangent_plane(placed)
    except ValueError as exc:
        raise ChainSplitRefused(f"partial at {record['boundary']}: {exc}") from exc


# -- the join ------------------------------------------------------------------

def join_split_checkpoint(space, boundary: int, *, n_probes: int, n_batches: int) -> dict:
    """Publish checkpoint ``boundary`` from every range's partial; returns its record.

    The partials must tile ``0 .. n_batches`` with no gap and no overlap,
    carry one session, and together hold every ``(probe, batch)`` of the
    ``n_probes x n_batches`` plane. The record is the one a single owner
    seals: the rows of every partial in name order, one pack whose members
    are every partial's members in name order, byte for byte, and the
    manifest serialization the writer uses.
    """
    from .joint_adjoint_checkpoints import (
        SHARED_STATE_PACK_FILENAME, SHARED_STATE_PACK_NAME, _publish_streamed_file,
        _shared_state_pack_index, _shared_state_pack_trailer, checkpoint_directory,
        unpack_shared_states)
    from .joint_adjoint_slices import checkpoint_manifest_bytes, checkpoint_seal_sha256

    space = Path(space)
    boundary = int(boundary)
    partials = _partial_ranges(space, boundary)
    expected = 0
    records = []
    for start, stop, directory in partials:
        if start != expected:
            raise ChainSplitRefused(
                f"the partials at boundary {boundary} do not tile the samples: "
                f"{'gap' if start > expected else 'overlap'} at {min(start, expected)}")
        if not (directory / "checkpoint.json").is_file():
            raise ChainSplitRefused(f"partial {directory} is not sealed")
        records.append(read_partial(directory, boundary=boundary))
        expected = stop
    if expected != int(n_batches):
        raise ChainSplitRefused(
            f"the partials at boundary {boundary} cover samples 0:{expected}, "
            f"not 0:{n_batches}")
    sessions = {json.dumps(record["session"], sort_keys=True) for record in records}
    if len(sessions) != 1:
        # Not an identity gate: a checkpoint record carries one session and
        # every reader holds each row to it (checkpoint_cotangent_plane), so
        # a mixed join would publish a checkpoint no reader accepts.
        raise ChainSplitRefused(
            f"the partials at boundary {boundary} name {len(sessions)} sessions; "
            "a checkpoint record holds one and every reader holds its rows to it")
    target = checkpoint_directory(space, boundary)
    pack_path = target / "entries" / SHARED_STATE_PACK_FILENAME
    rows = sorted((row for record in records for row in record["activation_entries"]),
                  key=lambda row: row["name"])
    joined = {
        "schema": records[0]["schema"],
        "boundary": boundary,
        "session": records[0]["session"],
        "activation_entries": rows,
        "shared_state_entries": [{"name": SHARED_STATE_PACK_NAME, "path": str(pack_path),
                                  "sha256": "0" * 64, "file_bytes": 0}],
    }
    plane = _partial_plane({**joined, "cotangent_sha256": None}, space=space)
    whole = {(probe, batch) for probe in range(int(n_probes))
             for batch in range(int(n_batches))}
    if set(plane) != whole:
        missing = sorted(whole - set(plane))
        raise ChainSplitRefused(
            f"the partials at boundary {boundary} miss {len(missing)} of the "
            f"{n_probes} x {n_batches} cotangents, first {missing[:4]}")
    payloads = [Path(record["shared_state_entries"][0]["path"]).read_bytes()
                for record in records]
    members = sorted((member for payload in payloads
                      for member in unpack_shared_states(payload)),
                     key=lambda member: member[0])
    names = [name for name, _ in members]
    if len(set(names)) != len(names):
        raise ChainSplitRefused(
            f"the partials at boundary {boundary} repeat a shared state")

    index, offset = [], 0
    for name, member in members:
        index.append({"name": name, "offset": offset, "bytes": member.nbytes,
                      "sha256": hashlib.sha256(member).hexdigest()})
        offset += member.nbytes
    encoded = _shared_state_pack_index(index)
    trailer = _shared_state_pack_trailer(len(encoded))

    def body(sink):
        for _name, member in members:
            sink.write(member)
        sink.write(encoded)
        sink.write(trailer)

    if (target / "checkpoint.json").exists():
        raise ChainSplitRefused(f"{target} is already sealed: a checkpoint is joined once")
    if target.exists():
        # A join that died before its manifest published nothing: set its
        # directory aside, never delete it, and join again.
        attempt = 1
        while target.with_name(f"{target.name}.partial-join-{attempt:03d}").exists():
            attempt += 1
        os.rename(target, target.with_name(f"{target.name}.partial-join-{attempt:03d}"))
    target.mkdir(parents=True, exist_ok=False)
    joined["shared_state_entries"] = [_publish_streamed_file(
        pack_path, SHARED_STATE_PACK_NAME, body,
        max_file_bytes=offset + len(encoded) + len(trailer),
        label="joined shared-state pack")]
    joined["cotangent_sha256"] = checkpoint_seal_sha256(joined)
    atomic_write_bytes(target / "checkpoint.json", checkpoint_manifest_bytes(joined))
    return joined


def join_receipt(space, record: dict) -> dict:
    """The join's own record: what it published, for the log and the report."""
    manifest = Path(space) / "checkpoints" / f"boundary-{int(record['boundary']):03d}" \
        / "checkpoint.json"
    return {"schema": JOIN_RECEIPT_SCHEMA, "boundary": int(record["boundary"]),
            "checkpoint": {"path": str(manifest), "sha256": _file_sha256(manifest)},
            "cotangent_sha256": record["cotangent_sha256"],
            "cotangents": len(record["activation_entries"])}


def main(argv=None) -> int:
    """Join one boundary's partials (CPU only, one PrismaBuild row)."""
    from .joint_adjoint_checkpoints import adjoint_space
    from .stage_a_chain_resume import ChainResumeRefused, read_chain_state

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--receipt", default=None,
                        help="also write the join's receipt to this path")
    args = parser.parse_args(argv)
    space = adjoint_space(args.output_root)
    try:
        state = read_chain_state(space)
        record = join_split_checkpoint(
            space, args.boundary, n_probes=int(state["run_identity"]["n_probes"]),
            n_batches=int(state["n_batches"]))
    except (ChainSplitRefused, ChainResumeRefused) as exc:
        print(f"stage_a_chain_split: join refused: {exc}", file=sys.stderr, flush=True)
        return 2
    receipt = join_receipt(space, record)
    if args.receipt is not None:
        atomic_write_bytes(Path(args.receipt), (json.dumps(
            receipt, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
