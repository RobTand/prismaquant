"""Stage A forward split: one run's forward capture as partition-range quanta (PQ #738).

A fresh Stage A run captures every calibration partition's forward
boundaries through all layers, then rolls each partition's tail cotangents
and seals the tail checkpoint. Partitions never communicate on the way: each
entry is written under its own ``(batch, boundary)`` coordinates, and a
partition's forward and tail read nothing another partition wrote. So the
capture splits by contiguous partition range, and each part is a PrismaBuild
row that any free GPU can run:

* **A prep row, run once.** It mints the run's exact boundary generation,
  leaves the generation's status file at ``running`` for the quanta to
  rebind, and seals the forward prep record (``split/forward/prep.json``):
  every field of the chain state except the ones the quanta produce (the
  boundary entries and the tail checkpoint), plus the session and the
  ranges. It captures nothing.
* **Quanta.** Each rebinds the prep's session as one owner of the generation
  (``owners/forward-samples-SSSSSS-EEEEEE.json``), captures its range's
  boundaries through every layer under their global batch indices, rolls its
  range's tail cotangents, and seals a **partial** tail checkpoint
  (``split/boundary-NNN/samples-SSSSSS-EEEEEE``) in the layout of the chain
  split's partials. It writes its boundary entry records beside the prep
  record. A range is whole read windows (``prefetch_batches``), so each
  quantum's produced-output groups are its own.
* **One join.** It holds the ranges to the partition plane with
  :func:`~prismaquant.cost_streaming.verify_boundary_partition_coverage`,
  requires every quantum's owner status to be ``complete``, publishes the
  tail checkpoint from the partials with the chain split's own join
  (:func:`~prismaquant.stage_a_chain_split.join_split_checkpoint`), and
  writes the chain state a single owner writes after its tail. The chain
  state names no producer: the join runs only over owners whose own status
  says they finished, so no interrupted writer is left to contain.

From there the run is a chain resume at the tail checkpoint, and the chain
split (``stage_a_chain_split``) rolls it by sample range from the top.

The forward relaunch compares full bind identities through the existing
chain-resume seal/comparability owner; it never changes the session hash.
The join checks immutable records and whole-plane coverage as before.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
    publish_new_bytes,
)
from .digests import bytes_sha256hex, canonical_json_bytes, indent2_json_file_bytes
from .stage_a_chain_split import (
    ChainSplitRefused,
    parse_ranges,
    quantum_label,
    range_name,
    require_whole_plane,
    split_root,
)

PREP_RECORD_SCHEMA = "prismaquant.stage_a.forward_split_prep.v1"
PREP_RECEIPT_SCHEMA = "prismaquant.stage_a.forward_split_prep_receipt.v1"
QUANTUM_RECEIPT_SCHEMA = "prismaquant.stage_a.forward_split_quantum.v1"
FRAGMENT_SCHEMA = "prismaquant.stage_a.forward_split_entries.v1"
JOIN_RECEIPT_SCHEMA = "prismaquant.stage_a.forward_split_join.v1"
PREP, QUANTUM = "prep", "quantum"
PREP_OWNER_LABEL = "forward-prep"

#: The chain-state fields the prep seals; the join adds the rest.
CHAIN_STATE_FIELDS = ("run_identity", "stride", "boundary_storage", "bind_identity",
                      "arithmetic", "n_batches", "num_layers",
                      "artifact_budget_override")


class ForwardSplitRefused(ChainSplitRefused):
    """The forward split cannot run or join without losing or corrupting bytes."""


# -- layout --------------------------------------------------------------------

def forward_root(space) -> Path:
    return split_root(space) / "forward"


def prep_record_path(space) -> Path:
    return forward_root(space) / "prep.json"


def fragment_path(space, start: int, stop: int) -> Path:
    """One quantum's boundary entry records."""
    return forward_root(space) / f"{range_name(start, stop)}.entries.json"


def owner_status_path(generation_directory, start: int, stop: int) -> Path:
    label = quantum_label(start, stop, role="forward")
    return Path(generation_directory) / "owners" / f"{label}.json"


# -- the split spec ------------------------------------------------------------

def normalize_forward_split(spec) -> dict:
    """``{role: prep, ranges}`` or ``{role: quantum, samples}``, in shape only.

    The run's own numbers are checked by
    :func:`~prismaquant.stage_a_chain_split.check_ranges` once the run is known.
    """
    if not isinstance(spec, dict) or spec.get("role") not in (PREP, QUANTUM):
        raise ForwardSplitRefused("a forward split is a prep or a quantum")
    if spec["role"] == PREP:
        if set(spec) != {"role", "ranges"}:
            raise ForwardSplitRefused("a forward split prep names its ranges")
        ranges = [_pair(value, "a prep range") for value in spec["ranges"] or ()]
        if not ranges:
            raise ForwardSplitRefused("a forward split prep names at least one range")
        return {"role": PREP, "ranges": [list(pair) for pair in sorted(ranges)]}
    if set(spec) != {"role", "samples"}:
        raise ForwardSplitRefused("a forward split quantum names its samples")
    return {"role": QUANTUM, "samples": list(_pair(spec["samples"], "a quantum's samples"))}


def _pair(value, where) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ForwardSplitRefused(f"{where} is a [start, stop) pair of sample indices")
    return int(value[0]), int(value[1])


def parse_quantum(text) -> list[int]:
    """``START:STOP`` from the command line."""
    ranges = parse_ranges(text)
    if len(ranges) != 1:
        raise ForwardSplitRefused(f"a forward quantum is one START:STOP, not {text!r}")
    return ranges[0]


# -- the prep record -------------------------------------------------------------

def write_prep_record(space, *, chain_state: dict, session: dict, ranges, n_probes: int,
                      group_size: int) -> dict:
    """Seal the prep record once; a second prep of the run refuses."""
    if set(chain_state) != set(CHAIN_STATE_FIELDS):
        raise ForwardSplitRefused("the prep seals every chain-state field but the produced ones")
    body = canonical_json({"schema": PREP_RECORD_SCHEMA, "chain_state": chain_state,
                           "session": session, "ranges": [list(pair) for pair in ranges],
                           "n_probes": int(n_probes), "group_size": int(group_size)},
                          where="forward split prep")
    document = {**body, "prep_sha256": canonical_json_sha256(body, where="forward split prep")}
    path = prep_record_path(space)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = indent2_json_file_bytes(document)
    if not publish_new_bytes(path, payload):
        raise ForwardSplitRefused(f"{path} exists: a run's forward split is prepped once")
    return {"path": str(path), "sha256": bytes_sha256hex(payload),
            "document": document}


def read_prep_record(space) -> dict:
    """The prep record, its own seal checked."""
    path = prep_record_path(space)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ForwardSplitRefused(f"the run has no forward split prep at {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != PREP_RECORD_SCHEMA:
        raise ForwardSplitRefused(f"{path} is not a {PREP_RECORD_SCHEMA} document")
    body = {key: value for key, value in document.items() if key != "prep_sha256"}
    if canonical_json_sha256(body, where="forward split prep") != document.get("prep_sha256"):
        raise ForwardSplitRefused(f"{path} does not seal its own content")
    return document


def adopt_forward_bind_identity(prep, running_identity):
    """Compare full retained/run identities, then reuse the original session identity."""
    from .stage_a_chain_resume import ChainResumeRefused, require_chain_fields_equal

    retained = prep["chain_state"]["bind_identity"]
    try:
        require_chain_fields_equal(
            {"bind_identity": retained}, {"bind_identity": running_identity},
            fields=("bind_identity",), where="Stage A forward split")
    except ChainResumeRefused as exc:
        raise ForwardSplitRefused(str(exc)) from exc
    return retained

def generation_directory(prep: dict) -> Path:
    storage = prep["chain_state"]["boundary_storage"]
    return Path(storage["directory"]) / str(storage["session"]["generation"])


# -- a quantum's entry records ---------------------------------------------------

def write_fragment(space, samples, *, session: dict, boundary_entries: dict) -> Path:
    """A quantum's boundary entry records, by boundary, in global batch order."""
    start, stop = (int(value) for value in samples)
    path = fragment_path(space, start, stop)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": FRAGMENT_SCHEMA, "samples": [start, stop], "session": session,
                "boundary_entries": boundary_entries}
    atomic_write_bytes(path, canonical_json_bytes(document, where="forward entries") + b"\n")
    return path


def _read_fragment(space, start, stop, *, session, num_layers) -> dict:
    path = fragment_path(space, start, stop)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ForwardSplitRefused(f"range {start}:{stop} wrote no entry records") from exc
    if (document.get("schema") != FRAGMENT_SCHEMA
            or document.get("samples") != [start, stop]
            or document.get("session") != session):
        raise ForwardSplitRefused(f"{path} is not range {start}:{stop}'s record of this run")
    entries = document.get("boundary_entries")
    if not isinstance(entries, dict) or set(entries) != {
            str(boundary) for boundary in range(int(num_layers))}:
        raise ForwardSplitRefused(f"{path} does not name every forward boundary")
    for boundary, rows in entries.items():
        if len(rows) != stop - start:
            raise ForwardSplitRefused(
                f"{path} names {len(rows)} entries at boundary {boundary}, not {stop - start}")
        for batch, row in zip(range(start, stop), rows):
            identity = row["metadata"]["identity"]
            if (identity["session"] != session or identity["kind"] != "boundary"
                    or identity["coordinates"] != {"batch": batch, "boundary": int(boundary),
                                                   "probe": None}):
                raise ForwardSplitRefused(
                    f"{path} names {row['name']} for batch {batch} at boundary {boundary}")
            try:
                size = Path(row["path"]).stat().st_size
            except OSError as exc:
                raise ForwardSplitRefused(f"{row['path']} is missing") from exc
            if size != int(row["file_bytes"]):
                raise ForwardSplitRefused(f"{row['path']} is not the size its record names")
    return entries


# -- the join ------------------------------------------------------------------

def join_forward_split(space) -> dict:
    """Publish the tail checkpoint and the chain state; returns the join receipt.

    The prep's ranges must tile the ``n_batches`` partitions, every range's
    owner must have finished (``complete``) under the prep's session, and
    every range's entry records must name its own batches at every forward
    boundary. The tail checkpoint is the chain split's join of the
    quanta's partials; the chain state is the one a single owner writes,
    with the entry records in global batch order and no producer.
    """
    from .joint_adjoint_checkpoints import checkpoint_directory
    from .stage_a_chain_resume import build_chain_state, chain_state_path, write_chain_state
    from .stage_a_chain_split import join_split_checkpoint

    space = Path(space)
    prep = read_prep_record(space)
    state = prep["chain_state"]
    n_batches, num_layers = int(state["n_batches"]), int(state["num_layers"])
    ranges = [tuple(pair) for pair in prep["ranges"]]
    try:
        require_whole_plane(ranges, n_batches=n_batches)
    except ChainSplitRefused as exc:
        raise ForwardSplitRefused(str(exc)) from exc
    if chain_state_path(space).exists():
        raise ForwardSplitRefused(
            f"{chain_state_path(space)} exists: a run's forward split is joined once")
    session = prep["session"]
    generation = generation_directory(prep)
    for start, stop in ranges:
        status_path = owner_status_path(generation, start, stop)
        try:
            status = json.loads(status_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ForwardSplitRefused(
                f"range {start}:{stop} has no owner status at {status_path}") from exc
        if status.get("session") != session or status.get("status") != "complete":
            raise ForwardSplitRefused(
                f"range {start}:{stop}'s owner is {status.get('status')!r}, not a "
                "complete owner of this run's generation")
    entries = {str(boundary): [] for boundary in range(num_layers)}
    for start, stop in ranges:
        fragment = _read_fragment(space, start, stop, session=session,
                                  num_layers=num_layers)
        for boundary, rows in fragment.items():
            entries[boundary].extend(rows)
    tail = join_split_checkpoint(space, num_layers, n_probes=int(prep["n_probes"]),
                                 n_batches=n_batches)
    written = write_chain_state(space, build_chain_state(
        **state, boundary_entries=entries, tail_checkpoint=tail, producer=None))
    manifest = checkpoint_directory(space, num_layers) / "checkpoint.json"
    return {"schema": JOIN_RECEIPT_SCHEMA, "boundary": num_layers,
            "ranges": [list(pair) for pair in ranges],
            "tail_checkpoint": {"path": str(manifest),
                                "cotangent_sha256": tail["cotangent_sha256"],
                                "cotangents": len(tail["activation_entries"])},
            "chain_state": {"path": written["path"], "sha256": written["sha256"]},
            "prep_sha256": prep["prep_sha256"]}


def main(argv=None) -> int:
    """Join a run's forward split (CPU only, one PrismaBuild row)."""
    from .joint_adjoint_checkpoints import adjoint_space

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt", default=None,
                        help="also write the join's receipt to this path")
    args = parser.parse_args(argv)
    try:
        receipt = join_forward_split(adjoint_space(args.output_root))
    except ChainSplitRefused as exc:
        print(f"stage_a_forward_split: join refused: {exc}", file=sys.stderr, flush=True)
        return 2
    if args.receipt is not None:
        atomic_write_bytes(Path(args.receipt), indent2_json_file_bytes(receipt))
    print(indent2_json_file_bytes(receipt).decode(), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
