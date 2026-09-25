"""PB-publishable verification quanta for the joint head walk (#765).

The head walk (``tessera_joint_aura.load_measured_anchor_input``) verifies one
unit at a time -- identity reads plus checksums over wire blobs -- and, since
#754, banks each verified unit under the campaign's own checkpoint machinery
with commitment in the roster's one deterministic order. What it still does
not do is leave its container: the whole 36,423-unit roster walks inside one
PB action.

This module is the data half of distributing that walk. It names the quantum
(a deterministic ``lo:hi`` slice of the sorted census roster), the check a
caller applies before a slice may bank (the descriptor must name this exact
roster and this exact slice), the coverage proof a set of quanta must carry
(disjoint slices covering ``[0, N)`` under one roster digest), and the
collector that joins per-quantum journals into the one journal a resumed walk
consumes unchanged.

Deliberately absent: any dispatcher. Sharding, placement, admission and
balancing are PrismaBuild's job -- these descriptors are rows a submitter may
publish through ``pbcampaign`` (each quantum one action, residency-gated on
the stage tier like a movement node), never an application-side scheduler.
Per the repo's work-granularity policy, if PrismaBuild cannot subdivide a
workload the response is a reported capability gap, not a second dispatcher.

Fleet execution of these quanta additionally waits on the D45 measurement
gate: the campaign's own resubmission must first say whether the walk is
checksum-bound or read-bound at full-core fan-out, because a read-bound walk
is fed by the tier rate, not by core count. Until then this machinery is
opt-in (``head_walk_quantum=``) and the production prepare path is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from .schemas import Contract


#: The schema one quantum descriptor carries. Bumped only by a change to the
#: descriptor's key set; the journal schemas it banks under stay owned by
#: ``tessera_joint_aura`` and ``cost_stage_checkpoint``.
QUANTUM_SCHEMA = "prismaquant.joint_head_walk_quanta.v1"


def roster_digest(names) -> str:
    """The digest that binds a descriptor to its exact roster.

    The same construction the head-walk journal already uses for its
    ``roster_sha256`` (``tessera_joint_aura.load_measured_anchor_input``), so
    a descriptor built here and a journal banked there either agree or refuse.
    """
    roster = sorted(names)
    return hashlib.sha256("\n".join(roster).encode("utf-8")).hexdigest()


_require = Contract(ValueError).require


def _checked_names(names) -> list:
    _require(isinstance(names, (list, tuple)) or
             (isinstance(names, Sequence) and not isinstance(names, str)),
             "head-walk quanta need a sequence of unit names")
    roster = [name for name in names]
    _require(bool(roster), "head-walk quanta need a non-empty roster")
    _require(all(isinstance(name, str) and name for name in roster),
             "head-walk quanta need non-empty string unit names")
    _require(len(set(roster)) == len(roster), "head-walk roster has duplicate units")
    return sorted(roster)


def head_walk_quanta(names, *, max_units_per_quantum) -> list:
    """Cut the roster into deterministic verification quanta (data only).

    The roster is ``sorted(names)`` -- the one order every reader already
    walks -- cut into contiguous half-open ``[lo, hi)`` slices of at most
    ``max_units_per_quantum`` units. The output is JSON data: no threads, no
    placement, no execution. Publishing these rows (e.g. one ``pbcampaign``
    row per quantum) is the submitter's job; running them is PrismaBuild's.
    """
    roster = _checked_names(names)
    _require(type(max_units_per_quantum) is int and max_units_per_quantum > 0,
             "max_units_per_quantum must be a positive integer")
    digest = roster_digest(roster)
    total = len(roster)
    quanta = (total + max_units_per_quantum - 1) // max_units_per_quantum
    descriptors = []
    for index in range(quanta):
        low, high = index * max_units_per_quantum, min(total, (index + 1) * max_units_per_quantum)
        units = roster[low:high]
        descriptors.append({
            "schema": QUANTUM_SCHEMA,
            "quantum_id": f"quantum-{index:04d}",
            "lo": low,
            "hi": high,
            "units": list(units),
            "roster_sha256": digest,
            "slice_sha256": hashlib.sha256("\n".join(units).encode("utf-8")).hexdigest(),
            "quanta": quanta,
        })
    return descriptors


def check_quantum_for_roster(quantum, names) -> tuple:
    """Bind one descriptor to the live roster, fail-closed. Returns ``(lo, hi)``.

    Every field is checked against the roster in front of the caller, not
    against a cached claim: the schema, the full-roster digest, the slice
    bounds, the exact unit list at those bounds, and the slice digest. A
    descriptor from another campaign, another roster revision, or another
    slice is refused here rather than banking a journal shard nothing can
    join.
    """
    _require(isinstance(quantum, Mapping), "head-walk quantum must be a mapping")
    _require(quantum.get("schema") == QUANTUM_SCHEMA,
             f"head-walk quantum schema must be {QUANTUM_SCHEMA}")
    roster = _checked_names(names)
    _require(quantum.get("roster_sha256") == roster_digest(roster),
             "head-walk quantum names another roster")
    low, high = quantum.get("lo"), quantum.get("hi")
    _require(type(low) is int and type(high) is int and 0 <= low < high <= len(roster),
             f"head-walk quantum slice [{low}:{high}) is outside the {len(roster)}-unit roster")
    _require(list(quantum.get("units") or []) == roster[low:high],
             "head-walk quantum units differ from the roster at its slice")
    _require(quantum.get("slice_sha256") == hashlib.sha256(
        "\n".join(roster[low:high]).encode("utf-8")).hexdigest(),
        "head-walk quantum slice digest differs from the roster at its slice")
    _require(isinstance(quantum.get("quantum_id"), str) and quantum["quantum_id"],
             "head-walk quantum needs a non-empty quantum_id")
    _require(type(quantum.get("quanta")) is int and quantum["quanta"] >= 1,
             "head-walk quantum needs a positive quanta count")
    return (low, high)


def verify_quanta_coverage(quanta, names) -> list:
    """The coverage proof: these descriptors verify this roster exactly once.

    Every descriptor must check out against the roster (see
    ``check_quantum_for_roster``), all must share one roster digest, ids must
    be unique, and the slices must tile ``[0, N)`` contiguously -- no gap no
    overlap. Returns the descriptors in walk order. This is the proof the
    collector replays before joining, and the reason a lost or duplicated
    quantum is a refusal rather than a silent hole.
    """
    _require(isinstance(quanta, (list, tuple)) and bool(quanta),
             "head-walk coverage needs a non-empty list of quanta")
    _require(all(isinstance(quantum, Mapping) for quantum in quanta),
             "head-walk coverage needs quantum descriptor mappings")
    roster = _checked_names(names)
    ordered = sorted(quanta, key=lambda quantum: (quantum.get("lo"), quantum.get("hi")))
    for quantum in ordered:
        check_quantum_for_roster(quantum, roster)
    # One shared roster digest holds transitively: every descriptor above was
    # checked against this same roster, so a mixed set refuses there.
    ids = [quantum["quantum_id"] for quantum in ordered]
    _require(len(set(ids)) == len(ids), "head-walk quanta reuse a quantum_id")
    cursor = 0
    for quantum in ordered:
        _require(quantum["lo"] == cursor,
                 f"head-walk coverage breaks at roster position {cursor}")
        cursor = quantum["hi"]
    _require(cursor == len(roster),
             f"head-walk quanta cover [0:{cursor}) of a {len(roster)}-unit roster")
    return ordered


def _quantum_journal_identity(journal_dir) -> dict:
    """Read one quantum journal's manifest identity (the custody root)."""
    manifest_path = Path(journal_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"head-walk quantum journal at {journal_dir} has no readable manifest") from exc
    _require(isinstance(manifest, Mapping), "head-walk quantum manifest is not an object")
    identity = manifest.get("identity")
    _require(isinstance(identity, Mapping), "head-walk quantum manifest carries no identity")
    return dict(identity)


def join_head_walk_journals(receipts, *, dest, names) -> dict:
    """Join per-quantum journals into the one journal a resumed walk consumes.

    ``receipts`` is one ``{"quantum": descriptor, "journal_dir": path}`` per
    quantum; ``names`` is the full census roster the join must reproduce. The
    roster comes from the submitter, never from the receipts: deriving it
    from the surviving shards would shrink the roster to fit a loss, so a
    lost quantum fails the coverage proof instead of passing silently. Each
    journal is then opened strictly -- its manifest must match the slice
    identity it claims, every envelope's identity binding and payload
    digest is checked on load -- then custody is verified across journals
    (one shared input set, one roster digest, coverage exactly once), and the
    verified unit states are re-enveloped under the full-roster identity into
    ``dest``, which must not exist yet.

    The joined journal is indistinguishable from a single-container walk's:
    the same stage, the same identity keys minus the per-quantum marker, one
    envelope per roster unit. The consumer's resume path re-verifies every
    banked unit against the live bytes before trusting it, so the join never
    has to be trusted -- only the coverage proof has to hold, and it is
    checked here.
    """
    from .cost_stage_checkpoint import prepare_journal, write_unit

    from . import tessera_joint_aura as bridge

    stage = bridge.HEAD_WALK_STAGE
    _require(isinstance(receipts, (list, tuple)) and bool(receipts),
             "head-walk join needs a non-empty list of quantum receipts")
    dest = Path(dest)
    _require(not dest.exists(), f"head-walk join destination already exists: {dest}")
    full_roster = _checked_names(names)
    verify_quanta_coverage(
        [receipt.get("quantum") for receipt in receipts
         if isinstance(receipt, Mapping)], full_roster)

    opened = []
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "head-walk join receipt must be a mapping")
        quantum, journal_dir = receipt.get("quantum"), receipt.get("journal_dir")
        _require(isinstance(quantum, Mapping) and journal_dir is not None,
                 "head-walk join receipt needs a quantum descriptor and a journal_dir")
        identity = _quantum_journal_identity(journal_dir)
        stored_quantum = identity.get("quantum")
        _require(dict(stored_quantum or {}) == dict(quantum),
                 "head-walk quantum journal was banked under another descriptor")
        expected_units = list(quantum.get("units") or [])
        root, seal, completed = prepare_journal(
            journal_dir, stage=stage, resume=True, identity=identity,
            qnames=expected_units)
        _require(set(completed) == set(expected_units),
                 "head-walk quantum journal is missing banked units")
        for name, state in completed.items():
            _require(isinstance(state, dict)
                     and state.get("schema") == bridge.HEAD_WALK_STATE_SCHEMA,
                     f"head-walk quantum unit {name} carries no verified walk state")
        opened.append({"quantum": dict(quantum), "identity": identity,
                       "completed": completed})

    # One shared input set across every quantum: the inputs digests, the
    # checkpoint seal, the roster digest and the encoder-reuse binding must be
    # identical, and only the per-quantum marker may differ.
    shared_keys = ("inputs", "required_source_units", "required_campaign_groups",
                   "checkpoint_seal", "roster_sha256", "render_mirror_root",
                   "encoder_source_reuse_sha256")
    first = opened[0]["identity"]
    for key in shared_keys:
        _require(all(entry["identity"].get(key) == first.get(key) for entry in opened),
                 f"head-walk quanta disagree on journal identity field {key!r}")
    _require(first.get("roster_sha256") == roster_digest(full_roster),
             "head-walk quantum journals name another roster")
    full_identity = {key: value for key, value in first.items() if key != "quantum"}
    _require("quantum" in first, "head-walk quantum journal carries no quantum marker")
    _require(full_identity.get("schema") == bridge.HEAD_WALK_JOURNAL_SCHEMA,
             "head-walk quantum journal carries a foreign journal schema")

    root, seal, _ = prepare_journal(dest, stage=stage, resume=False,
                                    identity=full_identity, qnames=full_roster)
    for entry in opened:
        for name, state in entry["completed"].items():
            write_unit(root, stage=stage, qname=name, identity_sha256=seal, state=state)
    _, _, joined = prepare_journal(dest, stage=stage, resume=True,
                                   identity=full_identity, qnames=full_roster)
    _require(set(joined) == set(full_roster), "head-walk join published an incomplete journal")
    return {"dest": str(root), "identity_sha256": seal, "units": list(full_roster),
            "quanta": len(opened)}
