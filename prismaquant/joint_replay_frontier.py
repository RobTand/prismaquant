"""The sealed replay frontier of a resumed windowed joint preparation.

A fresh windowed ``prepare`` walks the layer/part order its data manifest
declares, so the phases it seals are the order it reads: the head, then
``layer-<L>-part-<P>`` over complete qualification units. A *resumed* prepare
does not start there. ``_qualification_replay`` first re-authenticates every
unit the qualification journal already holds -- its X/H capture, then each
measured rung's wire and render -- and only then does the layer walk qualify
the units the journal does not hold.

Those are two different orders over the same files. Declaring the fresh
phases during a replay would name bytes that have not been read yet, and
PrismaBuild releases the manifest prefix that precedes a phase it is told
about; a false transition there releases bytes still in flight and the claim
stops being true. That is why the resumed path sealed no phases at all
(RobTand/prismaquant#607): reporting nothing is honest, but it leaves the
storage role charging the whole manifest against the ARC budget.

This module is the one definition of the resumed order, and therefore of what
may be sealed. It is deliberately free of torch, of the format registry and
of any PrismaQuant import, because both ends of the contract need it:

* the manifest producer, which runs in a CPU environment without the GPU
  image (``experiments/glm_data_manifests.py``), loads it by file path;
* the preparing action, which walks the order and reports each transition
  only after the files before it are on the end of a completed read.

The roster and its read order are bound to the *journal identity* that was on
disk when the action was sealed. A journal, checkpoint or unit set that
changed after submission is refused rather than reported: a resumed action
that announced a consumed prefix for files it did not read would keep a
broken row alive, which is the one thing the watchdog exists not to do.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Sequence


#: The qualification journal's stage string. Held against
#: ``tessera_joint_aura.QUALIFICATION_STAGE`` and
#: ``cost_stage_checkpoint``'s envelope check by
#: ``tests/test_joint_replay_frontier_607.py``: the producer reads journals the
#: preparing action wrote, and it cannot import either module to ask.
QUALIFICATION_STAGE = "Tessera joint anchor qualification"

#: ``cost_stage_checkpoint.MANIFEST_SCHEMA`` / ``UNIT_SCHEMA``, restated for
#: the same reason. A journal written by any other schema is not this
#: contract.
JOURNAL_MANIFEST_SCHEMA = "prismaquant.cost_stage_checkpoint.manifest.v1"
JOURNAL_UNIT_SCHEMA = "prismaquant.cost_stage_checkpoint.unit.v1"

#: The three file roles a replay read has, in the order one unit is read.
CAPTURE = "capture"
WIRE = "wire"
RENDER = "render"

#: The annotations a sealed resumed prepare carries. ``replay_roster`` is the
#: sorted unit set the journal held at submission, ``replay_read_order_sha256``
#: is the digest of the exact read order sealed beside it, and
#: ``replay_journal_identity_sha256`` is the journal identity those units were
#: committed under.
ROSTER_KEY = "replay_roster"
ROSTER_SHA256_KEY = "replay_roster_sha256"
READ_ORDER_SHA256_KEY = "replay_read_order_sha256"
JOURNAL_IDENTITY_KEY = "replay_journal_identity_sha256"
PHASE_START_UNITS_KEY = "replay_phase_start_units"

#: Every phase this module names starts here, so the producing side and the
#: preparing side can tell a replay phase from a layer phase without a table.
REPLAY_PHASE_PREFIX = "replay-"


def replay_phase_name(index: int) -> str:
    """``replay-0007``: a phase name for the eighth replayed unit.

    Indexed rather than qname-derived because a unit qname is a dotted module
    path, and a phase name is spelled into a command line and a sealed policy.
    The mapping from qname to phase is sealed beside it in
    ``replay_phase_start_units``, which is where a reader should look.
    """
    if type(index) is not int or index < 0:
        raise ValueError(f"replay phase index must be a non-negative int: {index!r}")
    return f"{REPLAY_PHASE_PREFIX}{index:04d}"


def is_replay_phase(name: object) -> bool:
    return isinstance(name, str) and name.startswith(REPLAY_PHASE_PREFIX)


def replay_read_items(
    roster: Iterable[str], cells_by_unit: Mapping[str, Iterable[str]]
) -> list[tuple[str, str, str | None]]:
    """The exact order a replay reads, one complete unit at a time.

    ``_qualification_replay`` authenticates a unit's X/H capture and then each
    of its measured rungs' wire and render. The unit is the durable boundary,
    so the order is grouped by unit -- capture, then that unit's cells -- not
    two whole-roster passes. Items are ``(kind, qname, fmt)`` with ``fmt``
    ``None`` for the capture.
    """
    items: list[tuple[str, str, str | None]] = []
    for name in sorted(roster):
        cells = cells_by_unit.get(name)
        if cells is None:
            raise ValueError(f"replay roster names a unit with no cells: {name}")
        items.append((CAPTURE, name, None))
        for fmt in sorted(cells):
            items.append((WIRE, name, fmt))
            items.append((RENDER, name, fmt))
    return items


def _framed(items: Sequence[tuple[str, str, str | None]]) -> bytes:
    """Length-framed items, so adjacent rows cannot be read two ways."""
    digest = hashlib.sha256(b"prismaquant.joint_prepare.replay_read_order.v1\n")
    for kind, name, fmt in items:
        row = json.dumps((kind, name, fmt), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.digest()


def roster_sha256(roster: Iterable[str]) -> str:
    digest = hashlib.sha256(b"prismaquant.joint_prepare.replay_roster.v1\n")
    for name in roster:
        row = json.dumps(name, ensure_ascii=False, separators=(",", ":")).encode()
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.hexdigest()


def read_order_sha256(items: Sequence[tuple[str, str, str | None]]) -> str:
    return _framed(items).hex()


def seal_frontier(
    roster: Iterable[str],
    cells_by_unit: Mapping[str, Iterable[str]],
    *,
    journal_identity_sha256: str,
) -> dict[str, object]:
    """The annotations that bind a resumed submission to one journal state.

    ``roster`` must already be the completed set read from the journal; this
    function does not decide it, it seals what the producer read and checks it
    is internally consistent (sorted, unique, every unit has cells).
    """
    ordered = list(roster)
    if ordered != sorted(set(ordered)):
        raise ValueError("a replay roster is a sorted set of unit qnames")
    if not isinstance(journal_identity_sha256, str) or len(journal_identity_sha256) != 64:
        raise ValueError("a sealed replay frontier needs the journal identity sha256")
    items = replay_read_items(ordered, cells_by_unit)
    starts = {name: replay_phase_name(index) for index, name in enumerate(ordered)}
    return {
        ROSTER_KEY: ordered,
        ROSTER_SHA256_KEY: roster_sha256(ordered),
        READ_ORDER_SHA256_KEY: read_order_sha256(items),
        JOURNAL_IDENTITY_KEY: journal_identity_sha256,
        PHASE_START_UNITS_KEY: starts,
    }


def sealed_from_annotations(annotations: Mapping[str, object]) -> dict[str, object] | None:
    """The sealed frontier in a data manifest's annotations, or ``None``.

    Absent means the submission sealed none, which is the fresh-run shape and
    the shape older sealed actions carry. Present-but-inconsistent is a
    refusal: a half-sealed frontier cannot be checked against anything.
    """
    if not isinstance(annotations, Mapping):
        raise ValueError("a data manifest's annotations are an object")
    present = [key for key in (ROSTER_KEY, ROSTER_SHA256_KEY, READ_ORDER_SHA256_KEY,
                               JOURNAL_IDENTITY_KEY, PHASE_START_UNITS_KEY)
               if key in annotations]
    if not present:
        return None
    if len(present) != 5:
        raise ValueError(
            "a sealed replay frontier is all of " + ", ".join(sorted(
                (ROSTER_KEY, ROSTER_SHA256_KEY, READ_ORDER_SHA256_KEY,
                 JOURNAL_IDENTITY_KEY, PHASE_START_UNITS_KEY)))
            + f"; this manifest carries only {sorted(present)}")
    roster = annotations[ROSTER_KEY]
    starts = annotations[PHASE_START_UNITS_KEY]
    if (not isinstance(roster, list)
            or roster != sorted(set(roster))
            or any(not isinstance(name, str) for name in roster)):
        raise ValueError("a sealed replay roster is a sorted, unique list of qnames")
    if not isinstance(starts, dict) or set(starts) != set(roster):
        raise ValueError("a sealed replay phase table names exactly its roster")
    if any(not is_replay_phase(phase) for phase in starts.values()):
        raise ValueError("a replayed unit starts in a replay phase")
    if roster_sha256(roster) != annotations[ROSTER_SHA256_KEY]:
        raise ValueError("the sealed replay roster does not hash to its own digest")
    return {
        ROSTER_KEY: list(roster),
        ROSTER_SHA256_KEY: annotations[ROSTER_SHA256_KEY],
        READ_ORDER_SHA256_KEY: annotations[READ_ORDER_SHA256_KEY],
        JOURNAL_IDENTITY_KEY: annotations[JOURNAL_IDENTITY_KEY],
        PHASE_START_UNITS_KEY: dict(starts),
    }


def require_replay_matches(
    sealed: Mapping[str, object],
    *,
    completed: Iterable[str],
    cells_by_unit: Mapping[str, Iterable[str]],
    journal_identity_sha256: str,
) -> None:
    """Refuse a journal, checkpoint or unit set that moved after submission.

    Called by the preparing action before it reports anything. Every clause
    raises: a mismatch here means the sealed phase table describes reads this
    action is not going to make, and reporting it would release bytes that
    nothing consumed.
    """
    roster = sealed[ROSTER_KEY]
    actual = sorted(completed)
    if actual != list(roster):
        missing = sorted(set(roster) - set(actual))
        extra = sorted(set(actual) - set(roster))
        raise RuntimeError(
            "the qualification journal no longer holds the sealed replay "
            f"roster: {len(missing)} sealed unit(s) are gone and {len(extra)} "
            "appeared; refusing to report a consumed prefix for reads this "
            f"action will not make (gone: {missing[:3]}, new: {extra[:3]})")
    if sealed[JOURNAL_IDENTITY_KEY] != journal_identity_sha256:
        raise RuntimeError(
            "the qualification journal identity changed after submission "
            f"(sealed {sealed[JOURNAL_IDENTITY_KEY]}, now "
            f"{journal_identity_sha256}); refusing to report a consumed prefix "
            "against a different checkpoint, capture or window contract")
    if sealed[READ_ORDER_SHA256_KEY] != read_order_sha256(
            replay_read_items(actual, cells_by_unit)):
        raise RuntimeError(
            "the sealed replay read order does not match the journal's units; "
            "refusing to report a consumed prefix for a different order")
