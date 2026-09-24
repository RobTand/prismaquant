"""Read a source shard's tensors off PrismaBuild's stage tier, ranges included.

``layer_streaming._source_safe_open`` is the one seam every BF16 source-shard
read passes through, and until now it opened the declared pool path and nothing
else. The stage held the bytes and no reader asked for them (PQ #732).

A shard is not staged the way the two existing consumers' files are. PrismaBuild
stages the byte *ranges* a data manifest declares, and its mover writes each
range as a file of its own whose byte 0 is the range's first byte
(``<shard>.safetensors.pbrange/<offset>-<length>``; PrismaBuild
``tools/fleet/stage_move.py``). On the live GLM-5.3-Flash run 34 of 55 staged
shard entries are ranges, so a path rewrite serves at most the other 21: byte 0
of a range file is tensor payload, not safetensors' header-length prefix, and
``safe_open`` on it parses garbage or a header whose spans overrun the file.

So the redirect is per tensor, not per file:

* The **header** always comes from the declared file. It sits at offset 0 and
  is generally outside any staged extent, and it is what turns a tensor name
  into an absolute byte span.
* Each tensor's **payload** comes from the staged range that covers its span
  outright, when the map has one, and from the declared file otherwise. A span
  straddling two staged ranges is read from the pool: never partial bytes, and
  never a concatenation this reader has not fenced.
* With no map the reader is not built at all. ``staged_shard_opener`` hands the
  caller back its own ``safe_open``, so an unmapped run's behaviour and syscalls
  are what they were.

**What binds a staged read here, and what does not.** The fences are
structural: the entry has to fit inside the declared file, the staged copy has
to be a regular file of exactly the entry's length, and its stat signature has
to be unchanged across every read this reader makes from it. They are the
pre-open fences ``ResidencyResolver.staged_read`` applies, asked of a range.
They are **not** a digest. The two whole-file consumers
(``production_weight_cache._read_file_tensor``,
``tessera_joint_aura._read_wire_bytes``) hash what they read and hold it to the
map's ``sha256``; a shard read cannot, because it reads one tensor's span out
of a multi-gigabyte range and the map's digest covers the whole range. Hashing
the range on first touch would be a second full read of every staged extent
before its first tensor could be served. So on this path the map's own check is
the only check, which is the coverage decision ``docs/ARCHITECTURE.md`` §12 D43
names and this reader makes explicit rather than quiet.

**How a span is read.** One tensor's payload is one span, and until PQ #746 it
was one sequential ``preadv`` loop on one descriptor -- one stream, whatever the
link underneath could carry. It is now cut on the mount's ``rsize`` and read on
at most the mount's ``nconnect`` threads into disjoint slices of the tensor's own
buffer, so the bytes are what the sequential loop produced and the wait is not.
Both numbers come from the mount rather than from us (principle 2); a mount that
is not NFS, or one publishing neither, is read the way it was. The threads are
one process-wide pool, so the layer gather's own reader threads
(``layer_streaming.layer_read_threads``) and this split cannot multiply into
more in-flight reads than the client has transports.

That has one consequence worth stating where it is made: under
``source_authentication`` (``tessera_calibration_cache``'s
``_CaptureSourceSafeOpen``) the payload the caller receives may come from the
stage while the ``prismaquant.selected_source_authentication.v1`` receipt's
hash is of the declared pool file. The declared file's identity fences still
run; the bytes handed over are the stage's, admitted on the map.
"""
from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
import errno
import json
import math
import os
import re
import stat
import threading
import time

import torch

from .residency_map import RANGE_HIT, RANGE_UNCOVERED, residency_resolver
from .staged_tier_policy import (
    StagedRangeNotLanded,
    active_policy,
    policy_is_active,
    refuse_pool_bulk_read,
)
from .staged_lease import LeaseRefused, acquire_entry_window


#: How long a strict read waits for a declared range when PrismaBuild
#: publishes no landing record for it.
#:
#: Since PB #989 the tier loop writes ``<consumer>.landing.json`` beside the
#: map: pending ranges with their mover, the mover's state and the tier's
#: expectation of when it lands. Since PB #1018 a record that carries a
#: ``horizon`` key lists every leg the consumer has still to read while it
#: has a refill horizon, a leg past the horizon with ``deferred_by``; an
#: older record lists only the legs inside the horizon. Where the record
#: covers the range, the reader waits on the mover's state instead of this
#: bound (see :func:`landing_verdict`), and this constant does not apply.
#: Under a record that lists every leg, a span outside the bound read order
#: refuses at once, naming the span (PQ #1113), and this constant does not
#: apply either. It remains the bound for a generation that writes no
#: landing record, for a record without the ``horizon`` key, for a span the
#: record does not list (before the consumer's first accepted progress,
#: a leg past the window's one step of run-ahead has no row), and for a read
#: whose sealed read order is not bound here.
#:
#: A policy bound, not a derived threshold: without the landing record the
#: consumer cannot see PrismaBuild's mover queue, so nothing in this process
#: can compute when a published range will land. It is set from what movers on this fleet
#: measurably take. On the Stage A run behind PQ #874 the stage mover for one
#: 14.8 GB forward phase ran 16:55:57 -> 16:56:41 (44 s), another finished its
#: RAM leg 96 s after its stage leg, and the phase the consumer refused was
#: claimed by its own mover 7 s after the consumer had already died. Five
#: minutes covers that class of lag and is still a bound: a range that never
#: arrives refuses exactly as it did before.
#:
#: ``PRISMAQUANT_STAGED_RANGE_WAIT_S`` overrides it; ``0`` restores the
#: pre-#874 behaviour of refusing on the first uncovered span that no landing
#: record covers. It does not shorten a wait the landing record governs.
STAGED_RANGE_WAIT_ENV = "PRISMAQUANT_STAGED_RANGE_WAIT_S"
STAGED_RANGE_WAIT_S = 300.0
#: Between polls. ``ResidencyResolver._read_map`` is identity-gated, so a poll
#: that finds the map unchanged costs one ``lstat`` and no parse.
STAGED_RANGE_POLL_S = 1.0
#: The longest a staged-range wait goes without a log line (PQ #1167). A
#: policy bound, not a derived threshold: no phase of a campaign may run two
#: minutes with no output, and a line a minute meets that with room for a
#: slow poll. The line also repeats at the first poll after the expectation
#: it printed passes, so a wait that outlives its expectation says so.
STAGED_WAIT_REPORT_S = 60.0


#: One process can have several reads blocked at once (the layer gather's
#: reader threads). PrismaBuild reads one staged-wait record per action, so
#: the declaration is the union of every live wait's movers.
_STAGED_WAITS_LOCK = threading.Lock()
_STAGED_WAITS: dict[int, tuple[float, frozenset]] = {}


def _declare_staged_waits() -> None:
    """Write the union of live waits to PrismaBuild, or clear it. Lock held."""
    from . import prismabuild_progress

    movers = set()
    for _since, names in _STAGED_WAITS.values():
        movers.update(names)
    if movers:
        since = min(since for since, _names in _STAGED_WAITS.values())
        prismabuild_progress.declare_staged_wait(movers, since_unix=since)
    else:
        prismabuild_progress.clear_staged_wait()


def _staged_wait(token: int, since_unix: float | None, movers) -> None:
    """Record (or, with no movers, end) one call's staged wait."""
    with _STAGED_WAITS_LOCK:
        before = _STAGED_WAITS.get(token)
        if movers:
            now = (since_unix, frozenset(movers))
            if before == now:
                return
            _STAGED_WAITS[token] = now
        elif before is None:
            return
        else:
            del _STAGED_WAITS[token]
        _declare_staged_waits()


def _landing_rows(record, positions):
    """The landing ranges overlapping any of ``positions``."""
    rows = []
    for low, high in positions:
        for row in record["ranges"]:
            if row["range_start_bytes"] < high and low < row["range_end_bytes"]:
                rows.append(row)
    return rows


def _describe_landing(row, now_unix, horizon=None) -> str:
    """One range's state, expectation and the numbers behind it, for a log.

    A leg PrismaBuild lists as deferred by the refill horizon (PB #1018)
    says so, where the horizon ends and the consumption that moves it, from
    the record's ``horizon`` block.
    """
    text = f"mover {row['mover_action_key'][:12]} is {row['state']}"
    deferred = row.get("deferred_by")
    if deferred == "horizon":
        text += ", deferred by the refill horizon"
        if isinstance(horizon, dict):
            end = horizon.get("end_bytes")
            rate = horizon.get("consumption_bytes_per_s")
            if type(end) is int:
                text += f" at byte {end}"
            if horizon.get("reading_phase"):
                text += f" while the consumer reads {horizon['reading_phase']}"
            if type(rate) in (int, float) and rate > 0:
                text += f" at {rate / 1e6:.1f} MB/s"
    elif deferred is not None:
        text += f", deferred by {deferred}"
    expected = row.get("expected_landing_unix")
    if type(expected) in (int, float):
        delta = expected - now_unix
        text += (f", expected to land in {delta:.0f} s" if delta >= 0 else
                 f", expected to land {-delta:.0f} s ago")
        if row.get("queue_position") is not None:
            text += f" (queue position {row['queue_position']}"
            if row.get("bytes_ahead") is not None:
                text += f", {row['bytes_ahead'] / 1e9:.1f} GB ahead"
            text += ")"
    else:
        text += ", no expected landing time"
    if row.get("waiting_for"):
        text += f", waiting for {row['waiting_for']}"
    return text


def _lists_every_leg(record) -> bool:
    """Whether PrismaBuild's record answers for every leg (PB #1018).

    Such a record carries the ``horizon`` key, ``null`` when the
    consumer's horizon is undefined. An older record lists only the legs
    inside the refill horizon, so a span it does not place says nothing.
    """
    return "horizon" in record


def landing_verdict(resolver, rows):
    """What PrismaBuild's landing record says about ``rows``.

    Returns ``(kind, detail, movers)``:

    * ``"wait"``: every row is covered by a range whose mover is ``ready``
      or ``claimed``, or that is ``unpublished``, and the tier loop that
      publishes and prices it is alive. ``movers`` names the ranges' movers,
      which the caller declares to PrismaBuild so the wait is not counted as
      quiet (PB #989). The expectation is logged, never enforced: a copy
      slower than every earlier receipt is still a copy.
    * ``"refuse"``: evidence that the bytes will not come. A row whose every
      covering range is ``terminal-no-receipt``, or a tier loop that has not
      announced its tier within the bound the record names
      (``tier_loop_liveness_s``, PrismaBuild's own offer freshness bound,
      the judgment ``PoolQueue._tier_loop_alive`` makes).
    * ``"absent"``: nothing published to wait on. No landing record, a
      span when no read order is bound, a span outside the read order under
      a record that lists only the legs inside the refill horizon (written
      before PB #1018), or a range the record does not list. The caller
      keeps its bounded wait.

    A record that lists every leg (PB #1018) refuses a span outside the
    bound read order at once, naming it (PQ #1113): no leg of the
    consumer's plan covers it, so no mover will stage it. A leg PrismaBuild
    defers past the refill horizon, or until the consumer's first accepted
    progress, is ``unpublished``: it is waited on like any other while the
    tier loop lives, and the detail says it is deferred and what moves it.
    A span such a record does not list keeps the bounded wait: the tier
    loop composes the map before it rewrites the record, so a range that
    has just become resident can leave the record one poll before this
    reader sees it in the map.

    Each check is one ``lstat`` of the record (identity-cached) and one
    small read of the tier record.
    """
    return _landing_verdict(resolver, rows)[:3]


def _landing_verdict(resolver, rows):
    """:func:`landing_verdict`, plus what PrismaBuild priced a wait on.

    The fourth value is ``None`` unless the verdict is ``"wait"``. Then it
    is ``{"expected_unix", "rate", "basis"}``: the expectation of the range
    the detail describes, and the record's landing rate and its basis, each
    ``None`` when the record does not carry it (PQ #1167).
    """
    fetch = getattr(resolver, "landing_record", None)
    record = fetch() if callable(fetch) else None
    if record is None:
        return "absent", "PrismaBuild published no landing record", (), None
    locate = getattr(resolver, "read_order_positions", None)
    age_of = getattr(resolver, "tier_record_age", None)
    if not callable(locate) or not callable(age_of):
        return "absent", "this resolver cannot read a landing record", (), None
    now_unix = time.time()
    horizon = record.get("horizon")
    covering = []
    for declared, start, end, _size in rows:
        positions = locate(declared, start, end)
        if not positions:
            bound = getattr(resolver, "read_order_bound", None)
            if _lists_every_leg(record) and callable(bound) and bound():
                return ("refuse", f"{declared} [{start}, {end}) is not in the "
                        "bound read order: no leg of this consumer's plan "
                        "covers it, so no mover will stage it", (), None)
            return ("absent", f"{declared} [{start}, {end}) is not in the bound "
                    "read order", (), None)
        found = _landing_rows(record, positions)
        if not found:
            return ("absent", f"the landing record lists no pending range for "
                    f"{declared} [{start}, {end})", (), None)
        if all(row["state"] == "terminal-no-receipt" for row in found):
            return ("refuse", f"{declared} [{start}, {end}): "
                    + "; ".join(_describe_landing(row, now_unix, horizon)
                                for row in found),
                    (), None)
        covering.extend(row for row in found
                        if row["state"] != "terminal-no-receipt")
    liveness = float(record["tier_loop_liveness_s"])
    age = age_of(record["tier_id"])
    head = _describe_landing(covering[0], now_unix, horizon)
    deferred = sum(1 for row in covering if row.get("deferred_by"))
    if len(covering) > 1 and deferred:
        head += (f" ({deferred} of the {len(covering)} ranges waited on are "
                 "deferred by PrismaBuild's window)")
    if age is None or age > liveness:
        silent = ("the tier loop's record is unreadable" if age is None else
                  f"the tier loop last announced {record['tier_id']} {age:.0f} s ago")
        return ("refuse", f"{head}; {silent}, beyond PrismaBuild's "
                f"{liveness:g} s liveness bound, so nothing will land it", (), None)
    expected = covering[0].get("expected_landing_unix")
    rate = record.get("landing_bytes_per_s")
    basis = record.get("landing_basis")
    priced = {
        "expected_unix": float(expected) if type(expected) in (int, float) else None,
        "rate": float(rate) if type(rate) in (int, float) and rate > 0 else None,
        "basis": basis if isinstance(basis, str) and basis else None,
    }
    return ("wait", head, tuple(sorted({row["mover_action_key"] for row in covering})),
            priced)


def _priced_text(priced) -> str:
    """The rate and basis behind a wait's expectation, for its log line."""
    if not priced or priced["expected_unix"] is None or priced["rate"] is None:
        return ""
    text = f"; PrismaBuild priced it at {priced['rate'] / 1e6:.1f} MB/s"
    if priced["basis"]:
        text += f" (basis {priced['basis']})"
    return text


def await_staged_spans(resolver, wanted, *, deadline, published=None, cancel=None,
                       published_batch=None) -> str:
    """Wait for PrismaBuild's movers to land ``wanted``.

    **What bounds the wait (PB #989, PQ #1107).** While PrismaBuild's
    landing record covers every pending span, the wait follows the record
    (:func:`landing_verdict`): it continues while the span's mover is queued
    or copying, or while its range is unpublished and the tier loop is
    alive, and it refuses at once on a ``terminal-no-receipt`` range or a
    silent tier loop, naming the state. ``deadline`` does not apply then;
    a wait of a mover still coming is declared to PrismaBuild so its
    ``no_progress`` rung does not count it as quiet. Where no landing record
    covers the pending spans (an older generation, a produced-output map),
    ``deadline`` bounds the wait exactly as before, and the refusal says so.
    The bounded clock runs only while nothing is published to wait on.

    ``wanted`` is ``[(declared path, start, end, declared size), ...]`` --
    every span one read is about to need, across every shard it touches,
    each with the size the caller's own header parse already produced.
    Returns the verdict the read should expect: ``RANGE_HIT`` when every
    span is covered, or the first non-covered outcome otherwise.

    **One deadline for the whole call, not one per span.** A read touching
    four cold shards waits the bound once; per-span budgets would multiply
    it, and choosing an order and a share per span would be a scheduler --
    which is PrismaBuild's job, not this reader's. Nothing here moves
    anything or asks for anything to be moved: it re-reads the map PB's
    movers publish into, and that map is the only thing that changes.

    Stops early, without waiting, on anything that is neither a covered
    span nor a mid-flight miss: ``RANGE_UNDECLARED`` (PB's sealed readset
    never named these bytes, so no mover will ever produce them) and
    ``RANGE_REFUSED`` (an entry covers the span and failed a hard check --
    wrong size, non-regular, permission or integrity, evidence in hand which
    re-asking cannot improve). A covering entry whose staged file is merely
    missing reports ``RANGE_UNCOVERED`` when the span is declared (PQ #903)
    and is waited on like any other mid-flight miss. Waiting on either
    terminal verdict would be the same conflation this exists to fix,
    pointed the other way.

    Cheap to poll: an uncovered span returns before ``staged_range``
    stats anything, and ``ResidencyResolver._read_map`` is identity-gated,
    so a poll that finds the map unchanged costs one ``lstat``.

    ``published(resolver, declared, entry) -> bool`` extends "landed" to the
    entry's proof. A mover writes its fragment, which puts the row in the
    map, and then the sidecar a lease needs; a covered span whose sidecar is
    not there yet is still landing, and is waited on exactly like an
    uncovered one (PQ #905). Asked once per staged entry per poll, never per
    tensor, and an entry that answered yes is not asked again.

    ``published_batch(resolver, [(declared, entry), ...]) -> True | None``
    asks the same question for every entry a poll still needs, at once
    (PQ #997: one PrismaBuild cover lookup instead of one per entry, which
    was 5.5% of Stage A R12's main thread). ``True`` proves them all;
    ``None`` means the batched answer cannot say which entry is missing,
    and that poll asks ``published`` one entry at a time instead. The
    proof lives for this call only, as before: nothing is carried to the
    next read, and the lease that follows re-verifies every key.

    ``cancel`` is the owning context's ``threading.Event`` (PQ #907), or
    ``None`` for the historical bounded wait. A set event aborts the wait
    by raising ``CancelledError``: a cancelled wait never resolves as a
    verdict and never falls through to payload reading -- the caller owns
    no bytes it did not ask to keep reading. ``Event.wait`` wakes the
    sleep promptly.
    """
    started = time.monotonic()
    bound = max(0.0, deadline - started)
    token = object()
    try:
        verdict, polls, pending, detail = _await_loop(
            resolver, list(wanted), published=published, cancel=cancel,
            published_batch=published_batch, token=id(token),
            started=started, bound=bound)
    finally:
        _staged_wait(id(token), None, ())
    declared = pending[0][0] if pending else wanted[0][0]
    if polls:
        resolver.record_range_wait(
            declared, polls=polls,
            seconds=time.monotonic() - started, served=verdict == RANGE_HIT)
    note = getattr(resolver, "record_range_refusal", None)
    if detail and callable(note):
        note(declared, detail)
    return verdict


def _await_loop(resolver, pending, *, published, cancel, published_batch,
                token, started, bound):
    """The poll loop of :func:`await_staged_spans`.

    Returns ``(verdict, polls, pending, refusal detail or None)``.
    """
    proven = set()
    polls = 0
    verdict = RANGE_HIT
    detail = None
    absent_since = started
    waiting_since = None
    # The last line printed: its (kind, first clause), when, and the
    # expectation it named. A line repeats when the clause changes, when
    # that expectation passes, or after STAGED_WAIT_REPORT_S (PQ #1167).
    said = None
    said_unix = 0.0
    said_expected = None
    while pending:
        if cancel is not None and cancel.is_set():
            raise CancelledError("staged-range wait cancelled")
        still = []
        unproven = set()
        asked = {}
        hits = []
        for row in pending:
            declared, start, end, size = row
            entry, outcome = resolver.staged_range_outcome(
                declared, start, end, declared_size=size)
            if outcome == RANGE_HIT:
                if published is None:
                    continue
                key = (declared, entry["offset"], entry["bytes"])
                if key in proven:
                    continue
                if published_batch is not None:
                    asked.setdefault(key, (declared, entry))
                    hits.append((row, key))
                    continue
                if key not in unproven and published(resolver, declared, entry):
                    proven.add(key)
                    continue
                unproven.add(key)
                still.append(row)
                continue
            if outcome != RANGE_UNCOVERED:
                verdict = outcome
                still = []
                break
            still.append(row)
        else:
            if asked:
                if published_batch(resolver, list(asked.values())) is True:
                    proven.update(asked)
                else:
                    for key, (declared, entry) in asked.items():
                        if published(resolver, declared, entry):
                            proven.add(key)
                still.extend(row for row, key in hits if key not in proven)
            pending = still
            if not pending:
                break
            now = time.monotonic()
            kind, why, movers, priced = _landing_verdict(resolver, pending)
            if kind == "refuse":
                verdict = RANGE_UNCOVERED
                detail = f"PrismaBuild's landing record refuses the wait: {why}"
                break
            if kind == "wait":
                absent_since = None
                if waiting_since is None:
                    waiting_since = time.time()
                _staged_wait(token, waiting_since, movers)
                pause = STAGED_RANGE_POLL_S
                line = (f"following PrismaBuild's landing record: {why}"
                        + _priced_text(priced))
            else:
                waiting_since = None
                _staged_wait(token, None, ())
                if absent_since is None:
                    absent_since = now
                remaining = absent_since + bound - now
                if remaining <= 0:
                    verdict = RANGE_UNCOVERED
                    # With no poll there was no wait to describe: a zero
                    # bound refuses on the first miss and records nothing,
                    # as it always has.
                    detail = None if not polls else (
                        f"no landing record covers the wait ({why}); the "
                        f"bounded wait of {bound:g} s applies "
                        f"({STAGED_RANGE_WAIT_ENV}) and ran out")
                    break
                pause = min(STAGED_RANGE_POLL_S, remaining)
                line = f"{why}; bounded wait of {bound:g} s, {remaining:.0f} s left"
            now_unix = time.time()
            clause = (kind, why.split(",")[0])
            if (clause != said or now_unix - said_unix >= STAGED_WAIT_REPORT_S
                    or (said_expected is not None
                        and said_unix <= said_expected < now_unix)):
                said, said_unix = clause, now_unix
                said_expected = priced["expected_unix"] if priced else None
                print("[residency] staged-range wait"
                      + (f", {now - started:.0f} s so far" if polls else "")
                      + f": {line}", flush=True)
            if cancel is None:
                time.sleep(pause)
            else:
                cancel.wait(pause)
                if cancel.is_set():
                    raise CancelledError("staged-range wait cancelled")
            polls += 1
            continue
        break
    return verdict, polls, pending, detail


def staged_range_wait_s() -> float:
    """The configured wait bound in seconds: finite and not negative.

    ``math.isfinite`` rather than a ``>= 0`` test alone, because the one
    property this design rests on is that the wait *ends*: ``float("inf")``
    is neither negative nor NaN, and a deadline of ``started + inf`` never
    expires. An unbounded wait is not a longer bound, it is no bound.
    """
    return staged_range_wait_from_env(os.environ)


def staged_range_wait_from_env(env) -> float:
    """:func:`staged_range_wait_s` read from ``env`` rather than the process.

    The joint dispatcher reads a campaign spec's ``env`` block with the same
    rules before it submits, so the wait it compares with the row's progress
    grace is the wait the reader inside the container will use.
    """
    raw = env.get(STAGED_RANGE_WAIT_ENV)
    if raw is None or not str(raw).strip():
        return STAGED_RANGE_WAIT_S
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise ValueError(
            f"{STAGED_RANGE_WAIT_ENV} must be a number of seconds, "
            f"not {raw!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{STAGED_RANGE_WAIT_ENV} must be a finite number of seconds "
            f">= 0, not {raw!r}")
    return value


try:
    # safetensors' own dtype table, so the reader reads the format's spelling
    # rather than asserting one of its own. Without it no tensor is served
    # from the stage; a hand-rolled table would drift silently.
    from safetensors.torch import _TYPES as _SAFETENSORS_DTYPES
except ImportError:  # pragma: no cover - a safetensors that moved the table
    _SAFETENSORS_DTYPES = None

# The same bound ``layer_streaming._advise_consumed_safetensors_pages`` puts on
# the same structure.
MAX_HEADER_BYTES = 100_000_000


#: The environment's override for the number of concurrent reads one staged
#: span is split into. ``1`` restores the single-stream read this reader
#: shipped with; unset takes the mount's own ``nconnect``.
STREAMS_ENV = "PRISMAQUANT_STAGED_READ_STREAMS"

#: Where the kernel publishes the mount table this reader reads its two numbers
#: from. Named so a test can point it at a table of its own.
MOUNTS_PATH = "/proc/self/mounts"

#: ``/proc/self/mounts`` escapes space, tab, newline and backslash in a mount
#: point as a backslash and three octal digits. Decoded in one pass, so a mount
#: point holding a literal backslash cannot be decoded twice.
_OCTAL = re.compile(r"\\([0-7]{3})")

_READ_SHAPE_LOCK = threading.Lock()
_READ_SHAPE_CACHE: dict[str, tuple[int, int] | None] = {}
_CHUNK_POOL_LOCK = threading.Lock()
_CHUNK_POOL: ThreadPoolExecutor | None = None
_CHUNK_POOL_SIZE = 0


def _mount_options(path: str) -> tuple[str, str, str] | None:
    """``(mount point, fstype, options)`` of the mount ``path`` is on.

    The longest mount point that prefixes the path wins, and on a tie the one
    later in the table does, because that is how the kernel resolves it: mounts
    stack, and the last one on a point is the one a read reaches. The tie is
    not hypothetical -- ``/stage/prewarm`` is an autofs trigger with the NFS
    mount on top of it, and reading the autofs row's options instead of the
    NFS row's is a mount with no ``nconnect``, which is a reader that quietly
    stays serial. ``/proc/self/mounts`` escapes space, tab, newline and
    backslash in the mount point as octal, so they are decoded before the
    comparison rather than compared raw.
    """
    target = os.path.abspath(path)
    best: tuple[int, str, str, str] | None = None
    try:
        with open(MOUNTS_PATH) as handle:
            rows = handle.read().splitlines()
    except OSError:
        return None
    for row in rows:
        fields = row.split(" ")
        if len(fields) < 4:
            continue
        point = _OCTAL.sub(lambda m: chr(int(m.group(1), 8)), fields[1])
        if target == point or target.startswith(point.rstrip("/") + "/"):
            if best is None or len(point) >= best[0]:
                best = (len(point), point, fields[2], fields[3])
    if best is None:
        return None
    return best[1], best[2], best[3]


def _read_shape_for(mount: tuple[str, str, str]) -> tuple[int, int] | None:
    """``(streams, chunk_bytes)`` for reads on ``mount``, or None to stay serial.

    Both numbers are the mount's, not ours (principle 2). ``nconnect`` is how
    many transports the NFS client actually holds open to the server, so it is
    the ceiling on how many of this span's reads can be in flight at once;
    ``rsize`` is the size of the read the client issues, so it is the unit a
    span is cut on and a cut anywhere else only splits one wire read in two.

    ``None`` -- a mount that is not NFS, or an NFS mount publishing neither --
    means there is no explicit to read, and a reader with no explicit reads the
    way it read before. ``STREAMS_ENV`` overrides the stream count for an A/B;
    ``1`` is the single-stream read and is what the before arm sets.
    """
    _, fstype, raw = mount
    if not fstype.startswith("nfs"):
        return None
    streams = chunk = 0
    for option in raw.split(","):
        name, _, value = option.partition("=")
        if name == "nconnect" and value.isdigit():
            streams = int(value)
        elif name == "rsize" and value.isdigit():
            chunk = int(value)
    override = str(os.environ.get(STREAMS_ENV, "")).strip()
    if override:
        try:
            streams = max(1, int(override))
        except ValueError:
            pass
    if streams <= 1 or chunk <= 0:
        return None
    return streams, chunk


def _read_shape(path: str) -> tuple[int, int] | None:
    """The read shape for ``path``, deriving it once per mount point.

    The mount table is read either way -- it is what says which mount the path
    is on -- and the cache saves the option parse and the environment read, not
    the table read.
    """
    mount = _mount_options(path)
    if mount is None:
        return None
    with _READ_SHAPE_LOCK:
        if mount[0] in _READ_SHAPE_CACHE:
            return _READ_SHAPE_CACHE[mount[0]]
    shape = _read_shape_for(mount)
    with _READ_SHAPE_LOCK:
        _READ_SHAPE_CACHE[mount[0]] = shape
    return shape


def reset_read_shape_cache_for_tests() -> None:
    with _READ_SHAPE_LOCK:
        _READ_SHAPE_CACHE.clear()


def _chunk_pool(streams: int) -> ThreadPoolExecutor:
    """One pool for the whole process, sized by the mount's transport count.

    Shared on purpose. ``layer_streaming.read_prefix_tensors`` already reads a
    layer's tensors on several threads, so a per-reader pool would multiply
    (gather threads x chunk threads) into more in-flight reads than the client
    has transports to carry. The chunk tasks submit nothing themselves, so a
    gather thread waiting on this pool cannot deadlock against it.
    """
    global _CHUNK_POOL, _CHUNK_POOL_SIZE
    with _CHUNK_POOL_LOCK:
        if _CHUNK_POOL is None or _CHUNK_POOL_SIZE < streams:
            if _CHUNK_POOL is not None:
                _CHUNK_POOL.shutdown(wait=False)
            _CHUNK_POOL = ThreadPoolExecutor(
                max_workers=streams, thread_name_prefix="pq-staged-read")
            _CHUNK_POOL_SIZE = streams
        return _CHUNK_POOL


def _cuts(offset: int, count: int, chunk: int) -> list[tuple[int, int]]:
    """``[(offset, length), ...]`` covering ``[offset, offset + count)`` exactly.

    Aligned to ``chunk`` in the staged file's own offset space, so every read
    but the first and last is one whole ``rsize`` request. The pieces are
    disjoint and their lengths sum to ``count``; nothing is read twice and no
    byte is left out.
    """
    cuts = []
    position = offset
    end = offset + count
    while position < end:
        boundary = ((position // chunk) + 1) * chunk
        stop = min(boundary, end)
        cuts.append((position, stop - position))
        position = stop
    return cuts


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_shard_header(path: str) -> tuple[dict, int, int]:
    """The shard's header, its payload base, and the declared file's length.

    The 8-byte little-endian length prefix and JSON header ``safetensors``
    itself reads, under the bounds
    ``layer_streaming._advise_consumed_safetensors_pages`` applies to the same
    bytes. The parse is repeated there rather than shared because that function
    is on the install hot path and this change has to stay expressible as
    minimal source hunks for the joint run's closed source transition.
    """
    handle = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("source shard is not a regular file")
        raw = os.pread(handle, 8, 0)
        if len(raw) != 8:
            raise ValueError("source shard has no safetensors header length")
        header_bytes = int.from_bytes(raw, "little")
        if not 0 < header_bytes <= min(MAX_HEADER_BYTES, info.st_size - 8):
            raise ValueError("source shard header length is out of range")
        blob = os.pread(handle, header_bytes, 8)
        if len(blob) != header_bytes:
            raise ValueError("source shard header is shorter than it declares")
        header = json.loads(blob)
        if type(header) is not dict:
            raise ValueError("source shard header is not an object")
        return header, 8 + header_bytes, info.st_size
    finally:
        os.close(handle)


def _pread_into(fd: int, view: memoryview, offset: int) -> None:
    """Fill ``view`` from ``offset``, in as many reads as the kernel needs.

    A single ``pread`` is capped near 2 GiB on Linux and an NFS read can be
    short for its own reasons, so the loop is the contract rather than an
    optimization.
    """
    count = len(view)
    done = 0
    while done < count:
        moved = os.preadv(fd, [view[done:]], offset + done)
        if moved <= 0:
            raise OSError(errno.EIO, "staged range ended before the tensor's bytes")
        done += moved


def _read_span(fd: int, count: int, offset: int,
               shape: tuple[int, int] | None) -> bytearray:
    """``count`` bytes at ``offset``, on as many streams as the mount holds.

    One buffer, cut into disjoint pieces that are read at the same time and
    written straight into their own slice of it, so the bytes are the bytes a
    single sequential read would have produced, piece by piece and offset by
    offset. The buffer is writable, which is what ``torch.frombuffer`` wants,
    and it is the tensor's own storage afterwards.

    A span shorter than one read per stream is read on one stream: splitting it
    would hand some streams nothing and cost a round trip to find out. Every
    piece is waited for before the result is looked at, including on a failure,
    so no thread is still writing into the buffer -- or reading the descriptor
    the caller is about to close -- when this returns.
    """
    buffer = bytearray(count)
    view = memoryview(buffer)
    if shape is None or count < shape[0] * shape[1]:
        _pread_into(fd, view, offset)
        return buffer
    streams, chunk = shape
    cuts = _cuts(offset, count, chunk)
    pool = _chunk_pool(streams)
    futures = [pool.submit(_pread_into, fd, view[at - offset:at - offset + size], at)
               for at, size in cuts]
    failure = None
    for future in futures:
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - re-raised after the join
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return buffer


class _StrictSliceProxy:
    """Header-only slice metadata; payload materializes via the staged reader.

    ``get_shape``/``get_dtype`` are served from the bounded header parse
    (metadata, allowed). Any indexing materializes the full tensor through
    the reader's staged path — the same fences, the same refusal, no pool
    fallback — then applies the index. This is what keeps the shape
    estimators working under policy without a payload exemption by naming.
    """

    def __init__(self, reader, name, dtype, shape):
        self._reader = reader
        self._name = name
        self._dtype = dtype
        self._shape = tuple(shape)

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return self._dtype

    def __getitem__(self, index):
        tensor = self._reader._staged_tensor(self._name)
        if tensor is None:
            raise refuse_pool_bulk_read(
                self._reader._declared, "pool-fallback")
        return tensor[index]


class StagedShardReader:
    """A ``safe_open`` handle whose payload comes off the stage when it can.

    Drop-in at the ``_source_safe_open`` seam, including through
    ``source_authentication.safe_open(opener, path, **kwargs)``, which calls the
    opener with ``/proc/self/fd/<fd>`` rather than the declared path. The
    declared path travels in the closure ``staged_shard_opener`` builds; the
    handed path is what is opened and what the header is read from, and the two
    are the same file.

    Under the active allowed-tier policy no pool ``safe_open`` handle is
    constructed at all — no pool mmap merely for header. Keys, metadata,
    shapes and dtypes come from the existing bounded header reader
    (``_read_shard_header``); payload comes from staged ranges or refuses.

    Concurrency is threads, never fork: payload and lifecycle operations
    (``get_tensor``, ``get_slice``, ``__enter__``, ``__exit__``) reject
    inherited handle operations in a forked child, so a child cannot
    drive reads through state the parent's release retires. ``keys``/
    ``metadata`` stay available: stateless bounded-header metadata with
    no retained descriptor. Raw descriptor escape past these operations
    is unsupported (not policed here): a forked child must not perform
    I/O on any inherited descriptor or mapping and should ``_exit``
    without it.
    """

    def __init__(self, pool_open, path, declared, resolver, kwargs):
        self._declared = declared
        self._path = os.fspath(path)
        self._resolver = resolver
        self._device = kwargs.get("device")
        self._owner_pid = os.getpid()
        # Captured at construction (opener time): the entrypoints activate
        # the process-global policy before any read, so every reader built
        # afterwards — on any thread — sees the same verdict.
        self._strict = policy_is_active()
        # The allowed set beside the verdict: tier checks read the same
        # snapshot the reader was built under, not a later global.
        self._allowed = active_policy() if self._strict else None
        if self._strict:
            self._handle = None
        else:
            # First, so a ``safe_open`` that rejects the ``device`` kwarg raises
            # where it raises today and the callers' retry without it still works.
            self._handle = pool_open(path, **kwargs)
        self._header = None
        self._base = 0
        self._declared_size = 0
        self._parsed = False
        self._shape: tuple[int, int] | None = None
        self._bound: list[tuple] = []

    def _require_owner(self, operation: str) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                f"StagedShardReader.{operation} from a forked child is "
                "unsupported: inherited bound handles must not be driven "
                "past the holder's lifecycle. Use threads, which share the "
                "holder pid, or fork with no live reader.")

    # -- the handle interface --------------------------------------------

    def __enter__(self):
        self._require_owner("__enter__")
        if self._handle is not None:
            self._handle.__enter__()
        return self

    def __exit__(self, *args):
        self._require_owner("__exit__")
        # The window owns every bound descriptor: close each through it,
        # then release its exact ref. Release-before-close is forbidden,
        # and a forked child never releases (the window's pid guard makes
        # that explicit).
        failure = None
        while self._bound:
            row = self._bound.pop()
            window = row[6] if len(row) > 6 else None
            if window is not None:
                try:
                    window.close_fd(row[2])
                except OSError as exc:
                    if failure is None:
                        failure = exc
                try:
                    window.__exit__(None, None, None)
                except LeaseRefused as exc:
                    if failure is None:
                        failure = exc
            else:
                try:
                    os.close(row[2])
                except OSError as exc:
                    if failure is None:
                        failure = exc
        if self._handle is not None:
            try:
                result = self._handle.__exit__(*args)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                result = False
            if failure is not None:
                raise failure
            return result
        if failure is not None:
            raise failure
        return False

    def keys(self):
        if not self._strict:
            return self._handle.keys()
        self._parse()
        if self._header is None:
            raise refuse_pool_bulk_read(self._declared, "header-unreadable")
        return [name for name in self._header if name != "__metadata__"]

    def metadata(self):
        if not self._strict:
            return self._handle.metadata()
        self._parse()
        if self._header is None:
            raise refuse_pool_bulk_read(self._declared, "header-unreadable")
        meta = self._header.get("__metadata__")
        return dict(meta) if type(meta) is dict else {}

    def get_slice(self, name):
        """Shape/dtype metadata from the header; payload via the staged path.

        The returned proxy serves ``get_shape``/``get_dtype`` without
        touching payload bytes. Indexing it materializes through the
        reader's staged tensor path — same fences, same refusal. No pool
        fallback either way. Inactive policy delegates to the pool handle
        as before.
        """
        if not self._strict:
            return self._handle.get_slice(name)
        self._require_owner("get_slice")
        span = self._span(name)
        if span is None:
            raise refuse_pool_bulk_read(self._declared, "span-not-bound")
        _, _, dtype, shape = span
        return _StrictSliceProxy(self, name, dtype, shape)

    def get_tensor(self, name):
        """The tensor, off the stage when a staged range covers its whole span.

        ``bytes_from_pool`` counts what this reader falls back to, not every
        shard byte the run reads: a shard the map never names is opened by
        ``safe_open`` itself and no reader sees it. The stage-side count has no
        such gap, so read the two as "what the stage served" and "what this
        reader could not get from it", not as a partition of the run.

        Under the active allowed-tier policy there is no pool fallback:
        a span no staged range covers, or a fence the staged copy fails,
        raises ``TierPolicyRefused`` before a pool payload byte is read.
        Zero-size tensors carry no payload bytes and are built locally.
        """
        if self._strict:
            self._require_owner("get_tensor")
        served = self._staged_tensor(name)
        if served is not None:
            return served
        if self._strict:
            raise refuse_pool_bulk_read(self._declared, "pool-fallback")
        tensor = self._handle.get_tensor(name)
        if self._resolver is not None:
            self._resolver.record_pool_read(self._declared, tensor.nbytes)
        return tensor

    # -- the stage -------------------------------------------------------

    def _parse(self) -> None:
        """Read the header once, on the first payload read, never before.

        An open that only asks for ``keys()``, ``metadata()`` or a slice pays
        nothing for this reader beyond the object.
        """
        if self._parsed:
            return
        self._parsed = True
        if _SAFETENSORS_DTYPES is None:
            if self._resolver is not None:
                self._resolver.record_fallback(
                    self._declared,
                    "safetensors publishes no dtype table this reader can read")
            return
        try:
            self._header, self._base, self._declared_size = _read_shard_header(self._path)
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            self._header = None
            if self._resolver is not None:
                self._resolver.record_fallback(
                    self._declared, f"source shard header is unreadable: {error}")

    def _span(self, name):
        """``(start, end, dtype, shape)`` in the declared file, or None."""
        self._parse()
        if self._header is None:
            return None
        row = self._header.get(name)
        if type(row) is not dict:
            return None
        dtype = _SAFETENSORS_DTYPES.get(row.get("dtype"))
        offsets, shape = row.get("data_offsets"), row.get("shape")
        if (dtype is None or type(offsets) is not list or len(offsets) != 2
                or type(shape) is not list
                or any(type(dim) is not int or isinstance(dim, bool) or dim < 0
                       for dim in shape)):
            return None
        begin, end = offsets
        if (type(begin) is not int or type(end) is not int
                or isinstance(begin, bool) or isinstance(end, bool)
                or not 0 <= begin <= end <= self._declared_size - self._base):
            return None
        count = 1
        for dim in shape:
            count *= dim
        if count * dtype.itemsize != end - begin:
            return None
        return self._base + begin, self._base + end, dtype, tuple(shape)

    def _range_for(self, start, end):
        """The bound staged range covering ``[start, end)``, or None.

        Bound ranges are held open for the reader's life, so a layer's several
        hundred tensors cost one resolver lookup and one open per staged range
        rather than one per tensor.

        Under the active allowed-tier policy the range is pinned for the
        reader's life: one lifetime window per staged entry (never per
        tensor), payload read through the held descriptor, the SDK's own
        serving record registered at the successful actual open, and the
        exact ref released after the bound descriptors close. A RAM leg
        needs RAM-mover covers the composed map does not carry, so it
        refuses fast and the SSD copy acquires honestly with its own
        material and lifetime. A typed retiring refusal may select another
        checked overlap through its own lease; other refusals propagate,
        never becoming pool reads. Inactive policy keeps the legacy
        stage-only open order.
        """
        for row in self._bound:
            if row[0] <= start and end <= row[1]:
                return row
        strict = self._strict
        if self._resolver is None:
            if strict:
                raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
            return None
        entry, outcome = self._resolver.staged_range_outcome(
            self._declared, start, end, declared_size=self._declared_size)
        # Nothing waits here. This runs on a worker of the shared, bounded
        # ``layer_streaming._LAYER_READ_POOL``, and a worker sleeping on a
        # cold future range is a worker the current layer's already-staged
        # reads queue behind. Readiness is decided one level up, before
        # these slots are occupied: ``layer_streaming._await_layer_readset``
        # (PQ #874). By the time a chunk asks, the answer is final.
        if entry is None:
            if strict:
                if outcome == RANGE_UNCOVERED:
                    # Declared and simply not landed yet: the one transient
                    # cause a demand-side retry may match on. Undeclared
                    # spans and failed covering entries keep the generic
                    # refusal below: nothing about them is on its way.
                    raise StagedRangeNotLanded(self._declared, start, end)
                raise refuse_pool_bulk_read(self._declared, "readset-not-staged")
            return None
        if not strict:
            try:
                fd = os.open(entry["stage_path"], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except OSError as error:
                self._resolver.record_fallback(
                    self._declared, f"staged range is unreadable: {error.strerror}")
                return None
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("staged range is not a regular file")
                if info.st_size != entry["bytes"]:
                    raise ValueError("staged range size differs from the map")
            except (OSError, ValueError) as error:
                os.close(fd)
                self._resolver.record_fallback(self._declared, f"staged range {error}")
                return None
            if not self._bound:
                # Every range of one declared shard is staged under the same root,
                # so the mount's read shape is read once per reader, not per tensor.
                self._shape = _read_shape(entry["stage_path"])
            row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
                   _signature(info), entry, "stage")
            self._bound.append(row)
            return row
        from .staged_lease import LeaseRefused, acquire_entry_window
        alternatives = None
        hard_refusal = None
        while True:
            try:
                window, key = acquire_entry_window(
                    self._resolver, self._declared, entry)
                window.__enter__()
                break
            except LeaseRefused as refusal:
                # A retired phase may still have a physically valid file
                # while an older reader drains. Its closed generation cannot
                # hide another overlapping entry that admits its OWN pin.
                # Unknown/identity/integrity refusals never take this path.
                if refusal.kind != 'availability' or refusal.reason != 'retiring':
                    self._resolver.record_fallback(self._declared, str(refusal))
                    raise
                if alternatives is None:
                    candidates, hard_refusal = self._resolver.staged_range_alternatives(
                        self._declared, start, end,
                        rejected_offset=entry['offset'], declared_size=self._declared_size)
                    alternatives = iter(candidates)
                entry = next(alternatives, None)
                if entry is None:
                    if hard_refusal is not None:
                        refusal = LeaseRefused(hard_refusal, kind='integrity')
                    self._resolver.record_fallback(self._declared, str(refusal))
                    raise refusal
        try:
            fd, serving = window.open(key)
            info = os.fstat(fd)
            if info.st_size != entry["bytes"]:
                raise LeaseRefused("lease-open-size-changed", kind="integrity")
        except (OSError, LeaseRefused) as error:
            try:
                window.__exit__(None, None, None)
            except (LeaseRefused, RuntimeError):
                pass
            if isinstance(error, LeaseRefused):
                self._resolver.record_fallback(self._declared, str(error))
                raise
            reason = f"staged range is unreadable: {error.strerror}"
            self._resolver.record_fallback(self._declared, reason)
            raise refuse_pool_bulk_read(self._declared, reason)
        # The open fence passed on a held descriptor: the SDK's own
        # serving record is registered at this successful actual open.
        tier = window.serving_tier or "stage"
        self._resolver.record_serving_tier(
            self._declared, tier,
            pin_id=str(serving.get("pin_id") or ""),
            range_ref=str(serving.get("range_ref") or ""))
        if not self._bound:
            # Every range of one declared shard is staged under the same
            # root, so the mount's read shape is read once per reader, not
            # per tensor — from the path actually opened (the RAM copy for
            # a RAM window), never an assumed tier.
            self._shape = _read_shape(window.stage_path(key)
                                      or entry["stage_path"])
        row = (entry["offset"], entry["offset"] + entry["bytes"], fd,
               _signature(info), entry, tier, window)
        self._bound.append(row)
        return row

    def _drop(self, row) -> None:
        if row in self._bound:
            self._bound.remove(row)
        window = row[6] if len(row) > 6 else None
        if window is not None:
            # The window owns the descriptor: close through it, then
            # release its exact ref now (after the descriptor above)
            # rather than lending it to a later tensor. A release failure
            # is recorded, never silent; the caller's read error still
            # raises.
            try:
                window.close_fd(row[2])
            except OSError:
                pass
            try:
                window.__exit__(None, None, None)
            except LeaseRefused as exc:
                if self._resolver is not None:
                    self._resolver.record_fallback(self._declared, str(exc))
            return
        try:
            os.close(row[2])
        except OSError:
            pass

    def _staged_tensor(self, name):
        span = self._span(name)
        if span is None:
            if self._strict:
                raise refuse_pool_bulk_read(self._declared, "span-not-bound")
            return None
        start, end, dtype, shape = span
        if end == start:
            # No bytes to serve: an empty tensor is built locally rather
            # than read from any tier, pool included.
            if self._strict:
                tensor = torch.empty(shape, dtype=dtype)
                if self._device is not None:
                    tensor = tensor.to(self._device)
                return tensor
            return None
        row = self._range_for(start, end)
        if row is None:
            return None
        fd, signature, entry, tier = row[2], row[3], row[4], row[5]
        try:
            raw = _read_span(fd, end - start, start - entry["offset"], self._shape)
            if _signature(os.fstat(fd)) != signature:
                raise ValueError("changed during its content read")
        except (OSError, ValueError) as error:
            self._drop(row)
            reason = f"staged range {getattr(error, 'strerror', None) or error}"
            if self._resolver is not None:
                self._resolver.record_fallback(self._declared, reason)
            if self._strict:
                raise refuse_pool_bulk_read(self._declared, reason)
            return None
        tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)
        if self._device is not None:
            tensor = tensor.to(self._device)
        if self._resolver is not None:
            if tier == "ram":
                # A range read the ram tier served: ram bytes, not stage
                # bytes. ``range_hits`` stays the stage-range count, so the
                # two tiers never double-count the same bytes.
                self._resolver.record_ram_read(self._declared, end - start)
            else:
                self._resolver.record_stage_range_read(self._declared, end - start)
        return tensor


def staged_shard_opener(declared, pool_open):
    """The opener for ``declared``: ``pool_open`` itself unless the stage holds it.

    Handing the caller back its own opener is what keeps the unmapped path
    identical -- same callable, same arguments, no wrapper and no extra
    syscall. A file the map never names takes that path too, so wrapping is
    paid for only where it can pay off.

    Under the active allowed-tier policy there is no unwrapped path: a file
    the map never names (or no map at all) still opens through a reader
    whose header/keys/metadata come from the pool handle — bounded header
    bytes, allowed — but whose payload reads refuse with
    ``readset-not-staged`` instead of serving pool bytes.
    """
    resolver = residency_resolver()
    if resolver is None or not resolver.stages(declared):
        if policy_is_active():
            path = os.fspath(declared)

            def strict_opener(handed, **kwargs):
                return StagedShardReader(pool_open, handed, path, resolver, kwargs)

            return strict_opener
        return pool_open
    path = os.fspath(declared)

    def opener(handed, **kwargs):
        return StagedShardReader(pool_open, handed, path, resolver, kwargs)

    return opener
