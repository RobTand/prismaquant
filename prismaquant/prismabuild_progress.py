"""Report committed pricing work to PrismaBuild's stall watchdog.

PrismaBuild bounds an action either by how long it has run or by how long it
has gone without committing work, and which of the two it uses is a property
of the sealed request (``prismabuild.action_progress_policy.v1``, PB #480).
A campaign row that declares the progress contract is not killed for taking a
long time; it is killed for stopping.  This is the half of that the row owes:
the report that says it has not stopped.

Written against the *wire format* rather than against ``prismabuild.core``.
The pricing rows execute inside a pinned producer image, and making an
already-qualified image depend on PrismaBuild being importable inside it
would put a serving-side dependency in front of every campaign.  The record
is four fields and a token; the schema string below is the contract.

Report only what is **durable**.  ``tessera_campaign`` calls this after the
identity-bound journal shard is on disk, never on entering a batch: a counter
that ran ahead of the work it stands for would keep a broken row alive, which
is the one thing the watchdog exists not to do.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


#: The record schema PrismaBuild's ``ProgressWatch`` accepts.  A record with
#: any other value here is not advancement, so this string is a contract with
#: the deployed fleet generation and not a label.
RECORD_SCHEMA = "prismabuild.action_progress.v1"

PATH_ENV = "PRISMABUILD_ACTION_PROGRESS_PATH"
TOKEN_ENV = "PRISMABUILD_ACTION_PROGRESS_TOKEN"


def report(phase: str, units_completed: int, *, unit: str = "anchors") -> bool:
    """Say that ``units_completed`` units are durably committed in ``phase``.

    ``units_completed`` must be monotone across the whole run -- a resumed row
    continues from what its journal already holds rather than restarting at
    zero -- and ``phase`` must be one the submitted row declared.  Neither is
    checked here: the worker refuses what it cannot accept and says why on the
    receipt, and a second opinion computed from a stale copy of the policy
    would only disagree with it.

    Returns whether a record was written.  ``False`` when the row was not
    admitted under the contract, which is the ordinary case for every
    campaign that does not declare phases, so callers may call unconditionally.
    Never raises: a row must not fail because it could not describe itself.
    """

    destination = os.environ.get(PATH_ENV) or ""
    token = os.environ.get(TOKEN_ENV) or ""
    if not destination or not token:
        return False
    record = {
        "schema": RECORD_SCHEMA,
        "token": token,
        "phase": str(phase),
        "units_completed": int(units_completed),
        "unit": str(unit),
        "reported_unix": time.time(),
    }
    path = Path(destination)
    try:
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # A box that cannot write to its own queue directory reads as a stall,
        # which is the honest verdict rather than a reason to stop pricing.
        return False
    return True


#: What a consumer blocked on its own staged range writes beside its progress
#: report (PrismaBuild #989).  PrismaBuild's ``no_progress`` rung reads it,
#: checks every named mover against the consumer's own plan, and leaves the
#: blocked time out of the quiet only while a named mover is still coming.
#: The wire format of ``prismabuild.progress.declare_staged_wait``, written
#: here for the same reason ``report`` is: the rows run where PrismaBuild may
#: not be importable.
STAGED_WAIT_SCHEMA = "prismabuild.staged_wait.v1"
STAGED_WAIT_SUFFIX = ".staged-wait"


def declare_staged_wait(movers, *, since_unix: float) -> bool:
    """Say that this row is blocked until one of ``movers`` lands its range.

    Replaces any earlier record, so a caller with several waits passes their
    union.  Returns whether a record was written; ``False`` without a
    progress channel, which leaves the wait counted as quiet, as before.
    Never raises.
    """

    destination = os.environ.get(PATH_ENV) or ""
    token = os.environ.get(TOKEN_ENV) or ""
    names = sorted({str(mover) for mover in movers})
    if not destination or not token or not names:
        return False
    record = {"schema": STAGED_WAIT_SCHEMA, "token": token,
              "since_unix": float(since_unix), "movers": names}
    path = Path(destination + STAGED_WAIT_SUFFIX)
    try:
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        return False
    return True


def clear_staged_wait() -> bool:
    """End the declared wait.  Never raises."""

    destination = os.environ.get(PATH_ENV) or ""
    if not destination:
        return False
    try:
        os.unlink(destination + STAGED_WAIT_SUFFIX)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


#: What an owner blocked on its own produced-output exports writes beside its
#: progress report (PrismaBuild #1035): at an ordering barrier, or with its
#: local window full (``ProducedOutputSpool``). PrismaBuild's ``no_progress``
#: rung reads it after the staged-wait record, checks every named export
#: against the owner's own sealed exports, and leaves the blocked time out of
#: the quiet only while one of them shows progress. A record of its own, not a
#: field of the staged-wait one: an owner can wait on both at once. The wire
#: format of ``prismabuild.progress.declare_export_wait``, written here for
#: the same reason ``report`` is.
EXPORT_WAIT_SCHEMA = "prismabuild.export_wait.v1"
EXPORT_WAIT_SUFFIX = ".export-wait"


def declare_export_wait(exports, *, since_unix: float) -> bool:
    """Say that this action is blocked until one of its own ``exports`` lands.

    ``exports`` are export action keys (``ProducedSpool.submit_group``'s
    ``export_key``). Replaces any earlier record, so a caller with several
    waits passes their union. Returns whether a record was written; ``False``
    without a progress channel, which leaves the wait counted as quiet, as
    before. Never raises.
    """

    destination = os.environ.get(PATH_ENV) or ""
    token = os.environ.get(TOKEN_ENV) or ""
    names = sorted({str(export) for export in exports})
    if not destination or not token or not names:
        return False
    record = {"schema": EXPORT_WAIT_SCHEMA, "token": token,
              "since_unix": float(since_unix), "exports": names}
    path = Path(destination + EXPORT_WAIT_SUFFIX)
    try:
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        return False
    return True


def clear_export_wait() -> bool:
    """End the declared export wait.  Never raises."""

    destination = os.environ.get(PATH_ENV) or ""
    if not destination:
        return False
    try:
        os.unlink(destination + EXPORT_WAIT_SUFFIX)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True
