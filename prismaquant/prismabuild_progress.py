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
