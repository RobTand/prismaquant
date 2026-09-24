"""``PRISMAQUANT_DEV_MODE``: sealing is off unless a run asks for it (PQ #1147).

Rob, 2026-09-24: "All sealing should be disabled until further notice. That's
what dev mode should do but somehow we keep burning cycles on this
ill-conceived notion." The earlier opt-in switch (#771, #826, #833) did not
hold: every gate written after it sealed by default, and where it did fire it
archived a mismatched lineage and recomputed it. This module makes the ruling
structural.

The contract:

* Dev mode is ON unless ``PRISMAQUANT_DEV_MODE`` is exactly ``0``. Unset,
  empty, ``1`` and any other value are dev mode. ``0`` is certified mode, and
  certified mode behaves exactly as the gates did before #1147: every seal
  refuses with the same exception type and message.
* A run-gate seal compares a recorded source, implementation, plan, policy,
  budget, scope, session, image or generation identity with the running one.
  Every such seal goes through :func:`seal_check`. In dev mode it prints one
  ``[DEV-MODE]`` line naming both values and returns, and the caller continues
  with the stored data. It never archives and never recomputes.
* Dev mode computes no digest over existing data only to seal a run. A digest
  written beside new bytes, and the check that reads those bytes back against
  it, is integrity, not sealing, and stays in both modes: a missing unit, a
  corrupt file and bytes that do not match their own digest still refuse.
  ``tests/test_no_new_seals.py`` holds that line.
* A code change that alters stored numerics is the one case where reuse is
  wrong. Nothing detects it automatically. The ``[DEV-MODE]`` line names the
  recorded and running implementation, so a human decides, and the author of
  such a change starts the run fresh on purpose.
* A dev result never passes as a certified one: results carry the
  ``dev_uncertified`` stamp from :func:`dev_stamp`, and progress records carry
  it under the ``PRISMAQUANT_DEV_PROGRESS_STAMP=1`` opt-in (PR #828).
* The merged campaign checkpoint's canonical seal in
  ``load_measured_anchor_input`` (302.653 s on the real 7.2 GB input, action
  ``282c61140ba7``) is not recomputed in dev mode. Dev mode requires a full
  64-hex ``identity_sha256`` shape, names the digest in the ``[DEV-MODE]``
  line, and hands it to the per-unit envelope fences. The raw checkpoint file
  keeps its ``_bound`` SHA-256 check above the gate. (2026-09-20, #833.)

Nothing in this module decides which checks are seals; the gates stay in the
modules that own them, and each one hands its comparison to
:func:`seal_check`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from typing import Any

#: The one environment variable. Read at the gates, never cached, so a
#: subprocess or a container inherits it through the environment alone.
DEV_MODE_ENV = "PRISMAQUANT_DEV_MODE"

#: The one value that turns sealing back on (PQ #1147). Everything else,
#: unset included, is dev mode.
CERTIFIED_VALUE = "0"

#: What a dev-mode run passes, and records, for an identity input it would
#: compute only to compare it (PQ #1147). :func:`seal_check` names it in the
#: ``[DEV-MODE]`` line; certified mode never passes it, and a certified check
#: that meets it refuses, because it differs from every recorded identity.
NOT_COMPUTED = "not computed"

#: The stamp's fixed top-level marker. Always present in results.json under
#: dev mode; progress records carry it only when PRISMAQUANT_DEV_PROGRESS_STAMP
#: is exactly "1" (PR #828). Absent in certified mode.
DEV_UNCERTIFIED_KEY = "dev_uncertified"
DEV_MODE_KEY = "dev_mode"


def dev_mode_enabled(environ=None) -> bool:
    """Whether dev mode is ON: ``PRISMAQUANT_DEV_MODE`` is anything but ``0``.

    The one definition. Every gate reads the switch through this function;
    nothing else reads the variable (PQ #1147).
    """
    environ = os.environ if environ is None else environ
    return environ.get(DEV_MODE_ENV) != CERTIFIED_VALUE


def dev_stamp(producer_source_sha256: str | None = None, *, timestamped: bool = True) -> dict:
    """The unmistakable mark of a dev run.

    ``{"dev_uncertified": true, "dev_mode": {...}}`` at the top level of
    whatever record accepts it. ``dev_mode`` carries the executing package's
    ACTUAL tree digest when known -- a record of what ran, never a gate --
    and, only where equality is not compared across processes (results.json,
    progress records), a UTC timestamp. Callers that compare the stamp for
    equality across a resume (unit-checkpoint provenance) pass
    ``timestamped=False`` so a resumed run reproduces the identical record.
    """
    block: dict = {DEV_MODE_ENV: "1"}
    if producer_source_sha256 is not None:
        block["producer_source_sha256"] = str(producer_source_sha256)
    if timestamped:
        block["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {DEV_UNCERTIFIED_KEY: True, DEV_MODE_KEY: block}


def dev_warning(message: str) -> None:
    """The loud line every suspended gate prints instead of refusing.

    One stable ``[DEV-MODE]`` prefix so a dev run's suspended gates are as
    grep-able as its stamps: ``grep DEV-MODE`` finds every place a certified
    run would have stopped.
    """
    print(f"[DEV-MODE] {message}", flush=True)


#: Longest rendering of one side of a mismatch in a ``[DEV-MODE]`` line.
_SHOWN_CHARS = 160


def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= _SHOWN_CHARS:
        return text
    try:
        encoded = json.dumps(value, sort_keys=True, default=repr).encode()
    except (TypeError, ValueError):
        encoded = text.encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{text[:_SHOWN_CHARS]}... ({len(text)} chars, sha256 {digest}...)"


def _first_difference(expected: Any, actual: Any, path: str = "") -> tuple[str, Any, Any]:
    """The first field two identities disagree on, for the stamp line."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            left, right = expected.get(key), actual.get(key)
            if key not in expected or key not in actual or left != right:
                return _first_difference(left, right, f"{path}.{key}" if path else str(key))
    return path, expected, actual


def _refusal(refusal, kind: str, expected: Any, actual: Any, where: str) -> BaseException:
    message = f"{where}: {kind} differs: expected {expected!r}, actual {actual!r}"
    if refusal is None:
        return RuntimeError(message)
    if isinstance(refusal, BaseException):
        return refusal
    if isinstance(refusal, type) and issubclass(refusal, BaseException):
        return refusal(message)
    return refusal()


def seal_check(kind: str, expected: Any, actual: Any, *, where: str,
               refusal: BaseException | type | Callable[[], BaseException] | None = None,
               same: bool | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Compare a recorded identity with the running one: the one seal helper.

    Returns ``True`` when the two agree. On a mismatch, certified mode
    (``PRISMAQUANT_DEV_MODE=0``) raises ``refusal`` -- the exception the site
    raised before #1147, so its type and message are unchanged -- and dev mode
    prints one ``[DEV-MODE]`` line naming both values and returns ``False``.
    The caller then continues with the stored data.

    ``refusal`` is an exception instance, an exception class (called with a
    generated message) or a zero-argument callable that returns one. ``same``
    replaces ``expected == actual`` for a site whose comparison is not plain
    equality (canonical bytes, a set, a combined condition).

    Only seals go through here. A check of bytes against the digest written
    with them is integrity and raises in both modes.
    """
    agree = (expected == actual) if same is None else bool(same)
    if agree:
        return True
    if not dev_mode_enabled(environ):
        raise _refusal(refusal, kind, expected, actual, where)
    field, left, right = _first_difference(expected, actual)
    named = f" at {field}" if field else ""
    if isinstance(right, str) and right == NOT_COMPUTED:
        dev_warning(
            f"seal {kind}{named} not computed ({where}): recorded {_shown(left)}; "
            "sealing is off (PQ #1147), continuing with the stored data")
        return False
    dev_warning(
        f"seal {kind} differs{named} ({where}): expected {_shown(left)}, "
        f"actual {_shown(right)}; sealing is off (PQ #1147), continuing with "
        "the stored data")
    return False
