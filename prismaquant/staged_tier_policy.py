"""Strict allowed-tier policy for GPU-consumed bulk inputs (PQ:readers lane).

The staged-read contract (INV-03/SAFE-02) forbids bulk-input reads from the
pool/HDD tier: every bulk byte comes from the SSD stage. The
readers historically fall back to the declared pool path on every
miss/fence failure and record only counters. This module is the shared
policy those readers enforce (principle 8: one abstraction, not per-callsite
knobs).

Activation is explicit and process-global. The joint campaign entrypoints
(``joint_cost_quantum``, ``joint_cost_stage_a``) activate it from their
sealed ``--allowed-tiers`` flag — always, after the identity gates and
before any bulk byte, so a campaign run cannot waive silently. There is
deliberately no ambient-environment fallback in the production path.
Tests use the explicitly test-only :func:`staged_tier_policy_test_context`.

Process-global (never ContextVar) is load-bearing, not incidental: tensor
payloads are read on prefetch worker threads that inherit no context, and
the policy must reach them deterministically.

Inactive by default: ordinary offline/library paths outside campaign scope
keep their legacy fallback behavior (explicitly scoped, no automatic
campaign waiver).

Refusal semantics: :class:`TierPolicyRefused` is deliberately NOT a
``StagedReadRefused`` — existing ``except StagedReadRefused`` handlers read
the declared path, which is exactly what a strict refusal must not do.
Callers that catch ``StagedReadRefused`` for pool fallback must let
``TierPolicyRefused`` propagate (or convert it explicitly, never into a
pool read).

This module enforces tier *choice*: which staged tier a bulk open may use.
Supported staged readers pair that policy with :mod:`staged_lease`, which
owns descriptors under immutable manifests and retained native lifetimes.
Async reads join before descriptor close and window release;
CPU/GPU data outlives a window only in an owned buffer charged to the action.
The successful open's serving record supplies its pin identity. Activating
this policy alone does not add lifetime ownership to an arbitrary reader.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

#: The campaign default: the SSD stage. Pool/HDD are
#: never bulk tiers under this policy.
DEFAULT_ALLOWED_TIERS = "ssd"

PATH_ENV = 'PRISMAQUANT_STAGED_MANIFEST'
SHA_ENV = 'PRISMAQUANT_STAGED_MANIFEST_SHA256'
READ_ENV = 'PRISMAQUANT_READ_CONTRACT_SHA256'

_KNOWN_TIERS = ("ssd",)


class TierPolicyRefused(Exception):
    """A bulk open the allowed-tier policy forbids. Never read the pool."""


_LOCK = threading.Lock()
_ACTIVE: frozenset[str] | None = None
#: Thread ident holding test-only scopes, plus the stack of priors they
#: must restore LIFO. Production never takes a scope (it activates), so a
#: set owner always means test code is inside.
_OWNER: int | None = None
_STACK: list[frozenset[str] | None] = []


def parse_allowed_tiers(value: str) -> frozenset[str]:
    """Parse ``--allowed-tiers`` into the permitted staged-tier set.

    Grammar: ``ssd``, the only staged tier. ``pool``/``hdd`` are refused
    outright: legacy pool reads need explicit scoped user authorization on
    the sealed request (TIER-04), which this lane does not implement — the
    error says so rather than admitting the tier.
    """
    if type(value) is not str:
        raise ValueError("--allowed-tiers must be a comma-separated string")
    tiers = frozenset(part.strip().lower() for part in value.split(",") if part.strip())
    if tiers != frozenset(_KNOWN_TIERS):
        raise ValueError(
            "--allowed-tiers must be ssd; "
            f"got {value!r}. pool/hdd bulk reads need explicit scoped user "
            "authorization on the sealed request (TIER-04, not implemented "
            "in this lane)")
    return tiers


def activate_staged_tier_policy(value: str) -> frozenset[str]:
    """Activate strict enforcement for this process. Returns the allowed set.

    ``value`` is the sealed ``--allowed-tiers`` declaration the campaign
    entrypoints always pass explicitly. No ambient fallback: callers that
    have no sealed declaration have no policy. The explicit production
    install wins over any test-only scope state.
    """
    allowed = parse_allowed_tiers(value)
    with _LOCK:
        global _ACTIVE, _OWNER
        _ACTIVE = allowed
        _OWNER = None
        _STACK.clear()
    return allowed


def deactivate_staged_tier_policy_for_tests() -> None:
    """Restore legacy fallback behavior (tests only, explicit lifetime)."""
    with _LOCK:
        global _ACTIVE, _OWNER
        _ACTIVE = None
        _OWNER = None
        _STACK.clear()


@contextmanager
def staged_tier_policy_test_context(value: str):
    """Explicitly TEST-ONLY scoped strict policy. No production library use.

    Atomically installed; the prior verdict is restored on exit, even on
    error. Scopes are thread-exclusive: entering
    while another thread holds a scope refuses, so overlapping contexts
    can never restore inactive underneath a live reader. Same-thread
    ``with`` nesting is the only supported overlap (LIFO by construction).

    The lock is held only for the atomic capture/install and restore —
    never across the yield — so prefetch threads calling
    :func:`active_policy` cannot deadlock against a live scope.
    """
    requested = parse_allowed_tiers(value)
    me = threading.get_ident()
    with _LOCK:
        global _ACTIVE, _OWNER
        if _OWNER is not None and _OWNER != me:
            raise RuntimeError(
                "overlapping staged-tier scopes on multiple threads are "
                "unsupported; production activates once from sealed args")
        _OWNER = me
        _STACK.append(_ACTIVE)
        _ACTIVE = requested
    try:
        yield active_policy()
    finally:
        with _LOCK:
            if _OWNER != me:
                raise RuntimeError(
                    "staged-tier scope exited from a different thread "
                    "than entered")
            _ACTIVE = _STACK.pop()
            if not _STACK:
                _OWNER = None


def active_policy() -> frozenset[str] | None:
    """The active allowed-tier set, or None when policy is inactive."""
    with _LOCK:
        return _ACTIVE


def policy_is_active() -> bool:
    return active_policy() is not None


def refuse_pool_bulk_read(path: str, reason: str) -> TierPolicyRefused:
    """Build the refusal for a forbidden pool/HDD bulk open.

    Callers raise it (never catch it into a pool read). ``reason`` is one
    of the named causes: missing/stale-epoch/corrupt/wrong-digest/
    wrong-size/partial-span/readset-not-staged/
    missing-digest-binding/get-slice-is-a-data-reader.
    """
    return TierPolicyRefused(f"staged-tier-forbidden: {reason}: {path}")


class StagedRangeNotLanded(TierPolicyRefused):
    """A declared staged range that is not currently landed (availability).

    Raised at the read seam when the resolver reports ``RANGE_UNCOVERED``
    for the span: the sealed readset declares these bytes but has no serving
    cover, including stale covering rows whose staged files are missing.
    The message keeps the established
    ``readset-not-staged`` wording so existing log greps still match; the
    type is what distinguishes a proven transient cause from the generic
    refusal. An undeclared span (``RANGE_UNDECLARED``) and a failed
    covering entry (``RANGE_REFUSED``) keep the generic refusal: nothing
    about them says the bytes are on their way. Never read the pool.
    """

    def __init__(self, declared: str, start: int, end: int):
        super().__init__(f"staged-tier-forbidden: readset-not-staged: {declared}")
        self.declared = str(declared)
        self.start = int(start)
        self.end = int(end)


def _activate_native_launch():
    import os
    setting = os.environ.get('PRISMAQUANT_STAGED_ALLOWED_TIERS')
    if setting is not None:
        if setting != 'ssd' or not all(os.environ.get(key) for key in (PATH_ENV, SHA_ENV, READ_ENV)):
            raise ValueError('incomplete explicit native staged-reader configuration')
        activate_staged_tier_policy(setting)


_activate_native_launch()
