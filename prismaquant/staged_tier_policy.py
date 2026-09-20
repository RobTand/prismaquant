"""Strict allowed-tier policy for GPU-consumed bulk inputs (PQ:readers lane).

The staged-read contract (INV-03/SAFE-02) forbids bulk-input reads from the
pool/HDD tier: RAM first, SSD only when the sealed declaration permits. The
readers historically fall back to the declared pool path on every
miss/fence failure and record only counters. This module is the shared
policy those readers enforce (principle 8: one abstraction, not per-callsite
knobs).

Activation is explicit and process-global. The joint campaign entrypoints
(``joint_cost_quantum``, ``joint_cost_stage_a``) activate it from their
sealed ``--allowed-tiers`` flag — always, so a campaign run cannot waive
silently. There is deliberately no ambient-environment fallback in the
production path: tests use :func:`staged_tier_policy_context` (or the
explicit setter), whose lifetime is explicit.

Process-global (never ContextVar) is load-bearing, not incidental: tensor
payloads are read on prefetch worker threads that inherit no context, and
the policy must reach them deterministically. Reset happens only at
explicit lifetimes (the context exit, or the test-only deactivator).

Inactive by default: ordinary offline/library paths outside campaign scope
keep their legacy fallback behavior (explicitly scoped, no automatic
campaign waiver).

Refusal semantics: :class:`TierPolicyRefused` is deliberately NOT a
``StagedReadRefused`` — existing ``except StagedReadRefused`` handlers read
the declared path, which is exactly what a strict refusal must not do.
Callers that catch ``StagedReadRefused`` for pool fallback must let
``TierPolicyRefused`` propagate (or convert it explicitly, never into a
pool read).

Lease posture: the PB reader-lease API (RNG-02/SM-03) is PB-owned and
pending (window-level Lease/Pin handle per root direction; exact
signatures come from the PB worker — this module proposes no stub).
This module enforces tier choice without leases and refuses clearly where
staged/lease support is absent. Owned pread buffers conceptually hold
their lease from acquire through the last pread: async chunk futures are
already joined before fd close/release on every path including
cancellation/error (see ``_read_span``), CPU/GPU copies outlive fd close
only after the owned buffer is fully read, and memory charge stays the
normal action budget. Full lease-pin integration follows once the PB API
lands; this policy is not claimed complete without it.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

#: The campaign default: RAM first, SSD stage when declared. Pool/HDD are
#: never bulk tiers under this policy.
DEFAULT_ALLOWED_TIERS = "ram,ssd"

_KNOWN_TIERS = ("ram", "ssd")


class TierPolicyRefused(Exception):
    """A bulk open the allowed-tier policy forbids. Never read the pool."""


_LOCK = threading.Lock()
_ACTIVE: frozenset[str] | None = None


def parse_allowed_tiers(value: str) -> frozenset[str]:
    """Parse ``--allowed-tiers`` into the permitted staged-tier set.

    Grammar: comma-separated subset of ``{ram,ssd}``. ``pool``/``hdd`` are
    refused outright: legacy pool reads need explicit scoped user
    authorization on the sealed request (TIER-04), which this lane does not
    implement — the error says so rather than admitting the tier.
    """
    if type(value) is not str:
        raise ValueError("--allowed-tiers must be a comma-separated string")
    tiers = frozenset(part.strip().lower() for part in value.split(",") if part.strip())
    if not tiers or not tiers <= frozenset(_KNOWN_TIERS):
        raise ValueError(
            "--allowed-tiers must be a comma-separated subset of {ram,ssd}; "
            f"got {value!r}. pool/hdd bulk reads need explicit scoped user "
            "authorization on the sealed request (TIER-04, not implemented "
            "in this lane)")
    return tiers


def activate_staged_tier_policy(value: str) -> frozenset[str]:
    """Activate strict enforcement for this process. Returns the allowed set.

    ``value`` is the sealed ``--allowed-tiers`` declaration the campaign
    entrypoints always pass explicitly. No ambient fallback: callers that
    have no sealed declaration have no policy.
    """
    allowed = parse_allowed_tiers(value)
    with _LOCK:
        global _ACTIVE
        _ACTIVE = allowed
    return allowed


def deactivate_staged_tier_policy_for_tests() -> None:
    """Restore legacy fallback behavior (tests only, explicit lifetime)."""
    with _LOCK:
        global _ACTIVE
        _ACTIVE = None


@contextmanager
def staged_tier_policy_context(value: str):
    """Explicit-lifetime strict policy for tests and scoped library use.

    Saves the prior policy on entry and restores it on exit — even on
    error — so a nested context can never clear (or permanently weaken)
    an outer campaign's enforcement: the inner policy governs only
    inside, and the outer verdict is intact afterwards. Production
    campaign entrypoints never use this helper; they activate explicitly
    from sealed args once at startup. The process-global cell is what
    reaches prefetch worker threads.
    """
    global _ACTIVE
    with _LOCK:
        prior = _ACTIVE
    activate_staged_tier_policy(value)
    try:
        yield active_policy()
    finally:
        with _LOCK:
            _ACTIVE = prior


def active_policy() -> frozenset[str] | None:
    """The active allowed-tier set, or None when policy is inactive."""
    with _LOCK:
        return _ACTIVE


def policy_is_active() -> bool:
    return active_policy() is not None


def tier_is_allowed(tier: str) -> bool:
    """Is ``tier`` (``ram``/``ssd``) openable under the active policy?"""
    allowed = active_policy()
    if allowed is None:
        return True
    return tier in allowed


def refuse_pool_bulk_read(path: str, reason: str) -> TierPolicyRefused:
    """Build the refusal for a forbidden pool/HDD bulk open.

    Callers raise it (never catch it into a pool read). ``reason`` is one
    of the named causes: missing/stale-epoch/corrupt/wrong-digest/
    wrong-size/partial-span/readset-not-staged/ssd-not-allowed/
    missing-digest-binding/get-slice-is-a-data-reader.
    """
    return TierPolicyRefused(f"staged-tier-forbidden: {reason}: {path}")
