"""Strict allowed-tier policy for GPU-consumed bulk inputs (PQ:readers lane).

The staged-read contract (INV-03/SAFE-02) forbids bulk-input reads from the
pool/HDD tier: RAM first, SSD only when the sealed declaration permits. The
readers historically fall back to the declared pool path on every
miss/fence failure and record only counters. This module is the shared
policy those readers enforce (principle 8: one abstraction, not per-callsite
knobs).

Inactive by default: ordinary offline/library paths keep their legacy
fallback behavior (explicitly scoped, no automatic campaign waiver). The
joint campaign entrypoints (``joint_cost_quantum``, ``joint_cost_stage_a``)
activate it with their sealed ``--allowed-tiers`` declaration, so a
campaign run cannot waive silently. Env ``PRISMAQUANT_ALLOWED_TIERS`` is a
fallback for tests only; the sealed binding rides the payload flag through
the container, never ambient env alone.

Refusal semantics: :class:`TierPolicyRefused` is deliberately NOT a
``StagedReadRefused`` — existing ``except StagedReadRefused`` handlers read
the declared path, which is exactly what a strict refusal must not do.
Callers that catch ``StagedReadRefused`` for pool fallback must let
``TierPolicyRefused`` propagate (or convert it explicitly, never into a
pool read).

Lease posture: the PB reader-lease API (RNG-02/SM-03) is PB-owned and
pending. This module enforces tier choice without leases and refuses
clearly where staged/lease support is absent. Async prefetch and mmap pool
reads are refused as bulk opens; the strict staged path itself avoids mmap
(owned pread buffers), so no mapping outlives an unheld pin there. Full
lease-pin integration follows once the PB API lands; this policy is not
claimed complete without it.
"""
from __future__ import annotations

import os
import threading

#: The test-only ambient fallback. Production carries the declaration on the
#: sealed payload flag (``--allowed-tiers``), threaded by the dispatcher
#: through the container boundary.
ENV_VAR = "PRISMAQUANT_ALLOWED_TIERS"

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


def activate_staged_tier_policy(value: str | None = None) -> frozenset[str]:
    """Activate strict enforcement for this process. Returns the allowed set.

    ``None`` resolves the test-only env fallback, else the campaign default.
    The joint entrypoints always pass their parsed flag explicitly.
    """
    global _ACTIVE
    if value is None:
        value = os.environ.get(ENV_VAR, DEFAULT_ALLOWED_TIERS)
    allowed = parse_allowed_tiers(value)
    with _LOCK:
        _ACTIVE = allowed
    return allowed


def deactivate_staged_tier_policy_for_tests() -> None:
    """Restore legacy fallback behavior (tests only)."""
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None


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
    wrong-size/partial-span/readset-not-staged/ssd-not-allowed.
    """
    return TierPolicyRefused(f"staged-tier-forbidden: {reason}: {path}")
