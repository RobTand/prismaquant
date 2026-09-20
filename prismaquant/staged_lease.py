"""Lifetime-pinned staged reads through the PB reader-lease SDK (PQ #850).

Tier *choice* is enforced by :mod:`prismaquant.staged_tier_policy`; this
module adds lifetime *pinning* on top, through the exact PB signature
family (`injected_context` / `acquire_for` / `open_pinned` / `release`,
capability `reader-lease-v1`). The PB worker owns that implementation —
nothing here re-implements, shadows, or diverges from it:

- The SDK is imported only from a sealed root: the explicit override
  (tests/wiring) else the authoritative PB-injected
  ``PRISMABUILD_READER_HELPER_ROOT`` (sealed generation path, forwarded
  read-only) — never a user knob, never mutable active ``/repo``
  resolution, no vendoring. The imported tree is verified to be that
  root, and a divergent pre-import refuses.
- Identity comes only from the SDK's ``injected_context`` (PB-owned
  env + live claim row); anything missing refuses, nothing guessed.
- One :class:`LeaseWindow` per bounded read window — one composed-map
  entry, never per tensor. Acquire once, read through ``open_pinned``
  descriptors, join all async work, close every descriptor, then release
  exactly the acquired ref, on success, error, and cancellation alike.
- The serving record is the SDK's own return at the successful actual
  open — never a path candidate that merely passed an ``lstat``.
- Material integrity failure fails clear with no alternate-copy
  adoption; availability refuses the leg (a RAM leg may then acquire
  SSD covers honestly, with its own material and lifetime).
- Covers come from actual PB identity: the composed map's leads plus its
  manifest for the stage tier. RAM movers are not in the composed map,
  so a RAM leg refuses ``ram-covers-unresolved`` until the PB owner
  provides mover resolution (requested via root) — SSD leads are never
  pretended to identify a RAM mover.
- Fork safety is enforced, not documented-only: the holder pid is
  recorded at entry and every window operation (open, close, exit) plus
  every reader payload/lifecycle operation refuses loudly in a forked
  child, so inherited descriptors and mappings can never be used past
  the parent's release. Normal PQ readers use threads, which share the
  holder pid; no second process manager is invented.
  ``register_inherited_ref`` is never used — unregistered inheritance is
  unsupported, and the guards make that explicit rather than lucky. A
  forked child must fork with no live window and `_exit` without I/O.
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path

from .staged_tier_policy import TierPolicyRefused

#: Approved PB candidate commit (pristine checkout; tests assert it).
PINNED_SDK_COMMIT = "d079ad33f82976875a992cc68fd5c1bde6d7bd20"

#: Names PQ actually calls. Anything else is not our protocol.
_REQUIRED_NAMES = ("injected_context", "acquire_for", "open_pinned",
                   "release", "covers_for_keys", "READER_LEASE_TAG")

#: Refusals that prove content/identity failure: fail clear, no alternate.
_INTEGRITY_REFUSALS = ("file-identity-changed", "generation-changed",
                       "source-coverage-gap", "ownership-uncertain")

#: Refusals that prove unavailability: the leg refuses; another permitted
#: tier may acquire honestly. Anything unlisted fails clear.
_AVAILABILITY_REFUSALS = ("unpublished", "stale-epoch", "retiring",
                          "file-missing", "no-file-identity",
                          "ram-covers-unresolved", "lease-helper-unavailable",
                          "lease-context-unavailable")

_HELPER_LOCK = threading.Lock()
_HELPER_ROOT: str | None = None

#: Authoritative PB-injected helper root: the sealed generation path PB
#: forwards core+container read-only. Read automatically as the production
#: discovery — never a user knob, never mutable-`/repo` resolution, never
#: a test fallback. The explicit setter above wins when set (tests).
HELPER_ROOT_ENV_VAR = "PRISMABUILD_READER_HELPER_ROOT"

#: Caller-owned pre-check cache shared process-wide (mirrors the SDK's
#: ``context`` argument): fragment/material/epoch reads cached across
#: calls so a window batch does not re-stat over NFS per tensor. Successes
#: may cache; the SDK re-reads fresh under its lock before anything pins,
#: so no cached pre-check is ever admission proof. Plain dict: concurrent
#: inserts race benignly (idempotent values), correctness never depends
#: on it.
_ACQUIRE_CONTEXT: dict = {}


class LeaseRefused(TierPolicyRefused):
    """A lifetime-pinned open the lease path forbids. Never read the pool.

    ``kind`` is ``"integrity"`` (fail clear, no alternate copy) or
    ``"availability"`` (the leg refuses; another permitted tier may
    acquire with its own material and lifetime).
    """

    def __init__(self, reason: str, *, kind: str):
        super().__init__(f"staged-tier-forbidden: {reason}")
        self.kind = kind


def _refuse(reason: str, *, kind: str) -> LeaseRefused:
    return LeaseRefused(reason, kind=kind)


def _classify(refusal: str) -> str:
    """Integrity or availability for an SDK refusal string."""
    head = str(refusal).split(":", 1)[0]
    if head in _INTEGRITY_REFUSALS:
        return "integrity"
    if head in _AVAILABILITY_REFUSALS:
        return "availability"
    return "integrity"


def set_lease_helper_root(path: str | Path | None) -> None:
    """Name the sealed PB generation root for tests/wiring (explicit only)."""
    global _HELPER_ROOT
    with _HELPER_LOCK:
        _HELPER_ROOT = None if path is None else str(path)


def lease_helper_root() -> str | None:
    """The sealed helper root: explicit override, else the authoritative
    PB-injected ``PRISMABUILD_READER_HELPER_ROOT`` (sealed generation
    path, forwarded read-only). Anything else — including a missing
    variable — means no helper, and strict lifetime refuses."""
    with _HELPER_LOCK:
        if _HELPER_ROOT is not None:
            return _HELPER_ROOT
    return os.environ.get(HELPER_ROOT_ENV_VAR)


def _sdk():
    """The pinned SDK module, or a clear availability refusal.

    Imports ``prismabuild.reader_lease`` from the explicit sealed root
    only, verifies the module lives under that root (no shadow, no
    divergent pre-import), and requires the exact approved names.
    """
    root = lease_helper_root()
    if root is None:
        raise _refuse("lease-helper-unavailable", kind="availability")
    src = str(Path(root) / "src")
    with _HELPER_LOCK:
        present = sys.modules.get("prismabuild.reader_lease")
        if present is not None:
            if not str(getattr(present, "__file__", "")).startswith(src + os.sep):
                raise _refuse("lease-helper-divergent", kind="integrity")
            module = present
        else:
            sys.path.insert(0, src)
            try:
                import prismabuild.reader_lease as module  # noqa: PLC0415
            except ImportError as exc:
                raise _refuse(f"lease-helper-unavailable: {exc}",
                              kind="availability") from None
            if not str(getattr(module, "__file__", "")).startswith(src + os.sep):
                raise _refuse("lease-helper-divergent", kind="integrity")
    for name in _REQUIRED_NAMES:
        if not hasattr(module, name):
            raise _refuse(f"lease-helper-unsupported: no {name}",
                          kind="availability")
    return module


def resolve_context(*, env=None):
    """The SDK's injected identity, or a clear availability refusal.

    Strict context requires a launch-env positive claim match: the
    identity comes only from ``injected_context`` (PB-owned env + the
    live claim row), refusals propagate as-is, and nothing here ever
    synthesizes, guesses, or falls back to a map-derived identity.
    """
    sdk = _sdk()
    try:
        answer = sdk.injected_context(env=env)
    except Exception as exc:  # PB-internal failure: fail clear, never guess
        raise _refuse(f"lease-context-error: {exc}", kind="integrity") from None
    if not isinstance(answer, dict) or not answer.get("ok"):
        refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
        raise _refuse(f"lease-context-unavailable: {refusal}", kind="availability")
    return sdk, answer["ctx"]


def covers_for_leads(leads, manifest_sha256) -> list[dict[str, str]]:
    """Stage-tier covers from actual composed-map identity.

    Every lead genuinely vouched the composed manifest (compose refuses
    disagreement), and ``acquire`` proves per-key presence against
    ``expected`` — so no per-key mover mapping is pretended. RAM movers
    are not in the composed map: see :func:`ram_covers`.
    """
    heritage = sorted({str(lead) for lead in leads or ()})
    if not heritage or any(len(mover) != 64 for mover in heritage):
        raise _refuse("source-coverage-gap: no lead covers", kind="integrity")
    return [{"mover_action_key": mover, "manifest_sha256": str(manifest_sha256)}
            for mover in heritage]


def resolve_ram_covers(resolver, declared, entry):
    """RAM-tier covers via the PB cover lookup, cross-checked, never equated.

    Returns ``(covers, key)`` for a RAM-tier acquire: the SDK resolves the
    minimal covering movers from ram-tier fragments/material at the
    announced epoch, and PQ cross-checks the helper's expected bytes/digest
    against the sealed composed-map entry before adopting anything — a
    divergent proof fails clear as integrity. Refusals (no material, stale
    epoch, contradiction) are availability: the SSD leg may acquire
    honestly. SSD leads are never consulted for RAM identity.
    """
    from .residency_map import residency_map_key
    sdk, ctx = resolve_context()
    identity = resolver.lease_identity()
    ram_tier = identity["ram_tier_id"]
    ram_epoch = identity["ram_epoch"]
    if not ram_tier or not ram_epoch:
        raise _refuse("ram-not-announced", kind="availability")
    key = residency_map_key(str(declared), entry["offset"])
    if not hasattr(sdk, "covers_for_keys"):
        raise _refuse("ram-covers-unresolved: no cover lookup",
                      kind="availability")
    try:
        answer = sdk.covers_for_keys(
            identity["residency_root"], str(ctx["action_key"]), [key],
            tier_id=ram_tier, manifest_sha256=identity["manifest_sha256"],
            epoch=ram_epoch, context=_ACQUIRE_CONTEXT)
    except Exception as exc:
        raise _refuse(f"ram-cover-lookup-error: {exc}",
                      kind="integrity") from None
    if not isinstance(answer, dict) or not answer.get("ok"):
        refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
        raise _refuse(str(refusal), kind=_classify(refusal))
    expected = answer.get("expected")
    got = expected.get(key) if isinstance(expected, dict) else None
    if (not isinstance(got, dict) or got.get("bytes") != entry["bytes"]
            or str(got.get("sha256") or "") != str(entry["sha256"] or "")):
        raise _refuse("ram-cover-proof-divergent", kind="integrity")
    covers = answer.get("covers")
    if (not isinstance(covers, list) or not covers
            or any(not isinstance(cover, dict)
                   or len(str(cover.get("mover_action_key") or "")) != 64
                   for cover in covers)):
        raise _refuse("ram-cover-proof-divergent", kind="integrity")
    return covers, key


def _acquire_ram_window(resolver, declared, entry):
    """Enter a RAM-tier lifetime window for one entry. Returns (window, key).

    Acquires at the announced RAM epoch with resolved RAM covers; the
    payload then serves from the tmpfs copy. Any refusal carries its
    integrity/availability kind for the caller to split.
    """
    covers, key = resolve_ram_covers(resolver, declared, entry)
    identity = resolver.lease_identity()
    window = LeaseWindow({
        "tier_id": identity["ram_tier_id"],
        "epoch": identity["ram_epoch"],
        "covers": covers,
        "expected": {key: {"bytes": entry["bytes"],
                           "sha256": entry["sha256"]}},
        "span": {"start_bytes": entry["offset"],
                 "end_bytes": entry["offset"] + entry["bytes"]},
    })
    try:
        window.__enter__()
    except LeaseRefused as refusal:
        resolver.record_ram_fallback(declared, str(refusal))
        raise
    return window, key


def ram_covers(map_entry: dict) -> list[dict[str, str]]:
    """RAM-tier covers: refused until the PB owner resolves RAM movers.

    Retained for the pre-lookup SDK pin only; current code resolves via
    :func:`resolve_ram_covers` instead. Direct callers get the honest
    refusal: SSD leads never identify a RAM mover.
    """
    raise _refuse("ram-covers-unresolved", kind="availability")


def acquire_entry_window(resolver, declared, entry: dict):
    """Acquire a lifetime window for one composed-map entry.

    RAM first where offered and allowed: the RAM leg resolves real
    RAM-mover covers and pins at the announced epoch, so a live tmpfs
    copy serves with its own material and lifetime — the full goal, not
    a fallback. An availability refusal on the RAM leg records and falls
    through to SSD; an integrity refusal propagates with no alternate
    adoption. SSD acquires honestly with the map's leads. ``ssd`` outside
    the allowed tiers refuses. Any ``LeaseRefused`` propagates with its
    integrity/availability kind; the caller opens keys, reads through
    held descriptors, and exits the window (close-then-release) on every
    path. One window per entry, never per tensor.
    """
    from .residency_map import residency_map_key
    from .staged_tier_policy import tier_is_allowed
    if entry.get("ram_path") is not None and tier_is_allowed("ram"):
        try:
            return _acquire_ram_window(resolver, declared, entry)
        except LeaseRefused as refusal:
            if refusal.kind != "availability":
                raise
            resolver.record_ram_fallback(declared, str(refusal))
    if not tier_is_allowed("ssd"):
        raise _refuse("ssd-not-allowed", kind="availability")
    key = residency_map_key(str(declared), entry["offset"])
    identity = resolver.lease_identity()
    window = LeaseWindow({
        "tier_id": identity["tier_id"],
        "epoch": "",
        "covers": covers_for_leads(identity["leads"],
                                   identity["manifest_sha256"]),
        "expected": {key: {"bytes": entry["bytes"],
                           "sha256": entry["sha256"]}},
        "span": {"start_bytes": entry["offset"],
                 "end_bytes": entry["offset"] + entry["bytes"]},
    })
    try:
        window.__enter__()
    except LeaseRefused as refusal:
        resolver.record_fallback(declared, str(refusal))
        raise
    return window, key


class LeaseWindow:
    """One bounded read window, pinned for exactly its holder's lifetime.

    ``spec`` names the window: ``tier_id``, ``epoch``, ``covers``,
    ``expected`` (map key -> ``{bytes, sha256}``), ``span``
    (``{start_bytes, end_bytes}`` in entry coordinates), and optional
    ``ram`` passthrough. ``acquire_token`` defaults to a fresh uuid4;
    pass it back to retry idempotently onto the same ref.

    Use as a context manager: ``__enter__`` acquires (one
    ``generation-changed`` re-resolve, then fail clear),
    :meth:`open` returns held ``(fd, serving)`` per entry key and
    registers every descriptor, ``__exit__`` closes all descriptors
    first and then releases exactly the acquired ref — on success,
    error, and cancellation. A forked child (different pid) never
    releases: orphan refs retain safely.
    """

    def __init__(self, spec: dict, *, acquire_token: str | None = None,
                 helper_root: str | Path | None = None, env=None):
        self._spec = dict(spec)
        self._token = acquire_token or uuid.uuid4().hex
        self._helper_root = helper_root
        self._env = env
        self._pin: dict | None = None
        self._pin_id: str | None = None
        self._ref_id: str | None = None
        self._consumer: str | None = None
        self._queue_root: str | None = None
        self._pool_mod = None
        self._pid: int | None = None
        self._owner_pid: int | None = None
        self._fds: dict[int, str] = {}
        self._released = False

    def _require_owner(self, operation: str) -> None:
        """Refuse forked-child use loudly: only the acquiring process may
        open, close, or release. Inherited descriptors and mappings stay
        the parent's: the child must not perform I/O on them — use
        threads, or fork with no live window and `_exit` without I/O.
        Unregistered inheritance is unsupported, and this (not a
        docstring) enforces it on every state-changing operation."""
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            raise RuntimeError(
                f"LeaseWindow.{operation} from a forked child is "
                "unsupported: fork with a live window must not use "
                "inherited descriptors, mappings, or refs — the parent "
                "releases them on its own lifecycle and the child must "
                "not outlive that protection. Use threads, which share "
                "the holder pid, or fork with no live window.")

    # -- acquire ------------------------------------------------------

    def __enter__(self) -> "LeaseWindow":
        if self._helper_root is not None:
            set_lease_helper_root(self._helper_root)
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            self._require_owner("__enter__")
        sdk, ctx = resolve_context(env=self._env)
        spec = self._spec
        answer = sdk.acquire_for(
            ctx, tier_id=str(spec["tier_id"]), epoch=str(spec["epoch"]),
            covers=spec["covers"], expected=spec.get("expected"),
            span=spec["span"], acquire_token=self._token,
            ram=spec.get("ram"), context=_ACQUIRE_CONTEXT)
        if not isinstance(answer, dict) or not answer.get("ok"):
            refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
            if str(refusal).split(":", 1)[0] == "generation-changed":
                answer = sdk.acquire_for(
                    ctx, tier_id=str(spec["tier_id"]), epoch=str(spec["epoch"]),
                    covers=spec["covers"], expected=spec.get("expected"),
                    span=spec["span"], acquire_token=self._token,
                    ram=spec.get("ram"), context=_ACQUIRE_CONTEXT)
                if not isinstance(answer, dict) or not answer.get("ok"):
                    refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
            if not isinstance(answer, dict) or not answer.get("ok"):
                raise _refuse(str(refusal), kind=_classify(refusal))
        self._pin = answer["pin"]
        self._pin_id = str(answer["pin_id"])
        self._ref_id = str(answer["ref_id"])
        self._consumer = str(ctx["action_key"])
        self._queue_root = str(ctx["queue_root"])
        self._pid = os.getpid()
        self._owner_pid = os.getpid()
        import prismabuild.pool as pool_mod  # noqa: PLC0415, sealed tree
        self._pool_mod = pool_mod
        return self

    @property
    def serving_tier(self) -> str | None:
        if self._pin is None:
            return None
        tier = str(self._pin.get("tier_id") or "")
        return "ram" if tier.startswith("ram:") else "stage"

    def stage_path(self, key: str) -> str | None:
        """The pinned stage path one entry key opened from, if held."""
        if self._pin is None:
            return None
        for entry in self._pin.get("entries") or ():
            if isinstance(entry, dict) and entry.get("key") == key:
                return str(entry.get("stage_path") or "")
        return None

    # -- open ----------------------------------------------------------

    def open(self, key: str) -> tuple[int, dict]:
        """Hold ``(fd, serving)`` for one pinned entry key.

        The serving record is the SDK's own return at this successful
        actual open. The caller reads through the descriptor and returns
        it via :meth:`close_fd`; the ref outlives every descriptor.
        A forked child is refused here: opening on the parent's ref and
        outliving the parent's release would leave the child's FD live
        with no pin.
        """
        self._require_owner("open")
        if self._pin is None or self._pin_id is None or self._ref_id is None:
            raise _refuse("lease-not-acquired", kind="integrity")
        sdk, ctx = resolve_context(env=self._env)
        queue = self._pool_mod.PoolQueue(self._queue_root)
        try:
            fd, serving = sdk.open_pinned(queue, self._pin, self._ref_id, str(key))
        except Exception as exc:
            # Open-time refusal (stale ref, changed bytes, unknown key) or
            # PB-internal failure: fail clear either way, never guess.
            raise _refuse(f"lease-open-refused: {exc}", kind="integrity") from None
        self._fds[int(fd)] = str(key)
        return int(fd), dict(serving)

    def close_fd(self, fd: int) -> None:
        """Close one held descriptor. Release-before-close is forbidden:
        the ref is released only in :meth:`__exit__`, after every fd."""
        self._require_owner("close_fd")
        if int(fd) in self._fds:
            try:
                os.close(int(fd))
            finally:
                del self._fds[int(fd)]

    def close_fds(self) -> None:
        """Close every held descriptor, first failure raised last."""
        self._require_owner("close_fds")
        failure = None
        for fd in sorted(self._fds):
            try:
                os.close(fd)
            except OSError as exc:
                if failure is None:
                    failure = exc
        self._fds.clear()
        if failure is not None:
            raise failure

    # -- release --------------------------------------------------------

    def __exit__(self, *args) -> bool:
        # Forked-child teardown is refused loudly by close_fds below: a
        # child unwinding a parent's window must not close inherited
        # descriptors or release the parent's ref.
        failure = None
        try:
            self.close_fds()
        except (OSError, RuntimeError) as exc:
            failure = exc
        try:
            self._release_exact()
        except LeaseRefused as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure
        return False

    def _release_exact(self) -> None:
        """Release exactly the acquired ref, once, from the holder only."""
        if self._released or self._pin_id is None or self._ref_id is None:
            return
        if self._pid is not None and os.getpid() != self._pid:
            # Unreachable through __exit__ (refused above); kept as the
            # last-resort guard for direct callers. Deliberately does NOT
            # mark released: the holder's own later exit must still
            # release exactly, and the orphan retains safely until then
            # on the PB side.
            return
        sdk, _ctx = resolve_context(env=self._env)
        queue = self._pool_mod.PoolQueue(self._queue_root)
        try:
            sdk.release(queue, self._pin_id, self._ref_id,
                        consumer_action_key=self._consumer)
        finally:
            self._released = True
