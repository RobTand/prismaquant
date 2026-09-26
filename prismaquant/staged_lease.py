"""Lifetime-pinned staged reads through the PB reader-lease SDK (PQ #850).

Tier *choice* is enforced by :mod:`prismaquant.staged_tier_policy`; this
module adds lifetime *pinning* on top, through the exact PB signature
family (`injected_context` / `acquire_for` / `open_pinned` / `release`,
capability `reader-lease-v1`). The PB worker owns that implementation —
nothing here re-implements, shadows, or diverges from it:

- The SDK resolves from a sealed tree (explicit override, else the
  authoritative PB-injected ``PRISMABUILD_READER_HELPER_ROOT``) or, in
  tests only, the explicitly injected reviewed install — never an
  implicit installed fallback in production, which refuses fail-closed.
  Every imported PB submodule must share one immutable package root or
  entry refuses as divergent BEFORE any pin is created. No vendoring.
- Identity comes only from the SDK's ``injected_context`` (PB-owned
  env + live claim row); anything missing refuses, nothing guessed.
- One :class:`LeaseWindow` per bounded read window, never per tensor: one
  composed-map entry (:func:`acquire_entry_window`), or every entry of a
  window that resolves through one material namespace on one tier
  (:func:`acquire_entries_window`, PQ #997). Acquire once, read through
  ``open_pinned`` descriptors, join all async work, close every
  descriptor, then release exactly the acquired ref, on success, error,
  and cancellation alike.
- The serving record is the SDK's own return at the successful actual
  open — never a path candidate that merely passed an ``lstat``.
- Material integrity failure fails clear with no alternate-copy
  adoption; availability refuses the leg (a RAM leg may then acquire
  SSD covers honestly, with its own material and lifetime).
- Covers come from actual PB identity: the PB `covers_for_keys` lookup
  resolves the minimal covering movers for the requested key on BOTH
  tiers — stage-tier selection is bounded to the current consumer,
  manifest, tier and the SSD empty epoch and cross-checked against the
  sealed entry before adoption; RAM covers resolve at the announced
  epoch, never equated. The composed map's leads are heritage, not a
  covering set: an egress legitimately retires a lead's fragment and
  material once its phase is read past, and ``acquire`` requires every
  named cover to exist, so naming all leads let one retired, unrelated
  mover poison any later entry (Stage A ``f9951e60``: every model-87..90
  SSD entry refused ``unpublished`` while its own mover was complete).
  SSD leads are never consulted for RAM identity.
- Fork safety: supported PQ readers use threads and owned pread
  buffers, and every window operation (open, close, exit) plus every
  reader payload/lifecycle operation rejects inherited handle
  operations loudly in a forked child. Returned tensors ride owned
  buffers, so they need no file-backed mapping lease once the buffer is
  owned. Arbitrary raw descriptor escape (a dup'd fd read directly, an
  unregistered mapping used past release) and unregistered fork are
  unsupported; PB's explicitly registered inheritance remains its API.
  No second process manager is invented. A forked child must fork with
  no live window and `_exit` without I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from .staged_tier_policy import TierPolicyRefused

#: Accepted PB730/PB741 source pin for the PB reader lease.
#: Deployment qualification is separate; this pin advertises no capability.
#: The owning literal the test resolver reads — see
#: tools/resolve_prismabuild_dev_pin.py. No capability assertion rides it.
PB_READER_LEASE_PIN_COMMIT = "461728e4dcc08123d5fdb410eb2f18772fdb3fe0"
PINNED_SDK_COMMIT = PB_READER_LEASE_PIN_COMMIT

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

#: A pool row is small structured JSON; this only bounds a pathological read.
_MAX_POOL_ROW_BYTES = 8 * 1024 * 1024

_HELPER_LOCK = threading.Lock()
_HELPER_ROOT: str | None = None
#: Test-only injected SDK module (see inject_installed_sdk_for_tests).
#: Production never sets this: it resolves the sealed tree or refuses.
_INJECTED = None

#: The ``prismabuild.*`` modules the process already had when the current
#: test-only injection imported the installed distribution. The teardown
#: removes exactly what the injection added and nothing else (PQ #963).
#: ``None`` means no injection is on record, and then ``sys.modules`` is
#: left alone.
_INJECTED_MODULES_BEFORE: frozenset[str] | None = None

#: Authoritative PB-injected helper root: the sealed generation path PB
#: forwards core+container read-only. Read automatically as the production
#: discovery — never a user knob, never mutable-`/repo` resolution, never
#: a test fallback. The explicit setter above wins when set (tests).
HELPER_ROOT_ENV_VAR = "PRISMABUILD_READER_HELPER_ROOT"

def _call_context() -> dict:
    """The SDK's ``context`` pre-check cache, for one call and never kept.

    The SDK caches each mover's fragment and material in it: by generation in
    ``covers_for_keys``, unconditionally in ``acquire``. A stage mover
    republishes both documents as its entries land, all under ONE generation
    for its whole run, so a cache that outlives the call answers for a mover
    as it stood at the first look. Stage A r2 (2026-09-21) leased a shard's
    1 MiB header entry, waited 37 s for the same mover's 5 GB body entry, and
    was refused ``unpublished`` for bytes that were on the stage (PQ #905,
    PrismaBuild #823). A stale pair can only ever hide a key, so the cost of
    keeping it was a dead run and the cost of dropping it is one small
    fragment read per mover per lease -- leases are per staged entry, never
    per tensor.
    """
    return {}


class LeaseRefused(TierPolicyRefused):
    """A lifetime-pinned open the lease path forbids. Never read the pool.

    ``kind`` is ``"integrity"`` (fail clear, no alternate copy) or
    ``"availability"`` (the leg refuses; another permitted tier may
    acquire with its own material and lifetime).
    """

    def __init__(self, reason: str, *, kind: str):
        super().__init__(f"staged-tier-forbidden: {reason}")
        self.reason = str(reason)
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


def _sdk_accepts_material_namespace(sdk) -> bool:
    """Does this pinned SDK take a produced batch's material namespace?

    Asked of the SDK itself rather than assumed from its version: the
    owner/material-namespace split is an ADDITIVE parameter on
    ``acquire_for``, so a pin predating it would silently bind an output
    namespace as the running action. A pin that does not carry it refuses
    the produced read outright; nothing falls back to the owner namespace
    and nothing reads the pool.

    COST, stated rather than assumed away: this runs once per acquired
    window, and a window is one entry, so it is per-entry reflection on a
    path that then reads a multi-megabyte payload. Caching it per pinned
    module (the module is immutable, so one validation would do) is the
    obvious shape and is deliberately NOT applied here: nothing has
    measured it against the read it sits beside, and a performance change
    justified by a guess is the thing this repo refuses. Measure first.
    """

    import inspect

    acquire_for = getattr(sdk, "acquire_for", None)
    if acquire_for is None:
        return False
    try:
        parameters = inspect.signature(acquire_for).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return ("material_namespace" in parameters
            and "residency_root" in parameters)


def _sdk():
    """The pinned SDK module, or a clear availability refusal.

    Two attributable sources, in order. (1) A sealed tree: the explicit
    override else the authoritative PB-injected ``PRISMABUILD_READER_HELPER_ROOT``
    (generation root; ``src/`` appended per the agreed contract), verified
    to actually serve the imported module — no shadow, no divergent
    pre-import. (2) The TEST-ONLY injected reviewed install (see
    :func:`inject_installed_sdk_for_tests`): the pbtest pin guard proves
    its commit, RECORD bytes, and shadow-freedom before pytest starts.
    Production with no sealed tree refuses fail-closed
    (``lease-helper-unavailable``) — an installed package is never an
    implicit production fallback. Neither path vendors code.
    """
    root = lease_helper_root()
    if root is not None:
        return _sdk_from_tree(root)
    with _HELPER_LOCK:
        injected = _INJECTED
    if injected is not None:
        for name in _REQUIRED_NAMES:
            if not hasattr(injected, name):
                raise _refuse(f"lease-helper-unsupported: no {name}",
                              kind="availability")
        return injected
    raise _refuse("lease-helper-unavailable", kind="availability")


def sdk_submodule(name: str):
    """One ``prismabuild.<name>`` from the SAME generation as the SDK.

    Every PrismaBuild module this package uses must come from ONE sealed
    generation. A bare ``import prismabuild.produced_output`` does not
    guarantee that: production forwards an immutable
    ``PRISMABUILD_READER_HELPER_ROOT`` and mounts it, but it does not
    populate ``sys.path``, so a bare import can land on an older container
    distribution -- or, worse, a MIXTURE, with produced_output from one
    generation and pool from another. That is fatal to a lane whose whole
    output is qualified provenance: bytes you cannot name are not a pin.

    So the reader-lease SDK is resolved first (it owns the generation
    discovery and its ``src`` insertion), the requested submodule is
    imported after it, and the existing coherence check then proves every
    imported ``prismabuild.*`` -- the new one included -- resolves inside
    that one package directory. The refusal is the ordinary lease refusal,
    named, never a fallback.
    """

    import importlib                                     # noqa: PLC0415

    sdk = _sdk()
    expected = _package_dir_of(sdk)
    try:
        module = importlib.import_module(f"prismabuild.{name}")
    except ImportError as exc:
        raise _refuse(f"lease-helper-unavailable: prismabuild.{name}: {exc}",
                      kind="availability") from None
    _check_package_coherence(expected)
    served = Path(getattr(module, "__file__", ""))
    try:
        inside = served.resolve().is_relative_to(expected.resolve())
    except OSError:
        inside = False
    if not inside:
        raise _refuse(
            f"lease-helper-divergent: prismabuild.{name} serves {served}, "
            f"outside the SDK generation at {expected}",
            kind="integrity")
    return module


def inject_installed_sdk_for_tests():
    """TEST-ONLY explicit injection of the reviewed installed distribution.

    Binds the ``prismabuild`` resolved by normal import after verifying it
    is a single installed distribution at exactly
    :data:`PB_READER_LEASE_PIN_COMMIT` (no worktree shadow, no editable
    install — the same properties the pbtest pin guard proves worker-side
    before pytest). Raises ``RuntimeError`` (never a tier refusal) when
    the environment does not provide it: tests fail loudly on a missing
    dependency, never silently skip. Production never calls this.
    """
    global _INJECTED, _INJECTED_MODULES_BEFORE
    import importlib.metadata as metadata
    import json as _json
    owners = metadata.packages_distributions().get("prismabuild", [])
    if len(owners) != 1:
        raise RuntimeError(
            "test SDK injection needs exactly one installed distribution "
            f"owning prismabuild, found {owners}")
    dist = metadata.distribution(owners[0])
    try:
        direct = _json.loads(dist.read_text("direct_url.json") or "{}")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "test SDK injection cannot prove install provenance") from exc
    vcs = direct.get("vcs_info", {})
    if (direct.get("dir_info", {}).get("editable")
            or vcs.get("vcs") != "git"
            or vcs.get("commit_id") != PB_READER_LEASE_PIN_COMMIT):
        raise RuntimeError(
            "test SDK injection needs a non-editable Git install at "
            f"{PB_READER_LEASE_PIN_COMMIT}, found {vcs}")
    # Before the import, so the teardown can tell the modules this
    # injection adds from the ones the process already had (PQ #963).
    # A second injection with one still on record keeps the first
    # snapshot: the teardown is what clears it.
    with _HELPER_LOCK:
        if _INJECTED_MODULES_BEFORE is None:
            _INJECTED_MODULES_BEFORE = frozenset(_prismabuild_modules())
    import prismabuild.reader_lease as module  # noqa: PLC0415
    # The checks above prove what is installed, not what the import
    # served. A ``prismabuild`` preimported from another tree (a sealed
    # generation's ``src`` put on ``sys.path`` by an earlier test) answers
    # this import instead, and the coherence check below cannot see it: it
    # is relative to the module it was handed. That module then binds as
    # the "installed" SDK, with another generation's behaviour
    # (PQ #1281). The raise leaves the snapshot on record, as the
    # coherence raise below does; the teardown's clear drops both.
    installed = Path(dist.locate_file("prismabuild/reader_lease.py")).resolve()
    served = Path(getattr(module, "__file__", None) or "").resolve()
    if served != installed:
        raise RuntimeError(
            f"test SDK injection: prismabuild.reader_lease serves {served}, "
            f"not the installed distribution's {installed}; a prismabuild "
            "imported from another tree shadows it (PQ #1281)")
    try:
        expected = _package_dir_of(module)
    except LeaseRefused as exc:
        raise RuntimeError(f"test SDK injection: {exc}") from None
    try:
        _check_package_coherence(expected)
    except LeaseRefused as exc:
        raise RuntimeError(
            "test SDK injection refuses divergent preimported "
            f"prismabuild.* outside {expected}: {exc}") from None
    origin = str(Path(getattr(module, "__file__", "")).resolve())
    for name in _REQUIRED_NAMES:
        if not hasattr(module, name):
            raise RuntimeError(f"test SDK surface missing {name} at {origin}")
    with _HELPER_LOCK:
        _INJECTED = module
    return module


def _prismabuild_modules():
    """Every ``prismabuild`` entry ``sys.modules`` holds right now."""
    return [name for name in list(sys.modules)
            if name == "prismabuild" or name.startswith("prismabuild.")]


def clear_injected_sdk_for_tests() -> None:
    """Drop the test-only injection, and the modules it imported.

    Resetting the module reference alone is not enough (PQ #963). The
    injection also leaves the installed distribution in ``sys.modules``,
    and :func:`_sdk_from_tree` returns a preimported
    ``prismabuild.reader_lease`` as it is: a leftover venv module then
    fails containment against the sealed generation tree, so every later
    test in the same pytest worker refuses ``lease-helper-divergent``.
    Only the modules an injection added are removed, and with no
    injection on record ``sys.modules`` is left alone.
    """
    global _INJECTED, _INJECTED_MODULES_BEFORE
    with _HELPER_LOCK:
        _INJECTED = None
        before, _INJECTED_MODULES_BEFORE = _INJECTED_MODULES_BEFORE, None
        if before is None:
            return
        for name in _prismabuild_modules():
            if name not in before:
                del sys.modules[name]


def _package_dir_of(module) -> Path:
    """The immutable package root one SDK module proves: its parent dir."""
    location = getattr(module, "__file__", None)
    if not location:
        raise _refuse("lease-helper-divergent: package has no file",
                      kind="integrity")
    return Path(location).resolve().parent


def _check_package_coherence(expected_dir: Path) -> None:
    """Refuse any preimported ``prismabuild.*`` outside the expected root.

    Component-wise path containment against the actual package directory
    proven by the resolved SDK module (plus the package itself) — never a
    substring or private-path blacklist. Runs BEFORE any acquire, so a
    divergent preimport cannot strand a pin.
    """
    expected = Path(expected_dir).resolve()
    # Keep the same point-in-time module mapping without allocating one
    # tracked tuple per unrelated import. On a loaded model those temporary
    # tuples repeatedly trigger cyclic GC on the lease hot path. A dict copy
    # retains the identical names and module objects; all path checks below
    # still run, including newly imported modules and changed symlink targets.
    for name, mod in sys.modules.copy().items():
        if name != "prismabuild" and not name.startswith("prismabuild."):
            continue
        if mod is None:
            continue
        location = getattr(mod, "__file__", None)
        if location is None:
            paths = getattr(mod, "__path__", None)
            if paths is None:
                continue
            try:
                entries = [Path(entry).resolve() for entry in paths]
            except OSError:
                raise _refuse(
                    f"lease-helper-divergent: {name} has unresolvable path",
                    kind="integrity")
            if not any(entry == expected for entry in entries):
                raise _refuse(
                    f"lease-helper-divergent: {name} resolves elsewhere",
                    kind="integrity")
            continue
        try:
            resolved = Path(location).resolve()
        except OSError:
            raise _refuse(
                f"lease-helper-divergent: {name} has unresolvable file",
                kind="integrity")
        if resolved != expected and expected not in resolved.parents:
            raise _refuse(
                f"lease-helper-divergent: {name} resolves elsewhere",
                kind="integrity")


def _sdk_from_tree(root: str):
    """Import the SDK from one sealed generation tree.

    The tree's package root is ``<root>/src/prismabuild``; the serving
    module plus every preimported ``prismabuild.*`` must resolve inside
    it (component-wise containment), else entry refuses divergent before
    any pin. No substring matching, no vendoring.
    """
    src = Path(root) / "src"
    expected = src / "prismabuild"
    with _HELPER_LOCK:
        present = sys.modules.get("prismabuild.reader_lease")
        if present is not None:
            module = present
        else:
            sys.path.insert(0, str(src))
            try:
                import prismabuild.reader_lease as module  # noqa: PLC0415
            except ImportError as exc:
                raise _refuse(f"lease-helper-unavailable: {exc}",
                              kind="availability") from None
        try:
            served = Path(getattr(module, "__file__", "")).resolve()
            want = expected.resolve()
        except OSError:
            raise _refuse("lease-helper-divergent: unresolvable SDK file",
                          kind="integrity")
        try:
            served.relative_to(want)
        except ValueError:
            raise _refuse("lease-helper-divergent: SDK resolves elsewhere",
                          kind="integrity")
    for name in _REQUIRED_NAMES:
        if not hasattr(module, name):
            raise _refuse(f"lease-helper-unsupported: no {name}",
                          kind="availability")
    _check_package_coherence(_package_dir_of(module))
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


class ReadsetUnbound(Exception):
    """PB's sealed readset could not be reached. Never a refusal to read.

    Callers treat this as "this process cannot tell a declared range from
    an undeclared one" and fall back to the conservative behaviour, which
    is to do nothing extra. It is deliberately NOT a ``TierPolicyRefused``:
    not knowing the readset never makes a range unreadable.
    """


def resolve_sealed_readset(*, env=None):
    """``(cas_root, manifest_sha256, manifest_bytes)`` for this action.

    PB publishes both halves into the action's own pool row at publish
    time (``pool.py`` ``"cas_root"`` and the ``"residency"`` block), the
    row moves ready -> claimed unchanged, and the sealed data manifest is
    an ordinary CAS blob under ``<cas_root>/blobs/<d[:2]>/<d>``. So the
    authority for "is this range declared" is PB's own request context,
    reached from the identity the SDK already vouches: this reads the
    claim row that :func:`resolve_context` has just matched against the
    launch env, and nothing else.

    **This is a protocol extension.** ``_REQUIRED_NAMES`` is the SDK
    surface PQ calls; ``queue_root`` is an exported ``ctx`` field but the
    row's ``cas_root``/``residency`` keys are read directly rather than
    handed over by an SDK call. The clean shape is a PB-side ``ctx``
    field carrying the sealed readset, and until that exists this reads
    PB's own published row and nothing derived from it -- no sibling-path
    guess at the CAS root, no manifest path assembled from a convention.

    Raises :class:`ReadsetUnbound` with a reason whenever any hop is
    missing or disagrees. It never returns a partial answer.
    """
    try:
        _sdk_module, ctx = resolve_context(env=env)
    except TierPolicyRefused as refusal:
        raise ReadsetUnbound(f"lease context: {refusal}") from None
    queue_root = str(ctx.get("queue_root") or "")
    action_key = str(ctx.get("action_key") or "")
    if not queue_root or len(action_key) != 64:
        raise ReadsetUnbound("lease context names no queue row")
    row_path = Path(queue_root) / "claimed" / f"{action_key}.json"
    try:
        with open(row_path, "rb") as handle:
            row = json.loads(handle.read(_MAX_POOL_ROW_BYTES + 1).decode("utf-8"))
    except FileNotFoundError:
        raise ReadsetUnbound(f"no claim row at {row_path}") from None
    except (OSError, UnicodeError, ValueError) as error:
        raise ReadsetUnbound(f"claim row is unreadable: {error}") from None
    if not isinstance(row, dict) or row.get("action_key") != action_key:
        raise ReadsetUnbound("claim row names another action")
    cas_root = row.get("cas_root")
    residency = row.get("residency")
    if not isinstance(cas_root, str) or not cas_root:
        raise ReadsetUnbound("claim row publishes no cas_root")
    if not isinstance(residency, dict):
        raise ReadsetUnbound("claim row publishes no residency block")
    digest = residency.get("manifest_sha256")
    size = residency.get("manifest_bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ReadsetUnbound("residency block names no manifest digest")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReadsetUnbound("residency block names no manifest size")
    return cas_root, digest, size


def _load_sealed_payload(bound_manifest_sha256: str) -> dict:
    """PB's sealed data manifest for this action, decoded by PB's own reader.

    The declared-range authority, read from PrismaBuild's own request
    context and decoded by PrismaBuild's own validator:

    1. :func:`resolve_sealed_readset` gives the ``cas_root`` and manifest
       digest PB published into this action's pool row.
    2. That row's digest must equal ``bound_manifest_sha256`` -- the digest
       this run was submitted with. A readset for another submission is
       not this run's readset.
    3. Both the row's stated size and the blob's actual size are checked
       against ``core.DATA_MANIFEST_MAX_BYTES`` -- PB's own fixed bound --
       **before the blob is opened**, and only then is it read and hashed.
       The row's ``manifest_bytes`` is an input, not an established size:
       the same record carries ``detail.prewarm.manifest_bytes``, which is
       1,244,988,662,830 on the GLM-5.3-Flash campaign because it measures
       the payload those entries *describe*. What is hashed here is the
       small metadata -- ``residency.manifest_bytes``, 1.5 MB on that same
       campaign -- and the fixed ceiling is what keeps it that way.
       Content addressing is then what makes the manifest self-attesting:
       the bytes prove themselves or nothing is adopted.
    4. ``prismabuild.core.read_data_manifest`` decodes and validates it --
       gzip detected by header rather than suffix, stored and decoded bytes
       bounded independently before JSON parsing, trailing bytes and
       concatenated members refused, then PB's own schema validation. PQ
       does not re-implement any of that: a second, more permissive parser
       is a second contract.

    ``entries`` carry ``{path, offset, bytes}`` in **file offsets** -- the
    read-order caveat in ``core.residency_descriptor`` is about a
    descriptor's ``range_start_bytes``/``range_end_bytes``, not about
    entries, and the live campaign's entries 36447-36448 for
    ``model-00087-of-00120`` end at exactly that file's 5,354,098,040
    bytes. ``sha256`` may be null and is not read here.

    Raises :class:`ReadsetUnbound` on any missing or disagreeing hop.
    """
    cas_root, digest, size = resolve_sealed_readset()
    if digest != bound_manifest_sha256:
        raise ReadsetUnbound(
            f"the claim row names manifest {digest[:12]}, this run reads "
            f"{str(bound_manifest_sha256)[:12]}")
    try:
        from prismabuild.core import DATA_MANIFEST_MAX_BYTES, read_data_manifest
    except ImportError as error:
        raise ReadsetUnbound(f"PB manifest reader unavailable: {error}") from None
    # The ceiling is PB's own fixed bound, applied BEFORE anything is opened.
    # ``size`` came off the claim row: it is an input, not an established
    # fact, and the same record carries ``detail.prewarm.manifest_bytes`` --
    # 1,244,988,662,830 on the live campaign, the payload those entries
    # describe. A wrong field or a wrong value must not be able to spend a
    # read and a hash on a terabyte. Small metadata is what is hashed here,
    # and this is what keeps it small. (One bound, PB's: a second constant
    # here would be a second contract, free to drift from the reader that
    # enforces it.)
    if size > DATA_MANIFEST_MAX_BYTES:
        raise ReadsetUnbound(
            f"the claim row says the sealed manifest is {size} bytes, past "
            f"PrismaBuild's own {DATA_MANIFEST_MAX_BYTES}-byte manifest bound")
    blob = Path(cas_root) / "blobs" / digest[:2] / digest
    try:
        actual = os.lstat(blob).st_size
    except OSError as error:
        raise ReadsetUnbound(
            f"sealed manifest is unstatable: {error.strerror}") from None
    if actual > DATA_MANIFEST_MAX_BYTES:
        raise ReadsetUnbound(
            f"sealed manifest is {actual} bytes, past PrismaBuild's own "
            f"{DATA_MANIFEST_MAX_BYTES}-byte manifest bound")
    try:
        with open(blob, "rb") as handle:
            raw = handle.read(size + 1)
    except OSError as error:
        raise ReadsetUnbound(
            f"sealed manifest is unreadable: {error.strerror}") from None
    if len(raw) != size:
        raise ReadsetUnbound(
            f"sealed manifest is {len(raw)} bytes, the claim row says {size}")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ReadsetUnbound("sealed manifest does not hash to its digest")
    try:
        payload, _encoding = read_data_manifest(blob)
    except Exception as error:
        raise ReadsetUnbound(f"PB refused the sealed manifest: {error}") from None
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ReadsetUnbound("sealed manifest declares no entries")
    return payload


def load_sealed_readset(bound_manifest_sha256: str) -> dict[str, list[tuple[int, int]]]:
    """PB's sealed readset as ``{declared path: merged [start, end) spans}``.

    Read through :func:`_load_sealed_payload`, whose docstring states every
    hop and bound.
    """
    payload = _load_sealed_payload(bound_manifest_sha256)
    entries = payload["entries"]
    spans: dict[str, list[tuple[int, int]]] = {}
    for row in entries:
        path, offset, count = row["path"], row["offset"], row["bytes"]
        spans.setdefault(str(path), []).append((int(offset), int(offset) + int(count)))
    return spans


def load_sealed_read_order(bound_manifest_sha256: str) -> list[tuple[str, int, int]]:
    """The sealed manifest's entries in the order the action reads them.

    ``[(path, file offset, bytes), ...]``, in read order: the order whose
    running byte sum PrismaBuild's movers, plans and landing records name
    ranges in (PB ``core.residency_descriptor``). Cut by PB's own
    ``storage_tiers.manifest_read_entries`` -- list order for a v1 manifest,
    the ``read_plan`` expansion for v2, revisits included -- so this reader
    and the plan agree about one order by construction rather than by a
    second parser. The module comes through :func:`sdk_submodule`, from the
    same sealed generation as the lease SDK, never from whatever
    ``prismabuild`` a bare import would find. Raises :class:`ReadsetUnbound`
    when PB's helper is not available or the manifest describes no read
    order.
    """
    payload = _load_sealed_payload(bound_manifest_sha256)
    try:
        tiers = sdk_submodule("storage_tiers")
        manifest_read_entries = tiers.manifest_read_entries
    except (LeaseRefused, AttributeError) as error:
        raise ReadsetUnbound(f"PB read-order helper unavailable: {error}") from None
    entries = manifest_read_entries(payload)
    if not entries:
        raise ReadsetUnbound("PB's read order does not describe the sealed manifest")
    return [(str(row["path"]), int(row["offset"]), int(row["bytes"]))
            for row in entries]


def _material_consumer(identity, ctx) -> str:
    """Whose fragments vouch this window: the owner, or a named namespace.

    PrismaBuild splits OWNER from MATERIAL NAMESPACE for produced output:
    the owner is the running action that holds the pin, the namespace is
    the producing consumer whose fragments and sidecars vouch the bytes.
    For every ordinary input read they are the same string, which is why
    ``covers_for_keys`` and ``acquire_for`` both default the namespace to
    the owner and why this returns the action key unchanged there. A
    resolver built for one produced batch names its namespace explicitly,
    and it is never inferred from a path or from a terminal record.
    """

    namespace = identity.get("material_namespace")
    if namespace:
        return str(namespace)
    return str(ctx["action_key"])


def _namespace_spec(identity) -> dict:
    """The window-spec fields a produced-output read needs, or nothing.

    Empty for an input map, so the window a sealed input builds is
    byte-identical to the one it built before produced output existed.
    """

    namespace = identity.get("material_namespace")
    if not namespace:
        return {}
    return {"material_namespace": str(namespace),
            "residency_root": str(identity["residency_root"])}


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
            identity["residency_root"], _material_consumer(identity, ctx),
            [key],
            tier_id=ram_tier, manifest_sha256=identity["manifest_sha256"],
            epoch=ram_epoch, context=_call_context())
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


def resolve_stage_covers(resolver, declared, entry):
    """Stage-tier covers via the PB cover lookup: minimal, per requested key.

    The composed map's leads are heritage, not a covering set: an egress
    legitimately retires a lead's fragment and material once its phase is
    read past, and ``acquire`` requires every named cover to exist — so
    naming all leads let one retired, unrelated mover poison any later
    entry with ``unpublished``. ``covers_for_keys`` resolves the minimal
    covering movers for the ACTUAL requested key from the current
    consumer's fragments plus publish-time sidecars, bounded to this
    consumer, manifest, tier and the SSD empty epoch; PQ cross-checks the
    helper's expected bytes/digest against the sealed composed-map entry
    before adopting anything — a divergent proof fails clear as
    integrity. Selection only, exactly like the RAM leg: the SDK's
    ownership-lock acquire stays the admission authority, and movers are
    never invented from paths.
    """
    from .residency_map import residency_map_key
    sdk, ctx = resolve_context()
    identity = resolver.lease_identity()
    stage_tier = identity["tier_id"]
    if not stage_tier:
        raise _refuse("stage-not-announced", kind="availability")
    key = residency_map_key(str(declared), entry["offset"])
    if not hasattr(sdk, "covers_for_keys"):
        raise _refuse("stage-covers-unresolved: no cover lookup",
                      kind="availability")
    try:
        answer = sdk.covers_for_keys(
            identity["residency_root"], _material_consumer(identity, ctx),
            [key],
            tier_id=stage_tier, manifest_sha256=identity["manifest_sha256"],
            epoch="", context=_call_context())
    except Exception as exc:
        raise _refuse(f"stage-cover-lookup-error: {exc}",
                      kind="integrity") from None
    if not isinstance(answer, dict) or not answer.get("ok"):
        refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
        raise _refuse(str(refusal), kind=_classify(refusal))
    expected = answer.get("expected")
    got = expected.get(key) if isinstance(expected, dict) else None
    if (not isinstance(got, dict) or got.get("bytes") != entry["bytes"]
            or str(got.get("sha256") or "") != str(entry["sha256"] or "")):
        raise _refuse("stage-cover-proof-divergent", kind="integrity")
    covers = answer.get("covers")
    if (not isinstance(covers, list) or not covers
            or any(not isinstance(cover, dict)
                   or len(str(cover.get("mover_action_key") or "")) != 64
                   for cover in covers)):
        raise _refuse("stage-cover-proof-divergent", kind="integrity")
    return covers, key


def stage_cover_is_published(resolver, declared, entry) -> bool:
    """Whether PrismaBuild has published the proof for a staged map entry.

    A mover writes its fragment first and the material sidecar that dates it
    second, and the composed map is built from fragments. A reader polling
    for a row can therefore see it before its sidecar exists, and a lease
    asked in that window answers ``unpublished``. That is a range that has
    not finished landing, which is what the readiness wait is for.

    Selection only, nothing is pinned. False for ``unpublished`` alone; every
    other refusal -- integrity, a missing helper, a tier that is not
    announced -- is True here, because waiting cannot improve it and the read
    that follows owns the refusal.
    """
    try:
        resolve_stage_covers(resolver, declared, entry)
    except LeaseRefused as refusal:
        return refusal.reason.split(":", 1)[0] != "unpublished"
    return True


def _select_ram_window(resolver, declared, entry):
    """Build (unentered) a RAM-tier lifetime window for one entry.

    Pins at the announced RAM epoch with resolved RAM covers; the payload
    then serves from the tmpfs copy. Any refusal carries its
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
        **_namespace_spec(identity),
    })
    return window, key


def _resolve_window_covers(resolver, items, *, tier_id: str, epoch: str,
                           label: str):
    """Covers and map keys for several entries, in one PB cover lookup.

    The per-entry lookups above (:func:`resolve_ram_covers`,
    :func:`resolve_stage_covers`) each scan the consumer's material
    directory and read every mover's documents, so a 64-entry window paid
    that scan 64 times (PQ #997: 11% of Stage A R12's main thread).
    ``covers_for_keys`` takes the window's whole key list, which is the
    granularity PrismaBuild batches at. Every key's expected bytes and
    digest are still cross-checked against its own sealed map entry, and
    the context stays per call (PQ #905).

    ``items`` is ``[(declared, entry), ...]`` from one resolver. Returns
    ``(covers, keys)`` with ``keys[i]`` the map key of ``items[i]``.
    """
    from .residency_map import residency_map_key
    sdk, ctx = resolve_context()
    identity = resolver.lease_identity()
    if not tier_id:
        raise _refuse(f"{label}-not-announced", kind="availability")
    if not hasattr(sdk, "covers_for_keys"):
        raise _refuse(f"{label}-covers-unresolved: no cover lookup",
                      kind="availability")
    keys = [residency_map_key(str(declared), entry["offset"])
            for declared, entry in items]
    if len(set(keys)) != len(keys):
        raise _refuse(f"{label}-window-repeats-a-key", kind="integrity")
    try:
        answer = sdk.covers_for_keys(
            identity["residency_root"], _material_consumer(identity, ctx),
            list(keys), tier_id=tier_id,
            manifest_sha256=identity["manifest_sha256"], epoch=epoch,
            context=_call_context())
    except Exception as exc:
        raise _refuse(f"{label}-cover-lookup-error: {exc}",
                      kind="integrity") from None
    if not isinstance(answer, dict) or not answer.get("ok"):
        refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
        raise _refuse(str(refusal), kind=_classify(refusal))
    expected = answer.get("expected")
    for key, (_declared, entry) in zip(keys, items):
        got = expected.get(key) if isinstance(expected, dict) else None
        if (not isinstance(got, dict) or got.get("bytes") != entry["bytes"]
                or str(got.get("sha256") or "") != str(entry["sha256"] or "")):
            raise _refuse(f"{label}-cover-proof-divergent", kind="integrity")
    covers = answer.get("covers")
    if (not isinstance(covers, list) or not covers
            or any(not isinstance(cover, dict)
                   or len(str(cover.get("mover_action_key") or "")) != 64
                   for cover in covers)):
        raise _refuse(f"{label}-cover-proof-divergent", kind="integrity")
    return covers, keys


def acquire_entries_window(resolver, items, *, tier: str):
    """Build (but do not enter) ONE lifetime window for several entries.

    The multi-entry form of :func:`acquire_entry_window`, for a read window
    whose entries all resolve through one resolver (one material
    namespace) on one tier. PrismaBuild's ``acquire`` pins a window's whole
    key set under one ownership-lock hold and ``release`` drops it under
    one more, so a 64-entry window takes the stage-root lock twice instead
    of 128 times (PQ #997: lock waits were 10.5% of Stage A R12's main
    thread, and six other actions contend for the same NFS lock).

    ``tier`` is ``"ram"`` (every entry offers a RAM copy; pins at the
    announced RAM epoch) or ``"ssd"``. Refusals carry their kind exactly as
    the single-entry legs do. The caller owns fallback: this never tries
    another tier, so a caller that must keep the per-entry RAM-first
    semantics retries refused entries one at a time.

    Returns ``(window, keys)``; the caller opens each key once, closes each
    descriptor, and exits the window once on every path. The pin records
    the first entry's span, since a multi-entry window carries each key's
    range through its map key and never one cumulative source offset (the
    SDK's own range contract).
    """
    from .staged_tier_policy import tier_is_allowed
    items = list(items)
    if not items:
        raise ValueError("a lease window needs at least one entry")
    identity = resolver.lease_identity()
    if tier == "ram":
        if not tier_is_allowed("ram"):
            raise _refuse("ram-not-allowed", kind="availability")
        if any(entry.get("ram_path") is None for _declared, entry in items):
            raise _refuse("ram-not-offered", kind="availability")
        tier_id, epoch = identity["ram_tier_id"], identity["ram_epoch"]
        if not tier_id or not epoch:
            raise _refuse("ram-not-announced", kind="availability")
        covers, keys = _resolve_window_covers(
            resolver, items, tier_id=tier_id, epoch=epoch, label="ram")
    elif tier == "ssd":
        if not tier_is_allowed("ssd"):
            raise _refuse("ssd-not-allowed", kind="availability")
        tier_id, epoch = identity["tier_id"], ""
        covers, keys = _resolve_window_covers(
            resolver, items, tier_id=tier_id, epoch=epoch, label="stage")
    else:
        raise ValueError(f"unknown lease tier {tier!r}")
    first = items[0][1]
    window = LeaseWindow({
        "tier_id": tier_id,
        "epoch": epoch,
        "covers": covers,
        "expected": {key: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                     for key, (_declared, entry) in zip(keys, items)},
        "span": {"start_bytes": first["offset"],
                 "end_bytes": first["offset"] + first["bytes"]},
        **_namespace_spec(identity),
    })
    return window, keys


def stage_covers_are_published(resolver, items) -> bool | None:
    """Whether PrismaBuild published the proof for every staged entry, at once.

    The batched form of :func:`stage_cover_is_published`: one cover lookup
    for the whole list. ``True`` when every key is proven; ``None`` when
    the batched answer cannot say which key is missing, and then the caller
    asks :func:`stage_cover_is_published` one entry at a time. Selection
    only, like the single form: nothing is pinned, and the lease that
    follows re-verifies every key under the ownership lock.
    """
    identity = resolver.lease_identity()
    try:
        _resolve_window_covers(resolver, items, tier_id=identity["tier_id"],
                               epoch="", label="stage")
    except LeaseRefused:
        return None
    return True


def ram_covers(map_entry: dict) -> list[dict[str, str]]:
    """RAM-tier covers without a lookup: always refused.

    Retained for SDK pins predating `covers_for_keys`; current code
    resolves via :func:`resolve_ram_covers` instead. Direct callers get
    the honest refusal: SSD leads never identify a RAM mover.
    """
    raise _refuse("ram-covers-unresolved", kind="availability")


def acquire_entry_window(resolver, declared, entry: dict):
    """Build (but do not enter) a lifetime window for one composed-map entry.

    One unambiguous contract: the returned window is fresh and unentered;
    the caller enters exactly once (``with`` for function scope, manual
    enter/exit for bound lifetimes) and exits exactly once on every path.
    Entering an entered or released window refuses — reuse and nesting
    are programming errors, never silent reacquisition.

    RAM first where offered and allowed: the RAM leg resolves real
    RAM-mover covers and pins at the announced epoch, so a live tmpfs
    copy serves with its own material and lifetime — the full goal, not
    a fallback. An availability refusal on the RAM leg records and falls
    through to SSD; an integrity refusal propagates with no alternate
    adoption. SSD acquires with minimal per-key covers resolved through
    the PB cover lookup — never the whole lead set, which a legitimately
    retired, unrelated mover would poison into ``unpublished``.
    ``ssd`` outside the allowed tiers refuses.

    Enter-time races fail this window clearly; it never re-enters or
    silently changes covers. A source reader may handle a typed retiring
    refusal by selecting another covering entry and entering a NEW window
    for it. The selected entry's pin and descriptor own the full read
    lifetime; unknown and integrity refusals still fail closed.
    """
    from .staged_tier_policy import tier_is_allowed
    if entry.get("ram_path") is not None and tier_is_allowed("ram"):
        try:
            return _select_ram_window(resolver, declared, entry)
        except LeaseRefused as refusal:
            if refusal.kind != "availability":
                raise
            resolver.record_ram_fallback(declared, str(refusal))
    if not tier_is_allowed("ssd"):
        raise _refuse("ssd-not-allowed", kind="availability")
    identity = resolver.lease_identity()
    covers, key = resolve_stage_covers(resolver, declared, entry)
    window = LeaseWindow({
        "tier_id": identity["tier_id"],
        "epoch": "",
        "covers": covers,
        "expected": {key: {"bytes": entry["bytes"],
                           "sha256": entry["sha256"]}},
        "span": {"start_bytes": entry["offset"],
                 "end_bytes": entry["offset"] + entry["bytes"]},
        **_namespace_spec(identity),
    })
    return window, key


#: Backoff between release attempts: doubles from the first delay up to the
#: cap. Neither changes what a retry can do (release is idempotent); they
#: only set how soon a transient clears is noticed.
_RELEASE_RETRY_FIRST_S = 0.05
_RELEASE_RETRY_CAP_S = 5.0


def _mount_options(path: str, mountinfo: str = "/proc/self/mountinfo"):
    """``(fstype, {option: value})`` of the mount holding ``path``, or ``None``.

    The longest mount point that is a path prefix wins, as the kernel
    resolves it. Super options (after the `` - `` separator) carry the NFS
    ``timeo``/``retrans`` values; per-mount options are merged under them.
    """
    target = os.path.abspath(path)
    best = None
    try:
        with open(mountinfo, encoding="utf-8") as stream:
            rows = stream.read().splitlines()
    except OSError:
        return None
    for row in rows:
        head, sep, tail = row.partition(" - ")
        fields = head.split()
        if not sep or len(fields) < 6:
            continue
        point = fields[4].replace("\\040", " ")
        if not (target == point or target.startswith(point.rstrip("/") + "/")
                or point == "/"):
            continue
        tail_fields = tail.split()
        if len(tail_fields) < 3:
            continue
        options = {}
        for text in (fields[5], tail_fields[2]):
            for item in text.split(","):
                name, _eq, value = item.partition("=")
                options[name] = value
        if best is None or len(point) > len(best[0]):
            best = (point, tail_fields[0], options)
    if best is None:
        return None
    return best[1], best[2]


def release_retry_horizon_s(path: str, mountinfo: str = "/proc/self/mountinfo") -> float:
    """How long a failed pin release is retried: the mount's NFS major timeout.

    PB's ``reader_lease.release`` returns ``False`` -- never the reason --
    when it cannot read, rewrite or unlink a pin. On the fleet's NFS mount
    such a failure can be transient: R13 (PB action ``556d7a803098``) died on
    one in forward-002 on 2026-09-23 after hundreds of clean releases, and
    the pin it left read and validated cleanly afterwards.

    The horizon is the one the mount itself already treats as transient:
    ``timeo`` (deciseconds) x (``retrans`` + 1) is how long the NFS client
    retries one request before it reports the server as not responding. A
    pin on any other filesystem has no such horizon -- a failure there is not
    transient -- so it is not retried (``0``).
    """
    found = _mount_options(path, mountinfo)
    if found is None or not found[0].startswith("nfs"):
        return 0.0
    options = found[1]
    try:
        timeo_s = int(options.get("timeo", "")) / 10.0
        retrans = int(options.get("retrans", ""))
    except ValueError:
        return 0.0
    if timeo_s <= 0 or retrans < 0:
        return 0.0
    return timeo_s * (retrans + 1)


def _observe_pin(sdk, queue, consumer, pin_id, ref_id, where) -> str:
    """What the pin file shows right now, for a release-failure record.

    PB's release does not say which step failed; reading the pin the way it
    does recovers the errno when the read is what failed, and shows whether
    the ref is still held when a rewrite or unlink failed.
    """
    leases_root = getattr(sdk, "leases_root", None)
    if leases_root is None:
        return "unobservable: the SDK names no leases_root"
    try:
        root = Path(leases_root(queue, where.get("residency_root")))
        path = root / str(consumer) / f"{pin_id}.lease.json"
    except Exception as exc:  # noqa: BLE001 -- diagnostics only
        return f"unobservable: {type(exc).__name__}: {exc}"
    try:
        with open(path, encoding="utf-8") as stream:
            pin = json.load(stream)
    except FileNotFoundError:
        return f"{path}: absent"
    except OSError as exc:
        return f"{path}: unreadable: {type(exc).__name__} errno {exc.errno}: {exc}"
    except ValueError as exc:
        return f"{path}: unparseable: {exc}"
    refs = pin.get("refs") if isinstance(pin, dict) else None
    if not isinstance(refs, dict):
        return f"{path}: present, no refs object"
    held = "held" if ref_id in refs else "gone"
    return f"{path}: present, {len(refs)} ref(s), this ref {held}"


class LeaseWindow:
    """One bounded read window, pinned for exactly its holder's lifetime.

    ``spec`` names the window: ``tier_id``, ``epoch``, ``covers``,
    ``expected`` (map key -> ``{bytes, sha256}``), ``span``
    (``{start_bytes, end_bytes}`` in entry coordinates), and optional
    ``ram`` passthrough. ``acquire_token`` defaults to a fresh uuid4;
    pass it back to retry idempotently onto the same ref.

    Single-shot lifetime: enter exactly once, exit exactly once.
    Re-entering (nested or after exit) and re-exiting refuse as
    programming errors — reuse is never silent reacquisition. Exiting a
    never-entered window is a finally-safe no-op.

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
        self._entered = False
        self._exited = False
        self._fds: dict[int, str] = {}
        self._released = False

    def _require_owner(self, operation: str) -> None:
        """Reject inherited handle operations in a forked child.

        Supported readers use threads and owned buffers; an operation
        issued from another pid would act on inherited handles outside
        the holder's lifecycle, so it fails loudly instead. Raw
        descriptor escape past these operations is unsupported (not
        policed here): fork with no live window and `_exit` without I/O.
        """
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            raise RuntimeError(
                f"LeaseWindow.{operation} from a forked child is "
                "unsupported: inherited handles must not be operated "
                "past the holder's lifecycle — the parent releases them "
                "on its own schedule. Use threads, which share the "
                "holder pid, or fork with no live window.")

    # -- acquire ------------------------------------------------------

    def __enter__(self) -> "LeaseWindow":
        if self._exited:
            raise RuntimeError(
                "LeaseWindow re-enter after exit is refused: a released "
                "manager is never reused — acquire a fresh window per read.")
        if self._entered:
            raise RuntimeError(
                "LeaseWindow nested/reentrant enter is refused: one enter "
                "per window, no silent reacquisition.")
        if self._helper_root is not None:
            set_lease_helper_root(self._helper_root)
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            self._require_owner("__enter__")
        sdk, ctx = resolve_context(env=self._env)
        # Provenance BEFORE side effects: the serving module proves the
        # expected package root; pool plus every preimported prismabuild.*
        # (and the package itself) must resolve inside it, component-wise.
        # A divergent preimport refuses here — before any pin exists, so no
        # `with` teardown is needed and nothing can strand.
        expected = _package_dir_of(sdk)
        try:
            import prismabuild.pool as pool_mod  # noqa: PLC0415
        except ImportError as exc:
            raise _refuse(f"lease-helper-unavailable: {exc}",
                          kind="availability") from None
        _check_package_coherence(expected)
        spec = self._spec
        # Owner and material namespace are the same action for every input
        # read, so these stay absent there and the call is byte-identical
        # to the one this window made before produced output existed. A
        # produced-output window names both: the pin files under the owner
        # from ``ctx`` either way, and only the proof resolves in the
        # batch's namespace.
        namespace: dict = {}
        if spec.get("material_namespace"):
            namespace = {"material_namespace": str(spec["material_namespace"]),
                         "residency_root": str(spec["residency_root"])}
            if not _sdk_accepts_material_namespace(sdk):
                raise _refuse(
                    "produced-namespace-unsupported: the pinned reader-lease "
                    "SDK has no material_namespace parameter, so a produced "
                    "batch's material cannot be vouched without treating an "
                    "output namespace as the running action",
                    kind="availability")
        answer = sdk.acquire_for(
            ctx, tier_id=str(spec["tier_id"]), epoch=str(spec["epoch"]),
            covers=spec["covers"], expected=spec.get("expected"),
            span=spec["span"], acquire_token=self._token,
            ram=spec.get("ram"), context=_call_context(), **namespace)
        if not isinstance(answer, dict) or not answer.get("ok"):
            refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
            if str(refusal).split(":", 1)[0] == "generation-changed":
                answer = sdk.acquire_for(
                    ctx, tier_id=str(spec["tier_id"]), epoch=str(spec["epoch"]),
                    covers=spec["covers"], expected=spec.get("expected"),
                    span=spec["span"], acquire_token=self._token,
                    ram=spec.get("ram"), context=_call_context(),
                    **namespace)
                if not isinstance(answer, dict) or not answer.get("ok"):
                    refusal = answer.get("refusal", "unknown") if isinstance(answer, dict) else "unknown"
            if not isinstance(answer, dict) or not answer.get("ok"):
                raise _refuse(str(refusal), kind=_classify(refusal))
        # Successful acquire: bind state without reporting entry until the
        # very end. Any failure below releases exactly the just-acquired
        # ref (ids known) or retains the idempotent token for an explicit
        # retry (ids unknown) — never a silent strand, never entered=True.
        pin_id = ref_id = consumer = queue_root = None
        pin = None
        try:
            pin = answer["pin"]
            pin_id = str(answer["pin_id"])
            ref_id = str(answer["ref_id"])
            consumer = str(ctx["action_key"])
            queue_root = str(ctx["queue_root"])
            self._pin = pin
            self._pin_id = pin_id
            self._ref_id = ref_id
            self._consumer = consumer
            self._queue_root = queue_root
            self._pool_mod = pool_mod
            self._pid = os.getpid()
            self._owner_pid = os.getpid()
        except Exception as exc:
            if pin_id is not None and ref_id is not None:
                try:
                    queue = pool_mod.PoolQueue(queue_root)
                    released = sdk.release(
                        queue, pin_id, ref_id,
                        consumer_action_key=consumer)
                except Exception as rel_exc:
                    raise _refuse(
                        f"lease-enter-failed: {exc}; cleanup-error: {rel_exc}",
                        kind="integrity") from None
                if released is not True:
                    raise _refuse(
                        f"lease-enter-failed: {exc} (pin retained for retry)",
                        kind="integrity") from None
                self._pin = None
                self._pin_id = None
                self._ref_id = None
            raise _refuse(f"lease-enter-failed: {exc}",
                          kind="integrity") from None
        self._entered = True
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
        # The pin lives under the residency root it was ACQUIRED in. A
        # produced-output window acquires under the produced-output
        # fragment root, so opening without naming it reads the tier's
        # default root, finds nothing, and refuses "pin is not live" for a
        # pin that is perfectly live one directory over.
        where = ({"residency_root": str(self._spec["residency_root"])}
                 if self._spec.get("material_namespace") else {})
        try:
            fd, serving = sdk.open_pinned(queue, self._pin, self._ref_id,
                                          str(key), **where)
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
        # Exiting a never-entered window is a finally-safe no-op;
        # re-exiting a released window refuses (no reuse). A forked-child
        # teardown is refused loudly by close_fds below: a child unwinding
        # a parent's window must not close inherited descriptors or
        # release the parent's ref.
        if not self._entered:
            return False
        if self._exited:
            raise RuntimeError(
                "LeaseWindow re-exit is refused: a released manager is "
                "never reused.")
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
            # State unchanged: a failed teardown may be retried by the
            # holder; only a completed exit marks the manager released.
            raise failure
        self._exited = True
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
        # The pin lives under its OWNER directory either way, but a produced
        # window's owner directory is under the produced-output fragment
        # root, not the tier's default residency root -- so the release names
        # the root it acquired in rather than letting the SDK fall back to a
        # full scan of the wrong one.
        where = ({"residency_root": str(self._spec["residency_root"])}
                 if self._spec.get("material_namespace") else {})
        # The SDK returns False (never an exception, never the reason) for a
        # pin it could not read, rewrite or unlink: only an exact True
        # releases. ``release`` is idempotent -- a ref already dropped
        # answers True -- so a failed attempt is retried, with its record,
        # for as long as the mount itself treats a fault as transient
        # (:func:`release_retry_horizon_s`). Past that, or at once on a
        # filesystem with no such horizon, the exit refuses and keeps full
        # retry state, so a failed exit never silently strands a ref.
        deadline = None
        delay = _RELEASE_RETRY_FIRST_S
        attempt = 0
        while True:
            attempt += 1
            error = None
            try:
                released = sdk.release(queue, self._pin_id, self._ref_id,
                                       consumer_action_key=self._consumer,
                                       **where)
            except OSError as exc:
                released, error = None, exc
            except Exception as exc:
                raise LeaseRefused(f"lease-release-failed: {exc}",
                                   kind="integrity") from exc
            if released is True:
                break
            if deadline is None:
                deadline = time.monotonic() + release_retry_horizon_s(
                    str(self._queue_root))
            observed = _observe_pin(sdk, queue, self._consumer, self._pin_id,
                                    self._ref_id, where)
            answer = (f"{type(error).__name__}: {error}" if error is not None
                      else f"returned {released!r}")
            remaining = deadline - time.monotonic()
            print(f"[staged-lease] release of pin {self._pin_id} ref "
                  f"{self._ref_id} attempt {attempt} {answer}; pin: {observed}; "
                  + (f"retrying for up to {remaining:.1f} s more"
                     if remaining > 0 else "not retrying"),
                  flush=True)
            if remaining <= 0:
                if error is not None:
                    raise LeaseRefused(f"lease-release-failed: {error}",
                                       kind="integrity") from error
                raise LeaseRefused(
                    f"lease-release-failed: pin not released ({observed})",
                    kind="integrity")
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _RELEASE_RETRY_CAP_S)
        self._released = True
