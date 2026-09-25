"""Stage A's own boundary entries, staged through PrismaBuild produced output.

The live failure this closes: an admitted Stage A action writes its exact
boundary/cotangent entries with :func:`prismaquant.perturbed_x_cache.
write_exact_activation_cache_entry` and then reads them back through the
strict allowed-tier path, which resolves the process residency map -- the
map of the run's *sealed inputs*.  Its own just-written outputs are not in
that map, so ``ResidencyResolver.staged_read`` misses and the read refuses
``staged-not-serving``.  There is no own-session exemption: the physical
tier policy is not waived for bytes this action produced.

The fix is the one PrismaBuild already owns.  An output entry becomes a
readable staged object by going through the produced-output lifecycle:
``require_prewrite`` (durable-class budget, BEFORE the first byte) -> the
existing writer, writing its unchanged canonical name -> one descriptor per
entry carrying the writer's OWN inline digest -> ``publish_prepaid_batch``
(PB seals a real mover off this action's own sealed request and funds it by
exact transfer from this owner's window) -> the mover stages the bytes and
files a residency fragment under the BATCH NAMESPACE -> this adapter
composes that namespace's fragments into a supplemental reader context ->
the unchanged strict prefetch reads through it -> ``retire_batch`` frees the
stage window and ``reclaim_origin`` frees the durable charge.

Three boundaries this module does not cross:

* **It is not a second cache, dispatcher or scheduler.**  PB owns movement
  and placement.  This composes published records and calls published
  entry points; the polling, the copy and the eviction are PB's.
* **It never rehashes a payload.**  ``write_exact_activation_cache_entry``
  already computes the serialized digest inline (``SerializedEntryDigest``)
  and hands it back on the reference.  That exact digest is what the
  descriptor carries and what the strict reader demands, so the producer's
  receipt, PB's manifest and the reader's fence are one number.  No sealing
  pass is added and no file is read a second time.
* **It never swaps the run's input map.**  The supplemental reader context
  is an explicitly namespaced batch resolver, bound to the batch's own
  manifest digest and material namespace.  The process resolver
  (:func:`prismaquant.residency_map.residency_resolver`) and every
  read-only or foreign attached generation keep resolving exactly as
  before: they are ordinary input-map entries and cannot declare an owner
  prewrite.

Binding is derived, never operator-supplied, exactly as the render lane's
publication derives it: the owner action key, nonce and scope come from the
launch environment PB injected for THIS attempt; the template comes from the
submission's own ``--produced-output-template`` declaration; the instance
comes from ``bind_declared_instance`` against the live claim.

The bounded cycle is closed: a group is published once, staged once,
read, and its STAGE COPY released -- and a later read of the same unchanged
logical batch goes through PrismaBuild's own repeat-materialization surface
(``produced_output.materialization_state`` to ask, then
``produced_output.ensure_batch_materialized`` only when the answer says the
copy is gone).  One logical batch, one durable origin charge, across the
forward and reverse cycle; the successor's mover and funding key are
PrismaBuild's, sealed over the filed materialization generation, never a
caller nonce.  A pinned candidate without those two entry points refuses
loudly (:class:`BoundaryRepeatMaterializationUnsupported`); it never
degrades to a new batch id or an origin read.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class BoundaryProducedBindingError(RuntimeError):
    """The boundary publication could not be bound or used."""


class BoundaryProducedPrewriteRefused(RuntimeError):
    """The prewrite budget refused this group BEFORE any byte.

    Carries the typed refusal; no boundary entry was written.
    """

    def __init__(self, *, batch_id: str, refusal: Any) -> None:
        self.batch_id = batch_id
        self.refusal = refusal
        super().__init__(
            f"produced-output prewrite refused for boundary group "
            f"{batch_id!r} before any byte was written: {refusal}")


class BoundaryProducedPublicationFailed(RuntimeError):
    """The entries are durable but the batch publication refused.

    Retryable: identical descriptors re-derive the same content-addressed
    mover and every step answers a typed duplicate.
    """

    def __init__(self, *, batch_id: str, refusal: Any) -> None:
        self.batch_id = batch_id
        self.refusal = refusal
        super().__init__(
            f"produced-output boundary group {batch_id!r} failed to "
            f"publish (entries are durable; retry re-derives the mover): "
            f"{refusal}")


class BoundaryMaterializationIncomplete(RuntimeError):
    """The batch's mover has not staged the whole group yet.

    Not a refusal to read and not a fallback: the caller waits for PB's own
    records to say the batch is staged, or fails its own way.  Nothing here
    reads the pool.
    """


#: The Stage A boundary publication's template identity and its one slot.
#: The slot name is what a bound owner's descriptors declare, so it is a
#: contract between the pre-submit declaration and the runtime writer.
BOUNDARY_TEMPLATE_ID = "pq-stagea-boundary-entries-v1"
BOUNDARY_SLOT = "boundary_entries"

#: PrismaBuild's own tier unit. Quoted, not assumed: `storage_tiers`
#: prices a reservation per mover and rounds UP to a whole token, so every
#: derivation here uses the same ceiling rather than a nearest or a floor.
GIB = 1 << 30

#: The writer's own per-entry envelope: `StreamedBoundaryArtifacts.write`
#: bounds an entry at ``nbytes + 65536`` for the PyTorch zip header, and
#: the prewrite ceiling is that same bound so the two cannot drift.
HEADER_ENVELOPE_BYTES = 65536


def boundary_group_ceiling_bytes(*, group_size: int,
                                 max_entry_tensor_bytes: int,
                                 header_envelope_bytes: int = HEADER_ENVELOPE_BYTES
                                 ) -> int:
    """The conservative payload ceiling of ONE publication group.

    The group is the existing read window (``prefetch_batches`` entries),
    and its ceiling is the writer's own per-entry bound times the entries
    the group actually holds -- so the LAST, partial group is priced on
    its own byte range and never on an assumed full window.
    """

    for name, value in (("group_size", group_size),
                        ("max_entry_tensor_bytes", max_entry_tensor_bytes),
                        ("header_envelope_bytes", header_envelope_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"boundary group {name} must be a positive int")
    return group_size * (max_entry_tensor_bytes + header_envelope_bytes)


def boundary_window_gib(*, group_size: int, max_entry_tensor_bytes: int,
                        concurrent_groups: int = 2,
                        header_envelope_bytes: int = HEADER_ENVELOPE_BYTES
                        ) -> int:
    """The tier window this producer reserves, in PrismaBuild tokens.

    Derived from the ACTUAL MAXIMUM GROUP, never from the retained origin
    peak -- two different quantities, and conflating them sizes the window
    off the wrong number entirely. The origin peak is how many durable
    bytes the whole capture keeps (hundreds of GiB); the window is how much
    STAGE space the bounded reader borrows at once, which is the groups
    that can be live together.

    ``concurrent_groups`` defaults to 2 because that is what one window
    needs: a window may not overlap another (``cost_streaming`` refuses it),
    and the read window plus the incoming cotangent plane are the two groups
    a single window can span. Each is priced with PrismaBuild's own
    per-mover ceiling, ``ceil(bytes / GiB)``, so the production geometry --
    64 entries of 16 MiB plus envelope -- is 2 tokens a group and 4 for the
    pair.

    A larger count funds read-ahead (#887). The bound owner derives the
    count back from the sealed ``window_gib`` and treats every group past
    two as credit for staging work it has not been asked for yet; at two it
    is the synchronous loop, unchanged.
    """

    if type(concurrent_groups) is not int or concurrent_groups <= 0:
        raise ValueError("boundary concurrent_groups must be a positive int")
    per_group = boundary_group_ceiling_bytes(
        group_size=group_size,
        max_entry_tensor_bytes=max_entry_tensor_bytes,
        header_envelope_bytes=header_envelope_bytes)
    return -(-per_group // GIB) * concurrent_groups


def build_boundary_template(*, output_prefix, tier: str,
                            artifact_max_bytes: int,
                            group_size: int, max_entry_tensor_bytes: int,
                            checkpoint_max_bytes: int | None = None,
                            concurrent_groups: int = 2,
                            template_id: str = BOUNDARY_TEMPLATE_ID,
                            write_only: bool = False,
                            export_rate_family: str | None = None) -> dict:
    """The pre-submit produced-output template for a Stage A capture.

    Sealed BEFORE submission (it is an input of the owner's own action) and
    bound after admission, so every number in it comes from geometry the
    submitter already has.

    ``artifact_max_bytes`` is THE configured artifact max -- the plan's
    sealed ``boundary_storage.max_artifact_bytes`` or the run's explicit
    override, whichever this invocation will actually run under -- and it
    is the single source of the durable origin class maximum. Nothing here
    carries a default budget, and no figure is written into this module:
    a planning number that can move is an argument, not a constant.

    The temp class matches the payload class because every final passes
    through a staging file of its own size before the rename; stating the
    relationship beats leaving it a coincidence of defaults.

    ``write_only`` declares outputs the owner's own action never reads
    again (PrismaBuild #912): the band-serial handoff (PQ #1075). Such a
    template reserves no stage window, so every tier's minimum and window
    are zero and ``concurrent_groups`` does not apply. Its owner commits
    each group at its origin (``commit_origin_batch``), and a later action
    stages the group as an ordinary input.

    ``export_rate_family`` (PrismaBuild #1126, PQ #1254) names the templates
    whose spool exports share one learned export rate on a host. A template
    per row is never measured on its own, so the band-serial handoff's
    templates declare one family and each row starts from what the earlier
    rows' exports measured. ``None`` declares none, and the template's bytes,
    and so its id, are what they were before the field existed.
    """

    po = _produced_output_module()
    if type(artifact_max_bytes) is not int or artifact_max_bytes <= 0:
        raise ValueError(
            "the Stage A artifact max must be a positive integer byte count: "
            "it is the configured budget this invocation runs under, never "
            "a default this module supplies")
    checkpoint = (artifact_max_bytes if checkpoint_max_bytes is None
                  else int(checkpoint_max_bytes))
    if type(write_only) is not bool:
        raise ValueError("the boundary template's write_only is True or False")
    if write_only:
        # Nothing is staged for the owner to read back, so it borrows no
        # stage space at all (PrismaBuild refuses anything but zero here).
        demands = {"minimum_gib": 0, "window_gib": 0}
    else:
        window = boundary_window_gib(
            group_size=group_size,
            max_entry_tensor_bytes=max_entry_tensor_bytes,
            concurrent_groups=concurrent_groups)
        # The working MINIMUM is one group, the window is
        # ``concurrent_groups`` of them: an owner that cannot hold a single
        # publication group cannot make progress at all, while the window is
        # what it wants in order to keep the next group ahead of the reader.
        demands = {"minimum_gib": max(window // max(concurrent_groups, 1), 1),
                   "window_gib": window}
    body = {
        "schema": po.TEMPLATE_SCHEMA_V1,
        "version": 1,
        "template_id": str(template_id),
        "output_prefix": str(output_prefix),
        "slots": {BOUNDARY_SLOT: {"class": "payload"}},
        "durable_maxima": {"payload_max_bytes": int(artifact_max_bytes),
                           "checkpoint_max_bytes": int(checkpoint),
                           "temp_max_bytes": int(artifact_max_bytes)},
        "working_demands": {tier: demands},
        "permitted_tiers": [str(tier)]}
    if write_only:
        # Only when true: a read-back template keeps the exact bytes, and so
        # the template id, it had before PrismaBuild #912.
        body["write_only"] = True
    if export_rate_family is not None:
        # Only when declared, for the same reason (PrismaBuild #1126); the
        # PrismaBuild module checks that it is an identifier.
        body["export_rate_family"] = export_rate_family
    return dict(po.validate_template(body))


class BoundaryStagingTimeout(TimeoutError):
    """The batch's mover did not land inside this window's staging budget.

    NAMED and BOUNDED, and it is the only thing this lane does about an
    asynchronous mover. PrismaBuild schedules and places the movement; the
    reader's only job is to not proceed until PB's own records say the
    bytes are whole. There is no fallback read, no direct origin open and
    no second publication on this path: the caller withdraws (the group's
    prewrite/commit state is PB's and survives), and the next attempt
    resumes the same logical batch.
    """


#: The egress receipt key that says an own-copy deferral occurred. A NEW
#: TOP-LEVEL key, a sibling of ``deferred_handoffs``, ``live_pins``,
#: ``entries_deferred`` and ``errors`` -- not an entry inside any of them.
DEFERRED_OWN_FIELD = "deferred_own"

#: The ONE ``deferred_own`` reason this lane knows how to wait out, spelled
#: exactly as PrismaBuild's egress writes it (stage_release.py, the
#: ``"deferred_own": ["own-copy-in-flight"] if own_deferred else []`` line).
#: Matched as a string, not as a substring and not as truthiness: a reason
#: this lane has never seen is not a reason it may wait on.
OWN_COPY_IN_FLIGHT = "own-copy-in-flight"

#: The ``deferred_own`` reason ``retire_batch`` reports while the retirement's
#: own egress action is queued or running on the tier host
#: (``produced_output.OWN_EGRESS_IN_FLIGHT``, RobTand/prismabuild#801). Only
#: the tier host mounts the stage read-write, so an owner on a GPU host cannot
#: delete its staged files in its own process: PrismaBuild publishes the egress
#: as an action placed on the tier host and answers this until that action has
#: ended. Bytes, proof and full credit are kept exactly as for an own-copy
#: deferral, and the same ordinary retry completes the retirement.
OWN_EGRESS_IN_FLIGHT = "own-egress-in-flight"

#: Every reason a bounded wait is correct for. A non-empty ``deferred_own``
#: carrying anything outside this set is surfaced, because "PrismaBuild
#: deferred for a reason I do not recognise" and "PrismaBuild deferred on
#: its own in-flight work" are different facts and only the second one is
#: known to clear on retry.
RECOGNISED_DEFERRED_OWN_REASONS = frozenset(
    {OWN_COPY_IN_FLIGHT, OWN_EGRESS_IN_FLIGHT})


#: What each surfaced classification means, in the one place a reader of
#: the exception will look. Every one of these is a receipt this lane
#: cannot turn into a decision -- never a decision it made.
UNCLASSIFIED_CAUSES = {
    "unknown":
        f"carries no {DEFERRED_OWN_FIELD!r} key at all and no other positive "
        "cause (no live pins, no deferred handoffs, no errors). A receipt "
        "that predates the field cannot say whether this was an own-copy "
        "deferral, so it is neither waited on nor called final. That is an "
        "older PrismaBuild with no opinion, not a negative answer",
    "egress-deferral-unrecognised":
        f"carries a non-empty {DEFERRED_OWN_FIELD!r} naming a reason this "
        f"lane does not recognise (known: {sorted(RECOGNISED_DEFERRED_OWN_REASONS)}). "
        "PrismaBuild deferred for something, and an unrecognised something "
        "is not known to clear on retry",
    "egress-deferral-malformed":
        f"carries a {DEFERRED_OWN_FIELD!r} that is not a list. The shape is "
        "wrong, so nothing may be read out of it -- neither a deferral nor "
        "the absence of one",
}


class BoundaryEgressUnclassified(RuntimeError):
    """A retirement refused and the receipt could not say whether it deferred.

    Raised for the three receipts that cannot be turned into a decision: no
    ``deferred_own`` key at all and no other positive cause, a non-empty
    ``deferred_own`` naming an unrecognised reason, and a ``deferred_own``
    of the wrong shape. None of the three is "no deferral"; collapsing any
    of them into that is the fail-open shape that cost a stage token per
    occurrence, and the handler written for it does not get to repeat it.

    Its opposite is deliberate and is NOT raised: a receipt that carries
    ``deferred_own`` as an empty list has been asked and has answered. The
    rule is act on positive observations, surface only true silence.
    """

    def __init__(self, batch_id: str, outcome, receipt,
                 kind: str = "unknown") -> None:
        self.batch_id = batch_id
        self.outcome = outcome
        self.receipt = receipt
        self.kind = kind
        cause = UNCLASSIFIED_CAUSES.get(
            kind, f"was classified {kind!r}, which is not a decision")
        super().__init__(
            f"produced-output boundary group {batch_id!r} was refused by an "
            f"egress receipt that {cause}. Surfaced rather than decided. "
            "Run against a PrismaBuild generation that publishes "
            f"{DEFERRED_OWN_FIELD!r} as documented. Outcome: {outcome!r}")


class BoundaryProducedReleaseRefused(RuntimeError):
    """A retirement refused terminally: retrying it cannot help."""

    def __init__(self, batch_id: str, outcome, cause) -> None:
        self.batch_id = batch_id
        self.outcome = outcome
        self.cause = cause
        super().__init__(
            f"produced-output boundary group {batch_id!r} was refused "
            f"terminally ({cause}); its stage credit stays held by the "
            f"material that is still there. Outcome: {outcome!r}")


class BoundaryProducedReleaseDeferred(TimeoutError):
    """An own-copy deferral did not clear inside the staging budget."""

    def __init__(self, batch_id: str, *, waited_s: float, timeout_s: float,
                 attempts: int, outcome) -> None:
        self.batch_id = batch_id
        self.waited_s = waited_s
        self.timeout_s = timeout_s
        self.attempts = attempts
        self.outcome = outcome
        super().__init__(
            f"produced-output boundary group {batch_id!r} deferred its "
            f"retirement on its own in-flight copy and did not clear in "
            f"{waited_s:.1f}s of a {timeout_s:.1f}s budget over {attempts} "
            "re-drives. The stage credit is still held; the next window "
            "will not fund. Reported rather than waited on further -- this "
            f"lane does not own scheduling. Last outcome: {outcome!r}")


def classify_egress_outcome(outcome) -> str:
    """Name one ``retire_batch`` outcome from PrismaBuild's egress receipt.

    ``retire_batch`` returns ``{ok: False, refusal: "egress-incomplete",
    receipt: <egress receipt>}`` for every incomplete egress, so the
    refusal is a category and the receipt is the cause. Exactly one of the
    answers below is waited on; every other one is visible.

    ``own-copy-deferral`` -- WAIT
        ``deferred_own`` is a list naming only reasons in
        :data:`RECOGNISED_DEFERRED_OWN_REASONS`. PrismaBuild deferred this
        retirement on this lane's OWN in-flight work -- the evicted mover's
        still-live claimed copy (:data:`OWN_COPY_IN_FLIGHT`), or the
        retirement's own egress action on the tier host
        (:data:`OWN_EGRESS_IN_FLIGHT`): bytes, proof and full credit are
        kept, and ordinary retry returns the token once that child action
        reaches terminal. The only case this lane waits on.
    ``egress-deferral-unrecognised`` -- SURFACE
        ``deferred_own`` is a non-empty list naming something else. A
        deferral this lane has never seen is not one it may wait out.
    ``egress-deferral-malformed`` -- SURFACE
        ``deferred_own`` is present but is not a list. Nothing may be read
        out of a shape that is wrong.
    ``foreign-pin``
        A live reader holds the staged bytes. A real failure to preserve,
        never something to wait out.
    ``promotion-handoff``
        A deferred promotion handoff: a different lifecycle, not this
        lane's wait.
    ``egress-error``
        The receipt carries errors.
    ``egress-incomplete``
        ``deferred_own`` is present and EMPTY, with no pin, handoff or
        error. A complete answer: PrismaBuild looked and found no own-copy
        deferral. Recorded and drained, never waited on.
    ``unknown`` -- SURFACE
        The receipt has NO ``deferred_own`` key and no positive cause in
        it. That is an older PrismaBuild with no opinion, not a negative
        answer, and a missing key is not an empty list.

    The rule underneath all of it: **act on positive observations, and
    surface only true silence.** A ``deferred_own`` this lane can read
    decides first, because an unreadable or unrecognised deferral is the
    one thing no other field can rule out. After that a non-empty
    ``live_pins``, ``deferred_handoffs`` or ``errors`` classifies whether
    or not ``deferred_own`` is there, because each is a POSITIVE
    observation of a different cause that an older generation reports
    exactly as a newer one does -- which is also what keeps the deployed
    generation, whose receipts have no ``deferred_own`` at all, working.
    What the missing key removes is the ability to conclude anything from
    silence, and that case alone returns ``unknown``.
    """

    if isinstance(outcome, Mapping) and outcome.get("ok"):
        return "retired"
    receipt = outcome.get("receipt") if isinstance(outcome, Mapping) else None
    if not isinstance(receipt, Mapping):
        return "unknown"
    if DEFERRED_OWN_FIELD in receipt:
        deferred_own = receipt[DEFERRED_OWN_FIELD]
        # A list, because that is what the egress writes. A string that
        # happens to read "own-copy-in-flight" is NOT this field; reading
        # one would be inventing an interface PrismaBuild does not have.
        if not isinstance(deferred_own, list):
            return "egress-deferral-malformed"
        if deferred_own:
            if all(reason in RECOGNISED_DEFERRED_OWN_REASONS
                   for reason in deferred_own):
                return "own-copy-deferral"
            # Some reason outside the known set. Not "not a deferral", and
            # not a deferral this lane may sit on either.
            return "egress-deferral-unrecognised"
    if receipt.get("live_pins"):
        return "foreign-pin"
    if receipt.get("deferred_handoffs"):
        return "promotion-handoff"
    if receipt.get("errors"):
        return "egress-error"
    if DEFERRED_OWN_FIELD not in receipt:
        return "unknown"
    return "egress-incomplete"


#: The classifications that are NOT a decision. Each is raised as
#: :class:`BoundaryEgressUnclassified` and recorded in its own debt bucket,
#: never folded into the abandoned one, which would claim a decision was
#: made.
UNCLASSIFIED_OUTCOMES = frozenset(UNCLASSIFIED_CAUSES)


#: PrismaBuild's own name for a step that found a transition lock busy.
#: ``pool.fund_output_batch`` takes the owner lock and then the mover lock
#: with ``blocking=False`` and returns this refusal when either is
#: contended; ``pool.stage_output_intent`` answers the same way on the same
#: owner lock, and the refusal also appears on PrismaBuild's drive, release
#: and adoption paths.
#:
#: It is an explicit transient, and the transient is narrow: THAT
#: non-blocking call moved no tokens. It does NOT mean nothing happened.
#: ``publish_prepaid_batch`` seals and files the request, stages the
#: funding intent and publishes the READY row BEFORE the fund step runs,
#: and a deferral rolls none of that back. What licenses the re-drive is
#: not an absence of side effects; it is that the mover key is derived by
#: content address and every step is idempotent, so identical inputs meet
#: what already exists instead of duplicating it.
#:
#: It surfaces verbatim from both ``publish_prepaid_batch`` and
#: ``ensure_batch_materialized``, stamped with the step that met the
#: contention (produced_output.py, ``funded["step"] = "fund"``).
FUNDING_RACE_REFUSAL = "funding-race-deferred"


def is_funding_race_deferral(outcome) -> bool:
    """True for PrismaBuild's typed funding transient, and nothing else.

    Matched on the exact refusal string. Deliberately NOT a category: a
    refusal that merely looks retryable, an ``errors`` list, a timeout or
    any other step's refusal is terminal to this lane.

    What makes the retry safe is NOT a belief that the deferred call did
    nothing -- the sequence has already sealed a request, staged an intent
    and published a row by the time the funding step runs. It is that the
    mover key is derived by content address and every step is idempotent,
    so identical inputs meet whatever already exists instead of
    duplicating it. That is a property of the call, not of the refusal.
    """

    return (isinstance(outcome, Mapping)
            and outcome.get("ok") is not True
            and outcome.get("refusal") == FUNDING_RACE_REFUSAL)


class BoundaryProducedFundingDeferred(TimeoutError):
    """A funding transient did not clear inside the staging budget."""

    def __init__(self, batch_id: str, *, step: str, waited_s: float,
                 timeout_s: float, attempts: int, outcome) -> None:
        self.batch_id = batch_id
        self.step = step
        self.waited_s = waited_s
        self.timeout_s = timeout_s
        self.attempts = attempts
        self.outcome = outcome
        super().__init__(
            f"produced-output boundary group {batch_id!r} was deferred by "
            f"PrismaBuild's funding lock ({FUNDING_RACE_REFUSAL!r} at step "
            f"{step!r}) and did not clear in {waited_s:.1f}s of a "
            f"{timeout_s:.1f}s staging budget over {attempts} identical "
            "re-drives. What that establishes, and only this: the logical "
            "batch's COMMIT is unfinished. The sealed request, the funding "
            "intent and the mover row may already exist -- the deferred "
            "step runs after them -- and nothing here rolled any of it "
            "back, so the credits stay PrismaBuild-accounted wherever it "
            "put them. Infer no rollback from this failure; re-driving the "
            "identical inputs is the documented resumption. Reported "
            "rather than waited on further -- this lane does not own "
            f"scheduling. Last outcome: {outcome!r}")


class BoundaryRepeatMaterializationUnsupported(RuntimeError):
    """The pinned PrismaBuild candidate has no repeat-materialization API.

    A loud refusal, never a silent skip and never a substitute.  The two
    ways around a missing entry point are both forbidden: a NEW batch id
    over the same origin bytes is a caller-chosen successor and a second
    durable charge, and reading the origin path directly is exactly the
    pool read the strict policy exists to refuse.  The names are
    PrismaBuild's own and are used unaliased --
    ``produced_output.materialization_state`` and
    ``produced_output.ensure_batch_materialized`` -- so a candidate that
    lacks them is reported as the old candidate it is.
    """


def _produced_output_module() -> Any:
    from .staged_lease import LeaseRefused, sdk_submodule
    try:
        module = sdk_submodule("produced_output")
    except LeaseRefused as exc:
        raise BoundaryProducedBindingError(
            "the Stage A boundary publication needs PrismaBuild's\n"
            "produced_output API from the SAME sealed generation as the\n"
            f"reader SDK: {exc}") from exc
    po = module
    for name in ("declared_template", "bind_declared_instance",
                 "declare_instance", "admit_instance", "admit_funded_window",
                 "require_prewrite", "abort_prewrite", "publish_prepaid_batch",
                 "commit_batch", "retire_batch", "reclaim_origin",
                 "safe_release_instance", "refill_window",
                 "validate_descriptor", "output_manifest_sha256",
                 "batch_namespace", "output_fragment_root", "instance_dir",
                 "validate_instance", "owner_demand_terms",
                 "recover_batches", "due_mover_rows"):
        if not callable(getattr(po, name, None)):
            raise BoundaryProducedBindingError(
                "the installed prismabuild.produced_output lacks "
                f"{name!r}: a runtime without the produced-output API "
                "cannot stage Stage A's own boundary entries")
    return po


def _pool_module() -> Any:
    from .staged_lease import LeaseRefused, sdk_submodule
    try:
        module = sdk_submodule("pool")
    except LeaseRefused as exc:
        raise BoundaryProducedBindingError(
            "the Stage A boundary publication needs PrismaBuild's pool\n"
            "module from the SAME sealed generation as the reader SDK: "
            f"{exc}") from exc
    return module


def _residency_map_module() -> Any:
    from .staged_lease import LeaseRefused, sdk_submodule
    try:
        module = sdk_submodule("residency_map")
    except LeaseRefused as exc:
        raise BoundaryProducedBindingError(
            "composing a produced batch's reader context needs\n"
            "PrismaBuild's residency_map module from the SAME sealed\n"
            f"generation as the reader SDK: {exc}") from exc
    return module


def launch_queue_root(env: Mapping[str, str] | None = None) -> Path:
    """The queue root PrismaBuild gave THIS action, or a named refusal.

    Derived exactly as the reader SDK derives it, from the shape of the
    launcher's own ``PRISMABUILD_RESIDENCY_MAP``
    (``<queue>/residency/<key>.json`` -> ``parent.parent``). Never guessed
    from topology and never ``None``: a missing launch context is a
    refusal with the variable named, not a ``Path(None)`` that fails three
    calls later as something else.
    """

    source = dict(os.environ) if env is None else dict(env)
    raw = source.get("PRISMABUILD_RESIDENCY_MAP", "")
    if not raw:
        raise BoundaryProducedBindingError(
            "no PrismaBuild launch context: PRISMABUILD_RESIDENCY_MAP is "
            "unset, so there is no queue root to bind an owner on")
    root = Path(raw).parent.parent
    if not root.is_dir():
        raise BoundaryProducedBindingError(
            f"the queue root derived from PRISMABUILD_RESIDENCY_MAP ({raw}) "
            f"is not a directory: {root}")
    return root


def open_pool_queue(queue_root: str | Path | None,
                    env: Mapping[str, str] | None = None) -> Any:
    """The queue the admitted owner lives on (fail closed, named error)."""

    pool_mod = _pool_module()
    resolved = launch_queue_root(env) if queue_root is None else Path(queue_root)
    try:
        return pool_mod.PoolQueue(resolved)
    except Exception as exc:
        raise BoundaryProducedBindingError(
            f"cannot open the PrismaBuild queue at {resolved}: {exc}"
        ) from exc


def _launch_owner(env: Mapping[str, str] | None) -> str:
    source = dict(os.environ) if env is None else dict(env)
    owner = source.get("PRISMABUILD_ACTION_KEY", "")
    if not owner:
        raise BoundaryProducedBindingError(
            "the Stage A boundary publication needs this action's own key "
            "(PRISMABUILD_ACTION_KEY): it binds to the admitted owner's "
            "request, never to a caller-supplied identity")
    return str(owner)


class BoundaryProducedPublication:
    """One Stage A action's bound produced-output publication.

    Constructed from the admitted owner's request alone (see
    :meth:`bind_from_admitted_owner`); every later call reuses the bound
    template and instance.  Nothing here re-seals the source tree: each
    group's mover is sealed by PrismaBuild off the owner's own sealed
    request.
    """

    #: The template slot every boundary/cotangent entry is filed under.
    #: A template that does not authorize it refuses at the first
    #: descriptor rather than at the mover.
    DEFAULT_SLOT = "boundary_entries"

    def __init__(self, *, queue, template, instance, tier: str,
                 cas_root: str, slot: str | None = None,
                 env: Mapping[str, str] | None = None,
                 command_extra: tuple[str, ...] = ()) -> None:
        self.queue = queue
        self.template = template
        self.instance = instance
        self.tier = str(tier)
        self.cas_root = str(cas_root)
        self.slot = str(slot or self.DEFAULT_SLOT)
        self.env = dict(os.environ) if env is None else dict(env)
        # Dev/fixture-only passthrough for the mover argv (e.g. ``--unpaced``
        # where no ZFS pacer exists); production stays empty.
        self.command_extra = tuple(command_extra)
        self._po = _produced_output_module()
        if self.write_only and not callable(
                getattr(self._po, "commit_origin_batch", None)):
            raise BoundaryProducedBindingError(
                "the declared template is write-only, and the loaded "
                "prismabuild.produced_output has no commit_origin_batch "
                "(PrismaBuild #912): its groups could never be committed")
        self._generation: str | None = None
        # batch_id -> the manifest digest its descriptors sealed. Needed to
        # re-derive the batch namespace for the reader context and for
        # retirement, and deliberately not a second durable store: it is
        # rebuilt from PB's own records by :meth:`recover_batch_states`.
        self._manifest_digests: dict[str, str] = {}

    # -- binding -----------------------------------------------------------

    @classmethod
    def bind_from_admitted_owner(cls, *, queue_root: str | Path | None = None,
                                 tier: str | None = None,
                                 slot: str | None = None,
                                 env: Mapping[str, str] | None = None,
                                 command_extra: tuple[str, ...] = (),
                                 ) -> "BoundaryProducedPublication":
        """Bind template + instance from the admitted owner's request.

        ``queue_root`` and ``tier`` are both normally omitted: the queue
        root is derived from the launcher's own
        ``PRISMABUILD_RESIDENCY_MAP`` and the tier from the declaration,
        so a real action supplies neither and a caller that supplies one
        is overriding, not configuring.

        The launch environment names this attempt; the submission's own
        ``--produced-output-template`` declaration provides the template
        (``declared_template`` -- no operator dictionary);
        ``bind_declared_instance`` checks the live claim and both identity
        halves.  Declares and admits the runtime instance, and refuses
        closed on every mismatch.
        """

        po = _produced_output_module()
        pool_mod = _pool_module()
        owner = _launch_owner(env)
        queue = open_pool_queue(queue_root, env)
        try:
            template = po.declared_template(queue, owner)
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"the owner action {owner[:12]} declares no "
                f"produced-output template: {exc}") from exc
        permitted = list(template.get("permitted_tiers", []))
        if tier is None:
            # The runtime is told which tier CLASSES it may read; the tier
            # ID is the declaration's own. Deriving it is only honest when
            # the declaration leaves no choice, so more than one permitted
            # tier refuses rather than picking.
            if len(permitted) != 1:
                raise BoundaryProducedBindingError(
                    "no tier was named and the declared template permits "
                    f"{permitted!r}: a produced-output owner binds ONE tier, "
                    "and choosing between several is the submitter's to "
                    "declare, not this process's to guess")
            tier = str(permitted[0])
        if str(tier) not in permitted:
            raise BoundaryProducedBindingError(
                f"tier {tier!r} is not in the declared template's "
                "permitted tiers")
        wanted = str(slot or cls.DEFAULT_SLOT)
        if wanted not in template.get("slots", {}):
            raise BoundaryProducedBindingError(
                f"the declared template authorizes no {wanted!r} slot: "
                "Stage A's boundary entries have nowhere to be filed")
        try:
            claim_snapshot = pool_mod._read_json(
                queue.item_path(pool_mod.CLAIMED, owner))
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"the owner claim is unreadable: {exc}") from exc
        if not isinstance(claim_snapshot, Mapping):
            raise BoundaryProducedBindingError(
                "the owner action is not claimed: a boundary publication "
                "binds to the live admitted owner only")
        cas_root = claim_snapshot.get("cas_root")
        if not isinstance(cas_root, str) or not cas_root:
            raise BoundaryProducedBindingError(
                "the admitted owner's row files no cas_root: the "
                "publication refuses to guess the CAS topology")
        try:
            instance = po.bind_declared_instance(
                queue, owner_action_key=owner,
                claim_snapshot=claim_snapshot, env=env)
            instance = cls._adopt_filed_instance(po, queue, instance)
            po.declare_instance(queue.root, instance)
            admitted = po.admit_instance(queue, instance, template)
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"cannot bind the produced-output instance: {exc}") from exc
        if not admitted.get("ok"):
            raise BoundaryProducedBindingError(
                f"the produced-output instance was not admitted: {admitted}")
        return cls(queue=queue, template=template, instance=instance,
                   tier=tier, cas_root=cas_root, slot=wanted, env=env,
                   command_extra=command_extra)

    @staticmethod
    def _adopt_filed_instance(po, queue, bound: Mapping) -> dict:
        """Prefer the instance already filed for this owner/template/nonce.

        RESTART PATH, and the same one the render lane documents:
        ``bind_instance`` stamps ``bound_unix`` into the body and
        ``declare_instance`` files that body immutably, so a process that
        rebinds after a crash would otherwise be unable to declare its own
        instance at all.  ``instance_dir`` is derived from identity a
        restarted process still holds, so the filed body is located by the
        same rule that wrote it.  Every field except ``bound_unix`` must
        match the freshly bound one, so this can never substitute a foreign
        instance.
        """

        path = po.instance_dir(queue.root, bound) / "instance.json"
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return dict(bound)                  # nothing filed yet
        filed = po.validate_instance(raw)
        differing = sorted(
            key for key in set(filed) | set(bound)
            if key != "bound_unix" and filed.get(key) != bound.get(key))
        if differing:
            raise BoundaryProducedBindingError(
                "the filed produced-output instance disagrees with the "
                f"freshly bound one on {differing}: refusing to adopt a "
                "record this attempt did not produce")
        return filed

    # -- geometry ----------------------------------------------------------

    @property
    def output_prefix(self) -> str:
        """The bound prefix every own-generation path must sit inside."""

        return str(self.template["output_prefix"])

    @property
    def write_only(self) -> bool:
        """Whether the declared template is write-only (PrismaBuild #912).

        A write-only owner never reads its groups back in its own action.
        It commits each one at its origin (:meth:`commit_origin`), and a
        later action declares the committed batch as an input.
        """

        return self.template.get("write_only") is True

    def contains(self, path: str | Path) -> bool:
        """Is ``path`` inside the bound output prefix, symlinks resolved?

        PrismaBuild's own containment rule, restated rather than imported
        from a private helper (the same way ``residency_map`` restates the
        tier-record schema): both sides through ``os.path.realpath``, then
        ``os.path.commonpath`` -- a string prefix alone never proves
        identity.  Asked here so a caller can refuse an escaped entry
        directory when it binds, instead of at the first descriptor.  PB
        re-checks every planned path and every descriptor itself; this
        never substitutes for that.
        """

        try:
            real_prefix = os.path.realpath(self.output_prefix)
            real_path = os.path.realpath(os.fspath(path))
            return os.path.commonpath([real_prefix, real_path]) == real_prefix
        except (OSError, ValueError, TypeError):
            return False

    def durable_maxima(self) -> dict[str, int]:
        """The instance's per-class durable ceilings (PB's own reading)."""

        return dict(self._po.checked_instance_maxima(self.template))

    def group_ceiling_bytes(self, *, entries: int,
                            max_entry_tensor_bytes: int,
                            header_envelope_bytes: int = 65536) -> int:
        """A conservative per-group payload ceiling from bound geometry.

        The exact serialized length of an entry is not known before it is
        written, and ``require_prewrite`` must be answered before the first
        byte.  The writer's own bound is the tensor's byte count plus a
        fixed envelope for the PyTorch zip header -- the same
        ``nbytes + 65536`` ceiling ``StreamedBoundaryArtifacts.write``
        already enforces per entry -- so the group's ceiling is that bound
        times the number of entries the group will hold.  A ceiling is an
        upper bound by contract (``_actual_within_ceiling``): the commit's
        actual bytes must land at or under it, and exact equality is only
        the special case of an exact ceiling.
        """

        if type(entries) is not int or entries <= 0:
            raise ValueError("a boundary group holds a positive entry count")
        if type(max_entry_tensor_bytes) is not int or max_entry_tensor_bytes <= 0:
            raise ValueError(
                "a boundary group needs the bound per-entry tensor ceiling")
        return entries * (max_entry_tensor_bytes + int(header_envelope_bytes))

    # -- identity ----------------------------------------------------------

    @property
    def generation(self) -> str:
        """This producer's logical work-unit identity, as a short token.

        The OWNER ACTION KEY, deliberately not the attempt nonce: a PB retry
        is a new attempt of the same action key, so every group id survives a
        restart and a rebuilt publication re-derives the same ids to ask PB
        what happened to them.  Costs no payload byte.
        """

        if self._generation is None:
            self._generation = hashlib.sha256(
                str(self.instance["owner_action_key"]).encode()).hexdigest()[:16]
        return self._generation

    def batch_id_for(self, *, boundary_index: int, group_index: int,
                     probe_index: int | None = None,
                     kind: str = "boundary") -> str:
        """A deterministic, generation-stable batch id for one group.

        INJECTIVE on the whole coordinate, not on the readable part: the
        suffix digests the exact ``(kind, boundary, probe, group,
        generation)`` tuple with an unambiguous separator, so two distinct
        groups can never share a prewrite envelope, a commit or a
        retirement. Readable-but-lossy ids alias, and two groups sharing an
        id would mean retiring one frees the other's credit.

        Retry-stable by construction: the generation is the owner action
        key, so a restart inside the same admitted attempt re-derives every
        id and asks PrismaBuild what happened to it.
        """

        if any(type(v) is not int or v < 0
               for v in (boundary_index, group_index)):
            raise ValueError(
                "boundary group coordinates must be nonnegative integers")
        if probe_index is not None and (type(probe_index) is not int
                                        or probe_index < 0):
            raise ValueError("a boundary group probe index is nonnegative")
        if not kind or "/" in kind:
            raise ValueError("a boundary group kind is a bare name")
        probe = "-" if probe_index is None else str(probe_index)
        ident = hashlib.sha256("\x00".join((
            str(kind), str(boundary_index), probe, str(group_index),
            self.generation)).encode()).hexdigest()[:12]
        readable = (f"b{boundary_index}" if probe_index is None
                    else f"b{boundary_index}p{probe_index}")
        return (f"stagea-{kind}-{readable}-g{group_index}"
                f"-{self.generation}-{ident}")

    # -- window ------------------------------------------------------------

    def admit_window(self, need_gib: int | None = None) -> dict:
        """The funded-window binding report (declaration-only by design)."""

        demands = self.template["working_demands"][self.tier]
        need = int(demands["window_gib"]) if need_gib is None else int(need_gib)
        out = self._po.admit_funded_window(
            self.queue, self.instance, self.template,
            need_gib_per_tier={self.tier: need})
        if not out.get("ok"):
            raise BoundaryProducedBindingError(
                f"the funded window was not admitted: {out}")
        return dict(out)

    def refill(self) -> dict:
        """Bounded lifecycle refill of this producer's own window."""

        return dict(self._po.refill_window(
            self.queue, self.instance, self.template, tier=self.tier))

    # -- per group ---------------------------------------------------------

    def require_prewrite(self, *, batch_id: str, payload_ceiling_bytes: int,
                         paths: list[str], temp_ceiling_bytes: int = 0) -> dict:
        """Claim the group's durable budget BEFORE its first byte.

        ``paths`` is the group's PLANNED durable-origin superset: the exact
        canonical entry names the existing writer will write, plus the exact
        ``.pt.tmp`` staging names it renames away.  A conservative superset
        is allowed by contract and the rename makes the temporaries absent
        again, which is what the commit's planned-omitted-absent proof
        requires.  Nothing here writes, copies or hashes a file.
        """

        out = dict(self._po.require_prewrite(
            self.queue, self.instance, self.template, batch_id=batch_id,
            tier=self.tier,
            class_bytes={"payload": int(payload_ceiling_bytes),
                         "checkpoint": 0, "temp": int(temp_ceiling_bytes)},
            paths=list(paths)))
        if not out.get("ok"):
            raise BoundaryProducedPrewriteRefused(
                batch_id=batch_id, refusal=out.get("refusal", out))
        return out

    def abort_prewrite(self, *, batch_id: str) -> dict:
        """Release a group's prewrite after an abort with no output.

        Safe only when every planned path is absent, which PB proves
        itself (``abort_prewrite`` lstats each one and retains on a present
        or unstatable file).  This never deletes a durable origin: disposal
        is the producer's, the proof of absence is PrismaBuild's.
        """

        return dict(self._po.abort_prewrite(
            self.queue, self.instance, self.template, batch_id=batch_id))

    def descriptor_for(self, reference, *, producer_generation: str) -> dict:
        """One validated descriptor for an entry the writer already wrote.

        The digest is the writer's OWN inline serialized digest, carried on
        the reference -- not a reread, not a second hash, and not the DEV
        null.  It is the same number the strict reader demands of the map
        entry, so the producer's receipt, PB's manifest and the reader's
        fence cannot disagree.  The byte count is the reference's recorded
        file length, which the writer fenced against its own stat at
        publication.
        """

        if not isinstance(producer_generation, str) or not producer_generation:
            raise BoundaryProducedBindingError(
                "a descriptor needs a retry-stable producer_generation "
                "(the group's batch id), never a fresh mint")
        return self._po.validate_descriptor({
            "schema": self._po.DESCRIPTOR_SCHEMA_V2,
            "slot": self.slot,
            "artifact_class": "payload",
            "path": str(reference.path),
            "bytes": int(reference.file_bytes),
            "sha256": str(reference.sha256),
            "producer_generation": str(producer_generation),
            "owner_action_key": self.instance["owner_action_key"],
            "owner_attempt": dict(self.instance["owner_attempt"]),
        }, self.template, self.instance)

    def manifest_digest_for(self, descriptors: list) -> str:
        """The batch's own manifest digest over its descriptor list."""

        return str(self._po.output_manifest_sha256(list(descriptors)))

    def namespace_for(self, *, batch_id: str, manifest_digest: str) -> str:
        """The batch namespace PB's mover files its fragment under."""

        return str(self._po.batch_namespace(
            self.instance, batch_id, str(manifest_digest)))

    #: How long one paced re-drive waits before asking PrismaBuild's
    #: funding path again. A floor, not a schedule: without it a wait burns
    #: the whole staging budget in milliseconds and reports a timeout that
    #: never waited for anything.
    FUNDING_DEFERRAL_POLL_S = 0.25

    #: How many typed funding transients this publication has waited out,
    #: over its whole life. Counted because an invisible retry is how a
    #: contended fleet starts looking like a fast one.
    funding_deferrals = 0

    def _require_staging_budget(self, batch_id: str, deadline, *,
                                step: str) -> None:
        """Refuse to START a mutation on a budget that is already spent.

        THE RULE, stated once because four variants of its violation have
        been written in this lane: a deadline bounds SIDE EFFECTS, not
        attempts. It is read immediately before the mutation -- not after
        it, not once per iteration, and with no exemption for being the
        first call. An expiry observed here is this process's own fact; it
        does not need PrismaBuild to hand back a refusal, and it must not
        perform a mutation in order to obtain one.

        Every call this guards can change durable state: ``refill_window``
        takes a reservation, ``publish_prepaid_batch`` seals a request and
        publishes a row, ``ensure_batch_materialized`` seals a successor.
        A deadline that stops the funding but not the refill is not one
        budget.

        ``deadline`` of ``None`` is the legacy one-shot: no budget was
        opened, so nothing is bounded and nothing is refused.
        """

        import time

        if deadline is None or time.monotonic() < deadline:
            return
        raise BoundaryStagingTimeout(
            f"produced-output boundary group {batch_id!r} did not START its "
            f"{step} step: the staging budget this group was given was "
            "already spent when the step was reached. Nothing was called, "
            "so there is no PrismaBuild outcome to report -- this is a "
            "locally observed expiry, and doing work known to be doomed "
            "just to be handed a refusal is how a bounded run stops being "
            "bounded.")

    def _drive_prepaid_step(self, call, *, batch_id: str, deadline):
        """Re-drive ONE prepaid step while PrismaBuild says it is contended.

        ``call`` is re-invoked with IDENTICAL inputs, which is the whole
        licence for this, and BOTH callers document it themselves rather
        than one being inferred from the other. ``publish_prepaid_batch``
        derives the mover key by content address, so the same batch,
        descriptors and instance re-derive the same key: "retrying with
        identical inputs re-derives the same key and every step is
        idempotent". ``ensure_batch_materialized`` seals its successor's
        key over the filed materialization generation and says the same of
        a restart: it "re-drives that exact row's sealed mover key through
        the same idempotent stage/publish/fund steps", with the intent
        filed under the ownership lock before any side effect and two
        concurrent callers collapsing onto one generation. Nothing here
        mints a successor, relaxes a lock, or touches capacity.

        Bounded by the caller's ABSOLUTE ``deadline`` -- the same one the
        rest of this group's staging spends, never a fresh budget minted
        here. ``deadline`` of ``None`` means no budget was opened, so this
        drives exactly once and the transient surfaces like any other
        refusal; this lane does not invent time to spend.

        Only :func:`is_funding_race_deferral` is waited on. Every other
        refusal returns immediately, to its caller, unchanged.
        """

        import time

        started = time.monotonic()
        attempts = 0
        out = None
        while True:
            # READ THE DEADLINE IMMEDIATELY BEFORE THE SIDE EFFECT, on
            # EVERY pass including the first. A budget checked only after
            # the call -- or only after the sleep that reached it, or only
            # for retries -- still issues a mutating call past expiry.
            if attempts and deadline is not None and (
                    time.monotonic() >= deadline):
                # A retry past expiry: report the last ACTUAL outcome,
                # never a refusal nobody returned.
                raise BoundaryProducedFundingDeferred(
                    batch_id, step=str(out.get("step") or "fund"),
                    waited_s=time.monotonic() - started,
                    timeout_s=max(deadline - started, 0.0),
                    attempts=attempts, outcome=out)
            if not attempts:
                # The first call is bounded too. An inherited deadline
                # that a previous step already spent stops this one before
                # it touches PrismaBuild at all.
                self._require_staging_budget(batch_id, deadline,
                                             step="prepaid")
            out = dict(call())
            attempts += 1
            if not is_funding_race_deferral(out):
                return out
            if deadline is None:
                return out
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BoundaryProducedFundingDeferred(
                    batch_id, step=str(out.get("step") or "fund"),
                    waited_s=time.monotonic() - started,
                    timeout_s=max(deadline - started, 0.0),
                    attempts=attempts, outcome=out)
            # Per instance, never on the class: a count shared between
            # owners would report another owner's contention as this
            # one's.
            self.funding_deferrals = self.funding_deferrals + 1
            time.sleep(min(self.FUNDING_DEFERRAL_POLL_S, remaining))

    def publish(self, *, batch_id: str, descriptors: list,
                deadline=None) -> dict:
        """Publish one finished group; PB seals the mover off this owner.

        The bounded window refill runs first: retirement returned spent
        credits to free, and the group ahead funds only by exact transfer
        from this owner's own holdings.

        ``deadline`` is the absolute monotonic instant this group's WHOLE
        staging must end by -- publish, any re-materialization and the wait
        for the mover's receipt share it. It is spent, never reset, so a
        funding transient waited out here shortens the wait that follows
        instead of extending the total.
        """

        # Before the refill, not after it: a refill takes a reservation,
        # so it is a mutation and the budget governs it too.
        self._require_staging_budget(batch_id, deadline, step="refill")
        refill = self.refill()
        if not refill.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id,
                refusal={"step": "refill", "refusal": refill.get("refusal"),
                         "refill": refill})
        descriptors = list(descriptors)
        out = self._drive_prepaid_step(
            lambda: self._po.publish_prepaid_batch(
                self.queue, self.instance, self.template, list(descriptors),
                batch_id=batch_id, tier=self.tier, cas_root=self.cas_root,
                producer_action_key=str(self.instance["owner_action_key"]),
                command_extra=tuple(self.command_extra)),
            batch_id=batch_id, deadline=deadline)
        if not out.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id, refusal=out)
        self._manifest_digests[str(batch_id)] = self.manifest_digest_for(
            descriptors)
        return out

    def commit_origin(self, *, batch_id: str, descriptors: list,
                      lifetime: str) -> dict:
        """Commit one finished group of a write-only owner at its origin.

        PrismaBuild's ``commit_origin_batch`` (#912): no stage copy, no
        mover and no window. It consumes the group's prewrite, records each
        origin's identity and returns the batch ``ref`` a consumer declares.
        ``lifetime`` (#914) is ``retain`` (PrismaBuild never deletes it) or
        ``consumed`` (its retirement tick deletes it once every declared
        consumer has succeeded, or once the producer attempt is dead and no
        consumer declared it). Every descriptor carries its sha256, which is
        what binds the bytes a consumer's mover copies to these.

        This is the direct path, for files the owner wrote at their
        canonical paths itself. A group exported through the local spool
        commits through the spool instead, against the identities its
        export receipt recorded (``ProducedOutputSpool.commit_origin``).
        A refusal raises: the entries are durable, and a retry of the same
        descriptors answers PrismaBuild's typed duplicate.
        """

        if not self.write_only:
            raise BoundaryProducedBindingError(
                "only a write-only template commits a group at its origin: a "
                "read-back template's groups are staged by publish()")
        descriptors = list(descriptors)
        out = dict(self._po.commit_origin_batch(
            self.queue, self.instance, self.template, descriptors,
            batch_id=batch_id, lifetime=str(lifetime)))
        if not out.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id, refusal=out)
        self._manifest_digests[str(batch_id)] = str(out["manifest_digest"])
        return out

    # -- reading this producer's own output --------------------------------

    def fragment_root(self) -> Path:
        """Where the produced movers file their fragments and material."""

        pool_mod = _pool_module()
        return Path(self._po.output_fragment_root(
            Path(self.queue.root) / pool_mod.RESIDENCY))

    def reader_context_root(self) -> Path:
        """Where this adapter composes its supplemental reader maps.

        Under the instance directory, which is metadata only: a composed map
        is a projection of PB's own fragments, never a second store of the
        bytes and never a parallel cache.
        """

        root = Path(self._po.instance_dir(
            self.queue.root, self.instance)) / "reader-context"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def materialize(self, *, batch_id: str,
                    manifest_digest: str | None = None) -> dict:
        """Compose this group's staged fragments into a reader context.

        Selection, not movement: ``read_fragments`` and ``compose`` are
        PrismaBuild's own published primitives over the fragments its mover
        filed, and ``compose`` refuses any disagreement between them.  The
        composed map is written once per group under the instance's metadata
        directory and the returned block names everything a scoped resolver
        needs -- the map path, the batch's own manifest digest, the material
        namespace and the fragment root.  Nothing polls: an unstaged group
        raises :class:`BoundaryMaterializationIncomplete` and the caller
        decides.
        """

        map_mod = _residency_map_module()
        digest = str(manifest_digest or self._manifest_digests.get(
            str(batch_id), ""))
        if len(digest) != 64:
            raise BoundaryProducedBindingError(
                f"boundary group {batch_id!r} has no known manifest digest: "
                "materialization is bound to the exact descriptor list its "
                "publication sealed, never to a re-derived one")
        namespace = self.namespace_for(batch_id=batch_id,
                                       manifest_digest=digest)
        root = self.fragment_root()
        try:
            fragments = map_mod.read_fragments(root, namespace)
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"cannot read the fragments for boundary group "
                f"{batch_id!r}: {exc}") from exc
        if not fragments:
            raise BoundaryMaterializationIncomplete(
                f"boundary group {batch_id!r} (namespace {namespace[:12]}) "
                "has no staged fragment yet: its mover has not published")
        # Fragments cannot answer whether the batch is WHOLE. `stage_move`
        # publishes one fragment per entry as the bytes land and files its
        # receipt once at the end, so a half-staged batch composes exactly
        # like a finished one. PrismaBuild's own receipt is the readiness
        # fact, and this lane reads it rather than inferring completeness
        # from what happens to be on the tier.
        if hasattr(self._po, "materialization_state"):
            state = self.materialization_state(batch_id=batch_id)
            if state.get("mover_receipt_complete") is not True:
                raise BoundaryMaterializationIncomplete(
                    f"boundary group {batch_id!r} has fragments but no "
                    f"complete mover receipt (mover "
                    f"{str(state.get('mover_key'))[:12]}, queue state "
                    f"{state.get('mover_queue_state')!r}): a half-staged "
                    "batch composes like a whole one, so the receipt is "
                    "what this read waits on")
        try:
            composed = map_mod.compose(fragments)
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"the fragments for boundary group {batch_id!r} do not "
                f"compose: {exc}") from exc
        if str(composed.get("manifest_sha256") or "") != digest:
            raise BoundaryProducedBindingError(
                f"the composed context for boundary group {batch_id!r} names "
                f"manifest {str(composed.get('manifest_sha256'))[:12]}, not "
                f"this group's {digest[:12]}")
        path = self.reader_context_root() / f"{namespace}.map.json"
        map_mod.write_map(path, composed)
        entries = composed.get("entries")
        return {"batch_id": str(batch_id), "map_path": str(path),
                "manifest_sha256": digest, "material_namespace": namespace,
                "residency_root": str(root),
                "tiers_dir": str(Path(self.queue.root) / "tiers"),
                "entries": len(entries) if isinstance(entries, Mapping) else 0}

    def reader_context(self, *, batch_id: str,
                       manifest_digest: str | None = None):
        """A supplemental, explicitly namespaced resolver for one group.

        Never the process input map: this resolver is constructed for this
        batch, bound to this batch's manifest digest, and handed to exactly
        the reads that want this batch's entries.  Every other read -- the
        run's sealed inputs, a read-only attached generation, a foreign
        generation -- keeps resolving through the unchanged process
        resolver.
        """

        from .residency_map import namespaced_residency_resolver

        block = self.materialize(batch_id=batch_id,
                                 manifest_digest=manifest_digest)
        resolver = namespaced_residency_resolver(
            block["map_path"], manifest_sha256=block["manifest_sha256"],
            residency_root=block["residency_root"],
            material_namespace=block["material_namespace"],
            tiers_dir=block["tiers_dir"])
        return resolver, block

    # -- lifetime ----------------------------------------------------------

    def stage_root(self) -> str:
        """The tier's announced stage root, read from PB's own tier record.

        The storage role announces ``mountpoint`` in
        ``<queue>/tiers/<tier>.json`` -- the same record
        ``residency_map`` reads the RAM half's identity from -- so the
        retirement names the root the tier announced and never one derived
        from queue topology.
        """

        for record in self.queue.tiers():
            if not isinstance(record, Mapping):
                continue
            if str(record.get("tier_id") or "") != self.tier:
                continue
            root = record.get("mountpoint")
            if isinstance(root, str) and root:
                return root
        raise BoundaryProducedBindingError(
            f"tier {self.tier!r} announces no stage mountpoint")

    def retire(self, batch_id: str, *, stage_root: str | Path | None = None,
               residency_root: str | Path | None = None) -> dict:
        """Release the group's stage copy; the origin stays durable."""

        root = (self.fragment_root() if residency_root is None
                else Path(residency_root))
        return dict(self._po.retire_batch(
            self.queue, self.instance, self.template, batch_id,
            stage_root=(self.stage_root() if stage_root is None
                        else str(stage_root)),
            residency_root=root))

    def reclaim_origin(self, batch_id: str) -> dict:
        """Free the group's durable charge once its entries are all gone.

        Disposal is the producer's -- ``StreamedBoundaryArtifacts`` unlinks
        its own entries -- and the proof of absence is PrismaBuild's.
        """

        return dict(self._po.reclaim_origin(
            self.queue, self.instance, self.template, batch_id=batch_id))

    def durable_charge(self) -> dict:
        """Committed origin bytes still charged, read from PB's own record.

        The durable charge is a property of the LOGICAL batch, not of its
        stage copy: retirement returns the tier window and leaves this
        untouched, and only ``reclaim_origin`` -- which proves every origin
        path absent -- stops a batch counting. Read here from the
        instance's ``commitments.json`` (PrismaBuild's record, in the
        instance directory it owns) rather than recomputed from
        descriptors, so a drift between what this lane thinks it wrote and
        what PB filed shows up as a difference instead of being
        reconstructed away.
        """

        path = Path(self._po.instance_dir(
            self.queue.root, self.instance)) / "commitments.json"
        try:
            batches = json.loads(path.read_text()).get("batches") or {}
        except FileNotFoundError:
            batches = {}
        sums = {"payload": 0, "checkpoint": 0, "temp": 0}
        for record in batches.values():
            if not isinstance(record, Mapping) or record.get("origin_reclaimed"):
                continue
            classes = record.get("class_bytes")
            if not isinstance(classes, Mapping):
                raise BoundaryProducedBindingError(
                    "a committed batch records no class bytes")
            for name in sums:
                sums[name] += int(classes.get(name) or 0)
        return sums

    def origin_only_lifetimes(self) -> dict:
        """``{batch_id: lifetime}`` of every origin-only batch PB filed.

        Read from the instance's ``commitments.json``, like
        :meth:`durable_charge`. A staged batch (``publish_prepaid_batch``)
        is absent. An origin-only entry without a lifetime retains (PB #914
        writes ``consumed`` only). A referenced checkpoint (PQ #1036) refuses
        to seal over a ``consumed`` batch, because PB's retirement tick
        unlinks those origins once their consumers succeed.
        """

        path = Path(self._po.instance_dir(
            self.queue.root, self.instance)) / "commitments.json"
        try:
            batches = json.loads(path.read_text()).get("batches") or {}
        except FileNotFoundError:
            batches = {}
        return {str(batch_id): str(record.get("lifetime") or "retain")
                for batch_id, record in batches.items()
                if isinstance(record, Mapping) and record.get("origin_only") is True}

    def recover_batch_states(self) -> dict:
        """What PrismaBuild's own records say about this owner's groups."""

        states: dict[str, str] = {}
        for event in self._po.recover_batches(
                self.queue, self.instance, self.template):
            batch_id = event.get("batch_id")
            if isinstance(batch_id, str) and batch_id:
                states[batch_id] = str(event.get("event") or "")
        return states

    def materialization_state(self, *, batch_id: str) -> dict:
        """PrismaBuild's own read-only answer: which copy is current, where.

        The question a bounded-window reader asks BEFORE deciding whether
        it needs a re-materialization -- is a stage copy of this batch live
        now, under which mover and generation, and has that mover's receipt
        said the bytes landed whole.  Mutates nothing and takes no lock.
        The spelling is PrismaBuild's, deliberately unaliased.
        """

        po = self._po
        if not hasattr(po, "materialization_state"):
            raise BoundaryRepeatMaterializationUnsupported(
                "the pinned PrismaBuild candidate has no "
                "produced_output.materialization_state: this lane reads a "
                "batch's residency from PB's records or not at all")
        return dict(po.materialization_state(
            self.queue, self.instance, self.template, batch_id=batch_id))

    def await_materialized(self, *, batch_id: str, timeout_s: float,
                           poll_s: float = 0.25, deadline=None) -> dict:
        """Wait, bounded, for PrismaBuild to say this group is whole.

        Staging is ASYNCHRONOUS: publishing seals a mover row and the fleet
        claims, places and runs it. Nothing here schedules, places, or
        re-submits anything -- the only decision is when to stop waiting.

        The readiness fact is PrismaBuild's own: ``materialization_state``
        reports ``mover_receipt_complete`` from the mover's filed receipt,
        which is the one signal that distinguishes a half-staged batch from
        a finished one. PB publishes no blocking wait for a produced
        mover's row (``pbwait`` is a client CLI over an action key, not a
        library call this path may shell out to), so this polls those
        records at a bounded budget and raises :class:`BoundaryStagingTimeout`
        -- a named failure and a withdrawal, never a fallback.

        A complete receipt still wins: ``mover_receipt_complete is True``
        returns even if the queue row has since moved terminal. An
        attributable terminal mover fails fast: after a valid binding
        (``ok is True``) with an incomplete receipt, a current mover
        already in ``failed`` or ``withdrawn`` raises
        :class:`BoundaryStagingTimeout` at once with the batch, mover and
        generation attached, rather than waiting out the full budget.
        Every other state -- absent, unknown/unreadable, ready, claimed,
        done -- keeps the existing bounded wait. This never infers spent
        funding from absent tokens and never scans ``recover_batches``;
        the single ``materialization_state`` snapshot is the identity
        proof, so a retired predecessor cannot fail a live successor:
        once a restage files, the active mover is the successor.
        """

        import time

        # The caller's absolute instant when it opened one, so a funding
        # transient already waited out inside publish() is spent from THIS
        # budget rather than forgiven by a second one.
        if deadline is None:
            deadline = time.monotonic() + max(float(timeout_s), 0.0)
        last: dict = {}
        while True:
            last = self.materialization_state(batch_id=batch_id)
            if last.get("mover_receipt_complete") is True:
                return last
            if last.get("ok") is False:
                raise BoundaryProducedBindingError(
                    f"boundary group {batch_id!r} has no readable "
                    f"materialization: {last.get('refusal')!r}")
            # Fail fast on an attributable terminal mover. The snapshot
            # above is the ACTIVE materialization, so a retired
            # predecessor cannot trigger this once a successor is filed:
            # the active mover would then be the successor. Only
            # failed/withdrawn are terminal here; absent, unknown,
            # ready, claimed and done keep the bounded wait, and no
            # funding record or census is read on this path.
            if (last.get("ok") is True
                    and last.get("mover_receipt_complete") is not True
                    and last.get("mover_queue_state") in (
                        "failed", "withdrawn")):
                raise BoundaryStagingTimeout(
                    f"boundary group {batch_id!r} will not stage: mover "
                    f"{str(last.get('mover_key'))[:12]} (generation "
                    f"{last.get('generation')!r}) is terminal "
                    f"{last.get('mover_queue_state')!r} and its receipt "
                    f"reads {last.get('mover_receipt_complete')!r} "
                    f"(refusal {last.get('mover_refusal')!r}); failing fast "
                    f"without waiting the full {timeout_s}s budget")
            if time.monotonic() >= deadline:
                raise BoundaryStagingTimeout(
                    f"boundary group {batch_id!r} was not staged within "
                    f"{timeout_s}s: mover "
                    f"{str(last.get('mover_key'))[:12]} is "
                    f"{last.get('mover_queue_state')!r} and its receipt "
                    f"reads {last.get('mover_receipt_complete')!r}")
            time.sleep(min(poll_s, max(deadline - time.monotonic(), 0.0)))

    def ensure_batch_materialized(self, *, batch_id: str,
                                  deadline=None) -> dict:
        """Make one already-committed group resident on its tier again.

        Asks first.  ``materialization_state`` is the intended gate: a
        batch whose active copy is neither retired nor incomplete is
        already readable, and calling the transition unconditionally would
        ask PrismaBuild to re-answer a question its own records already
        settle.  Only when the state says the copy is gone (or was never
        completed) does this drive the transition.

        Nothing here chooses an origin, a token or a successor id: the
        descriptors come from the immutable batch record and the
        successor's mover/funding key is the content-addressed key of a
        request PrismaBuild seals over the filed materialization
        generation.  The window is refilled first for the same reason
        :meth:`publish` refills -- retirement returned the spent credit to
        free, and a successor funds only by exact transfer from this
        owner's own holdings.  The durable origin charge is untouched: one
        logical batch, one charge, across the forward and reverse cycle.
        """

        self._require_staging_budget(batch_id, deadline, step="ensure")
        po = self._po
        if not hasattr(po, "ensure_batch_materialized"):
            raise BoundaryRepeatMaterializationUnsupported(
                "the pinned PrismaBuild candidate has no "
                "produced_output.ensure_batch_materialized: re-staging a "
                "retired group is not available, and this lane will not "
                "invent a successor key, a second durable charge or a "
                "pool read to fake it")
        state = self.materialization_state(batch_id=batch_id)
        if (state.get("ok") and not state.get("stage_retired")
                and state.get("mover_receipt_complete")):
            return {"ok": True, "state": "live", "step": "none",
                    "ensured": False, "queried": state,
                    "batch_id": str(batch_id),
                    "mover_key": state.get("mover_key"),
                    "generation": state.get("generation"),
                    "manifest_digest": state.get("manifest_digest")}
        # Immediately before the mutation, again: asking PrismaBuild for
        # the state above is a read and costs nothing, taking a
        # reservation is not.
        self._require_staging_budget(batch_id, deadline, step="refill")
        refill = self.refill()
        if not refill.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id,
                refusal={"step": "refill", "refusal": refill.get("refusal"),
                         "refill": refill})
        out = self._drive_prepaid_step(
            lambda: po.ensure_batch_materialized(
                self.queue, self.instance, self.template, batch_id=batch_id,
                cas_root=self.cas_root,
                producer_action_key=str(self.instance["owner_action_key"]),
                command_extra=tuple(self.command_extra)),
            batch_id=batch_id, deadline=deadline)
        if not out.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id, refusal=out)
        digest = str(out.get("manifest_digest") or "")
        if len(digest) == 64:
            # The same logical batch keeps the same manifest; re-record it
            # so a reader that only ever saw the successor can still
            # compose (the map is bound to the digest, not to the mover).
            self._manifest_digests[str(batch_id)] = digest
        out["ensured"] = True
        out["queried"] = state
        return out

    def release(self, lease_sdk=None) -> dict:
        """Close the instance and reclaim leftover holdings."""

        if lease_sdk is None:
            try:
                from prismabuild import reader_lease as lease_sdk
            except ImportError:
                lease_sdk = None
        out = self._po.safe_release_instance(
            self.queue, self.instance, self.template, lease_sdk=lease_sdk)
        return dict(out) if isinstance(out, dict) else {"ok": True}


__all__ = [
    "BoundaryProducedBindingError",
    "BoundaryProducedPrewriteRefused",
    "BoundaryProducedFundingDeferred",
    "BoundaryProducedPublicationFailed",
    "BoundaryMaterializationIncomplete",
    "BoundaryRepeatMaterializationUnsupported",
    "BoundaryStagingTimeout",
    "FUNDING_RACE_REFUSAL",
    "is_funding_race_deferral",
    "launch_queue_root",
    "BOUNDARY_SLOT",
    "BOUNDARY_TEMPLATE_ID",
    "boundary_group_ceiling_bytes",
    "boundary_window_gib",
    "build_boundary_template",
    "BoundaryProducedPublication",
    "open_pool_queue",
]
