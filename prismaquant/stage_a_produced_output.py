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

One transition is NOT implemented here because PrismaBuild does not publish
it yet: re-materializing an unchanged, already-retired logical batch.  See
:class:`BoundaryRematerializationUnavailable` for the exact call this lane
needs and why no substitute is acceptable.
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


class BoundaryRematerializationUnavailable(NotImplementedError):
    """Re-staging an unchanged, already-retired batch is not yet publishable.

    ``publish_prepaid_batch`` short-circuits on a filed commitments entry:
    a second call after ``retire_batch`` replays the idempotent commit and
    answers the duplicate rather than sealing a new mover, so the bytes are
    never re-staged.  The two ways around that are both forbidden:

    * a NEW batch id for the same origin bytes is a caller-chosen successor
      and a second durable charge (the PB ruling: unique successor key from
      the PB-sealed materialization record, immutable origin charge constant
      across forward and reverse staging); and
    * reading the origin path directly is the pool read the strict policy
      exists to refuse.

    THE CAPABILITY REQUESTED, not a name this lane has chosen. The
    spelling belongs to the PrismaBuild lane that owns produced output;
    two lanes shipping different names for one call is a merge problem,
    so nothing here calls, imports or depends on any such symbol. What
    this lane needs is one entry point that takes the owner's bound
    (queue, instance, template) plus the EXISTING ``batch_id``, tier and
    cas_root, and re-stages that unchanged logical batch, returning the
    new mover key, the PB-sealed materialization sequence and the batch
    namespace its fragments will be vouched under -- idempotent on replay.

    The properties the produced-output ruling already fixes, restated so
    the request is unambiguous: the successor mover and funding key derive
    from PrismaBuild's own sealed materialization record and never from a
    caller nonce; the old terminal key stays terminal and spent; at most
    one live-or-pending materialization exists per logical batch; the
    immutable origin identity tuple captured at first commit is re-checked
    before funding; and the durable origin charge is constant across the
    forward and reverse cycle.
    """


def _produced_output_module() -> Any:
    try:
        from prismabuild import produced_output as po
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise BoundaryProducedBindingError(
            "the Stage A boundary publication needs PrismaBuild's "
            f"produced_output API: {exc}") from exc
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
    try:
        from prismabuild import pool as pool_mod
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise BoundaryProducedBindingError(
            "the Stage A boundary publication needs PrismaBuild's pool "
            f"module: {exc}") from exc
    return pool_mod


def _residency_map_module() -> Any:
    try:
        from prismabuild import residency_map as map_mod
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise BoundaryProducedBindingError(
            "composing a produced batch's reader context needs "
            f"PrismaBuild's residency_map module: {exc}") from exc
    return map_mod


def open_pool_queue(queue_root: str | Path) -> Any:
    """The queue the admitted owner lives on (fail closed, named error)."""

    pool_mod = _pool_module()
    try:
        return pool_mod.PoolQueue(Path(queue_root))
    except Exception as exc:
        raise BoundaryProducedBindingError(
            f"cannot open the PrismaBuild queue at {queue_root}: {exc}"
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
        self._generation: str | None = None
        # batch_id -> the manifest digest its descriptors sealed. Needed to
        # re-derive the batch namespace for the reader context and for
        # retirement, and deliberately not a second durable store: it is
        # rebuilt from PB's own records by :meth:`recover_batch_states`.
        self._manifest_digests: dict[str, str] = {}

    # -- binding -----------------------------------------------------------

    @classmethod
    def bind_from_admitted_owner(cls, *, queue_root: str | Path, tier: str,
                                 slot: str | None = None,
                                 env: Mapping[str, str] | None = None,
                                 command_extra: tuple[str, ...] = (),
                                 ) -> "BoundaryProducedPublication":
        """Bind template + instance from the admitted owner's request.

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
        queue = open_pool_queue(queue_root)
        try:
            template = po.declared_template(queue, owner)
        except Exception as exc:
            raise BoundaryProducedBindingError(
                f"the owner action {owner[:12]} declares no "
                f"produced-output template: {exc}") from exc
        if str(tier) not in template.get("permitted_tiers", []):
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

    def publish(self, *, batch_id: str, descriptors: list) -> dict:
        """Publish one finished group; PB seals the mover off this owner.

        The bounded window refill runs first: retirement returned spent
        credits to free, and the group ahead funds only by exact transfer
        from this owner's own holdings.
        """

        refill = self.refill()
        if not refill.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id,
                refusal={"step": "refill", "refusal": refill.get("refusal"),
                         "refill": refill})
        out = dict(self._po.publish_prepaid_batch(
            self.queue, self.instance, self.template, list(descriptors),
            batch_id=batch_id, tier=self.tier, cas_root=self.cas_root,
            producer_action_key=str(self.instance["owner_action_key"]),
            command_extra=tuple(self.command_extra)))
        if not out.get("ok"):
            raise BoundaryProducedPublicationFailed(
                batch_id=batch_id, refusal=out)
        self._manifest_digests[str(batch_id)] = self.manifest_digest_for(
            descriptors)
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

    def recover_batch_states(self) -> dict:
        """What PrismaBuild's own records say about this owner's groups."""

        states: dict[str, str] = {}
        for event in self._po.recover_batches(
                self.queue, self.instance, self.template):
            batch_id = event.get("batch_id")
            if isinstance(batch_id, str) and batch_id:
                states[batch_id] = str(event.get("event") or "")
        return states

    def rematerialize(self, *, batch_id: str) -> dict:
        """Re-stage an unchanged, already-retired group. NOT AVAILABLE.

        Refuses loudly rather than approximating.  See
        :class:`BoundaryRematerializationUnavailable` for the exact call
        this lane needs from PrismaBuild and why a new batch id, a caller
        nonce or a direct origin read are each forbidden substitutes.
        """

        raise BoundaryRematerializationUnavailable(
            f"re-materializing retired boundary group {batch_id!r} needs "
            "PrismaBuild's repeat-materialization entry point; "
            "publish_prepaid_batch replays the committed duplicate instead "
            "of sealing a successor mover, and this lane will not invent a "
            "successor key, a second durable charge, or a pool read")

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
    "BoundaryProducedPublicationFailed",
    "BoundaryMaterializationIncomplete",
    "BoundaryRematerializationUnavailable",
    "BoundaryProducedPublication",
    "open_pool_queue",
]
