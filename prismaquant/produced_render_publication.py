"""Produced-render publication: PQ's narrow adapter onto PrismaBuild's
prepaid produced-output lifecycle.

Hook and lifetime (the connection this module implements): the renderer's
anchor publication — ``tessera_campaign._finish_anchor`` — writes the
canonical rendered tensor through ``production_weight_cache.
_store_rendered_weight_entry`` (torch.save temporary -> ``os.replace`` ->
weights entry).  ``ProductionWeightCache.store_rendered_weight_published``
either REUSES an already-filed render (no prewrite, no publish, no
rewrite — a cache hit never incurs a fresh producer charge merely to
exercise the API) or wraps that exact atomic writer in the PB prepaid
sequence: ``require_prewrite`` BEFORE the first byte (a refusal leaves no
file), the existing atomic write, one payload descriptor, and
``publish_prepaid_batch`` — which seals the mover off the OWNER'S OWN
SEALED REQUEST (``producer_action_key``), so movement inputs, code
closure and execution scope derive from the admitted request and nothing
re-seals or re-hashes the source tree per output batch.  Retrying with
identical inputs re-derives the same content-addressed mover and every
step answers a typed duplicate.  ``retire_batch`` and
``safe_release_instance`` close the lifetime and return capacity.

Binding is derived, never operator-supplied: the owner action key,
nonce and scope come from the launch environment PB injected for THIS
attempt; the template comes from the submission's own
``--produced-output-template`` declaration (``declared_template``); the
instance comes from ``bind_declared_instance`` against the live claim.
The adapter imports ``prismabuild`` lazily and fails closed with a named
error when the runtime does not provide the produced-output API (the
qualified runtime pin carries it; see tests/produced_render_pb_pin.json).
"""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable


class ProducedRenderBindingError(RuntimeError):
    """The produced-render publication could not be bound or used."""


class ProducedRenderPrewriteRefused(RuntimeError):
    """The prewrite budget refused this batch BEFORE any byte.

    Carries the typed refusal; no file was written.
    """

    def __init__(self, *, batch_id: str, refusal) -> None:
        self.batch_id = batch_id
        self.refusal = refusal
        super().__init__(
            f"produced-output prewrite refused for batch {batch_id!r} "
            f"before any byte was written: {refusal}")


class ProducedRenderPublicationFailed(RuntimeError):
    """The atomic write landed but the batch publication refused.

    Retryable: identical inputs re-derive the same content-addressed
    mover and every step answers a typed duplicate.
    """

    def __init__(self, *, batch_id: str, refusal) -> None:
        self.batch_id = batch_id
        self.refusal = refusal
        super().__init__(
            f"produced-output batch {batch_id!r} failed to publish "
            f"(file is durable; retry re-derives the mover): {refusal}")


def _produced_output_module() -> Any:
    try:
        from prismabuild import produced_output as po
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ProducedRenderBindingError(
            "the produced-render publication needs PrismaBuild's "
            f"produced_output API (qualified runtime pin): {exc}") from exc
    for name in ("declare_template", "bind_declared_instance",
                 "declare_instance", "admit_instance", "admit_funded_window",
                 "require_prewrite", "publish_prepaid_batch", "commit_batch",
                 "retire_batch", "safe_release_instance",
                 "validate_descriptor", "owner_demand_terms"):
        if not callable(getattr(po, name, None)):
            raise ProducedRenderBindingError(
                "the installed prismabuild.produced_output lacks "
                f"{name!r}: a runtime without the produced-output API "
                "cannot publish renders")
    return po


def _pool_module() -> Any:
    try:
        from prismabuild import pool as pool_mod
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ProducedRenderBindingError(
            "the produced-render publication needs PrismaBuild's pool "
            f"module: {exc}") from exc
    return pool_mod


def open_pool_queue(queue_root: str | Path) -> Any:
    """The queue the admitted owner lives on (fail closed, named error)."""

    pool_mod = _pool_module()
    try:
        return pool_mod.PoolQueue(Path(queue_root))
    except Exception as exc:
        raise ProducedRenderBindingError(
            f"cannot open the PrismaBuild queue at {queue_root}: {exc}"
        ) from exc


def produced_batch_binding_validator(
        queue_root: str | Path) -> Callable[[Mapping], dict]:
    """The closed binding validator for executable readsets.

    Returns a callable that runs PrismaBuild's real
    ``PoolQueue.validate_produced_output_batch`` over a candidate
    ``render_prerequisite.binding`` with the demand the reference itself
    names (the range floor on its own tier — single-tier output movers),
    so ``joint_layer_quanta`` and the dispatcher accept exactly what the
    queue's closed validator accepts and refuse everything else.
    """

    pool_mod = _pool_module()
    try:
        from prismabuild import storage_tiers as tiers_mod
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ProducedRenderBindingError(
            f"the binding validator needs storage_tiers: {exc}") from exc

    def validate(binding: Mapping) -> dict:
        queue = open_pool_queue(queue_root)
        if not isinstance(binding, Mapping):
            raise ValueError("a render binding must be an object")
        tier = binding.get("tier_id")
        total = binding.get("range_end_bytes")
        if not isinstance(tier, str) or not tier:
            raise ValueError("a render binding names no tier")
        if not isinstance(total, int) or isinstance(total, bool) \
                or int(total) <= 0:
            raise ValueError("a render binding names no positive range")
        kind = tiers_mod.capacity_kind_of(tier)
        demand = {f"{kind}@{tier}": int(
            tiers_mod.stage_tokens_for_bytes(int(total)))}
        try:
            return dict(queue.validate_produced_output_batch(
                binding, demand))
        except pool_mod.PoolContractError as exc:
            raise ValueError(
                f"the queue refuses this produced-output batch reference: "
                f"{exc}") from exc

    return validate


def _launch_owner(env: Mapping[str, str] | None) -> str:
    source = dict(os.environ) if env is None else dict(env)
    owner = source.get("PRISMABUILD_ACTION_KEY", "")
    if not owner:
        raise ProducedRenderBindingError(
            "the produced-render publication needs this action's own key "
            "(PRISMABUILD_ACTION_KEY): it binds to the admitted owner's "
            "request, never to a caller-supplied identity")
    return str(owner)


class ProducedRenderPublication:
    """One producer action's bound produced-output publication.

    Constructed from the admitted owner's request alone (see
    :meth:`bind_from_admitted_owner`); every later call reuses the bound
    template/instance.  Nothing here re-seals the source tree: the per
    batch mover is sealed by PrismaBuild off the owner's own sealed
    request.
    """

    def __init__(self, *, queue, template, instance, tier: str,
                 cas_root: str, env: Mapping[str, str] | None = None,
                 command_extra: tuple[str, ...] = ()) -> None:
        self.queue = queue
        self.template = template
        self.instance = instance
        self.tier = str(tier)
        # The CAS root the admitted owner's own row files (its request, its
        # inputs, its batches' manifests live there).  Never guessed from
        # queue topology: the live layout is a sibling CAS, not a child.
        self.cas_root = str(cas_root)
        self.env = dict(os.environ) if env is None else dict(env)
        # Dev/fixture-only passthrough for the mover argv (e.g.
        # ``--unpaced`` where no pacer exists); production stays empty.
        self.command_extra = tuple(command_extra)
        self._po = _produced_output_module()

    # -- binding -----------------------------------------------------------

    @classmethod
    def bind_from_admitted_owner(cls, *, queue_root: str | Path, tier: str,
                                 env: Mapping[str, str] | None = None,
                                 ) -> "ProducedRenderPublication":
        """Bind template + instance from the admitted owner's request.

        The launch environment names this attempt (action key, nonce,
        scope); the submission's own ``--produced-output-template``
        declaration provides the template (``declared_template`` — no
        operator dictionary); ``bind_declared_instance`` checks the live
        claim and both identity halves.  Declares and admits the runtime
        instance, and refuses closed on every mismatch.
        """

        po = _produced_output_module()
        pool_mod = _pool_module()
        owner = _launch_owner(env)
        queue = open_pool_queue(queue_root)
        try:
            template = po.declared_template(queue, owner)
        except Exception as exc:
            raise ProducedRenderBindingError(
                f"the owner action {owner[:12]} declares no "
                f"produced-output template: {exc}") from exc
        if str(tier) not in template.get("permitted_tiers", []):
            raise ProducedRenderBindingError(
                f"tier {tier!r} is not in the declared template's "
                "permitted tiers")
        try:
            claim_snapshot = pool_mod._read_json(
                queue.item_path(pool_mod.CLAIMED, owner))
        except Exception as exc:
            raise ProducedRenderBindingError(
                f"the owner claim is unreadable: {exc}") from exc
        if not isinstance(claim_snapshot, Mapping):
            raise ProducedRenderBindingError(
                "the owner action is not claimed: a produced-render "
                "publication binds to the live admitted owner only")
        cas_root = claim_snapshot.get("cas_root")
        if not isinstance(cas_root, str) or not cas_root:
            raise ProducedRenderBindingError(
                "the admitted owner's row files no cas_root: the "
                "publication refuses to guess the CAS topology (the live "
                "CAS is a sibling of the queue, not a child)")
        try:
            instance = po.bind_declared_instance(
                queue, owner_action_key=owner,
                claim_snapshot=claim_snapshot, env=env)
            po.declare_instance(queue.root, instance)
            admitted = po.admit_instance(queue, instance, template)
        except Exception as exc:
            raise ProducedRenderBindingError(
                f"cannot bind the produced-output instance: {exc}") from exc
        if not admitted.get("ok"):
            raise ProducedRenderBindingError(
                f"the produced-output instance was not admitted: "
                f"{admitted}")
        return cls(queue=queue, template=template, instance=instance,
                   tier=tier, cas_root=cas_root, env=env)

    # -- window ------------------------------------------------------------

    def admit_window(self, need_gib: int | None = None) -> dict:
        """The funded-window binding report (declaration-only by design)."""

        demands = self.template["working_demands"][self.tier]
        need = int(demands["window_gib"]) if need_gib is None \
            else int(need_gib)
        out = self._po.admit_funded_window(
            self.queue, self.instance, self.template,
            need_gib_per_tier={self.tier: need})
        if not out.get("ok"):
            raise ProducedRenderBindingError(
                f"the funded window was not admitted: {out}")
        return dict(out)

    # -- per batch ---------------------------------------------------------

    def batch_id_for(self, qname: str, fmt: str) -> str:
        """A deterministic batch id for one rendered weight.

        Retry-stable: the same (qname, fmt) re-derives the same batch id,
        and PrismaBuild's content-addressed mover key follows from the
        same inputs, so a restart re-calls publish and meets typed
        duplicates instead of double-charging.
        """

        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{qname}.{fmt}")
        return f"render-{safe}"[:120]

    def require_prewrite(self, *, batch_id: str, class_bytes: Mapping[str, int],
                         paths: list[str]) -> dict:
        out = self._po.require_prewrite(
            self.queue, self.instance, self.template, batch_id=batch_id,
            tier=self.tier, class_bytes=class_bytes, paths=paths)
        return dict(out)

    def descriptor_for(self, path: str | Path, *, slot: str,
                       artifact_class: str,
                       producer_generation: str | None = None,
                       ) -> dict:
        """One validated descriptor for an already-written file.

        DEV contract: NO payload reread or hash.  The digest is JSON null
        (the produced-output DEV path -- identity is path+size+order
        through the manifest digest over the descriptor list, plus the
        mover's own necessary-copy/material evidence); the byte count is
        the file's current size, read from its stat.  ``producer_generation``
        must be STABLE across retries: pass the batch id (the default
        caller spelling), never a fresh mint.
        """

        file_path = Path(path)
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            raise ProducedRenderBindingError(
                f"cannot describe {file_path}: {exc}") from exc
        if size <= 0:
            raise ProducedRenderBindingError(
                f"cannot describe {file_path}: empty or missing")
        if producer_generation is None:
            raise ProducedRenderBindingError(
                "a descriptor needs a retry-stable producer_generation "
                "(the batch id), never a fresh mint")
        return self._po.validate_descriptor({
            "schema": self._po.DESCRIPTOR_SCHEMA_V2,
            "slot": slot, "artifact_class": artifact_class,
            "path": str(file_path), "bytes": int(size),
            "sha256": None,
            "producer_generation": str(producer_generation),
            "owner_action_key": self.instance["owner_action_key"],
            "owner_attempt": dict(self.instance["owner_attempt"]),
        }, self.template, self.instance)

    def publish(self, *, batch_id: str, descriptors: list,
                command_extra: tuple[str, ...] = ()) -> dict:
        """Publish one finished batch; the mover derives from the owner's
        own sealed request (``producer_action_key``), never from a
        caller-supplied template.  The CAS is the one the owner's row
        files, never a guessed queue topology."""

        out = self._po.publish_prepaid_batch(
            self.queue, self.instance, self.template, descriptors,
            batch_id=batch_id, tier=self.tier,
            cas_root=self.cas_root,
            producer_action_key=str(self.instance["owner_action_key"]),
            command_extra=tuple(command_extra))
        return dict(out)

    # -- lifetime ----------------------------------------------------------

    def retire(self, batch_id: str, *, stage_root: str | Path,
               residency_root: str | Path | None = None) -> dict:
        root = (Path(self.queue.root) / "residency"
                if residency_root is None else Path(residency_root))
        out = self._po.retire_batch(
            self.queue, self.instance, self.template, batch_id,
            stage_root=stage_root, residency_root=root)
        return dict(out)

    def release(self) -> dict:
        out = self._po.safe_release_instance(
            self.queue, self.instance, self.template)
        return dict(out) if isinstance(out, dict) else {"ok": True}
