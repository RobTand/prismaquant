"""Render-specific binding and planning on PrismaBuild's existing lifecycle.

This adapter does not enable produced-output writes in ProductionWeightCache.
Its plans accept explicit reservation envelopes, not tensors or archive-size
estimates. Writer and reader integration remain separate. PrismaBuild owns
admission, immutable prewrite accounting, publication and retirement.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

from .digests import DIRECT_ASCII_STRICT
from .stage_a_produced_output import BoundaryProducedPublication


@dataclass(frozen=True, slots=True)
class ProducedRenderWritePlan:
    """Derived destinations and ceilings; not permission to write or read."""

    qname: str
    fmt: str
    owner_action_key: str
    template_sha256: str
    owner_nonce: str
    owner_scope_id: str
    batch_id: str
    origin_root: str
    attempt_component: str
    relative_final_path: str
    final_path: str
    temporary_path: str
    payload_ceiling_bytes: int
    temp_ceiling_bytes: int
    producer_generation: str


def _render_coordinate(qname: str, fmt: str) -> tuple[str, str]:
    if (not isinstance(qname, str) or not qname
            or not isinstance(fmt, str) or not fmt):
        raise ValueError("render coordinates must be nonempty strings")
    from .format_registry import canonical_format_name

    fmt = canonical_format_name(fmt.strip().upper())
    if not fmt:
        raise ValueError("render coordinates must be nonempty strings")
    return qname, fmt


def _require_available_destination(root: Path, path: Path) -> None:
    """Refuse known occupancy, symlinks or unreadable metadata, without IO writes."""
    relative = path.relative_to(root)
    candidates = [root] + [root.joinpath(*relative.parts[:i])
                           for i in range(1, len(relative.parts) + 1)]
    for current in candidates:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("cannot stat render destination") from exc
        if stat.S_ISLNK(mode):
            raise ValueError("symlink render destination is unsupported")
        if current == path or not stat.S_ISDIR(mode):
            raise ValueError("occupied render destination is unsupported")


class ProducedRenderPublication(BoundaryProducedPublication):
    """One admitted action's rendered-weight payload publication.

    Bind using the owner's sealed template and live attempt. Rendered weights
    are read by their producer, so a write-only template is not a render lane.
    No caller-supplied owner, generation, CAS topology or budget is invented.
    """

    DEFAULT_SLOT = "renders"
    REQUIRED_SLOT_CLASS = "payload"
    REQUIRES_READBACK = True

    def render_batch_id_for(self, qname: str, fmt: str) -> str:
        """Stable identity for the full render coordinate of this owner.

        Use the weight writer's registry-canonical format spelling. Do not
        strip, sanitize or truncate the qualified Linear name: distinct entries
        must not share prewrite credit. The bound template and full owner key
        scope this identity; rebinding and retrying that owner remain stable,
        as in the shared boundary adapter. PB still authenticates the attempt.
        This is not a payload/content digest or a PB generation identifier.
        """
        qname, fmt = _render_coordinate(qname, fmt)
        coordinate = [
            "prismaquant.render_batch.v1", self.instance["owner_action_key"],
            self.instance["template_sha256"], qname, fmt,
        ]
        return "render-" + DIRECT_ASCII_STRICT.sha256(coordinate)

    def plan_render_prewrite(self, qname: str, fmt: str, *,
                             archive_max_bytes: int) -> ProducedRenderWritePlan:
        """Plan disjoint origins and an explicit payload/temp reservation.

        No directories, tensors or archives are created. The envelope must be
        supplied by the caller; this method does not prove a tensor's eventual
        serialized size. The deterministic temporary path is for planning,
        not authorization for concurrent writers or existing-byte adoption.
        """
        qname, fmt = _render_coordinate(qname, fmt)
        if type(archive_max_bytes) is not int or archive_max_bytes <= 0:
            raise ValueError("render archive ceiling must be a positive integer")
        maxima = self.durable_maxima()
        if archive_max_bytes > min(maxima["payload_max_bytes"], maxima["temp_max_bytes"]):
            raise ValueError("render archive ceiling exceeds sealed payload/temp maxima")

        from .production_weight_cache import _cache_weight_filename

        leaf = _cache_weight_filename(qname, fmt)
        if (Path(leaf).name != leaf or "\\" in leaf
                or any(ord(c) < 32 or ord(c) == 127 for c in leaf)):
            raise ValueError("unsafe render archive leaf")
        try:
            leaf_bytes = len((leaf + ".tmp").encode("utf-8"))
        except UnicodeError as exc:
            raise ValueError("unsupported render archive filename encoding") from exc
        if leaf_bytes > 255:
            raise ValueError("render archive filename exceeds the supported leaf bound")

        owner = str(self.instance["owner_action_key"])
        template = str(self.instance["template_sha256"])
        attempt = self.instance["owner_attempt"]
        nonce, scope = str(attempt["nonce"]), str(attempt["scope_id"])
        component = "attempt-" + DIRECT_ASCII_STRICT.sha256([
            "prismaquant.render_attempt.v1", owner, template, nonce, scope])
        batch = self.render_batch_id_for(qname, fmt)
        root = Path(os.path.normpath(self.output_prefix))
        if not root.is_absolute():
            raise ValueError("render origin root must be absolute")
        relative = Path("renders-v1") / component / batch / leaf
        final = root / relative
        temporary = final.with_name(final.name + ".tmp")
        for path in (final, temporary):
            if not self.contains(path):
                raise ValueError("render destination fails origin containment")
            _require_available_destination(root, path)
        return ProducedRenderWritePlan(
            qname=qname, fmt=fmt, owner_action_key=owner,
            template_sha256=template, owner_nonce=nonce, owner_scope_id=scope,
            batch_id=batch, origin_root=str(root), attempt_component=component,
            relative_final_path=str(relative), final_path=str(final),
            temporary_path=str(temporary), payload_ceiling_bytes=archive_max_bytes,
            temp_ceiling_bytes=archive_max_bytes, producer_generation=batch)

    def require_render_prewrite(self, plan: ProducedRenderWritePlan) -> dict:
        """Revalidate a plan, then delegate immutable accounting to PrismaBuild."""
        if type(plan) is not ProducedRenderWritePlan:
            raise ValueError("render plan must be ProducedRenderWritePlan")
        expected = self.plan_render_prewrite(
            plan.qname, plan.fmt, archive_max_bytes=plan.payload_ceiling_bytes)
        if plan != expected:
            raise ValueError("render plan does not match this publication")
        return self.require_prewrite(
            batch_id=plan.batch_id, payload_ceiling_bytes=plan.payload_ceiling_bytes,
            temp_ceiling_bytes=plan.temp_ceiling_bytes,
            paths=[plan.final_path, plan.temporary_path])
