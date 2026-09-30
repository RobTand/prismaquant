"""Render-specific binding on the existing PrismaBuild publication lifecycle.

This adapter does not enable produced-output writes in ProductionWeightCache.
The writer and reader must opt into that integration separately. Admission,
prewrite credit, descriptor validation, publication and retirement remain in
BoundaryProducedPublication and PrismaBuild's public produced-output API.
"""
from __future__ import annotations

from .digests import DIRECT_ASCII_STRICT
from .stage_a_produced_output import BoundaryProducedPublication


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

        if (not isinstance(qname, str) or not qname
                or not isinstance(fmt, str) or not fmt):
            raise ValueError("render coordinates must be nonempty strings")
        from .format_registry import canonical_format_name

        fmt = canonical_format_name(fmt.strip().upper())
        if not fmt:
            raise ValueError("render coordinates must be nonempty strings")
        coordinate = [
            "prismaquant.render_batch.v1", self.instance["owner_action_key"],
            self.instance["template_sha256"], qname, fmt,
        ]
        return "render-" + DIRECT_ASCII_STRICT.sha256(coordinate)
