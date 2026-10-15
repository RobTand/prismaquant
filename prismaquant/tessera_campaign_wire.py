"""Campaign wire names, coordinate custody, producer receipts and seed adoption.

The existing rendered cache owns its wire coordinate set. Selected-wire
materialization reserves the whole group before iteration; ordinary campaign
resume and seed paths reserve each admitted state before reading its wires.
Publication checks the accumulated coordinate set. Wire spellings, producer
byte checks, optional source-seal policy and copy/link behavior are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path


def wire_path(wire_dir: Path, qname: str, format_name: str) -> Path:
    return wire_dir / f"{qname.replace('.', '__')}__{format_name}.tessera"


def wire_coordinates(cache, wire_dir):
    """The wires this cache owns, independent of its rendered-only entries.

    A pre-existing manifest contributes only destinations already present in
    the wire directory. Snapshot that roster once: checking existence again
    after an alias is published would misidentify a dense-only entry as the
    new wire's owner. Fresh publications and resume/seed admission explicitly
    add their coordinates, including wires with no rendered-manifest entry.
    This is process-local campaign ownership, not a cross-process index.
    """
    coordinates = getattr(cache, "_campaign_wire_coordinates", None)
    if coordinates is None:
        coordinates = {key for key in cache.weights
                       if wire_path(wire_dir, *key).exists()}
        # Batch admission may bootstrap while the ordered writer publishes.
        # Install only once and adopt the winner's object, so a stale snapshot
        # can never replace a roster another thread has created or updated.
        coordinates = vars(cache).setdefault("_campaign_wire_coordinates", coordinates)
    return coordinates


def require_injective_anchor_filenames(cache, wire_dir, coordinates):
    """Check actual rendered and wire destinations before either write."""
    from .render_cache_paths import require_injective_cache_filenames

    coordinates = set(coordinates)
    if cache.cache_dir:
        require_injective_cache_filenames(
            set(cache.weights).union(coordinates), where="campaign rendered weights")
    require_injective_wire_filenames(
        wire_dir, wire_coordinates(cache, wire_dir).union(coordinates))


def register_wire_coordinates(cache, wire_dir, coordinates):
    """Reserve resume/seed wires before their links or receipt reads."""
    # The roster belongs to the cache object: one cache, one wire_dir.
    # This check/update is not atomic with the writer; register before publication.
    coordinates = set(coordinates)
    owned = wire_coordinates(cache, wire_dir)
    require_injective_wire_filenames(wire_dir, owned.union(coordinates))
    owned.update(coordinates)


def require_injective_wire_filenames(wire_dir, coordinates):
    # Resume and export consume wire_path: its legacy dot-to-double-
    # underscore spelling is different from the rendered-weight leaf.
    owners = {}
    for coordinate in sorted(coordinates):
        filename = str(wire_path(wire_dir, *coordinate).relative_to(wire_dir))
        previous = owners.setdefault(filename, coordinate)
        if previous != coordinate:
            raise ValueError(
                f"campaign wire shards: coordinates {previous!r} and "
                f"{coordinate!r} map to one wire destination {filename!r}; "
                "the wire shards would silently overwrite each other")


def require_checkpoint_identity_api():
    try:
        from tessera import cached_unit
    except ImportError as exc:
        raise RuntimeError(
            "Tessera campaign checkpoint identity requires the producer's "
            "cached_unit input/byte receipt API; refusing unbound resume") from exc
    return cached_unit


#: Producer provenance only; every recipe, input and sampling field still refuses.
CAMPAIGN_CHECKPOINT_SEAL_FIELDS = frozenset({"prismaquant_source_sha256", "encoder_source_sha256"})


def checkpoint_wire_record(anchor, wire_dir, identity, *, existing=None):
    """Use the producer's one receipt grammar for fresh and resumed bytes."""
    path = wire_path(wire_dir, anchor.qname, anchor.format_name)
    if path.is_symlink() or path.resolve().parent != wire_dir.resolve():
        raise RuntimeError(f"checkpoint cached wire escapes its directory: {path}")
    try:
        blob = path.read_bytes()
        api = require_checkpoint_identity_api()
        if existing is None:
            record = api.make_unit_record(blob, identity, filename=path.name)
        else:
            record = existing
            verification_identity = identity
            producer_field = "encoder_source_sha256"
            observed = record.get("identity") if isinstance(record, dict) else None
            if (producer_field in CAMPAIGN_CHECKPOINT_SEAL_FIELDS
                    and isinstance(observed, dict)
                    and producer_field in observed and producer_field in identity
                    and observed[producer_field] != identity[producer_field]):
                # Defer only this declared producer seal. The real reader still
                # verifies own bytes, grammar and every comparability field.
                verification_identity = {**identity, producer_field: observed[producer_field]}
            api.verify_cached_unit(blob, record, verification_identity)
            if record.get("file") != path.name:
                raise ValueError("cached wire filename differs from the priced unit/rung")
        # CampaignAnchor.wire_bytes means the full blob, not plane-region bytes.
        if anchor.wire_bytes != record["blob_bytes"]:
            raise ValueError("cached wire length differs from the measured anchor")
        if existing is not None and verification_identity is not identity:
            from .dev_mode import seal_check

            seal_check(f"campaign wire {producer_field}", observed[producer_field], identity[producer_field],
                       where=f"{anchor.qname}@{anchor.format_name}",
                       refusal=ValueError(f"cached unit {producer_field} identity mismatch"))
        return record
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"checkpoint cached wire identity refused for {anchor.qname} "
            f"{anchor.format_name}: {exc}") from exc


def require_seed_wire_filename(wire_dir: Path, filename, *,
                                qname: str, format_name: str) -> None:
    """Refuse a seed receipt whose file name is not its unit/rung destination.

    Pure name arithmetic: callers run it before they read or link the wire.
    """
    if not isinstance(filename, str) or not filename or "/" in filename:
        raise RuntimeError(f"seed wire receipt names an unusable file: {filename!r}")
    if filename != wire_path(wire_dir, qname, format_name).name:
        raise RuntimeError(f"{qname}: seed wire filename differs from priced unit/rung")


def link_seed_wire(seed_wire: Path, wire_dir: Path, filename, *,
                    qname: str, format_name: str) -> None:
    """Put a seed's priced wire where this run's receipt check will read it."""
    require_seed_wire_filename(wire_dir, filename, qname=qname, format_name=format_name)
    target = wire_dir / filename
    if target.exists():
        return
    source = seed_wire / filename
    if not source.is_file():
        raise RuntimeError(f"seed checkpoint has no priced wire at {source}")
    try:
        os.link(source, target)
    except OSError:
        target.write_bytes(source.read_bytes())


def expert_input_identity(api, weight, unit, fmt, activation):
    """The same served expert identity at completion, resume and finalization."""
    from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe
    from .tessera_render import rung_accepts_hessian

    family, rung = parse_tessera_format_name(fmt)
    wire = tessera_served_wire_recipe(family, rung, structure='routed_moe')
    active = (activation if activation is not None and rung_accepts_hessian(fmt, wire)
              else None)
    return api.unit_input_identity(weight, unit, family.payload_grid(), rung,
                                   activation=active, structure='routed_moe')
