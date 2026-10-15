"""Stable rendered-cache filenames and coordinate collision refusal."""
from __future__ import annotations

from collections.abc import Iterable


def cache_weight_leaf(qname: str) -> str:
    """The mangled, filesystem-safe leaf body of a qualified name.

    One home for the mangling (#2219): ``cache_weight_filename`` spells the
    shard leaf with it and ``require_injective_cache_filenames`` tests
    injectivity against it, so the two can never disagree.
    """
    return str(qname).replace("/", "__").replace(".", "_")


def cache_weight_filename(qname: str, fmt: str) -> str:
    return f"{cache_weight_leaf(qname)}__{fmt}.pt"


def require_injective_cache_filenames(
    coordinates: Iterable[tuple[str, str]],
    *,
    where: str,
) -> None:
    """Refuse two render coordinates that share one cache filename (#2219).

    The leaf mangles ``.`` -> ``_`` and ``/`` -> ``__`` to stay
    filesystem-safe, so distinct qualified names can share one shard leaf
    (``a.b`` and ``a_b``; ``layer/a`` and ``layer__a``). The stored payload is
    the bare tensor -- nothing in the file names the Linear it belongs to --
    so two coordinates that land on one file would silently read and
    overwrite each other's shard. The mangled spelling itself is load-bearing
    (archive names, existing caches), so instead of changing it this fails
    closed on the whole SET: call it where a cache directory is opened for a
    model's rendered coordinate set, before any shard is written or read --
    that is what covers resumes, pre-existing caches and concurrent
    producers. Injectivity is filename-level over ``(qname, canonical
    format)`` coordinates: ``layer.a`` and ``layer_a`` at two different
    formats name two different files and are admitted -- the layout #1859
    deliberately admits -- while the same pair at one format refuses.
    Callers pass canonical formats; ``_check_rendered_cache_destinations``
    canonicalizes the identity's ``"qname|FMT"`` pairs before delegating.
    """
    owners: dict[str, tuple[str, str]] = {}
    for coordinate in sorted(
        {(str(qname), str(fmt)) for qname, fmt in coordinates}
    ):
        filename = cache_weight_filename(*coordinate)
        previous = owners.setdefault(filename, coordinate)
        if previous != coordinate:
            raise ValueError(
                f"{where}: coordinates {previous!r} and {coordinate!r} map "
                f"to one rendered cache destination {filename!r}; the "
                f"mangled leaf is not injective and the rendered shards "
                f"would silently overwrite each other"
            )
