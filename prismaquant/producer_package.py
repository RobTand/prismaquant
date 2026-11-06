"""The producer package digest, independent of Git provenance."""
from __future__ import annotations


def producer_package_sha256() -> str:
    """Bind AURA to the complete producer package, including dependencies."""
    from prismaquant.production_weight_cache import (
        _production_cache_source_sha256,
    )

    return _production_cache_source_sha256()
