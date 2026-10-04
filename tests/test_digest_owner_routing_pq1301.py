"""Self-source binding for the PQ #1301 digest-owner routing.

The routed call sites keep the consumer behavior tests that already exercise
them; owner-versus-hashlib equivalence is a property of the owners and is
not re-asserted here. This file holds only the self-source binding:
``kernels.kda_chunk.source_sha256`` must digest this tree's exact kernel
bytes, whichever owner performs the hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def test_migrated_self_source_digest_still_binds_this_tree():
    """The kda_chunk self-source digest: same bytes, whatever hashes them."""
    from prismaquant.kernels import kda_chunk

    expected = hashlib.sha256(
        Path(kda_chunk.__file__).read_bytes()).hexdigest()
    assert kda_chunk.source_sha256() == expected
