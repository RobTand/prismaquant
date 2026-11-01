"""The public cache ceiling boundary accepts nonnegative integer byte counts."""
from __future__ import annotations

import pytest

from prismaquant.container_cache_peak import GIB_BYTES, derive_cache_ceiling


@pytest.mark.parametrize("peak,headroom", [
    (1.5, 0), (0, 1.5), (-0.5, 0), (0, -0.5),
    (True, 0), (False, 0), (0, True), (0, False),
], ids=[
    "fractional_peak", "fractional_headroom", "negative_fraction_peak",
    "negative_fraction_headroom", "true_peak", "false_peak",
    "true_headroom", "false_headroom",
])
def test_public_ceiling_refuses_noninteger_byte_inputs(peak, headroom):
    with pytest.raises(ValueError):
        derive_cache_ceiling(peak, headroom_bytes=headroom)


@pytest.mark.parametrize("peak,headroom,ceiling", [
    (0, 0, GIB_BYTES),
    (1, 0, GIB_BYTES),
    (GIB_BYTES, 0, GIB_BYTES),
    (GIB_BYTES, 1, 2 * GIB_BYTES),
    (GIB_BYTES + 1, GIB_BYTES, 3 * GIB_BYTES),
    (2**60 + 1, 0, 2**60 + GIB_BYTES),
], ids=["zero", "one_byte", "exact_gib", "headroom_crosses_gib", "multi_gib", "exact_large"])
def test_public_ceiling_preserves_integer_byte_contract(peak, headroom, ceiling):
    result = derive_cache_ceiling(peak, headroom_bytes=headroom)
    assert result == {
        "peak_bytes": peak,
        "headroom_bytes": headroom,
        "gib_bytes": GIB_BYTES,
        "ceiling_bytes": ceiling,
        "formula": "C = G * ceil((P + H) / G)",
    }


@pytest.mark.parametrize("peak,headroom", [(-1, 0), (0, -1)])
def test_public_ceiling_preserves_negative_integer_refusal(peak, headroom):
    with pytest.raises(ValueError, match="non-negative"):
        derive_cache_ceiling(peak, headroom_bytes=headroom)
