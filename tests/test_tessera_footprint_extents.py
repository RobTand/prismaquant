"""Use the producer's payload-free API when the installed producer supplies it."""
import pytest

from prismaquant import tessera_footprint as footprint
from tessera import layout


def test_modern_producer_pricing_never_materializes_payloads(monkeypatch):
    if not hasattr(layout, "build_plane_extents"):
        pytest.skip("installed Tessera predates the payload-free extent API")

    def forbid(*args, **kwargs):
        raise AssertionError("pricing allocated or hashed a placeholder payload")

    monkeypatch.setattr(footprint, "bytes", forbid, raising=False)
    monkeypatch.setattr(layout, "bytes", forbid, raising=False)
    monkeypatch.setattr(footprint, "build_planes", forbid)
    monkeypatch.setattr(footprint, "build_terminal", forbid)
    result = footprint.tessera_tensor_payload_breakdown(
        (2048, 4096), family="TESSERA_E4M3_K1", body_rate_q256=1024)
    assert result["payload_bytes"] > 0
    assert result["plane_elements"]


def test_older_producer_path_preserves_exact_report(monkeypatch):
    if not hasattr(layout, "build_plane_extents"):
        pytest.skip("comparison needs both producer pricing interfaces")
    kwargs = dict(family="TESSERA_E4M3_K1", body_rate_q256=1024)
    expected = footprint.tessera_tensor_payload_breakdown((2048, 4096), **kwargs)
    monkeypatch.setattr(footprint, "_build_plane_extents", None)
    monkeypatch.setattr(footprint, "_build_terminal_extent", None)
    assert footprint.tessera_tensor_payload_breakdown((2048, 4096), **kwargs) == expected
