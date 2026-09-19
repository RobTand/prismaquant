"""PQ #725: ``verify_anchor_render`` binds the read-ahead reader's fenced
``(blob, digest)`` pair instead of hashing the same bytes a third time.
The receipt checksum still refuses; only the redundant pass is gone."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _anchor_cell(tmp_path):
    import hashlib

    source = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
    rendered = torch.ones_like(source)
    anchor = dict(qname="model.layers.0.q_proj", format_name="TESSERA_E4M3_K1_R1024",
        family="TESSERA_E4M3_K1", body_rate_q256=1024, dloss=0.25, dloss_stderr=0.0,
        memory_bytes=256, bits_per_param=8.0, activation_contract="fp8_e4m3",
        activation_quantized=True, wire_bytes=4, seconds=0.1, hessian_applied=True,
        input_global_scale=None)
    wire = tmp_path / "fixture.wire"; wire.write_bytes(b"wire")
    digest = hashlib.sha256(b"wire").hexdigest()
    receipt = {"expected": "derived", "blob_bytes": 4, "blob_sha256": digest}
    cell = {"anchor": anchor, "record": receipt, "wire": str(wire),
            "render_file_sha256": "a" * 64, "render_origin": "encoded"}
    return source, rendered, cell, digest


def _patched_verify(monkeypatch, rendered):
    from prismaquant import tessera_campaign as tc
    from tessera import unit_artifact

    monkeypatch.setattr(tc, "_checkpoint_anchor_identity",
                        lambda *_a, **_k: {"expected": "derived"})
    monkeypatch.setattr(tc, "_checkpoint_identity_api",
                        lambda: SimpleNamespace(verify_cached_unit=lambda *_a: None))
    monkeypatch.setattr(unit_artifact, "read_unit_artifact",
                        lambda blob, device="cpu": rendered.clone())


def test_binds_reader_digest_without_rehash(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import verify_anchor_render

    source, rendered, cell, digest = _anchor_cell(tmp_path)
    _patched_verify(monkeypatch, rendered)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={},
                  wire_blob=b"wire", wire_sha256=digest)
    assert verify_anchor_render(cell, source, rendered, **kwargs)["wire_sha256"] == digest


def test_refuses_digest_receipt_mismatch(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import verify_anchor_render

    source, rendered, cell, digest = _anchor_cell(tmp_path)
    _patched_verify(monkeypatch, rendered)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={},
                  wire_blob=b"wire", wire_sha256="0" * 64)
    with pytest.raises(ValueError, match="wire checksum"):
        verify_anchor_render(cell, source, rendered, **kwargs)


def test_refuses_malformed_reader_digest(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import verify_anchor_render

    source, rendered, cell, digest = _anchor_cell(tmp_path)
    _patched_verify(monkeypatch, rendered)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={},
                  wire_blob=b"wire", wire_sha256="not-a-digest")
    with pytest.raises(ValueError, match="64-hex digest required"):
        verify_anchor_render(cell, source, rendered, **kwargs)


def test_hashes_without_reader_digest(tmp_path, monkeypatch):
    """No bound pair, no trust: the bytes are hashed as before."""
    from prismaquant.tessera_joint_aura import verify_anchor_render

    source, rendered, cell, digest = _anchor_cell(tmp_path)
    _patched_verify(monkeypatch, rendered)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={},
                  wire_blob=b"wire")
    assert verify_anchor_render(cell, source, rendered, **kwargs)["wire_sha256"] == digest
