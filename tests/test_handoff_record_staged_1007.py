"""Band-serial record metadata is declared and read from existing PB material."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_quantum_handoff as handoffs
from prismaquant.staged_tier_policy import TierPolicyRefused
from test_band_serial_dispatch import _campaign, _emit, _slice
from test_streamed_metadata_staged_reads import _activate, _deny_pool_opens
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


@pytest.fixture
def emitted(tmp_path, monkeypatch):
    bound, _receipt = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    source = _slice(consumer)
    document = handoffs.load_quantum_handoff(
        published["path"], published["sha256"], record=consumer,
        adjoint_slice=source, kda_capture_kernel=None)
    path = Path(published["path"])
    anchor = tmp_path / "anchor.meta"
    anchor.write_bytes(b"independent declared metadata keeps a missing-record map valid")
    manifest = {"entries": [
        {"path": str(path), "offset": 0, "bytes": path.stat().st_size,
         "sha256": published["sha256"]},
        {"path": str(anchor), "offset": 0, "bytes": anchor.stat().st_size,
         "sha256": hashlib.sha256(anchor.read_bytes()).hexdigest()},
    ]}
    return consumer, source, published, document, manifest


def test_bootstrap_head_declares_exact_handoff_record(emitted, tmp_path):
    consumer, source, published, document, _manifest = emitted
    wire = handoffs.band_serial_manifest_bytes(
        consumer, document, source["checkpoint"], output_root=tmp_path)
    derived = json.loads(gzip.decompress(wire))
    head = derived["read_plan"]["phases"][0]
    assert head["name"] == "head"
    rows = [derived["entries"][i] for i in head["entry_indices"]]
    expected = {"path": published["path"], "offset": 0,
                "bytes": len(handoffs.handoff_record_bytes(document)),
                "sha256": published["sha256"]}
    assert expected in rows
    assert len([r for r in rows if r["path"] == expected["path"]]) == 1


def test_record_reader_uses_material_without_pool_open(emitted, tmp_path, monkeypatch):
    consumer, source, published, document, manifest = emitted
    _activate(tmp_path, monkeypatch, manifest)
    path = Path(published["path"])
    _deny_pool_opens(monkeypatch, path.parent, names=(path.name,))
    actual = handoffs.load_quantum_handoff(
        path, published["sha256"], record=consumer, adjoint_slice=source,
        kda_capture_kernel=None)
    assert actual == document


@pytest.mark.parametrize("fault", ["missing", "digest"])
def test_record_material_failure_never_falls_back(emitted, tmp_path, monkeypatch, fault):
    consumer, source, published, _document, manifest = emitted
    path = Path(published["path"])
    _activate(tmp_path, monkeypatch, manifest,
              skip={str(path)} if fault == "missing" else (),
              corrupt={str(path)} if fault == "digest" else ())
    _deny_pool_opens(monkeypatch, path.parent, names=(path.name,))
    with pytest.raises((TierPolicyRefused, RuntimeError), match="material|digest|staged|bound"):
        handoffs.load_quantum_handoff(
            path, published["sha256"], record=consumer, adjoint_slice=source,
            kda_capture_kernel=None)


def test_inactive_record_bytes_and_identity_checks_are_unchanged(emitted):
    consumer, source, published, document, _manifest = emitted
    path = Path(published["path"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == published["sha256"]
    assert handoffs.load_quantum_handoff(
        path, published["sha256"], record=consumer, adjoint_slice=source,
        kda_capture_kernel=None) == document
    with pytest.raises(handoffs.QuantumHandoffRefused, match="does not hash"):
        handoffs.load_quantum_handoff(
            path, "0" * 64, record=consumer, adjoint_slice=source,
            kda_capture_kernel=None)
