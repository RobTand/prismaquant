"""Bound intake bytes precede scientific/capture work in the staged readset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from prismaquant.stage_inputs import read_bound
from prismaquant.tessera_acquisition_inputs import joint_campaign_acquisition_control_inputs
from prismaquant.tessera_full_domain_acquisition import load_joint_campaign_acquisition
from test_campaign_acquisition_intake import bound_document, two_unit_payload
from test_campaign_acquisition_projection import full_document


def test_actual_request_and_cost_are_ordered_whole_bound_inputs(
        two_unit_payload, full_document, tmp_path):
    binding, document = bound_document(tmp_path, full_document, two_unit_payload)
    request_path, cost_path = tmp_path / "request.json", tmp_path / "joint.pkl"
    request_bytes, cost_bytes = request_path.read_bytes(), cost_path.read_bytes()
    assert joint_campaign_acquisition_control_inputs(binding) == [
        {**binding, "bytes": len(request_bytes)},
        {"path": str(cost_path), "sha256": document["cost_sha256"], "bytes": len(cost_bytes)},
    ]
    assert load_joint_campaign_acquisition(binding)["identity"]["cost_sha256"] == document["cost_sha256"]
    assert request_path.read_bytes() == request_bytes
    assert cost_path.read_bytes() == cost_bytes


@pytest.mark.parametrize("filename", ["request.json", "joint.pkl"])
@pytest.mark.parametrize("reader", [None, read_bound], ids=["metadata_stream", "staged_bound_cache"])
def test_control_input_drift_refuses_even_after_a_cached_bound_read(
        filename, reader, two_unit_payload, full_document, tmp_path):
    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    joint_campaign_acquisition_control_inputs(binding, reader=reader)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        joint_campaign_acquisition_control_inputs(binding, reader=reader)


@pytest.mark.parametrize("raw, message", [
    (b'{"schema":"first","schema":"second"}', "duplicate JSON key"),
    (b'{"schema":NaN}', "nonfinite JSON value"),
    (b'{"schema":"unrelated"}', "campaign request schema"),
])
def test_readset_reuses_strict_bound_request_document_owner(raw, message, tmp_path):
    path = tmp_path / "request.json"
    path.write_bytes(raw)
    binding = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match=message):
        joint_campaign_acquisition_control_inputs(binding)


def test_real_bound_control_inputs_construct_without_site_or_package_bootstrap(
        two_unit_payload, full_document, tmp_path):
    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    owner = Path(__file__).resolve().parents[1] / "prismaquant" / "tessera_acquisition_inputs.py"
    program = (
        "import importlib.util,json,sys; "
        "spec=importlib.util.spec_from_file_location('acquisition_metadata_owner',sys.argv[1]); "
        "owner=importlib.util.module_from_spec(spec); spec.loader.exec_module(owner); "
        "entries=owner.joint_campaign_acquisition_control_inputs(json.loads(sys.argv[2])); "
        "assert 'prismaquant' not in sys.modules and 'torch' not in sys.modules; "
        "print(json.dumps(entries))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(owner), json.dumps(binding)],
        check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == joint_campaign_acquisition_control_inputs(binding)


def test_one_metadata_pair_reuses_cost_hash_but_not_mutable_return_values(
        two_unit_payload, full_document, tmp_path, monkeypatch):
    from prismaquant import tessera_acquisition_inputs as metadata

    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    monkeypatch.setattr(metadata, "_VERIFIED_CONTROL_INPUTS", None)
    actual_hash = metadata._digests.file_sha256hex
    hashed = []

    def counted_hash(path):
        hashed.append(str(path))
        return actual_hash(path)

    monkeypatch.setattr(metadata._digests, "file_sha256hex", counted_hash)
    first = metadata.joint_campaign_acquisition_control_inputs(binding)
    expected = [dict(entry) for entry in first]
    first[0]["sha256"] = "0" * 64
    first[1]["bytes"] = -1
    assert metadata.joint_campaign_acquisition_control_inputs(binding) == expected
    assert hashed == [str(tmp_path / "joint.pkl")]
    # No raw request/cost bytes are retained by the bounded metadata memo.
    held, fences = metadata._VERIFIED_CONTROL_INPUTS
    assert held == expected and len(held) == len(fences) == 2
    assert all(set(entry) == {"path", "sha256", "bytes"} for entry in held)


def test_request_drift_during_cost_hash_cannot_seed_a_reusable_metadata_pair(
        two_unit_payload, full_document, tmp_path, monkeypatch):
    from prismaquant import tessera_acquisition_inputs as metadata

    binding, _ = bound_document(tmp_path, full_document, two_unit_payload)
    monkeypatch.setattr(metadata, "_VERIFIED_CONTROL_INPUTS", None)
    actual_hash = metadata._digests.file_sha256hex

    def hash_then_drift_request(path):
        digest = actual_hash(path)
        request = tmp_path / "request.json"
        request.write_bytes(request.read_bytes() + b"\n")
        return digest

    monkeypatch.setattr(metadata._digests, "file_sha256hex", hash_then_drift_request)
    with pytest.raises(ValueError, match="changed during declaration"):
        metadata.joint_campaign_acquisition_control_inputs(binding)
    assert metadata._VERIFIED_CONTROL_INPUTS is None
