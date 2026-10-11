"""PQ #1293: the producer's result binds the exact published counters.

The completion also inlines the ORIGINAL quantum wire bytes the producer's own
identity gate authenticated, so a consumer can bind a pilot to the record that
actually ran without reopening a mutable path (root decision 2, 2026-10-02).
"""
import hashlib
import json

import pytest

import prismaquant.joint_cost_quantum as quantum
from prismaquant import joint_dispatch_pilot as pilot


def _record_bytes(quantum_id="layer-007"):
    return json.dumps({"quantum_id": quantum_id, "schema": "fixture"}).encode()


def _completion(result, record_bytes, **overrides):
    completion = quantum.quantum_completion_record(result, record_bytes=record_bytes)
    completion.update(overrides)
    return completion


@pytest.mark.parametrize("complete", [False, True])
def test_publisher_binds_counter_bytes_before_results_and_status(tmp_path, complete):
    record = {
        "quantum_id": "layer-007", "identity_sha256": "a" * 64,
        "output_space": {"root": str(tmp_path),
                         "cost_payload": str(tmp_path / "cost.pkl"),
                         "counters": str(tmp_path / "counters.json"),
                         "results": str(tmp_path / "results.json")},
    }
    counters = {"schema": "prismaquant.joint_layer_quantum.counters.v1",
                "quantum_id": record["quantum_id"],
                "pilot": {"action_key": "b" * 64, "binding": {"scope": "test"}},
                "units": [int(complete), 1]}
    result = {"quantum_id": record["quantum_id"]}
    wire = _record_bytes(record["quantum_id"])
    status = quantum.publish_quantum_outputs(
        record, payload={"costs": {"linear": {}}} if complete else None,
        result=result, counters=counters, units_total=1)
    raw = (tmp_path / "counters.json").read_bytes()
    reference = result["counters"]
    assert reference == {"path": str(tmp_path / "counters.json"),
                         "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert json.loads((tmp_path / "results.json").read_bytes())["counters"] == reference
    assert json.loads(raw)["outcome"]["status"] == status["status"]
    completion = quantum.quantum_completion_record(result, record_bytes=wire)
    assert completion["counters"] == reference
    assert completion["schema"] == "prismaquant.joint_layer_quantum.completion.v1"
    assert completion["passed"] is complete
    assert completion["quantum_record_bytes"] == len(wire)
    assert completion["quantum_record_sha256"] == hashlib.sha256(wire).hexdigest()
    # A rewritten document cannot retain the producer's reference, even
    # when all self-reported numeric relationships remain internally valid.
    altered = json.loads(raw)
    altered["pilot"]["binding"]["scope"] = "another-test"
    changed = (json.dumps(altered, sort_keys=True, indent=2) + "\n").encode()
    assert hashlib.sha256(changed).hexdigest() != reference["sha256"]
    if complete:
        completion = quantum.quantum_completion_record(result, record_bytes=wire)
        assert pilot.validate_pilot_completion(
            completion, counters=counters, counters_sha256=reference["sha256"],
            counters_bytes=len(raw), quantum_record_sha256=hashlib.sha256(wire).hexdigest(),
            quantum_record_bytes=len(wire))["quantum_record"] == wire
        with pytest.raises(pilot.PilotRefused, match="bytes differ"):
            pilot.validate_pilot_completion(
                completion, counters=altered, counters_sha256=hashlib.sha256(changed).hexdigest(),
                counters_bytes=len(changed), quantum_record_sha256=hashlib.sha256(wire).hexdigest(),
                quantum_record_bytes=len(wire))


def test_failed_counter_write_publishes_no_reference_or_success_files(tmp_path, monkeypatch):
    record = {"quantum_id": "layer-000", "identity_sha256": "a" * 64,
              "output_space": {"root": str(tmp_path),
                               "cost_payload": str(tmp_path / "cost.pkl"),
                               "counters": str(tmp_path / "counters.json"),
                               "results": str(tmp_path / "results.json")}}
    result = {}
    real_write = quantum.atomic_write_bytes

    def fail_counter(path, raw):
        if path.name == "counters.json":
            raise OSError("counter publication failed")
        return real_write(path, raw)

    monkeypatch.setattr(quantum, "atomic_write_bytes", fail_counter)
    with pytest.raises(OSError, match="counter publication"):
        quantum.publish_quantum_outputs(record, payload={"costs": {"a": {}}},
                                        result=result, counters={}, units_total=1)
    assert "counters" not in result
    assert not (tmp_path / "results.json").exists()
    assert not (tmp_path / "status.json").exists()


@pytest.mark.parametrize("fault", ["schema", "quantum", "status", "passed", "units",
                                    "digest", "length", "bool_length", "missing"])
def test_pb_completion_refuses_foreign_or_malformed_reference(fault):
    counters = {"quantum_id": "layer-007", "units": [3, 3]}
    wire = _record_bytes()
    completion = {"schema": pilot.QUANTUM_COMPLETION_SCHEMA,
                  "quantum_id": "layer-007", "passed": True, "status": "complete",
                  "units_done": 3, "units_total": 3,
                  "counters": {"path": "/original/counters.json",
                               "sha256": "a" * 64, "bytes": 100},
                  "quantum_record": __import__("base64").b64encode(wire).decode(),
                  "quantum_record_bytes": len(wire),
                  "quantum_record_sha256": hashlib.sha256(wire).hexdigest()}
    if fault == "schema":
        completion["schema"] = "another-producer"
    elif fault == "quantum":
        completion["quantum_id"] = "layer-008"
    elif fault == "status":
        completion["status"] = "gapped"
    elif fault == "passed":
        completion["passed"] = 1
    elif fault == "units":
        completion["units_done"] = 1
    elif fault == "digest":
        completion["counters"]["sha256"] = "b" * 64
    elif fault == "length":
        completion["counters"]["bytes"] = 99
    elif fault == "bool_length":
        completion["counters"]["bytes"] = True
    elif fault == "missing":
        del completion["counters"]
    with pytest.raises(pilot.PilotRefused):
        pilot.validate_pilot_completion(completion, counters=counters,
                                        counters_sha256="a" * 64, counters_bytes=100,
                                        quantum_record_sha256=hashlib.sha256(wire).hexdigest(),
                                        quantum_record_bytes=len(wire))


@pytest.mark.parametrize("fault", ["missing_record", "bad_base64", "wrong_length",
                                   "wrong_digest", "oversized", "other_record_sha"])
def test_pb_completion_refuses_a_foreign_or_malformed_quantum_record(fault):
    """The inlined ORIGINAL record must decode to the sealed bytes."""
    import base64

    counters = {"quantum_id": "layer-007", "units": [3, 3]}
    wire = _record_bytes()
    sha = hashlib.sha256(wire).hexdigest()
    completion = {"schema": pilot.QUANTUM_COMPLETION_SCHEMA,
                  "quantum_id": "layer-007", "passed": True, "status": "complete",
                  "units_done": 3, "units_total": 3,
                  "counters": {"path": "/original/counters.json",
                               "sha256": "a" * 64, "bytes": 100},
                  "quantum_record": base64.b64encode(wire).decode(),
                  "quantum_record_bytes": len(wire),
                  "quantum_record_sha256": sha}
    sealed_sha, sealed_bytes = sha, len(wire)
    if fault == "missing_record":
        del completion["quantum_record"]
    elif fault == "bad_base64":
        completion["quantum_record"] = "not base64 !!!"
    elif fault == "wrong_length":
        sealed_bytes = len(wire) + 1
    elif fault == "wrong_digest":
        sealed_sha = "b" * 64
    elif fault == "oversized":
        completion["quantum_record"] = "A" * (4 * ((len(wire) + 2) // 3) + 4)
    elif fault == "other_record_sha":
        completion["quantum_record"] = base64.b64encode(
            _record_bytes("layer-999")).decode()
        completion["quantum_record_bytes"] = len(wire)
    with pytest.raises(pilot.PilotRefused):
        pilot.validate_pilot_completion(completion, counters=counters,
                                        counters_sha256="a" * 64, counters_bytes=100,
                                        quantum_record_sha256=sealed_sha,
                                        quantum_record_bytes=sealed_bytes)


@pytest.mark.parametrize("over_cap", [False, True])
def test_record_cap_is_enforced_before_decode_or_encode(monkeypatch, over_cap):
    wire = b"x" * (pilot.QUANTUM_RECORD_MAX_BYTES + int(over_cap))
    result = {"quantum_id": "layer-007", "passed": True, "status": "complete",
              "units_done": 1, "units_total": 1,
              "counters": {"path": "/counters", "sha256": "a" * 64, "bytes": 3}}
    if over_cap:
        monkeypatch.setattr(quantum.base64, "b64encode", lambda raw: pytest.fail("must refuse before encoding"))
        with pytest.raises(quantum.QuantumIdentityRefused, match="cap"):
            quantum.quantum_completion_record(result, record_bytes=wire)
        completion = {**result, "schema": pilot.QUANTUM_COMPLETION_SCHEMA,
                      "quantum_record": "AAAA", "quantum_record_bytes": len(wire),
                      "quantum_record_sha256": hashlib.sha256(wire).hexdigest()}
        monkeypatch.setattr(pilot.base64, "b64decode", lambda *a, **k: pytest.fail("must refuse before decoding"))
        with pytest.raises(pilot.PilotRefused, match="cap"):
            pilot.validate_pilot_completion(
                completion, counters={"quantum_id": "layer-007", "units": [1, 1]},
                counters_sha256="a" * 64, counters_bytes=3,
                quantum_record_sha256=completion["quantum_record_sha256"])
    else:
        completion = quantum.quantum_completion_record(result, record_bytes=wire)
        evidence = pilot.validate_pilot_completion(
            completion, counters={"quantum_id": "layer-007", "units": [1, 1]},
            counters_sha256="a" * 64, counters_bytes=3,
            quantum_record_sha256=completion["quantum_record_sha256"])
        assert evidence["quantum_record"] == wire


def test_sealed_record_passes_raised_cap():
    """PQ #2603: the 2.7 MB sealed record passes the 4 MiB cap."""
    wire = b"y" * int(2.7 * 1024 * 1024)
    assert len(wire) < pilot.QUANTUM_RECORD_MAX_BYTES
    result = {"quantum_id": "layer-007", "passed": True, "status": "complete",
              "units_done": 1, "units_total": 1,
              "counters": {"path": "/counters", "sha256": "a" * 64, "bytes": 3}}
    completion = quantum.quantum_completion_record(result, record_bytes=wire)
    evidence = pilot.validate_pilot_completion(
        completion, counters={"quantum_id": "layer-007", "units": [1, 1]},
        counters_sha256="a" * 64, counters_bytes=3,
        quantum_record_sha256=completion["quantum_record_sha256"])
    assert evidence["quantum_record"] == wire


def test_record_over_4mib_refuses():
    """PQ #2603: a record above 4 MiB still refuses the cap."""
    wire = b"z" * (4 * 1024 * 1024 + 1)
    result = {"quantum_id": "layer-007", "passed": True, "status": "complete",
              "units_done": 1, "units_total": 1,
              "counters": {"path": "/counters", "sha256": "a" * 64, "bytes": 3}}
    with pytest.raises(quantum.QuantumIdentityRefused, match="cap"):
        quantum.quantum_completion_record(result, record_bytes=wire)
