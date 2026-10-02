"""PQ #1293: the producer's result binds the exact published counters."""
import hashlib
import json

import pytest

import prismaquant.joint_cost_quantum as quantum
from prismaquant import joint_dispatch_pilot as pilot


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
    status = quantum.publish_quantum_outputs(
        record, payload={"costs": {"linear": {}}} if complete else None,
        result=result, counters=counters, units_total=1)
    raw = (tmp_path / "counters.json").read_bytes()
    reference = result["counters"]
    assert reference == {"path": str(tmp_path / "counters.json"),
                         "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert json.loads((tmp_path / "results.json").read_bytes())["counters"] == reference
    assert json.loads(raw)["outcome"]["status"] == status["status"]
    assert quantum.quantum_completion_record(result)["counters"] == reference
    assert quantum.quantum_completion_record(result)["schema"] == (
        "prismaquant.joint_layer_quantum.completion.v1")
    assert quantum.quantum_completion_record(result)["passed"] is complete
    # A rewritten document cannot retain the producer's reference, even
    # when all self-reported numeric relationships remain internally valid.
    altered = json.loads(raw)
    altered["pilot"]["binding"]["scope"] = "another-test"
    changed = (json.dumps(altered, sort_keys=True, indent=2) + "\n").encode()
    assert hashlib.sha256(changed).hexdigest() != reference["sha256"]
    if complete:
        completion = quantum.quantum_completion_record(result)
        assert pilot.validate_pilot_completion(
            completion, counters=counters, counters_sha256=reference["sha256"],
            counters_bytes=len(raw)) == reference
        with pytest.raises(pilot.PilotRefused, match="bytes differ"):
            pilot.validate_pilot_completion(
                completion, counters=altered, counters_sha256=hashlib.sha256(changed).hexdigest(),
                counters_bytes=len(changed))


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
    completion = {"schema": pilot.QUANTUM_COMPLETION_SCHEMA,
                  "quantum_id": "layer-007", "passed": True, "status": "complete",
                  "units_done": 3, "units_total": 3,
                  "counters": {"path": "/original/counters.json",
                               "sha256": "a" * 64, "bytes": 100}}
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
                                        counters_sha256="a" * 64, counters_bytes=100)
