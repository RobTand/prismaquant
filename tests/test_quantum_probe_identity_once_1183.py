"""A layer quantum validates and hashes its probe identity once (PQ #1183).

The probe identity carries the source model's identity, 8.8 MB of JSON for
GLM-5.3. v7 layer-044 spent about 57 s of main thread per window, with the
GPU idle, serializing it again for every operator and every (unit, format)
row. These tests count those serializations inside the quantum, which must
not grow with the number of rows, and check that the rows the quantum
returns stay ordinary JSON data carrying the identity that was validated.
"""
import pickle

import pytest

import prismaquant.aura_cost as aura
from prismaquant import cost_streaming, joint_aura
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_cost_stage_a import run_adjoint_capture_core

import test_joint_cost_quantum_runtime as runtime
from test_layer_major_boundary_capture import draw
from test_streamed_cost_checkpoints import _model_identity

PROBES_SCHEMA = "prismaquant.joint_aura.probes.v2"


def _campaign(tmp_path, monkeypatch):
    single_root = tmp_path / "single"
    single = runtime._single_run(single_root, monkeypatch,
                                 checkpoint=single_root / "checkpoints")
    output_root = tmp_path / "campaign"
    runner_a, _ = runtime._stage_a(tmp_path, monkeypatch)
    runner_a.context.settle_prefetch_layers = lambda layers: None
    receipt = run_adjoint_capture_core(
        runner_a, draw(), execution=runtime._execution(tmp_path),
        output_root=output_root, stride=2,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256=runtime._hex("a"), plan_sha256=runtime._hex("d"),
        prepared_sha256=runtime._hex("e"), read_manifest_sha256=runtime._hex("f"),
        implementation_sha256=aura._aura_source_sha256())
    return single, receipt, output_root


def _counted_quantum(tmp_path, monkeypatch, single, receipt, output_root):
    """Run one quantum; count probe-identity work inside the core only."""
    counts = {"validations": 0, "plain_probe_hashes": 0}
    active = []
    validate = cost_streaming.validate_streamed_model_identity
    digest = joint_aura.identity_sha256
    core = runtime.run_layer_quantum_core

    def counted_validate(*args, **kwargs):
        if active:
            counts["validations"] += 1
        return validate(*args, **kwargs)

    def counted_digest(value):
        if active and type(value) is dict and value.get("schema") == PROBES_SCHEMA:
            counts["plain_probe_hashes"] += 1
        return digest(value)

    def counted_core(*args, **kwargs):
        active.append(True)
        try:
            return core(*args, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(cost_streaming, "validate_streamed_model_identity",
                        counted_validate)
    monkeypatch.setattr(joint_aura, "identity_sha256", counted_digest)
    monkeypatch.setattr(runtime, "run_layer_quantum_core", counted_core)
    payload, _record, _counters = runtime._run_quantum(
        tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
        output_root=output_root, plan_sha=runtime._hex("d"),
        prepared_sha=runtime._hex("e"))
    return payload, counts


def test_a_quantum_validates_and_hashes_its_probe_identity_once(tmp_path, monkeypatch):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    payload, counts = _counted_quantum(tmp_path, monkeypatch, single, receipt,
                                       output_root)
    rows = [row for per_unit in payload["costs"].values() for row in per_unit.values()]
    assert len(rows) >= 3
    # Once when the head reads the source model identity, and once when the
    # probe identity is final. Never once per row.
    assert counts["validations"] <= 2, (
        f"the source model identity was validated {counts['validations']} times "
        f"for {len(rows)} rows")
    # The payload's provenance digest is the one plain serialization left.
    assert counts["plain_probe_hashes"] <= 1, (
        f"the probe identity was serialized {counts['plain_probe_hashes']} times "
        f"for {len(rows)} rows")


def test_the_rows_carry_the_ordinary_probe_identity(tmp_path, monkeypatch):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    payload, _counts = _counted_quantum(tmp_path, monkeypatch, single, receipt,
                                        output_root)
    probe = payload["provenance"]["probe_identity"]
    digest = joint_aura.identity_sha256(probe)
    assert type(probe) is dict
    assert payload["provenance"]["probe_identity_sha256"] == digest
    # In memory, as the quantum returns them: consumers digest rows as JSON
    # before any pickle restores them.
    for per_unit in payload["costs"].values():
        for row in per_unit.values():
            assert row["probe_identity"] is probe
            canonical_json_sha256(row, where="cost row")
            assert row["probe_identity_sha256"] == digest
            assert row["joint_operator_identity"]["probe_identity_sha256"] == digest
            assert joint_aura.validate_joint_aura_entry(row)
    # cost.pkl and the unit checkpoints are pickles; the shared identity
    # restores as one ordinary dict.
    restored = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    for per_unit in restored["costs"].values():
        for row in per_unit.values():
            assert type(row["probe_identity"]) is dict
            assert row["probe_identity"] == probe


def test_a_probe_identity_changed_after_validation_refuses(tmp_path, monkeypatch):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    validated = joint_aura.validated_probe_identity

    def validated_then_changed(probe):
        result = validated(probe)
        probe["changed_after_validation"] = True
        return result

    monkeypatch.setattr(joint_aura, "validated_probe_identity", validated_then_changed)
    with pytest.raises(RuntimeError, match="changed after it was validated"):
        _counted_quantum(tmp_path, monkeypatch, single, receipt, output_root)
