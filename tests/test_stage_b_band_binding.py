"""Stage B binds each quantum's Stage A slice, so a checkpoint band serves it.

RobTand/prismaquant#993. A layer quantum reads one checkpoint and the forward
boundary entries of its own chain. Those are sealed hours before Stage A
writes its receipt, one band per stride checkpoint. The producer must accept
a band, emit records only for the layers that band serves, and emit exactly
the records the completed receipt would.

The Stage A documents here are pure data: a completed receipt whose
checkpoint records seal their own manifests, and the bands it implies.
``tests/test_stage_a_bands.py`` proves the band tool reconstructs real bands
byte for byte; this file proves what the consumers do with them. It imports
only names that exist before the change, so on a checkout without slice
binding it fails on the refusal itself, not on an import.
"""
from __future__ import annotations

import copy
import hashlib

import pytest

from prismaquant import joint_layer_quanta as jl
from prismaquant.cost_stage_checkpoint import canonical_json_sha256

from test_joint_layer_quanta import _synthetic_inputs

RECEIPT_SCHEMA = "prismaquant.joint_adjoint_capture.v1"
BAND_SCHEMA = "prismaquant.joint_adjoint_capture.band.v1"
CHECKPOINT_SCHEMA = "prismaquant.joint_adjoint_checkpoint.v1"
PARENT_SHA256 = "2" * 64


def _row(path: str) -> dict:
    name = path.rsplit("/", 1)[-1].removesuffix(".pt")
    return {"name": name, "path": path,
            "sha256": hashlib.sha256(path.encode()).hexdigest(), "file_bytes": 4096}


def synthetic_receipt(*, plan_sha256: str, prepared_sha256: str, scope: dict,
                      num_layers: int, stride: int, n_batches: int = 2,
                      prefetch_batches: int = 1,
                      root: str = "/mnt/shared/fixture/out",
                      generation: str = "fixture-generation",
                      run_identity_sha256: str = "ab" * 32,
                      identity: dict | None = None) -> dict:
    """A completed Stage A receipt with every field a slice reads, as data.

    Checkpoints seal their own manifests the way ``write_adjoint_checkpoint``
    does; boundary entries are named the way the capture names them.
    ``identity`` overrides run-identity fields (calibration, roster, probes).
    """
    space = f"{root}/layer-quanta/adjoint"
    directory = f"{space}/exact-boundaries"
    boundaries = [num_layers, *[mark for mark in range(stride, num_layers, stride)][::-1]]
    session = {"generation": generation, "run_identity_sha256": run_identity_sha256}
    checkpoints = []
    for boundary in boundaries:
        entries = f"{space}/checkpoints/boundary-{boundary:03d}/entries"
        record = {
            "schema": CHECKPOINT_SCHEMA, "boundary": boundary,
            "session": {"generation": generation, "kind": "adjoint_checkpoint",
                        "run_identity_sha256": run_identity_sha256},
            "activation_entries": [_row(f"{entries}/cotangent-{probe}-{batch}.pt")
                                   for probe in range(2) for batch in range(n_batches)],
            "shared_state_entries": [_row(f"{entries}/shared-{batch}.pt")
                                     for batch in range(n_batches)],
        }
        record["cotangent_sha256"] = canonical_json_sha256(
            {key: record[key] for key in ("schema", "boundary", "session",
                                          "activation_entries", "shared_state_entries")},
            where="adjoint checkpoint")
        checkpoints.append(record)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "run_identity": {"plan_sha256": plan_sha256, "prepared_sha256": prepared_sha256,
                         "read_manifest_sha256": PARENT_SHA256,
                         "implementation_sha256": "9" * 64,
                         "unit_roster_sha256": "8" * 64, "campaign_scope": scope,
                         "n_probes": 2, "seed_base": 7, "calibration_shape": [n_batches, 8],
                         "calibration_sha256": "7" * 64, **(identity or {})},
        "stride": {"value": stride, "source": None, "boundaries": boundaries,
                   "max_chain_layers": stride - 1},
        "boundary_storage": {"session": session,
                             "policy": {"prefetch_batches": prefetch_batches},
                             "directory": directory},
        "boundary_entries": {
            str(k): [_row(f"{directory}/{generation}/entries/boundary-{batch}-{k}-at-{k}.pt")
                     for batch in range(n_batches)]
            for k in range(num_layers)},
        "checkpoints": checkpoints,
        # Receipt-only fields: nothing of these may reach a quantum.
        "retention": {"retained": True},
        "telemetry": {"wall_s": 1.0},
    }


def band_from_receipt(receipt: dict, boundary: int) -> dict:
    """The band of ``boundary`` a completed receipt implies: same field names."""
    marks = sorted(receipt["stride"]["boundaries"])
    lower = max([mark for mark in marks if mark < boundary], default=0)
    layers = list(range(boundary - 1, lower - 1, -1))
    return copy.deepcopy({
        "schema": BAND_SCHEMA, "status": "band",
        "band": {"boundary": boundary, "layers": layers},
        "run_identity": receipt["run_identity"], "stride": receipt["stride"],
        "boundary_storage": receipt["boundary_storage"],
        "boundary_entries": {str(k): receipt["boundary_entries"][str(k)] for k in layers},
        "checkpoints": [record for record in receipt["checkpoints"]
                        if record["boundary"] == boundary],
    })


def _campaign():
    plan, prepared, parent = _synthetic_inputs()
    receipt = synthetic_receipt(
        plan_sha256=parent["annotations"]["plan_sha256"], prepared_sha256="1" * 64,
        scope=parent["annotations"]["campaign_scope"], num_layers=2, stride=1)
    return plan, prepared, parent, receipt


def _quanta(plan, prepared, parent, **proof):
    return jl.layer_quanta(plan, prepared, parent, parent_manifest_sha256=PARENT_SHA256,
                           stride=1, **proof)


def test_a_band_binds_the_records_of_its_own_layers():
    """Red before #993: ``bind_adjoint_receipt`` refused any band's schema."""
    plan, prepared, parent, receipt = _campaign()
    tail = band_from_receipt(receipt, 2)
    built = _quanta(plan, prepared, parent, adjoint_receipt=tail)
    assert [record["layer"] for record in built["records"]] == [1]
    complete = _quanta(plan, prepared, parent, adjoint_receipt=receipt)
    assert [record["layer"] for record in complete["records"]] == [0, 1]
    # Same Stage A output, same record: the band-derived layer-1 record is the
    # receipt-derived one, byte for byte, and its identity names the slice.
    assert jl.canonical_bytes(built["records"][0]) == jl.canonical_bytes(complete["records"][1])
    assert jl.canonical_bytes(built["adjoint_slices"]) == jl.canonical_bytes(
        {"layer-001": complete["adjoint_slices"]["layer-001"]})
    adjoint = built["records"][0]["adjoint"]
    assert "receipt_sha256" not in adjoint
    assert adjoint["slice_sha256"] == canonical_json_sha256(
        built["adjoint_slices"]["layer-001"], where="slice")
    # The receipt-only fields never enter a slice.
    for adjoint_slice in complete["adjoint_slices"].values():
        assert set(adjoint_slice) == {"run_identity", "stride", "boundary_storage",
                                      "checkpoint", "boundary_entries"}


def test_more_bands_add_records_and_change_none():
    plan, prepared, parent, receipt = _campaign()
    first = _quanta(plan, prepared, parent, adjoint_receipts=[band_from_receipt(receipt, 2)])
    both = _quanta(plan, prepared, parent, adjoint_receipts=[
        band_from_receipt(receipt, 1), band_from_receipt(receipt, 2)])
    complete = _quanta(plan, prepared, parent, adjoint_receipt=receipt)
    assert [record["layer"] for record in both["records"]] == [0, 1]
    assert jl.canonical_bytes(both["records"][1]) == jl.canonical_bytes(first["records"][0])
    assert jl.canonical_bytes(both["records"]) == jl.canonical_bytes(complete["records"])
    # The coverage proof still tiles every parent layer.
    assert first["coverage"]["coverage_sha256"] == complete["coverage"]["coverage_sha256"]


def test_band_refusals_at_the_producer():
    plan, prepared, parent, receipt = _campaign()
    tail = band_from_receipt(receipt, 2)
    # A band whose checkpoint is not the one its layers read.
    wrong = copy.deepcopy(tail)
    wrong["band"] = {"boundary": 1, "layers": [0]}
    with pytest.raises(ValueError):
        _quanta(plan, prepared, parent, adjoint_receipt=wrong)
    # Bands of two Stage A runs: another generation sealed the lower band.
    other = synthetic_receipt(
        plan_sha256=parent["annotations"]["plan_sha256"], prepared_sha256="1" * 64,
        scope=parent["annotations"]["campaign_scope"], num_layers=2, stride=1,
        generation="another-generation")
    with pytest.raises(ValueError, match="run header|mixed"):
        _quanta(plan, prepared, parent,
                adjoint_receipts=[tail, band_from_receipt(other, 1)])
    # A band that answers for another campaign plan.
    foreign = copy.deepcopy(tail)
    foreign["run_identity"]["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="plan_sha256"):
        _quanta(plan, prepared, parent, adjoint_receipt=foreign)
    # A band whose checkpoint record no longer seals its manifest.
    edited = copy.deepcopy(tail)
    edited["checkpoints"][0]["shared_state_entries"] = []
    with pytest.raises(ValueError):
        _quanta(plan, prepared, parent, adjoint_receipt=edited)
    # A record still binding a whole receipt is refused by the campaign check.
    record = copy.deepcopy(_quanta(plan, prepared, parent, adjoint_receipt=tail)["records"][0])
    record["adjoint"] = {**{k: v for k, v in record["adjoint"].items()
                            if k not in ("slice_sha256", "slice_path")},
                         "receipt_sha256": "a" * 64}
    record["identity_sha256"] = jl.canonical_sha256(
        {k: v for k, v in record.items() if k != "identity_sha256"})
    with pytest.raises(ValueError, match="whole stage-A receipt"):
        jl.check_quantum_for_campaign(record, record["campaign"])
