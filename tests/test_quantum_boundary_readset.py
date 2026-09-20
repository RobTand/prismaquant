"""Producer tests: quantum boundary/checkpoint bulk readset binding.

PQ #848. Fixture-scale only: a synthetic completed adjoint receipt plus the
real recovery record metadata (4 KiB JSON, no payloads). Proves the derived
v2 manifest binds the exact path/length/digest triple in true reader order
(checkpoint plane, then chain windows descending, then the quantum's own
boundary) with byte-exact phase accounting, deterministic sealing, and
fail-closed refusals. Full-chain staging tests belong to the integration
worker, not this file.
"""

import hashlib
import json
import os

import pytest

from prismaquant.joint_layer_quanta import (
    ADJOINT_CAPTURE_SCHEMA,
    MANIFEST_SCHEMA_V2,
    bind_adjoint_receipt,
    build_quantum_boundary_readset,
    quantum_boundary_read_phase_names,
    seal_manifest_bytes,
)

RECOVERY_RECORDS = (
    "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913"
    "/allocation/joint-panel/stage-a-recovery-20260920/records")
STRIDED = [8, 16, 24, 32, 40, 45]

# Real record metadata lives on the fleet's shared mount; skip -- never
# fail -- where it is absent (GitHub CI has no /mnt/shared, the convention
# tests/test_joint_layer_quanta.py established).
needs_recovery = pytest.mark.skipif(
    not os.path.isdir(RECOVERY_RECORDS),
    reason="the recovery records are absent (GitHub CI has no /mnt/shared)")


def _exact(name, nbytes, seed):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return {"name": name,
            "path": f"/mnt/shared/adjoint/exact-boundaries/gen/{name}.pt",
            "sha256": digest, "tensor_bytes": nbytes, "file_bytes": nbytes + 64,
            "shape": [1, 8], "dtype": "torch.float32",
            "metadata": {"identity": {"session": "gen", "slot": name}}}


def _receipt(campaign, n_batches=5, prefetch_batches=2):
    checkpoints = []
    for boundary in STRIDED:
        full = boundary == 8
        checkpoints.append({
            "boundary": boundary,
            "session": {"generation": "gen", "kind": "adjoint_checkpoint"},
            "activation_entries": [
                _exact(f"cotangent-{p}-{b}-at-{boundary}", 2048,
                       f"ckpt{boundary}")
                for p in range(2) for b in range(n_batches)] if full else [],
            "shared_state_entries": [
                _exact(f"shared-pass-{b}", 128, f"ckpt{boundary}")
                for b in range(n_batches)] if full else [],
        })
    return {
        "schema": ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"]},
        "boundary_storage": {
            "session": {"generation": "gen", "run_identity_sha256": "0" * 64},
            "policy": {"prefetch_batches": prefetch_batches},
        },
        "boundary_entries": {
            str(boundary): [
                _exact(f"boundary-{b}-{boundary}-at-{boundary}", 1024,
                       f"b{boundary}")
                for b in range(n_batches)]
            for boundary in (3, 4, 5, 6, 7, 44)},
        "checkpoints": checkpoints,
    }


def _record(layer, chain, checkpoint):
    campaign = json.load(
        open(f"{RECOVERY_RECORDS}/layer-{layer:03d}.json"))["campaign"]
    return {"quantum_id": f"layer-{layer:03d}", "layer": layer,
            "adjoint": {"checkpoint_boundary": checkpoint,
                        "chain_layers": list(chain),
                        "receipt_sha256": None},
            "campaign": campaign}


def test_phase_names_frozen_reader_order():
    names = quantum_boundary_read_phase_names([7, 6, 5, 4], 3, batch_windows=3)
    assert names[0] == "checkpoint"
    assert names[1:4] == ("chain-007-w00", "chain-007-w01", "chain-007-w02")
    assert names[-3:] == ("chain-003-w00", "chain-003-w01", "chain-003-w02")
    assert len(names) == 1 + 5 * 3


def test_tail_quantum_chains_nothing_but_own_boundary():
    names = quantum_boundary_read_phase_names([], 44, batch_windows=2)
    assert names == ("checkpoint", "chain-044-w00", "chain-044-w01")


@needs_recovery
def test_build_real_record_metadata_binds_exact_triple():
    record = _record(3, [7, 6, 5, 4], 8)
    manifest = build_quantum_boundary_readset(
        record, _receipt(record["campaign"]), strided_boundaries=STRIDED)
    assert manifest["schema"] == MANIFEST_SCHEMA_V2
    assert "phases" not in manifest["annotations"]
    assert manifest["annotations"]["quantum_id"] == "layer-003"
    assert manifest["annotations"]["receipt_sha256"] == bind_adjoint_receipt(
        _receipt(record["campaign"]), plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"], checkpoints=STRIDED)
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    assert names == list(quantum_boundary_read_phase_names(
        [7, 6, 5, 4], 3, batch_windows=3))
    # Every entry referenced exactly once; accounting is exact.
    seen = [i for p in manifest["read_plan"]["phases"]
            for i in p["entry_indices"]]
    assert sorted(seen) == list(range(len(manifest["entries"])))
    assert manifest["read_plan"]["read_bytes"] == sum(
        p["bytes"] for p in manifest["read_plan"]["phases"])
    assert manifest["total_bytes"] == sum(
        e["bytes"] for e in manifest["entries"])
    assert manifest["entry_count"] == len(manifest["entries"]) == (
        2 * 5 + 5) + 5 * 5
    for entry in manifest["entries"]:
        assert set(entry) == {"path", "offset", "bytes", "sha256"}
        assert entry["offset"] == 0 and entry["bytes"] > 0


@needs_recovery
def test_build_tail_quantum_empty_chain():
    record = _record(44, [], 45)
    receipt = _receipt(record["campaign"])
    receipt["checkpoints"] = [
        dict(c, activation_entries=c["activation_entries"],
             shared_state_entries=c["shared_state_entries"])
        if c["boundary"] == 45 else c for c in receipt["checkpoints"]]
    full = [_exact(f"cotangent-{p}-{b}-at-45", 2048, "ckpt45")
            for p in range(2) for b in range(5)]
    shared = [_exact(f"shared-pass-{b}", 128, "ckpt45") for b in range(5)]
    for c in receipt["checkpoints"]:
        if c["boundary"] == 45:
            c["activation_entries"] = full
            c["shared_state_entries"] = shared
    manifest = build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED)
    names = [p["name"] for p in manifest["read_plan"]["phases"]]
    assert names == list(quantum_boundary_read_phase_names(
        [], 44, batch_windows=3))


@needs_recovery
def test_seal_deterministic_wire():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    first = seal_manifest_bytes(build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED))
    second = seal_manifest_bytes(build_quantum_boundary_readset(
        record, receipt, strided_boundaries=STRIDED))
    assert first == second
    assert hashlib.sha256(first).hexdigest() != \
        manifest_annotations_digest(record, receipt)


def manifest_annotations_digest(record, receipt):
    return bind_adjoint_receipt(
        receipt, plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"], checkpoints=STRIDED)


@needs_recovery
def test_refuses_chain_outside_stride_owner():
    record = _record(3, [7, 6, 5, 3], 8)
    with pytest.raises(ValueError, match="sealed stride chain"):
        build_quantum_boundary_readset(
            record, _receipt(record["campaign"]), strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_missing_checkpoint_boundary():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["checkpoints"] = [
        c for c in receipt["checkpoints"] if c["boundary"] != 8]
    with pytest.raises(ValueError, match="checkpoints differ"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_missing_needed_boundary():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    del receipt["boundary_entries"]["5"]
    with pytest.raises(ValueError, match="no boundary 5 entries"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_malformed_exact_record():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["boundary_entries"]["4"][0]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="no entry digest"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_duplicate_staged_path():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["boundary_entries"]["4"][0]["path"] = \
        receipt["boundary_entries"]["6"][0]["path"]
    with pytest.raises(ValueError, match="duplicate staged path"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_mixed_batch_counts():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["boundary_entries"]["4"].pop()
    with pytest.raises(ValueError, match="different batch counts"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_foreign_campaign():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["run_identity"]["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another plan_sha256"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_refuses_missing_prefetch_window():
    record = _record(3, [7, 6, 5, 4], 8)
    receipt = _receipt(record["campaign"])
    receipt["boundary_storage"]["policy"] = {}
    with pytest.raises(ValueError, match="no prefetch batch window"):
        build_quantum_boundary_readset(
            record, receipt, strided_boundaries=STRIDED)


@needs_recovery
def test_old_records_slices_parent_untouched():
    before = {layer: open(f"{RECOVERY_RECORDS}/layer-{layer:03d}.json").read()
              for layer in (0, 3, 44)}
    record = _record(3, [7, 6, 5, 4], 8)
    build_quantum_boundary_readset(
        record, _receipt(record["campaign"]), strided_boundaries=STRIDED)
    after = {layer: open(f"{RECOVERY_RECORDS}/layer-{layer:03d}.json").read()
             for layer in (0, 3, 44)}
    assert before == after
