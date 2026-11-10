"""Own-plane read windows and capture receipts retain scientific coverage."""
import json
from pathlib import Path
import pytest
from produced_native_fixtures import owner
from test_stage_a_produced_boundary_chain import write_group, read_group, GROUP_SIZE


def test_a_retained_boundary_group_survives_four_passes_without_rematerializing(tmp_path, monkeypatch):
    storage, pub = owner(tmp_path, n_batches=GROUP_SIZE, group_size=GROUP_SIZE, tensor_bytes=32)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        refs = write_group(storage)
        spool = storage._local_output_spool
        spool.join_copies()
        group = next(iter(storage._produced_groups.values()))
        paths = [Path(row[0].path) for row in spool._groups[group['batch_id']]['references']]
        original = [path.stat().st_ino for path in paths]
        with storage.retain_produced_boundary(0):
            for ref in refs:
                storage.release_produced_group(ref)
            for _ in range(4):
                read_group(storage, refs)
                assert [path.stat().st_ino for path in paths] == original
        assert all(not path.exists() for path in paths)
        assert len(pub._groups) == 1
        read_group(storage, refs)
        assert len(pub._groups) == 1
        assert storage._produced_retained_boundary is None


def test_an_unbound_owner_reports_nothing(tmp_path):
    from test_streamed_boundary_artifacts import _policy
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
    storage = StreamedBoundaryArtifacts(_policy(tmp_path))
    assert storage.produced_output_report() is None
def test_the_capture_declares_its_last_roll_unread(tmp_path, monkeypatch):
    """The line that keeps the last roll off the stage tier, pinned."""

    from test_layer_major_boundary_capture import fixture, draw
    from test_streamed_boundary_artifacts import _policy
    from test_streamed_cost_checkpoints import _model_identity
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core

    seen = []
    real = StreamedBoundaryArtifacts.write

    def recording(self, tensor, **kwargs):
        if kwargs.get("probe_index") is not None:
            seen.append((kwargs["boundary_index"],
                         kwargs.get("read_back", True)))
        return real(self, tensor, **kwargs)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "write", recording)
    _, _, runner, _ = fixture()
    runner.context.settle_prefetch_layers = lambda layers: None
    receipt = run_adjoint_capture_core(
        runner, draw(), execution={
            "n_probes": 1, "seed_base": 7000, "probe_microbatch": 0,
            "boundary_storage": {
                **_policy(tmp_path / "b", cap=1 << 24, aux=1 << 22,
                          disk=1 << 26),
                "schema": "prismaquant.aura.boundary_storage.v2",
                "capture_order": "layer_major"}},
        output_root=tmp_path / "out", stride=8,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256="a" * 64, plan_sha256="d" * 64,
        prepared_sha256="e" * 64, read_manifest_sha256="f" * 64,
        implementation_sha256="b" * 64)
    assert "produced_output" not in receipt["telemetry"], (
        "an unbound owner's receipt is unchanged")
    boundaries = sorted({boundary for boundary, _ in seen})
    assert boundaries[0] == 0 and len(boundaries) > 1
    assert all(read_back is (boundary > 0) for boundary, read_back in seen), seen


def test_the_closing_facts_reach_a_receipt_that_still_seals(
        tmp_path, monkeypatch):
    """The real capture carries actual copy/release facts into its sealed receipt."""
    from test_stage_a_retirement_1073 import scientific_case
    from prismaquant.joint_adjoint_checkpoints import adjoint_receipt_path, write_adjoint_receipt
    case = scientific_case(tmp_path, monkeypatch)
    receipt = json.loads(adjoint_receipt_path(case.space).read_text())
    block = receipt["telemetry"]["produced_output"]
    assert block['schema'] == 'prismaquant.boundary_produced_output.v2'
    assert block['declaration'] == case.pub.binding
    spool = block['local_spool']
    assert spool['groups'] > 0 and spool['durable_groups'] == spool['groups']
    assert spool['pending_groups'] == spool['retained_ceiling_bytes'] == 0
    assert all(row['durable'] and row['released'] and row['reading'] == 0
               and row['publication_error'] is None for row in spool['exports'])
    assert block['deferred_unlinks'] == {} and block['release_errors'] == []
    space = tmp_path / "sealed"
    space.mkdir()
    assert write_adjoint_receipt(space, receipt)


@pytest.mark.parametrize("report, expected", [
    (lambda: 1 / 0, "ZeroDivisionError"),
    (lambda: {"telemetry": {"produced_group_stage_wait_s": float("nan")}},
     "ValueError"),
    (lambda: {"release_debt": {("boundary", 3): "a tuple is not a key"}},
     "TypeError"),
])
def test_a_report_that_cannot_be_carried_never_fails_a_finished_capture(
        report, expected):
    from prismaquant.joint_cost_stage_a import _produced_output_block

    class Owner:
        def produced_output_report(self):
            return report()

    block = _produced_output_block(Owner())
    assert expected in block["produced_output"]["report_error"]
    json.dumps(block, sort_keys=True, allow_nan=False)
