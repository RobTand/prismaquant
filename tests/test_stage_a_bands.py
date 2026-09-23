"""Stage B consumes Stage A by checkpoint band (RobTand/prismaquant#993).

A quantum reads only its slice of Stage A. A band sealed from one durable
checkpoint must give every layer it serves the same slice, byte for byte, as
the complete receipt the run writes hours later. The runs below are the real
fixture Stage A (``run_adjoint_capture_core``), one resumed from a forward
recovery capsule (R12's shape) and one uninterrupted.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.joint_adjoint_band import (
    BandRefused,
    band_summary,
    build_band_receipt,
    main as band_main,
    stage_a_argv,
)
from prismaquant.joint_adjoint_checkpoints import (
    STAGE_A_SLICE_FIELDS,
    AdjointSliceRefused,
    adjoint_slice_sha256,
    band_set,
    load_stage_a_receipt_like,
    missing_band_boundaries,
    stage_a_run_header,
    stage_a_slice,
    write_band_receipt,
)

from test_joint_forward_resume import _bound, _chain_sdk, _owner_spool
from test_layer_major_boundary_capture import draw

CAMPAIGN = {"plan_sha256": "b" * 64, "prepared_sha256": "c" * 64,
            "read_manifest_sha256": "d" * 64, "unit_roster_sha256": "a" * 64,
            "campaign_scope": None}
DIGESTS = {"plan_sha256": "b" * 64, "prepared_sha256": "c" * 64,
           "read_manifest_sha256": "d" * 64, "stride_value": 1, "stride_source": None}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """R9 contained after boundary 1, R10 resumed from its capsule, a baseline."""
    from prismaquant import joint_cost_stage_a as stage_a
    from prismaquant import joint_forward_resume as recovery_mod
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from test_joint_cost_quantum_runtime import _execution, _stage_a
    from test_streamed_cost_checkpoints import _model_identity

    monkeypatch.setattr(recovery_mod, "_sdk", _chain_sdk)
    identities = []
    bind = StreamedBoundaryArtifacts.bind

    def recorded(self, identity, **kw):
        identities.append(copy.deepcopy(identity))
        return bind(self, identity, **kw)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "bind", recorded)

    def capture(path, implementation, bound=None, stop_after=None):
        retained = {}
        write = StreamedBoundaryArtifacts.write

        def interrupted(self, tensor, **kw):
            ref = write(self, tensor, **kw)
            if kw.get("probe_index") is None:
                retained[(kw["boundary_index"], kw["batch_index"])] = ref
                if kw["boundary_index"] == stop_after and kw["batch_index"] == len(draw()) - 1:
                    raise InterruptedError("contained after a whole boundary")
            return ref

        runner, _ = _stage_a(path, monkeypatch)
        runner.context.settle_prefetched_layers = lambda *a, **kw: None
        with monkeypatch.context() as patch:
            patch.setattr(StreamedBoundaryArtifacts, "write", interrupted)

            def run():
                return stage_a.run_adjoint_capture_core(
                    runner, draw(), execution=_execution(path), output_root=path, stride=1,
                    source_model_identity=_model_identity("joint-source"),
                    unit_roster_sha256="a" * 64, plan_sha256="b" * 64,
                    prepared_sha256="c" * 64, read_manifest_sha256="d" * 64,
                    implementation_sha256=implementation, forward_recovery=bound)
            if stop_after is None:
                return run(), retained
            with pytest.raises(InterruptedError):
                run()
        return None, retained

    _, first = capture(tmp_path / "r9", "1" * 64, stop_after=1)
    spool, instance, template = _owner_spool(tmp_path, "e" * 64, first)
    ref = first[0, 0]
    capsule = tmp_path / "r9-to-r10.json"
    recovery_mod.build_forward_recovery(specification={
        "schema": recovery_mod.SCHEMA, "queue_root": str(tmp_path / "queue"),
        "instance": instance, "template": template,
        "session": json.loads(ref.metadata_json)["identity"]["session"],
        "original_bind_identity": identities[-1], "campaign_identity": CAMPAIGN,
        "implementation_compatibility": {"original": "1" * 64, "recovery": "2" * 64,
                                         "scope": "forward-identical-memory-only"},
        "n_batches": len(draw()), "entry_shape": list(ref.shape), "entry_dtype": ref.dtype},
        spool_directory=spool, output=capsule, frontier=1)
    bound = _bound(capsule)
    resumed, _ = capture(tmp_path / "r10", "2" * 64, bound=bound)
    baseline, _ = capture(tmp_path / "baseline", "2" * 64)
    # The receipt a quantum reads is the sealed file, so compare JSON forms.
    return SimpleNamespace(
        root=tmp_path, capsule=bound, baseline_identity=identities[-1],
        resumed=json.loads(json.dumps(resumed)), baseline=json.loads(json.dumps(baseline)))


def _band(runs, name, boundary, **overrides):
    sources = ({"forward_recovery": runs.capsule} if name == "r10" else
               {"bind_identity": runs.baseline_identity, "unit_roster_sha256": "a" * 64})
    return build_band_receipt(output_root=runs.root / name, boundary=boundary,
                              **{**DIGESTS, **sources, **overrides})


def test_band_slices_equal_the_complete_receipt_slices(runs, tmp_path):
    """Every band's slice of every layer it serves is the receipt's, byte for byte.

    Compared: the canonical bytes of the whole slice -- ``run_identity``,
    ``stride``, ``boundary_storage`` (``session``, ``policy``, ``directory``,
    ``forward_recovery``), the checkpoint record and every boundary entry
    record -- and its digest, for the recovered run (entries from the capsule
    chain) and the uninterrupted one (entries from the generation's files).
    """
    for name, receipt in (("r10", runs.resumed), ("baseline", runs.baseline)):
        assert receipt["stride"]["boundaries"] == [2, 1]
        served = []
        for boundary in receipt["stride"]["boundaries"]:
            band = _band(runs, name, boundary)
            path = tmp_path / "bands" / name / f"band-{boundary:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            file_sha256 = write_band_receipt(path, band)
            assert write_band_receipt(path, band) == file_sha256, "re-sealing is a no-op"
            sealed = load_stage_a_receipt_like(path, file_sha256)
            assert stage_a_run_header(sealed) == stage_a_run_header(receipt)
            for layer in band["band"]["layers"]:
                from_band = stage_a_slice(sealed, layer)
                from_receipt = stage_a_slice(receipt, layer)
                assert set(from_band) == set(STAGE_A_SLICE_FIELDS)
                assert canonical_json_bytes(from_band, where="band slice") == \
                    canonical_json_bytes(from_receipt, where="receipt slice"), (name, layer)
                assert adjoint_slice_sha256(from_band) == adjoint_slice_sha256(from_receipt)
                served.append(layer)
            summary = band_summary(sealed, path=path, file_sha256=file_sha256)
            assert summary["slices"] == {
                str(layer): adjoint_slice_sha256(stage_a_slice(receipt, layer))
                for layer in band["band"]["layers"]}
        assert sorted(served) == [0, 1], "the bands serve every layer exactly once"
    recovery = runs.resumed["boundary_storage"]["forward_recovery"]
    assert _band(runs, "r10", 2)["boundary_storage"]["forward_recovery"] == recovery


def test_band_refusals(runs):
    band_tail = _band(runs, "r10", 2)
    # A band for the wrong boundary: layer 0 reads checkpoint 1, not 2.
    with pytest.raises(AdjointSliceRefused, match="cannot serve layer 0"):
        stage_a_slice(band_tail, 0)
    # A band with a different run header: another generation's band.
    other = _band(runs, "baseline", 1)
    with pytest.raises(AdjointSliceRefused, match="another Stage A run header"):
        band_set([band_tail, other])
    # A band set with a missing stride checkpoint.
    assert missing_band_boundaries(band_set([band_tail])) == [1]
    assert missing_band_boundaries(band_set([band_tail, _band(runs, "r10", 1)])) == []
    # A header that does not answer for the generation that sealed the checkpoint.
    with pytest.raises(BandRefused, match="does not hash"):
        _band(runs, "baseline", 2,
              bind_identity={**runs.baseline_identity, "seed_base": 1})
    # A boundary that is no stride checkpoint of the run.
    with pytest.raises((BandRefused, OSError)):
        _band(runs, "r10", 3)
    # An edited checkpoint manifest no longer seals itself.
    from prismaquant.joint_adjoint_checkpoints import adjoint_space, checkpoint_directory
    manifest = checkpoint_directory(adjoint_space(runs.root / "baseline"), 1) / "checkpoint.json"
    record = json.loads(manifest.read_text())
    record["activation_entries"] = record["activation_entries"][1:]
    manifest.write_text(json.dumps(record))
    with pytest.raises(BandRefused, match="does not seal its own manifest"):
        _band(runs, "baseline", 1)


def test_the_tool_never_writes_under_the_run_output_root(tmp_path):
    root = tmp_path / "run"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"action_key": "f" * 64, "params": {"command": [
        "python3", "-m", "tools.tessera_campaign_container", "--spec", "{}", "--",
        "python3", "-m", "prismaquant.joint_adjoint_capture",
        "--plan", str(tmp_path / "plan.json"), "--plan-sha256", "b" * 64,
        "--prepared", str(tmp_path / "prepared.json"), "--prepared-sha256", "c" * 64,
        "--output-root", str(root), "--resume"]}}))
    assert stage_a_argv(json.loads(request.read_text())["params"]["command"])[
        "--output-root"] == str(root)
    inside = root / "layer-quanta" / "adjoint" / "band-045.json"
    assert band_main(["--request", str(request), "--boundary", "45",
                      "--output", str(inside)]) == 3
    assert not inside.exists()
