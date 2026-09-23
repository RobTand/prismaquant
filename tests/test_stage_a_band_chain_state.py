"""A band's run header comes from the run's sealed chain state (PQ #1122).

A dev-mode Stage A run binds the producer that is *running*, which may differ
from the implementation its prepared completion recorded (the preflight
records that drift rather than refusing it). The band tool used to rebuild the
bind identity from the prepared completion, so such a run got no band, and it
dropped the #1028 bf16 stamp, so its header was not the receipt's.

The runs below are the real fixture Stage A (``run_adjoint_capture_core``)
with bf16 reduced-precision reduction off, driven through
``band_from_request`` as the band action runs it: a sealed PB request whose
container spec carries the environment.
"""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.joint_adjoint_band import BandRefused, band_from_request
from prismaquant.joint_adjoint_checkpoints import (
    STAGE_A_SLICE_FIELDS,
    adjoint_slice_sha256,
    adjoint_space,
    load_stage_a_receipt_like,
    stage_a_run_header,
    stage_a_slice,
    write_band_receipt,
)
from prismaquant.matmul_arithmetic import BF16_REDUCTION_ENV, BF16_REDUCTION_FIELD
from prismaquant.stage_a_chain_resume import _seal, chain_state_path

from test_layer_major_boundary_capture import draw

PREPARED = "1" * 64   # the implementation the prepared completion recorded
RUNNING = "2" * 64    # the producer the dev-mode run actually bound
READ_MANIFEST = "d" * 64
FORMATS = {"model.layers.0.mlp.down_proj": "NVFP4", "model.layers.1.mlp.down_proj": "FP8"}
ROSTER = hashlib.sha256("".join(f"{name}\n" for name in sorted(FORMATS)).encode()).hexdigest()


def _write_json(path, document):
    path.write_text(json.dumps(document, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(path, *, run, plan=None, prepared=None, env=None, extra=()):
    """A sealed PB request of R13's shape: the container spec carries the env."""
    plan = plan or (run.plan, run.plan_sha256)
    prepared = prepared or (run.prepared, run.prepared_sha256)
    spec = {"env": {BF16_REDUCTION_ENV: "off"} if env is None else env}
    path.write_text(json.dumps({"action_key": "f" * 64, "params": {"command": [
        "python3", "-m", "tools.tessera_campaign_container", "--spec", json.dumps(spec), "--",
        "python3", "-m", "prismaquant.joint_cost_stage_a",
        "--plan", str(plan[0]), "--plan-sha256", plan[1],
        "--prepared", str(prepared[0]), "--prepared-sha256", prepared[1],
        "--read-manifest-sha256", READ_MANIFEST,
        "--output-root", str(run.root), "--resume", *extra]}}))
    return path


def _run(tmp_path, monkeypatch, name, *, prepared_implementation):
    """One fixture Stage A run with bf16 reduction off, plus its plan and preparation."""
    from prismaquant import joint_cost_stage_a as stage_a
    from test_joint_cost_quantum_runtime import _execution, _stage_a
    from test_streamed_cost_checkpoints import _model_identity

    base = tmp_path / name
    base.mkdir()
    root = base / "run"
    ids = draw()
    plan = base / "plan.json"
    plan_sha256 = _write_json(plan, {
        "output_root": str(root), "campaign_scope": None,
        "distributed_campaign": {"cotangent_checkpoint_stride": 1},
        "calibration_input": {"path": str(base / "calibration.pt"), "sha256": "e" * 64},
        "execution": {"n_calib_samples": ids.shape[0], "calib_seqlen": ids.shape[1],
                      "n_probes": 4, "seed_base": 7000, "probe_microbatch": 1}})
    prepared = base / "prepared.json"
    prepared_sha256 = _write_json(prepared, {
        "implementation_sha256": prepared_implementation,
        "source_model_identity": _model_identity("joint-source"),
        "formats_by_qname": FORMATS})
    runner, _ = _stage_a(root, monkeypatch)
    runner.context.settle_prefetched_layers = lambda *a, **kw: None
    with monkeypatch.context() as patch:
        # The entry point pins this from the container env (PQ #1028).
        patch.setattr(torch.backends.cuda.matmul,
                      "allow_bf16_reduced_precision_reduction", False)
        receipt = stage_a.run_adjoint_capture_core(
            runner, ids, execution=_execution(root), output_root=root, stride=1,
            source_model_identity=_model_identity("joint-source"),
            unit_roster_sha256=ROSTER, plan_sha256=plan_sha256,
            prepared_sha256=prepared_sha256, read_manifest_sha256=READ_MANIFEST,
            implementation_sha256=RUNNING)
    receipt = json.loads(json.dumps(receipt))
    # The entry point fills the stride source it resolved (joint_cost_stage_a.py).
    receipt["stride"]["source"] = "plan"
    return SimpleNamespace(root=root, base=base, receipt=receipt, plan=plan,
                           plan_sha256=plan_sha256, prepared=prepared,
                           prepared_sha256=prepared_sha256)


@pytest.fixture
def calibration(monkeypatch):
    """The fallback path loads the plan's calibration input; serve the fixture draw."""
    from prismaquant import calibration_data

    monkeypatch.setattr(calibration_data, "load_calibration_input",
                        lambda path, **kw: (draw(), None))


@pytest.fixture
def drifted(tmp_path, monkeypatch, calibration):
    """R13's shape: the running producer is not the prepared completion's."""
    return _run(tmp_path, monkeypatch, "drifted", prepared_implementation=PREPARED)


def _assert_band_is_the_receipt(run, request, tmp_path):
    receipt = run.receipt
    assert receipt["run_identity"][BF16_REDUCTION_FIELD] is False
    assert receipt["run_identity"]["implementation_sha256"] == RUNNING
    served = []
    for boundary in receipt["stride"]["boundaries"]:
        band = band_from_request(request, boundary=boundary)
        path = tmp_path / "bands" / run.base.name / f"band-{boundary:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        sealed = load_stage_a_receipt_like(path, write_band_receipt(path, band))
        assert canonical_json_bytes(stage_a_run_header(sealed), where="band header") == \
            canonical_json_bytes(stage_a_run_header(receipt), where="receipt header")
        for layer in band["band"]["layers"]:
            from_band = stage_a_slice(sealed, layer)
            from_receipt = stage_a_slice(receipt, layer)
            assert set(from_band) == set(STAGE_A_SLICE_FIELDS)
            assert canonical_json_bytes(from_band, where="band slice") == \
                canonical_json_bytes(from_receipt, where="receipt slice"), layer
            assert adjoint_slice_sha256(from_band) == adjoint_slice_sha256(from_receipt)
            served.append(layer)
    assert sorted(served) == [0, 1]
    return band


def test_a_drifted_dev_mode_run_bands_from_its_chain_state(drifted, tmp_path):
    """The #1122 fixture: refused on main, byte-equal to the receipt with the fix."""
    request = _request(tmp_path / "request.json", run=drifted)
    band = _assert_band_is_the_receipt(drifted, request, tmp_path)
    state = chain_state_path(adjoint_space(drifted.root))
    assert band["sources"]["chain_state"] == {
        "path": str(state), "sha256": hashlib.sha256(state.read_bytes()).hexdigest()}


def test_without_a_chain_state_the_band_stamps_bf16_from_the_request(
        tmp_path, monkeypatch, calibration):
    """The fallback path: the request's sealed environment carries the #1028 stamp."""
    run = _run(tmp_path, monkeypatch, "legacy", prepared_implementation=RUNNING)
    state = chain_state_path(adjoint_space(run.root))
    state.rename(state.with_name("chain-state.json.aside"))
    request = _request(tmp_path / "request.json", run=run)
    band = _assert_band_is_the_receipt(run, request, tmp_path)
    assert "chain_state" not in band["sources"]
    # The same request without the setting describes another arithmetic.
    unset = _request(tmp_path / "unset.json", run=run, env={})
    assert BF16_REDUCTION_FIELD not in band_from_request(unset, boundary=2)["run_identity"]
    malformed = _request(tmp_path / "malformed.json", run=run, env={BF16_REDUCTION_ENV: "on"})
    with pytest.raises(BandRefused, match=BF16_REDUCTION_ENV):
        band_from_request(malformed, boundary=2)


def _reseal(run, edit):
    path = chain_state_path(adjoint_space(run.root))
    document = json.loads(path.read_bytes())
    edit(document)
    path.write_text(json.dumps(_seal(document), sort_keys=True))


@pytest.mark.parametrize("field", ["plan_sha256", "prepared_sha256"])
def test_a_request_for_another_plan_or_preparation_refuses(drifted, tmp_path, field):
    source = drifted.plan if field == "plan_sha256" else drifted.prepared
    other = tmp_path / f"other-{source.name}"
    document = json.loads(source.read_text())
    document["unrelated"] = True
    digest = _write_json(other, document)
    request = _request(tmp_path / "request.json", run=drifted,
                       **{("plan" if field == "plan_sha256" else "prepared"): (other, digest)})
    with pytest.raises(BandRefused, match=field):
        band_from_request(request, boundary=2)


def test_a_request_for_another_read_manifest_refuses(drifted, tmp_path):
    request = _request(tmp_path / "request.json", run=drifted)
    document = json.loads(request.read_text())
    command = document["params"]["command"]
    command[command.index("--read-manifest-sha256") + 1] = "f" * 64
    request.write_text(json.dumps(document))
    with pytest.raises(BandRefused, match="read_manifest_sha256"):
        band_from_request(request, boundary=2)


def test_a_request_whose_arithmetic_or_regime_differs_refuses(drifted, tmp_path):
    unset = _request(tmp_path / "unset.json", run=drifted, env={})
    with pytest.raises(BandRefused, match=BF16_REDUCTION_FIELD):
        band_from_request(unset, boundary=2)
    batched = _request(tmp_path / "batched.json", run=drifted,
                       extra=("--chain-batch-size", "2"))
    with pytest.raises(BandRefused, match="chain_regime"):
        band_from_request(batched, boundary=2)


def test_a_chain_state_that_does_not_seal_itself_refuses(drifted, tmp_path):
    path = chain_state_path(adjoint_space(drifted.root))
    document = json.loads(path.read_bytes())
    document["run_identity"]["seed_base"] = 1
    path.write_text(json.dumps(document, sort_keys=True))
    with pytest.raises(BandRefused, match="does not seal its own content"):
        band_from_request(_request(tmp_path / "request.json", run=drifted), boundary=2)


def test_a_resealed_chain_state_for_another_run_refuses(drifted, tmp_path):
    """A self-sealed state whose bind identity is not the generation's run."""
    request = _request(tmp_path / "request.json", run=drifted)

    def other_seed(document):
        document["bind_identity"]["seed_base"] = 1
        document["run_identity"]["seed_base"] = 1

    _reseal(drifted, other_seed)
    with pytest.raises(BandRefused, match="does not hash"):
        band_from_request(request, boundary=2)


def test_a_chain_state_header_that_disagrees_with_its_bind_identity_refuses(drifted, tmp_path):
    request = _request(tmp_path / "request.json", run=drifted)

    def other_implementation(document):
        document["run_identity"]["implementation_sha256"] = PREPARED

    _reseal(drifted, other_implementation)
    with pytest.raises(BandRefused, match="implementation_sha256"):
        band_from_request(request, boundary=2)
