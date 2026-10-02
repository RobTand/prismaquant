"""CPU selector/plumbing controls; these do not qualify original CUDA loads."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_band import BandRefused, refuse_seed_space, stage_a_argv
from prismaquant.joint_adjoint_checkpoints import adjoint_receipt_path, adjoint_space
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.stage_a_selected_row_diagnostic import SPEC_SCHEMA

from test_joint_cost_quantum_runtime import _boundary_policy, _execution
from test_layer_major_boundary_capture import draw
from test_stage_a_chain_resume import CAP, _offline_tier_policy, _run  # noqa: F401


@pytest.fixture(autouse=True)
def _diagnostic_dev_mode(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")


def _spec(ids, execution, *, row=3):
    return {
        "schema": SPEC_SCHEMA,
        "calibration_shape": list(ids.shape),
        "calibration_dtype": str(ids.dtype),
        "calibration_tensor_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "selected_global_row": row,
        "probe_seed": execution["seed_base"],
        "global_token_count": ids.numel(),
        "vocab_size": 23,
        "through": 2,
    }


def _execution_one(root):
    execution = _execution(root)
    execution.update(n_probes=1, probe_microbatch=1)
    execution["boundary_storage"] = _boundary_policy(root / "boundaries")
    execution["boundary_storage"]["max_resident_bytes"] = CAP
    return execution


@pytest.mark.parametrize("model", ["dense", "shared"])
@pytest.mark.parametrize("row", [0, 3])
def test_fresh_selected_tail_matches_full_draw_row(tmp_path, monkeypatch, model, row):
    ids = draw()
    execution = _execution_one(tmp_path)
    full, selected = [], []
    _run(tmp_path / "full", monkeypatch, model=model, calib=ids,
         execution=execution, writes=full)
    root = tmp_path / "selected"
    spec = _spec(ids, execution, row=row)
    spec["vocab_size"] = 11 if model == "shared" else 23
    receipt = _run(root, monkeypatch, model=model, calib=ids,
                   execution=execution, writes=selected,
                   selected_row_diagnostic=spec)
    expected = {boundary: digest for (boundary, probe, batch), digest in full
                if probe == 0 and batch == row and boundary >= 2}
    observed = {boundary: digest for (boundary, probe, batch), digest in selected}
    assert observed == expected
    assert all(probe == 0 and batch == 0 for (_, probe, batch), _ in selected)
    assert receipt["bandable"] is False
    assert receipt["diagnostic"]["selected_global_row"] == row
    assert receipt["diagnostic"]["global_token_count"] == ids.numel()
    assert receipt["run_identity"]["calibration_shape"] == list(ids.shape)
    assert receipt["tail_probe"]["fresh"] is True
    assert receipt["tail_probe"]["global_row_offset"] == row
    assert receipt["tail_probe"]["teacher_logits_sha256"]
    assert not chain_state_path(adjoint_space(root)).exists()
    assert not adjoint_receipt_path(adjoint_space(root)).exists()
    with pytest.raises(BandRefused, match="diagnostic"):
        refuse_seed_space(adjoint_space(root))


@pytest.mark.parametrize("broken", ["row", "seed", "N", "vocab", "shape", "tensor", "dtype",
                                    "through", "probe-count", "microbatch", "bool"])
def test_wrong_diagnostic_geometry_refuses_before_output(tmp_path, monkeypatch, broken):
    ids = draw()
    execution = _execution_one(tmp_path)
    spec = _spec(ids, execution)
    if broken == "probe-count":
        execution["n_probes"] = 2
    elif broken == "microbatch":
        execution["probe_microbatch"] = 0
    else:
        key, value = {
            "row": ("selected_global_row", len(ids)),
            "seed": ("probe_seed", execution["seed_base"] + 1),
            "N": ("global_token_count", ids.shape[1]),
            "vocab": ("vocab_size", 99),
            "shape": ("calibration_shape", [1, ids.shape[1]]),
            "tensor": ("calibration_tensor_sha256", "0" * 64),
            "dtype": ("calibration_dtype", "torch.int32"),
            "through": ("through", 99),
            "bool": ("selected_global_row", True),
        }[broken]
        spec[key] = value
        if broken == "shape":
            # Keep the declared subset internally legal so the actual full
            # input-shape wall, rather than the selector grammar, rejects it.
            spec["selected_global_row"] = 0
            spec["global_token_count"] = ids.shape[1]
    root = tmp_path / "refused"
    message = {"row": "selected_global_row", "seed": "seed differs",
               "N": "global_token_count", "vocab": "vocabulary differs",
               "shape": "shape differs", "tensor": "tensor differs", "dtype": "dtype differs",
               "through": "outside the chain", "probe-count": "one probe",
               "microbatch": "probe_microbatch", "bool": "selected_global_row"}[broken]
    with pytest.raises(stage_a.AdjointIdentityRefused, match=message):
        _run(root, monkeypatch, execution=execution, calib=ids,
             selected_row_diagnostic=spec)
    assert not root.exists()


@pytest.mark.parametrize("extra", ["chain_resume", "chain_seed", "chain_split",
                                   "forward_split", "forward_recovery"])
def test_diagnostic_never_borrows_a_tail(tmp_path, monkeypatch, extra):
    ids = draw()
    execution = _execution_one(tmp_path)
    with pytest.raises(stage_a.AdjointIdentityRefused, match="diagnostic"):
        _run(tmp_path / "refused", monkeypatch, execution=execution, calib=ids,
             selected_row_diagnostic=_spec(ids, execution), **{extra: {}})
    assert not (tmp_path / "refused").exists()


def test_nonfinite_tail_cannot_publish_success(tmp_path, monkeypatch):
    from test_stage_a_chain_resume import _dense_runner
    execution = _execution_one(tmp_path)
    def runner_factory():
        runner = _dense_runner()
        original = runner.tail_logits
        runner.tail_logits = lambda *a, **kw: original(*a, **kw) * float("nan")
        return runner
    with pytest.raises(stage_a.AdjointIdentityRefused, match="nonfinite"):
        _run(tmp_path / "bad", monkeypatch, execution=execution,
             runner_factory=runner_factory,
             selected_row_diagnostic=_spec(draw(), execution))
    assert not adjoint_receipt_path(adjoint_space(tmp_path / "bad")).exists()


def test_sealed_diagnostic_request_cannot_feed_band():
    with pytest.raises(BandRefused, match="diagnostic"):
        stage_a_argv(["python", "-m", "prismaquant.joint_cost_stage_a",
                      "--selected-row-diagnostic", "spec.json"])


def test_missing_original_owner_refuses_before_device_or_reads(tmp_path, monkeypatch):
    from prismaquant import gpu_guard
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path",
                        lambda *a: pytest.fail("device reached before source admission"))
    execution = _execution_one(tmp_path)
    with pytest.raises(stage_a.AdjointIdentityRefused, match="original.*owner"):
        stage_a.run_adjoint_capture(
            {"model": "unopened", "execution": execution},
            plan_sha256="1" * 64, prepared={}, output_root=tmp_path,
            selected_row_diagnostic=_spec(draw(), execution))


def test_spec_checksum_and_duplicate_keys_refuse(tmp_path):
    from prismaquant.stage_a_selected_row_diagnostic import load_diagnostic_spec
    path = tmp_path / "spec.json"
    raw = json.dumps(_spec(draw(), _execution_one(tmp_path))).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_diagnostic_spec(path, "0" * 64)
    raw = b'{"schema":"one","schema":"two"}'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        load_diagnostic_spec(path, hashlib.sha256(raw).hexdigest())


def test_certified_mode_refuses_diagnostic(tmp_path, monkeypatch):
    execution = _execution_one(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(stage_a.AdjointIdentityRefused, match="dev mode"):
        _run(tmp_path / "refused", monkeypatch, execution=execution,
             selected_row_diagnostic=_spec(draw(), execution))
    assert not (tmp_path / "refused").exists()


def test_direct_cuda_core_requires_original_owner(tmp_path, monkeypatch):
    from test_stage_a_chain_resume import _dense_runner
    def runner_factory():
        runner = _dense_runner()
        runner.device = torch.device("cuda")
        return runner
    execution = _execution_one(tmp_path)
    with pytest.raises(stage_a.AdjointIdentityRefused, match="original.*owner"):
        _run(tmp_path / "refused", monkeypatch, execution=execution,
             runner_factory=runner_factory,
             selected_row_diagnostic=_spec(draw(), execution))
    assert not (tmp_path / "refused").exists()


def test_cli_diagnostic_flags_must_be_paired():
    with pytest.raises(SystemExit) as exc:
        stage_a.main(["--plan", "unopened", "--plan-sha256", "1" * 64,
                      "--prepared", "unopened", "--prepared-sha256", "2" * 64,
                      "--output-root", "uncreated", "--selected-row-diagnostic", "unopened"])
    assert exc.value.code == 2


def test_diagnostic_root_cannot_be_adopted_by_ordinary_capture(tmp_path, monkeypatch):
    execution = _execution_one(tmp_path)
    root = tmp_path / "one"
    _run(root, monkeypatch, execution=execution,
         selected_row_diagnostic=_spec(draw(), execution))
    with pytest.raises(stage_a.AdjointIdentityRefused, match="diagnostic root"):
        _run(root, monkeypatch, execution=execution)
