"""The Stage B quantum runs Stage A's bf16 reduction setting (PQ #1065).

Stage A stamps ``allow_bf16_reduced_precision_reduction: false`` into its run
identity when it ran with the flag off, and nothing at PyTorch's default
(PQ #1028). A quantum rebuilds Stage A's chain from its checkpoint, so its
own setting must be the one the slice records: certified mode refuses a
mismatch and names both settings, dev mode records it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant.dev_mode import DEV_MODE_ENV  # noqa: E402
from prismaquant.joint_cost_quantum import (  # noqa: E402
    QuantumIdentityRefused,
    require_slice_bf16_reduction,
    run_layer_quantum,
)
from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV  # noqa: E402
from prismaquant.matmul_arithmetic import (  # noqa: E402
    BF16_REDUCTION_ENV,
    BF16_REDUCTION_FIELD,
)

MATMUL = torch.backends.cuda.matmul


class _Stop(Exception):
    pass


@pytest.fixture
def launcher(monkeypatch):
    """A CPU launcher: no CUDA gate, no replay regime, flags restored after."""
    import prismaquant.gpu_guard as gpu_guard

    saved = (MATMUL.allow_bf16_reduced_precision_reduction,
             MATMUL.allow_fp16_reduced_precision_reduction, MATMUL.allow_tf32)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.delenv(REPLAY_REGIME_ENV, raising=False)
    monkeypatch.delenv(BF16_REDUCTION_ENV, raising=False)
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    yield monkeypatch
    (MATMUL.allow_bf16_reduced_precision_reduction,
     MATMUL.allow_fp16_reduced_precision_reduction, MATMUL.allow_tf32) = saved


def _launch(tmp_path, adjoint_slice):
    record = {"layer": 1, "quantum_id": "layer-001", "campaign": {},
              "output_space": {"root": str(tmp_path / "space")},
              "executable_readset": {"head_slice": {"path": "slice.json",
                                                    "sha256": "a" * 64}}}
    return run_layer_quantum(
        {"execution": {}}, record=record, adjoint_slice=adjoint_slice,
        plan_sha256="c" * 64, prepared={"path": "p", "sha256": "b" * 64},
        output_root=tmp_path)


def _stop_at_the_head(monkeypatch):
    import prismaquant.joint_stage_b_head as head_mod

    def read_slice(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setattr(head_mod, "read_quantum_head_slice", read_slice)


def test_a_slice_stamped_off_refuses_a_quantum_at_the_default(tmp_path, launcher):
    slice_off = {"run_identity": {BF16_REDUCTION_FIELD: False}}
    with pytest.raises(QuantumIdentityRefused) as refused:
        _launch(tmp_path, slice_off)
    message = str(refused.value)
    assert "quantum layer-001" in message
    assert f"{BF16_REDUCTION_FIELD}=False ({BF16_REDUCTION_ENV}=off)" in message
    assert f"{BF16_REDUCTION_FIELD}=True ({BF16_REDUCTION_ENV} unset)" in message
    assert not (tmp_path / "space").exists()


def test_a_quantum_run_off_refuses_an_unstamped_slice(tmp_path, launcher):
    launcher.setenv(BF16_REDUCTION_ENV, "off")
    with pytest.raises(QuantumIdentityRefused) as refused:
        _launch(tmp_path, {"run_identity": {}})
    message = str(refused.value)
    assert message.index(f"({BF16_REDUCTION_ENV} unset)") < message.index(
        f"({BF16_REDUCTION_ENV}=off)")


@pytest.mark.parametrize("env, stamp", [
    (None, {}),
    ("off", {BF16_REDUCTION_FIELD: False}),
])
def test_matching_settings_reach_the_head(tmp_path, launcher, env, stamp):
    if env is not None:
        launcher.setenv(BF16_REDUCTION_ENV, env)
    _stop_at_the_head(launcher)
    with pytest.raises(_Stop):
        _launch(tmp_path, {"run_identity": stamp})


def test_dev_mode_records_the_mismatch_and_goes_on(tmp_path, launcher, capsys):
    launcher.setenv(DEV_MODE_ENV, "1")
    _stop_at_the_head(launcher)
    with pytest.raises(_Stop):
        _launch(tmp_path, {"run_identity": {BF16_REDUCTION_FIELD: False}})
    out = capsys.readouterr().out
    assert "[DEV-MODE] quantum layer-001: Stage A ran its chain with" in out
    assert "recorded, not gated (dev mode)" in out


@pytest.mark.parametrize("identity", [
    {BF16_REDUCTION_FIELD: True},
    {BF16_REDUCTION_FIELD: "off"},
    {BF16_REDUCTION_FIELD: 0},
])
def test_a_malformed_stamp_refuses_in_either_mode(monkeypatch, identity):
    for dev in (False, True):
        if dev:
            monkeypatch.setenv(DEV_MODE_ENV, "1")
        else:
            monkeypatch.setenv(DEV_MODE_ENV, "0")
        with pytest.raises(QuantumIdentityRefused, match="bf16 reduction stamp"):
            require_slice_bf16_reduction({"run_identity": identity}, True, where="q")


@pytest.mark.parametrize("adjoint_slice", [{}, {"run_identity": None}, None])
def test_a_slice_without_a_run_identity_refuses(adjoint_slice):
    with pytest.raises(QuantumIdentityRefused, match="carries no run identity"):
        require_slice_bf16_reduction(adjoint_slice, True, where="q")
