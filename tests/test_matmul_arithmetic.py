"""The shared matmul pin and the bf16 reduction flag's stamp (PQ #1028).

Every Stage A, Stage B and joint AURA entry point pins its matmul settings
through ``matmul_arithmetic.pin_matmul_arithmetic``. The bf16 reduced-precision
reduction flag is a launch setting: unset keeps PyTorch's default and stamps
nothing, so every identity written before the flag was pinned keeps its bytes;
``off`` stamps it into every arithmetic identity, so a quantum run under
another setting cannot join the rest.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prismaquant.joint_aura import arithmetic_identity, identity_sha256
from prismaquant.joint_quanta_join import JoinRefused, join_joint_quanta
from prismaquant.joint_replay_regime import stamp_replay_regime
from prismaquant.joint_statistics_replay import statistics_arithmetic_identity
from prismaquant.matmul_arithmetic import (
    BF16_REDUCTION_ENV,
    BF16_REDUCTION_FIELD,
    MatmulArithmeticRefused,
    bf16_reduction_from_environment,
    bf16_reduction_of,
    bf16_reduction_stamp,
    pin_matmul_arithmetic,
)
from prismaquant.stage_a_chain_resume import chain_arithmetic_stamp
from tests.test_joint_quanta_allocator_bridge import _generated_outputs
from tests.test_joint_quanta_join import campaign, probe  # noqa: F401 (fixtures)
from tests.test_stageb_replay_regime import PINNED_DEFAULT, _quanta, _restamp

MATMUL = torch.backends.cuda.matmul

#: ``arithmetic_identity``'s fields before the flag was pinned, in order.
DEFAULT_FIELDS = [
    "projection_backend", "projection_dtype", "delta_dtype", "measurement_dtype",
    "matmul_precision", "allow_tf32", "weight_projection", "residual", "aggregation"]


@pytest.fixture
def flags():
    """Restore every setting the pin touches, whatever a test leaves."""
    saved = (torch.get_float32_matmul_precision(), MATMUL.allow_tf32,
             MATMUL.allow_bf16_reduced_precision_reduction)
    torch.set_float32_matmul_precision("highest")
    MATMUL.allow_tf32 = False
    MATMUL.allow_bf16_reduced_precision_reduction = True
    yield
    torch.set_float32_matmul_precision(saved[0])
    MATMUL.allow_tf32 = saved[1]
    MATMUL.allow_bf16_reduced_precision_reduction = saved[2]


# ---- the launch setting ----------------------------------------------------

def test_unset_is_the_default_and_off_is_the_only_value():
    assert bf16_reduction_from_environment({}) is True
    assert bf16_reduction_from_environment({BF16_REDUCTION_ENV: "off"}) is False


@pytest.mark.parametrize("text,message", [
    ("on", "spells the default"),
    ("", "the only value"),
    ("OFF", "the only value"),
    (" off", "the only value"),
    ("0", "the only value"),
    ("false", "the only value"),
])
def test_any_other_setting_is_refused(text, message):
    with pytest.raises(MatmulArithmeticRefused, match=message):
        bf16_reduction_from_environment({BF16_REDUCTION_ENV: text})


def test_the_pin_sets_every_setting_the_entry_points_pinned(flags):
    torch.set_float32_matmul_precision("medium")
    MATMUL.allow_tf32 = True
    MATMUL.allow_bf16_reduced_precision_reduction = False
    assert pin_matmul_arithmetic({}) is True
    assert torch.get_float32_matmul_precision() == "highest"
    assert MATMUL.allow_tf32 is False
    assert MATMUL.allow_bf16_reduced_precision_reduction is True
    assert pin_matmul_arithmetic({BF16_REDUCTION_ENV: "off"}) is False
    assert MATMUL.allow_bf16_reduced_precision_reduction is False


def test_a_malformed_setting_refuses_before_anything_is_pinned(flags):
    torch.set_float32_matmul_precision("medium")
    with pytest.raises(MatmulArithmeticRefused):
        pin_matmul_arithmetic({BF16_REDUCTION_ENV: "on"})
    assert torch.get_float32_matmul_precision() == "medium"
    assert MATMUL.allow_bf16_reduced_precision_reduction is True


def test_the_pin_reads_the_process_environment_by_default(flags, monkeypatch):
    monkeypatch.setenv(BF16_REDUCTION_ENV, "off")
    assert pin_matmul_arithmetic() is False
    monkeypatch.delenv(BF16_REDUCTION_ENV)
    assert pin_matmul_arithmetic() is True


# ---- the stamp -------------------------------------------------------------

def test_the_default_stamps_nothing(flags):
    assert bf16_reduction_stamp() == {}
    assert list(arithmetic_identity(torch.bfloat16)) == DEFAULT_FIELDS
    assert statistics_arithmetic_identity(torch.bfloat16, None) == PINNED_DEFAULT
    assert bf16_reduction_of(PINNED_DEFAULT) is True


def test_the_stamp_differs_when_the_flag_differs(flags):
    default = statistics_arithmetic_identity(torch.bfloat16, None)
    MATMUL.allow_bf16_reduced_precision_reduction = False
    assert bf16_reduction_stamp() == {BF16_REDUCTION_FIELD: False}
    off = statistics_arithmetic_identity(torch.bfloat16, None)
    assert off == {**PINNED_DEFAULT, BF16_REDUCTION_FIELD: False}
    assert identity_sha256(off) != identity_sha256(default)
    assert bf16_reduction_of(off) is False
    assert arithmetic_identity(torch.bfloat16)[BF16_REDUCTION_FIELD] is False


def test_the_flag_and_the_replay_regime_are_independent_axes(flags):
    regime = "capture_batch=8,accumulation=operator_gemm,chunk_rows=65536"
    digests = set()
    for allow in (True, False):
        MATMUL.allow_bf16_reduced_precision_reduction = allow
        for text in (None, regime):
            digests.add(identity_sha256(statistics_arithmetic_identity(
                torch.bfloat16, None, replay_regime=text)))
    assert len(digests) == 4


@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_a_stamped_default_or_malformed_stamp_is_refused(value):
    with pytest.raises(MatmulArithmeticRefused, match="is stamped"):
        bf16_reduction_of({**PINNED_DEFAULT, BF16_REDUCTION_FIELD: value})


def test_the_stage_a_chain_stamp_differs_when_the_flag_differs(flags):
    runner = SimpleNamespace(device="cpu", dtype=torch.bfloat16)
    default = chain_arithmetic_stamp(runner)
    assert BF16_REDUCTION_FIELD not in default
    MATMUL.allow_bf16_reduced_precision_reduction = False
    off = chain_arithmetic_stamp(runner)
    assert off == {**default, BF16_REDUCTION_FIELD: False}


# ---- the entry points ------------------------------------------------------

@pytest.mark.parametrize("text", ["on", "0"])
def test_both_launchers_refuse_a_malformed_setting_before_any_work(monkeypatch, text):
    import prismaquant.gpu_guard as gpu_guard
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused, run_layer_quantum
    from prismaquant.joint_cost_stage_a import AdjointIdentityRefused, run_adjoint_capture
    from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: None)
    monkeypatch.delenv(REPLAY_REGIME_ENV, raising=False)
    monkeypatch.setenv(BF16_REDUCTION_ENV, text)
    with pytest.raises(QuantumIdentityRefused, match=BF16_REDUCTION_ENV):
        run_layer_quantum({"execution": {}}, record={}, adjoint_slice={}, plan_sha256="",
                          prepared="", output_root="")
    with pytest.raises(AdjointIdentityRefused, match=BF16_REDUCTION_ENV):
        run_adjoint_capture({}, plan_sha256="", prepared="", output_root="")


def test_the_container_forwards_the_sealed_setting_verbatim():
    from tools.tessera_campaign_container import docker_command

    spec = {"model": "/mnt/shared/model", "cwd": "/original/checkout",
            "python": "python3", "campaign_argv": [],
            "env": {"PYTHONPATH": ".", BF16_REDUCTION_ENV: "off"},
            "container": {"image": "qualified:fixed", "mounts": [
                {"source": "/mnt/shared", "target": "/mnt/shared"}]}}
    argv = docker_command(spec, ["python3"], cwd="/snapshot", uid=1, gid=1,
                          image_id="sha256:x", environ={})
    assert f"{BF16_REDUCTION_ENV}=off" in argv


# ---- the campaign join -----------------------------------------------------

def _off(arithmetic):
    return {**arithmetic, BF16_REDUCTION_FIELD: False}


def test_join_refuses_a_quantum_whose_flag_differs(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    quanta = _quanta(root)
    assert len(quanta) == 3
    _restamp(root, quanta[1], _off)
    with pytest.raises(JoinRefused, match="probe or measurement identity differs"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "refused")


def test_join_accepts_one_uniform_off_setting(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    for quantum in _quanta(root):
        _restamp(root, quantum, lambda arithmetic: _off(stamp_replay_regime(
            arithmetic, "capture_batch=8,accumulation=operator_gemm,chunk_rows=65536")))
    result = join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                               output_dir=tmp_path / "joined")
    assert result["status"] == "complete"


def test_a_stamped_default_row_fails_its_currency_check(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    _restamp(root, _quanta(root)[0],
             lambda arithmetic: {**arithmetic, BF16_REDUCTION_FIELD: True})
    with pytest.raises(JoinRefused, match="is stamped"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "refused")
