"""The native-cell driver declares which activation arithmetic priced its rows.

``nvfp4_activation_contract.quantize_dequantize`` refuses a priced path that no
caller bound, and it does not fall back to PrismaQuant's own Torch model: the
model and ``torch.ops._C.scaled_fp4_quant`` disagreed on 24 of 172,032 probed
elements by one E2M1 code (RobTand/prismaquant#567).  Before this binding existed
no production caller bound anything, so every fp4 cell refused at preparation and
the refusal was invisible -- fp8 and bf16 do not reach the static activation
contract, so the 2026-09-17 49-cell freeze never exercised it.

These tests mutate the DRIVER, never a fixture: they move what the process can
register and assert the driver's own answer moves with it.
"""
from __future__ import annotations

import sys
import types

import pytest

from experiments import pq_frontier_native_cells as driver
from nvfp4_served_qdq_fixtures import cpu_kernels
from prismaquant import nvfp4_activation_contract as owner


@pytest.fixture(autouse=True)
def _fresh_process_binding():
    owner._reset_served_quantizer_identity_for_tests()
    yield
    owner._reset_served_quantizer_identity_for_tests()


def _registers(monkeypatch):
    """Everything a box that CAN price these rows reports, and nothing more.

    The axes are stubbed because a CPU test box has no sm_121 device and no vLLM
    build; what is under test is the driver's declaration, not this host.  That
    includes the leg's Triton dequantisation kernels (#1211): a box that can
    price these rows loads them, so they are stood in for too (#1224).
    """
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: True)
    monkeypatch.setattr(owner, "_served_dequant_kernels", cpu_kernels)
    monkeypatch.setattr(owner, "_served_quantizer_platform", lambda: "sm_121")
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.28.1rc1"))
    monkeypatch.setenv("PRISMAQUANT_CONTAINER_CONTENT_SHA256", "a" * 64)


def test_the_driver_binds_the_registered_operator_when_it_registers(monkeypatch):
    _registers(monkeypatch)

    bound = driver.bind_priced_arithmetic()

    assert bound.backend == owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP
    assert bound.op == owner.SERVED_QUANTIZER_OP
    assert bound.image_content_sha256 == "a" * 64
    assert bound.platform == "sm_121"
    assert owner.active_served_quantizer_identity() == bound


def test_the_driver_refuses_rather_than_pricing_under_its_own_model(monkeypatch):
    _registers(monkeypatch)
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: False)

    with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
        driver.bind_priced_arithmetic()

    assert "native dense cell preparation" in str(refusal.value)
    assert owner.active_served_quantizer_identity() is None


def test_an_unstamped_image_is_refused_not_published(monkeypatch):
    """A binding that cannot name the image it ran in is unusable, not weaker."""
    _registers(monkeypatch)
    monkeypatch.delenv("PRISMAQUANT_CONTAINER_CONTENT_SHA256", raising=False)

    with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
        driver.bind_priced_arithmetic()

    assert "image_content_sha256" in str(refusal.value)
    assert owner.active_served_quantizer_identity() is None
