"""Real resident source delivery uses the published decoder and memory guard."""
from __future__ import annotations

import os

import pytest
import torch

from prismaquant import memory_management as memory
from prismaquant import streaming_model as owner
from tests.native_source_contract_fixtures import (
    producer_source as _source, producer_cpu_guard as _cpu_guard,
)


def _cpu_guard(tmp_path, monkeypatch, *, capacity=4 << 30):
    root = tmp_path / "cgroup"
    scope = root / "task"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(capacity))
    (scope / "memory.current").write_text("0")
    (scope / "memory.stat").write_text("anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
    membership = tmp_path / "membership"
    membership.write_text("0::/task\n")
    monkeypatch.setattr(memory, "_host_memory_info", lambda: (64 << 30, 128 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    return memory.CaptureMemoryGuard(torch.device("cpu"), cgroup_root=root, membership=membership)


def _load(root, tmp_path, guard, *, dtype=torch.bfloat16, requires_grad=False, device="cpu"):
    return owner.load_resident_source_model(
        str(root), device=torch.device(device), dtype=dtype,
        offload_folder=str(tmp_path / "offload"), resource_check=guard.check,
        attn_implementation="eager", requires_grad=requires_grad)


def _check_values(model, expected, dtype, requires_grad):
    plan = owner._model_tensor_dtypes(model, dtype)
    for name, parameter in model.named_parameters():
        wanted = expected[name].to(dtype=plan.get(name, dtype), device=parameter.device)
        assert not parameter.is_meta, name
        assert parameter.dtype == wanted.dtype, name
        assert torch.equal(parameter, wanted), name
        assert parameter.requires_grad is requires_grad, name
    assert model.config._attn_implementation == "eager"
    assert model._prismaquant_resident_source_initialization["status"] == "completed"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("requires_grad", [False, True], ids=["frozen", "gradient"])
def test_real_resident_native_source_values_and_requested_semantics(tmp_path, monkeypatch, dtype, requires_grad):
    root, expected = _source(tmp_path)
    guard = _cpu_guard(tmp_path, monkeypatch)
    from transformers import AutoModelForCausalLM

    def no_payload_load(*args, **kwargs):
        pytest.fail("resident delivery must not run HF payload loading first")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", no_payload_load)
    model = _load(root, tmp_path, guard, dtype=dtype, requires_grad=requires_grad)
    _check_values(model, expected, dtype, requires_grad)
    assert guard.last["label"] == "resident_source:complete"


@pytest.mark.parametrize("native,tied", [(False, False), (True, True)], ids=["bf16_source", "tied_head"])
def test_real_resident_preserves_unscaled_source_and_tied_head(tmp_path, monkeypatch, native, tied):
    root, expected = _source(tmp_path, native=native, tied=tied)
    model = _load(root, tmp_path, _cpu_guard(tmp_path, monkeypatch), dtype=torch.float32)
    _check_values(model, expected, torch.float32, False)
    if tied:
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_real_guard_refuses_aggregate_before_source_payload(tmp_path, monkeypatch):
    root, _ = _source(tmp_path)
    guard = _cpu_guard(tmp_path, monkeypatch, capacity=memory.CaptureMemoryGuard.MARGIN_BYTES + 1)
    original = owner._materialize
    payload_calls = []

    def observed_payload(*args, **kwargs):
        payload_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_materialize", observed_payload)
    with pytest.raises(RuntimeError):
        _load(root, tmp_path, guard)
    assert not payload_calls
    assert guard.last["label"] == "resident_source:before_payload"
    assert guard.last["future_allocation_bytes"] > 1


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_real_prefetch_failure_shuts_down_existing_pool(tmp_path, monkeypatch, tied):
    root, _ = _source(tmp_path, tied=tied)
    guard = _cpu_guard(tmp_path, monkeypatch)
    pools = []
    contexts = []
    original_pool = owner.ThreadPoolExecutor
    original_context = owner._build_streaming_context
    original_materialize = owner._materialize

    def remove_source_after_head(*args, **kwargs):
        result = original_materialize(*args, **kwargs)
        (root / "model.safetensors").unlink()
        return result

    def pool(*args, **kwargs):
        result = original_pool(*args, **kwargs)
        pools.append(result)
        return result

    monkeypatch.setattr(owner, "ThreadPoolExecutor", pool)
    monkeypatch.setattr(owner, "_materialize", remove_source_after_head)
    def context(*args, **kwargs):
        result = original_context(*args, **kwargs)
        contexts.append(result)
        return result

    monkeypatch.setattr(owner, "_build_streaming_context", context)
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        _load(root, tmp_path, guard)
    assert pools
    assert all(not thread.is_alive() for pool in pools for thread in pool._threads)
    assert contexts
    assert all(parameter.is_meta for _, parameter in
               contexts[0].model.named_parameters(remove_duplicate=False))


def test_resident_requires_existing_memory_guard_before_loading(tmp_path):
    with pytest.raises(ValueError, match="memory guard"):
        owner.load_resident_source_model(
            str(tmp_path / "missing"), device=torch.device("cpu"), dtype=torch.bfloat16,
            offload_folder=str(tmp_path / "offload"), resource_check=None)


@pytest.mark.skipif(os.environ.get("PRISMAQUANT_TEST_RESIDENT_SOURCE_CUDA") != "1",
                    reason="CUDA source admission requires an explicitly admitted GPU run")
def test_real_resident_cuda_source_decode(tmp_path):
    assert torch.cuda.is_available(), "explicit CUDA admission requires a visible GPU"
    root, expected = _source(tmp_path)
    device = torch.device("cuda")
    envelope = min(torch.cuda.get_device_properties(device).total_memory // 2, 4 << 30)
    memory.enforce_device_envelope(device, envelope, where="resident source audit")
    guard = memory.CaptureMemoryGuard(device, device_bytes=envelope)
    model = _load(root, tmp_path, guard, device=device, dtype=torch.float32)
    _check_values(model, expected, torch.float32, False)
