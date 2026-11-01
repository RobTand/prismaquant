"""Actual resident admission versus an observed FP8 tail allocation request.

Both resident cases use real intake and decoding. The large-block case observes the existing
repeat_interleave request and stops before allocating more than 16 MiB. That
is an allocation-shape boundary, not a measured peak or serving claim.
"""
from __future__ import annotations

import json
import threading

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant import layer_streaming as decoder
from tests.test_resident_source_decode_audit import (
    _check_values, _cpu_guard, _load, _source,
)


class _LargeTailAllocation(RuntimeError):
    pass


@pytest.mark.parametrize("block", [48, 1 << 20], ids=["existing-tail-block", "large-declared-tail-block"])
def test_resident_reservation_covers_observed_tail_allocation(tmp_path, monkeypatch, block):
    root, expected = _source(tmp_path)
    if block != 48:
        tensors = load_file(root / "model.safetensors")
        for scale_name in list(tensors):
            if scale_name.endswith("_scale_inv"):
                weight = tensors[scale_name.removesuffix("_scale_inv")]
                grid = tuple((size + block - 1) // block for size in weight.shape)
                tensors[scale_name] = torch.ones(grid, dtype=torch.float32)
        replacement = root / "model.replacement.safetensors"
        save_file(tensors, replacement)
        replacement.replace(root / "model.safetensors")
        del tensors, weight
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"]["weight_block_size"] = [block, block]
        config_path.write_text(json.dumps(config))

    guard = _cpu_guard(tmp_path, monkeypatch)
    check = guard.check
    admissions = []

    def observe_admission(label, **kwargs):
        check(label, **kwargs)
        admissions.append((label, kwargs))

    guard.check = observe_admission
    active = threading.local()
    dequant = decoder._dequant_fp8_block_weight
    repeat = torch.Tensor.repeat_interleave
    allocations = []
    stopped = []

    def observe_decode(*args, **kwargs):
        active.inside_tail = True
        try:
            return dequant(*args, **kwargs)
        finally:
            active.inside_tail = False

    def observe_repeat(value, repeats, dim=None, *args, **kwargs):
        if getattr(active, "inside_tail", False):
            assert type(repeats) is int and dim in (0, 1)
            shape = list(value.shape)
            shape[dim] *= repeats
            allocation_bytes = value.numel() * repeats * value.element_size()
            label, reservation = admissions[-1]
            observed = (tuple(shape), allocation_bytes, label, reservation)
            allocations.append(observed)
            if allocation_bytes > 16 << 20:
                stopped.append(observed)
                raise _LargeTailAllocation("stopped before large padded tail allocation")
        return repeat(value, repeats, dim, *args, **kwargs)

    monkeypatch.setattr(decoder, "_dequant_fp8_block_weight", observe_decode)
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", observe_repeat)
    try:
        model = _load(root, tmp_path, guard)
    except _LargeTailAllocation:
        assert block != 48
        assert len(stopped) == 1
        shape, allocation_bytes, label, reservation = stopped[0]
        assert label.startswith("resident_source:before_layer:")
        assert reservation["reserve_bytes"] >= allocation_bytes, (
            f"Resident admission {reservation['reserve_bytes']} cannot cover even "
            f"one observed tail allocation {shape} ({allocation_bytes} bytes)")
    else:
        if block == 48:
            _check_values(model, expected, torch.bfloat16, False)
            assert allocations
        else:
            # A decoder that avoids padded repeats may complete instead.
            assert model._prismaquant_resident_source_initialization["status"] == "completed"
            assert all(not value.is_meta and torch.isfinite(value).all()
                       for value in model.parameters())
        assert guard.last["label"] == "resident_source:complete"


@pytest.mark.parametrize("shape,block", [
    ((7, 11), (4, 8)),
    ((8, 9), (4, 8)),
    ((9, 7), (1 << 20, 1 << 20)),
    ((9, 7), (1 << 20, 3)),
], ids=["two-tails", "one-tail", "large-block", "large-row-block"])
def test_tail_decode_matches_logical_coordinate_scale_oracle(shape, block):
    rows, columns = shape
    codes = torch.linspace(-3.0, 3.0, rows * columns).reshape(shape).to(torch.float8_e4m3fn)
    grid = tuple((size + width - 1) // width for size, width in zip(shape, block))
    scales = torch.linspace(0.2, 2.3, grid[0] * grid[1]).reshape(grid)
    # Index logical source coordinates directly; do not repeat a padded block.
    row_scales = scales.to(torch.bfloat16)[torch.arange(rows) // block[0]]
    logical_scales = row_scales[:, torch.arange(columns) // block[1]]
    expected = codes.to(torch.bfloat16) * logical_scales

    actual = decoder._dequant_fp8_block_weight(codes, scales, block, name="tail-oracle")

    assert actual.shape == shape and actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
