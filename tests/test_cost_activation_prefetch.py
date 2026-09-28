"""Measured activation-chunk budgets and shared-engine lifetime (#1294)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from prismaquant import format_registry as fr
from prismaquant import io_engine
from prismaquant import measure_quant_cost as mqc


def _cache(tmp_path):
    names = ["a", "b", "c"]
    for name, rows in zip(names, (1, 17, 5), strict=True):
        torch.save(
            {"inputs": torch.arange(rows * 64).reshape(rows, 64).float() / 100,
             "row_indices": torch.arange(rows)},
            tmp_path / f"{name}.pt",
        )
    return mqc.ActivationIndex(tmp_path, names), names


def _chunks(names):
    return [[(name, None)] for name in names]


def test_budget_is_derived_from_measured_adjacent_chunks(tmp_path, monkeypatch):
    cache, names = _cache(tmp_path)
    captured = {}

    def capture(entries, budget):
        captured.update(entries=entries, budget=budget)
        return captured

    monkeypatch.setattr(io_engine, "read_stream", capture)
    assert mqc._activation_chunk_stream(cache, _chunks(names)) is captured
    sizes = [(tmp_path / f"{name}.pt").stat().st_size for name in names]
    budget = captured["budget"]
    assert [entry.size for entry in captured["entries"]] == sizes
    assert budget.buffer_bytes == max(sizes)
    assert budget.capacity_bytes == max(sizes[0] + sizes[1], sizes[1] + sizes[2])
    assert budget.headroom_bytes(sizes[0]) == budget.capacity_bytes - sizes[0]
    assert budget.headroom_bytes(budget.capacity_bytes + 1) == 0
    # Changing the actual chunk, not a global byte/depth constant, changes the cap.
    mqc._activation_chunk_stream(cache, [[(name, None) for name in names]])
    assert captured["budget"].capacity_bytes == sum(sizes)


def test_stream_is_byte_identical_and_legacy_load_policy_is_unchanged(tmp_path, monkeypatch):
    cache, names = _cache(tmp_path)
    expected = [cache.load_with_row_indices(name) for name in names]
    original_load = torch.load
    kwargs_seen = []

    def load(*args, **kwargs):
        kwargs_seen.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    with mqc._activation_chunk_stream(cache, _chunks(names)) as stream:
        for i, want in enumerate(expected):
            got = stream.take(i)[0].value[0]
            for a, b in zip(got, want, strict=True):
                assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                assert torch.equal(a, b)
            del got
            stream.release()
    assert len(kwargs_seen) == len(names)
    assert all(kwargs == {"map_location": "cpu", "weights_only": False}
               for kwargs in kwargs_seen)
    assert stream._closed


def test_underpriced_retained_storage_refuses_delivery():
    class Underpriced:
        def prefetch_bytes(self, name):
            return 1

        def load_with_row_indices(self, name):
            return torch.ones(64), None

    with mqc._activation_chunk_stream(Underpriced(), _chunks(["a"])) as stream:
        with pytest.raises(io_engine.EntryError, match="beyond its 1-byte charge"):
            stream.take(0)


def test_storage_measure_counts_views_once():
    value = torch.arange(10)
    assert mqc._activation_chunk_bytes([(value[:2], value[2:])]) == 80


def _model(names):
    torch.manual_seed(7)
    model = nn.Module()
    for name in names:
        model.add_module(name, nn.Linear(64, 32, bias=False))
    return model


def test_prefetch_does_not_change_cost_rows(tmp_path, monkeypatch):
    cache, names = _cache(tmp_path)
    model = _model(names)
    specs = [fr.get_format("BF16"), fr.get_format("INT8_W8A16")]
    monkeypatch.setenv("PRISMAQUANT_COST_PREFETCH_ACT", "0")
    sync = mqc.measure_batched_gpu(
        model, cache, set(names), specs, "cpu", torch.float32, chunk_size=1,
    )
    monkeypatch.delenv("PRISMAQUANT_COST_PREFETCH_ACT")
    async_rows = mqc.measure_batched_gpu(
        model, cache, set(names), specs, "cpu", torch.float32, chunk_size=1,
    )
    assert async_rows == sync


def test_measurement_failure_closes_read_ahead(tmp_path, monkeypatch):
    cache, names = _cache(tmp_path)
    created = []
    original_stream = io_engine.read_stream

    def create(*args, **kwargs):
        stream = original_stream(*args, **kwargs)
        created.append(stream)
        return stream

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate measurement failure")

    monkeypatch.setattr(io_engine, "read_stream", create)
    monkeypatch.setattr(mqc, "_batched_quantize", fail)
    monkeypatch.setattr(mqc, "_COST_FAIL_FAST", True)
    monkeypatch.setenv("PRISMAQUANT_COST_PREFETCH_ACT", "1")
    with pytest.raises(RuntimeError, match="deliberate measurement failure"):
        mqc.measure_batched_gpu(
            _model(names), cache, set(names), [fr.get_format("BF16")],
            "cpu", torch.float32, chunk_size=1,
        )
    assert created and all(stream._closed for stream in created)
