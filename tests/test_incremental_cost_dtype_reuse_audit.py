"""Real CLI shard reuse must preserve the requested measurement dtype."""
from __future__ import annotations

import json
import pickle
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import gpu_guard
from prismaquant import incremental_measure_quant_cost as owner


QNAME = "model.layers.0.proj"


class _TinyContext:
    def __init__(self, dtype):
        self.model = nn.Module()
        self.model.config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
        self.model.model = nn.Module()
        self.model.model.layers = nn.ModuleList([nn.Module()])
        self.model.model.layers[0].proj = nn.Linear(16, 4, bias=False, dtype=dtype)
        source = torch.arange(1, 65, dtype=torch.float32).reshape(4, 16).sin() * 0.31 + 0.137
        with torch.no_grad():
            self.model.model.layers[0].proj.weight.copy_(source.to(dtype))
        self.num_layers = 1
        self.layers_prefix = "model.layers."
        self.layer_cache = self

    def set_priority_layers(self, layers):
        assert set(layers) <= {0}

    def configure_runtime_pressure_floor(self):
        pass

    def install(self, layer):
        assert layer == 0

    def unload(self, layer):
        assert layer == 0
        return 0

    def shutdown(self):
        pass


@pytest.mark.parametrize("next_dtype, next_mode, legacy_unmarked", [
    ("bf16", "local", False), ("fp16", "local", False),
    ("fp32", "local", False), ("bf16", "aura", False),
    ("bf16", "local", True),
], ids=["unchanged", "fp16", "fp32", "cross_mode_same_dtype", "legacy_unmarked"])
def test_real_cost_main_reuses_only_matching_dtype(
    tmp_path, monkeypatch, next_dtype, next_mode, legacy_unmarked,
):
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1,
    }))
    probe = tmp_path / "probe.pkl"
    with probe.open("wb") as file:
        pickle.dump({"stats": {QNAME: {"h_trace": 1.0, "n_params": 64}}, "meta": {}}, file)
    activations = tmp_path / "act"
    activations.mkdir()
    rows = torch.arange(1, 113, dtype=torch.float32).reshape(7, 16).cos() / 3
    filename = owner.ActivationIndex._FNAME_SUB.sub("__", QNAME) + ".pt"
    torch.save({"inputs": rows, "name": QNAME}, activations / filename)
    loads = []

    def context(model_path, **kwargs):
        assert model_path == str(root)
        assert kwargs["device"] == torch.device("cpu")
        loads.append(kwargs["dtype"])
        return _TinyContext(kwargs["dtype"])

    monkeypatch.setattr(owner, "_build_streaming_context", context)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(owner, "_PIPELINE_COST_MODE", "")

    def run(work, dtype, mode):
        output = work / "cost.pkl"
        monkeypatch.setattr(sys, "argv", [
            "incremental_measure_quant_cost", "--model", str(root),
            "--probe", str(probe), "--activation-cache-dir", str(activations),
            "--work-dir", str(work), "--output", str(output),
            "--device", "cpu", "--dtype", dtype, "--mode", "unbatched",
            "--render-path", "registry", "--formats", "FP8_E4M3",
            "--cost-mode", mode, "--layers-per-shard", "1", "--threads", "1",
            "--no-include-mtp", "--no-include-visual", "--no-include-lm-head",
            "--no-watchdog",
        ])
        owner.main()
        with output.open("rb") as file:
            payload = pickle.load(file)
        record = payload["costs"][QNAME]["FP8_E4M3"]
        assert "error" not in record, record
        return record

    initial = run(tmp_path / "shared", "bf16", "local")
    shard = tmp_path / "shared" / "shards" / "cost_shard_000.pkl"
    if legacy_unmarked:
        with shard.open("rb") as file:
            old_payload = pickle.load(file)
        del old_payload["meta"]["incremental_shard"]["dtype"]
        with shard.open("wb") as file:
            pickle.dump(old_payload, file)
    resumed = run(tmp_path / "shared", next_dtype, next_mode)
    after_resume = list(loads)
    with shard.open("rb") as file:
        resumed_shard = pickle.load(file)
    expected_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
    }[next_dtype]
    assert resumed_shard["meta"]["incremental_shard"]["dtype"] == str(expected_dtype)
    fresh = run(tmp_path / "fresh", next_dtype, next_mode)
    fields = ("weight_mse", "output_mse", "rel_output_mse")
    if next_dtype != "bf16":
        assert any(initial[field] != fresh[field] for field in fields), (
            "dtype differential fixture produced identical actual measurements", initial, fresh,
        )
    assert all(resumed[field] == fresh[field] for field in fields), (
        f"real CLI resumed bf16 shard under requested {next_dtype}: "
        f"initial={initial}, resumed={resumed}, fresh={fresh}, "
        f"context_loads_after_resume={after_resume}"
    )
    expected_loads = [torch.bfloat16]
    if next_dtype != "bf16" or legacy_unmarked:
        expected_loads.append(expected_dtype)
    assert after_resume == expected_loads
