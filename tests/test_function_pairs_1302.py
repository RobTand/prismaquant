"""Each consolidated near-duplicate function pair keeps its outcomes (PQ #1302).

The duplication baseline listed these pairs as one responsibility implemented
twice. Each now has one owner, and each old site binds or calls it. Every call
below goes through the old site's name. Its outcome, a return or a refusal, was
frozen from the pre-consolidation code into
``tests/fixtures/function_pairs_1302.json`` (``tests/golden_table.py``).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("function_pairs_1302")


def _site(ref):
    module, name = ref.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


# --- canonical JSON value (production_weight_cache -> digests) -----------------

_JSON_VALUES = [
    {"b": 1, "a": [1, 2.5, "x"]},
    {"é": "ü", "n": None, "t": True},
    [float("nan")],
    {1: "int key"},
    {"s": {1, 2}},
    (1, 2),
]


@pytest.mark.parametrize("index", range(len(_JSON_VALUES)))
def test_canonical_json_value(index):
    site = _site("prismaquant.production_weight_cache._canonical_json_value")
    GOLDEN.call(lambda: site(_JSON_VALUES[index], where="w"))


# --- git HEAD commit (emu_forward_kl, fisher_col_weights -> aura_cost) ---------

_GIT_RESULTS = ["abc123\n", "", "  \n", OSError("no git"), subprocess.TimeoutExpired(["git"], 5)]


def _fake_run(result, calls):
    def run(argv, **kwargs):
        cwd = kwargs.pop("cwd", None)
        calls.append((list(argv), None if cwd is None else Path(cwd).name, sorted(kwargs.items())))
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(stdout=result, returncode=0)
    return run


@pytest.mark.parametrize("ref", [
    "prismaquant.emu_forward_kl._git_commit",
    "prismaquant.fisher_col_weights._git_commit",
    "tools.tp2_budget_plan._git_sha",
    "tools.tp_decode_feasibility._git_sha",
])
@pytest.mark.parametrize("index", range(len(_GIT_RESULTS)))
def test_git_commit(monkeypatch, ref, index):
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", raising=False)
    site = _site(ref)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(_GIT_RESULTS[index], calls))
    GOLDEN.call(site)
    GOLDEN.value(calls)


def test_aura_git_commit_keeps_its_override(monkeypatch):
    from prismaquant import aura_cost
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run("f" * 40 + "\n", calls))
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", raising=False)
    GOLDEN.call(aura_cost._git_commit)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "A" * 40)
    GOLDEN.call(aura_cost._git_commit)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "not-a-commit")
    GOLDEN.call(aura_cost._git_commit)
    GOLDEN.value(calls)


# --- routed-expert profile accessor calls --------------------------------------

class _Profile:
    def works(self, *args):
        return ("ok", args)

    def breaks(self, *args):
        raise KeyError("inner")

    not_callable = 3


@pytest.mark.parametrize("ref", [
    "prismaquant.routed_experts._profile_call",
    "prismaquant.expert_empirical_cost._expert_profile_call",
])
@pytest.mark.parametrize("accessor", ["works", "breaks", "not_callable", "missing"])
def test_profile_call(ref, accessor):
    site = _site(ref)
    GOLDEN.call(lambda: site(_Profile(), accessor, "a", 2))


# --- packed-expert helpers (export_native_compressed, measure_quant_cost ->
#     sensitivity_probe) -------------------------------------------------------

class _NamesProfile:
    def __init__(self, names=None, parent=None, fail=False):
        self._names, self._parent, self._fail = names, parent, fail

    def packed_expert_param_names(self):
        if self._fail:
            raise RuntimeError("no names")
        return self._names

    def packed_expert_parent_for_projection(self, projection):
        if self._fail:
            raise RuntimeError("no parent")
        return None if self._parent is None else f"{self._parent}:{projection}"


_PROFILES = [None, _NamesProfile(names=["gate_up_proj", "down_proj"], parent="experts"),
             _NamesProfile(fail=True), _NamesProfile(names=[], parent=None)]


class Qwen3MoeExperts(nn.Module):
    def __init__(self, name, shape):
        super().__init__()
        self.register_parameter(name, nn.Parameter(torch.zeros(shape)))


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 2, 2))


_MODULES = [Qwen3MoeExperts("gate_up_proj", (2, 2, 2)), Qwen3MoeExperts("gate_up_proj", (2, 2)),
            Qwen3MoeExperts("other", (2, 2, 2)), Conv()]


@pytest.mark.parametrize("module_name", [
    "prismaquant.export_native_compressed", "prismaquant.sensitivity_probe"])
@pytest.mark.parametrize("profile_index", range(len(_PROFILES)))
def test_packed_expert_param_names_and_module_test(module_name, profile_index):
    module = importlib.import_module(module_name)
    profile = _PROFILES[profile_index]
    GOLDEN.call(lambda: sorted(module._packed_expert_param_name_set(profile)))
    for candidate in _MODULES:
        GOLDEN.call(lambda: module._is_packed_experts_module(candidate, profile))
        GOLDEN.call(lambda: module._packed_experts_param_names(candidate, profile))


@pytest.mark.parametrize("module_name", [
    "prismaquant.export_native_compressed", "prismaquant.measure_quant_cost"])
@pytest.mark.parametrize("profile_index", range(len(_PROFILES)))
@pytest.mark.parametrize("projection", ["gate_proj", "down_proj", "unknown"])
def test_packed_expert_parent_for_projection(module_name, profile_index, projection):
    module = importlib.import_module(module_name)
    GOLDEN.call(lambda: module._packed_expert_parent_for_projection(
        _PROFILES[profile_index], projection))


# --- control-metadata directories (joint_layer_quanta, joint_stage_b_head) -----

_ROOTS = [("/out", None), ("/out/", None), ("/out", "/meta"), ("/out", "/meta/"),
          ("relative", None), (None, None), ("/out", "meta"), ("/out", "/a/../b"),
          ("/out", "/")]


@pytest.mark.parametrize("ref", [
    "prismaquant.joint_layer_quanta.bound_readset_directory",
    "prismaquant.joint_stage_b_head.head_slice_directory",
])
@pytest.mark.parametrize("index", range(len(_ROOTS)))
def test_control_directory(ref, index):
    site = _site(ref)
    output_root, metadata_root = _ROOTS[index]
    GOLDEN.call(lambda: site(output_root, metadata_root=metadata_root))


# --- first tensor of a forward call (layer_state_cache -> perturbed_x_cache) ---

_T = torch.arange(3.0)
_CALLS = [((), {}), ((1, _T), {}), ((), {"x": 1, "input": _T}),
          ((), {"other": _T, "hidden_states": _T + 1}), (("a",), {"z": _T}),
          ((), None), ((1,), {"inputs_embeds": 1})]


@pytest.mark.parametrize("ref", [
    "prismaquant.layer_state_cache._first_tensor_location",
    "prismaquant.perturbed_x_cache._first_tensor_location",
])
@pytest.mark.parametrize("index", range(len(_CALLS)))
def test_first_tensor_location(ref, index):
    site = _site(ref)
    args, kwargs = _CALLS[index]
    GOLDEN.call(lambda: site(args, kwargs))


# --- model-profile name claims -------------------------------------------------

_CLAIMS = [("gemma4", []), ("gemma4_text", []), ("", ["Gemma4ForCausalLM"]),
           ("lfm2_moe", []), ("lfm2-moe", []), ("", ["Lfm2MoeForCausalLM"]),
           ("minimax_m2.7", []), ("", ["MiniMax-M2X"]), ("", ["MiniMaxM2ForCausalLM"]),
           ("deepseek-v4", []), ("", ["DeepSeek-V4-Custom"]), ("", ["DeepseekV4ForCausalLM"]),
           ("glm5_next_text", []), ("", ["Glm5NextForCausalLM"]), ("laguna", []),
           ("", ["LagunaForCausalLM"]), ("hy_v3", []), ("", ["HYV3ForCausalLM"]),
           ("qwen3", ["Qwen3ForCausalLM"]), ("", ["gemma4"]), ("", ["X", "Laguna1"]),
           (None, ["Gemma4X"])]
_PROFILE_CLASSES = [
    "prismaquant.model_profiles.gemma4.Gemma4Profile",
    "prismaquant.model_profiles.lfm2_moe.Lfm2MoeProfile",
    "prismaquant.model_profiles.minimax_m2.MiniMaxM2Profile",
    "prismaquant.model_profiles.deepseek_v4.DeepseekV4Profile",
    "prismaquant.model_profiles.glm5_next.Glm5NextProfile",
    "prismaquant.model_profiles.laguna.LagunaProfile",
    "prismaquant.model_profiles.hy_v3.HyV3Profile",
]


@pytest.mark.parametrize("ref", _PROFILE_CLASSES)
def test_profile_matches(ref):
    profile = _site(ref)
    for model_type, architectures in _CLAIMS:
        GOLDEN.call(lambda: profile.matches(model_type, architectures))


# --- gold measurement provenance (measure_vllm_full_kl, _wikitext_ppl) ---------

def _gold_args(**overrides):
    fields = dict(model="artifact", dtype="bfloat16", gpu_memory_utilization=0.5,
                  seqlen=512, max_logprobs=100, enforce_eager=True,
                  quantization=None, max_num_batched_tokens=1024,
                  tensor_parallel_size=2, nnodes=2, node_rank=0,
                  master_addr="192.0.2.1", master_port=29501,
                  distributed_executor_backend="mp", data_parallel_backend="mp",
                  moe_backend="triton", kv_cache_memory_bytes=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("runner", ["tools.measure_vllm_full_kl", "tools.measure_vllm_wikitext_ppl"])
@pytest.mark.parametrize("case", ["tp2", "tp1", "peer", "no_image"])
def test_gold_provenance(monkeypatch, runner, case):
    module = importlib.import_module(runner)
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setattr(module, "_ENGINE_KWARGS", {"model": "m", "tensor_parallel_size": 2}
                        if case == "peer" else None)
    monkeypatch.setattr(module, "_SPEC_DECODE_DETECTED", case == "tp1")
    monkeypatch.setattr(module, "gold_producer_identity",
                        lambda name: {"git_commit": "a" * 40, "tool": name})

    def image(args):
        if case == "no_image":
            raise RuntimeError("no image")
        return "image@sha256:" + "b" * 64
    monkeypatch.setattr(module, "_resolve_serve_image", image)
    monkeypatch.setattr(module, "self_manifest",
                        lambda **kwargs: {"serve_fingerprint": "c" * 64, **kwargs})
    args = _gold_args(nnodes=1, tensor_parallel_size=1) if case == "tp1" else _gold_args()
    GOLDEN.call(lambda: module._provenance(args))


# --- dropping a file's client page cache (benches -> io_spans) -----------------

@pytest.mark.parametrize("ref", [
    "tools.render_window_bench._drop_pages",
    "tools.pwc_window_load_bench._drop_client_pages",
    "tools.staged_exact_read_bench.drop_client_cache",
    "tools.staged_read_stream_ab.drop_client_cache",
])
def test_drop_page_cache(tmp_path, ref):
    site = _site(ref)
    present = tmp_path / "present.bin"
    present.write_bytes(b"x" * 10)
    (tmp_path / "empty.bin").write_bytes(b"")
    GOLDEN.call(lambda: site([]), tmp=tmp_path)
    GOLDEN.call(lambda: site([present, str(tmp_path / "empty.bin")]), tmp=tmp_path)
    GOLDEN.call(lambda: site([present, tmp_path / "missing.bin", present]), tmp=tmp_path)
    GOLDEN.call(lambda: site([tmp_path]), tmp=tmp_path)
    assert present.read_bytes() == b"x" * 10
    assert os.path.exists(present)
