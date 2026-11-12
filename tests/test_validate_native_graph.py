"""Native smoke observations never construct original equality-measurement evidence."""
import copy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import types

import pytest

from prismaquant.digests import bytes_sha256hex
from prismaquant.shipcard import (
    load_shipcard, open_shipcard, verify, write_shipcard,
)
from tests.graph_receipt_fixtures import graph_fixture


@pytest.fixture
def native_graph(tmp_path, monkeypatch):
    from prismaquant import validate_native_export as native

    model = tmp_path / "native"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"tiny native weights")
    card = open_shipcard(model, build={"achieved_bpp": {"value": 4.75}})
    write_shipcard(model / "shipcard.json", card)
    producer, _ = graph_fixture(model, tensor_parallel_size=1)
    scope = producer["metrics"]["serve_scope"]

    @dataclass
    class Compilation:
        mode: int = 3
        custom_ops: list = field(default_factory=lambda: ["all"])
        cudagraph_capture_sizes: list = field(default_factory=lambda: [1, 2])

    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model=str(model), max_model_len=8448),
        scheduler_config=types.SimpleNamespace(max_num_seqs=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None, compilation_config=Compilation(),
        kernel_config=types.SimpleNamespace(ir_op_priority={"rms_norm": ["native"]}),
    )
    observed_ranks = [{"rank": 0, "tensor_parallel_size": 1,
                       "image": scope["image"], "fabric": "none",
                       "tessera_src_sha256": scope["tessera_src_sha256"],
                       "model_config_sha256": scope["model_config_sha256"]}]
    calls = []

    class Engine:
        llm_engine = types.SimpleNamespace(vllm_config=config)

        def __init__(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            if "compilation_config" in kwargs:
                kwargs["compilation_config"]["mutated_by_engine"] = True

        def apply_model(self, observer):
            return copy.deepcopy(observed_ranks)

        def generate(self, prompts, params):
            return [types.SimpleNamespace(prompt=prompts[0],
                outputs=[types.SimpleNamespace(text="native output")])]

    module = types.ModuleType("vllm")
    module.LLM = Engine
    module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(native, "_graph_image", lambda: scope["image"], raising=False)
    monkeypatch.setattr(native, "_graph_tessera_source_sha256",
                        lambda: scope["tessera_src_sha256"], raising=False)
    args = types.SimpleNamespace(
        graph_receipt=producer["metrics"]["graph_receipt_path"],
        compilation_config='{"mode":3,"cudagraph_capture_sizes":[1,2]}',
        max_model_len=9000, max_num_seqs=8, tensor_parallel_size=2,
        gpu_memory_utilization=0.5, max_new_tokens=16, prompt="native prompt",
        shipcard=str(model / "shipcard.json"), route_sweep_out=None,
    )
    return native, args, model, config, observed_ranks, calls


def test_graph_native_observes_actual_engine_without_original_equality_link(native_graph):
    native, args, model, config, ranks, calls = native_graph
    result = native._run_arm(args, model, None, enforce_eager=False)
    assert result["passed"], result
    assert calls[0]["quantization"] == "compressed-tensors"
    assert calls[0]["compilation_config"] == {"mode": 3, "cudagraph_capture_sizes": [1, 2]}
    assert calls[0]["max_num_seqs"] == 8
    assert calls[0]["tensor_parallel_size"] == 2
    metrics = result["metrics"]
    assert metrics["serve_scope"]["max_model_len"] == 8448
    assert metrics["serve_scope"]["max_num_seqs"] == 4
    assert metrics["serve_scope"]["speculative_tokens"] == 0
    assert metrics["serve_scope"]["tensor_parallel_size"] == 1
    assert "mutated_by_engine" not in metrics["serve_scope"]["compilation_config"]
    assert metrics["graph_receipt_sha256"] == bytes_sha256hex(
        Path(args.graph_receipt).read_bytes())
    assert metrics["graph_smoke_observation"]["ranks"] == ranks
    assert metrics["graph_smoke_observation"]["resolved"]["compilation_config"]["custom_ops"] == ["all"]
    assert "graph_observation" not in metrics
    native._record_arm(args, model, None, result)
    problems = verify(load_shipcard(model / "shipcard.json"), model_dir=model,
                      required=["native_export.graph"])
    assert any("missing producer graph_observation linkage" in problem for problem in problems), problems


@pytest.mark.parametrize("missing", ["receipt", "compilation", "config", "resolved", "operator", "ranks", "image", "source"])
def test_graph_native_missing_observations_refuse(native_graph, monkeypatch, missing):
    native, args, model, config, ranks, calls = native_graph
    if missing == "receipt":
        args.graph_receipt = None
    elif missing == "compilation":
        args.compilation_config = None
    elif missing == "config":
        del config.scheduler_config.max_num_seqs
    elif missing == "resolved":
        config.compilation_config = None
    elif missing == "operator":
        del config.kernel_config.ir_op_priority
    elif missing == "ranks":
        ranks.clear()
    else:
        monkeypatch.setattr(native, "_graph_image" if missing == "image"
                            else "_graph_tessera_source_sha256", lambda: None)
    result = native._run_arm(args, model, None, enforce_eager=False)
    assert not result["passed"], result


def test_graph_native_rank_transport_disagreement_refuses(native_graph):
    native, args, model, config, ranks, calls = native_graph
    config.parallel_config.tensor_parallel_size = 2
    ranks[0].update(tensor_parallel_size=2, fabric="socket")
    ranks.append({**ranks[0], "rank": 1, "fabric": "roce"})
    result = native._run_arm(args, model, None, enforce_eager=False)
    assert not result["passed"], result
    assert "fabric" in result["detail"]


def test_graph_native_changed_config_bytes_refuse(native_graph, monkeypatch):
    native, args, model, config, ranks, calls = native_graph
    original = sys.modules["vllm"].LLM.generate

    def generate(self, *arguments):
        (model / "config.json").write_text('{"model_type":"changed"}')
        return original(self, *arguments)

    monkeypatch.setattr(sys.modules["vllm"].LLM, "generate", generate)
    result = native._run_arm(args, model, None, enforce_eager=False)
    assert not result["passed"], result


def test_eager_native_needs_no_graph_receipt_or_observations(native_graph):
    native, args, model, config, ranks, calls = native_graph
    args.graph_receipt = None
    args.compilation_config = None
    config.compilation_config = None
    result = native._run_arm(args, model, None, enforce_eager=True)
    assert result["passed"], result
    assert "compilation_config" not in calls[0]
    assert "serve_scope" not in result["metrics"]


def test_graph_cli_requires_receipt_and_compilation_before_preflight(monkeypatch):
    from prismaquant.validate_native_export import main

    monkeypatch.setattr(sys, "argv", ["validate", "--model", "absent", "--both-arms"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize("nccl_disable", [None, "0", "1"])
def test_native_rank_observer_never_infers_tp2_transport_from_environment(monkeypatch, nccl_disable):
    from prismaquant.validate_native_export import _graph_rank_observation

    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: types.SimpleNamespace(world_size=2, rank_in_group=0)
    package = types.ModuleType("vllm")
    package.__path__ = []
    package.distributed = distributed
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    if nccl_disable is None:
        monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    else:
        monkeypatch.setenv("NCCL_IB_DISABLE", nccl_disable)
    with pytest.raises(ValueError, match="NCCL environment request is not transport evidence"):
        _graph_rank_observation(None, model_dir="unused")


def test_native_resolved_compilation_preserves_boolean_numeric_distinction(native_graph):
    native, args, model, config, ranks, calls = native_graph
    args.compilation_config = '{"mode":true}'
    config.compilation_config.mode = 1
    result = native._run_arm(args, model, None, enforce_eager=False)
    assert not result["passed"], result
    assert "resolved compilation_config" in result["detail"]


def test_graph_image_observes_container_manifest_instead_of_environment(monkeypatch):
    from prismaquant import validate_native_export as native
    import json

    container = "a" * 64
    image = "registry/native@sha256:" + "b" * 64
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        "0::/docker/" + container if str(path) == "/proc/self/cgroup"
                        else "" if str(path) == "/proc/self/mountinfo"
                        else original(path, *args, **kwargs))
    monkeypatch.setenv("TESSERA_RUNTIME_IMAGE", "unobserved caller image")
    commands = []

    def inspect(command, **kwargs):
        commands.append(command)
        if command[1] == "container":
            return json.dumps([{"Id": container, "Image": "sha256:" + "c" * 64,
                                "State": {"Running": True}, "Config": {"Image": image}}])
        return json.dumps([{"Id": "sha256:" + "c" * 64, "RepoDigests": [image]}])

    monkeypatch.setattr(native.subprocess, "check_output", inspect)
    assert native._graph_image() == image
    assert commands[0][-1] == container
