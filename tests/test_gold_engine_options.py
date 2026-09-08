"""Gold runners must pass the declared stock topology to their actual LLM."""
import importlib
import sys
import types

import pytest


RUNNERS = ("tools.measure_vllm_full_kl", "tools.measure_vllm_wikitext_ppl")


def _args(**overrides):
    fields = dict(model="artifact", dtype="bfloat16", gpu_memory_utilization=0.5,
                  seqlen=512, max_logprobs=100, enforce_eager=True,
                  quantization=None, max_num_batched_tokens=1024,
                  tensor_parallel_size=2, nnodes=2, node_rank=0,
                  master_addr="192.0.2.1", master_port=29501,
                  distributed_executor_backend="mp", data_parallel_backend="mp",
                  moe_backend="triton")
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize("runner", RUNNERS)
def test_llm_receives_two_node_topology(monkeypatch, runner):
    module = importlib.import_module(runner)
    seen = {}
    sentinel = object()
    def llm(**kwargs):
        seen.update(kwargs)
        return sentinel
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=llm))
    monkeypatch.setattr(module, "refuse_if_spec_decode", lambda **kwargs: False)
    args = _args()
    result = (module._load_llm(args, max_model_len=513) if runner.endswith("full_kl")
              else module._load_llm(args))
    assert result is sentinel
    assert {key: seen.get(key) for key in (
        "tensor_parallel_size", "nnodes", "node_rank", "master_addr", "master_port",
        "distributed_executor_backend", "data_parallel_backend", "moe_backend")} == {
        "tensor_parallel_size": 2, "nnodes": 2, "node_rank": 0,
        "master_addr": "192.0.2.1", "master_port": 29501,
        "distributed_executor_backend": "mp", "data_parallel_backend": "mp",
        "moe_backend": "triton"}
    assert seen["enforce_eager"] is True
    assert seen["max_num_batched_tokens"] == 1024


@pytest.mark.parametrize("runner", RUNNERS)
def test_public_cli_accepts_declared_two_node_topology(monkeypatch, runner):
    module = importlib.import_module(runner)
    class ReachedImageValidation(Exception):
        pass
    def image(args):
        assert args.tensor_parallel_size == args.nnodes == 2
        raise ReachedImageValidation
    monkeypatch.setattr(module, "_resolve_serve_image", image)
    argv = [runner, "--model", "artifact", "--output", "result.json",
            "--tensor-parallel-size", "2", "--nnodes", "2", "--node-rank", "0",
            "--master-addr", "192.0.2.1", "--master-port", "29501",
            "--distributed-executor-backend", "mp", "--data-parallel-backend", "mp",
            "--moe-backend", "triton"]
    if runner.endswith("full_kl"):
        argv += ["--mode", "teacher"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedImageValidation):
        module.main()


@pytest.mark.parametrize("runner", RUNNERS)
def test_default_llm_options_preserve_single_node_behavior(monkeypatch, runner):
    module = importlib.import_module(runner)
    args = _args()
    for field in ("tensor_parallel_size", "nnodes", "node_rank", "master_addr", "master_port",
                  "distributed_executor_backend", "data_parallel_backend", "moe_backend"):
        delattr(args, field)
    seen = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: seen.update(kw)))
    monkeypatch.setattr(module, "refuse_if_spec_decode", lambda **kw: False)
    if runner.endswith("full_kl"):
        module._load_llm(args, max_model_len=513)
    else:
        module._load_llm(args)
    assert seen == {"model": "artifact", "trust_remote_code": True, "dtype": "bfloat16",
                    "tensor_parallel_size": 1, "gpu_memory_utilization": 0.5,
                    "max_model_len": 513, "max_num_seqs": 1,
                    "enforce_eager": True, "disable_log_stats": True,
                    "max_num_batched_tokens": 1024,
                    **({"max_logprobs": 100} if runner.endswith("full_kl") else {})}


@pytest.mark.parametrize("fields", [
    {"tensor_parallel_size": 0}, {"tensor_parallel_size": True},
    {"tensor_parallel_size": 2.0}, {"nnodes": 0}, {"nnodes": 3},
    {"node_rank": 1}, {"node_rank": False},
    {"master_addr": ""}, {"master_addr": None},
    {"master_port": 0}, {"master_port": 65536}, {"master_port": True},
    {"master_port": None}, {"distributed_executor_backend": None},
    {"data_parallel_backend": None}, {"distributed_executor_backend": "ray"},
    {"moe_backend": "unknown"},
])
def test_incomplete_or_incompatible_topology_refuses_before_engine(fields):
    from tools.gold_engine_options import gold_engine_kwargs
    with pytest.raises(ValueError):
        gold_engine_kwargs(_args(**fields))


@pytest.mark.parametrize("runner", RUNNERS)
def test_public_cli_refuses_worker_rank_before_model_access(monkeypatch, runner):
    module = importlib.import_module(runner)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid topology reached model/image work")
    monkeypatch.setattr(module, "_resolve_serve_image", forbidden)
    argv = [runner, "--model", "artifact", "--output", "result.json", "--node-rank", "1"]
    if runner.endswith("full_kl"):
        argv += ["--mode", "teacher"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("runner", RUNNERS)
def test_measurement_manifest_binds_topology_and_helper_source(monkeypatch, runner):
    module = importlib.import_module(runner)
    from tools.serve_fingerprint import _GOLD_PRODUCER_TOOL_FILES
    tool = runner.split(".")[-1]
    assert "tools/gold_engine_options.py" in _GOLD_PRODUCER_TOOL_FILES[tool]
    monkeypatch.setattr(module, "gold_producer_identity", lambda name: {"git_commit": "a" * 40})
    monkeypatch.setattr(module, "_resolve_serve_image", lambda args: "image@sha256:" + "b" * 64)
    def manifest(**kwargs):
        return {"serve_fingerprint": "c" * 64, **kwargs["extra"]}
    monkeypatch.setattr(module, "self_manifest", manifest)
    result = module._provenance(_args())
    assert result["serve_manifest"]["gold_engine_configuration"]["tensor_parallel_size"] == 2
    assert result["serve_manifest"]["gold_engine_configuration"]["nnodes"] == 2
