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


# --- the fabric a multi-node number crossed ---------------------------------

@pytest.mark.parametrize("environment,transport", [
    ({"NCCL_IB_DISABLE": "1"}, "sockets"),
    ({"NCCL_IB_DISABLE": "0"}, "ib_or_roce"),
    ({}, "unset_nccl_default"),
    ({"NCCL_IB_DISABLE": "maybe"}, "declared_other"),
])
def test_fabric_request_derives_transport_only_from_the_documented_values(
    environment, transport,
):
    """Two of NCCL_IB_DISABLE's values have a documented meaning. Anything else
    is reported verbatim rather than mapped onto one of them by guess."""
    from tools.gold_engine_options import gold_fabric_request

    fabric = gold_fabric_request(environment)
    assert fabric["transport"] == transport
    assert fabric["values"] == environment


def test_fabric_request_is_labelled_a_request_and_not_an_observation():
    """Principle 14: the environment is what the operator ASKED for. What NCCL
    then did is its own `NET/Socket`/`NET/IB` line, which this does not read."""
    from tools.gold_engine_options import gold_fabric_request

    fabric = gold_fabric_request({"NCCL_IB_DISABLE": "1"})
    assert fabric["observation"] == "requested_not_observed"
    assert "NET/Socket" in fabric["observed_by"]


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_manifest_records_the_fabric_so_two_fabrics_are_distinguishable(
    monkeypatch, runner,
):
    """A TP2 KL on sockets and the same measurement on RoCE must not read as
    one number: the fabric travels with the receipt."""
    module = importlib.import_module(runner)
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    # `_ENGINE_KWARGS` is a module global another test in this worker may have
    # filled; pin it so this test reads the same way in any order.
    monkeypatch.setattr(module, "_ENGINE_KWARGS", None)
    monkeypatch.setattr(module, "gold_producer_identity",
                        lambda name: {"git_commit": "a" * 40})
    monkeypatch.setattr(module, "_resolve_serve_image",
                        lambda args: "image@sha256:" + "b" * 64)
    monkeypatch.setattr(module, "self_manifest",
                        lambda **kwargs: {"serve_fingerprint": "c" * 64,
                                          **kwargs["extra"]})
    fabric = module._provenance(_args())["serve_manifest"]["gold_fabric_request"]
    assert fabric["transport"] == "sockets"
    assert fabric["values"]["NCCL_IB_DISABLE"] == "1"


def test_the_fabric_names_ride_the_performance_stack_fingerprint():
    """The refusal that matters: `tools/kl_ab.py` keys cross-arm comparability
    on the performance-stack fingerprint, so two arms that crossed different
    fabrics must not hash alike."""
    from tools.gold_engine_options import NCCL_FABRIC_ENV
    from tools.serve_fingerprint import (
        SERVER_ENV_ALLOWLIST,
        performance_stack_payload,
    )

    for name in NCCL_FABRIC_ENV:
        assert name in SERVER_ENV_ALLOWLIST

    def manifest(values):
        return {"server_process_environment": {"values": values}}

    sockets = performance_stack_payload(manifest({"NCCL_IB_DISABLE": "1"}))
    roce = performance_stack_payload(manifest({"NCCL_IB_DISABLE": "0"}))
    assert sockets != roce


# --- the peer argv this coordinator's own kwargs imply -----------------------

def test_peer_argv_is_derived_from_the_coordinators_own_engine_kwargs():
    """Not retyped: a peer that joins with different engine arguments than the
    coordinator built is a measurement of neither configuration."""
    from tools.gold_engine_options import headless_peer_argv

    argv = headless_peer_argv(
        {"model": "/models/stub", "tensor_parallel_size": 2, "nnodes": 2,
         "master_addr": "10.100.96.1", "master_port": 29531,
         "distributed_executor_backend": "mp", "data_parallel_backend": "mp",
         "trust_remote_code": True, "enforce_eager": True,
         "max_model_len": 513, "node_rank": 0},
        node_rank=1)
    assert argv[:5] == ["serve", "/models/stub", "--node-rank", "1", "--headless"]
    assert "--tensor-parallel-size" in argv and "2" in argv
    assert "--master-addr" in argv and "10.100.96.1" in argv
    assert "--enforce-eager" in argv and "--trust-remote-code" in argv
    # rank 0's own rank is never forwarded, and the model is the positional.
    assert "--node-rank" == argv[2] and argv.count("--node-rank") == 1
    assert "--model" not in argv


def test_peer_argv_refuses_an_engine_kwarg_it_cannot_spell():
    """Dropping an unknown kwarg silently is how the two ranks come to run
    different engines; the refusal names the kwarg instead."""
    from tools.gold_engine_options import headless_peer_argv

    with pytest.raises(ValueError, match="invented_knob"):
        headless_peer_argv(
            {"model": "/models/stub", "invented_knob": 7}, node_rank=1)


@pytest.mark.parametrize("node_rank", [0, -1, True, 1.0])
def test_peer_argv_refuses_a_rank_that_is_not_a_peer(node_rank):
    from tools.gold_engine_options import headless_peer_argv

    with pytest.raises(ValueError):
        headless_peer_argv({"model": "/models/stub"}, node_rank=node_rank)


def test_a_boolean_engine_kwarg_that_is_false_is_stated_not_omitted():
    """`enable_prefix_caching=False` must reach the peer as its negated flag:
    leaving it out serves the peer's default, which is the other value."""
    from tools.gold_engine_options import headless_peer_argv

    argv = headless_peer_argv(
        {"model": "/m", "enable_prefix_caching": False,
         "enable_chunked_prefill": False}, node_rank=1)
    assert "--no-enable-prefix-caching" in argv
    assert "--no-enable-chunked-prefill" in argv


@pytest.mark.parametrize("runner", RUNNERS)
def test_a_single_node_receipt_carries_no_peer_argv(monkeypatch, runner):
    """TP1 is unchanged: there is no peer, so the receipt claims none."""
    module = importlib.import_module(runner)
    monkeypatch.setattr(module, "gold_producer_identity",
                        lambda name: {"git_commit": "a" * 40})
    monkeypatch.setattr(module, "_resolve_serve_image",
                        lambda args: "image@sha256:" + "b" * 64)
    monkeypatch.setattr(module, "self_manifest",
                        lambda **kwargs: {"serve_fingerprint": "c" * 64,
                                          **kwargs["extra"]})
    monkeypatch.setattr(module, "_ENGINE_KWARGS", None)
    args = _args(tensor_parallel_size=1, nnodes=None, master_addr=None,
                 master_port=None, distributed_executor_backend=None,
                 data_parallel_backend=None, moe_backend=None)
    manifest = module._provenance(args)["serve_manifest"]
    assert "headless_peer_argv" not in manifest


@pytest.mark.parametrize("runner", RUNNERS)
def test_a_multi_node_receipt_states_the_peer_argv_its_own_engine_implies(
    monkeypatch, runner,
):
    """The end-to-end shape of the peer contract: the argv stamped on the
    receipt is derived from the kwargs THIS engine was built with, so the
    launcher that started rank 1 can be checked against it."""
    from tools.gold_engine_options import headless_peer_argv

    module = importlib.import_module(runner)
    seen = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=lambda **kwargs: seen.update(kwargs)))
    monkeypatch.setattr(module, "refuse_if_spec_decode", lambda **kwargs: False)
    args = _args()
    if runner.endswith("full_kl"):
        module._load_llm(args, max_model_len=513)
    else:
        module._load_llm(args)

    monkeypatch.setattr(module, "gold_producer_identity",
                        lambda name: {"git_commit": "a" * 40})
    monkeypatch.setattr(module, "_resolve_serve_image",
                        lambda args: "image@sha256:" + "b" * 64)
    monkeypatch.setattr(module, "self_manifest",
                        lambda **kwargs: {"serve_fingerprint": "c" * 64,
                                          **kwargs["extra"]})
    manifest = module._provenance(args)["serve_manifest"]
    assert manifest["headless_peer_argv"] == headless_peer_argv(
        seen, node_rank=1)
    assert manifest["headless_peer_argv"][:5] == [
        "serve", "artifact", "--node-rank", "1", "--headless"]
