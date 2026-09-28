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
                  moe_backend="triton", kv_cache_memory_bytes=None)
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


def test_the_explicit_kv_bound_and_cutlass_are_shared_engine_options():
    """A4's serve flags must travel through the shared options rather than
    being retyped at a call site: the explicit positive KV byte bound and the
    widened backend menu are selections this tool validates and records. Which
    backend name a given image maps for its NVFP4 MoE oracle is that image's
    fact, not this repository's, so no requirement is asserted here."""
    from tools.gold_engine_options import gold_engine_kwargs

    result = gold_engine_kwargs(_args(kv_cache_memory_bytes=4294967296,
                                      moe_backend="flashinfer_cutlass"))
    assert result["kv_cache_memory_bytes"] == 4294967296
    assert result["moe_backend"] == "flashinfer_cutlass"


def test_an_omitted_kv_bound_is_absent_and_not_none():
    """Omission preserves the original kwargs: vLLM sizes KV from
    `gpu_memory_utilization` and must not be handed a `None` bound."""
    from tools.gold_engine_options import gold_engine_kwargs

    omitted = gold_engine_kwargs(_args())
    assert "kv_cache_memory_bytes" not in omitted
    assert omitted["moe_backend"] == "triton"


@pytest.mark.parametrize("runner", RUNNERS)
def test_explicit_kv_bound_and_cutlass_reach_the_engine_and_the_peer(monkeypatch, runner):
    from tools.gold_engine_options import headless_peer_argv

    module = importlib.import_module(runner)
    seen = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: seen.update(kw)))
    monkeypatch.setattr(module, "refuse_if_spec_decode", lambda **kw: False)
    args = _args(kv_cache_memory_bytes=4294967296, moe_backend="flashinfer_cutlass")
    if runner.endswith("full_kl"):
        module._load_llm(args, max_model_len=513)
    else:
        module._load_llm(args)
    assert seen["kv_cache_memory_bytes"] == 4294967296
    assert seen["moe_backend"] == "flashinfer_cutlass"
    peer = headless_peer_argv(seen, node_rank=1)
    assert peer[peer.index("--kv-cache-memory-bytes") + 1] == "4294967296"
    assert peer[peer.index("--moe-backend") + 1] == "flashinfer_cutlass"


@pytest.mark.parametrize("runner", RUNNERS)
def test_public_cli_accepts_the_explicit_kv_bound_and_cutlass(monkeypatch, runner):
    module = importlib.import_module(runner)
    class ReachedImageValidation(Exception):
        pass
    def image(args):
        assert args.kv_cache_memory_bytes == 4294967296
        assert args.moe_backend == "flashinfer_cutlass"
        raise ReachedImageValidation
    monkeypatch.setattr(module, "_resolve_serve_image", image)
    argv = [runner, "--model", "artifact", "--output", "result.json",
            "--moe-backend", "flashinfer_cutlass",
            "--kv-cache-memory-bytes", "4294967296"]
    if runner.endswith("full_kl"):
        argv += ["--mode", "teacher"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedImageValidation):
        module.main()


@pytest.mark.parametrize("runner", RUNNERS)
def test_public_cli_refuses_a_nonpositive_kv_bound(monkeypatch, runner):
    module = importlib.import_module(runner)
    def forbidden(*args, **kwargs):
        raise AssertionError("an invalid KV bound reached model/image work")
    monkeypatch.setattr(module, "_resolve_serve_image", forbidden)
    argv = [runner, "--model", "artifact", "--output", "result.json",
            "--kv-cache-memory-bytes", "0"]
    if runner.endswith("full_kl"):
        argv += ["--mode", "teacher"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


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
    {"moe_backend": "unknown"}, {"moe_backend": "cutlass"},
    {"kv_cache_memory_bytes": 0}, {"kv_cache_memory_bytes": -1},
    {"kv_cache_memory_bytes": True}, {"kv_cache_memory_bytes": 4096.0},
    {"kv_cache_memory_bytes": "4096"},
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


def test_peer_argv_spells_the_kv_bound_and_states_its_omission():
    """The peer must carry the same explicit KV byte bound as rank 0, and a
    run that declared none must not invent one for the peer."""
    from tools.gold_engine_options import headless_peer_argv

    argv = headless_peer_argv(
        {"model": "/m", "kv_cache_memory_bytes": 4294967296,
         "moe_backend": "flashinfer_cutlass"}, node_rank=1)
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == "4294967296"
    assert argv[argv.index("--moe-backend") + 1] == "flashinfer_cutlass"
    assert "--kv-cache-memory-bytes" not in headless_peer_argv(
        {"model": "/m"}, node_rank=1)


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


def _glm_tr3_scorer_kwargs():
    """The engine kwargs the GLM TR3 scorer builds for a TP2 NoPE run (#1473)."""
    return {"model": "/models/glm", "trust_remote_code": True, "dtype": "bfloat16",
            "language_model_only": True, "kv_cache_dtype": "fp8_ds_mla",
            "enforce_eager": True, "enable_prefix_caching": False, "enable_chunked_prefill": False,
            "max_model_len": 2049, "max_num_batched_tokens": 2049, "max_num_seqs": 1,
            "max_logprobs": 1, "disable_log_stats": True, "logprobs_mode": "raw_logprobs",
            "gpu_memory_utilization": 0.5, "tensor_parallel_size": 2, "nnodes": 2,
            "node_rank": 0, "master_addr": "192.0.2.1", "master_port": 29531,
            "distributed_executor_backend": "mp", "data_parallel_backend": "mp",
            "moe_backend": "triton", "kv_cache_memory_bytes": 1073741824,
            "attention_backend": "CUSTOM",
            "kernel_config": {"enable_flashinfer_autotune": False}}


def test_peer_argv_spells_the_glm_attention_backend_and_kernel_config():
    """PQ #1473: both NoPE selections reach the peer with stock spellings."""
    from tools.gold_engine_options import headless_peer_argv

    argv = headless_peer_argv(_glm_tr3_scorer_kwargs(), node_rank=1)
    assert argv[argv.index("--attention-backend") + 1] == "CUSTOM"
    assert argv[argv.index("--kernel-config") + 1] == '{"enable_flashinfer_autotune":false}'


def test_peer_argv_round_trips_to_the_coordinators_engine_kwargs():
    """Parsing the peer argv back yields exactly the coordinator's kwargs
    minus rank 0's own (model positional, node_rank)."""
    from tools.gold_engine_options import headless_peer_argv, parse_headless_peer_argv

    kwargs = _glm_tr3_scorer_kwargs()
    model, node_rank, back = parse_headless_peer_argv(headless_peer_argv(kwargs, node_rank=1))
    assert (model, node_rank) == (kwargs["model"], 1)
    want = {k: v for k, v in kwargs.items() if k not in ("model", "node_rank")}
    assert set(back) == set(want)
    for key, value in want.items():
        if isinstance(value, (bool, dict)):
            assert back[key] == value, key
        else:
            assert back[key] == str(value), key


def test_a_compiled_coordinator_states_eager_off_and_its_compilation_config():
    """PQ #1634: the peer builds the compiled engine rank 0 built rather than a
    default, and its argv parses back to the coordinator's kwargs."""
    from tools.gold_engine_options import headless_peer_argv, parse_headless_peer_argv

    compilation = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                   "cudagraph_capture_sizes": [1, 2, 3, 4]}
    kwargs = {**_glm_tr3_scorer_kwargs(), "enforce_eager": False, "compilation_config": compilation}
    argv = headless_peer_argv(kwargs, node_rank=1)
    assert "--no-enforce-eager" in argv and "--enforce-eager" not in argv
    assert argv[argv.index("--compilation-config") + 1] == (
        '{"cudagraph_capture_sizes":[1,2,3,4],"cudagraph_mode":"FULL_DECODE_ONLY","mode":"NONE"}')
    _, _, back = parse_headless_peer_argv(argv)
    assert back["enforce_eager"] is False and back["compilation_config"] == compilation


@pytest.mark.parametrize("runner", RUNNERS)
def test_a_gold_tools_default_engine_states_nothing_for_eager_off(monkeypatch, runner):
    """PQ #1634 review: only a coordinator that declares a compilation config
    states eager off. A gold tool's own default (no `--enforce-eager`, so
    `enforce_eager` False, and no compilation config) emits the peer argv it
    emitted before #1634, so the argv its receipt stamps and its fingerprint
    do not move."""
    from tools.gold_engine_options import headless_peer_argv

    module = importlib.import_module(runner)
    seen = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: seen.update(kw)))
    monkeypatch.setattr(module, "refuse_if_spec_decode", lambda **kw: False)
    args = _args(enforce_eager=False)
    if runner.endswith("full_kl"):
        module._load_llm(args, max_model_len=513)
    else:
        module._load_llm(args)
    assert seen["enforce_eager"] is False and "compilation_config" not in seen
    argv = headless_peer_argv(seen, node_rank=1)
    assert "--no-enforce-eager" not in argv and "--enforce-eager" not in argv
    assert argv == headless_peer_argv(
        {k: v for k, v in seen.items() if k != "enforce_eager"}, node_rank=1)


def test_the_peer_parser_refuses_eager_off_without_a_compilation_config():
    """The parser stays the exact inverse: headless_peer_argv never states
    `--no-enforce-eager` without `--compilation-config`, so an argv that does
    was not derived from a coordinator's kwargs."""
    from tools.gold_engine_options import headless_peer_argv, parse_headless_peer_argv

    argv = headless_peer_argv({**_glm_tr3_scorer_kwargs(), "enforce_eager": False}, node_rank=1)
    assert "--no-enforce-eager" not in argv
    with pytest.raises(ValueError, match="--no-enforce-eager without --compilation-config"):
        parse_headless_peer_argv(argv + ["--no-enforce-eager"])


def test_kernel_config_must_be_a_json_object():
    from tools.gold_engine_options import headless_peer_argv

    with pytest.raises(ValueError, match="kernel_config"):
        headless_peer_argv({"model": "/models/stub", "kernel_config": '{"a":1}'}, node_rank=1)


@pytest.mark.parametrize("argv", [
    ["serve", "/m", "--node-rank", "1", "--headless", "--invented-flag", "1"],
    ["serve", "/m", "--node-rank", "1", "--headless", "--dtype", "bf16", "--dtype", "bf16"],
    ["serve", "/m", "--node-rank", "1", "--headless", "--dtype"],
    ["serve", "/m", "--node-rank", "1"],
])
def test_parse_peer_argv_refuses_what_the_tables_cannot_spell(argv):
    from tools.gold_engine_options import parse_headless_peer_argv

    with pytest.raises(ValueError):
        parse_headless_peer_argv(argv)
