import json
import sys
import types

import pytest

from prismaquant.validate_native_export import (
    _flashinfer_runtime_package,
    _resolve_validation_target_profile,
    _speculative_config_uses_embedded_mtp,
    maybe_upgrade_flashinfer,
    require_native_lane_artifact,
)


def test_speculative_config_embedded_mtp_detection():
    assert _speculative_config_uses_embedded_mtp({"method": "mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_5_mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_next_mtp"})

    assert not _speculative_config_uses_embedded_mtp({"method": "ngram"})
    assert not _speculative_config_uses_embedded_mtp({"method": "draft_model"})
    assert not _speculative_config_uses_embedded_mtp({})


def test_flashinfer_runtime_package_comes_from_serving_profile():
    version, packages, env = _flashinfer_runtime_package("vllm_packed_moe")

    assert version == "0.6.8.post1"
    assert packages == ("flashinfer-python", "flashinfer-cubin")
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a stub `flashinfer` module and record any pip call."""
    calls = []
    monkeypatch.setattr(
        "prismaquant.validate_native_export.subprocess.check_call",
        lambda cmd, *a, **k: calls.append(cmd),
    )

    def install(version):
        mod = types.ModuleType("flashinfer")
        mod.__version__ = version
        monkeypatch.setitem(sys.modules, "flashinfer", mod)
        return calls

    return install


@pytest.mark.parametrize(
    "installed, pinned, should_install",
    [
        # The bug: the serving container ships NEWER than the profile pin, and
        # `== version` made that look wrong. It downgraded a container that had
        # just served the artifact cleanly and vLLM 0.26 died on the missing
        # `set_autotune_process_group`. Measured 2026-08-14, Qwen3.8-27B gate.
        ("0.6.18", "0.6.8.post1", False),
        # Same version, both spellings.
        ("0.6.8.post1", "0.6.8.post1", False),
        ("0.6.8", "0.6.8.post1", False),
        # A post-release must not read as older than its own base version.
        ("0.6.9.post2", "0.6.9", False),
        # Genuinely too old: the pin's original purpose (an image that cannot
        # dispatch the NVFP4 MoE backend on Blackwell) still upgrades.
        ("0.6.7", "0.6.8.post1", True),
        ("0.5.20", "0.6.0", True),
    ],
)
def test_flashinfer_pin_is_a_floor_and_never_downgrades(
    fake_flashinfer, installed, pinned, should_install
):
    calls = fake_flashinfer(installed)
    maybe_upgrade_flashinfer(pinned)

    assert bool(calls) is should_install, (
        f"installed={installed} pinned={pinned}: "
        f"{'expected an upgrade' if should_install else 'must not touch it'}"
    )
    if should_install:
        assert any(f"flashinfer-python=={pinned}" in part for part in calls[0])


def test_flashinfer_absent_still_installs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "prismaquant.validate_native_export.subprocess.check_call",
        lambda cmd, *a, **k: calls.append(cmd),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", None)  # import -> ImportError

    maybe_upgrade_flashinfer("0.6.8.post1")

    assert calls, "no flashinfer at all must still install the pinned version"


def test_validation_target_profile_defaults_from_model_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}'
    )

    assert _resolve_validation_target_profile(tmp_path, None) == "vllm_packed_moe"
    assert _resolve_validation_target_profile(tmp_path, "research") == "research"


# ---------------------------------------------------------------------------
# The gate's lane (RobTand/prismaquant#119 half 1)
# ---------------------------------------------------------------------------
def _write_config(model_dir, quant_method=None, nested=False):
    cfg = {"model_type": "qwen3"}
    if quant_method is not None:
        qc = {"quant_method": quant_method, "config_groups": {}}
        if nested:
            cfg["text_config"] = {"quantization_config": qc}
        else:
            cfg["quantization_config"] = qc
    (model_dir / "config.json").write_text(json.dumps(cfg))


def test_a_tessera_artifact_is_refused_before_vllm_loads(tmp_path):
    """`validate_native_export` is the NATIVE lane's load gate: it builds
    `LLM(..., quantization="compressed-tensors")`, so pointing it at a
    Tessera artifact must refuse with the lane named -- not die inside vLLM
    on bytes it was never going to dispatch."""
    _write_config(tmp_path, "tessera")
    with pytest.raises(SystemExit, match="tessera"):
        require_native_lane_artifact(tmp_path)


def test_a_nested_tessera_quant_config_is_refused_too(tmp_path):
    """Multimodal checkpoints nest `quantization_config` under `text_config`;
    the lane check must read where the loader reads."""
    _write_config(tmp_path, "tessera", nested=True)
    with pytest.raises(SystemExit, match="tessera"):
        require_native_lane_artifact(tmp_path)


def test_a_compressed_tensors_artifact_passes_the_lane_check(tmp_path):
    _write_config(tmp_path, "compressed-tensors")
    assert require_native_lane_artifact(tmp_path) == "compressed-tensors"


def test_an_unquantized_checkpoint_keeps_its_legacy_behavior(tmp_path):
    """No declared `quant_method` is not a foreign lane: refusing here would
    change what a dense smoke does today, which is not this issue's bargain."""
    _write_config(tmp_path, None)
    assert require_native_lane_artifact(tmp_path) is None


def _graph_arm_fixture(tmp_path, monkeypatch):
    from prismaquant import validate_native_export as owner
    from test_shipcard import _graph_receipt_metrics

    (tmp_path / "config.json").write_bytes(b'{"model_type":"glm5next"}')
    metrics = _graph_receipt_metrics(tmp_path)
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model=str(tmp_path), max_model_len=8448),
        scheduler_config=types.SimpleNamespace(max_num_seqs=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=2),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=1),
    )
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            llm_engine=types.SimpleNamespace(vllm_config=config),
            generate=lambda *a: [types.SimpleNamespace(
                prompt="test", outputs=[types.SimpleNamespace(text="ok")])],
        )

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=load, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(owner, "_graph_image", lambda: metrics["serve_scope"]["image"],
                        raising=False)
    monkeypatch.setattr(owner, "_graph_tessera_source_sha256",
                        lambda: metrics["serve_scope"]["tessera_src_sha256"],
                        raising=False)
    args = types.SimpleNamespace(
        graph_receipt=metrics["graph_receipt_path"], compilation_config='{"mode":"NONE"}',
        max_model_len=8448, max_num_seqs=4, tensor_parallel_size=2,
        gpu_memory_utilization=0.5, max_new_tokens=2, prompt="test",
        both_arms=False, shipcard=None, route_sweep_out=None,
    )
    return owner, args, config, calls, metrics


def test_graph_arm_stamps_derived_scope_and_receipt(tmp_path, monkeypatch):
    import hashlib

    owner, args, config, calls, expected = _graph_arm_fixture(tmp_path, monkeypatch)
    # Requested settings are not authority for the resolved engine's settings.
    args.max_model_len = 9000
    args.max_num_seqs = 8
    args.tensor_parallel_size = 1
    result = owner._run_arm(args, tmp_path, {"num_speculative_tokens": 3},
                            enforce_eager=False)
    assert result["passed"], result
    metrics = result["metrics"]
    expected["serve_scope"]["model_config_sha256"] = hashlib.sha256(
        (tmp_path / "config.json").read_bytes()).hexdigest()
    assert metrics["serve_scope"] == expected["serve_scope"]
    assert metrics["graph_receipt_path"] == expected["graph_receipt_path"]
    assert metrics["graph_receipt_sha256"] == expected["graph_receipt_sha256"]
    assert calls[0]["compilation_config"] == {"mode": "NONE"}
    assert calls[0]["max_num_seqs"] == 8
    assert calls[0]["tensor_parallel_size"] == 1


@pytest.mark.parametrize("parent,field", [
    ("model_config", "max_model_len"), ("scheduler_config", "max_num_seqs"),
    ("parallel_config", "tensor_parallel_size"),
    ("speculative_config", "num_speculative_tokens"),
])
def test_graph_arm_refuses_underived_scope(tmp_path, monkeypatch, parent, field):
    owner, args, config, calls, expected = _graph_arm_fixture(tmp_path, monkeypatch)
    delattr(getattr(config, parent), field)
    result = owner._run_arm(args, tmp_path, None, enforce_eager=False)
    assert not result["passed"], result
    name = "speculative_tokens" if field == "num_speculative_tokens" else field
    assert name in result["detail"]


def test_graph_arm_requires_receipt_before_load(tmp_path, monkeypatch):
    owner, args, config, calls, expected = _graph_arm_fixture(tmp_path, monkeypatch)
    args.graph_receipt = None
    result = owner._run_arm(args, tmp_path, None, enforce_eager=False)
    assert not result["passed"], result
    assert "graph_receipt_path" in result["detail"]
    assert calls == []


def test_graph_arm_no_speculation_is_derived_as_zero(tmp_path, monkeypatch):
    owner, args, config, calls, expected = _graph_arm_fixture(tmp_path, monkeypatch)
    config.speculative_config = None
    result = owner._run_arm(args, tmp_path, None, enforce_eager=False)
    assert result["passed"], result
    assert result["metrics"]["serve_scope"]["speculative_tokens"] == 0


def test_eager_arm_needs_no_graph_scope(tmp_path, monkeypatch):
    owner, args, config, calls, expected = _graph_arm_fixture(tmp_path, monkeypatch)
    args.graph_receipt = None
    del config.scheduler_config.max_num_seqs
    result = owner._run_arm(args, tmp_path, None, enforce_eager=True)
    assert result["passed"], result
    assert "serve_scope" not in result["metrics"]
    assert "compilation_config" not in calls[0]


def test_graph_cli_requires_receipt_before_preflight(monkeypatch):
    from prismaquant.validate_native_export import main

    monkeypatch.setattr(sys, "argv", ["validate", "--model", "absent",
                                     "--both-arms"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_graph_image_uses_kernel_container_and_daemon_digest(tmp_path, monkeypatch):
    from pathlib import Path
    from prismaquant import validate_native_export as owner

    container = "a" * 64
    image = "registry/serve@sha256:" + "b" * 64
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw:
                        "0::/docker/" + container if str(p) == "/proc/self/cgroup"
                        else "" if str(p) == "/proc/self/mountinfo"
                        else original(p, *a, **kw))
    monkeypatch.setenv("TESSERA_RUNTIME_IMAGE", "caller supplied image")
    commands = []

    def inspect(command, **kw):
        commands.append(command)
        if command[1] == "container":
            return json.dumps([{"Id": container, "Image": "sha256:" + "c" * 64,
                                "Config": {"Image": image},
                                "State": {"Running": True}}])
        return json.dumps([{"Id": "sha256:" + "c" * 64, "RepoDigests": [image]}])

    monkeypatch.setattr(owner.subprocess, "check_output", inspect)
    assert owner._graph_image() == image
    assert commands[0][-1] == container


def test_graph_image_refuses_when_container_is_not_observed(monkeypatch):
    from pathlib import Path
    from prismaquant import validate_native_export as owner

    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: "0::/host")
    with pytest.raises(ValueError, match="image"):
        owner._graph_image()


def test_graph_source_digest_matches_tessera_receipt_recipe(monkeypatch):
    import hashlib
    import subprocess
    from pathlib import Path
    from tessera import graph_receipt
    from prismaquant import validate_native_export as owner
    from prismaquant import tessera_serving_runtime_pin as pin

    # This is a digest unit test, not a serving-pin qualification.
    monkeypatch.setattr(pin, "require_exact_tessera_runtime_pin", lambda *a, **kw: None)
    root = Path(graph_receipt.__file__).resolve().parents[2]
    expected = subprocess.check_output([
        "bash", "-c", "find src -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum",
    ], cwd=root)
    assert owner._graph_tessera_source_sha256() == hashlib.sha256(expected).hexdigest()

