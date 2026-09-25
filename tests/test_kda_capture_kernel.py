"""The Stage B KDA capture kernel's contract (PQ #1199).

A launch names the kernel with ``PRISMAQUANT_STAGE_B_KDA_KERNEL``. The GLM
derivative contract declares it, ``CaptureKernelDispatch`` substitutes it for
the verified fallback for one block at a time, and the layer quantum admits
it for a KDA target layer only, runs every target-layer pass inside its
scope, stamps its identity into the probe arithmetic and records it in the
counters. Every step refuses rather than fall back to Torch.

These tests need no GPU. The kernel's numerics are
``tests/test_kda_chunk_kernel.py``'s.
"""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from prismaquant import glm_kda_capture_kernel as capture
from prismaquant import glm_source_derivative as derivative
from prismaquant.kernels import kda_chunk
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

EVIDENCE = (Path(__file__).resolve().parents[1]
            / "experiments/measurements/glm-derivative-contract-20260908")
MODELING = "transformers.models.glm5_next.modeling_glm5_next"
ONE_PASS = {"calls": 1, "gram_forward": 2, "gram_backward": 2}


def fallback(*args, **kwargs):
    """Stands in for the image's decorated Torch fallback."""
    raise AssertionError("the fixture fallback is never called")


@pytest.fixture
def modeling(monkeypatch):
    """A fake GLM modeling module whose dispatch global is the fixture fallback."""
    package = ModuleType("transformers.models.glm5_next")
    module = ModuleType(MODELING)
    module.chunk_kimi_delta_attention = fallback
    package.modeling_glm5_next = module
    monkeypatch.setitem(sys.modules, "transformers.models.glm5_next", package)
    monkeypatch.setitem(sys.modules, MODELING, module)
    return module


@pytest.fixture
def bound_model(tmp_path, monkeypatch, modeling):
    """A model bound to the GLM derivative, as ``test_glm_source_derivative`` binds one."""
    build = json.loads((EVIDENCE / "image-build-result.json").read_text())
    path = tmp_path / "build.json"
    path.write_text(json.dumps(build, sort_keys=True))
    import hashlib
    policy = dict(schema=derivative.SCHEMA, version=derivative.VERSION,
                  image_build=dict(path=str(path),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    current = dict(declaration=derivative.declaration(), gates={"unit": {"lower_bound": -5}},
                   image_content_sha256=derivative.CORRECTED_IMAGE_CONTENT_SHA256)
    monkeypatch.setattr(derivative, "_observe", lambda *_: copy.deepcopy(current))
    monkeypatch.setenv("PRISMAQUANT_CONTAINER_CONTENT_SHA256",
                       derivative.CORRECTED_IMAGE_CONTENT_SHA256)
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(_attn_implementation="eager")
    derivative.bind_source_derivative(model, Glm5NextProfile(), policy)
    return model


class KdaAttention(torch.nn.Module):
    """A module of the GLM KDA attention class, by module path and name."""


KdaAttention.__module__, KdaAttention.__name__ = capture.KDA_ATTENTION_CLASS
KdaAttention.__qualname__ = KdaAttention.__name__


def kda_layer():
    layer = torch.nn.Module()
    layer.self_attn = KdaAttention()
    return layer


def count_one_kernel_pass():
    for key, value in ONE_PASS.items():
        kda_chunk._COUNTS[key] += value


# ---- the launch setting ----------------------------------------------------

def test_the_launch_setting_names_a_known_kernel_or_nothing():
    assert capture.kda_capture_kernel_from_environment({}) is None
    assert capture.kda_capture_kernel_from_environment(
        {capture.KDA_KERNEL_ENV: "kda_gram_v1"}) == "kda_gram_v1"
    for text in ("", "kda_gram_fp32_v1", "torch"):
        with pytest.raises(capture.KdaCaptureKernelRefused, match="names no known capture kernel"):
            capture.kda_capture_kernel_from_environment({capture.KDA_KERNEL_ENV: text})


def test_every_launchable_kernel_is_declared_by_the_derivative_contract():
    for name in capture.KERNELS:
        declaration = derivative.capture_kernel_declaration(name)
        assert declaration["implements"] == derivative.VERSION
        assert declaration["replaces"] == "chunk_kimi_delta_attention"
        assert declaration["scope"] == "stage_b_target_layer_pass"
        assert (declaration["module"], declaration["entry"]) == (
            "prismaquant.kernels.kda_chunk", "chunk_kimi_delta_attention")
    assert kda_chunk.NAME in capture.KERNELS
    assert kda_chunk.IMPLEMENTS == derivative.VERSION
    with pytest.raises(ValueError, match="undeclared capture kernel"):
        derivative.capture_kernel_declaration("fla_chunk_kda")


# ---- the dispatch ----------------------------------------------------------

def test_dispatch_substitutes_the_kernel_for_one_block_and_restores_the_fallback(
        bound_model, modeling):
    dispatch = derivative.CaptureKernelDispatch(bound_model, "kda_gram_v1")
    assert modeling.chunk_kimi_delta_attention is fallback
    with dispatch:
        assert modeling.chunk_kimi_delta_attention is kda_chunk.chunk_kimi_delta_attention
        with pytest.raises(ValueError, match="do not nest"):
            dispatch.__enter__()
    assert modeling.chunk_kimi_delta_attention is fallback
    with dispatch:
        pass
    assert modeling.chunk_kimi_delta_attention is fallback


def test_dispatch_refuses_a_global_changed_outside_or_inside_its_block(bound_model, modeling):
    dispatch = derivative.CaptureKernelDispatch(bound_model, "kda_gram_v1")

    def other(*args, **kwargs):
        raise AssertionError("never called")

    modeling.chunk_kimi_delta_attention = other
    with pytest.raises(ValueError, match="changed outside a capture kernel block"):
        with dispatch:
            pass
    assert modeling.chunk_kimi_delta_attention is other
    modeling.chunk_kimi_delta_attention = fallback
    with pytest.raises(ValueError, match="changed inside a capture kernel block"):
        with dispatch:
            modeling.chunk_kimi_delta_attention = other
    # The fallback is restored whatever the block did.
    assert modeling.chunk_kimi_delta_attention is fallback


def test_dispatch_requires_the_bound_derivative(modeling):
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="requires the bound GLM derivative"):
        derivative.CaptureKernelDispatch(model, "kda_gram_v1")
    assert modeling.chunk_kimi_delta_attention is fallback


# ---- admission --------------------------------------------------------------

def test_admission_skips_a_target_without_kda_before_any_kernel_work(monkeypatch):
    monkeypatch.setattr(kda_chunk, "probe_digest", lambda device: pytest.fail("probed"))
    layer = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert capture.layer_runs_kda(layer) is False
    assert capture.layer_runs_kda(kda_layer()) is True
    assert capture.admit_kda_capture_kernel("kda_gram_v1", None, layer, device="cpu") is None
    with pytest.raises(capture.KdaCaptureKernelRefused, match="unknown KDA capture kernel"):
        capture.admit_kda_capture_kernel("other", None, layer, device="cpu")


def test_admission_refuses_a_band_serial_handoff(monkeypatch):
    monkeypatch.setattr(kda_chunk, "probe_digest", lambda device: pytest.fail("probed"))
    with pytest.raises(capture.KdaCaptureKernelRefused, match="band-serial producer"):
        capture.admit_kda_capture_kernel("kda_gram_v1", None, kda_layer(), device="cpu",
                                         emits_handoff=True)


def test_admission_refuses_an_unbound_model_and_a_kernel_that_cannot_run(
        bound_model, modeling, monkeypatch):
    with pytest.raises(capture.KdaCaptureKernelRefused, match="bound GLM derivative"):
        capture.admit_kda_capture_kernel("kda_gram_v1", torch.nn.Linear(2, 2), kda_layer(),
                                         device="cpu")

    def broken(device):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(kda_chunk, "probe_digest", broken)
    with pytest.raises(capture.KdaCaptureKernelRefused, match="probe failed.*no CUDA device"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(), device="cpu")
    monkeypatch.setattr(kda_chunk, "IMPLEMENTS", "glm_kda_causal_exp_v0")
    with pytest.raises(capture.KdaCaptureKernelRefused, match="not the declared kernel"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(), device="cpu")
    assert modeling.chunk_kimi_delta_attention is fallback


def _admitted(monkeypatch, tmp_path, *, qualified):
    """Admit the kernel with a stubbed probe and compile, against a chosen qualification."""
    compiled = {f"{kernel}_{'inclusive' if inclusive else 'strict'}":
                {"code_sha256": "c" * 64, "triton_hash": "t" * 64}
                for kernel, inclusive in kda_chunk.KERNEL_KEYS}
    probe = {"shape": [1, 128, 2, 128], "sha256": "p" * 64}
    runtime = {"torch": "fixture"}
    monkeypatch.setattr(kda_chunk, "probe_digest", lambda device: probe)
    monkeypatch.setattr(kda_chunk, "compiled_kernels", lambda: compiled)
    monkeypatch.setattr(capture, "_runtime_identity", lambda device: runtime)
    path = tmp_path / "kda_chunk_qualification.json"
    if qualified is None:
        path.unlink(missing_ok=True)
    else:
        identity = {"name": "kda_gram_v1", "source_sha256": kda_chunk.source_sha256(),
                    "compiled": compiled, "probe": probe, "runtime": runtime, **qualified}
        path.write_text(json.dumps({"schema": capture.QUALIFICATION_SCHEMA,
                                    "status": "qualified", "identity": identity}))
    monkeypatch.setattr(capture, "QUALIFICATION_PATH", path)
    return path


def test_admission_compares_the_runtime_with_the_packaged_qualification(
        bound_model, modeling, monkeypatch, tmp_path, capsys):
    import hashlib

    path = _admitted(monkeypatch, tmp_path, qualified={})
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(),
                                                device="cpu")
    assert admitted.qualification_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    # The arithmetic identity names what runs; the evidence digest stays out of it.
    assert "qualification_sha256" not in admitted.identity
    assert admitted.identity["declaration"] == derivative.capture_kernel_declaration("kda_gram_v1")
    assert admitted.record()["qualification_sha256"] == admitted.qualification_sha256
    assert admitted.record()["qualification_matched"] is True

    _admitted(monkeypatch, tmp_path, qualified={"probe": {"shape": [1], "sha256": "0" * 64}})
    with pytest.raises(capture.KdaCaptureKernelRefused, match="outside its packaged qualification: probe"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(), device="cpu")
    _admitted(monkeypatch, tmp_path, qualified=None)
    with pytest.raises(capture.KdaCaptureKernelRefused, match="qualification file"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(), device="cpu")
    # Dev mode prints the difference and runs the kernel in this runtime.
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(),
                                                device="cpu")
    assert admitted.qualification_sha256 is None
    assert admitted.record()["qualification_matched"] is False
    assert "seal KDA capture kernel qualification differs" in capsys.readouterr().out


def test_a_dev_mode_mismatch_records_that_the_runtime_is_not_the_qualified_one(
        bound_model, modeling, monkeypatch, tmp_path, capsys):
    """Dev mode runs a kernel that differs from its qualification, and says so (#1199 D5)."""
    import hashlib

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    path = _admitted(monkeypatch, tmp_path, qualified={"probe": {"shape": [1], "sha256": "0" * 64}})
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(),
                                                device="cpu")
    assert "seal KDA capture kernel qualification differs" in capsys.readouterr().out
    record = admitted.record()
    # The digest names the file this runtime was compared with; the flag says
    # whether it matched, so the record never claims a qualification it lacks.
    assert record["qualification_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["qualification_matched"] is False
    assert admitted.qualification_matched is False

    _admitted(monkeypatch, tmp_path, qualified={})
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, kda_layer(),
                                                device="cpu")
    assert "seal KDA capture kernel qualification" not in capsys.readouterr().out
    assert admitted.record()["qualification_matched"] is True


# ---- the pass scope ---------------------------------------------------------

class CountingDispatch:
    """A dispatch whose block runs the kernel ``calls`` times (0: the fallback ran)."""

    def __init__(self, calls=1):
        self.calls = calls
        self.active = False
        self.entered = 0

    def __enter__(self):
        self.active = True
        self.entered += 1
        return self

    def __exit__(self, *exc_info):
        for _ in range(self.calls):
            count_one_kernel_pass()
        self.active = False
        return False


def test_a_pass_must_run_the_kernel_exactly_once():
    admitted = capture.AdmittedKdaKernel(CountingDispatch(), {"name": "kda_gram_v1"}, None)
    with admitted.scope():
        pass
    assert (admitted.passes, admitted.calls) == (1, 1)
    for calls in (0, 2):
        admitted = capture.AdmittedKdaKernel(CountingDispatch(calls), {"name": "kda_gram_v1"},
                                             None)
        with pytest.raises(capture.KdaCaptureKernelRefused, match="must run the kernel once"):
            with admitted.scope():
                pass
        assert admitted.passes == 0


# ---- the layer quantum -------------------------------------------------------

def test_the_quantum_refuses_the_kernel_as_a_plan_field_or_an_unknown_name(monkeypatch):
    import prismaquant.gpu_guard as gpu_guard
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused, run_layer_quantum
    from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV
    from prismaquant.matmul_arithmetic import BF16_REDUCTION_ENV

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: None)
    for name in (REPLAY_REGIME_ENV, BF16_REDUCTION_ENV, capture.KDA_KERNEL_ENV):
        monkeypatch.delenv(name, raising=False)
    arguments = dict(record={}, adjoint_slice={}, plan_sha256="", prepared="", output_root="")
    with pytest.raises(QuantumIdentityRefused, match="not a plan execution field"):
        run_layer_quantum({"execution": {"kda_capture_kernel": "kda_gram_v1"}}, **arguments)
    monkeypatch.setenv(capture.KDA_KERNEL_ENV, "kda_gram_fp32_v1")
    with pytest.raises(QuantumIdentityRefused, match="names no known capture kernel"):
        run_layer_quantum({"execution": {}}, **arguments)


def test_the_runtime_execution_and_the_counters_carry_the_kernel_only_when_named():
    from prismaquant.joint_cost_quantum import QuantumCounters, quantum_runtime_execution

    config = {"execution": {"a": 1}, "max_gpu_bytes": 7}
    assert "kda_capture_kernel" not in quantum_runtime_execution(config, replay_regime=None)
    assert quantum_runtime_execution(config, replay_regime=None,
                                     kda_capture_kernel="kda_gram_v1")[
        "kda_capture_kernel"] == "kda_gram_v1"
    counters = QuantumCounters(quantum_id="q", identity_sha256="i", chunks=[], frontier=None)
    assert "kda_capture_kernel" not in counters.finish(units_done=0, units_total=0)
    counters.kda_capture_kernel = capture.not_executed_record("kda_gram_v1", "no KDA")
    assert counters.finish(units_done=0, units_total=0)["kda_capture_kernel"] == {
        "executed": False, "name": "kda_gram_v1", "reason": "no KDA"}


# The core, on the CPU fixture campaign of test_joint_cost_quantum_runtime. A
# quantum refuses to rerun into its own checkpoints, so each run gets its own
# campaign; the fixture is seeded, so their rows agree bit for bit.

def _quantum(tmp_path, monkeypatch, *, kernel=None):
    import test_joint_cost_quantum_runtime as runtime
    import test_quantum_probe_identity_once_1183 as once

    single, receipt, output_root = once._campaign(tmp_path, monkeypatch)
    execution = runtime._execution
    if kernel is not None:
        monkeypatch.setattr(runtime, "_execution", lambda path, **kwargs: {
            **execution(path, **kwargs), "kda_capture_kernel": kernel})
    try:
        return runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
            output_root=output_root, plan_sha=runtime._hex("d"),
            prepared_sha=runtime._hex("e"))
    finally:
        monkeypatch.setattr(runtime, "_execution", execution)


def _passes(monkeypatch, dispatch):
    """Every isolated-layer call inside the quantum core, and whether the kernel block was open.

    The reference run and Stage A, which build the campaign before the
    quantum runs, call the same method; only calls made while
    ``run_layer_quantum_core`` runs are recorded.
    """
    import test_joint_cost_quantum_runtime as runtime
    from prismaquant import cost_streaming

    events, inside = [], []
    original = cost_streaming.StreamedCausalLM.isolated_layer
    core = runtime.run_layer_quantum_core

    def recorded(self, batch, layer, *args, **kwargs):
        if inside:
            events.append((layer, dispatch.active))
        return original(self, batch, layer, *args, **kwargs)

    def counted_core(*args, **kwargs):
        inside.append(True)
        try:
            return core(*args, **kwargs)
        finally:
            inside.pop()

    monkeypatch.setattr(cost_streaming.StreamedCausalLM, "isolated_layer", recorded)
    monkeypatch.setattr(runtime, "run_layer_quantum_core", counted_core)
    return events


def _rows(payload):
    return {(name, fmt): (row["signed_components_per_probe"], row["x2_per_probe"])
            for name, rows in payload["costs"].items() for fmt, row in rows.items()}


def _probe(payload):
    return payload["provenance"]["probe_identity"]


def test_the_core_runs_every_target_pass_on_the_kernel_and_stamps_its_identity(
        tmp_path, monkeypatch):
    baseline, _, baseline_counters = _quantum(tmp_path / "baseline", monkeypatch)
    assert "kda_capture_kernel" not in _probe(baseline)["arithmetic"]
    assert "kda_capture_kernel" not in baseline_counters

    dispatch = CountingDispatch()
    identity = {"schema": capture.IDENTITY_SCHEMA, "name": "kda_gram_v1", "probe": "fixture"}
    admissions = []

    def admit(name, model, layer_module, *, device, emits_handoff=False):
        admissions.append((name, emits_handoff))
        return capture.AdmittedKdaKernel(dispatch, identity, "q" * 64,
                                         qualification_matched=True)

    monkeypatch.setattr(capture, "admit_kda_capture_kernel", admit)
    events = _passes(monkeypatch, dispatch)
    payload, _, counters = _quantum(tmp_path / "kernel", monkeypatch, kernel="kda_gram_v1")
    assert admissions == [("kda_gram_v1", False)]
    target = [active for layer, active in events if layer == 1]
    assert target and all(target), events
    assert not any(active for layer, active in events if layer != 1)
    assert _probe(payload)["arithmetic"]["kda_capture_kernel"] == identity
    record = counters["kda_capture_kernel"]
    assert record["executed"] is True and record["name"] == "kda_gram_v1"
    assert record["qualification_sha256"] == "q" * 64
    assert record["qualification_matched"] is True
    assert record["passes"] == record["calls"] == len(target) == dispatch.entered
    # The fixture dispatch changes no arithmetic, so the rows are the baseline's.
    assert _rows(payload) == _rows(baseline)


def test_the_core_refuses_a_pass_the_kernel_did_not_run(tmp_path, monkeypatch):
    silent = CountingDispatch(calls=0)
    monkeypatch.setattr(capture, "admit_kda_capture_kernel",
                        lambda *args, **kwargs: capture.AdmittedKdaKernel(
                            silent, {"name": "kda_gram_v1"}, None))
    with pytest.raises(capture.KdaCaptureKernelRefused, match="must run the kernel once"):
        _quantum(tmp_path, monkeypatch, kernel="kda_gram_v1")
    assert silent.entered == 1


def test_a_target_without_kda_records_the_kernel_as_not_executed(tmp_path, monkeypatch):
    baseline, _, _ = _quantum(tmp_path / "baseline", monkeypatch)
    payload, _, counters = _quantum(tmp_path / "kernel", monkeypatch, kernel="kda_gram_v1")
    assert counters["kda_capture_kernel"] == {
        "executed": False, "name": "kda_gram_v1",
        "reason": "target layer 1 has no KDA attention"}
    assert _probe(payload) == _probe(baseline)
    assert _rows(payload) == _rows(baseline)
