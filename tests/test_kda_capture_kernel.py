"""The Stage B KDA capture kernel's contract (PQ #1199, #1214).

A launch names the kernel with ``PRISMAQUANT_STAGE_B_KDA_KERNEL``. The GLM
derivative contract declares it, ``CaptureKernelDispatch`` substitutes it for
the verified fallback for one block at a time, and the layer quantum admits
it once per launch (kernel mode, PQ #1214). Every Stage B layer pass then
runs inside the dispatch: a pass of a KDA layer must run the kernel once
forward and once per backward, and a pass of any other layer must call no
kernel. Every step refuses rather than fall back to Torch.

These tests need no GPU. The kernel's numerics are
``tests/test_kda_chunk_kernel.py``'s; the quantum core in kernel mode is
``tests/test_kda_kernel_mode.py``'s. The tests that touch
``prismaquant.kernels.kda_chunk`` (directly, or through the dispatch, the
admission or the layer pass, which import it) need Triton importable, because
that module defines its kernels with ``@triton.jit`` at import; they are marked
``needs_triton`` and skip on hosted CPU CI (#1224). The rest run everywhere.
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
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

try:
    from prismaquant.kernels import kda_chunk
except ModuleNotFoundError as exc:  # only a missing Triton is a skip; anything else fails
    if exc.name != "triton" and not str(exc.name).startswith("triton."):
        raise
    kda_chunk = None

needs_triton = pytest.mark.skipif(
    kda_chunk is None,
    reason="prismaquant.kernels.kda_chunk defines its kernels with Triton at import")

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


# ---- the launch setting ----------------------------------------------------

def test_the_launch_setting_names_a_known_kernel_or_nothing():
    assert capture.kda_capture_kernel_from_environment({}) is None
    assert capture.kda_capture_kernel_from_environment(
        {capture.KDA_KERNEL_ENV: "kda_gram_v1"}) == "kda_gram_v1"
    for text in ("", "kda_gram_fp32_v1", "torch"):
        with pytest.raises(capture.KdaCaptureKernelRefused, match="names no known capture kernel"):
            capture.kda_capture_kernel_from_environment({capture.KDA_KERNEL_ENV: text})


@needs_triton
def test_every_launchable_kernel_is_declared_by_the_derivative_contract():
    for name in capture.KERNELS:
        declaration = derivative.capture_kernel_declaration(name)
        assert declaration["implements"] == derivative.VERSION
        assert declaration["replaces"] == "chunk_kimi_delta_attention"
        # Kernel mode (PQ #1214): every Stage B pass of a KDA layer, the
        # target's capture passes and the chain rolls; never Stage A.
        assert declaration["scope"] == "stage_b_kda_layer_passes"
        assert (declaration["module"], declaration["entry"]) == (
            "prismaquant.kernels.kda_chunk", "chunk_kimi_delta_attention")
    assert kda_chunk.NAME in capture.KERNELS
    assert kda_chunk.IMPLEMENTS == derivative.VERSION
    with pytest.raises(ValueError, match="undeclared capture kernel"):
        derivative.capture_kernel_declaration("fla_chunk_kda")


# ---- the dispatch ----------------------------------------------------------

@needs_triton
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


@needs_triton
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

@needs_triton
def test_admission_is_per_launch_and_refuses_an_unknown_kernel_before_any_work(monkeypatch):
    """Kernel mode admits once per launch, whatever the target layer (PQ #1214)."""
    monkeypatch.setattr(kda_chunk, "probe_digest", lambda device: pytest.fail("probed"))
    with pytest.raises(capture.KdaCaptureKernelRefused, match="unknown KDA capture kernel"):
        capture.admit_kda_capture_kernel("other", None, device="cpu")
    # The class check each layer pass applies.
    layer = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert capture.layer_runs_kda(layer) is False
    assert capture.layer_runs_kda(kda_layer()) is True


@needs_triton
def test_admission_refuses_an_unbound_model_and_a_kernel_that_cannot_run(
        bound_model, modeling, monkeypatch):
    with pytest.raises(capture.KdaCaptureKernelRefused, match="bound GLM derivative"):
        capture.admit_kda_capture_kernel("kda_gram_v1", torch.nn.Linear(2, 2), device="cpu")

    def broken(device):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(kda_chunk, "probe_digest", broken)
    with pytest.raises(capture.KdaCaptureKernelRefused, match="probe failed.*no CUDA device"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    monkeypatch.setattr(kda_chunk, "IMPLEMENTS", "glm_kda_causal_exp_v0")
    with pytest.raises(capture.KdaCaptureKernelRefused, match="not the declared kernel"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
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


@needs_triton
def test_admission_compares_the_runtime_with_the_packaged_qualification(
        bound_model, modeling, monkeypatch, tmp_path, capsys):
    import hashlib

    path = _admitted(monkeypatch, tmp_path, qualified={})
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    assert admitted.qualification_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    # The arithmetic identity names what runs; the evidence digest stays out of it.
    assert "qualification_sha256" not in admitted.identity
    assert admitted.identity["declaration"] == derivative.capture_kernel_declaration("kda_gram_v1")
    assert admitted.record()["qualification_sha256"] == admitted.qualification_sha256
    assert admitted.record()["qualification_matched"] is True

    _admitted(monkeypatch, tmp_path, qualified={"probe": {"shape": [1], "sha256": "0" * 64}})
    with pytest.raises(capture.KdaCaptureKernelRefused, match="outside its packaged qualification: probe"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    _admitted(monkeypatch, tmp_path, qualified=None)
    with pytest.raises(capture.KdaCaptureKernelRefused, match="qualification file"):
        capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    # Dev mode prints the difference and runs the kernel in this runtime.
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    assert admitted.qualification_sha256 is None
    assert admitted.record()["qualification_matched"] is False
    assert "seal KDA capture kernel qualification differs" in capsys.readouterr().out


@needs_triton
def test_a_dev_mode_mismatch_records_that_the_runtime_is_not_the_qualified_one(
        bound_model, modeling, monkeypatch, tmp_path, capsys):
    """Dev mode runs a kernel that differs from its qualification, and says so (#1199 D5)."""
    import hashlib

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    path = _admitted(monkeypatch, tmp_path, qualified={"probe": {"shape": [1], "sha256": "0" * 64}})
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    assert "seal KDA capture kernel qualification differs" in capsys.readouterr().out
    record = admitted.record()
    # The digest names the file this runtime was compared with; the flag says
    # whether it matched, so the record never claims a qualification it lacks.
    assert record["qualification_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["qualification_matched"] is False
    assert admitted.qualification_matched is False

    _admitted(monkeypatch, tmp_path, qualified={})
    admitted = capture.admit_kda_capture_kernel("kda_gram_v1", bound_model, device="cpu")
    assert "seal KDA capture kernel qualification" not in capsys.readouterr().out
    assert admitted.record()["qualification_matched"] is True


# ---- the layer pass ---------------------------------------------------------

class CountingDispatch:
    """A dispatch whose block adds ``counts`` to the kernel's counters on exit.

    The counts are this unit test's input: the arithmetic of the pass check,
    not whether a kernel ran. ``tests/test_kda_kernel_mode.py`` counts real
    forwards and backwards through an autograd function instead.
    """

    def __init__(self, counts=None):
        self.counts = dict(counts or {})
        self.active = False
        self.entered = 0

    def __enter__(self):
        self.active = True
        self.entered += 1
        return self

    def __exit__(self, *exc_info):
        for key, value in self.counts.items():
            kda_chunk._COUNTS[key] += value
        self.active = False
        return False


def _kernel(counts=None, identity=None):
    return capture.AdmittedKdaKernel(CountingDispatch(counts),
                                     identity or {"name": "kda_gram_v1"}, None)


@needs_triton
def test_a_kda_layer_pass_runs_the_kernel_once_forward_and_once_per_backward():
    admitted = _kernel(ONE_PASS)
    with admitted.layer_pass(kda_layer(), site="target", layer=3):
        assert admitted._dispatch.active
    # A fused chain roll: one forward, then one backward per probe on the
    # retained graph.
    admitted._dispatch.counts = {"calls": 1, "gram_forward": 2, "gram_backward": 8}
    with admitted.layer_pass(kda_layer(), site="chain", layer=4, backwards=4):
        pass
    record = admitted.record()
    assert (record["executed"], record["passes"], record["calls"]) == (True, 1, 1)
    assert record["chain"] == {"layers": [4], "passes": 1, "calls": 1, "gram_backward": 8}
    assert record["scoped_passes"] == 2 and "reason" not in record
    # Any other count refuses and is not counted: no call (the fallback ran),
    # two calls, a forward without its backward, a fused pass one backward short.
    for counts, backwards in (({}, 1),
                              ({"calls": 2, "gram_forward": 4, "gram_backward": 4}, 1),
                              ({"calls": 1, "gram_forward": 2}, 1),
                              ({"calls": 1, "gram_forward": 2, "gram_backward": 6}, 4)):
        refused = _kernel(counts)
        with pytest.raises(capture.KdaCaptureKernelRefused, match="layer 5 chain pass"):
            with refused.layer_pass(kda_layer(), site="chain", layer=5, backwards=backwards):
                pass
        assert refused.record()["scoped_passes"] == 0
        assert refused.record()["chain"]["passes"] == 0


@needs_triton
def test_a_pass_of_a_layer_without_kda_runs_in_the_dispatch_and_calls_no_kernel():
    dense = torch.nn.Sequential(torch.nn.Linear(2, 2))
    admitted = _kernel()
    with admitted.layer_pass(dense, site="target", layer=39):
        assert admitted._dispatch.active
    with admitted.layer_pass(dense, site="chain", layer=43, backwards=4):
        pass
    record = admitted.record()
    assert record["executed"] is False and record["scoped_passes"] == 2
    assert record["passes"] == record["calls"] == 0
    assert record["chain"] == {"layers": [], "passes": 0, "calls": 0, "gram_backward": 0}
    assert "no KDA" in record["reason"]
    # A layer the class check reads as having no KDA attention, but whose pass
    # calls the kernel, refuses: outside the dispatch that pass would have run
    # the fallback.
    calling = _kernel(ONE_PASS)
    with pytest.raises(capture.KdaCaptureKernelRefused,
                       match="layer 39 target pass.*no KDA attention"):
        with calling.layer_pass(dense, site="target", layer=39):
            pass


def test_a_layer_pass_names_a_stage_b_site_and_its_backwards():
    for kwargs, match in ((dict(site="stage_a", layer=1), "site"),
                          (dict(site="chain", layer=1, backwards=0), "backward")):
        with pytest.raises(capture.KdaCaptureKernelRefused, match=match):
            with _kernel(ONE_PASS).layer_pass(kda_layer(), **kwargs):
                pass


def test_the_handoff_names_the_kernel_by_name_and_identity():
    admitted = _kernel(identity={"name": "kda_gram_v1", "probe": "fixture"})
    assert admitted.handoff_stamp() == {"name": "kda_gram_v1",
                                        "identity_sha256": admitted.identity_sha256}


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
    counters.kda_capture_kernel = _kernel()
    block = counters.finish(units_done=0, units_total=0)["kda_capture_kernel"]
    assert block["executed"] is False and block["name"] == "kda_gram_v1"
    assert block["scoped_passes"] == 0
