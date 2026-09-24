"""The Stage B KDA capture kernel: launch setting, admission and pass scope (PQ #1199).

A GLM-5.3 KDA layer's Stage B capture spends most of its time in the pinned
Torch fallback ``chunk_kimi_delta_attention``, which is limited by memory
bandwidth. ``PRISMAQUANT_STAGE_B_KDA_KERNEL`` names a capture kernel that the
GLM derivative contract declares (``glm_source_derivative.
capture_kernel_declaration``). A layer quantum launched with it runs that
kernel in its target layer's forward and backward passes, when the target
layer has KDA attention. Everything else keeps the fallback:

* the chain rolls, which must be Stage A's arithmetic (PQ #997);
* a target layer without KDA attention, where the kernel would never be
  called, so nothing is stamped;
* every process that leaves the variable unset.

The setting is a launch setting, like ``PRISMAQUANT_STAGE_B_REPLAY_REGIME``:
the campaign container spec carries it, and a plan's ``execution`` block
must not.

The kernel computes the fallback's function and rounds differently, so it is
an arithmetic change. The admitted kernel's identity goes into the probe
identity's ``arithmetic`` block under ``kda_capture_kernel``. ``arithmetic``
is a run seal (``cost_currency.PROBE_IDENTITY_SEAL_FIELDS``): dev mode joins
rows with and without the kernel and prints the difference, and certified
mode refuses to join them. The identity names what ran (source, compiled
code, probe result, runtime); the digest of the qualification that vouches
for it goes into the result's ``kda_capture_kernel`` record instead, so new
evidence for one kernel does not reseal the rows it ran.

Admission fails closed. It refuses when the model has no bound GLM
derivative, when dispatch is not the verified fallback, when the kernel does
not compile or run, and when the quantum would hand its capture plane to
the next quantum (a band-serial producer, PQ #996): that plane would be the
kernel's arithmetic, and the consumer's chain rebuild runs the fallback.
Each pass then refuses unless the kernel ran exactly once, forward and
backward. There is no silent fallback to Torch.

The identity is compared with the packaged qualification
(``kernels/kda_chunk_qualification.json``) as a run seal, like the joint
projection backend's (PQ #1147): certified mode refuses a difference, and
dev mode prints it and runs the kernel in this runtime.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import platform
from pathlib import Path

KDA_KERNEL_ENV = "PRISMAQUANT_STAGE_B_KDA_KERNEL"
#: The key the identity occupies in ``statistics_arithmetic_identity``.
ARITHMETIC_FIELD = "kda_capture_kernel"
IDENTITY_SCHEMA = "prismaquant.stage_b.kda_capture_kernel.v1"
QUALIFICATION_SCHEMA = "prismaquant.kda_capture_kernel_qualification.v1"
QUALIFICATION_PATH = Path(__file__).with_name("kernels") / "kda_chunk_qualification.json"
#: The kernels a launch may name. Each must be declared by the GLM contract.
KERNELS = ("kda_gram_v1",)
#: The attention class whose forward calls ``chunk_kimi_delta_attention``.
KDA_ATTENTION_CLASS = ("transformers.models.glm5_next.modeling_glm5_next",
                       "Glm5NextTextLinearAttention")


class KdaCaptureKernelRefused(ValueError):
    """A malformed setting, an unadmittable kernel, or a pass the kernel did not run."""


def kda_capture_kernel_from_environment(environ: Mapping) -> str | None:
    """The kernel a launch names, or ``None`` (the fallback) when unset."""
    text = environ.get(KDA_KERNEL_ENV)
    if text is None:
        return None
    if text not in KERNELS:
        raise KdaCaptureKernelRefused(
            f"{KDA_KERNEL_ENV}={text!r} names no known capture kernel; known: {list(KERNELS)}")
    return text


def layer_runs_kda(layer_module) -> bool:
    """True when ``layer_module`` holds GLM KDA attention, by class, not by name."""
    for module in layer_module.modules():
        cls = type(module)
        if (cls.__module__, cls.__name__) == KDA_ATTENTION_CLASS:
            return True
    return False


def _runtime_identity(device) -> dict:
    import torch
    import triton

    from .joint_projection_backend import executing_image
    props = torch.cuda.get_device_properties(device)
    return {"torch": str(torch.__version__), "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda, "triton": triton.__version__,
            "machine": platform.machine(), "image": executing_image(),
            "device": {"name": props.name, "major": props.major, "minor": props.minor,
                       "multi_processor_count": props.multi_processor_count}}


def _qualification():
    try:
        raw = QUALIFICATION_PATH.read_bytes()
    except FileNotFoundError:
        return None, None
    value = json.loads(raw)
    if value.get("schema") != QUALIFICATION_SCHEMA or value.get("status") != "qualified":
        raise KdaCaptureKernelRefused("the packaged KDA kernel qualification is not qualified")
    return value, hashlib.sha256(raw).hexdigest()


#: The identity fields the qualification binds, compared as one run seal.
QUALIFIED_FIELDS = ("name", "source_sha256", "compiled", "probe", "runtime")


def _require_qualified(identity) -> str | None:
    from .dev_mode import seal_check
    qualification, digest = _qualification()
    expected = None if qualification is None else {
        key: qualification["identity"].get(key) for key in QUALIFIED_FIELDS}
    actual = {key: identity.get(key) for key in QUALIFIED_FIELDS}
    changed = (["qualification file"] if expected is None else
               sorted(key for key in QUALIFIED_FIELDS if actual[key] != expected[key]))
    seal_check("KDA capture kernel qualification", expected, actual,
               where="packaged kernels/kda_chunk_qualification.json versus this runtime",
               refusal=lambda: KdaCaptureKernelRefused(
                   "KDA capture kernel runs outside its packaged qualification: "
                   + ", ".join(changed)))
    return digest


class AdmittedKdaKernel:
    """One quantum's admitted kernel: its identity, its pass scope and its counts."""

    def __init__(self, dispatch, identity, qualification_sha256):
        self._dispatch = dispatch
        self.identity = identity
        self.qualification_sha256 = qualification_sha256
        self.identity_sha256 = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.passes = 0
        self.calls = 0

    @contextmanager
    def scope(self):
        """One target-layer pass, forward and backward, on the kernel."""
        from .kernels import kda_chunk
        before = kda_chunk.counts()
        with self._dispatch:
            yield
        after = kda_chunk.counts()
        delta = {key: after[key] - before[key] for key in after}
        if delta != {"calls": 1, "gram_forward": 2, "gram_backward": 2}:
            raise KdaCaptureKernelRefused(
                "a KDA capture kernel pass must run the kernel once, forward and "
                f"backward (1 call, 2 Gram forwards, 2 Gram backwards); it ran {delta}")
        self.passes += 1
        self.calls += delta["calls"]

    def record(self) -> dict:
        return {"executed": True, "name": self.identity["name"],
                "identity_sha256": self.identity_sha256,
                "qualification_sha256": self.qualification_sha256,
                "passes": self.passes, "calls": self.calls}


def not_executed_record(name: str, reason: str) -> dict:
    return {"executed": False, "name": name, "reason": reason}


def admit_kda_capture_kernel(name, model, layer_module, *, device, emits_handoff=False):
    """The admitted kernel for this target layer, or ``None`` when it has no KDA.

    Compiles the kernel and runs its fixed probe on ``device`` before any
    capture work, so a kernel that cannot run refuses here.
    """
    from .glm_source_derivative import CaptureKernelDispatch, capture_kernel_declaration

    if name not in KERNELS:
        raise KdaCaptureKernelRefused(f"unknown KDA capture kernel {name!r}")
    if not layer_runs_kda(layer_module):
        return None
    if emits_handoff:
        raise KdaCaptureKernelRefused(
            "a band-serial producer hands off the plane its capture pass wrote, and under "
            "the KDA capture kernel that plane is the kernel's arithmetic, while the "
            "consumer's chain rebuild runs the fallback; emit no handoff, or launch "
            f"without {KDA_KERNEL_ENV}")
    try:
        declaration = capture_kernel_declaration(name)
        dispatch = CaptureKernelDispatch(model, name)
    except ValueError as exc:
        raise KdaCaptureKernelRefused(str(exc)) from exc
    from .kernels import kda_chunk
    if kda_chunk.NAME != name or kda_chunk.IMPLEMENTS != declaration["implements"]:
        raise KdaCaptureKernelRefused("the kernel module is not the declared kernel")
    try:
        probe = kda_chunk.probe_digest(device)
    except Exception as exc:  # noqa: BLE001 - any probe failure refuses the kernel
        raise KdaCaptureKernelRefused(f"the KDA kernel probe failed: {exc!r}") from exc
    compiled = kda_chunk.compiled_kernels()
    if len(compiled) != len(kda_chunk.KERNEL_KEYS):
        raise KdaCaptureKernelRefused(f"the KDA kernel probe compiled {sorted(compiled)}")
    identity = {"schema": IDENTITY_SCHEMA, "name": name, "declaration": declaration,
                "source_sha256": kda_chunk.source_sha256(), "compiled": compiled,
                "probe": probe, "runtime": _runtime_identity(device)}
    return AdmittedKdaKernel(dispatch, identity, _require_qualified(identity))


def qualification_candidate(device) -> dict:
    """The identity fields a qualification binds, read from this runtime.

    For the numerics harness, which records them next to the numbers that
    qualify them. Compiles the kernel and runs its probe.
    """
    from .kernels import kda_chunk
    probe = kda_chunk.probe_digest(device)
    return {"name": kda_chunk.NAME, "source_sha256": kda_chunk.source_sha256(),
            "compiled": kda_chunk.compiled_kernels(), "probe": probe,
            "runtime": _runtime_identity(device)}
