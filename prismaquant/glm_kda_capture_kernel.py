"""The Stage B KDA capture kernel: launch setting, admission and layer passes.

A GLM-5.3 KDA layer's Stage B capture spends most of its time in the pinned
Torch fallback ``chunk_kimi_delta_attention``, which is limited by memory
bandwidth (PQ #1199). ``PRISMAQUANT_STAGE_B_KDA_KERNEL`` names a capture
kernel that the GLM derivative contract declares (``glm_source_derivative.
capture_kernel_declaration``). A layer quantum launched with it runs in
**kernel mode** (PQ #1214): every Stage B layer pass runs inside the
kernel's dispatch (:meth:`AdmittedKdaKernel.layer_pass`), the target
layer's capture passes, a band-serial producer's among them, and the chain
rolls. A pass of a layer with KDA attention runs the kernel; a pass of any
other layer calls it zero times. Everything else keeps the fallback:

* Stage A, whose checkpoint planes stay the fallback's arithmetic;
* every process that leaves the variable unset.

The setting is a launch setting, like ``PRISMAQUANT_STAGE_B_REPLAY_REGIME``:
the campaign container spec carries it, and a plan's ``execution`` block
must not. One dispatch seals one spec for every row, so a band runs in one
mode.

The kernel computes the fallback's function and rounds differently, so it is
an arithmetic change, and **the mode is the seal**: in kernel mode the
admitted kernel's identity goes into the probe identity's ``arithmetic``
block under ``kda_capture_kernel`` for every quantum, including one whose
target and chain layers have no KDA attention. ``arithmetic`` is a run seal
(``cost_currency.PROBE_IDENTITY_SEAL_FIELDS``): dev mode joins rows with and
without the kernel and prints the difference, and certified mode refuses to
join them. The identity names what ran (source, compiled code, probe result,
runtime); the digest of the qualification that vouches for it goes into the
result's ``kda_capture_kernel`` record instead, so new evidence for one
kernel does not reseal the rows it ran.

The chain rolls of a KDA layer run the kernel too, so a quantum's chained
cotangents are the kernel's arithmetic, not Stage A's; the chain regime
(PQ #997) still sets their grouping. A band-serial producer's handoff plane
is then the plane its consumer's kernel-mode chain rebuild ends on (PQ #996
within the mode), and the handoff names its producer's kernel
(:meth:`AdmittedKdaKernel.handoff_stamp`). A consumer launched in another
mode refuses it (``joint_quantum_handoff.load_quantum_handoff``), and one
whose admitted kernel is another build refuses it in the core.

Admission fails closed, once per launch. It refuses when the model has no
bound GLM derivative, when dispatch is not the verified fallback, and when
the kernel does not compile or run. Each layer pass then refuses unless the
kernel ran exactly once forward and once per backward on a KDA layer, and
not at all on any other. There is no silent fallback to Torch.

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


def _require_qualified(identity) -> tuple[str | None, bool]:
    """The packaged qualification's digest, and whether this runtime matches it.

    Certified mode refuses a mismatch. Dev mode prints it and returns
    ``False``, so the kernel record says the run is not the qualified one.
    """
    from .dev_mode import seal_check
    qualification, digest = _qualification()
    expected = None if qualification is None else {
        key: qualification["identity"].get(key) for key in QUALIFIED_FIELDS}
    actual = {key: identity.get(key) for key in QUALIFIED_FIELDS}
    changed = (["qualification file"] if expected is None else
               sorted(key for key in QUALIFIED_FIELDS if actual[key] != expected[key]))
    matched = seal_check(
        "KDA capture kernel qualification", expected, actual,
        where="packaged kernels/kda_chunk_qualification.json versus this runtime",
        refusal=lambda: KdaCaptureKernelRefused(
            "KDA capture kernel runs outside its packaged qualification: "
            + ", ".join(changed)))
    return digest, matched


#: The Stage B sites a layer pass runs at: the target layer's capture passes
#: and the chain rolls. Stage A has none; its planes stay the fallback's.
PASS_SITES = ("target", "chain")


class AdmittedKdaKernel:
    """One quantum's admitted kernel: its identity, its layer passes and their counts.

    Kernel mode (PQ #1214): every Stage B layer pass of the quantum runs in
    :meth:`layer_pass`. ``passes`` and ``calls`` count the target layer's
    KDA passes, ``chain`` the chain rolls' KDA passes, and ``scoped_passes``
    every layer pass that ran inside the dispatch, KDA or not.
    """

    def __init__(self, dispatch, identity, qualification_sha256, *,
                 qualification_matched=False):
        self._dispatch = dispatch
        self.identity = identity
        # The digest of the packaged qualification this runtime was compared
        # with (None without one), and whether the identities were equal.
        self.qualification_sha256 = qualification_sha256
        self.qualification_matched = bool(qualification_matched)
        self.identity_sha256 = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.passes = 0
        self.calls = 0
        self.chain = {"layers": [], "passes": 0, "calls": 0, "gram_backward": 0}
        self.scoped_passes = 0

    @contextmanager
    def layer_pass(self, layer_module, *, site, layer, backwards=1):
        """One Stage B layer pass inside the kernel's dispatch.

        A pass is one forward of ``layer_module`` and ``backwards`` backwards
        through it: one per capture pass and per unfused or batched roll, one
        per probe on the retained graph of a fused roll. The kernel counts
        one call and two Gram passes forward, and two Gram passes per
        backward, so a pass of a layer with KDA attention (by class) must add
        exactly ``{calls: 1, gram_forward: 2, gram_backward: 2 * backwards}``
        to them. A pass of any other layer must add nothing: it runs inside
        the dispatch too, so a layer the class check misreads refuses here
        instead of running the fallback. A refused pass is not counted.
        """
        if site not in PASS_SITES:
            raise KdaCaptureKernelRefused(
                f"a Stage B layer pass site is one of {list(PASS_SITES)}, not {site!r}")
        if type(backwards) is not int or backwards < 1:
            raise KdaCaptureKernelRefused(
                f"layer {layer} {site} pass: a layer pass runs at least one "
                f"backward, not {backwards!r}")
        from .kernels import kda_chunk
        kda = layer_runs_kda(layer_module)
        before = kda_chunk.counts()
        with self._dispatch:
            yield
        after = kda_chunk.counts()
        delta = {key: after[key] - before[key] for key in after}
        where = f"layer {int(layer)} {site} pass"
        if kda:
            expected = {"calls": 1, "gram_forward": 2, "gram_backward": 2 * backwards}
            if delta != expected:
                raise KdaCaptureKernelRefused(
                    f"{where}: a KDA layer pass must run the kernel once forward and "
                    f"once per backward ({backwards}): {expected}; it ran {delta}")
        elif any(delta.values()):
            raise KdaCaptureKernelRefused(
                f"{where}: the layer has no KDA attention by its class, yet its pass "
                f"ran the kernel ({delta}); outside the dispatch it would have run "
                "the fallback")
        self.scoped_passes += 1
        if not kda:
            return
        if site == "target":
            self.passes += 1
            self.calls += delta["calls"]
            return
        chain = self.chain
        if int(layer) not in chain["layers"]:
            # In roll order: the chain walks down from its checkpoint.
            chain["layers"].append(int(layer))
        chain["passes"] += 1
        chain["calls"] += delta["calls"]
        chain["gram_backward"] += delta["gram_backward"]

    def handoff_stamp(self) -> dict:
        """What a band-serial producer's handoff names: this kernel, by name and identity."""
        return {"name": self.identity["name"], "identity_sha256": self.identity_sha256}

    def record(self) -> dict:
        executed = self.passes + self.chain["passes"] > 0
        record = {"executed": executed, "name": self.identity["name"],
                  "identity_sha256": self.identity_sha256,
                  "qualification_sha256": self.qualification_sha256,
                  "qualification_matched": self.qualification_matched,
                  "passes": self.passes, "calls": self.calls,
                  "chain": {**self.chain, "layers": list(self.chain["layers"])},
                  "scoped_passes": self.scoped_passes}
        if not executed:
            record["reason"] = (
                "no KDA layer pass ran: neither the target layer nor a chain layer "
                "has KDA attention; the mode still seals the rows")
        return record


def admit_kda_capture_kernel(name, model, *, device):
    """The kernel a kernel-mode launch runs, admitted once per quantum.

    Compiles the kernel and runs its fixed probe on ``device`` before any
    chain or capture work, so a kernel that cannot run refuses here, whatever
    the quantum's layers are.
    """
    from .glm_source_derivative import CaptureKernelDispatch, capture_kernel_declaration

    if name not in KERNELS:
        raise KdaCaptureKernelRefused(f"unknown KDA capture kernel {name!r}")
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
    digest, matched = _require_qualified(identity)
    return AdmittedKdaKernel(dispatch, identity, digest, qualification_matched=matched)


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
