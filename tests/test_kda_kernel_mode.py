"""Stage B kernel mode (PQ #1214): every KDA layer pass on the admitted kernel.

A launch that names the KDA capture kernel (``PRISMAQUANT_STAGE_B_KDA_KERNEL``)
runs every Stage B pass of a KDA layer on it: the target layer's capture
passes, a band-serial producer's among them, and the chain rolls of KDA
layers. The Stage A checkpoint planes stay the fallback's. The mode is the
seal: the kernel's identity goes into every row's arithmetic, and a handoff
carries its producer's kernel, which a consumer launched in another mode
refuses. Within the mode, band-serial equals chain mode byte for byte (#996).

The CPU fixture is the five-layer band-serial campaign of
``tests/test_quantum_band_serial.py`` (bands 5: [4, 3] and 3: [2, 1, 0]).
Layers 1, 2 and 4 carry a module of the GLM KDA attention class whose
forward calls a dispatch global at call time, as the image's attention calls
``chunk_kimi_delta_attention``. The fallback returns its input. The fixture
kernel is an autograd function that counts its Gram passes as the real
kernel counts them (two per call forward, two per backward) and scales the
activation and its gradient, so a pass that ran the fallback changes the
bytes. Its counts come from real forwards and real backwards on a retained
graph, never from the dispatch.

Every test here runs kernel mode, whose layer pass reads the counters of
``prismaquant.kernels.kda_chunk``. That module defines its kernels with
``@triton.jit`` at import, so the whole module skips where Triton is not
installed (hosted CPU CI, #1224); the fleet has Triton and runs all of it.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import json
import pickle
import shutil
import sys

import pytest
import torch

import prismaquant.joint_adjoint_checkpoints as roll_owner
from prismaquant import glm_kda_capture_kernel as capture
from prismaquant import glm_source_derivative as derivative
from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.joint_cost_quantum import QuantumIdentityRefused
from prismaquant.joint_quantum_handoff import (
    HANDOFF_RECORD_NAME,
    QuantumHandoffRefused,
    handoff_record_bytes,
    handoff_seal_sha256,
    load_quantum_handoff,
)

import test_joint_cost_quantum_runtime as runtime
import test_streamed_cost_checkpoints as tiny
from test_quantum_band_serial import (
    BANDS,
    _campaign,
    _consumer_record,
    _emitter,
    _handoff_plane,
    _quantum,
    _state_digest,
    _tensor_digest,
)

try:
    from prismaquant.kernels import kda_chunk
except ModuleNotFoundError as exc:  # only a missing Triton is a skip; anything else fails
    if exc.name != "triton" and not str(exc.name).startswith("triton."):
        raise
    kda_chunk = None

pytestmark = pytest.mark.skipif(
    kda_chunk is None,
    reason="prismaquant.kernels.kda_chunk defines its kernels with Triton at import")

KERNEL = "kda_gram_v1"
#: The fixture layers with KDA attention: KDA and DSA targets, KDA chain layers.
KDA_LAYERS = frozenset({1, 2, 4})
N_PROBES, N_BATCHES = 4, 5
#: The fixture kernel's factor per Gram pass: one binade step, so it moves bits.
FACTOR = 1.0 + 2.0 ** -7
#: The dispatch global the fixture's KDA attention reads at call time.
GLM = SimpleNamespace(chunk_kimi_delta_attention=None)
#: The launch's execution before any test adds the kernel setting to it.
_EXECUTION = runtime._execution


class FakeKdaAttention(torch.nn.Module):
    """A module of the GLM KDA attention class; its forward calls the global."""

    def forward(self, hidden):
        return GLM.chunk_kimi_delta_attention(hidden)


FakeKdaAttention.__module__, FakeKdaAttention.__name__ = capture.KDA_ATTENTION_CLASS
FakeKdaAttention.__qualname__ = FakeKdaAttention.__name__


class _Gram(torch.autograd.Function):
    """One Gram pass of the fixture kernel, counted as ``_KdaGram`` counts."""

    @staticmethod
    def forward(ctx, hidden):
        kda_chunk._COUNTS["gram_forward"] += 1
        return hidden * FACTOR

    @staticmethod
    def backward(ctx, grad):
        kda_chunk._COUNTS["gram_backward"] += 1
        return grad * FACTOR


class FixtureDispatch:
    """``CaptureKernelDispatch`` on the fixture global: one block, then the fallback."""

    def __init__(self, entry):
        self.entry = entry
        self.fallback = GLM.chunk_kimi_delta_attention
        self.active = False
        self.entered = 0

    def __enter__(self):
        if self.active:
            raise ValueError("capture kernel blocks do not nest")
        if GLM.chunk_kimi_delta_attention is not self.fallback:
            raise ValueError("KDA dispatch changed outside a capture kernel block")
        GLM.chunk_kimi_delta_attention = self.entry
        self.active = True
        self.entered += 1
        return self

    def __exit__(self, *exc_info):
        current = GLM.chunk_kimi_delta_attention
        GLM.chunk_kimi_delta_attention = self.fallback
        self.active = False
        if current is not self.entry:
            raise ValueError("KDA dispatch changed inside a capture kernel block")
        return False


def _identity_sha256(identity):
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity(**changes):
    return {"schema": capture.IDENTITY_SCHEMA, "name": KERNEL,
            "declaration": derivative.capture_kernel_declaration(KERNEL),
            "probe": "fixture", **changes}


def install_fixture_kernel(monkeypatch, *, kda_layers=KDA_LAYERS):
    """Point the fixture global at the fallback; returns the call log.

    ``calls.events`` records each attention call as ``(implementation,
    in_core)``, ``in_core`` being whether a quantum core was running
    (:func:`track_core`). ``calls.fallback`` and ``calls.kernel`` are the two
    implementations a dispatch may swap between.

    The fixture attention's class names the GLM modeling module, so the
    source identity (``glm_source_derivative.source_derivative_identity``)
    looks that module up and hashes its file. It is registered here as a
    module loaded from this test file: a GLM runtime that is not the
    corrected one, which needs no derivative binding.
    """
    modeling = ModuleType(capture.KDA_ATTENTION_CLASS[0])
    modeling.__file__ = __file__
    monkeypatch.setitem(sys.modules, capture.KDA_ATTENTION_CLASS[0], modeling)
    calls = SimpleNamespace(events=[], core=False, kda_layers=kda_layers)

    def fallback(hidden):
        calls.events.append(("fallback", calls.core))
        return hidden

    def kernel(hidden):
        kda_chunk._COUNTS["calls"] += 1
        calls.events.append(("kernel", calls.core))
        return _Gram.apply(_Gram.apply(hidden))

    calls.fallback, calls.kernel = fallback, kernel
    calls.in_core = lambda kind: sum(1 for name, core in calls.events
                                     if core and name == kind)
    monkeypatch.setattr(GLM, "chunk_kimi_delta_attention", fallback)
    return calls


def attach_kda(monkeypatch, layer_class):
    """Route each ``layer_class`` output through the layer's ``kda`` child, if any."""
    forward = layer_class.forward

    def layer_forward(self, hidden_states, **kwargs):
        out = forward(self, hidden_states, **kwargs)
        kda = self._modules.get("kda")
        return out if kda is None else kda(out)

    monkeypatch.setattr(layer_class, "forward", layer_forward)


def track_core(monkeypatch, owner, calls):
    """Mark the calls made while ``owner.run_layer_quantum_core`` runs."""
    core = owner.run_layer_quantum_core

    def counted(*args, **kwargs):
        calls.core = True
        try:
            return core(*args, **kwargs)
        finally:
            calls.core = False

    monkeypatch.setattr(owner, "run_layer_quantum_core", counted)


def admit_fixture_kernel(monkeypatch, calls, *, entry=None, identity=None):
    """Admit the fixture kernel wherever the core admits one; returns the admissions."""
    admitted = []

    def admit(name, model, *args, device, **kwargs):
        assert name == KERNEL
        kernel = capture.AdmittedKdaKernel(
            FixtureDispatch(calls.kernel if entry is None else entry),
            _identity() if identity is None else identity, "q" * 64,
            qualification_matched=True)
        admitted.append(kernel)
        return kernel

    monkeypatch.setattr(capture, "admit_kda_capture_kernel", admit)
    return admitted


@pytest.fixture
def glm(monkeypatch):
    """KDA attention on ``glm.kda_layers`` of every tiny fixture model built from here on."""
    calls = install_fixture_kernel(monkeypatch)
    init = tiny._DenseTinyLM.__init__

    def with_kda(self, *args, **kwargs):
        init(self, *args, **kwargs)
        for index, layer in enumerate(self.model.layers):
            if index in calls.kda_layers:
                layer.kda = FakeKdaAttention()

    monkeypatch.setattr(tiny._DenseTinyLM, "__init__", with_kda)
    attach_kda(monkeypatch, tiny._DenseLayer)
    track_core(monkeypatch, runtime, calls)
    return calls


def _admit(monkeypatch, glm, **kwargs):
    return admit_fixture_kernel(monkeypatch, glm, **kwargs)


def _mode(monkeypatch, kernel):
    """Launch every later quantum with ``kernel`` (``None``: the fallback)."""
    if kernel is None:
        monkeypatch.setattr(runtime, "_execution", _EXECUTION)
    else:
        monkeypatch.setattr(runtime, "_execution", lambda path, **kwargs: {
            **_EXECUTION(path, **kwargs), "kda_capture_kernel": kernel})


def _binder(handoff, kernel):
    def bind(record, adjoint_slice):
        return load_quantum_handoff(handoff["path"], handoff["sha256"], record=record,
                                    adjoint_slice=adjoint_slice, kda_capture_kernel=kernel)
    return bind


def _published(record):
    path = next((Path(record["output_space"]["root"]) / "handoff").glob(
        f"*/{HANDOFF_RECORD_NAME}"))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _rows(payload):
    return {(name, fmt): (row["signed_components_per_probe"], row["x2_per_probe"])
            for name, rows in payload["costs"].items() for fmt, row in rows.items()}


def _arithmetic(payload):
    return payload["provenance"]["probe_identity"]["arithmetic"]


def _chain_passes(layers, *, fusion):
    """Layer passes a chain rolls at batch size one: per batch fused, else per (probe, batch)."""
    return len(layers) * (N_BATCHES if fusion else N_PROBES * N_BATCHES)


# ---- every KDA pass runs the kernel ------------------------------------------

@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_every_kda_pass_of_a_kernel_mode_quantum_runs_the_kernel(
        tmp_path, monkeypatch, glm, fusion):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch, fusion=fusion)
    # Stage A and the reference run built the campaign: the fallback only.
    assert {name for name, _ in glm.events} == {"fallback"}
    assert glm.in_core("fallback") == glm.in_core("kernel") == 0
    admitted = _admit(monkeypatch, glm)
    _mode(monkeypatch, KERNEL)
    for layers in BANDS.values():
        for layer in layers:
            glm.events.clear()
            payload, record, counters = _quantum(
                tmp_path, monkeypatch, single=single, receipt=receipt,
                output_root=output_root, layer=layer)
            chain = [int(c) for c in record["adjoint"]["chain_layers"]]
            kda_chain = [c for c in chain if c in KDA_LAYERS]
            # No KDA pass inside the core ran the fallback, target or chain.
            assert glm.in_core("fallback") == 0, (layer, glm.events)
            block = counters["kda_capture_kernel"]
            # ``layer_passes`` counts the target's replays; each passes every batch.
            replayed = counters["replay"]["layer_passes"] * N_BATCHES
            target = replayed if layer in KDA_LAYERS else 0
            rolled = _chain_passes(kda_chain, fusion=fusion)
            assert block["passes"] == block["calls"] == target, layer
            assert block["chain"] == {
                "layers": kda_chain, "passes": rolled, "calls": rolled,
                "gram_backward": 2 * N_PROBES * N_BATCHES * len(kda_chain)}, layer
            assert glm.in_core("kernel") == target + rolled, layer
            assert block["executed"] is (target + rolled > 0)
            # Every layer pass the core ran, KDA or not, ran inside the dispatch.
            assert block["scoped_passes"] == replayed + _chain_passes(chain, fusion=fusion)
            assert block["name"] == KERNEL and block["qualification_matched"] is True
            # The mode is the seal, whatever this quantum's layers are.
            assert _arithmetic(payload)["kda_capture_kernel"] == _identity()
    assert len(admitted) == sum(len(layers) for layers in BANDS.values())


def test_the_mode_is_the_seal_where_no_kda_pass_ran(tmp_path, monkeypatch, glm):
    """Layer 1's quantum rolls layer 2 and captures layer 1; neither has KDA here."""
    glm.kda_layers = frozenset({4})
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    _admit(monkeypatch, glm)
    _mode(monkeypatch, None)
    fallback, record, fallback_counters = _quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=1)
    assert record["adjoint"]["chain_layers"] == [2]
    assert "kda_capture_kernel" not in fallback_counters
    space = Path(record["output_space"]["root"])
    shutil.move(str(space), str(tmp_path / "fallback-space"))
    _mode(monkeypatch, KERNEL)
    payload, _record, counters = _quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=1)
    block = counters["kda_capture_kernel"]
    assert block["executed"] is False
    assert block["passes"] == block["calls"] == 0
    assert block["chain"]["passes"] == 0 and block["chain"]["layers"] == []
    assert block["scoped_passes"] > 0
    assert "no KDA" in block["reason"]
    assert _arithmetic(payload)["kda_capture_kernel"] == _identity()
    assert "kda_capture_kernel" not in _arithmetic(fallback)
    # No KDA pass ran, so the numbers are the fallback's.
    assert _rows(payload) == _rows(fallback)


def test_a_kda_pass_that_ran_the_fallback_refuses_at_the_target_and_at_the_roll(
        tmp_path, monkeypatch, glm):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    _mode(monkeypatch, KERNEL)
    # A dispatch whose block runs the fallback: every KDA pass calls no kernel.
    _admit(monkeypatch, glm, entry=glm.fallback)
    # Layer 2 tops band 3 and has KDA: its first capture pass refuses.
    with pytest.raises(capture.KdaCaptureKernelRefused, match="layer 2 target pass"):
        _quantum(tmp_path, monkeypatch, single=single, receipt=receipt,
                 output_root=output_root, layer=2)
    # Layer 3 has no KDA, but its chain rolls layer 4, which has: the roll refuses.
    with pytest.raises(capture.KdaCaptureKernelRefused, match="layer 4 chain pass"):
        _quantum(tmp_path, monkeypatch, single=single, receipt=receipt,
                 output_root=output_root, layer=3)


# ---- #996 within the mode ------------------------------------------------------

@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_band_serial_is_chain_mode_byte_for_byte_within_kernel_mode(
        tmp_path, monkeypatch, glm, fusion):
    from tools.compare_joint_layer_gate import compare_layer

    single, receipt, output_root = _campaign(tmp_path, monkeypatch, fusion=fusion)
    _admit(monkeypatch, glm)
    kept = tmp_path / "chain-spaces"
    kept.mkdir()

    def chain_mode(kernel, rolled=None):
        original = roll_owner.render_free_layer_roll
        current = {}

        def spy(runner, *, layer, roll, cotangents, **kwargs):
            plane = {}

            def keep(tensor, batch, probe):
                plane[(probe, batch)] = tensor.clone()
                roll(tensor, batch, probe)

            backwards = original(runner, layer=layer, roll=keep,
                                 cotangents=cotangents, **kwargs)
            if rolled is not None:
                rolled.setdefault(current["layer"], {})[layer] = (
                    plane, [[owner.state_dict() for owner in row] for row in cotangents])
            return backwards

        _mode(monkeypatch, kernel)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(roll_owner, "render_free_layer_roll", spy)
            for layers in BANDS.values():
                for layer in layers:
                    current["layer"] = layer
                    payload, record, _counters = _quantum(
                        tmp_path, monkeypatch, single=single, receipt=receipt,
                        output_root=output_root, layer=layer)
                    results[layer] = (payload, record)
                    space = Path(record["output_space"]["root"])
                    shutil.move(str(space),
                                str(kept / f"{kernel or 'fallback'}-{space.name}"))
        return results

    fallback = chain_mode(None)
    rolled = {}
    chain = chain_mode(KERNEL, rolled)
    # Not vacuous: every quantum has a KDA layer in its target or its chain,
    # so the kernel moved every row.
    for layer, (payload, _record) in chain.items():
        assert _rows(payload) != _rows(fallback[layer][0]), layer

    handoffs = {}
    with monkeypatch.context() as patch:
        def no_chain(*args, **kwargs):
            raise AssertionError("a band-serial quantum walked the chain")
        patch.setattr(roll_owner, "render_free_layer_roll", no_chain)
        for layers in BANDS.values():
            for position, layer in enumerate(layers):
                emits = position + 1 < len(layers)
                consumes = position > 0
                payload, record, counters = _quantum(
                    tmp_path, monkeypatch, single=single, receipt=receipt,
                    output_root=output_root, layer=layer,
                    adjoint_handoff=(_binder(handoffs[layer + 1], KERNEL)
                                     if consumes else None),
                    handoff_emitter=_emitter if emits else None)
                chain_payload, chain_record = chain[layer]
                assert canonical_json_bytes(record, where="record") == \
                    canonical_json_bytes(chain_record, where="record")
                assert pickle.dumps(payload) == pickle.dumps(chain_payload), layer
                assert counters["chain"]["layers"] == 0
                verdict = compare_layer(
                    kept / f"{KERNEL}-layer-{layer:03d}" / "checkpoints",
                    Path(record["output_space"]["checkpoint_dir"]),
                    layer=layer, qname_filter=None)
                assert verdict["verdict"] == "match", verdict
                if emits:
                    handoffs[layer] = _published(record)
                    document = json.loads(Path(handoffs[layer]["path"]).read_bytes())
                    assert document["producer"]["kda_capture_kernel"] == {
                        "name": KERNEL, "identity_sha256": _identity_sha256(_identity())}

    # Each handoff plane is the plane its consumer's kernel-mode chain ends on.
    assert sorted(handoffs) == [1, 2, 4]
    for producer, handoff in handoffs.items():
        document, plane, states = _handoff_plane(handoff)
        chain_plane, chain_states = rolled[producer - 1][producer]
        assert set(plane) == set(chain_plane)
        for key in sorted(plane):
            assert _tensor_digest(plane[key]) == _tensor_digest(chain_plane[key]), \
                (producer, key)
        assert {key: _state_digest(state) for key, state in states.items()} == {
            (probe, batch): _state_digest(chain_states[probe][batch])
            for probe in range(len(chain_states))
            for batch in range(len(chain_states[probe]))}


# ---- the handoff carries its mode ------------------------------------------------

def _reseal_in_place(path, mutate):
    document = json.loads(path.read_bytes())
    mutate(document)
    document["handoff_sha256"] = handoff_seal_sha256(document)
    raw = handoff_record_bytes(document)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_a_handoff_carries_its_mode_and_a_consumer_in_another_mode_refuses(
        tmp_path, monkeypatch, glm, capsys):
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    _admit(monkeypatch, glm)
    _mode(monkeypatch, KERNEL)
    # Layer 2 is a KDA producer in band 3: the kernel captures the plane it hands off.
    _payload, producer_record, _counters = _quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=2, handoff_emitter=_emitter)
    handoff = _published(producer_record)
    path, sha = Path(handoff["path"]), handoff["sha256"]
    raw = path.read_bytes()
    stamp = {"name": KERNEL, "identity_sha256": _identity_sha256(_identity())}
    assert json.loads(raw)["producer"]["kda_capture_kernel"] == stamp
    record, adjoint_slice = _consumer_record(producer_record, layer=1, receipt=receipt)

    def load(kernel, digest=sha):
        return load_quantum_handoff(path, digest, record=record,
                                    adjoint_slice=adjoint_slice, kda_capture_kernel=kernel)

    assert load(KERNEL)["producer"]["kda_capture_kernel"] == stamp
    # A consumer launched on the fallback refuses a kernel-mode plane.
    with pytest.raises(QuantumHandoffRefused, match="kernel mode"):
        load(None)
    # A handoff naming another kernel refuses under this one.
    digest = _reseal_in_place(path, lambda document: document["producer"][
        "kda_capture_kernel"].update(name="kda_gram_v0"))
    with pytest.raises(QuantumHandoffRefused, match="kda_gram_v0"):
        load(KERNEL, digest)
    # A fallback producer's handoff carries no stamp: the fallback binds it,
    # the kernel refuses it.
    digest = _reseal_in_place(path, lambda document: document["producer"].pop(
        "kda_capture_kernel"))
    assert "kda_capture_kernel" not in load(None, digest)["producer"]
    with pytest.raises(QuantumHandoffRefused, match="fallback"):
        load(KERNEL, digest)
    path.write_bytes(raw)
    # The mode and the kernel's name bind at load, in both modes: another
    # mode is another arithmetic. The identity binds in the core, after
    # admission, and it is a run seal (PQ #1147): a consumer whose admitted
    # kernel is another build prints the difference in dev mode and runs,
    # and refuses before any capture work when certified.
    _admit(monkeypatch, glm, identity=_identity(probe="another runtime"))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    capsys.readouterr()
    payload, consumer_record, _counters = _quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=1, adjoint_handoff=_binder(handoff, KERNEL))
    assert payload is not None
    assert "seal KDA capture kernel handoff differs" in capsys.readouterr().out
    space = Path(consumer_record["output_space"]["root"])
    shutil.move(str(space), str(tmp_path / "dev-mode-consumer"))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(QuantumIdentityRefused, match="KDA capture kernel"):
        _quantum(tmp_path, monkeypatch, single=single, receipt=receipt,
                 output_root=output_root, layer=1,
                 adjoint_handoff=_binder(handoff, KERNEL))
