"""The KDA capture kernel is the Stage B default (PQ #1252).

An unset ``PRISMAQUANT_STAGE_B_KDA_KERNEL`` used to run the image's Torch
fallback. It now resolves at launch (``launch_kda_capture_kernel``):

* a band-serial consumer takes its producer's mode from the handoff's stamp,
  so a band already in flight on the fallback stays on it;
* any other quantum runs ``kda_gram_v1`` in kernel mode (PQ #1214) when a
  decoder layer of the model has KDA attention, and the fallback when none
  does.

``fallback`` selects the fallback explicitly and stamps nothing, so its rows
are an unset launch's from before PQ #1252, byte for byte.

The resolver tests run everywhere. The quantum tests run the five-layer
band-serial CPU campaign of ``tests/test_kda_kernel_mode.py`` (KDA attention
on layers 1, 2 and 4), whose kernel-mode layer pass reads the counters of
``prismaquant.kernels.kda_chunk``; they skip where Triton is not installed,
as that file does. Each quantum goes through the core with the execution the
launcher builds: the fixture's execution plus the kernel
``launch_kda_capture_kernel`` resolves from the fixture's runner and the
bound handoff, as ``run_layer_quantum`` passes it to
``quantum_runtime_execution``.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil

import pytest
import torch

from prismaquant import glm_kda_capture_kernel as capture
from prismaquant import joint_cost_quantum as quantum_module
from prismaquant.cost_stage_checkpoint import canonical_json_bytes
from prismaquant.joint_cost_quantum import quantum_runtime_execution
from prismaquant.joint_quantum_handoff import (
    QuantumHandoffRefused,
    handoff_kernel_refusal,
    load_quantum_handoff,
)

import test_joint_cost_quantum_runtime as runtime
from test_kda_kernel_mode import (  # noqa: F401 - glm is a fixture
    KERNEL,
    _admit,
    _arithmetic,
    _identity,
    _identity_sha256,
    _published,
    _rows,
    glm,
    kda_chunk,
)
from test_quantum_band_serial import _campaign, _emitter, _quantum

needs_triton = pytest.mark.skipif(
    kda_chunk is None,
    reason="prismaquant.kernels.kda_chunk defines its kernels with Triton at import")

STAMP = {"name": KERNEL, "identity_sha256": "e" * 64}


class _KdaAttention(torch.nn.Module):
    """A module of the GLM KDA attention class."""


_KdaAttention.__module__, _KdaAttention.__name__ = capture.KDA_ATTENTION_CLASS


def _layers(*kda):
    layers = []
    for has_kda in kda:
        layer = torch.nn.Module()
        layer.proj = torch.nn.Linear(2, 2)
        if has_kda:
            layer.attention = _KdaAttention()
        layers.append(layer)
    return layers


def _runner(*kda):
    layers = _layers(*kda)
    return SimpleNamespace(layers=layers, num_layers=len(layers))


def _set(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv(capture.KDA_KERNEL_ENV, raising=False)
    else:
        monkeypatch.setenv(capture.KDA_KERNEL_ENV, setting)


def _launcher_kernel(runner, adjoint_handoff):
    """The kernel ``run_layer_quantum`` hands ``quantum_runtime_execution``.

    Read from this process's environment, as the launcher reads it. Before PQ
    #1252 the launcher passed the setting itself, unset being the fallback;
    that branch lets these tests run, red, on a checkout without the default.
    """
    launch = getattr(quantum_module, "launch_kda_capture_kernel", None)
    if launch is None:
        return capture.kda_capture_kernel_from_environment(os.environ)
    return launch(capture.kda_capture_kernel_setting(os.environ), runner=runner,
                  adjoint_handoff=adjoint_handoff)


def _consumer_mode(setting):
    """The mode a consumer launched with ``setting`` loads its handoff in."""
    mode = getattr(capture, "consumer_handoff_mode", None)
    if mode is None:  # before PQ #1252: the setting, unset being the fallback
        return setting
    return mode(setting)


def _launched(monkeypatch, setting, runner, adjoint_handoff=None):
    _set(monkeypatch, setting)
    return _launcher_kernel(runner, adjoint_handoff)


# ---- the resolution --------------------------------------------------------------

def test_unset_selects_the_kernel_on_a_model_with_kda_attention(monkeypatch, capsys):
    assert _launched(monkeypatch, None, _runner(False, True, False)) == KERNEL
    assert "KDA capture: kda_gram_v1" in capsys.readouterr().out
    assert capture.DEFAULT_KERNEL == KERNEL


def test_unset_on_a_model_without_kda_attention_changes_nothing(monkeypatch):
    assert _launched(monkeypatch, None, _runner(False, False)) is None
    config = {"execution": {"a": 1}, "max_gpu_bytes": 7, "min_free_gib": 3}
    today = quantum_runtime_execution(config, replay_regime=None)
    assert "kda_capture_kernel" not in today
    assert json.dumps(quantum_runtime_execution(
        config, replay_regime=None,
        kda_capture_kernel=_launched(monkeypatch, None, _runner(False))),
        sort_keys=True) == json.dumps(today, sort_keys=True)


def test_fallback_selects_the_fallback_and_an_explicit_kernel_is_what_it_names(monkeypatch):
    kda = _runner(True, True)
    config = {"execution": {"a": 1}, "max_gpu_bytes": 7}
    assert json.dumps(quantum_runtime_execution(
        config, replay_regime=None,
        kda_capture_kernel=_launched(monkeypatch, "fallback", kda)), sort_keys=True) == \
        json.dumps(quantum_runtime_execution(config, replay_regime=None), sort_keys=True)
    # An explicit setting is not overridden by a handoff: the handoff check refuses instead.
    assert _launched(monkeypatch, "fallback", kda,
                     {"producer": {"kda_capture_kernel": STAMP}}) is None
    assert _launched(monkeypatch, KERNEL, _runner(False)) == KERNEL


def test_an_unset_consumer_follows_its_producers_mode(monkeypatch, capsys):
    kda = _runner(True, True)
    fallback_producer = {"producer": {"layer": 2}}
    kernel_producer = {"producer": {"layer": 2, "kda_capture_kernel": STAMP}}
    # A band in flight on the fallback stays on it, although the model has KDA.
    assert _launched(monkeypatch, None, kda, fallback_producer) is None
    assert "the band-serial producer's mode" in capsys.readouterr().out
    assert _launched(monkeypatch, None, kda, kernel_producer) == KERNEL
    # The consumer's handoff check: unset follows either mode; an explicit one
    # refuses the other mode and names the setting that binds it.
    follow = _consumer_mode(None)
    assert _consumer_mode("fallback") is None
    assert _consumer_mode(KERNEL) == KERNEL
    for producer in (fallback_producer, kernel_producer):
        assert handoff_kernel_refusal(producer["producer"], follow) is None
    refusal = handoff_kernel_refusal(fallback_producer["producer"], KERNEL)
    assert "one mode" in refusal and f"{capture.KDA_KERNEL_ENV} unset" in refusal
    assert "set it to fallback" in refusal
    refusal = handoff_kernel_refusal(kernel_producer["producer"], None)
    assert "one mode" in refusal and f"set it to {KERNEL}" in refusal
    # A malformed stamp refuses whatever the mode.
    assert "not a name and an identity digest" in handoff_kernel_refusal(
        {"kda_capture_kernel": {"name": KERNEL}}, follow)


# ---- the quantum, through the core -------------------------------------------------

def _launch_through_the_core(monkeypatch, setting):
    """Run each later fixture quantum with the kernel the launcher resolves for ``setting``."""
    _set(monkeypatch, setting)
    core = runtime.run_layer_quantum_core
    resolved = []

    def launched(runner, *args, execution, adjoint_handoff=None, **kwargs):
        kernel = _launcher_kernel(runner, adjoint_handoff)
        resolved.append(kernel)
        if kernel is not None:
            execution = {**execution, "kda_capture_kernel": kernel}
        return core(runner, *args, execution=execution,
                    adjoint_handoff=adjoint_handoff, **kwargs)

    monkeypatch.setattr(runtime, "run_layer_quantum_core", launched)
    return resolved


def _bind(handoff, setting):
    def bind(record, adjoint_slice):
        return load_quantum_handoff(handoff["path"], handoff["sha256"], record=record,
                                    adjoint_slice=adjoint_slice,
                                    kda_capture_kernel=_consumer_mode(setting))
    return bind


def _move(record, tmp_path, name):
    space = Path(record["output_space"]["root"])
    shutil.move(str(space), str(tmp_path / name))


def _identity_bytes(payload):
    return canonical_json_bytes(payload["provenance"]["probe_identity"], where="probe identity")


@needs_triton
def test_unset_runs_kernel_mode_on_a_kda_model_and_fallback_reproduces_today(
        tmp_path, monkeypatch, glm):
    """Layer 3 has no KDA and rolls layer 4, which has; layer 2 has KDA."""
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    admitted = _admit(monkeypatch, glm)
    common = dict(single=single, receipt=receipt, output_root=output_root)
    # Today's unset launch: the fixture's execution, no kernel key.
    today, record, today_counters = _quantum(tmp_path, monkeypatch, layer=3, **common)
    assert record["adjoint"]["chain_layers"] == [4]
    assert "kda_capture_kernel" not in today_counters
    _move(record, tmp_path, "today")
    # fallback: the same identity, byte for byte, and the same rows.
    _launch_through_the_core(monkeypatch, "fallback")
    fallback, record, counters = _quantum(tmp_path, monkeypatch, layer=3, **common)
    assert "kda_capture_kernel" not in counters and admitted == []
    assert _identity_bytes(fallback) == _identity_bytes(today)
    assert _rows(fallback) == _rows(today)
    _move(record, tmp_path, "fallback")
    # Unset: kernel mode. The target has no KDA, so the mode is the seal, and
    # the chain roll of layer 4 runs the kernel (PQ #1214's rule).
    glm.events.clear()
    resolved = _launch_through_the_core(monkeypatch, None)
    default, record, counters = _quantum(tmp_path, monkeypatch, layer=3, **common)
    assert resolved == [KERNEL] and len(admitted) == 1
    assert _arithmetic(default)["kda_capture_kernel"] == _identity()
    block = counters["kda_capture_kernel"]
    assert block["passes"] == 0 and block["chain"]["layers"] == [4]
    assert block["chain"]["calls"] > 0 and glm.in_core("fallback") == 0
    assert _rows(default) != _rows(today)
    _move(record, tmp_path, "default-3")
    # A KDA target runs its capture passes on the kernel.
    default, record, counters = _quantum(tmp_path, monkeypatch, layer=2, **common)
    assert counters["kda_capture_kernel"]["passes"] > 0
    assert _arithmetic(default)["kda_capture_kernel"] == _identity()


@needs_triton
def test_unset_on_a_model_without_kda_runs_nothing_new(tmp_path, monkeypatch, glm):
    glm.kda_layers = frozenset()
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    admitted = _admit(monkeypatch, glm)
    common = dict(single=single, receipt=receipt, output_root=output_root)
    today, record, _counters = _quantum(tmp_path, monkeypatch, layer=3, **common)
    _move(record, tmp_path, "today")
    resolved = _launch_through_the_core(monkeypatch, None)
    default, _record, counters = _quantum(tmp_path, monkeypatch, layer=3, **common)
    assert resolved == [None] and admitted == []
    assert "kda_capture_kernel" not in counters
    assert _identity_bytes(default) == _identity_bytes(today)
    assert _rows(default) == _rows(today)


@needs_triton
def test_an_in_flight_fallback_band_stays_on_the_fallback(tmp_path, monkeypatch, glm):
    """Band 3 is [2, 1, 0]. Its producer ran on the fallback before the default moved."""
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    admitted = _admit(monkeypatch, glm)
    common = dict(single=single, receipt=receipt, output_root=output_root)
    # Today's unset launch of the producer: a fallback handoff, no stamp.
    _payload, producer, _counters = _quantum(
        tmp_path, monkeypatch, layer=2, handoff_emitter=_emitter, **common)
    handoff = _published(producer)
    assert "kda_capture_kernel" not in json.loads(
        Path(handoff["path"]).read_bytes())["producer"]
    # An explicit kernel consumer refuses, naming the settings that bind it.
    _launch_through_the_core(monkeypatch, KERNEL)
    with pytest.raises(QuantumHandoffRefused,
                       match="unset to follow the producer, or set it to fallback"):
        _quantum(tmp_path, monkeypatch, layer=1, adjoint_handoff=_bind(handoff, KERNEL),
                 **common)
    # Unset: the consumer follows its producer onto the fallback, although
    # its target has KDA, and hands the fallback on to layer 0.
    glm.events.clear()
    resolved = _launch_through_the_core(monkeypatch, None)
    payload, consumer, counters = _quantum(
        tmp_path, monkeypatch, layer=1, adjoint_handoff=_bind(handoff, None),
        handoff_emitter=_emitter, **common)
    assert resolved == [None] and admitted == []
    assert "kda_capture_kernel" not in counters
    assert "kda_capture_kernel" not in _arithmetic(payload)
    assert glm.in_core("kernel") == 0 and glm.in_core("fallback") > 0
    onward = _published(consumer)
    assert "kda_capture_kernel" not in json.loads(
        Path(onward["path"]).read_bytes())["producer"]


@needs_triton
def test_an_unset_band_serial_band_runs_in_kernel_mode(tmp_path, monkeypatch, glm):
    """A band-serial producer launched unset on a KDA model stamps the kernel."""
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    admitted = _admit(monkeypatch, glm)
    common = dict(single=single, receipt=receipt, output_root=output_root)
    resolved = _launch_through_the_core(monkeypatch, None)
    _payload, producer, _counters = _quantum(
        tmp_path, monkeypatch, layer=2, handoff_emitter=_emitter, **common)
    assert resolved == [KERNEL]
    handoff = _published(producer)
    stamp = {"name": KERNEL, "identity_sha256": _identity_sha256(_identity())}
    assert json.loads(Path(handoff["path"]).read_bytes())["producer"][
        "kda_capture_kernel"] == stamp
    # An explicit fallback consumer refuses, naming the setting that binds it.
    _launch_through_the_core(monkeypatch, "fallback")
    with pytest.raises(QuantumHandoffRefused,
                       match=f"unset to follow the producer, or set it to {KERNEL}"):
        _quantum(tmp_path, monkeypatch, layer=1,
                 adjoint_handoff=_bind(handoff, "fallback"), **common)
    # Unset: the consumer follows its producer's kernel mode.
    resolved = _launch_through_the_core(monkeypatch, None)
    payload, _consumer, counters = _quantum(
        tmp_path, monkeypatch, layer=1, adjoint_handoff=_bind(handoff, None), **common)
    assert resolved == [KERNEL] and len(admitted) == 2
    assert counters["kda_capture_kernel"]["passes"] > 0
    assert _arithmetic(payload)["kda_capture_kernel"] == _identity()
