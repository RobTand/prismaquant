"""The no-slice Stage B head walk honours the plan's encoder reuse (PQ #1023).

A record without a head slice runs the whole-journal intake inside the
quantum. That call omitted ``historical_encoder_reuse``, so a plan that admits
a historical encoder seal (the GLM plan does: the Stage B pin's seal is not
the checkpoint's) was admitted by Stage A, the prepare and the slice producer
and refused only here. The walk now passes the plan's allowance, like every
other intake of the same journal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _Stop(Exception):
    pass


POLICY = {"schema": "prismaquant.tessera_joint_aura.historical_encoder_reuse.v1",
          "fixture": True}


@pytest.fixture
def legacy_walk(tmp_path, monkeypatch):
    """``run_layer_quantum`` up to its no-slice intake, GPU body stubbed."""
    import prismaquant.gpu_guard as gpu_guard
    import prismaquant.joint_cost_quantum as quantum
    import prismaquant.joint_projection_backend as backend
    import prismaquant.tessera_joint_aura as aura

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(backend, "executing_image", lambda: "fixture-image")
    monkeypatch.setattr(backend, "prewarm_projection_backend",
                        lambda *a, **k: SimpleNamespace(identity={"name": "torch"}))
    monkeypatch.setattr(quantum, "bind_joint_served_quantizer", lambda formats: {})
    monkeypatch.setattr(aura, "_preflight_run_prepared", lambda *a, **k: None)
    monkeypatch.setenv(aura.ACTIVATION_SCALE_ENV, "0")
    seen = []

    def intake(inputs, **kwargs):
        seen.append(kwargs)
        raise _Stop()

    monkeypatch.setattr(aura, "load_measured_anchor_input", intake)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"formats_by_qname": {}}))
    record = {"layer": 1, "quantum_id": "layer-001", "identity_sha256": "a" * 64,
              "campaign": {}, "adjoint": {"slice_sha256": "b" * 64},
              "output_space": {"root": str(tmp_path / "space")}}
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32

    def run(config):
        with pytest.raises(_Stop):
            quantum.run_layer_quantum(
                config, record=record, adjoint_slice={}, plan_sha256="c" * 64,
                prepared={"path": str(prepared),
                          "sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()},
                output_root=tmp_path)
        return seen[-1]

    try:
        yield run
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _config(**extra):
    return {"model": "fixture", "inputs": {}, "execution": {"production_act_scales": "0"},
            **extra}


def test_legacy_walk_passes_the_plan_reuse_policy(legacy_walk):
    kwargs = legacy_walk(_config(historical_encoder_reuse=POLICY))
    assert kwargs["historical_encoder_reuse"] == POLICY


def test_legacy_walk_without_a_policy_stays_strict(legacy_walk):
    kwargs = legacy_walk(_config())
    assert kwargs["historical_encoder_reuse"] is None
