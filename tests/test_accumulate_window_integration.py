"""Installed Arm-A callback integration, not a numerical experiment receipt.

Use a tiny random MTP layer with 710 synthetic aliases of an unchanged weight.
The real MTP backward, planner, statistics lease, prewarmed torch projection and
report writer run on CPU; only GPU telemetry and the priced roster are replaced.
"""
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from experiments import accumulate_window_research as research
from prismaquant import format_registry as fr, glm_mtp, glm_mtp_quantum as quantum
from prismaquant import joint_statistics_replay as replay
from prismaquant.joint_aura import activation_identity
from prismaquant.joint_projection_backend import prewarm_projection_backend
from prismaquant.production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity
from tests.test_glm_mtp_quantum import _Env, _source_model


@pytest.mark.parametrize("published_total,valid", [(0.0, True), (1.0, False), (-0.0, False)],
                         ids=["matching", "different-value", "different-sign-bit"])
def test_installed_mtp_callback_finishes_arm_a(tmp_path, monkeypatch, published_total, valid):
    env = _Env(tmp_path)
    name = next(name for name in env.modules if "shared_experts.gate_proj" in name)
    module = env.modules[name]
    original_spec = fr.get_format("TESSERA_BF16_K1_R1024")
    specs = {f"synthetic_{i}": replace(original_spec, name=f"synthetic_{i}") for i in range(710)}
    get_format = fr.get_format
    monkeypatch.setattr(fr, "get_format", lambda fmt: specs[fmt] if fmt in specs else get_format(fmt))
    monkeypatch.setattr(quantum, "mtp_priced_modules", lambda *_: {name: module})

    # All 710 unchanged renders have analytically zero weight/activation/mixed
    # terms. Keep real cache tensor identities and admission, not a helper mock.
    path = tmp_path / "unchanged.pt"
    torch.save(module.weight.detach().clone(), path)
    identity = _cb_cache_tensor_identity(module.weight.detach())
    cache = ProductionWeightCache({(name, fmt): str(path) for fmt in specs}, {},
                                  activation_max_abs={name: 1.0}, metadata={})
    cache.metadata["verified_cells"] = {
        (name, fmt): {"source_weight": identity, "rendered_weight": identity,
                      "activation": activation_identity(spec, cache.activation_max_abs, name)}
        for fmt, spec in specs.items()}
    reference = tmp_path / "reference.pkl"
    cells = {fmt: {"signed_components_per_probe": [{"total": 0.0}] * 4,
                   "signed_per_probe": [0.0] * 4} for fmt in specs}
    cells[next(iter(specs))] = {"signed_components_per_probe": [{"total": published_total}] * 4,
                               "signed_per_probe": [published_total] * 4}
    reference.write_bytes(pickle.dumps({"costs": {name: cells}}))
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setattr(research, "STATE", {"phases": [], "arms": {}})
    monkeypatch.setattr(research, "_sync", lambda: None)
    monkeypatch.setattr(research, "_cuda_mem", lambda: {})
    monkeypatch.setattr(research, "_netdata_power", lambda *_: {"attempted": False})
    monkeypatch.setattr(research, "_commit_progress", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    calls = []
    probe_scalar = glm_mtp.mtp_probe_scalar

    def recorded_scalar(*args, **kwargs):
        calls.append((kwargs["global_row_offset"], kwargs["n_sequences"], kwargs["seed"]))
        return probe_scalar(*args, **kwargs)

    def completed(code):
        raise SystemExit(code)

    monkeypatch.setattr(glm_mtp, "mtp_probe_scalar", recorded_scalar)
    monkeypatch.setattr(research.os, "_exit", completed)
    # Register both overwritten owners with monkeypatch so install is reversible.
    monkeypatch.setattr(quantum, "mtp_backward", quantum.mtp_backward)
    monkeypatch.setattr(replay, "observe_and_project_windows", replay.observe_and_project_windows)
    research.install(SimpleNamespace(out=out, window=0, max_sequences=None, arms="A",
                                     buffer_sizes=[], max_buffer_bytes=0, reference_cost=reference))
    backend = prewarm_projection_backend({"name": "torch"}, device="cpu")
    options = dict(env.windows, max_candidate_bytes=1 << 24,
                   max_render_resident_bytes=1 << 24)
    exception = SystemExit if valid else RuntimeError
    with pytest.raises(exception) as caught:
        quantum.compute_mtp_cost(
            env.model, env.embed, env.head, env.ids, lambda i: env.hidden[i:i + 1],
            production_cache=cache, formats_by_qname={name: list(specs)},
            wire_bytes={name: {fmt: 1 for fmt in specs}},
            census={"anchor_groups": {"synthetic": [name]}}, profile=None,
            n_probes=4, seed_base=7000, source_model=_source_model(),
            projection_backend=backend, operator_windows=options, device="cpu")
    if valid:
        assert caught.value.code == 0
    else:
        assert "substitution is invalid" in str(caught.value)
    report = json.loads((out / "report.json").read_text())
    assert report["backend_substitution_valid"] is valid
    assert report["reference_check"]["compared"] == 710
    assert report["reference_check"]["bitwise_equal"] == (710 if valid else 709)
    arm = report["arms"]["A"]
    assert arm["window_sequences"] == len(env.ids)
    assert len(arm["term_totals"]) == 710
    assert arm["window_s"] >= 0
    assert calls == [(i, len(env.ids), 7000) for i in range(len(env.ids))]
    assert not module._forward_hooks
