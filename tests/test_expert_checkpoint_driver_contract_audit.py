"""RE-06 baseline draft for real journal publication and driver resume."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import cost_stage_checkpoint as journal
from prismaquant import expert_empirical_cost as expert


@pytest.mark.parametrize("driver", ["window", "forked"])
def test_expert_driver_resumes_its_own_published_payload(tmp_path, monkeypatch, driver):
    qname = "model.layers.0.mlp.experts"
    member = qname + ".down_proj"
    state = {
        "stats": {member: {"n_params": 16, "in_features": 4, "out_features": 4}},
        "costs": {member: {"NVFP4": {"predicted_dloss": 0.25,
                                    "cost_source": "empirical_unit_kl",
                                    "output_mse_measured": False}}},
        "unit_kls": {qname: {"NVFP4": 0.25}},
    }
    if driver == "forked":
        state["kl_windows"] = {"NVFP4": [0.2, 0.3]}
    identity = {"schema": "audit.expert_resume.v1", "driver": driver}
    directory, digest, _ = journal.prepare_journal(
        tmp_path / driver, stage=expert.EXPERT_CHECKPOINT_STAGE,
        resume=False, identity=identity, qnames=[qname],
    )
    expert._write_expert_unit_checkpoint(
        directory, qname=qname, identity_sha256=digest, state=state)

    profile = SimpleNamespace(new_forward_pass_state=lambda: {})
    monkeypatch.setattr(expert, "resolve_routed_expert_profile", lambda *a: profile)
    monkeypatch.setattr(expert, "_streamed_expert_unit_records", lambda *a, **k: (
        [("packed", qname, object())], [],
    ))

    def checkpoint_identity(**kwargs):
        assert kwargs["identity_extra"].get("eval_driver") == (
            "forked-stream/1" if driver == "forked" else None)
        return identity

    monkeypatch.setattr(expert, "_expert_checkpoint_identity", checkpoint_identity)

    def refuse_recompute(*args, **kwargs):
        raise AssertionError("completed journal must resume without model execution")

    runner = SimpleNamespace(
        model=nn.Module(), device=torch.device("cpu"), dtype=torch.float32,
        num_layers=1, layer_index_for_qname=lambda name: 0,
        capture_boundaries=refuse_recompute,
    )
    measure = (expert.measure_expert_unit_costs_forked if driver == "forked"
               else expert.measure_expert_unit_costs_streamed)
    actual = measure(
        runner, profile, torch.tensor([[1, 2], [2, 3]]), ["NVFP4", "BF16"],
        checkpoint_dir=directory, resume=True,
        model_identity={"fixture": "identity seam is isolated"}, progress=False,
    )
    assert actual == (state["stats"], state["costs"], state["unit_kls"])
    _, _, restored = journal.prepare_journal(
        directory, stage=expert.EXPERT_CHECKPOINT_STAGE,
        resume=True, identity=identity, qnames=[qname],
    )
    assert restored[qname] == state


@pytest.mark.parametrize("forked", [False, True])
def test_expert_checkpoint_refuses_the_other_driver_payload(forked):
    qname = "model.layers.0.mlp.experts"
    state = {"stats": {}, "costs": {}, "unit_kls": {qname: {"NVFP4": 0.25}}}
    if not forked:
        state["kl_windows"] = {"NVFP4": [0.2, 0.3]}
    with pytest.raises(RuntimeError, match="has fields"):
        expert._validate_expert_checkpoint_state(qname, state, forked=forked)


@pytest.mark.parametrize("windows", [None, [], 0])
def test_forked_checkpoint_requires_a_window_mapping(windows):
    qname = "model.layers.0.mlp.experts"
    state = {"stats": {}, "costs": {}, "unit_kls": {qname: {"NVFP4": 0.25}},
             "kl_windows": windows}
    with pytest.raises(RuntimeError, match="invalid kl_windows"):
        expert._validate_expert_checkpoint_state(qname, state, forked=True)
