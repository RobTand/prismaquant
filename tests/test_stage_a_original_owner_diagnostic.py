"""Source admission/plumbing on CPU; positive device seams are explicitly stubbed."""
from __future__ import annotations

import pytest

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_checkpoints import adjoint_space, adjoint_receipt_path
from prismaquant.stage_a_selected_row_diagnostic import diagnostic_marker_path, RECEIPT_NAME

from test_capture_original_material import material, _owner, _forget_state  # noqa: F401
from test_layer_major_boundary_capture import draw
from test_stage_a_chain_resume import _dense_runner, ONE
from test_stage_a_head_skip import _campaign, _config, _entry
from test_stage_a_selected_row_diagnostic import _spec, _diagnostic_dev_mode  # noqa: F401

pytestmark = pytest.mark.own_process


def test_core_uses_context_owner_without_invented_model_path(material, monkeypatch):
    from test_stage_a_selected_row_diagnostic import _execution_one
    from test_stage_a_chain_resume import _run
    import torch
    owner = _owner(material)
    try:
        def runner_factory():
            runner = _dense_runner()
            runner.device = torch.device("cuda")
            runner.context.source_authentication = owner
            return runner
        root = material["tmp"] / "uncreated"
        execution = _execution_one(root)
        with pytest.raises(stage_a.AdjointIdentityRefused, match="GPU loads/transfers are not qualified"):
            _run(root, monkeypatch, execution=execution, runner_factory=runner_factory,
                 selected_row_diagnostic=_spec(draw(), execution))
        assert not root.exists()
    finally:
        owner.close()


def test_actual_original_owner_still_refuses_cuda_before_reads(material, monkeypatch):
    from prismaquant import gpu_guard
    owner = _owner(material)
    try:
        monkeypatch.setattr(gpu_guard, "require_cuda_hot_path",
                            lambda *a: pytest.fail("CUDA path reached before source device refusal"))
        with pytest.raises(stage_a.AdjointIdentityRefused, match="GPU loads/transfers are not qualified"):
            stage_a.run_adjoint_capture(
                {"model": str(owner.root)}, plan_sha256="1" * 64,
                prepared={}, output_root=material["tmp"] / "uncreated",
                source_authentication=owner,
                selected_row_diagnostic=_spec(draw(), {"seed_base": 7000}))
        assert not (material["tmp"] / "uncreated").exists()
    finally:
        owner.close()


def test_enclosing_api_threads_same_owner_and_owned_profile(material, monkeypatch):
    """Only wiring is tested: CUDA admission/backend/model calls are CPU stubs."""
    from prismaquant import cost_streaming, layer_streaming, model_profiles
    owner = _owner(material)
    try:
        campaign = _campaign(material["tmp"] / "campaign", implementation=ONE)
        root = material["tmp"] / "diagnostic"
        config = _config(campaign, root)
        config["model"] = str(owner.root)
        config["execution"]["n_probes"] = 1
        config["execution"]["probe_microbatch"] = 1
        _entry(monkeypatch, implementation=ONE, walk=False, generation=1030)
        observed = []
        # Explicit test-local device gate override; no actual CUDA load or qualification.
        monkeypatch.setattr(owner, "require_material_device", lambda device: observed.append(device))
        profile = _dense_runner().profile
        def owned_profile(model, authority):
            assert model == str(owner.root) and authority is owner
            return profile
        monkeypatch.setattr(layer_streaming, "_source_profile", owned_profile)
        monkeypatch.setattr(model_profiles, "detect_profile",
                            lambda *a, **kw: pytest.fail("mutable path profile discovery used"))
        def build(model, **kw):
            assert kw["source_authentication"] is owner
            assert kw["profile"] is profile
            return _dense_runner()
        monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
        result = stage_a.run_adjoint_capture(
            config, plan_sha256="b" * 64, prepared=campaign.prepared,
            output_root=root, stride=2, source_authentication=owner,
            selected_row_diagnostic=_spec(draw(), config["execution"]))
        assert result["passed"] is True
        assert observed == ["cuda"]
        assert diagnostic_marker_path(adjoint_space(root)).is_file()
        assert (adjoint_space(root) / RECEIPT_NAME).is_file()
        assert not adjoint_receipt_path(adjoint_space(root)).exists()
        assert result["diagnostic_receipt"]["sha256"]
    finally:
        owner.close()
