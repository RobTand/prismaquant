"""Actual CPU original owners; no original CUDA/provider admission (PQ #2143)."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import joint_cost_quantum as quantum
from prismaquant import model_profiles
from test_capture_original_material import _owner, material  # noqa: F401
from test_original_streaming_bootstrap_2010 import original_model  # noqa: F401
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


def _config(model):
    return {
        "model": str(model), "execution": {"source_derivative": None},
        "source_prefetch": {
            "max_cache_slots": 2, "prefetch_workers": 1,
            "prefetch_lookahead": 1, "cache_headroom_gb": 1,
            "prefetch_min_available_gb": 1, "require_prefetched_residency": True,
        },
    }


def test_actual_original_predicate_precedes_every_profile_and_pool_read(material, monkeypatch):
    with _owner(material) as owner:
        before = list(material["checks"])
        forbidden = {owner.root / "config.json", owner.root / "model.safetensors.index.json"}
        opened = []
        original_open = Path.open

        def open_path(path, *args, **kwargs):
            opened.append(path)
            if path in forbidden:
                pytest.fail("mutable logical source metadata opened before original device refusal")
            return original_open(path, *args, **kwargs)

        def profile(*args, **kwargs):
            pytest.fail("profile discovery reached before original device refusal")

        monkeypatch.setattr(Path, "open", open_path)
        monkeypatch.setattr(model_profiles, "detect_profile", profile)
        with pytest.raises(RuntimeError, match="GPU loads/transfers are not qualified"):
            quantum.build_quantum_source_runner(
                {"model": str(owner.root)}, offload_folder="uncreated",
                source_authentication=owner)
        assert not forbidden.intersection(opened)
        assert material["checks"] == before
        assert owner.material_live_bytes == 0
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("owned", [True, False])
def test_actual_cpu_builder_preserves_owned_and_legacy_profile_intake(original_model, monkeypatch, owned):
    """Only the quantum's device construction is CPU; the owner's predicate is real."""
    fixture = original_model
    owner = _owner(fixture) if owned else None
    model = owner.root if owned else Path(fixture["paths"]["config.json"]).parent
    expected_config = json.loads(fixture["raws"]["config.json"])
    profiles = []
    original_detect = model_profiles.detect_profile

    def detect(path, *, config=None):
        if owned:
            assert config == expected_config
        else:
            assert config is None
        profile = original_detect(path, config=config)
        profiles.append(profile)
        return profile

    # This test-only CPU seam does not patch require_material_device or
    # fabricate _original. The real builder and sealed native owner run on CPU.
    monkeypatch.setattr(quantum, "torch", SimpleNamespace(
        device=lambda _device: torch.device("cpu"), bfloat16=torch.bfloat16))
    monkeypatch.setattr(model_profiles, "detect_profile", detect)
    runner = None
    try:
        runner = quantum.build_quantum_source_runner(
            _config(model), offload_folder=fixture["tmp"] / "quantum-offload",
            source_authentication=owner)
        assert runner.context.source_authentication is owner
        assert runner.profile is profiles[0]
        assert runner.model.config.model_type == "llama"
        assert runner.device.type == "cpu"
        runner.context.install(0)
        for name, value in runner.model.state_dict().items():
            if not value.is_meta:
                assert torch.equal(value, fixture["expected"][name].to(value.dtype)), name
        del value
        assert not torch.cuda.is_initialized()
    finally:
        if runner is not None:
            runner.shutdown()
        runner = None
        gc.collect()
        if owner is not None:
            owner.close()
