"""Packed resume must release prior modules' routed state before the next score."""
from __future__ import annotations

import weakref

import pytest

from prismaquant import production_weight_cache as pwc
from prismaquant import routed_experts
from test_packed_expert_hook_scope import (
    MODULE_TOKEN_BUDGET, TwoLayerMoe, all_tensor_qnames, frozen_calib, frozen_state,
)


@pytest.mark.parametrize("arm", ["fresh", "full_resume", "render_gate_up", "score_down"])
def test_packed_resume_releases_previous_module_routed_state(monkeypatch, arm):
    model = TwoLayerMoe().eval()
    model.load_state_dict(frozen_state())
    calib = frozen_calib()
    names = all_tensor_qnames()
    assignment = {name: "FP8_E4M3" for name in names}
    cache = pwc.ProductionWeightCache(weights={}, levers={"gptq": False})
    monkeypatch.setattr(
        pwc, "render_production_weight", lambda weight, *args, **kwargs: weight.clone(),
    )

    def fill(selected):
        return pwc.fill_packed_expert_cache_entries(
            cache, model, calib,
            render_assignment=selected,
            levers=cache.levers,
            profile=None,
            module_token_budget=MODULE_TOKEN_BUDGET,
            progress=False,
        )

    if arm == "render_gate_up":
        fill({name: fmt for name, fmt in assignment.items() if name.endswith(".down_proj")})
    elif arm != "fresh":
        fill(assignment)
        if arm == "score_down":
            scores = cache.metadata["render_scores"]
            for name, fmt in assignment.items():
                if name.endswith(".down_proj"):
                    scores["records"].pop(pwc._render_score_record_key(name, fmt))
            scores["entries"] = len(scores["records"])

    module_names = {
        id(model.get_submodule(name.rsplit(".", 1)[0])): name.rsplit(".", 1)[0]
        for name in names
    }
    routed_refs = {}
    scored_modules = []
    real_derive = routed_experts.derive_per_expert_activations
    real_score = pwc._packed_expert_render_score_record

    def derive(experts_mod, *args, **kwargs):
        result = real_derive(experts_mod, *args, **kwargs)
        routed_refs[module_names[id(experts_mod)]] = [
            weakref.ref(rows) for rows in result["gate_weights"] if rows.numel()
        ]
        return result

    def score(**kwargs):
        current = kwargs["qname"].rsplit(".", 1)[0]
        scored_modules.append(current)
        for previous, refs in routed_refs.items():
            if previous != current:
                assert refs and all(ref() is None for ref in refs), (
                    f"{previous} retains routed tensors while scoring {current} "
                    f"under {arm}"
                )
        return real_score(**kwargs)

    monkeypatch.setattr(routed_experts, "derive_per_expert_activations", derive)
    monkeypatch.setattr(pwc, "_packed_expert_render_score_record", score)
    fill(assignment)
    assert set(cache.weights) == {(name, fmt) for name, fmt in assignment.items()}
    if arm == "full_resume":
        assert not routed_refs
        assert not scored_modules
    else:
        assert set(routed_refs) == set(module_names.values())
        assert set(scored_modules) == set(module_names.values())
