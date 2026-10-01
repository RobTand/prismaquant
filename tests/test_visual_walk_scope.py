"""Full-gamut discovery includes real vision operands without moving A8S defaults."""

import pytest
import torch
from torch import nn

from prismaquant.model_profiles.registry import detect_profile
from prismaquant.model_walk import find_decided_but_unpriced, walk_model
from test_visual_shard_coverage import CASES, _config


class OperandTree(nn.Module):
    def __init__(self, names):
        super().__init__()
        self.names = names
        for name in names:
            owner = self
            parts = name.split('.')
            for part in parts[:-1]:
                if part not in owner._modules:
                    owner.add_module(part, nn.Module())
                owner = owner._modules[part]
            owner.add_module(parts[-1], nn.Linear(4, 4, bias=False))

    def forward(self, x):
        result = x
        for name in self.names:
            result = result + self.get_submodule(name)(x)
        return result


def _fixture(tmp_path, case):
    model_type, architecture, block_prefix, extras = case
    profile = detect_profile(_config(tmp_path, model_type, architecture))
    visual = [f'{block_prefix}.0.attn.qkv', *extras]
    names = ['model.layers.0.mlp.up_proj', 'lm_head', *visual]
    mtp = profile.mtp_source_prefix()
    mtp_name = f'{mtp.rstrip(".")}.mlp.up_proj' if mtp else None
    if mtp_name:
        names.append(mtp_name)
    with torch.device('meta'):
        model = OperandTree(names).eval()
    return profile, model, visual, mtp_name


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_full_gamut_walk_claims_each_real_visual_and_merger_operand(tmp_path, case):
    profile, model, visual, mtp = _fixture(tmp_path, case)
    result = walk_model(model, (torch.zeros(1, 2, 4, device='meta'),),
                        claim_rules=profile.walk_claim_rules(include_visual=True))
    assert result.ok
    for name in visual:
        node = f'{name}.weight'
        assert result.edges_for(node), f'{node} was not used by the traced graph'
        assert result.claims[node].disposition == 'decide'
    assert result.claims['lm_head.weight'].disposition == 'pin'
    assert result.claims['model.layers.0.mlp.up_proj.weight'].disposition == 'decide'
    if mtp:
        assert result.claims[f'{mtp}.weight'].disposition == 'exclude'
    assert find_decided_but_unpriced(result, model, profile) == ()


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_default_discovery_scope_remains_the_explicit_text_artifact_control(tmp_path, case):
    profile, model, visual, mtp = _fixture(tmp_path, case)
    result = walk_model(model, (torch.zeros(1, 2, 4, device='meta'),),
                        claim_rules=profile.walk_claim_rules())
    explicit = walk_model(model, (torch.zeros(1, 2, 4, device='meta'),),
                          claim_rules=profile.walk_claim_rules(include_visual=False))
    assert result.ok and explicit.ok
    assert result.claims == explicit.claims
    assert result.claims[f'{visual[0]}.weight'].disposition == 'exclude'
    assert result.claims['lm_head.weight'].disposition == 'pin'
    if mtp:
        assert result.claims[f'{mtp}.weight'].disposition == 'exclude'
