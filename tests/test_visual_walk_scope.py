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
        owner = model.get_submodule(visual[0].rsplit('.', 1)[0])
        owner.add_module('norm', nn.LayerNorm(4))
        owner.add_module('embedding', nn.Embedding(4, 4))
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
    prefix = visual[0].rsplit('.', 1)[0]
    assert result.claims[f'{prefix}.norm.weight'].disposition == 'exclude'
    assert result.claims[f'{prefix}.embedding.weight'].disposition == 'exclude'
    unpriced = find_decided_but_unpriced(result, model, profile)
    if case[0] == 'glm5_next':
        # Discovery does not overrule GLM's BF16/source-format serve contract.
        assert {item['node'] for item in unpriced} == {
            f'{name}.weight' for name in visual}
        assert {item['reason_code'] for item in unpriced} == {'probe_linear_excluded'}
    else:
        assert unpriced == ()


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


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_full_gamut_does_not_override_a_profile_pinned_visual_leaf(tmp_path, monkeypatch, case):
    profile, model, visual, _ = _fixture(tmp_path, case)
    original = profile.is_pinned_name
    pinned = f'{visual[0]}.weight'
    monkeypatch.setattr(profile, 'is_pinned_name',
                        lambda name: name == pinned or original(name))
    result = walk_model(model, (torch.zeros(1, 2, 4, device='meta'),),
                        claim_rules=profile.walk_claim_rules(include_visual=True))
    assert result.claims[pinned].disposition == 'pin'
    assert result.claims[f'{visual[1]}.weight'].disposition == 'decide'


def _cli_loader(monkeypatch, model):
    from transformers import AutoConfig, AutoModelForCausalLM
    from prismaquant import model_walk

    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: object())
    monkeypatch.setattr(AutoModelForCausalLM, 'from_config', lambda *a, **kw: model)
    original = model_walk.walk_model
    # Adapt only the synthesized input to this small real operand graph.
    # The real tracer, claims, priceability check and gate still execute.
    monkeypatch.setattr(model_walk, 'walk_model',
                        lambda model, **kw: original(
                            model, (torch.zeros(1, 2, 4, device='meta'),), **kw))
    return model_walk


@pytest.mark.parametrize('include_visual', [False, True])
def test_cli_records_scope_and_keeps_the_independent_gate(tmp_path, monkeypatch, include_visual):
    import json

    _, model, visual, _ = _fixture(tmp_path, CASES[1])
    walker = _cli_loader(monkeypatch, model)
    out = tmp_path / 'scope.json'
    args = ['--model', str(tmp_path), '--output', str(out)]
    if include_visual:
        args.append('--include-visual')
    assert walker.main(args) == 0
    report = json.loads(out.read_text())
    assert report['context']['claim_scope'] == (
        'full_gamut' if include_visual else 'text_artifact')
    assert bool(report['context']['visual_roots']) is include_visual
    assert report['gate']['refused'] is False
    # Legacy Qwen scope excludes the block tower, not its non-block mergers.
    # The opt-in must not quietly change that existing default control.
    expected_visual = visual if include_visual else visual[1:]
    assert report['gate']['claims_by_disposition']['decide'] == 1 + len(expected_visual)


@pytest.mark.parametrize('override', [None, 'trace override cannot admit GLM vision'])
def test_full_gamut_cli_does_not_waive_the_profile_priceability_gate(tmp_path, monkeypatch, override):
    import json

    _, model, visual, _ = _fixture(tmp_path, CASES[0])
    walker = _cli_loader(monkeypatch, model)
    out = tmp_path / 'refused.json'
    args = ['--model', str(tmp_path), '--output', str(out), '--include-visual']
    if override:
        args.extend(['--override-reason', override])
    assert walker.main(args) == 2
    report = json.loads(out.read_text())
    assert report['context']['claim_scope'] == 'full_gamut'
    assert report['gate']['refused'] is True
    assert report['gate']['claims_by_disposition']['decide'] == 1 + len(visual)
    assert report['gate']['refusal_kinds'] == ['decided_but_unpriced_node']
    assert {item['node'] for item in report['gate']['decided_but_unpriced_nodes']} == {
        f'{name}.weight' for name in visual}


def test_cli_cannot_excuse_a_missing_declared_visual_root(tmp_path, monkeypatch, capsys):
    _config(tmp_path, CASES[0][0], CASES[0][1])
    with torch.device('meta'):
        model = OperandTree(['model.layers.0.mlp.up_proj', 'lm_head'])
    walker = _cli_loader(monkeypatch, model)
    with pytest.raises(SystemExit) as exc:
        walker.main(['--model', str(tmp_path), '--include-visual',
                     '--override-reason', 'this cannot excuse missing scope'])
    assert exc.value.code == 2
    assert 'missing declared vision roots: model.visual' in capsys.readouterr().err


@pytest.mark.parametrize('rules', ['profile', 'none'])
def test_cli_rejects_full_gamut_without_a_declared_scope(tmp_path, capsys, rules):
    from prismaquant.model_walk import main

    _config(tmp_path, 'qwen3', 'Qwen3ForCausalLM')
    with pytest.raises(SystemExit) as exc:
        main(['--model', str(tmp_path), '--include-visual', '--rules', rules])
    assert exc.value.code == 2
    assert 'requires profile rules and declared vision roots' in capsys.readouterr().err
