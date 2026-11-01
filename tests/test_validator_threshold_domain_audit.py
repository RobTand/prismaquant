import math

import pytest

from prismaquant import validate_quantized_model as validator


@pytest.mark.parametrize('name', ['max_ppl', 'max_mean_nll', 'max_p99_nll'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True])
def test_perplexity_refuses_invalid_thresholds(monkeypatch, name, value):
    monkeypatch.setattr(validator, '_spec_decode_on', lambda url: False)
    monkeypatch.setattr(validator, 'EVAL_PROMPTS', ['a'])
    monkeypatch.setattr(validator, '_post_json', lambda *a, **k: {
        'choices': [{'logprobs': {'token_logprobs': [None, -1.0]}}]})
    thresholds = dict(max_ppl=10.0, max_mean_nll=2.0, max_p99_nll=2.0)
    thresholds[name] = value
    result = validator.check_perplexity('http://fixture', 'model', **thresholds)
    assert not result.passed
    assert name in result.detail
    assert 'finite' in result.detail


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True, -0.01, 1.01])
def test_mtp_refuses_invalid_threshold_even_without_drafts(monkeypatch, value):
    monkeypatch.setattr(validator, '_get_text', lambda *a, **k: '')
    result = validator.check_mtp_acceptance('http://fixture', value)
    assert not result.passed
    assert 'min_p0' in result.detail


def test_valid_thresholds_keep_the_recorded_population(monkeypatch):
    monkeypatch.setattr(validator, '_spec_decode_on', lambda url: False)
    monkeypatch.setattr(validator, 'EVAL_PROMPTS', ['a'])
    monkeypatch.setattr(validator, '_post_json', lambda *a, **k: {
        'choices': [{'logprobs': {'token_logprobs': [None, -1.0, -3.0]}}]})
    result = validator.check_perplexity('http://fixture', 'model', 10, 3, 3)
    assert result.passed
    assert result.metrics['n_tokens'] == 2
    assert result.metrics['mean_nll_per_tok'] == 2
