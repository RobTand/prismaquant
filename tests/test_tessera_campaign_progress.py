"""Adaptive scheduler progress and durable resume, with controlled producer calls."""
from collections import Counter
import copy
import json
import pickle
from types import SimpleNamespace

import pytest

from test_tessera_campaign_resume import _main_fixture, UNIT


OTHER = 'model.layers.0.proj2'
FAMILY = 'TESSERA_E4M3_K1'


def fixture(monkeypatch, tmp_path):
    campaign, checkpoint, argv, model, inputs = _main_fixture(monkeypatch, tmp_path, priced=True)
    model.model.layers[0].proj2 = copy.deepcopy(model.model.layers[0].proj)
    inputs['menu'] = [SimpleNamespace(format_name=f'{FAMILY}_R{r}', family=FAMILY,
        body_rate_q256=r, bpp=r/256, admission=SimpleNamespace(activation_contract='a8'))
        for r in (1024, 1280, 1536)]
    monkeypatch.setattr(campaign, '_collect_activations', lambda *_a, **_k: (
        {n: inputs['rows'] for n in (UNIT, OTHER)}, {}, {n: 0 for n in (UNIT, OTHER)},
        {n: inputs['max_abs'] for n in (UNIT, OTHER)}))
    monkeypatch.setattr(campaign, 'resolve_anchor_groups',
                        lambda targets, **_: {'g:controlled-fused-group': sorted(targets)})
    # The loop/journal is real; these records deliberately stand in for producer
    # wire verification, which the existing campaign resume tests cover.
    def record(anchor, _wire_dir, identity, *, existing=None):
        expected = dict(name=anchor.qname, format=anchor.format_name, identity=identity)
        if existing is not None:
            assert existing == expected
        return expected
    monkeypatch.setattr(campaign, '_checkpoint_wire_record', record)
    calls, controls = [], {'fail': True, 'hard': None}
    def measure(*, qname, format_name, **_):
        rate = int(format_name.rsplit('R', 1)[1])
        calls.append((qname, rate))
        if qname == OTHER and rate == 1280 and controls['fail']:
            if controls['hard'] is not None:
                raise controls['hard']
            raise RuntimeError('controlled permanent adaptive encode rejection')
        return campaign.CampaignAnchor(qname=qname, family=FAMILY,
            format_name=format_name, body_rate_q256=rate, dloss=1/rate,
            dloss_stderr=0., memory_bytes=32, bits_per_param=rate/256,
            activation_contract='a8', activation_quantized=True, wire_bytes=32,
            seconds=float(len(calls)), hessian_applied=False)
    monkeypatch.setattr(campaign, '_measure_anchor', measure)
    # This test-only safety bound makes the unfixed loop finish and expose its
    # repeated successful member. Production remains governed by PB timeout.
    argv[argv.index('--max-rounds')+1] = '4'
    argv += ['--anchors', '2', '--anchor-budget', '3']
    return campaign, checkpoint, argv, calls, controls


def states(checkpoint):
    from prismaquant.cost_stage_checkpoint import prepare_journal
    identity = json.loads(checkpoint.read_text())['identity']
    _, _, values = prepare_journal(checkpoint.with_name(checkpoint.name+'.parts'),
        manifest_path=checkpoint, stage='Tessera campaign', resume=True,
        identity=identity, qnames=[UNIT, OTHER])
    return values


def test_permanent_adaptive_rejection_stops_and_journals_successes(monkeypatch, tmp_path):
    campaign, checkpoint, argv, calls, _controls = fixture(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='no progress'):
        campaign.main(argv)
    counts = Counter(calls)
    assert counts[(UNIT, 1280)] == 1, 'already successful member was re-encoded'
    assert counts[(OTHER, 1280)] == 2  # One partial-success round, then no progress.
    saved = states(checkpoint)
    assert sorted(a['body_rate_q256'] for a in saved[UNIT]['anchors']) == [1024, 1280, 1536]
    assert [a['body_rate_q256'] for a in saved[OTHER]['anchors']] == [1024, 1536]
    assert not (tmp_path/'cost.pkl').exists()


def test_retry_encodes_only_missing_member_and_preserves_prior_prices(monkeypatch, tmp_path):
    campaign, checkpoint, argv, calls, controls = fixture(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='no progress'):
        campaign.main(argv)
    before = states(checkpoint)
    calls.clear()
    controls['fail'] = False
    assert campaign.main(argv) == 0
    assert calls == [(OTHER, 1280)]
    after = states(checkpoint)
    assert after[UNIT] == before[UNIT]
    old_by_rate = {a['body_rate_q256']: a for a in before[OTHER]['anchors']}
    assert all(a == old_by_rate[a['body_rate_q256']] for a in after[OTHER]['anchors']
               if a['body_rate_q256'] in old_by_rate)
    calls.clear()
    assert campaign.main(argv) == 0
    assert not calls


def test_successful_refinement_is_unchanged(monkeypatch, tmp_path):
    campaign, _checkpoint, argv, calls, controls = fixture(monkeypatch, tmp_path)
    controls['fail'] = False
    assert campaign.main(argv) == 0
    assert Counter(calls) == Counter((name, rate) for name in (UNIT, OTHER)
                                    for rate in (1024, 1280, 1536))
    payload = pickle.loads((tmp_path/'cost.pkl').read_bytes())
    assert set(payload['costs']) == {UNIT, OTHER}


@pytest.mark.parametrize('kind', ['HessianContractError', 'ActivationScaleContractError'])
def test_hard_contract_failures_remain_immediate(monkeypatch, tmp_path, kind):
    campaign, _checkpoint, argv, calls, controls = fixture(monkeypatch, tmp_path)
    from prismaquant.tessera_render import HessianContractError
    error_type = HessianContractError if kind == 'HessianContractError' else campaign.ActivationScaleContractError
    error = error_type('original hard refusal')
    controls['hard'] = error
    with pytest.raises(type(error), match='original hard refusal') as caught:
        campaign.main(argv)
    assert caught.value is error
    assert Counter(calls)[(OTHER, 1280)] == 1
