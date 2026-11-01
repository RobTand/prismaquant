"""A bounded observer must preserve the real campaign's journal boundary."""
from prismaquant import tessera_cell_measurement as cell_measurement
from types import SimpleNamespace

import pytest
import torch

from experiments.campaign_prefix_profile import run_prefix, PrefixEncodingFailed


def fixture(monkeypatch, *, batch=2, fail=False, omit_prefetch=False, available=8):
    journal, encoded = [], []
    def encode(**kwargs):
        if fail:
            raise ValueError('real encoder error')
        encoded.extend(kwargs['qnames'])
        return kwargs['qnames']
    def prefetch(*a, **kw):
        acts = {str(i): torch.ones(2, 2) for i in range(8)}
        return (acts, acts.copy(), {}, {}), {'sha256': 'capture'}, {'sha256': 'receipt'}
    monkeypatch.setattr(cell_measurement, "measure_cells", encode)
    campaign = SimpleNamespace(_prefetch_selected_capture=prefetch)
    def main(command):
        if not omit_prefetch:
            campaign._prefetch_selected_capture(names=[str(i) for i in range(8)])
        for start in range(0, available, batch):
            try:
                rows = cell_measurement.measure_cells(qnames=[str(i) for i in range(start, start+batch)],
                    format_name='same-rung')
            except Exception:
                continue
            journal.extend(rows)
    campaign.main = main
    observer = SimpleNamespace(result={}, wrap_anchor=lambda fn: fn)
    return campaign, observer, journal, encoded


def test_stops_before_next_encode_after_last_return_was_journaled(monkeypatch):
    campaign, observer, journal, encoded = fixture(monkeypatch)
    original = cell_measurement.measure_cells
    run_prefix(campaign, [], observer, limit=4, expected_source_units=8)
    assert journal == encoded == ['0', '1', '2', '3']
    assert cell_measurement.measure_cells is original
    assert observer.result['campaign_completed'] is False
    assert observer.result['resident_prefetch']['units'] == 8
    assert observer.result['completed_anchor_units'] == 4


@pytest.mark.parametrize('options,limit,error', [
    ({'batch': 3}, 4, PrefixEncodingFailed),
    ({'fail': True}, 4, PrefixEncodingFailed),
    ({'omit_prefetch': True}, 4, PrefixEncodingFailed),
    ({'available': 2}, 4, RuntimeError),
])
def test_refuses_different_work_and_restores_the_campaign(monkeypatch, options, limit, error):
    campaign, observer, _, _ = fixture(monkeypatch, **options)
    original = cell_measurement.measure_cells
    with pytest.raises(error):
        run_prefix(campaign, [], observer, limit=limit, expected_source_units=8)
    assert cell_measurement.measure_cells is original
