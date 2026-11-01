"""Real coherent source identities with explicit synthetic prices."""
from copy import deepcopy
import pickle

import pytest

from prismaquant import glm53_stock_reprice as stock
from tests.native_source_contract_fixtures import coherent_glm_campaign


@pytest.mark.parametrize('calibration', ['absent', 'different-draw'])
def test_stock_preserves_independent_empirical_population(tmp_path, calibration):
    probe, census, harvest, expert = coherent_glm_campaign(tmp_path)
    provenance = expert['provenance']
    if calibration != 'absent':
        provenance.update(dataset='different-corpus', n_calib_samples=3,
                          calib_seqlen=17, calib_seed=99, calib_sha256='f' * 64)
    before = pickle.dumps(expert)
    payload = stock.run_campaign_from_artifacts(
        probe_census=probe, checkpoint_census=census, harvest=harvest, expert_payload=expert)
    assert pickle.dumps(expert) == before
    for name, rows in expert['costs'].items():
        for fmt, row in rows.items():
            assert payload['costs'][name][fmt] == row
    retained = payload['provenance']['campaign_identity']['expert_empirical']
    assert retained['eval_driver'] == provenance['eval_driver']
    assert retained['calib_batch'] == provenance['calib_batch']


def test_stock_refuses_foreign_per_cell_source_before_merge(tmp_path):
    probe, census, harvest, expert = coherent_glm_campaign(tmp_path)
    for rows in expert['costs'].values():
        for row in rows.values():
            row['source_binding'] = deepcopy(row['source_binding'])
            row['source_binding']['effective_dtype'] = 'torch.float32'
    with pytest.raises(ValueError, match='source binding'):
        stock.run_campaign_from_artifacts(
            probe_census=probe, checkpoint_census=census, harvest=harvest, expert_payload=expert)
