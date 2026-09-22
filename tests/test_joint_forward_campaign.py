"""Published Stage A identities reuse the campaign's canonical authority."""
import copy
import hashlib
import json

import pytest

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_forward_campaign import resolve_forward_campaign
from prismaquant.joint_layer_quanta import (
    ADJOINT_CAPTURE_SCHEMA, LAYER_QUANTUM_SCHEMA, bind_adjoint_receipt,
    canonical_sha256, roster_digest,
)


def fixture(tmp_path):
    names = ['model.layers.0.a', 'model.layers.1.b']
    scope = {'kind': 'complete_campaign', 'source_unit_count': 2,
             'source_roster_sha256': canonical_json_sha256(names, where='fixture'),
             'window_count': 5, 'campaign_window_count': 5, 'calib_seqlen': 4}
    campaign = dict(plan_sha256='a' * 64, prepared_sha256='b' * 64,
                    read_manifest_sha256='c' * 64, unit_roster_sha256=roster_digest(names),
                    campaign_scope=scope)
    record = {'schema': LAYER_QUANTUM_SCHEMA, 'quantum_id': 'layer-000',
              'campaign': campaign, 'adjoint': {}}
    record['identity_sha256'] = canonical_sha256(record, where='fixture')
    path = tmp_path / 'record.json'
    raw = json.dumps(record).encode(); path.write_bytes(raw)
    old = {**campaign, 'campaign_scope': None,
           'unit_roster_sha256': hashlib.sha256(''.join(n + '\n' for n in names).encode()).hexdigest()}
    document = {'source_campaign_record': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()},
                'campaign_identity': old, 'published_campaign_identity': campaign}
    args = dict(plan_sha256='a' * 64, prepared_sha256='b' * 64,
                read_manifest_sha256='c' * 64, formats_by_qname={n: [] for n in names},
                calibration_shape=[5, 4])
    return document, args


def test_original_caller_fails_consumer_and_corrected_new_receipt_binds(tmp_path):
    doc, args = fixture(tmp_path)
    original = copy.deepcopy(doc)
    old_receipt = {'schema': ADJOINT_CAPTURE_SCHEMA, 'run_identity': doc['campaign_identity'],
                   'checkpoints': [{'boundary': 2}]}
    expected = dict(plan_sha256=args['plan_sha256'], prepared_sha256=args['prepared_sha256'],
                    scope=doc['published_campaign_identity']['campaign_scope'], checkpoints=[2])
    with pytest.raises(ValueError, match='another scope'):
        bind_adjoint_receipt(old_receipt, **expected)
    published = resolve_forward_campaign(doc, **args)
    assert published['unit_roster_sha256'] != doc['campaign_identity']['unit_roster_sha256']
    assert bind_adjoint_receipt({**old_receipt, 'run_identity': published}, **expected)
    assert doc == original


@pytest.mark.parametrize('kind', ['roster', 'shape', 'plan', 'old-roster', 'old-scope', 'published'])
def test_correction_refuses_any_other_campaign_change(tmp_path, kind):
    doc, args = fixture(tmp_path)
    if kind == 'roster':
        args['formats_by_qname'] = {'model.layers.0.a': [], 'model.layers.1.c': []}
    elif kind == 'shape':
        args['calibration_shape'] = [4, 4]
    elif kind == 'plan':
        args['plan_sha256'] = 'd' * 64
    elif kind == 'old-roster':
        doc['campaign_identity']['unit_roster_sha256'] = 'd' * 64
    elif kind == 'old-scope':
        doc['campaign_identity']['campaign_scope'] = {'kind': 'diagnostic'}
    else:
        doc['published_campaign_identity'] = {**doc['published_campaign_identity'],
                                             'unit_roster_sha256': 'd' * 64}
    with pytest.raises(RuntimeError):
        resolve_forward_campaign(doc, **args)


def test_changed_source_record_bytes_refuse(tmp_path):
    doc, args = fixture(tmp_path)
    (tmp_path / 'record.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError)):
        resolve_forward_campaign(doc, **args)


def test_already_canonical_caller_is_preserved(tmp_path):
    doc, args = fixture(tmp_path)
    doc['campaign_identity'] = copy.deepcopy(doc['published_campaign_identity'])
    assert resolve_forward_campaign(doc, **args) == doc['campaign_identity']
