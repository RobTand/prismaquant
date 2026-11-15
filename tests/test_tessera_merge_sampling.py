"""Fanout retains the draw and every row's refusal/identity evidence."""
import sys
from pathlib import Path

import pytest

from prismaquant import tessera_campaign_table as campaign_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from test_tessera_campaign_fanout import _payloads


def test_merge_unions_row_sampling_and_empty_menu_evidence():
    rows = _payloads()
    for name, payload in zip(('a', 'b'), rows.values()):
        payload['provenance'].update(no_admitted_rung=[name], unit_selection_sample={
            'audit_units': [name], 'inclusion_probability': {name: .5}})
    merged = campaign_table.merge_payloads(rows, census={'counts': {}}, capture_sha256='merged')
    prov = merged['provenance']
    assert prov['no_admitted_rung'] == ['a', 'b']
    assert prov['unit_selection_sample'] == {
        'audit_units': ['a', 'b'], 'inclusion_probability': {'a': .5, 'b': .5}}


def test_merge_refuses_different_rate_bands():
    rows = _payloads()
    rows['row-0000']['provenance']['rate_band'] = [512, 896]
    rows['row-0001']['provenance']['rate_band'] = [896, 1792]
    with pytest.raises(campaign_table.MergeRefused, match='rate_band'):
        campaign_table.merge_payloads(rows, census={'counts': {}}, capture_sha256='merged')


def _rows(tmp_path, extras):
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit
    result = {}
    for name, extra in zip(('a', 'b'), extras):
        directory = tmp_path / name
        parts, digest, _ = prepare_journal(directory / 'cost.anchors.json.parts',
            stage='Tessera campaign', resume=True,
            identity={'units': {name: {'weight': name}}, **extra}, qnames=[name],
            manifest_path=directory / 'cost.anchors.json')
        write_unit(parts, stage='Tessera campaign', qname=name,
                   identity_sha256=digest, state={'measured': {}})
        result[name] = directory
    return result


def test_checkpoint_unions_stack_sampling_identity(tmp_path):
    rows = _rows(tmp_path, [
        {'stack_sampling_identity': {'packed-a': {'seed': 3}}},
        {'stack_sampling_identity': {'packed-b': {'seed': 7}}}])
    merged = campaign_table.merge_checkpoint(rows, tmp_path / 'merged.json')
    assert merged['identity']['stack_sampling_identity'] == {
        'packed-a': {'seed': 3}, 'packed-b': {'seed': 7}}


def test_checkpoint_refuses_a_shared_field_missing_from_second_row(tmp_path):
    rows = _rows(tmp_path, [{'calibration': 'a'}, {}])
    with pytest.raises(campaign_table.MergeRefused, match='calibration'):
        campaign_table.merge_checkpoint(rows, tmp_path / 'merged.json')


@pytest.mark.parametrize('field,left,right', [
    ('serving_scope', {'target': 'target', 'by_unit': {'owner': {'lane': 'dense'}}},
     {'target': 'target', 'by_unit': {'owner': {'lane': 'routed'}}}),
    ('expert_projection', {'source': 'source', 'stacks': {'stack': {'slice': 1}}},
     {'source': 'source', 'stacks': {'stack': {'slice': 2}}}),
])
def test_checkpoint_refuses_conflicting_overlapping_scope_records(tmp_path, field, left, right):
    rows = _rows(tmp_path, [{field: left}, {field: right}])
    with pytest.raises(campaign_table.MergeRefused, match='(owner|stack)'):
        campaign_table.merge_checkpoint(rows, tmp_path / 'merged.json')
