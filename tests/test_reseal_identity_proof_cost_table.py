"""experiments/reseal_identity_proof.py: a prefix run's cost table compares on its own units.

A prefix arm measures a few units of a stored row.  If its campaign wrote a cost
table, that table has fewer units than the stored row's, so a whole-table
comparison fails on every per-unit key.  The comparator restricts those keys to
the prefix's units on both sides and requires exact equality there; it never
drops a key, and it refuses a produced entry for any other unit.
"""
from __future__ import annotations

import copy
import importlib

import pytest

PREFIX = ('layers.10.mlp.experts.0.down_proj', 'layers.10.mlp.experts.0.gate_proj')
OTHERS = ('layers.10.mlp.experts.1.down_proj', 'layers.10.mlp.experts.1.gate_proj')


def _proof():
    return importlib.import_module('experiments.reseal_identity_proof')


def _row(unit, index):
    return {fmt: dict(output_mse=0.5 + index + offset, tessera_body_rate_q256=rate, encode_seconds=3.0 + index,
                      encode_seconds_accounting='batch_wall_time_divided_by_batch_size', encoding_batch_size=8)
            for fmt, rate, offset in (('TESSERA_BF16_K1_R1024', 1024, 0.0), ('TESSERA_E4M3_K1_R1024', 1024, 0.25))}


def _table(units, *, seal='old'):
    """A cost table in the campaign's shape, over ``units`` of a four-unit row."""
    everything = PREFIX + OTHERS
    costs = {unit: _row(unit, everything.index(unit)) for unit in units}
    if OTHERS[1] in units:
        # a format only another unit prices
        costs[OTHERS[1]]['TESSERA_E2M1_K2_R896'] = dict(output_mse=9.0, tessera_body_rate_q256=896)
    loo = {PREFIX[0]: {'TESSERA_BF16_K1': dict(error=0.125)}, OTHERS[0]: {'TESSERA_BF16_K1': dict(error=0.5)}}
    refusals = [dict(qname=PREFIX[1], family='TESSERA_E4M3_K1', reason='non_interpolable_anchors'),
                dict(qname=OTHERS[1], family='TESSERA_BF16_K1', reason='non_interpolable_anchors')]
    return dict(
        schema='prismaquant.tessera_campaign_cost.v1', currency='dloss', provenance=dict(seconds=len(units)),
        costs=costs,
        formats=sorted({fmt for rows in costs.values() for fmt in rows}),
        anchor_counts={unit: {'TESSERA_BF16_K1': 1, 'TESSERA_E4M3_K1': 1} for unit in units},
        leave_one_anchor_out={unit: dict(v) for unit, v in loo.items() if unit in units},
        non_interpolable=[entry for entry in refusals if entry['qname'] in units],
        tessera_expert_wires={unit: {'TESSERA_BF16_K1_R1024': dict(blob_sha256=f'{unit}-blob', identity=dict(
            encoder_source_sha256=seal, recipe=dict(body='window', plane='channel')))} for unit in units},
        # built for the whole selected scope in both runs, so it compares whole
        menu_sizes={unit: 2 for unit in everything},
    )


def _prefix_table():
    table = _table(PREFIX, seal='new')
    for rows in table['costs'].values():
        for row in rows.values():
            row['encode_seconds'] += 1.5
    return table


def _stored():
    return _table(PREFIX + OTHERS)


def _what(failures):
    return sorted((failure['what'], failure.get('key')) for failure in failures)


def test_a_prefix_table_compares_exactly_on_its_own_units():
    proof = _proof()
    failures, report = proof.compare_cost_tables(_prefix_table(), _stored(), units=PREFIX)
    assert failures == []
    assert report['restricted_to_units'] == 2
    assert set(report['keys_compared']) == {'anchor_counts', 'costs', 'currency', 'formats', 'leave_one_anchor_out',
                                            'menu_sizes', 'non_interpolable', 'schema', 'tessera_expert_wires'}
    # The same table is not a whole row: compared whole it fails on every per-unit key.
    failures, report = proof.compare_cost_tables(_prefix_table(), _stored())
    assert report['restricted_to_units'] is None
    assert _what(failures) == [('cost_content', key) for key in (
        'anchor_counts', 'costs', 'formats', 'leave_one_anchor_out', 'non_interpolable', 'tessera_expert_wires')]


@pytest.mark.parametrize('key,remove', [
    ('anchor_counts', lambda t: t['anchor_counts'].pop(PREFIX[1])),
    ('costs', lambda t: t['costs'].pop(PREFIX[0])),
    ('costs', lambda t: t['costs'][PREFIX[0]].pop('TESSERA_E4M3_K1_R1024')),
    ('leave_one_anchor_out', lambda t: t['leave_one_anchor_out'].pop(PREFIX[0])),
    ('non_interpolable', lambda t: t['non_interpolable'].clear()),
    ('tessera_expert_wires', lambda t: t['tessera_expert_wires'].pop(PREFIX[1])),
])
def test_a_prefix_unit_entry_the_stored_row_has_and_the_table_lacks_is_refused(key, remove):
    table = _prefix_table()
    remove(table)
    failures, _ = _proof().compare_cost_tables(table, _stored(), units=PREFIX)
    assert ('cost_content', key) in _what(failures)


@pytest.mark.parametrize('key,change', [
    ('anchor_counts', lambda t: t['anchor_counts'][PREFIX[0]].update(TESSERA_BF16_K1=2)),
    ('costs', lambda t: t['costs'][PREFIX[1]]['TESSERA_BF16_K1_R1024'].update(output_mse=0.75)),
    ('leave_one_anchor_out', lambda t: t['leave_one_anchor_out'][PREFIX[0]]['TESSERA_BF16_K1'].update(error=0.25)),
    ('non_interpolable', lambda t: t['non_interpolable'][0].update(reason='other')),
    ('tessera_expert_wires', lambda t: t['tessera_expert_wires'][PREFIX[0]]['TESSERA_BF16_K1_R1024'].update(blob_sha256='x')),
    ('formats', lambda t: t['formats'].append('TESSERA_E2M1_K2_R896')),
    ('menu_sizes', lambda t: t['menu_sizes'].pop(OTHERS[0])),
])
def test_a_value_that_moved_on_a_prefix_unit_is_refused(key, change):
    table = _prefix_table()
    change(table)
    failures, _ = _proof().compare_cost_tables(table, _stored(), units=PREFIX)
    assert ('cost_content', key) in _what(failures)


@pytest.mark.parametrize('key,add', [
    ('anchor_counts', lambda t, s: t['anchor_counts'].update({OTHERS[0]: copy.deepcopy(s['anchor_counts'][OTHERS[0]])})),
    ('costs', lambda t, s: t['costs'].update({OTHERS[0]: copy.deepcopy(s['costs'][OTHERS[0]])})),
    ('non_interpolable', lambda t, s: t['non_interpolable'].append(copy.deepcopy(s['non_interpolable'][-1]))),
])
def test_a_prefix_table_entry_for_another_unit_is_refused(key, add):
    table, stored = _prefix_table(), _stored()
    add(table, stored)
    failures, _ = _proof().compare_cost_tables(table, stored, units=PREFIX)
    assert ('cost_outside_prefix', key) in _what(failures)


def test_a_key_missing_from_the_prefix_table_is_refused():
    table = _prefix_table()
    del table['anchor_counts']
    failures, report = _proof().compare_cost_tables(table, _stored(), units=PREFIX)
    assert ('cost_keys', None) in _what(failures)
    assert 'anchor_counts' not in report['keys_compared']
