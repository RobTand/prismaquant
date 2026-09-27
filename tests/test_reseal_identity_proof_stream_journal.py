"""experiments/reseal_identity_proof.py: a stream-head prefix compares journal to journal.

A stream-head campaign row (PQ #640) writes its checkpoint manifest and unit
shards only at finalize, and a prefix arm stops before finalize, so the only
journal the produced run has is the stream journal (``<checkpoint>.stream``,
PQ #1403).  ``compare_rows`` then compares it with the stored row's stream
journal, checks each unit's reader receipts, and accepts the stored stream
journal as the reference only where it agrees with the same row's finalized
shards.

The rows are written with the campaign's own journal writer
(``prepare_journal``/``write_unit``), so the on-disk shapes are the real ones.
Each refusal is shown on a copy that differs from the passing one in exactly
the named way.
"""
from __future__ import annotations

import copy
import hashlib
import importlib

import pytest

from prismaquant.cost_stage_checkpoint import _load_unit, prepare_journal, unit_path, write_unit

OLD = dict(prismaquant_source_sha256='a'*64, encoder_source_sha256='b'*64)
NEW = dict(prismaquant_source_sha256='c'*64, encoder_source_sha256='d'*64)
FIXTURE = 'f'*64
PARTS_STAGE = 'Tessera campaign'
STREAM_STAGE = 'Tessera campaign stream head'
UNITS = tuple(f'model.layers.16.mlp.experts.{i}.{p}' for i in (0, 1) for p in ('down_proj', 'gate_proj'))
FORMATS = ('TESSERA_BF16_K1_R1088', 'TESSERA_BF16_K1_R1152')
DEFERRED = 'bound per unit in its stream journal shard'


def _proof():
    return importlib.import_module('experiments.reseal_identity_proof')


def _receipts(unit):
    def one(kind, shape, dtype):
        return dict(algorithm='sha256.dtype_shape_contiguous.v1', dtype=dtype, shape=list(shape),
                    sha256=hashlib.sha256(f'{unit}/{kind}'.encode()).hexdigest())
    return dict(weight=one('weight', (4096, 2048), 'torch.bfloat16'),
                scoring_rows=one('scoring_rows', (512, 2048), 'torch.float32'),
                hessian=one('hessian', (2048, 2048), 'torch.float32'))


def _identity(pins, *, deferred):
    units = {u: dict(menu=list(FORMATS), input_global_scale=0.5,
                     **({k: DEFERRED for k in ('weight', 'scoring_rows', 'hessian')} if deferred else _receipts(u)))
             for u in UNITS}
    return dict(campaign_schema='prismaquant.tessera_campaign.v1', currency='output_mse',
                settings=dict(nsamples=512, seqlen=512, rate_band=[1088, 1152]),
                calibration=dict(fit_tokens=3), serving_scope=None, encoder_recipe=dict(body='window'),
                input_global_scale_policy='static', expert_projection=None, family_restriction=None,
                units=units, **pins)


def _wire(root, unit, fmt, seal):
    blob = (unit + fmt + 'wire').encode() * 11
    path = root/'cache'/'wire'/f'{unit}__{fmt}.tessera'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return dict(file=path.name, blob_sha256=hashlib.sha256(blob).hexdigest(), blob_bytes=len(blob),
                identity=dict(unit=unit, encoder_fixture_id=FIXTURE, encoder_source_sha256=seal,
                              recipe=dict(body='window', plane='CHANNEL', q256=int(fmt.rsplit('_R', 1)[1])),
                              schema='tessera.cached_unit_inputs.v1'))


def _anchor(unit, fmt, seconds):
    return dict(qname=unit, format_name=fmt, family=fmt.rsplit('_R', 1)[0], body_rate_q256=int(fmt.rsplit('_R', 1)[1]),
                dloss=0.25 + len(unit) / 1000 + int(fmt[-4:]) / 1e6, dloss_stderr=0.01, seconds=seconds,
                encoding_batch_size=16, wire_bytes=123, bits_per_param=4.25)


def _state(root, unit, formats, seal, *, seconds, stream):
    state = dict(anchors=[_anchor(unit, fmt, seconds) for fmt in formats],
                 wire_records={fmt: _wire(root, unit, fmt, seal) for fmt in formats})
    if stream:
        state['stream_receipts'] = _receipts(unit)
    return state


def _journal(root, directory, stage, identity, states, manifest_path=None):
    journal, sha, _ = prepare_journal(root/directory, stage=stage, resume=False, identity=identity,
                                      qnames=UNITS, manifest_path=manifest_path)
    for unit, state in states.items():
        write_unit(journal, stage=stage, qname=unit, identity_sha256=sha, state=state)


def stored_row(root):
    """A finalized stream-head row: its stream journal and its finalized shards."""
    stream = {u: _state(root, u, FORMATS, OLD['encoder_source_sha256'], seconds=2.5, stream=True) for u in UNITS}
    _journal(root, 'cost.anchors.json.stream', STREAM_STAGE, _identity(OLD, deferred=True), stream)
    parts = {u: {k: v for k, v in s.items() if k != 'stream_receipts'} for u, s in stream.items()}
    _journal(root, 'cost.anchors.json.parts', PARTS_STAGE, _identity(OLD, deferred=False), parts,
             manifest_path=root/'cost.anchors.json')
    return root


# The prefix: three units journaled, one of them with only its first rung.
PREFIX = {UNITS[0]: FORMATS, UNITS[1]: FORMATS, UNITS[2]: FORMATS[:1]}
PREFIX_CELLS = sum(len(v) for v in PREFIX.values())


def produced_run(root):
    """A stream-head prefix under the new pins, stopped before finalize."""
    states = {u: _state(root, u, fmts, NEW['encoder_source_sha256'], seconds=1.9, stream=True)
              for u, fmts in PREFIX.items()}
    _journal(root, 'cost.anchors.json.stream', STREAM_STAGE, _identity(NEW, deferred=True), states)
    (root/'cache').mkdir(parents=True, exist_ok=True)
    return root


def compare(produced, stored, expected_cells=PREFIX_CELLS):
    return _proof().compare_rows(produced, stored, old=OLD, new=NEW, expected_cells=expected_cells, prefix=True)


def rewrite_unit(root, directory, stage, unit, edit):
    """Rewrite one shard through the journal writer, so only its content changes."""
    proof = _proof()
    manifest = proof.load_manifest(root, 'stream' if directory.endswith('.stream') else 'parts')
    state = _load_unit(unit_path(root/directory, unit), stage=stage, qname=unit,
                       identity_sha256=manifest['identity_sha256'])
    state = copy.deepcopy(state)
    edit(state)
    write_unit(root/directory, stage=stage, qname=unit, identity_sha256=manifest['identity_sha256'], state=state)


def failure_names(result):
    return {f['what'] for f in result['failures']}


@pytest.fixture
def rows(tmp_path):
    return produced_run(tmp_path/'produced'), stored_row(tmp_path/'stored')


def test_an_unmodified_stream_prefix_passes_on_every_cell(rows):
    produced, stored = rows
    result = compare(produced, stored)
    assert result['failures'] == []
    assert result['ok'] is True
    assert result['journal'] == 'stream'
    assert len(result['cells']) == PREFIX_CELLS
    assert all(cell['byte_identical'] for cell in result['cells'])
    assert result['identity_matches_with_pins_substituted'] is True


def test_a_flipped_wire_byte_in_one_produced_unit_refuses(rows):
    produced, stored = rows
    path = produced/'cache'/'wire'/f'{UNITS[1]}__{FORMATS[1]}.tessera'
    blob = bytearray(path.read_bytes())
    blob[17] ^= 0x01
    path.write_bytes(bytes(blob))
    result = compare(produced, stored)
    assert result['ok'] is False
    assert 'cells' in failure_names(result)
    bad = [c for c in result['cells'] if not c['ok']]
    assert [(c['qname'], c['format_name']) for c in bad] == [(UNITS[1], FORMATS[1])]
    assert 'wire bytes differ' in bad[0]['problems']


def test_a_changed_stream_receipt_refuses(rows):
    produced, stored = rows

    def edit(state):
        state['stream_receipts']['hessian']['sha256'] = '0'*64
    rewrite_unit(produced, 'cost.anchors.json.stream', STREAM_STAGE, UNITS[0], edit)
    result = compare(produced, stored)
    assert result['ok'] is False
    receipts = [f for f in result['failures'] if f['what'] == 'stream_receipts']
    assert [f['unit'] for f in receipts] == [UNITS[0]]
    assert receipts[0]['diffs'][0][0] == 'stream_receipts.hessian.sha256'
    # Nothing else differs: the refusal is the receipt check's alone.
    assert failure_names(result) == {'stream_receipts'}


def test_a_missing_produced_unit_refuses_on_the_cell_count(rows):
    produced, stored = rows
    unit_path(produced/'cost.anchors.json.stream', UNITS[2]).unlink()
    result = compare(produced, stored)
    assert result['ok'] is False
    counts = [f for f in result['failures'] if f['what'] == 'cell_count']
    assert counts == [dict(what='cell_count', produced=PREFIX_CELLS - 1, expected=PREFIX_CELLS)]


def test_a_stale_stored_stream_journal_refuses_even_when_the_produced_run_matches_it(rows):
    """The reference check is load-bearing: without it this comparison passes."""
    produced, stored = rows

    def edit(state):
        state['anchors'][0]['dloss'] = 9.0
    rewrite_unit(stored, 'cost.anchors.json.stream', STREAM_STAGE, UNITS[0], edit)
    rewrite_unit(produced, 'cost.anchors.json.stream', STREAM_STAGE, UNITS[0], edit)
    result = compare(produced, stored)
    assert result['ok'] is False
    assert failure_names(result) == {'stored_stream_vs_parts'}
    stale = result['failures'][0]
    assert (stale['unit'], stale['field']) == (UNITS[0], 'anchors')
    # Every cell matches the stale journal; only the reference check refuses.
    assert all(cell['ok'] for cell in result['cells'])


def test_a_finalized_produced_run_still_compares_its_finalized_shards(tmp_path):
    """A dense arm finalizes, so it keeps the parts comparison it always had."""
    stored = stored_row(tmp_path/'stored')
    produced = tmp_path/'produced'
    stream = {u: _state(produced, u, FORMATS, NEW['encoder_source_sha256'], seconds=1.9, stream=True) for u in UNITS}
    _journal(produced, 'cost.anchors.json.stream', STREAM_STAGE, _identity(NEW, deferred=True), stream)
    parts = {u: {k: v for k, v in s.items() if k != 'stream_receipts'} for u, s in stream.items()}
    _journal(produced, 'cost.anchors.json.parts', PARTS_STAGE, _identity(NEW, deferred=False), parts,
             manifest_path=produced/'cost.anchors.json')
    result = compare(produced, stored, expected_cells=len(UNITS)*len(FORMATS))
    assert result['journal'] == 'parts'
    assert result['ok'] is True, result['failures']


def test_a_run_with_neither_journal_is_refused(tmp_path):
    stored = stored_row(tmp_path/'stored')
    (tmp_path/'produced').mkdir()
    with pytest.raises(ValueError, match='no checkpoint manifest and no stream journal'):
        compare(tmp_path/'produced', stored)


class _Campaign:
    STREAM_JOURNAL_SUFFIX = '.stream'


def test_the_prefix_helper_reads_the_row_head_from_the_files_written_before_the_first_encode(tmp_path):
    helper = importlib.import_module('experiments.campaign_prefix_profile')
    run = tmp_path/'run'
    command = ['--out', str(run/'cost.pkl'), '--cache-dir', str(run/'cache'),
               '--checkpoint', str(run/'cost.anchors.json')]
    (run/'cache').mkdir(parents=True)
    # Neither file yet, or a tree without the stream head: load-all's precondition applies.
    assert helper.row_head(_Campaign(), command) == ('load-all', None)
    assert helper.row_head(object(), command) == ('load-all', None)
    manifest = run/'cost.anchors.json.stream'/'manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"units": []}')
    assert helper.row_head(_Campaign(), command) == ('stream', manifest)
    # The load-all head writes its execution record before encoding; the stream head only at finalize.
    (run/'cache'/'row-head-execution.json').write_text('{}')
    assert helper.row_head(_Campaign(), command) == ('load-all', None)


# ---------------------------------------------------------------------------
# Produced-only settings: a flag newer than the stored row (PQ #1520)
# ---------------------------------------------------------------------------
#
# tessera_campaign builds its identity settings from vars(args), so a flag
# added after a row was stored lands in every produced identity, set or not.
# ``--produced-only-setting`` sets exactly the declared names aside before the
# identity is compared.  Each refusal below differs from the passing run in
# one named way.

PQ_1520 = ('source_identity_cache', 'source_identity_cache_sha256')


def produced_run_binding(root, extra):
    """``produced_run`` whose identity settings also bind ``extra``."""
    states = {u: _state(root, u, fmts, NEW['encoder_source_sha256'], seconds=1.9, stream=True)
              for u, fmts in PREFIX.items()}
    identity = _identity(NEW, deferred=True)
    identity['settings'].update(extra)
    _journal(root, 'cost.anchors.json.stream', STREAM_STAGE, identity, states)
    (root/'cache').mkdir(parents=True, exist_ok=True)
    return root


def compare_normalized(produced, stored, names=PQ_1520):
    return _proof().compare_rows(produced, stored, old=OLD, new=NEW, expected_cells=PREFIX_CELLS,
                                 prefix=True, produced_only_settings=names)


@pytest.fixture
def pq1520_rows(tmp_path):
    produced = produced_run_binding(tmp_path/'produced', {name: None for name in PQ_1520})
    return produced, stored_row(tmp_path/'stored')


def test_the_pq1520_settings_alone_fail_the_identity_without_normalization(pq1520_rows):
    produced, stored = pq1520_rows
    result = compare(produced, stored)
    assert failure_names(result) == {'identity', 'identity_sha256'}
    assert [f['field'] for f in result['failures'] if f['what'] == 'identity'] == ['settings.source_identity_cache']
    assert all(cell['ok'] for cell in result['cells'])


def test_setting_the_pq1520_settings_aside_restores_the_exact_expected_digest(pq1520_rows):
    produced, stored = pq1520_rows
    result = compare_normalized(produced, stored)
    assert result['failures'] == []
    assert result['ok'] is True
    assert result['identity_matches_with_pins_substituted'] is False
    assert result['identity_matches_after_normalization'] is True
    assert result['normalized_produced_identity_sha256'] == result['expected_identity_sha256']
    assert result['produced_identity_sha256'] != result['expected_identity_sha256']
    proof = _proof()
    assert result['produced_only_settings'] == {
        name: dict(reason=proof.PRODUCED_ONLY_SETTINGS[name], produced_value=None) for name in PQ_1520}


def test_another_flipped_setting_still_refuses_after_normalization(tmp_path):
    extra = {name: None for name in PQ_1520}
    extra['nsamples'] = 513
    produced = produced_run_binding(tmp_path/'produced', extra)
    result = compare_normalized(produced, stored_row(tmp_path/'stored'))
    assert result['ok'] is False
    assert failure_names(result) == {'identity', 'identity_sha256'}
    assert [f['field'] for f in result['failures'] if f['what'] == 'identity'] == ['settings.nsamples']
    assert result['identity_matches_after_normalization'] is False


def test_a_setting_the_stored_row_binds_cannot_be_set_aside(tmp_path):
    produced = produced_run_binding(tmp_path/'produced', {name: None for name in PQ_1520})
    stored = tmp_path/'stored'
    stream = {u: _state(stored, u, FORMATS, OLD['encoder_source_sha256'], seconds=2.5, stream=True) for u in UNITS}
    identity = _identity(OLD, deferred=True)
    identity['settings']['source_identity_cache'] = None
    _journal(stored, 'cost.anchors.json.stream', STREAM_STAGE, identity, stream)
    with pytest.raises(ValueError, match=r"stored identity binds setting\(s\) \['source_identity_cache'\]"):
        compare_normalized(produced, stored)


def test_an_undeclared_setting_cannot_be_set_aside(pq1520_rows):
    produced, stored = pq1520_rows
    with pytest.raises(ValueError, match=r"\['nsamples'\] are not declared produced-only settings"):
        compare_normalized(produced, stored, names=PQ_1520 + ('nsamples',))


def test_a_declared_setting_the_produced_run_lacks_is_refused(tmp_path):
    produced = produced_run_binding(tmp_path/'produced', {'source_identity_cache': None})
    with pytest.raises(ValueError, match=r"produced identity binds no setting\(s\) \['source_identity_cache_sha256'\]"):
        compare_normalized(produced, stored_row(tmp_path/'stored'))
