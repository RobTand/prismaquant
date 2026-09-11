"""tools/reseal_campaign_identity.py on synthetic campaign rows.

The rows mimic the on-disk shapes of a Tessera campaign row (manifest,
unit shards, cost.pkl, wire files) with two source pins, so the tests can
exercise dry-run, migrate, verify, idempotency, the two-step migration and
every refusal without a GPU or a real checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import reseal_campaign_identity as tool  # noqa: E402

OLD = dict(prismaquant_source_sha256='a'*64, encoder_source_sha256='b'*64)
NEW = dict(prismaquant_source_sha256='c'*64, encoder_source_sha256='d'*64)
FIXTURE = 'f'*64


def _wire(root, name, fmt, seal):
    blob = (name+fmt+'wire').encode()*7
    path = root/'cache'/'wire'/f'{name}__{fmt}.tessera'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return dict(file=path.name, blob_sha256=hashlib.sha256(blob).hexdigest(), blob_bytes=len(blob),
                identity=dict(unit=name, encoder_fixture_id=FIXTURE, encoder_source_sha256=seal,
                              recipe=dict(q256=int(fmt.rsplit('R', 1)[1])), schema='tessera.cached_unit_inputs.v1'))


def make_row(root, *, pins=OLD, units=('layers.0.mlp.experts.0.down_proj', 'layers.0.mlp.experts.1.down_proj'),
             with_cost=True, expert_wires=True):
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(campaign_schema='prismaquant.tessera_campaign.v1', currency='output_mse',
                    settings=dict(nsamples=4, seqlen=8), calibration=dict(fit_tokens=3), serving_scope=None,
                    encoder_recipe=dict(body='window'), **pins, input_global_scale_policy='static',
                    expert_projection=None, units={u: dict(menu=['TESSERA_E4M3_K1_R832', 'TESSERA_E4M3_K1_R960'],
                                                            weight=dict(sha256='e'*64)) for u in units})
    sha = tool.identity_sha256(identity)
    parts = root/'cost.anchors.json.parts'
    manifest = dict(schema=tool.MANIFEST_SCHEMA, stage=tool.STAGE, identity_sha256=sha,
                    identity=json.loads(tool.canonical_bytes(identity)),
                    units=[dict(qname=u, file=str(tool.unit_path(parts, u).relative_to(parts))) for u in units])
    (root/'cost.anchors.json').write_bytes(tool.manifest_bytes(manifest))
    wires = {}
    for u in units:
        records = {fmt: _wire(root, u, fmt, pins['encoder_source_sha256']) for fmt in ('TESSERA_E4M3_K1_R832', 'TESSERA_E4M3_K1_R960')}
        wires[u] = records
        state = dict(anchors=[dict(qname=u, format_name=fmt, family='TESSERA_E4M3_K1', body_rate_q256=int(fmt[-3:]),
                                   dloss=0.5, seconds=1.0) for fmt in records], wire_records=records)
        payload = pickle.dumps(state, protocol=5)
        envelope = dict(schema=tool.UNIT_SCHEMA, stage=tool.STAGE, qname=u, identity_sha256=sha,
                        payload_sha256=hashlib.sha256(payload).hexdigest(), payload=payload)
        path = tool.unit_path(parts, u)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(envelope, protocol=5))
    if with_cost:
        cost = dict(provenance=dict(model='m', wall_seconds=3.0, tessera_commit=''), schema='prismaquant.tessera_campaign.v1',
                    costs={u: {fmt: dict(output_mse=0.5, encode_seconds=1.0) for fmt in ('TESSERA_E4M3_K1_R832', 'TESSERA_E4M3_K1_R960')} for u in units},
                    formats=['TESSERA_E4M3_K1_R832'], currency='output_mse', leave_one_anchor_out={}, non_interpolable=[],
                    menu_sizes={u: 2 for u in units}, anchor_counts={u: 2 for u in units})
        if expert_wires:
            cost['tessera_expert_wires'] = wires
        (root/'cost.pkl').write_bytes(pickle.dumps(cost, protocol=4))
    return sha


def write_pins(path, old=OLD, new=NEW):
    path.write_text(json.dumps(dict(schema=tool.PINS_SCHEMA, old=old, new=new)))
    return path


def write_bundle(path, old=OLD, new=NEW, ok=True):
    bundle = dict(schema=tool.BUNDLE_SCHEMA, ok=ok, pins=dict(old=old, new=new), encoder_fixture_id_equal=True,
                  arms=[], fixture_id=dict(result='synthetic-fixture-id-arm', ids={'old': FIXTURE, 'new': FIXTURE}),
                  cell_count=24, pb_actions=['k1'])
    path.write_text(json.dumps(bundle))
    return path


def run(*argv):
    return tool.main([str(a) for a in argv])


def test_dry_run_lists_every_edit_and_writes_nothing(tmp_path, capsys):
    row = tmp_path/'row-0001'
    make_row(row)
    before = {p: p.read_bytes() for p in row.rglob('*') if p.is_file()}
    pins = write_pins(tmp_path/'pins.json')
    assert run('dry-run', '--pins', pins, '--row', row, '--verbose', '--report', tmp_path/'r.json') == 0
    out = capsys.readouterr().out
    assert 'identity_sha256' in out and 'wire_records[*].identity.encoder_source_sha256' in out
    report = json.loads((tmp_path/'r.json').read_text())
    plan = report['rows'][0]
    assert plan['state'] == 'pending' and plan['shard_count'] == 2 and plan['receipt_seals'] == 4 and plan['cost_seals'] == 4
    assert plan['bytes_to_write'] > 0
    assert {p: p.read_bytes() for p in row.rglob('*') if p.is_file()} == before


def test_migrate_then_verify_is_consistent_and_idempotent(tmp_path, capsys):
    row = tmp_path/'row-0001'
    old_sha = make_row(row)
    old_cost = (row/'cost.pkl').read_bytes()
    pins = write_pins(tmp_path/'pins.json')
    bundle = write_bundle(tmp_path/'bundle.json')
    assert run('migrate', '--pins', pins, '--proof', bundle, '--row', row, '--report', tmp_path/'m.json') == 0
    manifest = json.loads((row/'cost.anchors.json').read_text())
    assert manifest['identity']['prismaquant_source_sha256'] == NEW['prismaquant_source_sha256']
    assert manifest['identity_sha256'] == tool.identity_sha256(manifest['identity']) != old_sha
    record = manifest['identity_migration'][-1]
    assert record['old_identity_sha256'] == old_sha and record['proof_bundle_sha256'] == tool.sha256_file(bundle)
    cost = pickle.loads((row/'cost.pkl').read_bytes())
    assert cost['provenance']['identity_migration'] == manifest['identity_migration']
    assert all(r['identity']['encoder_source_sha256'] == NEW['encoder_source_sha256']
               for u in cost['tessera_expert_wires'].values() for r in u.values())
    content = tool.content_equality(old_cost, (row/'cost.pkl').read_bytes(), encoder_moves=True)
    assert content['identical']
    previous = [p for p in row.iterdir() if p.name.startswith('.reseal-previous-')]
    assert len(previous) == 1 and (previous[0]/'cost.pkl').read_bytes() == old_cost
    assert not [p for p in row.iterdir() if p.name.startswith('.reseal-stage-')]
    assert json.loads((row/'identity_migration.json').read_text())['phase'] == 'done'
    assert run('verify', '--pins', pins, '--row', row) == 0
    # a second run finds the row migrated and leaves it alone
    snapshot = {p: p.read_bytes() for p in row.rglob('*') if p.is_file()}
    assert run('migrate', '--pins', pins, '--proof', bundle, '--row', row) == 0
    assert 'already carries the new pins' in capsys.readouterr().out
    assert {p: p.read_bytes() for p in row.rglob('*') if p.is_file()} == snapshot


def test_two_step_migration_prismaquant_then_encoder(tmp_path):
    row = tmp_path/'row-0002'
    make_row(row)
    mid = dict(OLD, prismaquant_source_sha256=NEW['prismaquant_source_sha256'])
    pins1 = write_pins(tmp_path/'p1.json', OLD, mid)
    bundle1 = write_bundle(tmp_path/'b1.json', OLD, mid)
    assert run('migrate', '--pins', pins1, '--proof', bundle1, '--row', row, '--discard-previous') == 0
    state = pickle.loads(pickle.loads((row/'cost.anchors.json.parts'/'units'/next(
        (row/'cost.anchors.json.parts'/'units').iterdir()).name).read_bytes())['payload'])
    assert next(iter(state['wire_records'].values()))['identity']['encoder_source_sha256'] == OLD['encoder_source_sha256']
    pins2 = write_pins(tmp_path/'p2.json', mid, NEW)
    bundle2 = write_bundle(tmp_path/'b2.json', mid, NEW)
    assert run('migrate', '--pins', pins2, '--proof', bundle2, '--row', row, '--discard-previous') == 0
    assert run('verify', '--pins', pins2, '--row', row) == 0
    manifest = json.loads((row/'cost.anchors.json').read_text())
    assert len(manifest['identity_migration']) == 2
    assert not [p for p in row.iterdir() if p.name.startswith('.reseal-')]


def test_journal_only_row_is_resealed(tmp_path):
    row = tmp_path/'row-0003'
    make_row(row, with_cost=False)
    pins = write_pins(tmp_path/'pins.json')
    bundle = write_bundle(tmp_path/'bundle.json')
    assert run('migrate', '--pins', pins, '--proof', bundle, '--row', row) == 0
    assert run('verify', '--pins', pins, '--row', row) == 0


def test_refusals(tmp_path, capsys):
    row = tmp_path/'row-0004'
    make_row(row)
    pins = write_pins(tmp_path/'pins.json')
    assert run('migrate', '--pins', pins, '--row', row) == 2
    assert 'requires --proof' in capsys.readouterr().err
    wrong = write_bundle(tmp_path/'wrong.json', OLD, dict(NEW, encoder_source_sha256='9'*64))
    assert run('migrate', '--pins', pins, '--proof', wrong, '--row', row) == 2
    assert 'differ from the pins file' in capsys.readouterr().err
    notok = write_bundle(tmp_path/'notok.json', ok=False)
    assert run('migrate', '--pins', pins, '--proof', notok, '--row', row) == 2
    foreign = tmp_path/'row-0005'
    make_row(foreign, pins=dict(prismaquant_source_sha256='1'*64, encoder_source_sha256='2'*64))
    assert run('dry-run', '--pins', pins, '--row', foreign) == 2
    assert 'neither the old nor the new' in capsys.readouterr().err
    live = tmp_path/'first-proof-anchor-preparation-05'/'workspace'/'rows'/'row-0006'
    make_row(live)
    assert run('dry-run', '--pins', pins, '--row', live) == 2
    assert 'live campaign workspace' in capsys.readouterr().err
    assert run('dry-run', '--pins', pins, '--row', live, '--allow-live') == 0
    # the row is untouched by every refusal above
    assert json.loads((row/'cost.anchors.json').read_text())['identity']['encoder_source_sha256'] == OLD['encoder_source_sha256']


def test_verify_detects_a_tampered_wire_and_a_stale_seal(tmp_path):
    row = tmp_path/'row-0007'
    make_row(row)
    pins = write_pins(tmp_path/'pins.json')
    bundle = write_bundle(tmp_path/'bundle.json')
    assert run('migrate', '--pins', pins, '--proof', bundle, '--row', row, '--discard-previous') == 0
    wire = next((row/'cache'/'wire').iterdir())
    wire.write_bytes(wire.read_bytes()+b'x')
    assert run('verify', '--pins', pins, '--row', row) == 1
    assert run('verify', '--pins', pins, '--row', row, '--skip-wire-bytes') == 0
    stale = write_pins(tmp_path/'stale.json', OLD, dict(NEW, encoder_source_sha256='9'*64))
    assert run('verify', '--pins', stale, '--row', row, '--skip-wire-bytes') == 1


def test_hash_definitions_match_the_campaign(tmp_path):
    pq = tmp_path/'pq'/'prismaquant'
    pq.mkdir(parents=True)
    (pq/'a.py').write_bytes(b'x')
    (pq/'__pycache__').mkdir()
    (pq/'__pycache__'/'a.pyc').write_bytes(b'ignored')
    digest = hashlib.sha256()
    digest.update((4).to_bytes(4, 'big')+b'a.py'+(1).to_bytes(8, 'big')+b'x')
    assert tool.prismaquant_tree_sha256(pq) == (digest.hexdigest(), 1)
    src = tmp_path/'producer'/'src'/'tessera'
    src.mkdir(parents=True)
    (src/'b.py').write_bytes(b'y')
    (src/'README.md').write_bytes(b'not hashed')
    digest = hashlib.sha256(b'b.py\0y\0')
    assert tool.encoder_tree_sha256(src) == (digest.hexdigest(), 1)
    sources = dict(prismaquant=dict(commit='deadbeef', tree=str(tmp_path/'pq')), encoder=dict(commit='cafe', tree=str(tmp_path/'producer')))
    good = tmp_path/'good.json'
    good.write_text(json.dumps(dict(schema=tool.PINS_SCHEMA, old=OLD, sources=sources,
                                    new=dict(prismaquant_source_sha256=tool.prismaquant_tree_sha256(pq)[0],
                                             encoder_source_sha256=tool.encoder_tree_sha256(src)[0]))))
    assert tool.load_pins(good)['source_checks']['prismaquant']['files'] == 1
    bad = tmp_path/'bad.json'
    bad.write_text(json.dumps(dict(schema=tool.PINS_SCHEMA, old=OLD, new=NEW, sources=sources)))
    with pytest.raises(tool.Refused):
        tool.load_pins(bad)


def _arm_result(path, old, new, *, routed_rates=(832, 960, 1088), dense_rates=(832, 960, 1088)):
    cells = []
    def cell(qname, family, rate):
        cells.append(dict(ok=True, byte_identical=True, dloss=0.5, stored_dloss=0.5, qname=qname, family=family,
                          format_name=f'{family}_R{rate}', body_rate_q256=rate, blob_sha256='e'*64, blob_bytes=10,
                          encoder_fixture_id=FIXTURE))
    for rate in routed_rates:
        for j in range(4):
            cell(f'layers.0.mlp.experts.{j}.down_proj', 'TESSERA_E4M3_K1', rate)
    for rate in dense_rates:
        for family in ('TESSERA_BF16_K1', 'TESSERA_E4M3_K1'):
            for j in range(2):
                cell(f'layers.{j}.mlp.gate_proj', family, rate)
    cell('layers.0.mlp.up_proj', 'TESSERA_E2M1_K2', 896)
    path.write_text(json.dumps(dict(kind='dense', comparison=dict(
        kind='comparison', ok=True, old_pins=old, new_pins=new, identity_matches_with_pins_substituted=True, cells=cells))))
    return path


def test_proof_bundle_needs_a_fixture_arm_only_when_the_encoder_pin_moves(tmp_path, capsys):
    pq_only = dict(NEW, encoder_source_sha256=OLD['encoder_source_sha256'])
    pins_moving = write_pins(tmp_path/'pins-both.json')
    pins_pq_only = write_pins(tmp_path/'pins-pq.json', new=pq_only)
    arm_moving = _arm_result(tmp_path/'arm-both.json', OLD, NEW)
    arm_pq_only = _arm_result(tmp_path/'arm-pq.json', OLD, pq_only)
    fixture = tmp_path/'fixture.json'
    fixture.write_text(json.dumps(dict(kind='fixture_id', ok=True, fixture_id_equal=True,
                                       encoder_source_sha256={'old': OLD['encoder_source_sha256'], 'new': NEW['encoder_source_sha256']},
                                       encoder_fixture_ids={'old': FIXTURE, 'new': FIXTURE})))
    # Encoder moves: the fixture-id arm is mandatory.
    assert run('proof-bundle', '--pins', pins_moving, '--arm', arm_moving, '--out', tmp_path/'b1.json') == 2
    assert 'fixture-id arm result is required' in capsys.readouterr().err
    assert run('proof-bundle', '--pins', pins_moving, '--fixture-id', fixture, '--arm', arm_moving,
               '--out', tmp_path/'b1.json') == 0
    assert json.loads((tmp_path/'b1.json').read_text())['ok'] is True
    # PQ-only step: the producer is unchanged, a fixture-id arm is refused and the bundle still certifies.
    assert run('proof-bundle', '--pins', pins_pq_only, '--fixture-id', fixture, '--arm', arm_pq_only,
               '--out', tmp_path/'b2.json') == 2
    assert 'does not move' in capsys.readouterr().err
    assert run('proof-bundle', '--pins', pins_pq_only, '--arm', arm_pq_only, '--out', tmp_path/'b2.json') == 0
    bundle = json.loads((tmp_path/'b2.json').read_text())
    assert bundle['ok'] and bundle['encoder_fixture_id_equal'] and bundle['fixture_id']['encoder_pin_unchanged']
    assert bundle['cell_count'] >= tool.MIN_CELLS and not bundle['strata_missing']
    # And that bundle drives a migration whose record survives an unchanged producer.
    row = tmp_path/'row'; make_row(row)
    assert run('migrate', '--pins', pins_pq_only, '--proof', tmp_path/'b2.json', '--row', row) == 0
    assert run('verify', '--pins', pins_pq_only, '--row', row) == 0
    loaded = tool.load_bundle(tmp_path/'b2.json', tool.load_pins(pins_pq_only))
    assert loaded['fixture_id']['ids'] is None
