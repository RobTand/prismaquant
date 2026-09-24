"""Synthetic catalog/capture artifacts; these tests qualify no model bytes."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_catalog_extension import (
    ADDED_FORMAT, ADDED_RECIPE, ADOPTION_SCHEMA, create_extension,
    require_extension, verify_catalog_pair,
)
from prismaquant.joint_adjoint_slices import (
    adjoint_slice_sha256, stage_a_run_header, stage_a_run_header_sha256, write_adjoint_slice,
    write_band_receipt)
from prismaquant.joint_layer_quanta import check_adjoint_run_header, layer_quanta
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.tessera_joint_aura import PREPARED_SCHEMA
from tests.test_joint_quanta_join import campaign, probe
from tests.test_stage_b_band_binding import band_from_receipt, synthetic_receipt


def _write(root, name, value, *, binary=False):
    raw = pickle.dumps(value) if binary else json.dumps(value, sort_keys=True).encode()
    path = root / name
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _encoder_proof(tmp_path):
    from tests.test_reseal_campaign_identity import _arm_result, _fixture_result
    old = {'prismaquant_source_sha256': 'a'*64, 'encoder_source_sha256': '4'*64}
    new = {**old, 'encoder_source_sha256': '8'*64}
    arm_path = _arm_result(tmp_path/'source-proof-arm.json', old, new)
    arm = json.loads(arm_path.read_text())
    routed = copy.deepcopy(arm['comparison']['cells'][-1])
    routed['qname'] = 'model.layers.0.mlp.experts.0.down_proj'
    arm['comparison']['cells'].append(routed)
    arm_bound = _write(tmp_path, arm_path.name, arm)
    fixture_path = _fixture_result(tmp_path/'source-fixture.json', old['encoder_source_sha256'], new['encoder_source_sha256'])
    fixture_bound = {'path': str(fixture_path), 'sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest()}
    cells, strata = [], {}
    for cell in arm['comparison']['cells']:
        kind = 'routed' if '.experts.' in cell['qname'] else 'dense'
        cells.append({**cell, 'kind': kind})
        strata.setdefault(kind, {}).setdefault(cell['family'], []).append(cell['body_rate_q256'])
    proof = {'schema': 'prismaquant.reseal_proof_bundle.v1', 'ok': True,
        'pins': {'old': old, 'new': new}, 'encoder_fixture_id_equal': True,
        'arms': [{'result': arm_bound['path'], 'result_sha256': arm_bound['sha256']}],
        'fixture_id': {'result': fixture_bound['path'], 'result_sha256': fixture_bound['sha256'],
                       'ids': {'old': 'f'*64, 'new': 'f'*64}},
        'source_checks': {'encoder': {'sha256': '8'*64, 'tree': str(tmp_path/'candidate-producer')}},
        'cells': cells, 'strata': strata, 'cell_count': len(cells), 'min_cells': 24}
    return _write(tmp_path, 'source-proof.json', proof)


def _scope_bindings(tmp_path, qnames):
    """A census, campaign plan, calibration, capture and checkpoint the
    original plan binds, the way a real joint plan binds them, so the one
    scope builder (``joint_campaign_scope``) can derive the campaign's scope
    from the plan alone."""
    from tests.test_joint_campaign_scope import _bind
    return _bind(tmp_path / 'scope', list(qnames), {'g0': list(qnames)})


def _receipt(inputs, campaign, probe, *, scope, identity=None):
    """The original run's completed receipt over the pair ``inputs``."""
    qnames = campaign['roster']
    original = json.loads(Path(inputs['original_prepared']['path']).read_bytes())
    oldplan = json.loads(Path(inputs['original_plan']['path']).read_bytes())
    return synthetic_receipt(
        plan_sha256=inputs['original_plan']['sha256'],
        prepared_sha256=inputs['original_prepared']['sha256'], scope=scope,
        num_layers=3, stride=8, root=oldplan['output_root'],
        identity={'calibration_sha256': original['calibration_input']['calibration_sha256'],
                  'calibration_shape': original['calibration_input']['shape'],
                  'n_probes': probe['n_probes'], 'seed_base': probe['seed_base'],
                  'unit_roster_sha256': hashlib.sha256(
                      ''.join(n+'\n' for n in sorted(qnames)).encode()).hexdigest(),
                  **(identity or {})})


def _pair(tmp_path, campaign, probe, *, scoped=False, identity=None):
    """``(inputs, receipt, capture)`` of a synthetic catalog pair.

    With ``scoped``, the original plan binds real scope artifacts
    (:func:`_scope_bindings`) and the receipt seals ``campaign_scope: None``,
    the way R13 sealed it (PQ #1126): the run declared no scope, and only the
    plan it ran can say which campaign it answers for. ``identity`` overrides
    run-identity fields of the receipt.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    qnames = campaign['roster']
    oldfmt = 'TESSERA_E4M3_K1_R1024'
    source = {'content_sha256': '1'*64, 'shape': [4, 4], 'dtype': 'torch.bfloat16', 'logical_bytes': 32}
    proof = _encoder_proof(tmp_path)
    old_weights, old_cells, new_weights, new_cells = {}, {}, {}, {}
    for name in qnames:
        identity = {'unit': name, 'source': {'sha256': '2'*64, 'shape': [4, 4]},
            'projection': {'kind': 'synthetic'}, 'calibration': {'hessian_sha256': '3'*64},
            'encoder_fixture_id': 'f'*64, 'encoder_source_sha256': '4'*64,
            'recipe': {**ADDED_RECIPE, 'grid': 'E4M3', 'q256': 1024, 'span': 1}}
        oldcell = {'source_weight': source, 'rendered_weight': {**source, 'content_sha256': '5'*64},
            'activation': {'input_global_scale': None}, 'encoding_identity_sha256': canonical_json_sha256(identity, where="synthetic identity"),
            'wire_sha256': '6'*64, 'render_file_sha256': '7'*64,
            'render_origin': 'encoded', 'render_comparison': 'independent_render_vs_wire'}
        old_weights[name, oldfmt] = '/fixture/' + name + '.old.pt'
        old_cells[name, oldfmt] = oldcell
        candidate = {**copy.deepcopy(identity), 'recipe': copy.deepcopy(ADDED_RECIPE), 'encoder_source_sha256': '8'*64}
        newcell = {**copy.deepcopy(oldcell), 'activation': {'input_global_scale': 0.5},
            'encoding_identity_sha256': canonical_json_sha256(candidate, where="synthetic candidate"),
            'catalog_source_adoption': {'schema': ADOPTION_SCHEMA,
                'reference_pair': [name, oldfmt], 'reference_encoding_identity': identity,
                'candidate_encoding_identity': candidate, 'encoder_source_proof': proof}}
        new_weights[name, ADDED_FORMAT] = '/fixture/' + name + '.a4.pt'
        new_cells[name, ADDED_FORMAT] = newcell
    new_weights.update(old_weights)
    new_cells.update(old_cells)
    common = {'source_model_identity': probe['source_model'],
        'source_execution': {'synthetic': True}, 'reader_identity': {'reader': 'synthetic'},
        'projection_backend': {'backend': 'synthetic'},
        'calibration_input': {'artifact_sha256': '9'*64, 'calibration_sha256': probe['calibration_sha256'],
                              'shape': probe['calibration_shape']}}
    oldplan = {'schema': 'prismaquant.tessera_joint_aura.plan.v1', 'model': '/mnt/shared/models/TEST',
        'calibration_input': {'path': '/fixture/calib.pt', 'sha256': '9'*64},
        'inputs': {'fixture': 'old'}, 'output_root': str(tmp_path / 'old-output'),
        'retained_window_budget_derivation': {'windows_by_layer': {str(i): 1 for i in range(3)}},
        'execution': {'n_probes': probe['n_probes'], 'seed_base': probe['seed_base'],
                      'source_derivative': {'fixture': 'bf16-exact'}, 'temperature': 1.0}}
    if scoped:
        bindings = _scope_bindings(tmp_path, qnames)
        oldplan['calibration_input'] = bindings['calibration_input']
        oldplan['canonical_capture'] = bindings['canonical_capture']
        oldplan['inputs'] = {'census': bindings['census'], 'campaign_plan': bindings['campaign_plan'],
                             'merged_checkpoint': bindings['merged_checkpoint'],
                             'required_source_units': len(qnames), 'required_campaign_groups': 1}
        oldplan['execution'].update(calib_seqlen=512, n_calib_samples=512)
        common['calibration_input']['artifact_sha256'] = bindings['calibration_input']['sha256']
    newplan = {**copy.deepcopy(oldplan), 'inputs': {'fixture': 'new'},
               'output_root': str(tmp_path / 'new-output')}
    inputs = {'original_plan': _write(tmp_path, 'old-plan.json', oldplan),
              'extended_plan': _write(tmp_path, 'new-plan.json', newplan)}
    for label, weights, cells, plan in [('original', old_weights, old_cells, oldplan),
                                       ('extended', new_weights, new_cells, newplan)]:
        metadata = {**common, 'schema': PREPARED_SCHEMA, 'inputs': plan['inputs'], 'verified_cells': cells}
        cache = ProductionWeightCache(weights=weights, metadata=metadata, levers={},
                                       activation_max_abs=dict.fromkeys(qnames, 12.0))
        prepared = {**common, 'schema': PREPARED_SCHEMA, 'status': 'complete',
            'plan_sha256': inputs[label+'_plan']['sha256'],
            'production_cache': _write(tmp_path, label+'.pkl', cache, binary=True),
            'formats_by_qname': {name: [oldfmt] + ([ADDED_FORMAT] if label == 'extended' else []) + ['BF16']
                                 for name in qnames}, 'measured_cells': len(cells)}
        inputs[label+'_prepared'] = _write(tmp_path, label+'-prepared.json', prepared)
    receipt = _receipt(inputs, campaign, probe, scope=None if scoped else campaign['scope'],
                       identity=identity)
    capture = _write(tmp_path, 'capture.json', receipt)
    return inputs, receipt, capture


def _campaign_identity(tmp_path, inputs):
    """The frozen campaign identity an operator seals once, bound by path and digest."""
    from tools.dispatch_tessera_campaign import campaign_identity, joint_campaign_scope
    plan = json.loads(Path(inputs['original_plan']['path']).read_bytes())
    return _write(tmp_path, 'campaign-identity.json', campaign_identity(joint_campaign_scope(plan)))


def _expected_scope(inputs, identity):
    """The scope the dispatcher stamps on a campaign-scoped submission
    (``dispatch_tessera_campaign.cmd_submit_joint``)."""
    from tools.dispatch_tessera_campaign import joint_campaign_scope
    plan = json.loads(Path(inputs['original_plan']['path']).read_bytes())
    return {**joint_campaign_scope(plan), 'campaign_identity_sha256': identity['sha256']}


def _adopt_scope(campaign, scope):
    """The parent manifest the Stage B prepare extends annotates ``scope``."""
    campaign['scope'] = scope
    campaign['parent_manifest']['annotations']['campaign_scope'] = scope


def _scoped_case(tmp_path, campaign, probe, *, identity=None):
    """A null-scope original run over a scoped plan, its identity file, and
    the parent whose annotated scope is the one the plan derives."""
    inputs, receipt, capture = _pair(tmp_path, campaign, probe, scoped=True, identity=identity)
    assert receipt['run_identity']['campaign_scope'] is None
    campaign_identity = _campaign_identity(tmp_path, inputs)
    scope = _expected_scope(inputs, campaign_identity)
    _adopt_scope(campaign, scope)
    return inputs, receipt, capture, campaign_identity, scope


def _site_args(inputs, campaign):
    return dict(plan_sha256=inputs['extended_plan']['sha256'],
                prepared_sha256=inputs['extended_prepared']['sha256'],
                scope=campaign['scope'], checkpoints=[3])


def _join_campaign(inputs, receipt, campaign):
    """The joiner's view of the campaign: the proof and what the parent annotates."""
    return {'adjoint_receipt': receipt, 'adjoint_bands': [],
            'plan_sha256': inputs['extended_plan']['sha256'],
            'prepared_sha256': inputs['extended_prepared']['sha256'], 'scope': campaign['scope']}


def _all_four_sites(tmp_path, inputs, receipt, capture, campaign, bound):
    """Run the generator, the dispatcher, the quantum gate and the joiner's
    header check on one proof set.

    Returns the generator's records. Every site raises its own refusal type
    when the run header does not answer for the campaign.
    """
    from prismaquant.joint_cost_quantum import verify_quantum_identity
    from prismaquant.joint_quanta_join import _check_stage_a_header
    from tools.dispatch_joint_quanta import check_stage_a_proofs
    plan = json.loads(Path(inputs['extended_plan']['path']).read_bytes())
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    args = _site_args(inputs, campaign)
    produced = layer_quanta(plan, prepared, campaign['parent_manifest'],
        parent_manifest_sha256=campaign['manifest_sha256'],
        plan_path=inputs['extended_plan']['path'], prepared_path=inputs['extended_prepared']['path'],
        plan_sha256=args['plan_sha256'], prepared_sha256=args['prepared_sha256'],
        catalog_extension=bound, stride=8, adjoint_receipt=receipt)
    rows = []
    for record in produced['records']:
        adjoint_slice = produced['adjoint_slices'][record['quantum_id']]
        write_adjoint_slice(record['adjoint']['slice_path'], adjoint_slice, layer=record['layer'])
        rows.append((tmp_path/f"{record['quantum_id']}.json", record))
    check_stage_a_proofs([(Path(capture['path']), receipt)], rows)
    first = produced['records'][0]
    quantum = _write(tmp_path, 'quantum.json', first)
    verify_quantum_identity(
        quantum_path=Path(quantum['path']), quantum_sha256=quantum['sha256'],
        plan_path=Path(inputs['extended_plan']['path']), plan_sha256=args['plan_sha256'],
        prepared_path=Path(inputs['extended_prepared']['path']), prepared_sha256=args['prepared_sha256'],
        adjoint_path=Path(first['adjoint']['slice_path']), adjoint_sha256=first['adjoint']['slice_sha256'],
        output_root=Path(plan['output_root']))
    _check_stage_a_header(_join_campaign(inputs, receipt, campaign),
                          {record['quantum_id']: record for record in produced['records']})
    return produced['records']


def _rescoped_record(record, scope):
    """An honest record of a campaign whose parent annotates ``scope``."""
    from prismaquant.joint_layer_quanta import canonical_sha256
    other = copy.deepcopy(record)
    other['campaign']['campaign_scope'] = scope
    body = {key: value for key, value in other.items() if key != 'identity_sha256'}
    other['identity_sha256'] = canonical_sha256(body, where='rescoped record')
    return other


def test_a_null_scope_run_is_admitted_through_the_derived_scope_at_every_site(tmp_path, campaign, probe):
    """R13 sealed ``campaign_scope: null`` (no forward-recovery capsule, a plan
    that declares no scope), while the parent it read annotates the campaign's
    ``complete_campaign`` scope (PQ #1126). The extension derives the scope
    once from the original plan with the one scope builder, binds the frozen
    identity file, and the generator, the dispatcher, the quantum and the joiner admit
    the run under it."""
    from prismaquant.joint_catalog_extension import SCHEMA_V3, DERIVED_SCOPE_SCHEMA
    inputs, receipt, capture, identity, scope = _scoped_case(tmp_path, campaign, probe)
    header = stage_a_run_header(receipt)
    args = _site_args(inputs, campaign)
    # Without the identity file, the null scope cannot be derived: refuse.
    with pytest.raises(ValueError, match='campaign-identity'):
        create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'no-identity.json')
    assert not (tmp_path/'no-identity.json').exists()
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json',
                             campaign_identity=identity)
    document = json.loads(Path(bound['path']).read_bytes())
    assert document['schema'] == SCHEMA_V3
    block = document['original_campaign_scope']
    assert block['schema'] == DERIVED_SCOPE_SCHEMA
    assert block['scope'] == scope
    assert block['derived_from']['original_plan'] == inputs['original_plan']
    assert block['derived_from']['campaign_identity'] == identity
    assert block['derived_from']['require_scope'] == 'complete_campaign'
    # The sealed identity is null; the effective one is the derived scope.
    effective = require_extension(bound, run_header=header, plan_sha256=args['plan_sha256'],
                                  prepared_sha256=args['prepared_sha256'])
    assert effective['campaign_scope'] == scope
    assert {k: v for k, v in effective.items() if k != 'campaign_scope'} == \
        {k: v for k, v in header['run_identity'].items() if k != 'campaign_scope'}
    assert check_adjoint_run_header(header, **args, catalog_extension=bound) == \
        stage_a_run_header_sha256(header)
    records = _all_four_sites(tmp_path, inputs, receipt, capture, campaign, bound)
    assert len(records) == 3
    assert all(record['campaign']['campaign_scope'] == scope for record in records)
    # The first sealed band of the run creates the same bytes as the receipt.
    band = band_from_receipt(receipt, 3)
    band_path = tmp_path/'band-003.json'
    band_file = {'path': str(band_path), 'sha256': write_band_receipt(band_path, band)}
    from_band = create_extension(inputs=inputs, adjoint_capture=band_file, campaign_identity=identity,
                                 output=tmp_path/'extension-from-band.json')
    assert from_band['sha256'] == bound['sha256']


def test_a_derived_scope_that_is_not_the_parents_refuses_at_every_site(tmp_path, campaign, probe):
    """The parent annotates another campaign (here, another checkpoint digest):
    the derived scope is still this plan's, and it is not the parent's, so the
    generator, the dispatcher, the quantum and the joiner refuse."""
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused, verify_quantum_identity
    from prismaquant.joint_quanta_join import JoinRefused, _check_stage_a_header
    from tools.dispatch_joint_quanta import DispatchRefused, check_stage_a_proofs
    inputs, receipt, capture, identity, scope = _scoped_case(tmp_path, campaign, probe)
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json',
                             campaign_identity=identity)
    records = _all_four_sites(tmp_path, inputs, receipt, capture, campaign, bound)
    other = {**scope, 'campaign_checkpoint_sha256': '0' * 64}
    header = stage_a_run_header(receipt)
    with pytest.raises(ValueError, match='another scope'):
        check_adjoint_run_header(header, **{**_site_args(inputs, campaign), 'scope': other},
                                 catalog_extension=bound)
    _adopt_scope(campaign, other)
    plan = json.loads(Path(inputs['extended_plan']['path']).read_bytes())
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    with pytest.raises(ValueError, match='another scope'):
        layer_quanta(plan, prepared, campaign['parent_manifest'],
            parent_manifest_sha256=campaign['manifest_sha256'],
            plan_path=inputs['extended_plan']['path'], prepared_path=inputs['extended_prepared']['path'],
            plan_sha256=inputs['extended_plan']['sha256'], prepared_sha256=inputs['extended_prepared']['sha256'],
            catalog_extension=bound, stride=8, adjoint_receipt=receipt)
    rescoped = [_rescoped_record(record, other) for record in records]
    with pytest.raises(DispatchRefused, match='another scope'):
        check_stage_a_proofs([(Path(capture['path']), receipt)],
                             [(tmp_path/f"{r['quantum_id']}.json", r) for r in rescoped])
    quantum = _write(tmp_path, 'rescoped-quantum.json', rescoped[0])
    with pytest.raises(QuantumIdentityRefused, match='another scope'):
        verify_quantum_identity(
            quantum_path=Path(quantum['path']), quantum_sha256=quantum['sha256'],
            plan_path=Path(inputs['extended_plan']['path']), plan_sha256=inputs['extended_plan']['sha256'],
            prepared_path=Path(inputs['extended_prepared']['path']),
            prepared_sha256=inputs['extended_prepared']['sha256'],
            adjoint_path=Path(rescoped[0]['adjoint']['slice_path']),
            adjoint_sha256=rescoped[0]['adjoint']['slice_sha256'],
            output_root=Path(plan['output_root']))
    with pytest.raises(JoinRefused, match='another scope'):
        _check_stage_a_header(_join_campaign(inputs, receipt, campaign),
                              {record['quantum_id']: record for record in rescoped})


def test_a_null_scope_proof_without_a_derived_scope_refuses(tmp_path, campaign, probe):
    """No extension, or an extension that derives nothing (a v2 document
    bound to the null-scope header), leaves the sealed null, which never
    equals the parent's scope. ``null == null`` never admits either."""
    from prismaquant.joint_catalog_extension import SCHEMA
    from prismaquant.joint_layer_quanta import check_adjoint_run_identity
    inputs, receipt, capture, identity, scope = _scoped_case(tmp_path, campaign, probe)
    header = stage_a_run_header(receipt)
    args = _site_args(inputs, campaign)
    # Without an extension the header answers for the original plan; asked
    # under that plan, the sealed null is what refuses, not the plan digest.
    with pytest.raises(ValueError, match='another scope'):
        check_adjoint_run_header(header, **{**args,
                                            'plan_sha256': inputs['original_plan']['sha256'],
                                            'prepared_sha256': inputs['original_prepared']['sha256']})
    with pytest.raises(ValueError):
        check_adjoint_run_header(header, **args)
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json',
                             campaign_identity=identity)
    document = json.loads(Path(bound['path']).read_bytes())
    del document['original_campaign_scope']
    document['schema'] = SCHEMA
    v2 = _write(tmp_path, 'v2-on-null.json', document)
    with pytest.raises(ValueError, match='no campaign scope'):
        require_extension(v2, run_header=header, plan_sha256=args['plan_sha256'],
                          prepared_sha256=args['prepared_sha256'])
    with pytest.raises(ValueError, match='no campaign scope|another scope'):
        check_adjoint_run_header(header, **args, catalog_extension=v2)
    # A consumer handed no scope at all is refused before any comparison.
    for unset in (None, {}):
        with pytest.raises(ValueError, match='scope'):
            check_adjoint_run_identity(header, plan_sha256=args['plan_sha256'],
                                       prepared_sha256=args['prepared_sha256'], scope=unset,
                                       catalog_extension=bound)
        with pytest.raises(ValueError, match='scope'):
            check_adjoint_run_identity(
                {**header, 'run_identity': {**header['run_identity'],
                                            'plan_sha256': args['plan_sha256'],
                                            'prepared_sha256': args['prepared_sha256']}},
                plan_sha256=args['plan_sha256'], prepared_sha256=args['prepared_sha256'], scope=unset)


def test_a_sealed_scope_other_than_the_parents_refuses_even_with_an_extension(tmp_path, campaign, probe):
    """A run that sealed a scope answers for that scope: the extension derives
    nothing over it, and a parent annotating another scope refuses."""
    inputs, receipt, capture = _pair(tmp_path, campaign, probe)
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json')
    header = stage_a_run_header(receipt)
    args = _site_args(inputs, campaign)
    assert check_adjoint_run_header(header, **args, catalog_extension=bound)
    other = {**campaign['scope'], 'campaign': 'another-campaign'}
    with pytest.raises(ValueError, match='another scope'):
        check_adjoint_run_header(header, **{**args, 'scope': other}, catalog_extension=bound)
    _adopt_scope(campaign, other)
    with pytest.raises(ValueError, match='another scope'):
        _all_four_sites(tmp_path, inputs, receipt, capture, campaign, bound)
    # A sealed scope with a derived block on top refuses: the sealed scope rules.
    inputs2, receipt2, capture2, identity, scope = _scoped_case(tmp_path/'scoped', campaign, probe)
    derived = create_extension(inputs=inputs2, adjoint_capture=capture2, campaign_identity=identity,
                               output=tmp_path/'derived.json')
    document = json.loads(Path(derived['path']).read_bytes())
    sealed = copy.deepcopy(receipt2)
    sealed['run_identity']['campaign_scope'] = scope
    document['adjoint_run_header'] = stage_a_run_header(sealed)
    document['adjoint_run_header_sha256'] = canonical_json_sha256(
        document['adjoint_run_header'], where='sealed header')
    forged = _write(tmp_path, 'derived-over-sealed.json', document)
    with pytest.raises(ValueError, match='sealed'):
        require_extension(forged, run_header=stage_a_run_header(sealed),
                          plan_sha256=inputs2['extended_plan']['sha256'],
                          prepared_sha256=inputs2['extended_prepared']['sha256'])


@pytest.mark.parametrize('change', ['bytes', 'identity_field', 'checkpoint', 'kind', 'windows',
                                    'identity_binding', 'plan_binding', 'artifact_binding',
                                    'identity_stamp', 'rule'])
def test_a_tampered_derived_scope_refuses(tmp_path, campaign, probe, change):
    """Every field of the embedded scope is rechecked cheaply by consumers:
    the identity file's fields, the plan's stated digests and counts, and the
    bindings the derivation names. Changed bytes under the bound digest
    refuse before any field is read."""
    from tools.dispatch_tessera_campaign import campaign_identity
    inputs, receipt, capture, identity, scope = _scoped_case(tmp_path, campaign, probe)
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json',
                             campaign_identity=identity)
    header = stage_a_run_header(receipt)
    args = _site_args(inputs, campaign)
    document = json.loads(Path(bound['path']).read_bytes())
    block = document['original_campaign_scope']
    if change == 'bytes':
        Path(bound['path']).write_bytes(Path(bound['path']).read_bytes() + b'\n')
        tampered = bound
    else:
        if change == 'identity_field': block['scope']['source_unit_count'] += 1
        elif change == 'checkpoint': block['scope']['campaign_checkpoint_sha256'] = '0' * 64
        elif change == 'kind': block['scope']['kind'] = 'diagnostic'
        elif change == 'windows': block['scope']['window_count'] = 16
        elif change == 'identity_stamp': block['scope']['campaign_identity_sha256'] = '0' * 64
        elif change == 'rule': block['rule'] = 'operator_declared'
        elif change == 'plan_binding':
            block['derived_from']['original_plan'] = {**inputs['original_plan'], 'sha256': '0' * 64}
        elif change == 'artifact_binding':
            block['derived_from']['bound_artifacts']['census']['sha256'] = '0' * 64
        elif change == 'identity_binding':
            # Another identity file whose fields agree with the tampered scope:
            # the plan's stated digests and the parent's scope still refuse.
            forged = campaign_identity({**block['scope'], 'campaign_checkpoint_sha256': '0' * 64})
            block['derived_from']['campaign_identity'] = _write(tmp_path, 'forged-identity.json', forged)
            block['scope']['campaign_checkpoint_sha256'] = '0' * 64
        tampered = _write(tmp_path, 'tampered.json', document)
    with pytest.raises(ValueError):
        require_extension(tampered, run_header=header, plan_sha256=args['plan_sha256'],
                          prepared_sha256=args['prepared_sha256'])
    with pytest.raises(ValueError):
        check_adjoint_run_header(header, **args, catalog_extension=tampered)


def test_a_sealed_scope_header_keeps_the_v2_document_and_its_identity(tmp_path, campaign, probe):
    """A run that sealed its scope creates the v2 bytes it always did, with
    or without an identity binding in hand, and consumers get the sealed
    identity unchanged. Existing v2 documents keep verifying."""
    from prismaquant.joint_catalog_extension import SCHEMA
    inputs, receipt, capture = _pair(tmp_path, campaign, probe, scoped=True)
    receipt['run_identity']['campaign_scope'] = campaign['scope']
    capture = _write(tmp_path, 'sealed-capture.json', receipt)
    identity = _campaign_identity(tmp_path, inputs)
    plain = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'plain.json')
    with_identity = create_extension(inputs=inputs, adjoint_capture=capture, campaign_identity=identity,
                                     output=tmp_path/'with-identity.json')
    assert plain['sha256'] == with_identity['sha256']
    document = json.loads(Path(plain['path']).read_bytes())
    assert document['schema'] == SCHEMA
    assert 'original_campaign_scope' not in document
    header = stage_a_run_header(receipt)
    args = _site_args(inputs, campaign)
    assert require_extension(plain, run_header=header, plan_sha256=args['plan_sha256'],
                             prepared_sha256=args['prepared_sha256']) == header['run_identity']
    assert check_adjoint_run_header(header, **args, catalog_extension=plain)


def test_additive_catalog_binds_original_capture_without_relabelling(tmp_path, campaign, probe):
    inputs, receipt, capture = _pair(tmp_path, campaign, probe)
    original = Path(capture['path']).read_bytes()
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json')
    args = dict(plan_sha256=inputs['extended_plan']['sha256'],
                prepared_sha256=inputs['extended_prepared']['sha256'],
                scope=campaign['scope'], checkpoints=[3])
    header = stage_a_run_header(receipt)
    with pytest.raises(ValueError, match='another plan_sha256'):
        check_adjoint_run_header(header, **args)
    result = check_adjoint_run_header(header, **args, catalog_extension=bound)
    assert result == stage_a_run_header_sha256(header)
    assert Path(capture['path']).read_bytes() == original
    # PQ #993: the extension binds the original run header, so the first
    # sealed band of the run creates the same extension bytes as the receipt.
    band = band_from_receipt(receipt, 3)
    band_path = tmp_path/'band-003.json'
    band_file = {'path': str(band_path), 'sha256': write_band_receipt(band_path, band)}
    from_band = create_extension(inputs=inputs, adjoint_capture=band_file,
                                 output=tmp_path/'extension-from-band.json')
    assert from_band['sha256'] == bound['sha256']
    plan = json.loads(Path(inputs['extended_plan']['path']).read_bytes())
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    common = dict(parent_manifest_sha256=campaign['manifest_sha256'],
        plan_path=inputs['extended_plan']['path'], prepared_path=inputs['extended_prepared']['path'],
        plan_sha256=args['plan_sha256'], prepared_sha256=args['prepared_sha256'],
        catalog_extension=bound, stride=8)
    produced = layer_quanta(plan, prepared, campaign['parent_manifest'],
                            adjoint_receipt=receipt, **common)
    banded = layer_quanta(plan, prepared, campaign['parent_manifest'],
                          adjoint_receipts=[band], **common)
    assert json.dumps(banded['records'], sort_keys=True) == json.dumps(produced['records'], sort_keys=True)
    assert len(produced['records']) == 3
    for record in produced['records']:
        adjoint_slice = produced['adjoint_slices'][record['quantum_id']]
        assert record['adjoint']['slice_sha256'] == adjoint_slice_sha256(adjoint_slice)
        assert 'receipt_sha256' not in record['adjoint']
        assert record['campaign']['prepared_sha256'] == args['prepared_sha256']
        assert record['catalog_extension'] == bound
        write_adjoint_slice(record['adjoint']['slice_path'], adjoint_slice, layer=record['layer'])
    from tools.dispatch_joint_quanta import check_stage_a_proofs, DispatchRefused
    rows = [(tmp_path/f"{record['quantum_id']}.json", record) for record in produced['records']]
    assert check_stage_a_proofs([(Path(capture['path']), receipt)], rows) == {
        record['quantum_id']: record['adjoint']['slice_sha256'] for record in produced['records']}
    unbound = {**produced['records'][1], 'catalog_extension': None}
    with pytest.raises(DispatchRefused, match='extension bindings differ'):
        check_stage_a_proofs([(Path(capture['path']), receipt)],
                             [rows[0], (tmp_path/'r.json', unbound)])
    from prismaquant.joint_cost_quantum import verify_quantum_identity
    first = produced['records'][0]
    quantum = _write(tmp_path, 'quantum.json', first)
    loaded, adjoint_slice = verify_quantum_identity(
        quantum_path=Path(quantum['path']), quantum_sha256=quantum['sha256'],
        plan_path=Path(inputs['extended_plan']['path']), plan_sha256=args['plan_sha256'],
        prepared_path=Path(inputs['extended_prepared']['path']), prepared_sha256=args['prepared_sha256'],
        adjoint_path=Path(first['adjoint']['slice_path']),
        adjoint_sha256=first['adjoint']['slice_sha256'],
        output_root=Path(plan['output_root']))
    assert loaded == first
    assert adjoint_slice == produced['adjoint_slices'][first['quantum_id']]
    # The production metadata CLI carries the explicit bridge into immutable
    # records; it does not rewrite or relocate the original adjoint capture.
    from tools.regenerate_joint_quanta import main as regenerate
    manifest = _write(tmp_path, 'parent.json', campaign['parent_manifest'])
    derivation = _write(tmp_path, 'derivation.json', {'stride': 8})
    metadata = tmp_path/'extension-metadata'
    assert regenerate([
        '--plan', inputs['extended_plan']['path'], '--plan-sha256', args['plan_sha256'],
        '--prepared', inputs['extended_prepared']['path'], '--prepared-sha256', args['prepared_sha256'],
        '--parent-manifest', manifest['path'], '--parent-manifest-sha256', manifest['sha256'],
        '--derivation', derivation['path'], '--metadata-root', str(metadata),
        '--adjoint-receipt', capture['path'], '--catalog-extension', bound['path'],
        '--catalog-extension-sha256', bound['sha256']]) == 0
    written = [json.loads(path.read_bytes()) for path in (metadata/'records').glob('layer-*.json')]
    assert len(written) == 3
    assert all(record['catalog_extension'] == bound for record in written)
    assert Path(capture['path']).read_bytes() == original


def test_a_recovered_stage_a_binds_its_canonical_roster_spelling(tmp_path, campaign, probe):
    """A Stage A resumed from a forward-recovery capsule (R10 onward) seals
    the campaign's canonical roster, ``roster_digest``, not a fresh run's
    newline spelling. The extension checks the spelling the header implies,
    so a recovered run's band can create it; either spelling on the other
    kind of run refuses."""
    from prismaquant.joint_catalog_extension import extension_run_header
    from prismaquant.joint_layer_quanta import roster_digest
    inputs, receipt, _ = _pair(tmp_path, campaign, probe)
    newline = receipt['run_identity']['unit_roster_sha256']
    canonical = roster_digest(campaign['roster'])
    assert canonical != newline
    recovered = copy.deepcopy(receipt)
    recovered['boundary_storage']['forward_recovery'] = {
        'path': '/fixture/forward-recovery.json', 'sha256': 'e' * 64}
    recovered['run_identity']['unit_roster_sha256'] = canonical
    bound = create_extension(inputs=inputs, output=tmp_path/'ext-recovered.json',
                             adjoint_capture=_write(tmp_path, 'recovered.json', recovered))
    assert extension_run_header(bound) == stage_a_run_header(recovered)
    stale = copy.deepcopy(recovered)
    stale['run_identity']['unit_roster_sha256'] = newline
    fresh = copy.deepcopy(receipt)
    fresh['run_identity']['unit_roster_sha256'] = canonical
    for name, document in (('stale', stale), ('fresh', fresh)):
        with pytest.raises(ValueError, match='capture qname roster'):
            create_extension(inputs=inputs, output=tmp_path/f'ext-{name}.json',
                             adjoint_capture=_write(tmp_path, f'{name}.json', document))


@pytest.mark.parametrize('change', ['source', 'calibration', 'derivative', 'qnames', 'old_cell',
                                    'old_path', 'unqualified', 'adopted_hessian', 'recipe'])
def test_extension_refuses_scientific_or_original_candidate_drift(tmp_path, campaign, probe, change):
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    cache = pickle.loads(Path(prepared['production_cache']['path']).read_bytes())
    pair = next(p for p in cache.weights if p[1] == ADDED_FORMAT)
    oldpair = (pair[0], 'TESSERA_E4M3_K1_R1024')
    if change == 'source': prepared['source_model_identity']['content_sha256'] = '0'*64
    elif change == 'calibration': prepared['calibration_input']['calibration_sha256'] = '0'*64
    elif change == 'qnames': del prepared['formats_by_qname'][pair[0]]
    elif change == 'derivative':
        plan = json.loads(Path(inputs['extended_plan']['path']).read_bytes())
        plan['execution']['source_derivative'] = {'different': True}
        inputs['extended_plan'] = _write(tmp_path, 'new-plan.json', plan)
        prepared['plan_sha256'] = inputs['extended_plan']['sha256']
    elif change == 'old_cell': cache.metadata['verified_cells'][oldpair]['wire_sha256'] = '0'*64
    elif change == 'old_path': cache.weights[oldpair] = '/changed.pt'
    elif change == 'unqualified': del cache.metadata['verified_cells'][pair]['render_file_sha256']
    elif change in ('adopted_hessian', 'recipe'):
        cell = cache.metadata['verified_cells'][pair]
        identity = cell['catalog_source_adoption']['candidate_encoding_identity']
        if change == 'adopted_hessian': identity['calibration']['hessian_sha256'] = '0'*64
        else: identity['recipe']['seed'] = 99
        cell['encoding_identity_sha256'] = canonical_json_sha256(identity, where="synthetic identity")
    prepared['production_cache'] = _write(tmp_path, 'extended.pkl', cache, binary=True)
    inputs['extended_prepared'] = _write(tmp_path, 'extended-prepared.json', prepared)
    with pytest.raises(ValueError):
        verify_catalog_pair(inputs)


def test_incomplete_capture_and_retargeted_proof_never_authorize_extension(tmp_path, campaign, probe):
    inputs, receipt, capture = _pair(tmp_path, campaign, probe)
    receipt['status'] = 'running'
    bad = _write(tmp_path, 'incomplete.json', receipt)
    with pytest.raises(ValueError, match='completed Stage A'):
        create_extension(inputs=inputs, adjoint_capture=bad, output=tmp_path/'refused.json')
    assert not (tmp_path/'refused.json').exists()
    receipt['status'] = 'complete'
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json')
    header = stage_a_run_header(receipt)
    with pytest.raises(ValueError, match='extended plan binding'):
        require_extension(bound, run_header=header, plan_sha256='0'*64,
                          prepared_sha256=inputs['extended_prepared']['sha256'])
    # A header of another Stage A run never matches the bound one.
    other = copy.deepcopy(header)
    other['boundary_storage']['session']['generation'] = 'another-generation'
    with pytest.raises(ValueError, match='original Stage A run header'):
        require_extension(bound, run_header=other, plan_sha256=inputs['extended_plan']['sha256'],
                          prepared_sha256=inputs['extended_prepared']['sha256'])


def test_cached_pair_revalidates_changed_metadata_bytes(tmp_path, campaign, probe):
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    assert verify_catalog_pair(inputs)['added_cells'] == 6
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    path = Path(prepared['production_cache']['path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_catalog_pair(inputs)


def test_cached_pair_revalidates_source_proof_and_publication_is_no_clobber(tmp_path, campaign, probe):
    inputs, receipt, capture = _pair(tmp_path, campaign, probe)
    out = tmp_path/'extension.json'
    create_extension(inputs=inputs, adjoint_capture=capture, output=out)
    before = out.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        create_extension(inputs=inputs, adjoint_capture=capture, output=out)
    assert out.read_bytes() == before
    (tmp_path/'source-proof.json').write_text('changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_catalog_pair(inputs)


_SYNTHETIC_HESSIAN = {'supplied': True, 'capture_sha256': 'f'*64, 'text_sha256': 'e'*64,
                      'fit_ids_sha256': 'd'*64, 'fit_tokens': 4}


def _overlay_case(tmp_path, campaign, probe, *, row_hessian=_SYNTHETIC_HESSIAN,
                  panel_hessian=_SYNTHETIC_HESSIAN, overlay_hessian=None):
    """A bound candidate overlay over the synthetic pair, and the base it attaches to.

    ``row_hessian`` is what every overlay scalar row carries,
    ``panel_hessian`` the base payload's ``provenance.hessian`` and
    ``overlay_hessian`` the overlay cost payload's own ``provenance.hessian``.
    """
    from types import SimpleNamespace
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    original = json.loads(Path(inputs['original_prepared']['path']).read_bytes())
    extended = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    cache = pickle.loads(Path(extended['production_cache']['path']).read_bytes())
    base_cells, rows, scalar_costs = {}, [], {}
    for (name, fmt), verified in sorted(cache.metadata['verified_cells'].items()):
        if fmt != ADDED_FORMAT:
            continue
        adoption = verified['catalog_source_adoption']
        reference_pair = tuple(adoption['reference_pair'])
        base_cells[reference_pair] = {'record': {'identity': adoption['reference_encoding_identity']}}
        anchor = {'dloss': 1.0, 'family': 'TESSERA_E2M1_K2', 'body_rate_q256': 896,
                  'input_global_scale': 0.5, 'wire_bytes': 4,
                  'activation_contract': 'fp4_e2m1', 'activation_quantized': True}
        scalar_costs[name] = {fmt: {'output_mse_measured': True, 'cost_source': 'tessera_campaign_measured',
            'tessera_provenance': 'measured', 'currency': 'output_mse_under_route_activation_contract',
            'output_mse': 1.0, 'tessera_family': anchor['family'], 'tessera_body_rate_q256': 896,
            'input_global_scale': 0.5, 'wire_bytes': 4, 'activation_contract': anchor['activation_contract'],
            'activation_quantized': True, 'hessian_identity': copy.deepcopy(row_hessian)}}
        paths = {}
        for field in ('wire', 'render'):
            path = tmp_path/(name+'.'+field)
            path.write_bytes(field.encode())
            s = path.stat()
            paths[field] = str(path)
            paths[field+'_stat'] = dict(inode=s.st_ino, bytes=s.st_size,
                                       mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
        rows.append({**copy.deepcopy(verified), **paths, 'qname': name, 'format': fmt,
            'anchor': anchor, 'adopted_source_hessian': {'synthetic': True},
            'record': {'identity': adoption['candidate_encoding_identity'],
                       'blob_sha256': hashlib.sha256(b'wire').hexdigest()}})

    def bind(mutate=None):
        if mutate is not None:
            mutate(rows, scalar_costs)
        cost = {'costs': scalar_costs}
        if overlay_hessian is not None:
            cost['provenance'] = {'hessian': overlay_hessian}
        catalog = {'schema': 'prismaquant.t4_adopted_catalog.v1', 'format': ADDED_FORMAT,
            'old_prepared': inputs['original_prepared'], 'old_pwc': original['production_cache'],
            'cost': _write(tmp_path, 'scalar.pkl', cost, binary=True),
            'reseal_proof': rows[0]['catalog_source_adoption']['encoder_source_proof'], 'cells': rows,
            **{key: original[key] for key in ('source_model_identity', 'source_execution', 'calibration_input',
                                             'reader_identity', 'projection_backend')}}
        bound = _write(tmp_path, 'catalog.json', catalog)
        data = SimpleNamespace(inputs={'fixture': 'old', 'candidate_overlay': bound},
            cells=copy.deepcopy(base_cells), formats_by_qname=copy.deepcopy(original['formats_by_qname']),
            payload={'provenance': {'hessian': copy.deepcopy(panel_hessian)},
                     'costs': {name: {} for name in original['formats_by_qname']}})
        return data, bound

    return SimpleNamespace(bind=bind, rows=rows, scalar_costs=scalar_costs, base_cells=base_cells,
                           qnames=sorted(original['formats_by_qname']))


@pytest.mark.parametrize('change', [None, 'hessian', 'scalar', 'missing', 'render_changed'])
def test_overlay_intake_preserves_base_and_consumes_only_bound_measured_additions(tmp_path, campaign, probe, change):
    from prismaquant.joint_catalog_extension import attach_candidate_overlay
    case = _overlay_case(tmp_path, campaign, probe)

    def mutate(rows, scalar_costs):
        if change == 'scalar': scalar_costs[rows[0]['qname']][ADDED_FORMAT]['output_mse'] = 2.0
        elif change == 'hessian':
            rows[0]['record']['identity']['calibration']['hessian_sha256'] = '0'*64
        elif change == 'missing': rows.pop()
        elif change == 'render_changed': Path(rows[0]['render']).write_bytes(b'drifted')

    data, bound = case.bind(mutate)
    if change is not None:
        with pytest.raises(ValueError):
            attach_candidate_overlay(data, bound, verify_payloads=True)
    else:
        result = attach_candidate_overlay(data, bound, verify_payloads=True)
        assert {pair: result.cells[pair] for pair in case.base_cells} == case.base_cells
        assert len(result.cells) == 12
        assert all(ADDED_FORMAT in row for row in result.formats_by_qname.values())
        assert result.payload['costs'] == case.scalar_costs


# Two workspaces commit Hessians from one canonical capture: the overlay's cost
# run covers the routed experts it priced, the panel covers every unit plus a
# dense unit the overlay never prices. The run of 2026-09-22 (PB 53a9d3399086)
# had exactly this shape and the loader refused it on the references files'
# capture seals, which differ whenever the unit rosters do.
_EXTRA_DENSE = 'model.layers.0.mlp.shared_expert.down_proj'
_REFERENCE_POLICY = dict(schema='tessera.hessian_reference_load.v1', max_metadata_bytes=1024**2,
                         max_file_bytes=1024**2, max_hessian_bytes=1024**2)


def _canonical_capture(root, names, *, max_abs=4.0, act_value=1.0, census=None):
    """Publish one complete canonical capture; ``census`` reuses an existing census file."""
    import torch
    from prismaquant import tessera_calibration_cache as cc
    from tests.test_tessera_calibration_cache import canonical_fields, identity
    from tests.test_tessera_priced_export_inputs import TRIPLE
    root.mkdir(parents=True)
    if census is None:
        model = root/'source'
        model.mkdir()
        (model/'config.json').write_text('{}')
        (model/'model.safetensors').write_bytes(b'fixture')
        document = dict(model=str(model), counts=dict.fromkeys(names, 5), max_abs=dict.fromkeys(names, max_abs),
                        unit_shapes={name: [3, 2] for name in names}, layer_stride=1,
                        anchor_groups={'u:'+name: [name] for name in names}, **canonical_fields())
        census = root/'census.json'
        census.write_text(json.dumps(document))
    document = json.loads(census.read_text())
    calibration = dict(TRIPLE, model=document['model'], seqlen=8, source='fixture')
    H = {name: torch.eye(2)*(3+i) for i, name in enumerate(names)}
    record = cc.publish_capture(root/'capture', census_path=census,
        identity=identity(census, calibration=calibration, max_act_rows=2),
        acts={name: torch.full((2, 2), act_value) for name in names}, hessians=H,
        counts=document['counts'], maxima=document['max_abs'])
    return dict(record=record, census=census, calibration=calibration, H=H, counts=document['counts'])


def _workspace_hessian(root, canonical, units, *, census_copy=False, override=None):
    """One workspace's reference file over ``units`` and the H provenance its cost run stamps."""
    import shutil
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant import tessera_campaign as tc
    root.mkdir(parents=True)
    census = canonical['census']
    if census_copy:
        # Same census bytes under another workspace's path, as each GLM
        # extension workspace keeps its own census.json.
        census = root/'census.json'
        shutil.copyfile(canonical['census'], census)
    hessians = {name: canonical['H'][name] for name in units}
    hessians.update(override or {})
    path, _, digest = tc.write_export_inputs(root, hessians=hessians, hessian_rows=canonical['counts'],
        hessian_identity=canonical['calibration'], static_scales={}, static_scale_policy='fixture',
        hessian_reference=dict(canonical_capture=canonical['record'], census_path=census,
                               load_policy=_REFERENCE_POLICY))
    binding = cc.hessian_reference_binding(canonical['record']['sha256'], cc.sha256(census))
    triple = {key: canonical['calibration'][key] for key in ('text_sha256', 'fit_ids_sha256', 'fit_tokens')}
    row = dict(triple, supplied=True, capture_sha256=digest, reference_binding=binding)
    return row, dict(row, capture_path=str(path))


@pytest.mark.parametrize('change', [None, 'unit_digest', 'canonical', 'census', 'absent_from_panel',
                                    'stale_row_seal', 'unbound_panel'])
def test_overlay_intake_compares_hessian_content_not_reference_files(tmp_path, campaign, probe, change):
    """The H an overlay row was priced under is its content, not a references file's seal."""
    from prismaquant.joint_catalog_extension import attach_candidate_overlay
    # ``_pair`` adds the overlay format to every roster unit, so the overlay
    # prices all of them; the panel additionally commits one dense unit.
    qnames = priced = sorted(campaign['roster'])
    shared = _canonical_capture(tmp_path/'canonical', [*qnames, _EXTRA_DENSE])
    overlay_units = priced
    panel_units = [*qnames, _EXTRA_DENSE]
    if change == 'absent_from_panel':
        panel_units = [name for name in panel_units if name != priced[0]]
    override = {priced[0]: shared['H'][priced[0]]*7} if change == 'unit_digest' else None
    row, overlay_prov = _workspace_hessian(tmp_path/'overlay', shared, overlay_units, census_copy=True,
                                           override=override)
    panel_capture = shared
    if change == 'canonical':
        # Same census and same H, sealed as a different canonical capture.
        panel_capture = _canonical_capture(tmp_path/'canonical-2', [*qnames, _EXTRA_DENSE],
                                           act_value=2.0, census=shared['census'])
    elif change == 'census':
        panel_capture = _canonical_capture(tmp_path/'canonical-3', [*qnames, _EXTRA_DENSE], max_abs=9.0)
    _, panel_prov = _workspace_hessian(tmp_path/'panel', panel_capture, panel_units)
    if change == 'stale_row_seal':
        row = dict(row, capture_sha256='0'*64)
    elif change == 'unbound_panel':
        panel_prov = {key: value for key, value in panel_prov.items() if key != 'reference_binding'}
    # The shape the loader met: one draw, different reference files and seals.
    assert overlay_prov['capture_path'] != panel_prov['capture_path']
    assert row['capture_sha256'] != panel_prov['capture_sha256']
    case = _overlay_case(tmp_path/'case', campaign, probe, row_hessian=row,
                         panel_hessian=panel_prov, overlay_hessian=overlay_prov)
    data, bound = case.bind()
    if change is None:
        result = attach_candidate_overlay(data, bound, verify_payloads=True)
        assert len(result.cells) == 12
        assert all(ADDED_FORMAT in formats for formats in result.formats_by_qname.values())
        assert result.payload['costs'] == case.scalar_costs
        return
    expected = {'unit_digest': 'unit Hessian', 'canonical': 'canonical capture and census',
                'census': 'canonical capture and census', 'absent_from_panel': 'no Hessian in the panel',
                'stale_row_seal': 'capture seal', 'unbound_panel': 'capture_sha256 differs'}[change]
    with pytest.raises(ValueError, match=expected):
        attach_candidate_overlay(data, bound, verify_payloads=True)


# The sealed GLM Stage A prepared completion ends every roster in BF16, and
# the T4 overlay was assembled by inserting the added format before it.
# Stage B re-checks the prepared roster against the loaded one in order
# (joint_cost_quantum.run_layer_quantum), so the loader, the assembler and
# the pair check must all produce exactly this order (RobTand/prismaquant#990).
_SEALED_ROSTERS = json.loads((Path(__file__).parent / 'fixtures'
                              / 'glm_sealed_prepared_rosters.json').read_text())


def test_sealed_roster_fixture_is_bf16_terminal_and_extends_before_it():
    assert _SEALED_ROSTERS['added_format'] == ADDED_FORMAT
    for name, unit in _SEALED_ROSTERS['units'].items():
        base = unit['base']
        assert base[-1] == 'BF16' and base[:-1] == sorted(base[:-1]), name
        if unit['extended'] is not None:
            assert unit['extended'] == [*base[:-1], ADDED_FORMAT, 'BF16'], name


def _extended_units():
    return {name: unit for name, unit in sorted(_SEALED_ROSTERS['units'].items())
            if unit['extended'] is not None}


def test_overlay_intake_inserts_added_format_before_terminal_bf16(tmp_path, campaign, probe):
    """Loader order on sealed rosters equals the order the overlay was assembled in."""
    from prismaquant.joint_catalog_extension import attach_candidate_overlay
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind()
    sealed = list(_extended_units().values())
    expected = {}
    for index, name in enumerate(case.qnames):
        unit = sealed[index % len(sealed)]
        data.formats_by_qname[name] = tuple(unit['base'])
        expected[name] = tuple(unit['extended'])
    result = attach_candidate_overlay(data, bound, verify_payloads=True)
    assert dict(result.formats_by_qname) == expected


def test_overlay_intake_refuses_a_base_roster_without_terminal_bf16(tmp_path, campaign, probe):
    from prismaquant.joint_catalog_extension import attach_candidate_overlay
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind()
    name = case.qnames[0]
    data.formats_by_qname[name] = tuple(data.formats_by_qname[name])[:-1]
    with pytest.raises(ValueError, match='terminal BF16'):
        attach_candidate_overlay(data, bound, verify_payloads=True)


def test_catalog_pair_refuses_an_extension_appended_after_bf16(tmp_path, campaign, probe):
    """The on-disk proposed-overlay-01 pair appended the format after BF16; refuse it."""
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    verify_catalog_pair(inputs)
    path = Path(inputs['extended_prepared']['path'])
    prepared = json.loads(path.read_bytes())
    prepared['formats_by_qname'] = {name: [*[f for f in formats if f != ADDED_FORMAT], ADDED_FORMAT]
                                    for name, formats in prepared['formats_by_qname'].items()}
    inputs['extended_prepared'] = _write(tmp_path, 'appended-prepared.json', prepared)
    with pytest.raises(ValueError, match='terminal BF16'):
        verify_catalog_pair(inputs)


def test_overlay_assembly_inserts_before_terminal_bf16():
    """tools/assemble_t4_overlay.py writes the order the loader reads."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import assemble_t4_overlay as assemble
    rosters = {name: list(unit['base']) for name, unit in _extended_units().items()}
    for name in rosters:
        assemble.add_overlay_format(rosters, name, ADDED_FORMAT)
    assert rosters == {name: unit['extended'] for name, unit in _extended_units().items()}
