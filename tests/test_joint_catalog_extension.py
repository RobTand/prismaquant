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
from prismaquant.joint_layer_quanta import bind_adjoint_receipt, layer_quanta
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.tessera_joint_aura import PREPARED_SCHEMA
from tests.test_joint_quanta_join import campaign, probe


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


def _pair(tmp_path, campaign, probe):
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
    receipt = {'schema': 'prismaquant.joint_adjoint_capture.v1', 'status': 'complete',
        'run_identity': {'plan_sha256': inputs['original_plan']['sha256'],
            'prepared_sha256': inputs['original_prepared']['sha256'], 'campaign_scope': campaign['scope'],
            'calibration_sha256': common['calibration_input']['calibration_sha256'],
            'calibration_shape': common['calibration_input']['shape'],
            'n_probes': probe['n_probes'], 'seed_base': probe['seed_base'],
            'unit_roster_sha256': hashlib.sha256(''.join(n+'\n' for n in sorted(qnames)).encode()).hexdigest()},
        'checkpoints': [{'boundary': 3}]}
    capture = _write(tmp_path, 'capture.json', receipt)
    return inputs, receipt, capture


def test_additive_catalog_binds_original_capture_without_relabelling(tmp_path, campaign, probe):
    inputs, receipt, capture = _pair(tmp_path, campaign, probe)
    original = Path(capture['path']).read_bytes()
    bound = create_extension(inputs=inputs, adjoint_capture=capture, output=tmp_path/'extension.json')
    args = dict(plan_sha256=inputs['extended_plan']['sha256'],
                prepared_sha256=inputs['extended_prepared']['sha256'],
                scope=campaign['scope'], checkpoints=[3])
    with pytest.raises(ValueError, match='another plan_sha256'):
        bind_adjoint_receipt(receipt, **args)
    result = bind_adjoint_receipt(receipt, **args, catalog_extension=bound)
    assert result == canonical_json_sha256(receipt, where="synthetic capture")
    assert Path(capture['path']).read_bytes() == original
    from tools.dispatch_joint_quanta import check_adjoint_receipt, DispatchRefused
    dispatch_record = {'campaign': {'plan_sha256': inputs['extended_plan']['sha256'],
        'prepared_sha256': inputs['extended_prepared']['sha256']},
        'adjoint': {'receipt_sha256': result}, 'catalog_extension': bound}
    assert check_adjoint_receipt(Path(capture['path']), [(tmp_path/'q.json', dispatch_record)]) == receipt
    unbound = {**dispatch_record, 'catalog_extension': None}
    with pytest.raises(DispatchRefused, match='extension bindings differ'):
        check_adjoint_receipt(Path(capture['path']), [(tmp_path/'q.json', dispatch_record), (tmp_path/'r.json', unbound)])
    plan = json.loads(Path(inputs['extended_plan']['path']).read_bytes())
    prepared = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    produced = layer_quanta(plan, prepared, campaign['parent_manifest'],
        parent_manifest_sha256=campaign['manifest_sha256'],
        plan_path=inputs['extended_plan']['path'], prepared_path=inputs['extended_prepared']['path'],
        plan_sha256=args['plan_sha256'], prepared_sha256=args['prepared_sha256'],
        adjoint_receipt=receipt, catalog_extension=bound, stride=8)
    assert len(produced['records']) == 3
    for record in produced['records']:
        assert record['adjoint']['receipt_sha256'] == result
        assert record['campaign']['prepared_sha256'] == args['prepared_sha256']
        assert record['catalog_extension'] == bound
    from prismaquant.joint_cost_quantum import verify_quantum_identity
    first = produced['records'][0]
    quantum = _write(tmp_path, 'quantum.json', first)
    loaded, original_receipt = verify_quantum_identity(
        quantum_path=Path(quantum['path']), quantum_sha256=quantum['sha256'],
        plan_path=Path(inputs['extended_plan']['path']), plan_sha256=args['plan_sha256'],
        prepared_path=Path(inputs['extended_prepared']['path']), prepared_sha256=args['prepared_sha256'],
        adjoint_path=Path(capture['path']), adjoint_sha256=capture['sha256'],
        output_root=Path(plan['output_root']))
    assert loaded == first and original_receipt == receipt
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
    with pytest.raises(ValueError, match='extended plan binding'):
        require_extension(bound, receipt=receipt, plan_sha256='0'*64,
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


@pytest.mark.parametrize('change', [None, 'hessian', 'scalar', 'missing', 'render_changed'])
def test_overlay_intake_preserves_base_and_consumes_only_bound_measured_additions(tmp_path, campaign, probe, change):
    from types import SimpleNamespace
    from prismaquant.joint_catalog_extension import attach_candidate_overlay
    inputs, _, _ = _pair(tmp_path, campaign, probe)
    original = json.loads(Path(inputs['original_prepared']['path']).read_bytes())
    extended = json.loads(Path(inputs['extended_prepared']['path']).read_bytes())
    cache = pickle.loads(Path(extended['production_cache']['path']).read_bytes())
    base_cells, rows, scalar_costs = {}, [], {}
    hessian = {'supplied': True, 'capture_sha256': 'f'*64, 'text_sha256': 'e'*64,
               'fit_ids_sha256': 'd'*64, 'fit_tokens': 4}
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
            'activation_quantized': True, 'hessian_identity': hessian}}
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
    if change == 'scalar': scalar_costs[rows[0]['qname']][ADDED_FORMAT]['output_mse'] = 2.0
    elif change == 'hessian':
        rows[0]['record']['identity']['calibration']['hessian_sha256'] = '0'*64
    elif change == 'missing': rows.pop()
    elif change == 'render_changed': Path(rows[0]['render']).write_bytes(b'drifted')
    catalog = {'schema': 'prismaquant.t4_adopted_catalog.v1', 'format': ADDED_FORMAT,
        'old_prepared': inputs['original_prepared'], 'old_pwc': original['production_cache'],
        'cost': _write(tmp_path, 'scalar.pkl', {'costs': scalar_costs}, binary=True),
        'reseal_proof': rows[0]['catalog_source_adoption']['encoder_source_proof'], 'cells': rows,
        **{key: original[key] for key in ('source_model_identity', 'source_execution', 'calibration_input',
                                         'reader_identity', 'projection_backend')}}
    bound = _write(tmp_path, 'catalog.json', catalog)
    data = SimpleNamespace(inputs={'fixture': 'old', 'candidate_overlay': bound},
        cells=copy.deepcopy(base_cells), formats_by_qname=copy.deepcopy(original['formats_by_qname']),
        payload={'provenance': {'hessian': hessian},
                 'costs': {name: {} for name in original['formats_by_qname']}})
    if change is not None:
        with pytest.raises(ValueError):
            attach_candidate_overlay(data, bound, verify_payloads=True)
    else:
        result = attach_candidate_overlay(data, bound, verify_payloads=True)
        assert {pair: result.cells[pair] for pair in base_cells} == base_cells
        assert len(result.cells) == 12
        assert all(ADDED_FORMAT in row for row in result.formats_by_qname.values())
        assert result.payload['costs'] == scalar_costs
