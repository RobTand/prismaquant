"""Explicit compatibility of an original GLM capture with a corrected derivative.

This consumer receipt never rewrites a capture or supplies fake producer runtime
fields. Issuance requires the completed original PB action and native evidence.
"""
from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path

from .glm_source_derivative import (
    VERSION, ORIGINAL_IMAGE_CONTENT_SHA256, ORIGINAL_MODELING_SHA256,
    CORRECTED_MODELING_SHA256, ORIGINAL_EXPRESSION, CORRECTED_EXPRESSION,
    bound_json, _require, source_derivative_identity,
)

SCHEMA = 'prismaquant.glm_capture_derivative_compatibility.v1'
CAPTURE_ACTION = '8740a0b3456bb6cb334ae80b0e35fc3da31c62918abdda8c41ad226020c4e88a'
CAPTURE_REQUEST_SHA256 = '31d64b92bcc26c3288976a5e7f540778e2cc3a95cf9748ca187d45f81155d082'
CAPTURE_SOURCE = 'e22a0286a820b23f2aaaa6a9232912abf49394c5'


def _bytes(binding, label):
    _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, label + ' requires path/SHA256')
    raw = Path(binding['path']).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == binding['sha256'], label + ' bytes changed')
    return raw


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _completion_literal(text):
    """The reviewed producer prints a Python dict repr, with two string fields."""
    _require(len(text) <= 4096, 'capture completion repr exceeds its bound')
    node = ast.parse(text, mode='eval').body
    _require(isinstance(node, ast.Dict) and len(node.keys) == 2 and
             all(isinstance(x, ast.Constant) and type(x.value) is str for x in [*node.keys, *node.values]),
             'capture completion is not the reviewed string dictionary')
    value = {key.value: item.value for key, item in zip(node.keys, node.values)}
    _require(set(value) == {'path', 'sha256'}, 'capture completion fields differ')
    return value


def _verify_cas_receipt(receipt, snapshot, output):
    _require(set(receipt) == {'schema', 'action_key', 'action_manifest_sha256', 'producer', 'result', 'receipt_sha256'} and
             receipt['schema'] == 'prismaquant.prismabuild.cas_receipt.v3', 'canonical v3 CAS receipt required')
    body = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
    _require(_digest(body) == receipt['receipt_sha256'], 'CAS receipt body digest differs')
    producer = receipt['producer']
    _require(producer.get('schema') == 'prismaquant.prismabuild.worker_attestation.v2' and
             producer.get('action_key') == receipt['action_key'] and
             _digest({key: value for key, value in producer.items() if key != 'attestation_sha256'}) == producer.get('attestation_sha256'),
             'CAS producer attestation digest differs')
    _require(snapshot['input'] in producer.get('inputs', []), 'CAS producer does not bind the reviewed source snapshot input')
    _require(receipt['result'] == dict(sha256=hashlib.sha256(output).hexdigest(), bytes=len(output)),
             'original CAS output is not bound by its receipt')


def _producer(evidence, capture):
    from tools.container_runtime_identity import image_content_sha256
    _require(isinstance(evidence, dict) and set(evidence) ==
             {'request', 'terminal', 'receipt', 'output', 'image_inspection', 'modeling_source'}, 'closed original producer evidence required')
    _require(evidence['request']['sha256'] == CAPTURE_REQUEST_SHA256, 'original capture request is not the reviewed action')
    request = bound_json(evidence['request'], 'original capture request')
    terminal = bound_json(evidence['terminal'], 'original capture terminal')
    receipt = bound_json(evidence['receipt'], 'original capture CAS receipt')
    output = _bytes(evidence['output'], 'original capture CAS output')
    _require(request.get('action_key') == terminal.get('action_key') == receipt.get('action_key') == CAPTURE_ACTION,
             'original producer action differs')
    _require(terminal.get('schema') == 'prismaquant.prismabuild.pool_outcome.v1' and terminal.get('status') == 'executed' and
             type(terminal.get('detail', {}).get('returncode')) is int and terminal['detail']['returncode'] == 0 and
             terminal.get('resource_scope_cleanup', {}).get('complete') is True, 'original capture has not completed successfully')
    snapshot = request['params']['checkout_snapshot']
    _require(snapshot['parent'] == CAPTURE_SOURCE and terminal.get('checkout_snapshot') == snapshot,
             'original producer source snapshot differs')
    _verify_cas_receipt(receipt, snapshot, output)
    command = request['params']['command']
    _require(command[:3] == ['python3', '-m', 'tools.tessera_campaign_container'], 'original producer wrapper differs')
    spec = json.loads(command[command.index('--spec')+1])
    _require(spec['container'].get('content_sha256') == ORIGINAL_IMAGE_CONTENT_SHA256, 'original requested image differs')
    containers, completions = [], []
    prefix = '[campaign] complete streamed calibration capture: '
    for line in output.decode().splitlines():
        if line.startswith(prefix):
            completions.append(_completion_literal(line[len(prefix):]))
        elif line.startswith('{'):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('schema') == 'prismaquant.tessera_campaign_container.v1':
                containers.append(row)
    _require(len(containers) == 1 and containers[0].get('image_content_sha256') == ORIGINAL_IMAGE_CONTENT_SHA256,
             'original actual container content is absent or differs')
    _require(completions == [capture], 'complete original capture output does not bind this manifest')
    inspected = bound_json(evidence['image_inspection'], 'original image inspection')
    images = list(inspected.values()) if isinstance(inspected, dict) else []
    matched = [row for row in images if isinstance(row, dict) and row.get('Id') == containers[0].get('image_id')]
    _require(len(matched) == 1 and image_content_sha256(matched[0]) == ORIGINAL_IMAGE_CONTENT_SHA256,
             'original image inspection does not bind actual image/config/layers')
    _require(hashlib.sha256(_bytes(evidence['modeling_source'], 'original modeling source')).hexdigest() == ORIGINAL_MODELING_SHA256,
             'original modeling source differs')
    return dict(action_key=CAPTURE_ACTION, source_snapshot=snapshot,
                image_content_sha256=ORIGINAL_IMAGE_CONTENT_SHA256, modeling_sha256=ORIGINAL_MODELING_SHA256)


def _forward_and_graph(evidence, derivative):
    _require(isinstance(evidence, dict) and set(evidence) ==
             {'cpu_reproduction', 'original_layer0', 'corrected_layer0', 'corrected_graph'}, 'closed forward/graph evidence required')
    cpu = bound_json(evidence['cpu_reproduction'], 'CPU causal transform evidence')
    _require(cpu.get('schema') == 'prismaquant.glm_kda_source_cpu_reproduction.v1' and
             cpu.get('source', {}).get('modeling_source_sha256') == ORIGINAL_MODELING_SHA256,
             'CPU source evidence differs')
    cases = cpu.get('cases', [])
    expected_cases = [(64, -5., 'torch.float32', False, False), (65, -5., 'torch.float32', True, True),
                      (64, -.1, 'torch.float32', True, False), (65, -.1, 'torch.float32', False, True),
                      (64, -5., 'torch.bfloat16', True, False), (65, -5., 'torch.bfloat16', True, True)]
    _require(cpu.get('proposed_expression') == dict(before=ORIGINAL_EXPRESSION, after=CORRECTED_EXPRESSION) and
             [tuple(row.get(key) for key in ('length', 'gate', 'dtype', 'use_norm', 'initial_state')) for row in cases] == expected_cases,
             'CPU transform or exact case roster differs')
    _require(len(cases) == 6 and all(row.get('forward_byte_equal') is True and
             (not row.get('initial_state') or row.get('final_state_byte_equal') is True) for row in cases),
             'core and final-state forward byte equivalence is incomplete')
    original = bound_json(evidence['original_layer0'], 'original native layer0')
    corrected = bound_json(evidence['corrected_layer0'], 'corrected native layer0')
    graph = bound_json(evidence['corrected_graph'], 'corrected native graph')
    _require(all(result.get('schema') == 'prismaquant.glm_original_graph_qualification.v1'
                 for result in (original, corrected, graph)), 'native graph schema differs')
    _require(original.get('mode') == corrected.get('mode') == 'layer0_diagnostic_not_qualification',
             'native forward comparison must use the actual layer0 diagnostic')
    _require(corrected.get('status') == 'diagnostic_complete_not_qualification' and
             corrected.get('source_derivative') == derivative, 'corrected native diagnostic is incomplete or mismatched')
    before, after = original.get('primary_outputs', []), corrected.get('primary_outputs', [])
    _require(len(before) == len(after) == 1 and before[0]['layer'] == after[0]['layer'] == 0 and
             before[0]['original_row'] == after[0]['original_row'] == 0 and
             before[0]['statistics']['nonfinite'] == after[0]['statistics']['nonfinite'] == 0 and
             before[0]['identity'] == after[0]['identity'], 'actual original/corrected native forward bytes differ')
    _require(graph.get('status') == 'complete' and graph.get('mode') == 'bounded_prefix_qualification' and
             graph.get('source_derivative') == derivative and len(graph.get('backwards', [])) == 72 and
             graph.get('source_owners_expired') is True, 'corrected native graph qualification is incomplete')
    for result, diagnostic in [(corrected, True), (graph, False)]:
        _require(result.get('runtime_modeling_sha256') == CORRECTED_MODELING_SHA256 and
                 result.get('runtime_image_content_sha256') == derivative['image_content_sha256'] and
                 result.get('source_execution', {}).get('source_derivative') == derivative and
                 not result.get('cleanup_errors') and not result.get('telemetry_errors'),
                 'corrected native execution or cleanup differs')
        _verify_native_schedule(result, diagnostic=diagnostic)
    for name, result in [('original', original), ('corrected', corrected), ('graph', graph)]:
        source = result.get('source_final', {})
        _require(source.get('descriptors_open') == 0 and not source.get('violations'), name + ' source lifetime is unverified')
        authenticated = source.get('authenticated', [])
        _require(authenticated and all(row.get('actual_sha256') == row.get('expected_sha256') for row in authenticated),
                 name + ' source payload identity is unverified')
    # The closed source transformation supplies the causal-domain equivalence
    # argument. Native equality above is a layer0 check, not a full-model A/B.
    return dict(scope='closed_causal_expression_proof_plus_native_layer0_and_bounded_graph',
                corrected_modeling_sha256=CORRECTED_MODELING_SHA256)


def _verify_native_schedule(result, *, diagnostic):
    """Recheck measured rows, numerical identities and gradients at consumption."""
    layers, rows, seeds = ((0,), (0,), (7000,)) if diagnostic else ((0, 3, 4), (0, 511), (7000, 7001, 7002, 7003))
    arms = ('unobserved_isolated_baseline',) if diagnostic else (
        'unobserved_isolated_baseline', 'nonfinal_fork_replay', 'final_original_owner_replay')
    expected = [(layer, row, seed, arm) for layer in layers for row in rows for seed in seeds for arm in arms]
    keys = ('layer', 'original_row', 'seed', 'arm')
    backwards, diagnostics = result.get('backwards', []), result.get('replay_diagnostics', [])
    _require(all([tuple(row.get(key) for key in keys) for row in sequence] == expected
                 for sequence in (backwards, diagnostics)), 'native backward/diagnostic schedule differs')
    primary = result.get('primary_outputs', [])
    _require([(row.get('layer'), row.get('original_row')) for row in primary] ==
             [(layer, row) for layer in layers for row in rows] and
             all(row.get('statistics', {}).get('nonfinite') == 0 for row in primary), 'native primary schedule or finiteness differs')
    outputs = {(row['layer'], row['original_row']): row['identity'] for row in primary}
    baselines = {}
    for row, observed in zip(backwards, diagnostics):
        group = tuple(row[key] for key in keys[:3])
        output = outputs[group[:2]]
        gradient = observed.get('leaf_gradient', {})
        _require(row.get('output') == output == observed.get('output_identity') and
                 observed.get('output_matches_primary') is True and observed.get('backward_completed') is True and
                 observed.get('output', {}).get('nonfinite') == 0 and gradient.get('nonfinite') == 0 and
                 type(gradient.get('finite_nonzero')) is int and gradient['finite_nonzero'] > 0,
                 'native forward or finite nonzero backward evidence differs')
        if row['arm'] == arms[0]:
            baselines[group] = row
        else:
            baseline = baselines[group]
            _require(row.get('cotangent') == baseline.get('cotangent') and row.get('stimulus') == baseline.get('stimulus'),
                     'native replay cotangent or stimulus differs from baseline')
            _require(row.get('activity') is not None, 'native replay activity evidence is absent')
    if not diagnostic:
        _require(all(backwards[index+1]['activity'] == backwards[index+2]['activity']
                     for index in range(0, len(backwards), 3)), 'native fork/final route activity differs')


def _verify(record, *, capture, derivative):
    from .tessera_calibration_cache import require_capture_contract
    _require(isinstance(record, dict) and set(record) ==
             {'schema', 'version', 'capture', 'derivative_identity_sha256', 'producer', 'forward_equivalence'},
             'closed compatibility receipt required')
    _require(record['schema'] == SCHEMA and record['version'] == VERSION and record['capture'] == capture and
             record['derivative_identity_sha256'] == _digest(derivative), 'compatibility receipt identity differs')
    manifest = require_capture_contract(capture['path'], expected_sha256=capture['sha256'])
    producer = _producer(record['producer'], capture)
    proof = _forward_and_graph(record['forward_equivalence'], derivative)
    return dict(capture_identity=manifest['identity'], producer=producer, proof=proof)


def require_capture_compatibility(binding, *, capture, model):
    derivative = source_derivative_identity(model)
    if derivative is None:
        _require(binding is None, 'compatibility receipt requires an explicitly corrected consumer')
        return None
    _require(binding is not None, 'corrected consumer requires a completed original-capture compatibility receipt')
    return _verify(bound_json(binding, 'capture compatibility receipt'), capture=capture, derivative=derivative)


def create_capture_compatibility(*, capture, producer, forward_equivalence, model, output):
    derivative = source_derivative_identity(model)
    _require(derivative is not None, 'compatibility issuance requires an observed corrected runtime')
    record = dict(schema=SCHEMA, version=VERSION, capture=capture, derivative_identity_sha256=_digest(derivative),
                  producer=producer, forward_equivalence=forward_equivalence)
    _verify(record, capture=capture, derivative=derivative)
    raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with Path(output).open('xb') as stream:
        stream.write(raw)
    return dict(path=str(output), sha256=hashlib.sha256(raw).hexdigest())
