"""Bind completed joint prices to their original Tessera export metadata.

This is an opt-in metadata handoff, not a probe, renderer or export gate. The
original joint table stays immutable. Current wire bytes are checked by the
existing exporter; this handoff authenticates the recorded preparation that
proved which original wire produced each tensor measured by joint AURA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle

from .cluster_campaign import _atomic_write_new_bytes as atomic_write_bytes
from .cost_stage_checkpoint import canonical_json_sha256
from .joint_aura import (
    prepare_joint_aura_identities, release_joint_aura_identities, validated_identity_memo,
)
from .tessera_joint_aura import (
    HISTORICAL_WIRE_VALIDATION, PREPARED_SCHEMA, RENDER_COMPARISON_BY_ORIGIN, SCHEMA, _require, _same,
    cell_render_census, render_origin_census,
)

HANDOFF_SCHEMA = 'prismaquant.tessera_joint_allocation.v1'
ROW_FIELDS = ('hessian_identity', 'tessera_family', 'tessera_body_rate_q256',
              'wire_bytes', 'input_global_scale', 'activation_contract', 'activation_quantized')
PROVENANCE_FIELDS = ('model', 'nsamples', 'seqlen', 'layer_stride', 'max_act_rows',
                     'hessian', 'activation_static_scales', 'wire_dir', 'calibration_cache',
                     'population', 'tessera_expert_projection', 'tessera_serving_scope')


def _add(target, key, value, where):
    if key in target:
        _same(target[key], value, f'{where}: existing {key}')
    else:
        target[key] = copy.deepcopy(value)


def bind_allocation_payload(joint, data, prepared, cache_metadata, *, plan_sha256, prepared_binding,
                            scope='ordinary_handoff'):
    """Join already-authenticated artifacts without replacing any joint field."""
    from .cost_currency import require_run_currency, require_sampled_joint_run_currency
    from .schemas import validate_cost_payload, validate_probe_payload
    from . import tessera_expert_projection as tep

    _require(scope in ('ordinary_handoff', 'sampled_joint_panel'), 'unknown joint binding scope')
    if scope == 'ordinary_handoff':
        _require('joint_eval' not in joint.get('provenance', {}).get('tessera_joint_anchors', {})
                 and 'joint_eval' not in joint.get('provenance', {}),
                 'diagnostic joint evaluation requires a separate sampled-proposal path or validated promotion; ordinary allocation/export remains closed')
    else:
        _require('joint_eval' in joint.get('provenance', {}),
                 'sampled research binding requires a diagnostic joint panel')

    validate_cost_payload(joint)
    validate_probe_payload(joint)
    currency_gate = (require_run_currency if scope == 'ordinary_handoff'
                     else require_sampled_joint_run_currency)
    currency = currency_gate(joint)
    _require(currency.get('joint_aura_rows', 0) > 0, 'a complete joint table is required')
    evidence = joint['provenance'].get('tessera_joint_anchors', {})
    _same(evidence.get('plan_sha256'), plan_sha256, 'joint plan')
    _same(evidence.get('prepared'), prepared_binding, 'joint prepared binding')
    _same(evidence.get('inputs'), data.inputs, 'joint original anchor inputs')
    if scope == 'sampled_joint_panel':
        _same(evidence.get('joint_eval'), joint['provenance']['joint_eval'], 'joint pilot identity')
    _same(prepared.get('schema'), PREPARED_SCHEMA, 'prepared schema')
    _same(prepared.get('status'), 'complete', 'prepared completion')
    _same(prepared.get('plan_sha256'), plan_sha256, 'prepared plan')
    _same(prepared.get('calibration_input'), evidence.get('calibration_input'), 'prepared calibration')
    roster = {name: list(formats) for name, formats in data.formats_by_qname.items()}
    _same(prepared.get('formats_by_qname'), roster, 'prepared candidate roster')
    _same(set(joint['costs']), set(roster), 'joint unit roster')
    _same(set(joint['stats']), set(roster), 'joint statistic roster')
    for item in (prepared, evidence):
        _same(item.get('measured_cells'), len(data.cells), 'measured anchor count')
    _same(cache_metadata.get('schema'), PREPARED_SCHEMA, 'prepared cache schema')
    _same(cache_metadata.get('inputs'), data.inputs, 'prepared cache inputs')
    for key in ('reader_identity', 'projection_backend'):
        _same(cache_metadata.get(key), prepared.get(key), f'prepared cache {key}')
    verified = cache_metadata.get('verified_cells', {})
    _same(set(verified), set(data.cells), 'prepared tensor receipt roster')
    # Carried, not recomputed from file state: which rungs had an independent
    # render/wire comparison and which had only the wire round-trip is a fact
    # about the qualification that ran, and it travels with this table.
    render_census = cell_render_census(data.cells)
    _same(render_origin_census(receipt['render_origin'] for receipt in verified.values()),
          render_census, 'prepared render origin census')
    for key, value in render_census.items():
        _same(cache_metadata.get(key), value, f'prepared cache {key}')
        _same(prepared.get(key), value, f'prepared {key}')
        _same(evidence.get(key), value, f'joint {key}')
    original = data.payload['provenance']
    calibration = prepared['calibration_input']
    pilot = joint['provenance']['joint_eval'] if scope == 'sampled_joint_panel' else None
    if pilot is not None:
        from .tessera_joint_eval_panel import validate_panel_descriptor
        validate_panel_descriptor(pilot, n_samples=calibration['shape'][0],
                                  seqlen=calibration['shape'][1],
                                  artifact_sha256=calibration['artifact_sha256'])
        _same(pilot['shape'][0], pilot['selection']['size'], 'pilot selected windows')
    original_draw = original['hessian']['calibration_identity']
    _same(calibration['provenance'], {key: original_draw.get(key) for key in calibration['provenance']},
          'anchor calibration draw')
    source_by_unit = {}
    # One probe identity object serves every row of a quantum (pickle keeps
    # the shared reference), and on GLM-5.3 its source model is 8.9 MB of
    # weight map. Compare its fields once per object, not once per row
    # (PQ #1256). Keeping each probe as the value means no keyed id can be
    # reused by another object while this loop runs.
    checked_probes = {}
    result = copy.deepcopy(joint, validated_identity_memo(joint))
    policy_binding = prepared.get('served_activation_policy')
    _same(joint['provenance'].get('stage_b_resource_policy'), prepared.get('stage_b_resource_policy'),
          'joint Stage B resource policy')
    _same(joint['provenance'].get('served_activation_policy'), policy_binding, 'joint served activation policy')
    policy = None
    if policy_binding is not None:
        from .joint_served_activation import verify_policy, require_priced_activation
        from .joint_catalog_extension import extension_run_header, require_extension
        extension = joint['provenance'].get('catalog_extension')
        _require(isinstance(extension, dict), 'served policy requires an authenticated catalog extension')
        # The full extension independently binds the old capture and new plan.
        extension_doc = json.loads(_read_bound(extension, 'served policy catalog extension'))
        _same(extension_doc['inputs']['extended_prepared'], prepared_binding, 'served policy extended preparation')
        _same(extension_doc['inputs']['extended_plan']['sha256'], plan_sha256, 'served policy extended plan')
        require_extension(extension, run_header=extension_run_header(extension),
                          plan_sha256=plan_sha256, prepared_sha256=prepared_binding['sha256'])
        policy = verify_policy(policy_binding, original_prepared=extension_doc['inputs']['original_prepared'])
    for name, formats in roster.items():
        _same(set(joint['costs'][name]), set(formats), f'{name}: measured candidate roster')
        shape = data.census['unit_shapes'][name]
        stats = joint['stats'][name]
        _same([stats.get('out_features'), stats.get('in_features')], shape, f'{name}: statistics shape')
        _same(stats['n_params'], math.prod(shape), f'{name}: parameter count')
        for fmt in formats:
            row = joint['costs'][name][fmt]
            operator, probe = row['joint_operator_identity'], row['probe_identity']
            _same(operator['arithmetic'].get('served_activation_policy'), policy_binding,
                  f'{name}@{fmt}: priced served policy arithmetic')
            _same(operator['arithmetic'].get('stage_b_resource_policy'), prepared.get('stage_b_resource_policy'),
                  f'{name}@{fmt}: priced resource policy arithmetic')
            if id(probe) not in checked_probes:
                source_model = probe['source_model']
                _same(source_model, prepared['source_model_identity'], f'{name}: source model')
                _same(source_model['source'], original['model'], f'{name}: source model path')
                _same(probe['calibration_sha256'],
                      calibration['calibration_sha256'] if pilot is None else pilot['eval_ids_sha256'],
                      f'{name}: probe calibration')
                _same(probe.get('calibration_shape'),
                      calibration['shape'] if pilot is None else pilot['shape'],
                      f'{name}: calibration shape')
                checked_probes[id(probe)] = probe
            _same(operator['arithmetic']['projection_backend'], prepared['projection_backend'], f'{name}: projection backend')
            _same(operator['source_weight']['shape'], shape, f'{name}: source shape')
            source = operator['source_weight']
            _same(source, source_by_unit.setdefault(name, source), f'{name}: source changed across formats')
            if fmt == 'BF16':
                _same(operator['rendered_weight'], source, f'{name}: BF16 source passthrough')
                _require(not operator['activation']['quantizes_input'] and
                         all(value == 0 for value in row['x2_per_probe']), f'{name}: BF16 is not measured zero')
                continue
            pair = (name, fmt)
            receipt = verified[pair]
            for key in ('source_weight', 'rendered_weight'):
                _same(operator[key], receipt[key], f'{name}@{fmt}: prepared {key}')
            priced_scale = None
            if policy is None:
                _same(operator['activation'], receipt['activation'], f'{name}@{fmt}: prepared activation')
                _require(operator.get('served_activation_policy') is None, 'unbound served activation override')
            else:
                priced_scale = require_priced_activation(policy_binding, policy, name, fmt, receipt['activation'], operator)
            _same(receipt['render_comparison'],
                  RENDER_COMPARISON_BY_ORIGIN[data.cells[pair]['render_origin']],
                  f'{name}@{fmt}: prepared render comparison')
            record = data.cells[pair]['record']
            _same(receipt['wire_sha256'], record['blob_sha256'], f'{name}@{fmt}: original wire')
            source_record = data.manifest['identity']['units'][name]['weight']
            # Tessera hashes dtype/shape plus values; PWC hashes raw values.
            # Preparation verified the original encoding identity against the
            # actual source tensor and retained BOTH owner-defined receipts.
            # Join through that authenticated encoding receipt, never compare
            # hashes from different grammars or manufacture a conversion.
            _same(receipt['encoding_identity_sha256'],
                  canonical_json_sha256(record['identity'], where='original encoding'),
                  f'{name}@{fmt}: prepared encoding identity')
            _same(record['identity']['source'], source_record, f'{name}: original source identity')
            _same(source['shape'], source_record['shape'], f'{name}: original source dimensions')
            _same(source['dtype'].removeprefix('torch.'), source_record['dtype'].removeprefix('torch.'), f'{name}: original source dtype')
            anchor_row = data.payload['costs'][name][fmt]
            _same(receipt['activation']['input_global_scale'], anchor_row.get('input_global_scale'), f'{name}@{fmt}: qualified original static scale')
            if priced_scale is None:
                _same(operator['activation']['input_global_scale'], anchor_row.get('input_global_scale'), f'{name}@{fmt}: original static scale')
            for key in ROW_FIELDS:
                if key in anchor_row:
                    value = priced_scale if key == 'input_global_scale' and priced_scale is not None else anchor_row[key]
                    _add(result['costs'][name][fmt], key, value, f'{name}@{fmt}')
    for key in PROVENANCE_FIELDS:
        if key in original:
            _add(result['provenance'], key, original[key], 'joint provenance')
    if tep.PROJECTION_KEY in original:
        _source, units, _stacks = tep.carried_units(original[tep.PROJECTION_KEY])
        _require(set(units) <= set(roster), 'original projection exceeds joint roster')
        original_wires = data.payload.get(tep.EXPERT_WIRES_KEY, {})
        wires = {}
        for name in units:
            wires[name] = {}
            for fmt in roster[name]:
                if fmt == 'BF16':
                    continue
                record = data.cells[name, fmt]['record']
                _same(original_wires.get(name, {}).get(fmt), record, f'{name}@{fmt}: projected wire receipt')
                wires[name][fmt] = copy.deepcopy(record)
        _add(result, tep.EXPERT_WIRES_KEY, wires, 'joint payload')
        # The existing population/projection validator checks coverage without
        # choosing a quantization: BF16 is only an inert receipt-validation arm.
        tep.allocation_expert_projection_block(result, {name: 'BF16' for name in roster})
    _add(result['provenance'], 'tessera_joint_allocation', {
        'schema': HANDOFF_SCHEMA,
        'status': ('research_metadata_handoff' if scope == 'ordinary_handoff'
                   else 'research_sampled_joint_panel'),
        **({'export_authority': False} if scope == 'sampled_joint_panel' else {}),
        'plan_sha256': plan_sha256, 'prepared': prepared_binding,
        'units': len(roster), 'measured_cells': len(data.cells),
        'cost_fields': 'all_original_joint_fields_unchanged',
        'wire_validation': HISTORICAL_WIRE_VALIDATION,
        **render_census,
    }, 'joint provenance')
    _same(currency_gate(result), currency, 'unchanged joint currency')
    release_joint_aura_identities(result)
    return result


def _bound_stat_fence(path):
    """The stat identity a memoized bound read trusts a hit on (P3 #682)."""
    value = path.stat()
    return (value.st_mode, value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


#: Process-scoped bound bytes by ``(path, sha256)`` with the stat fence the
#: bytes were verified under. The digest check is what authenticates the
#: bytes; the fence only admits reusing them without re-reading and
#: re-hashing. A fence drift re-reads and re-verifies, and a digest mismatch
#: still refuses -- a memo hit never authenticates anything.
_BOUND_BYTES = {}


#: A process-wide reader for :func:`_read_bound`'s bytes, ``(path, sha256,
#: label) -> bytes``. None reads the path. The Stage B preparation installs
#: its strict staged reader here (``stage_b_prep_io.bind_staged_reads``,
#: PQ #1092), so the control documents it reads come off the stage.
BOUND_READER = None


def _read_bound(record, label):
    _require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, f'{label}: bound path and SHA256 required')
    path = Path(record['path'])
    key = (str(path), record['sha256'])
    try:
        fence = _bound_stat_fence(path)
    except OSError:
        fence = None
    if fence is not None:
        hit = _BOUND_BYTES.get(key)
        if hit is not None and hit[0] == fence:
            return hit[1]
    raw = (path.read_bytes() if BOUND_READER is None
           else BOUND_READER(path, record['sha256'], label))
    _same(hashlib.sha256(raw).hexdigest(), record['sha256'], f'{label}: owned bytes')
    if fence is not None and len(raw) == fence[3]:
        _BOUND_BYTES[key] = (fence, raw)
    return raw


def handoff(*, joint_binding, plan_binding, output_path):
    """Authenticate inputs and publish a new table, then its success receipt.

    Both publications refuse concurrent destinations atomically. If receipt
    publication fails, the incomplete own table remains for diagnosis; no
    success receipt is returned and no competing bytes are replaced.
    """
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_aura import load_measured_anchor_input

    joint = pickle.loads(_read_bound(joint_binding, 'joint cost'))
    plan = json.loads(_read_bound(plan_binding, 'joint plan'))
    _same(plan.get('schema'), SCHEMA, 'joint plan schema')
    _require('joint_eval' not in plan,
             'diagnostic joint evaluation requires a separate sampled-proposal path or validated promotion; ordinary allocation/export remains closed')
    if joint.get('provenance', {}).get('join_schema') is not None:
        joint = bind_joined_anchors(joint, plan, plan_binding=plan_binding)
    evidence = joint['provenance']['tessera_joint_anchors']
    _same(evidence['plan_sha256'], plan_binding['sha256'], 'joint plan binding')
    _same(evidence['inputs'], plan['inputs'], 'joint plan original inputs')
    prepared_binding = evidence['prepared']
    prepared = json.loads(_read_bound(prepared_binding, 'prepared completion'))
    _same(prepared.get('served_activation_policy'), plan.get('served_activation_policy'), 'planned served activation policy')
    _same(prepared.get('stage_b_resource_policy'), plan.get('stage_b_resource_policy'), 'planned Stage B resource policy')
    _same(prepared['calibration_input']['artifact_sha256'], plan['calibration_input']['sha256'], 'planned calibration artifact')
    cache = pickle.loads(_read_bound(prepared['production_cache'], 'prepared cache'))
    _require(isinstance(cache, ProductionWeightCache), 'prepared cache owner is not ProductionWeightCache')
    # This joins historical identities; it does not re-read all decoded model
    # tensors or weaken the exporter's current-byte verification. The plan's
    # named historical encoder seals travel with its inputs: the anchor
    # checkpoint was priced by the package the plan names, not the installed one.
    data = load_measured_anchor_input(plan['inputs'], verify_payloads=False,
                                      historical_encoder_reuse=plan.get('historical_encoder_reuse'))
    _same(cache.weights, {pair: cell['render'] for pair, cell in data.cells.items()}, 'prepared render paths')
    # Every row's currency check re-validates its probe identity; validate and
    # hash each shared identity once instead (PQ #1256), as the allocator does.
    prepare_joint_aura_identities(joint)
    result = bind_allocation_payload(joint, data, prepared, cache.metadata,
        plan_sha256=plan_binding['sha256'], prepared_binding=prepared_binding)
    result['provenance']['tessera_joint_allocation']['original_joint_cost'] = dict(joint_binding)
    output = Path(output_path)
    _require(not output.exists() and output.resolve() != Path(joint_binding['path']).resolve(), 'handoff output must be new')
    receipt_path = output.with_suffix(output.suffix + '.receipt.json')
    _require(not receipt_path.exists(), 'handoff receipt must be new')
    raw = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {'schema': HANDOFF_SCHEMA, 'original_joint_cost': joint_binding, 'plan': plan_binding,
               'prepared': prepared_binding, 'prepared_cache': prepared['production_cache'],
               'output': {'path': str(output.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
               'units': len(result['costs']), 'joint_fields_unchanged': True, 'research_only': True}
    atomic_write_bytes(output, raw)
    atomic_write_bytes(receipt_path,
                       (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    return receipt


def bind_joined_anchors(joint, plan, *, plan_binding):
    """Restore the ordinary handoff from bound preparation, never row guesses.

    The distributed join preserves measured statistics and every quantum's
    provenance. Its shared anchor metadata is the same plan/preparation pair
    the monolithic producer carries; read that pair by digest and let the
    existing handoff validate every source, render, calibration and wire.
    """
    from .joint_quanta_join import JOINED_RESULTS_SCHEMA
    provenance = joint['provenance']
    _same(provenance.get('join_schema'), JOINED_RESULTS_SCHEMA, 'joint join schema')
    coverage = provenance.get('coverage', {})
    _require(coverage.get('status') == 'complete' and not coverage.get('gaps'),
             'joint join is gapped; complete all quanta before allocation handoff')
    _same(provenance.get('plan_sha256'), plan_binding['sha256'], 'joined plan binding')
    prepared_binding = provenance.get('prepared')
    prepared = json.loads(_read_bound(prepared_binding, 'joined prepared completion'))
    _same(prepared_binding['sha256'], provenance.get('prepared_sha256'), 'joined preparation binding')
    _same(prepared.get('plan_sha256'), plan_binding['sha256'], 'joined prepared plan')
    _same(prepared.get('schema'), PREPARED_SCHEMA, 'joined prepared schema')
    _same(prepared.get('status'), 'complete', 'joined prepared status')
    _same(set(joint.get('stats', {})), set(prepared['formats_by_qname']), 'joined statistics roster')
    result = copy.deepcopy(joint)
    _add(result['provenance'], 'tessera_joint_anchors', {
        'plan_sha256': plan_binding['sha256'], 'prepared': prepared_binding,
        'inputs': copy.deepcopy(plan['inputs']),
        'calibration_input': copy.deepcopy(prepared['calibration_input']),
        'measured_cells': prepared['measured_cells'],
        'wire_validation': HISTORICAL_WIRE_VALIDATION,
        **{key: copy.deepcopy(prepared[key]) for key in ('render_origins', 'render_comparisons')},
    }, 'joined provenance')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-cost', required=True)
    parser.add_argument('--joint-cost-sha256', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(handoff(joint_binding={'path': args.joint_cost, 'sha256': args.joint_cost_sha256},
        plan_binding={'path': args.plan, 'sha256': args.plan_sha256}, output_path=args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
