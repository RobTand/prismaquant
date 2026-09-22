"""Truthful late joint-cost binding of immutable pre-cost native measurements.

Execution panels omit cost/probe coordinates that do not exist yet. A later
binding retains the raw receipt verbatim and requires equality of every measured
coordinate with the independently frozen final joint panel. It never overwrites
or backdates the producer's execution panel.
"""
from __future__ import annotations

import copy
from .joint_aura import identity_sha256

RAW_PANEL_SCHEMA = 'tessera.native_dense_execution_panel.v1'
RAW_RECEIPT_SCHEMA = 'tessera.native_dense_execution_receipt.v1'
BOUND_SCHEMA = 'prismaquant.native_dense_late_binding.v1'
_OPERATOR_FIELDS = ('qname', 'format', 'source_weight', 'rendered_weight', 'activation')


def execution_panel_from_joint(panel):
    if panel.get('schema') != 'tessera.native_dense_panel.v1':
        raise ValueError('late binding requires final joint dense panel')
    value = copy.deepcopy(panel)
    joint = value.pop('joint_operator_identity')
    if identity_sha256(joint) != value.pop('joint_operator_identity_sha256'):
        raise ValueError('final joint operator digest differs')
    value.pop('cost_sha256')
    value.pop('probe_identity_sha256')
    value['schema'] = RAW_PANEL_SCHEMA
    value['operator_identity'] = {key: joint[key] for key in _OPERATOR_FIELDS}
    value['operator_identity_sha256'] = identity_sha256(value['operator_identity'])
    return value


def freeze_execution_panel(inputs, preflight, *, source_sha256):
    from .native_operator_panel import EXECUTION, INPUT_SCHEMA, PHASES, _equal, _sha
    _sha(source_sha256, 'source model')
    if inputs.get('schema') != INPUT_SCHEMA:
        raise ValueError('execution panel needs independently prepared PWC inputs')
    if (preflight.get('schema') != 'tessera.native_dense_preflight.v1'
            or preflight.get('status') != 'untimed_preparation'):
        raise ValueError('execution panel needs untimed native preflight')
    operator = preflight['operator']
    for key in ('source_weight', 'rendered_weight'):
        _equal(operator[key], inputs[key], key)
    _equal(operator['input_global_scale'], inputs['activation']['input_global_scale'], 'input scale')
    _equal(operator['clip_enabled'], False, 'native clip')
    _equal(operator['wire_sha256'], inputs['wire']['blob_sha256'], 'wire bytes')
    _equal(operator['wire_record_sha256'], identity_sha256(inputs['wire']['record']), 'wire record')
    _equal(preflight['runtime_sha256'], identity_sha256(preflight['runtime']), 'runtime digest')
    _equal(preflight['native_tensors_sha256'], identity_sha256(operator['native_tensors']), 'native tensors')
    _equal(preflight['scheme_sha256'], identity_sha256(operator['scheme']), 'native scheme')
    _equal(preflight['runtime']['execution'], EXECUTION, 'execution')
    _equal(preflight['runtime']['image'], inputs['runtime_image'], 'runtime image')
    route = operator['declared_route']
    _equal(route['contract'], operator['activation_contract'], 'activation route')
    identity = {'qname': inputs['unit'], 'format': inputs['format'],
                **{key: inputs[key] for key in ('source_weight','rendered_weight','activation')}}
    return copy.deepcopy({
        'schema': RAW_PANEL_SCHEMA, 'unit': inputs['unit'], 'format': inputs['format'],
        'shape': inputs['shape'], 'source_sha256': source_sha256,
        'calibration_sha256': inputs['calibration']['calibration_sha256'],
        'operator_identity': identity, 'operator_identity_sha256': identity_sha256(identity),
        'wire': inputs['wire'], 'execution': dict(EXECUTION), 'runtime': preflight['runtime'],
        'native_tensors_sha256': preflight['native_tensors_sha256'],
        'scheme_sha256': preflight['scheme_sha256'], 'numerics': inputs['numerics'],
        'numerics_derivation': inputs['numerics_derivation'],
        'activation_quantizer_attestation': inputs['activation_quantizer_attestation'],
        'phases': {phase: {**inputs['phases'][phase], 'expected_route': route} for phase in PHASES}})


def bind_execution_receipt(raw_receipt, final_panel):
    """Preserve the raw evidence and join only an identical final execution."""
    if (raw_receipt.get('schema') != RAW_RECEIPT_SCHEMA
            or raw_receipt.get('status') != 'timing_admissible'):
        raise ValueError('late binding requires numerically admitted execution receipt')
    panel = execution_panel_from_joint(final_panel)
    if (raw_receipt['panel'] != panel
            or raw_receipt['panel_sha256'] != identity_sha256(panel)):
        raise ValueError('late joint binding changes measured execution identity')
    return copy.deepcopy({'schema': BOUND_SCHEMA,
        'raw_receipt': raw_receipt, 'raw_receipt_sha256': identity_sha256(raw_receipt),
        'panel': final_panel, 'panel_sha256': identity_sha256(final_panel)})


def resolve_execution_binding(binding, final_panel):
    """Verified in-memory view for existing numerical/resource consumers.

    The on-disk schema remains explicitly late-bound and includes original raw
    evidence. The final cost row is independently checked by the table builder.
    """
    if binding.get('schema') != BOUND_SCHEMA:
        raise ValueError('unsupported execution binding')
    raw = binding['raw_receipt']
    if identity_sha256(raw) != binding['raw_receipt_sha256']:
        raise ValueError('raw receipt digest differs')
    if binding != bind_execution_receipt(raw, final_panel):
        raise ValueError('late execution binding differs from final panel')
    view = copy.deepcopy(raw)
    view.update(schema='tessera.native_dense_operator_receipt.v1',
                panel=copy.deepcopy(final_panel), panel_sha256=identity_sha256(final_panel))
    return view
