"""Freeze dense execution inputs before final joint-cost coordinates exist.

The schema registry in native_panel_kind owns receipt projection and late binding.
"""
from __future__ import annotations

import copy
from .joint_aura import identity_sha256

from .native_panel_kind import DENSE_EXECUTION_PANEL_SCHEMA as RAW_PANEL_SCHEMA


def require_reference_quantizer(data, activation, probe=None):
    """A served static oracle is bound to the actual quality producer build."""
    if not (activation.get('static_contract') or {}).get('measured_as_served'):
        return None
    reference=data.get('reference_served_quantizer')
    if not isinstance(reference,dict) or reference.get('backend')!='registered_scaled_fp4_quant':
        raise ValueError('static native references require their registered served quantizer identity')
    if probe is not None:
        # Compared on the fields that change the priced number (schema and
        # the reuse axes), not as a whole record: ``dequant_kernel`` (#1211)
        # names bit-identical implementations, so a reference recorded before
        # the field existed still binds a row that carries it.
        from .nvfp4_activation_contract import served_quantizer_reuse_differences
        actual = (probe.get('arithmetic') or {}).get('served_quantizer')
        if (not isinstance(actual, dict)
                or served_quantizer_reuse_differences(reference, actual)):
            raise ValueError('native reference quantizer differs from actual joint quality arithmetic')
    return copy.deepcopy(reference)




def freeze_execution_panel(inputs, preflight, *, source_sha256):
    from .native_operator_panel import INPUT_SCHEMA, PHASES, _equal, _sha, require_native_execution
    _sha(source_sha256, 'source model')
    if inputs.get('schema') != INPUT_SCHEMA:
        raise ValueError('execution panel needs independently prepared PWC inputs')
    if (preflight.get('schema') != 'tessera.native_dense_preflight.v1'
            or preflight.get('status') != 'untimed_preparation'):
        raise ValueError('execution panel needs untimed native preflight')
    reference=require_reference_quantizer(inputs,inputs['activation'])
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
    execution = require_native_execution(inputs, preflight)
    _equal(preflight['runtime']['image'], inputs['runtime_image'], 'runtime image')
    route = operator['declared_route']
    _equal(route['contract'], operator['activation_contract'], 'activation route')
    identity = {'qname': inputs['unit'], 'format': inputs['format'],
                **{key: inputs[key] for key in ('source_weight','rendered_weight','activation')}}
    return copy.deepcopy({
        'schema': RAW_PANEL_SCHEMA, **({'reference_served_quantizer':reference} if reference is not None else {}), 'unit': inputs['unit'], 'format': inputs['format'],
        'shape': inputs['shape'], 'source_sha256': source_sha256,
        'calibration_sha256': inputs['calibration']['calibration_sha256'],
        'operator_identity': identity, 'operator_identity_sha256': identity_sha256(identity),
        'wire': inputs['wire'], 'execution': execution, 'runtime': preflight['runtime'],
        'native_tensors_sha256': preflight['native_tensors_sha256'],
        'scheme_sha256': preflight['scheme_sha256'], 'numerics': inputs['numerics'],
        'numerics_derivation': inputs['numerics_derivation'],
        'activation_quantizer_attestation': inputs['activation_quantizer_attestation'],
        'phases': {phase: {**inputs['phases'][phase], 'expected_route': route} for phase in PHASES}})
