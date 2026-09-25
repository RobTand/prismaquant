"""Common runtime projection retaining each validated operator's full context.

This protocol is opt-in: legacy receipts remain under whole-runtime equality.
The table caller must consume numerical/resource receipts before projection.
No operator field is silently treated as a shared runtime coordinate.
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
from .measured_runtime_prices import RuntimePriceError, identity_sha256
from .schemas import Contract

SCHEMA='prismaquant.native_runtime_cohort.v1'
BUNDLE_SCHEMA='tessera.native_operator_source_bundle.v1'
_BUNDLE_FIELDS={'schema','dense_harness_sha256','routed_harness_sha256',
                'resource_analysis_sha256','resource_collector_source_sha256'}
_COMMON_FIELDS={'image','image_declaration','arithmetic','versions','gpu','resource_collector'}
_BASE_FIELDS=_COMMON_FIELDS|{'schema','execution','source','native_libraries'}


_fail = Contract(RuntimePriceError, "native runtime cohort: ").fail


def project(panel):
    """Project only the versioned producer bundle, retaining every original field."""
    runtime=panel['runtime'];routed=panel['schema']=='tessera.native_moe_panel.v1'
    if panel['schema'] not in ('tessera.native_dense_panel.v1','tessera.native_moe_panel.v1'):
        _fail('requires an independently bound final native panel')
    expected=_BASE_FIELDS|({'collective'} if routed else set())
    if set(runtime)!=expected:_fail('unknown or missing runtime fields')
    schema='tessera.native_moe_runtime.v1' if routed else 'tessera.native_dense_runtime.v1'
    if runtime['schema']!=schema:_fail('operator schema differs')
    source=runtime['source'];fields={'tessera_package_sha256','runtime_contract_sha256',
        'harness_sha256','native_cohort_bundle'}|({'routed_harness_sha256'} if routed else set())
    if set(source)!=fields:_fail('unknown or missing source fields')
    bundle=source['native_cohort_bundle']
    if not isinstance(bundle,dict) or set(bundle)!=_BUNDLE_FIELDS or bundle['schema']!=BUNDLE_SCHEMA:
        _fail('versioned complete source bundle is required')
    import re
    for key,value in bundle.items():
        if key!='schema' and (not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)):
            _fail('invalid source bundle digest')
    if bundle['dense_harness_sha256']!=source['harness_sha256']:
        _fail('dense harness differs from common bundle')
    if routed and bundle['routed_harness_sha256']!=source['routed_harness_sha256']:
        _fail('routed harness differs from common bundle')
    if bundle['resource_analysis_sha256']!=runtime['resource_collector']['analysis_source_sha256']:
        _fail('resource analysis differs from common bundle')
    execution=runtime['execution']
    shared_execution={key:execution[key] for key in ('mode','execution_mode','tensor_parallel')}
    common={key:runtime[key] for key in _COMMON_FIELDS}
    common.update(execution=shared_execution,source={key:source[key] for key in
        ('tessera_package_sha256','runtime_contract_sha256','native_cohort_bundle')})
    libraries=runtime['native_libraries']
    if not isinstance(libraries,Mapping):_fail('missing native libraries')
    # Retain the complete runtime and panel operator coordinates. Consumers can
    # inspect schema/collective/workspace/route without reconstructing defaults.
    operator={'unit':panel['unit'],'format':panel['format'],
        'structure':'routed_moe' if routed else 'dense','runtime':copy.deepcopy(runtime),
        'shape':copy.deepcopy(panel['shape']),
        'routes':{phase:copy.deepcopy(value['expected_route']) for phase,value in panel['phases'].items()}}
    for key in ('workspace','routing','routing_capture_sha256','serving_config_sha256','source_execution',
                'config_sha256','scheme_sha256','native_tensors_sha256','execution','runtime_binding',
                'joint_operator_identity_sha256','source_sha256','calibration_sha256','source_acquisition'):
        if key in panel:operator[key]=copy.deepcopy(panel[key])
    return {'common':copy.deepcopy(common),'libraries':dict(libraries),'operator':operator}


def bind_cohort(panels):
    if not panels:_fail('empty cohort')
    projections=[project(panel) for panel in panels]
    first=projections[0]['common'];libraries={};operators={}
    for projected in projections:
        if identity_sha256(projected['common'])!=identity_sha256(first):
            _fail('image, GPU, shared execution, source bundle or arithmetic differs')
        for path,digest in projected['libraries'].items():
            if path in libraries and libraries[path]!=digest:
                _fail('shared library bytes differ: '+path)
            libraries[path]=digest
        operator=projected['operator'];key=operator['unit']+'@'+operator['format']
        if key in operators:_fail('duplicate operator context')
        operators[key]=operator
    return {'schema':SCHEMA,'common':first,'native_libraries':dict(sorted(libraries.items())),
            'operator_contexts':operators}


def validate_cohort(cohort):
    """Recompute the projection from all retained operator contexts."""
    if not isinstance(cohort,dict) or set(cohort)!={'schema','common','native_libraries','operator_contexts'} or cohort['schema']!=SCHEMA:
        _fail('unknown cohort grammar')
    panels=[]
    for key,operator in cohort['operator_contexts'].items():
        if key!=operator['unit']+'@'+operator['format']:
            _fail('operator context key differs')
        if operator['structure'] not in ('dense','routed_moe'):
            _fail('unknown operator structure')
        panel={name:copy.deepcopy(value) for name,value in operator.items() if name not in ('structure','routes')}
        panel['schema']=('tessera.native_moe_panel.v1' if operator['structure']=='routed_moe' else 'tessera.native_dense_panel.v1')
        panel['phases']={phase:{'expected_route':route} for phase,route in operator['routes'].items()}
        panels.append(panel)
    if identity_sha256(bind_cohort(panels))!=identity_sha256(cohort):
        _fail('retained context projection differs')
    return cohort
