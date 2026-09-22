"""Whole packed MoE execution receipts before final distortion rows exist."""
import copy
from .joint_aura import identity_sha256

RAW_PANEL_SCHEMA='tessera.native_moe_execution_panel.v1'
RAW_RECEIPT_SCHEMA='tessera.native_moe_execution_receipt.v1'
BOUND_SCHEMA='prismaquant.native_moe_late_binding.v1'


def member_identity(member):
    return {'qname':member['unit'],**{key:member[key] for key in
        ('format','source_weight','rendered_weight','activation')}}


def execution_panel_from_joint(panel):
    if panel.get('schema')!='tessera.native_moe_panel.v1':
        raise ValueError('whole MoE late binding requires final joint panel')
    value=copy.deepcopy(panel);value['schema']=RAW_PANEL_SCHEMA
    value.pop('cost_sha256');value.pop('probe_identity_sha256')
    value['runtime_binding'].pop('member_operator_identity_sha256')
    value['runtime_binding']['member_execution_identity_sha256']={
        m['unit']:identity_sha256(member_identity(m)) for m in value['members']}
    return value


def freeze_execution_panel(inputs,preflight,*,source_sha256):
    from . import native_moe_panel as owner
    from .native_operator_panel import _equal,_sha,operator_route_identity
    _sha(source_sha256,'source model')
    if inputs.get('schema')!=owner.INPUT_SCHEMA or preflight.get('schema')!='tessera.native_moe_preflight.v1':
        raise ValueError('whole MoE execution requires prepared inputs and native preflight')
    if preflight.get('status')!='untimed_preparation':
        raise ValueError('whole MoE preflight must precede timing')
    members=owner._member_roster(inputs['unit'],inputs['members'],inputs['shape'])
    owner._calibration_and_capture(inputs['calibration'],inputs['routing_capture'],
        unit=inputs['unit'],shape=inputs['shape'],routing=inputs['routing'])
    _equal(inputs['routing_capture_sha256'],identity_sha256(inputs['routing_capture']),'routing capture')
    from .native_execution_binding import require_reference_quantizer
    reference=require_reference_quantizer(inputs,members[0]['activation'])
    operator=preflight['operator']
    _equal(operator['members'],[owner._native_member_identity(m) for m in members],'native members')
    for key in ('shape','routing','profile_role_order','routing_capture_sha256','serving_config_sha256'):
        _equal(operator[key],inputs[key],'native '+key)
    for key in ('native_tensors','scheme','config'):
        _equal((operator if key=='config' else preflight)[key+'_sha256'],identity_sha256(operator[key]),key+' digest')
    _equal(preflight['runtime_sha256'],identity_sha256(preflight['runtime']),'runtime digest')
    _equal(preflight['runtime']['image'],inputs['runtime_image'],'runtime image')
    _equal(preflight['runtime']['execution'],inputs['execution'],'runtime execution')
    owner._workspace_identity(preflight['workspace'])
    _equal(preflight['workspace_sha256'],identity_sha256(preflight['workspace']),'workspace digest')
    route=operator['declared_route'];owner.validate_compact_native_route(inputs,route)
    execution=owner._source_execution(inputs['routing_capture']['source_execution'],unit=inputs['unit'])
    phases={}
    for phase in owner.PHASES:
        expected=inputs['phases'][phase]
        owner._transport_identity(expected)
        _equal(operator['phases'][phase]['transport'],expected['transport'],phase+' transport')
        for key in ('input','topk_ids','topk_weights'):
            _equal(expected[key],inputs['routing_capture']['phases'][phase][key],phase+' capture '+key)
        phases[phase]={**expected,'expected_route':route}
    binding={'member_formats':{m['unit']:m['format'] for m in members},
        'member_execution_identity_sha256':{m['unit']:identity_sha256(member_identity(m)) for m in members},
        'member_shapes':{m['unit']:owner.rank_local_member_shape(inputs['shape'],m['role']) for m in members},
        'operator_route':operator_route_identity(route)}
    return copy.deepcopy({'schema':RAW_PANEL_SCHEMA,**({'reference_served_quantizer':reference} if reference is not None else {}),'unit':inputs['unit'],'format':inputs['format'],
        'shape':inputs['shape'],'members':members,'profile_role_order':inputs['profile_role_order'],
        'routing':inputs['routing'],'routing_capture_sha256':inputs['routing_capture_sha256'],
        'source_sha256':source_sha256,'calibration_sha256':inputs['calibration']['calibration_sha256'],
        'source_execution':execution,'source_execution_qualification_sha256':None,'probe_scope':None,
        'runtime_binding':binding,'execution':inputs['execution'],'runtime':preflight['runtime'],
        'native_tensors_sha256':preflight['native_tensors_sha256'],'scheme_sha256':preflight['scheme_sha256'],
        'config_sha256':operator['config_sha256'],'serving_config_sha256':inputs['serving_config_sha256'],
        'workspace':preflight['workspace'],'workspace_sha256':preflight['workspace_sha256'],
        'numerics':inputs['numerics'],'phases':phases})


def bind_execution_receipt(raw,final_panel):
    if raw.get('schema')!=RAW_RECEIPT_SCHEMA or raw.get('status')!='timing_admissible':
        raise ValueError('whole MoE late binding requires admitted raw execution')
    panel=execution_panel_from_joint(final_panel)
    if identity_sha256(raw['panel'])!=identity_sha256(panel) or raw['panel_sha256']!=identity_sha256(panel):
        raise ValueError('whole MoE late binding changes measured execution')
    return copy.deepcopy({'schema':BOUND_SCHEMA,'raw_receipt':raw,'raw_receipt_sha256':identity_sha256(raw),
        'panel':final_panel,'panel_sha256':identity_sha256(final_panel)})


def resolve_execution_binding(binding,final_panel):
    raw=binding['raw_receipt']
    if identity_sha256(raw)!=binding['raw_receipt_sha256'] or binding!=bind_execution_receipt(raw,final_panel):
        raise ValueError('whole MoE late binding differs from immutable evidence')
    view=copy.deepcopy(raw);view.update(schema='tessera.native_moe_operator_receipt.v1',
        panel=copy.deepcopy(final_panel),panel_sha256=identity_sha256(final_panel))
    return view
