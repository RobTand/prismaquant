import copy
import pytest
from prismaquant.native_runtime_cohort import bind_cohort, BUNDLE_SCHEMA
from prismaquant.measured_runtime_prices import RuntimePriceError


def panel(unit, *, routed=False):
    bundle={'schema':BUNDLE_SCHEMA,'dense_harness_sha256':'a'*64,'routed_harness_sha256':'b'*64,
        'resource_analysis_sha256':'c'*64,'resource_collector_source_sha256':'d'*64}
    source={'tessera_package_sha256':'e'*64,'runtime_contract_sha256':'f'*64,
            'harness_sha256':'a'*64,'native_cohort_bundle':bundle}
    runtime={'schema':'tessera.native_moe_runtime.v1' if routed else 'tessera.native_dense_runtime.v1',
        'image':'test/runtime@sha256:'+'7'*64,'image_declaration':{},'arithmetic':{},'versions':{},'gpu':{'uuid':'same','capability':[12,1]},
        'resource_collector':{'library_sha256':'1'*64,'analysis_source_sha256':'c'*64},
        'execution':{'mode':'resident','execution_mode':'eager','tensor_parallel':1},
        'source':source,'native_libraries':{'common.so':'2'*64}}
    if routed:
        source['routed_harness_sha256']='b'*64
        runtime['collective']={'world_size':1,'required_by_this_owner':False}
        runtime['execution']['owner_kind']='complete_routed_stack'
    return {'schema':'tessera.native_moe_panel.v1' if routed else 'tessera.native_dense_panel.v1',
        'unit':unit,'format':'TESSERA_E4M3_K1_R1024','shape':{},'runtime':runtime,
        'source_sha256':'8'*64,'calibration_sha256':'9'*64,
        'phases':{phase:{'m':512 if phase=='prefill' else 1,'expected_route':{'kind':'moe' if routed else 'dense'}} for phase in ('prefill','decode')}}


def test_mixed_operator_context_is_retained_under_one_explicit_runtime():
    dense,moe=panel('dense'),panel('moe',routed=True)
    cohort=bind_cohort([dense,moe]);contexts=cohort['operator_contexts']
    assert len(contexts)==2
    assert contexts['moe@TESSERA_E4M3_K1_R1024']['runtime']==moe['runtime']
    assert contexts['dense@TESSERA_E4M3_K1_R1024']['runtime']==dense['runtime']


@pytest.mark.parametrize('change',[
    lambda r:r['gpu'].update(uuid='other'),
    lambda r:r['source']['native_cohort_bundle'].update(routed_harness_sha256='0'*64),
    lambda r:r['source'].update(unknown='ignored'),
    lambda r:r.update(unknown='ignored'),
    lambda r:r['source'].pop('native_cohort_bundle'),
    lambda r:r['resource_collector'].update(library_sha256='0'*64),
])
def test_no_shared_runtime_coordinate_or_unknown_field_is_dropped(change):
    a,b=panel('a'),panel('b',routed=True);change(b['runtime'])
    with pytest.raises(RuntimePriceError):bind_cohort([a,b])


def test_library_conflict_between_later_panels_is_not_hidden_by_first():
    rows=[panel('a'),panel('b'),panel('c',routed=True)]
    rows[1]['runtime']['native_libraries']['route.so']='4'*64
    rows[2]['runtime']['native_libraries']['route.so']='5'*64
    with pytest.raises(RuntimePriceError,match='library bytes'):bind_cohort(rows)
    rows[2]['runtime']['native_libraries']['route.so']='4'*64
    assert bind_cohort(rows)['native_libraries']['route.so']=='4'*64


def test_mixed_context_roundtrips_with_all_operator_contexts_and_refuses_relabel():
    import json
    from prismaquant.native_receipt_table import derive_context
    from prismaquant.measured_runtime_prices import parse_runtime_context
    rows=[panel('dense'),panel('moe',routed=True)]
    context=derive_context(rows,relation={'fixture':'no scientific evidence'})
    context['operator_routes']={p['unit']:{p['format']:json.dumps(p['phases']['prefill']['expected_route'],sort_keys=True,separators=(',',':'))} for p in rows}
    assert context['schema']=='prismaquant.measured_runtime_context.v3'
    assert context['serving_context']['structure']=='mixed_native.v1'
    assert parse_runtime_context(context).as_dict()==context
    bad=copy.deepcopy(context);bad['serving_context']['structure']='dense'
    with pytest.raises(RuntimePriceError,match='structure'):parse_runtime_context(bad)
    bad=copy.deepcopy(context);bad.pop('native_cohort')
    with pytest.raises(RuntimePriceError):parse_runtime_context(bad)
    bad=copy.deepcopy(context);bad['operator_routes']['moe'][rows[1]['format']]='{}'
    with pytest.raises(RuntimePriceError,match='route'):parse_runtime_context(bad)
