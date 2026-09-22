"""One original GLM routed owner: existing PWC references and actual routing.

This is a PB-admitted research preparation/measurement adapter, not a sharder.
Source/H provenance comes from independently bound original qualification;
source bytes, render bytes and every original wire are checked again here.
"""
import argparse,hashlib,inspect,json,pickle,io,os
from pathlib import Path

IMAGE='eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def checked(binding):
    p=Path(binding['path'])
    if sha(p)!=binding['sha256']:raise ValueError('bound input changed: '+str(p))
    return p


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def prepare(spec,out,local_root,device_bytes):
    import torch
    from prismaquant.memory_management import enforce_device_envelope
    envelope=enforce_device_envelope("cuda",device_bytes,where="native GLM reference preparation")
    from experiments.native_local_handoff import LocalHandoff,LocalSourceMapping,strict_read,bind_strict_inputs
    from prismaquant.prismabuild_progress import report
    bind_strict_inputs()
    if not report("head",0,unit="native_operators"):
        raise RuntimeError("native input read-phase channel is unavailable")
    local=LocalHandoff(local_root,max_bytes=40<<30)
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from prismaquant import native_moe_panel as pq
    from prismaquant.production_weight_cache import ProductionWeightCache,_cb_cache_tensor_identity
    from prismaquant.model_profiles import profile_from_config
    from prismaquant.tessera_expert_projection import source_unit_weight
    from prismaquant.tessera_calibration_cache import require_capture_contract
    from prismaquant.nvfp4_activation_contract import bind_served_quantizer_identity
    metadata=json.loads(strict_read(spec['metadata']))
    fmt=spec['format'];layer=metadata['layer'];unit=f'model.language_model.layers.{layer}.mlp.experts'
    rows={r['qname']:r for r in metadata['rows']}
    payload=torch.load(io.BytesIO(strict_read(spec['boundary'])),map_location='cpu',weights_only=True)
    capture_ref=local.write('canonical-capture.json',raw=strict_read(spec['canonical_capture']))
    capture=require_capture_contract(capture_ref['path'],expected_sha256=spec['canonical_capture']['sha256'])
    transported={'source':'routed_boundary_capture','boundary_metadata':payload['metadata'],**payload['tensors']}
    routed,phases,bias=pq.routed_boundary_inputs(transported,calibration_receipt=metadata['calibration_input'],
        capture_manifest=capture,device='cuda:0',source_model_identity=metadata['source_model_identity'])
    if routed['unit']!=unit:raise ValueError('routing and original wire layer differ')
    shape=routed['shape'];source=routed['producer_source'];model=metadata['source_model_identity']['source']
    members=[]
    for expert in range(shape['n_routed_experts']):
        for role,projection in (('w1','gate_proj'),('w3','up_proj'),('w2','down_proj')):
            q=f'{unit}.{expert}.{projection}';entry=rows[q]['formats'][fmt]
            members.append({'unit':q,'expert':expert,'role':role,'format':fmt,
                'shape':entry['record']['identity']['source']['shape']})
    wire_bytes=sum(rows[m['unit']]['formats'][fmt]['record']['blob_bytes'] for m in members)
    tensor_bytes=sum(__import__('math').prod(m['shape'])*2 for m in members)
    if wire_bytes+tensor_bytes+(4<<30)>40<<30 or 2*tensor_bytes+(8<<30)>device_bytes:
        raise ValueError('whole-owner preparation exceeds declared CPU/device envelope')
    if set(rows)!={m['unit'] for m in members}:raise ValueError('metadata must cover the whole exact routed owner')
    config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=True)
    profile=profile_from_config(config)
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
    with torch.device('meta'):experts=Glm5NextTextExperts(config.text_config)
    original_cache=pickle.loads(strict_read(metadata['inputs']['production_cache']))
    paths={(m['unit'],fmt):str(rows[m['unit']]['formats'][fmt]['render']) for m in members}
    maxima={m['unit']:original_cache.activation_max_abs[m['unit']] for m in members}
    policy=None;reference_quantizer=None
    if fmt.startswith('TESSERA_E2M1'):
        from prismaquant.joint_served_activation import verify_policy
        policy=verify_policy(spec['activation_policy'],original_prepared=metadata['inputs']['prepared'],read_bound=strict_read)
        maxima={name:policy['effective_max_abs'][name] for name in maxima}
        reference_quantizer=bind_served_quantizer_identity(require=True,context='whole GLM native reference').as_record()
    cache=ProductionWeightCache(weights=paths,levers=original_cache.levers,activation_max_abs=maxima)
    del original_cache
    cache.enable_lru(24<<30);cache.enable_file_load_receipts(max_file_bytes=64<<20)
    plan=json.loads(strict_read(spec['scientific_plan']));serving_bytes=strict_read(spec['serving_config'])
    local_serving=local.write('serving-config.json',raw=serving_bytes)
    source_bindings={};blobs={};records={};encodings={};refs=[]
    from prismaquant.layer_streaming import _source_safe_open,_await_layer_readset
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    for index,member in enumerate(members):
        if index%32==0:
            if not report(f'members-{index//32:03d}',0,unit='native_operators'):
                raise RuntimeError('native input read-phase report refused')
            by_shard={}
            for upcoming in members[index:index+32]:
                row=rows[upcoming['unit']]['formats'][fmt]
                key=row['source_projection']['source_tensor'];file=str(Path(model)/source['tensors'][key])
                by_shard.setdefault(file,[]).append((key,key))
            _await_layer_readset(by_shard)
            import time
            from prismaquant.residency_map import residency_resolver,RANGE_HIT
            from prismaquant.residency_shard_reader import await_staged_spans
            wanted=[]
            for upcoming in members[index:index+32]:
                row=rows[upcoming['unit']]['formats'][fmt]
                for key in ('wire','render'):
                    path=Path(row[key]);size=path.stat().st_size
                    wanted.append((path,0,size,size))
            if await_staged_spans(residency_resolver(),wanted,deadline=time.monotonic()+300)!=RANGE_HIT:
                raise RuntimeError('native wire/render window is not staged')
        q=member['unit'];entry=rows[q]['formats'][fmt];record=entry['record']
        # The separate bound original qualification owns source/H/recipe
        # provenance. This is reuse of that identity, never a new encode.
        if record['identity']['calibration']['settings']['hessian']['fit_ids_sha256']!=metadata['calibration_input']['provenance']['fit_ids_sha256']:
            raise ValueError('original wire calibration differs')
        source_tensor=entry['source_projection']['source_tensor']
        source_file=str(Path(model)/source['tensors'][source_tensor])
        with _source_safe_open(source_file,framework='pt',device='cpu') as source_reader:
            source_weight=source_reader.get_tensor(source_tensor)
        if _cb_cache_tensor_identity(source_weight)!=rows[q]['source_weight']:
            raise ValueError('actual source member changed: '+q)
        source_bindings[q]=local.write(f'source-{index:04d}.pt',tensor=source_weight)
        del source_weight
        records[q]=record;encodings[q]=record['identity'];blobs[q]=_read_verified_wire_blob(entry)[0]
        local_wire=local.write(f'wire-{index:04d}.tessera',raw=blobs[q])
        render=cache.get(q,fmt);original_render_receipt=cache.file_load_receipt((q,fmt),render)
        expected_render_sha=entry['render_file_sha256']
        if expected_render_sha is not None and original_render_receipt['sha256']!=expected_render_sha:
            raise ValueError('qualified PWC render bytes changed')
        local_render=local.write(f'render-{index:04d}.pt',tensor=render)
        refs.append({'unit':q,'expert':member['expert'],'role':member['role'],'format':fmt,
            'wire':local_wire['path'],'record':record,'render':local_render['path'],
            'render_file_sha256':local_render['sha256'],'original_render_file_sha256':original_render_receipt['sha256'],
            'original_render_path':entry['render'],'source_local':source_bindings[q],
            'source_file':source_file,'source_tensor':source_tensor})
    weights=LocalSourceMapping(source_bindings)
    execution=plan['execution']
    request={key:execution[key] for key in ('n_probes','seed_base','token_scope','temperature')}
    request.update(distribution='rademacher',normalization='global_kl_fisher',source_model=model,
        source_shards=source['files'],source_config_sha256=source['config_sha256'],source_auxiliary_sha256=source['auxiliary_sha256'])
    serving=Path(local_serving['path'])
    # Existing whole-owner gate, declared before execution; QDQ is additionally
    # required to be exactly equal. 2^-6 is four BF16 unit roundoffs.
    numerics={'atol':4*2.0**-8,'rtol':4*2.0**-8}
    inputs,tensors=pq.prepare_moe_inputs(cache,weights,phases,unit=unit,members=members,shape=shape,
        routing=routed['routing'],calibration_receipt=metadata['calibration_input'],routing_capture=routed,
        experts_module=experts,profile=profile,wire_blobs=blobs,wire_records=records,encoding_identities=encodings,
        numerics=numerics,max_resident_bytes=24<<30,max_temporary_bytes=24<<30,runtime_image=IMAGE,
        serving_config_sha256=spec['serving_config']['sha256'],probe_request=request,format_name=fmt,retain_source_tensors=False)
    if reference_quantizer is not None:inputs['reference_served_quantizer']=reference_quantizer
    for ref,member in zip(refs,inputs['members']):
        tensor=cache.get(ref['unit'],fmt)
        receipt=cache.file_load_receipt((ref['unit'],fmt),tensor)
        if ref['original_render_file_sha256']!=receipt['sha256']:raise ValueError('PWC identity changed after preparation')
        expected=rows[ref['unit']]['formats'][fmt]['render_file_sha256']
        if expected is not None and expected!=ref['original_render_file_sha256']:raise ValueError('qualified render bytes changed')
        if member['activation']['input_global_scale'] is not None:ref['input_global_scale']=member['activation']['input_global_scale']
    out.mkdir(parents=True,exist_ok=False)
    phase_data={key:value.detach().cpu().contiguous().clone() for key,value in tensors.items()
                if key.split('.',1)[0] in pq.PHASES}
    phase_data['routing_bias']=bias.detach().cpu().contiguous()
    save_file(phase_data,str(out/'phase-tensors.safetensors'))
    dump(out/'inputs.json',inputs);dump(out/'weight-references.json',refs)
    bundle=local.seal(os.environ['PRISMABUILD_ACTION_KEY'])
    dump(out/'origin.json',{'device_envelope':envelope,'resource_budget':{'cpu_container_bytes':40<<30,'device_bytes':device_bytes,'aggregate_bytes':96<<30,'wire_bytes':wire_bytes,'render_tensor_bytes':tensor_bytes},'spec':spec,'local_serving_config':local_serving,'local_bundle':bundle,'source_sha256':metadata['source_model_identity']['content_sha256'],
        'encoding_provenance':'reused independently bound original full512 source/H/recipe qualification; no re-encode',
        'artifacts':{name:sha(out/name) for name in ('inputs.json','phase-tensors.safetensors','weight-references.json')}})
    print(json.dumps({'status':'prepared','unit':unit,'format':fmt,'members':len(members),
        'inputs_sha256':sha(out/'inputs.json'),'origin_sha256':sha(out/'origin.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--spec-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True);p.add_argument('--device-bytes',type=int,required=True);a=p.parse_args()
    spec=json.loads(checked({'path':a.spec,'sha256':a.spec_sha256}).read_text());prepare(spec,a.out,a.local_root,a.device_bytes)
