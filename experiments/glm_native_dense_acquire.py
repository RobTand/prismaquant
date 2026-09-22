#!/usr/bin/env python3
"""Original GLM dense wire -> independent inputs -> unbound native receipt.

Run only inside an admitted PB action. No encoding or joint distortion is done.
The historical encoder stamp stays historical; actual source/H/recipe/fixture
coordinates are rederived and compared before the original wire is accepted.
"""
import argparse,hashlib,json,pickle
from pathlib import Path

BASE=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
PANEL=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel')
PREPARED=PANEL/'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json'
PREPARED_SHA='962207a3385e9531adaf951b823871a2fb7ff4684320e7a8e19a1d0aa85d8f16'
ROW=BASE/'activation-runtime-allocation-20260911/extension-r1024-02/workspace/rows/row-0087'
CAPTURE=BASE/'workspace/calibration-cache/capture_manifest.json'
CAPTURE_SHA='f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c'
CENSUS=ROW.parents[1]/'census.json'
UNIT='model.language_model.layers.0.mlp.down_proj'
IMAGE='eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def prepare(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera import cached_unit
    from prismaquant.native_operator_panel import prepare_native_inputs
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.tessera_calibration_cache import require_capture_contract,prefetch_capture
    from prismaquant.tessera_hessian import activation_source
    from prismaquant.tessera_formats import parse_tessera_format_name
    assert sha(PREPARED)==PREPARED_SHA
    prepared=json.loads(PREPARED.read_text());assert args.format in prepared['formats_by_qname'][UNIT]
    capture=require_capture_contract(CAPTURE,expected_sha256=CAPTURE_SHA)
    census=json.loads(CENSUS.read_text());assert sha(CENSUS)==capture['identity']['census_sha256']
    name=UNIT+'.weight'; source=census['expert_projection']['producer']['source']
    with safe_open(str(Path(census['model'])/source['tensors'][name]),framework='pt',device='cpu') as f:
        weight=f.get_tensor(name).to('cuda')
    (acts,hessians,counts,maxima),_=prefetch_capture(CAPTURE,expected_sha256=CAPTURE_SHA,
        expected_identity=capture['identity'],census=census,names=[UNIT],device='cuda')
    checkpoint=ROW/'cost.anchors.json.parts/units'/f'{hashlib.sha256(UNIT.encode()).hexdigest()}.pkl'
    envelope=pickle.loads(checkpoint.read_bytes());assert hashlib.sha256(envelope['payload']).hexdigest()==envelope['payload_sha256']
    record=pickle.loads(envelope['payload'])['wire_records'][args.format]
    blob=(ROW/'cache/wire'/record['file']).read_bytes()
    family,rung=parse_tessera_format_name(args.format)
    encoding=cached_unit.encoding_input_identity(weight,UNIT,family.payload_grid(),int(rung),
        activation=activation_source(hessians,capture['identity']['calibration']))
    historical=record['identity']; changed=[]
    for key in encoding:
        if key=='encoder_source_sha256':continue
        if encoding[key]!=historical[key]:raise ValueError('actual encoding input differs: '+key)
    reuse={'encoded_now':False,'historical_encoder_source_sha256':historical['encoder_source_sha256'],
           'current_encoder_source_sha256':encoding['encoder_source_sha256'],
           'verified_coordinates':[key for key in encoding if key!='encoder_source_sha256']}
    encoding['encoder_source_sha256']=historical['encoder_source_sha256']
    cache_path=Path(prepared['production_cache']['path']);assert sha(cache_path)==prepared['production_cache']['sha256']
    full=pickle.loads(cache_path.read_bytes())
    render=full.weights[(UNIT,args.format)]
    cache=ProductionWeightCache(weights={(UNIT,args.format):render},levers=full.levers,
        activation_max_abs={UNIT:full.activation_max_abs[UNIT]})
    del full,hessians
    quantizers=None
    if args.activation_quantizers:
        from prismaquant.tessera_runtime_contract import _parse_activation_contract,_parse_activation_generated
        block=json.loads(args.activation_quantizers.read_text())
        # The existing producer validates the emitted image-scoped table.
        # Parse its actual contract rows through the typed reader API; retain
        # the original artifact instead of relabelling its schema or image.
        if block['schema']!='tessera.activation-quantizer.v2':raise ValueError('quantizer producer schema differs')
        if set(block['platforms'])!={'sm_121'}:raise ValueError('quantizer platform differs')
        entry=block['platforms']['sm_121']
        if entry['generated']['image']!=IMAGE:raise ValueError('quantizer runtime differs')
        generated=_parse_activation_generated(entry['generated'],'native measurement generated')
        quantizers={'sm_121':{name:_parse_activation_contract(value,platform='sm_121',name=name,
            where='native measurement '+name,generated=generated) for name,value in entry['contracts'].items()}}
    inputs,tensors=prepare_native_inputs(cache,weight,acts[UNIT].to(torch.bfloat16),unit=UNIT,
        format_name=args.format,calibration_receipt=prepared['calibration_input'],wire_blob=blob,
        wire_record=record,encoding_identity=encoding,prefill_rows=512,decode_rows=1,
        max_resident_bytes=2<<30,activation_quantizers=quantizers)
    inputs['runtime_image']=IMAGE
    args.out.mkdir(parents=True,exist_ok=False)
    if args.activation_quantizers:(args.out/'activation-quantizers.json').write_bytes(args.activation_quantizers.read_bytes())
    (args.out/'weight.tessera').write_bytes(blob);dump(args.out/'wire-record.json',record)
    save_file({key:t.detach().cpu().contiguous().clone() for key,t in tensors.items()},str(args.out/'tensors.safetensors'))
    dump(args.out/'inputs.json',inputs)
    dump(args.out/'origin.json',{'prepared_sha256':PREPARED_SHA,'capture_sha256':CAPTURE_SHA,
        'source_sha256':prepared['source_model_identity']['content_sha256'],'encoder_reuse':reuse,
        'source_unit':UNIT,'format':args.format,'pwc_path':render,'capture_entry':capture['entries'][UNIT],
        'artifacts':{name:sha(args.out/name) for name in ('weight.tessera','wire-record.json','tensors.safetensors','inputs.json')}})
    print(json.dumps({'status':'independent_inputs_prepared','path':str(args.out),'inputs_sha256':sha(args.out/'inputs.json')}),flush=True)


def measure(args):
    # Must precede Torch, vLLM and CUDA imports; the collector includes startup.
    from experiments.native_operator_resources import NativeMemoryCollector
    collector=NativeMemoryCollector(args.resource_library)
    from experiments import bench_native_operator as bench
    from prismaquant.native_execution_binding import freeze_execution_panel
    from safetensors.torch import load_file
    inputs=json.loads((args.out/'inputs.json').read_text());origin=json.loads((args.out/'origin.json').read_text())
    for name,digest in origin['artifacts'].items():assert sha(args.out/name)==digest
    tracepath=args.out/'memory.json'; finished=False
    try:
        with bench.native_runtime_context():
            tensors=load_file(str(args.out/'tensors.safetensors'),device='cuda')
            prepared=bench.prepare_native_operator((args.out/'weight.tessera').read_bytes(),
                json.loads((args.out/'wire-record.json').read_text()),tensors['source_weight'],tensors['rendered_weight'],
                unit=inputs['unit'],format_name=inputs['format'],runtime_image=IMAGE,
                input_global_scale=inputs['activation']['input_global_scale'])
            prepared['runtime']['resource_collector']={'library_sha256':collector.library_sha256,
                'analysis_source_sha256':sha(Path(bench.__file__).with_name('native_operator_resources.py'))}
            preflight={'schema':'tessera.native_dense_preflight.v1','status':'untimed_preparation',
                'operator':prepared['operator'],'runtime':prepared['runtime'],
                'runtime_sha256':bench.identity_sha256(prepared['runtime']),
                'native_tensors_sha256':bench.identity_sha256(prepared['operator']['native_tensors']),
                'scheme_sha256':prepared['operator']['scheme_sha256']}
            panel=freeze_execution_panel(inputs,preflight,source_sha256=origin['source_sha256'])
            dump(args.out/'preflight.json',preflight);dump(args.out/'execution-panel.json',panel)
            phases={p:{k:tensors[f'{p}.{k}'] for k in ('input','reference_qdq','reference_output')} for p in bench.PHASES}
            receipt=bench.measure_prepared_operator(prepared,panel,phases,warmup_iterations=8,iterations=32,resource_collector=collector)
            trace=collector.finish(tracepath);finished=True;bench.attach_resource_trace(receipt,trace)
            if receipt['status']=='resources_observed' and receipt['resources']['status']=='complete_operator_bound':
                bench.time_after_resource_collection(prepared,panel,phases,receipt,collector=collector,warmup_iterations=8,iterations=32)
            dump(args.out/'execution-receipt.json',receipt)
            print(json.dumps({'status':receipt['status'],'resources':receipt['resources']['status'],
                'receipt_path':str(args.out/'execution-receipt.json'),'receipt_sha256':sha(args.out/'execution-receipt.json'),
                'memory_sha256':sha(tracepath),'panel_sha256':sha(args.out/'execution-panel.json')}),flush=True)
            if receipt['status']!='timing_admissible':raise RuntimeError('native timing/resource gate refused')
    finally:
        if not finished:collector.finish(tracepath)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','measure']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--activation-quantizers',type=Path)
    p.add_argument('--format',default='TESSERA_E4M3_K1_R1024');p.add_argument('--resource-library',type=Path)
    args=p.parse_args();globals()[args.phase](args)
