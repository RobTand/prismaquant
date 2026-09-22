"""Actual PB staging/lease to private-local handoff regression, CPU only."""
import argparse,hashlib,json,os,struct,tempfile,sys
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def build(root):
    import torch
    from safetensors.torch import save_file
    root.mkdir(parents=True,exist_ok=False)
    tensor=torch.arange(4096,dtype=torch.bfloat16).reshape(64,64)
    save_file({'weight':tensor},str(root/'source.safetensors'))
    torch.save(tensor,root/'render.pt');(root/'wire.bin').write_bytes(b'qualified wire bytes'*100)
    files={n:{'path':str(root/n),'sha256':digest(root/n),'bytes':(root/n).stat().st_size}
           for n in ('source.safetensors','render.pt','wire.bin')}
    dump(root/'files.json',files)
    with (root/'source.safetensors').open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
    start,end=header['weight']['data_offsets']
    entries=[{'path':str(root/'source.safetensors'),'offset':8+size+start,'bytes':end-start,'sha256':None}]
    entries += [{'path':str(root/n),'offset':0,'bytes':(root/n).stat().st_size,'sha256':digest(root/n)} for n in ('render.pt','wire.bin','files.json')]
    total=sum(e['bytes'] for e in entries)
    dump(root/'data-manifest.json',{'schema':'prismaquant.prismabuild.data_manifest.v1','mount_prefix':'/mnt/shared','produced_by':{'program':'experiments/qualify_native_local_handoff.py','source_sha256':digest(__file__)},
        'entries':entries,'entry_count':len(entries),'total_bytes':total,'annotations':{'phases':[{'name':'head','bytes':total,'cumulative_bytes':total}]}})
    print(json.dumps({'status':'fixture_written','bytes':total,'manifest_sha256':digest(root/'data-manifest.json')}))

def read(root,tessera):
    import torch
    from experiments.native_local_handoff import bind_strict_inputs,strict_read,LocalHandoff
    from prismaquant.staged_lease import sdk_submodule
    os.environ.update(sdk_submodule('core')._reader_identity_environment(os.environ['PRISMABUILD_ACTION_KEY']))
    bind_strict_inputs()
    from prismaquant.prismabuild_progress import report
    assert report('head',0,unit='handoff')
    files=json.loads(strict_read({'path':str(root/'files.json'),'sha256':digest(root/'files.json')}))
    from prismaquant.layer_streaming import _source_safe_open,_await_layer_readset
    from prismaquant.production_weight_cache import ProductionWeightCache
    source=files['source.safetensors']['path'];_await_layer_readset({source:[('weight','weight')]})
    with _source_safe_open(source,framework='pt',device='cpu') as handle:weight=handle.get_tensor('weight')
    cache=ProductionWeightCache(weights={('unit','BF16'):files['render.pt']['path']},levers={})
    cache.enable_lru(1<<20);cache.enable_file_load_receipts(max_file_bytes=1<<20)
    rendered=cache.get('unit','BF16');receipt=cache.file_load_receipt(('unit','BF16'),rendered)
    assert receipt['serving_tier'] in ('ram','ssd','stage'),receipt
    assert torch.equal(weight,rendered)
    wire=strict_read(files['wire.bin']);assert wire==b'qualified wire bytes'*100
    with tempfile.TemporaryDirectory(prefix='native-handoff-proof-',dir=Path.cwd().parent) as temporary:
        owner=LocalHandoff(Path(temporary)/'payload',max_bytes=1<<20)
        source_ref=owner.write('source.pt',tensor=weight);wire_ref=owner.write('wire.bin',raw=wire)
        bundle=owner.seal(os.environ['PRISMABUILD_ACTION_KEY'])
        import importlib.util
        spec=importlib.util.spec_from_file_location('independent_native_reader',Path(tessera)/'experiments/native_local_inputs.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        local=module.LocalNativeInputs(bundle,os.environ['PRISMABUILD_ACTION_KEY'])
        try:
            assert local.read(wire_ref['path'],wire_ref['sha256'],1<<20)==wire
            import io
            assert torch.equal(torch.load(io.BytesIO(local.read(source_ref['path'],source_ref['sha256'],1<<20)),weights_only=True),weight)
        finally:local.close()
    result={'status':'passed','source_tensor_bytes':weight.numel()*weight.element_size(),
        'pwc_serving_tier':receipt['serving_tier'],'pwc_sha256':receipt['sha256'],
        'wire_sha256':hashlib.sha256(wire).hexdigest(),'strict_allowed_tiers':['ram','ssd'],
        'action_key':os.environ['PRISMABUILD_ACTION_KEY'],'independent_tessera_reader_sha256':digest(Path(tessera)/'experiments/native_local_inputs.py')}
    dump(root/'proof.json',result);assert report('head',1,unit='handoff');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('build','read'));p.add_argument('--root',type=Path,required=True);p.add_argument('--tessera');a=p.parse_args()
    if a.mode=='build':build(a.root)
    else:read(a.root,a.tessera)
