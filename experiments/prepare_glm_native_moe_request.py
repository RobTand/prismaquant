"""Seal exact whole-owner reads; PB owns every stage, placement and retry."""
import argparse,hashlib,json,struct
from pathlib import Path

def binding(path):
    path=Path(path);raw=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}

def write(path,value):
    with Path(path).open('x') as handle:json.dump(value,handle,sort_keys=True,indent=2);handle.write('\n')
    return binding(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--metadata-sha256',required=True)
    p.add_argument('--boundary',required=True);p.add_argument('--boundary-sha256',required=True)
    p.add_argument('--format',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    metadata_bound=binding(args.metadata)
    if metadata_bound['sha256']!=args.metadata_sha256:raise ValueError('metadata changed')
    metadata=json.loads(Path(args.metadata).read_text())
    base=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913')
    native=base/'native-pact-acquisition-20260922'
    spec={'metadata':metadata_bound,'boundary':{'path':args.boundary,'sha256':args.boundary_sha256},
        'format':args.format,'canonical_capture':binding('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/workspace/calibration-cache/capture_manifest.json'),
        'scientific_plan':binding(base/'allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json'),
        'serving_config':binding(native/'glm-whole-owner-serving-config.json')}
    args.out.mkdir(parents=True,exist_ok=False)
    entries=[];seen=set();phases=[];total=0
    def add(path,offset,size,digest=None):
        nonlocal total
        key=(str(path),offset,size)
        if key in seen:return
        seen.add(key);entries.append({'path':str(path),'offset':offset,'bytes':size,'sha256':digest});total+=size
    def full(bound):add(bound['path'],0,Path(bound['path']).stat().st_size,bound.get('sha256'))
    for key in ('metadata','boundary','canonical_capture','scientific_plan','serving_config'):full(spec[key])
    full(metadata['inputs']['production_cache'])
    if args.format.startswith('TESSERA_E2M1'):
        spec['activation_policy']=binding(base/'t4-reuse-20260922/proposed-served-activation-policy.json')
        policy=json.loads(Path(spec['activation_policy']['path']).read_text());full(spec['activation_policy'])
        for key in ('original_prepared','original_cache','census'):full(policy[key])
    spec_bound=write(args.out/'spec.json',spec);full(spec_bound)
    phases.append({'name':'head','bytes':total,'cumulative_bytes':total})
    rows={r['qname']:r for r in metadata['rows']};unit=f"model.language_model.layers.{metadata['layer']}.mlp.experts"
    names=[f'{unit}.{expert}.{projection}' for expert in range(288) for projection in ('gate_proj','up_proj','down_proj')]
    headers={};source=metadata['source_model_identity'];model=Path(source['source'])
    for start in range(0,len(names),32):
        before=total
        for name in names[start:start+32]:
            entry=rows[name]['formats'][args.format];tensor=entry['source_projection']['source_tensor']
            path=model/source['checkpoint_weight_map'][tensor]
            if str(path) not in headers:
                with path.open('rb') as handle:
                    size=struct.unpack('<Q',handle.read(8))[0]
                    if size>100_000_000:raise ValueError('source header exceeds bound')
                    headers[str(path)]=(8+size,json.loads(handle.read(size)))
            offset,header=headers[str(path)];begin,end=header[tensor]['data_offsets']
            add(path,offset+begin,end-begin)
            add(entry['wire'],0,entry['record']['blob_bytes'],entry['record']['blob_sha256'])
            add(entry['render'],0,Path(entry['render']).stat().st_size,entry['render_file_sha256'])
        phases.append({'name':f'members-{start//32:03d}','bytes':total-before,'cumulative_bytes':total})
    manifest={'schema':'prismaquant.prismabuild.data_manifest.v1',
        'produced_by':{'program':'experiments/prepare_glm_native_moe_request.py','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'mount_prefix':'/mnt/shared','entries':entries,'entry_count':len(entries),'total_bytes':total,
        'annotations':{'phases':phases,'scope':'one complete GLM288-expert owner; exact original source spans, PWC renders and wires; no scientific subdivision'}}
    result=write(args.out/'data-manifest.json',manifest)
    print(json.dumps({'spec':spec_bound,'data_manifest':result,'phases':[v['name'] for v in phases],'bytes':total}))

if __name__=='__main__':main()
