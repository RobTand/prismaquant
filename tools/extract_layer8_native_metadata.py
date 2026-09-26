"""Bounded whole-operator inputs: metadata only, no wire/render/H payload reads."""
import argparse,json,pickle,os
from pathlib import Path
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_aura import activation_identity
from prismaquant import format_registry as fr
from prismaquant.digests import bytes_sha256hex
ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922')
BASE=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911')
PREP=ROOT.parent/'allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json'
FORMATS=('TESSERA_E4M3_K1_R1024','TESSERA_BF16_K1_R1024','TESSERA_E2M1_K2_R896')
sha = bytes_sha256hex
def binding(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def unit(root,q):
 path=root/'cost.anchors.json.parts/units'/(sha(q.encode())+'.pkl');raw=path.read_bytes();env=pickle.loads(raw);assert sha(env['payload'])==env['payload_sha256'];return pickle.loads(env['payload']),{'path':str(path),'sha256':sha(raw),'payload_sha256':env['payload_sha256']}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--layer',type=int,default=8);args=parser.parse_args();layer=args.layer;assert 3<=layer<=44
 prepared=json.loads(PREP.read_bytes());pwc=prepared['production_cache'];raw=Path(pwc['path']).read_bytes();assert sha(raw)==pwc['sha256'];cache=pickle.loads(raw);del raw
 oldroot=BASE/'extension-r1024-02/workspace/merged';a4root=BASE/'extension-e2m1-01/workspace/merged-c92826fa4';planpath=BASE/'extension-e2m1-01/workspace/plan.json';plan=json.loads(planpath.read_bytes());owners={q:Path(row['dir']) for row in plan['rows'] for q in row['members']}
 names=sorted(q for q in prepared['formats_by_qname'] if q.startswith(f'model.language_model.layers.{layer}.mlp.experts.'));assert len(names)==864
 rows=[];groups={'w13':{'activation_max_abs':set(),'a4_input_global_scale':set()},'w2':{'activation_max_abs':set(),'a4_input_global_scale':set()}}
 for q in names:
  old,oldbound=unit(oldroot,q);a4,a4bound=unit(a4root,q);byfmt={r['format_name']:r for r in a4['anchors']};formats={}
  for fmt in FORMATS:
   state=a4 if fmt==FORMATS[2] else old;record=state['wire_records'][fmt];row=next(r for r in state['anchors'] if r['format_name']==fmt);wire=(a4root if fmt==FORMATS[2] else oldroot)/'cache/wire'/record['file'];assert wire.stat().st_size==record['blob_bytes']
   if fmt==FORMATS[2]:
    render=owners[q]/'cache'/(q.replace('.','_')+'__'+fmt+'.pt');qualified=ROOT/'qualified'/(sha(q.encode())+'.json');proof=None if not qualified.exists() else json.loads(qualified.read_bytes())['verified_cell']
   else:
    assert fmt in prepared['formats_by_qname'][q];render=Path(cache.weights[q,fmt]);proof=cache.metadata['verified_cells'][q,fmt];assert proof['wire_sha256']==record['blob_sha256'];assert proof['encoding_identity_sha256']==canonical_json_sha256(record['identity'],where='layer8 originalwire')
   assert render.is_file();act=activation_identity(fr.get_format(fmt),cache.activation_max_abs,q)
   if fmt==FORMATS[2]:
    assert act['input_global_scale']==row['input_global_scale'];group=groups['w2' if q.endswith('.down_proj') else 'w13'];group['activation_max_abs'].add(cache.activation_max_abs[q]);group['a4_input_global_scale'].add(act['input_global_scale'])
   formats[fmt]={'record':record,'wire':str(wire),'render':str(render),'render_file_sha256':None if proof is None else proof['render_file_sha256'],'rendered_weight':None if proof is None else proof['rendered_weight'],'qualification_status':'pending_current_render_qualification' if proof is None else 'qualified','source_projection':record['identity']['projection'],'activation':act,'anchor':row}
  rows.append({'qname':q,'source_weight':cache.metadata['verified_cells'][q,FORMATS[0]]['source_weight'],'source_unit_envelopes':{'original':oldbound,'a4':a4bound},'formats':formats})
 for group in groups.values():
  for key,value in list(group.items()):group[key]=sorted(value)
  group['uniform_group_maximum']=len(group['activation_max_abs'])==1;group['uniform_a4_input_scale']=len(group['a4_input_global_scale'])==1
 result={'schema':'prismaquant.glm_native_operator_inputs.v1','layer':layer,'experts':288,'projections_per_expert':3,'qnames':864,'formats':list(FORMATS),'inputs':{'prepared':binding(PREP),'production_cache':pwc,'a4_plan':binding(planpath),'adopted_catalog':{'path':str(ROOT/'adopted-catalog.json'),'sha256':'71238bdab85bdccabed921ce94b03459d5aba3607021b85007bf41f3e376fbc8'}},'source_model_identity':prepared['source_model_identity'],'calibration_input':prepared['calibration_input'],'executed_group_static_scales':groups,'rows':rows,'scope':'metadata census only; no wire/render/H payload reads; pending A4 render hashes remain null'}
 out=ROOT/f'layer{layer}-native-metadata.json';assert not out.exists();raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode();out.write_bytes(raw);print(json.dumps({'path':str(out),'sha256':sha(raw),'qnames':len(rows),'cells':len(rows)*len(FORMATS),'group_static_scales':groups}),flush=True)
if __name__=='__main__':main()
