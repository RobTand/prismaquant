"""Declare one original sample's BF16 prefix reads with existing source-range tooling."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from glm_arc_prewarm import Campaign
from glm_data_manifests import _source_tensor_extents,check_manifest,SCHEMA
ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/native-pact-acquisition-20260922/routing-prefix-layer3-phased-02')
PANEL=ROOT.parents[1]/'allocation/joint-panel'
PLAN=PANEL/'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json'
PREP=PANEL/'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def write(p,value):
 assert not p.exists();raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();p.write_bytes(raw);return {'path':str(p),'sha256':sha(raw)}
def main():
 ROOT.mkdir(parents=True,exist_ok=True);plan=json.loads(PLAN.read_bytes());prepared=json.loads(PREP.read_bytes());assert binding(PLAN)['sha256']==prepared['plan_sha256']
 spec=write(ROOT/'spec.json',{'schema':'prismaquant.glm_routing_capture_spec.v1','mode':'fresh_source_prefix','read_phases':['head','layer-0','layer-1','layer-2','layer-3'],'sample':0,'layer':3,'plan':binding(PLAN),'prepared':binding(PREP),'scope':'one original sequence, coordinates sample0 token0..511 in unchanged full512 calibration; no quality-panel reduction'})
 campaign=Campaign(str(Path(plan['inputs']['campaign_plan']['path']).parent),plan={'model':plan['model'],'rows':[]})
 groups=[('head',['model.language_model.embed_tokens.','model.language_model.norm.','model.language_model.rotary_emb.','lm_head.'])]+[(f'layer-{i}',[f'model.language_model.layers.{i}.']) for i in range(4)]
 metadata=[PLAN,PREP,Path(spec['path']),Path(plan['calibration_input']['path']),Path(plan['inputs']['census']['path']),Path(plan['source_identity_cache']['path'])]
 for path in Path(plan['model']).iterdir():
  if path.is_file() and path.suffix in ('.json','.model','.txt','.jinja','.py'):metadata.append(path)
 entries=[];phases=[];total=0
 for name,prefixes in groups:
  spans={}
  # Exact tensor payload intervals keep headers in head without a repeated
  # rounded offset0 that would promote a future whole layer into head.
  for tensor,shard in campaign.weight_map.items():
   if not tensor.startswith(tuple(prefixes)):continue
   header=campaign._header(shard);lo,hi=header[tensor]['data_offsets'];start=header['__data_start__']
   path=str(Path(campaign.model_dir)/shard);spans.setdefault(path,[]).append((start+lo,start+hi))
  if name=='head':
   for shard in sorted(set(campaign.weight_map.values())):
    path=str(Path(campaign.model_dir)/shard);count=campaign._header(shard)['__data_start__'];spans.setdefault(path,[]).append((0,count))
   for path in metadata:spans.setdefault(str(path),[]).append((0,path.stat().st_size))
  phase_entries=[]
  for path,ranges in sorted(spans.items()):
   merged=[]
   for lo,hi in sorted(ranges):
    if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
    else:merged.append([lo,hi])
   phase_entries.extend({'path':path,'offset':lo,'bytes':hi-lo,'sha256':None} for lo,hi in merged)
  count=sum(e['bytes'] for e in phase_entries);total+=count;entries.extend(phase_entries);phases.append({'name':name,'bytes':count,'cumulative_bytes':total})
 manifest={'schema':SCHEMA,'produced_by':{'program':'tools/prepare_glm_routing_prefix_phased.py','source_sha256':sha(Path(__file__).read_bytes())},'mount_prefix':'/mnt/shared','entries':entries,'entry_count':len(entries),'total_bytes':total,'annotations':{'phases':phases,'scope':'unchanged original source head and layers0..3, semantic streamed source phases; no speculative future layers'}}
 check_manifest(manifest);data=write(ROOT/'data-manifest.json',manifest)
 print(json.dumps({'spec':spec,'data_manifest':data,'payload_bytes':total,'entries':len(entries)}),flush=True)
if __name__=='__main__':main()
