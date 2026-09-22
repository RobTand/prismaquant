"""Recover failed logical members; PB retains all partition and placement ownership."""
import argparse,copy,hashlib,json,os
from pathlib import Path
ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922')
PARENT='7ba9f2e4120e8067e7aa6cc9a7fa16e2c87c766b376ad5817dd392f845d51d82'
QUEUE=Path('/mnt/shared/prismabuild-fleet/pb-queue')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--qualifier-checkout',default='/home/rob/tmp/pq-t4-readiness-20260922');args=p.parse_args();out=Path(args.out);assert not out.exists()
 dep=Path('/mnt/shared/prismabuild-fleet/cas/decompositions')/PARENT[:2]/PARENT
 publication=json.loads((dep/'publication.json').read_bytes());plan=json.loads((dep/'plan.json').read_bytes());keys=publication['child_action_keys'];assert len(keys)==len(plan['partitions'])
 failed={e.name[:-5] for e in os.scandir(QUEUE/'failed') if e.name.endswith('.json')}
 selected={q for key,part in zip(keys,plan['partitions']) if key in failed for q in part};assert selected
 raw=(ROOT/'logical-qualification-37.json').read_bytes();assert sha(raw)=='1902e677b456b0a6e40da2fe45fe0bd38fccee5226e238359d4f9b060a963bf5';request=json.loads(raw);del raw
 tasks=[];reused=0
 for task in request['roster']['tasks']:
  if task['id'] not in selected:continue
  payload=task['payload'];result=Path(payload['output'])
  if result.exists():
   raw=result.read_bytes();value=json.loads(raw);cell=payload['cell'];assert value['cell_sha256']==sha(json.dumps(cell,sort_keys=True,separators=(',',':')).encode())
   if 'existing_result_sha256' in payload:assert sha(raw)==payload['existing_result_sha256']
   else:assert value['verified_cell_sha256']==sha(json.dumps(value['verified_cell'],sort_keys=True,separators=(',',':')).encode())
   payload['existing_result_sha256']=sha(raw);payload['reads']=[];task['estimated_seconds']=.001;reused+=1
  tasks.append(task)
 assert len(tasks)==len(selected)
 request['roster']['tasks']=tasks;request['common']['cwd']=args.qualifier_checkout;request['common']['timeout_s']=600
 out.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(request,sort_keys=True,separators=(',',':'))+'\n').encode()
 with out.open('xb') as f:f.write(raw)
 proof={'schema':'prismaquant.t4_failed_logical_members.v1','parent':PARENT,'publication':{'path':str(dep/'publication.json'),'sha256':sha((dep/'publication.json').read_bytes())},'failed_children':[k for k in keys if k in failed],'members':len(tasks),'reused_results':reused,'request':{'path':str(out),'sha256':sha(raw)}}
 out.with_suffix('.recovery.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n');print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
