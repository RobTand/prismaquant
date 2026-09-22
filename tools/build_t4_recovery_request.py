"""Recover failed logical members; PB retains all partition and placement ownership."""
import argparse,copy,hashlib,json,os
from pathlib import Path
QUEUE=Path('/mnt/shared/prismabuild-fleet/pb-queue')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--all-members',action='store_true');p.add_argument('--qualifier-checkout',required=True);p.add_argument('--parent',required=True,help='PB decomposition parent key of the logical request');p.add_argument('--request',required=True,help='logical request built by build_t4_logical_request.py');p.add_argument('--request-sha256',required=True);args=p.parse_args();PARENT=args.parent;out=Path(args.out);assert not out.exists()
 dep=Path('/mnt/shared/prismabuild-fleet/cas/decompositions')/PARENT[:2]/PARENT
 publication=json.loads((dep/'publication.json').read_bytes());plan=json.loads((dep/'plan.json').read_bytes());keys=publication['child_action_keys'];assert len(keys)==len(plan['partitions'])
 failed={e.name[:-5] for e in os.scandir(QUEUE/'failed') if e.name.endswith('.json')}
 selected={q for key,part in zip(keys,plan['partitions']) if args.all_members or key in failed for q in part};assert selected
 raw=Path(args.request).read_bytes();assert sha(raw)==args.request_sha256;request=json.loads(raw);del raw
 tasks=[];reused=0
 for task in request['roster']['tasks']:
  if task['id'] not in selected:continue
  payload=task['payload'];result=Path(payload['output'])
  if result.exists():
   raw=result.read_bytes();value=json.loads(raw);cell=payload['cell'];assert value['cell_sha256']==sha(json.dumps(cell,sort_keys=True,separators=(',',':')).encode())
   if 'existing_result_sha256' in payload:assert sha(raw)==payload['existing_result_sha256']
   else:assert value['verified_cell_sha256']==sha(json.dumps(value['verified_cell'],sort_keys=True,separators=(',',':')).encode())
   assert value['qname']==cell['qname'] and value['format']==cell['format']
   for kind in ('wire','render'):
    stat=Path(cell[kind]).stat();assert cell[kind+'_stat']=={'inode':stat.st_ino,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns}
   payload['existing_result_sha256']=sha(raw);payload['reads']=[];task['estimated_seconds']=.001;reused+=1
  tasks.append(task)
 assert len(tasks)==len(selected)
 request['roster']['tasks']=tasks;request['common']['cwd']=args.qualifier_checkout;request['common']['timeout_s']=600
 out.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(request,sort_keys=True,separators=(',',':'))+'\n').encode()
 with out.open('xb') as f:f.write(raw)
 proof={'schema':'prismaquant.t4_recovery_logical_members.v1','scope':'all_original_members' if args.all_members else 'failed_original_members','parent':PARENT,'publication':{'path':str(dep/'publication.json'),'sha256':sha((dep/'publication.json').read_bytes())},'failed_children':[k for k in keys if k in failed],'members':len(tasks),'reused_results':reused,'request':{'path':str(out),'sha256':sha(raw)}}
 out.with_suffix('.recovery.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n');print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
