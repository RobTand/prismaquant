"""Describe immutable logical cells; PrismaBuild alone partitions them."""
import hashlib,json
from pathlib import Path
ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 raw=(ROOT/'adopted-catalog.json').read_bytes();assert sha(raw)=='71238bdab85bdccabed921ce94b03459d5aba3607021b85007bf41f3e376fbc8';catalog=json.loads(raw)
 pilot=json.loads((ROOT/'pilot24-results.json').read_bytes());existing={r['task_id']:r['value_sha256'] for r in pilot['results']};tasks=[]
 for cell in catalog['cells']:
  q=cell['qname'];payload={'cell':cell,'output':str(ROOT/'qualified'/(sha(q.encode())+'.json')),'reads':[]}
  if q in existing:
   assert sha(Path(payload['output']).read_bytes())==existing[q];payload['existing_result_sha256']=existing[q]
  else:
   for kind in ('wire','render'):
    payload['reads'].append({'path':str(Path(cell[kind]).relative_to('/mnt/shared')),'offset':0,'bytes':cell[kind+'_stat']['bytes'],'sha256':cell['record']['blob_sha256'] if kind=='wire' else None})
  tasks.append({'id':q,'payload':payload,'residency_key':'retained-glm-a4','estimated_seconds':0.001 if q in existing else 0.2,'estimate_evidence':'PB f0fc0fe4f5e09ed29cb3f1225c60f09966779fa4a3cfa034c368975268e105c0; warm median 0.1644s, 0.2s planning allowance; reused pilot results metadata-only' ,'output_id':q})
 request={'schema':'prismabuild.logical_request.v1','common':{'argv':['/home/rob/venvs/pq846-pb461728e4/bin/python','tools/qualify_t4_overlay.py','--pb-task-batch','{pb.task_batch}','--allowed-tiers','ram,ssd'],'cwd':'/home/rob/tmp/pq-t4-reuse-audit-20260922','demand':{'cpu':4,'mem_gb':8,'gpu':1},'gpu_memory_gb':2,'data_manifest':None,'env':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':'.','T4_QUALIFY_READ_WORKERS':'2'},'tags':['gb10'],'timeout_s':600},'roster':{'schema':'prismabuild.logical_task_roster.v1','tasks':tasks},'batch_policy':{'schema':'prismabuild.roster_batch_policy.v1','residencies':[{'key':'retained-glm-a4','setup_seconds':9,'setup_evidence':'PB pilot24 wall15s minus ~10.5s percell work plus cold firstcell4.4s; conservative startup allowance, not throughput measurement'}],'max_setup_fraction':0.1,'max_estimated_wall_seconds':120},'task_data_manifest':{'schema':'prismabuild.task_data_manifest.v1','payload_field':'reads','mount_prefix':'/mnt/shared','residency_tier':None,'residency_ram':'off','mover_readers':2,'mover_mem_gb':1}}
 out=ROOT/'logical-qualification.json';assert not out.exists();raw=(json.dumps(request,sort_keys=True,separators=(',',':'))+'\n').encode();out.write_bytes(raw);print(json.dumps({'path':str(out),'sha256':sha(raw),'tasks':len(tasks),'reused':len(existing),'status':'requires_new_PB_client_task_data_manifest_support'}))
if __name__=='__main__':main()
