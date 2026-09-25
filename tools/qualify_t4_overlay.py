"""PB task-batch adapter: qualify retained renders, without encoding or rewriting them."""
import argparse,functools,hashlib,io,json,os,time
from pathlib import Path
import torch
from tessera.cached_unit import verify_cached_unit
from prismaquant.tessera_joint_aura import _decode_wire, _resolve_render_origin, RENDER_COMPARISON_BY_ORIGIN, _read_verified_wire_blob, _drive_ordered_walk, _decoder_identity
from prismaquant.production_weight_cache import _cb_cache_tensor_identity, ProductionWeightCache
from prismaquant.io_engine import read_file
from prismaquant.residency_map import residency_resolver
from prismaquant.staged_tier_policy import policy_is_active, refuse_pool_bulk_read

@functools.lru_cache(maxsize=1)
def decoder_provenance():
 from prismaquant.tessera_campaign import _checkpoint_identity_api
 import importlib.metadata
 distribution=importlib.metadata.distribution('tessera-quant');direct_url=distribution.read_text('direct_url.json')
 return {'schema':'prismaquant.catalog_qualification_decoder.v1','decoder':_decoder_identity(None),'distribution':{'name':distribution.metadata['Name'],'version':distribution.version,'direct_url':None if direct_url is None else json.loads(direct_url),'direct_url_sha256':None if direct_url is None else hashlib.sha256(direct_url.encode()).hexdigest()},'installed_encoder_source_sha256':_checkpoint_identity_api().encoder_source_sha256(),'torch':torch.__version__,'cuda':torch.version.cuda,'scope':'actual installed decoder used for this cell; no encoding performed'}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def stamp(path):
 s=path.stat();return dict(inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def read_cell(cell):
 started=time.monotonic();wire=Path(cell['wire']);render=Path(cell['render'])
 assert stamp(wire)==cell['wire_stat'];assert stamp(render)==cell['render_stat'];origin=_resolve_render_origin(render,wire=wire,record=cell['record'],name=cell['qname'],fmt=cell['format'],shape=cell['source_weight']['shape'],reader=None);assert origin==cell['render_origin'];assert RENDER_COMPARISON_BY_ORIGIN[origin]==cell['render_comparison']
 if policy_is_active():
  from prismaquant.residency_shard_reader import await_staged_spans, staged_range_wait_s
  from prismaquant.staged_lease import stage_cover_is_published
  from prismaquant.residency_map import RANGE_HIT
  resolver=residency_resolver()
  if resolver is None:raise refuse_pool_bulk_read(str(wire),'readset-not-staged')
  wanted=[(cell[k],0,cell[k+'_stat']['bytes'],cell[k+'_stat']['bytes']) for k in ('wire','render')]
  verdict=await_staged_spans(resolver,wanted,deadline=time.monotonic()+staged_range_wait_s(),published=stage_cover_is_published)
  if verdict!=RANGE_HIT:raise refuse_pool_bulk_read(str(wire),'qualification-readiness-'+verdict)
 wire_start=time.monotonic();blob,_wire_digest=_read_verified_wire_blob(cell);verify_cached_unit(blob,cell['record'],cell['record']['identity']);assert stamp(wire)==cell['wire_stat']
 wire_seconds=time.monotonic()-wire_start;render_start=time.monotonic();resolver=residency_resolver();staged=None if resolver is None else resolver.staged_read(render)
 if policy_is_active() and staged is None:raise refuse_pool_bulk_read(str(render),'new-candidate-render-not-staged')
 # First qualification has no prior digest. Existing bounded reader checks
 # PB copy-time digest and pinned lease, then this leg establishes decoder equality.
 raw,receipt,_signature=read_file(render,cell['render_stat']['bytes'],staged=staged);tensor,_guard=ProductionWeightCache._decode_file_tensor(ProductionWeightCache.__new__(ProductionWeightCache),None,raw,receipt,staged is not None);del raw;observed=(receipt,);file_sha=receipt['sha256']
 if resolver is not None:
  if staged is None:resolver.record_pool_read(render,observed[0]['bytes'])
  elif observed[0].get('serving_tier')=='ram':resolver.record_ram_read(render,observed[0]['bytes'])
  else:resolver.record_stage_read(render,observed[0]['bytes'])
 assert tensor.dtype==torch.bfloat16 and list(tensor.shape)==cell['source_weight']['shape']
 assert bool(torch.isfinite(tensor).all());identity=_cb_cache_tensor_identity(tensor)
 render_seconds=time.monotonic()-render_start
 return (cell,started,blob,tensor,identity,file_sha,wire_seconds,render_seconds)
def finish_cell(loaded):
 cell,started,blob,tensor,identity,file_sha,wire_seconds,render_seconds=loaded;render=Path(cell['render']);wire=Path(cell['wire']);decode_start=time.monotonic();decoded=_decode_wire(blob,reader=None,device='cuda').to(torch.bfloat16).cpu()
 decode_seconds=time.monotonic()-decode_start;assert torch.equal(decoded,tensor),cell['qname'];assert stamp(render)==cell['render_stat'];assert stamp(wire)==cell['wire_stat']
 return {**{k:cell[k] for k in ('source_weight','activation','encoding_identity_sha256','render_origin','render_comparison','catalog_source_adoption','adopted_source_hessian')},'rendered_weight':identity,'render_file_sha256':file_sha,'wire_sha256':cell['record']['blob_sha256'],'render_stat':cell['render_stat'],'wire_stat':cell['wire_stat'],'qualification_decoder':decoder_provenance(),'qualification_seconds':time.monotonic()-started,'phase_seconds':{'wire_read_verify':wire_seconds,'render_read_hash_tensor':render_seconds,'decode_cpu_compare_transfer':decode_seconds}}
def qualify(cell):return finish_cell(read_cell(cell))
def main():
 p=argparse.ArgumentParser();p.add_argument('--pb-task-batch');p.add_argument('--pilot-batch');p.add_argument('--allowed-tiers');args=p.parse_args();
 if args.allowed_tiers:
  from prismaquant.staged_tier_policy import activate_staged_tier_policy
  activate_staged_tier_policy(args.allowed_tiers)
 assert bool(args.pb_task_batch)!=bool(args.pilot_batch);batch=json.loads(Path(args.pb_task_batch or args.pilot_batch).read_text());assert batch['schema']==('prismabuild.task_batch.v1' if args.pb_task_batch else 'prismaquant.t4_qualification_pilot.v1');results=[]
 if args.allowed_tiers and any(t['payload'].get('reads') for t in batch['tasks']):
  from prismaquant.staged_lease import resolve_sealed_readset, load_sealed_readset
  from prismaquant.residency_map import bind_residency_manifest
  _cas_root,manifest_digest,_manifest_bytes=resolve_sealed_readset();readset=load_sealed_readset(manifest_digest);bind_residency_manifest(manifest_digest)
  print(json.dumps({'sealed_readset_sha256':manifest_digest,'declared_files':len(readset)}),flush=True)
 def read_task(task):
  cell=task['payload']['cell'];out=Path(task['payload']['output'])
  if out.exists():
   raw=out.read_bytes();expected=task['payload'].get('existing_result_sha256');assert expected is None or digest(raw)==expected;value=json.loads(raw);assert value['cell_sha256']==digest(json.dumps(cell,sort_keys=True,separators=(',',':')).encode());assert stamp(Path(cell['render']))==cell['render_stat'];assert stamp(Path(cell['wire']))==cell['wire_stat']
   if expected is None:assert value.get('verified_cell_sha256')==digest(json.dumps(value['verified_cell'],sort_keys=True,separators=(',',':')).encode())
   return ('existing',raw,value)
  if task['payload'].get('existing_result_sha256'):raise RuntimeError('previously qualified result is missing; sealed task declares no payload reads')
  return ('new',read_cell(cell))
 def commit_task(task,loaded):
  cell=task['payload']['cell'];out=Path(task['payload']['output']);out.parent.mkdir(parents=True,exist_ok=True)
  if loaded[0]=='existing':raw,value=loaded[1:]
  else:
   value={'qname':cell['qname'],'format':cell['format'],'cell_sha256':digest(json.dumps(cell,sort_keys=True,separators=(',',':')).encode()),'verified_cell':finish_cell(loaded[1])};value['verified_cell_sha256']=digest(json.dumps(value['verified_cell'],sort_keys=True,separators=(',',':')).encode());raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();tmp=out.with_suffix('.tmp')
   with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
   os.replace(tmp,out)
  results.append({'task_id':task['id'],'output_id':task['output_id'],'value_sha256':digest(raw)})
  print(json.dumps({'qname':cell['qname'],'seconds':value['verified_cell']['qualification_seconds'],'committed':len(results)}),flush=True)
 workers=int(os.environ.get('T4_QUALIFY_READ_WORKERS','1'));assert 0<workers<=len(os.sched_getaffinity(0))
 _drive_ordered_walk(batch['tasks'],read_task,commit_task,workers=workers)
 manifest=({k:batch[k] for k in ('parent_key','plan_key','child_ordinal')} if args.pb_task_batch else {});manifest.update(schema=('prismabuild.child_result_manifest.v1' if args.pb_task_batch else 'prismaquant.t4_qualification_pilot_result.v1'),results=results);Path(batch['result_manifest_path']).write_text(json.dumps(manifest,sort_keys=True)+'\n')
 from prismaquant.residency_map import residency_report
 print(json.dumps({'residency_report':residency_report()},sort_keys=True),flush=True)
if __name__=='__main__':main()
