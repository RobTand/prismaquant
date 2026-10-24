@functools.lru_cache(maxsize=1)
def decoder_provenance():
 from prismaquant.tessera_campaign_wire import require_checkpoint_identity_api
 import importlib.metadata
 distribution=importlib.metadata.distribution('tessera-quant');direct_url=distribution.read_text('direct_url.json')
 return {'schema':'prismaquant.catalog_qualification_decoder.v1','decoder':_decoder_identity(None),'distribution':{'name':distribution.metadata['Name'],'version':distribution.version,'direct_url':None if direct_url is None else json.loads(direct_url),'direct_url_sha256':None if direct_url is None else hashlib.sha256(direct_url.encode()).hexdigest()},'installed_encoder_source_sha256':require_checkpoint_identity_api().encoder_source_sha256(),'torch':torch.__version__,'cuda':torch.version.cuda,'scope':'actual installed decoder used for this cell; no encoding performed'}

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
   raw=out.read_bytes();expected=task['payload'].get('existing_result_sha256');assert expected is None or digest(raw)==expected;value=json.loads(raw);assert value['cell_sha256']==digest(json.dumps(cell,sort_keys=True,separators=(',',':')).encode());assert catalog_stat_fence(Path(cell['render']).stat())==cell['render_stat'];assert catalog_stat_fence(Path(cell['wire']).stat())==cell['wire_stat']
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
 manifest=({k:batch[k] for k in ('parent_key','plan_key','child_ordinal')} if args.pb_task_batch else {});manifest.update(schema=('prismabuild.child_result_manifest.v1' if args.pb_task_batch else 'prismaquant.t4_qualification_pilot_result.v1'),results=results);Path(batch['result_manifest_path']).write_text(DIRECT_ASCII_SPACED_LAX.text(manifest)+'\n')
 from prismaquant.residency_map import residency_report
 print(json.dumps({'residency_report':residency_report()},sort_keys=True),flush=True)

