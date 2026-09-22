"""Prepare immutable candidate metadata after every new render is qualified.

Does not issue any Stage A capture reuse authority or activate Stage B.
"""
import copy,hashlib,json,os,pickle
from pathlib import Path
from prismaquant.tessera_joint_aura import render_origin_census
ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922')
PANEL=ROOT.parent/'allocation/joint-panel'
OLDPLAN=PANEL/'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json'
OLDPREP=PANEL/'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json'
CATALOG_SHA='71238bdab85bdccabed921ce94b03459d5aba3607021b85007bf41f3e376fbc8'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def doc(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def bound(path):return dict(path=str(path),sha256=sha(path.read_bytes()))
def publish(path,raw):
 assert not path.exists(),path;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(tmp,path);tmp.unlink();return {'path':str(path),'sha256':sha(raw)}
def main():
 catalogpath=ROOT/'adopted-catalog.json';raw=catalogpath.read_bytes();assert sha(raw)==CATALOG_SHA;catalog=json.loads(raw);oldprep=json.loads(OLDPREP.read_bytes());plan=json.loads(OLDPLAN.read_bytes());raw=Path(oldprep['production_cache']['path']).read_bytes();assert sha(raw)==oldprep['production_cache']['sha256'];cache=pickle.loads(raw);del raw
 overlay=ROOT/'proposed-overlay';assert not overlay.exists(),'immutable proposed overlay already exists'
 plan['inputs']['candidate_overlay']={'path':str(catalogpath),'sha256':CATALOG_SHA};plan['output_root']=str(overlay)
 prepared=copy.deepcopy(oldprep)
 for cell in catalog['cells']:
  q,fmt=cell['qname'],cell['format'];pair=(q,fmt);assert pair not in cache.weights
  value=json.loads((ROOT/'qualified'/(sha(q.encode())+'.json')).read_bytes());assert value['cell_sha256']==sha(json.dumps(cell,sort_keys=True,separators=(',',':')).encode());receipt=value['verified_cell']
  if 'verified_cell_sha256' in value:assert value['verified_cell_sha256']==sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
  for key in ('source_weight','activation','encoding_identity_sha256','render_origin','render_comparison','catalog_source_adoption'):
   assert receipt[key]==cell[key],(pair,key)
  for key in ('wire','render'):
   s=Path(cell[key]).stat();assert cell[key+'_stat']==dict(inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
  assert receipt['render_file_sha256'] and receipt['rendered_weight']['content_sha256'];cache.weights[pair]=cell['render'];cache._lru_paths[pair]=cell['render'];cache.metadata['verified_cells'][pair]=receipt;prepared['formats_by_qname'][q].append(fmt)
 assert len(cache.weights)==234278
 planbinding=publish(overlay/'plan.json',doc(plan));cache.metadata['inputs']=plan['inputs'];cache.metadata['plan_sha256']=planbinding['sha256'];census=render_origin_census(r['render_origin'] for r in cache.metadata['verified_cells'].values());cache.metadata.update(census)
 prepared.update(census);prepared['measured_cells']=len(cache.weights);prepared['plan_sha256']=planbinding['sha256'];prepared['production_cache']=publish(overlay/'prepare/production.pkl',pickle.dumps(cache,protocol=pickle.HIGHEST_PROTOCOL));prepbinding=publish(overlay/'prepare/prepared.json',doc(prepared))
 inputs={'original_plan':bound(OLDPLAN),'original_prepared':bound(OLDPREP),'extended_plan':planbinding,'extended_prepared':prepbinding};publish(overlay/'catalog-pair-inputs.json',doc(inputs));print(json.dumps({'status':'proposed_candidate_metadata_only','inputs':inputs,'cells':len(cache.weights)}),flush=True)
if __name__=='__main__':main()
