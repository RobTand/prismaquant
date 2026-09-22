"""Freeze missing expert A4 cells by adopting exact already-qualified source/H commitments."""
import concurrent.futures,copy,hashlib,json,os,pickle,time
from pathlib import Path
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_aura import activation_identity
from prismaquant import format_registry as fr

ROOT=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922')
BASE=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911')
PREP=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json')
FMT='TESSERA_E2M1_K2_R896'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def stamp(path):
 s=path.stat();return {'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def main():
 started=time.monotonic();prepared=json.loads(PREP.read_bytes());pwc_path=Path(prepared['production_cache']['path']);raw=pwc_path.read_bytes();assert sha(raw)==prepared['production_cache']['sha256'];cache=pickle.loads(raw);del raw
 cost_path=BASE/'extension-e2m1-01/workspace/merged-c92826fa4/cost.pkl';raw=cost_path.read_bytes();cost_sha=sha(raw);cost=pickle.loads(raw);del raw
 proof_path=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/rollout-inputs/proof-bundle-9753a5b7c5-c92826fa4.json');proof_sha=sha(proof_path.read_bytes());assert proof_sha=='15b373db8429240ef29c0641818d1f7e705d18154a8c2d841381a00213fd86c9'
 oldplan=json.loads((BASE/'extension-r1024-02/workspace/plan.json').read_text());a4plan=json.loads((BASE/'extension-e2m1-01/workspace/plan.json').read_text());oldowners={q:Path(row['dir']) for row in oldplan['rows'] for q in row['members']};a4owners={q:Path(row['dir']) for row in a4plan['rows'] for q in row['members']}
 expert_names=sorted(q for q in prepared['formats_by_qname'] if '.experts.' in q);assert set(expert_names)==set(cost['costs'])==set(a4owners)
 prior={}
 for key,verified in cache.metadata['verified_cells'].items():
  q,fmt=key
  if q in a4owners and q not in prior:prior[q]=(fmt,verified)
 spec=fr.get_format(FMT)
 def cell(q):
  oldfmt,verified=prior[q];oldpath=oldowners[q]/'cost.anchors.json.parts/units'/(sha(q.encode())+'.pkl');envelope_raw=oldpath.read_bytes();envelope=pickle.loads(envelope_raw);assert sha(envelope['payload'])==envelope['payload_sha256'];unit=pickle.loads(envelope['payload']);old=unit['wire_records'][oldfmt]
  assert old['blob_sha256']==verified['wire_sha256'],q
  normalized=copy.deepcopy(old['identity']);normalization=verified.get('encoder_source_reuse',{}).get('recorded_encoder_source_sha256');
  if normalization:normalized['encoder_source_sha256']=normalization
  assert canonical_json_sha256(normalized,where='adopted old encoding identity')==verified['encoding_identity_sha256'],q
  record=cost['tessera_expert_wires'][q][FMT]
  for field in ('unit','source','projection','calibration','encoder_fixture_id'):
   assert record['identity'][field]==old['identity'][field],(q,field)
  anchor=cost['costs'][q][FMT];activation=activation_identity(spec,cache.activation_max_abs,q);assert activation['input_global_scale']==anchor['input_global_scale'],q
  render=a4owners[q]/'cache'/(q.replace('/','__').replace('.','_')+'__'+FMT+'.pt');wire=Path(cost['provenance']['wire_dir'])/record['file']
  assert wire.stat().st_size==record['blob_bytes']
  return {'qname':q,'format':FMT,'render':str(render),'render_stat':stamp(render),'wire':str(wire),'wire_stat':stamp(wire),'record':record,'anchor':anchor,'source_weight':verified['source_weight'],'activation':activation,'encoding_identity_sha256':canonical_json_sha256(record['identity'],where='adopted A4 encoding identity'),'render_origin':'encoded','render_comparison':'independent_render_vs_wire','catalog_source_adoption':{'schema':'prismaquant.joint_catalog_source_adoption.v1','reference_pair':[q,oldfmt],'reference_encoding_identity':normalized,'candidate_encoding_identity':record['identity'],'encoder_source_proof':{'path':str(proof_path),'sha256':proof_sha}},'adopted_source_hessian':{'schema':'prismaquant.adopted_source_hessian.v1','old_format':oldfmt,'old_pwc_sha256':prepared['production_cache']['sha256'],'old_verified_record_sha256':canonical_json_sha256(verified,where='old verified cell'),'old_wire_sha256':old['blob_sha256'],'old_encoding_identity_sha256':verified['encoding_identity_sha256'],'old_unit_envelope':str(oldpath),'old_unit_envelope_sha256':sha(envelope_raw),'encoder_source_normalized_to':normalization,'reseal_proof_sha256':proof_sha,'scope':'source/H/projection/calibration commitments adopted by exact equality to an independently qualified old cell; no new source/H payload recomputation'}}
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:cells=list(pool.map(cell,expert_names))
 catalog={'schema':'prismaquant.t4_adopted_catalog.v1','old_prepared':{'path':str(PREP),'sha256':sha(PREP.read_bytes())},'old_pwc':prepared['production_cache'],'cost':{'path':str(cost_path),'sha256':cost_sha},'reseal_proof':{'path':str(proof_path),'sha256':proof_sha},'source_model_identity':prepared['source_model_identity'],'calibration_input':prepared['calibration_input'],'source_execution':prepared['source_execution'],'reader_identity':prepared['reader_identity'],'projection_backend':prepared['projection_backend'],'format':FMT,'cells':cells,'status':'source_hessian_adopted_render_qualification_pending'}
 out=ROOT/'adopted-catalog.json';assert not out.exists();raw=json.dumps(catalog,sort_keys=True,separators=(',',':')).encode()+b'\n';out.write_bytes(raw)
 print(json.dumps({'path':str(out),'sha256':sha(raw),'cells':len(cells),'seconds':time.monotonic()-started,'status':catalog['status']}),flush=True)
if __name__=='__main__':main()
