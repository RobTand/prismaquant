"""Metadata-only exact-source cache qualification; any bulk SHA call is a failure."""
import argparse,hashlib,json
from pathlib import Path
from prismaquant import cost_streaming as source
from prismaquant.glm_routing_replay import require_cached_prefix_source_identity
from prismaquant.tessera_joint_allocation import _read_bound
from prismaquant.dev_mode import dev_stamp

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 spec_binding={'path':a.spec,'sha256':a.spec_sha256};spec=json.loads(_read_bound(spec_binding,'routing spec'));plan=json.loads(_read_bound(spec['plan'],'original plan'));prepared=json.loads(_read_bound(spec['prepared'],'original prepared'));cache=json.loads(_read_bound(plan['source_identity_cache'],'source cache'))
 expected={r['path']:r for r in cache['fingerprints']};observed={};calls=[];original_stat=source._streamed_identity_stat_fingerprint
 def stat(path):
  value=original_stat(path)
  if value['path'] in expected:observed[value['path']]=value
  return value
 def forbidden(path):
  calls.append(str(path));raise RuntimeError('unexpected source payload SHA call: '+str(path))
 source._streamed_identity_stat_fingerprint=stat;source._file_sha256=forbidden
 identity=require_cached_prefix_source_identity(plan['model'],plan['source_identity_cache']['path'],prepared['source_model_identity'])
 assert set(observed)==set(expected) and len(observed)==120 and not calls
 portable=0
 for name,live in observed.items():
  changed={k for k in live if live[k]!=expected[name][k]};assert changed<= {'device'}
  portable+=bool(changed)
 proof={'schema':'prismaquant.glm_prefix_source_cache_proof.v1','spec':spec_binding,'source_cache':plan['source_identity_cache'],'full_identity_content_sha256':identity['content_sha256'],'fingerprints_checked':len(observed),'device_only_portable_fingerprints':portable,'bulk_sha_calls':len(calls),'torch':__import__('torch').__version__,**dev_stamp(timestamped=False)}
 raw=(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode();out=Path(a.out)
 with out.open('xb') as f:f.write(raw)
 print(json.dumps({'path':str(out),'sha256':hashlib.sha256(raw).hexdigest(),'proof':proof}),flush=True)
if __name__=='__main__':main()
