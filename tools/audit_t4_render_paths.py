"""Stat every retained E2M1 render the extension plan names.

Runs only as a script; nothing executes at import. The output path (render-paths.json in the
campaign root when it was first written) is an argument and must not exist.
"""
import argparse
import json,hashlib,concurrent.futures,os
from pathlib import Path


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
 root=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-e2m1-01/workspace')
 p=json.loads((root/'plan.json').read_text());tasks=[(q,Path(row['dir'])/'cache'/(q.replace('/','__').replace('.','_')+'__TESSERA_E2M1_K2_R896.pt')) for row in p['rows'] for q in row['members']]
 def entry(item):
  q,path=item
  try:
   s=path.stat();return {'qname':q,'path':str(path),'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns}
  except FileNotFoundError:return {'qname':q,'path':str(path),'missing':True}
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as ex:rows=list(ex.map(entry,tasks))
 proof=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/rollout-inputs/proof-bundle-9753a5b7c5-c92826fa4.json');raw=proof.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest=='15b373db8429240ef29c0641818d1f7e705d18154a8c2d841381a00213fd86c9';x=json.loads(raw)
 summary={'render_count':len(rows),'existing':sum('missing' not in r for r in rows),'missing':sum('missing' in r for r in rows),'bytes':sum(r.get('bytes',0) for r in rows),'payloads_rehashed':0,'proof_bundle_sha256':digest,'proof_ok':x['ok'],'proof_cells':x['cell_count'],'proof_pb_actions':x['pb_actions'],'rows':rows}
 out=Path(args.out);assert not out.exists(),out;out.write_text(json.dumps(summary,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__ == '__main__':
    main()
