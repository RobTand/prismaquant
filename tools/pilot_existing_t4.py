"""Decode one retained E2M1 wire and compare it with its render (GPU).

Runs only as a script; nothing executes at import. The output path (pilot.json in the
campaign root when it was first written) is an argument and must not exist.
"""
import argparse
import json,pickle,hashlib,time
from pathlib import Path
import torch
from tessera.cached_unit import verify_cached_unit
from prismaquant.tessera_joint_aura import _decode_wire
from prismaquant.production_weight_cache import _canonical_rendered_weight_tensor


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
 base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911')
 q='model.language_model.layers.10.mlp.experts.0.down_proj';fmt='TESSERA_E2M1_K2_R896';fname=hashlib.sha256(q.encode()).hexdigest()+'.pkl'
 def unit(folder):
  outer=pickle.loads((folder/'cost.anchors.json.parts/units'/fname).read_bytes());assert hashlib.sha256(outer['payload']).hexdigest()==outer['payload_sha256'];return pickle.loads(outer['payload'])
 a4=unit(base/'extension-e2m1-01/workspace/merged-c92826fa4')
 a8=unit(base/'extension-r1024-02/workspace/rows/row-0045')
 r=a4['wire_records'][fmt]; old=next(iter(a8['wire_records'].values()))
 shared={k:r['identity'][k]==old['identity'][k] for k in ['source','projection','calibration','encoder_fixture_id']};assert all(shared.values()),shared
 wire=base/'extension-e2m1-01/workspace/merged-c92826fa4/cache/wire'/r['file'];blob=wire.read_bytes();verify_cached_unit(blob,r,r['identity'])
 render=base/'extension-e2m1-01/workspace/rows/row-0045/cache'/(q.replace('.','_')+'__'+fmt+'.pt')
 st=render.stat(); value=torch.load(render,map_location='cpu',weights_only=False)
 print('render_type',type(value).__name__, 'keys',list(value) if isinstance(value,dict) else None,flush=True)
 if isinstance(value,dict):
  tensors=[v for v in value.values() if isinstance(v,torch.Tensor)];assert len(tensors)==1;value=tensors[0]
 start=time.monotonic();decoded=_decode_wire(blob,reader=None,device='cuda').to(torch.bfloat16).cpu();elapsed=time.monotonic()-start
 assert torch.equal(decoded,value),(decoded.dtype,value.dtype,decoded.shape,value.shape)
 assert (render.stat().st_ino,render.stat().st_size,render.stat().st_mtime_ns)==(st.st_ino,st.st_size,st.st_mtime_ns)
 result={'qname':q,'format':fmt,'wire_sha256':r['blob_sha256'],'shared_source_identity':shared,'render_equal':True,'shape':list(decoded.shape),'dtype':str(decoded.dtype),'decode_seconds':elapsed,'device':torch.cuda.get_device_name(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'render_stat':{'inode':st.st_ino,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},'scope':'one current decoder consumption witness; no new encoding, no general per-cell digest claims'}
 out=Path(args.out);assert not out.exists(),out;out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
