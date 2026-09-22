"""Launch exact source image with PB-owned CPU, reader and residency contracts."""
import json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.container_runtime_identity import image_content_sha256
IMAGE='prismaquant-glm-derivative:causal-exp-v1-20260908'
PIN='/mnt/shared/tessera-pins/tessera-4c384e6049dca3eeaf503bb2c9cd1cd2778978d1'
CONTENT='d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
def launch_argv(cwd,environ,affinity,args):
 mounts=[{'source':str(cwd),'target':'/pq','readonly':True},{'source':str(Path(PIN).parent),'target':str(Path(PIN).parent),'readonly':True},{'source':'/mnt/shared','target':'/mnt/shared','readonly':False}]
 progress=environ.get('PRISMABUILD_ACTION_PROGRESS_PATH')
 if progress and not Path(progress).is_relative_to('/mnt/shared'):
  parent=str(Path(progress).parent);mounts.append({'source':parent,'target':parent,'readonly':False})
 env={'PRISMAQUANT_CONTAINER_CONTENT_SHA256':CONTENT,'PYTHONPATH':f'/pq:{PIN}/src','TESSERA_REPO':PIN,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PRISMAQUANT_PROD_ACT_SCALES':'0','PRISMAQUANT_TMPDIR':'/tmp/prismaquant','TMPDIR':'/tmp','HF_HOME':'/tmp/huggingface','XDG_CACHE_HOME':'/tmp/cache','TRITON_CACHE_DIR':'/tmp/triton','TORCHINDUCTOR_COMPILE_THREADS':'1','MAX_JOBS':'1'}
 from tools.tessera_campaign_container import residency_environment, reader_context_environment, progress_environment
 spec={'container':{'mounts':mounts},'env':env}
 residency_env,residency_mounts=residency_environment(spec,environ)
 reader_env,reader_mounts=reader_context_environment(spec,environ)
 env.update(progress_environment(spec,environ));env.update(residency_env);env.update(reader_env)
 mounts+=residency_mounts+reader_mounts
 argv=['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','--memory','32g','--memory-swap','32g','--cpuset-cpus',','.join(map(str,sorted(affinity))),'--workdir','/pq','--entrypoint','']
 for m in mounts:argv+=['--mount',f"type=bind,src={m['source']},dst={m['target']}"+(',readonly' if m['readonly'] else '')]
 for key,value in env.items():argv+=['--env',f'{key}={value}']
 return argv+[IMAGE,'python3','tools/capture_glm_routing_replay.py','--device-bytes',str(32<<30),*args]
def main():
 actual=image_content_sha256(json.loads(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))[0]);assert actual==CONTENT
 from prismaquant.staged_lease import sdk_submodule
 environ=dict(os.environ);environ.update(sdk_submodule('core')._reader_identity_environment(environ['PRISMABUILD_ACTION_KEY']))
 return subprocess.call(launch_argv(Path.cwd(),environ,os.sched_getaffinity(0),sys.argv[1:]))
if __name__=='__main__':raise SystemExit(main())
