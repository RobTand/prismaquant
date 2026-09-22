"""Run the bounded source prefix inside the exact already-qualified source image."""
import json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.container_runtime_identity import image_content_sha256
image='prismaquant-glm-derivative:causal-exp-v1-20260908'
pin='/mnt/shared/tessera-pins/tessera-4c384e6049dca3eeaf503bb2c9cd1cd2778978d1'
actual=image_content_sha256(json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]);assert actual=='d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
env={'PRISMAQUANT_CONTAINER_CONTENT_SHA256':actual,'PYTHONPATH':'/pq:/tessera-pin/src','TESSERA_REPO':'/tessera-pin','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PRISMAQUANT_PROD_ACT_SCALES':'0','TRITON_CACHE_DIR':'/tmp/triton','TORCHINDUCTOR_COMPILE_THREADS':'1','MAX_JOBS':'1'}
argv=['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','--memory','48g','--memory-swap','48g','--cpuset-cpus',','.join(map(str,sorted(os.sched_getaffinity(0)))),'--workdir','/pq','--entrypoint','','--mount',f'type=bind,src={Path.cwd()},dst=/pq,readonly','--mount',f'type=bind,src={pin},dst=/tessera-pin,readonly','--mount','type=bind,src=/mnt/shared,dst=/mnt/shared']
from prismaquant.staged_lease import sdk_submodule
core=sdk_submodule('core')
key=os.environ['PRISMABUILD_ACTION_KEY'];env.update(core._reader_identity_environment(key));env['PRISMABUILD_ACTION_KEY']=key
env['PRISMABUILD_RESIDENCY_MAP']=os.environ['PRISMABUILD_RESIDENCY_MAP']
for name in ('PRISMABUILD_ACTION_PROGRESS_PATH','PRISMABUILD_ACTION_PROGRESS_TOKEN','PRISMABUILD_ACTION_PROGRESS_PHASES'):
 if name in os.environ:env[name]=os.environ[name]
progress_path=env.get('PRISMABUILD_ACTION_PROGRESS_PATH')
if progress_path and not Path(progress_path).is_relative_to('/mnt/shared'):
 parent=Path(progress_path).parent
 argv+=['--mount',f'type=bind,src={parent},dst={parent}']
for key,value in env.items():argv+=['--env',f'{key}={value}']
argv += [image,'python3','tools/capture_glm_routing_replay.py',*sys.argv[1:]]
raise SystemExit(subprocess.call(argv))
