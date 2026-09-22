"""PB-admitted native acquisition; prepare A4 under the actual quality image.

Both images must already be present and declared to PB. No pull or deployment.
"""
import argparse,json,os,subprocess,sys,tempfile,shutil
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.container_runtime_identity import image_content_sha256

TS=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/native-pact-acquisition-20260922/tessera-32e92e85e')
TS_CONTAINER=Path('/native-sources')/TS.name
IMAGE='eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'
QUALITY_IMAGE='prismaquant-glm-derivative:causal-exp-v1-20260908'
QUALITY_CONTENT='d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--run',required=True);p.add_argument('--inputs-root',type=Path);p.add_argument('--origin-sha256')
args=p.parse_args()
if '/' in args.run or '..' in args.run:raise ValueError('run must be local basename')
out=TS.parent/args.run
from prismaquant.staged_lease import sdk_submodule
reader_environment=dict(os.environ)
reader_environment.update(sdk_submodule('core')._reader_identity_environment(reader_environment['PRISMABUILD_ACTION_KEY']))
local_owner=tempfile.TemporaryDirectory(prefix='pq-native-handoff-',dir=Path.cwd().parent)
local_path=Path(local_owner.name)
filesystem=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(local_path)],text=True).strip()
if filesystem in ('tmpfs','ramfs','nfs','nfs4','cifs','smb3','fuse.sshfs'):
    raise ValueError('native handoff requires physical worker-local SSD storage')
local_stat=local_path.stat()
if local_stat.st_uid!=os.getuid():raise ValueError('local handoff directory owner differs')

# Use the pinned producer's CLI to resolve its serving image declaration.
cli_env={**os.environ,'PYTHONPATH':str(TS/'src')}
record=json.loads(subprocess.check_output([sys.executable,'-m','tessera.serving.runtime_image','resolve','--image',IMAGE],env=cli_env,text=True))
if record['refused']:raise RuntimeError(record['reason'])
declared=subprocess.check_output([sys.executable,'-m','tessera.serving.runtime_image','container-env'],
    input=json.dumps(record),env=cli_env,text=True)
serving_env=dict(line.split('=',1) for line in declared.splitlines() if line)


def run(image,command,*,serving):
    inspected=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
    content=image_content_sha256(inspected)
    if not serving and content!=QUALITY_CONTENT:raise ValueError('quality producer image changed')
    env={'PRISMAQUANT_CONTAINER_CONTENT_SHA256':content,'PYTHONPATH':f'/pq:{TS_CONTAINER}:{TS_CONTAINER}/src',
         'TESSERA_REPO':str(TS_CONTAINER),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MAX_JOBS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1','PRISMAQUANT_PROD_ACT_SCALES':'0',
         'TMPDIR':'/tmp','HF_HOME':'/tmp/huggingface','XDG_CACHE_HOME':'/tmp/cache','PRISMAQUANT_TMPDIR':'/tmp/prismaquant','TRITON_CACHE_DIR':'/tmp/triton',**(serving_env if serving else {})}
    from tools.tessera_campaign_container import residency_environment,reader_context_environment,progress_environment
    mounts=[{'source':str(Path.cwd()),'target':'/pq','readonly':True},
        {'source':str(TS.parent),'target':'/native-sources','readonly':True},
        {'source':str(local_path),'target':'/native-spool','readonly':False},
        {'source':'/mnt/shared','target':'/mnt/shared','readonly':False}]
    progress=reader_environment.get('PRISMABUILD_ACTION_PROGRESS_PATH')
    if progress and not Path(progress).is_relative_to('/mnt/shared'):
        parent=str(Path(progress).parent);mounts.append({'source':parent,'target':parent,'readonly':False})
    declaration={'container':{'mounts':mounts},'env':env}
    residency_env,residency_mounts=residency_environment(declaration,reader_environment)
    read_env,read_mounts=reader_context_environment(declaration,reader_environment)
    env.update(residency_env);env.update(read_env);env.update(progress_environment(declaration,reader_environment))
    env['PRISMABUILD_ACTION_KEY']=reader_environment['PRISMABUILD_ACTION_KEY']
    mounts+=residency_mounts+read_mounts
    argv=['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
          '--memory','40g','--memory-swap','40g','--cpuset-cpus',','.join(map(str,sorted(os.sched_getaffinity(0)))),
          '--workdir','/pq','--entrypoint','']
    for mount in mounts:
        argv+=['--mount',f"type=bind,src={mount['source']},dst={mount['target']}"+(',readonly' if mount['readonly'] else '')]
    for key,value in env.items():argv+=['--env',f'{key}={value}']
    subprocess.run([*argv,image,*command],check=True)


import hashlib
if hashlib.sha256(args.spec.read_bytes()).hexdigest()!=args.spec_sha256:raise ValueError('execution specification changed')
if args.inputs_root:
    raise ValueError('local handoff cannot be reused across admitted actions')
else:
    inputs=Path('/native-spool/inputs')
    run(QUALITY_IMAGE,['python3','experiments/glm_native_moe_acquire.py','--spec',str(args.spec),
        '--spec-sha256',args.spec_sha256,'--out',str(inputs),'--local-root','/native-spool/blobs','--device-bytes',str(48<<30)],serving=False)
    origin_sha=hashlib.sha256((local_path/'inputs/origin.json').read_bytes()).hexdigest()
code="""import glob,subprocess,os
roots=['/usr/local/cuda','/usr/local/cuda-13.0']
headers=[p for r in roots for p in glob.glob(r+'/**/cupti.h',recursive=True)]
libs=[p for r in roots for p in glob.glob(r+'/**/libcupti.so*',recursive=True)]
assert headers and libs, 'CUDA CUPTI development artifacts absent'
subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
 '-I'+os.path.dirname(headers[0]),'-I/usr/local/cuda/include',
 TSROOT+'/experiments/csrc/native_operator_resources.cpp',libs[0],
 '-Wl,-rpath,'+os.path.dirname(libs[0]),'-o','/tmp/libtessera_native_memory.so'],check=True)
subprocess.run(['python3',TSROOT+'/experiments/measure_glm_native_execution.py',
 '--inputs-root',INPUTS,'--origin-sha256',ORIGIN_SHA,'--out',OUT,
 '--resource-library','/tmp/libtessera_native_memory.so','--device-bytes',str(48<<30)],check=True)
"""
run(IMAGE,['python3','-c','TSROOT='+repr(str(TS_CONTAINER))+'\nINPUTS='+repr(str(inputs))+'\nORIGIN_SHA='+repr(origin_sha)+'\nOUT='+repr(str(out))+'\n'+code],serving=True)

# Preserve small independent identities; bulk handoff data dies with this owner.
for name in ('inputs.json','origin.json','weight-references.json'):
    target=out/('preparation-'+name)
    with target.open('xb') as handle:handle.write((local_path/'inputs'/name).read_bytes())
local_owner.cleanup()
