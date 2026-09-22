"""PB-admitted host launcher: inspected image, ordinary contained Docker, no pulls."""
import argparse,json,os,subprocess,sys
from pathlib import Path
TS=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/native-pact-acquisition-20260922/tessera-ccc75b877691')
IMAGE='eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'
parser=argparse.ArgumentParser();parser.add_argument('--format',required=True);parser.add_argument('--run',required=True)
args=parser.parse_args()
if '/' in args.run or '..' in args.run:raise ValueError('run must be local basename')
OUT=TS.parent/args.run
sys.path.insert(0,str(TS/'src'))
from tessera.serving.runtime_image import resolve,container_env
record=resolve(IMAGE)
if record['refused']:raise RuntimeError(record['reason'])
env={'PYTHONPATH':'/pq:/tessera:/tessera/src','TESSERA_REPO':'/tessera','OMP_NUM_THREADS':'1',
     'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MAX_JOBS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1',
     'PRISMAQUANT_PROD_ACT_SCALES':'0','HOME':'/tmp','TRITON_CACHE_DIR':'/tmp/triton',
     **container_env(record)}
argv=['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
      '--memory','16g','--memory-swap','16g','--workdir','/pq','--entrypoint','',
      '--mount',f'type=bind,src={Path.cwd()},dst=/pq,readonly',
      '--mount',f'type=bind,src={TS},dst=/tessera,readonly',
      '--mount','type=bind,src=/mnt/shared,dst=/mnt/shared']
for k,v in env.items():argv+=['--env',f'{k}={v}']
# Compile the existing collector under the same admitted resources; it starts
# only in the subsequent fresh measurement process, before any CUDA import.
code="""import glob,subprocess
roots=['/usr/local/cuda','/usr/local/cuda-13.0']
headers=[p for r in roots for p in glob.glob(r+'/**/cupti.h',recursive=True)]
libs=[p for r in roots for p in glob.glob(r+'/**/libcupti.so*',recursive=True)]
assert headers and libs, 'CUDA 13 CUPTI development artifacts absent'
import os
subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
 '-I'+os.path.dirname(headers[0]),'-I/usr/local/cuda/include',
 '/tessera/experiments/csrc/native_operator_resources.cpp',libs[0],
 '-Wl,-rpath,'+os.path.dirname(libs[0]),'-o','/tmp/libtessera_native_memory.so'],check=True)
extra=[]
if FORMAT.startswith('TESSERA_E2M1'):
 subprocess.run(['python3','/tessera/experiments/attest_activation_quantizer.py','emit','--platform','sm_121','--image',IMAGE,'--out','/tmp/activation-quantizers.json'],check=True)
 extra=['--activation-quantizers','/tmp/activation-quantizers.json']
subprocess.run(['python3','experiments/glm_native_dense_acquire.py','prepare','--out',OUT,'--format',FORMAT,*extra],check=True)
subprocess.run(['python3','experiments/glm_native_dense_acquire.py','measure','--out',OUT,
 '--resource-library','/tmp/libtessera_native_memory.so'],check=True)
"""
argv += [IMAGE,'python3','-c','OUT='+repr(str(OUT))+'\nFORMAT='+repr(args.format)+'\nIMAGE='+repr(IMAGE)+'\n'+code]
raise SystemExit(subprocess.call(argv))
