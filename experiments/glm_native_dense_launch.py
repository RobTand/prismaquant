"""PB-admitted host launcher: inspected image, ordinary contained Docker, no pulls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


TS = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/native-pact-acquisition-20260922/tessera-ccc75b877691')
IMAGE = 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'

_MEASUREMENT_CODE = """import glob,subprocess
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args(argv)
    if '/' in args.run or '..' in args.run:
        raise ValueError('run must be local basename')
    out = TS.parent / args.run

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tools.tessera_campaign_container import inspect_or_load

    runtime_identity = runpy.run_path(str(root / 'prismaquant/container_runtime_identity.py'))
    inspected = inspect_or_load({'image': IMAGE})[0]
    repo_digests = inspected.get('RepoDigests')
    if not isinstance(repo_digests, list) or IMAGE not in repo_digests:
        raise RuntimeError('inspected image does not carry the requested RepoDigest: ' + IMAGE)
    content = runtime_identity['image_content_sha256'](inspected)
    declaration = {
        'schema': 'tessera.runtime_image/1',
        'requested': IMAGE,
        'required': IMAGE,
        'resolved_reference': IMAGE,
        'repo_digests': repo_digests,
        'local_id': inspected.get('Id'),
        'present': True,
        'gated': True,
        'refused': False,
        'reason': 'explicit_digest',
    }
    env = {
        'PYTHONPATH': '/pq:/tessera:/tessera/src',
        'TESSERA_REPO': '/tessera',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MAX_JOBS': '2',
        'TORCHINDUCTOR_COMPILE_THREADS': '1',
        'PRISMAQUANT_PROD_ACT_SCALES': '0',
        'HOME': '/tmp',
        'TRITON_CACHE_DIR': '/tmp/triton',
        'PRISMAQUANT_CONTAINER_CONTENT_SHA256': content,
        'TESSERA_CENSUS_RUNTIME_IMAGE': IMAGE,
        'TESSERA_CENSUS_RUNTIME_IMAGE_DECLARATION': json.dumps(declaration, sort_keys=True),
    }
    command = [
        'docker', 'run', '--rm', '--gpus', 'all', '--ipc=host',
        '--user', f'{os.getuid()}:{os.getgid()}',
        '--memory', '16g', '--memory-swap', '16g',
        '--cpuset-cpus', ','.join(map(str, sorted(os.sched_getaffinity(0)))),
        '--workdir', '/pq', '--entrypoint', '',
        '--mount', f'type=bind,src={Path.cwd()},dst=/pq,readonly',
        '--mount', f'type=bind,src={TS},dst=/tessera,readonly',
        '--mount', 'type=bind,src=/mnt/shared,dst=/mnt/shared',
    ]
    for key, value in env.items():
        command += ['--env', f'{key}={value}']
    code = 'OUT=' + repr(str(out)) + '\nFORMAT=' + repr(args.format) + '\nIMAGE=' + repr(IMAGE) + '\n' + _MEASUREMENT_CODE
    command += [IMAGE, 'python3', '-c', code]
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
