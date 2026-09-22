"""PB host launcher for the exact campaign image and Tessera pin; no pulls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.container_runtime_identity import image_content_sha256

image = "prismaquant-glm-derivative:causal-exp-v1-20260908"
parser = argparse.ArgumentParser()
parser.add_argument('--group-policy')
parser.add_argument('--group-policy-sha256')
args = parser.parse_args()
if bool(args.group_policy) != bool(args.group_policy_sha256):
    parser.error('group policy path and SHA256 are required together')
pin = "/mnt/shared/tessera-pins/tessera-4c384e6049dca3eeaf503bb2c9cd1cd2778978d1"
inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
content = image_content_sha256(inspected)
assert content == "d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026"
env = {"PRISMAQUANT_CONTAINER_CONTENT_SHA256": content,
       "PYTHONPATH": "/pq:/tessera-pin/src", "TESSERA_REPO": "/tessera-pin",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "PRISMAQUANT_PROD_ACT_SCALES": "0", "TRITON_CACHE_DIR": "/tmp/triton",
       "TORCHINDUCTOR_COMPILE_THREADS": "1", "MAX_JOBS": "1"}
argv = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "8g", "--memory-swap", "8g",
        "--cpuset-cpus", ",".join(map(str, sorted(os.sched_getaffinity(0)))),
        "--workdir", "/pq", "--entrypoint", "",
        "--mount", f"type=bind,src={Path.cwd()},dst=/pq,readonly",
        "--mount", f"type=bind,src={pin},dst=/tessera-pin,readonly"]
for key, value in env.items():
    argv += ["--env", f"{key}={value}"]
if args.group_policy:
    argv += ['--mount', 'type=bind,src=/mnt/shared,dst=/mnt/shared,readonly']
    argv += [image, 'python3', 'tools/check_stageb_a4_group_quantizer.py',
             '--policy', args.group_policy, '--policy-sha256', args.group_policy_sha256]
else:
    argv += [image, "python3", "tools/check_stageb_a4_quantizer.py"]
raise SystemExit(subprocess.call(argv))
