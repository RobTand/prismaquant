
set -euo pipefail
CAND=/mnt/shared/tessera-pins/fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb
IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5
mkdir -p /mnt/shared/tessera-measurements/pq1317-gpu-tests
OUT=$(mktemp -d /mnt/shared/tessera-measurements/pq1317-gpu-tests/dense.XXXXXXXX)
mkdir -p "$OUT/tmp" "$OUT/home" "$OUT/test-deps" "$OUT/torch-ext"
chmod 0777 "$OUT" "$OUT/tmp" "$OUT/home" "$OUT/test-deps" "$OUT/torch-ext"
CPUS=$(python3 -c "import os; print(\",\".join(map(str,sorted(os.sched_getaffinity(0)))))")
echo "DENSE_OUT=$OUT CPUS=$CPUS"
docker run --rm --gpus all --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" --shm-size 1g -v "$CAND:$CAND:ro" -v "$OUT:$OUT:rw" -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet:ro -w "$CAND" -e TS611_OUT="$OUT" -e TMPDIR="$OUT/tmp" -e HOME="$OUT/home" -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e PYTHONPATH="$CAND/src:$OUT/test-deps" "$IMAGE" bash -c 'set -euo pipefail; python3 - <<BOOT 2>&1 | tee "$TS611_OUT/boot.log"
import importlib.metadata, importlib.util, json, os, subprocess, sys
print("TS611_CONTAINER_USER " + json.dumps({"uid": os.getuid(), "gid": os.getgid()}), flush=True)
pins={"pytest":"pytest==8.4.2","xdist":"pytest-xdist==3.8.0","execnet":"execnet==2.1.1","pluggy":"pluggy==1.6.0","iniconfig":"iniconfig==2.1.0","pygments":"Pygments==2.19.2","packaging":"packaging==25.0"}
missing=[pin for mod,pin in pins.items() if importlib.util.find_spec(mod) is None]
print("MISSING "+json.dumps(missing), flush=True)
if missing:
    subprocess.run([sys.executable,"-m","pip","install","--no-deps","--no-cache-dir","--target",os.environ["TS611_OUT"]+"/test-deps",*missing],check=True)
import pytest, xdist
print("TS611_TEST_TOOLS "+json.dumps({"pytest":pytest.__version__,"xdist":importlib.metadata.version("pytest-xdist")}),flush=True)
import torch, vllm, tessera
print("TS611_RUNTIME "+json.dumps({"torch":torch.__version__,"vllm":vllm.__version__,"tessera_source":tessera.__file__,"cuda_available":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"capability":list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}),flush=True)
BOOT
set +e
python3 -m pytest -p xdist.plugin -n 2 --dist worksteal --durations=20 --strict-cuda --surface-json "$TS611_OUT/surface.json" --basetemp "$TS611_OUT/tmp/pytest" --junitxml "$TS611_OUT/junit.xml" -o cache_dir="$TS611_OUT/pytest-cache" tests/test_serving_bf16_gemv.py tests/test_serving_fp8_gemv.py 2>&1 | tee "$TS611_OUT/pytest.log"
rc=${PIPESTATUS[0]}
set -e
python3 - <<POP 2>&1 | tee -a "$TS611_OUT/pytest.log"
import json, os, pathlib
p=pathlib.Path(os.environ["TS611_OUT"])/"surface.json"
print("TS611_SURFACE "+(p.read_text() if p.exists() else json.dumps({"missing":str(p)})),flush=True)
POP
echo "TS611_RC $rc" | tee -a "$TS611_OUT/pytest.log"
find "$TS611_OUT/torch-ext" -name "*.so" -exec sha256sum {} \; 2>/dev/null | tee "$TS611_OUT/native-so.sha256" || true
exit "$rc"'
echo "DOCKER_RC=$?"
