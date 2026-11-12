
set -euo pipefail
CAND=/mnt/shared/tessera-pins/fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb
IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5
mkdir -p /mnt/shared/tessera-measurements/pq1317-gpu-tests
OUT=$(mktemp -d /mnt/shared/tessera-measurements/pq1317-gpu-tests/preflight.XXXXXXXX)
mkdir -p "$OUT/tmp" "$OUT/home" "$OUT/test-deps"
chmod 0777 "$OUT" "$OUT/tmp" "$OUT/home" "$OUT/test-deps"
CPUS=$(python3 -c "import os; print(\",\".join(map(str,sorted(os.sched_getaffinity(0)))))")
echo "OUT=$OUT"
docker run --rm --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" --shm-size 1g -v "$CAND:$CAND:ro" -v "$OUT:$OUT:rw" -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet:ro -w "$CAND" -e CUDA_VISIBLE_DEVICES= -e TS_OUT="$OUT" -e TMPDIR="$OUT/tmp" -e HOME="$OUT/home" -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e PYTHONPATH="$CAND/src:$OUT/test-deps" "$IMAGE" bash -c 'set -euo pipefail; python3 -m pip install --no-cache-dir --target "$TS_OUT/test-deps" --no-deps "pytest==8.4.2" "pytest-xdist==3.8.0" "execnet==2.1.1" "pluggy==1.6.0" "iniconfig==2.1.0" "Pygments==2.19.2" "packaging==25.0" 2>&1 | tail -n 3; python3 -m pytest --collect-only -q tests/test_native_window_moe.py tests/test_native_window_moe_method.py tests/test_serving_moe_tp2.py tests/test_serving_bf16_gemv.py tests/test_serving_fp8_gemv.py 2>&1 | tee "$TS_OUT/collect.txt"; echo COLLECT_DONE'
echo "DOCKER_RC=$?"
tail -n 60 "$OUT/collect.txt"
