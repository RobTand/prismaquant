#!/usr/bin/env bash
# PQ #2459 gang member: start one TP2 rank container, run one entry role.
#
# PrismaBuild admits both members of the gang together. Each member runs
# this script on its own box. The head member starts the ray head and runs
# the census through tools/pq2459_serve_census.py --mode head. The worker
# member joins the cluster and holds the second rank through --mode worker.
# No member reaches the other box over ssh; the gang owns both admissions.
#
# Usage: pq2459_gang_member.sh <head|worker> <out.json>
set -euo pipefail
ROLE=${1:?role: head or worker}
OUT=${2:?out path}
IMG=${PQ2459_IMAGE:-'localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a'}
TS_Q=${PQ2459_TS:-/home/rob/tessera-2459q}
RUNS=${PQ2459_RUNS:-/home/rob/tessera-runs/tsplugin}
EXT=${PQ2459_EXT:-$RUNS/ext}
HEAD_ADDR=${PQ2459_HEAD_ADDR:-10.100.96.1}
RAY_PORT=${PQ2459_RAY_PORT:-6379}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON=${PQ2459_PYTHON:-/home/rob/venvs/pb-cpu/bin/python}
OUT=$(realpath -m "$OUT")
OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR"
if [ -e "$OUT" ] || [ -e "$OUT.raw.json" ]; then
  echo "[pq2459] REFUSED: output already exists: $OUT" >&2
  exit 2
fi

if [ "$ROLE" != "head" ] && [ "$ROLE" != "worker" ]; then
  echo "[pq2459] REFUSED: role must be head or worker, got $ROLE" >&2
  exit 64
fi
if [ ! -f "$TS_Q/pyproject.toml" ] || [ ! -d "$TS_Q/src/tessera/serving" ]; then
  echo "[pq2459] REFUSED: no staged tree at $TS_Q on $(hostname)" >&2
  exit 2
fi

# Apply the recorded-versus-running image comparison through D32.
"$PYTHON" "$REPO/tools/pq2459_serve_census.py" --mode image-env \
  --runtime-image "$IMG" --out "$OUT.image.json"
IMAGE_ENV=()
OBSERVED_IMAGE=
while IFS= read -r kv; do
  [ -n "$kv" ] && IMAGE_ENV+=(-e "$kv")
  case "$kv" in
    TESSERA_CENSUS_RUNTIME_IMAGE=*) OBSERVED_IMAGE=${kv#*=} ;;
  esac
done < "$OUT.image.json.env"
source "$TS_Q/experiments/serve_lock.sh"
SERVE_LOCK_OWNER="$0 pq2459-$ROLE"
serve_lock_acquire

NAME="pq2459-tp-$ROLE"
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  serve_lock_release
}
trap cleanup EXIT
mkdir -p "$EXT" "$RUNS"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# One fabric on both members: sockets on the bootstrap NIC, IB off.
# It matches the historical R1 serve and the attempt-3 socket arm.
FABRIC=(
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0"
  -e "NCCL_IB_DISABLE=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_CUMEM_HOST_ENABLE=0"
  -e "NCCL_DMABUF_ENABLE=0"
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"
  -e "RAY_memory_monitor_refresh_ms=0"
)
FABRIC+=("${IMAGE_ENV[@]}" -e "TESSERA_ROUTE_TRACE=$RUNS/trace-$ROLE.json")

PREPARE='
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera >/dev/null 2>&1
python3 -c "import importlib.metadata as m; print(\"[tp] plugins:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])"
'

if [ "$ROLE" = "head" ]; then
  docker run -d --name "$NAME" --rm --network host --ipc host \
    --device /dev/infiniband --gpus all \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -v "$TS_Q/src":/tessera/src:ro -v "$TS_Q/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$TS_Q/tools":/tessera/tools:ro -v "$TS_Q/tests":/tessera/tests:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared \
    -v "$OUT_DIR":"$OUT_DIR" -v "$RUNS":"$RUNS" \
    -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
    -e TESSERA_SERVE_MODE=resident -w /tessera "${FABRIC[@]}" \
    --entrypoint bash "$IMG" -c "
$PREPARE
ray start --head --node-ip-address='$HEAD_ADDR' --port=$RAY_PORT >/dev/null
sleep infinity" >/dev/null
  echo "[pq2459] head $HEAD_ADDR:$RAY_PORT on $(hostname), image 5be13705"
  "$PYTHON" "$REPO/tools/pq2459_serve_census.py" \
    --mode head --out "$OUT" --container "$NAME" --head-addr "$HEAD_ADDR" \
    --ray-port "$RAY_PORT" --tessera-src "$TS_Q" \
    --runs-dir "$RUNS" --ext-dir "$EXT" --runtime-image "$OBSERVED_IMAGE"
  rc=$?
else
  docker run -d --name "$NAME" --rm --network host --ipc host \
    --device /dev/infiniband --gpus all \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -v "$TS_Q/src":/tessera/src:ro -v "$TS_Q/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$TS_Q/tools":/tessera/tools:ro -v "$TS_Q/tests":/tessera/tests:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared \
    -v "$OUT_DIR":"$OUT_DIR" -v "$RUNS":"$RUNS" \
    -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
    -e TESSERA_SERVE_MODE=resident -w /tessera "${FABRIC[@]}" \
    --entrypoint bash "$IMG" -c "
$PREPARE
sleep infinity" >/dev/null
  echo "[pq2459] worker on $(hostname), head $HEAD_ADDR:$RAY_PORT"
  "$PYTHON" "$REPO/tools/pq2459_serve_census.py" \
    --mode worker --out "$OUT" --container "$NAME" --head-addr "$HEAD_ADDR" \
    --ray-port "$RAY_PORT" --tessera-src "$TS_Q" \
    --runs-dir "$RUNS" --ext-dir "$EXT" --runtime-image "$OBSERVED_IMAGE"
  rc=$?
fi
echo "[pq2459] member $ROLE exit $rc -> $OUT"
exit "$rc"
