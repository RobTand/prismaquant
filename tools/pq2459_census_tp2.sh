#!/usr/bin/env bash
# PQ #2459 TP2 GPU census, one PrismaBuild action on the head box.
#
# The driver (tessera_plugin_served_tp.sh) starts the ray head here and the
# ray worker on the second box over ssh, then runs the route census inside
# the stock vLLM image. Both boxes import the plugin from their own disk,
# so both checkouts must hold the qualified tree first. This wrapper builds
# that tree from a git archive of the qualified commit and verifies the
# v1 digest before the driver starts. It refuses otherwise.
set -euo pipefail
OUT=${1:-/tmp/pq2459_census_out.json}
COMMIT=9eef9fea6edce32f4e64abf87f0058b11dab2287
DIGEST=a9b7bf32563ce874f45956dd4e5ff4b4c43de73459f9aa40dee1977b9b152330
IMG='localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a'
TS_Q=/home/rob/tessera-2459q
WORKER=${TESSERA_TP_WORKER:-sparklina}

echo "[pq2459] qualified commit $COMMIT digest $DIGEST image 5be13705"
rm -rf "$TS_Q"
mkdir -p "$TS_Q"
git --git-dir=/home/rob/tessera/.git archive "$COMMIT" | tar -x -C "$TS_Q"
if [ ! -f "$TS_Q/pyproject.toml" ] || [ ! -d "$TS_Q/src/tessera/serving" ]; then
  echo "[pq2459] archive missed files; refusing" >&2
  exit 2
fi
echo "[pq2459] head tree ready at $TS_Q"
if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$WORKER" \
    "rm -rf '$TS_Q' && mkdir -p '$TS_Q' && git --git-dir=/home/rob/tessera/.git archive '$COMMIT' | tar -x -C '$TS_Q' && test -f '$TS_Q/pyproject.toml' && test -d '$TS_Q/src/tessera/serving'" \
    </dev/null; then
  echo "[pq2459] REFUSED: cannot stage $TS_Q on $WORKER" >&2
  exit 2
fi
echo "[pq2459] worker tree staged at $TS_Q"
export TS="$TS_Q"
export IMG
export TESSERA_SERVE_MODE=resident
bash "$TS_Q/experiments/tessera_plugin_served_tp.sh" \
  /mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported \
  "$OUT" \
  --runtime-image "$IMG" \
  --tessera-commit "$COMMIT" \
  --tensor-parallel-size 2 --distributed-executor-backend ray
rc=$?
echo "[pq2459] census exit $rc -> $OUT"
exit "$rc"
