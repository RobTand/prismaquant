#!/usr/bin/env bash
# Before/after profile of the campaign row head (RobTand/prismaquant#492).
#
# Same inputs, same box, same interpreter, both arms on this checkout:
#   before = the serial phases the head ran until #492 (hold at 1 thread seals
#            the owner inside its first receipt; commitments digest every H;
#            wireverify inline), i.e. the code paths the default flags select;
#   after  = the sequence #492 runs with --campaign-identity-threads N
#            (seal ahead, hold on N, commitments from the sealed receipts,
#            wire verify on N).
# The after records carry the same identity_digest / capture_sha256 as the
# before records, so the receipt equality is checked on the results, not
# assumed. Everything read is read-only; writes go under $SCRATCH only.
set -u

CHECKOUT="${CHECKOUT:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRATCH="${SCRATCH:-/home/rob/tmp/pq-rowhead-prof}"
pick() { for c in "$@"; do [ -x "$c" ] && { echo "$c"; return; }; done; }
PYTHON="${PYTHON:-$(pick /home/rob/venvs/pq-cpu312/bin/python \
                         /home/rob/venvs/pq-cu130/bin/python \
                         /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python)}"
PYSPY="${PYSPY:-$(pick /home/rob/venvs/pb-cpu/bin/py-spy \
                       /home/rob/venvs/pq-cu130/bin/py-spy \
                       /home/rob/.local/bin/py-spy \
                       /home/rob/venvs/pq-release/bin/py-spy)}"
echo "interpreter=$PYTHON pyspy=$PYSPY host=$(hostname) checkout=$CHECKOUT $(git -C "$CHECKOUT" rev-parse --short HEAD 2>/dev/null)"
WS=/mnt/shared/tessera-measurements/glm-canonical-census-20260908
PRODUCER=$WS/identity-reseal-20260911/producer-source-d403cc5a31
ROW=$WS/first-proof-anchor-preparation-05/workspace/rows/row-0065
EXPERTS="${EXPERTS:-48}"
THREADS="${THREADS:-8}"

mkdir -p "$SCRATCH/out" "$SCRATCH/tmp"
RESULTS="$SCRATCH/out/phases.jsonl"
export PYTHONPATH="$CHECKOUT:$PRODUCER/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TESSERA_REPO="$PRODUCER" TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/cache"
export HF_HUB_OFFLINE=1

run() {   # run <tag> <pyspy-mode|none> -- <harness args...>
  local tag="$1" mode="$2"; shift 3
  local prefix=()
  if [ "$mode" != "none" ] && [ -n "$PYSPY" ]; then
    prefix=("$PYSPY" record --rate 100 --format speedscope \
            -o "$SCRATCH/out/${tag}.${mode}.speedscope.json")
    [ "$mode" = "idle" ] && prefix+=(--idle)
    [ "$mode" = "gil" ] && prefix+=(--gil)
    prefix+=(--)
  fi
  echo "=== $tag mode=$mode $(date -Is)"
  "${prefix[@]}" "$PYTHON" -u "$CHECKOUT/experiments/profile_row_head.py" "$@" 2>&1 | tail -6
  echo "--- exit ${PIPESTATUS[0]}"
}

COMMON=(--model /mnt/shared/models/GLM-5.3-Flash-BF16
        --census "$WS/first-proof-anchor-preparation-05/workspace/census.json"
        --units "$WS/first-proof-anchor-preparation-05/workspace/units/row-0065.json"
        --calibration-inputs "$WS/workspace/calibration-cache/inputs"
        --input-scales "$ROW/cache/input_scales.safetensors"
        --hessian-references "$ROW/cache/hessian_capture.references.json"
        --experts "$EXPERTS" --out "$RESULTS")
JOURNAL=(--journal-manifest "$ROW/cost.anchors.json"
         --journal-dir "$ROW/cost.anchors.json.parts")

case "${STAGE:-ab}" in
  ab)
    # before: the serial head, on this checkout (default flags = serial paths)
    run hold-before none -- --phase hold "${COMMON[@]}" --label before
    run commitments-before none -- --phase commitments "${COMMON[@]}" --label before
    run journal none -- --phase journal "${JOURNAL[@]}" --out "$RESULTS" --label after-digest-first
    run wireverify-before none -- --phase wireverify "${COMMON[@]}" "${JOURNAL[@]}" \
      --wire-dir "$ROW/cache/wire" --label before
    # after: #492's sequence, N threads, once plain and once with the
    # sleeping threads visible so the sha256 workers show as such
    run headafter-t$THREADS none -- --phase headafter "${COMMON[@]}" --threads "$THREADS" --label after
    run headafter-t$THREADS idle -- --phase headafter "${COMMON[@]}" --threads "$THREADS" --label after-idle
    run wireverifyafter-t$THREADS none -- --phase wireverifyafter "${COMMON[@]}" "${JOURNAL[@]}" \
      --wire-dir "$ROW/cache/wire" --threads "$THREADS" --label after
    ;;
  smoke)
    run journal none -- --phase journal "${JOURNAL[@]}" --out "$RESULTS" --label smoke
    ;;
esac
echo "=== done $(date -Is)"
echo "=== results $RESULTS"
"$PYTHON" - "$RESULTS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def show(r, key):
    print(f"{r.get('label','?'):>18} {key:>12} wall={r.get('wall_seconds')} cpu={r.get('cpu_seconds')} t={r.get('threads')} "
          f"digest={str(r.get('identity_digest', r.get('capture_sha256','')))[:16]}")
for r in rows:
    if r.get('phase') == 'headafter':
        for key in ('seal', 'hold', 'commitments'):
            show({**r[key], 'label': r['label'], 'threads': r['threads']}, key)
        print(f"{'':>18} {'head total':>12} wall={r['wall_seconds']}")
    else:
        show(r, r.get('name', r.get('phase', '?')))
PY
