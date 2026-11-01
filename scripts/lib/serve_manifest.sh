#!/usr/bin/env bash
# ============================================================================
# serve_manifest.sh — shared helper: write the R15 serve fingerprint.
#
# Every gold-lane number is only comparable to another taken on the SAME
# serving stack: loading any CUDA extension shifts allocator addresses, which
# flips alignment-sensitive kernel selection, which moved the same 27B artifact
# between 0.01134 and 0.01328 conf-KL (±17%) on residency alone (§7.4).
#
# The read has to happen INSIDE the container: the measuring client cannot see
# the server's address space, which is exactly why that drift stayed invisible.
# `tools/serve_fingerprint.py` walks the server's /proc/<pid>/maps, records the
# launch argv / image / versions / GPU, and records both fingerprints.
# `tools/kl_ab.py` validates each arm's `serve_fingerprint` and compares
# `performance_stack_fingerprint` across arms.
#
# Never fatal: a serve that came up must not be torn down because a JSON could
# not be written. The refusal lives in the comparator, not here.
#
# Usage (after the READY probe):  write_serve_manifest "$NAME" "$MODEL" [image]
#   NAME  = docker container name
#   MODEL = artifact path AS SEEN INSIDE THE CONTAINER (manifest lands beside it)
# ============================================================================

write_serve_manifest() {
  local name="$1" model="$2" image="${3:-vllm-node:latest}"
  if docker exec "$name" python3 /repo/tools/serve_fingerprint.py write \
       --out "${model}/serve_manifest.json" --image "$image" 2>&1; then
    echo "[serve] serve_manifest.json written beside the artifact"
  else
    echo "[serve] WARN serve_manifest.json NOT written — gold-lane results from"
    echo "[serve]      this serve will carry no serve_fingerprint, and kl_ab.py"
    echo "[serve]      will compare them only with a legacy warning."
  fi
}
