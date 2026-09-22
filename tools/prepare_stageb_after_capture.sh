#!/usr/bin/env bash
# Publish the GLM Stage B metadata generation after Stage A's capture completes.
#
# This submits one CPU action through PrismaBuild that runs
# tools/prepare_extended_joint_quanta.py from a snapshot of this checkout. It
# publishes metadata only (catalog extension, extended parent manifest, layer
# records and a launch recipe under FRESH_METADATA_ROOT); it submits no Stage B
# quantum. The coordinator runs the recipe's dispatcher argv afterwards.
#
# Arguments:
#   PAIR_JSON PAIR_SHA        catalog-pair-inputs.json written by
#                             tools/assemble_t4_overlay.py, and its SHA-256
#   CAPTURE_JSON CAPTURE_SHA  the complete Stage A adjoint-capture.json, and
#                             its SHA-256
#   FRESH_METADATA_ROOT       a new directory for the published metadata
#
# Environment overrides:
#   PQ_CHECKOUT    checkout to snapshot (default: the one holding this script)
#   PQ_PYTHON      interpreter on the target box (default: the pinned venv
#                  named after tools/resolve_tessera_dev_pin.py's commit)
#   STAGEB_SPEC, STAGEB_SPEC_SHA256
#                  reviewed Stage B container spec. Its
#                  PRISMAQUANT_STAGED_RANGE_WAIT_S must be below the 900 s
#                  chunk grace; the preflight refuses the spec otherwise.
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo 'usage: prepare_stageb_after_capture.sh PAIR_JSON PAIR_SHA CAPTURE_JSON CAPTURE_SHA FRESH_METADATA_ROOT' >&2
  exit 2
fi
checkout=${PQ_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
pin=$(python3 "$checkout/tools/resolve_tessera_dev_pin.py")
python=${PQ_PYTHON:-/home/rob/venvs/pq-pb461728e4-tessera-${pin:0:8}/bin/python}
campaign=/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913
panel=$campaign/allocation/joint-panel
spec=${STAGEB_SPEC:-$campaign/campaign-final-union-20260922/stage-b-spec-local-scratch.json}
spec_sha=${STAGEB_SPEC_SHA256:-a52b53594cae071dd0f38616674acd9072b4b85f3ebd447f8fbdfd3aa1f8e2b9}
exec python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd "$checkout" --tag gb10 --cpus 4 --demand mem_gb=20 --priority -10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --timeout-s 1800 --wait-s 2400 -- \
  "$python" -m tools.prepare_extended_joint_quanta \
  --pair-inputs "$1" --pair-inputs-sha256 "$2" \
  --adjoint-receipt "$3" --adjoint-receipt-sha256 "$4" \
  --metadata-root "$5" \
  --parent-manifest "$panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/data-manifests/prismaquant.tessera_joint_aura.run.json.gz" \
  --parent-manifest-sha256 d6481239715c44c8f3c9972708f5fcc8f533d175f49a2163deba9cdb6e0a3f55 \
  --derivation "$panel/stage-a-recovery-20260920/records/derivation.json" \
  --derivation-sha256 50bc8899938b1f6e61bc34efdd7752ab8c8a016889a3994481de3f3848ea79e3 \
  --spec "$spec" --spec-sha256 "$spec_sha"
