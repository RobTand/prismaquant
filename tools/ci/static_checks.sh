#!/bin/bash
# The merge gate's checks that execute no test: generated inputs (BUILD files, requirements), the queue policy and
# the graph impact with its recorded baseline. The gate's Graph impact job runs this script. Before the `queue` label,
# run it on the PR head merged with the current main: tools/ci/static_checks.sh --merge-main
set -Eeuo pipefail
if [ "${1:-}" = --merge-main ]; then
  git fetch -q origin main
  tree="$(mktemp -d "${TMPDIR:-$HOME/tmp}/static-checks-XXXXXX")"
  git worktree add -q --detach "$tree" HEAD
  trap 'git worktree remove --force "$tree"' EXIT
  git -C "$tree" -c user.name=checks -c user.email=checks@example.invalid merge -q --no-edit origin/main
  cd "$tree"
  export GATE_BASE="$(git rev-parse origin/main)" GATE_GRAPH_BASE_MODE=batch
fi
base="${GATE_BASE:?GATE_BASE names the base commit}"
work="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-$HOME/tmp}}/static-checks-XXXXXX")"
python tools/ci/bazel_generated.py & generated=$!
python tools/ci/gate_select.py --base "$base" --head "$(git rev-parse HEAD)" --mode full --output "$work/plan.json" > /dev/null & policy=$!
python tools/ci/graph_impact_pq.py --write base --snapshots "$work" & graph_base=$!
python tools/ci/graph_impact_pq.py --write head --snapshots "$work" & graph_head=$!
status=0
for job in $generated $policy $graph_base $graph_head; do wait "$job" || status=1; done
[ "$status" = 0 ] && python tools/ci/graph_impact_pq.py --snapshots "$work" || status=1
rm -rf "$work"
exit "$status"
