#!/bin/bash
# Run pytest on the gate's pinned CPU stack inside a Flux job.
# Usage: run_tests.sh LOGDIR [PYTEST_ARGS...]
set -euo pipefail
log=$(realpath -m "${1:?usage: run_tests.sh LOGDIR [PYTEST_ARGS...]}")
shift
cd "$(dirname "$0")/../.."
V=$(bash tools/flux/test_venv.sh)
mkdir -p "$log/tmp"
# torch's cpp_extension finds ninja on PATH, so the venv bin comes first.
export PATH=$V/bin:/usr/bin:/bin PYTHONPATH=. TMPDIR=$log/tmp
exec "$V/bin/python" -m pytest -q -p no:cacheprovider -n 4 -rfE \
  --basetemp="$log/basetemp" "$@"
