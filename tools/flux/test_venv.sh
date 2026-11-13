#!/bin/bash
# Print the path of the CPU test venv that holds the gate's hashed requirements.
# Build it first if it is absent. Runs only inside a Flux job.
set -euo pipefail
: "${FLUX_JOB_ID:?test_venv.sh runs only inside a Flux job}"
cd "$(dirname "$0")/../.."
req=tools/bazel/requirements_linux_x86_64.txt
V=$HOME/venvs/pq-test-$(sha256sum "$req" | cut -c1-12)
if [ ! -x "$V/bin/python" ]; then
  # The gate's Bazel toolchain and its torch index are the one source of each.
  py=$(sed -n 's/^python\.toolchain(python_version = "\(.*\)")$/\1/p' MODULE.bazel)
  index=$(sed -n 's/^# torch: \([^ ]*\).*/\1/p' "$req")
  [ -n "$py" ] && [ -n "$index" ] || { echo "test_venv: no Python version or torch index" >&2; exit 2; }
  tmp=$V.tmp.$FLUX_JOB_ID
  trap 'rm -rf "$tmp"' EXIT
  rm -rf "$tmp"
  uv=$HOME/.local/bin/uv
  "$uv" venv --no-config --relocatable --python "$py" "$tmp" >&2
  "$uv" pip sync --no-config --require-hashes --python "$tmp/bin/python" \
    --extra-index-url "$index" --index-strategy unsafe-best-match "$req" >&2
  # A concurrent job can win the rename; then use its venv.
  mv -T "$tmp" "$V" 2>/dev/null || [ -x "$V/bin/python" ] || exit 1
  echo "test_venv: venv ready after ${SECONDS}s" >&2
fi
echo "$V"
