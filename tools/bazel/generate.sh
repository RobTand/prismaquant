#!/usr/bin/env bash
# Regenerate every derived Bazel input: pip requirements from what the hosted
# gate installs, then every BUILD.bazel from PrismaQuant's dependency evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 tools/bazel/requirements.py
python3 tools/bazel/build_files.py
