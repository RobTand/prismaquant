#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$PWD/.pb-action-tmp"
mkdir -p "$TMPDIR"
base=/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python
"$base" -m venv --without-pip "$PWD/.pb-b40-runtime"
py="$PWD/.pb-b40-runtime/bin/python"
base_site=$("$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
private_site=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
mkdir -p "$private_site" "$PWD/.pb-cpu-dependency-links"
"$base" - "$base_site" "$PWD/.pb-cpu-dependency-links" <<'PY'
from pathlib import Path
import sys
source, destination = map(Path, sys.argv[1:])
for entry in source.iterdir():
    name = entry.name.lower()
    if (any(part in name for part in ('tessera', 'prismabuild', 'xxhash', 'pytest_timeout', 'pytest-timeout'))
            or entry.suffix == '.pth' or entry.name == '__pycache__'):
        continue
    target = destination / entry.name
    if not target.exists() and not target.is_symlink():
        target.symlink_to(entry, target_is_directory=entry.is_dir())
PY
printf '%s\n' "$PWD/.pb-cpu-dependency-links" > "$private_site/zz_cpu_dependency_base.pth"
pb_commit=$("$base" tools/resolve_prismabuild_dev_pin.py)
"$py" -m pip install --quiet --no-deps --force-reinstall "prismabuild @ git+https://github.com/RobTand/prismabuild.git@$pb_commit" 'tessera-quant @ git+https://github.com/RobTand/tessera.git@b40c93cb73745097e57a1ba4cf5b9eee166c759a' xxhash==3.7.0 pytest-timeout==2.4.0
export PYTHONPATH="$PWD:/mnt/shared/prismabuild-fleet/repo/tools"
"$py" -c 'from pbtest_pins import preflight; raise SystemExit(preflight())'
"$py" -m py_compile tools/dispatch_tessera_campaign.py tools/tessera_campaign_container.py tools/tessera_campaign_namespace.py tools/pq_profile_digest.py tests/test_tessera_campaign_namespace_1986.py
"$py" -m pytest -v --timeout=90 "$@"
