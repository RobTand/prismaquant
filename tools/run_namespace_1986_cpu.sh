#!/bin/bash
# Run only as an admitted native payload with private bootstrap destinations.
set -euo pipefail
base=${1:?base interpreter required}
work=${2:?new private work directory required}
clone=${3:?read-only Tessera Git clone required}
shift 3
: "${FLUX_JOB_ID:?native admission required}" "${FLUX_SCRATCH:?native scratch required}"
root=$(git rev-parse --show-toplevel)
job_scratch="$FLUX_SCRATCH/${FLUX_JOB_ID/#ƒ/f}"
"$base" -I -B - "$work" "$job_scratch" "$@" <<'PY'
from pathlib import Path
import sys
work, scratch = map(Path, sys.argv[1:3])
if not work.is_absolute() or work != work.resolve() or not work.is_relative_to(scratch.resolve()) or work == scratch.resolve():
    raise SystemExit('new work directory must be canonical and inside this native job scratch')
if any(not value.startswith('tests/') or value.startswith('-') for value in sys.argv[3:]):
    raise SystemExit('only explicit test selectors are accepted')
PY
mkdir "$work"
export TMPDIR="$work/tmp" PYTHONDONTWRITEBYTECODE=1
mkdir "$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1
"$base" -I -B -m venv "$work/venv"
py="$work/venv/bin/python"
"$base" -I -B - "$work/dependency-links" <<'PY'
from pathlib import Path
import sys, sysconfig
source, destination = Path(sysconfig.get_path('purelib')), Path(sys.argv[1])
destination.mkdir()
for entry in source.iterdir():
    name = entry.name.lower()
    if (any(part in name for part in ('tessera', 'prismabuild', 'pip-', 'pip.'))
            or entry.name == 'pip' or entry.suffix == '.pth' or entry.name == '__pycache__'):
        continue
    (destination / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
PY
site=$("$py" -I -B -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$work/dependency-links" > "$site/zz_private_dependency_base.pth"
"$base" -s -B "$root/tools/provision_tessera_pin.py" --python "$py" --prefix "$work/venv" \
  --clone "$clone" --pins-root "$work/pins" --staging-root "$work/build"
"$base" -s -B "$root/tools/check_tessera_dev_install.py" --python "$py" --clone "$clone"
export PYTHONPATH="$root:$root/tests" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
if [[ $# == 0 ]]; then set -- tests/test_tessera_campaign_namespace_1986.py; fi
"$py" -s -B -m pytest -v -o addopts= -p pytest_timeout --timeout=90 \
  --basetemp="$work/pytest" "$@"
