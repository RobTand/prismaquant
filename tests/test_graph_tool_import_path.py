import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROBE = """
from importlib.machinery import PathFinder
import os
import runpy
import sys

standard = PathFinder.find_spec('inspect')

def report_resolution():
    print(PathFinder.find_spec('inspect').origin == standard.origin)

class InspectFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'inspect':
            report_resolution()
            return standard

# Load only the standard library copy of inspect.
sys.meta_path.insert(0, InspectFinder())
if sys.argv[2] == 'early-root':
    sys.path[:0] = ['', os.getcwd()]
runpy.run_path(sys.argv[1])
report_resolution()
"""


@pytest.mark.parametrize('tool', [
    'tools/ci/graph_impact_pq.py',
    'tools/bazel/build_files.py',
    'tools/ci/gate_select.py',
])
@pytest.mark.parametrize('startup', ['safe-path', 'early-root'])
def test_a_root_file_named_like_a_stdlib_module_does_not_shadow_it(tmp_path, tool, startup):
    checkout = Path(__file__).resolve().parents[1]
    for path in (
        'tools/ci/graph_impact_pq.py',
        'tools/bazel/build_files.py',
        'tools/ci/gate_select.py',
        'tools/ci/gate_paths.py',
        'tools/ci/graph_impact.py',
        'tools/ci/graph_impact_imports.py',
        'tools/duplication_inventory.py',
        'tools/__init__.py',
        'prismaquant/__init__.py',
        'prismaquant/digests.py',
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / path, target)
    (tmp_path / 'inspect.py').touch()

    result = subprocess.run([sys.executable, '-IS', '-c', PROBE, tool, startup],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['True', 'True'], 'A root file took the place of inspect.'
