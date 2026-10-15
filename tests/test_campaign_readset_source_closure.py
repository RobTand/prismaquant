"""Readset producer provenance binds the code that derives paths and phases."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('dependency', [
    'render_cache_paths.py',
    'joint_prewarm_phases.py',
    'joint_replay_frontier.py',
    'joint_run_progress.py',
])
def test_readset_source_digest_changes_with_its_actual_dependency_bytes(tmp_path, dependency):
    source = Path(__file__).resolve().parents[1] / 'prismaquant'
    package = tmp_path / 'prismaquant'
    shutil.copytree(source, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from prismaquant import campaign_data_manifests as owner
root = Path.cwd()
assert Path(owner.__file__).resolve() == root / 'prismaquant' / 'campaign_data_manifests.py'
before = owner.producer_source_sha256()
dependency = root / 'prismaquant' / sys.argv[1]
dependency.write_bytes(dependency.read_bytes() + b'\\n# owned dependency mutation\\n')
after = owner.producer_source_sha256()
assert before != after, 'readset producer provenance ignored changed ' + dependency.name
print('BOUND-READSET-DEPENDENCY')
'''
    result = subprocess.run([sys.executable, '-c', script, dependency], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'BOUND-READSET-DEPENDENCY'
