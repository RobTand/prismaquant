"""Actual private interpreter/source controls for the installed Tessera gate."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from test_tessera_pin_provisioner import (
    CONTRACT, _git_clone_with_two_commits, _pin_module, _venv_with_tessera,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/check_tessera_dev_install.py'


def _load():
    return runpy.run_path(str(TOOL))


@pytest.fixture
def install(tmp_path):
    clone, _, commit = _git_clone_with_two_commits(tmp_path)
    pin = _pin_module(tmp_path, commit, hashlib.sha256(CONTRACT).hexdigest())
    python = _venv_with_tessera(tmp_path, 'private', 'VERSION = 2\n')
    site, = python.parent.parent.glob('lib/python*/site-packages')
    info = site / 'tessera_quant-0.1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: tessera-quant\nVersion: 0.1.0\n')
    (info / 'top_level.txt').write_text('tessera\n')
    (info / 'direct_url.json').write_text(json.dumps({'vcs_info': {'vcs': 'git', 'commit_id': commit}}))
    return python, site, info, clone, pin


def check(install):
    python, _, _, clone, pin = install
    return _load()['check_install'](str(python), clone=clone, pin_source=pin)


def test_correct_private_package_is_checked_against_real_git_source(install):
    ok, report = check(install)
    assert ok, report
    assert report['installed']['contract_sha256'] == hashlib.sha256(CONTRACT).hexdigest()
    assert report['installed']['package_path'].endswith('/tessera')
    assert report['expected']['package_files'] > 1


@pytest.mark.parametrize('mutation,reason', [
    ('wrong-commit', 'Git provenance'), ('editable', 'editable'),
    ('changed', 'installed bytes differ'), ('missing', 'installed bytes differ'),
    ('unlisted', 'installed bytes differ'), ('no-owner', 'import ownership'),
    ('ambiguous-owner', 'import ownership'), ('shadow', 'symlinked'),
])
def test_positive_then_real_install_mutations_refuse(install, mutation, reason):
    assert check(install)[0]
    python, site, info, _, _ = install
    if mutation in {'wrong-commit', 'editable'}:
        path = info / 'direct_url.json'
        value = json.loads(path.read_text())
        if mutation == 'wrong-commit':
            value['vcs_info']['commit_id'] = '0' * 40
        else:
            value['dir_info'] = {'editable': True}
        path.write_text(json.dumps(value))
    elif mutation == 'changed':
        (site / 'tessera/encode.py').write_text('VERSION = 99\n')
    elif mutation == 'missing':
        (site / 'tessera/encode.py').unlink()
    elif mutation == 'unlisted':
        (site / 'tessera/foreign.py').write_text('EXTRA = 1\n')
    elif mutation == 'no-owner':
        (info / 'top_level.txt').unlink()
    elif mutation == 'ambiguous-owner':
        other = site / 'foreign-1.dist-info'; other.mkdir()
        (other / 'METADATA').write_text('Metadata-Version: 2.1\nName: foreign\nVersion: 1\n')
        (other / 'top_level.txt').write_text('tessera\n')
    else:
        target = site / 'tessera'
        moved = site / 'other-tessera'
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
    ok, report = check(install)
    assert not ok, report
    assert reason in report['refusal']


def test_local_provenance_needs_independent_exact_bytes(install):
    _, site, info, _, _ = install
    (info / 'direct_url.json').write_text(json.dumps({'url': 'file:///private-source', 'dir_info': {}}))
    assert check(install)[0]
    (site / 'tessera/encode.py').write_text('VERSION = 0\n')
    assert not check(install)[0]


def test_missing_interpreter_is_a_named_refusal(install):
    _, site, _, clone, pin = install
    ok, report = _load()['check_install'](str(site / 'no-python'), clone=clone, pin_source=pin)
    assert not ok and 'FileNotFoundError' in report['refusal']


def test_checker_never_imports_tessera_or_serving(install):
    python, site, _, _, _ = install
    (site / 'tessera/__init__.py').write_text("raise AssertionError('package initialized')\n")
    provisioner = runpy.run_path(str(ROOT / 'tools/provision_tessera_pin.py'))
    identity, absent = provisioner['installed_identity'](str(python))
    assert absent is None and identity['contract_sha256'] == hashlib.sha256(CONTRACT).hexdigest()


def test_guard_reads_the_existing_pin_without_numerical_imports():
    mod = _load()
    printed = subprocess.check_output([sys.executable, '-I', str(ROOT / 'tools/resolve_tessera_dev_pin.py')], text=True).strip()
    assert mod['reviewed_commit']() == printed
