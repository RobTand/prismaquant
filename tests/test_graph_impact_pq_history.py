"""PQ edge attribution uses real git history, never persisted edge identities."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.ci import graph_impact_pq as pq

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def commit(root, message):
    git(root, 'add', '.')
    git(root, 'commit', '-qm', message)
    return git(root, 'rev-parse', 'HEAD')


def put(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key in ('GATE_BASE', 'GATE_HEAD', 'GATE_MODE', 'GATE_GRAPH_BASE_MODE'):
        monkeypatch.delenv(key, raising=False)
    git(tmp_path, 'init', '-qb', 'main')
    git(tmp_path, 'config', 'user.email', 'test@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Test')
    put(tmp_path, 'prismaquant/__init__.py', '')
    put(tmp_path, 'prismaquant/owner.py', 'VALUE = 1\n')
    put(tmp_path, 'tests/test_a.py', 'from prismaquant import owner\n' + '# padding for rename detection\n' * 12)
    put(tmp_path, 'tests/test_b.py', 'from prismaquant import owner\n' + '# distinct target padding\n' * 12)
    put(tmp_path, 'tests/conftest.py', '')
    put(tmp_path, 'tools/ci/gate_durations.json', '{"files": {}}')
    put(tmp_path, 'tools/ci/gate_quarantine.json', '{}')
    put(tmp_path, str(pq.ALLOWLIST), '{}')
    commit(tmp_path, 'source')
    return tmp_path


def freeze(root, mode='strict'):
    config = pq.record(pq.load_snapshot(root))
    config.pop('edge_groups', None)
    config['test_to_test'] = mode
    put(root, str(pq.BASELINE), json.dumps(config))
    return commit(root, 'record numeric limits')


def check(root, base, capsys, *args):
    result = pq.main(['--root', str(root), '--base', base, *args])
    output = capsys.readouterr()
    return result, output.out + output.err


@pytest.mark.parametrize('existing', ['test', 'conftest'])
def test_base_edge_missing_from_stored_list_passes(repo, capsys, existing):
    if existing == 'test':
        with (repo / 'tests/test_a.py').open('a') as out:
            out.write('from tests import test_b\n')
    else:
        put(repo, 'tests/conftest.py', 'from prismaquant import owner\n')
    commit(repo, 'pre-existing edge')
    base = freeze(repo)
    put(repo, 'README.md', 'unrelated documentation\n')
    commit(repo, 'unrelated head')
    code, output = check(repo, base, capsys)
    assert code == 0, output


def test_stored_edge_identities_are_ignored(repo, capsys):
    base = freeze(repo)
    config = json.loads((repo / pq.BASELINE).read_text())
    config['edge_groups'] = {'conftest_source_imports': ['stale -> imaginary']}
    put(repo, str(pq.BASELINE), json.dumps(config))
    commit(repo, 'obsolete stored identities')
    code, output = check(repo, base, capsys)
    assert code == 0, output


def test_same_basename_move_with_edits_preserves_base_edges(repo, capsys):
    with (repo / 'tests/test_a.py').open('a') as out:
        out.write('from tests import test_b\n')
    commit(repo, 'old sibling import')
    base = freeze(repo)
    (repo / 'tests/own_process').mkdir()
    for name in ('test_a.py', 'test_b.py'):
        git(repo, 'mv', 'tests/' + name, 'tests/own_process/' + name)
    path = repo / 'tests/own_process/test_a.py'
    path.write_text(path.read_text().replace('from tests import test_b', 'from tests.own_process import test_b') + '# edited while moving\n')
    head = commit(repo, 'move and edit tests')
    assert 'R' in git(repo, 'diff', '-M', '--name-status', base, head)
    code, output = check(repo, base, capsys)
    assert code == 0, output
    assert 'new tests 0' in output and 'new edges 0' in output


@pytest.mark.parametrize('mode,deepens,passes', [('strict', False, False),
                                               ('reach', False, True), ('reach', True, False)])
def test_new_test_edge_uses_strict_or_reach_policy(repo, capsys, mode, deepens, passes):
    base = freeze(repo, mode)
    with (repo / 'tests/test_a.py').open('a') as out:
        out.write('from tests import test_b\n')
    if deepens:
        put(repo, 'prismaquant/deep.py', 'VALUE = 2\n')
        with (repo / 'tests/test_b.py').open('a') as out:
            out.write('from prismaquant import deep\n')
    commit(repo, 'new test edge')
    code, output = check(repo, base, capsys)
    assert (code == 0) is passes, output
    if not passes:
        assert 'new test_to_test' in output
    if deepens:
        assert 'deepens package' in output


def test_reach_mode_enforces_total_edge_growth(repo, capsys):
    for index in range(6):
        put(repo, f'tests/test_target_{index}.py', 'from prismaquant import owner\n')
    commit(repo, 'existing targets')
    base = freeze(repo, 'reach')
    with (repo / 'tests/test_a.py').open('a') as out:
        for index in range(6):
            out.write(f'from tests import test_target_{index}\n')
    commit(repo, 'six new edges')
    code, output = check(repo, base, capsys)
    assert code == 1 and 'maximum growth 5' in output, output


def test_new_conftest_source_import_fails(repo, capsys):
    base = freeze(repo, 'reach')
    put(repo, 'tests/conftest.py', 'from prismaquant import owner\n')
    commit(repo, 'new conftest edge')
    code, output = check(repo, base, capsys)
    assert code == 1 and 'new conftest_source_imports edge' in output, output


def test_basename_change_is_delete_and_add(repo, capsys):
    with (repo / 'tests/test_a.py').open('a') as out:
        out.write('from tests import test_b\n')
    commit(repo, 'existing sibling edge')
    base = freeze(repo)
    git(repo, 'mv', 'tests/test_a.py', 'tests/test_renamed.py')
    commit(repo, 'change basename')
    code, output = check(repo, base, capsys)
    assert code == 1 and 'tests/test_renamed.py -> tests/test_b.py' in output, output


@pytest.mark.parametrize('mode', ['pr', 'batch'])
def test_base_selection_distinguishes_pr_merge_base_and_batch_base(repo, monkeypatch, mode):
    base = freeze(repo)
    git(repo, 'checkout', '-qb', 'feature')
    put(repo, 'README.md', 'feature\n')
    commit(repo, 'feature head')
    git(repo, 'checkout', '-q', 'main')
    put(repo, 'CHANGELOG.md', 'new base branch commit\n')
    target = commit(repo, 'advance base')
    git(repo, 'checkout', '-q', 'feature')
    monkeypatch.setenv('GATE_BASE', target)
    monkeypatch.setenv('GATE_GRAPH_BASE_MODE', mode)
    assert pq.base_revision(repo) == (base if mode == 'pr' else target)


def test_baseline_write_keeps_no_edge_identities(repo, capsys):
    base = freeze(repo)
    code, output = check(repo, base, capsys, '--write-baseline')
    assert code == 0, output
    assert 'edge_groups' not in json.loads((repo / pq.BASELINE).read_text())


def test_committed_baseline_contains_only_limits():
    config = json.loads((ROOT / pq.BASELINE).read_text())
    assert set(config) == {'schema', 'figures', 'tolerances', 'reach_budget', 'test_to_test',
                           'test_to_test_count', 'test_to_test_strict_by', 'analyzer_sha256'}
    assert not (ROOT / 'tests/fixtures/import_fanout_baseline.json').exists()


def test_removing_a_base_edge_needs_no_list_edit(repo, capsys):
    with (repo / 'tests/test_a.py').open('a') as out:
        out.write('from tests import test_b\n')
    commit(repo, 'existing sibling edge')
    base = freeze(repo)
    path = repo / 'tests/test_a.py'
    path.write_text(path.read_text().replace('from tests import test_b\n', ''))
    commit(repo, 'remove edge')
    code, output = check(repo, base, capsys)
    assert code == 0, output


@pytest.mark.parametrize('budget,passes', [(None, True), (0, False)])
def test_no_history_with_edges_enforces_only_absolute_limits(repo, capsys, budget, passes):
    import shutil
    with (repo / 'tests/test_a.py').open('a') as out:
        out.write('from tests import test_b\n')
    put(repo, 'tests/conftest.py', 'from prismaquant import owner\n')
    commit(repo, 'existing edges')
    freeze(repo)
    config = json.loads((repo / pq.BASELINE).read_text())
    if budget is not None:
        config['reach_budget'] = budget
        put(repo, str(pq.BASELINE), json.dumps(config))
    shutil.rmtree(repo / '.git')
    code = pq.main(['--root', str(repo)])
    output = capsys.readouterr()
    assert (code == 0) is passes, output.out + output.err
    assert 'no base available' in output.out
    assert 'new test_to_test' not in output.err and 'new conftest' not in output.err


def test_workflow_supplies_exact_base_for_authenticated_batches_only():
    import yaml
    workflow = yaml.safe_load((ROOT / '.github/workflows/gate.yml').read_text())
    assert workflow['env']['GATE_BASE'] == '${{ github.event.pull_request.base.sha }}'
    mode = workflow['env']['GATE_GRAPH_BASE_MODE']
    for required in ('head.repo.full_name == github.repository', 'mergify/merge-queue/',
                     "user.login == 'mergify[bot]'", 'user.id == 37929162', "'batch'", "'pr'"):
        assert required in mode
    assert 'gate-canary' not in mode
