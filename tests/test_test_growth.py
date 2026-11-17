"""A PR above the test-growth limit needs a reason; a queue draft counts each PR alone."""
import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_test_growth', ROOT / 'tools/ci/test_growth.py')
growth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(growth)


def git(repo, *arguments):
    return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *arguments], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


def pr(repo, base, name, lines, message):
    git(repo, 'checkout', '-q', '-b', name, base)
    (repo / 'tests').mkdir(exist_ok=True)  # the base tracks no tests/ directory
    (repo / 'tests' / f'test_{name}.py').write_text('x = 1\n' * lines)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', message)
    return name


def test_only_the_pr_above_the_limit_needs_a_reason(tmp_path, monkeypatch):
    (tmp_path / 'tests').mkdir()
    git(tmp_path, 'init', '-q', '-b', 'main')
    git(tmp_path, 'commit', '-q', '--allow-empty', '-m', 'base')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    small = pr(tmp_path, base, 'small', growth.LIMIT, 'small')
    large = pr(tmp_path, base, 'large', growth.LIMIT + 1, 'large')
    said = pr(tmp_path, base, 'said', growth.LIMIT + 1, 'said\n\nTest-lines: one fixture table per format')
    monkeypatch.chdir(tmp_path)
    assert growth.refusals(base, large) == [
        f'this PR adds {growth.LIMIT + 1} net lines to tests/*.py (limit {growth.LIMIT}); '
        'add a "Test-lines: <reason>" trailer']
    assert growth.refusals(base, said) == []
    git(tmp_path, 'checkout', '-q', '-b', 'draft', base)
    for number, branch in enumerate((small, large, said), 1):
        git(tmp_path, 'merge', '-q', '--no-ff', '-m', f'Merge of #{number}', branch)
    assert [line.split(' adds')[0] for line in growth.refusals(base)] == ['Merge of #2']
