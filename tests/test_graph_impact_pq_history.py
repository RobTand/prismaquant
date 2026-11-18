"""PQ edge attribution uses real git history, never persisted edge identities."""
import json
from pathlib import Path
import pickle
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
    put(tmp_path, str(pq.ALLOWLIST), '{}')
    commit(tmp_path, 'source')
    return tmp_path


def freeze(root, mode='strict'):
    config = pq.record(pq.load_snapshot(root))
    config.pop('edge_groups', None)
    config['test_to_test'] = mode
    put(root, str(pq.BASELINE), json.dumps(config))
    return commit(root, 'record numeric limits')


def test_warm_cache_matches_uncached_snapshot(repo, monkeypatch):
    source = ('import hashlib\n'
              'def digest_parts(parts):\n'
              '    result = []\n'
              '    for part in parts:\n'
              '        digest = hashlib.sha256(part)\n'
              '        if digest is not None:\n'
              '            result.append(digest.hexdigest())\n'
              '    return tuple(result)\n')
    put(repo, 'prismaquant/owner.py', source)
    put(repo, 'prismaquant/peer.py', source)
    git(repo, 'add', '.')
    monkeypatch.delenv('GRAPH_IMPACT_CACHE', raising=False)
    expected = pq.load_snapshot(repo)
    assert expected['duplication']['near_duplicates'] == [
        ['prismaquant/owner.py::digest_parts', 'prismaquant/peer.py::digest_parts']]
    monkeypatch.setenv('GRAPH_IMPACT_CACHE', str(repo / '.cache'))
    assert pq.load_snapshot(repo) == expected

    def refuse_recompute(path, *args, **kwargs):
        pytest.fail(f'Recomputed cache entry for {path}')

    monkeypatch.setattr(pq.build, 'source_facts', refuse_recompute)
    monkeypatch.setattr(pq.duplication_inventory, 'file_inventory', refuse_recompute)
    assert pq.load_snapshot(repo) == expected


def test_warm_cache_preserves_surrogate_code_points(repo, monkeypatch):
    source = 'tests/test_a.py'
    target = 'tests/test_\U0001f600.py'
    surrogates = 'tests/test_\ud83d\ude00.py'
    put(repo, source, f'import runpy\nrunpy.run_path({surrogates!r})\n')
    put(repo, target, '')
    monkeypatch.setenv('GRAPH_IMPACT_CACHE', str(repo / '.cache'))

    cold = pq.build.Evidence(repo, [source, target])
    warm = pq.build.Evidence(repo, [source, target])
    assert surrogates in cold.facts[source].loaders
    assert target not in cold.of(source)[0]
    assert warm.syntax[source] == cold.syntax[source]
    assert warm.of(source) == cold.of(source)


@pytest.mark.parametrize('source', [
    'import importlib as loader\nloader.import_module(name)\n',
    'from importlib import import_module as load\nload(name=name)\n',
    '__import__(name)\n',
    'from runpy import run_path\nrun_path(path_name=name)\n',
    'import runpy\nrunpy.run_module(name)\n',
])
def test_tool_hash_rejects_computed_imports(tmp_path, source):
    put(tmp_path, 'tools/ci/graph_impact_pq.py', 'from tools.ci import helper\n')
    put(tmp_path, 'tools/ci/helper.py', source)
    with pytest.raises(ValueError, match=r'Computed import in tools/ci/helper\.py'):
        pq.build.tool_hash(tmp_path)


def test_tool_source_change_invalidates_cached_file_facts(repo, monkeypatch):
    tools = repo / 'tool-copy'
    for path, source in pq.build.tool_sources(ROOT).items():
        target = tools / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    monkeypatch.setattr(pq.build, 'ROOT', tools)
    monkeypatch.setenv('GRAPH_IMPACT_CACHE', str(repo / '.cache'))
    expected = pq.load_snapshot(repo)
    inputs = pq.build.tool_sources(tools)
    helper = next(path for path in inputs if Path(path).name == 'gate_paths.py')
    old_hash = pq.build.tool_hash(tools)
    (tools / helper).write_bytes(inputs[helper] + b'\n# Change the tool source.\n')
    assert pq.build.tool_hash(tools) != old_hash

    collected = set()

    def record(owner, name):
        collect = getattr(owner, name)

        def collect_and_record(path, *args, **kwargs):
            if path == 'prismaquant/owner.py':
                collected.add(name)
            return collect(path, *args, **kwargs)

        monkeypatch.setattr(owner, name, collect_and_record)

    record(pq.build, 'source_facts')
    record(pq.duplication_inventory, 'file_inventory')
    assert pq.load_snapshot(repo) == expected
    assert collected == {'source_facts', 'file_inventory'}


def test_changed_source_bytes_invalidate_cached_file_facts(repo, monkeypatch):
    monkeypatch.setenv('GRAPH_IMPACT_CACHE', str(repo / '.cache'))
    before = pq.load_snapshot(repo)
    assert 'prismaquant/owner.py' in before['reach']['tests/test_a.py']
    assert before['duplication']['primitive_digest_sites'] == []

    put(repo, 'tests/test_a.py', 'def test_sample():\n    assert True\n')
    put(repo, 'prismaquant/owner.py', 'import hashlib\nVALUE = hashlib.sha256(b"data")\n')
    after = pq.load_snapshot(repo)
    assert 'prismaquant/owner.py' not in after['reach']['tests/test_a.py']
    assert after['duplication']['primitive_digest_sites'] == ['hashlib:prismaquant/owner.py::<module>']


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


@pytest.mark.parametrize('base_scc,head_scc', [(2, 1), (1, 2)])
def test_scc_below_baseline_keeps_gains_against_base(repo, capsys, base_scc, head_scc):
    graphs = {size: pq.core.measure({}, {('a', 'b')} | ({('b', 'a')} if size == 2 else set()))
              for size in (1, 2)}
    for graph in graphs.values():
        graph['duplication'] = pq.duplication_inventory.scan(repo)
    before, after = graphs[base_scc], graphs[head_scc]
    path = repo / pq.BASELINE
    config = {**pq.core.baseline_record(before), 'analyzer_sha256': pq.file_sha256hex(pq.ROOT / pq.ANALYZER)}
    config['figures']['scc_max'] = 3  # An unrecorded base gain stays legal while no ratchet input changes.
    before.update(recorded_baseline=config, recorded_inputs=pq.ratchet_digests(pq.ROOT))
    put(repo, str(pq.BASELINE), json.dumps(config))
    assert pq.check(before, after, path, repo / pq.ALLOWLIST) == int(head_scc > base_scc)
    assert ('scc_max rose from 1 to 2 against the base commit' in capsys.readouterr().err) == (head_scc > base_scc)
    assert json.loads(path.read_text()) == config


def test_a_head_recorded_before_a_cut_passes_after_it(repo, capsys):
    graphs = {size: pq.core.measure({}, {('a', 'b')} | ({('b', 'a')} if size == 2 else set())) for size in (1, 2)}
    for graph in graphs.values():
        graph['duplication'] = pq.duplication_inventory.scan(repo)
    # The head recorded its baseline while the cycle was 2; a cut on main made it 1 before the head's draft ran.
    stale = {**pq.core.baseline_record(graphs[2]), 'analyzer_sha256': pq.file_sha256hex(pq.ROOT / pq.ANALYZER)}
    before, after = graphs[1], dict(graphs[1])
    before.update(recorded_baseline={**stale, 'analyzer_sha256': 'older'}, recorded_inputs=pq.ratchet_digests(pq.ROOT))
    put(repo, str(pq.BASELINE), json.dumps(stale))
    assert pq.check(before, after, repo / pq.BASELINE, repo / pq.ALLOWLIST) == 0, capsys.readouterr().err
    # A recorded baseline holds no scc_max to go stale: the live head-against-base rule is the only one.
    assert all('scc_max' not in pq.numeric_baseline(stale)[key] for key in ('figures', 'tolerances'))


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


@pytest.mark.parametrize('budget', [None, 0])
def test_no_history_fails_closed_and_compares_no_edges(repo, capsys, budget):
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
    assert code == 1 and 'no base graph' in output.err, output.out + output.err
    assert 'no base available' in output.out and (budget is None) is ('exceeds budget 0' not in output.err)
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


@pytest.mark.parametrize('remaining', [1, 2])
def test_duplication_shrink_and_return_without_fixture_edit(repo, capsys, remaining):
    body = ('import hashlib\ndef shared(rows):\n    total = 0\n    for row in rows:\n'
            '        if row.ready:\n            total += row.size\n'
            '        else:\n            total -= row.size\n'
            '    return hashlib.sha256(str(total).encode()).hexdigest()\n')
    for index in range(remaining + 1):
        put(repo, f'prismaquant/copy_{index}.py', body)
    fixture = 'tests/fixtures/ceiling.json'
    ceiling = json.dumps(pq.duplication_inventory.scan(repo))
    put(repo, fixture, ceiling)
    original = freeze(repo)
    removed = repo / f'prismaquant/copy_{remaining}.py'
    removed.unlink()
    (repo / 'tests/moved').mkdir()
    git(repo, 'mv', 'tests/test_a.py', 'tests/moved/test_a.py')
    base = commit(repo, 'remove copy and move test')
    code, output = check(repo, original, capsys)
    assert code == 0, output
    removed.write_text(body)
    code, output = check(repo, base, capsys)
    assert code == 1, output
    for kind in ('near_duplicates', 'same_name_helpers', 'primitive_digest_sites'):
        assert f'new {kind} against the base commit' in output
    assert (repo / fixture).read_text() == ceiling


def test_formatted_path_keeps_all_sources_in_its_declared_directory(tmp_path):
    sources = {
        'tests/test_reads.py': 'from pathlib import Path\n'
                               'ROOT = Path(__file__).resolve().parents[1]\n'
                               'path = ROOT / "tools/dynamic/%s.py" % "a"\n',
        'tools/dynamic/a.py': 'VALUE = 1\n',
        'tools/dynamic/b.py': 'VALUE = 2\n',
        'tools/other.py': 'VALUE = 3\n',
    }
    for path, text in sources.items():
        put(tmp_path, path, text)
    live = pq.load_snapshot(tmp_path, files=sorted(sources),
                            population=['tests/test_reads.py'], legacy=False)
    assert live['reach'] == {'tests/test_reads.py': ['tools/dynamic/a.py', 'tools/dynamic/b.py']}
    assert live['dependents'] == {'tools/dynamic/a.py': 1, 'tools/dynamic/b.py': 1, 'tools/other.py': 0}
    assert {path: (tmp_path / path).read_text() for path in sources} == sources


def test_graph_snapshots_built_for_another_base_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pq, 'base_revision', lambda *args, **kwargs: 'b' * 40)
    for name in ('base', 'head'):
        (tmp_path / f'{name}.pickle').write_bytes(pickle.dumps(('a' * 40, {})))
    with pytest.raises(SystemExit, match='graph snapshots are for base'):
        pq.main(['--root', str(tmp_path), '--snapshots', str(tmp_path)])


def test_computed_directory_paths_keep_exact_source_dependencies(tmp_path):
    a, b = 'tools/dynamic/a.py', 'tools/dynamic/b.py'
    cases = (
        ('Path(os.path.dirname(str(ROOT / "tools/dynamic/a.py"))) / unknown', [a, b]),
        ('os.path.join(ROOT, "tools/dynamic", unknown)', [a, b]),
        ('(ROOT / "tools/dynamic/a.py").with_suffix(".json") / "ignored"', [a, b]),
        ('ROOT / "tools/dynamic/a.py"', [a]),
    )
    base = {a: 'VALUE = 1\n', b: 'VALUE = 2\n', 'tools/other.py': 'VALUE = 3\n'}
    test_path = 'tests/test_reads.py'
    for expression, expected in cases:
        sources = {**base, test_path: 'from pathlib import Path\nimport os\n'
                                     'ROOT = Path(__file__).resolve().parents[1]\n'
                                     f'path = {expression}\n'}
        for path, text in sources.items():
            put(tmp_path, path, text)
        live = pq.load_snapshot(tmp_path, files=sorted(sources),
                                population=[test_path], legacy=False)
        assert live['reach'] == {test_path: expected}
        assert live['dependents'] == {path: int(path in expected) for path in base}
        assert {path: (tmp_path / path).read_text() for path in sources} == sources
