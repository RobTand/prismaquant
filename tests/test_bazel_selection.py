"""Cutover selection falls back safely and preserves wheel-level impact."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from prismaquant.digests import bytes_sha256hex

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_bazel_selection', ROOT / 'tools/ci/bazel_selection.py')
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
TESTS = {'//tests:test_consumer', '//tests:test_unrelated'}


def decision(changed, *, impacted=frozenset(), used=frozenset(), deleted=frozenset(), error=None):
    return selection.gate_decision(set(changed), set(deleted), set(impacted), set(used), TESTS,
                                   detection_error=error)


@pytest.mark.parametrize('path', [
    '.github/workflows/gate.yml', '.github/workflows/bazel-hosted.yml', '.bazelrc', '.bazelversion', '.bazelignore',
    'MODULE.bazel', 'MODULE.bazel.lock', 'BUILD.bazel', 'tests/BUILD.bazel',
    'tools/example/BUILD', 'tools/example/defs.bzl', 'tools/bazel/runtime.py',
    'tools/ci/bazel_selection.py', 'tools/ci/bazel_gate.py', 'tools/ci/remote_gate.py',
])
def test_gate_and_bazel_inputs_select_everything(path):
    found = decision({path})
    assert found['selection'] == 'fallback'
    assert found['selected'] == sorted(TESTS)
    assert found['build'] == ['//...']
    assert found['all_scopes'] is True


@pytest.mark.parametrize('path', ['tools/ci/bazel_artifacts.py', 'tools/ci/check_no_secret_leak.py'])
def test_artifact_boundary_changes_force_full_coverage_even_with_narrow_graph_evidence(path):
    found = decision({path}, impacted={'//tests:test_consumer'}, used={path})
    assert found['selection'] == 'fallback'
    assert found['selected'] == sorted(TESTS)
    assert found['all_scopes'] is True


@pytest.mark.parametrize('path', [
    'README.md', 'docs/bazel.md', '.github/workflows/ci.yml', '.github/scripts/check.py',
    '.mergify.yml',
])
def test_unread_other_workflows_and_docs_select_no_tests(path):
    found = decision({path})
    assert found['selected'] == []
    assert found['selection'] == 'irrelevant'
    assert found['all_scopes'] is False


def test_docs_read_by_a_test_select_their_consumer():
    found = decision({'docs/schema.md'}, impacted={'//tests:test_consumer'}, used={'docs/schema.md'})
    assert found['selected'] == ['//tests:test_consumer']


@pytest.mark.parametrize('path', ['uv.lock', 'pyproject.toml', 'pytest.ini'])
def test_lock_and_metadata_use_proven_consumers(path):
    found = decision({path}, impacted={'//tests:test_consumer'}, used={path})
    assert found['selection'] == 'impacted'
    assert found['selected'] == ['//tests:test_consumer']
    assert found['all_scopes'] is False


def test_a_changed_wheel_keeps_unrelated_wheel_consumers_unselected():
    found = decision({'uv.lock'}, impacted={'@@pq_pip_numpy//:pkg', '//:numeric',
                                         '//tests:test_consumer'}, used={'uv.lock'})
    assert found['selected'] == ['//tests:test_consumer']
    assert found['build'] == ['//:numeric']
    assert found['scopes'] == ['//', '//tests']


def test_query_population_preserves_build_rules_and_architecture_lanes():
    def rule(label, kind='py_test', tags=()):
        return {'rule': {'name': label, 'ruleClass': kind,
                         'attribute': [{'name': 'tags', 'stringListValue': list(tags)}]}}
    rows = [rule('//:core', 'py_library'), rule('//tests:cpu'),
            rule('//tests:mixed_gpu', tags=['gpu']),
            rule('//tests:runner', tags=['runner', 'no-remote-exec']),
            rule('//tests:disk', tags=['hosted-arm64']),
            rule('//tests:hosted', tags=['manual', 'gate-lane-hosted']),
            rule('//tests:deferred', tags=['manual', 'gate-lane-root']),
            rule('@@rules_python++pip+pq_pip_numpy//:pkg', 'py_library'),
            {'sourceFile': {'name': '//prismaquant:core.py'}},
            {'sourceFile': {'name': '//:README.md'}}]
    amd = selection.query_population(rows, 'amd64')
    arm = selection.query_population(list(reversed(rows)), 'arm64')
    assert amd['targets'] == ['//:core', '//tests:cpu', '//tests:disk',
                             '//tests:mixed_gpu', '//tests:runner']
    assert amd['tests'] == ['//tests:cpu', '//tests:disk', '//tests:mixed_gpu', '//tests:runner']
    assert arm['targets'] == ['//:core', '//tests:cpu', '//tests:mixed_gpu']
    assert arm['tests'] == ['//tests:cpu', '//tests:mixed_gpu']
    for population in (amd, arm):
        assert population['hosted'] == ['//tests:hosted']
        assert population['hosted_arm64'] == ['//tests:disk']
        assert population['used_sources'] == ['README.md', 'prismaquant/core.py']
        assert population['wheel_repositories'] == ['rules_python++pip+pq_pip_numpy']


@pytest.mark.parametrize('architecture', ['amd64', 'arm64'])
def test_query_population_accepts_protobuf_json_omitted_empty_tag_values(architecture):
    rows = [
        {'rule': {'name': '//:core', 'ruleClass': 'py_library',
                  'attribute': [{'name': 'tags', 'type': 'STRING_LIST'}]}},
        {'rule': {'name': '//tests:cpu', 'ruleClass': 'py_test',
                  'attribute': [{'name': 'tags', 'type': 'STRING_LIST'}]}},
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                  'ruleClass': 'py_library'}},
    ]
    population = selection.query_population(rows, architecture)
    assert population['targets'] == ['//:core', '//tests:cpu']
    assert population['tests'] == ['//tests:cpu']
    assert population['hosted'] == [] and population['hosted_arm64'] == []


@pytest.mark.parametrize('architecture', ['amd64', 'arm64'])
def test_query_population_keeps_package_groups_as_dependency_records(architecture):
    rows = [
        {'type': 'PACKAGE_GROUP', 'packageGroup': {'name': '//visibility:public'}},
        {'type': 'PACKAGE_GROUP', 'packageGroup': {'name': '//visibility:private'}},
        {'rule': {'name': '//:core', 'ruleClass': 'py_library'}},
        {'rule': {'name': '//tests:cpu', 'ruleClass': 'py_test'}},
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                  'ruleClass': 'py_library'}},
    ]
    population = selection.query_population(rows, architecture)
    assert population['targets'] == ['//:core', '//tests:cpu']
    assert population['tests'] == ['//tests:cpu']
    assert population['wheel_repositories'] == ['rules_python++pip+pq_pip_numpy']


@pytest.mark.parametrize('phase', ['command', 'json', 'population'])
def test_revision_query_refusal_names_check_without_private_diagnostics(tmp_path, capsys, phase):
    from types import SimpleNamespace
    def query(command, **kwargs):
        return SimpleNamespace(returncode=1 if phase == 'command' else 0,
            stdout='private-path-token-query' if phase == 'json' else '{}',
            stderr='private-path-token-query')
    with pytest.raises(selection.SelectionError):
        selection.query_revision(tmp_path, tmp_path / 'private', 'amd64', runner=query)
    output = capsys.readouterr()
    assert 'Revision query refused at ' + phase in output.err
    assert 'private-path-token-query' not in output.err + output.out


@pytest.mark.parametrize('rows', [
    [], [{'rule': {'name': '//tests:missing_kind'}}],
    [{'rule': {'name': '//tests:bad', 'ruleClass': 'py_test',
               'attribute': [{'name': 'tags', 'stringListValue': 'manual'}]}}],
    [{'sourceFile': {'name': '//:../outside.py'}}],
])
def test_missing_or_malformed_query_population_cannot_select_an_empty_pass(rows):
    with pytest.raises(selection.SelectionError, match='query'):
        selection.query_population(rows, 'amd64')


@pytest.mark.parametrize('path', ['prismaquant/new_module.py', 'tests/test_new.py'])
def test_an_unmapped_python_file_falls_back_instead_of_passing_empty(path):
    found = decision({path})
    assert found['selection'] == 'fallback'
    assert found['selected'] == sorted(TESTS)


def test_an_empty_impact_for_a_read_input_falls_back():
    assert decision({'tests/fixtures/values.json'}, used={'tests/fixtures/values.json'})['build'] == ['//...']


def test_graph_failure_falls_back_to_all_tests():
    found = decision({'README.md'}, error='bazel-diff did not produce a complete graph')
    assert found['build'] == ['//...']
    assert found['all_scopes'] is True


def test_deleted_inputs_keep_their_impacted_tests():
    found = decision(set(), deleted={'prismaquant/old.py'}, impacted={'//tests:test_consumer'})
    assert found['selected'] == ['//tests:test_consumer']


def test_missing_changed_refs_are_not_an_empty_pass():
    with pytest.raises(selection.SelectionError, match='empty diff'):
        decision(set())


def test_v42_hashes_each_actual_wheel_repository_and_keeps_edges_private(tmp_path):
    hashes = tmp_path / 'hashes.json'
    edges = tmp_path / 'edges.json'
    repos = tmp_path / 'wheel-repos.txt'
    command = selection.generate_hashes_command('java', tmp_path / 'v42.jar', ROOT,
                                                'bazel', hashes, edges, repos)
    assert command[:4] == ['java', '-jar', str(tmp_path / 'v42.jar'), 'generate-hashes']
    assert command[command.index('--fineGrainedHashExternalReposFile') + 1] == str(repos)
    assert command[command.index('--depEdgesFile') + 1] == str(edges)
    assert command[-1] == str(hashes)
    assert '--fineGrainedHashExternalRepos' not in command
    assert '--includeTargetType' in command
    assert '--bazelCommandOptions=--config=ci' in command


def test_missing_test_status_fails_and_budget_warning_cannot_change_status():
    verdict, failed = selection.verdict_of(TESTS, {'//tests:test_consumer': 'PASSED'})
    assert verdict == 'fail'
    assert failed == ['//tests:test_unrelated']
    assert selection.verdict_of({'//tests:test_consumer'}, {'//tests:test_consumer': 'PASSED'}) == ('pass', [])


def test_selection_manifest_preserves_exact_action_population_and_hosted_routes():
    populations = {
        'amd64': {'targets': ['//:core', '//tests:one', '//tests:two'],
                  'tests': ['//tests:one', '//tests:two']},
        'arm64': {'targets': ['//:core', '//tests:one'], 'tests': ['//tests:one']},
    }
    decision = selection.everything(['graph error'], {'//tests:one', '//tests:two', '//tests:hosted'})
    found = selection.selection_manifest(
        decision, populations, hosted={'//tests:hosted'},
        base='b' * 40, head='a' * 40, mode='pr', shard_counts={'amd64': 2})
    assert set(found['architectures']) == {'amd64'}
    assert found['hosted'] == ['//tests:hosted']
    row = found['architectures']['amd64']
    assert row['population'] == ['//:core', '//tests:one', '//tests:two']
    assert row['tests'] == ['//tests:one', '//tests:two']
    assert sorted(label for shard in row['shards'] for label in shard) == row['population']
    assert len(row['shards']) == 2


def test_postmerge_manifest_always_covers_both_architectures_and_hosted_arm64():
    populations = {
        'amd64': {'targets': ['//tests:one', '//tests:disk'],
                  'tests': ['//tests:one', '//tests:disk']},
        'arm64': {'targets': ['//tests:one'], 'tests': ['//tests:one']},
    }
    irrelevant = {'selection': 'irrelevant', 'selected': [], 'build': []}
    found = selection.selection_manifest(
        irrelevant, populations, hosted={'//tests:hosted'}, hosted_arm64={'//tests:disk'},
        base='b' * 40, head='a' * 40, mode='main', shard_counts={'amd64': 2, 'arm64': 4})
    assert set(found['architectures']) == {'amd64', 'arm64'}
    assert found['hosted'] == ['//tests:hosted']
    assert found['hosted_arm64'] == ['//tests:disk']
    assert found['architectures']['amd64']['population'] == ['//tests:disk', '//tests:one']
    assert found['architectures']['arm64']['population'] == ['//tests:one']


def test_empty_docs_plan_and_partial_selection_keep_their_exact_scope():
    populations = {'amd64': {'targets': ['//:core', '//tests:test_consumer',
                                        '//tests:test_unrelated'], 'tests': sorted(TESTS)}}
    for changed, impacted, expected in [({'README.md'}, set(), []),
            ({'prismaquant/core.py'}, {'//:core', '//tests:test_consumer'},
             ['//:core', '//tests:test_consumer'])]:
        selected = decision(changed, impacted=impacted, used={'prismaquant/core.py'})
        found = selection.selection_manifest(selected, populations, hosted=set(),
            base='b' * 40, head='a' * 40, mode='pr', shard_counts={'amd64': 2})
        assert found['architectures']['amd64']['population'] == expected
        assert found['architectures']['amd64']['shards'] == ([] if not expected else [
            ['//tests:test_consumer'], ['//:core']])


@pytest.mark.parametrize('fault', ['missing_arch', 'unknown_selected', 'foreign_test', 'overlap'])
def test_selection_manifest_refuses_incomplete_or_contradictory_population(fault):
    populations = {'amd64': {'targets': ['//tests:one'], 'tests': ['//tests:one']}}
    selected = {'selection': 'fallback', 'selected': ['//tests:one'], 'build': ['//...']}
    hosted = set()
    mode = 'main' if fault == 'missing_arch' else 'pr'
    if fault == 'unknown_selected':
        selected = {'selection': 'impacted', 'selected': ['//tests:missing'], 'build': []}
    if fault == 'foreign_test':
        populations['amd64']['tests'].append('//tests:foreign')
    if fault == 'overlap':
        hosted.add('//tests:one')
    with pytest.raises(selection.SelectionError):
        selection.selection_manifest(selected, populations, hosted=hosted,
            base='b' * 40, head='a' * 40, mode=mode, shard_counts={'amd64': 2, 'arm64': 4})


def test_wheel_repository_inventory_keeps_each_canonical_package_separate():
    found = selection.wheel_repositories([
        '@@rules_python++pip+pq_pip_linux_amd64_numpy//:pkg',
        '@@rules_python++pip+pq_pip_linux_amd64_torch//:pkg',
        '@@rules_python++pip+pq_pip_linux_arm64_numpy//:pkg',
        '@@rules_python++pip+pq_pip_linux_amd64_numpy//:site-packages/numpy/__init__.py',
        '@@rules_python++pip+pq_pip//numpy:pkg',
        '@@rules_python+//python:defs.bzl', '//tests:test_consumer',
    ])
    assert found == ['rules_python++pip+pq_pip_linux_amd64_numpy',
                     'rules_python++pip+pq_pip_linux_amd64_torch',
                     'rules_python++pip+pq_pip_linux_arm64_numpy']


@pytest.mark.parametrize('labels', [[], ['//tests:test_consumer'],
    ['@@rules_python++pip+pq_pip//numpy:pkg']])
def test_missing_canonical_wheel_inventory_refuses_opaque_hub_selection(labels):
    with pytest.raises(selection.SelectionError, match='wheel'):
        selection.wheel_repositories(labels)


def test_bazel_diff_impact_query_uses_both_private_graph_snapshots(tmp_path):
    command = selection.impacted_targets_command('java', tmp_path / 'v42.jar',
        tmp_path / 'head-source', tmp_path / 'pinned-bazel',
        tmp_path / 'base.json', tmp_path / 'head.json', tmp_path / 'impacted.txt')
    assert command == ['java', '-jar', str(tmp_path / 'v42.jar'), 'get-impacted-targets',
        '--workspacePath', str(tmp_path / 'head-source'),
        '--bazelPath', str(tmp_path / 'pinned-bazel'),
        '--targetType', 'Rule',
        '--startingHashes', str(tmp_path / 'base.json'),
        '--finalHashes', str(tmp_path / 'head.json'),
        '--output', str(tmp_path / 'impacted.txt')]


def test_revision_graph_retains_wheel_hashes_and_source_edges_privately(tmp_path, monkeypatch):
    rows = [
        {'rule': {'name': '//tests:test_consumer', 'ruleClass': 'py_test'}},
        {'sourceFile': {'name': '//prismaquant:core.py'}},
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                  'ruleClass': 'py_library'}},
    ]
    calls = []
    monkeypatch.setenv('MERGIFY_TOKEN', 'test-controller-only')
    monkeypatch.setenv('GITHUB_TOKEN', 'test-controller-only')

    def remote_command(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'] == ROOT
        assert 'MERGIFY_TOKEN' not in kwargs['env']
        assert 'GITHUB_TOKEN' not in kwargs['env']
        if command[1] == 'query':
            return SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, rows)))
        assert command[3] == 'generate-hashes'
        wheel_file = Path(command[command.index('--fineGrainedHashExternalReposFile') + 1])
        assert wheel_file.read_text() == '@@rules_python++pip+pq_pip_numpy\n'
        Path(command[-1]).write_text(json.dumps({'hashes': {'//tests:test_consumer': 'Rule#abc'},
                                               'metadata': {'moduleGraphJson': '{}'}}))
        Path(command[command.index('--depEdgesFile') + 1]).write_text(
            json.dumps({'//tests:test_consumer': ['//prismaquant:core.py']}))
        return SimpleNamespace(returncode=0, stdout='private diagnostic')

    graph = selection.collect_revision(ROOT, tmp_path, 'amd64', java='java',
        jar=tmp_path / 'v42.jar', runner=remote_command)
    assert graph['population']['tests'] == ['//tests:test_consumer']
    assert graph['population']['used_sources'] == ['prismaquant/core.py']
    assert graph['hashes'].parent == graph['edges'].parent == graph['query'].parent == tmp_path
    assert json.loads(graph['hashes'].read_text())['hashes']['//tests:test_consumer'] == 'Rule#abc'
    assert json.loads(graph['query'].read_text().splitlines()[0]) == rows[0]
    assert calls[0][:3] == ['bazel', 'query', 'deps(//...) union buildfiles(//...)']
    assert '--output=streamed_jsonproto' in calls[0]
    assert '--keep_going=false' in calls[0]


def test_v42_adapter_qualifies_the_native_sdist_repository_failure_shape(tmp_path):
    # Native ffaecdd3 refused the bare repo as an invalid package path.
    repo = 'rules_python++pip+pq_pip_312__sdist_52ec077b'
    rows = [{'rule': {'name': '//tests:consumer', 'ruleClass': 'py_test'}},
            {'rule': {'name': '@@' + repo + '//:pkg', 'ruleClass': 'py_library'}}]
    population = selection.query_population(rows, 'amd64')
    assert population['wheel_repositories'] == [repo]

    def hashes(command, **kwargs):
        wheels = Path(command[command.index('--fineGrainedHashExternalReposFile') + 1])
        patterns = [line + '//...:all-targets' for line in wheels.read_text().splitlines()]
        assert patterns == ['@@' + repo + '//...:all-targets']
        Path(command[-1]).write_text('{"//tests:consumer":"Rule#hash"}')
        Path(command[command.index('--depEdgesFile') + 1]).write_text('{}')
        return SimpleNamespace(returncode=0)

    selection.collect_revision(ROOT, tmp_path, 'amd64', java='java',
        jar=tmp_path / 'v42.jar', runner=hashes,
        queried={'population': population, 'query': tmp_path / 'query.jsonl'})


@pytest.mark.parametrize('fault', ['query_exit', 'query_invalid', 'missing_hashes', 'empty_hashes'])
def test_revision_graph_refuses_partial_external_evidence(tmp_path, fault):
    def incomplete(command, **kwargs):
        if command[1] == 'query':
            rows = [{'rule': {'name': '//tests:test_consumer', 'ruleClass': 'py_test'}},
                    {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                              'ruleClass': 'py_library'}}]
            return SimpleNamespace(returncode=1 if fault == 'query_exit' else 0,
                stdout='invalid' if fault == 'query_invalid' else '\n'.join(map(json.dumps, rows)))
        Path(command[command.index('--depEdgesFile') + 1]).write_text('{}')
        if fault == 'empty_hashes':
            Path(command[-1]).write_text('{"hashes": {}}')
        return SimpleNamespace(returncode=0, stdout='')

    with pytest.raises(selection.SelectionError, match='graph'):
        selection.collect_revision(ROOT, tmp_path, 'amd64', java='java',
            jar=tmp_path / 'v42.jar', runner=incomplete)


@pytest.mark.parametrize('cached', [None, b'wrong jar', b'pinned v42 jar'])
def test_v42_jar_is_hash_checked_before_use_or_reuse(tmp_path, monkeypatch, cached):
    content = b'pinned v42 jar'
    monkeypatch.setattr(selection, 'BAZEL_DIFF_SHA256', bytes_sha256hex(content))
    path = tmp_path / 'bazel-diff-v42.0.0.jar'
    if cached is not None:
        path.write_bytes(cached)
    requested = []

    def download(url):
        requested.append(url)
        return content

    assert selection.bazel_diff_jar(tmp_path, fetch=download) == path
    assert path.read_bytes() == content
    assert requested == ([] if cached == content else [selection.BAZEL_DIFF_URL])


def test_unverified_jar_is_never_written_or_executed(tmp_path):
    with pytest.raises(selection.SelectionError, match='checksum'):
        selection.bazel_diff_jar(tmp_path, fetch=lambda url: b'untrusted jar')
    assert not (tmp_path / 'bazel-diff-v42.0.0.jar').exists()


@pytest.mark.parametrize('module', ['bazel_selection.py', 'bazel_gate.py', 'remote_gate.py'])
def test_gate_modules_load_without_repository_pythonpath(tmp_path, module):
    # Run's allowlisted child environment intentionally omits PYTHONPATH.
    loaded = subprocess.run([sys.executable, '-I', '-c',
        'import runpy,sys;runpy.run_path(sys.argv[1],run_name="pq_gate_import_probe")',
        str(ROOT / 'tools/ci' / module)], cwd=tmp_path, env={},
        text=True, capture_output=True, check=False)
    assert loaded.returncode == 0, loaded.stderr


@pytest.mark.parametrize('changed,impacted,used,expected_scopes,all_scopes', [
    ({'prismaquant/core.py'}, {'//:core', '//tests:test_consumer'},
     {'prismaquant/core.py'}, ['//', '//tests'], False),
    ({'MODULE.bazel'}, set(), set(), [], True),
])
def test_manifest_preserves_mergify_scope_selection(
        changed, impacted, used, expected_scopes, all_scopes):
    selected = decision(changed, impacted=impacted, used=used)
    populations = {'amd64': {'targets': ['//:core', *sorted(TESTS)],
                             'tests': sorted(TESTS)}}
    found = selection.selection_manifest(selected, populations, hosted=set(),
        base='b' * 40, head='a' * 40, mode='pr')
    assert found['scopes'] == expected_scopes
    assert found['all_scopes'] is all_scopes
    assert found['reason'] == selected['reason']


@pytest.mark.parametrize('changed', ['uv.lock', 'prismaquant/core.py'])
def test_graph_comparison_uses_v42_results_and_keeps_wheel_selection_narrow(tmp_path, changed):
    population = {'tests': sorted(TESTS), 'used_sources': [changed], 'excluded': []}
    base = {'hashes': tmp_path / 'base.json'}
    head = {'hashes': tmp_path / 'head.json', 'population': population}

    def compare(command, **kwargs):
        assert command[3] == 'get-impacted-targets'
        assert command[command.index('--startingHashes') + 1] == str(base['hashes'])
        assert command[command.index('--finalHashes') + 1] == str(head['hashes'])
        assert '--targetType' in command and command[command.index('--targetType') + 1] == 'Rule'
        assert 'BUILDBUDDY_API_KEY' not in kwargs['env']
        assert 'MERGIFY_TOKEN' not in kwargs['env']
        Path(command[command.index('--output') + 1]).write_text('@//tests:test_consumer\n')
        return SimpleNamespace(returncode=0)

    found = selection.compare_graphs(base, head, {changed}, set(), tmp_path,
        java='java', jar=tmp_path / 'v42.jar', root=tmp_path / 'head-source',
        bazel=tmp_path / 'pinned-bazel', runner=compare)
    assert found['selection'] == 'impacted'
    assert found['selected'] == ['//tests:test_consumer']
    assert found['all_scopes'] is False


@pytest.mark.parametrize('fault', ['base_missing', 'diff_exit', 'invalid_output', 'output_missing'])
def test_graph_comparison_errors_fall_back_to_the_complete_queried_head(tmp_path, fault):
    head = {'hashes': tmp_path / 'head.json',
        'population': {'tests': sorted(TESTS), 'used_sources': ['prismaquant/core.py'], 'excluded': []}}
    base = None if fault == 'base_missing' else {'hashes': tmp_path / 'base.json'}

    def failed_comparison(command, **kwargs):
        assert base is not None, 'missing base cannot start a comparison'
        if fault == 'invalid_output':
            Path(command[command.index('--output') + 1]).write_text('not-a-bazel-label\n')
        return SimpleNamespace(returncode=1 if fault == 'diff_exit' else 0)

    found = selection.compare_graphs(base, head, {'prismaquant/core.py'}, set(), tmp_path,
        java='java', jar=tmp_path / 'v42.jar', root=tmp_path / 'head-source',
        runner=failed_comparison)
    assert found['selection'] == 'fallback'
    assert found['selected'] == sorted(TESTS)
    assert found['build'] == ['//...']
    assert found['all_scopes'] is True


def test_impacted_hosted_consumer_survives_query_to_manifest_routing():
    hosted = '//tests:test_hosted_consumer'
    population = selection.query_population([
        {'rule': {'name': '//tests:test_consumer', 'ruleClass': 'py_test'}},
        {'rule': {'name': hosted, 'ruleClass': 'py_test', 'attribute': [
            {'name': 'tags', 'stringListValue': ['manual', 'gate-lane-hosted']}]}},
        {'sourceFile': {'name': '//prismaquant:core.py'}},
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                  'ruleClass': 'py_library'}},
    ], 'amd64')
    selected = selection.gate_decision({'prismaquant/core.py'}, set(), {hosted},
        set(population['used_sources']), set(population['tests']))
    found = selection.selection_manifest(selected, {'amd64': population},
        hosted=population['hosted'], base='b' * 40, head='a' * 40, mode='pr')
    assert found['hosted'] == [hosted]
    assert found['architectures']['amd64']['population'] == []
    assert found['scopes'] == ['//tests']
    assert found['selection'] == 'impacted'


@pytest.mark.parametrize('unknown', [False, True])
def test_impacted_quarantined_rule_is_excluded_but_hosted_and_unknown_routes_are_preserved(
        tmp_path, unknown):
    hosted, quarantined = '//tests:hosted', '//tests:g3job/test_g3'
    population = selection.query_population([
        {'rule': {'name': '//tests:cpu', 'ruleClass': 'py_test'}},
        {'rule': {'name': hosted, 'ruleClass': 'py_test', 'attribute': [
            {'name': 'tags', 'stringListValue': ['manual', 'gate-lane-hosted']}]}},
        {'rule': {'name': quarantined, 'ruleClass': 'py_test', 'attribute': [
            {'name': 'tags', 'stringListValue': ['manual', 'gate-issue-3181']}]}},
        {'sourceFile': {'name': '//tools:tessera_campaign_container.py'}},
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg',
                  'ruleClass': 'py_library'}},
    ], 'amd64')
    impacted = ['//tests:cpu', hosted, quarantined]
    if unknown:
        impacted.append('//tests:unknown')

    def compare(command, **kwargs):
        Path(command[command.index('--output') + 1]).write_text('\n'.join(impacted) + '\n')
        return SimpleNamespace(returncode=0)

    decision = selection.compare_graphs({'hashes': tmp_path / 'base.json'},
        {'hashes': tmp_path / 'head.json', 'population': population},
        {'tools/tessera_campaign_container.py'}, set(), tmp_path,
        java='java', jar=tmp_path / 'v42.jar', root=tmp_path, runner=compare)
    if unknown:
        with pytest.raises(selection.SelectionError, match='outside the queried population'):
            selection.selection_manifest(decision, {'amd64': population},
                hosted=population['hosted'], base='b' * 40, head='a' * 40, mode='pr')
    else:
        manifest = selection.selection_manifest(decision, {'amd64': population},
            hosted=population['hosted'], base='b' * 40, head='a' * 40, mode='pr')
        assert manifest['hosted'] == [hosted]
        assert manifest['architectures']['amd64']['population'] == ['//tests:cpu']
        assert manifest['architectures']['amd64']['tests'] == ['//tests:cpu']
        assert manifest['selection'] == 'impacted'


@pytest.fixture
def planning_history(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True,
            text=True, capture_output=True).stdout.strip()
    git('init', '--quiet')
    source = root / 'prismaquant/core.py'
    source.parent.mkdir()
    revisions = []
    for content in ('base\n', 'head\n'):
        source.write_text(content)
        git('add', '.')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '--quiet', '-m', content.strip())
        revisions.append(git('rev-parse', 'HEAD'))
    return root, revisions


@pytest.mark.parametrize('timed', [False, True])
def test_fallback_planner_maps_measured_file_times_into_four_balanced_shards(
        tmp_path, planning_history, monkeypatch, timed):
    root, (_, head) = planning_history
    tests = [f'//tests:test_pq_fallback_cost_{index}' for index in range(8)]
    files = {'tests/' + f'test_pq_fallback_cost_{index}.py': seconds
             for index, seconds in enumerate([40, 39, 38, 37, 4, 3, 2, 1])}
    if timed:
        timing_file = root / selection.DURATIONS_PATH
        timing_file.parent.mkdir(parents=True)
        timing_file.write_text(json.dumps({'files': files}))
    population = selection.query_population([
        *[{'rule': {'name': label, 'ruleClass': 'py_test'}} for label in tests],
        {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg', 'ruleClass': 'py_library'}},
    ], 'amd64')
    monkeypatch.setattr(selection, 'query_revision', lambda *args, **kwargs: {'population': population})
    # An empty source diff takes the full fallback without launching an impact query.
    document = selection.plan_revisions(root, tmp_path / 'private', base=head, head=head, mode='pr')
    row = document['architectures']['amd64']
    assert document['selection'] == 'fallback'
    assert len(row['shards']) == 4
    assert sorted(label for shard in row['shards'] for label in shard) == tests
    if timed:
        assert [sum(files[label[2:].replace(':', '/') + '.py'] for label in shard)
                for shard in row['shards']] == [41] * 4
    else:
        assert [len(shard) for shard in row['shards']] == [2] * 4


@pytest.mark.parametrize('mode,fault', [
    ('pr', None), ('pr', 'missing_base'), ('pr', 'hash_failure'),
    ('pr', 'comparison_failure'), ('main', None),
])
def test_revision_planner_binds_real_git_sources_and_retains_full_fallback(
        tmp_path, planning_history, mode, fault):
    root, (base, head) = planning_history
    calls, hashed_sources = [], []
    rows = [
        {'rule': {'name': label, 'ruleClass': 'py_test'}} for label in sorted(TESTS)
    ] + [{'sourceFile': {'name': '//prismaquant:core.py'}},
         {'rule': {'name': '@@rules_python++pip+pq_pip_numpy//:pkg', 'ruleClass': 'py_library'}}]

    def planning_command(command, **kwargs):
        calls.append(command)
        assert not {'BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN', 'GITHUB_TOKEN'} & set(kwargs['env'])
        if command[0] == 'git':
            return subprocess.run(command, **kwargs)
        if command[1] == 'query':
            return SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, rows)))
        if command[3] == 'generate-hashes':
            hashed_sources.append((Path(kwargs['cwd']) / 'prismaquant/core.py').read_text())
            Path(command[-1]).write_text('{"hashes": {"//tests:test_consumer": "Rule#hash"}}')
            Path(command[command.index('--depEdgesFile') + 1]).write_text('{}')
            return SimpleNamespace(returncode=1 if fault == 'hash_failure' else 0)
        assert command[3] == 'get-impacted-targets'
        assert command[command.index('--workspacePath') + 1] == str(root)
        assert command[command.index('--bazelPath') + 1] == 'bazel'
        Path(command[command.index('--output') + 1]).write_text('//tests:test_consumer\n')
        return SimpleNamespace(returncode=1 if fault == 'comparison_failure' else 0)

    found = selection.plan_revisions(root, tmp_path / 'private',
        base='c' * 40 if fault == 'missing_base' else base, head=head, mode=mode,
        java='java', jar=tmp_path / 'v42.jar', runner=planning_command)
    assert found['head'] == head
    assert found['base'] == ('c' * 40 if fault == 'missing_base' else base)
    assert set(found['architectures']) == ({'amd64', 'arm64'} if mode == 'main' else {'amd64'})
    expected = sorted(TESTS) if fault or mode == 'main' else ['//tests:test_consumer']
    for row in found['architectures'].values():
        assert row['tests'] == expected
        assert sorted(label for shard in row['shards'] for label in shard) == row['population']
    if mode == 'pr' and not fault:
        assert hashed_sources == ['base\n', 'head\n']
        assert found['selection'] == 'impacted' and found['all_scopes'] is False
    else:
        assert found['selection'] == 'fallback' and found['all_scopes'] is True
    if mode == 'main':
        assert not hashed_sources
        assert not any(command[0] == 'java' for command in calls)


@pytest.mark.parametrize('fault', [None, 'leak', 'unscannable'])
def test_remote_plan_cli_publishes_only_scanned_manifest_and_matching_receipt(
        tmp_path, monkeypatch, fault):
    monkeypatch.setattr(selection.artifacts, 'require_scan_python', lambda: None)
    monkeypatch.setattr(selection.artifacts, 'scan_runtime_available', lambda: True)
    upload = tmp_path / 'upload'
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(upload))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'plan-test-secret-sentinel')
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': 'pr',
        'base': 'b' * 40, 'head': 'a' * 40, 'selection': 'fallback',
        'scopes': [], 'all_scopes': True, 'reason': [],
        'hosted': [], 'hosted_arm64': [], 'architectures': {'amd64': {
            'population': sorted(TESTS), 'tests': sorted(TESTS),
            'shards': [[label] for label in sorted(TESTS)]}}}
    calls = []
    def plan_sources(root, directory, **kwargs):
        assert not list(upload.iterdir())
        calls.append(kwargs)
        if fault == 'leak':
            document['reason'] = ['plan-test-secret-sentinel']
        if fault == 'unscannable':
            (upload / 'execution_log.binpb.zst').write_bytes(b'\x28\xb5\x2f\xfd')
        return document
    monkeypatch.setattr(selection, 'plan_revisions', plan_sources)
    code = selection.main(['plan', '--mode', 'pr', '--base', 'b' * 40,
        '--head', 'a' * 40, '--branch', 'bbrun/cutover'])
    assert len(calls) == 1 and calls[0]['head'] == 'a' * 40
    if fault:
        assert code == 75
        assert list(upload.iterdir()) == []
    else:
        assert code == 0
        assert {path.name for path in upload.iterdir()} == {'shard-manifest.json', 'gate-result.json'}
        raw = (upload / 'shard-manifest.json').read_bytes()
        assert json.loads(raw) == document
        receipt = json.loads((upload / 'gate-result.json').read_text())
        assert receipt['operation'] == 'plan'
        assert receipt['head'] == document['head'] and receipt['base'] == document['base']
        assert receipt['manifest_sha256'] == bytes_sha256hex(raw)


def test_v42_impact_adapter_supplies_the_required_workspace_and_bazel(tmp_path):
    # Native v42 declares --workspacePath required even for precomputed hashes.
    command = selection.impacted_targets_command('java', tmp_path / 'v42.jar',
        tmp_path / 'head-source', tmp_path / 'pinned-bazel',
        tmp_path / 'base.json', tmp_path / 'head.json', tmp_path / 'impacted.txt')
    assert command[command.index('--workspacePath') + 1] == str(tmp_path / 'head-source')
    assert command[command.index('--bazelPath') + 1] == str(tmp_path / 'pinned-bazel')
