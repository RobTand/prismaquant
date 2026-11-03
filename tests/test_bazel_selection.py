"""Cutover selection falls back safely and preserves wheel-level impact."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pq_bazel_selection', ROOT / 'tools/ci/bazel_selection.py')
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
TESTS = {'//tests:test_consumer', '//tests:test_unrelated'}


def decision(changed, *, impacted=frozenset(), used=frozenset(), deleted=frozenset(), error=None):
    return selection.gate_decision(set(changed), set(deleted), set(impacted), set(used), TESTS,
                                   detection_error=error)


@pytest.mark.parametrize('path', [
    '.github/workflows/gate.yml', '.bazelrc', '.bazelversion', '.bazelignore',
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
        tmp_path / 'base.json', tmp_path / 'head.json', tmp_path / 'impacted.txt')
    assert command == ['java', '-jar', str(tmp_path / 'v42.jar'), 'get-impacted-targets',
        '--startingHashes', str(tmp_path / 'base.json'),
        '--finalHashes', str(tmp_path / 'head.json'),
        '--output', str(tmp_path / 'impacted.txt')]
