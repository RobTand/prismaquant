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
