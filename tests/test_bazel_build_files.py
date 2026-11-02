"""The Bazel graph generator reads PrismaQuant's dependency evidence soundly (#3216).

A test that needs a file must depend on it: remote executors only stage what
the graph declares. These tests pin each evidence rule on a small tree.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bazel_build_files', ROOT / 'tools/bazel/build_files.py')
build_files = importlib.util.module_from_spec(SPEC)
sys.modules['bazel_build_files'] = build_files
SPEC.loader.exec_module(build_files)


def write(root, sources):
    for path, text in sources.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    return sorted(sources)


def evidence(tmp_path, sources, **options):
    return build_files.Evidence(tmp_path, write(tmp_path, sources), **options)


def graph(tmp_path, sources, **options):
    return build_files.Graph(evidence(tmp_path, sources, **options))


def test_an_import_cycle_becomes_one_strongly_connected_component():
    edges = {'a': {'b'}, 'b': {'c'}, 'c': {'a'}, 'd': {'a'}}
    assert build_files.strongly_connected(['a', 'b', 'c', 'd'], edges) == [['a', 'b', 'c'], ['d']]


def test_import_statements_anywhere_are_edges_and_bare_strings_are_not(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_a.py': ('import helper\n'
                            'def test_x():\n    from prismaquant import core\n'
                            "PATCHED = ('prismaquant.pool', 'launch')\n"),
        'tests/test_b.py': ("import importlib\nNAMES = ('prismaquant.pool', 'launch')\n"
                            'MODULES = [importlib.import_module(name) for name in NAMES]\n'),
        'tests/helper.py': '', 'prismaquant/__init__.py': '', 'prismaquant/core.py': '',
        'prismaquant/pool.py': '', 'tools/launch.py': ''})
    modules, _ = found.of('tests/test_a.py')
    assert modules == {'tests/helper.py', 'prismaquant/__init__.py', 'prismaquant/core.py'}
    assert found.test_modules('tests/test_a.py') == set()
    # A test that imports by name gets the modules it names, on its own target only.
    assert found.of('tests/test_b.py')[0] == set()
    assert found.test_modules('tests/test_b.py') == {'prismaquant/__init__.py', 'prismaquant/pool.py',
                                                      'tools/launch.py'}


def test_a_path_that_reaches_a_launch_counts_and_a_named_path_does_not(tmp_path):
    found = evidence(tmp_path, {
        'tools/worker.py': ("import subprocess, sys\nfrom pathlib import Path\n"
                            "EXEC = Path(__file__).resolve().parent / 'fleet/resource_exec.py'\n"
                            "def run():\n    subprocess.run([sys.executable, str(EXEC)], check=True)\n"),
        'tools/named.py': ("from pathlib import Path\n"
                           "EXEC = Path(__file__).resolve().parent / 'fleet/resource_exec.py'\n"
                           "print('would run', EXEC)\n"),
        'tools/fleet/resource_exec.py': 'from prismaquant import pool\n',
        'prismaquant/pool.py': ''})
    assert found.run_by_path({'tools/worker.py'}) == {'tools/fleet/resource_exec.py'}
    assert found.run_by_path({'tools/named.py'}) == set()
    # A launched program is the test's input, never a library edge.
    assert found.of('tools/worker.py')[0] == set()


def test_with_the_launch_rule_off_every_exact_production_path_counts(tmp_path):
    found = evidence(tmp_path, {
        'tools/named.py': ("from pathlib import Path\n"
                           "EXEC = Path(__file__).resolve().parent / 'fleet/resource_exec.py'\n"),
        'tools/fleet/resource_exec.py': ''}, launch_rule=False)
    assert found.run_by_path({'tools/named.py'}) == {'tools/fleet/resource_exec.py'}


def test_a_launch_whose_arguments_cannot_be_followed_counts_every_path_and_is_uncertain(tmp_path):
    found = evidence(tmp_path, {
        'tools/runner.py': ("import subprocess\nfrom pathlib import Path\n"
                            "HELPER = Path(__file__).with_name('helper.py')\n"
                            "def run(argv):\n    subprocess.run(argv, check=True)\n"),
        'tools/helper.py': ''})
    assert found.run_by_path({'tools/runner.py'}) == {'tools/helper.py'}
    assert 'launch arguments not followed' in found.uncertain['tools/runner.py']


def test_bare_file_names_count_only_when_a_path_is_built_from_them():
    tree = ast.parse(
        "SCRIPTS = ('worker_loop.py', 'tier_loop.py')\n"
        "DEPENDENCIES = ('resource_broker.py',)\n"
        "def busy(name):\n    return name in SCRIPTS\n"
        "def copy(source, target):\n    for dependency in DEPENDENCIES:\n"
        "        (target / dependency).write_bytes((source / dependency).read_bytes())\n")
    assert build_files.path_built_names(tree) == {'resource_broker.py'}


def test_a_module_launched_with_dash_m_is_a_program_in_production_and_an_edge_in_tests(tmp_path):
    found = evidence(tmp_path, {
        'prismaquant/__init__.py': '',
        'prismaquant/campaign.py': ("import subprocess, sys\n"
                                    "def run():\n"
                                    "    subprocess.run([sys.executable, '-m', 'prismaquant.child'], check=True)\n"),
        'prismaquant/child.py': 'from prismaquant import heavy\n', 'prismaquant/heavy.py': '',
        'tests/test_cli.py': ("import subprocess, sys\n"
                              "def test_cli():\n"
                              "    subprocess.run(f'{sys.executable} -m prismaquant.child --help', shell=True)\n")})
    assert found.of('prismaquant/campaign.py')[0] == set()
    assert found.run_by_path({'prismaquant/campaign.py'}) >= {'prismaquant/child.py'}
    assert found.of('tests/test_cli.py')[0] == {'prismaquant/__init__.py', 'prismaquant/child.py'}
    tests = build_files.Graph(found)
    assert {'prismaquant/child.py', 'prismaquant/heavy.py'} <= tests.test_files('tests/test_cli.py')


def test_snippets_dynamic_imports_and_pytest_plugins_are_edges(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_a.py': ('import importlib, textwrap\n'
                            'CHILD = textwrap.dedent("""\n    from prismaquant import pool\n    pool.run()\n""")\n'
                            "SCRIPT = f'from queue_host import install; install({1!r})'\n"
                            "importlib.import_module('prismaquant.core')\n"),
        'tests/conftest.py': 'pytest_plugins = ("prismaquant.bound",)\n',
        'tests/queue_host.py': '', 'prismaquant/pool.py': '', 'prismaquant/core.py': '',
        'prismaquant/bound.py': ''})
    assert found.of('tests/test_a.py')[0] == {'prismaquant/pool.py', 'prismaquant/core.py', 'tests/queue_host.py'}
    assert found.of('tests/conftest.py')[0] == {'prismaquant/bound.py'}


def test_paths_to_tools_are_edges_and_paths_to_tests_or_globs_are_data(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_a.py': ('from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n'
                            "TOOL = ROOT / 'tools/launch.py'\nOTHER = ROOT / 'tests/test_b.py'\n"
                            "FIXTURES = sorted((ROOT / 'tests/fixtures').glob('*.json'))\n"),
        'tests/test_b.py': '', 'tools/launch.py': '',
        'tests/fixtures/one.json': '', 'tests/fixtures/two.json': ''})
    modules, data = found.of('tests/test_a.py')
    assert modules == {'tools/launch.py'}
    assert {'tests/test_b.py', 'tests/fixtures/one.json', 'tests/fixtures/two.json'} <= data


def test_a_test_that_runs_a_child_pytest_needs_the_named_test_files_imports(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_parent.py': ('import subprocess, sys\nfrom pathlib import Path\n'
                                 "CHILD = Path(__file__).with_name('test_child.py')\n"
                                 "def test_run():\n"
                                 "    subprocess.run([sys.executable, '-m', 'pytest', str(CHILD)], check=True)\n"),
        'tests/test_child.py': 'from prismaquant import core\n', 'prismaquant/core.py': ''})
    assert found.test_modules('tests/test_parent.py') == {'tests/test_child.py'}
    assert 'prismaquant/core.py' in build_files.Graph(found).test_files('tests/test_parent.py')


def test_a_tool_module_resolves_from_the_directory_a_conftest_puts_on_sys_path(tmp_path):
    found = evidence(tmp_path, {
        'tests/g3job/conftest.py': ('import sys\nfrom pathlib import Path\n'
                                    'HARNESS = Path(__file__).resolve().parents[2] / "tools" / "g3job"\n'
                                    'sys.path.insert(0, str(HARNESS))\n'),
        'tests/g3job/test_a.py': 'from g3_policy import rules\n',
        'tools/g3job/g3_policy/__init__.py': '', 'tools/g3job/g3_policy/rules.py': ''})
    assert found.of('tests/g3job/test_a.py')[0] == {'tools/g3job/g3_policy/__init__.py',
                                                    'tools/g3job/g3_policy/rules.py'}


def test_a_directory_put_on_sys_path_outside_tools_makes_its_modules_importable(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_probe.py': ('import sys\nfrom pathlib import Path\n'
                                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/probes'))\n"
                                'import native_probe\n'),
        'docs/probes/native_probe.py': 'from prismaquant import core\n', 'prismaquant/core.py': ''})
    assert found.of('tests/test_probe.py')[0] == {'docs/probes/native_probe.py'}
    assert found.of('docs/probes/native_probe.py')[0] == {'prismaquant/core.py'}


def test_a_sys_path_entry_that_cannot_be_resolved_is_uncertain(tmp_path):
    found = evidence(tmp_path, {'tools/child.py': 'import sys\nsys.path.insert(0, sys.argv[1])\n'})
    assert 'sys.path entry not followed' in found.uncertain['tools/child.py']


def test_experiments_modules_resolve_by_their_repository_path(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_exp.py': 'from experiments.kl_sweep import runner\n',
        'experiments/kl_sweep/runner.py': 'from prismaquant import core\n', 'prismaquant/core.py': ''})
    assert found.of('tests/test_exp.py')[0] == {'experiments/kl_sweep/runner.py'}
    assert 'prismaquant/core.py' in build_files.Graph(found).test_files('tests/test_exp.py')


def test_a_lazy_package_export_resolves_to_the_module_that_defines_it(tmp_path):
    found = evidence(tmp_path, {
        'prismaquant/__init__.py': ('from importlib import import_module\n'
                                    '_LAZY_EXPORTS = {"FormatSpec": ".format_registry"}\n'
                                    'def __getattr__(name):\n'
                                    '    return getattr(import_module(_LAZY_EXPORTS[name], __name__), name)\n'),
        'prismaquant/format_registry.py': '', 'prismaquant/other.py': '',
        'tests/test_lazy.py': 'from prismaquant import FormatSpec\n',
        'tests/test_plain.py': 'from prismaquant import other\n'})
    assert found.of('tests/test_lazy.py')[0] == {'prismaquant/__init__.py', 'prismaquant/format_registry.py'}
    assert found.of('tests/test_plain.py')[0] == {'prismaquant/__init__.py', 'prismaquant/other.py'}
    # The table itself is followed, so the package's own non-literal import is not uncertain.
    assert 'prismaquant/__init__.py' not in found.uncertain


def test_a_non_literal_dynamic_import_is_uncertain(tmp_path):
    found = evidence(tmp_path, {'prismaquant/plugins.py': ('import importlib\n'
                                                           'def load(name):\n'
                                                           '    return importlib.import_module(name)\n')})
    assert 'dynamic import not followed' in found.uncertain['prismaquant/plugins.py']


def test_production_tree_reads_reach_only_tests_that_import_the_module(tmp_path):
    found = evidence(tmp_path, {
        'tools/scan.py': ('from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n'
                          "TABLE = Path(__file__).with_name('table.json')\n"
                          "FILES = sorted((ROOT / 'docs').rglob('*.md'))\n"),
        'tools/table.json': '{}', 'docs/a.md': '', 'docs/b.md': '',
        'tests/test_scan.py': 'import scan\n', 'tests/helper.py': 'import scan\n'})
    _, data = found.of('tools/scan.py')
    assert data - {'tools/scan.py'} == {'tools/table.json'}
    assert found.test_data('tests/test_scan.py', found.of('tests/test_scan.py')[0]) >= {'docs/a.md', 'docs/b.md'}


def test_conftest_contributes_imports_and_exact_paths_only(tmp_path):
    found = evidence(tmp_path, {
        'tests/conftest.py': ('from pathlib import Path\nimport helper\n'
                              'def pytest_terminal_summary(terminalreporter):\n'
                              '    root = Path(__file__).resolve().parents[1]\n'
                              '    for name in terminalreporter.files:\n        (root / name).read_bytes()\n'),
        'tests/helper.py': '', 'docs/a.md': ''})
    modules, data = found.of('tests/conftest.py')
    assert modules == {'tests/helper.py'} and data <= {'tests/conftest.py'}


def test_every_conftest_from_tests_down_to_the_test_applies(tmp_path):
    tests = graph(tmp_path, {
        'tests/conftest.py': 'from prismaquant import shared\n',
        'tests/g3job/conftest.py': 'from prismaquant import harness\n',
        'tests/test_top.py': '', 'tests/g3job/test_deep.py': '',
        'prismaquant/shared.py': '', 'prismaquant/harness.py': ''})
    assert tests.conftests_for('tests/test_top.py') == ['tests/conftest.py']
    assert tests.conftests_for('tests/g3job/test_deep.py') == ['tests/conftest.py', 'tests/g3job/conftest.py']
    assert 'prismaquant/harness.py' not in tests.test_files('tests/test_top.py')
    assert {'prismaquant/shared.py', 'prismaquant/harness.py', 'tests/g3job/conftest.py',
            'tests/g3job/test_deep.py'} <= tests.test_files('tests/g3job/test_deep.py')


def test_a_test_file_holds_its_closure_its_library_data_and_never_another_tests_own_inputs(tmp_path):
    tests = graph(tmp_path, {
        'tests/test_a.py': 'from test_b import HELPER\n',
        'tests/test_b.py': ("import importlib\nHELPER = 1\n"
                            "importlib.import_module(('prismaquant.pool', 'x')[0])\n"
                            "NAMES = ('prismaquant.pool',)\n"),
        'prismaquant/__init__.py': '', 'prismaquant/pool.py': '',
        'prismaquant/reader.py': ("from pathlib import Path\n"
                                  "TABLE = Path(__file__).with_name('table.json')\n"),
        'prismaquant/table.json': '{}',
        'tests/test_c.py': 'from prismaquant import reader\n'})
    assert 'tests/test_b.py' in tests.test_files('tests/test_a.py')
    assert 'prismaquant/pool.py' in tests.test_files('tests/test_b.py')
    assert 'prismaquant/pool.py' not in tests.test_files('tests/test_a.py')
    assert 'prismaquant/table.json' in tests.test_files('tests/test_c.py')


def test_the_graph_record_names_edges_cycles_tests_and_uncertain_modules(tmp_path):
    tests = graph(tmp_path, {
        'tests/conftest.py': '', 'tests/test_a.py': 'from tools import alpha\n',
        'tools/__init__.py': '', 'tools/alpha.py': 'from tools import beta\n',
        'tools/beta.py': 'from tools import alpha\nimport importlib\nimportlib.import_module(beta)\n'})
    record = json.loads(json.dumps(build_files.graph_record(tests, population=['tests/test_a.py'])))
    assert record['schema'] == 'prismaquant.bazel_graph.v1'
    assert record['edges']['tools/alpha.py'] == ['tools/__init__.py', 'tools/beta.py']
    assert record['cycles'] == [['tools/alpha.py', 'tools/beta.py']]
    assert record['tests']['tests/test_a.py']['conftests'] == ['tests/conftest.py']
    assert record['tests']['tests/test_a.py']['in_population'] is True
    assert record['uncertain'] == {'tools/beta.py': ['dynamic import not followed']}


PLAN = {'selected': [{'path': 'tests/test_in.py'}, {'path': 'tests/test_slow.py'}],
        'deselect': ['tests/test_in.py::test_flaky'],
        'deferred': [{'path': 'tests/test_cuda.py', 'lane': 'cuda', 'issue': 12, 'kind': 'environment_lanes'},
                     {'path': 'tests/test_costly.py', 'issue': 1997, 'kind': 'cost_exclusions'}]}


def test_population_files_keep_their_deselects_and_others_are_manual_with_owner():
    attributes = build_files.test_attributes('tests/test_in.py', PLAN, {})
    assert attributes['args'] == ['--deselect=tests/test_in.py::test_flaky']
    assert 'tags' not in attributes
    assert build_files.test_attributes('tests/test_cuda.py', PLAN, {})['tags'] == ['gate-lane-cuda', 'manual']
    assert build_files.test_attributes('tests/test_costly.py', PLAN, {})['tags'] == ['gate-issue-1997', 'manual']


def test_a_test_file_outside_the_plan_is_an_error():
    with pytest.raises(build_files.GenerationError, match='tests/test_unknown.py'):
        build_files.test_attributes('tests/test_unknown.py', PLAN, {})


def test_long_files_are_sharded_and_given_a_long_timeout_only_when_needed():
    slow = build_files.test_attributes('tests/test_slow.py', PLAN, {'tests/test_slow.py': 170.2})
    assert slow['shard_count'] == 5 and slow['timeout'] == 'moderate'
    quick = build_files.test_attributes('tests/test_in.py', PLAN, {'tests/test_in.py': 3.0})
    assert 'shard_count' not in quick and quick['timeout'] == 'moderate'
