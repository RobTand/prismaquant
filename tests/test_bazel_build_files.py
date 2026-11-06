"""The Bazel graph generator reads PrismaQuant's dependency evidence soundly (#3216).

A test that needs a file must depend on it: remote executors only stage what
the graph declares. These tests pin each evidence rule on a small tree.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
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


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, check=True).stdout


@pytest.fixture
def indexed_repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.invalid')
    git(root, 'config', 'user.name', 'Test')
    write(root, {
        '.gitignore': 'ignored-root.json\n',
        'tools/__init__.py': '',
        'tools/scan.py': ("from pathlib import Path\n"
                          "ROOT = Path(__file__).resolve().parents[1]\n"
                          "TABLE = ROOT / 'fixture.json'\n"
                          "FILES = sorted(ROOT.rglob('*'))\n"),
        'fixture.json': '{"tracked": true}\n',
        'tests/test_scan.py': 'from tools import scan\n',
        'tools/ci/gate_durations.json': '{"files": {}}\n',
        'tools/ci/gate_quarantine.json': '{}\n',
    })
    git(root, 'add', '.')
    git(root, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixed graph inputs')
    return root


def generate_graph_json(root, output):
    # Exercise the real CLI entry point, including discovery and hosted_plan.
    assert build_files.main(['--root', str(root), '--graph-json', str(output)]) == 0
    return output.read_bytes()


@pytest.mark.parametrize('name', ['.pbrun-closure.x.json', 'pbrun_result.x.txt'])
def test_untracked_root_files_leave_the_graph_json_byte_identical(indexed_repository, tmp_path, name):
    root = indexed_repository
    output = tmp_path / 'graph.json'  # Not an input to the miniature repository.
    before = generate_graph_json(root, output)
    record = json.loads(before)
    assert 'fixture.json' in record['data']['tools/scan.py']
    assert 'fixture.json' in record['tests']['tests/test_scan.py']['own_data']
    write(root, {name: 'unrelated admission bookkeeping\n'})
    assert git(root, 'ls-files', '--others', '--exclude-standard', '-z') == name.encode() + b'\0'
    assert generate_graph_json(root, output) == before


def test_repository_files_lists_the_index_and_never_untracked_files(indexed_repository):
    root = indexed_repository
    committed = set(build_files.repository_files(root))
    added = {
        'tools/staged.py': 'VALUE = 1\n',
        'space and\nnewline.txt': 'NUL-delimited Git paths survive intact\n',
        'BUILD.bazel': '# excluded\n',
        'nested/BUILD.bazel': '# also excluded\n',
        'bazel-output/file.txt': 'excluded prefix\n',
        'deleted.txt': 'tracked, but absent from the worktree\n',
        'directory.txt': 'tracked, but replaced by a directory\n',
    }
    write(root, added)
    (root / 'link.txt').symlink_to('fixture.json')
    git(root, 'add', *added, 'link.txt')
    (root / 'deleted.txt').unlink()
    (root / 'directory.txt').unlink()
    (root / 'directory.txt').mkdir()
    write(root, {'untracked.py': '', 'untracked.json': '{}', 'untracked.txt': ''})
    assert set(build_files.repository_files(root)) == committed | {'tools/staged.py', 'space and\nnewline.txt'}


def test_tracked_fixture_and_staged_module_participate_in_the_graph(indexed_repository, tmp_path):
    root = indexed_repository
    write(root, {'tools/staged.py': 'VALUE = 1\n',
                 'tests/test_scan.py': 'from tools import scan, staged\n'})
    git(root, 'add', 'tools/staged.py')
    assert git(root, 'ls-tree', '-r', '--name-only', 'HEAD', 'tools/staged.py') == b''
    assert git(root, 'diff', '--cached', '--name-only') == b'tools/staged.py\n'
    record = json.loads(generate_graph_json(root, tmp_path / 'graph.json'))
    assert 'tools/staged.py' in record['modules']
    assert 'tools/staged.py' in record['edges']['tests/test_scan.py']
    discovered = build_files.Graph(build_files.Evidence(root, build_files.repository_files(root)))
    assert 'tools/staged.py' in discovered.test_files('tests/test_scan.py')
    assert 'fixture.json' in record['data']['tools/scan.py']


def test_ignored_root_file_leaves_the_graph_json_byte_identical(indexed_repository, tmp_path):
    root = indexed_repository
    output = tmp_path / 'graph.json'
    before = generate_graph_json(root, output)
    write(root, {'ignored-root.json': '{"ignored": true}\n'})
    assert git(root, 'check-ignore', 'ignored-root.json') == b'ignored-root.json\n'
    assert generate_graph_json(root, output) == before


def test_deleted_tracked_fixture_is_not_listed_and_generation_succeeds(indexed_repository, tmp_path):
    root = indexed_repository
    output = tmp_path / 'graph.json'
    before = json.loads(generate_graph_json(root, output))
    assert 'fixture.json' in before['data']['tools/scan.py']
    (root / 'fixture.json').unlink()
    assert git(root, 'ls-files', '--error-unmatch', 'fixture.json') == b'fixture.json\n'
    assert 'fixture.json' not in build_files.repository_files(root)
    after = json.loads(generate_graph_json(root, output))
    assert 'fixture.json' not in after['data'].get('tools/scan.py', [])
    assert 'fixture.json' not in after['tests']['tests/test_scan.py']['own_data']


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


@pytest.mark.parametrize('plugins', ['("prismaquant.bound",)', '["prismaquant.bound"]'])
def test_annotated_pytest_plugins_tuple_and_list_give_the_same_edge_as_the_unannotated_control(tmp_path, plugins):
    found = evidence(tmp_path, {
        'tests/plain/conftest.py': f'pytest_plugins = {plugins}\n',
        'tests/annotated/conftest.py': f'pytest_plugins: tuple[str, ...] = {plugins}\n',
        'prismaquant/bound.py': '',
    })
    expected = {'prismaquant/bound.py'}
    assert found.of('tests/plain/conftest.py')[0] == expected
    assert found.of('tests/annotated/conftest.py')[0] == expected


def test_an_annotated_pytest_plugins_string_is_an_edge(tmp_path):
    found = evidence(tmp_path, {
        'tests/plain/conftest.py': 'pytest_plugins = "prismaquant.bound"\n',
        'tests/annotated/conftest.py': 'pytest_plugins: str = "prismaquant.bound"\n',
        'prismaquant/bound.py': '',
    })
    expected = {'prismaquant/bound.py'}
    assert found.of('tests/plain/conftest.py')[0] == expected
    assert found.of('tests/annotated/conftest.py')[0] == expected


def test_an_annotated_plugins_imports_reach_the_consuming_test_and_an_unrelated_module_does_not(tmp_path):
    tests = graph(tmp_path, {
        'tests/conftest.py': 'pytest_plugins: tuple[str, ...] = ("prismaquant.bound",)\n',
        'tests/test_a.py': 'def test_a():\n    pass\n',
        'prismaquant/bound.py': 'import prismaquant.helper\n',
        'prismaquant/helper.py': '',
        'prismaquant/unrelated.py': '',
    })
    files = tests.test_files('tests/test_a.py')
    assert {'tests/test_a.py', 'tests/conftest.py', 'prismaquant/bound.py', 'prismaquant/helper.py'} <= files
    assert 'prismaquant/unrelated.py' not in files


@pytest.mark.parametrize('declaration', [
    'pytest_plugins: tuple[str, ...]',
    'pytest_plugins: tuple[str, ...] = runtime_plugins',
    'pytest_plugins: tuple[str, ...] = load_plugins()',
])
def test_a_bare_or_non_literal_annotated_pytest_plugins_adds_no_edge_beside_a_valued_one(tmp_path, declaration):
    prefix = 'runtime_plugins = ("prismaquant.unrelated",)\n' + declaration + '\n'
    found = evidence(tmp_path, {
        'tests/empty/conftest.py': prefix,
        'tests/valued/conftest.py': prefix + 'pytest_plugins: tuple[str, ...] = ("prismaquant.bound",)\n',
        'prismaquant/bound.py': '',
        'prismaquant/unrelated.py': '',
    })
    assert found.of('tests/empty/conftest.py')[0] == set()
    assert found.of('tests/valued/conftest.py')[0] == {'prismaquant/bound.py'}


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


def test_a_test_importing_another_test_depends_on_its_library_not_its_test():
    assert build_files.library_name('tests/test_b.py') == 'test_b_lib'
    assert build_files.library_name('tests/helper.py') == 'helper'
    assert build_files.library_name('prismaquant/allocator.py') == 'prismaquant/allocator'


def test_whole_directories_compress_to_one_filegroup_and_the_rest_stay_files():
    files = ['tests/fixtures/a.json', 'tests/fixtures/b.json', 'tests/test_x.py', 'docs/one.md', 'docs/two.md']
    trees = build_files.Trees(files)
    labels = trees.labels({'tests/fixtures/a.json', 'tests/fixtures/b.json', 'docs/one.md'}, 'tests')
    assert labels == ['//:docs/one.md', ':files/fixtures']
    assert trees.used == {'tests': {'fixtures'}}


def test_a_module_depends_on_the_pip_distributions_it_imports(tmp_path):
    found = evidence(tmp_path, {
        'tools/bazel/pip_modules.json': json.dumps({'torch': ['torch'], 'yaml': ['pyyaml'], 'PIL': ['pillow']}),
        'prismaquant/loader.py': 'import yaml\ndef later():\n    import torch.nn\n',
        'prismaquant/plain.py': 'import json\n',
        'tests/test_cli.py': 'import subprocess\nsubprocess.run(["python", "-m", "PIL.Image"])\n'})
    assert found.packages('prismaquant/loader.py') == {'pyyaml', 'torch'}
    assert found.packages('prismaquant/plain.py') == set()
    assert found.packages('tests/test_cli.py') == {'pillow'}
    assert build_files.pip_label('pyyaml') == '@pq_pip//pyyaml'
    assert build_files.pip_label('tessera-quant') == '@pq_pip//tessera_quant'


SMALL = {
    'pytest.ini': '[pytest]\ntestpaths = tests\n',
    'tools/ci/gate_durations.json': json.dumps({'files': {'tests/test_slow.py': 100.0}}),
    'tools/ci/gate_quarantine.json': json.dumps({'entries': [
        {'target': 'tests/test_held.py', 'issue': 7, 'reason': 'owned elsewhere'},
        {'target': 'tests/test_slow.py::test_flaky', 'issue': 8, 'reason': 'flaky'}]}),
    'tools/bazel/pip_modules.json': json.dumps({'yaml': ['pyyaml']}),
    'tools/bazel/test_inputs.bzl': 'TEST_INPUTS = {}\n',
    'tests/conftest.py': 'from prismaquant import core\n',
    'tests/g3job/conftest.py': 'import sys\n',
    'tests/g3job/test_deep.py': 'def test_deep(): pass\n',
    'tests/test_slow.py': 'import yaml\nfrom prismaquant import runner\ndef test_flaky(): pass\n',
    'tests/test_held.py': 'def test_held(): pass\n',
    'prismaquant/__init__.py': '', 'prismaquant/core.py': '',
    'prismaquant/runner.py': ("import subprocess, sys\n"
                              "def run():\n    subprocess.run([sys.executable, '-m', 'prismaquant.child'])\n"),
    'prismaquant/child.py': 'import yaml\n',
}


def test_generated_packages_hold_libraries_tests_lanes_and_deselects(tmp_path):
    files = write(tmp_path, SMALL)
    generated = build_files.generate(tmp_path, files)
    assert set(generated) == {'', 'tests'}
    root, tests = generated[''], generated['tests']
    assert root.startswith(build_files.HEADER) and tests.startswith(build_files.HEADER)
    assert 'name = "prismaquant/child",' in root and '"@pq_pip//pyyaml"' in root
    slow = tests[tests.index('name = "test_slow"'):].split('\n)\n', 1)[0]
    # The conftest above it, the program its closure launches, its deselect, its shards.
    assert '":conftest"' in slow and '"//:prismaquant/child"' in slow
    assert '"--deselect=tests/test_slow.py::test_flaky"' in slow
    assert 'shard_count = 3' in slow
    deep = tests[tests.index('name = "g3job/test_deep"'):].split('\n)\n', 1)[0]
    assert '":conftest"' in deep and '":g3job/conftest"' in deep
    held = tests[tests.index('name = "test_held"'):].split('\n)\n', 1)[0]
    assert '"gate-issue-7"' in held and '"manual"' in held


def test_an_import_cycle_across_packages_refuses_generation(tmp_path):
    sources = dict(SMALL, **{'prismaquant/core.py': 'import tests.helper\n',
                             'tests/helper.py': 'from prismaquant import core\n'})
    with pytest.raises(build_files.GenerationError, match='crosses Bazel packages'):
        build_files.generate(tmp_path, write(tmp_path, sources))


def test_two_targets_with_one_name_refuse_generation():
    with pytest.raises(build_files.GenerationError, match='x_lib'):
        build_files.render_package('tests', [('x_lib', 'pq_py_library', {}), ('x_lib', 'pq_py_library', {})])


def test_check_names_each_stale_build_file_and_the_fix(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/BUILD.bazel').write_text('old\n')
    git(tmp_path, 'add', '.')
    monkeypatch.setattr(build_files, 'generate', lambda root: {'tests': 'new\n', '': 'root\n'})
    with pytest.raises(SystemExit) as stopped:
        build_files.main(['--check', '--root', str(tmp_path)])
    assert 'BUILD.bazel' in str(stopped.value) and 'tests/BUILD.bazel' in str(stopped.value)
    assert 'tools/bazel/generate.sh' in str(stopped.value)
    assert (tmp_path / 'tests/BUILD.bazel').read_text() == 'old\n'


@pytest.mark.parametrize('fault', [None, 'retired', 'deleted_retired', 'untracked_expected',
                                  'untracked_retired', 'ignored_retired'])
def test_check_requires_exact_tracked_generated_outputs(indexed_repository, monkeypatch, fault):
    root = indexed_repository
    generated = {'': build_files.HEADER + '# root\n', 'tests': build_files.HEADER + '# tests\n'}
    monkeypatch.setattr(build_files, 'generate', lambda root: generated)
    write(root, {'BUILD.bazel': generated[''], 'tests/BUILD.bazel': generated['tests'],
                 'tools/bazel/BUILD.bazel': '# handwritten\n',
                 'manual/BUILD.bazel': '# handwritten\n'})
    git(root, 'add', '.')
    # The handwritten subtree is excluded even when an untracked file has the header.
    write(root, {'tools/bazel/scratch/BUILD.bazel': build_files.HEADER})
    path = 'retired/BUILD.bazel'
    if fault in ('retired', 'deleted_retired', 'untracked_retired', 'ignored_retired'):
        write(root, {path: build_files.HEADER + '# retired\n'})
        if fault in ('retired', 'deleted_retired'):
            git(root, 'add', path)
        if fault == 'deleted_retired':
            (root / path).unlink()
        if fault == 'ignored_retired':
            (root / '.gitignore').write_text('retired/\n')
    if fault == 'untracked_expected':
        path = 'tests/BUILD.bazel'
        git(root, 'rm', '--cached', path)
        # Declared outputs are owned even without their generated header.
        generated['tests'] = '# declared output\n'
        (root / path).write_text(generated['tests'])
    if fault is None:
        assert build_files.main(['--check', '--root', str(root)]) == 0
    else:
        with pytest.raises(SystemExit) as stopped:
            build_files.main(['--check', '--root', str(root)])
        assert path in str(stopped.value)
        assert build_files.FIX in str(stopped.value)


def test_check_fails_when_generation_errors(tmp_path, monkeypatch):
    def fail(root):
        raise build_files.GenerationError('invalid dependency evidence')
    monkeypatch.setattr(build_files, 'generate', fail)
    with pytest.raises(SystemExit, match='Cannot generate BUILD files: invalid dependency evidence'):
        build_files.main(['--check', '--root', str(tmp_path)])


@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_gate_planner_rejects_stale_generated_build_before_query(indexed_repository, monkeypatch, mode):
    from tools.ci import bazel_generated, bazel_selection

    root = indexed_repository
    write(root, SMALL)
    git(root, 'add', '.')
    for package, text in build_files.generate(root).items():
        write(root, {str(Path(package) / 'BUILD.bazel'): text})
    git(root, 'add', '.')
    git(root, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'generated inputs')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    (root / 'tests/BUILD.bazel').write_text(build_files.HEADER + '# stale\n')
    monkeypatch.setattr(bazel_generated, 'commands', lambda: [[sys.executable,
        str(ROOT / 'tools/bazel/build_files.py'), '--check', '--root', str(root)]])

    def query(*args, **kwargs):
        pytest.fail('stale generated BUILD reached the graph query')
    monkeypatch.setattr(bazel_selection, 'query_revision', query)
    with pytest.raises(RuntimeError, match='Generated BUILD files are stale: tests/BUILD.bazel'):
        bazel_selection.plan_revisions(root, root / 'private', base=head, head=head, mode=mode)


def test_committed_build_files_carry_the_generated_header():
    for path in [ROOT / 'BUILD.bazel', ROOT / 'tests/BUILD.bazel']:
        assert path.read_text().startswith(build_files.HEADER), path


def test_files_a_module_globs_below_its_own_directory_are_library_data(tmp_path):
    found = evidence(tmp_path, {
        'prismaquant/lane_spec.py': ('from pathlib import Path\n'
                                     '_SPEC_DIR = Path(__file__).resolve().parent / "lane_specs"\n'
                                     'SPECS = sorted(_SPEC_DIR.glob("*.json"))\n'
                                     'SOURCES = sorted(Path(__file__).parent.glob("*.py"))\n'),
        'prismaquant/lane_specs/tessera.json': '{}', 'prismaquant/other.py': ''})
    _, data = found.of('prismaquant/lane_spec.py')
    assert 'prismaquant/lane_specs/tessera.json' in data
    # A scan of the module's own directory stays a tree read of direct importers.
    assert 'prismaquant/other.py' not in data


def test_a_module_that_imports_a_computed_name_depends_on_modules_its_data_names(tmp_path):
    found = evidence(tmp_path, {
        'prismaquant/lane_spec.py': ('import json\nfrom importlib import import_module\nfrom pathlib import Path\n'
                                     '_SPEC_DIR = Path(__file__).resolve().parent / "lane_specs"\n'
                                     'def plugin(name):\n'
                                     '    spec = json.loads((_SPEC_DIR / f"{name}.json").read_text())\n'
                                     '    return import_module(spec["plugin"])\n'),
        'prismaquant/lane_specs/tessera.json': '{"plugin": "prismaquant.tessera_lane"}',
        'prismaquant/tessera_lane.py': ''})
    assert 'prismaquant/tessera_lane.py' in found.of('prismaquant/lane_spec.py')[0]


def test_a_test_that_loads_a_module_by_path_depends_on_its_library(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_audit.py': ('import importlib.util\nfrom pathlib import Path\n'
                                'FIXTURE = Path.cwd() / "tests" / "test_fixture.py"\n'
                                'spec = importlib.util.spec_from_file_location("fixture", FIXTURE)\n'),
        'tests/test_fixture.py': 'from test_helper import X\n', 'tests/test_helper.py': 'X = 1\n'})
    assert 'tests/test_fixture.py' in found.of('tests/test_audit.py')[0]
    assert 'tests/test_helper.py' in build_files.Graph(found).test_files('tests/test_audit.py')


def test_a_directory_a_test_names_brings_its_files_and_a_top_level_one_does_not(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_copy.py': ('import shutil\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n'
                               'SOURCE = ROOT / "tests" / "fixture-data" / "figure"\n'
                               'TOP = ROOT / "docs"\n'),
        'tests/fixture-data/figure/a.json': '{}', 'tests/fixture-data/figure/b.json': '{}', 'docs/x.md': ''})
    data = found.of('tests/test_copy.py')[1]
    assert {'tests/fixture-data/figure/a.json', 'tests/fixture-data/figure/b.json'} <= data
    assert 'docs/x.md' not in data


def test_a_py_path_inside_a_longer_string_is_a_reference(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_snippet.py': ("import subprocess, sys\n"
                                  "CODE = \"import runpy; runpy.run_path(ROOT + '/experiments/manifests.py')\"\n"
                                  "subprocess.run([sys.executable, '-c', CODE])\n"),
        'experiments/manifests.py': 'from prismaquant import core\n', 'prismaquant/core.py': ''})
    assert 'experiments/manifests.py' in found.of('tests/test_snippet.py')[0]


def test_entry_points_a_test_runs_from_its_own_data_table_are_its_inputs(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_entrypoints.py': ('import json, subprocess, sys\nfrom pathlib import Path\n'
                                      'ROWS = json.loads((Path(__file__).parent / "fixtures" / "rows.json").read_text())\n'
                                      'def test_help():\n'
                                      '    for row in ROWS:\n'
                                      '        subprocess.run([sys.executable, *row["probe"]])\n'),
        'tests/fixtures/rows.json': '[{"file": "tools/capture.py", "probe": ["-m", "tools.capture", "--help"]}]',
        'tools/__init__.py': '', 'tools/capture.py': 'from prismaquant import replay\n', 'prismaquant/replay.py': ''})
    assert 'tools/capture.py' in found.test_modules('tests/test_entrypoints.py')
    assert 'prismaquant/replay.py' in build_files.Graph(found).test_files('tests/test_entrypoints.py')


def test_an_entrypoint_string_with_an_attribute_names_its_module(tmp_path):
    found = evidence(tmp_path, {
        'prismaquant/lane_spec.py': ('import json\nfrom importlib import import_module\nfrom pathlib import Path\n'
                                     '_SPEC_DIR = Path(__file__).resolve().parent / "lane_specs"\n'
                                     'def consumer(name):\n'
                                     '    spec = json.loads((_SPEC_DIR / f"{name}.json").read_text())\n'
                                     '    return import_module(spec["entrypoint"].split(":")[0])\n'),
        'prismaquant/lane_specs/ct.json': '{"entrypoint": "prismaquant.validate_kl:main"}',
        'prismaquant/validate_kl.py': ''})
    assert 'prismaquant/validate_kl.py' in found.of('prismaquant/lane_spec.py')[0]


def test_modules_a_test_inputs_entry_names_become_the_tests_dependencies(tmp_path):
    sources = dict(SMALL, **{
        'tools/bazel/test_inputs.bzl': ('TEST_INPUTS = {\n    "//tests:test_held": struct(\n'
                                        '        modules = ["prismaquant/child.py"],\n'
                                        '        reason = "it runs prismaquant.child from a computed name.",\n    ),\n}\n')})
    generated = build_files.generate(tmp_path, write(tmp_path, sources))['tests']
    held = generated[generated.index('name = "test_held"'):].split('\n)\n', 1)[0]
    assert '"//:prismaquant/child"' in held
    sources['tools/bazel/test_inputs.bzl'] = sources['tools/bazel/test_inputs.bzl'].replace('child.py', 'gone.py')
    with pytest.raises(build_files.GenerationError, match='prismaquant/gone.py'):
        build_files.generate(tmp_path, write(tmp_path, sources))


def test_a_deselected_node_with_a_space_stays_one_argument():
    plan = {'selected': [{'path': 'tests/test_in.py'}], 'deferred': [],
            'deselect': ["tests/test_in.py::test_x[a b]"]}
    args = build_files.test_attributes('tests/test_in.py', plan, {})['args']
    assert build_files.shlex.split(args[0]) == ["--deselect=tests/test_in.py::test_x[a b]"]


def test_a_package_that_redirects_its_path_serves_that_directorys_modules(tmp_path):
    found = evidence(tmp_path, {
        'tools/g3job/g3_policy/__init__.py': ('from pathlib import Path\n'
                                              '__path__ = [str(Path(__file__).resolve().parents[3] / "prismaquant")]\n'),
        'tests/g3job/test_a.py': 'from g3_policy.staged_lease import ReadsetUnbound\nfrom g3_policy import exl3\n',
        'prismaquant/staged_lease.py': '', 'prismaquant/exl3.py': ''})
    assert {'prismaquant/staged_lease.py', 'prismaquant/exl3.py'} <= found.of('tests/g3job/test_a.py')[0]


def test_exact_names_still_count_beside_a_whole_repository_read(tmp_path):
    found = evidence(tmp_path, {
        'tests/test_golden.py': ('import importlib.util\nfrom pathlib import Path\n'
                                 'REPOSITORY = Path(__file__).resolve().parents[1]\n'
                                 'def _load(name, path):\n'
                                 '    return importlib.util.spec_from_file_location(name, REPOSITORY / path)\n'
                                 'WIKITEXT = _load("wikitext", "tools/wikitext_inputs.py")\n'),
        'tools/wikitext_inputs.py': 'import torch\n'})
    modules, data = found.of('tests/test_golden.py')
    assert 'tools/wikitext_inputs.py' in modules
