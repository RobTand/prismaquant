"""A source change selects only the tests that depend on it, and that only improves (#3215).

``tools/ci/import_fanout.py`` follows each gate test's static imports,
including every conftest that applies to the test. It counts, for each
``prismaquant/`` and ``tools/`` module, how many gate tests depend on it.
``tests/fixtures/import_fanout_baseline.json`` records today's state. A new
test-to-test import fails, as does a new source import in a conftest or a
summary figure worse than the baseline. So does a baseline that no longer
matches the state: each gain shrinks it in the same change, by running
``python tools/ci/import_fanout.py --write-baseline``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('import_fanout', ROOT / 'tools/ci/import_fanout.py')
fanout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fanout)


def write_tree(root: Path, sources: dict[str, str]) -> Path:
    for name, content in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return root


def test_every_spelling_of_an_import_is_found():
    source = (
        'from prismaquant import (  # a comment with a ) in it\n'
        '    allocator,\n'
        '    digests as owner,\n'
        ')\n'
        'from prismaquant.model_profiles import \\\n'
        '    registry\n'
        'def later():\n'
        '    import g3_lib\n'
        'CODE = "import sys; from prismaquant import shipcard"\n'
        'loaded = importlib.import_module("tessera_campaign_container")\n'
        'pytest_plugins = ("prismaquant.test_bound",)\n'
        '"""\n'
        'import the data, then read it.\n'
        '"""\n'
    )
    names = fanout.imported_names(source, 'tests/test_x.py')
    assert {'prismaquant', 'prismaquant.allocator', 'prismaquant.digests', 'prismaquant.model_profiles',
            'prismaquant.model_profiles.registry', 'g3_lib', 'sys', 'prismaquant.shipcard',
            'tessera_campaign_container', 'prismaquant.test_bound'} <= names
    assert 'the' not in names


def test_a_relative_import_resolves_inside_its_package():
    assert 'prismaquant.digests' in fanout.imported_names('from .digests import x\n', 'prismaquant/allocator.py')
    assert 'prismaquant' in fanout.imported_names('from . import digests\n', 'prismaquant/allocator.py')
    assert 'prismaquant.model_profiles.base' in fanout.imported_names(
        'from .base import ModelProfile\n', 'prismaquant/model_profiles/qwen3.py')


def test_a_conftest_selects_every_test_beneath_it(tmp_path):
    root = write_tree(tmp_path, {
        'tests/conftest.py': 'def fixture():\n    from prismaquant import core\n',
        'tests/test_a.py': 'def test_a(): pass\n',
        'tests/test_b.py': 'from prismaquant import pool\n',
        'tests/g3job/conftest.py': 'import prismaquant.lease\n',
        'tests/g3job/test_c.py': 'def test_c(): pass\n',
        'prismaquant/__init__.py': '',
        'prismaquant/core.py': '',
        'prismaquant/pool.py': 'from prismaquant import core\n',
        'prismaquant/lease.py': '',
    })
    live = fanout.measure_fanout(root, population=['tests/test_a.py', 'tests/test_b.py', 'tests/g3job/test_c.py'])
    assert live['dependents']['prismaquant/core.py'] == 3
    assert live['dependents']['prismaquant/pool.py'] == 1
    assert live['dependents']['prismaquant/lease.py'] == 1
    assert live['conftest_source_modules'] == 3
    assert live['conftest_source_imports'] == [
        'tests/conftest.py -> prismaquant/__init__.py',
        'tests/conftest.py -> prismaquant/core.py',
        'tests/g3job/conftest.py -> prismaquant/__init__.py',
        'tests/g3job/conftest.py -> prismaquant/lease.py',
    ]
    # four modules: core 3/3, the package 3/3, pool 1/3 and lease 1/3
    assert (live['median_selected_pct'], live['p90_selected_pct'], live['modules_over_half']) == (33.3, 100.0, 2)


def test_a_tool_named_by_path_brings_its_imports_and_a_named_test_does_not(tmp_path):
    root = write_tree(tmp_path, {
        'tests/test_runs_a_tool.py': 'TOOL = ROOT / "tools/fleet/runner.py"\n',
        'tests/test_names_a_test.py': 'MATCHED = item.path.name == "test_cycle.py"\n',
        'tests/test_cycle.py': 'import alpha\n',
        'tools/fleet/runner.py': 'import heavy\n',
        'tools/heavy.py': '',
        'tools/alpha.py': 'import beta\n',
        'tools/beta.py': 'import alpha\n',
    })
    live = fanout.measure_fanout(root, population=['tests/test_runs_a_tool.py', 'tests/test_names_a_test.py',
                                            'tests/test_cycle.py'])
    # the alpha/beta cycle terminates, and the named test file's import of alpha is not followed
    assert live['dependents'] == {'tools/alpha.py': 1, 'tools/beta.py': 1,
                                  'tools/fleet/runner.py': 1, 'tools/heavy.py': 1}


def test_imports_through_experiments_are_followed_to_source_modules(tmp_path):
    root = write_tree(tmp_path, {
        'tests/test_exp.py': 'from experiments.sweep import runner\n',
        'experiments/sweep/runner.py': 'from prismaquant import allocator\n',
        'prismaquant/allocator.py': '',
    })
    live = fanout.measure_fanout(root, population=['tests/test_exp.py'])
    assert live['dependents'] == {'prismaquant/allocator.py': 1}


def test_only_imports_of_test_modules_are_test_to_test(tmp_path):
    root = write_tree(tmp_path, {
        'tests/test_a.py': 'from test_b import HELPER\nimport fleet_fixture\n',
        'tests/test_b.py': 'HELPER = 1\n',
        'tests/fleet_fixture.py': 'from test_b import HELPER\n',
    })
    live = fanout.measure_fanout(root, population=['tests/test_a.py', 'tests/test_b.py'])
    assert live['test_to_test_imports'] == ['tests/fleet_fixture.py -> tests/test_b.py',
                                            'tests/test_a.py -> tests/test_b.py']


def test_the_population_is_the_gates_own(tmp_path):
    root = write_tree(tmp_path, {
        'tools/ci/gate_durations.json': json.dumps({'files': {}}),
        'tools/ci/gate_quarantine.json': json.dumps({'cost_exclusions': [
            {'target': 'tests/test_costly.py', 'issue': 7, 'reason': 'slow'}]}),
        'tests/test_plain.py': '', 'tests/test_costly.py': '', 'tests/helper.py': '',
    })
    assert fanout.gate_population(root) == ['tests/test_plain.py']


BASE = {'evidence': 'imports', 'median_selected_pct': 10.0, 'p90_selected_pct': 50.0,
        'modules_over_half': 4, 'conftest_source_modules': 3,
        'conftest_source_imports': ['tests/conftest.py -> prismaquant/core.py'],
        'test_to_test_imports': ['tests/test_a.py -> tests/test_b.py']}


def test_a_regression_fails_and_noise_within_tolerance_does_not():
    noise = {**BASE, 'median_selected_pct': 10.9, 'modules_over_half': 6}
    assert fanout.regressions(noise, BASE) == []
    worse = {**BASE, 'p90_selected_pct': 51.1, 'modules_over_half': 7, 'conftest_source_modules': 4,
             'conftest_source_imports': [*BASE['conftest_source_imports'], 'tests/conftest.py -> tools/launch.py'],
             'test_to_test_imports': [*BASE['test_to_test_imports'], 'tests/test_c.py -> tests/test_b.py']}
    assert fanout.regressions(worse, BASE) == [
        'new conftest_source_imports entry: tests/conftest.py -> tools/launch.py',
        'new test_to_test_imports entry: tests/test_c.py -> tests/test_b.py',
        'p90_selected_pct rose from 50.0 to 51.1 (tolerance 1.0)',
        'modules_over_half rose from 4 to 7 (tolerance 2)',
        'conftest_source_modules rose from 3 to 4 (tolerance 0)',
    ]


def test_a_gain_must_be_recorded():
    better = {**BASE, 'median_selected_pct': 8.9, 'conftest_source_modules': 2,
              'conftest_source_imports': [], 'test_to_test_imports': []}
    assert fanout.stale_entries(better, BASE) == [
        'conftest_source_imports entry is gone: tests/conftest.py -> prismaquant/core.py',
        'test_to_test_imports entry is gone: tests/test_a.py -> tests/test_b.py',
        'median_selected_pct fell from 10.0 to 8.9',
        'conftest_source_modules fell from 3 to 2',
    ]
    assert fanout.stale_entries({**BASE, 'median_selected_pct': 9.5}, BASE) == []


def test_recording_a_gain_never_records_a_tolerated_rise():
    # Otherwise each unrelated gain would bank the rise within tolerance, and the
    # next change could rise again: the ratchet would loosen itself.
    live = {**BASE, 'median_selected_pct': 9.0, 'p90_selected_pct': 50.9, 'modules_over_half': 6}
    record = fanout.baseline_record(live, BASE)
    assert (record['median_selected_pct'], record['p90_selected_pct'], record['modules_over_half']) == (
        9.0, BASE['p90_selected_pct'], BASE['modules_over_half'])


def test_write_baseline_records_a_gain_and_refuses_a_loss(tmp_path, monkeypatch, capsys):
    root = write_tree(tmp_path, {'tests/test_a.py': 'from test_b import X\n', 'tests/test_b.py': 'X = 1\n'})
    monkeypatch.setattr(fanout, 'gate_population', lambda root: ['tests/test_a.py', 'tests/test_b.py'])
    baseline = root / 'tests/fixtures/import_fanout_baseline.json'
    baseline.parent.mkdir(parents=True)
    clean = {**BASE, 'median_selected_pct': 0.0, 'p90_selected_pct': 0.0, 'modules_over_half': 0,
             'conftest_source_modules': 0, 'conftest_source_imports': [], 'test_to_test_imports': []}
    baseline.write_text(json.dumps(clean))
    assert fanout.main(['--root', str(root), '--write-baseline']) == 1
    assert 'new test_to_test_imports entry: tests/test_a.py -> tests/test_b.py' in capsys.readouterr().err
    assert json.loads(baseline.read_text()) == clean
    baseline.write_text(json.dumps({**clean, 'test_to_test_imports': ['tests/test_a.py -> tests/test_b.py']}))
    (root / 'tests/test_a.py').write_text('X = 1\n')
    assert fanout.main(['--root', str(root), '--write-baseline']) == 0
    assert json.loads(baseline.read_text()) == clean


@pytest.fixture(scope='module')
def live_fanout():
    return fanout.measure_fanout(ROOT)


def baseline():
    return json.loads(fanout.BASELINE.read_text(encoding='utf-8'))


def test_the_import_fanout_does_not_regress(live_fanout):
    assert not fanout.regressions(live_fanout, baseline()), (
        'the test import graph grew: import a helper from a support module, not from a test '
        'module, and keep source imports out of conftest (#3215)')


def test_the_import_fanout_baseline_records_every_gain(live_fanout):
    assert not fanout.stale_entries(live_fanout, baseline()), (
        'record the gain: python tools/ci/import_fanout.py --write-baseline')
