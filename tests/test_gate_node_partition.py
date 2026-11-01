"""Heavy-file splitting keeps live pytest nodes covered exactly once."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'tools/ci/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


partition, runner, actions = [load(name) for name in ('gate_partition', 'gate_run', 'gate_actions')]
FILE = 'tests/test_heavy.py'
NODES = {f'{FILE}::test_case[{i}]': seconds for i, seconds in enumerate([90, 80, 70, 60, 50, 40])}


def assignments():
    return partition.shards([FILE, 'tests/test_light.py'], {FILE: 390, 'tests/test_light.py': 3},
                            3, unknown_seconds=2, node_durations=NODES)


def test_heavy_file_splits_but_ordinary_files_stay_whole():
    rows = assignments()
    owners = rows[0]['node_partition']['owners'][FILE]
    assert set(owners) == set(NODES)
    assert set(owners.values()) == {0, 1, 2}
    assert max(row['estimated_seconds'] for row in rows) <= 140
    assert sum('tests/test_light.py' in row['files'] for row in rows) == 1
    assert all(FILE in row['files'] for row in rows)


def test_all_shards_recompute_identical_assignment_despite_input_order_and_hash_seed(tmp_path):
    expected = assignments()
    assert expected == partition.shards(['tests/test_light.py', FILE], {'tests/test_light.py': 3, FILE: 390},
        3, unknown_seconds=2, node_durations=dict(reversed(list(NODES.items()))))
    population = tmp_path / 'files.txt'; population.write_text(FILE + '\ntests/test_light.py\n')
    costs = tmp_path / 'costs.json'; costs.write_text(json.dumps({'files': {FILE: 390, 'tests/test_light.py': 3}, 'nodes': NODES}))
    for index in range(3):
        output = tmp_path / f'assignment-{index}.json'
        run = subprocess.run([sys.executable, str(ROOT / 'tools/ci/gate_partition.py'), 'shard',
            '--files-from', str(population), '--durations', str(costs), '--shards', '3', '--index', str(index),
            '--output-files', str(tmp_path / 'assigned.txt'), '--output-assignment', str(output)],
            capture_output=True, text=True, check=False)
        assert run.returncode == 0, run.stderr
        assert json.loads(output.read_text()) == expected[index]


@pytest.mark.parametrize('changed', ['added', 'removed'])
def test_live_collection_runs_each_node_once_after_duration_inventory_changes(tmp_path, changed):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tools/ci').mkdir(parents=True)
    (tmp_path / 'tools/ci/gate_budget.json').write_text('{"seconds":20,"slow":{}}')
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    cases = list(range(7)) if changed == 'added' else list(range(5))
    (tmp_path / FILE).write_text('import pytest\n'
        f'@pytest.mark.parametrize("case", {cases!r}, ids=str)\n'
        'def test_case(case): assert case >= 0\n')
    (tmp_path / 'tests/test_light.py').write_text('def test_light(): pass\n')
    plan = {'mode': 'full', 'head_sha': 'head', 'base_sha': None, 'population': {'version': 'population'},
            'selected': [{'path': FILE}, {'path': 'tests/test_light.py'}], 'deselect': []}
    rows, records, executed = assignments(), {}, []
    for row in rows:
        index = row['index']; output = tmp_path / f'result-{index}.json'; junit = tmp_path / f'junit-{index}.xml'
        code = runner.run_plan(plan, assignment=row, root=tmp_path, output=output, junit=junit,
                               workers=1, affinity={0})
        assert code == 0, output.read_text()
        records[index] = json.loads(output.read_text())
        executed.extend(runner.gate_junit_identity(case)[1] for case in ET.parse(junit).iter('testcase'))
    expected = {f'{FILE}::test_case[{i}]' for i in cases} | {'tests/test_light.py::test_light'}
    assert set(executed) == expected and len(executed) == len(expected)
    result = actions.assess_attempts(plan, rows, {1: records})
    assert result['exit_code'] == 0
    assert result['node_coverage'] == {'split_files': 1, 'nodes': len(cases), 'exactly_once': True}
    # Both missing and duplicate execution must invalidate an otherwise passing shard.
    owner = next(index for index, record in records.items() if record['node_coverage']['executed'])
    original = records[owner]['node_coverage']['executed'][:]
    for corrupted in [original[1:], original + original[:1]]:
        records[owner]['node_coverage']['executed'] = corrupted
        assert actions.assess_attempts(plan, rows, {1: records})['exit_code'] == 75
    records[owner]['node_coverage']['executed'] = original
    # Collection drift between architectures is a missing verdict, never a pass.
    records[owner]['node_coverage']['collected'][FILE].append(FILE + '::test_architecture_only')
    assert actions.assess_attempts(plan, rows, {1: records})['exit_code'] == 75
