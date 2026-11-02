"""Duration balancing preserves coverage and uses both hosted architectures."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_partition', ROOT / 'tools/ci/gate_partition.py')
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)


def test_longest_files_balance_deterministically():
    files = ['a', 'b', 'c', 'd']
    rows = partition.shards(files, dict(zip(files, [10, 9, 2, 1])), 2, unknown_seconds=3)
    assert [row['files'] for row in rows] == [['a', 'd'], ['b', 'c']]
    assert [row['estimated_seconds'] for row in rows] == [11, 11]


def test_large_selection_fans_out_and_unknown_cost_is_conservative():
    files = [str(i) for i in range(30)]
    costs = {path: 100 for path in files}
    assert partition.selection_shard_count(files, costs, 12, workers=4, single_shard_seconds=90) == 12
    assert partition.default_unknown_seconds({'a': 2, 'b': 19}) == 19


def test_empty_selection_has_a_valid_noop_assignment():
    rows = partition.shards([], {}, 1, unknown_seconds=1)
    assert rows[0]['files'] == []


def test_runner_weighting_gives_x64_less_work_and_lowers_expected_tail():
    files = [f'tests/test_{i:02}.py' for i in range(20)]
    costs = dict.fromkeys(files, 100)
    ordinary = partition.shards(files, costs, 2, unknown_seconds=1)
    weighted = partition.shards(files, costs, 2, unknown_seconds=1, runner_factors=[1 / 1.6, 1])
    assert len(weighted[0]['files']) > len(weighted[1]['files'])
    assert max(row['estimated_seconds'] for row in weighted) < max(row['estimated_seconds'] for row in ordinary)
    swapped = partition.shards(files, costs, 2, unknown_seconds=1, runner_factors=[1, 1 / 1.6])
    assert len(swapped[1]['files']) > len(swapped[0]['files'])


def test_every_shard_recomputes_weighted_assignment_from_the_actual_matrix(tmp_path):
    import json
    files = ['tests/test_heavy.py', 'tests/test_light.py']
    costs = {files[0]: 300, files[1]: 10}
    nodes = {f'{files[0]}::test_{i}': 50 for i in range(6)}
    population = tmp_path / 'files.txt'; population.write_text('\n'.join(files) + '\n')
    durations = tmp_path / 'costs.json'; durations.write_text(json.dumps({'files': costs, 'nodes': nodes}))
    matrix = tmp_path / 'matrix.json'
    common = ['--files-from', str(population), '--durations', str(durations), '--shards', '2']
    assert partition.main(['matrix', *common, '--runners', 'ubuntu-24.04-arm,ubuntu-24.04', '--output', str(matrix)]) == 0
    recorded = json.loads(matrix.read_text())
    assert recorded['runner_cost_factors'] == [1 / 1.6, 1]
    expected = partition.shards(files, costs, 2, unknown_seconds=partition.default_unknown_seconds(costs),
                                node_durations=nodes, runner_factors=recorded['runner_cost_factors'])
    for index in range(2):
        output = tmp_path / f'assignment-{index}.json'
        assert partition.main(['shard', *common, '--matrix', str(matrix), '--index', str(index),
            '--output-files', str(tmp_path / 'assigned.txt'), '--output-assignment', str(output)]) == 0
        assert json.loads(output.read_text()) == expected[index]


def test_one_heavy_selected_file_can_use_its_recorded_node_parallelism():
    path = 'tests/test_heavy.py'
    nodes = {f'{path}::test_case[{i}]': 100 for i in range(4)}
    assert partition.selection_shard_count([path], {path: 400}, 12,
        workers=4, single_shard_seconds=90, node_durations=nodes) == 4


def test_selection_matrix_splits_a_single_heavy_file(tmp_path):
    import json
    path = 'tests/test_heavy.py'
    population = tmp_path / 'files.txt'; population.write_text(path + '\n')
    durations = tmp_path / 'costs.json'; durations.write_text(json.dumps({
        'files': {path: 400}, 'nodes': {f'{path}::test_case[{i}]': 100 for i in range(4)}}))
    output = tmp_path / 'matrix.json'
    assert partition.main(['matrix', '--files-from', str(population), '--durations', str(durations),
        '--shards', '12', '--single-shard-seconds', '90', '--runners',
        'ubuntu-24.04-arm,ubuntu-24.04', '--output', str(output)]) == 0
    matrix = json.loads(output.read_text())
    assert matrix['shards'] == 4


def test_dedicated_graph_check_has_no_shard_owner_or_cost():
    path = 'tests/test_graph_impact.py'
    live = path + '::test_graph_impact_gate'
    unit = path + '::test_vendored_core_hash'
    rows = partition.shards([path], {path: 175}, 2, unknown_seconds=1,
                            node_durations={live: 170, unit: 5}, excluded=[live])
    assert sum(row['estimated_seconds'] for row in rows) == 5
    assert all(live not in row.get('node_partition', {}).get('owners', {}).get(path, {})
               for row in rows)
    collected = {path: [live, unit]}
    assigned = [node for row in rows for node in partition.assigned_nodes(collected, row['node_partition'], [live])]
    assert assigned == [unit]


def test_matrix_and_shard_use_plan_exclusions_consistently(tmp_path):
    import json
    path = 'tests/test_graph_impact.py'
    live, unit = path + '::test_graph_impact_gate', path + '::test_vendored_core_hash'
    population = tmp_path / 'files.txt'; population.write_text(path + '\n')
    durations = tmp_path / 'costs.json'
    durations.write_text(json.dumps({'files': {path: 175}, 'nodes': {live: 170, unit: 5}}))
    plan = tmp_path / 'plan.json'; plan.write_text(json.dumps({'deselect': [live]}))
    common = ['--files-from', str(population), '--durations', str(durations), '--shards', '2',
              '--plan', str(plan)]
    matrix = tmp_path / 'matrix.json'
    assert partition.main(['matrix', *common, '--runners', 'ubuntu-24.04', '--output', str(matrix)]) == 0
    assert json.loads(matrix.read_text())['total_estimated_seconds'] == 5
    for index in range(2):
        output = tmp_path / f'assignment-{index}.json'
        assert partition.main(['shard', *common, '--matrix', str(matrix), '--index', str(index),
            '--output-files', str(tmp_path / 'assigned.txt'), '--output-assignment', str(output)]) == 0
        row = json.loads(output.read_text())
        assert live not in row.get('node_partition', {}).get('owners', {}).get(path, {})
