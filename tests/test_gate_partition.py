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
