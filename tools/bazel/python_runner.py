"""Run one gate test file as tools/ci/gate_run.py runs a hosted shard.

pytest runs from the repository root with pytest.ini and the conftest chain,
the gate's environment, pytest-timeout's 600 s and the gate's own_cgroup
exclusion, and its node-level deselects. A file whose every node is deselected
passes, as the gate accepts pytest's exit 5 for a complete collection. The
per-test budgets in tools/ci/gate_budget.json produce warnings without changing
the test verdict. A file Bazel shards
runs every TEST_TOTAL_SHARDS-th collected test. After the run,
test.outputs/inventory.json records each node's outcome for parity.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

# Import siblings by path: tools/ci has modules with common names.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import checkout_of, prepare_remote_test  # noqa: E402

SHARD_PLUGIN = '''
import os

def pytest_collection_modifyitems(config, items):
    index, total = int(os.environ['PQ_BAZEL_SHARD_INDEX']), int(os.environ['PQ_BAZEL_TOTAL_SHARDS'])
    keep = [item for position, item in enumerate(items) if position % total == index]
    config.hook.pytest_deselected(items=[item for item in items if item not in keep])
    items[:] = keep
'''
# gate_collect records each node's collected identity, which gate_run's
# budget timings read; a test defined in an imported module has no other.
IDENTITY_PLUGIN = '''
def pytest_collection_modifyitems(items):
    for item in items:
        item.user_properties.append(('gate_nodeid', item.nodeid))
'''
NO_TESTS_COLLECTED = 5


def outcomes(junit):
    """Node ID to passed, skipped or failed, from an xunit1 report."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ci'))
    from gate_run import gate_junit_identity

    found = {}
    for case in ET.parse(junit).iter('testcase'):
        _, node = gate_junit_identity(case)
        if node is None:
            continue
        if case.find('failure') is not None or case.find('error') is not None:
            found[node] = 'failed'
        elif case.find('skipped') is not None:
            found[node] = 'skipped' if found.get(node) != 'failed' else 'failed'
        else:
            found.setdefault(node, 'passed')
    return found


def budget_overruns(root, junit, path):
    """gate_run's per-test budgets. The gate checks separately that they never grow."""
    sys.path.insert(0, str(root / 'tools/ci'))
    import gate_run
    config = gate_run._budget_limits(json.loads((root / 'tools/ci/gate_budget.json').read_text()))
    return gate_run._budget_timings(junit, [path], config)


def file_verdict(status, nodes):
    """The file's exit status: a failed node fails it, and exit 5 with nothing failed passes."""
    if any(outcome == 'failed' for outcome in nodes.values()):
        return 1
    return 0 if status in (0, NO_TESTS_COLLECTED) else status


def main():
    root = Path.cwd().resolve()
    if os.environ.get('PQ_BAZEL_LANE') == 'runner':
        # Hosted shards run pytest in a git checkout with its own file modes.
        root = checkout_of(root)
    path, extra = sys.argv[1], sys.argv[2:]
    scratch, environment = prepare_remote_test(root)
    private = Path(os.environ['TEST_TMPDIR'])
    outputs = Path(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', private))
    junit = Path(os.environ.get('XML_OUTPUT_FILE') or outputs / 'junit.xml')
    (private / 'bazel_identity.py').write_text(IDENTITY_PLUGIN)
    environment['PYTHONPATH'] = str(private) + os.pathsep + environment['PYTHONPATH']
    plugins = ['-p', 'bazel_identity']
    if os.environ.get('TEST_TOTAL_SHARDS'):
        Path(os.environ['TEST_SHARD_STATUS_FILE']).touch()
        (private / 'bazel_shard.py').write_text(SHARD_PLUGIN)
        environment['PQ_BAZEL_SHARD_INDEX'] = os.environ['TEST_SHARD_INDEX']
        environment['PQ_BAZEL_TOTAL_SHARDS'] = os.environ['TEST_TOTAL_SHARDS']
        plugins += ['-p', 'bazel_shard']
    command = [sys.executable, '-m', 'pytest', path, '-q', '-ra', '--timeout=600', '-m', 'not own_cgroup',
               '--rootdir=' + str(root), '--junitxml=' + str(junit), '-o', 'junit_family=xunit1',
               '-o', 'cache_dir=' + str(private / 'pytest-cache'), *plugins, *extra]
    try:
        status = subprocess.run(command, cwd=root, env=environment, check=False).returncode
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    try:
        nodes = outcomes(junit)
    except (OSError, ET.ParseError) as error:
        print(f'no JUnit verdict for {path}: {error}', file=sys.stderr)
        return status or 1
    status = file_verdict(status, nodes)
    overruns = []
    if status == 0 and nodes:
        overruns = budget_overruns(root, junit, path)
        for row in overruns:
            print(f"BUDGET_WARNING: {row['nodeid']}: {row['seconds']:.3f}s > "
                  f"{row['budget_seconds']:.3f}s policy", file=sys.stderr)
    if status == 0:
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / 'inventory.json').write_text(json.dumps(
            {'file': path, 'shard': os.environ.get('TEST_SHARD_INDEX'), 'nodes': nodes,
             'budget_warnings': overruns}))
    return status


if __name__ == '__main__':
    sys.exit(main())
