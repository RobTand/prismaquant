#!/usr/bin/env python3
"""Run any CI-assigned gate files. Exit 0 pass, 1 test failure, 75 no verdict."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition import assigned_nodes, validate_node_partition

NO_VERDICT = 75
BUDGET_ENFORCEMENT_TOLERANCE = 1.5


class BudgetFailure(ValueError):
    """A definite policy failure, never an infrastructure retry."""


def _budget_limits(config):
    if not isinstance(config, dict) or not isinstance(config.get('slow'), dict):
        raise BudgetFailure('Test budget requires seconds and a slow map')
    file_limits = config.get('file_seconds', {})
    if not isinstance(file_limits, dict):
        raise BudgetFailure('File budgets require a map')
    limits = [config.get('seconds'), *config['slow'].values(), *file_limits.values()]
    if (any(isinstance(limit, bool) or not isinstance(limit, (int, float)) or
            not math.isfinite(limit) or limit <= 0 for limit in limits) or
            any(not isinstance(name, str) or not name for name in config['slow'])):
        raise BudgetFailure('Test budget limits must be finite positive seconds')
    if (any(not isinstance(name, str) or not name.startswith('tests/') or
            not name.endswith('.py') or '::' in name or '..' in Path(name).parts
            for name in file_limits) or
            any(name.split('::', 1)[0] in file_limits for name in config['slow'])):
        raise BudgetFailure('File budgets must name test files without overlapping node exceptions')
    return config


MIGRATIONS_PATH = 'tools/ci/gate_budget_migrations.json'
MIGRATION_RUNNERS = frozenset({'ubuntu-24.04', 'ubuntu-24.04-arm'})


def _finite_positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and value > 0)


def _migration_limits(root, base_policy):
    """Node limits a reviewed migration grants against this exact base policy.

    ``docs/ci_gate.md`` reserves budget growth for an explicit migration the
    readiness lead reviews after a verified runner or dependency change. Each
    record names the Git blob of the base ``gate_budget.json`` it was measured
    against and, per node, hosted evidence from both architectures; its limit
    is ``ceil(2 * largest hosted time)``. A malformed record fails closed.
    """
    path = root / MIGRATIONS_PATH
    if base_policy is None or not path.exists():
        return {}
    try:
        document = json.loads(path.read_text())
    except ValueError as error:
        raise BudgetFailure(f'Budget migration record is not JSON: {error}') from None
    if (not isinstance(document, dict) or
            document.get('schema') != 'prismaquant.gate_budget_migration.v1' or
            not isinstance(document.get('migrations'), list)):
        raise BudgetFailure('Budget migration record has an unknown shape')
    granted = {}
    for migration in document['migrations']:
        if (not isinstance(migration, dict) or
                not isinstance(migration.get('base_policy'), str) or
                len(migration['base_policy']) != 40 or
                not isinstance(migration.get('reason'), str) or not migration['reason'].strip() or
                not isinstance(migration.get('slow'), dict) or not migration['slow']):
            raise BudgetFailure('Budget migration needs a base policy blob, a reason and nodes')
        for name, entry in migration['slow'].items():
            rows = entry.get('evidence') if isinstance(entry, dict) else None
            if (not isinstance(name, str) or '::' not in name or not isinstance(rows, list) or
                    not rows or any(not isinstance(row, dict) or
                                    isinstance(row.get('run'), bool) or
                                    not isinstance(row.get('run'), int) or
                                    row.get('runner') not in MIGRATION_RUNNERS or
                                    not _finite_positive(row.get('seconds')) for row in rows) or
                    {row['runner'] for row in rows} != MIGRATION_RUNNERS):
                raise BudgetFailure(f'Budget migration for {name} needs hosted evidence '
                                    'from both architectures')
            limit = math.ceil(2 * max(row['seconds'] for row in rows))
            if entry.get('seconds') != limit:
                raise BudgetFailure(f'Budget migration for {name} must be ceil(2 x largest '
                                    f'hosted time) = {limit}')
            if migration['base_policy'] == base_policy:
                granted[name] = max(granted.get(name, 0), limit)
    return granted


def _budget_policy(root, base_sha):
    path = 'tools/ci/gate_budget.json'
    current = _budget_limits(json.loads((root / path).read_text()))
    baseline = current
    base_policy = None
    if base_sha:
        # Check the commit separately: an unavailable target is not a bootstrap.
        target = subprocess.run(['git', 'cat-file', '-e', f'{base_sha}^{{commit}}'],
                                cwd=root, capture_output=True, check=False)
        if target.returncode:
            raise RuntimeError('budget target commit is unavailable: ' + base_sha)
        tree = subprocess.run(['git', 'ls-tree', base_sha, '--', path],
                              cwd=root, text=True, capture_output=True, check=True)
        if tree.stdout.strip():
            base_policy = tree.stdout.split()[2]
            blob = subprocess.run(['git', 'show', f'{base_sha}:{path}'], cwd=root,
                                  text=True, capture_output=True, check=True)
            baseline = _budget_limits(json.loads(blob.stdout))
    granted = _migration_limits(root, base_policy)
    allowed = {**baseline['slow'], **{name: max(limit, baseline['slow'].get(name, 0))
                                      for name, limit in granted.items()}}
    if (baseline.get('enforced', True) and not current.get('enforced', True) or
            current['seconds'] > baseline['seconds'] or
            not current['slow'].keys() <= allowed.keys() or
            any(limit > allowed[name] for name, limit in current['slow'].items()) or
            not current.get('file_seconds', {}).keys() <= baseline.get('file_seconds', {}).keys() or
            any(limit > baseline['file_seconds'][name]
                for name, limit in current.get('file_seconds', {}).items())):
        raise BudgetFailure('Test budgets and the explicit slow list may not grow versus the target')
    return current


def gate_junit_identity(case, files=None):
    """Read collection identity even when pytest reports an imported definition."""
    identities = [prop.get('value', '') for prop in case.findall('properties/property')
                  if prop.get('name') == 'gate_nodeid']
    name, classname = case.get('name', ''), case.get('classname', '')
    filename = case.get('file', '').removeprefix('./')
    if identities:
        if len(identities) != 1 or not identities[0] or '::' not in identities[0]:
            raise ValueError('JUnit has invalid collected test identity')
        nodeid = identities[0]
        filename = nodeid.split('::', 1)[0]
        if not name or not nodeid.endswith('::' + name):
            raise ValueError('JUnit collected test name is inconsistent')
        prefix = nodeid[:-(len(name) + 2)]
        module = filename.removesuffix('.py').replace('/', '.')
        expected_class = module + prefix[len(filename):].replace('::', '.')
        if classname != expected_class:
            raise ValueError('JUnit collected test class is inconsistent')
    else:
        if not filename or not name:
            raise ValueError('JUnit timing has missing or unexpected test identity')
        module = filename.removesuffix('.py').replace('/', '.')
        if case.find('skipped') is not None and not classname:
            nodeid = None
        else:
            if classname != module and not classname.startswith(module + '.'):
                raise ValueError('JUnit timing cannot resolve test class: ' + classname)
            classes = classname[len(module):].lstrip('.').replace('.', '::')
            nodeid = filename + '::' + (classes + '::' if classes else '') + name
    if files is not None and filename not in files:
        raise ValueError('JUnit timing has missing or unexpected test identity')
    return filename, nodeid


def _collection_files(collection, target=None):
    files = set(collection['seen'])
    if target and '::' in target:
        # Pytest does not emit module collectreports for explicit node targets.
        # modifyitems still records their collection identities, including nodes
        # deselected from the parallel phase for own_cgroup execution.
        files.update(node.split('::', 1)[0] for node in collection['collected'])
    return files


def _budget_timings(junit, files, config, collected_files=None):
    observed, seconds, file_seconds = set(), {}, {}
    for case in ET.parse(junit).iter('testcase'):
        filename, nodeid = gate_junit_identity(case, files)
        observed.add(filename)
        if nodeid is None:
            continue
        try:
            elapsed = float(case.attrib['time'])
        except (KeyError, ValueError) as error:
            raise ValueError('JUnit timing missing or invalid for ' + nodeid) from error
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('JUnit timing must be finite nonnegative seconds: ' + nodeid)
        observed.add(filename)
        if filename in config.get('file_seconds', {}):
            file_seconds[filename] = file_seconds.get(filename, 0) + elapsed
        else:
            seconds[nodeid] = seconds.get(nodeid, 0) + elapsed
    if (observed if collected_files is None else set(collected_files)) != set(files):
        raise ValueError('JUnit timings do not cover every assigned file')
    rows = [{'nodeid': nodeid, 'seconds': elapsed,
             'budget_seconds': config['slow'].get(nodeid, config['seconds'])}
            for nodeid, elapsed in sorted(seconds.items())
            if elapsed > config['slow'].get(nodeid, config['seconds'])]
    rows.extend({'nodeid': filename, 'seconds': elapsed,
                 'budget_seconds': config['file_seconds'][filename]}
                for filename, elapsed in sorted(file_seconds.items())
                if elapsed > config['file_seconds'][filename])
    return rows


def _write_budget_warnings(junit, warnings):
    messages = [
        f"BUDGET_WARNING: {row['nodeid']}: {row['seconds']:.3f}s > "
        f"{row['budget_seconds']:.3f}s policy; within "
        f"{BUDGET_ENFORCEMENT_TOLERANCE:g}x tolerance "
        f"({row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE:.3f}s)"
        for row in warnings]
    text = '\n'.join(messages)
    print(text)
    tree = ET.parse(junit)
    suite = next(tree.iter('testsuite'), tree.getroot())
    output = suite.find('system-out')
    if output is None:
        output = ET.SubElement(suite, 'system-out')
    output.text = (output.text + '\n' if output.text else '') + text
    tree.write(junit, encoding='utf-8', xml_declaration=True)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a') as handle:
            handle.write('\n## Gate time budget warnings\n\n' +
                         f'{len(messages)} warnings; complete list in junit.xml and result.json.\n\n' +
                         '\n'.join('- ' + message[:1000] for message in messages[:25]) + '\n' +
                         (f'{len(messages) - 25} more warnings in junit.xml.\n' if len(messages) > 25 else ''))


def _write_budget_reruns(reruns):
    messages = [f"BUDGET_RERUN: {row['nodeid']}: first {row['first_seconds']:.3f}s, "
                f"rerun {row['rerun_seconds']}s, limit {row['limit']:.3f}s: {row['outcome']}"
                + (f"; infrastructure_reason: {row['infrastructure_reason']}"
                   if row.get('infrastructure_reason') else '')
                for row in reruns]
    print('\n'.join(messages))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a') as handle:
            handle.write('\n## Gate time budget reruns\n\n' +
                         f'{len(messages)} reruns; complete list in result.json.\n\n' +
                         '\n'.join('- ' + message[:1000] for message in messages[:25]) + '\n')


def worker_count(requested=None, *, affinity=None, available_bytes=None):
    cpus = os.sched_getaffinity(0) if affinity is None else affinity
    if available_bytes is None:
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available_bytes = int(memory['MemAvailable'].split()[0]) * 1024
    requested = len(cpus) if requested is None else requested
    if type(requested) is not int or requested < 1 or not cpus:
        raise ValueError('job needs a positive worker count and assigned CPUs')
    return min(requested, len(cpus), max(1, (available_bytes - 1024**3) // (2 * 1024**3)))


def run_plan(plan: dict, *, root: Path, output: Path, junit: Path,
             files: list[str] | None = None, workers: int | None = None,
             execute=subprocess.run, affinity=None, assignment: dict | None = None,
             _budget_target: str | None = None) -> int:
    print('GATE_PREFLIGHT_START', flush=True)
    record = {'schema': 'prismaquant.gate_result.v1',
              'population_version': plan.get('population', {}).get('version'),
              'mode': plan.get('mode'), 'base_sha': plan.get('base_sha'), 'head_sha': plan.get('head_sha'), 'files': [], 'verdict': None, 'pytest_exit_code': None,
              'budget_reruns': []}
    try:
        eligible = {row['path'] for row in plan['selected']}
        partition = assignment.get('node_partition') if assignment else None
        if assignment is not None:
            if files is not None:
                raise ValueError('specify assignment or files, not both')
            files = assignment['files']
        if partition:
            validate_node_partition(partition, files)
            if partition['index'] != assignment['index']:
                raise ValueError('assignment index differs from node partition')
            record['node_partition'] = partition
        files = sorted(eligible) if files is None else files
        if not files:
            if eligible:
                raise ValueError('empty assignment for a nonempty population')
            record.update(verdict='pass', pytest_exit_code=0, budget={'status': 'pass'}, noop=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record, indent=2) + '\n')
            return 0
        if len(files) != len(set(files)) or set(files) - eligible:
            raise ValueError('assignment contains duplicate files or files outside the plan')
        costs = {row['path']: row.get('estimated_seconds') for row in plan['selected']}
        known = sorted(value for value in costs.values() if value is not None)
        unknown = (known[math.ceil(len(known) * .9) - 1] or 1.0) if known else 1.0
        files = sorted(files, key=lambda path: (-(costs[path] if costs[path] is not None else unknown), path))
        record['files'] = files
        config = _budget_policy(root, plan.get('base_sha'))
        record['budget'] = {'status': 'pending', 'baseline_sha': plan.get('base_sha')}
        if _budget_target is None:
            workers = worker_count(workers, affinity=affinity)
        record['workers'] = workers
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1',
                           CUDA_VISIBLE_DEVICES='')
        # setup-python exports loader overrides; proof capture must see the actual
        # installed interpreter rather than an ambient dynamic-loader override.
        environment = {name: value for name, value in environment.items()
                       if not name.startswith('LD_') and name not in {'BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN'}}
        collection_path = junit.with_suffix('.collection.json')
        collection_path.unlink(missing_ok=True)
        environment['GATE_COLLECTION_REPORT'] = str(collection_path.resolve())
        environment['GATE_ASSIGNED_FILES'] = json.dumps(files)
        environment.pop('GATE_NODE_PARTITION', None)
        if partition:
            environment['GATE_NODE_PARTITION'] = json.dumps(partition)
        environment['GATE_DESELECT_NODEIDS'] = json.dumps(plan['deselect'])
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + environment.get('PYTHONPATH', '')
        junit.parent.mkdir(parents=True, exist_ok=True)
        profile = ['-X', 'importtime'] if os.environ.get('GATE_PROFILE_STARTUP') == '1' else []
        target = str(root) + '/' + _budget_target if _budget_target else str(root)
        command = [sys.executable, *profile, '-m', 'pytest', '-n', str(workers), '--dist=worksteal',
                   '-q', '-ra', '--timeout=600', '-m', 'not own_cgroup', '-p', 'gate_collect', '--rootdir=' + str(root), '--junitxml=' + str(junit), '-o', 'junit_family=xunit1',
                   target]
        print('GATE_PYTEST_START', flush=True)
        result = execute(command, cwd=root, env=environment, check=False)
        collection = json.loads(collection_path.read_text()) if collection_path.exists() else {}
        own_files = sorted(set(collection.get('own_cgroup_files', [])))
        if set(own_files) - set(files):
            raise RuntimeError('isolated cgroup assignment is outside the shard')
        if own_files:
            scoped_junit = junit.with_suffix('.cgroup.xml')
            scoped_collection = junit.with_suffix('.cgroup.collection.json')
            scoped_collection.unlink(missing_ok=True)
            scoped_env = {key: value for key, value in environment.items() if key in {
                'HOME', 'PATH', 'TMPDIR', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS'}}
            scoped_env.update(GATE_ASSIGNED_FILES=json.dumps(own_files),
                              GATE_DESELECT_NODEIDS=environment['GATE_DESELECT_NODEIDS'],
                              GATE_COLLECTION_REPORT=str(scoped_collection.resolve()))
            if partition:
                scoped_env['GATE_NODE_PARTITION'] = json.dumps(partition)
            scoped_command = [sys.executable, '-m', 'pytest', '-q', '-ra', '--timeout=600',
                '-m', 'own_cgroup', '-p', 'no:xdist', '-p', 'gate_collect', '--rootdir=' + str(root),
                '--junitxml=' + str(scoped_junit), '-o', 'junit_family=xunit1',
                target]
            scoped = execute(['sudo', '-n', 'systemd-run', '--scope', '--quiet', '--',
                'setpriv', '--reuid=' + str(os.getuid()), '--regid=' + str(os.getgid()),
                '--init-groups', 'env', *(key + '=' + value for key, value in scoped_env.items()),
                *scoped_command], cwd=root, env=environment, check=False)
            record['cgroup_exit_code'] = scoped.returncode
            if scoped.returncode not in (0, 5):
                # Preserve any definite test failure from either phase.
                result = type('Exit', (), {'returncode': 1 if 1 in (result.returncode, scoped.returncode) else 75})()
            if scoped.returncode == 5:
                coverage = json.loads(scoped_collection.read_text())
                if coverage['collection_failed'] or _collection_files(coverage, _budget_target) != set(own_files):
                    raise RuntimeError('isolated cgroup phase did not collect its complete assignment')
            if scoped_junit.exists():
                tree = ET.parse(junit)
                extra = ET.parse(scoped_junit).getroot()
                tree.getroot().extend(list(extra) if extra.tag == 'testsuites' else [extra])
                tree.write(junit, encoding='utf-8', xml_declaration=True)
            elif result.returncode != 1:
                raise RuntimeError('isolated cgroup phase has no JUnit verdict')
        print('GATE_PYTEST_END', flush=True)
        collection = json.loads(collection_path.read_text()) if collection_path.exists() else None
        collected_files = _collection_files(collection, _budget_target) if collection else None
        record['collection'] = collection
        pytest_code = result.returncode
        # A module-level skip or complete deselection legitimately has no tests.
        if (pytest_code == 5 and collection and not collection['collection_failed'] and
                set(collected_files) == set(files)):
            pytest_code = 0
        record['pytest_exit_code'] = pytest_code
        record['verdict'] = 'fail' if pytest_code == 1 else None
        try:
            # Preserve a recorded failure even if pytest subsequently interrupts
            # or leaves a truncated report during an internal error.
            for _, element in ET.iterparse(junit):
                if element.tag in {'failure', 'error'}:
                    record['verdict'] = 'fail'
        except (OSError, ET.ParseError) as error:
            if record['verdict'] != 'fail':
                raise RuntimeError('JUnit verdict unavailable: ' + str(error)) from error
        if record['verdict'] != 'fail':
            if pytest_code != 0:
                raise RuntimeError(f'pytest exited {result.returncode}, without a complete test verdict')
            record['verdict'] = 'pass'
        try:
            if _budget_target and '::' in _budget_target:
                if collection and collection['collection_failed']:
                    raise ValueError('budget rerun collection is incomplete')
                executed = [gate_junit_identity(case, files)[1]
                            for case in ET.parse(junit).iter('testcase')]
                if executed != [_budget_target]:
                    raise ValueError('budget rerun target did not execute exactly once')
            costs = _budget_timings(junit, files, config, collected_files)
            if partition:
                if not collection or collection['collection_failed']:
                    raise ValueError('split-file collection is incomplete')
                collected = collection.get('split_collected', {})
                if set(collected) != set(partition['owners']):
                    raise ValueError('split-file collection does not match the partition')
                identities = [(gate_junit_identity(case, files), float(case.get('time', 0)))
                              for case in ET.parse(junit).iter('testcase')]
                all_executed = sorted(node for (path, node), seconds in identities if node is not None)
                selected = collection['selected'][:]
                if own_files:
                    scoped_coverage = json.loads(scoped_collection.read_text())
                    if scoped_coverage['collection_failed']:
                        raise ValueError('serial-phase collection is incomplete')
                    selected.extend(scoped_coverage['selected'])
                record['execution_coverage'] = {'selected': sorted(selected), 'executed': all_executed}
                if sorted(selected) != all_executed:
                    raise ValueError('selected nodes did not execute exactly once')
                file_limits = {path: limit for path, limit in config.get('file_seconds', {}).items()
                               if path in partition['owners']}
                record['budget'].update(file_limits=file_limits, file_totals={
                    path: sum(seconds for (filename, node), seconds in identities if filename == path)
                    for path in file_limits})
                executed = [node for (path, node), seconds in identities
                            if path in partition['owners'] and node is not None]
                record['node_coverage'] = {'collected': collected, 'executed': sorted(executed)}
                if sorted(executed) != assigned_nodes(collected, partition, plan['deselect']):
                    raise ValueError('split-file nodes did not execute exactly once')
            overruns, warnings = [], []
            for row in costs:
                target = (overruns if row['seconds'] >
                          row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE else warnings)
                target.append(row)
            enforced = config.get('enforced', True)
            if (overruns and enforced and record['verdict'] == 'pass' and
                    pytest_code == 0 and _budget_target is None):
                rerun_codes = []
                remaining = []
                for index, row in enumerate(overruns):
                    nodeid = row['nodeid']
                    rerun_junit = junit.with_suffix(f'.budget-rerun-{index}.xml')
                    rerun_junit.unlink(missing_ok=True)
                    rerun_output = rerun_junit.with_suffix('.result.json')
                    # Reuse the normal verdict and cgroup paths, but without a
                    # node partition: a file budget measures the complete file.
                    rerun_code = run_plan(plan, root=root, output=rerun_output, junit=rerun_junit,
                        files=[nodeid.split('::', 1)[0]], workers=workers, execute=execute,
                        affinity=affinity, _budget_target=nodeid)
                    nested = json.loads(rerun_output.read_text())
                    reason = nested.get('infrastructure_reason')
                    elapsed = None
                    try:
                        timings = [(gate_junit_identity(case), float(case.attrib['time']))
                                   for case in ET.parse(rerun_junit).iter('testcase')]
                        measured = [seconds for (path, node), seconds in timings
                                    if node == nodeid or path == nodeid]
                        if not measured or any(not math.isfinite(s) or s < 0 for s in measured):
                            raise ValueError('budget rerun has no valid target timing')
                        elapsed = sum(measured)
                        if nodeid in record['budget'].get('file_totals', {}):
                            owned = set(record['node_coverage']['executed'])
                            record['budget']['file_totals'][nodeid] = sum(
                                seconds for (path, node), seconds in timings if node in owned)
                    except (OSError, ValueError, KeyError, ET.ParseError) as error:
                        if rerun_code != 1:
                            rerun_code = NO_VERDICT
                            reason = reason or str(error)
                    record['budget_reruns'].append({
                        'nodeid': nodeid, 'first_seconds': row['seconds'], 'rerun_seconds': elapsed,
                        'limit': row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE,
                        'outcome': {0: 'pass', 1: 'fail', NO_VERDICT: 'no_verdict'}[rerun_code]})
                    if rerun_code == NO_VERDICT:
                        record['budget_reruns'][-1]['infrastructure_reason'] = reason or 'budget rerun has no verdict'
                    rerun_codes.append(rerun_code)
                    if rerun_code != 0:
                        remaining.append(row)
                overruns = remaining
                if 1 in rerun_codes:
                    record['verdict'] = 'fail'
                elif NO_VERDICT in rerun_codes:
                    record['verdict'] = None
                _write_budget_reruns(record['budget_reruns'])
            record['budget'].update(enforced=enforced, status='fail' if overruns and enforced else 'pass',
                                    overruns=overruns, warnings=warnings)
            if overruns:
                if enforced and record['verdict'] is not None:
                    record['verdict'] = 'fail'
                print('Gate time budget exceeded:')
                for row in overruns:
                    limit = row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE
                    print(f"{row['nodeid']}: {row['seconds']:.3f}s > {limit:.3f}s "
                          f"(policy {row['budget_seconds']:.3f}s x {BUDGET_ENFORCEMENT_TOLERANCE:g})")
            if warnings:
                _write_budget_warnings(junit, warnings)
        except (OSError, ValueError, ET.ParseError) as error:
            record['budget'].update(status='unavailable', reason=str(error))
            # A definite test failure remains final even if timing evidence is lost.
            if record['verdict'] != 'fail':
                record['verdict'] = None
                raise RuntimeError(str(error)) from error
        code = {'pass': 0, 'fail': 1, None: NO_VERDICT}[record['verdict']]
    except BudgetFailure as error:
        record['verdict'] = 'fail'
        record['budget'] = {'status': 'fail', 'reason': str(error), 'baseline_sha': plan.get('base_sha')}
        print('GATE_BUDGET_FAILURE: ' + str(error))
        code = 1
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        record['infrastructure_reason'] = str(error)
        code = NO_VERDICT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--junitxml', type=Path, required=True)
    parser.add_argument('--workers', type=int, help='resource-class CPU allocation; clamped to inherited affinity')
    assigned = parser.add_mutually_exclusive_group()
    assigned.add_argument('--assignment', type=Path, help='complete file and node assignment JSON')
    assigned.add_argument('--files-from', type=Path, help='one assigned repository-relative file per line')
    assigned.add_argument('--files', nargs='+', help='explicit assigned test files')
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        files = ([line.strip() for line in args.files_from.read_text().splitlines() if line.strip()]
                 if args.files_from else args.files)
        return run_plan(plan, root=args.root, output=args.result, junit=args.junitxml,
                        files=files, workers=args.workers,
                        assignment=json.loads(args.assignment.read_text()) if args.assignment else None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'gate: no verdict: {error}', file=sys.stderr)
        return NO_VERDICT


if __name__ == '__main__':
    raise SystemExit(main())
