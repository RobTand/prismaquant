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

NO_VERDICT = 75
BUDGET_ENFORCEMENT_TOLERANCE = 1.5


class BudgetFailure(ValueError):
    """A definite policy failure, never an infrastructure retry."""


def _budget_limits(config):
    if not isinstance(config, dict) or not isinstance(config.get('slow'), dict):
        raise BudgetFailure('Test budget requires seconds and a slow map')
    limits = [config.get('seconds'), *config['slow'].values()]
    if (any(isinstance(limit, bool) or not isinstance(limit, (int, float)) or
            not math.isfinite(limit) or limit <= 0 for limit in limits) or
            any(not isinstance(name, str) or not name for name in config['slow'])):
        raise BudgetFailure('Test budget limits must be finite positive seconds')
    return config


def _budget_policy(root, base_sha):
    path = 'tools/ci/gate_budget.json'
    current = _budget_limits(json.loads((root / path).read_text()))
    baseline = current
    if base_sha:
        # Check the commit separately: an unavailable target is not a bootstrap.
        target = subprocess.run(['git', 'cat-file', '-e', f'{base_sha}^{{commit}}'],
                                cwd=root, capture_output=True, check=False)
        if target.returncode:
            raise RuntimeError('budget target commit is unavailable: ' + base_sha)
        tree = subprocess.run(['git', 'ls-tree', base_sha, '--', path],
                              cwd=root, text=True, capture_output=True, check=True)
        if tree.stdout.strip():
            blob = subprocess.run(['git', 'show', f'{base_sha}:{path}'], cwd=root,
                                  text=True, capture_output=True, check=True)
            baseline = _budget_limits(json.loads(blob.stdout))
    if (baseline.get('enforced', True) and not current.get('enforced', True) or
            current['seconds'] > baseline['seconds'] or
            not current['slow'].keys() <= baseline['slow'].keys() or
            any(limit > baseline['slow'][name] for name, limit in current['slow'].items())):
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


def _budget_timings(junit, files, config, collected_files=None):
    observed, seconds = set(), {}
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
        seconds[nodeid] = seconds.get(nodeid, 0) + elapsed
    if (observed if collected_files is None else set(collected_files)) != set(files):
        raise ValueError('JUnit timings do not cover every assigned file')
    return [{'nodeid': nodeid, 'seconds': elapsed,
             'budget_seconds': config['slow'].get(nodeid, config['seconds'])}
            for nodeid, elapsed in sorted(seconds.items())
            if elapsed > config['slow'].get(nodeid, config['seconds'])]


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
            handle.write('\n## Per-test budget warnings\n\n' +
                         f'{len(messages)} warnings; complete list in junit.xml and result.json.\n\n' +
                         '\n'.join('- ' + message[:1000] for message in messages[:25]) + '\n' +
                         (f'{len(messages) - 25} more warnings in junit.xml.\n' if len(messages) > 25 else ''))


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
             execute=subprocess.run, affinity=None) -> int:
    print('GATE_PREFLIGHT_START', flush=True)
    record = {'schema': 'prismaquant.gate_result.v1',
              'population_version': plan.get('population', {}).get('version'),
              'mode': plan.get('mode'), 'base_sha': plan.get('base_sha'), 'head_sha': plan.get('head_sha'), 'files': [], 'verdict': None, 'pytest_exit_code': None}
    try:
        eligible = {row['path'] for row in plan['selected']}
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
        workers = worker_count(workers, affinity=affinity)
        record['workers'] = workers
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1',
                           CUDA_VISIBLE_DEVICES='')
        # setup-python exports loader overrides; proof capture must see the actual
        # installed interpreter rather than an ambient dynamic-loader override.
        environment = {name: value for name, value in environment.items() if not name.startswith('LD_')}
        collection_path = junit.with_suffix('.collection.json')
        collection_path.unlink(missing_ok=True)
        environment['GATE_COLLECTION_REPORT'] = str(collection_path.resolve())
        environment['GATE_ASSIGNED_FILES'] = json.dumps(files)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + environment.get('PYTHONPATH', '')
        junit.parent.mkdir(parents=True, exist_ok=True)
        profile = ['-X', 'importtime'] if os.environ.get('GATE_PROFILE_STARTUP') == '1' else []
        command = [sys.executable, *profile, '-m', 'pytest', '-n', str(workers), '--dist=worksteal',
                   '-q', '-ra', '--timeout=600', '-m', 'not own_cgroup', '-p', 'gate_collect', '--rootdir=' + str(root), '--junitxml=' + str(junit), '-o', 'junit_family=xunit1',
                   *('--deselect=' + target for target in plan['deselect']
                     if target.split('::', 1)[0] in files), str(root)]
        print('GATE_PYTEST_START', flush=True)
        result = execute(command, cwd=root, env=environment, check=False)
        own_files = [path for path in files if (root / path).exists() and
                     'pytest.mark.own_cgroup' in (root / path).read_text()]
        if own_files:
            scoped_junit = junit.with_suffix('.cgroup.xml')
            scoped_collection = junit.with_suffix('.cgroup.collection.json')
            scoped_collection.unlink(missing_ok=True)
            scoped_env = {key: value for key, value in environment.items() if key in {
                'HOME', 'PATH', 'TMPDIR', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS'}}
            scoped_env.update(GATE_ASSIGNED_FILES=json.dumps(own_files),
                              GATE_COLLECTION_REPORT=str(scoped_collection.resolve()))
            scoped_command = [sys.executable, '-m', 'pytest', '-q', '-ra', '--timeout=600',
                '-m', 'own_cgroup', '-p', 'gate_collect', '--rootdir=' + str(root),
                '--junitxml=' + str(scoped_junit), '-o', 'junit_family=xunit1',
                *('--deselect=' + target for target in plan['deselect']
                  if target.split('::', 1)[0] in own_files), str(root)]
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
                if coverage['collection_failed'] or set(coverage['seen']) != set(own_files):
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
        collected_files = collection['seen'] if collection else None
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
            costs = _budget_timings(junit, files, config, collected_files)
            overruns, warnings = [], []
            for row in costs:
                target = (overruns if row['seconds'] >
                          row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE else warnings)
                target.append(row)
            enforced = config.get('enforced', True)
            record['budget'].update(enforced=enforced, status='fail' if overruns and enforced else 'pass',
                                    overruns=overruns, warnings=warnings)
            if overruns:
                if enforced:
                    record['verdict'] = 'fail'
                print('Per-test time budget exceeded:')
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
        code = 0 if record['verdict'] == 'pass' else 1
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
    assigned.add_argument('--files-from', type=Path, help='one assigned repository-relative file per line')
    assigned.add_argument('--files', nargs='+', help='explicit assigned test files')
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        files = ([line.strip() for line in args.files_from.read_text().splitlines() if line.strip()]
                 if args.files_from else args.files)
        return run_plan(plan, root=args.root, output=args.result, junit=args.junitxml,
                        files=files, workers=args.workers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'gate: no verdict: {error}', file=sys.stderr)
        return NO_VERDICT


if __name__ == '__main__':
    raise SystemExit(main())
