"""Admission-local ownership of an observer and one unchanged workload.

The fleet invokes this entrypoint only after admission. No PB submission,
placement, Docker invocation, affinity rewrite or workload signal occurs here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROFILE_LOCAL_ROOT = Path('/home/rob/tmp/claude-campaign-20260926/tmp/row-startup/profiles')


def run_observed(command, observer_command, observations: Path, *, target_out,
                 ready_timeout=60, observer_exit_timeout=540):
    observations = Path(observations)
    if observations.exists():
        raise FileExistsError('refusing stale observer output directory')
    observer = subprocess.Popen(observer_command)
    workload = None
    try:
        deadline = time.monotonic() + ready_timeout
        ready = observations / 'ready.json'
        while True:
            if observer.poll() is not None:
                raise RuntimeError('observer failed before workload readiness')
            if ready.is_file():
                record = json.loads(ready.read_text())
                if record.get('target_out') != target_out:
                    raise RuntimeError('observer readiness names a different target')
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('observer readiness timeout; workload not started')
            time.sleep(0.1)
        workload = subprocess.Popen(command)
        row_rc = workload.wait()
        (observations / 'workload_done').write_text(str(row_rc))
        observer_error = None
        try:
            observer_rc = observer.wait(timeout=observer_exit_timeout)
        except (subprocess.TimeoutExpired, OSError) as error:
            observer_rc = None
            observer_error = str(error)
        qualified = row_rc == 0 and observer_rc == 0 and observer_error is None
        with (observations / 'observer_result.json').open('x') as handle:
            json.dump({'schema': 'prismaquant.profile_observer_result.v1',
                       'workload_returncode': row_rc, 'returncode': observer_rc,
                       'error': observer_error, 'qualified': qualified}, handle)
        # Qualification failure cannot turn the original failed row into some
        # unrelated exception exit. Cleanup still joins/stops our observer only.
        if row_rc:
            return row_rc
        if not qualified:
            raise RuntimeError(f'observer failed with status {observer_rc}; '
                               f'profile not qualified: {observer_error}')
        return row_rc
    finally:
        # Own observer only. The workload is never signalled or retried here.
        if observer.poll() is None:
            observer.terminate()
            observer.wait(timeout=30)


def profile_row_parts(command):
    """Locate the existing row boundary without changing command operands."""
    boundary = 0
    if 'tools.tessera_campaign_container' in command:
        module = command.index('tools.tessera_campaign_container')
        boundary = command.index('--', module) + 1
    inner = command[boundary:]
    if not inner or not Path(inner[0]).name.startswith('python'):
        raise ValueError('profiling requires the original Python row argv')
    return boundary, inner


def profiled_row_command(command, *, destination, profiler_executable):
    """Keep container ownership and original argv; profile its same-UID child."""
    boundary, inner = profile_row_parts(command)
    return [*command[:boundary], inner[0], '-u', '-m', 'tools.pq_profile_child',
            '--profiler-executable', profiler_executable, '--out', str(destination),
            '--', *inner]


def profiled_workload_parts(command, *, destination, profiler_executable):
    """Accept only the exact existing canonical child wrapper; keep full argv."""
    boundary, inner = profile_row_parts(command)
    if len(inner) < 10 or inner[8] != '--':
        raise RuntimeError('namespace profile requires the canonical child wrapper')
    original = [*command[:boundary], *inner[9:]]
    if command != profiled_row_command(original, destination=destination,
                                       profiler_executable=profiler_executable):
        raise RuntimeError('namespace profile wrapper differs from its declaration')
    return boundary + 9


def profile_local_destination(request_key):
    """Same existing host-local policy; namespace owns one deterministic child."""
    return PROFILE_LOCAL_ROOT / request_key


def bound_profile_command(row, *, observations, profile_local, profiler_executable, row_s):
    """Admit the already sealed instrumented row, never rewrite its identity."""
    command = row['argv']
    if command[:4] != ['python3', '-m', 'tools.tessera_campaign_container', '--spec']:
        return None
    spec = json.loads(command[4])
    if 'namespace_binding' not in spec and 'namespace_profile' not in spec:
        return None
    from tools.tessera_campaign_namespace import (
        namespace_profile_record, namespace_request_parts, require_namespace_publication)
    profile = namespace_profile_record(spec)
    if profile is None or 'namespace_binding' not in spec:
        raise RuntimeError('namespace profiling must be prepared before binding and publication')
    _, _, inner_start = namespace_request_parts(row)
    require_namespace_publication(row)
    expected = (profile['observations'], profile['profile_local'], profile['profiler']['path'], profile['row_s'])
    if (observations, profile_local, profiler_executable, row_s) != expected:
        raise RuntimeError('namespace profile launch arguments differ from sealed instrumentation')
    local = Path(profile_local)
    if local.exists() or local.is_symlink():
        raise FileExistsError('refusing stale namespace profile-local output directory')
    target = command[command.index('--out', inner_start) + 1]
    return command, target, Path(profile['observations']) / 'child-profile.speedscope'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', required=True, help='Frozen JSON argv/env row descriptor')
    parser.add_argument('--observations', required=True)
    parser.add_argument('--profile-local', required=True)
    parser.add_argument('--profiler-executable', default='py-spy',
                        help='Pinned executable visible at the same path inside the container')
    parser.add_argument('--row-s', type=int, default=900)
    args = parser.parse_args(argv)
    row = json.loads(Path(args.launch).read_text())
    command = row['argv']
    if (not isinstance(command, list) or not command or
            any(not isinstance(part, str) for part in command)):
        raise RuntimeError('launch must carry a nonempty string argv')
    bound = bound_profile_command(row, observations=args.observations,
                                  profile_local=args.profile_local,
                                  profiler_executable=args.profiler_executable, row_s=args.row_s)
    if bound is None:
        target = command[command.index('--out') + 1]
        child_profile = Path(args.observations).resolve() / 'child-profile.speedscope'
        observed_command = profiled_row_command(command, destination=child_profile,
                                                profiler_executable=args.profiler_executable)
    else:
        observed_command, target, child_profile = bound
    observer = [sys.executable, '-u', '-m', 'tools.pq_row_profile_observer',
                '--out', args.observations, '--target-out', target,
                '--profile-local', args.profile_local, '--wait-s', '60',
                '--profiler-executable', args.profiler_executable,
                '--child-profile', str(child_profile),
                '--row-s', str(args.row_s)]
    return run_observed(observed_command, observer, Path(args.observations), target_out=target,
                        observer_exit_timeout=120)


if __name__ == '__main__':
    raise SystemExit(main())
