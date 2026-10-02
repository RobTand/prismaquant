"""Parent-launched same-UID py-spy; no sudo, attach escalation or policy edits.

Run this entrypoint in the workload's container/namespace. py-spy parents an
owned status worker and its original Python child, so Yama's parent relationship
works while PB's NoNewPrivileges stays enabled. No duration cutoff terminates a
row: the existing admitted row deadline bounds the whole process tree.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.pq_profile_artifact import finish_profile_status, workload_status_path


def same_uid_profile_command(executable, command, destination, status):
    if (not command or not isinstance(command[0], str) or not command[0]
            or any(not isinstance(part, str) for part in command)):
        raise ValueError('profile workload requires a nonempty executable and string operands')
    return [str(executable), 'record', '--idle', '--threads', '--subprocesses',
            '--rate', '50', '--format', 'speedscope', '-o', str(destination), '--',
            sys.executable, '-u', '-m', 'tools.pq_profile_child', '--execute',
            '--status', str(status), '--', *command]


def run_profile_workload(command, status):
    status = Path(status)
    if status.exists():
        raise FileExistsError('refusing stale workload status')
    workload = subprocess.Popen(command)
    rc = workload.wait()
    with status.open('x') as handle:
        json.dump({'schema': 'prismaquant.profile_workload_status.v1',
                   'returncode': rc, 'workload_pid': workload.pid}, handle)
    return rc


def run_child_profile(command, *, executable, destination):
    destination = Path(destination)
    status = workload_status_path(destination)
    if destination.exists() or status.exists():
        raise FileExistsError('refusing stale child profile or workload status')
    rc = subprocess.call(same_uid_profile_command(executable, command, destination, status))
    if rc and not status.is_file():
        raise RuntimeError(f'no-sudo child profiler failed with status {rc}')
    record = finish_profile_status(destination, rc)
    # A profiler shutdown error must not mask the original failed workload.
    # The separate profiler outcome still prevents successful publication.
    if record['returncode']:
        return record['returncode']
    if rc:
        raise RuntimeError(f'no-sudo child profiler failed with status {rc}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiler-executable', default='py-spy')
    parser.add_argument('--out')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--status')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('an unchanged workload argv is required')
    if args.execute:
        if not args.status:
            parser.error('--execute requires --status')
        return run_profile_workload(command, args.status)
    if not args.out:
        parser.error('--out is required')
    return run_child_profile(command, executable=args.profiler_executable, destination=args.out)


if __name__ == '__main__':
    raise SystemExit(main())
