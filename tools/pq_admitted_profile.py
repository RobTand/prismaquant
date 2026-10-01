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
        observer_rc = observer.wait(timeout=observer_exit_timeout)
        if observer_rc:
            raise RuntimeError(f'observer failed with status {observer_rc}; profile not qualified')
        return row_rc
    finally:
        # Own observer only. The workload is never signalled or retried here.
        if observer.poll() is None:
            observer.terminate()
            observer.wait(timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', required=True, help='Frozen JSON argv/env row descriptor')
    parser.add_argument('--observations', required=True)
    parser.add_argument('--profile-local', required=True)
    parser.add_argument('--row-s', type=int, default=900)
    parser.add_argument('--profile-s', type=int, default=420)
    args = parser.parse_args(argv)
    row = json.loads(Path(args.launch).read_text())
    command = row['argv']
    if (not isinstance(command, list) or not command or
            any(not isinstance(part, str) for part in command)):
        raise RuntimeError('launch must carry a nonempty string argv')
    target = command[command.index('--out') + 1]
    observer = [sys.executable, '-u', '-m', 'tools.pq_row_profile_observer',
                '--out', args.observations, '--target-out', target,
                '--profile-local', args.profile_local, '--wait-s', '60',
                '--row-s', str(args.row_s), '--profile-s', str(args.profile_s)]
    return run_observed(command, observer, Path(args.observations), target_out=target,
                        observer_exit_timeout=args.profile_s + 120)


if __name__ == '__main__':
    raise SystemExit(main())
