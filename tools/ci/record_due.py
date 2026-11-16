#!/usr/bin/env python3
"""Whether the scheduled post-merge record is due: main moved since the last record job that ran.

Only a record job that finished (success or failure) counts, never a run whose `due` job failed or
whose record was skipped. Any error reading the history runs the record.
"""
import json
import os
import subprocess

RECORD = 'Post-merge Bazel record (amd64 and arm64)'


def due(head, gh, depth=10):
    runs = [run for run in gh('run', 'list', '-w', 'ci.yml', '-b', 'main', '-L', '200',
                              '--json', 'databaseId,event,status,headSha')
            if run['event'] in ('schedule', 'workflow_dispatch') and run['status'] == 'completed']
    for run in runs[:depth]:
        jobs = gh('api', f"repos/{{owner}}/{{repo}}/actions/runs/{run['databaseId']}/jobs")['jobs']
        if any(job['name'] == RECORD and job['conclusion'] in ('success', 'failure') for job in jobs):
            return run['headSha'] != head
    return True


def main():
    gh = lambda *args: json.loads(subprocess.run(['gh', *args], capture_output=True, text=True, check=True).stdout)
    try:
        run = os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch' or due(os.environ['GITHUB_SHA'], gh)
    except (subprocess.CalledProcessError, ValueError, KeyError):
        run = True
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'run={str(run).lower()}\n')


if __name__ == '__main__':
    main()
