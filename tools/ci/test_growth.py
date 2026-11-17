#!/usr/bin/env python3
"""Refuse a PR that adds more than LIMIT net lines to tests/*.py unless one of its commits says why.

Usage: test_growth.py BASE [HEAD]. A queue draft holds one first-parent "Merge of #N" commit per PR; each is
counted alone. Any other range is one PR. The reason is a commit trailer: "Test-lines: <why the tests must grow>".
LIMIT: the PQ+T lead's number (2026-11-17). Over the 250 merges of 2026-10-18..11-17, 17 were above it and held
77 % of the net test growth.
"""
import re
import subprocess
import sys

LIMIT = 250


def git(*arguments):
    return subprocess.run(['git', *arguments], capture_output=True, text=True, check=True).stdout


def growth(base, head):
    rows = git('diff', '--numstat', base, head, '--', 'tests/*.py').splitlines()
    return sum(int(added) - int(removed) for added, removed, _ in (row.split('\t') for row in rows) if added != '-')


def refusals(base, head='HEAD'):
    drafts = [(line[:40], line[41:]) for line in git('log', '--first-parent', '--format=%H %s', f'{base}..{head}')
              .splitlines() if re.match(r'Merge of #\d+$', line[41:])]
    # (what the PR brought, its own commits, its name)
    prs = [((f'{merge}^1', merge), f'{merge}^1..{merge}^2', name) for merge, name in drafts] or [
        ((base, head), f'{base}..{head}', 'this PR')]
    return [f'{name} adds {lines} net lines to tests/*.py (limit {LIMIT}); add a "Test-lines: <reason>" trailer'
            for change, commits, name in prs if (lines := growth(*change)) > LIMIT
            and not re.search(r'(?m)^Test-lines: \S', git('log', '--format=%B', commits))]


if __name__ == '__main__':
    found = refusals(*sys.argv[1:])
    print('\n'.join(found) or f'test growth: every PR within {LIMIT} net lines or explained')
    sys.exit(bool(found))
