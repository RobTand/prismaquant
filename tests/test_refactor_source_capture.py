import subprocess

import pytest

from tools.freeze_refactor_checkout import freeze_refactor_checkout


def test_conflicted_index_cannot_publish_a_verification_checkout(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()

    def git(*args, check=True):
        return subprocess.run(['git', '-C', str(source), *args], check=check,
                              text=True, capture_output=True)

    git('init', '-q', '-b', 'base')
    git('config', 'user.name', 'Capture fixture')
    git('config', 'user.email', 'capture@example.invalid')
    file = source / 'contract.txt'
    file.write_text('original\n')
    git('add', '.')
    git('commit', '-qm', 'base')
    git('checkout', '-qb', 'left')
    file.write_text('left\n')
    git('commit', '-qam', 'left')
    git('checkout', '-qb', 'right', 'base')
    file.write_text('right\n')
    git('commit', '-qam', 'right')
    assert git('merge', 'left', check=False).returncode != 0
    before = git('worktree', 'list', '--porcelain').stdout
    target = tmp_path / 'verification'
    with pytest.raises(ValueError, match='unresolved merge conflicts'):
        freeze_refactor_checkout(source, target)
    assert not target.exists()
    assert git('worktree', 'list', '--porcelain').stdout == before
