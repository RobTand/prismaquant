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


def test_frozen_checkout_applies_tracked_binary_patch(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'verification'
    source.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])

    git('init', '-q', '-b', 'base')
    tracked = source / 'payload.bin'
    tracked.write_bytes(b'before\x00payload\xff')
    git('add', tracked.name)
    git('-c', 'user.name=Capture fixture', '-c', 'user.email=capture@example.invalid',
        'commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD').decode().strip()
    edited = b'after\x00payload\xfe'
    tracked.write_bytes(edited)
    patch = git('diff', '--binary', 'HEAD')
    status = git('status', '--porcelain')
    try:
        result = freeze_refactor_checkout(source, target)
        assert (target / tracked.name).read_bytes() == edited
        assert result['base'] == base
        assert result['patch_bytes'] == len(patch)
        assert result['untracked'] == []
        assert tracked.read_bytes() == edited
        assert git('diff', '--binary', 'HEAD') == patch
        assert git('status', '--porcelain') == status
        assert git('rev-parse', 'HEAD').decode().strip() == base
    finally:
        if target.exists():
            git('worktree', 'remove', '--force', str(target))


def test_frozen_checkout_preserves_untracked_source_symlink_and_file(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'verification'
    source.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])

    git('init', '-q', '-b', 'base')
    (source / 'tracked.txt').write_bytes(b'committed\n')
    git('add', 'tracked.txt')
    git('-c', 'user.name=Capture fixture', '-c', 'user.email=capture@example.invalid',
        'commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD').decode().strip()
    package = source / 'prismaquant'
    package.mkdir()
    contents = b'untracked source\n'
    regular, link = package / 'new.py', package / 'alias.py'
    regular.write_bytes(contents)
    regular.chmod(0o640)
    mode = regular.stat().st_mode
    link.symlink_to(regular.name)
    status = git('status', '--porcelain')
    try:
        result = freeze_refactor_checkout(source, target)
        frozen = target / package.name
        frozen_link = frozen / link.name
        assert frozen_link.is_symlink()
        assert frozen_link.readlink().as_posix() == regular.name
        assert (frozen / regular.name).read_bytes() == contents
        assert (frozen / regular.name).lstat().st_mode == mode
        assert result['base'] == base
        assert result['patch_bytes'] == 0
        assert result['untracked'] == ['prismaquant/alias.py', 'prismaquant/new.py']
        assert regular.read_bytes() == contents
        assert link.is_symlink()
        assert link.readlink().as_posix() == regular.name
        assert git('status', '--porcelain') == status
        assert git('rev-parse', 'HEAD').decode().strip() == base
    finally:
        if target.exists():
            git('worktree', 'remove', '--force', str(target))
