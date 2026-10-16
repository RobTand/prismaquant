"""One owner for the atomic-write recipe (#1573).

`prismaquant/cost_stage_checkpoint.py::atomic_write_bytes` is the owner of
"stage bytes beside the target, fsync, replace, fsync the directory".
`prismaquant/aura_cost.py` carried its own copy; the AURA checkpoint writer
now lives in `prismaquant/aura_checkpoint_journal.py` and shares the owner.
The reseal package owner must rewrite plain checkpoint data without numerical
or CLI dependencies. Its actual import and payload behavior is exercised in a
fresh child interpreter. The durable publication ordering is:

write -> flush -> fsync(file) -> os.replace -> fsync(containing directory)

The published bytes are the caller's payload (unchanged by this primitive),
so the byte-relevant encoders stay pinned where they already are
(tests/test_digest_calibration_cache_1512.py and friends); what this file
pins is the delegation and the fsync/replace sequence itself.
"""
from __future__ import annotations

import os

import pytest

from prismaquant import cost_stage_checkpoint as csc


def test_aura_checkpoint_journal_uses_the_owner_recipe() -> None:
    from prismaquant import aura_checkpoint_journal

    assert aura_checkpoint_journal.atomic_write_bytes is csc.atomic_write_bytes


def test_reseal_package_rewrites_plain_data_without_numerical_or_tool_imports(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.path.insert(0, ROOT)
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.startswith('tessera.serving') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools'}:
            raise AssertionError('reseal dependency: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
import pickle
from pathlib import Path
from prismaquant import tessera_campaign_reseal as owner
value = {'units': {'u': {'menu': ['TESSERA_E4M3_K1_R832']}}, 'unicode': 'héllo'}
raw = pickle.dumps(value, protocol=4)
assert owner.loads(raw) == value
assert owner.dumps_like(value, raw) == raw
assert owner.canonical_bytes(value) == b'{"unicode":"h\\xc3\\xa9llo","units":{"u":{"menu":["TESSERA_E4M3_K1_R832"]}}}'
for unsafe in (pickle.dumps(Exception('x')), b'not-a-pickle'):
    try:
        owner.loads(unsafe)
    except (owner.Refused, pickle.UnpicklingError):
        pass
    else:
        raise AssertionError('non-plain checkpoint admitted')
try:
    owner.discover_rows()
except owner.Refused as error:
    assert 'no rows' in str(error)
else:
    raise AssertionError('empty row scope admitted')
path = Path('metadata.json')
owner.write_json(path, value)
import json
assert json.loads(path.read_bytes()) == value
print('RESEAL-OWNER-OK')
""".replace('ROOT', repr(str(root)), 1)
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'RESEAL-OWNER-OK'


def test_owner_orders_fsync_replace_dirfsync(tmp_path, monkeypatch) -> None:
    target = tmp_path / "nested" / "file.json"
    calls: list[tuple[str, object]] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def record_replace(src, dst):
        calls.append(("replace", (src, dst)))
        return real_replace(src, dst)

    def record_fsync(fd):
        calls.append(("fsync", fd))
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", record_replace)
    monkeypatch.setattr(os, "fsync", record_fsync)

    csc.atomic_write_bytes(target, b'{"a": 1}')

    kinds = [kind for kind, _ in calls]
    # exactly one rename, file-fsync strictly before it, dir-fsync after
    assert kinds.count("replace") == 1
    assert kinds[: kinds.index("replace")].count("fsync") == 1
    assert kinds[kinds.index("replace") + 1 :].count("fsync") == 1
    assert target.read_bytes() == b'{"a": 1}'
    # no staging file is left behind under any suffix spelling
    leftovers = [p.name for p in target.parent.iterdir() if p.name != target.name]
    assert leftovers == [], leftovers


def test_owner_preserves_existing_file_mode(tmp_path) -> None:
    # The copies opened the staging file with plain "wb" and never chmod'ed;
    # the owner must not either -- a replaced file keeps the staging mode.
    target = tmp_path / "mode.bin"
    csc.atomic_write_bytes(target, b"first")
    first_mode = target.stat().st_mode & 0o777
    os.chmod(target, 0o600)
    csc.atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    # replaced by the staging inode: mode is the fresh-open mode, not the
    # pre-existing 0600 -- same behaviour the local copies had.
    assert (target.stat().st_mode & 0o777) == first_mode


@pytest.mark.parametrize("other_pid", [4242, 99999])
def test_two_writers_do_not_share_a_staging_file(tmp_path, monkeypatch, other_pid) -> None:
    # The deliberate delta of #1573: aura_cost and the reseal tool staged a
    # bare ".tmp" -- two concurrent writers to one target then interleaved on
    # one staging inode. The owner's suffix is per-process, so each writer
    # stages beside, never on, the other's file.
    monkeypatch.setattr(csc, "_TEMP_SUFFIX", None)
    mine = csc.unique_temp_suffix()
    monkeypatch.setattr(csc, "_TEMP_SUFFIX", None)
    monkeypatch.setattr(csc.os, "getpid", lambda: other_pid)
    theirs = csc.unique_temp_suffix()
    assert mine != theirs
    assert mine.startswith(".tmp") and theirs.startswith(".tmp")


def test_nofollow_publication_keeps_stage_link_cleanup_on_admitted_parent(tmp_path, monkeypatch):
    parent = tmp_path / "owner"
    parent.mkdir()
    retained = tmp_path / "retained-owner"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    target = parent / "artifact.json"
    link = csc.os.link
    fsync = csc.os.fsync
    syncs = []

    def swap_parent_at_link(source, destination, **kwargs):
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
        parent.rename(retained)
        parent.symlink_to(foreign, target_is_directory=True)
        return link(source, destination, **kwargs)

    def record_fsync(fd):
        syncs.append(fd)
        return fsync(fd)

    monkeypatch.setattr(csc.os, "link", swap_parent_at_link)
    monkeypatch.setattr(csc.os, "fsync", record_fsync)
    assert csc.publish_new_bytes(target, b"owned bytes", nofollow=True)
    assert list(foreign.iterdir()) == []
    assert (retained / "artifact.json").read_bytes() == b"owned bytes"
    assert len(syncs) == 2, "file then admitted directory durability"
    assert [path.name for path in retained.iterdir()] == ["artifact.json"]


def test_nofollow_publication_existing_destination_remains_unchanged(tmp_path):
    target = tmp_path / "owner" / "artifact.json"
    assert csc.publish_new_bytes(target, b"original", nofollow=True)
    assert not csc.publish_new_bytes(target, b"replacement", nofollow=True)
    assert target.read_bytes() == b"original"
    assert [path.name for path in target.parent.iterdir()] == ["artifact.json"]
