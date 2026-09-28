"""One owner for the atomic-write recipe (#1573).

`prismaquant/cost_stage_checkpoint.py::atomic_write_bytes` is the owner of
"stage bytes beside the target, fsync, replace, fsync the directory".
`prismaquant/aura_cost.py` carried its own copy and now shares the owner.
`tools/reseal_campaign_identity.py` keeps a deliberate standalone twin:
its row rewrites are stdlib-only (the retired must_differ entry recorded
that importing the package pulls in torch, and consumer_merge does the one
lazy import), so its copy stays local and cross-referenced. These tests pin
the delegation, the standalone boundary, and that the owner's durability
ordering is exactly the recipe the copies had:

write -> flush -> fsync(file) -> os.replace -> fsync(containing directory)

The published bytes are the caller's payload (unchanged by this primitive),
so the byte-relevant encoders stay pinned where they already are
(tests/test_digest_calibration_cache_1512.py and friends); what this file
pins is the delegation and the fsync/replace sequence itself.
"""
from __future__ import annotations

import os

import pytest

from prismaquant import aura_cost
from prismaquant import cost_stage_checkpoint as csc


def test_aura_cost_uses_the_owner_recipe() -> None:
    assert aura_cost.atomic_write_bytes is csc.atomic_write_bytes


def test_reseal_tool_stays_stdlib_only_for_row_rewrites() -> None:
    # The ratchet's retired must_differ entry recorded the design this slice
    # preserves: reseal keeps its row rewrites stdlib-only (importing the
    # package pulls in torch; consumer_merge does the one lazy import).
    # Mechanically: every prismaquant import in the tool sits inside a
    # function body, never at module level.
    import ast
    from pathlib import Path as _Path

    source = _Path("tools/reseal_campaign_identity.py").read_text()
    tree = ast.parse(source)
    module_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.Import) and any(a.name.split(".")[0] == "prismaquant" for a in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "prismaquant")
        )
    ]
    assert module_level_imports == [], module_level_imports


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
