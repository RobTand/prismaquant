"""The generator that writes a closed transition's literal source-rewrite table."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.generate_transition_rewrites import _hunks, _package_digest, main


def _tree(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return root


def _apply_reverse(executing, rewrites, new_files):
    """What ``source_proof`` does: reverse each hunk, omit the new files, hash."""
    reconstructed = {}
    for name, payload in executing.items():
        if name in new_files:
            continue
        for old, new in reversed(rewrites.get(name, ())):
            assert payload.count(new) == 1
            payload = payload.replace(new, old, 1)
        reconstructed[name] = payload.encode()
    return _package_digest(reconstructed)


@pytest.fixture
def trees(tmp_path):
    sealed = {"a.py": "one\ntwo\nthree\nfour\n", "b.py": "keep\n", "c.py": "same\n"}
    executing = {"a.py": "one\nTWO\nthree\nfour\n", "b.py": "keep\n", "c.py": "same\n",
                 "new.py": "added\n"}
    _tree(tmp_path / "sealed", sealed)
    _tree(tmp_path / "executing", executing)
    return {"sealed": sealed, "executing": executing, "root": tmp_path}


def test_the_generated_table_reconstructs_the_sealed_package(trees, capsys):
    argv = ["--sealed-dir", str(trees["root"] / "sealed"),
            "--executing-dir", str(trees["root"] / "executing"), "--new-file", "new.py"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("# BEGIN GENERATED REWRITES")
    namespace = {}
    exec(printed.split("\n", 1)[1].rsplit("# END", 1)[0], namespace)  # noqa: S102
    rewrites = namespace["_SOURCE_REWRITES"]
    assert set(rewrites) == {"a.py"}
    sealed_digest = _package_digest({name: text.encode() for name, text in trees["sealed"].items()})
    assert _apply_reverse(trees["executing"], rewrites, {"new.py"}) == sealed_digest


def test_an_unnamed_new_file_refuses(trees):
    with pytest.raises(SystemExit, match="new.py"):
        main(["--sealed-dir", str(trees["root"] / "sealed"),
              "--executing-dir", str(trees["root"] / "executing")])


def test_a_dropped_file_refuses(trees):
    (trees["root"] / "executing" / "c.py").unlink()
    with pytest.raises(SystemExit, match="dropped"):
        main(["--sealed-dir", str(trees["root"] / "sealed"),
              "--executing-dir", str(trees["root"] / "executing"), "--new-file", "new.py"])


def test_a_stated_sealed_digest_is_required_to_hold(trees):
    with pytest.raises(SystemExit, match="hashes to"):
        main(["--sealed-dir", str(trees["root"] / "sealed"),
              "--executing-dir", str(trees["root"] / "executing"), "--new-file", "new.py",
              "--expect-sealed-sha256", "0" * 64])


def test_each_hunk_is_unique_in_both_files():
    """A repeated line is grown with context until both sides name one place."""
    old = "x\ny\nx\ny\nEND\n"
    new = "x\ny\nx\nZ\nEND\n"
    pairs = _hunks(old, new)
    assert pairs
    for old_snippet, new_snippet in pairs:
        assert old.count(old_snippet) == 1
        assert new.count(new_snippet) == 1
    payload = new
    for old_snippet, new_snippet in reversed(pairs):
        payload = payload.replace(new_snippet, old_snippet, 1)
    assert payload == old
