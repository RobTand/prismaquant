"""PQ #1607: the transition twins' shared helpers, proved byte-identical.

The three ``joint_aura_*_transition`` modules plus the ``joint_aura_transitions``
router carried their own copies of ``_require``; the twins additionally shared
``_canonical`` / ``_sha`` (now the ``digests.py`` owners
``DIRECT_UTF8_STRICT`` / ``file_sha256hex``), raw-byte hashes (now
``bytes_sha256hex``), and -- for the retained-budget and run twins --
``_bytes_identity`` / ``checkout_head_commit`` / ``_committed_package`` (now
``joint_aura_transition_base``). This file pins the old spellings verbatim
against the new owners, the refusal vocabularies, and that the twins no longer
define the moved names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_aura_retained_budget_transition as budget
from prismaquant import joint_aura_run_transition as run
from prismaquant import joint_aura_source_transition as source
from prismaquant import joint_aura_transition_base as base
from prismaquant import joint_aura_transitions as router
from prismaquant.digests import (
    DIRECT_UTF8_STRICT,
    bytes_sha256hex,
    file_sha256hex,
)


def _old_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _old_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _old_require(ok, message):
    if not ok:
        raise ValueError(f"joint source transition: {message}")


VALUES = [
    {},
    {"b": 1, "a": [1, {"z": None, "y": True, "x": -0.0}]},
    {"unicode": "ünïcodé ✓", "escape": "a\nb\t\"q\"\\"},
    {"tuple": (1, 2)},
    [1, "two", {"three": 3.5}],
]


@pytest.mark.parametrize("value", VALUES)
def test_canonical_matches_direct_utf8_strict(value):
    assert DIRECT_UTF8_STRICT.encoded(value) == _old_canonical(value)


@pytest.mark.parametrize("value", VALUES)
def test_canonical_sha_matches_profile(value):
    assert (DIRECT_UTF8_STRICT.sha256(value)
            == hashlib.sha256(_old_canonical(value)).hexdigest())


def test_canonical_refuses_nan_like_the_old_spelling():
    with pytest.raises(ValueError, match="[Nn]an"):
        _old_canonical({"a": float("nan")})
    with pytest.raises(ValueError, match="[Nn]an"):
        DIRECT_UTF8_STRICT.encoded({"a": float("nan")})


@pytest.mark.parametrize("payload", [b"", b"\x00\xff binary \n", b'{"a": 1}\n' * 1000])
def test_file_sha_matches_old_sha_spelling(tmp_path, payload):
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert file_sha256hex(path) == _old_sha(path)


@pytest.mark.parametrize("raw", [b"", b"receipt bytes", bytes(range(256))])
def test_bytes_sha_matches_old_raw_hash(raw):
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()


def test_require_refusal_vocabulary_is_unchanged():
    with pytest.raises(ValueError, match="joint source transition: probe"):
        base._require(False, "probe")
    with pytest.raises(ValueError, match="joint source transition: probe"):
        _old_require(False, "probe")
    assert base._require(True, "probe") is None


def test_bound_accepts_a_matching_record(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"bound bytes")
    record = {"path": str(path), "sha256": file_sha256hex(path)}
    assert base._bound(record, "probe") == path


@pytest.mark.parametrize("record", [
    {"path": "x", "sha256": "0" * 64, "extra": 1},
    {"path": "x"},
])
def test_bound_refuses_a_misshapen_record(record):
    with pytest.raises(ValueError, match="requires independently bound"):
        base._bound(record, "probe")


def test_bound_refuses_changed_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        base._bound({"path": str(path), "sha256": "0" * 64}, "probe")


def test_bound_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="bytes changed"):
        base._bound({"path": str(tmp_path / "absent"), "sha256": "0" * 64}, "probe")


def test_bytes_identity_passes_the_triple_through():
    execution = {key: str(index) * 64 for index, key in enumerate(base._BYTES)}
    assert base._bytes_identity(execution) == execution


@pytest.mark.parametrize("execution", [
    {"producer_source_sha256": "1" * 64},
    {"producer_source_sha256": "1" * 63, "reconstructed_source_sha256": "2" * 64,
     "transition_module_sha256": "3" * 64},
])
def test_bytes_identity_refuses_a_short_or_missing_key(execution):
    with pytest.raises(ValueError, match="byte identity"):
        base._bytes_identity(execution)


def _gitdir(root, head, **files):
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text(head)
    for name, payload in files.items():
        target = git / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
    return root


def test_checkout_reads_a_detached_head(tmp_path):
    commit = "a" * 40
    assert base.checkout_head_commit(_gitdir(tmp_path, commit)) == commit


def test_checkout_resolves_a_loose_ref(tmp_path):
    commit = "b" * 64
    root = _gitdir(tmp_path, "ref: refs/heads/main", **{"refs/heads/main": commit})
    assert base.checkout_head_commit(root) == commit


def test_checkout_reads_through_a_gitdir_pointer(tmp_path):
    commit = "c" * 40
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "HEAD").write_text(commit)
    (tmp_path / ".git").write_text("gitdir: sub")
    assert base.checkout_head_commit(tmp_path) == commit


def test_checkout_refuses_an_unresolved_ref(tmp_path):
    with pytest.raises(ValueError, match="unresolved"):
        base.checkout_head_commit(_gitdir(tmp_path, "ref: refs/heads/absent"))


def test_checkout_refuses_a_directory_without_git(tmp_path):
    with pytest.raises(ValueError, match="no sealed Git checkout"):
        base.checkout_head_commit(tmp_path)


@pytest.mark.parametrize("module", [budget, run, source, router])
def test_require_is_the_shared_owner(module):
    assert module._require is base._require


def test_moved_budget_and_run_helpers_are_the_shared_owner():
    for module in (budget, run):
        assert module._bound is base._bound
        assert module._bytes_identity is base._bytes_identity
        assert module._committed_package is base._committed_package
        assert module._COMMIT == base._COMMIT


@pytest.mark.parametrize("module", [budget, run, source])
def test_old_helper_spellings_are_gone(module):
    assert not hasattr(module, "_canonical")
    assert not hasattr(module, "_sha")
