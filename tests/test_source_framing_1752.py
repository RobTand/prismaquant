"""Byte identity and owner routing for the length-framed source recipe (#1752)."""
from __future__ import annotations

import hashlib
from importlib import import_module
import struct

import pytest

from prismaquant import digests


MODULES = (
    "prismaquant.joint_aura_source_transition",
    "prismaquant.joint_aura_run_transition",
    "prismaquant.joint_aura_retained_budget_transition",
)
# Frozen from the literal legacy recipe by an admitted fixture-generation action.
GOLDEN = (
    ((), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ((("", b""),), "15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b"),
    ((("a", b"bc"),), "e7ab503819a08f424da9ba8421b745917f2f82f92b7e6a09ecce37d61289b827"),
    ((("ab", b"c"),), "2cfe02211ae98daa5145daefc9c3f6f87bd1c4d9231fdd7b7bc98aea18fb7215"),
    ((("nested/é.py", b"\x00\xff\r\n"), ("a", b"tail")),
     "2b2541143f63fc56d5fd5074fdd5b68c737fd52973d92df95217daebf3204c48"),
    ((("a", b"tail"), ("nested/é.py", b"\x00\xff\r\n")),
     "696aabeeef18a3c8afc6b32b28e85c8e827705eaabb39864da119f95e6ff67cb"),
    ((("same", b"one"), ("same", b"two")),
     "fb515c75a1352e81a6ba3247d21312d9f2b8b2bbbda8638fe9d2345e3d7aa5b6"),
)


def _legacy(entries):
    digest = hashlib.sha256()
    for name, payload in entries:
        encoded = name.encode()
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


@pytest.mark.parametrize("entries,expected", GOLDEN)
def test_profile_golden_table(entries, expected):
    profile = digests.LengthFramedSourceSha256()
    for name, payload in entries:
        profile.update(name, payload)
    assert profile.hexdigest() == expected == _legacy(entries)
    assert profile.hexdigest() == expected  # observing a digest does not finish it


def test_profile_streams_and_preserves_utf8_refusal():
    profile = digests.LengthFramedSourceSha256()
    profile.update("first", b"\x00\xff")
    first = profile.hexdigest()
    assert first == _legacy((("first", b"\x00\xff"),))
    profile.update("second", b"\r\n")
    assert profile.hexdigest() == _legacy((("first", b"\x00\xff"), ("second", b"\r\n")))
    assert profile.hexdigest() != first
    with pytest.raises(UnicodeEncodeError):
        profile.update("\ud800", b"untouched")
    assert profile.hexdigest() == _legacy((("first", b"\x00\xff"), ("second", b"\r\n")))


def _fixture(module, tmp_path, monkeypatch, payload):
    root = tmp_path / "package"
    root.mkdir()
    label = module.__name__.rsplit(".", 1)[-1] + ".py"
    data = {
        label: b"transition bytes\x00\r\n",
        "source.py": b"new-normalize\r\nnew-version\x00\xff",
        "nested/é.bin": payload,
    }
    for name, raw in data.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name in ("__pycache__/ignored.py", "nested/ignored.pyc", "ignored.pyo"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be hashed")
    monkeypatch.setattr(module, "_SOURCE_REWRITES", {
        "source.py": [("old-normalize", "new-normalize"), ("old-version", "new-version")],
    })
    if hasattr(module, "_NEW_FILES"):
        monkeypatch.setattr(module, "_NEW_FILES", {label})
    if hasattr(module, "dev_mode_enabled"):
        monkeypatch.setattr(module, "dev_mode_enabled", lambda: False)
    current = sorted(data.items())
    original = [(name, raw.replace(b"new-version", b"old-version").replace(
        b"new-normalize", b"old-normalize")) for name, raw in current if name != label]
    monkeypatch.setattr(module, "_CONTRACT", {"source_sha256": _legacy(original)})
    return root, label, current, original


def _spy(module, monkeypatch):
    batches = []

    class RecordingProfile:
        def __init__(self):
            self.owner = digests.LengthFramedSourceSha256()
            self.rows = []
            batches.append(self.rows)

        def update(self, name, payload):
            self.rows.append((name, payload))
            self.owner.update(name, payload)

        def hexdigest(self):
            return self.owner.hexdigest()

    # Before delegation the legacy code ignores this injected owner entirely.
    monkeypatch.setattr(module, "LengthFramedSourceSha256", RecordingProfile, raising=False)
    return batches


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("payload", (b"", b"\x00\xff\r\nUTF-8:\xc3\xa9"))
def test_source_proof_routes(module_name, payload, tmp_path, monkeypatch):
    module = import_module(module_name)
    root, label, current, original = _fixture(module, tmp_path, monkeypatch, payload)
    batches = _spy(module, monkeypatch)
    actual = module.source_proof(root)
    assert actual == {
        "producer_source_sha256": _legacy(current),
        "reconstructed_source_sha256": _legacy(original),
        "transition_module_sha256": hashlib.sha256((root / label).read_bytes()).hexdigest(),
    }
    assert batches == [current, original]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("fault,message", (
    ("missing", "unapproved or missing source hunk in source.py"),
    ("duplicate", "unapproved or missing source hunk in source.py"),
    ("incomplete", "incomplete source proof"),
    ("contract", "unapproved producer package change"),
))
def test_source_proof_refusals(module_name, fault, message, tmp_path, monkeypatch):
    module = import_module(module_name)
    root, _, _, _ = _fixture(module, tmp_path, monkeypatch, b"payload")
    if fault == "missing":
        (root / "source.py").write_bytes(b"no approved hunk")
    elif fault == "duplicate":
        (root / "source.py").write_bytes(b"new-normalize new-normalize new-version")
    elif fault == "incomplete":
        (root / "source.py").unlink()
    else:
        monkeypatch.setattr(module, "_CONTRACT", {"source_sha256": "0" * 64})
    with pytest.raises(ValueError, match=message):
        module.source_proof(root)


def test_run_dev_mode_records_actual_tree_without_reconstruction(tmp_path, monkeypatch):
    module = import_module(MODULES[1])
    root, label, current, _ = _fixture(module, tmp_path, monkeypatch, b"payload")
    monkeypatch.setattr(module, "dev_mode_enabled", lambda: True)
    monkeypatch.setattr(module, "_SOURCE_REWRITES", {"absent.py": [("old", "new")]})
    monkeypatch.setattr(module, "_CONTRACT", {"source_sha256": "0" * 64})
    warnings = []
    monkeypatch.setattr(module, "dev_warning", warnings.append)
    batches = _spy(module, monkeypatch)
    digest = _legacy(current)
    assert module.source_proof(root) == {
        "producer_source_sha256": digest,
        "reconstructed_source_sha256": digest,
        "transition_module_sha256": hashlib.sha256((root / label).read_bytes()).hexdigest(),
        "dev_uncertified": True,
    }
    assert batches == [current, []]
    assert len(warnings) == 1 and digest in warnings[0]
