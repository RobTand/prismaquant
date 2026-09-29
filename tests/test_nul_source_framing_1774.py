"""Literal legacy bytes and caller-owned fences; #1762 is not a migration here."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from prismaquant import digests
from prismaquant import runtime_provenance as provenance
from prismaquant import tessera_reader as reader


def _legacy(entries):
    digest = hashlib.sha256()
    for name, payload in entries:
        digest.update(name.encode() + b"\0" + payload + b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("entries,expected", [
    ([], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ([("a.py", b"")], "537140573a4a2ccb69aea685cb51908ed3a42601d4d9d34c89a314549e8d4f18"),
    ([("a.py", b"\0\xff\r\n")], "7c346c5bbc72ec5b5ee783e3935d3dab28fb80ff632c620a425e63966970c657"),
    ([("é.py", b"\0\xff"), ("b.cu", b"gpu\r\n")], "558388e7eecfa16ae667650385f114cb9725f516903d2bc0a5f47fc57e8b36d3"),
    ([("b.cu", b"gpu\r\n"), ("é.py", b"\0\xff")], "d2f998f66203e9a6180771f6e7d0c62facc631831810f398ffec028c186cd6bf"),
    ([("a.py", b"x"), ("a.py", b"y")], "06be3897a8f534711f873f2e0d34236a742252e0280316ffcffb02392d2ddb28"),
    ([("__init__.py", b""), ("a.py", b"first\0b.py\0second")], "c0d3b3d116abc2201d9f3b9574a45e795eb7219ed24f9a90b01bf61c3550396a"),
    ([("__init__.py", b""), ("a.py", b"first"), ("b.py", b"second")], "c0d3b3d116abc2201d9f3b9574a45e795eb7219ed24f9a90b01bf61c3550396a"),
])
def test_profile_golden_table(entries, expected):
    assert _legacy(entries) == expected
    profile = digests.LegacyNulSourceSha256()
    for name, raw in entries:
        profile.update(name, raw)
    assert profile.hexdigest() == expected


def test_profile_streaming_and_strict_utf8(monkeypatch):
    entries = [("é.py", b"\0\xff"), ("a.h", b"header")]
    expected = _legacy(entries)
    real = hashlib.sha256()
    chunks = []

    class ObservedHash:
        def update(self, raw):
            chunks.append(raw)
            real.update(raw)

        def hexdigest(self):
            return real.hexdigest()

    monkeypatch.setattr(digests.hashlib, "sha256", ObservedHash)
    profile = digests.LegacyNulSourceSha256()
    for name, raw in entries:
        profile.update(name, raw)
    assert chunks == [part for name, raw in entries for part in (name.encode(), b"\0", raw, b"\0")]
    assert profile.hexdigest() == expected
    assert set(vars(profile)) == {"_digest"}
    with pytest.raises(UnicodeEncodeError):
        profile.update("\udcff.py", b"x")
    assert profile.hexdigest() == expected


def _profile_spy(monkeypatch, module):
    batches = []
    owner = getattr(digests, "LegacyNulSourceSha256", None)

    class Observed:
        def __init__(self):
            assert owner is not None
            self.digest = owner()
            self.calls = []
            batches.append(self.calls)

        def update(self, name, payload):
            self.calls.append((name, payload))
            self.digest.update(name, payload)

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(module, "LegacyNulSourceSha256", Observed, raising=False)
    return batches


def _bytes_spy(monkeypatch, module):
    calls = []
    owner = digests.bytes_sha256hex

    def observed(raw):
        calls.append(raw)
        return owner(raw)

    monkeypatch.setattr(module, "bytes_sha256hex", observed, raising=False)
    return calls


@pytest.mark.parametrize("payload", [b"", b"\0\xff\r\n\xc3\xa9"])
def test_source_digest_routes(payload, monkeypatch):
    files = {"z.cpp": payload, "雪.py": b"x", "a.cuh": b"a", "note.json": b"excluded"}
    expected = [(name, files[name]) for name in sorted(files, key=Path)
                if Path(name).suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}]
    batches = _profile_spy(monkeypatch, provenance)
    assert provenance._source_digest(files) == _legacy(expected)
    assert batches == [expected]


def test_empty_source_digest_routes(monkeypatch):
    batches = _profile_spy(monkeypatch, provenance)
    assert provenance._source_digest({"note.json": b"ignored"}) == _legacy([])
    assert batches == [[]]


@pytest.mark.parametrize("payload", [b"", b"\0\xff\r\n\xc3\xa9"])
def test_source_tree_routes(payload, tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    (root / "__init__.py").write_bytes(b"")
    (root / "雪.py").write_bytes(payload)
    (root / "nested").mkdir()
    (root / "nested" / "a.cu").write_bytes(b"gpu\r\n")
    (root / "ignored.json").write_bytes(b"excluded")
    paths = sorted(p for p in root.rglob("*") if p.suffix in reader.SOURCE_SUFFIXES)
    entries = [(p.relative_to(root).as_posix(), p.read_bytes()) for p in paths]
    batches = _profile_spy(monkeypatch, reader)
    byte_calls = _bytes_spy(monkeypatch, reader)
    sha, files = reader._source_tree(root)
    assert sha == _legacy(entries)
    assert files == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert batches == [entries]
    assert byte_calls == [raw for _, raw in entries]


@pytest.mark.parametrize("payload", [b"", b"\0\xff\r\n"])
@pytest.mark.parametrize("matches", [True, False])
def test_artifact_reader_routes(payload, matches, tmp_path, monkeypatch):
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    calls = _bytes_spy(monkeypatch, provenance)
    expected = hashlib.sha256(payload).hexdigest() if matches else "0" * 64
    binding = {"path": path.name, "sha256": expected}
    if matches:
        assert provenance.ArtifactReader(tmp_path).bytes(binding, "fixture") == (path, payload)
    else:
        with pytest.raises(provenance.RuntimePriceError, match="artifact SHA-256: evidence mismatch"):
            provenance.ArtifactReader(tmp_path).bytes(binding, "fixture")
    assert calls == [payload]


@pytest.mark.parametrize("payload", [b"", b"\0\xff\r\n"])
def test_source_members_routes(payload, monkeypatch):
    tree = {"雪.py": payload, "z.json": b"metadata"}
    calls = _bytes_spy(monkeypatch, provenance)
    members = {name: hashlib.sha256(raw).hexdigest() for name, raw in tree.items()}
    expected = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert provenance._source_tree_identity(tree) == (expected, len(tree))
    assert calls == list(tree.values())


@pytest.mark.parametrize("matches", [True, False])
@pytest.mark.parametrize("absolute_import", [True, False])
def test_loader_routes(matches, absolute_import, tmp_path, monkeypatch):
    raw = b"import tessera.serving\n" if absolute_import else "# 雪\r\nvalue = 7\r\n".encode()
    path = tmp_path / "module.py"
    path.write_bytes(raw)
    calls = _bytes_spy(monkeypatch, reader)
    expected = hashlib.sha256(raw).hexdigest() if matches else "0" * 64
    loader = reader._ReaderSourceLoader("bound.module", str(path), expected)
    if not matches:
        with pytest.raises(ImportError, match="reader source changed after its package checksum"):
            loader.get_code("bound.module")
    elif absolute_import:
        with pytest.raises(ImportError, match="contains an absolute tessera import"):
            loader.get_code("bound.module")
    else:
        namespace = {}
        exec(loader.get_code("bound.module"), namespace)
        assert namespace["value"] == 7
    assert calls == [raw]


@pytest.mark.parametrize("payload", [b"", b"\0\xff\r\n"])
def test_package_installed_members_routes(payload, tmp_path, monkeypatch):
    tree = {"src/tessera/__init__.py": b"", "src/tessera/a.py": payload, "build/metadata.json": b"{}"}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, raw in tree.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    archive_bytes = buffer.getvalue()
    archive_binding = {"path": "source.tar", "sha256": hashlib.sha256(archive_bytes).hexdigest()}

    class FixedArchiveReader:
        def bytes(self, reference, where):
            assert reference == archive_binding
            assert where == "original plugin source archive"
            return tmp_path / "source.tar", archive_bytes

    calls = _bytes_spy(monkeypatch, provenance)
    declaration = {"archive": archive_binding, "prefix": "src/tessera", "excluded_files": ["__init__.py"]}
    actual = provenance._package_source(declaration, FixedArchiveReader())
    members = {name: hashlib.sha256(raw).hexdigest() for name, raw in tree.items()}
    tree_identity = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert actual == {"archive_sha256": archive_binding["sha256"],
                      "source_identity_sha256": tree_identity, "source_identity_members": len(tree),
                      "source_tree_sha256": _legacy([("__init__.py", b""), ("a.py", payload)]),
                      "installed_source_sha256": _legacy([("a.py", payload)]),
                      "installed_files": {"a.py": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}}}
    assert calls == [*tree.values(), payload]


@pytest.mark.parametrize("kind", ["missing-init", "symlink", "non-file", "read-error"])
def test_source_tree_fences_unchanged(kind, tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    if kind != "missing-init":
        (root / "__init__.py").write_bytes(b"")
    path = root / "z.py"
    if kind == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(b"x")
        path.symlink_to(outside)
    elif kind == "non-file":
        path.mkdir()
    else:
        path.write_bytes(b"x")
    if kind == "read-error":
        original = Path.read_bytes

        def broken(self):
            if self == path:
                raise OSError("fixture read failure")
            return original(self)

        monkeypatch.setattr(Path, "read_bytes", broken)
        with pytest.raises(OSError, match="fixture read failure"):
            reader._source_tree(root)
    else:
        message = "complete package directory" if kind == "missing-init" else "regular files"
        with pytest.raises(ValueError, match=message):
            reader._source_tree(root)


def test_strict_utf8_name_refusal():
    with pytest.raises(UnicodeEncodeError):
        provenance._source_digest({"\udcff.py": b"x"})


def test_legacy_ambiguity_is_preserved_not_silently_migrated(tmp_path):
    left = {"__init__.py": b"", "a.py": b"first\0b.py\0second"}
    right = {"__init__.py": b"", "a.py": b"first", "b.py": b"second"}
    expected = "c0d3b3d116abc2201d9f3b9574a45e795eb7219ed24f9a90b01bf61c3550396a"
    assert provenance._source_digest(left) == provenance._source_digest(right) == expected
    counts = []
    for label, entries in [("left", left), ("right", right)]:
        root = tmp_path / label
        root.mkdir()
        for name, raw in entries.items():
            (root / name).write_bytes(raw)
        sha, files = reader._source_tree(root)
        assert sha == expected
        counts.append(len(files))
    assert counts == [2, 3]
