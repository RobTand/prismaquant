"""PQ #1361 (#1301 part 2): file, bytes and text digest sites keep their bytes.

Each site below hashed a file, a byte string or a text with its own copy of
one recipe. Each now binds or calls its owner in ``prismaquant.digests``. The
outcomes of the pre-move code on every input here (the returned digest, or the
exception type, ``str()`` and chained cause) are frozen in
``fixtures/digest_profiles_1301_files.json`` (``tests/golden_table.py``), and
every call goes through the old site's name.

File inputs straddle each block size a site reads with (1, 4, 8 and 16 MiB,
and ``hashlib.file_digest``'s 256 KiB buffer). ``str`` paths are checked only
for sites whose old signature took them.

The pytest dispatch surface is grouped per input (PQ #2349): one node per
input iterates the original ordered site lists and makes every original call
-- same Path or ``str`` argument, same extra keywords, same call order -- so
each frozen outcome keeps its own fixture row under the node's own call
ordinal. Fewer dispatch nodes, not fewer asserted outcomes.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import pytest

from prismaquant import digests
from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("digest_profiles_1301_files")

_KIB = 1024
_MIB = 1024 * 1024
_SIZES = sorted({0, 1, *(edge + delta for edge in (256 * _KIB, _MIB, 4 * _MIB, 8 * _MIB, 16 * _MIB)
                          for delta in (-1, 0, 1))})


def _site(ref):
    module, _, name = ref.rpartition(".")
    return getattr(importlib.import_module(module), name)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("digest_files")
    pattern = bytes(range(256))
    for size in _SIZES:
        (root / f"b{size}").write_bytes((pattern * (size // 256 + 1))[:size])
    (root / "directory").mkdir()
    os.symlink(root / f"b{_MIB}", root / "link")
    return root


#: ``(site, takes_str, extra keyword arguments)``.
FILE_SITES = [
    ("prismaquant.anchored_cost._sha256_file", False, {}),
    ("prismaquant.cost_streaming._file_sha256", False, {}),
    ("prismaquant.joint_adjoint_band._sha256_file", True, {}),
    ("prismaquant.joint_projection_backend._sha", True, {}),
    ("prismaquant.joint_quanta_join._sha_file", False, {"where": "w"}),
    ("prismaquant.lane_eligibility._sha256", False, {}),
    ("prismaquant.native_receipt_table.file_sha256", True, {}),
    ("prismaquant.prismasnap_checkpoint._sha256_file", False, {}),
    ("prismaquant.prismasnap_validation._sha256_file", False, {}),
    ("prismaquant.production_cache_stripes._sha256", False, {}),
    ("prismaquant.sample_parallel_probe._sha256_file", True, {}),
    ("prismaquant.stage_a_chain_split._file_sha256", True, {}),
    ("prismaquant.tessera_anchored_surface._sha", True, {}),
    ("prismaquant.tessera_joint_aura._sha", True, {}),
    ("prismaquant.tessera_legal_domain._file_digest", True, {}),
    ("prismaquant.tessera_materialization._sha", True, {}),
    ("prismaquant.union_production_cache._file_sha256", False, {}),
    ("tools.measure_vllm_full_kl._file_sha256", True, {}),
]
_FILE_INPUTS = [f"b{size}" for size in _SIZES] + ["missing", "directory", "link"]


@pytest.mark.parametrize("name", _FILE_INPUTS)
def test_file_digest_site(files, name):
    path = files / name
    for ref, takes_str, kwargs in FILE_SITES:  # original site order
        site = _site(ref)
        GOLDEN.call(lambda: site(path, **kwargs), tmp=files)
        if takes_str:
            GOLDEN.call(lambda: site(str(path), **kwargs), tmp=files)


@pytest.mark.parametrize("chunk", [1, 7, _MIB])
def test_prismasnap_file_digest_keeps_its_chunk_argument(files, chunk):
    for ref in ["prismaquant.prismasnap_checkpoint._sha256_file",
                "prismaquant.prismasnap_validation._sha256_file"]:
        site = _site(ref)
        GOLDEN.call(lambda: site(files / f"b{_MIB + 1}", chunk), tmp=files)
        GOLDEN.call(lambda: site(files / f"b{_MIB + 1}", chunk_bytes=chunk), tmp=files)


BYTES_SITES = [
    "prismaquant.emu_forward_kl._sha256",
    "tools.assemble_t4_overlay.sha",
    "tools.build_t4_overlay_catalog.sha",
    "tools.extract_layer8_native_metadata.sha",
]
_BYTES_INPUTS = {
    "empty": b"",
    "ascii": b"abc",
    "high": bytes(range(256)),
    "bytearray": bytearray(b"\x00\xff"),
    "memoryview": memoryview(b"view"),
    "text": "not bytes",
    "none": None,
}


@pytest.mark.parametrize("name", list(_BYTES_INPUTS))
def test_bytes_digest_site(name):
    for ref in BYTES_SITES:  # original site order
        site = _site(ref)
        GOLDEN.call(lambda: site(_BYTES_INPUTS[name]))


_TEXT_INPUTS = {
    "empty": "",
    "ascii": "abc",
    "non_ascii": "café \U0001f600",
    "surrogate": "a\ud800",
    "number": 12,
    "none": None,
}


@pytest.mark.parametrize("name", list(_TEXT_INPUTS))
def test_text_digest_site(name):
    site = _site("prismaquant.tessera_hessian.text_sha256")
    GOLDEN.call(lambda: site(_TEXT_INPUTS[name]))


@pytest.mark.parametrize("name", list(_TEXT_INPUTS))
def test_unit_path_site(name):
    for ref in ["prismaquant.cost_stage_checkpoint.unit_path",
                "prismaquant.aura_cost._aura_unit_checkpoint_path"]:
        site = _site(ref)
        GOLDEN.call(lambda: site(Path("/root"), _TEXT_INPUTS[name]))


# --- the owners ---------------------------------------------------------------

_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_each_profile_pins_its_digest(tmp_path):
    assert digests.bytes_sha256hex(b"") == _EMPTY
    assert digests.bytes_sha256hex(memoryview(b"abc")) == _ABC
    assert digests.text_sha256hex("abc") == _ABC
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    for block_size in (1, 2, 3, digests.FILE_BLOCK_BYTES):
        assert digests.file_sha256hex(path, block_size=block_size) == _ABC
        assert digests.file_sha256hex(str(path), block_size=block_size) == _ABC


class _IndexZero:
    """An integer-index-like zero, the spelling ``read`` itself coerces."""

    def __index__(self) -> int:
        return 0


def test_file_sha256hex_refuses_zero_read_count(tmp_path):
    # A zero read count never advances, so the unguarded owner returned the
    # empty-input digest for any file (PQ #2344). The refusal is
    # content-independent -- an empty file must refuse too, not pass by
    # coincidence -- and covers every zero spelling ``read`` coerces.
    for name, payload in (("empty", b""), ("abc", b"abc")):
        path = tmp_path / name
        path.write_bytes(payload)
        for size in (0, False, _IndexZero()):
            with pytest.raises(ValueError, match="block_size"):
                digests.file_sha256hex(path, block_size=size)
            with pytest.raises(ValueError, match="block_size"):
                digests.file_sha256hex(str(path), block_size=size)


def test_file_sha256hex_keeps_read_all_and_type_refusals(tmp_path):
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    assert digests.file_sha256hex(path, block_size=-1) == _ABC
    assert digests.file_sha256hex(path, block_size=None) == _ABC
    with pytest.raises(TypeError):
        digests.file_sha256hex(path, block_size=0.0)


# --- the package-digest route (prismaquant#2648) ----------------------------------
#
# Census call ``hashlib:tools/generate_transition_rewrites.py::_package_digest@40:13``
# (refresh_2540 owner ``prismaquant.digests.LengthFramedSourceSha256``) keeps its
# exact bytes: be32 name length, strict UTF-8 name, be64 payload length, payload.
# The Path-component sort, the file reads, and the hexdigest wrapper stay with
# the caller. Every row below pins the pre-route outcome.

#: The census call this section freezes.
_PACKAGE_DIGEST_CENSUS_ID = (
    "hashlib:tools/generate_transition_rewrites.py::_package_digest@40:13")

#: Small inputs shaped like the real argument: plain, nested, non-ASCII, empty.
_PACKAGE_DIGEST_INPUTS = {
    "ascii": {"b.py": b"keep\n", "a.py": b"one\n"},
    "nested": {"a.py": b"payload for a.py\n", "a/b.py": b"payload for a/b.py\n"},
    "unicode": {"caf\u00e9.py": b"x", "\u00e9/b.py": b"\x00\xff", "a.py": b"first"},
    "empty": {},
}


def _package_digest_path_framed(files):
    """The exact pre-route byte recipe, in the caller's Path-component order."""
    framed = hashlib.sha256()
    for name, payload in sorted(files.items(), key=lambda item: Path(item[0])):
        encoded = name.encode("utf-8")
        framed.update(len(encoded).to_bytes(4, "big"))
        framed.update(encoded)
        framed.update(len(payload).to_bytes(8, "big"))
        framed.update(payload)
    return framed


@pytest.mark.parametrize("name", list(_PACKAGE_DIGEST_INPUTS))
def test_package_digest_keeps_its_bytes(name):
    """The routed site keeps its digest on each shaped input (PQ #2648)."""
    from tools.generate_transition_rewrites import _package_digest

    files = _PACKAGE_DIGEST_INPUTS[name]
    GOLDEN.value((_PACKAGE_DIGEST_CENSUS_ID, name))
    GOLDEN.call(lambda: _package_digest(files))
    assert _package_digest(files) == _package_digest_path_framed(files).hexdigest()


def test_package_digest_keeps_path_order_not_posix_order():
    """``a/b.py`` precedes ``a.py``: Path parts, not POSIX strings (PQ #2648)."""
    from tools.generate_transition_rewrites import _package_digest

    files = _PACKAGE_DIGEST_INPUTS["nested"]
    GOLDEN.value((_PACKAGE_DIGEST_CENSUS_ID, "path-order"))
    forward = _package_digest(files)
    backward = _package_digest(dict(reversed(list(files.items()))))
    GOLDEN.call(lambda: forward)
    GOLDEN.call(lambda: backward)
    assert forward == backward
    assert forward == _package_digest_path_framed(files).hexdigest()
    posix = hashlib.sha256()
    for name in sorted(files):
        encoded, payload = name.encode("utf-8"), files[name]
        posix.update(len(encoded).to_bytes(4, "big") + encoded)
        posix.update(len(payload).to_bytes(8, "big") + payload)
    assert forward != posix.hexdigest()


def test_package_digest_truncation_changes_the_digest():
    """A payload prefix never stands in for the full payload (PQ #2648)."""
    from tools.generate_transition_rewrites import _package_digest

    GOLDEN.value((_PACKAGE_DIGEST_CENSUS_ID, "truncation"))
    full = _package_digest({"a.py": b"abc"})
    short = _package_digest({"a.py": b"ab"})
    GOLDEN.call(lambda: full)
    GOLDEN.call(lambda: short)
    assert full != short
    assert full == _package_digest_path_framed({"a.py": b"abc"}).hexdigest()


@pytest.mark.parametrize(
    ("label", "files", "raised"),
    (("surrogate-name", {"a\ud800.py": b"x"}, "builtins.UnicodeEncodeError"),
     ("str-payload", {"a.py": "not bytes"}, "builtins.TypeError"),
     ("none-payload", {"a.py": None}, "builtins.TypeError")),
    ids=["surrogate-name", "str-payload", "none-payload"])
def test_package_digest_refuses(label, files, raised):
    """Strict UTF-8 names and byte payloads refuse as before (PQ #2648)."""
    from tools.generate_transition_rewrites import _package_digest

    GOLDEN.value((_PACKAGE_DIGEST_CENSUS_ID, label))
    row = GOLDEN.call(lambda: _package_digest(files))
    assert row["raised"] == raised


def test_package_digest_routes_through_the_framed_source_owner(monkeypatch):
    """The site feeds caller-ordered records to LengthFramedSourceSha256."""
    import tools.generate_transition_rewrites as generator
    from prismaquant import digests as owners

    seen = []
    real = owners.LengthFramedSourceSha256

    class _Spy(real):
        def update(self, name, payload):
            seen.append((name, payload))
            super().update(name, payload)

    monkeypatch.setattr(generator, "LengthFramedSourceSha256", _Spy)
    files = _PACKAGE_DIGEST_INPUTS["unicode"]
    Routable = getattr(generator, "LengthFramedSourceSha256", None)
    assert Routable is _Spy
    digest = generator._package_digest(files)
    ordered = sorted(files.items(), key=lambda item: Path(item[0]))
    assert seen == ordered
    check = real()
    for name, payload in ordered:
        check.update(name, payload)
    assert digest == check.hexdigest()


def test_package_digest_generator_output_matches_frozen(tmp_path, capsys):
    """A small sealed/executing pair emits the frozen rewrite block (PQ #2648)."""
    from tools.generate_transition_rewrites import main

    GOLDEN.value((_PACKAGE_DIGEST_CENSUS_ID, "generator"))
    sealed = tmp_path / "sealed"
    executing = tmp_path / "executing"
    (sealed / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (sealed / "a.py").write_text("one\ntwo\nthree\nfour\n")
    (sealed / "b.py").write_text("keep\n")
    (sealed / "c.py").write_text("same\n")
    (executing / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (executing / "a.py").write_text("one\ntwo\nthree\nfour\n")
    (executing / "b.py").write_text("keep\nchanged\n")
    (executing / "c.py").write_text("same\n")
    (executing / "new.py").write_text("verifier\n")
    assert main(["--sealed-dir", str(sealed), "--executing-dir", str(executing),
                 "--new-file", "new.py"]) == 0
    out, _ = capsys.readouterr()
    GOLDEN.value(out)
