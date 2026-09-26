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
"""
from __future__ import annotations

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
@pytest.mark.parametrize("ref,takes_str,kwargs", FILE_SITES, ids=[row[0] for row in FILE_SITES])
def test_file_digest_site(files, ref, takes_str, kwargs, name):
    site = _site(ref)
    path = files / name
    GOLDEN.call(lambda: site(path, **kwargs), tmp=files)
    if takes_str:
        GOLDEN.call(lambda: site(str(path), **kwargs), tmp=files)


@pytest.mark.parametrize("chunk", [1, 7, _MIB])
@pytest.mark.parametrize("ref", ["prismaquant.prismasnap_checkpoint._sha256_file",
                                 "prismaquant.prismasnap_validation._sha256_file"])
def test_prismasnap_file_digest_keeps_its_chunk_argument(files, ref, chunk):
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
@pytest.mark.parametrize("ref", BYTES_SITES)
def test_bytes_digest_site(ref, name):
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
@pytest.mark.parametrize("ref", ["prismaquant.cost_stage_checkpoint.unit_path",
                                 "prismaquant.aura_cost._aura_unit_checkpoint_path"])
def test_unit_path_site(ref, name):
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
    assert digests.FILE_BLOCK_BYTES == 8 << 20


#: ``(site, owner, block size the site keeps or None)``.
BINDINGS = [
    ("prismaquant.anchored_cost._sha256_file", digests.file_sha256hex, None),
    ("prismaquant.cost_streaming._file_sha256", digests.file_sha256hex, 16 * _MIB),
    ("prismaquant.joint_adjoint_band._sha256_file", digests.file_sha256hex, _MIB),
    ("prismaquant.joint_projection_backend._sha", digests.file_sha256hex, None),
    ("prismaquant.lane_eligibility._sha256", digests.file_sha256hex, None),
    ("prismaquant.native_receipt_table.file_sha256", digests.file_sha256hex, None),
    ("prismaquant.production_cache_stripes._sha256", digests.file_sha256hex, _MIB),
    ("prismaquant.sample_parallel_probe._sha256_file", digests.file_sha256hex, _MIB),
    ("prismaquant.stage_a_chain_split._file_sha256", digests.file_sha256hex, 4 * _MIB),
    ("prismaquant.tessera_anchored_surface._sha", digests.file_sha256hex, _MIB),
    ("prismaquant.tessera_joint_aura._sha", digests.file_sha256hex, None),
    ("prismaquant.tessera_legal_domain._file_digest", digests.file_sha256hex, None),
    ("prismaquant.tessera_materialization._sha", digests.file_sha256hex, None),
    ("prismaquant.union_production_cache._file_sha256", digests.file_sha256hex, None),
    ("prismaquant.emu_forward_kl._sha256", digests.bytes_sha256hex, None),
    ("tools.assemble_t4_overlay.sha", digests.bytes_sha256hex, None),
    ("tools.build_t4_overlay_catalog.sha", digests.bytes_sha256hex, None),
    ("tools.extract_layer8_native_metadata.sha", digests.bytes_sha256hex, None),
]


@pytest.mark.parametrize("ref,owner,block_size", BINDINGS, ids=[row[0] for row in BINDINGS])
def test_each_site_binds_its_owner(ref, owner, block_size):
    site = _site(ref)
    if block_size is None:
        assert site is owner
    else:
        assert site.func is owner and site.args == () and site.keywords == {"block_size": block_size}


def test_the_unit_path_and_the_full_kl_digest_have_one_owner():
    from prismaquant import aura_cost, cost_stage_checkpoint
    from tools import full_kl_teacher_payload, measure_vllm_full_kl

    assert aura_cost._aura_unit_checkpoint_path is cost_stage_checkpoint.unit_path
    assert measure_vllm_full_kl._file_sha256 is full_kl_teacher_payload.file_sha256
