"""One stat identity for every file fence (PQ #1531, epic #1295).

The sealed catalog persists four stat keys; they are projected from the one
file identity and must stay byte-identical to what earlier catalogs recorded.
"""
from __future__ import annotations

import json
import os

from prismaquant import joint_catalog_extension as jce
from prismaquant.file_identity import file_stat_signature


def test_the_signature_is_the_full_stat_identity(tmp_path):
    path = tmp_path / "wire.bin"
    path.write_bytes(b"x" * 11)
    value = os.stat(path)
    assert file_stat_signature(value) == (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def test_the_catalog_fence_keeps_its_sealed_wire_format(tmp_path):
    path = tmp_path / "wire.bin"
    path.write_bytes(b"x" * 11)
    value = os.stat(path)
    # The dict #1519 sealed into catalogs, spelled as it was then.
    sealed = {"inode": value.st_ino, "bytes": value.st_size,
              "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}
    fence = jce._catalog_fence(value)
    assert fence == sealed
    assert list(fence) == list(sealed)
    assert json.dumps(fence) == json.dumps(sealed)
    assert json.dumps(fence, sort_keys=True) == json.dumps(sealed, sort_keys=True)
