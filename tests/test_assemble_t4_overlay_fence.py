"""assemble_t4_overlay holds catalog artifacts to one fence, the catalog's own."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

import prismaquant.joint_catalog_extension as jce

TOOL = Path(__file__).resolve().parents[1] / "tools" / "assemble_t4_overlay.py"


def _tool():
    spec = importlib.util.spec_from_file_location("assemble_t4_overlay_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamp(path):
    s = path.stat()
    return dict(inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _cell(tmp_path):
    wire, render = tmp_path / "wire.bin", tmp_path / "render.bin"
    wire.write_bytes(b"wire-bytes")
    render.write_bytes(b"render-bytes")
    return {"wire": str(wire), "render": str(render), "wire_stat": _stamp(wire), "render_stat": _stamp(render),
            "record": {"blob_sha256": hashlib.sha256(b"wire-bytes").hexdigest()}}


def test_assemble_accepts_a_wire_whose_ctime_moved_under_a_new_hardlink(tmp_path):
    cell = _cell(tmp_path)
    os.link(cell["wire"], tmp_path / "linked-by-a-later-merge.bin")
    assert _stamp(Path(cell["wire"]))["ctime_ns"] != cell["wire_stat"]["ctime_ns"]
    getattr(jce, "FENCE_REHASHED", {}).clear()
    _tool().fence_cell_artifacts(cell)
    assert getattr(jce, "FENCE_REHASHED", None) == {"assemble catalog wire fence": 1}


def test_assemble_refuses_a_same_size_wire_with_different_bytes(tmp_path):
    cell = _cell(tmp_path)
    Path(cell["wire"]).write_bytes(b"WIRE-BYTES")
    with pytest.raises(Exception):
        _tool().fence_cell_artifacts(cell)
