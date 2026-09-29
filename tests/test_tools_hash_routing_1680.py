"""Tools hash one-liners route through the digests owners (issue #1680).

RED: each test patches the digests owner with a recorder and calls the tool
helper. Before delegation the helpers hash inline, so the recorder is never
consulted and the test fails with the unrouted digest observed. After
delegation the recorder sees every call. Chunk sizes differ across sites but
chunking never changes a hash value, so the values test holds on both sides.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAIR = _load("route_pair_1680", "tools/band_serial_handoff_live_pair.py")
BUDGET = _load("route_budget_1680", "tools/derive_retained_window_budget.py")
SPLIT = _load("route_split_1680", "tools/dispatch_stage_a_split.py")

from prismaquant import digests as owners  # noqa: E402


def test_pair_sha_routes_through_owner(monkeypatch):
    seen = []
    monkeypatch.setattr(owners, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "routed")
    assert PAIR._sha(b"abc") == "routed"
    assert seen == [b"abc"]


def test_budget_sha256_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "b.bin"
    target.write_bytes(b"abc" * 100)
    seen = []
    monkeypatch.setattr(owners, "file_sha256hex",
                        lambda path, **k: seen.append(Path(path).name) or "routed")
    assert BUDGET._sha256(target) == "routed"
    assert seen == ["b.bin"]


def test_split_sha_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "s.bin"
    target.write_bytes(b"abc" * 100)
    seen = []
    monkeypatch.setattr(owners, "file_sha256hex",
                        lambda path, **k: seen.append(Path(path).name) or "routed")
    assert SPLIT._sha(target) == "routed"
    assert seen == ["s.bin"]


def test_values_match_hashlib(tmp_path):
    blob = b"abc" * 100
    target = tmp_path / "v.bin"
    target.write_bytes(blob)
    want = hashlib.sha256(blob).hexdigest()
    assert PAIR._sha(blob) == want
    assert BUDGET._sha256(target) == want
    assert SPLIT._sha(target) == want
