"""Dispatch hashlib wrappers route through the digests owners (issue #1670).

RED: each test patches the digests owner with a recorder and calls the tool
wrapper. Before delegation the wrapper hashes inline, so the recorder is
never consulted and the test fails with the mutation (unrouted bytes)
observed. After delegation the recorder sees every call.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUANTA = _load("route_quanta_1670", "tools/dispatch_joint_quanta.py")
CAMPAIGN = _load("route_campaign_1670", "tools/dispatch_tessera_campaign.py")

from prismaquant import digests as owners  # noqa: E402


def test_sha_bytes_routes_through_owner(monkeypatch):
    seen = []
    monkeypatch.setattr(owners, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "routed")
    assert QUANTA._sha_bytes(b"abc") == "routed"
    assert seen == [b"abc"]


def test_sha256_of_routes_through_owner(monkeypatch, tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")
    seen = []
    monkeypatch.setattr(owners, "file_sha256hex",
                        lambda path, **k: seen.append(Path(path).name) or "routed")
    assert CAMPAIGN._sha256_of(target) == "routed"
    assert seen == ["f.bin"]


def test_argv_file_hash_routes_through_owner(monkeypatch):
    seen = []
    monkeypatch.setattr(owners, "bytes_sha256hex",
                        lambda data: seen.append(bytes(data)) or "deadbeef")
    import prismaquant.dev_mode as dev

    monkeypatch.setattr(dev, "dev_mode_enabled", lambda: True)
    compared = []
    monkeypatch.setattr(dev, "seal_check",
                        lambda *a, **k: compared.append(a) or None)
    campaign = {"plan_sha256": "deadbeef", "plan_path": "/nonexistent"}
    out = QUANTA._argv_file_sha256(campaign, "plan", where="t", raw=b"abc")
    assert out == "deadbeef"
    assert seen == [b"abc"]
    assert compared[0][1:] == ("deadbeef", "deadbeef")


def test_values_match_hashlib():
    assert QUANTA._sha_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
