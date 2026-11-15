"""The live strict-JSON hooks refuse duplicate members with their own errors."""

from __future__ import annotations

import importlib.util
import json
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


FINGERPRINT = _load("keep_fingerprint_1660", "tools/serve_fingerprint.py")
SNAPSHOT = _load("keep_snapshot_1660", "tools/prismaquant_runtime_snapshot.py")
IDENTITY = _load("keep_identity_1660", "prismaquant/container_runtime_identity.py")


DUP = '{"a": 1, "b": 2, "a": 3}'
OK = '{"a": 1, "b": 2}'
NAN = '{"a": NaN}'


def test_pin_hook_refuses_duplicates():
    with pytest.raises(ValueError, match=r"serving pin repeats JSON key 'a'"):
        json.loads(DUP, object_pairs_hook=FINGERPRINT._reject_duplicate_pin_keys)
    assert json.loads(OK, object_pairs_hook=FINGERPRINT._reject_duplicate_pin_keys) == {"a": 1, "b": 2}


def test_models_hook_refuses_duplicates():
    with pytest.raises(ValueError, match=r"models response repeats JSON key 'a'"):
        json.loads(DUP, object_pairs_hook=FINGERPRINT._reject_duplicate_json_keys)


def test_snapshot_hook_refuses_duplicates():
    with pytest.raises(SNAPSHOT.SnapshotError, match=r"duplicate manifest member 'a'"):
        json.loads(DUP, object_pairs_hook=SNAPSHOT._reject_duplicate_members)


def test_identity_hook_refuses_duplicates():
    with pytest.raises(IDENTITY.RuntimeIdentityError, match=r"duplicate JSON member 'a'"):
        json.loads(DUP, object_pairs_hook=IDENTITY._reject_duplicate_members)
