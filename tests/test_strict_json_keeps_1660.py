"""Strict-JSON keeps refuse duplicates with their own errors (issue #1660).

Slice 1 verified all six duplicate-key sites against the schemas owner.
``dsv4_wikitext_inputs`` already delegates to ``strict_json_loads`` (group 1);
the other five keep local hooks for documented standalone reasons (serving
container without the package, stdlib-only path-invoked helpers, standalone
torch-free snapshot load). This test pins each kept hook's exact refusal
through the real ``json.loads`` call, so a future dedup attempt knows the
contract it must preserve.
"""

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
IDENTITY = _load("keep_identity_1660", "tools/container_runtime_identity.py")

from prismaquant.dsv4_campaign_completion import (  # noqa: E402
    CampaignCompletionError,
    _reject_duplicate_members as completion_hook,
)

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


def test_completion_hook_refuses_duplicates():
    with pytest.raises(CampaignCompletionError, match=r"duplicate JSON member 'a'"):
        json.loads(DUP, object_pairs_hook=completion_hook)
    assert json.loads(OK, object_pairs_hook=completion_hook) == {"a": 1, "b": 2}


def test_completion_loader_end_to_end(tmp_path):
    from prismaquant.dsv4_campaign_completion import _load_json

    good = tmp_path / "good.json"
    good.write_text(OK)
    assert _load_json(good) == {"a": 1, "b": 2}
    bad = tmp_path / "dup.json"
    bad.write_text(DUP)
    with pytest.raises(CampaignCompletionError, match=r"duplicate JSON member 'a'"):
        _load_json(bad)
    nan = tmp_path / "nan.json"
    nan.write_text(NAN)
    with pytest.raises(CampaignCompletionError, match=r"non-finite JSON value"):
        _load_json(nan)
