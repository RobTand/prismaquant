"""PQ #1737: stage-a-retirement bytes hashes delegate to digests owners.

Routing: _successor must call bytes_sha256hex for the pinned check, on both
the matching and the refusal path. _load_chain_state + _checkpoint_planes +
plan_retirement need sealed chain states / occupied checkpoint directories /
full run spaces; they share the same one-line bytes-hash recipe and their
rows are pinned by the baseline tripwire. Values: byte-identical to hashlib.
"""
import hashlib
import json

import prismaquant.stage_a_retirement as retire
from prismaquant.digests import bytes_sha256hex


def _spies(monkeypatch):
    calls = {"hash": []}
    real_hash = getattr(retire, "bytes_sha256hex", None)

    def stand_in_hash(value):
        calls["hash"].append(value)
        return real_hash(value) if real_hash is not None else None

    monkeypatch.setattr(retire, "bytes_sha256hex", stand_in_hash, raising=False)
    return calls


def _successor_record(identity, generation):
    return {
        "boundary_storage": {
            "session": {"generation": generation, "run_identity_sha256": identity}
        }
    }


def test_successor_pinned_check_routes_to_owner(monkeypatch, tmp_path):
    calls = _spies(monkeypatch)
    record = _successor_record("a" * 64, "7")
    target = tmp_path / "successor.json"
    raw = (json.dumps(record, sort_keys=True) + "\n").encode()
    target.write_bytes(raw)
    space = tmp_path / "space"
    space.mkdir()
    session = {"generation": "6", "run_identity_sha256": "b" * 64}
    out = retire._successor(target, hashlib.sha256(raw).hexdigest(),
                            space=space, session=session)
    assert out["sha256"] == hashlib.sha256(raw).hexdigest()
    try:
        retire._successor(target, "0" * 64, space=space, session=session)
    except retire.RetirementRefused:
        pass
    assert len(calls["hash"]) == 2


def test_stage_retirement_values_match_verbatim_spelling():
    raw = b"\x00\x01stage-retirement"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
