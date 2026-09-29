"""PQ #1731: stage-A capture hashes delegate to digests owners.

Routing: load_prefetch_override must call bytes_sha256hex;
_write_split_receipt must call indent2_json_file_bytes + bytes_sha256hex.
run_adjoint_capture_core + run_adjoint_capture share the same one-line
recipes but are full campaign runners; their rows are pinned by the
baseline tripwire. Values: byte-identical to the verbatim old spellings.
"""
import hashlib
import json

import pytest

import prismaquant.joint_cost_stage_a as stage_a
from prismaquant.digests import bytes_sha256hex, indent2_json_file_bytes
from prismaquant.stage_a_chain_split import PREP_RECEIPT_SCHEMA


def _spies(monkeypatch):
    calls = {"dumps": [], "hash": []}
    real_dumps = getattr(stage_a, "indent2_json_file_bytes", None)
    real_hash = getattr(stage_a, "bytes_sha256hex", None)

    def stand_in_dumps(value):
        calls["dumps"].append(value)
        return real_dumps(value) if real_dumps is not None else None

    def stand_in_hash(value):
        calls["hash"].append(value)
        return real_hash(value) if real_hash is not None else None

    monkeypatch.setattr(stage_a, "indent2_json_file_bytes", stand_in_dumps,
                        raising=False)
    monkeypatch.setattr(stage_a, "bytes_sha256hex", stand_in_hash, raising=False)
    return calls


def _override():
    return {
        "schema": stage_a.PREFETCH_OVERRIDE_INPUT_SCHEMA,
        "reason": "test override",
        "source_prefetch": {
            "max_cache_slots": 8, "prefetch_workers": 2,
            "prefetch_lookahead": 4, "cache_headroom_gb": 1.0,
            "prefetch_min_available_gb": 0.5,
            "require_prefetched_residency": True,
        },
    }


def test_stage_a_hashes_route_to_owners(monkeypatch, tmp_path):
    calls = _spies(monkeypatch)
    override_path = tmp_path / "override.json"
    override_path.write_bytes(json.dumps(_override()).encode())
    loaded = stage_a.load_prefetch_override(override_path)
    assert loaded["reason"] == "test override"
    assert loaded["sha256"] == hashlib.sha256(override_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        stage_a.load_prefetch_override(tmp_path / "missing.json")
    receipt = {"schema": PREP_RECEIPT_SCHEMA, "resume": {"index": 0}}
    out = stage_a._write_split_receipt(tmp_path / "space", receipt)
    from pathlib import Path
    assert out["receipt"]["sha256"] == hashlib.sha256(
        Path(out["receipt"]["path"]).read_bytes()).hexdigest()
    assert len(calls["hash"]) == 2
    assert len(calls["dumps"]) == 1


def test_stage_a_values_match_verbatim_spellings():
    raw = b"\x00\x01stage-a"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    document = {"b": [1, 2], "a": "x"}
    assert indent2_json_file_bytes(document) == (
        json.dumps(document, sort_keys=True, indent=2,
                   allow_nan=False) + "\n").encode()
