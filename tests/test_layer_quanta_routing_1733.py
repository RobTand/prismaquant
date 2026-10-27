"""PQ #1733: layer-quanta hashes delegate to digests owners.

Routing: verify_quanta_coverage must call bytes_sha256hex once for the
first-campaign digest. layer_quanta + bind_quantum_boundary_readset +
bind_quantum_executable need full campaign plans / sealed record+receipt
derivations; they share the same one-line bytes-hash recipe and their rows
are pinned by the baseline tripwire. Values: byte-identical to hashlib.
"""
import hashlib

from prismaquant import joint_campaign_ledger as ledger

import prismaquant.joint_layer_quanta as quanta
from prismaquant.digests import bytes_sha256hex
from tests.delegation_recording import record_unary_calls


def _parent():
    return {
        "entries": [{"bytes": 10}],
        "annotations": {
            "phases": [{"name": "p0", "cumulative_bytes": 10}],
            "layers": [0],
        },
        "entry_count": 1,
        "total_bytes": 10,
    }


def _record(campaign):
    return {
        "schema": ledger.LAYER_QUANTUM_SCHEMA,
        "quantum_id": "layer-000",
        "layer": 0,
        "campaign": campaign,
        "read_set": {
            "source_phase": {"name": "p0", "start_bytes": 0, "end_bytes": 10},
            "entry_count": 1,
            "total_bytes": 10,
        },
        "chunks": [{"name": "c0", "start_bytes": 0, "end_bytes": 10}],
        "windows": [],
    }


def test_layer_quanta_hash_routes_to_owner(monkeypatch):
    calls = {
        "hash": record_unary_calls(monkeypatch, quanta, "bytes_sha256hex"),
    }
    campaign = {"plan": "p", "n": 1}
    proof = quanta.verify_quanta_coverage([_record(campaign)], _parent())
    assert proof["campaign_sha256"] == hashlib.sha256(
        quanta.canonical_bytes(campaign)).hexdigest()
    assert len(calls["hash"]) == 1


def test_layer_quanta_values_match_verbatim_spelling():
    raw = b"\x00\x01layer-quanta"
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
