"""One exact token recipe and three consumer retentions (PQ #2638).

The independent byte-contract goldens qualify the sole added owner recipe
and the retained chunked feeds. The scoped refresh and retention decisions
live under refresh_2540.verification.tensor_leaf_2638 in the digest census.
"""
from __future__ import annotations

import torch

INT32_TOKEN_STREAM_GOLDEN = "c72dd2e22cfe4b7a3023394a011364af88315aa6e10c0f8b4c955f8e17ae1c4c"
INT32_TOKEN_STREAM_OTHER = "3c359f809f17cb5781b9deb2d6c8008420d074640f27720ac1a62fdbacee0231"
DEVICE_CHUNKS_GOLDEN = "24ae2dfe8df57c1b80e54cef3d90ac3b417fd98973345a5f616bbc9a75dcc202"
VIEW_STREAM_GOLDEN = "ea2ea5050002b3fafb9c454c4b55fd0c5ca6ff82cbbd215d487f8d765f0681c6"
VIEW_STREAM_EMPTY_GOLDEN = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FP32_STREAM_GOLDEN = "1734fdd19dfd458a6e40bbfb030be6e84e4b0614b0cca857a6e8b6c593f5fb1a"
FP32_STREAM_LIST_GOLDEN = "b9c80b5adeca450753a16950c3cc655d271f7bef7a485bc83f112b72fef21d37"


def _token_batches():
    return [
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64),
        torch.tensor([[8, 9], [10, 11]], dtype=torch.int64),
        torch.tensor([12], dtype=torch.int32),
    ]


def test_the_int32_token_stream_matches_its_golden():
    """The multi-batch int32 fold keeps its byte order and cast."""
    from prismaquant.tensor_digests import token_ids_int32_sha256

    assert token_ids_int32_sha256(_token_batches()) == INT32_TOKEN_STREAM_GOLDEN


def test_the_int32_token_stream_tells_draws_apart():
    """A different id draw gives a different digest."""
    from prismaquant.tensor_digests import token_ids_int32_sha256

    other = [
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64),
        torch.tensor([[8, 9], [10, 11]], dtype=torch.int64),
        torch.tensor([12], dtype=torch.int32),
    ]
    assert token_ids_int32_sha256(other) == INT32_TOKEN_STREAM_OTHER
    assert token_ids_int32_sha256(other) != INT32_TOKEN_STREAM_GOLDEN


def test_the_hessian_wrapper_delegates_to_the_owner(monkeypatch):
    """The consumer passes the original batches without conversion or copies."""
    from prismaquant import tensor_digests, tessera_hessian

    batches = _token_batches()
    assert tessera_hessian.token_ids_sha256(batches) == INT32_TOKEN_STREAM_GOLDEN
    calls = []

    def owner(value):
        calls.append(value)
        return "owner-result"

    monkeypatch.setattr(tensor_digests, "token_ids_int32_sha256", owner)
    assert tessera_hessian.token_ids_sha256(batches) == "owner-result"
    assert len(calls) == 1 and calls[0] is batches


def test_the_chunked_payload_matches_its_golden_at_any_chunk_width():
    """Chunked feeds equal the whole-bytes digest at every width."""
    from prismaquant.stage_a_chain_seed import tensor_payload_sha256

    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    assert tensor_payload_sha256(tensor) == DEVICE_CHUNKS_GOLDEN
    assert tensor_payload_sha256(tensor, chunk_bytes=8) == DEVICE_CHUNKS_GOLDEN


def test_the_view_stream_identity_matches_its_golden():
    """The chunked memoryview feed keeps its schema and content."""
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    tensor = torch.tensor([[1.0, -2.0], [3.5, 0.25]], dtype=torch.float32)
    assert _cb_cache_tensor_identity(tensor) == {
        "shape": [2, 2],
        "dtype": "torch.float32",
        "logical_bytes": 16,
        "content_sha256": VIEW_STREAM_GOLDEN,
    }


def test_the_view_stream_identity_covers_the_empty_tensor():
    """An empty tensor feeds zero bytes and keeps its shape."""
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    identity = _cb_cache_tensor_identity(torch.empty(0))
    assert identity["shape"] == [0]
    assert identity["logical_bytes"] == 0
    assert identity["content_sha256"] == VIEW_STREAM_EMPTY_GOLDEN


def test_the_fp32_stream_matches_its_golden():
    """The C-order fp32 conversion keeps its bytes and shape."""
    from prismaquant.production_weight_cache import _source_weight_value_identity

    weight = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)
    assert _source_weight_value_identity(weight) == ([2, 2], FP32_STREAM_GOLDEN)
    assert _source_weight_value_identity([1.0, 2.0]) == ([2], FP32_STREAM_LIST_GOLDEN)


def test_the_scoped_manifest_records_all_tensor_decisions():
    """The refreshed list must not silently turn a retention into a route."""
    import ast
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    census = json.loads((root / "docs/audits/digest_site_census_pq1301_2026-10-04.json").read_text())
    refresh = census["refresh_2540"]
    scoped = refresh["verification"]["tensor_leaf_2638"]
    decisions = {site["id"]: site for site in scoped["sites"]}
    assert len(decisions) == 8  # four tensor sites, four capture framing calls
    rows = {row["id"]: row for row in refresh["calls"]}
    for call_id, decision in decisions.items():
        assert decision["manifest"] == "refresh_2540"
        row = rows[call_id]
        if decision["decision"].startswith("retain"):
            assert row["disposition"] == "retain" and not row["eligible"]
            assert row["cohort"] is None and row["retention_evidence"]
            source = (root / row["file"]).read_text()
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == row["scope"])
            actual = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
            expected = scoped["sources"][row["file"]]["functions"][row["scope"]]["sha256"]
            assert actual == expected  # unchanged bounded feed and seal grammar
        else:
            assert row["scope"] == "token_ids_sha256"
            assert decision["decision"] == "route:prismaquant.tensor_digests.token_ids_int32_sha256"

    owner = ast.parse((root / "prismaquant/tensor_digests.py").read_text())
    assert {n.name for n in owner.body if isinstance(n, ast.FunctionDef)} == {
        "_host", "_raw", "tensor_host_bytes", "tensor_identity",
        "tensor_value_stamp", "tensor_sha256", "tensor_hash_update", "token_ids_int32_sha256",
    }
