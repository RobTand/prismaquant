"""Qualify the one retained token recipe on current main (PQ #2638/#3012).

The byte-contract goldens and consumer-retention cases come from retained
commit a9fce4a588dd41ebffcb77756166c8394a21b378. Census decisions are deliberately
not asserted here: the retention-audit sibling owns that historical manifest.
Current integration must instead bind the owner source and shrink main's
primitive baseline by exactly the routed token scope.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import torch

INT32_TOKEN_STREAM_GOLDEN = "c72dd2e22cfe4b7a3023394a011364af88315aa6e10c0f8b4c955f8e17ae1c4c"
INT32_TOKEN_STREAM_OTHER = "3c359f809f17cb5781b9deb2d6c8008420d074640f27720ac1a62fdbacee0231"
DEVICE_CHUNKS_GOLDEN = "24ae2dfe8df57c1b80e54cef3d90ac3b417fd98973345a5f616bbc9a75dcc202"
VIEW_STREAM_GOLDEN = "ea2ea5050002b3fafb9c454c4b55fd0c5ca6ff82cbbd215d487f8d765f0681c6"
VIEW_STREAM_EMPTY_GOLDEN = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FP32_STREAM_GOLDEN = "1734fdd19dfd458a6e40bbfb030be6e84e4b0614b0cca857a6e8b6c593f5fb1a"
FP32_STREAM_LIST_GOLDEN = "b9c80b5adeca450753a16950c3cc655d271f7bef7a485bc83f112b72fef21d37"

REPO = Path(__file__).resolve().parents[1]
TOKEN_SCOPE = "hashlib:prismaquant/tessera_hessian.py::token_ids_sha256"
# Canonical JSON identity of supplied main 6d8288ad's baseline, with ONLY
# TOKEN_SCOPE removed (277 -> 276 primitive sites). No historical Git needed.
TOKEN_ONLY_BASELINE_SHA256 = "78d379bc79f16c8529736bdcc272a085ee6c2c21754d0d1887a5f73fd59249c9"
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))]


def _token_batches(device="cpu"):
    return [
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64, device=device),
        torch.tensor([[8, 9], [10, 11]], dtype=torch.int64, device=device),
        torch.tensor([12], dtype=torch.int32, device=device),
    ]


def _token_owner():
    from prismaquant import tensor_digests

    owner = getattr(tensor_digests, "token_ids_int32_sha256", None)
    assert callable(owner), "the sole tensor owner must expose the retained int32 token recipe"
    return owner


@pytest.mark.parametrize("device", DEVICES)
def test_the_int32_token_stream_matches_its_golden(device):
    """The multi-batch int32 fold keeps its byte order and cast."""
    assert _token_owner()(_token_batches(device)) == INT32_TOKEN_STREAM_GOLDEN


@pytest.mark.parametrize("device", DEVICES)
def test_the_int32_token_stream_tells_draws_apart(device):
    """A different id draw gives a different digest."""
    other = [
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64, device=device),
        torch.tensor([[8, 9], [10, 11]], dtype=torch.int64, device=device),
        torch.tensor([12], dtype=torch.int32, device=device),
    ]
    assert _token_owner()(other) == INT32_TOKEN_STREAM_OTHER
    assert _token_owner()(other) != INT32_TOKEN_STREAM_GOLDEN


@pytest.mark.parametrize("device", DEVICES)
def test_the_int32_token_stream_has_no_headers_or_batch_delimiters(device):
    owner = _token_owner()
    assert owner([]) == VIEW_STREAM_EMPTY_GOLDEN
    # Same logical stream, but noncontiguous storage, different shape and cuts.
    noncontiguous = torch.arange(26, dtype=torch.int64, device=device)[::2] // 2
    noncontiguous = torch.stack((noncontiguous, noncontiguous), dim=1)[:, 0]
    assert not noncontiguous.is_contiguous()
    assert owner([noncontiguous.reshape(13, 1)]) == INT32_TOKEN_STREAM_GOLDEN
    assert owner([torch.empty(0, dtype=torch.int64, device=device),
                  *_token_batches(device)]) == INT32_TOKEN_STREAM_GOLDEN


def test_the_hessian_wrapper_delegates_to_the_owner(monkeypatch):
    """The consumer passes the original batches without conversion or copies."""
    from prismaquant import tensor_digests, tessera_hessian

    _token_owner()
    batches = _token_batches()
    assert tessera_hessian.token_ids_sha256(batches) == INT32_TOKEN_STREAM_GOLDEN
    calls = []

    def owner(value):
        calls.append(value)
        return "owner-result"

    monkeypatch.setattr(tensor_digests, "token_ids_int32_sha256", owner)
    assert tessera_hessian.token_ids_sha256(batches) == "owner-result"
    assert len(calls) == 1 and calls[0] is batches


def test_only_the_retained_token_recipe_extends_the_owner_surface():
    owner = ast.parse((REPO / "prismaquant/tensor_digests.py").read_text())
    assert {n.name for n in owner.body if isinstance(n, ast.FunctionDef)} == {
        "_host", "_raw", "tensor_host_bytes", "tensor_identity",
        "tensor_value_stamp", "tensor_sha256", "tensor_hash_update", "token_ids_int32_sha256",
    }


def test_token_route_has_a_reviewed_current_owner_pin():
    """Check integration constants only; no dependence on census edits."""
    pins = runpy.run_path(str(REPO / "tests/test_guarded_read_digest_retention_2637.py"))
    rel = "prismaquant/tensor_digests.py"
    historical = (
        "5c364c5d7c8fadf78ad9917668e72e1262546278",
        "98406374a7859e9a1b9e490671432372ad5dbe84a7a58c2b3c34377125af2037",
    )
    assert pins["MANIFEST_SOURCE_BLOBS"][rel] == historical
    current = pins["CURRENT_SOURCE_BLOBS"][rel]
    assert current != historical, "admit the additive owner with a current pin, not a rewritten snapshot"
    raw = (REPO / rel).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    assert current == (blob, hashlib.sha256(raw).hexdigest()), "stale integration owner pin"


def test_token_route_removes_exactly_one_main_baseline_scope():
    baseline = json.loads((REPO / "tests/fixtures/duplication_baseline.json").read_text())
    assert TOKEN_SCOPE not in baseline["primitive_digest_sites"], "remove the routed consumer allowance"
    canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == TOKEN_ONLY_BASELINE_SHA256, (
        "regenerate from supplied main: only the token primitive scope may disappear; "
        "do not restore retained-branch allowances or change other baseline sections"
    )


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
