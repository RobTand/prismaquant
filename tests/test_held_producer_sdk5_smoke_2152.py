"""Self-contained CPU SDK5 native-carrier contract integration smoke.

Synthetic immutable evidence is authenticated by the real installed SDK and
joined to Original controls. This is not real fleet/native-GPU qualification
or recovery of the historical producer's retained evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fleet_sdk import require_prismabuild_sdk
from native_carrier_fixture import native_carrier
from prismaquant import source_generation as sg
from prismaquant.staged_lease import PB_CLIENT_SDK_VERSION

# Placeholder source-tree seals intentionally disagree with the private helper
# tree. Matching identity must reach that named refusal in certified mode.
PRODUCER_RUNTIME = {
    "schema": "prismaquant.original_source_runtime.v1",
    "prismaquant_source_sha256": "0" * 64,
    "tessera_source_sha256": "0" * 64,
    "modeling_source": {"path": "/held/modeling.py", "sha256": "0" * 64},
    "model_class": "transformers.models.held.HeldModel",
    "profile": "held", "config": {},
    "versions": {"python": "held", "torch": "held", "torch_git": None,
                 "cuda": None, "transformers": "held"},
    "container_content_sha256": None,
    "arithmetic": {"matmul_precision": "highest", "allow_tf32": False,
                   "allow_bf16_reduced_precision_reduction": False},
    "material_pipeline": {"decoder": "safetensors.safe_open", "framework": "pt",
                          "decoder_device": "cpu",
                          "cast_owner": "prismaquant.layer_streaming",
                          "direct_gpu_decode": False, "target_dtype": "torch.bfloat16",
                          "tensor_dtypes": {}, "scale_inv_map": {}},
    "prismabuild": {"sdk_version": 5, "helper_root": "fixture",
                    "source_tree": {"package_sha256": "0" * 64,
                                    "helper_tree_sha256": "0" * 64},
                    "runtime_generation": "fixture"},
}
PRODUCER_RESOURCES = {
    "schema": "prismaquant.original_source_resources.v1",
    "cpu_bytes": 1, "material_bytes": 1, "source_cache_bytes": 1,
    "source_prefetch": {"max_cache_slots": 2, "prefetch_workers": 1,
                        "prefetch_lookahead": 1, "cache_headroom_gb": 1,
                        "prefetch_min_available_gb": 1,
                        "require_prefetched_residency": True},
    "copy_bytes": 1, "gpu_bytes": 0, "native_bytes": 0, "serialization_bytes": 1,
    "artifact_bytes": 1, "deadline_seconds": 60, "stall_seconds": 10,
    "host_floor_bytes": 1, "margin_bytes": 0,
    "claim_demand": {},
}


def _bound(tmp, name, doc):
    path = tmp / name
    path.write_text(json.dumps(doc, sort_keys=True))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _producer_authority(tmp, context):
    runtime = json.loads(json.dumps(PRODUCER_RUNTIME))
    runtime['prismabuild'].update(helper_root=context['helper_root'],
                                runtime_generation=Path(context['helper_root']).name)
    resources = dict(PRODUCER_RESOURCES, claim_demand=context['resources'])
    return {"runtime": _bound(tmp, "runtime.json", runtime),
            "resources": _bound(tmp, "resources.json", resources),
            "source_model_identity": {"config": runtime["config"]}}


def _claim_receipt(claim):
    return {"deliveries": [{"native_delivery": {"claim": dict(
        claim, attempt_source="launch-env", map_path="/held/producer/map")}}]}


def test_held_producer_context_authenticates_and_joins(
        installed_client_sdk, tmp_path, monkeypatch, capsys):
    """Authenticate one private immutable attempt, then exercise real joins."""
    require_prismabuild_sdk()
    assert installed_client_sdk.SDK_VERSION == PB_CLIENT_SDK_VERSION == 5
    queue, action, record, receipt, payload, expected = native_carrier(tmp_path, monkeypatch)
    result = installed_client_sdk.read_verified_action_result(
        queue, action['action_key'],
        published_unix=record['published_unix'], attempt=expected['attempt'],
        max_result_bytes=64 * 1024, max_evidence_bytes=1024 * 1024,
        require_native_producer_context=True)
    context = result["producer_context"]
    assert context["schema"] == "prismabuild.native_producer_context.v1"
    for key, value in expected.items():
        assert context[key] == value, key
    for key in ('action_key', 'attempt', 'published_unix', 'generation', 'host'):
        assert context[key] == result[key], key
    assert context['worker'] == context['incarnation'] == result['worker_id']
    assert result["payload"] == payload
    assert result["request"]["action_key"] == action['action_key']
    assert result["receipt"]["receipt_sha256"] == receipt['receipt_sha256']
    assert result["payload"].count(b"passed")
    # The actual executable wrapper of the private attempt, bound from its
    # sealed request bytes — not params.command asserted alone.
    installed_client_sdk.bind_standard_capture_command(result["request"])
    # The consumer delivery join accepts the genuine context identity and
    # then stops at the sealed helper-tree check in certified mode: the private
    # helper has real on-disk digests, which
    # cannot agree with this control's placeholders. That named refusal is the
    # boundary.
    claim = {key: context[key] for key in ("queue_root", "action_key", "nonce",
                                           "scope_id", "worker", "host",
                                           "incarnation", "helper_root")}
    authority = _producer_authority(tmp_path, context)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match="selected reader actual complete helper tree"):
        sg._require_original_reader_producer(
            _claim_receipt(claim), authority, result, authority["runtime"])
    # No later consumer or foreign producer may substitute one identity axis.
    for axis in ("nonce", "scope_id"):
        foreign = dict(claim)
        foreign[axis] = "f" * 32
        with pytest.raises(RuntimeError, match="actual reader delivery selected producer"):
            sg._require_original_reader_producer(
                _claim_receipt(foreign), authority, result, authority["runtime"])
    # CEO D32: dev mode hashes no helper tree to seal a run. The same genuine
    # context completes the join with one `not computed` stamp, and a foreign
    # producer axis is stamped instead of refused.
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    capsys.readouterr()
    assert sg._require_original_reader_producer(
        _claim_receipt(claim), authority, result, authority["runtime"]) is None
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 1
    assert "seal selected reader actual complete helper tree not computed" in out
    foreign = dict(claim, nonce="f" * 32)
    assert sg._require_original_reader_producer(
        _claim_receipt(foreign), authority, result, authority["runtime"]) is None
    out = capsys.readouterr().out
    assert "seal actual reader delivery selected producer differs at nonce" in out
    assert "seal selected reader actual complete helper tree not computed" in out
