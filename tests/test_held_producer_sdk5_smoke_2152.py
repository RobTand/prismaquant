"""Held-producer SDK5 smoke: the selected attempt's native context, joined.

The producer is the SAME held immutable selected attempt this campaign
accepted its 24-case CPU negative qualification from
(``d3966020c8e8...``); it is read, never rerun. This smoke proves the
installed SDK5 library the consumer pins authenticates that attempt's
native producer context through the public
``require_native_producer_context=True`` contract and that the consumer's
delivery join binds exactly that context — the positive half of the
PQ2152 caller contract. The full material-receipt join stays with the
native reader fixture; a captured-log producer carries no deliveries.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from fleet_sdk import require_prismabuild_sdk
from prismaquant import source_generation as sg
from prismaquant.staged_lease import PB_CLIENT_SDK_VERSION

#: The held selected producer attempt and its pre-validated identity
#: (terminal ``done`` record + CAS receipt, parent record
#: ``prismaquant-2152.json``). The selector is the decision; the expected
#: context fields bind the read to exactly this attempt, so a superseded,
#: replaced or foreign ending cannot masquerade as the held producer.
QUEUE_ROOT = "/mnt/shared/prismabuild-fleet/pb-queue"
ACTION_KEY = "d3966020c8e8b236ad858a0d4569f107448b8003373ca8d9da97dbbd745bdbc4"
PUBLISHED_UNIX = 1791085133.0468068
ATTEMPT = 1
PAYLOAD_SHA256 = "47e694d6f223c7856568a87da67556b2574a033740ca22750a949b044b3db0a4"
RECEIPT_SHA256 = "6f95cf0cddf50c98fef3f7cce18f53f682cad32d7076fddb747d05453cf1db30"
HELPER_ROOT = ("/mnt/shared/prismabuild-fleet/runtime-generations/"
               "c43700f9d1d0-1791018557-b6c96bc4bbbb")
EXPECTED_CONTEXT = {
    "action_key": ACTION_KEY,
    "attempt": ATTEMPT,
    "published_unix": PUBLISHED_UNIX,
    "nonce": "8a5c8c1d583c44bb8fb04e199308bbea",
    "scope_id": "prismabuild-jobbef70c91d03414e28172ecff45dccf33.slice",
    "host": "dl380g10",
    "worker": "dl380g10:1565588:692a5e47",
    "incarnation": "dl380g10:1565588:692a5e47",
    "helper_root": HELPER_ROOT,
    "resources": {"cpu": 1, "mem_gb": 4},
    "resources_semantics": "selected-claim-sealed-demand",
    "attempt_source": "selected-immutable-attempt",
}

#: The producer's own runtime/resources control documents, bound exactly as
#: the gate binds them. Their digests are placeholders here — this join ends
#: at the sealed helper-tree check, which hashes the REAL held helper root
#: on disk and cannot agree with placeholder digests; that named refusal is
#: the honest boundary (the fixture's material-receipt join carries real
#: digests and is covered by the native reader fixture tests).
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
    "prismabuild": {"sdk_version": 5, "helper_root": HELPER_ROOT,
                    "source_tree": {"package_sha256": "0" * 64,
                                    "helper_tree_sha256": "0" * 64},
                    "runtime_generation": Path(HELPER_ROOT).name},
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
    "claim_demand": EXPECTED_CONTEXT["resources"],
}


def _bound(tmp, name, doc):
    path = tmp / name
    path.write_text(json.dumps(doc, sort_keys=True))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _producer_authority(tmp):
    return {"runtime": _bound(tmp, "runtime.json", PRODUCER_RUNTIME),
            "resources": _bound(tmp, "resources.json", PRODUCER_RESOURCES),
            "source_model_identity": {"config": PRODUCER_RUNTIME["config"]}}


def _claim_receipt(claim):
    return {"deliveries": [{"native_delivery": {"claim": dict(
        claim, attempt_source="launch-env", map_path="/held/producer/map")}}]}


def test_held_producer_context_authenticates_and_joins(installed_client_sdk, tmp_path):
    """Read the held attempt through the installed SDK5; never rerun it.

    The test depends on the fleet queue retaining the held attempt's terminal
    record. A box with no fleet queue mounted skips by name. A mounted queue
    that no longer holds the attempt fails loudly: that is lost evidence, not
    a missing prerequisite.
    """
    require_prismabuild_sdk()
    if not Path(QUEUE_ROOT).is_dir():
        pytest.skip(f"fleet queue not mounted: {QUEUE_ROOT}")
    assert installed_client_sdk.SDK_VERSION == PB_CLIENT_SDK_VERSION == 5
    result = installed_client_sdk.read_verified_action_result(
        installed_client_sdk.PoolQueue(QUEUE_ROOT), ACTION_KEY,
        published_unix=PUBLISHED_UNIX, attempt=ATTEMPT,
        max_result_bytes=64 * 1024, max_evidence_bytes=1024 * 1024,
        require_native_producer_context=True)
    context = result["producer_context"]
    assert context["schema"] == "prismabuild.native_producer_context.v1"
    for key, expected in EXPECTED_CONTEXT.items():
        assert context[key] == expected, key
    assert result["payload"] and result["request"]["action_key"] == ACTION_KEY
    assert result["receipt"]["receipt_sha256"] == RECEIPT_SHA256
    assert Path(context["queue_root"]).resolve() == Path(QUEUE_ROOT).resolve()
    assert result["payload"].count(b"passed")
    # The actual executable wrapper of the held attempt, bound from its
    # sealed request bytes — not params.command asserted alone.
    installed_client_sdk.bind_standard_capture_command(result["request"])
    # The consumer delivery join accepts the genuine context identity and
    # then stops at the sealed helper-tree check: the held c437 helper is
    # not the installed SDK5 tree, and the real on-disk digests cannot agree
    # with this control's placeholders. That named refusal is the boundary.
    claim = {key: context[key] for key in ("queue_root", "action_key", "nonce",
                                           "scope_id", "worker", "host",
                                           "incarnation", "helper_root")}
    authority = _producer_authority(tmp_path)
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
