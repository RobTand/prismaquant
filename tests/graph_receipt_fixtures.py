"""Synthetic producer records for boundary tests, never serving qualification."""
from __future__ import annotations

import json

from prismaquant.digests import bytes_sha256hex, file_sha256hex
from prismaquant.shipcard import compute_model_sha
from prismaquant.shipcard_slots import (
    GRAPH_OBSERVATION_SCHEMA,
    GRAPH_SCOPE_FIELDS,
    make_record,
)


def graph_fixture(model, *, owner=None, tensor_parallel_size=2, model_sha=None):
    """A complete synthetic equality population and its producer observations."""
    scope = {
        "image": "registry/fixture@sha256:" + "1" * 64,
        "model_config_sha256": file_sha256hex(model / "config.json"),
        "tessera_src_sha256": "3" * 64,
        "compilation_config": {"mode": "FULL_DECODE_ONLY"},
        "speculative_tokens": 0, "max_model_len": 8448,
        "max_num_seqs": 4, "tensor_parallel_size": tensor_parallel_size,
        "fabric": "none" if tensor_parallel_size == 1 else "socket",
    }
    resolved = {"compilation_config": dict(scope["compilation_config"]),
                "custom_ops": ["all"], "ir_op_priority": {}}
    choices = owner.CHOICES_PER_PASS if owner is not None else 48
    graph = {
        "name": "fixture-graph", "pb_action": "fixture-equality-action",
        **{key: scope[key] for key in (
            "compilation_config", "speculative_tokens", "max_model_len",
            "max_num_seqs", "tensor_parallel_size")},
        "resolved": {key: resolved[key] for key in ("custom_ops", "ir_op_priority")},
        "graph": {"managers": {"full": {
            "captured_sizes": [1, 2], "replayed_sizes": {"1": 1, "2": 1}}},
            "classes": {"captured": {"full": {"2048": {}}},
                        "replays": {"full|2048": 1}}},
        "passes": [{"name": name, "members": choices, "choices": choices}
                   for name in ("fixture-graph", "fixture-graph-r2")],
    }
    receipt = {
        "schema": "tessera.graph_equals_eager.v2",
        "runtime": {"image": scope["image"], "fabric": scope["fabric"]},
        "model": {"config_sha256": scope["model_config_sha256"]},
        "tessera": {"src_sha256": scope["tessera_src_sha256"]},
        "equality_set": {"choices_per_pass": choices},
        "eager_pool": [{"name": "fixture-eager", "passes": [
            {"name": name, "members": choices, "choices": choices}
            for name in ("fixture-eager", "fixture-eager-r2")]}],
        "arms": [graph], "verdict": "equal",
    }
    if owner is not None:
        receipt = owner.finish(receipt)
    path = model.parent / f"{model.name}-graph-equality.json"
    raw = json.dumps(receipt).encode()
    path.write_bytes(raw)
    digest = bytes_sha256hex(raw)
    model_sha = compute_model_sha(model) if model_sha is None else model_sha
    observed = {
        "schema": GRAPH_OBSERVATION_SCHEMA,
        "model_sha_before": model_sha, "model_sha_after": model_sha,
        "graph_receipt_sha256": digest,
        "compilation_config": dict(scope["compilation_config"]),
        "resolved": resolved,
        "ranks": [{"rank": rank, "image": scope["image"], "fabric": scope["fabric"]}
                  for rank in range(tensor_parallel_size)],
        "generation": {"prompt": "fixture prompt", "outputs": ["fixture output"]},
        "equality_arm": graph["name"], "equality_action": graph["pb_action"],
    }
    record = make_record(
        slot="native_export.graph", tool="fixture:equality-producer",
        passed=True, model_sha=model_sha, git_commit="a" * 40,
        spec_decode_detected=False,
        metrics={"arm": "graph", "enforce_eager": False,
                 "generated_chars": len("fixture output"), "max_new_tokens": 16,
                 "graph_receipt_path": str(path.resolve()),
                 "graph_receipt_sha256": digest, "serve_scope": scope,
                 "graph_observation": observed},
    )
    assert set(scope) == set(GRAPH_SCOPE_FIELDS)
    return record, receipt


def replace_receipt(record, receipt):
    """Update the producer-bound test payload after a deliberate mutation."""
    from pathlib import Path

    raw = json.dumps(receipt).encode()
    Path(record["metrics"]["graph_receipt_path"]).write_bytes(raw)
    digest = bytes_sha256hex(raw)
    record["metrics"]["graph_receipt_sha256"] = digest
    record["metrics"]["graph_observation"]["graph_receipt_sha256"] = digest
