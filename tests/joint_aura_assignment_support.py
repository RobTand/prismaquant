"""Shared synthetic inputs for joint aura assignment diagnostics tests."""
from __future__ import annotations

import copy

import torch

from prismaquant import joint_aura_rows
from prismaquant import format_registry as fr
from prismaquant import joint_aura as joint
from prismaquant.streamed_model_identity import STREAMED_MODEL_IDENTITY_SCHEMA


UNIT_A = "model.layers.0.mlp.down_proj"
UNIT_B = "model.layers.7.mlp.down_proj"


def _row(name, signed, *, fmt="FP8_E4M3"):
    # The allocator currency fixture runs a model to build rows. These scalar
    # oracles instead use the same complete row builder and identity contract.
    source_content = {
        "config": {"fixture": "assignment diagnostics"},
        "weight_map": {"fixture.weight": "fixture.safetensors"},
        "shards": [{"path": "/fixture/fixture.safetensors", "size": 1,
                    "sha256": "a" * 64}],
    }
    source_model = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic",
        "resolved_commit": None, **source_content,
        "content_sha256": joint_aura_rows.joint_identity_sha256(source_content),
    }
    arithmetic = joint.arithmetic_identity(torch.float32)
    probe = {
        "schema": "prismaquant.joint_aura.probes.v2", "seed_base": 237000,
        "n_probes": len(signed), "calibration_sha256": "c" * 64,
        "producer_source_sha256": "d" * 64, "source_model": source_model,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "temperature": 1.0, "arithmetic": arithmetic,
    }
    operator = {
        "schema": "prismaquant.joint_aura.operator.v2", "qname": name,
        "format": fmt, "probe_identity_sha256": joint_aura_rows.joint_identity_sha256(probe),
        "source_weight": {
            "content_sha256": joint_aura_rows.joint_identity_sha256({"source": name}),
            "shape": [2, 2], "dtype": "torch.float32", "logical_bytes": 16,
        },
        "rendered_weight": {
            "content_sha256": joint_aura_rows.joint_identity_sha256({"render": name, "format": fmt}),
            "shape": [2, 2], "dtype": "torch.float32", "logical_bytes": 16,
        },
        "activation": joint.activation_identity(fr.get_format(fmt), {}, name),
        "arithmetic": arithmetic,
    }
    return joint_aura_rows.make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[dict(weight=float(x), activation=0.0, mixed=0.0,
                                total=float(x)) for x in signed],
    )


def _rebuild(row, *, operator_change=None, probe_change=None):
    """Re-sign changed identities so pairing, not a stale digest, must refuse."""
    operator = copy.deepcopy(row["joint_operator_identity"])
    probe = copy.deepcopy(row["probe_identity"])
    if operator_change:
        operator_change(operator)
    if probe_change:
        probe_change(probe)
    operator["probe_identity_sha256"] = joint_aura_rows.joint_identity_sha256(probe)
    operator["arithmetic"] = copy.deepcopy(probe["arithmetic"])
    return joint_aura_rows.make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=copy.deepcopy(row["signed_components_per_probe"]),
    )
