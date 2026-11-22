from __future__ import annotations

from prismaquant.allocator import (
    _build_bit_attribution,
    _parse_role_from_qname,
    _write_bit_attribution_reports,
)
from prismaquant.allocator_solver import Candidate


def test_parse_role_distinguishes_dense_attn_mlp():
    assert _parse_role_from_qname("model.layers.3.self_attn.q_proj") == "q_proj"
    assert _parse_role_from_qname("model.layers.3.self_attn.o_proj") == "o_proj"
    assert _parse_role_from_qname("model.layers.3.mlp.gate_proj") == "gate_proj"
    assert _parse_role_from_qname("model.layers.3.mlp.down_proj") == "down_proj"


def test_parse_role_distinguishes_experts_from_dense():
    # Unpacked routed expert keeps an expert.* prefix so it never collapses
    # into the dense MLP bucket.
    assert _parse_role_from_qname(
        "model.layers.3.mlp.experts.7.down_proj") == "expert.down_proj"
    # Packed routed expert (3D param, no per-expert index).
    assert _parse_role_from_qname(
        "model.layers.3.mlp.experts.gate_up_proj") == "expert.gate_up_proj"
    # Shared expert.
    assert _parse_role_from_qname(
        "model.layers.3.mlp.shared_experts.up_proj") == "shared_expert.up_proj"


def test_parse_role_unknown_fallback():
    assert _parse_role_from_qname("some.weird.module") == "unknown"
    assert _parse_role_from_qname("lm_head") == "lm_head"


def test_build_bit_attribution_buckets_and_body_totals():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
        "model.layers.1.mlp.down_proj": "BF16",
        # Auxiliary entries must be excluded from the body buckets/totals.
        "mtp.0.self_attn.q_proj": "BF16",
        "model.visual.blocks.0.attn.qkv": "BF16",
    }
    candidates = {
        "model.layers.0.self_attn.q_proj": [Candidate("NVFP4", 4.0, 512, 0.01)],
        "model.layers.0.mlp.down_proj": [Candidate("FP8_E4M3", 8.0, 2048, 0.002)],
        "model.layers.1.mlp.down_proj": [Candidate("BF16", 16.0, 4096, 0.0)],
    }
    stats = {
        "model.layers.0.self_attn.q_proj": {"n_params": 1024},
        "model.layers.0.mlp.down_proj": {"n_params": 2048},
        "model.layers.1.mlp.down_proj": {"n_params": 2048},
    }

    buckets, per_linear, totals = _build_bit_attribution(
        assignment, candidates, stats.get, format_specs={})

    # mtp + visual excluded.
    assert totals["n_body_linears"] == 3
    qnames = {r["qname"] for r in per_linear}
    assert not any("mtp." in q or "visual" in q for q in qnames)

    # Bits include the exact serialized tensor payload plus format-global
    # sidecars. NVFP4 emits two fp32 scale scalars per 2-D Linear (8 B).
    q = next(r for r in per_linear if r["role"] == "q_proj")
    assert q["bits"] == 8.0 * (512 + 8)
    assert q["bpp"] == (8.0 * (512 + 8)) / 1024
    assert q["predicted_dloss"] == 0.01
    assert "h_trace" not in q

    # one bucket per (block, role); down_proj appears in two different blocks.
    keys = {(b["block_id"], b["role"]) for b in buckets}
    assert ("model.layers.0", "down_proj") in keys
    assert ("model.layers.1", "down_proj") in keys

    # body totals only count the 3 body linears.
    assert totals["body_quantizable_params"] == 1024 + 2048 + 2048


def test_build_bit_attribution_null_dloss_when_no_candidate():
    # Expanded fused-sibling member: in the assignment but no scored candidate.
    assignment = {"model.layers.0.self_attn.k_proj": "NVFP4"}
    stats = {"model.layers.0.self_attn.k_proj": {
        "n_params": 1024,
        "_memory_bytes_by_format": {"NVFP4": 512},
    }}
    buckets, per_linear, totals = _build_bit_attribution(
        assignment, candidates={}, stats_entry_for=stats.get, format_specs={})
    row = per_linear[0]
    # No fabricated dloss; bits still recovered from the memory map.
    assert row["predicted_dloss"] is None
    assert row["bits"] == 8.0 * (512 + 8)
    assert buckets[0]["sum_predicted_dloss"] is None
    assert buckets[0]["predicted_dloss_coverage"] == "0/1"


def test_write_bit_attribution_reports_emit_files(tmp_path):
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
    }
    candidates = {
        "model.layers.0.self_attn.q_proj": [Candidate("NVFP4", 4.0, 512, 0.01)],
        "model.layers.0.mlp.down_proj": [Candidate("FP8_E4M3", 8.0, 2048, 0.002)],
    }
    stats = {
        "model.layers.0.self_attn.q_proj": {"n_params": 1024},
        "model.layers.0.mlp.down_proj": {"n_params": 2048},
    }
    jpath = tmp_path / "attr.json"
    cpath = tmp_path / "attr.csv"
    exact_bpp = (8.0 * (512 + 8) + 8.0 * 2048) / (1024 + 2048)
    _write_bit_attribution_reports(
        str(jpath), str(cpath),
        target_bits=5.0, achieved_bits=exact_bpp,
        assignment_expanded=assignment, candidates=candidates,
        stats_entry_for=stats.get, format_specs={})

    assert jpath.read_bytes() == b'''{
  "schema": "prismaquant.allocator.bit_attribution.v2",
  "target_bits": 5.0,
  "achieved_bits": 6.6875,
  "body_bits_per_param": 6.6875,
  "body_assignment_payload_bits": 20544.0,
  "body_tensor_payload_bits": 20544.0,
  "body_quantizable_params": 3072,
  "n_body_linears": 2,
  "buckets": [
    {
      "block_id": "model.layers.0",
      "role": "down_proj",
      "n_linears": 1,
      "bits_total": 16384.0,
      "n_params_total": 2048,
      "mean_bpp": 8.0,
      "format_counts": {
        "FP8_E4M3": 1
      },
      "sum_predicted_dloss": 0.002,
      "predicted_dloss_coverage": "1/1"
    },
    {
      "block_id": "model.layers.0",
      "role": "q_proj",
      "n_linears": 1,
      "bits_total": 4160.0,
      "n_params_total": 1024,
      "mean_bpp": 4.0625,
      "format_counts": {
        "NVFP4": 1
      },
      "sum_predicted_dloss": 0.01,
      "predicted_dloss_coverage": "1/1"
    }
  ]
}
'''
    assert cpath.read_bytes() == (
        b"qname,block_id,role,format,bits,bpp,n_params,predicted_dloss\r\n"
        b"model.layers.0.mlp.down_proj,model.layers.0,down_proj,FP8_E4M3,16384,8,2048,0.002\r\n"
        b"model.layers.0.self_attn.q_proj,model.layers.0,q_proj,NVFP4,4160,4.0625,1024,0.01\r\n"
    )


def test_write_bit_attribution_noop_when_no_paths():
    # Should not raise and should write nothing when both paths are None.
    _write_bit_attribution_reports(
        None, None, target_bits=5.0, achieved_bits=4.9,
        assignment_expanded={}, candidates={}, stats_entry_for=lambda n: None,
        format_specs={})
