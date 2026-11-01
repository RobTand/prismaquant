"""Recorded artifacts retain their refusal, source, row and resume contracts."""
from __future__ import annotations
import hashlib
import json
import pickle
import subprocess
import sys
import numpy as np
import pytest
import torch
from prismaquant import sample_parallel_artifacts as artifacts
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.sample_parallel_artifacts import (
    ACTIVATION_PRIORITY_MAX_ROWS, activation_priority_key, activation_row_priorities,
    activation_row_priority_scalar, validate_activation_priority_domain,
    BODY_STATS_AND_ACTIVATION, BODY_TOKEN_ROWS, LM_HEAD_STATS_ONLY,
    LM_HEAD_TOKEN_ROWS, MTP_STATS_ONLY, MTP_TOKEN_ROWS, validate_qname_census,
)
BODY_QNAMES = ("model.layers.0.q_proj",)
RTX4090_QWEN38_POLICY_SCHEMA = "prismaquant.rtx4090_qwen38_fp8_policy.v1"
RTX4090_QWEN38_POLICY_ID = "qwen38_27b_rtx4090_fp8_cb"

def _digest(value, **_):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def _manifest(*, kind, source_census_sha256, entries):
    body = {"schema": artifacts.QNAME_MANIFEST_SCHEMA, "kind": kind,
            "source_census_sha256": source_census_sha256, "entries": entries}
    return {**body, "identity_sha256": _digest(body)}


def _qname_census(
    *,
    shard_sha256: str = "9" * 64,
    derivation: str = "test_v1",
    upstream_content_sha256: str | None = None,
    upstream_portable_content_sha256: str | None = None,
    model: str = "model-content-id",
    body_qnames: tuple[str, ...] = BODY_QNAMES,
):
    entries = {
        name: {
            "source_tensor": name + ".weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": BODY_STATS_AND_ACTIVATION,
            "token_rows_per_sample": BODY_TOKEN_ROWS,
            "terminal_format": None,
        }
        for name in body_qnames
    }
    entries.update({
        "lm_head": {
            "source_tensor": "lm_head.weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": LM_HEAD_STATS_ONLY,
            "token_rows_per_sample": LM_HEAD_TOKEN_ROWS,
            "terminal_format": "BF16",
        },
        "mtp.fc": {
            "source_tensor": "mtp.fc.weight",
            "source_dtype": "BF16", "shape": [2, 3],
            "disposition": MTP_STATS_ONLY,
            "token_rows_per_sample": MTP_TOKEN_ROWS,
            "terminal_format": "BF16",
        },
    })
    shard = {
        "name": "model.safetensors", "size": 1,
        "sha256": shard_sha256,
    }
    content_sha = _digest({
        "source_config_sha256": "1" * 64,
        "checkpoint_weight_map_sha256": "3" * 64,
        "shards": [shard],
    }, where="test model content")
    model_body = {
        "schema": artifacts.SOURCE_MODEL_CONTENT_SCHEMA,
        "derivation": derivation,
        "upstream_content_sha256": upstream_content_sha256,
        "upstream_portable_content_sha256": (
            upstream_portable_content_sha256
        ),
        "content_sha256": content_sha, "resolved_commit": None,
        "checkpoint_tensors": len(entries), "checkpoint_shards": 1,
        "checkpoint_weight_map_sha256": "3" * 64,
        "shards": [shard],
    }
    model_identity = {
        **model_body,
        "identity_sha256": _digest(
            model_body, where="test model identity"
        ),
    }
    source_body = {
        "schema": artifacts.SOURCE_QNAME_CENSUS_SCHEMA,
        "model": model,
        "producer_profile_schema": RTX4090_QWEN38_POLICY_SCHEMA,
        "producer_profile_id": RTX4090_QWEN38_POLICY_ID,
        "source_layout": "flattened_text",
        "source_config_sha256": "1" * 64,
        "source_tensor_manifest_sha256": "2" * 64,
        "source_weight_map_sha256": "3" * 64,
        "source_tensor_count": len(entries),
        "source_linear_count": len(entries),
        "source_model_identity": model_identity,
        "linear_entries": entries,
    }
    source = {
        **source_body,
        "identity_sha256": _digest(
            source_body, where="test source census"
        ),
    }
    source_sha = source["identity_sha256"]
    probe = _manifest(
        kind="full_probe", source_census_sha256=source_sha, entries=entries,
    )
    activation = _manifest(
        kind="dense_body_activation", source_census_sha256=source_sha,
        entries={name: entries[name] for name in body_qnames},
    )
    terminal = _manifest(
        kind="terminal_bf16_stats_only", source_census_sha256=source_sha,
        entries={name: entries[name] for name in ("lm_head", "mtp.fc")},
    )
    body = {
        "source_census": source,
        "probe_qname_manifest": probe,
        "activation_qname_manifest": activation,
        "terminal_qname_manifest": terminal,
    }
    return validate_qname_census({
        **body,
        "identity_sha256": _digest(
            body, where="test qname census"
        ),
    })

def _contract(index: int) -> dict[str, object]:
    start = index * 2
    local = torch.arange(start * 4, (start + 2) * 4).reshape(2, 4)
    return {
        "schema": "prismaquant.sample_parallel_probe.partition.v1",
        "global_calibration_hash": "a" * 32,
        "calibration_artifact_sha256": "b" * 64,
        "global_samples": 4,
        "seqlen": 4,
        "partition_count": 2,
        "partition_index": index,
        "sample_indices": list(range(start, start + 2)),
        "sample_start": start,
        "sample_stop": start + 2,
        "local_calibration_hash": calibration_data_hash(local),
        "local_samples": 2,
        "dataset": "cal.jsonl",
        "model": "toy",
        "calib_seed": 42,
    }

def _sample_precompute_fixture(tmp_path):
    from prismaquant import incremental_probe as ip
    from prismaquant.perturbed_x_cache import calibration_data_hash

    ids = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    contract = _contract(0)
    contract["local_calibration_hash"] = calibration_data_hash(ids)
    meta = {
        "sample_parallel": contract,
        "sample_parallel_qname_census": _qname_census(),
    }
    h_full = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32,
    )
    resident_stats = {
        "lm_head": {
            "h_trace_raw": 21.0,
            "h_w2_sum_raw": 7.0,
            "w_max_abs": 2.0,
            "w_norm_sq": 5.0,
            "n_params": 6,
            "in_features": 3,
            "out_features": 2,
            "n_tokens_seen": 6,
            "route_prob": None,
            "router_path": None,
            "expert_id": None,
            "fisher_row": np.array([6.0, 15.0], dtype=np.float32),
            "fisher_col": np.array([5.0, 7.0, 9.0], dtype=np.float32),
            "g_sq_sum": np.array([10.0, 11.0], dtype=np.float32),
            "act_sq_sum": np.array([8.0, 9.0, 10.0], dtype=np.float32),
            "act_absmax": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
    }
    precompute = ip.GlobalPrecompute(
        activations_cpu=[
            torch.zeros(2, 4, 3, dtype=torch.bfloat16),
            torch.ones(2, 4, 3, dtype=torch.bfloat16),
        ],
        grad_at_tail=torch.ones(2, 4, 3, dtype=torch.bfloat16),
        ids=ids,
        resident_stats=resident_stats,
        resident_h_full={"lm_head": h_full},
        resident_g2_per_token={
            "lm_head": torch.arange(1, 7, dtype=torch.float32),
        },
        resident_act_snaps={},
        resident_act_row_indices={},
        expert_info={},
        router_counts={},
        router_totals={},
        router_active_counts={},
        expert_route_stats={},
        shared_pass_state=None,
    )
    path = tmp_path / "precomputed.pt"
    ip._save_precompute_cache(path, precompute, meta)
    return path, meta, ids


def _mutate_sample_precompute_payload(payload, mutation):
    if mutation == "ids":
        payload["ids_cpu"] = payload["ids_cpu"].flip(1)
    elif mutation == "routed_map":
        payload["router_counts"] = {"router": {"0": 1.0}}
    elif mutation == "minimal_stat":
        payload["resident_stats"]["lm_head"] = {"h_trace_raw": 21.0}
    elif mutation == "missing_map_entry":
        payload["resident_h_full"].clear()
    elif mutation == "extra_map_entry":
        payload["resident_g2_per_token"]["mtp.fc"] = torch.ones(
            6, dtype=torch.float32
        )
    elif mutation == "h_dtype":
        payload["resident_h_full"]["lm_head"] = payload[
            "resident_h_full"
        ]["lm_head"].to(torch.float64)
    elif mutation == "h_shape":
        payload["resident_h_full"]["lm_head"] = torch.ones(
            3, 2, dtype=torch.float32
        )
    elif mutation == "h_nonfinite":
        payload["resident_h_full"]["lm_head"][0, 0] = float("nan")
    elif mutation == "g2_dtype":
        payload["resident_g2_per_token"]["lm_head"] = payload[
            "resident_g2_per_token"
        ]["lm_head"].to(torch.float64)
    elif mutation == "g2_shape":
        payload["resident_g2_per_token"]["lm_head"] = torch.ones(
            5, dtype=torch.float32
        )
    elif mutation == "g2_nonfinite":
        payload["resident_g2_per_token"]["lm_head"][0] = float("inf")
    elif mutation == "marginal_dtype":
        payload["resident_stats"]["lm_head"]["fisher_row"] = np.array(
            [6.0, 15.0], dtype=np.float64
        )
    elif mutation == "marginal_shape":
        payload["resident_stats"]["lm_head"]["fisher_col"] = np.ones(
            2, dtype=np.float32
        )
    elif mutation == "marginal_nonfinite":
        payload["resident_stats"]["lm_head"]["g_sq_sum"][0] = np.nan
    elif mutation == "trace_inconsistent":
        payload["resident_stats"]["lm_head"]["fisher_row"][0] += 1.0
    elif mutation == "token_type":
        payload["resident_stats"]["lm_head"]["n_tokens_seen"] = True
    elif mutation == "resident_activation":
        payload["resident_act_snaps"]["lm_head"] = [torch.ones(1, 3)]
    elif mutation == "shared_state":
        payload["shared_pass_state"] = {}
    else:  # pragma: no cover - keeps the table closed
        raise AssertionError(mutation)

def test_sample_precompute_cache_valid_resident_payload_round_trip(tmp_path):
    from prismaquant import incremental_probe as ip

    path, meta, ids = _sample_precompute_fixture(tmp_path)
    loaded = ip._load_precompute_cache(
        path, meta, torch.device("cpu"), sample_calibration=ids,
    )

    assert loaded is not None
    assert set(loaded.resident_stats) == {"lm_head"}
    assert loaded.resident_stats["lm_head"]["n_tokens_seen"] == 6
    assert loaded.resident_h_full["lm_head"].dtype == torch.float32
    assert loaded.resident_g2_per_token["lm_head"].shape == (6,)


@pytest.mark.parametrize("mutation", [
    "ids",
    "routed_map",
    "minimal_stat",
    "missing_map_entry",
    "extra_map_entry",
    "h_dtype",
    "h_shape",
    "h_nonfinite",
    "g2_dtype",
    "g2_shape",
    "g2_nonfinite",
    "marginal_dtype",
    "marginal_shape",
    "marginal_nonfinite",
    "trace_inconsistent",
    "token_type",
    "resident_activation",
    "shared_state",
])
def test_sample_precompute_cache_refuses_malformed_resident_payload(
    tmp_path, mutation,
):
    from prismaquant import incremental_probe as ip

    path, meta, ids = _sample_precompute_fixture(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _mutate_sample_precompute_payload(payload, mutation)
    torch.save(payload, path)

    assert ip._load_precompute_cache(
        path, meta, torch.device("cpu"), sample_calibration=ids,
    ) is None

def test_the_sample_parallel_worker_entry_refuses_the_retired_mode():
    """The mode is a recorded capability LOSS, and the refusal is the record.

    Until 2026-09-02 this asserted the argparse pairing message
    ("--sample-cover is required exactly"), which is a check on how to invoke
    a mode that still ran. The mode does not run: its run-contract minter and
    per-worker source census were built on `rtx4090_artifact_census`, archived
    with the Gridbook lane (archive/gridbook_lane_2026-09-02/), so nothing can
    mint a contract and a pre-retirement one on disk would be admitted with
    one leg of its identity replay missing.

    So the assertion moves to the refusal itself. It is checked through the
    real CLI, with a model path that does not exist, because that is the
    property that matters: the refusal must not depend on reaching a
    checkpoint. Deleting this test rather than re-pointing it would leave the
    loss recorded only in prose.
    """
    from prismaquant.incremental_probe import SAMPLE_PARALLEL_RETIRED

    command = [
        sys.executable, "-m", "prismaquant.incremental_probe",
        "--model", "no/such/model",
        "--global-calibration-tensor", "calibration.pt",
        "--sample-partition-index", "0",
        "--sample-run-contract", "run-contract.json",
        "--output", "probe.pkl", "--activation-cache-dir", "act",
        "--work-dir", "work",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert SAMPLE_PARALLEL_RETIRED in completed.stderr
    assert "archive/gridbook_lane_2026-09-02/" in completed.stderr
    # ...and it refuses even when every companion flag the old pairing checks
    # demanded is absent, i.e. the retirement gate is not one of those checks.
    minimal = subprocess.run(
        [sys.executable, "-m", "prismaquant.incremental_probe",
         "--model", "no/such/model",
         "--global-calibration-tensor", "calibration.pt",
         "--output", "probe.pkl", "--activation-cache-dir", "act",
         "--work-dir", "work"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert SAMPLE_PARALLEL_RETIRED in minimal.stderr


def test_activation_priority_known_vectors_and_uint32_boundaries():
    qname = "model.layers.0.q_proj"
    calibration_hash = "a" * 32
    rows = [
        0, 1, 65535, 65536, (1 << 31) - 1, 1 << 31,
        ACTIVATION_PRIORITY_MAX_ROWS - 1,
    ]
    assert activation_priority_key(calibration_hash, qname) == (
        2261972762, 2393839975,
    )
    expected = [
        484211665, 827814195, 36613847, 1464622029,
        3455518833, 2398396980, 2952284691,
    ]
    assert [
        activation_row_priority_scalar(calibration_hash, qname, row)
        for row in rows
    ] == expected
    observed = activation_row_priorities(
        calibration_hash, qname, torch.tensor(rows, dtype=torch.long)
    )
    torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.long))


def test_activation_priority_is_a_permutation_on_dense_prefix():
    rows = torch.arange(1 << 16, dtype=torch.long)
    priorities = activation_row_priorities(
        "a" * 32, "model.layers.3.self_attn.q_proj", rows
    )
    assert int(torch.unique(priorities).numel()) == int(rows.numel())


def test_activation_priority_domain_refuses_zero_and_over_uint32():
    assert validate_activation_priority_domain(1 << 20, 1 << 12) == 1 << 32
    with pytest.raises(ValueError, match="0 < global_samples"):
        validate_activation_priority_domain(0, 4)
    with pytest.raises(ValueError, match=r"2\*\*32"):
        validate_activation_priority_domain((1 << 32) + 1, 1)
    with pytest.raises(ValueError, match="uint32"):
        activation_row_priority_scalar(
            "a" * 32, "model.layers.0.q_proj", 1 << 32
        )


def test_divergent_duplicate_linear_cache_rows_refuse_synthesized_reuse(tmp_path):
    import pickle
    from prismaquant.incremental_probe import scan_cached_linear_stats

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for index, raw in enumerate((1.0, 2.0)):
        with (shard_dir / f"probe_shard_{index:03d}.pkl").open("wb") as handle:
            pickle.dump({
                "stats": {
                    "model.layers.0.q_proj": {
                        "h_trace_raw": raw,
                        "fisher_row": np.array([raw], dtype=np.float32),
                    },
                },
                "meta": {},
            }, handle)
    assert scan_cached_linear_stats(shard_dir, {}) == {}


def test_lm_head_linear_hook_observes_exact_t_minus_one_reference_rows():
    from prismaquant.incremental_probe import _scored_lm_head_logits

    torch.manual_seed(0)
    batch, seqlen, hidden_size, vocab = 3, 7, 5, 11
    hidden = torch.randn(
        batch, seqlen, hidden_size, requires_grad=True
    )
    head = torch.nn.Linear(hidden_size, vocab, bias=False)
    observed_rows: list[int] = []
    saved_inputs: list[torch.Tensor] = []
    stats = {
        "n_tokens_seen": 0,
        "act_sq_sum": torch.zeros(hidden_size, dtype=torch.float32),
        "act_absmax": torch.zeros(hidden_size, dtype=torch.float32),
    }

    def _forward_hook(_module, inputs, _output):
        value = inputs[0].detach()
        observed_rows.append(int(value.numel() // hidden_size))
        saved_inputs.append(value)

    def _backward_hook(_module, _grad_input, _grad_output):
        value = saved_inputs.pop().reshape(-1, hidden_size)
        stats["n_tokens_seen"] += int(value.shape[0])
        stats["act_sq_sum"].add_(
            value.pow(2).sum(dim=0, dtype=torch.float32)
        )
        torch.maximum(
            stats["act_absmax"],
            torch.maximum(
                value.amax(dim=0).abs(), value.amin(dim=0).abs()
            ).to(torch.float32),
            out=stats["act_absmax"],
        )

    handles = [
        head.register_forward_hook(_forward_hook),
        head.register_full_backward_hook(_backward_hook),
    ]
    try:
        actual = _scored_lm_head_logits(
            head, hidden, start=0, scored_tokens=seqlen - 1
        )
        actual.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    expected_hidden = hidden.detach()[:, :-1, :]
    expected = head(expected_hidden).float()
    expected_flat = expected_hidden.reshape(-1, hidden_size)
    torch.testing.assert_close(actual, expected)
    assert observed_rows == [batch * (seqlen - 1)]
    assert stats["n_tokens_seen"] == batch * (seqlen - 1)
    torch.testing.assert_close(
        stats["act_sq_sum"],
        expected_flat.pow(2).sum(dim=0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        stats["act_absmax"],
        torch.maximum(
            expected_flat.amax(dim=0).abs(),
            expected_flat.amin(dim=0).abs(),
        ).to(torch.float32),
    )


@pytest.mark.parametrize("mutation", ["shape", "token_count", "raw_nan", "extra_field", "marginal_trace", "routed"])
def test_recorded_dense_stat_refuses_contract_drift(mutation):
    row = {"h_trace_raw": 21., "h_w2_sum_raw": 7., "h_trace": 2.625,
           "h_w2_sum": .875, "h_trace_norm_tokens": 8,
           "w_max_abs": 2., "w_norm_sq": 5., "n_params": 6,
           "in_features": 3, "out_features": 2, "n_tokens_seen": 8,
           "route_prob": None, "router_path": None, "expert_id": None,
           "fisher_row": np.array([6., 15.], dtype=np.float32),
           "fisher_col": np.array([5., 7., 9.], dtype=np.float32),
           "g_sq_sum": np.array([10., 11.], dtype=np.float32),
           "act_sq_sum": np.array([8., 9., 10.], dtype=np.float32),
           "act_absmax": np.array([1., 2., 3.], dtype=np.float32)}
    artifacts.validate_sample_parallel_stat_row(row, qname=BODY_QNAMES[0],
        expected_tokens=8, expected_shape=(2, 3), require_marginals=True)
    if mutation == "shape": row["in_features"] = 4
    elif mutation == "token_count": row["n_tokens_seen"] = 7
    elif mutation == "raw_nan": row["h_trace_raw"] = float("nan")
    elif mutation == "extra_field": row["unclaimed"] = 1
    elif mutation == "marginal_trace": row["fisher_row"][0] += 1
    elif mutation == "routed": row["router_path"] = "router"
    with pytest.raises(ValueError):
        artifacts.validate_sample_parallel_stat_row(row, qname=BODY_QNAMES[0],
            expected_tokens=8, expected_shape=(2, 3), require_marginals=True)


@pytest.mark.parametrize("mutation", ["rows", "priorities", "stamp", "width", "dtype"])
def test_recorded_activation_blob_replays_exact_priority_contract(tmp_path, mutation):
    from prismaquant.incremental_probe import _validate_sample_activation_blob
    partition = _contract(0)
    all_rows = torch.arange(8, dtype=torch.int64)
    priorities = activation_row_priorities(partition["global_calibration_hash"], BODY_QNAMES[0], all_rows)
    selected = torch.topk(priorities, 3, largest=False, sorted=True).indices
    blob = {"name": BODY_QNAMES[0], "inputs": torch.ones(3, 3),
            "row_indices": all_rows[selected], "row_priorities": priorities[selected],
            "sample_parallel_activation": artifacts.activation_cache_shard_stamp(
                partition, qname=BODY_QNAMES[0], rows_limit=3, candidate_rows=8,
                execution_identity_sha256="e" * 64)}
    path = tmp_path / "activation.pt"
    kwargs = dict(qname=BODY_QNAMES[0], expected_width=3, partition_contract=partition,
                  rows_limit=3, execution_identity_sha256="e" * 64)
    torch.save(blob, path)
    _validate_sample_activation_blob(path, **kwargs)
    if mutation == "rows": blob["row_indices"][0] = (blob["row_indices"][0] + 1) % 8
    elif mutation == "priorities": blob["row_priorities"][0] += 1
    elif mutation == "stamp": blob["sample_parallel_activation"]["execution_identity_sha256"] = "f" * 64
    elif mutation == "width": blob["inputs"] = torch.ones(3, 4)
    elif mutation == "dtype": blob["inputs"] = blob["inputs"].to(torch.bfloat16)
    torch.save(blob, path)
    with pytest.raises(ValueError):
        _validate_sample_activation_blob(path, **kwargs)
