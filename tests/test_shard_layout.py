"""The shared safetensors shard layout — partition rule, names, index.

Pure planner: no torch, no filesystem beyond one index write. The lane-level
consequences (CB exporters, inventory gate) live in
``tests/test_cb_lane_sharding.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.shard_layout import (
    DEFAULT_SHARD_BYTES,
    SHARD_INDEX_NAME,
    SINGLE_CONTAINER_NAME,
    TENSOR_PAYLOAD_IDENTITY_SCHEMA,
    container_names,
    describe_container_layout,
    plan_shards,
    shard_name,
    tensor_payload_identity,
    write_shard_index,
)


def test_default_matches_the_compressed_tensors_lane():
    """One flag, one number: `EXPORT_SHARD_BYTES` and `--shard-bytes` agree."""
    assert DEFAULT_SHARD_BYTES == 1024 ** 3

    import prismaquant.export_native_compressed as native

    # The native lane spells the same default inline (`--shard-bytes`,
    # default 1024**3). A drift there is a lane disagreement, not a rename.
    source = Path(native.__file__).read_text().replace(" ", "")
    assert "default=1024**3" in source


def test_a_tensor_below_the_budget_shares_a_shard():
    plan = plan_shards([("a", 300), ("b", 300), ("c", 300)], 1000)
    assert plan == [["a", "b", "c"]]


def test_the_next_tensor_that_would_overflow_starts_a_new_shard():
    plan = plan_shards([("a", 600), ("b", 600), ("c", 100)], 1000)
    assert plan == [["a"], ["b", "c"]]


def test_an_exact_fit_closes_the_shard_without_stranding_a_tensor():
    plan = plan_shards([("a", 500), ("b", 500), ("c", 10)], 1000)
    assert plan == [["a", "b"], ["c"]]
    assert plan_shards([("a", 1000)], 1000) == [["a"]]


def test_a_tensor_larger_than_the_budget_gets_its_own_shard():
    """safetensors has no cross-file tensor: an oversize tensor is not split."""
    plan = plan_shards([("a", 10), ("big", 4000), ("b", 10)], 1000)
    assert plan == [["a"], ["big"], ["b"]]
    assert plan_shards([("big", 4000)], 1000) == [["big"]]


def test_the_layout_is_a_pure_function_of_order_and_budget():
    sizes = [(f"t{i}", 137 * (i % 7) + 11) for i in range(200)]
    first = plan_shards(sizes, 1024)
    second = plan_shards(list(sizes), 1024)
    assert first == second
    # Emit order is preserved, so the concatenation is the input sequence.
    assert [name for group in first for name in group] == [n for n, _ in sizes]


def test_there_is_no_zero_sentinel():
    """The native lane has none; inventing one here would split the flag."""
    with pytest.raises(ValueError, match="no zero sentinel"):
        plan_shards([("a", 1)], 0)
    with pytest.raises(ValueError, match="no zero sentinel"):
        plan_shards([("a", 1)], -1)
    with pytest.raises(ValueError, match="empty tensor set"):
        plan_shards([], 1024)


def test_a_budget_above_the_artifact_reproduces_the_legacy_layout():
    sizes = [("a", 10), ("b", 20), ("c", 30)]
    assert plan_shards(sizes, 10 ** 9) == [["a", "b", "c"]]
    assert container_names(1) == [SINGLE_CONTAINER_NAME]


def test_container_names_are_the_hf_spelling():
    assert container_names(3) == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    assert shard_name(7, 24) == "model-00007-of-00024.safetensors"
    with pytest.raises(ValueError):
        shard_name(0, 3)
    with pytest.raises(ValueError):
        shard_name(4, 3)
    with pytest.raises(ValueError):
        container_names(0)


def test_describe_container_layout_reads_the_layout_off_the_names():
    assert describe_container_layout([SINGLE_CONTAINER_NAME]) == ("single", 1)
    assert describe_container_layout(container_names(4)) == ("sharded", 4)
    # A run that is missing a member, or disagrees about its own total, is not
    # a layout any loader would accept.
    assert describe_container_layout([
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
    ])[0] == "other"
    assert describe_container_layout([
        "model-00001-of-00002.safetensors",
        "model-00002-of-00003.safetensors",
    ])[0] == "other"
    assert describe_container_layout([
        SINGLE_CONTAINER_NAME, "model-00001-of-00001.safetensors",
    ])[0] == "other"
    assert describe_container_layout([])[0] == "other"


def test_the_payload_identity_depends_on_tensors_and_nothing_else():
    """One digest over `name -> sha256(raw bytes)`; no file, no order, no size."""
    rows = {"b.weight": "b" * 64, "a.weight": "a" * 64}
    identity = tensor_payload_identity(rows)
    assert identity["schema"] == TENSOR_PAYLOAD_IDENTITY_SCHEMA
    assert identity["algorithm"] == "sha256"
    assert identity["tensors"] == 2
    # The routed text owner digests the old recipe's exact bytes: sorted
    # compact JSON, UTF-8, no spaces (PQ #1301 keeps the served value).
    assert identity["payload_sha256"] == hashlib.sha256(json.dumps(
        dict(sorted(rows.items())), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    # Insertion order is not part of the identity; the tensor set is.
    assert tensor_payload_identity(dict(sorted(rows.items()))) == identity
    assert tensor_payload_identity(
        {**rows, "c.weight": "c" * 64}) != identity
    with pytest.raises(ValueError, match="empty tensor payload"):
        tensor_payload_identity({})


def test_write_shard_index_is_the_layout_hf_and_vllm_read(tmp_path):
    path = write_shard_index(
        tmp_path, {"a.weight": container_names(2)[0]}, 4096)
    assert path.name == SHARD_INDEX_NAME
    payload = json.loads(path.read_text())
    assert payload == {
        "metadata": {"total_size": 4096},
        "weight_map": {"a.weight": "model-00001-of-00002.safetensors"},
    }
