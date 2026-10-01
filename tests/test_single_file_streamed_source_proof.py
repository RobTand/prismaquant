"""Complete single-file source coverage (PQ #1852, research prerequisite #275).

Real CPU safetensors exercise the shared identity builder and adoption gate.
The tiny runner is a unit-test double, never operational model evidence.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import cost_streaming
from prismaquant import tessera_calibration_cache as cc
from prismaquant.digests import canonical_json_sha256


@pytest.fixture
def single_file_source_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text("{}")
    tensors = {
        "lm_head.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.layers.0.proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.mtp.layers.0.proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"fixture": "single"})
    complete_map = dict.fromkeys(sorted(tensors), "model.safetensors")
    # The live decoder intentionally omits the auxiliary MTP namespace.
    live_map = {name: name for name in tensors if ".mtp." not in name}
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"model_type": "fixture"})),
        context=SimpleNamespace(
            weight_ckpt=live_map,
            weight_shard=dict.fromkeys(live_map, root / "model.safetensors"),
        ),
    )
    return root, runner, complete_map


def _build_single_file_proof_fixture(root, runner, directory):
    cache = directory / "identity.json"
    identity = cost_streaming.build_streamed_model_identity(
        runner, str(root), identity_cache_path=cache
    )
    return cache, identity


def _adopt_single_file_proof_fixture(root, cache, *, source_files=None):
    if source_files is None:
        source_files = {"model.safetensors": cc.sha256(root / "model.safetensors")}
    return cc.streamed_identity_proof_digests(
        root, cache, source_files,
        live_stat=lambda name: (root / name).stat(),
        expected_sha256=cc.sha256(cache),
    )


def _reseal_single_file_proof_fixture(cache, checkpoint_map):
    """A deliberately edited, self-consistent unit-test proof; not real evidence."""
    record = json.loads(cache.read_text())
    identity = record["identity"]
    identity["checkpoint_weight_map"] = checkpoint_map
    value = {key: identity[key] for key in (
        "config", "weight_map", "shards", "checkpoint_weight_map"
    )}
    identity["content_sha256"] = canonical_json_sha256(
        value, where="single-file tamper fixture"
    )
    cache.write_text(json.dumps(record))


def test_single_file_checkpoint_map_covers_auxiliary_header_names(single_file_source_checkpoint):
    root, _, expected = single_file_source_checkpoint
    mapping, shards = cost_streaming._local_checkpoint_shards(root)
    assert mapping == expected, "single-file header must define complete checkpoint coverage"
    assert shards == [(root / "model.safetensors").resolve()]
    assert not (root / "model.safetensors.index.json").exists()


def test_single_file_identity_includes_complete_map_not_only_live_decoder(
    single_file_source_checkpoint, tmp_path
):
    root, runner, expected = single_file_source_checkpoint
    _, identity = _build_single_file_proof_fixture(root, runner, tmp_path)
    assert identity.get("checkpoint_weight_map") == expected
    live_weight_map = identity["weight_map"]
    assert isinstance(live_weight_map, dict)
    assert "model.mtp.layers.0.proj.weight" not in live_weight_map
    assert cost_streaming.validate_streamed_model_identity(
        identity, where="actual single-file builder fixture"
    ) == identity


def test_single_file_real_builder_proof_is_adoptable_without_index(
    single_file_source_checkpoint, tmp_path
):
    root, runner, _ = single_file_source_checkpoint
    cache, _ = _build_single_file_proof_fixture(root, runner, tmp_path)
    digests, proof_sha = _adopt_single_file_proof_fixture(root, cache)
    assert digests == {"model.safetensors": cc.sha256(root / "model.safetensors")}
    assert proof_sha == cc.sha256(cache)
    assert not (root / "model.safetensors.index.json").exists()


def test_indexed_checkpoint_map_remains_the_authority(single_file_source_checkpoint):
    root, _, expected = single_file_source_checkpoint
    declared = {"lm_head.weight": expected["lm_head.weight"]}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": declared}))
    mapping, shards = cost_streaming._local_checkpoint_shards(root)
    assert mapping == declared
    assert shards == [(root / "model.safetensors").resolve()]


@pytest.mark.parametrize("tamper", ["omit_auxiliary", "wrong_shard", "extra_tensor"])
def test_resealed_single_file_map_tampering_is_refused(
    single_file_source_checkpoint, tmp_path, tamper
):
    root, runner, expected = single_file_source_checkpoint
    cache, _ = _build_single_file_proof_fixture(root, runner, tmp_path)
    mapping = copy.deepcopy(expected)
    if tamper == "omit_auxiliary":
        mapping.pop("model.mtp.layers.0.proj.weight")
    elif tamper == "wrong_shard":
        mapping["lm_head.weight"] = "foreign.safetensors"
    else:
        mapping["foreign.weight"] = "model.safetensors"
    _reseal_single_file_proof_fixture(cache, mapping)
    with pytest.raises(RuntimeError, match="complete checkpoint index"):
        _adopt_single_file_proof_fixture(root, cache)


def test_single_file_source_mutation_refuses_old_digest(single_file_source_checkpoint, tmp_path):
    root, runner, _ = single_file_source_checkpoint
    cache, _ = _build_single_file_proof_fixture(root, runner, tmp_path)
    path = root / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="SHA differs|another object"):
        _adopt_single_file_proof_fixture(root, cache)


def test_single_file_capture_roster_must_cover_the_shard(single_file_source_checkpoint, tmp_path):
    root, runner, _ = single_file_source_checkpoint
    cache, _ = _build_single_file_proof_fixture(root, runner, tmp_path)
    with pytest.raises(RuntimeError, match="SHA differs|omits canonical capture"):
        _adopt_single_file_proof_fixture(root, cache, source_files={})


def test_invalid_single_file_header_is_not_complete_checkpoint_coverage(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors checkpoint")
    with pytest.raises(RuntimeError, match="header|safetensors|checkpoint"):
        cost_streaming._local_checkpoint_shards(tmp_path)
