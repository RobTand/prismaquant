"""Complete single-file source coverage (PQ #1852, research prerequisite #275).

Real CPU safetensors exercise the shared identity builder and adoption gate.
The tiny runner is a unit-test double, never operational model evidence.
"""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import streamed_model_identity as cost_streaming_identity
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
    identity = cost_streaming_identity.build_streamed_model_identity(
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
    mapping, shards = cost_streaming_identity._local_checkpoint_shards(root)
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
    assert cost_streaming_identity.validate_streamed_model_identity(
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
    mapping, shards = cost_streaming_identity._local_checkpoint_shards(root)
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
        cost_streaming_identity._local_checkpoint_shards(tmp_path)


def test_streamed_identity_validation_uses_shared_sha256_grammar(
    single_file_source_checkpoint, tmp_path, monkeypatch
):
    """PQ #3319: both digest grammar checks use the shared owner."""
    consumer = sys.modules[cost_streaming_identity.__name__]
    digests = sys.modules["prismaquant.digests"]
    real = digests.is_sha256hex
    calls = []

    def recording_is_sha256hex(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(digests, "is_sha256hex", recording_is_sha256hex, raising=False)
    monkeypatch.setattr(consumer, "is_sha256hex", recording_is_sha256hex, raising=False)
    root, runner, _ = single_file_source_checkpoint
    _, identity = _build_single_file_proof_fixture(root, runner, tmp_path)
    result = consumer.validate_streamed_model_identity(
        identity, where="shared grammar routing fixture"
    )
    assert result == identity
    expected = [str(identity["content_sha256"]).lower()] + [
        str(shard["sha256"]).lower() for shard in identity["shards"]
    ]
    assert calls == expected, (
        "validate_streamed_model_identity must route content and shard digests "
        "through digests.is_sha256hex"
    )


_CONTROL_WHERE = "grammar control"
_CONTENT_REFUSAL = f"{_CONTROL_WHERE} requires exact model content_sha256"
_SHARD0_REFUSAL = f"{_CONTROL_WHERE} model shard 0 lacks content sha256"
_MISMATCH_REFUSAL = (
    f"{_CONTROL_WHERE} model content_sha256 does not match its source shard identity"
)
_NO_SHARDS_REFUSAL = f"{_CONTROL_WHERE} requires source shard content identities"


def _reseal_identity_value(identity):
    """A deliberately edited, self-consistent unit-test identity; not real evidence."""
    value = {key: identity[key] for key in (
        "config", "weight_map", "shards", "checkpoint_weight_map"
    ) if key in identity}
    identity["content_sha256"] = canonical_json_sha256(
        value, where="grammar control reseal fixture"
    )
    return identity


def _set_content(value):
    def mutate(identity):
        identity["content_sha256"] = value(identity) if callable(value) else value
        return identity
    return mutate


def _set_shard0(value, *, reseal=False):
    def mutate(identity):
        shard = identity["shards"][0]
        shard["sha256"] = value(shard["sha256"]) if callable(value) else value
        return _reseal_identity_value(identity) if reseal else identity
    return mutate


def _drop_content(identity):
    identity.pop("content_sha256")
    return identity


def _drop_shard0_digest(identity):
    identity["shards"][0].pop("sha256")
    return identity


def _set_shards(value):
    def mutate(identity):
        identity["shards"] = value
        return identity
    return mutate


def _append_shard(record):
    def mutate(identity):
        identity["shards"].append(record)
        return identity
    return mutate


def _reseal_checkpoint_map(value):
    def mutate(identity):
        identity["checkpoint_weight_map"] = value
        return _reseal_identity_value(identity)
    return mutate


_ACCEPTED = object()

# Frozen base outcomes (PQ #3319): exact RuntimeError text or acceptance.
_GRAMMAR_CONTROL_CASES = [
    ("content_upper", _set_content(lambda i: i["content_sha256"].upper()), _ACCEPTED),
    ("shard_upper_resealed", _set_shard0(str.upper, reseal=True), _ACCEPTED),
    ("shard_upper_not_resealed", _set_shard0(str.upper), _MISMATCH_REFUSAL),
    ("content_nonhex", _set_content("g" * 64), _CONTENT_REFUSAL),
    ("content_nonhex_punct", _set_content("-" * 64), _CONTENT_REFUSAL),
    ("shard_nonhex", _set_shard0("z" * 64), _SHARD0_REFUSAL),
    ("content_len63", _set_content(lambda i: i["content_sha256"][:63]), _CONTENT_REFUSAL),
    ("content_len65", _set_content(lambda i: i["content_sha256"] + "0"), _CONTENT_REFUSAL),
    ("content_empty", _set_content(""), _CONTENT_REFUSAL),
    ("content_missing", _drop_content, _CONTENT_REFUSAL),
    ("shard_len63", _set_shard0(lambda d: d[:63]), _SHARD0_REFUSAL),
    ("shard_len65", _set_shard0(lambda d: d + "a"), _SHARD0_REFUSAL),
    ("content_newline", _set_content(lambda i: i["content_sha256"] + "\n"), _CONTENT_REFUSAL),
    ("shard_newline", _set_shard0(lambda d: d + "\n", reseal=True), _SHARD0_REFUSAL),
    ("content_none", _set_content(None), _CONTENT_REFUSAL),
    ("content_int_zero", _set_content(0), _CONTENT_REFUSAL),
    ("content_bytes", _set_content(lambda i: i["content_sha256"].encode()), _CONTENT_REFUSAL),
    ("content_int_64_digits", _set_content(10 ** 63), _MISMATCH_REFUSAL),
    ("shard_none", _set_shard0(None), _SHARD0_REFUSAL),
    ("shard_int_64_digits", _set_shard0(10 ** 63), _MISMATCH_REFUSAL),
    ("shard_digest_missing", _drop_shard0_digest, _SHARD0_REFUSAL),
    ("shards_missing_list", _set_shards("model.safetensors"), _NO_SHARDS_REFUSAL),
    ("shards_empty", _set_shards([]), _NO_SHARDS_REFUSAL),
    ("shard_not_mapping", _set_shards(["model.safetensors"]),
     f"{_CONTROL_WHERE} model shard 0 is malformed"),
    ("second_shard_malformed", _append_shard("x"),
     f"{_CONTROL_WHERE} model shard 1 is malformed"),
    ("second_shard_no_digest", _append_shard({"path": "x"}),
     f"{_CONTROL_WHERE} model shard 1 lacks content sha256"),
    ("resealed_foreign_shard_digest", _set_shard0("0" * 64, reseal=True), _ACCEPTED),
    ("unresealed_foreign_shard_digest", _set_shard0("0" * 64), _MISMATCH_REFUSAL),
    ("resealed_malformed_checkpoint_map", _reseal_checkpoint_map({"lm_head.weight": ""}),
     f"{_CONTROL_WHERE} model checkpoint_weight_map is malformed"),
    ("resealed_checkpoint_map_not_mapping", _reseal_checkpoint_map(["model.safetensors"]),
     f"{_CONTROL_WHERE} model checkpoint_weight_map is malformed"),
    ("content_contradicts_shards", _set_content("f" * 64), _MISMATCH_REFUSAL),
]


@pytest.mark.parametrize(
    "mutate, expected",
    [case[1:] for case in _GRAMMAR_CONTROL_CASES],
    ids=[case[0] for case in _GRAMMAR_CONTROL_CASES],
)
def test_streamed_identity_digest_grammar_outcomes_are_frozen(
    single_file_source_checkpoint, tmp_path, mutate, expected
):
    """Behavior-preserving controls for PQ #3319; they pass on base and head."""
    root, runner, _ = single_file_source_checkpoint
    _, identity = _build_single_file_proof_fixture(root, runner, tmp_path)
    mutated = mutate(copy.deepcopy(identity))
    if expected is _ACCEPTED:
        result = cost_streaming_identity.validate_streamed_model_identity(
            mutated, where=_CONTROL_WHERE
        )
        assert result == mutated
        return
    with pytest.raises(RuntimeError) as refused:
        cost_streaming_identity.validate_streamed_model_identity(
            mutated, where=_CONTROL_WHERE
        )
    assert str(refused.value) == expected
