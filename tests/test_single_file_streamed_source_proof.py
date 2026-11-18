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


def test_identity_validation_refuses_corrupt_cache_without_source_changes(
    single_file_source_checkpoint, tmp_path
):
    root, runner, _ = single_file_source_checkpoint
    cache, _ = _build_single_file_proof_fixture(root, runner, tmp_path)
    cache.write_bytes(b"{")
    source_before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(RuntimeError) as refusal:
        cost_streaming_identity.validate_cached_streamed_model_identity(root, cache)
    assert str(refusal.value) == (
        f"streamed model identity cache {cache} is corrupt; refusing identity reuse"
    )
    assert cache.read_bytes() == b"{"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == source_before


def test_invalid_single_file_header_is_not_complete_checkpoint_coverage(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors checkpoint")
    with pytest.raises(RuntimeError, match="header|safetensors|checkpoint"):
        cost_streaming_identity._local_checkpoint_shards(tmp_path)


def _assert_streamed_identity_digest_outcomes():
    consumer = sys.modules["prismaquant.streamed_model_identity"]
    owner = sys.modules["prismaquant.digests"]
    where = "compatibility fixture"
    body = {
        "config": {"model_type": "fixture"},
        "weight_map": {"layer.weight": "fixture.safetensors"},
        "shards": [{"path": "fixture.safetensors", "size": 8, "sha256": "b" * 64}],
    }
    good = {"schema": consumer.STREAMED_MODEL_IDENTITY_SCHEMA, **body,
            "content_sha256": canonical_json_sha256(body, where=where)}
    values = [None, 1, int("1" * 64), "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"]
    exact = "459745ebbe76cae9e9180dca5443516726434c38e13ef0c226ff4ba038e1b058"
    mismatch = "c2c370868c61bfba7c050eb4080eeadabd956e365e1aef8e79cd0fa31fe9dbb4"
    lacks = "5ef1f8d6c1ed024c5b4d2cab7141a39003401e1f652c4419e4c38a3245008b6a"
    malformed = "3de69b17b3f9a81cc58593e72c790f01a708a36be3d465456778be6c38e065a6"
    refusals = {
        exact: "compatibility fixture requires exact model content_sha256",
        mismatch: "compatibility fixture model content_sha256 does not match its source shard identity",
        lacks: "compatibility fixture model shard 0 lacks content sha256",
        malformed: "compatibility fixture model shard 0 is malformed",
    }
    for case, expected in (
        ("valid", "febf379436d516cddd830c627ad8c0bee8958dd2630e27ad3cc13ff02c3b8e00"),
        *((f"content{i}", digest) for i, digest in enumerate(
            [exact, exact, mismatch, exact, exact, mismatch, exact, exact]
        )),
        *((f"shard{i}", digest) for i, digest in enumerate([
            lacks, lacks, "25653e233105ee6babff840463180b0586491d1a0a5378d07353e41a0214c6b8",
            lacks, lacks, "86205872cdb06870761b1a5df19829e6435234923dd68a6cc0ee1c3d088e9787",
            lacks, lacks,
        ])),
        ("uppercase-content", "99433c0e94fac86e860eb90e794fc61be205dac969bd710354771062298da868"),
        ("malformed-shard", malformed),
    ):
        identity = copy.deepcopy(good)
        if case.startswith("content"):
            identity["content_sha256"] = values[int(case[-1])]
        elif case.startswith("shard"):
            identity["shards"][0]["sha256"] = values[int(case[-1])]
            identity["content_sha256"] = canonical_json_sha256(
                {key: identity[key] for key in body}, where=where
            )
        elif case == "uppercase-content":
            identity["content_sha256"] = good["content_sha256"].upper()
        elif case == "malformed-shard":
            identity["shards"] = [None]
        try:
            outcome = ("return", consumer.validate_streamed_model_identity(identity, where=where))
        except Exception as error:
            kind = type(error)
            outcome = (f"{kind.__module__}.{kind.__qualname__}", str(error))
        if expected in refusals:
            assert outcome == ("builtins.RuntimeError", refusals[expected]), case
        assert owner.bytes_sha256hex(repr(outcome).encode("utf-8")) == expected, case

    contradictory = copy.deepcopy(good)
    contradictory["checkpoint_weight_map"] = {"layer.weight": None}
    contradictory["content_sha256"] = canonical_json_sha256(
        {key: contradictory[key] for key in (*body, "checkpoint_weight_map")}, where=where
    )
    with pytest.raises(RuntimeError) as refusal:
        consumer.validate_streamed_model_identity(contradictory, where=where)
    assert type(refusal.value) is RuntimeError
    assert str(refusal.value) == "compatibility fixture model checkpoint_weight_map is malformed"


def test_streamed_identity_validation_preserves_digest_outcomes():
    _assert_streamed_identity_digest_outcomes()


def test_streamed_identity_validation_uses_shared_sha256_grammar(
    single_file_source_checkpoint, tmp_path, monkeypatch
):
    _assert_streamed_identity_digest_outcomes()
    consumer = sys.modules["prismaquant.streamed_model_identity"]
    owner = sys.modules["prismaquant.digests"]
    root, runner, _ = single_file_source_checkpoint
    _, identity = _build_single_file_proof_fixture(root, runner, tmp_path)
    real_is_sha256hex = owner.is_sha256hex
    calls = []

    def record(value):
        calls.append(value)
        return real_is_sha256hex(value)

    monkeypatch.setattr(owner, "is_sha256hex", record, raising=False)
    monkeypatch.setattr(consumer, "is_sha256hex", record, raising=False)
    consumer.validate_streamed_model_identity(identity, where="single-file grammar fixture")
    assert calls == [str(identity["content_sha256"]).lower(), *(
        str(shard["sha256"]).lower() for shard in identity["shards"]
    )]
