"""File-source bindings guard resume and independent production-cache intake."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import pickle
import shutil

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import production_weight_cache as pwc
from prismaquant import streamed_model_identity as identity
from prismaquant import streaming_production_cache as streaming
from prismaquant import union_production_cache as union
from prismaquant.calibration_data import _dtype_from_name
from test_production_weight_cache import _TinyChain
from test_union_production_cache import _make_shard


def _source(tmp_path):
    model = _TinyChain().eval()
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    save_file(model.state_dict(), str(root / "model.safetensors"))
    return root, model


def _binding(root, dtype=torch.float32, **kwargs):
    return identity.build_source_model_binding(root, dtype=dtype, **kwargs)


def _fill(model, cache_dir, binding, *, empty=False):
    return pwc.fill_production_weight_cache(
        model, torch.tensor([[0, 1]], dtype=torch.long), ["l1", "l2"],
        formats=["FP8_E4M3"], levers={"gptq": False, "scale_sweep": False},
        cache_dir=cache_dir, source_binding=binding,
        render_qnames=[] if empty else None, max_act_rows=8, progress=False,
    )


def test_current_byte_binding_reuses_only_unchanged_digest_in_dev_mode(tmp_path, monkeypatch):
    root, model = _source(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    cache = tmp_path / "digests.json"
    hashes = []
    real_hash = identity._hash_one_source_shard

    def observe(path, fingerprint):
        hashes.append(path)
        return real_hash(path, fingerprint)

    monkeypatch.setattr(identity, "_hash_one_source_shard", observe)
    before = _binding(root, digest_cache_path=cache)
    assert len(hashes) == 1
    assert _binding(root, digest_cache_path=cache) == before
    assert len(hashes) == 1
    shard = root / "model.safetensors"
    previous_stat = shard.stat()
    with torch.no_grad():
        model.l1.weight.mul_(2)
    save_file(model.state_dict(), str(shard))
    assert shard.stat().st_size == previous_stat.st_size
    os.utime(shard, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    recorded = identity.build_source_checkpoint_identity(root, digest_cache_path=cache)
    assert recorded == before["checkpoint"]
    assert len(hashes) == 1
    after = _binding(root, digest_cache_path=cache)
    assert after != before
    assert len(hashes) == 2
    assert _binding(root, digest_cache_path=cache) == after
    assert len(hashes) == 2


def test_source_binding_is_portable_and_binds_resolved_dtype(tmp_path):
    root, _model = _source(tmp_path)
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    before = _binding(root, _dtype_from_name("bf16"))
    assert _binding(moved, _dtype_from_name("bfloat16")) == before
    assert _binding(root, torch.float16) != before
    assert identity.validate_source_model_binding(before) == before


def test_remote_source_binding_uses_one_original_local_hub_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    root, _model = _source(tmp_path)
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", resolve)
    binding = identity.build_source_model_binding(
        "test/source", dtype=torch.float32, local_files_only=True,
    )
    assert binding == _binding(root)
    assert calls == [{"repo_id": "test/source", "local_files_only": True}]


@pytest.mark.parametrize("damage", ["missing_dtype", "invalid_dtype", "missing_checkpoint", "bad_digest"])
def test_source_binding_parser_refuses_malformed_values(tmp_path, damage):
    root, _model = _source(tmp_path)
    binding = _binding(root)
    if damage == "missing_dtype":
        del binding["effective_dtype"]
    elif damage == "invalid_dtype":
        binding["effective_dtype"] = []
    elif damage == "missing_checkpoint":
        binding["checkpoint"] = None
    else:
        binding["checkpoint"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding|checkpoint"):
        identity.validate_source_model_binding(binding, where="audit")


@pytest.mark.parametrize("empty", [False, True], ids=["dense", "packed_only_base"])
def test_bound_directory_refuses_source_drift_and_binding_downgrade(tmp_path, empty):
    root, model = _source(tmp_path)
    cache_dir = tmp_path / "cache"
    binding = _binding(root)
    cache = _fill(model, cache_dir, binding, empty=empty)
    assert cache.metadata["source_binding"] == binding
    sidecar = cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    original = sidecar.read_bytes()
    _fill(model, cache_dir, binding, empty=empty)
    with pytest.raises(ValueError, match="source.*binding|source.*identity"):
        _fill(model, cache_dir, _binding(root, torch.float16), empty=empty)
    if not empty:
        with pytest.raises(ValueError, match="source.*binding|source.*identity"):
            _fill(model, cache_dir, None)
    assert sidecar.read_bytes() == original


def test_older_guarded_directory_refuses_binding_without_retroattestation(tmp_path):
    root, model = _source(tmp_path)
    cache_dir = tmp_path / "cache"
    _fill(model, cache_dir, None)
    sidecar = cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    original = sidecar.read_bytes()
    with pytest.raises(ValueError, match="missing base render identity fields"):
        _fill(model, cache_dir, _binding(root))
    assert sidecar.read_bytes() == original


def test_no_sidecar_admission_remains_named_unverified_trust(tmp_path, capsys):
    root, model = _source(tmp_path)
    cache_dir = tmp_path / "cache"
    original = _fill(model, cache_dir, None)
    old_values = {key: original.get(*key).clone() for key in original.weights}
    sidecar = cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    sidecar.unlink()
    binding = _binding(root)
    admitted = _fill(model, cache_dir, binding)
    trust = json.loads(sidecar.read_text())[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY]
    assert admitted.metadata[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] == trust
    assert all(torch.equal(admitted.get(*key), value) for key, value in old_values.items())
    restored = pickle.loads(pickle.dumps(admitted))
    restored.require_source_binding(binding, where="test attachment")
    assert "original source is unverified" in capsys.readouterr().out


def test_unbound_independent_pickle_refuses_attachment(tmp_path):
    root, _model = _source(tmp_path)
    cache = pwc.ProductionWeightCache(weights={}, levers={}, metadata={})
    with pytest.raises(ValueError, match="no source binding.*rebuild"):
        cache.require_source_binding(_binding(root), where="test attachment")


def test_append_only_source_adoption_retains_unverified_shard_history(tmp_path, capsys):
    root, model = _source(tmp_path)
    cache_dir = tmp_path / "cache"
    _fill(model, cache_dir, None)
    sidecar = cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    section = pwc.build_mtp_append_identity(
        max_act_rows=8, activation_source_hash="0" * 64,
        activation_rows={"mtp.fc": 2}, source_prefix="mtp.",
        source_tensor_count=1, profile_name="test",
    )
    sidecar.write_text(json.dumps({pwc.MTP_APPEND_SIDECAR_KEY: section}))
    binding = _binding(root)
    adopted = _fill(model, cache_dir, binding)
    stored = json.loads(sidecar.read_text())
    assert stored[pwc.MTP_APPEND_SIDECAR_KEY] == section
    assert adopted.metadata[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY] == stored[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY]
    assert stored[pwc.PRE_GUARD_ADMISSION_SIDECAR_KEY]["trusted_shards"]
    adopted.require_source_binding(binding, where="append adoption")
    assert "original source is unverified" in capsys.readouterr().out


@pytest.mark.parametrize("other", ["same", "different", "unbound"])
def test_exact_union_preserves_one_source_binding_or_refuses(tmp_path, other):
    root, _model = _source(tmp_path)
    binding = _binding(root)
    assignment = {"model.a": "FP8_E4M3", "model.b": "FP8_E4M3"}
    code_identity = union._current_code_identity()
    right = deepcopy(binding)
    if other == "different":
        right = _binding(root, torch.float16)
    left_path = _make_shard(
        tmp_path, shard_id="left", entries={("model.a", "FP8_E4M3"): torch.ones(4)},
        assignment=assignment, extra_metadata={"source_binding": binding},
        code_identity=code_identity,
    )
    right_path = _make_shard(
        tmp_path, shard_id="right", entries={("model.b", "FP8_E4M3"): torch.ones(4)},
        assignment=assignment,
        extra_metadata={} if other == "unbound" else {"source_binding": right},
        code_identity=code_identity,
    )
    output = tmp_path / "union"
    if other != "same":
        with pytest.raises(ValueError, match="source_binding"):
            union.union_shard_manifests([left_path, right_path], assignment=assignment, output_bundle=output)
        return
    union.union_shard_manifests([left_path, right_path], assignment=assignment, output_bundle=output)
    _manifest, merged = union.verify_union_manifest(output / "union_manifest.json", assignment=assignment)
    assert merged.metadata["source_binding"] == binding
    merged.require_source_binding(binding, where="union attachment")


def test_streamed_producer_stamps_original_file_source_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root, model = _source(tmp_path)
    returned = pwc.ProductionWeightCache(weights={}, levers={}, metadata={})
    closed = []
    context = SimpleNamespace(
        model=model, layers_prefix="layers.", num_layers=0,
        install=lambda *args: None, unload=lambda *args: None,
        shutdown=lambda: closed.append(True),
    )
    monkeypatch.setattr(streaming, "_build_streaming_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(streaming, "run_streaming_render", lambda *args, **kwargs: returned)
    monkeypatch.setattr(streaming, "_priority_setter", lambda context: None)
    result = streaming.fill_production_weight_cache_streaming(
        str(root), render_assignment={}, activation_cache_dir=tmp_path / "acts",
        formats=["FP8_E4M3"], levers={}, cache_dir=tmp_path / "cache",
        device="cpu", dtype=torch.float32,
    )
    assert closed == [True]
    assert result.metadata["source_binding"] == _binding(root)
    result.require_source_binding(_binding(root), where="streamed attachment")
