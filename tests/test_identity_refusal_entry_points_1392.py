"""Every GPU-row entry point refuses an uncovered source shard (PQ #1392).

#1374 (PR #1391) made ``build_streamed_model_identity`` refuse, before it
hashes a byte, when neither the identity cache nor the CPU-only identity
quantum's digest proof covers a shard -- but only the ``tessera_joint_aura``
reference pass passed the arguments. Stage A, Stage B and the sample-parallel
worker source cache still hashed the whole source under a GPU reservation.
Each now builds its identity through one shared helper
(``tessera_joint_aura.source_identity_proof_kwargs``). Per entry point:

* an uncovered source refuses with no payload byte hashed, and
* a covered source reproduces the identity a full hash produces.

The fixture shards are tiny local files; payload hashing is patched to refuse
or count, and nothing reads model bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import cost_streaming as cs

REFUSAL = r"would hash 196608 bytes across 2 shard.*run the identity quantum"


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    shards = {}
    weight_map = {}
    for name, payload in (("model-00001-of-00002.safetensors", b"a" * 65536),
                          ("model-00002-of-00002.safetensors", b"b" * 131072)):
        path = root / name
        path.write_bytes(payload)
        shards[name] = path
        weight_map[f"{name[:14]}.weight"] = name
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    return root, shards


def _runner(shards):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: {"model_type": "fixture"})),
        context=SimpleNamespace(
            weight_ckpt={},
            weight_shard={name: str(path) for name, path in shards.items()}))


def _payload_hashes(monkeypatch, *, refuse):
    calls = []
    real = cs._file_sha256

    def _hash(path):
        if str(path).endswith(".safetensors"):
            if refuse:
                raise AssertionError(f"payload hash called for {path}")
            calls.append(str(path))
        return real(path)
    monkeypatch.setattr(cs, "_file_sha256", _hash)
    return calls


def _identity_quantum(root, out):
    from prismaquant import tessera_joint_aura as bridge
    assert bridge.main(["identity", "--model", str(root), "--out", str(out)]) == 0
    return json.loads(Path(out).read_text())


def _binding(path):
    return {"path": str(path), "sha256": cs._file_sha256(path)}


def _reference(root, shards):
    return cs.build_streamed_model_identity(_runner(shards), str(root))


# -- Stage A ----------------------------------------------------------------

def test_stage_a_refuses_an_uncovered_source_before_hashing(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import _stage_a_source_identity
    root, shards = checkpoint
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match=REFUSAL):
        _stage_a_source_identity(
            _runner(shards), {"model": str(root)}, tmp_path / "id.json")


def test_stage_a_covered_source_reproduces_the_full_hash_identity(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import _stage_a_source_identity
    root, shards = checkpoint
    reference = _reference(root, shards)
    digests = tmp_path / "digests.json"
    _identity_quantum(root, digests)
    _payload_hashes(monkeypatch, refuse=True)
    identity = _stage_a_source_identity(
        _runner(shards),
        {"model": str(root), "source_digest_cache": _binding(digests)},
        tmp_path / "id.json")
    assert identity == reference


def test_stage_a_refuses_a_digest_proof_whose_bytes_changed(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import _stage_a_source_identity
    root, shards = checkpoint
    digests = tmp_path / "digests.json"
    _identity_quantum(root, digests)
    binding = _binding(digests)
    digests.write_bytes(digests.read_bytes() + b" ")
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(Exception, match="(?i)sha256|digest|differ|mismatch"):
        _stage_a_source_identity(
            _runner(shards), {"model": str(root), "source_digest_cache": binding},
            tmp_path / "id.json")


# -- Stage B ----------------------------------------------------------------

def test_stage_b_refuses_an_uncovered_source_before_hashing(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import _build_quantum_source_identity
    root, shards = checkpoint
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match=REFUSAL):
        _build_quantum_source_identity(
            _runner(shards), {"model": str(root)}, run_dir=tmp_path / "run")


def test_stage_b_covered_source_reproduces_the_full_hash_identity(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import _build_quantum_source_identity
    root, shards = checkpoint
    reference = _reference(root, shards)
    digests = tmp_path / "digests.json"
    _identity_quantum(root, digests)
    _payload_hashes(monkeypatch, refuse=True)
    identity = _build_quantum_source_identity(
        _runner(shards),
        {"model": str(root), "source_digest_cache": _binding(digests)},
        run_dir=tmp_path / "run")
    assert identity == reference


def test_stage_b_head_slice_carries_the_proof_as_declared_bytes(
        checkpoint, tmp_path, monkeypatch):
    """On the head-slice path the proof arrives as head-file bytes."""
    from prismaquant.joint_cost_quantum import _build_quantum_source_identity
    root, shards = checkpoint
    reference = _reference(root, shards)
    digests = tmp_path / "digests.json"
    _identity_quantum(root, digests)
    _payload_hashes(monkeypatch, refuse=True)
    identity = _build_quantum_source_identity(
        _runner(shards), {"model": str(root)}, run_dir=tmp_path / "run",
        digest_cache_bytes=digests.read_bytes())
    assert identity == reference
    # ... and without those bytes the same call refuses.
    with pytest.raises(RuntimeError, match=REFUSAL):
        _build_quantum_source_identity(
            _runner(shards), {"model": str(root)}, run_dir=tmp_path / "run2")


def test_stage_b_head_role_list_names_the_digest_proof():
    from prismaquant.joint_stage_b_head import HEAD_FILE_ROLES
    assert "source_digest_cache" in HEAD_FILE_ROLES


# -- sample-parallel worker -------------------------------------------------

def test_worker_refuses_an_uncovered_source_before_hashing(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.sample_parallel_probe import _build_worker_source_identity
    root, shards = checkpoint
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match=REFUSAL):
        _build_worker_source_identity(
            _runner(shards), root, tmp_path / "id.json")


def test_worker_covered_source_reproduces_the_full_hash_identity(
        checkpoint, tmp_path, monkeypatch):
    from prismaquant.sample_parallel_probe import _build_worker_source_identity
    root, shards = checkpoint
    reference = _reference(root, shards)
    digests = tmp_path / "digests.json"
    _identity_quantum(root, digests)
    _payload_hashes(monkeypatch, refuse=True)
    identity = _build_worker_source_identity(
        _runner(shards), root, tmp_path / "id.json",
        source_digest_cache=digests,
        source_digest_cache_sha256=cs._file_sha256(digests))
    assert identity == reference


def test_worker_source_cache_cli_names_the_digest_proof():
    from prismaquant.sample_parallel_probe import _build_parser
    args = _build_parser().parse_args([
        "prepare-worker-source-cache", "--model", "m",
        "--output", "o", "--offload-folder", "f",
        "--source-digest-cache", "d", "--source-digest-cache-sha256", "x"])
    assert args.source_digest_cache == "d"
    assert args.source_digest_cache_sha256 == "x"
