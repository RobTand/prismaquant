"""The source identity is built in a CPU-only quantum, not under a GPU (PQ #1374).

A first-time identity build hashed the whole GLM-5.3 source inside a GPU row:
about 70 minutes at 12.8 W of a 140 W envelope. The fix has four parts, and
each is pinned here:

1. ``tessera_joint_aura identity --model M --out D`` writes the runner-free
   source digest cache without a plan, lease or device.
2. ``build_streamed_model_identity(digest_cache_path=D)`` seeds every shard
   the identity cache misses from that digest cache, so the GPU pass hashes
   no payload byte, and it produces the identity a full hash would.
3. ``refuse_uncovered`` refuses before a byte is hashed when neither cache
   covers a shard, and names the quantum.
4. ``run`` starts from the proof its prepare wrote in the same output root.
   Before this, only the capture owner read it, and the run's streamed
   identity build rehashed the whole source.

Fixture shards are tiny local files. Shard payload hashing is mocked to
refuse or to count; nothing reads model bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import cost_streaming as cs


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
    return json.loads(out.read_text())


# -- 1. the CPU-only quantum ------------------------------------------------

def test_identity_command_writes_the_digest_cache_without_a_plan_or_device(
        checkpoint, tmp_path, monkeypatch, capsys):
    import torch
    from prismaquant import tessera_joint_aura  # noqa: F401 -- imported before CUDA is fenced
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: pytest.fail("the identity quantum touched CUDA"))
    root, shards = checkpoint
    out = tmp_path / "digests.json"
    payload = _identity_quantum(root, out)
    assert payload["schema"] == cs.SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA
    assert sorted(Path(row["fingerprint"]["path"]).name for row in payload["entries"]) \
        == sorted(shards)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["shards"] == 2 and printed["source_bytes"] == 65536 + 131072
    assert printed["sha256"] == cs._file_sha256(out)


def test_identity_command_refuses_a_plan(checkpoint, tmp_path):
    from prismaquant import tessera_joint_aura as bridge
    root, _ = checkpoint
    with pytest.raises(SystemExit):
        bridge.main(["identity", "--model", str(root), "--out", str(tmp_path / "d.json"),
                     "--plan", str(tmp_path / "plan.json")])


# -- 2. the GPU pass seeds from it and hashes nothing -----------------------

def test_the_streamed_identity_seeds_from_the_digest_quantum(
        checkpoint, tmp_path, monkeypatch):
    root, shards = checkpoint
    reference = cs.build_streamed_model_identity(_runner(shards), str(root))
    out = tmp_path / "digests.json"
    _identity_quantum(root, out)
    _payload_hashes(monkeypatch, refuse=True)
    cache = tmp_path / "run" / "source-identity.json"
    cache.parent.mkdir()
    identity = cs.build_streamed_model_identity(
        _runner(shards), str(root), identity_cache_path=cache,
        digest_cache_path=out, refuse_uncovered="bind the digest quantum")
    assert identity == reference
    assert json.loads(cache.read_text())["identity"] == reference


def test_a_changed_shard_is_not_admitted_from_the_digest_cache(
        checkpoint, tmp_path, monkeypatch):
    root, shards = checkpoint
    out = tmp_path / "digests.json"
    _identity_quantum(root, out)
    changed = shards["model-00002-of-00002.safetensors"]
    changed.write_bytes(b"c" * 131072)
    calls = _payload_hashes(monkeypatch, refuse=False)
    identity = cs.build_streamed_model_identity(
        _runner(shards), str(root), digest_cache_path=out)
    assert [Path(call).name for call in calls] == [changed.name]
    import hashlib
    assert hashlib.sha256(b"c" * 131072).hexdigest() in {
        row["sha256"] for row in identity["shards"]}


# -- 3. refusal before any hash ---------------------------------------------

def test_an_uncovered_source_refuses_before_hashing_and_names_the_quantum(
        checkpoint, monkeypatch):
    root, shards = checkpoint
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match=r"would hash 196608 bytes across 2 shard.*"
                                           r"run the identity quantum"):
        cs.build_streamed_model_identity(
            _runner(shards), str(root), refuse_uncovered="run the identity quantum")


def test_a_partly_covered_source_refuses_only_for_what_is_missing(
        checkpoint, tmp_path, monkeypatch):
    root, shards = checkpoint
    out = tmp_path / "digests.json"
    _identity_quantum(root, out)
    grown = shards["model-00001-of-00002.safetensors"]
    grown.write_bytes(b"a" * 65537)
    _payload_hashes(monkeypatch, refuse=True)
    with pytest.raises(RuntimeError, match=r"would hash 65537 bytes across 1 shard"):
        cs.build_streamed_model_identity(
            _runner(shards), str(root), digest_cache_path=out,
            refuse_uncovered="run the identity quantum")


# -- 4. run starts from the prepare's proof ---------------------------------

def test_run_is_seeded_from_the_prepare_proof_in_the_same_output_root(tmp_path):
    from prismaquant import tessera_joint_aura as bridge
    prepared = tmp_path / "prepare" / "source-identity.json"
    prepared.parent.mkdir()
    prepared.write_bytes(b'{"prepare": "proof"}\n')
    run_root = tmp_path / "run"
    run_root.mkdir()
    slot = bridge._seed_source_identity_cache({"output_root": str(tmp_path)}, run_root)
    assert slot == run_root / "source-identity.json"
    assert slot.read_bytes() == prepared.read_bytes()
    # A run's own proof is never overwritten by the prepare's.
    slot.write_bytes(b"run's own proof")
    bridge._seed_source_identity_cache({"output_root": str(tmp_path)}, run_root)
    assert slot.read_bytes() == b"run's own proof"
    # The prepare's own slot is left alone.
    bridge._seed_source_identity_cache({"output_root": str(tmp_path)}, prepared.parent)
    assert prepared.read_bytes() == b'{"prepare": "proof"}\n'


def test_an_explicit_binding_outranks_the_prepare_proof(tmp_path):
    from prismaquant import tessera_joint_aura as bridge
    prepared = tmp_path / "prepare" / "source-identity.json"
    prepared.parent.mkdir()
    prepared.write_bytes(b"prepare proof")
    bound = tmp_path / "bound.json"
    bound.write_bytes(b"bound proof")
    run_root = tmp_path / "run"
    run_root.mkdir()
    slot = bridge._seed_source_identity_cache(
        {"output_root": str(tmp_path),
         "source_identity_cache": {"path": str(bound), "sha256": bridge._sha(bound)}},
        run_root)
    assert slot.read_bytes() == b"bound proof"
