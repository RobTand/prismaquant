"""Dev-portable source-identity reuse across hosts (PQ #843).

RED-first: the stat fingerprint keys cached digests on host-local
``st_dev``, so a cache seeded on one Spark misses on every shard of the
other and rehashes the whole checkpoint. Only long-landed names are
imported at module scope; the new predicate is imported lazily where the
test needs it so RED collects. Fixture shards are tiny local files --
payload hashing is mocked to refuse (dev reuse) or count (certified
rehash); no model bytes are ever read here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import cost_streaming as cs

DEV_ENV = "PRISMAQUANT_DEV_MODE"


def _fingerprint(**overrides):
    base = {"path": "/mnt/shared/model/a.safetensors", "device": 64,
            "inode": 1368, "size": 5368221856, "mtime_ns": 1788063368825961508,
            "ctime_ns": 1788063368825961508}
    base.update(overrides)
    return base


def _dev_on(monkeypatch):
    monkeypatch.setenv(DEV_ENV, "1")


def _dev_off(monkeypatch):
    monkeypatch.setenv(DEV_ENV, "0")


# -- the shared predicate ------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("device", 75),  # the cross-host case: same file, other mount
])
def test_device_only_difference_reuses_in_dev(monkeypatch, field, value):
    _dev_on(monkeypatch)
    from prismaquant.cost_streaming import stat_fingerprint_reusable
    assert stat_fingerprint_reusable(_fingerprint(), _fingerprint(**{field: value}))


def test_device_only_difference_is_strict_in_certified(monkeypatch):
    _dev_off(monkeypatch)
    from prismaquant.cost_streaming import stat_fingerprint_reusable
    assert not stat_fingerprint_reusable(_fingerprint(), _fingerprint(device=75))


@pytest.mark.parametrize("field,value", [
    ("path", "/mnt/shared/model/b.safetensors"),
    ("inode", 1369),
    ("size", 5368221857),
    ("mtime_ns", 1788063368825961509),
    ("ctime_ns", 1788063368825961509),
    ("device", "64"),
    ("device", True),
])
def test_any_other_difference_never_reuses(monkeypatch, field, value):
    from prismaquant.cost_streaming import stat_fingerprint_reusable
    live, cached = _fingerprint(), _fingerprint(**{field: value})
    _dev_on(monkeypatch)
    assert not stat_fingerprint_reusable(live, cached)
    _dev_off(monkeypatch)
    assert not stat_fingerprint_reusable(live, cached)
    assert not stat_fingerprint_reusable(live, "not-a-dict")


def test_malformed_fingerprints_never_reuse(monkeypatch):
    """Missing fields, extra fields, and empty rows never match -- a
    missing field is not a match and an extra field could carry mutation
    signal no comparison reads."""
    from prismaquant.cost_streaming import stat_fingerprint_reusable
    full = _fingerprint()
    missing = {key: value for key, value in full.items() if key != "device"}
    extra = dict(full, provenance="elsewhere")
    for live, cached in ((full, missing), (missing, full), (full, extra),
                         (extra, full), ({}, {}), (full, None)):
        _dev_on(monkeypatch)
        assert not stat_fingerprint_reusable(live, cached)
        _dev_off(monkeypatch)
        assert not stat_fingerprint_reusable(live, cached)


# -- build-level reuse (fixture checkpoint, mocked hashing) ---------------

@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    shards = {}
    for name, payload in (("a.safetensors", b"a" * 65536),
                          ("b.safetensors", b"b" * 131072)):
        path = root / name
        path.write_bytes(payload)
        shards[name] = path
    return root, shards


def _runner(shard_paths):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: {"model_type": "fixture"})),
        context=SimpleNamespace(
            weight_ckpt={},
            weight_shard={name: str(path)
                          for name, path in shard_paths.items()}))


def _build_cache(root, shards):
    """First build (certified, real tiny-figure hashes) to seed a cache."""
    cache = root / "identity-cache.json"
    runner = _runner(shards)
    identity = cs.build_streamed_model_identity(
        runner, str(root), identity_cache_path=cache)
    assert cache.is_file()
    return identity


def _retarget_device(cache_path, device):
    payload = json.loads(cache_path.read_text())
    for row in payload["fingerprints"]:
        row["device"] = device
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _refusing_hash(monkeypatch):
    """Refuse shard payload hashes; metadata kilobytes still hash for real."""
    real = cs._file_sha256

    def _refuse(path):
        if str(path).endswith(".safetensors"):
            raise AssertionError(f"payload hash called for {path}")
        return real(path)
    monkeypatch.setattr(cs, "_file_sha256", _refuse)


def _counting_hash(monkeypatch):
    """Count shard payload hashes; metadata kilobytes still hash for real."""
    calls = []
    real = cs._file_sha256

    def _count(path):
        if str(path).endswith(".safetensors"):
            calls.append(str(path))
        return real(path)
    monkeypatch.setattr(cs, "_file_sha256", _count)
    return calls


def test_dev_reuses_across_device_difference_without_hashing(
        monkeypatch, checkpoint, tmp_path):
    root, shards = checkpoint
    cache = root / "identity-cache.json"
    first = _build_cache(root, shards)
    live_device = next(iter(json.loads(cache.read_text())["fingerprints"]))["device"]
    _retarget_device(cache, live_device + 1000)
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    runner = _runner(shards)
    identity = cs.build_streamed_model_identity(
        runner, str(root), identity_cache_path=cache)
    assert identity["content_sha256"] == first["content_sha256"]
    assert [row["sha256"] for row in identity["shards"]] == [
        row["sha256"] for row in first["shards"]]


def test_certified_rehashes_across_device_difference(
        monkeypatch, checkpoint):
    root, shards = checkpoint
    cache = root / "identity-cache.json"
    _build_cache(root, shards)
    live_device = next(iter(json.loads(cache.read_text())["fingerprints"]))["device"]
    _retarget_device(cache, live_device + 1000)
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    cs.build_streamed_model_identity(
        _runner(shards), str(root), identity_cache_path=cache)
    assert len(calls) == len(shards)


def test_dev_refuses_mutated_cache_with_byte_count(
        monkeypatch, checkpoint, capsys):
    root, shards = checkpoint
    cache = root / "identity-cache.json"
    _build_cache(root, shards)
    payload = json.loads(cache.read_text())
    names = sorted(shards)
    grown = root / names[0]
    grown.write_bytes(b"a" * 65537)
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        cs.build_streamed_model_identity(
            _runner(shards), str(root), identity_cache_path=cache)
    assert "DEV-MODE" not in capsys.readouterr().out


def test_certified_still_rehashes_mutated_cache(monkeypatch, checkpoint):
    root, shards = checkpoint
    cache = root / "identity-cache.json"
    _build_cache(root, shards)
    names = sorted(shards)
    (root / names[0]).write_bytes(b"a" * 65537)
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    cs.build_streamed_model_identity(
        _runner(shards), str(root), identity_cache_path=cache)
    assert len(calls) == 1


def test_dev_missing_cache_refuses_with_byte_count(
        monkeypatch, checkpoint):
    root, shards = checkpoint
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        cs.build_streamed_model_identity(
            _runner(shards), str(root),
            identity_cache_path=root / "identity-cache.json")


def test_certified_missing_cache_hashes(monkeypatch, checkpoint):
    root, shards = checkpoint
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    cs.build_streamed_model_identity(
        _runner(shards), str(root),
        identity_cache_path=root / "identity-cache.json")
    assert len(calls) == len(shards)


def test_dev_top_up_without_contract_refuses(monkeypatch, tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    shard_a = root / "a.safetensors"
    shard_a.write_bytes(b"a" * 65536)
    cache = root / "identity-cache.json"
    cs.build_streamed_model_identity(
        _runner({"a": shard_a}), str(root), identity_cache_path=cache)
    (root / "b.safetensors").write_bytes(b"b" * 131072)
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        cs.build_streamed_model_identity(
            _runner({"a": shard_a, "b": root / "b.safetensors"}), str(root),
            identity_cache_path=cache)


def test_certified_top_up_hashes_only_new_shards(monkeypatch, tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    shard_a = root / "a.safetensors"
    shard_a.write_bytes(b"a" * 65536)
    cache = root / "identity-cache.json"
    cs.build_streamed_model_identity(
        _runner({"a": shard_a}), str(root), identity_cache_path=cache)
    shard_b = root / "b.safetensors"
    shard_b.write_bytes(b"b" * 131072)
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    identity = cs.build_streamed_model_identity(
        _runner({"a": shard_a, "b": shard_b}), str(root),
        identity_cache_path=cache)
    assert len(calls) == 1
    assert [row["path"] for row in identity["shards"]] == [
        str(shard_a.resolve()), str(shard_b.resolve())]
    refreshed = json.loads(cache.read_text())["fingerprints"]
    assert {row["path"] for row in refreshed} == {
        str(shard_a.resolve()), str(shard_b.resolve())}


def test_dev_portable_reuse_is_stamped_uncertified(
        monkeypatch, checkpoint, capsys):
    root, shards = checkpoint
    cache = root / "identity-cache.json"
    _build_cache(root, shards)
    live_device = next(iter(json.loads(cache.read_text())["fingerprints"]))["device"]
    _retarget_device(cache, live_device + 1000)
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    cs.build_streamed_model_identity(
        _runner(shards), str(root), identity_cache_path=cache)
    assert "DEV-MODE" in capsys.readouterr().out


# -- digest-cache memo path ----------------------------------------------

def _memo_fixture(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    index = {"metadata": {"total_size": 196608},
             "weight_map": {"a.weight": "a.safetensors",
                            "b.weight": "b.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    (root / "a.safetensors").write_bytes(b"a" * 65536)
    (root / "b.safetensors").write_bytes(b"b" * 131072)
    return root


def test_digest_cache_memo_reuses_across_device_in_dev(
        monkeypatch, tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    cache_path = tmp_path / "digest-cache.json"
    live = [cs._streamed_identity_stat_fingerprint(root / name)
            for name in ("a.safetensors", "b.safetensors")]
    real = {str(root / name): hashlib.sha256(
        (root / name).read_bytes()).hexdigest()
        for name in ("a.safetensors", "b.safetensors")}
    entries = []
    for fingerprint in live:
        moved = dict(fingerprint)
        moved["device"] = fingerprint["device"] + 1000
        entries.append({"fingerprint": moved,
                        "sha256": real[str(Path(moved["path"]))]})
    cache_path.write_text(json.dumps(
        {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
         "entries": entries}))
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    identity = build_source_checkpoint_identity(
        str(root), digest_cache_path=cache_path)
    assert [row["sha256"] for row in identity["shards"]] == [
        real[str(root / "a.safetensors")], real[str(root / "b.safetensors")]]


def test_digest_cache_memo_stays_strict_in_certified(
        monkeypatch, tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    cache_path = tmp_path / "digest-cache.json"
    live = [cs._streamed_identity_stat_fingerprint(root / name)
            for name in ("a.safetensors", "b.safetensors")]
    entries = []
    for fingerprint in live:
        moved = dict(fingerprint)
        moved["device"] = fingerprint["device"] + 1000
        entries.append({"fingerprint": moved, "sha256": "0" * 64})
    cache_path.write_text(json.dumps(
        {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
         "entries": entries}))
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    build_source_checkpoint_identity(str(root), digest_cache_path=cache_path)
    assert len(calls) == 2


def test_digest_cache_memo_miss_refuses_in_dev(monkeypatch, tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    cache_path = tmp_path / "digest-cache.json"
    live = [cs._streamed_identity_stat_fingerprint(root / name)
            for name in ("a.safetensors", "b.safetensors")]
    entries = [{"fingerprint": live[0],
                "sha256": hashlib.sha256(
                    (root / "a.safetensors").read_bytes()).hexdigest()}]
    cache_path.write_text(json.dumps(
        {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
         "entries": entries}))
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        build_source_checkpoint_identity(
            str(root), digest_cache_path=cache_path)


def test_digest_cache_memo_none_path_refuses_in_dev(monkeypatch, tmp_path):
    """Omitting the cache path is not an escape hatch: dev still refuses
    the hidden giant hash; certified preparation is the explicit path."""
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        build_source_checkpoint_identity(str(root), digest_cache_path=None)


def test_digest_cache_memo_none_path_hashes_in_certified(monkeypatch, tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    build_source_checkpoint_identity(str(root), digest_cache_path=None)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", [
    "missing-device",
    "wrong-typed-device",
    "extra-unknown-field",
])
def test_digest_cache_memo_malformed_rows_refuse_in_dev(
        monkeypatch, tmp_path, mutation):
    """A stored row without the exact six-field shape must never reuse --
    without the gate a device-less row would match every host's portable
    key. Refusal, no payload hash; certified behavior covered beside."""
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    cache_path = tmp_path / "digest-cache.json"
    live = [cs._streamed_identity_stat_fingerprint(root / name)
            for name in ("a.safetensors", "b.safetensors")]
    real = {str(root / name): hashlib.sha256(
        (root / name).read_bytes()).hexdigest()
        for name in ("a.safetensors", "b.safetensors")}
    entries = []
    for fingerprint in live:
        stored = dict(fingerprint)
        if mutation == "missing-device":
            del stored["device"]
        elif mutation == "wrong-typed-device":
            stored["device"] = str(stored["device"])
        else:
            stored["provenance"] = "elsewhere"
        entries.append({"fingerprint": stored,
                        "sha256": real[str(Path(stored["path"]))]})
    cache_path.write_text(json.dumps(
        {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
         "entries": entries}))
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        build_source_checkpoint_identity(
            str(root), digest_cache_path=cache_path)


def test_digest_cache_memo_malformed_rows_hash_in_certified(
        monkeypatch, tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    root = _memo_fixture(tmp_path)
    cache_path = tmp_path / "digest-cache.json"
    live = [cs._streamed_identity_stat_fingerprint(root / name)
            for name in ("a.safetensors", "b.safetensors")]
    entries = []
    for fingerprint in live:
        stored = dict(fingerprint)
        del stored["device"]
        entries.append({"fingerprint": stored, "sha256": "0" * 64})
    cache_path.write_text(json.dumps(
        {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
         "entries": entries}))
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    build_source_checkpoint_identity(str(root), digest_cache_path=cache_path)
    assert len(calls) == 2


def test_build_none_path_refuses_in_dev(monkeypatch, checkpoint):
    root, shards = checkpoint
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    with pytest.raises(RuntimeError, match="[Bb]ytes"):
        cs.build_streamed_model_identity(
            _runner(shards), str(root), identity_cache_path=None)


def test_build_none_path_hashes_in_certified(monkeypatch, checkpoint):
    root, shards = checkpoint
    _dev_off(monkeypatch)
    calls = _counting_hash(monkeypatch)
    cs.build_streamed_model_identity(
        _runner(shards), str(root), identity_cache_path=None)
    assert len(calls) == len(shards)


# -- seed path -----------------------------------------------------------

def test_seed_copies_the_bound_cache_byte_identical(tmp_path):
    from prismaquant.tessera_joint_aura import _seed_source_identity_cache
    import hashlib as _hashlib
    source = tmp_path / "plan-cache.json"
    source.write_bytes(b'{"identity": "fixture"}')
    digest = _hashlib.sha256(source.read_bytes()).hexdigest()
    out_root = tmp_path / "run"
    out_root.mkdir()
    slot = _seed_source_identity_cache(
        {"source_identity_cache": {"path": str(source), "sha256": digest}},
        out_root)
    assert slot == out_root / "source-identity.json"
    assert slot.read_bytes() == b'{"identity": "fixture"}'


def test_seed_refuses_a_different_preexisting_cache(tmp_path):
    from prismaquant.tessera_joint_aura import _seed_source_identity_cache
    import hashlib as _hashlib
    source = tmp_path / "plan-cache.json"
    source.write_bytes(b'{"identity": "fixture"}')
    digest = _hashlib.sha256(source.read_bytes()).hexdigest()
    out_root = tmp_path / "run"
    out_root.mkdir()
    (out_root / "source-identity.json").write_bytes(b"something else")
    with pytest.raises(Exception):
        _seed_source_identity_cache(
            {"source_identity_cache": {"path": str(source),
                                       "sha256": digest}},
            out_root)


# -- validate path, end to end -------------------------------------------

def _validate_fixture(tmp_path):
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import LlamaConfig
    except ImportError:
        pytest.skip("no LlamaConfig in installed transformers")
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(LlamaConfig().to_dict()))
    index = {"metadata": {"total_size": 196608},
             "weight_map": {"a.weight": "a.safetensors",
                            "b.weight": "b.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    shards = {}
    for name, payload in (("a.safetensors", b"a" * 65536),
                          ("b.safetensors", b"b" * 131072)):
        path = root / name
        path.write_bytes(payload)
        shards[name] = path
    return root, shards


def _validate_cache(root, shards):
    from transformers import LlamaConfig
    cache = root / "identity-cache.json"
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            to_dict=lambda: LlamaConfig().to_dict())),
        context=SimpleNamespace(
            weight_ckpt={},
            weight_shard={name: str(path)
                          for name, path in shards.items()}))
    cs.build_streamed_model_identity(
        runner, str(root), identity_cache_path=cache)
    return cache


def test_validate_reuses_across_device_in_dev(monkeypatch, tmp_path):
    from prismaquant.cost_streaming import validate_cached_streamed_model_identity
    root, shards = _validate_fixture(tmp_path)
    cache = _validate_cache(root, shards)
    payload = json.loads(cache.read_text())
    live_device = payload["fingerprints"][0]["device"]
    for row in payload["fingerprints"]:
        row["device"] = live_device + 1000
    cache.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _dev_on(monkeypatch)
    _refusing_hash(monkeypatch)
    identity = validate_cached_streamed_model_identity(str(root), cache)
    assert identity["content_sha256"] == payload["identity"]["content_sha256"]


def test_validate_stays_strict_in_certified(monkeypatch, tmp_path):
    from prismaquant.cost_streaming import validate_cached_streamed_model_identity
    root, shards = _validate_fixture(tmp_path)
    cache = _validate_cache(root, shards)
    payload = json.loads(cache.read_text())
    live_device = payload["fingerprints"][0]["device"]
    for row in payload["fingerprints"]:
        row["device"] = live_device + 1000
    cache.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _dev_off(monkeypatch)
    with pytest.raises(RuntimeError, match="[Dd]rift"):
        validate_cached_streamed_model_identity(str(root), cache)
