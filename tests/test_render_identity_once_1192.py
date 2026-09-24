"""A Stage B render window hashes each resident render once per load (PQ #1192).

Row 43 (PB ``d106121a45e4``, sparky, render windows 08-11) spent 24% of its
main thread in ``_cb_cache_tensor_identity``, called from the quantum's
operator record (``_record_joint_operator``) once per probe. The rendered
tensor is the same resident object for every probe of a window, so three of
every four hashes repeated work. The comparison with the prepared render
identity is a byte-integrity check and stays; only the repeated hash goes.

The counter below sees every call to the module's tensor-identity function
made while the quantum core runs, on any thread, and matches each call to
the resident render objects ``get_resident`` handed out. One resident object
is one load of one ``(name, fmt)`` in one window.
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pytest
import torch

import prismaquant.production_weight_cache as pwc

import test_joint_cost_quantum_runtime as runtime
from test_quantum_probe_identity_once_1183 import _campaign


def _counted_quantum(tmp_path, monkeypatch):
    """Run the layer-1 quantum on the stub; record renders served and hashed."""
    single, receipt, output_root = _campaign(tmp_path, monkeypatch)
    active, served, hashed = [], [], []
    identity = pwc._cb_cache_tensor_identity
    get_resident = pwc.ProductionWeightCache.get_resident
    core = runtime.run_layer_quantum_core

    def counted_identity(tensor):
        if active:
            hashed.append(tensor)
        return identity(tensor)

    def counted_get_resident(self, name, fmt):
        tensor = get_resident(self, name, fmt)
        if active:
            served.append(((name, fmt), tensor))
        return tensor

    def counted_core(*args, **kwargs):
        active.append(True)
        try:
            return core(*args, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(pwc, "_cb_cache_tensor_identity", counted_identity)
    monkeypatch.setattr(pwc.ProductionWeightCache, "get_resident", counted_get_resident)
    monkeypatch.setattr(runtime, "run_layer_quantum_core", counted_core)
    payload, record, _counters = runtime._run_quantum(
        tmp_path, monkeypatch, single=single, layer=1, receipt=receipt,
        output_root=output_root, plan_sha=runtime._hex("d"),
        prepared_sha=runtime._hex("e"))
    return payload, record, served, hashed


def _loads(served):
    """One entry per resident render object: its (name, fmt) and the object."""
    loads = []
    for pair, tensor in served:
        if not any(tensor is seen for _pair, seen in loads):
            loads.append((pair, tensor))
    return loads


def test_each_resident_render_is_hashed_once_per_load(tmp_path, monkeypatch):
    payload, _record, served, hashed = _counted_quantum(tmp_path, monkeypatch)
    loads = _loads(served)
    rows = {(name, fmt) for name, per_unit in payload["costs"].items()
            for fmt in per_unit}
    # The counter is not vacuous: the stub prices two rendered formats per
    # unit, each loaded once in its window and read by four probes.
    assert {pair for pair, _tensor in loads} == {
        pair for pair in rows if pair[1] != "BF16"}
    assert len(served) >= 4 * len(loads)
    counts = {pair: sum(1 for value in hashed if value is tensor)
              for pair, tensor in loads}
    assert all(count == 1 for count in counts.values()), (
        f"each resident render must be hashed once per load; hashes per "
        f"(name, fmt) load: {counts}")


# --------------------------------------------------------------------------
# The memo's lifetime is the resident load's
# --------------------------------------------------------------------------

FMT = "FP8_E4M3"


def _file_cache(tmp_path, count=2):
    paths = {}
    for index in range(count):
        key = (f"unit{index}", FMT)
        paths[key] = tmp_path / f"unit{index}.pt"
        torch.save(torch.arange(64, dtype=torch.bfloat16).reshape(8, 8) + index,
                   paths[key])
    cache = pwc.ProductionWeightCache(
        weights={key: str(path) for key, path in paths.items()}, levers={})
    cache.enable_lru(1 << 20)
    return cache, paths


def _window(cache, keys):
    return cache.retained_window(list(keys), max_resident_bytes=1 << 20,
                                 max_workers=1, max_load_buffer_bytes=1 << 20,
                                 release_file_pages=False)


def _counting(monkeypatch):
    hashed = []
    identity = pwc._cb_cache_tensor_identity

    def counted(tensor):
        hashed.append(tensor)
        return identity(tensor)

    monkeypatch.setattr(pwc, "_cb_cache_tensor_identity", counted)
    return hashed, identity


def test_the_identity_is_hashed_once_and_dies_with_the_window(tmp_path, monkeypatch):
    cache, paths = _file_cache(tmp_path)
    hashed, identity = _counting(monkeypatch)
    key = next(iter(paths))
    with _window(cache, paths):
        first = cache.get_resident(*key)
        served = cache.resident_render_identity(*key, first)
        # A caller that edits its copy cannot edit what the next probe reads.
        served["shape"].append(0)
        served["content_sha256"] = "0" * 64
        again = cache.resident_render_identity(*key, first)
        assert again == identity(first)
        assert [value is first for value in hashed] == [True]
    # The window released the tensor, and the identity with its receipt.
    assert isinstance(cache.weights[key], str)
    with pytest.raises(RuntimeError, match="no matching resident load"):
        cache.resident_render_identity(*key, first)
    # The next window's load is a new object and is hashed again.
    with _window(cache, paths):
        second = cache.get_resident(*key)
        assert second is not first
        cache.resident_render_identity(*key, second)
        cache.resident_render_identity(*key, second)
    assert [value is second for value in hashed] == [False, True]


def test_the_identity_refuses_a_tensor_changed_or_replaced_after_its_load(tmp_path):
    cache, paths = _file_cache(tmp_path)
    key, other = paths
    with _window(cache, paths):
        tensor = cache.get_resident(*key)
        cache.resident_render_identity(*key, tensor)
        # Another key's tensor, or a tensor this cache did not load, has no
        # receipt for this key.
        with pytest.raises(RuntimeError, match="no matching resident load"):
            cache.resident_render_identity(*key, cache.get_resident(*other))
        with pytest.raises(RuntimeError, match="no matching resident load"):
            cache.resident_render_identity(*key, tensor.clone())
        # An in-place write moves the version counter the load recorded.
        tensor[0, 0] += 1
        with pytest.raises(RuntimeError, match="changed after its load"):
            cache.resident_render_identity(*key, tensor)
        tensor = None


def test_an_lru_eviction_drops_the_identity(tmp_path):
    cache, paths = _file_cache(tmp_path)
    first, second = paths
    size = 8 * 8 * 2
    cache.enable_lru(size)
    cache.enable_file_load_receipts(max_file_bytes=max(
        path.stat().st_size for path in paths.values()))
    old = cache.get(*first)
    cache.resident_render_identity(*first, old)
    cache.get(*second)
    assert isinstance(cache.weights[first], str)
    with pytest.raises(RuntimeError, match="no matching resident load"):
        cache.resident_render_identity(*first, old)
    new = cache.get(*first)
    assert new is not old and torch.equal(new, old)
    assert cache.resident_render_identity(*first, new) == \
        pwc._cb_cache_tensor_identity(new)


# --------------------------------------------------------------------------
# Payload and evidence are byte-identical with and without the change
# --------------------------------------------------------------------------

def _output_bytes(record):
    root = Path(record["output_space"]["root"])
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _arm(tmp_path, monkeypatch, campaign, patches):
    """One quantum under ``patches``; its pickled payload and output files.

    Every arm runs in the same directories, so paths in the payload agree,
    and starts from an empty output space, not a resume.
    """
    single, receipt, output_root = campaign
    with monkeypatch.context() as patch:
        for target, attribute, value in patches:
            patch.setattr(target, attribute, value)
        payload, record, _counters = runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=1,
            receipt=receipt, output_root=output_root,
            plan_sha=runtime._hex("d"), prepared_sha=runtime._hex("e"))
    files = _output_bytes(record)
    shutil.rmtree(record["output_space"]["root"])
    return pickle.dumps(payload), files, payload


def _rehash_every_read(self, name, fmt, tensor):
    """Main's behaviour before PQ #1192: hash the render on every read."""
    return pwc._cb_cache_tensor_identity(tensor)


def test_the_payload_and_evidence_are_byte_identical_with_and_without_the_memo(
        tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)
    before, before_files, payload = _arm(tmp_path, monkeypatch, campaign, [
        (pwc.ProductionWeightCache, "resident_render_identity", _rehash_every_read)])
    after, after_files, _ = _arm(tmp_path, monkeypatch, campaign, [])
    assert payload["costs"] and payload["provenance"]["joint_operator_windows"]
    assert before_files and set(before_files) == set(after_files)
    for path in before_files:
        assert before_files[path] == after_files[path], path
    assert before == after
