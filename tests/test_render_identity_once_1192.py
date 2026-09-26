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

The rest of the window's main-thread load work moves to the PWC loader
pool: each loader hashes the render it loaded before the tensor is handed
out, and each file's archive directory is parsed on its loader, on the bytes
it read (14% of row 43's main thread was the preflight's cold scan of every
candidate; PQ #1210 dropped that scan). The payload and the files the
quantum writes are byte-identical with and without each change.
"""
from __future__ import annotations

import pickle
import shutil
import threading
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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


MAIN = [(pwc.ProductionWeightCache, "resident_render_identity", _rehash_every_read),
        (pwc.ProductionWeightCache, "_loaded_render_identity",
         lambda self, tensor, observed, requested=None: None)]
# The memo alone: hashed lazily, on the main thread, once per load.
MEMO = MAIN[1:]


def test_the_payload_and_evidence_are_byte_identical_with_and_without_each_change(
        tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)
    arms = {name: _arm(tmp_path, monkeypatch, campaign, patches)
            for name, patches in (("main", MAIN), ("memo", MEMO), ("head", []))}
    reference, reference_files, payload = arms["main"]
    assert payload["costs"] and payload["provenance"]["joint_operator_windows"]
    assert reference_files
    for name in ("memo", "head"):
        pickled, files, _payload = arms[name]
        assert set(files) == set(reference_files), name
        for path in reference_files:
            assert files[path] == reference_files[path], (name, path)
        assert pickled == reference, name


# --------------------------------------------------------------------------
# The window's load work runs on the loader pool, not the main thread
# --------------------------------------------------------------------------

def _threaded_counts(monkeypatch, *, core_only=False):
    """Record the thread of every render hash and every archive scan.

    A scan is recorded as ``(source, on the main thread)``, where ``source``
    is the file's path for a scan that opens the file and ``"bytes"`` for a
    scan of a buffer already read.

    With ``core_only``, only calls made while the quantum core runs count,
    on any thread: the single run and Stage A the fixture runs first are
    not the code under test.
    """
    calls = {"hash": [], "scan": []}
    active = [] if core_only else [True]
    identity = pwc._cb_cache_tensor_identity
    scan = pwc.ProductionWeightCache._window_archive_storage_bytes

    def on_main():
        return threading.current_thread() is threading.main_thread()

    def counted_identity(tensor):
        if active:
            calls["hash"].append((tensor, on_main()))
        return identity(tensor)

    def counted_scan(source):
        if active:
            calls["scan"].append((str(source) if isinstance(source, (str, Path))
                                  else "bytes", on_main()))
        return scan(source)

    monkeypatch.setattr(pwc, "_cb_cache_tensor_identity", counted_identity)
    monkeypatch.setattr(pwc.ProductionWeightCache, "_window_archive_storage_bytes",
                        staticmethod(counted_scan))
    if core_only:
        core = runtime.run_layer_quantum_core

        def counted_core(*args, **kwargs):
            active.append(True)
            try:
                return core(*args, **kwargs)
            finally:
                active.pop()

        monkeypatch.setattr(runtime, "run_layer_quantum_core", counted_core)
    return calls


def test_the_quantum_main_thread_neither_hashes_a_render_nor_scans_an_archive(
        tmp_path, monkeypatch):
    calls = _threaded_counts(monkeypatch, core_only=True)
    payload, _record, served, _hashed = _counted_quantum(tmp_path, monkeypatch)
    loads = _loads(served)
    assert payload["costs"] and loads
    render_hashes = [(tensor, main) for tensor, main in calls["hash"]
                     if any(tensor is loaded for _pair, loaded in loads)]
    assert len(render_hashes) == len(loads)
    assert not [main for _tensor, main in render_hashes if main], (
        "a render was hashed on the main thread")
    # Every loaded render's archive directory was parsed once, on the bytes
    # its loader read, and none of the parses ran on the main thread.
    assert len(calls["scan"]) == len(loads)
    assert {source for source, _main in calls["scan"]} == {"bytes"}
    assert not [source for source, main in calls["scan"] if main], calls["scan"]


def test_the_loaders_hash_only_when_the_window_asks(tmp_path, monkeypatch):
    cache, paths = _file_cache(tmp_path, count=3)
    unpatched = pwc._cb_cache_tensor_identity
    calls = _threaded_counts(monkeypatch)
    with _window(cache, paths):
        assert calls["hash"] == []
    with cache.retained_window(list(paths), max_resident_bytes=1 << 20,
                               max_workers=2, max_load_buffer_bytes=1 << 20,
                               render_identities=True):
        tensors = {key: cache.get_resident(*key) for key in paths}
        assert len(calls["hash"]) == 3
        assert not [main for _tensor, main in calls["hash"] if main]
        for key, tensor in tensors.items():
            assert cache.resident_render_identity(*key, tensor) == unpatched(tensor)
        # Served from the loaders' hashes: no second hash of any render.
        assert len(calls["hash"]) == 3
        tensors = tensor = None
    with pytest.raises(ValueError, match="render identities must be boolean"):
        with cache.retained_window(list(paths), max_resident_bytes=1 << 20,
                                   max_workers=1, render_identities=1):
            pass


def test_the_loader_hash_equals_the_main_thread_hash(tmp_path):
    cache, paths = _file_cache(tmp_path, count=2)
    with cache.retained_window(list(paths), max_resident_bytes=1 << 20,
                               max_workers=2, max_load_buffer_bytes=1 << 20,
                               render_identities=True):
        for key in paths:
            tensor = cache.get_resident(*key)
            assert cache.resident_render_identity(*key, tensor) == \
                pwc._cb_cache_tensor_identity(tensor)
        tensor = None


def test_the_archive_scans_run_off_the_main_thread_once_per_file(tmp_path, monkeypatch):
    cache, paths = _file_cache(tmp_path, count=4)
    calls = _threaded_counts(monkeypatch)
    with cache.retained_window(list(paths), max_resident_bytes=1 << 20,
                               max_workers=2, max_load_buffer_bytes=1 << 20):
        pass
    # One parse per file, on the bytes its loader read (PQ #1210): the
    # declared file is never opened a second time to read its directory.
    assert [source for source, _main in calls["scan"]] == ["bytes"] * len(paths)
    assert not [main for _source, main in calls["scan"] if main]


@pytest.mark.parametrize("damage", ["not-a-zip", "missing"])
def test_a_bad_file_refuses_its_window_and_leaves_nothing_resident(tmp_path, damage):
    """A missing file refuses at preflight; a bad archive at its own read.

    The retained window charges each file its length before the first load
    (PQ #1210), so a file that is not an ordinary uncompressed Torch archive
    is refused when its loader parses the bytes it read, before they are
    deserialized, with the same error the preflight scan raised.
    """
    cache, paths = _file_cache(tmp_path, count=3)
    target = list(paths.values())[1]
    if damage == "not-a-zip":
        target.write_bytes(b"not a torch archive")
        expected = (RuntimeError, "PWC window has an unaccountable Torch archive")
    else:
        target.unlink()
        expected = (FileNotFoundError, str(target.absolute()))
    loads = []
    original = torch.load

    def counted_load(source, *args, **kwargs):
        loads.append(source.getvalue() if hasattr(source, "getvalue") else source)
        return original(source, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "load", counted_load)
        with pytest.raises(expected[0]) as caught:
            with cache.retained_window(list(paths), max_resident_bytes=1 << 20,
                                       max_workers=1, max_load_buffer_bytes=1 << 20):
                pytest.fail("a window with a bad file was exposed")
    assert expected[1] in str(caught.value)
    # The bad file was never deserialized (its siblings may have been: the
    # IO engine reads them concurrently); a missing file refused before any
    # load.
    assert b"not a torch archive" not in loads
    assert len(loads) <= (len(paths) - 1 if damage == "not-a-zip" else 0)
    assert getattr(cache, "_resident_window_files", None) is None
    assert all(isinstance(value, str) for value in cache.weights.values())
