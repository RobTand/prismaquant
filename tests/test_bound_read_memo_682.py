"""Bound preparation reads happen once per process, not once per panel (#682).

``native_moe_panel._qualified_quality_members`` reaches the bound
``ProductionWeightCache`` through ``tessera_joint_allocation._read_bound``,
which did a full ``read_bytes`` plus ``pickle.loads`` of the entire cache --
once per ``prepare_moe_inputs`` and once per ``freeze_moe_panel``, so twice
per panel, unprofiled at real PWC size. Two properties pin the fix:

* **One read, one unpickle** -- a second load of the same bound bytes
  reuses the verified bytes without re-reading or re-hashing, and a second
  qualification of the same preparation reuses the unpickled cache.
* **Fail-closed memo** -- the memo key carries the expected digest and a
  hit additionally requires the file's stat fence to be unchanged. Drifted
  bytes are re-read and re-verified, so a digest mismatch still refuses and
  a replaced preparation still reloads.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path


def _bind(path: Path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _clear_memos():
    from prismaquant import tessera_joint_allocation as allocation
    from prismaquant import native_moe_panel as panel
    allocation._BOUND_BYTES.clear()
    panel._QUALIFIED_QUALITY_MEMO.clear()


def test_bound_bytes_are_read_and_hashed_once_per_process(tmp_path, monkeypatch):
    """The second load of the same bound file does no I/O."""
    from prismaquant import tessera_joint_allocation as allocation

    _clear_memos()
    target = tmp_path / "completion.json"
    target.write_bytes(b'{"status": "complete"}')
    record = _bind(target)
    reads = []
    real_read_bytes = Path.read_bytes

    def counting(self, *args, **kwargs):
        if str(self) == str(target):
            reads.append(str(self))
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting)
    first = allocation._read_bound(record, "test completion")
    second = allocation._read_bound(record, "test completion")
    assert first == second == b'{"status": "complete"}'
    assert len(reads) == 1, "the second load must not re-read the file"


def test_drifted_bound_bytes_are_reverified_not_trusted(tmp_path):
    """A changed file under the same record re-reads and refuses."""
    from prismaquant import tessera_joint_allocation as allocation

    _clear_memos()
    target = tmp_path / "completion.json"
    target.write_bytes(b'{"status": "complete"}')
    record = _bind(target)
    assert allocation._read_bound(record, "test completion")
    target.write_bytes(b'{"status": "tampered with!!"}')
    try:
        allocation._read_bound(record, "test completion")
    except ValueError:
        pass
    else:
        raise AssertionError("drifted bound bytes must refuse at the digest")


def _qualified_fixture(tmp_path):
    """A minimal bound preparation the qualifier accepts."""
    from prismaquant.production_weight_cache import ProductionWeightCache

    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(tmp_path),
                                  metadata={"schema": "prismaquant.test.pwc.v1"})
    pwc_path = tmp_path / "cache.pkl"
    pwc_path.write_bytes(pickle.dumps(cache))
    completion = {"plan_sha256": "p" * 64, "production_cache": _bind(pwc_path)}
    return completion


def test_qualified_cache_unpickles_once_for_both_panel_loads(tmp_path, monkeypatch):
    """``prepare`` and ``freeze`` share one unpickled cache per preparation."""
    from prismaquant import native_moe_panel as panel
    from prismaquant.production_weight_cache import ProductionWeightCache

    _clear_memos()
    completion = _qualified_fixture(tmp_path)
    loads = []
    real_loads = pickle.loads

    class CountingPickle:
        @staticmethod
        def loads(raw):
            loads.append(len(raw))
            return real_loads(raw)

    first = panel._memoized_qualified_cache(completion, CountingPickle,
                                            ProductionWeightCache)
    second = panel._memoized_qualified_cache(completion, CountingPickle,
                                             ProductionWeightCache)
    assert isinstance(first, ProductionWeightCache)
    assert second is first, "the second panel load must reuse the cache object"
    assert len(loads) == 1, "the cache must be unpickled exactly once"


def test_replaced_preparation_reloads_instead_of_reusing(tmp_path):
    """A new PWC under a new receipt is loaded, not mistaken for the old one."""
    from prismaquant import native_moe_panel as panel
    from prismaquant.production_weight_cache import ProductionWeightCache

    _clear_memos()
    completion = _qualified_fixture(tmp_path)
    first = panel._memoized_qualified_cache(completion, pickle, ProductionWeightCache)
    other = ProductionWeightCache(weights={}, levers={}, cache_dir=str(tmp_path),
                                  metadata={"schema": "prismaquant.test.pwc.v1",
                                            "generation": 2})
    pwc_path = Path(completion["production_cache"]["path"])
    pwc_path.write_bytes(pickle.dumps(other))
    completion["production_cache"] = _bind(pwc_path)
    second = panel._memoized_qualified_cache(completion, pickle, ProductionWeightCache)
    assert second is not first
    assert second.metadata["generation"] == 2
    assert json.dumps(completion["production_cache"]["sha256"])  # the new receipt binds
