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
