"""A chain quantum waits for a range only under a phase that stages it (PQ #1166).

PrismaBuild stages a row's ranges in the order of its sealed phase list, one
phase ahead of the phase the row last reported. A layer quantum with a chain
installs each chain layer under ``chain-NNN-source`` and then prefetches the
next layer of its install order. Under operator windows the chain step used
to settle that prefetch before it reported ``chain-NNN-bound``. For the last
chain layer the next layer is the quantum's own layer, whose source is staged
by ``own-LLL-source``, two phases later. The consumer then blocked, under
``chain-NNN-source``, on bytes the plan orders after ``chain-NNN-bound``.

These tests drive the real ``run_layer_quantum_core`` and record every point
where the consumer thread waits for staged bytes: a source ``install``, a
``settle_prefetched_layers`` call, and an exact boundary or checkpoint read.
Each wait's ranges are mapped to the first sealed phase that stages them.
That phase must be the consumer's current phase or an earlier one. A
prefetch's own read, which runs on the loader thread while the consumer
works, is not a wait and is not checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def _record_waits(monkeypatch, setup):
    """Log every consumer wait with the phase current when it began.

    Installed before the harness's first drive, so the harness's own
    install and prefetch seams delegate to these. Returns the log, a list
    of ``(kind, detail, paths, phase)``.
    """
    import prismaquant.joint_cost_quantum as qc
    import prismaquant.perturbed_x_cache as pxc

    assert "_orig_seams" not in setup, "install before the first drive"
    runner = setup["runner"]
    context = runner.context
    sources = _layer_source_paths(setup["parent"])
    live = {}
    original_enter = qc.QuantumProgress._enter

    def enter(self, name):
        live["progress"] = self
        return original_enter(self, name)

    monkeypatch.setattr(qc.QuantumProgress, "_enter", enter)

    def phase():
        progress = live.get("progress")
        return None if progress is None else progress._phase

    log = []
    original_install = context.install
    original_settle = context.settle_prefetched_layers

    def install(layer, *, require_prefetched=False, prefetch_following=True):
        log.append(("install", int(layer), sources[int(layer)], phase()))
        return original_install(
            layer, require_prefetched=require_prefetched,
            prefetch_following=prefetch_following)

    def settle(layers, *, retry_availability=False):
        for layer in layers:
            log.append(("settle", int(layer), sources[int(layer)], phase()))
        return original_settle(layers, retry_availability=retry_availability)

    monkeypatch.setattr(context, "install", install)
    monkeypatch.setattr(context, "settle_prefetched_layers", settle)
    original_prefetch = pxc.prefetch_exact_activation_cache_entries

    def prefetch(references, **kwargs):
        paths = tuple(str(ref.path) for ref in references)
        log.append(("exact-read", len(paths), paths, phase()))
        return original_prefetch(references, **kwargs)

    monkeypatch.setattr(pxc, "prefetch_exact_activation_cache_entries", prefetch)
    return log


def _layer_source_paths(parent):
    """Each layer's source files, from the parent manifest's ``layer-L`` phases."""
    phases = parent["annotations"]["phases"]
    ends = [(phase["name"], int(phase["cumulative_bytes"])) for phase in phases]
    by_layer, offset, index = {}, 0, 0
    for entry in parent["entries"]:
        offset += int(entry["bytes"])
        while offset > ends[index][1]:
            index += 1
        name = ends[index][0]
        if name.startswith("layer-"):
            by_layer.setdefault(int(name.split("-")[1]), []).append(
                str(entry["path"]))
    return {layer: tuple(paths) for layer, paths in by_layer.items()}


def _late_waits(log, manifest):
    """Waits whose ranges no phase up to the current one stages."""
    names = [phase["name"] for phase in manifest["read_plan"]["phases"]]
    entries = manifest["entries"]
    first = {}
    for index, phase in enumerate(manifest["read_plan"]["phases"]):
        for entry_index in phase["entry_indices"]:
            first.setdefault(str(entries[entry_index]["path"]), index)
    late = []
    for kind, detail, paths, phase in log:
        assert phase in names, (kind, detail, phase)
        current = names.index(phase)
        missing = [path for path in paths if path not in first]
        assert not missing, (
            f"{kind} {detail} waited on ranges the sealed manifest does not "
            f"stage: {missing}")
        owners = sorted({first[path] for path in paths})
        if owners and owners[-1] > current:
            late.append((kind, detail, phase, names[owners[-1]]))
    return late


def _assert_waits_in_phase(log, manifest, layer):
    assert [entry for entry in log if entry[0] == "install"], log
    assert any(entry[0] == "settle" and entry[1] == layer for entry in log), (
        f"no settle of layer {layer} was recorded; the test would pass "
        "without exercising the chain step's prefetch")
    late = _late_waits(log, manifest)
    assert not late, (
        "the consumer waited on ranges a later phase stages (PQ #1166): "
        + "; ".join(f"{kind} of {detail} under {phase}, staged by {owner}"
                    for kind, detail, phase, owner in late))


def test_a_chain_quantum_waits_only_on_ranges_its_phase_stages(
        tmp_path, monkeypatch):
    """Windowed replay, layer 0 with chain [1]."""
    import test_quantum_executable_readset as harness

    setup = harness._acceptance_setup(tmp_path, monkeypatch)
    record = setup["records"]["layer-000"]
    assert list(record["adjoint"]["chain_layers"]) == [1]
    assert setup["runner"].prefetch_lookahead >= 1
    log = _record_waits(monkeypatch, setup)
    _events, manifest, _payload, _resolved = harness._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False)
    _assert_waits_in_phase(log, manifest, layer=0)


def test_a_spill_chain_quantum_waits_only_on_ranges_its_phase_stages(
        tmp_path, monkeypatch):
    """The production regime: spill replay, layer 0 with chain [1]."""
    import test_quantum_executable_readset as harness
    import test_stageb_one_pass_spill as one_pass
    import prismaquant.joint_replay_spill as spill_mod

    monkeypatch.setattr(harness, "_expert_fixture", one_pass._bf16_expert_fixture)
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    setup = harness._acceptance_setup_expert(tmp_path, monkeypatch)
    record = setup["records"]["layer-000"]
    assert list(record["adjoint"]["chain_layers"]) == [1]
    assert setup["runner"].prefetch_lookahead >= 1
    monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(one_pass._spill_root(tmp_path)))
    monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(1 << 30))
    log = _record_waits(monkeypatch, setup)
    _events, manifest, _payload, _resolved = harness._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False,
        replay_mode="spill")
    _assert_waits_in_phase(log, manifest, layer=0)
