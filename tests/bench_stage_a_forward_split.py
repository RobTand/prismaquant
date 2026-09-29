"""The forward capture's wall, compute and exposed wait: one owner against N quanta (PQ #738).

The suite does not collect this file (it is ``bench_*``). PrismaBuild runs it
by name:

    python -m pytest -rA tests/bench_stage_a_forward_split.py

``test_chain_resume``'s five-layer dense model captures ``ROWS`` calibration
rows (one partition each, read in windows of two) through
``run_adjoint_capture_core``, once as the single owner and once as a forward
split of ``QUANTA`` quanta after its prep, then the join. Each layer forward
of one partition sleeps ``COMPUTE_S`` as a stand-in for the GPU step (a sleep
releases the GIL as a CUDA synchronize does), so compute has a known size.

For every role the bench records, on the calling thread:

* ``wall_s``: the role's own wall seconds, up to its receipt (the single
  owner's up to its chain state, the forward segment a split replaces).
* ``compute_s``: seconds inside the layer forwards and the tail logits.
* ``exposed_wait_s``: ``wall_s - compute_s``, the time the role spends not
  computing: staging, entry writes and reads, checkpoint sealing, records.
* ``installs``: source layer installs, each a whole layer's weights read. A
  single owner installs each layer once; each quantum installs every layer,
  so a split reads the model once per quantum. That is the trade: duplicated
  weight reads for idle GPUs.
* the produced-output wait counters of the boundary owner (zero here: the
  fixture binds no PrismaBuild publication) and cProfile's top functions.

The critical path of a split is the prep, the slowest quantum and the join,
when each quantum runs on its own box; the serial sum is what one box would
spend. The result prints as one JSON line and lands in ``PQ_FORWARD_BENCH_OUT``
(a directory) when set. CPU wall on a toy model is overhead-dominated; the
real-scale numbers come from the pilot rows and their Netdata series.
"""
from __future__ import annotations

import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import time

import torch

from prismaquant.joint_adjoint_checkpoints import adjoint_space
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.stage_a_forward_split import join_forward_split
from prismaquant.stage_a_chain_split import even_ranges

import test_stage_a_chain_resume as resume
from test_stage_a_chain_resume import _run

ROWS = 16
WINDOW = 2
QUANTA = (2, 4, 8)
COMPUTE_S = 0.01


def _calibration():
    torch.manual_seed(738)
    return torch.randint(1, 5, (ROWS, 4))


def _execution(root):
    from test_joint_cost_quantum_runtime import _boundary_policy, _execution as base

    execution = base(root)
    policy = _boundary_policy(root / "boundaries", window=WINDOW)
    policy["max_resident_bytes"] = resume.CAP
    policy["max_artifact_bytes"] = 1 << 28
    policy["max_auxiliary_bytes"] = 1 << 24
    execution["boundary_storage"] = policy
    return execution


class _Meter:
    """Counts installs and times the compute of one role's runner."""

    def __init__(self):
        self.installs = 0
        self.compute_s = 0.0

    def runner(self):
        runner = resume._dense_runner()
        install, call, tail = runner.context.install, runner._call, runner.tail_logits

        def counted(layer, **kw):
            self.installs += 1
            return install(layer, **kw)

        def timed_call(*args, **kw):
            started = time.perf_counter()
            try:
                time.sleep(COMPUTE_S)
                return call(*args, **kw)
            finally:
                self.compute_s += time.perf_counter() - started

        def timed_tail(*args, **kw):
            started = time.perf_counter()
            try:
                return tail(*args, **kw)
            finally:
                self.compute_s += time.perf_counter() - started

        runner.context.install = counted
        runner._call = timed_call
        runner.tail_logits = timed_tail
        return runner


def _profiled(role, call):
    meter = _Meter()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    try:
        receipt = call(meter)
    finally:
        profiler.disable()
    wall = time.perf_counter() - started
    stats = pstats.Stats(profiler)
    top = sorted(((row[3], f"{Path(key[0]).name}:{key[1]}:{key[2]}", row[1])
                  for key, row in stats.stats.items()), reverse=True)[:12]
    telemetry = (receipt or {}).get("retention", {}).get("telemetry", {})
    return {"role": role, "wall_s": wall, "compute_s": meter.compute_s,
            "exposed_wait_s": wall - meter.compute_s, "installs": meter.installs,
            "produced_waits": {key: telemetry.get(key) for key in (
                "produced_group_stage_wait_s", "produced_group_release_wait_s",
                "produced_group_ahead_wait_s", "produced_group_credit_waits")},
            "top_cumulative": [{"function": name, "cumulative_s": cumulative,
                                "calls": calls} for cumulative, name, calls in top]}


def _single_owner(root, monkeypatch):
    """The single owner up to its chain state: the segment a forward split replaces."""
    import prismaquant.stage_a_chain_resume as chain_resume

    write = chain_resume.write_chain_state

    def stopped(space, document):
        write(space, document)
        raise resume._Interrupted("the forward segment ends at the chain state")

    def owner(meter):
        with monkeypatch.context() as patch:
            patch.setattr(chain_resume, "write_chain_state", stopped)
            try:
                _run(root, monkeypatch, calib=_calibration(), execution=_execution(root),
                     runner_factory=meter.runner)
            except resume._Interrupted:
                pass
        return None

    return _profiled("single-owner", owner)


def _split(root, monkeypatch, quanta):
    ranges = even_ranges(ROWS, WINDOW, quanta)
    kw = {"calib": _calibration(), "execution": _execution(root)}
    rows = [_profiled("prep", lambda meter: _run(
        root, monkeypatch, forward_split={"role": "prep", "ranges": ranges},
        runner_factory=meter.runner, **kw))]
    for samples in ranges:
        rows.append(_profiled(f"quantum {samples[0]}:{samples[1]}", lambda meter, s=samples: _run(
            root, monkeypatch, forward_split={"role": "quantum", "samples": s},
            runner_factory=meter.runner, **kw)))
    started = time.perf_counter()
    join_forward_split(adjoint_space(root))
    join = {"role": "join", "wall_s": time.perf_counter() - started}
    quanta_rows = rows[1:]
    return {
        "quanta": quanta, "ranges": ranges, "rows": [*rows, join],
        "critical_path_s": rows[0]["wall_s"] + max(r["wall_s"] for r in quanta_rows)
        + join["wall_s"],
        "serial_sum_s": sum(r["wall_s"] for r in rows) + join["wall_s"],
        "max_quantum_exposed_wait_s": max(r["exposed_wait_s"] for r in quanta_rows),
        "installs": sum(r["installs"] for r in quanta_rows),
        "chain_state_sha256": hashlib.sha256(
            chain_state_path(adjoint_space(root)).read_bytes()).hexdigest(),
    }


def test_forward_split_profile(tmp_path, monkeypatch):
    single = _single_owner(tmp_path / "single", monkeypatch)
    # The single owner stopped after writing its chain state: its bytes are
    # the reference each split must reproduce.
    want = hashlib.sha256(chain_state_path(adjoint_space(tmp_path / "single"))
                          .read_bytes()).hexdigest()
    (tmp_path / "single").rename(tmp_path / "single-done")
    splits = []
    for quanta in QUANTA:
        root = tmp_path / "single"
        splits.append(_split(root, monkeypatch, quanta))
        root.rename(tmp_path / f"split-{quanta}")
    result = {"schema": "prismaquant.bench.stage_a_forward_split.v1",
              "rows": ROWS, "window": WINDOW, "layers": 5, "compute_s_per_forward": COMPUTE_S,
              "single_owner": single, "single_owner_chain_state_sha256": want,
              "splits": splits}
    for split in splits:
        assert split["chain_state_sha256"] == want, f"{split['quanta']} quanta"
    line = json.dumps(result, sort_keys=True)
    print(line)
    out = os.environ.get("PQ_FORWARD_BENCH_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "forward-split.json").write_text(line + "\n")
