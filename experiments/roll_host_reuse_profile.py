"""Bounded CPU allocation evidence for #2097; CUDA copies/events are simulated.

Use only through PB --measurement. This profiles real CPU allocations and
copies through _RollPipeline, replacing pinned allocation and CUDA events.
It makes no device, pinned-page, wall-time, residency or energy claim.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant.joint_adjoint_checkpoints import _RollPipeline


class CpuEvent:
    def record(self, stream):
        pass

    def synchronize(self):
        pass


def arm(out, *, reuse, iteration):
    allocate = torch.empty
    allocations = []
    digest = hashlib.sha256()
    name = f"{iteration}-{'reuse' if reuse else 'allocating'}"
    # Four individually compact 1 MiB rows, 32 fixed groups per arm.
    gradient = torch.arange(4 * 262144, dtype=torch.float32).reshape(4, 1, 262144)

    def empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        result = allocate(*args, **kwargs)
        if pinned:
            allocations.append(result.numel() * result.element_size())
        return result

    def roll(row, batch, probe):
        digest.update(memoryview(row.view(torch.uint8).numpy()))

    profiler = cProfile.Profile()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "empty", empty)
        patch.setattr(torch.cuda, "Event", CpuEvent)
        patch.setattr(torch.cuda, "current_stream", lambda device: None)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True) as trace:
            profiler.enable()
            pipeline = _RollPipeline(roll, device="cuda", roll_may_keep=False,
                                     reuse_host_buffers=reuse)
            for step in range(32):
                pipeline.submit(gradient, list(range(step * 4, step * 4 + 4)), step % 4)
            pipeline.drain()
            profiler.disable()
        # Keep the banks alive through profiler exit; lifetime is part of
        # the allocation corpus rather than an allocator-cache inference.
        del pipeline
    profiler.dump_stats(str(out / f"{name}.pstats"))
    trace.export_chrome_trace(str(out / f"{name}.trace.json"))
    empty_events = [event for event in trace.events() if event.name == "aten::empty"]
    report = dict(arm=name, reuse=reuse, groups=32, rows=128,
                  allocation_calls=len(allocations), allocated_bytes=sum(allocations),
                  profiler_empty_calls=len(empty_events),
                  profiler_empty_allocated_bytes=sum(max(0, event.cpu_memory_usage)
                                                     for event in empty_events),
                  output_sha256=digest.hexdigest())
    (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    with (args.out / "netdata.jsonl").open("x") as handle:
        writer = NetdataWriter(handle)
        for iteration, reuse in enumerate((False, True, False, True)):
            for phase in ("before", "after"):
                if phase == "after":
                    reports.append(arm(args.out, reuse=reuse, iteration=iteration))
                for host in ("sparky.lan", "sparklina.lan"):
                    record = sample_netdata(host)
                    writer.write({**record, "arm": iteration, "phase": phase})
    if len({report["output_sha256"] for report in reports}) != 1:
        raise RuntimeError("reusable rows changed the exact output bytes")
    for report in reports:
        expected = 8 if report["reuse"] else 128
        if report["allocation_calls"] != expected or report["profiler_empty_calls"] != expected:
            raise RuntimeError(f"allocation profiler and oracle disagree: {report}")
    artifacts = [dict(path=str(path), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in sorted(args.out.iterdir()) if path.is_file()]
    print(json.dumps(dict(scope="cpu_allocations_with_simulated_cuda_only", reports=reports,
                          artifacts=artifacts, affinity=sorted(os.sched_getaffinity(0)))))


if __name__ == "__main__":
    main()
