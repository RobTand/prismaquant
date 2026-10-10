"""Paired before/after measure for PQ #2039 staged-copy cost.

Run one admitted action on a GB10 worker. Arm ``before`` stages each unit
through one private ``pinned.copy_(weight)``. Arm ``after`` adopts a pinned
source tensor with no second host copy. Both arms use the same units, shapes,
dtype, live bytes, and the same prepare/launch/settle path. The driver keeps
local timing, in-process Torch profiles, power samples, and Netdata raw
responses separate. Clock alignment and energy stay on HOLD.
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request


def _power_samples(stop, out, errors):
    while not stop.is_set():
        try:
            raw = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if raw.returncode == 0 and raw.stdout.strip():
                out.append(float(raw.stdout.strip().splitlines()[0]))
            else:
                errors.append("nvidia-smi returned no power value")
        except Exception as exc:  # noqa: BLE001 - counted, never raised
            errors.append(str(exc))
        stop.wait(0.25)


def _netdata_raw():
    """Fetch one local Netdata response, or record why it stays on HOLD."""
    for url in ("http://localhost:19999/api/v1/info",
                "http://127.0.0.1:19999/api/v1/info"):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read(65536)
            return {"url": url, "status": response.status,
                    "bytes": len(body), "ok": True}
        except Exception as exc:  # noqa: BLE001 - HOLD, never raised
            last = str(exc)
    return {"url": "http://localhost:19999/api/v1/info",
            "ok": False, "error": last}


def _profile_top(table, names):
    out = {}
    for event in table:
        name = getattr(event, "key", "")
        if name in names:
            entry = out.setdefault(name, {"count": 0, "self_cpu_seconds": 0.0})
            entry["count"] += int(getattr(event, "count", 1))
            entry["self_cpu_seconds"] += float(getattr(
                event, "self_cpu_time_total", 0.0)) / 1e6
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch
    from prismaquant import tessera_campaign as campaign

    if not torch.cuda.is_available():
        print(json.dumps({"skipped": True,
                          "reason": "no CUDA device on this worker"}))
        return 0
    torch.cuda.set_device(0)

    units = {f"u{i}": {"source_tensor": f"w{i}",
                       "rows": args.rows, "cols": args.cols}
             for i in range(args.units)}
    shape = (args.rows, args.cols)
    # One pinned pool (the staged-reader result) and one pageable pool (the
    # legacy reader result) hold the same bytes. Live CUDA views match them,
    # so every pass takes the equal path through the device compare.
    with torch.no_grad():
        pageable = [torch.full(shape, 2.0, dtype=torch.bfloat16)
                    for _ in range(args.units)]
        pinned_pool = [t.pin_memory() for t in pageable]
        live = [torch.full(shape, 2.0, dtype=torch.bfloat16,
                           device="cuda") for _ in range(args.units)]
    for tensor in pinned_pool:
        assert tensor.is_pinned() and tensor.is_contiguous()
    names = sorted(units)

    original_read = campaign._read_projected_unit
    staging_copies = {"count": 0}
    real_copy = torch.Tensor.copy_

    def _count(target, source, *a, **k):
        staging_copies["count"] += 1
        return real_copy(target, source, *a, **k)

    torch.Tensor.copy_ = _count
    try:
        def run_arm(kind):
            pools = pinned_pool if kind == "after" else pageable
            idx = {"i": 0}
            released = {"n": 0}

            def read(name, unit, **kwargs):
                if kind == "after":
                    assert kwargs.get("pinned_host") is True
                pos = names.index(name)
                base = pools[pos]
                if kind == "after":
                    # The staged reader owns this pinned buffer. Hand out a
                    # fresh alias per pass so release accounting stays exact.
                    # No host copy happens here.
                    view = base
                else:
                    view = base
                return view, lambda: released.__setitem__("n", released["n"] + 1)

            campaign._read_projected_unit = read
            staging_copies["count"] = 0
            torch.cuda.synchronize()
            stop = threading.Event()
            powers, errors = [], []
            sampler = threading.Thread(target=_power_samples,
                                       args=(stop, powers, errors))
            sampler.start()
            calls, walls = [], []
            torch.cuda.synchronize()
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            profiler.start()
            try:
                for _ in range(args.passes):
                    start = time.perf_counter()
                    checks = []
                    for pos, name in enumerate(names):
                        check = campaign._prepare_device_projected_check(
                            name, units[name], live_shape=shape,
                            live_dtype=torch.bfloat16, model_path="unused",
                            source={}, release_source_pages=False,
                            source_authentication=None)
                        if check.pinned is not None:
                            campaign._launch_prepared_projected_check(
                                check, live[pos])
                            check.pinned = None
                        checks.append(check)
                    outcomes = campaign._settle_projected_unit_checks(checks)
                    torch.cuda.synchronize()
                    walls.append(time.perf_counter() - start)
                    calls.append(len(names))
                    assert all(o is None for o in outcomes), outcomes[:1]
            finally:
                profiler.stop()
                stop.set()
                sampler.join()
            table = profiler.key_averages()
            top = _profile_top(table, {"aten::copy_", "aten::_to_copy",
                                       "aten::ne", "aten::any",
                                       "cudaMemcpyAsync", "cudaLaunchKernel"})
            total_copy = top.get("aten::copy_", {}).get("self_cpu_seconds", 0.0)
            return {
                "kind": kind,
                "passes": args.passes,
                "units_per_pass": args.units,
                "staging_copy_calls": staging_copies["count"],
                "wall_seconds_mean": statistics.fmean(walls),
                "wall_seconds_median": statistics.median(walls),
                "wall_seconds": walls,
                "profile_top_self_cpu": top,
                "aten_copy_self_cpu_seconds": total_copy,
                "power_mean_w": (statistics.fmean(powers) if powers else None),
                "power_max_w": (max(powers) if powers else None),
                "power_samples": len(powers),
                "power_errors": len(errors),
            }
        # ABBA order cancels linear drift between the two arms.
        arms = [run_arm(k) for k in ("before", "after", "after", "before")]
    finally:
        campaign._read_projected_unit = original_read
        torch.Tensor.copy_ = real_copy

    before_walls = arms[0]["wall_seconds"] + arms[3]["wall_seconds"]
    after_walls = arms[1]["wall_seconds"] + arms[2]["wall_seconds"]
    before_copy = arms[0]["staging_copy_calls"] + arms[3]["staging_copy_calls"]
    after_copy = arms[1]["staging_copy_calls"] + arms[2]["staging_copy_calls"]
    before_prof = sum(a["aten_copy_self_cpu_seconds"] for a in (arms[0], arms[3]))
    after_prof = sum(a["aten_copy_self_cpu_seconds"] for a in (arms[1], arms[2]))
    result = {
        "schema": "prismaquant.pq2039_paired_copy_measure.v1",
        "issue": 2039,
        "units": args.units,
        "shape": list(shape),
        "dtype": "bfloat16",
        "order": ["before", "after", "after", "before"],
        "arms": arms,
        "paired": {
            "before_wall_mean_s": statistics.fmean(before_walls),
            "after_wall_mean_s": statistics.fmean(after_walls),
            "before_staging_copies": before_copy,
            "after_staging_copies": after_copy,
            "before_aten_copy_self_cpu_s": before_prof,
            "after_aten_copy_self_cpu_s": after_prof,
        },
        "power_envelope_w": 140.0,
        "netdata": _netdata_raw(),
        "reservations": {"cpus": os.environ.get("PB_CPUS", "4"),
                         "mem_gb": os.environ.get("PB_MEM_GB", "16"),
                         "gpu_memory_gb": os.environ.get("PB_GPU_MEM_GB", "8"),
                         "host_class": "gb10"},
        "hold": ["clock alignment across hosts stays on HOLD",
                 "energy integration and work-per-joule stay on HOLD; "
                 "watts are descriptive samples against the 140 W envelope",
                 "no full-campaign, export, serving, KL, or bpp claim"],
    }
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
