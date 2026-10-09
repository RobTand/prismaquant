"""One Stage B-shaped compilation-cache row with a measured peak (PQ #2463).

Part of PQ #1091. The cache ROOT/MAX pair declares an operator-chosen
reservation. This row runs real compilation work a Stage B quantum runs --
the served activation quantiser compile and the KDA capture-kernel probe --
under the cache-peak sampler, so the ceiling derives from a recorded peak.

Workload scope: the served NVFP4 RTN activation quantiser
(``format_registry._make_rtn`` with ``torch.compile``) on one fixed tensor,
then the KDA probe (``kda_chunk.probe_digest``) when CUDA is present. Both
write under the routed cache root: inductor writes under
``TORCHINDUCTOR_CACHE_DIR``, Triton writes under ``TRITON_CACHE_DIR``. The
sampler records allocated bytes every ``interval_s``. The receipt binds the
command, the runtime versions, the initial cache state, the samples, the
in-process profile and the PB action identity.

``--cache-root`` holds the four compilation caches (hf, triton, inductor,
xdg) as the launcher routes them. ``--workspace`` is the separate charged
temporary workspace (``PRISMAQUANT_TMPDIR``): never the cache root. ``--out``
is the receipt path. ``--profile-out`` is the cProfile dump path; without
it no in-process profile is written.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import container_cache_peak as peak_mod  # noqa: E402
from prismaquant.io_spans import (  # noqa: E402
    GpuPowerSampler,
    read_proc_io,
    read_proc_status,
    stage_span_log,
)

SCHEMA = "prismaquant.container_cache_row.v1"
#: Fixed workload tensors: one shape, one seed, every run.
TENSOR_SHAPE = (64, 256)
TENSOR_SEED = 2463


def _runtime_versions() -> dict:
    versions = {"schema": SCHEMA}
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["device_name"] = torch.cuda.get_device_name(0)
            versions["device_capability"] = list(
                torch.cuda.get_device_capability(0))
    except ImportError:
        versions["torch"] = None
    try:
        import triton

        versions["triton"] = str(triton.__version__)
    except ImportError:
        versions["triton"] = None
    try:
        import transformers

        versions["transformers"] = str(transformers.__version__)
    except ImportError:
        versions["transformers"] = None
    return versions


def _run_compilation_workload() -> dict:
    """Compile and run the served quantiser; probe the KDA kernel on CUDA."""
    import torch

    from prismaquant.format_registry import _make_rtn

    torch.manual_seed(TENSOR_SEED)
    quantise = _make_rtn("fp4_e2m1", 16)
    tensor = torch.randn(*TENSOR_SHAPE)
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    started = time.time()
    result = quantise(tensor)
    quantise_s = time.time() - started
    workload = {"quantiser": "fp4_e2m1/g16", "shape": list(TENSOR_SHAPE),
                "seed": TENSOR_SEED, "quantise_s": round(quantise_s, 3),
                "output_mean": float(result.float().mean()),
                "device": str(result.device)}
    try:
        from prismaquant.kernels import kda_chunk
    except ImportError as exc:
        workload["kda_probe"] = {"status": "missing", "error": str(exc)}
        return workload
    if not torch.cuda.is_available():
        workload["kda_probe"] = {"status": "skipped_no_cuda"}
        return workload
    started = time.time()
    digest = kda_chunk.probe_digest("cuda")
    workload["kda_probe"] = {"status": "ran", "sha256": digest["sha256"],
                             "shape": digest["shape"],
                             "probe_s": round(time.time() - started, 3),
                             "compiled": sorted(kda_chunk.compiled_kernels())}
    return workload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, default=None)
    parser.add_argument("--interval-s", type=float,
                        default=peak_mod.SAMPLE_INTERVAL_S)
    args = parser.parse_args(argv)
    cache_root = args.cache_root
    workspace = args.workspace
    if workspace == cache_root or cache_root in workspace.parents:
        parser.error("--workspace must be separate from --cache-root")
    for name in ("HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "XDG_CACHE_HOME"):
        value = os.environ.get(name)
        if value is None:
            parser.error(f"{name} is not set; launch under the charged cache root")
    workspace.mkdir(parents=True, exist_ok=True)
    if args.profile_out is not None:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    # The initial inventory is captured BEFORE the sampler starts and before
    # the workload runs: a row that starts on a warm root must record it.
    # A receipt whose initial totals equal the final ones on a cold row is
    # correct only when the root starts empty; the regression test pins a
    # warm start where they differ.
    initial_state = peak_mod.describe_initial_state(cache_root)
    command = [sys.executable, "-m", "tools.measure_container_cache_row",
               "--cache-root", str(cache_root), "--workspace", str(workspace),
               "--out", str(args.out)]
    sampler = peak_mod.CachePeakSampler(cache_root, interval_s=args.interval_s)
    power = GpuPowerSampler().start()
    spans = stage_span_log("container-cache-row", power_sampler=power)
    io_before = read_proc_io()
    wall_before = time.time()
    profile = cProfile.Profile()
    failure = None
    workload = {}
    with spans.span("row"), sampler:
        try:
            profile.enable()
            workload = _run_compilation_workload()
            profile.disable()
        except BaseException as exc:  # noqa: BLE001 - receipt records it
            failure = f"{type(exc).__name__}: {exc}"
            try:
                profile.disable()
            except RuntimeError:
                pass
            raise
        finally:
            sampler.stop()
    power_summary = power.stop()
    measurement = sampler.result(incomplete_scan=failure is not None)
    profile_io = io.StringIO()
    pstats.Stats(profile, stream=profile_io).strip_dirs().sort_stats(
        "tottime").print_stats(30)
    profile_text = profile_io.getvalue()
    if args.profile_out is not None:
        args.profile_out.write_text(profile_text)
    receipt = {
        "schema": SCHEMA,
        "command": command,
        "host": socket.gethostname().split(".")[0],
        "runtime": _runtime_versions(),
        "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
        "cache_env": {name: os.environ.get(name) for name in (
            "HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "XDG_CACHE_HOME", "PRISMAQUANT_TMPDIR",
            "PRISMAQUANT_CONTAINER_CACHE_ROOT",
            "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES")},
        "initial_state": initial_state,
        "measurement": measurement,
        "workload": workload if failure is None else {"failure": failure},
        "profile_top": profile_text.splitlines()[:40],
        "proc_io_delta": _delta_io(io_before, read_proc_io()),
        "peak_rss_kib": _peak_rss_kib(),
        "wall_s": round(time.time() - wall_before, 3),
        "power": {"summary": power_summary,
                  "samples": list(power.samples), "times": list(power.times),
                  "error": power.error},
        "byte_convention": ("allocated bytes count 512 * st_blocks for each "
                            "unique device/inode, directories included; "
                            "apparent bytes sum st_size"),
        "failure": failure,
    }
    digest = peak_mod.write_receipt(args.out, receipt)
    print(json.dumps({"receipt": str(args.out), "sha256": digest,
                      "peak_allocated_bytes": measurement["peak_allocated_bytes"],
                      "valid": measurement["valid"]}, sort_keys=True))
    return 0


def _delta_io(before: dict, after: dict) -> dict:
    return {key: (after.get(key) - before.get(key)
                  if isinstance(after.get(key), int)
                  and isinstance(before.get(key), int) else None)
            for key in set(before) | set(after)}


def _peak_rss_kib() -> int | None:
    status = read_proc_status()
    peak = status.get("VmHWM")
    return None if peak is None else peak // 1024


if __name__ == "__main__":
    raise SystemExit(main())
