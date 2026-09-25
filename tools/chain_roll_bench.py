"""Chain roll bench: one Stage A render-free layer roll on real R13 data.

PQ #1162 moves the chain roll's host work (the input cotangent's copy to the
host, the next operand's staging and the entry writes) off the GPU's critical
path. The claim is a delta in time per layer and in the main thread's share
outside the backward and the GPU wait, so this tool measures both trees on
the same bytes, in one admitted action, on one box, alternating the arms.

The workload is the production roll, not a model of it:

* the R13 source model, built as Stage A builds it
  (``build_streamed_causal_lm`` with the plan's source derivative and the
  run's prefetch settings), with only layer ``--layer`` installed;
* the calibration rows the plan names, partitioned as Stage A partitions
  them;
* a fresh writable ``StreamedBoundaryArtifacts`` owner in local scratch,
  under the plan's boundary storage policy, that borrows R13's boundary
  entries at the layer and checkpoint ``layer + 1``'s cotangent plane, as a
  chain seed does (``authorize_forward_inputs`` and
  ``authorize_seed_checkpoint``);
* ``render_free_layer_roll`` over the first ``--batches`` batches in the
  campaign regime (batch 4, probes fused), with Stage A's ``roll``: each row
  is written through ``storage.write(previous=...)``. The head arm passes
  ``roll_may_keep=False`` as Stage A does; a tree without the parameter is
  called without it.

What differs from a campaign quantum: the entries are written to the owner's
own directory in local scratch, not to the produced-output spool, so there is
no export to the pool behind them, and the roll stages no next layer
(``then=None``). Both arms run the same way.

Subcommands:

* ``manifest`` (host, before submission) writes the PrismaBuild data
  manifest the action declares: the head tensors and layer ``--layer``'s
  tensors as byte ranges of the source shards, the calibration rows, and the
  borrowed entries in the order the roll reads them.
* ``spec`` (host, before submission) prints the campaign container spec:
  the R13 spec plus the scratch and py-spy mounts, without the produced
  spool.
* ``host`` (the action's command, on the worker) extracts the base arm's
  ``prismaquant`` and ``tools`` from the action's own Git snapshot into
  scratch, then runs ``drive`` in the campaign container.
* ``drive`` (in the container) runs each repeat's arms in alternating order,
  each as one ``child`` process under ``py-spy record``.
* ``child`` (in the container) builds, installs, rolls once, digests the
  rolled payloads after the timed window, and writes one result record.
* ``analyze`` (host, after) reports time per layer, the main thread's
  shares from the py-spy samples, and whether both arms wrote the same
  payloads.

Reads only the declared inputs; writes under ``--scratch`` and ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "CHAIN-ROLL-BENCH "
HEAD_EXCLUDE = ("model.visual.",)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_json(path, sha256):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path}: sha256 {digest} is not {sha256}")
    return json.loads(raw)


def slice_records(slice_doc, *, layer, batches):
    """R13's boundary entries at ``layer`` and plane ``layer + 1``, by batch."""
    checkpoint = slice_doc["checkpoint"]
    if int(checkpoint["boundary"]) != layer + 1:
        raise SystemExit(f"slice checkpoint is boundary {checkpoint['boundary']}, "
                         f"not {layer + 1}")
    rows = {}
    for record in slice_doc["boundary_entries"][str(layer)]:
        batch = int(record["metadata"]["identity"]["coordinates"]["batch"])
        if batch < batches:
            rows[batch] = record
    plane = {}
    for record in checkpoint["activation_entries"]:
        coordinates = record["metadata"]["identity"]["coordinates"]
        if int(coordinates["batch"]) < batches:
            plane[int(coordinates["probe"]), int(coordinates["batch"])] = record
    if sorted(rows) != list(range(batches)):
        raise SystemExit(f"slice holds boundary batches {sorted(rows)[:4]}..., "
                         f"not 0..{batches - 1}")
    probes = sorted({probe for probe, _ in plane})
    if len(plane) != len(probes) * batches:
        raise SystemExit("slice plane is incomplete for the requested batches")
    return rows, plane, probes


# -- manifest (host) ------------------------------------------------------------

def _safetensors_spans(model_dir: Path, names) -> list[dict]:
    """Byte ranges of ``names`` and each shard's header, adjacent ranges merged."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard = defaultdict(list)
    for name in names:
        by_shard[index["weight_map"][name]].append(name)
    entries = []
    for shard in sorted(by_shard):
        path = model_dir / shard
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        base = 8 + length
        spans = sorted((base + header[name]["data_offsets"][0],
                        base + header[name]["data_offsets"][1])
                       for name in by_shard[shard])
        # The header too: the loader reads it before any tensor.
        spans.insert(0, (0, base))
        merged = [list(spans[0])]
        for start, stop in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        entries += [{"path": str(path), "offset": start, "bytes": stop - start,
                     "sha256": None} for start, stop in merged]
    return entries


def cmd_manifest(args) -> int:
    plan = load_verified_json(args.plan, args.plan_sha256)
    slice_doc = load_verified_json(args.slice, args.slice_sha256)
    model = Path(plan["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    prefix = re.compile(rf"^(?:model\.)?(?:language_model\.)?layers\.{args.layer}\.")
    layer = sorted(name for name in index["weight_map"] if prefix.match(name))
    head = sorted(name for name in index["weight_map"]
                  if ".layers." not in name and not name.startswith(HEAD_EXCLUDE))
    if not layer:
        raise SystemExit(f"no tensors of layer {args.layer} in {model}")
    calibration = Path(plan["calibration_input"]["path"])
    entries = [{"path": str(calibration), "offset": 0,
                "bytes": calibration.stat().st_size, "sha256": None}]
    entries += _safetensors_spans(model, head) + _safetensors_spans(model, layer)
    rows, plane, probes = slice_records(slice_doc, layer=args.layer, batches=args.batches)
    # In the order the fused roll reads them: each window's boundaries,
    # then every probe's cotangents of the same batches.
    for start in range(0, args.batches, args.window):
        group = range(start, min(start + args.window, args.batches))
        records = [rows[b] for b in group] + [plane[p, b] for p in probes for b in group]
        entries += [{"path": r["path"], "offset": 0, "bytes": int(r["file_bytes"]),
                     "sha256": r["sha256"]} for r in records]
    total = sum(entry["bytes"] for entry in entries)
    document = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "mount_prefix": "/mnt/shared",
        "produced_by": {"tool": "tools/chain_roll_bench.py",
                        "issue": "RobTand/prismaquant#1162",
                        "plan_sha256": args.plan_sha256,
                        "slice_sha256": args.slice_sha256},
        "annotations": {"purpose": f"chain roll bench, layer {args.layer}, "
                                   f"{args.batches} batches",
                        "phases": [{"name": "bench", "cumulative_bytes": total}]},
        "entries": entries, "entry_count": len(entries), "total_bytes": total,
    }
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    print(f"{args.out}: {len(entries)} entries, {total / 2**30:.2f} GiB "
          f"(head {len(head)} tensors, layer {len(layer)} tensors, "
          f"{len(rows) + len(plane)} entries); sha256 {_sha256_file(args.out)}")
    return 0


# -- spec (host) -------------------------------------------------------------------

def cmd_spec(args) -> int:
    spec = json.loads(Path(args.base_spec).read_text())
    spool = spec["env"].get("PRISMABUILD_PRODUCED_SPOOL_ROOT")
    spec["env"] = {key: value for key, value in spec["env"].items()
                   if not key.startswith("PRISMABUILD_PRODUCED_SPOOL_")}
    mounts = [m for m in spec["container"]["mounts"] if m["source"] != spool]
    mounts.append({"source": args.scratch, "target": args.scratch, "readonly": False})
    if args.py_spy:
        mounts.append({"source": args.py_spy, "target": "/opt/bench/py-spy",
                       "readonly": True})
    spec["container"]["mounts"] = mounts
    print(json.dumps(spec, sort_keys=True))
    return 0


# -- host (the action's command) ------------------------------------------------------

def cmd_host(args) -> int:
    scratch = Path(args.scratch) / args.label
    base = scratch / "arm-base"
    if base.exists():
        raise SystemExit(f"{base} exists: use a fresh --label")
    base.mkdir(parents=True)
    # ``tools`` too: prismaquant imports a few of its modules
    # (``glm_source_derivative`` reads ``tools.container_runtime_identity``).
    archive = subprocess.run(["git", "-c", "safe.directory=*", "archive", args.base_ref,
                              "prismaquant", "tools"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive, check=True)
    commit = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", args.base_ref],
                            check=True, capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    print(MARKER + json.dumps({"base_ref": args.base_ref, "base_commit": commit,
                               "head_commit": head, "base_tree": str(base)}), flush=True)
    drive = ["python3", "/workspace/tools/chain_roll_bench.py", "drive",
             "--arm", f"base={base}", "--arm", "head=/workspace",
             "--out", str(scratch / "out"), "--scratch", str(scratch),
             *args.drive_args]
    command = [sys.executable, "-m", "tools.tessera_campaign_container",
               "--spec", args.spec, "--", *drive]
    return subprocess.run(command).returncode


# -- drive (in the container) -----------------------------------------------------------

def cmd_drive(args) -> int:
    arms = dict(item.split("=", 1) for item in args.arm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    child_args = ["--plan", args.plan, "--plan-sha256", args.plan_sha256,
                  "--slice", args.slice, "--slice-sha256", args.slice_sha256,
                  "--manifest-sha256", args.manifest_sha256,
                  "--layer", str(args.layer), "--batches", str(args.batches),
                  "--batch-size", str(args.batch_size),
                  "--allowed-tiers", args.allowed_tiers]
    if args.prefetch_override:
        child_args += ["--prefetch-override", args.prefetch_override]
    if not args.fused:
        child_args += ["--probe-major"]
    names = list(arms)
    keep = set(args.keep_arm or ())
    unknown = keep - set(names)
    if unknown:
        raise SystemExit(f"--keep-arm names no arm: {sorted(unknown)}")
    runs = []
    for repeat in range(args.repeats):
        # Two arms alternate; more rotate, so each arm runs in each position.
        order = ((names if repeat % 2 == 0 else names[::-1]) if len(names) <= 2
                 else names[repeat % len(names):] + names[:repeat % len(names)])
        for arm in order:
            label = f"{arm}-r{repeat}"
            env = {**os.environ,
                   "PYTHONPATH": ":".join([arms[arm], *[p for p in os.environ.get(
                       "PYTHONPATH", "").split(":") if p and p != "/workspace"]])}
            command = [sys.executable, "/workspace/tools/chain_roll_bench.py", "child",
                       *child_args, *(["--roll-may-keep"] if arm in keep else []),
                       "--arm", arm, "--label", label,
                       "--scratch", str(Path(args.scratch) / label),
                       "--result", str(out / f"{label}.json")]
            if args.py_spy and Path(args.py_spy).exists():
                command = [args.py_spy, "record", "--rate", str(args.py_spy_rate),
                           "--format", "raw", "--idle", "--nonblocking",
                           "--output", str(out / f"{label}.raw"), "--", *command]
            started = time.time()
            print(f"{MARKER}start {label}: {' '.join(command[:6])} ...", flush=True)
            code = subprocess.run(command, env=env).returncode
            runs.append({"label": label, "arm": arm, "repeat": repeat,
                         "returncode": code, "started_unix": started,
                         "ended_unix": time.time()})
            (out / "drive.json").write_text(json.dumps({"arms": arms, "runs": runs},
                                                       indent=1))
            print(f"{MARKER}end {label}: rc={code} in {time.time() - started:.1f}s",
                  flush=True)
            if code != 0:
                return code
    return 0


# -- child (in the container) -------------------------------------------------------------

class _MemoryWatch:
    """MemAvailable low-water mark and the CUDA reservation, every 0.2 s."""

    def __init__(self, torch):
        from prismaquant.io_spans import MemAvailableFloor

        self.torch = torch
        self._floor = MemAvailableFloor(0.2, name="memwatch").__enter__()

    def stop(self):
        from prismaquant.io_spans import read_proc_status

        self._floor.__exit__(None, None, None)
        status = {key: f"{value // 1024} kB" for key, value in read_proc_status().items()
                  if key in ("VmHWM", "VmRSS")}
        cuda = self.torch.cuda
        baseline, low = self._floor.first["bytes"], self._floor.minimum["bytes"]
        return {"mem_available_baseline_bytes": baseline,
                "mem_available_min_bytes": low,
                "mem_available_drop_bytes": baseline - low,
                "cuda_max_reserved_bytes": cuda.max_memory_reserved(),
                "cuda_max_allocated_bytes": cuda.max_memory_allocated(),
                "process": status}


def _drop_client_cache(resolver, paths) -> int:
    done = 0
    for path in paths:
        staged = resolver.staged_read(Path(path))
        for copy in (() if staged is None else
                     (staged.get("ram_path"), staged.get("stage_path"))):
            if not copy:
                continue
            try:
                fd = os.open(copy, os.O_RDONLY | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                done += 1
            finally:
                os.close(fd)
    return done


def cmd_child(args) -> int:
    import inspect

    import torch

    import prismaquant
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.cost_streaming import (
        StreamedBoundaryArtifacts, build_streamed_causal_lm, normalize_boundary_storage)
    from prismaquant.joint_adjoint_checkpoints import reference_from_record, render_free_layer_roll
    from prismaquant.joint_cost_quantum import _install_with_settlement, _rebuild_batches
    from prismaquant.model_profiles import detect_profile
    from prismaquant.produced_output_spool import plane_partitions
    from prismaquant.residency_map import bind_residency_manifest, residency_resolver
    from prismaquant.sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled
    from prismaquant.stage_a_chain_seed import tensor_payload_sha256
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    phases = {}
    started = time.perf_counter()

    def mark(name):
        phases[name] = round(time.perf_counter() - started, 3)

    result = {"schema": "prismaquant.chain_roll_bench.child.v1", "arm": args.arm,
              "label": args.label, "prismaquant": str(Path(prismaquant.__file__).parent),
              "layer": args.layer, "batches": args.batches,
              "batch_size": args.batch_size, "fused": not args.probe_major}
    keeps = "roll_may_keep" in inspect.signature(render_free_layer_roll).parameters
    result["roll_may_keep_supported"] = keeps
    watch = _MemoryWatch(torch)
    activate_staged_tier_policy(args.allowed_tiers)
    bind_residency_manifest(args.manifest_sha256)
    resolver = residency_resolver()
    if resolver is None:
        raise SystemExit("no residency map in this action's environment")
    plan = load_verified_json(args.plan, args.plan_sha256)
    slice_doc = load_verified_json(args.slice, args.slice_sha256)
    execution = plan["execution"]
    prefetch = (json.loads(Path(args.prefetch_override).read_text())["source_prefetch"]
                if args.prefetch_override else plan["source_prefetch"])
    rows_records, plane_records, probes = slice_records(
        slice_doc, layer=args.layer, batches=args.batches)
    result["dropped_client_cache_files"] = _drop_client_cache(
        resolver, [r["path"] for r in rows_records.values()]
        + [r["path"] for r in plane_records.values()])
    mark("bound")

    ids, _calibration = load_calibration_input(
        plan["calibration_input"]["path"],
        expected_sha256=plan["calibration_input"]["sha256"],
        n_samples=int(execution["n_calib_samples"]), seqlen=int(execution["calib_seqlen"]))
    batch_rows, row_offsets = plane_partitions(
        n_rows=len(ids), probe_microbatch=int(execution.get("probe_microbatch", 0)))
    partitions = [ids[offset:offset + batch_rows] for offset in row_offsets[:args.batches]]
    scratch = Path(args.scratch)
    (scratch / "entries").mkdir(parents=True, exist_ok=True)
    runner = build_streamed_causal_lm(
        plan["model"], device=torch.device("cuda"), dtype=torch.bfloat16,
        offload_folder=str(scratch / "offload"), profile=detect_profile(plan["model"]),
        attn_implementation="eager", source_authentication=None,
        source_derivative=execution.get("source_derivative"), **prefetch)
    mark("built")
    _install_with_settlement(runner, args.layer, operator_windows=None, order=(args.layer,))
    torch.cuda.synchronize()
    mark("installed")
    batches = _rebuild_batches(runner, partitions=partitions, shared_pass={})
    rows = [reference_from_record(rows_records[b]) for b in range(args.batches)]
    plane = {key: reference_from_record(record) for key, record in plane_records.items()}
    for batch_index, batch in enumerate(batches):
        batch.activations_cpu = ([rows[batch_index] if boundary == args.layer else None
                                  for boundary in range(runner.num_layers)]
                                 + [torch.empty(0)])
    policy = normalize_boundary_storage({**execution["boundary_storage"],
                                         "directory": str(scratch / "entries")})
    n_probes = len(probes)
    mark("prepared")
    with StreamedBoundaryArtifacts(policy) as storage:
        storage.bind({"bench": "chain_roll_bench", "label": args.label},
                     n_probes=n_probes)
        storage.authorize_forward_inputs(rows)
        storage.authorize_seed_checkpoint(list(plane.values()), boundary=args.layer + 1,
                                          session=slice_doc["checkpoint"]["session"])
        cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                       for _ in batches] for _ in range(n_probes)]
        storage.watch_auxiliary(batches, cotangents)
        storage.check_auxiliary(batches, cotangents=cotangents)
        grad_outs = [[plane[(probe, batch)] for batch in range(args.batches)]
                     for probe in range(n_probes)]

        def roll(tensor, batch_index, probe_index):
            # Stage A's roll (joint_cost_stage_a.run_adjoint_capture_core),
            # without its digest and checkpoint legs.
            grad_outs[probe_index][batch_index] = storage.write(
                tensor, batch_index=batch_index, boundary_index=args.layer,
                probe_index=probe_index, previous=grad_outs[probe_index][batch_index])

        # Stage A's setting is False (its roll keeps no row); --roll-may-keep
        # measures the pageable-row path on the same tree.
        kwargs = {"roll_may_keep": bool(args.roll_may_keep)} if keeps else {}
        if args.roll_may_keep and not keeps:
            raise SystemExit("--roll-may-keep needs a tree whose roll takes roll_may_keep")
        result["roll_may_keep"] = kwargs.get("roll_may_keep")
        torch.cuda.synchronize()
        result["roll_started_unix"] = time.time()
        roll_started = time.perf_counter()
        backwards = render_free_layer_roll(
            runner, storage=storage, batches=batches, layer=args.layer,
            cotangents=cotangents, n_probes=n_probes, incoming_entries=grad_outs,
            incoming_tensor=None, roll=roll, min_free_gib=float(plan["min_free_gib"]),
            batch_size=args.batch_size, probe_fusion=not args.probe_major, **kwargs)
        torch.cuda.synchronize()
        result["roll_s"] = round(time.perf_counter() - roll_started, 4)
        result["roll_ended_unix"] = time.time()
        result["backwards"] = backwards
        mark("rolled")
        # After the timed window: the payload of every rolled row, which is
        # what both arms must agree on (an entry file also carries its
        # owner's generation, so file digests differ run to run).
        digests = {}
        for probe in range(n_probes):
            for batch in range(args.batches):
                reference = grad_outs[probe][batch]
                payload = torch.load(reference.path, map_location="cpu",
                                     weights_only=True)["inputs"]
                digests[f"{probe}-{batch}"] = tensor_payload_sha256(payload)
                storage.retire(reference)
                grad_outs[probe][batch] = None
        result["payload_sha256"] = digests
        result["plane_sha256"] = hashlib.sha256(json.dumps(
            digests, sort_keys=True).encode()).hexdigest()
        result["storage_telemetry"] = {key: value for key, value in storage.telemetry.items()
                                       if isinstance(value, (int, float))}
        mark("digested")
    result["memory"] = watch.stop()
    result["phases_s"] = phases
    runner.shutdown()
    Path(args.result).write_text(json.dumps(result, indent=1) + "\n")
    print(MARKER + json.dumps({key: result[key] for key in (
        "label", "roll_s", "backwards", "plane_sha256")}), flush=True)
    return 0


# -- analyze (host) ---------------------------------------------------------------------

#: Main-thread leaf classes, first match from the leaf up. The base tree
#: copies out in ``_roll_rows``; the head tree waits on the copy's event in
#: ``_deliver``. ``find_packed_sequence_indices`` line 754 (transformers
#: 5.16.1, the campaign image) tests ``(...).all()`` of a CUDA tensor as a
#: Python bool: a host sync on the device queue, so it is GPU wait too.
CLASSES = (
    ("gpu_wait", re.compile(r"^(_roll_rows|synchronize) |^find_packed_sequence_indices "
                            r"\(transformers/masking_utils\.py:754\)")),
    ("backward", re.compile(r"^(backward|_engine_run_backward) \(torch/autograd/")),
    ("forward", re.compile(r"^isolated_layer ")),
    ("stage", re.compile(r"^(_stack_to_device|_stage_to_device) ")),
    ("write", re.compile(r"^(write_exact_activation_cache_entry|write) ")),
    ("window", re.compile(r"^(prefetched_fused_boundary_windows|prefetched_boundary_batches"
                          r"|__next__|_window|result|wait) ")),
)


def _classify(frames):
    for frame in reversed(frames):
        name = frame.split(" (", 1)[0] + " "
        for label, pattern in CLASSES:
            if pattern.match(frame if label in ("backward", "gpu_wait") else name):
                return label
    return "other"


def _profile_shares(path: Path) -> dict:
    counts = Counter()
    leaves = Counter()
    for line in path.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        frames = stack.split(";")
        if frames[0].startswith("_bootstrap"):
            continue  # a worker thread
        if not any(f.startswith("render_free_layer_roll ") for f in frames):
            continue  # the main thread outside the timed roll
        label = _classify(frames)
        counts[label] += int(count)
        leaves[(label, frames[-1])] += int(count)
    total = sum(counts.values())
    return {"samples": total,
            "shares": {k: round(v / total, 4) for k, v in counts.most_common()} if total else {},
            "top_leaves": [(label, leaf, round(n / total, 4))
                           for (label, leaf), n in leaves.most_common(12)] if total else []}


#: Netdata charts read over each roll window: the GPU's power and the host's
#: CPU, local disk and NFS client load. The stage and RAM tiers are NFS mounts
#: from the file server, so their reads appear on ``nfs.proc4`` and the
#: high-speed links, not on the local disk.
NETDATA_CHARTS = ("system.cpu", "system.io", "disk.nvme0n1", "nfs.proc4",
                  "net.enP2p1s0f0np0", "net.enP2p1s0f1np1",
                  "system.memory_full_pressure")


def _netdata_window(base: str, chart: str, after: float, before: float) -> dict:
    """Mean and max of each dimension over the chart's own points in the window.

    The points are read at the chart's native interval (``points=0``) and
    averaged here. A one-point server-side average is not used: on a chart
    collected every 10 s (the GPU power chart) it can land on a partial slot
    and report a mean far from every sample in the window.
    """
    url = (f"{base}/api/v1/data?chart={chart}&after={int(after)}&before={int(before)}"
           f"&points=0&group=average&format=json&options=abs")
    try:
        with urllib.request.urlopen(url, timeout=20) as reply:
            doc = json.load(reply)
    except Exception as exc:  # noqa: BLE001 - a missing series is reported, not fatal
        return {"error": str(exc)[:160]}
    labels, data = doc.get("labels", []), doc.get("data", [])
    out = {}
    for column, label in enumerate(labels[1:], start=1):
        values = [row[column] for row in data if row[column] is not None]
        if values:
            out[label] = {"mean": round(sum(values) / len(values), 3),
                          "max": round(max(values), 3), "points": len(values)}
    return out


def _netdata(base: str, power_chart: str, envelope_w: float, after: float,
             before: float) -> dict:
    power = _netdata_window(base, power_chart, after, before)
    series = next(iter(power.values()), None) if "error" not in power else None
    watts = series["mean"] if series else None
    row = {"window_s": round(before - after, 1), "gpu_power_w_mean": watts,
           "gpu_power_w_max": series["max"] if series else None,
           "gpu_power_points": series["points"] if series else 0,
           "gpu_power_envelope_fraction": (round(watts / envelope_w, 4)
                                           if watts is not None else None),
           "gpu_joules": (round(watts * (before - after), 1)
                          if watts is not None else None)}
    for chart in NETDATA_CHARTS:
        row[chart] = _netdata_window(base, chart, after, before)
    return row


def cmd_analyze(args) -> int:
    out = Path(args.out)
    drive = json.loads((out / "drive.json").read_text())
    by_arm = defaultdict(list)
    for run in drive["runs"]:
        path = out / f"{run['label']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        raw = out / f"{run['label']}.raw"
        record["profile"] = _profile_shares(raw) if raw.exists() else None
        by_arm[run["arm"]].append(record)
    report = {"arms": {}, "planes": {}}
    for arm, records in by_arm.items():
        rolls = sorted(r["roll_s"] for r in records)
        merged = Counter()
        samples = 0
        seconds = defaultdict(list)
        undersampled = []
        for r in records:
            if not r["profile"]:
                continue
            expected = r["roll_s"] * args.py_spy_rate
            if r["profile"]["samples"] < args.min_sample_fraction * expected:
                # py-spy fell behind: its shares are not a fair sample.
                undersampled.append((r["label"], r["profile"]["samples"], round(expected)))
                continue
            samples += r["profile"]["samples"]
            for label, share in r["profile"]["shares"].items():
                merged[label] += share * r["profile"]["samples"]
                seconds[label].append(share * r["roll_s"])
        shares = {k: round(v / samples, 4) for k, v in merged.most_common()} if samples else {}
        outside = (round(1 - shares.get("backward", 0) - shares.get("gpu_wait", 0), 4)
                   if shares else None)
        report["arms"][arm] = {
            "repeats": len(records), "roll_s": rolls,
            "roll_s_median": rolls[len(rolls) // 2],
            "main_thread_shares": shares, "outside_backward_and_gpu_wait": outside,
            "main_thread_seconds_mean": {label: round(sum(v) / len(v), 2)
                                         for label, v in seconds.items()},
            "undersampled_profiles": undersampled,
            "memory": [r["memory"] for r in records],
            "windows": [(r["label"], r["roll_started_unix"], r["roll_ended_unix"])
                        for r in records],
            "top_leaves": [r["profile"]["top_leaves"] for r in records if r["profile"]],
        }
        if args.netdata:
            host = [{"label": r["label"], **_netdata(
                        args.netdata, args.power_chart, args.envelope_w,
                        r["roll_started_unix"], r["roll_ended_unix"])} for r in records]
            report["arms"][arm]["netdata"] = host
            watts = [h["gpu_power_w_mean"] for h in host if h["gpu_power_w_mean"] is not None]
            report["arms"][arm]["gpu_power_w_mean"] = (round(sum(watts) / len(watts), 2)
                                                       if watts else None)
        report["planes"][arm] = sorted({r["plane_sha256"] for r in records})
    planes = {p for values in report["planes"].values() for p in values}
    report["payloads_identical_across_arms_and_repeats"] = len(planes) == 1
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for arm, row in report["arms"].items():
        print(f"{arm}: roll_s {row['roll_s']} (median {row['roll_s_median']}); "
              f"outside backward+gpu_wait {row['outside_backward_and_gpu_wait']}; "
              f"shares {row['main_thread_shares']}"
              + (f"; GPU {row['gpu_power_w_mean']} W of {args.envelope_w:g} W"
                 if args.netdata else ""))
    print("payloads identical:", report["payloads_identical_across_arms_and_repeats"])
    return 0


# -- CLI ------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def inputs(p, *, manifest=True):
        p.add_argument("--plan", required=True)
        p.add_argument("--plan-sha256", required=True)
        p.add_argument("--slice", required=True)
        p.add_argument("--slice-sha256", required=True)
        if manifest:
            p.add_argument("--manifest-sha256", required=True)
        p.add_argument("--layer", type=int, default=44)
        p.add_argument("--batches", type=int, default=64)

    p = sub.add_parser("manifest")
    inputs(p, manifest=False)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--out", required=True)

    p = sub.add_parser("spec")
    p.add_argument("--base-spec", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--py-spy", default=None)

    p = sub.add_parser("host")
    p.add_argument("--spec", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--base-ref", required=True)
    p.add_argument("drive_args", nargs=argparse.REMAINDER)

    for name in ("drive", "child"):
        p = sub.add_parser(name)
        inputs(p)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--allowed-tiers", default="ram,ssd")
        p.add_argument("--prefetch-override", default=None)
        p.add_argument("--scratch", required=True)
        if name == "drive":
            p.add_argument("--arm", action="append", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--repeats", type=int, default=2)
            p.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
            p.add_argument("--py-spy", default="/opt/bench/py-spy")
            p.add_argument("--py-spy-rate", type=int, default=100)
            p.add_argument("--keep-arm", action="append",
                           help="an arm that runs with roll_may_keep=True (pageable rows)")
        else:
            p.add_argument("--probe-major", action="store_true")
            p.add_argument("--roll-may-keep", action="store_true")
            p.add_argument("--arm", required=True)
            p.add_argument("--label", required=True)
            p.add_argument("--result", required=True)

    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    p.add_argument("--py-spy-rate", type=int, default=100)
    p.add_argument("--min-sample-fraction", type=float, default=0.5,
                   help="a profile with fewer main-thread samples than this fraction "
                        "of rate x roll_s is reported and left out of the shares")
    p.add_argument("--netdata", default=None,
                   help="the measured box's Netdata, e.g. http://sparky:19999")
    p.add_argument("--power-chart", default=(
        "nvidia_smi.gpu_gpu-e76c7efc-c157-b1f4-1348-83e4eb5092f4_power_draw"))
    p.add_argument("--envelope-w", type=float, default=140.0)

    args = parser.parse_args(argv)
    if args.command == "host":
        args.drive_args = [a for a in args.drive_args if a != "--"]
    return {"manifest": cmd_manifest, "spec": cmd_spec, "host": cmd_host,
            "drive": cmd_drive, "child": cmd_child, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
