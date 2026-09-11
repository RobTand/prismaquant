"""Profile the single-threaded CPU phases at the head of a Tessera campaign row.

Read-only reproduction of four per-unit loops that idle the GPU:

* ``projected``    -- ``_checked_projected_units``: re-read every unit's source
                      tensor from the shard the producer hashed and compare it
                      byte for byte with the live view.
* ``hold``         -- ``_campaign_bound_identities``: one producer receipt
                      template per priced unit (sha256 over the source weight
                      and the Hessian, plus two ``isfinite`` passes).
* ``commitments``  -- ``canonical_hessian_reference_descriptor``'s
                      ``{name: tensor_identity(H)}`` plus the capture seal.
* ``journal``      -- ``cost_stage_checkpoint.prepare_journal`` over a completed
                      row's unit shards.
* ``wireverify``   -- the resumed row's ``_checkpoint_wire_record`` loop, which
                      reads and verifies every cached wire blob.

Nothing here writes to the campaign workspace, and no campaign module is
changed: the phases call the production functions on copies of their inputs.
``--threads`` runs the same loop through a thread pool so the measured speedup,
not an assumption about the GIL, decides whether a pool is the right fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def _io_counters() -> dict:
    out = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, _, value = line.partition(":")
            out[key.strip()] = int(value)
    except OSError:
        pass
    return out


class Phase:
    def __init__(self, name: str, units: int):
        self.name, self.units = name, units

    def _mark(self, edge):
        """A distinctive ENOENT stat, so a syscall trace can be cut to the phase.

        ``strace`` counts a whole process; the per-unit question is about the
        loop, not the imports and the input load that precede it.
        """
        try:
            os.stat("/PQHEADPROF-%s-%s" % (edge, self.name))
        except OSError:
            pass

    def __enter__(self):
        self._mark("BEGIN")
        self._io = _io_counters()
        self._cpu = time.process_time()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        self.seconds = time.perf_counter() - self._t
        self._mark("END")
        self.cpu_seconds = time.process_time() - self._cpu
        after = _io_counters()
        self.io = {k: after.get(k, 0) - v for k, v in self._io.items()}
        return False

    def record(self) -> dict:
        return {
            "phase": self.name,
            "units": self.units,
            "seconds": round(self.seconds, 3),
            "cpu_seconds": round(self.cpu_seconds, 3),
            "units_per_second": round(self.units / self.seconds, 3) if self.seconds else None,
            "rchar_bytes": self.io.get("rchar"),
            "read_bytes": self.io.get("read_bytes"),
        }


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_inputs(args):
    import torch
    from safetensors.torch import load_file

    census = json.loads(Path(args.census).read_text())
    projection = census["expert_projection"]
    source = projection["producer"]["source"]
    stacks = projection["stacks"]

    selection = json.loads(Path(args.units).read_text())
    group = selection["groups"][0]
    members = sorted(group["members"])
    stack = group["key"].split(":", 1)[1]
    bound = stacks[stack]

    # One contiguous slice of experts, so a subset keeps whole experts and the
    # gate/up Hessian sharing the row actually has.
    experts = sorted({int(name.rsplit(".", 2)[-2]) for name in members})
    keep = set(experts[: args.experts]) if args.experts else set(experts)
    names = [n for n in members if int(n.rsplit(".", 2)[-2]) in keep]

    weights, hessians = {}, {}
    from prismaquant.tessera_expert_projection import source_unit_weight
    for name in names:
        weights[name] = source_unit_weight(args.model, source, bound[name])
    if args.need_hessians:
        for name in names:
            path = Path(args.calibration_inputs) / (name.replace(".", "__") + ".pt")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            hessians[name] = payload["hessian"].contiguous()
            del payload

    scales = {}
    if Path(args.input_scales).is_file():
        table = load_file(args.input_scales)
        for name in names:
            key = f"{name}.input_global_scale"
            if key in table:
                scales[name] = float(table[key].reshape(-1)[0])

    rungs = args.rungs.split(",") if args.rungs else []
    menus = {name: [SimpleNamespace(format_name=fmt) for fmt in rungs] for name in names}
    projected_units = {name: bound[name] for name in names}
    return SimpleNamespace(names=names, weights=weights, hessians=hessians, menus=menus,
                           scales=scales, projected_units=projected_units, source=source,
                           bound=bound, census=census)


def activation_source(data, args):
    from tessera.export import ActivationSource
    provenance = json.loads(Path(args.hessian_references).read_text())["provenance"]
    return ActivationSource(hessians={name: h for name, h in data.hessians.items()},
                            provenance=dict(provenance))


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def phase_micro(args):
    import hashlib
    import torch
    results = []
    buffer = bytes(os.urandom(1 << 20)) * 64          # 64 MiB, no page sharing
    for label, payload in (("sha256_64MiB", buffer),):
        start = time.perf_counter()
        rounds = 4
        for _ in range(rounds):
            hashlib.sha256(payload).hexdigest()
        seconds = time.perf_counter() - start
        results.append({"kernel": label, "bytes": len(payload) * rounds,
                        "seconds": round(seconds, 4),
                        "bytes_per_second": round(len(payload) * rounds / seconds)})
    for label, tensor in (("isfinite_bf16_4096x2048", torch.randn(4096, 2048, dtype=torch.bfloat16)),
                          ("isfinite_fp32_4096x4096", torch.randn(4096, 4096))):
        start = time.perf_counter()
        rounds = 4
        for _ in range(rounds):
            bool(torch.isfinite(tensor).all())
        seconds = time.perf_counter() - start
        nbytes = tensor.numel() * tensor.element_size() * rounds
        results.append({"kernel": label, "bytes": nbytes, "seconds": round(seconds, 4),
                        "bytes_per_second": round(nbytes / seconds)})
    # thread scaling of the one kernel every phase leans on
    for workers in (1, 2, 4, 8):
        chunks = [buffer] * 8
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda b: hashlib.sha256(b).hexdigest(), chunks))
        seconds = time.perf_counter() - start
        results.append({"kernel": "sha256_pool", "threads": workers,
                        "bytes": len(buffer) * len(chunks), "seconds": round(seconds, 4),
                        "bytes_per_second": round(len(buffer) * len(chunks) / seconds)})
    return {"phase": "micro", "results": results}


def phase_projected(args, data):
    """_checked_projected_units: re-read the producer's source bytes and compare."""
    import torch
    from prismaquant.tessera_expert_projection import source_unit_weight

    def one(name):
        weight = source_unit_weight(args.model, data.source, data.bound[name])
        live = data.weights[name].detach().cpu()
        ok = live.dtype == weight.dtype and torch.equal(live, weight)
        del weight, live
        return ok

    with Phase("projected", len(data.names)) as phase:
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                results = list(pool.map(one, data.names))
        else:
            results = [one(name) for name in data.names]
    assert all(results), "live view disagreed with the producer's source bytes"
    return phase.record()


def phase_hold(args, data):
    """_campaign_bound_identities: the producer receipt template per unit."""
    from prismaquant.tessera_campaign import (
        _campaign_bound_identities, _campaign_identity_anchor_roster,
        _campaign_identity_metadata_plan, bind_checkpoint_unit_identity,
    )
    source = activation_source(data, args)
    bounds, scratch = _campaign_identity_metadata_plan(
        weights=data.weights, menus=data.menus, calibration_source=source,
        projected_units=data.projected_units, static_scales=data.scales)

    if args.threads > 1:
        def one(name):
            anchors = _campaign_identity_anchor_roster(
                name, data.menus[name], calibration_source=source, static_scales=data.scales)
            return name, bind_checkpoint_unit_identity(
                anchors, source_weight=data.weights[name], calibration_source=source,
                projected_unit=data.projected_units.get(name), static_scales=data.scales,
                retain_source_receipt=False)
        with Phase("hold", len(data.names)) as phase:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                produced = dict(pool.map(one, sorted(data.weights)))
            held = {name: produced[name] for name in sorted(data.weights)}
    else:
        with Phase("hold", len(data.names)) as phase:
            held = _campaign_bound_identities(
                weights=data.weights, menus=data.menus, calibration_source=source,
                projected_units=data.projected_units, static_scales=data.scales,
                metadata_bounds=bounds)

    record = phase.record()
    record["identity_digest"] = _identity_digest(held)
    record["observed_metadata_bytes"] = sum(u.observed_metadata_bytes() for u in held.values())
    record["planned_metadata_bytes"] = sum(bounds.values())
    record["planning_scratch_bytes"] = scratch
    record["insertion_order_is_sorted"] = list(held) == sorted(held)
    for unit in held.values():
        unit.close()
    return record


def _identity_digest(held) -> str:
    """One digest over every retained producer receipt, in insertion order.

    A parallel hold must produce byte-identical receipts in the same order, or
    the run-level identity and every downstream pickle stop being the bytes the
    serial hold produced.
    """
    import hashlib
    digest = hashlib.sha256()
    for name, unit in held.items():
        digest.update(name.encode() + b"\0")
        digest.update(json.dumps(unit.campaign_inputs(), sort_keys=True,
                                 separators=(",", ":")).encode() + b"\0")
    return digest.hexdigest()


def phase_commitments(args, data):
    """The H commitments writer's {name: tensor_identity(H)} plus the seal."""
    from tessera.cached_unit import tensor_identity
    from tessera.hessian_capture import capture_sha256_from_units
    provenance = json.loads(Path(args.hessian_references).read_text())["provenance"]

    with Phase("commitments", len(data.names)) as phase:
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                produced = dict(pool.map(
                    lambda n: (n, tensor_identity(data.hessians[n])), sorted(data.hessians)))
            identities = {n: produced[n] for n in sorted(data.hessians)}
        else:
            identities = {n: tensor_identity(v) for n, v in data.hessians.items()}
        digest = capture_sha256_from_units(dict(provenance),
                                           {n: v["sha256"] for n, v in identities.items()})
    record = phase.record()
    record["capture_sha256"] = digest
    record["insertion_order_is_sorted"] = list(identities) == sorted(identities)
    return record


def _nfs_proc4() -> dict:
    """``/proc/net/rpc/nfs`` counters, so a wait can be attributed to the wire."""
    out = {}
    try:
        for line in Path("/proc/net/rpc/nfs").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in ("rpc", "proc4", "proc3"):
                out[parts[0]] = [int(x) for x in parts[1:]]
    except OSError:
        pass
    return out


def _self_state() -> tuple:
    try:
        stat = Path("/proc/self/stat").read_text()
        state = stat[stat.rindex(")") + 2]
    except (OSError, ValueError):
        state = "?"
    try:
        wchan = Path("/proc/self/wchan").read_text().strip() or "-"
    except OSError:
        wchan = "-"
    return state, wchan


def phase_journalshards(args):
    """Only the ``_load_unit`` loop of prepare_journal, read-only, over NFS.

    ``prepare_journal`` also parses an 80 MB manifest and writes when no
    manifest exists; neither belongs in a latency measurement and neither runs
    here. Nothing under the workspace is written.
    """
    import threading
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path

    manifest = json.loads(Path(args.journal_manifest).read_text())
    root = Path(args.journal_dir)
    qnames = [entry["qname"] for entry in manifest["units"]]
    identity_sha256 = manifest["identity_sha256"]
    stage = manifest["stage"]
    paths = [(unit_path(root, q), q) for q in qnames]

    samples = collections_counter()
    stop = threading.Event()

    def sample():
        while not stop.wait(0.02):                     # ~50 Hz
            samples[_self_state()] += 1

    watcher = threading.Thread(target=sample, daemon=True)
    before_rpc = _nfs_proc4()
    watcher.start()
    latencies = []
    lock = threading.Lock()

    def one(job):
        path, qname = job
        start = time.perf_counter()
        if path.is_file():
            _load_unit(path, stage=stage, qname=qname,
                       identity_sha256=identity_sha256)
        taken = time.perf_counter() - start
        with lock:
            latencies.append(taken)

    with Phase("journalshards", len(paths)) as phase:
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                list(pool.map(one, paths))
        else:
            for job in paths:
                one(job)
    stop.set()
    watcher.join(timeout=2)
    after_rpc = _nfs_proc4()

    latencies.sort()
    record = phase.record()
    record["shard_seconds_mean"] = round(sum(latencies) / len(latencies), 6)
    record["shard_seconds_p50"] = round(latencies[len(latencies) // 2], 6)
    record["shard_seconds_p99"] = round(latencies[int(len(latencies) * 0.99)], 6)
    record["sampler_states"] = {f"{state}:{wchan}": n for (state, wchan), n in
                                sorted(samples.items(), key=lambda kv: -kv[1])[:10]}
    record["sampler_total"] = sum(samples.values())
    record["nfs_rpc_calls_delta"] = (
        (after_rpc.get("rpc", [0])[0] - before_rpc.get("rpc", [0])[0])
        if before_rpc.get("rpc") and after_rpc.get("rpc") else None)
    for key in ("proc4", "proc3"):
        if before_rpc.get(key) and after_rpc.get(key):
            record[f"nfs_{key}_delta_total"] = sum(
                a - b for a, b in zip(after_rpc[key], before_rpc[key]))
    return record


def collections_counter():
    import collections
    return collections.Counter()


def phase_journal(args):
    """cost_stage_checkpoint.prepare_journal over a completed row's shards."""
    from prismaquant.cost_stage_checkpoint import prepare_journal
    manifest = json.loads(Path(args.journal_manifest).read_text())
    qnames = [entry["qname"] for entry in manifest["units"]]
    with Phase("journal", len(qnames)) as phase:
        root, digest, completed = prepare_journal(
            args.journal_dir, stage=manifest["stage"], resume=True,
            identity=manifest["identity"], qnames=qnames,
            manifest_path=args.journal_manifest)
    record = phase.record()
    record["completed_units"] = len(completed)
    record["identity_sha256_matches"] = digest == manifest["identity_sha256"]
    record["source"] = str(args.journal_dir)
    return record


def phase_wireverify(args, data):
    """The resumed row's per-anchor _checkpoint_wire_record loop."""
    from prismaquant.cost_stage_checkpoint import prepare_journal
    from prismaquant.tessera_campaign import (
        CampaignAnchor, _campaign_bound_identities, _checkpoint_anchor_identity,
        _checkpoint_wire_record,
    )
    manifest = json.loads(Path(args.journal_manifest).read_text())
    qnames = [entry["qname"] for entry in manifest["units"]]
    _root, _digest, completed = prepare_journal(
        args.journal_dir, stage=manifest["stage"], resume=True,
        identity=manifest["identity"], qnames=qnames, manifest_path=args.journal_manifest)

    names = [n for n in data.names if n in completed]
    rungs = sorted({row["format_name"] for n in names for row in completed[n]["anchors"]})
    menus = {n: [SimpleNamespace(format_name=f) for f in rungs] for n in names}
    source = activation_source(data, args)
    held = _campaign_bound_identities(
        weights=data.weights, menus=menus, calibration_source=source,
        projected_units=data.projected_units, static_scales=data.scales)
    wire_dir = Path(args.wire_dir)

    jobs = []
    for name in names:
        for row in completed[name]["anchors"]:
            jobs.append((name, CampaignAnchor(**row), completed[name]["wire_records"][row["format_name"]]))

    def one(job):
        name, anchor, existing = job
        identity = _checkpoint_anchor_identity(
            anchor, weights=data.weights, menus=menus, calibration_source=source,
            static_scales=data.scales, projected_units=data.projected_units,
            bound_unit=held[name])
        return _checkpoint_wire_record(anchor, wire_dir, identity, existing=existing)

    with Phase("wireverify", len(jobs)) as phase:
        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                records = list(pool.map(one, jobs))
        else:
            records = [one(job) for job in jobs]
    record = phase.record()
    record["anchors"] = len(records)
    record["units"] = len(names)
    record["blob_bytes"] = sum(r["blob_bytes"] for r in records)
    for unit in held.values():
        unit.close()
    return record


# ---------------------------------------------------------------------------
# the branch's head (RobTand/prismaquant#492): seal ahead, hold on N threads,
# commitments from the hold's sealed receipts, resumed wire verify on N threads
# ---------------------------------------------------------------------------
def phase_headafter(args, data):
    """The campaign's own head sequence as #492 runs it, one record per step.

    ``hold`` here is ``_campaign_bound_identities(threads=N)`` -- the real
    function, not the harness's own pool -- after ``_SealAhead`` has sealed
    the owner; ``commitments`` are ``_bound_hessian_identities`` over the held
    units plus the same ``capture_sha256_from_units``.  The digests recorded
    (``identity_digest``, ``capture_sha256``) are what the serial ``hold`` and
    ``commitments`` phases record on the same inputs, so an after-run is
    checked against a before-run by equality, not by trust.
    """
    from prismaquant.tessera_campaign import (
        _SealAhead, _bound_hessian_identities, _campaign_bound_identities,
        _campaign_identity_metadata_plan,
    )
    from tessera.hessian_capture import capture_sha256_from_units
    provenance = json.loads(Path(args.hessian_references).read_text())["provenance"]
    source = activation_source(data, args)
    ahead = _SealAhead(source)
    bounds, scratch = _campaign_identity_metadata_plan(
        weights=data.weights, menus=data.menus, calibration_source=source,
        projected_units=data.projected_units, static_scales=data.scales,
        threads=args.threads)
    with Phase("sealwait", len(data.names)) as phase:
        waited = ahead.wait()
    seal = phase.record()
    seal["seal_seconds"] = ahead.seconds
    seal["waited_seconds"] = waited
    with Phase("hold", len(data.names)) as phase:
        held = _campaign_bound_identities(
            weights=data.weights, menus=data.menus, calibration_source=source,
            projected_units=data.projected_units, static_scales=data.scales,
            metadata_bounds=bounds, threads=args.threads)
    hold = phase.record()
    hold["identity_digest"] = _identity_digest(held)
    hold["observed_metadata_bytes"] = sum(u.observed_metadata_bytes() for u in held.values())
    hold["planned_metadata_bytes"] = sum(bounds.values())
    hold["planning_scratch_bytes"] = scratch
    hold["insertion_order_is_sorted"] = list(held) == sorted(held)
    with Phase("commitments", len(data.names)) as phase:
        identities = _bound_hessian_identities(held, data.hessians)
        digest = capture_sha256_from_units(dict(provenance),
                                           {n: v["sha256"] for n, v in identities.items()})
    commitments = phase.record()
    commitments["capture_sha256"] = digest
    commitments["units"] = len(identities)
    commitments["insertion_order_is_sorted"] = list(identities) == sorted(identities)
    for unit in held.values():
        unit.close()
    return {"phase": "headafter", "seal": seal, "hold": hold, "commitments": commitments,
            "wall_seconds": round(seal["wall_seconds"] + hold["wall_seconds"]
                                  + commitments["wall_seconds"], 3)}


def phase_wireverifyafter(args, data):
    """The resumed row's adopt loop as #492 runs it: gates in order, verify on N."""
    from prismaquant.cost_stage_checkpoint import prepare_journal
    from prismaquant.tessera_campaign import (
        CampaignAnchor, _SealAhead, _campaign_bound_identities,
        _checkpoint_anchor_identity, _verify_wire_records_on_threads,
    )
    manifest = json.loads(Path(args.journal_manifest).read_text())
    qnames = [entry["qname"] for entry in manifest["units"]]
    _root, _digest, completed = prepare_journal(
        args.journal_dir, stage=manifest["stage"], resume=True,
        identity=manifest["identity"], qnames=qnames, manifest_path=args.journal_manifest)
    names = [n for n in data.names if n in completed]
    rungs = sorted({row["format_name"] for n in names for row in completed[n]["anchors"]})
    menus = {n: [SimpleNamespace(format_name=f) for f in rungs] for n in names}
    source = activation_source(data, args)
    _SealAhead(source).wait()
    held = _campaign_bound_identities(
        weights=data.weights, menus=menus, calibration_source=source,
        projected_units=data.projected_units, static_scales=data.scales,
        threads=args.threads)
    wire_dir = Path(args.wire_dir)
    with Phase("wireverify", sum(len(completed[n]["anchors"]) for n in names)) as phase:
        pending = []
        for name in names:
            for row in completed[name]["anchors"]:
                anchor = CampaignAnchor(**row)
                identity = _checkpoint_anchor_identity(
                    anchor, weights=data.weights, menus=menus, calibration_source=source,
                    static_scales=data.scales, projected_units=data.projected_units,
                    bound_unit=held[name])
                pending.append((anchor, identity, completed[name]["wire_records"][row["format_name"]]))
        records = _verify_wire_records_on_threads(pending, wire_dir, threads=args.threads)
    record = phase.record()
    record["anchors"] = len(records)
    record["units"] = len(names)
    record["blob_bytes"] = sum(r["blob_bytes"] for r in records)
    for unit in held.values():
        unit.close()
    return record


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["micro", "projected", "hold", "commitments", "journal",
                                 "journalshards", "wireverify", "headafter", "wireverifyafter"])
    parser.add_argument("--model", default="/mnt/shared/models/GLM-5.3-Flash-BF16")
    parser.add_argument("--census")
    parser.add_argument("--units")
    parser.add_argument("--calibration-inputs")
    parser.add_argument("--input-scales")
    parser.add_argument("--hessian-references")
    parser.add_argument("--journal-manifest")
    parser.add_argument("--journal-dir")
    parser.add_argument("--wire-dir")
    parser.add_argument("--rungs", default="TESSERA_E4M3_K1_R832,TESSERA_E4M3_K1_R1088")
    parser.add_argument("--experts", type=int, default=48)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    args.need_hessians = args.phase in ("hold", "commitments", "wireverify",
                                        "headafter", "wireverifyafter")
    started = time.time()
    if args.phase == "micro":
        record = phase_micro(args)
    elif args.phase == "journal":
        record = phase_journal(args)
    elif args.phase == "journalshards":
        record = phase_journalshards(args)
    else:
        load = time.perf_counter()
        data = load_inputs(args)
        load = time.perf_counter() - load
        if args.phase == "projected":
            record = phase_projected(args, data)
        elif args.phase == "hold":
            record = phase_hold(args, data)
        elif args.phase == "commitments":
            record = phase_commitments(args, data)
        elif args.phase == "headafter":
            record = phase_headafter(args, data)
        elif args.phase == "wireverifyafter":
            record = phase_wireverifyafter(args, data)
        else:
            record = phase_wireverify(args, data)
        record["input_load_seconds"] = round(load, 3)
    record["threads"] = args.threads
    record["label"] = args.label
    record["started_unix"] = started
    record["hostname"] = os.uname().nodename
    record["python"] = sys.version.split()[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
