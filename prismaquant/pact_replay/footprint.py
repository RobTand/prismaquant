"""Complete resident and transient footprint evidence for PACT jobs.

The recorder observes the live objects the lead passes in and writes one
plain-data dict per phase. It creates no thread, timer, cache, preload, or
memory budget. It retains no tensor, future payload, or owner. It only reads
public metadata and releases every reference before it returns.

Observed categories:

- resident cache storage: the runner's StreamingContext through
  source_residency_snapshot, which never waits on a prefetch and counts
  each storage once. Resident layers come from non-touching peek calls.
  A completed prefetch future names the next layer. The current layer is
  the phase label the lead wires, cross-checked by shared storage below.
- render planes: current-layer decoded weights in weights, plus the
  source tensor each plane was decoded from.
- transient queue buffers: pending read-ahead entries. Done futures report
  exact bytes. Live futures report in_flight with unknown bytes, because
  inspection never waits on a worker.
- replay states: boundary owner telemetry copied verbatim, plus measured
  state tensors the lead passes in.
- shared storage views: every member names its storage key. A member that
  shares storage with another category sets shares_storage. Physical bytes
  sum each storage once per phase.

Sampled values bound observed instants only. Maxima across phases are the
largest sampled values, not the unobserved transient peaks between samples.
CUDA maxima are process-lifetime peaks from the driver and also bound only
what the process did up to the sample. Missing inputs and missing platform
data appear as explicit unavailable entries with a reason, never as zero.
"""

from __future__ import annotations

import os
import resource
import time

PHASE_SCHEMA = "pact.footprint_phase.v1"
EVIDENCE_SCHEMA = "pact.footprint_evidence.v1"

LIMITS = (
    "Sampled values bound observed instants only. "
    "Unobserved transient peaks between samples are unknown. "
    "Maxima across phases are the largest sampled values, "
    "not the true peaks. "
    "CUDA maxima are process-lifetime driver peaks up to the sample. "
    "Reserved bytes include allocator slack and are not assignment bytes."
)


def _unavailable(reason):
    return {"status": "unavailable", "reason": str(reason)}


def _torch():
    try:
        import torch  # noqa: WPS433
        return torch
    except Exception:
        return None


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


class _Registry:
    """One storage ledger per phase. Members are metadata, never tensors."""

    def __init__(self):
        self.storages = {}

    def add_tensor(self, tensor, *, category, label):
        torch = _torch()
        entry = {"category": category, "label": str(label)}
        if torch is None or not isinstance(tensor, torch.Tensor):
            entry["tensor_bytes"] = _unavailable("not a torch tensor")
            entry["shares_storage"] = False
            return entry
        try:
            if tensor.is_meta:
                entry["tensor_bytes"] = _unavailable("meta tensor has no storage")
                entry["shares_storage"] = False
                return entry
            element = int(tensor.element_size())
            count = int(tensor.numel())
            key = _storage_key(tensor)
        except Exception as exc:
            entry["tensor_bytes"] = _unavailable("metadata read failed: %r" % (exc,))
            entry["shares_storage"] = False
            return entry
        record = self.storages.get(key)
        if record is None:
            record = {"device": key[0], "bytes": key[2], "members": []}
            self.storages[key] = record
            entry["shares_storage"] = False
        else:
            entry["shares_storage"] = True
        entry.update({
            "shape": [int(n) for n in tensor.shape],
            "dtype": str(tensor.dtype),
            "tensor_bytes": count * element,
            "storage_offset": int(tensor.storage_offset()),
        })
        record["members"].append(entry)
        return entry

    def add_snapshot_storage(self, device, pointer, size, *, category, label):
        key = (str(device), int(pointer), int(size))
        record = self.storages.get(key)
        shared = record is not None
        if record is None:
            record = {"device": key[0], "bytes": key[2], "members": []}
            self.storages[key] = record
        record["members"].append({
            "category": category, "label": str(label),
            "tensor_bytes": _unavailable("owner-counted extent, not re-measured"),
            "shares_storage": shared,
        })

    def total(self):
        return sum(record["bytes"] for record in self.storages.values())

    def describe(self):
        return [
            {"device": record["device"], "bytes": record["bytes"],
             "members": record["members"]}
            for _, record in sorted(self.storages.items())
        ]


def _resident_layers(context):
    """Resident layer indices through non-touching peek calls only."""
    cache = getattr(context, "layer_cache", None)
    peek = getattr(cache, "peek", None)
    total = getattr(context, "num_layers", None)
    if not callable(peek) or type(total) is not int:
        return None, "context lacks layer_cache.peek or num_layers"
    try:
        return [layer for layer in range(total) if peek(layer)], None
    except Exception as exc:
        return None, "peek walk failed: %r" % (exc,)


def _describe_resident(runner, registry):
    if runner is None:
        return _unavailable("no runner passed for this phase")
    context = getattr(runner, "context", None)
    if context is None:
        return _unavailable("runner has no context")
    snapshotter = getattr(context, "source_residency_snapshot", None)
    if not callable(snapshotter):
        return _unavailable("context lacks source_residency_snapshot")
    layers, error = _resident_layers(context)
    if error is not None:
        return _unavailable(error)
    try:
        snapshot = snapshotter(range(context.num_layers), include_head=True)
    except Exception as exc:
        return _unavailable("source_residency_snapshot failed: %r" % (exc,))
    try:
        for owner in snapshot.get("owners", []):
            layer = owner.get("layer")
            for storage in owner.get("storages", []):
                registry.add_snapshot_storage(
                    storage.get("device"), storage.get("pointer"),
                    storage.get("bytes"), category="resident_layer",
                    label="layer %s %s" % (layer, owner.get("owner")))
    except Exception as exc:
        return _unavailable("resident merge failed: %r" % (exc,))
    return {"status": "measured", "layers": list(layers), "snapshot": snapshot}


def _option_name(option, index):
    if isinstance(option, dict):
        name = option.get("name")
        if name is not None:
            return str(name)
    return "option-%d" % index


def _describe_weights(weights, registry):
    if weights is None:
        return _unavailable("no weights passed for this phase")
    members = []
    try:
        names = sorted(weights.keys(), key=str)
    except Exception as exc:
        return _unavailable("weights keys unreadable: %r" % (exc,))
    for name in names:
        try:
            entry = weights[name]
            source = entry["source"] if isinstance(entry, dict) else None
            options = entry.get("options", []) if isinstance(entry, dict) else []
        except Exception as exc:
            members.append(_unavailable("entry %s unreadable: %r" % (name, exc)))
            continue
        members.append(registry.add_tensor(
            source, category="weight_source", label="%s source" % (name,)))
        for index, item in enumerate(options):
            option, decoded = item if isinstance(item, tuple) else ({}, item)
            label = "%s %s" % (name, _option_name(option, index))
            if hasattr(decoded, "values") and hasattr(decoded, "row_scales"):
                members.append(registry.add_tensor(decoded.values, category="render_plane", label=label + " values"))
                members.append(registry.add_tensor(decoded.row_scales, category="render_plane", label=label + " row_scales"))
            else:
                members.append(registry.add_tensor(decoded, category="render_plane", label=label))
    return {"status": "measured", "members": members}


def _future_bytes(future):
    """Exact bytes of a done future without waiting. Never blocks."""
    if future is None:
        return {"status": "no_buffer", "bytes": 0,
                "note": "passthrough option stages no byte buffer"}
    try:
        done = future.done()
    except Exception as exc:
        return {"status": "unknown", "bytes": _unavailable("done() failed: %r" % (exc,))}
    if not done:
        return {"status": "in_flight", "bytes": _unavailable(
            "inspection never waits on a read worker")}
    if getattr(future, "cancelled", lambda: False)():
        return {"status": "cancelled", "bytes": _unavailable("future cancelled")}
    try:
        payload = future.result()
    except Exception as exc:
        return {"status": "failed", "bytes": _unavailable("future failed: %r" % (exc,))}
    try:
        torch = _torch()
        if isinstance(payload, tuple) and payload and isinstance(payload[0], (bytes, bytearray, memoryview)):
            return {"status": "done", "bytes": len(payload[0])}
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return {"status": "done", "bytes": int(len(payload))}
        if torch is not None and isinstance(payload, torch.Tensor):
            if payload.is_meta:
                return {"status": "done",
                        "bytes": _unavailable("meta tensor has no storage")}
            storage = payload.untyped_storage()
            return {"status": "done", "bytes": int(storage.nbytes())}
        return {"status": "done",
                "bytes": _unavailable("payload type %s has no byte count" % type(payload).__name__)}
    finally:
        del payload


def _describe_pending(pending):
    if pending is None:
        return _unavailable("no pending queue passed for this phase")
    try:
        entries = list(pending)
    except Exception as exc:
        return _unavailable("pending queue not iterable: %r" % (exc,))
    buffers, exact, unknown = [], 0, False
    for index, item in enumerate(entries):
        future = item[3] if isinstance(item, tuple) and len(item) == 4 else None
        outcome = _future_bytes(future)
        status = outcome["status"]
        if status == "done" and type(outcome["bytes"]) is int:
            exact += outcome["bytes"]
        elif status in ("in_flight", "unknown"):
            unknown = True
        buffers.append({"index": index, "status": status, "bytes": outcome["bytes"]})
    return {"status": "measured", "buffers": buffers,
            "done_buffer_bytes": exact,
            "in_flight_unknown": unknown,
            "note": "Done buffers are exact. In-flight bytes stay unknown; "
                    "they are not estimated and not treated as zero."}


def _describe_owner(owner):
    telemetry = getattr(owner, "telemetry", None)
    if not isinstance(telemetry, dict):
        return _unavailable("owner has no telemetry dict")
    counters = {}
    for key, value in telemetry.items():
        if type(value) in (int, float, str, bool):
            counters[str(key)] = value
    return {"status": "measured", "telemetry": counters,
            "note": "Owner-counted tensor bytes, not allocator peaks."}


def _walk_state_tensors(value, registry, *, label, out):
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        out.append(registry.add_tensor(value, category="replay_state", label=label))
        return
    if isinstance(value,(bytes,bytearray,memoryview)):
        out.append({"category":"host_read_buffer","label":label,"bytes":len(value),
                    "storage_aliasing":"Not included in the tensor storage total"})
        return
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=str):
            _walk_state_tensors(value[key], registry,
                                label="%s/%s" % (label, key), out=out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_state_tensors(item, registry,
                                label="%s[%d]" % (label, index), out=out)


def _describe_replay(state, registry):
    if state is None:
        return _unavailable("no replay state passed for this phase")
    owners, tensors = [], []
    candidates = list(state.values()) if isinstance(state, dict) else [state]
    for index, candidate in enumerate(candidates):
        if hasattr(candidate, "telemetry"):
            owners.append({"index": index,
                           "report": _describe_owner(candidate)})
        else:
            _walk_state_tensors(candidate, registry,
                                label="state-%d" % index, out=tensors)
    if not owners and not tensors:
        return _unavailable("state holds no owner telemetry or tensors")
    return {"status": "measured", "owners": owners, "tensors": tensors}


def _proc_value(path, key):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    raise KeyError(key)


def _describe_host():
    missing = []
    try:
        rss = int(_proc_value("/proc/self/status", "VmRSS").split()[0]) * 1024
    except Exception:
        rss, missing = None, missing + ["VmRSS"]
    try:
        available = int(_proc_value("/proc/meminfo", "MemAvailable").split()[0]) * 1024
    except Exception:
        available, missing = None, missing + ["MemAvailable"]
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        peak, missing = None, missing + ["ru_maxrss"]
    return {"rss_bytes": rss, "peak_rss_bytes": peak,
            "mem_available_bytes": available,
            "missing": missing,
            "note": "rss is the current sample; peak_rss is the process peak. "
                    "Neither bounds unobserved transient use between samples."}


def _describe_cuda():
    torch = _torch()
    if torch is None:
        return _unavailable("torch is not importable, no CUDA counters exist")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return _unavailable("torch has no cuda module")
    try:
        if not cuda.is_available():
            return _unavailable("no CUDA device is visible")
    except Exception as exc:
        return _unavailable("cuda availability check failed: %r" % (exc,))
    try:
        report = {"status": "measured",
                  "allocated_bytes": int(cuda.memory_allocated()),
                  "reserved_bytes": int(cuda.memory_reserved()),
                  "peak_allocated_bytes": int(cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(cuda.max_memory_reserved())}
    except Exception as exc:
        return _unavailable("cuda counter read failed: %r" % (exc,))
    report["note"] = ("Driver peaks cover the process lifetime up to this sample. "
                       "Reserved bytes include allocator slack.")
    return report


def _describe_io():
    try:
        fields = {}
        with open("/proc/self/io", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                fields[key.strip()] = int(value.strip())
        return {"status": "measured", "counters": fields,
                "note": "Cumulative process counters, not phase deltas."}
    except Exception as exc:
        return _unavailable("process I/O unavailable: %r" % (exc,))


class FootprintRecorder:
    """One evidence log per run. The lead wires one record call per phase."""

    def __init__(self):
        self._records = []

    def record(self, phase, *, runner=None, weights=None, pending=None,
               state=None):
        """Observe one phase and return its plain-data evidence dict."""
        if type(phase) is not str or not phase:
            raise ValueError("phase must be a nonempty string")
        registry = _Registry()
        resident = _describe_resident(runner, registry)
        planes = _describe_weights(weights, registry)
        replay = _describe_replay(state, registry)
        entry = {
            "schema": PHASE_SCHEMA,
            "phase": phase,
            "unix_time": time.time(),
            "resident": resident,
            "planes": planes,
            "pending": _describe_pending(pending),
            "replay": replay,
            "unique_storage_bytes": registry.total(),
            "storages": registry.describe(),
            "host": _describe_host(),
            "cuda": _describe_cuda(),
            "io": _describe_io(),
            "limits": LIMITS,
        }
        del registry
        self._records.append(entry)
        return entry

    def finish(self):
        """Return the accumulated evidence with largest sampled values."""
        largest = {"resident_unique_bytes": 0, "tensor_unique_bytes": 0,
                   "host_rss_bytes": 0, "cuda_peak_allocated_bytes": 0}
        for entry in self._records:
            resident = entry.get("resident", {})
            snapshot = resident.get("snapshot", {}) if isinstance(resident, dict) else {}
            if type(snapshot.get("unique_storage_bytes")) is int:
                largest["resident_unique_bytes"] = max(
                    largest["resident_unique_bytes"],
                    snapshot["unique_storage_bytes"])
            if type(entry.get("unique_storage_bytes")) is int:
                largest["tensor_unique_bytes"] = max(
                    largest["tensor_unique_bytes"],
                    entry["unique_storage_bytes"])
            host = entry.get("host", {})
            if type(host.get("rss_bytes")) is int:
                largest["host_rss_bytes"] = max(
                    largest["host_rss_bytes"], host["rss_bytes"])
            cuda = entry.get("cuda", {})
            if type(cuda.get("peak_allocated_bytes")) is int:
                largest["cuda_peak_allocated_bytes"] = max(
                    largest["cuda_peak_allocated_bytes"],
                    cuda["peak_allocated_bytes"])
        return {"schema": EVIDENCE_SCHEMA,
                "phase_count": len(self._records),
                "phases": list(self._records),
                "largest_sampled": largest,
                "limits": LIMITS}
