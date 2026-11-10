"""Producer-bound admission for joint-quantum fanout (PQ #1293).

This is a measurement gate, not a scheduler or a performance estimator.
Missing measurements refuse; an operator may explicitly override the gate.
"""
from __future__ import annotations

import re
import math
from collections.abc import Mapping
from pathlib import Path

from .digests import bytes_sha256hex, canonical_json_sha256, is_sha256hex
from .io_spans import EXPOSED_WAIT_SCHEMA, GB10_POWER_ENVELOPE_W, derive_wait_bound
from .joint_replay_regime import normalize_replay_regime

PILOT_SCHEMA = "prismaquant.joint_dispatch_pilot.v2"
QUANTUM_COMPLETION_SCHEMA = "prismaquant.joint_layer_quantum.completion.v1"
#: The CEO approved this 4 MiB quantum control bound in PQ #2603.
#: The producer and consumer use the same bound through the shared reader.
#: The completion carries a descriptor, never the wire or its base64 form.
QUANTUM_RECORD_MAX_BYTES = 4 * 1024 * 1024
PILOT_SOURCE_SCHEMA = "prismaquant.joint_dispatch_pilot.source.v2"
PILOT_PRODUCER_ID = "prismaquant.joint_dispatch_pilot.producer"
#: Aggregate byte ceiling of one pilot publication: completion + counters + original quantum record.
PILOT_PUBLICATION_MAX_BYTES = 16 * 1024 * 1024
PILOT_LAUNCHER = ["python3", "-m", "tools.tessera_campaign_container"]
PILOT_ENTRY = ["python3", "-m", "prismaquant.joint_cost_quantum"]
#: The completion's external original-quantum-record entry names these three
#: fields. Its path is provenance; the consumer supplies an original wire copy
#: and checks its bytes against the declared length and sealed digest.
QUANTUM_RECORD_FIELDS = (
    "path", "sha256", "bytes")


class PilotRefused(ValueError):
    """The counters cannot certify the proposed row."""


def validate_pilot_completion(completion: Mapping, *, counters: Mapping,
                              counters_sha256: str, counters_bytes: int,
                              quantum_record_sha256: str, quantum_record_path,
                              quantum_record_bytes: int | None = None) -> dict:
    """Match supplied counters to an authenticated native producer completion.

    The caller must obtain ``completion`` from the selected native result of
    the successful job. A caller-supplied completion or terminal key alone
    does not establish this provenance. Paths are retained as provenance;
    the byte digest permits an exact copy of the counters to be consumed.

    The completion names the original wire digest and length.
    Both must match the sealed invocation and any independently known length.
    The caller supplies quantum_record_path for an exact original wire copy.
    The producer-local path remains provenance only.
    The declared length bounds the read before allocation.
    The caller derives the pilot binding from these authenticated original bytes.
    Different producer and consumer namespaces do not change that binding.
    """
    try:
        if not isinstance(completion, Mapping) or completion.get("schema") != QUANTUM_COMPLETION_SCHEMA:
            raise PilotRefused("pilot native result has no quantum completion reference")
        if (completion.get("quantum_id") != counters.get("quantum_id")
                or not isinstance(counters.get("quantum_id"), str)
                or not counters["quantum_id"]):
            raise PilotRefused("pilot native result belongs to another quantum")
        units = counters.get("units")
        if (not isinstance(units, list) or len(units) != 2
                or any(type(unit) is not int for unit in units)
                or units[0] != units[1] or units[1] <= 0
                or completion.get("status") != "complete"
                or completion.get("passed") is not True
                or type(completion.get("units_done")) is not int
                or type(completion.get("units_total")) is not int
                or [completion["units_done"], completion["units_total"]] != units):
            raise PilotRefused("pilot native completion and counters disagree on completed units")
        reference = completion["counters"]
        if (not isinstance(reference, Mapping)
                or set(reference) != {"path", "sha256", "bytes"}
                or not isinstance(reference["path"], str) or not reference["path"]
                or type(reference["bytes"]) is not int or reference["bytes"] <= 0
                or type(counters_bytes) is not int or counters_bytes <= 0):
            raise PilotRefused("pilot native counters reference is malformed")
        _pilot_binding_digest(counters_sha256, "supplied counters digest")
        _pilot_binding_digest(reference["sha256"], "producer counters digest")
        if (reference["sha256"] != counters_sha256 or reference["bytes"] != counters_bytes):
            raise PilotRefused("pilot counters bytes differ from the native producer result")
        record = _authenticated_quantum_record(
            completion, quantum_record_sha256=quantum_record_sha256,
            quantum_record_bytes=quantum_record_bytes,
            quantum_record_path=quantum_record_path)
        return {**dict(reference), "quantum_record": record}
    except (KeyError, TypeError, AttributeError) as error:
        raise PilotRefused("pilot native completion reference is incomplete or malformed") from error


def _authenticated_quantum_record(completion, *, quantum_record_sha256,
                                  quantum_record_bytes, quantum_record_path):
    """The completion's external original record entry, authenticated.

    The entry is the producer's own descriptor for the wire its identity
    gate read, bounded before the read and matched byte for byte against
    the sealed --quantum-sha256. The caller supplies a consumer-visible copy.
    The producer path remains provenance and is never reopened here.
    The shared reader bounds the copy before allocation and verifies its digest.
    Missing, malformed, oversized, unreadable, or changed bytes refuse.
    """
    entry = completion.get("quantum_record")
    if not isinstance(entry, Mapping) or set(entry) != set(QUANTUM_RECORD_FIELDS):
        raise PilotRefused("pilot native completion carries no original quantum record")
    path = entry["path"]
    declared_bytes = entry["bytes"]
    declared_sha = entry["sha256"]
    if not isinstance(path, str) or not path:
        raise PilotRefused("pilot completion quantum record path is malformed")
    if type(declared_bytes) is not int or declared_bytes <= 0:
        raise PilotRefused("pilot completion quantum record length is malformed")
    if declared_bytes > QUANTUM_RECORD_MAX_BYTES:
        raise PilotRefused("pilot completion quantum record exceeds the control-byte cap")
    _pilot_binding_digest(declared_sha, "completion quantum record digest")
    _pilot_binding_digest(quantum_record_sha256, "sealed quantum record digest")
    if quantum_record_bytes is not None and (
            type(quantum_record_bytes) is not int or quantum_record_bytes <= 0):
        raise PilotRefused("sealed quantum record length is malformed")
    if ((quantum_record_bytes is not None and declared_bytes != quantum_record_bytes)
            or declared_sha != quantum_record_sha256):
        raise PilotRefused("pilot completion names another quantum record than the sealed argv")
    if not isinstance(quantum_record_path, (str, Path)) or not str(quantum_record_path):
        raise PilotRefused("pilot consumer quantum record path is malformed")
    from .stage_inputs import read_bound as _read_bound
    try:
        raw = _read_bound({"path": str(quantum_record_path), "sha256": declared_sha},
                          "pilot completion quantum record", max_bytes=declared_bytes)
    except (ValueError, OSError) as error:
        raise PilotRefused(
            "pilot completion quantum record bytes do not match their digest") from error
    if len(raw) != declared_bytes or bytes_sha256hex(raw) != declared_sha:
        raise PilotRefused("pilot completion quantum record bytes do not match their digest")
    return raw


def _pilot_binding_digest(value, where):
    if not is_sha256hex(value):
        raise PilotRefused(f"pilot {where}: expected a full SHA-256")
    return value


def validate_pilot_source_contract(value, *, implementation_sha256) -> dict:
    """An independently supplied, reviewed source/invocation contract.

    The source-to-package relation is an explicit reviewer input. Neither
    a result publication nor its counter table can introduce an accepted
    source. The native launch authenticates the executed source (commit and
    tree); this domain contract supplies the source and container spec the
    reviewer accepted. The outer environment is the finite native allowlist,
    not reviewer input.
    """
    fields = {"schema", "source", "implementation_sha256", "launcher_argv",
              "quantum_argv", "container_spec"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != PILOT_SOURCE_SCHEMA:
        raise PilotRefused("pilot source contract is missing, malformed or unsupported")
    if _pilot_binding_digest(value["implementation_sha256"], "accepted code digest") != implementation_sha256:
        raise PilotRefused("pilot source contract does not name the proposed implementation")
    source = value["source"]
    if (not isinstance(source, dict) or set(source) != {"commit", "tree"}
            or any(not isinstance(source[key], str)
                   or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source[key]) is None
                   for key in ("commit", "tree"))):
        raise PilotRefused("pilot source contract needs the exact commit and tree")
    if value["launcher_argv"] != PILOT_LAUNCHER or value["quantum_argv"] != PILOT_ENTRY:
        raise PilotRefused("pilot source contract names an unsupported launcher or quantum entry")
    spec = value["container_spec"]
    if not isinstance(spec, dict) or not isinstance(spec.get("container"), dict) or not isinstance(spec.get("env", {}), dict):
        raise PilotRefused("pilot source contract needs an explicit container spec")
    return value


def publish_pilot_result(publication_path, *, completion: Mapping, counters_path, quantum_record: Mapping) -> dict:
    """Publish one quantum completion as a native result: the completion and counters are
    the artifacts, the original quantum record is the input. Returns the publication binding."""
    from . import native_results
    from .staged_manifest import publish_staged_json
    publication_path = Path(publication_path)
    completion_path = publication_path.with_name(publication_path.name + ".completion.json")
    publish_staged_json(completion_path, dict(completion))
    return native_results.publish_result(
        publication_path, producer_id=PILOT_PRODUCER_ID,
        inputs={"quantum_record": {key: quantum_record[key] for key in QUANTUM_RECORD_FIELDS}},
        artifacts={"completion": completion_path, "counters": Path(counters_path)},
        max_artifact_bytes=PILOT_PUBLICATION_MAX_BYTES)


def pilot_binding(record: Mapping, *, implementation_sha256: str,
                  execution_plan_sha256: str, replay_regime: Mapping | None,
                  cotangent_source: str, emits_handoff: bool) -> dict:
    """Bind the producer's code, actual plan, arithmetic and sealed geometry.

    Input values and output paths do not define geometry. The sealed plan and
    prepared/read-roster identities still constrain the model and calibration;
    window indices, chunk extents and chain geometry constrain the row. This
    deliberately does not generalize a pilot to a different layer or campaign.
    """
    campaign = record["campaign"]
    shape = {
        "layer": record["layer"],
        "campaign": {key: campaign.get(key) for key in (
            "plan_sha256", "prepared_sha256", "read_manifest_sha256",
            "unit_roster_sha256", "campaign_scope")},
        "source_phase": record["read_set"]["source_phase"],
        "chunks": record["chunks"], "windows": record["windows"],
        "adjoint": {key: record["adjoint"][key]
                    for key in ("checkpoint_boundary", "chain_layers")},
        "executable_readset": record.get("executable_readset"),
        "cotangent_source": cotangent_source,
        "emits_handoff": bool(emits_handoff),
    }
    return {
        "schema": PILOT_SCHEMA,
        "implementation_sha256": _pilot_binding_digest(implementation_sha256, "code digest"),
        "execution_plan_sha256": _pilot_binding_digest(execution_plan_sha256, "execution plan digest"),
        "replay_regime": normalize_replay_regime(replay_regime),
        "row_shape_sha256": canonical_json_sha256(shape, where="pilot row shape"),
    }


def _pilot_measurement_number(value, where, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotRefused(f"pilot {where}: missing numeric measurement")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise PilotRefused(f"pilot {where}: invalid measurement {value!r}")
    return float(value)


def _same_number(actual, expected, where):
    actual = _pilot_measurement_number(actual, where)
    # Floating-point reconciliation, not an allowed wait-budget excess.
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise PilotRefused(f"pilot {where}: recorded {actual:g}, derived {expected:g}")


def _pilot_measurement_phase(counters, start):
    phases = [p for p in counters.get("phases", [])
              if isinstance(p, dict) and isinstance(p.get("entered_unix"), (int, float))
              and p["entered_unix"] <= start]
    return max(phases, key=lambda p: p["entered_unix"])["name"] if phases else "outside-phases"


def validate_pilot(counters: Mapping, expected: Mapping) -> dict:
    """Check a published producer receipt; re-derive bounds from its rates.

    The caller binds the exact counter bytes to the selected native result,
    reviewed source contract and authenticated invocation/record. Aggregate
    bound/excess fields alone never authorize a fanout.
    """
    try:
        pilot = counters["pilot"]
        if pilot.get("binding") != expected:
            raise PilotRefused("pilot code, plan, replay regime or row shape does not match")
        job = pilot.get("job_id")
        try:
            from .flux_jobs import job_id_number
            job_id_number(job)
        except ValueError as exc:
            raise PilotRefused("pilot counters name no native producer job") from exc
        units = counters["units"]
        if (not isinstance(units, list) or len(units) != 2
                or any(type(n) is not int for n in units)
                or units[1] <= 0 or units[0] != units[1]
                or counters.get("outcome", {}).get("status") != "complete"):
            raise PilotRefused("pilot outcome or units are not complete")
        samples = counters.get("gpu_sampler_samples")
        if type(samples) is not int or samples <= 0 or counters.get("gpu_sampler_error"):
            raise PilotRefused("pilot has no valid GPU power samples")
        # Kernel timing is optional, including when another session owns
        # the profiler. This gate certifies measured power and wait/rates,
        # not kernel-active time; preserve the producer's diagnostic only.
        power = _pilot_measurement_number(counters.get("gpu_power_w_p50"), "median GPU power", positive=True)
        envelope = _pilot_measurement_number(counters.get("gpu_power_envelope_w"), "GPU envelope", positive=True)
        _same_number(envelope, GB10_POWER_ENVELOPE_W, "GB10 power envelope")
        fraction = power / envelope
        _same_number(pilot.get("gpu_envelope_fraction"), fraction, "GPU envelope fraction")
        report = counters["exposed_wait"]
        if report.get("schema") != EXPOSED_WAIT_SCHEMA or not report.get("instrumented", True):
            raise PilotRefused("pilot exposed wait is not instrumented")
        _same_number(report.get("gpu_power_envelope_w"), envelope, "wait power envelope")
        if not report.get("baseline") or report.get("power_samples", 0) <= 0:
            raise PilotRefused("pilot has no measured idle baseline or wait power samples")
        total, bound = report["total"], report["bound"]
        _pilot_measurement_number(report.get("idle_ceiling_w"), "idle ceiling")
        _same_number(total.get("wait_s"), math.fsum(
            _pilot_measurement_number(total.get(name), name)
            for name in ("idle_band_s", "busy_band_s", "unsampled_s")),
            "power-band wait")
        _same_number(total.get("wait_s"), math.fsum(
            _pilot_measurement_number(block.get("wait_s"), f"{kind} wait")
            for kind, block in total["by_kind"].items())
            - _pilot_measurement_number(total.get("overlap_s"), "overlapping wait"), "kind wait")
        if _pilot_measurement_number(total.get("unsampled_s"), "unsampled wait") > 0:
            raise PilotRefused("pilot exposed wait has unsampled seconds")
        takes = bound["per_take"]
        if not isinstance(takes, list) or not takes:
            raise PilotRefused("pilot has no measured load takes")
        waits, steady_wait, first_wait, bound_sum = {}, 0.0, 0.0, 0.0
        for take in takes:
            start = _pilot_measurement_number(take.get("start_unix"), "take start")
            end = _pilot_measurement_number(take.get("end_unix"), "take end")
            wait = _pilot_measurement_number(take.get("wait_s"), "take wait")
            _same_number(wait, end - start, "take interval")
            kind = take["kind"]
            waits[kind] = waits.get(kind, 0.0) + wait
            if take.get("regime") == "first_fill":
                if take.get("work_before_s") is not None:
                    raise PilotRefused("pilot first fill carries steady-state work")
                first_wait += wait
                continue
            nbytes = _pilot_measurement_number(take.get("bytes"), "take bytes", positive=True)
            work = _pilot_measurement_number(take.get("work_before_s"), "work before take", positive=True)
            load = _pilot_measurement_number(take.get("load_bytes_per_s"), "load_bytes_per_s", positive=True)
            derived = derive_wait_bound(wait_s=wait, nbytes=nbytes,
                                       work_before_s=work, load_bytes_per_s=load)
            if derived["excess_s"] > 0:
                phase = _pilot_measurement_phase(counters, start)
                raise PilotRefused(
                    f"pilot phase {phase}: excess {derived['excess_s']:g}s "
                    f"(wait {wait:g}s, bound {derived['bound_s']:g}s, "
                    f"load_bytes_per_s={load:g}, "
                    f"consume_bytes_per_s={derived['consume_bytes_per_s']:g})")
            steady_wait += wait
            bound_sum += derived["bound_s"]
            _same_number(take.get("bound_s"), derived["bound_s"], "take bound")
            _same_number(take.get("excess_s"), 0.0, "take excess")
        _same_number(bound.get("takes"), len(takes), "take count")
        _same_number(bound.get("steady_wait_s"), steady_wait, "steady wait")
        _same_number(bound.get("first_fill_wait_s"), first_wait, "first-fill wait")
        _same_number(bound.get("bound_s"), bound_sum, "total bound")
        _same_number(bound.get("excess_s"), 0.0, "total excess")
        _same_number(bound.get("unmeasured_takes"), 0.0, "unmeasured takes")
        first_take = min(t["start_unix"] for t in takes)
        for kind, block in total["by_kind"].items():
            wait = _pilot_measurement_number(block.get("wait_s"), f"{kind} wait")
            if wait == 0:
                continue
            # The span wrapper may cover the same render takes. It cannot
            # certify additional blocked seconds not present in those takes.
            covered = waits.get(kind, waits.get("window-load", 0.0)
                                if kind == "window-wait" else 0.0)
            if wait <= covered:
                continue
            # Initial checkpoint/source/handoff loads precede replay, like
            # first fill. An identically named steady-state wait is not exempt.
            spans = [s for s in counters.get("io_spans", []) if s.get("span") == kind]
            if (kind in {"checkpoint-load", "handoff-load", "own-source"}
                    and spans and all(s["end_unix"] <= first_take for s in spans)):
                continue
            raise PilotRefused(f"pilot underived wait kind {kind}: {wait:g}s; no rate bound")
        return {"schema": PILOT_SCHEMA, "mode": "verified", "job_id": job,
                "binding": dict(expected), "gpu_envelope_fraction": fraction,
                "steady_wait_s": steady_wait, "bound_s": bound_sum}
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise PilotRefused(f"pilot counters are incomplete or malformed: {exc}") from exc
