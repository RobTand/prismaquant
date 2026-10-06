"""Producer-bound admission for joint-quantum fanout (PQ #1293).

This is a measurement gate, not a scheduler or a performance estimator.
Missing measurements refuse; an operator may explicitly override the gate.
"""
from __future__ import annotations

import base64
import re
import math
from collections.abc import Mapping

from .digests import bytes_sha256hex, canonical_json_sha256, is_sha256hex
from .io_spans import EXPOSED_WAIT_SCHEMA, GB10_POWER_ENVELOPE_W, derive_wait_bound
from .joint_replay_regime import normalize_replay_regime

PILOT_SCHEMA = "prismaquant.joint_dispatch_pilot.v1"
QUANTUM_COMPLETION_SCHEMA = "prismaquant.joint_layer_quantum.completion.v1"
QUANTUM_RECORD_MAX_BYTES = 1024 * 1024
PILOT_SOURCE_SCHEMA = "prismaquant.joint_dispatch_pilot.source.v1"
PILOT_LAUNCHER = ["python3", "-m", "tools.tessera_campaign_container"]
PILOT_ENTRY = ["python3", "-m", "prismaquant.joint_cost_quantum"]

#: The completion's inlined original-quantum-record block names these three
#: fields; the wire bytes are base64 and must decode to exactly
#: ``quantum_record_bytes`` bytes hashing to ``quantum_record_sha256``.
QUANTUM_RECORD_FIELDS = (
    "quantum_record", "quantum_record_bytes", "quantum_record_sha256")


class PilotRefused(ValueError):
    """The counters cannot certify the proposed row."""


def validate_pilot_completion(completion: Mapping, *, counters: Mapping,
                              counters_sha256: str, counters_bytes: int,
                              quantum_record_sha256: str,
                              quantum_record_bytes: int | None = None) -> dict:
    """Match supplied counters to an authenticated PB producer completion.

    The caller must obtain ``completion`` from the successful action's
    verified CAS result. A caller-supplied completion or terminal key alone
    does not establish this provenance. Paths are retained as provenance;
    the byte digest permits an exact copy of the counters to be consumed.

    The completion also inlines the ORIGINAL quantum wire bytes the producer
    authenticated (PQ #1293). ``quantum_record_sha256``/``quantum_record_bytes``
    authenticates the sealed ``--quantum-sha256``; an optional independently
    known length must agree too. The declared length is capped before decode,
    then checked against the owned bytes. The caller derives the pilot binding
    from those authenticated bytes, never from the record path, so
    cross-output-namespace equivalence is preserved.
    """
    try:
        if not isinstance(completion, Mapping) or completion.get("schema") != QUANTUM_COMPLETION_SCHEMA:
            raise PilotRefused("pilot PB result has no quantum completion reference")
        if (completion.get("quantum_id") != counters.get("quantum_id")
                or not isinstance(counters.get("quantum_id"), str)
                or not counters["quantum_id"]):
            raise PilotRefused("pilot PB result belongs to another quantum")
        units = counters.get("units")
        if (not isinstance(units, list) or len(units) != 2
                or any(type(unit) is not int for unit in units)
                or units[0] != units[1] or units[1] <= 0
                or completion.get("status") != "complete"
                or completion.get("passed") is not True
                or type(completion.get("units_done")) is not int
                or type(completion.get("units_total")) is not int
                or [completion["units_done"], completion["units_total"]] != units):
            raise PilotRefused("pilot PB completion and counters disagree on completed units")
        reference = completion["counters"]
        if (not isinstance(reference, Mapping)
                or set(reference) != {"path", "sha256", "bytes"}
                or not isinstance(reference["path"], str) or not reference["path"]
                or type(reference["bytes"]) is not int or reference["bytes"] <= 0
                or type(counters_bytes) is not int or counters_bytes <= 0):
            raise PilotRefused("pilot PB counters reference is malformed")
        _pilot_binding_digest(counters_sha256, "supplied counters digest")
        _pilot_binding_digest(reference["sha256"], "producer counters digest")
        if (reference["sha256"] != counters_sha256 or reference["bytes"] != counters_bytes):
            raise PilotRefused("pilot counters bytes differ from the PB producer result")
        record = _authenticated_quantum_record(
            completion, quantum_record_sha256=quantum_record_sha256,
            quantum_record_bytes=quantum_record_bytes)
        return {**dict(reference), "quantum_record": record}
    except (KeyError, TypeError, AttributeError) as error:
        raise PilotRefused("pilot PB completion reference is incomplete or malformed") from error


def _authenticated_quantum_record(completion, *, quantum_record_sha256,
                                  quantum_record_bytes):
    """The completion's inlined original record bytes, authenticated.

    The inlined block is the producer's own owned read, bounded before decode
    and matched byte for byte against the sealed ``--quantum-sha256``. A
    missing, oversized, over-large-after-decode or digest-mismatched block
    refuses; no path is reopened and nothing is canonical-reencoded.
    """
    if set(completion) & set(QUANTUM_RECORD_FIELDS) != set(QUANTUM_RECORD_FIELDS):
        raise PilotRefused("pilot PB completion carries no original quantum record")
    encoded = completion["quantum_record"]
    declared_bytes = completion["quantum_record_bytes"]
    declared_sha = completion["quantum_record_sha256"]
    if not isinstance(encoded, str) or not encoded:
        raise PilotRefused("pilot completion quantum record is not a base64 string")
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
    if len(encoded) > _MAX_BASE64_CHARS(declared_bytes):
        raise PilotRefused("pilot completion quantum record exceeds its declared length")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise PilotRefused("pilot completion quantum record is not valid base64") from error
    if len(raw) != declared_bytes or bytes_sha256hex(raw) != declared_sha:
        raise PilotRefused("pilot completion quantum record bytes do not match their digest")
    return raw


def _MAX_BASE64_CHARS(declared_bytes: int) -> int:
    """The most base64 characters that can carry ``declared_bytes`` bytes.

    Computed before decoding, so an oversized or hostile block is refused on
    its encoded length rather than after allocating the decoded form. Four
    characters per three bytes, rounded up to the next 4-character group.
    """
    return 4 * ((declared_bytes + 2) // 3)


def _pilot_binding_digest(value, where):
    if not is_sha256hex(value):
        raise PilotRefused(f"pilot {where}: expected a full SHA-256")
    return value


def validate_pilot_source_contract(value, *, implementation_sha256) -> dict:
    """An independently supplied, reviewed source/invocation contract.

    The snapshot-to-package relation is an explicit reviewer input. Neither
    a result payload nor its counter table can introduce an accepted source.
    PB authenticates the matching declared input descriptor; this domain
    contract supplies the source and environment the reviewer accepted.
    """
    fields = {"schema", "snapshot", "snapshot_selection", "implementation_sha256", "launcher_argv",
              "quantum_argv", "container_spec", "outer_environment"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != PILOT_SOURCE_SCHEMA:
        raise PilotRefused("pilot source contract is missing, malformed or unsupported")
    if _pilot_binding_digest(value["implementation_sha256"], "accepted code digest") != implementation_sha256:
        raise PilotRefused("pilot source contract does not name the proposed implementation")
    descriptor = value["snapshot"]
    if (not isinstance(descriptor, dict) or set(descriptor) != {"id", "sha256", "bytes"}
            or descriptor["id"] != "pbrun.checkout-snapshot"
            or type(descriptor["bytes"]) is not int or descriptor["bytes"] <= 0):
        raise PilotRefused("pilot source contract needs an exact snapshot input descriptor")
    _pilot_binding_digest(descriptor["sha256"], "accepted snapshot digest")
    selection = value["snapshot_selection"]
    if (not isinstance(selection, dict) or selection.get("input") != descriptor
            or not isinstance(selection.get("commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", selection["commit"]) is None):
        raise PilotRefused("pilot source contract needs the exact snapshot selection")
    if value["launcher_argv"] != PILOT_LAUNCHER or value["quantum_argv"] != PILOT_ENTRY:
        raise PilotRefused("pilot source contract names an unsupported launcher or quantum entry")
    spec, environment = value["container_spec"], value["outer_environment"]
    if (not isinstance(spec, dict) or not isinstance(spec.get("container"), dict)
            or not isinstance(spec.get("env", {}), dict)
            or not isinstance(environment, dict) or not environment.get("PATH")
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())):
        raise PilotRefused("pilot source contract needs explicit container and outer environments")
    return value


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

    The caller binds the exact counter bytes to the selected PB result,
    reviewed source contract and authenticated invocation/record. Aggregate
    bound/excess fields alone never authorize a fanout.
    """
    try:
        pilot = counters["pilot"]
        if pilot.get("binding") != expected:
            raise PilotRefused("pilot code, plan, replay regime or row shape does not match")
        key = _pilot_binding_digest(pilot.get("action_key"), "PB action key")
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
        return {"schema": PILOT_SCHEMA, "mode": "verified", "action_key": key,
                "binding": dict(expected), "gpu_envelope_fraction": fraction,
                "steady_wait_s": steady_wait, "bound_s": bound_sum}
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise PilotRefused(f"pilot counters are incomplete or malformed: {exc}") from exc
