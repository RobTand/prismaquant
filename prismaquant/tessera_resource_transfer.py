"""Allocator-side native resource-transfer qualification over Tessera run reports.

Verification model (fixed by the coordinator, 2026-09-27): PQ **verifies** a
``tessera.native_persistent_run.v1`` report -- it re-hashes the report file,
binds it to the trace digest and the collector library digest it names, and
checks every window's digest and interval against the report's own records.
PQ **trusts** Tessera's window derivation as an attested measurement. When PQ
needs an independent re-derivation for window exactness, it runs Tessera's
published analyzer (``experiments/native_resource_trace.py``) **as a
subprocess** from a pinned Tessera checkout; it never imports it, and it never
reimplements CUPTI row semantics inside PrismaQuant.

The producer is pinned by commit: the driver takes the Tessera checkout path
and commit as explicit arguments, stamps both into the qualification
artifact, and refuses when the checkout HEAD is not the declared commit. A
qualification inherits the scope it was measured on.

This module makes the reuse decision and nothing else: stratified sampling,
the raw-only FOLLOWUP-7 noise gate, the three-way comparison of
fresh-process ground truth against the persistent run's resource windows and
timing records, and the consumer-side recomputation. Unknown report schema
versions are refused. A failing family keeps fresh-process measurement; no
check here can be loosened by a caller-asserted number.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

PHASES = ("prefill", "decode")
QUALIFICATION_SCHEMA = "prismaquant.native_resource_transfer_qualification.v1"
REPORT_SCHEMA = "tessera.native_persistent_run.v1"
REPORT_SCHEMA_VERSION = 1
TRACE_ANALYZER = "experiments/native_resource_trace.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stratified_rates(rates, count):
    """Stratified integer rates: first, true median and last always present.

    The interior points are the evenly spaced grid positions resolved onto
    unused indices, so the last rate can never be crowded out by the median.
    """
    if (not isinstance(rates, list) or len(rates) < 5 or any(type(r) is not int for r in rates)
            or rates != list(range(rates[0], rates[-1] + 1))):
        raise ValueError("qualification requires the full consecutive integer rate domain")
    if type(count) is not int or not 5 <= count <= len(rates):
        raise ValueError("stratified sample needs between five and every rate")
    last = len(rates) - 1
    middle = last // 2
    picks = {0, last, middle}
    for anchor in (round(i * last / (count - 1)) for i in range(1, count - 1)):
        if len(picks) >= count:
            break
        picks.add(anchor)
    index = 1
    while len(picks) < count:
        picks.add(index)
        index += 1
    if not {0, middle, last} <= picks or len(picks) != count:
        raise ValueError("stratified sample must keep the first, median and last rate")
    return [rates[i] for i in sorted(picks)]



def _samples(values):
    """Median of single-apply samples; three is time_apply's own iterations
    floor (bench_native_operator refuses iterations < 3), not a statistic
    chosen here — the timing authority is the band, never this median."""
    if (not isinstance(values, list) or len(values) < 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError("qualification timing requires positive finite single-apply samples")
    return statistics.median(values)



BAND_SOURCE = "fresh_process_repeat_r5_pooled_log"
GATE_KIND = "not_detected"
FRESH_REPEATS = 5
ALPHA = 0.01


def _band_raw(noise_band):
    """Validate the raw band shape; no caller-asserted number survives this."""
    if (not isinstance(noise_band, dict)
            or set(noise_band) != {"source", "gate_kind", "raw"}
            or noise_band["source"] != BAND_SOURCE
            or noise_band["gate_kind"] != GATE_KIND):
        raise ValueError("timing evidence must be raw fresh-process repeats under a "
                         "not-detected gate; caller-asserted band numbers are refused")
    raw = noise_band["raw"]
    if (not isinstance(raw, dict) or set(raw) != {"eps", "phases"}
            or not isinstance(raw["eps"], dict)
            or set(raw["eps"]) != {"samples_ms", "eps_source"}
            or not isinstance(raw["eps"]["samples_ms"], list) or not raw["eps"]["samples_ms"]
            or not isinstance(raw["eps"]["eps_source"], str) or not raw["eps"]["eps_source"]
            or not isinstance(raw["phases"], dict) or set(raw["phases"]) != set(PHASES)):
        raise ValueError("timing raw evidence needs both phases and the timer's own samples")
    return raw


def _phase_gate(phase, raw_phase, selected, eps_ms, K):
    """FOLLOWUP-7 per-phase gate over raw evidence.

    Everything is re-derived here from raw samples: the r fresh-process
    medians per rate, the pooled log-scale s, the timer floor, the per-rate
    variance screen, the residuals against the common factor and the drift
    test. The gate is a not-detected gate; its MDE is stamped, never hidden.
    """
    from scipy import stats as _st

    if (not isinstance(raw_phase, dict)
            or set(raw_phase) != {"fresh", "persistent"}):
        raise ValueError(f"{phase} raw timing evidence needs fresh repeats and "
                         "the persistent pass")
    fresh, persistent = raw_phase["fresh"], raw_phase["persistent"]
    if not isinstance(fresh, dict) or not fresh:
        return None, [f"{phase}: phase has no raw fresh timing evidence"]
    if len(fresh) < 2:
        return None, [f"{phase}: timing gate requires at least two sampled rates"]
    if set(fresh) != {str(rate) for rate in selected}:
        raise ValueError(f"{phase} raw fresh rates must key the sampled roster exactly")
    r, k = FRESH_REPEATS, len(selected)
    x, medians = {}, {}
    phase_fresh_ids = []
    for key, legs in fresh.items():
        if not isinstance(legs, list) or len(legs) != r:
            raise ValueError(f"{phase} rate {key} needs exactly {r} fresh processes")
        values, procs = [], []
        for leg in legs:
            if (not isinstance(leg, dict) or set(leg) != {"process", "samples_ms"}
                    or not isinstance(leg["process"], dict)):
                raise ValueError(f"{phase} rate {key} fresh legs carry process and samples only")
            values.append(_samples(leg["samples_ms"]))
            procs.append(digest(leg["process"]))
        if len(set(procs)) != r:
            return None, [f"{phase} rate {key}: fresh repeats must be separate processes"]
        # Cross-rate distinctness: the same r processes must not time every
        # rate. All k*r identities in a phase are pairwise distinct.
        phase_fresh_ids.extend(procs)
        medians[key] = values
        x[key] = [math.log(v) for v in values]
    if len(set(phase_fresh_ids)) != k * r:
        return None, [f"{phase}: fresh repeats must be separate processes "
                      "across rates"]
    if (not isinstance(persistent, dict)
            or set(persistent) != {"process", "rates"}
            or not isinstance(persistent["process"], dict)
            or not isinstance(persistent["rates"], list) or not persistent["rates"]):
        raise ValueError(f"{phase} persistent timing evidence names one process and its rates")
    entries = {}
    for entry in persistent["rates"]:
        if (not isinstance(entry, dict)
                or set(entry) != {"q256", "samples_ms", "time_in_process"}
                or entry["q256"] not in selected):
            raise ValueError(f"{phase} persistent rates must carry q256, samples and time in process")
        entries[entry["q256"]] = entry
    if len(entries) != len(persistent["rates"]) or set(entries) != set(selected):
        raise ValueError(f"{phase} persistent rates must key the sampled roster exactly")
    ordered = sorted(persistent["rates"], key=lambda e: e["time_in_process"])
    times = [e["time_in_process"] for e in ordered]
    if (any(type(t) is not int for t in times)
            or any(times[i + 1] <= times[i] for i in range(len(times) - 1))):
        return None, [f"{phase}: persistent rates are not ordered in time"]
    persistent_id = digest(persistent["process"])
    # Mixed-source table: every persistent row must come from the one pass-T
    # process, and no fresh-process row may sit inside the persistent table.
    if persistent_id in set(phase_fresh_ids):
        return None, [f"{phase}: persistent table mixes fresh-process rows "
                      "into the pass-T evidence"]
    y = {str(entry["q256"]): math.log(_samples(entry["samples_ms"])) for entry in ordered}

    reasons = []
    if any(len(set(vals)) < 2 for vals in x.values()):
        reasons.append("timer_cannot_resolve_process_noise: a rate has fewer than "
                       "two distinct fresh medians")
        return None, reasons
    m = {key: statistics.fmean(vals) for key, vals in x.items()}
    s_i = {key: statistics.stdev(vals) for key, vals in x.items()}
    df = k * (r - 1)
    s = math.sqrt(sum((value - m[key]) ** 2 for key, vals in x.items()
                      for value in vals) / df)
    eps_log = {key: math.log(1.0 + eps_ms / math.exp(m[key])) for key in m}
    if s <= 0.0 or max(eps_log.values()) >= s:
        reasons.append("timer_cannot_resolve_process_noise: the timer cannot "
                       "resolve process noise")
        return None, reasons

    factor = math.sqrt((1.0 + 1.0 / r) * (1.0 - 1.0 / k))
    t_crit = float(_st.t.ppf(1.0 - ALPHA / (2.0 * K), df))
    fallback, ratios = [], {}
    for key in sorted(x, key=int):
        others = math.sqrt(sum((value - m[other]) ** 2
                               for other in x if other != key
                               for value in x[other]) / ((k - 1) * (r - 1)))
        ratio = float("inf") if others == 0.0 else (s_i[key] ** 2) / (others ** 2)
        p_screen = float(_st.f.sf(ratio, r - 1, (k - 1) * (r - 1)))
        ratios[key] = {"variance_ratio": ratio if math.isfinite(ratio) else None,
                       "p": p_screen}
        if p_screen < ALPHA / k:
            fallback.append(key)

    d = {key: y[key] - m[key] for key in x}
    d_bar = statistics.fmean(d.values())
    residuals = {}
    for key in sorted(x, key=int):
        if key in fallback:
            v = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[key] ** 2
                                   + sum(s_i[other] ** 2 for other in s_i
                                         if other != key) / k ** 2)
            c_ii = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2
            c_ij = (1.0 + 1.0 / r) / k ** 2
            den = ((c_ii * s_i[key] ** 2) ** 2
                   + sum((c_ij * s_i[other] ** 2) ** 2
                         for other in s_i if other != key))
            df_i = (r - 1) * v ** 2 / den
            threshold = float(_st.t.ppf(1.0 - ALPHA / (2.0 * K), df_i)) * math.sqrt(v)
            threshold += eps_log[key]
            variance_log = v
        else:
            threshold = t_crit * factor * s + eps_log[key]
            variance_log = (s * factor) ** 2
        delta = d[key] - d_bar
        residuals[key] = {"d_minus_d_bar": delta, "threshold_log": threshold,
                          "variance_log": variance_log, "fallback": key in fallback,
                          "passed": abs(delta) <= threshold}
        if not residuals[key]["passed"]:
            reasons.append(f"{phase} rate {key}: residual exceeds the not-detected gate")

    if len(set(d.values())) > 1:
        spearman = cast("tuple[Any, Any]", _st.spearmanr([d[str(e["q256"])] for e in ordered],
                                                         times))
        drift = {"rho": float(spearman[0]), "p": float(spearman[1])}
    else:
        drift = {"rho": None, "p": None}
    if drift["p"] is not None and drift["p"] < ALPHA / K:
        reasons.append(f"persistent_mode_drifts: {phase} (Spearman p={drift['p']:.4g})")

    f_ratio = (sum((value - d_bar) ** 2 for value in d.values()) / (k - 1)
               / (s ** 2 * (1.0 + 1.0 / r)))
    f_p = float(_st.f.sf(f_ratio, k - 1, df))
    failing = [key for key, row in residuals.items() if not row["passed"]]
    if failing and f_p < ALPHA and not (set(failing) & set(fallback)):
        reasons.append(f"persistent_noise_exceeds_fresh: {phase}")
    try:
        wilcoxon_p = float(cast(Any, _st.wilcoxon(list(d.values()))).pvalue)
    except ValueError:
        wilcoxon_p = None
    spread = statistics.stdev(list(d.values())) / math.sqrt(k)
    hw = float(_st.t.ppf(0.995, k - 1)) * spread
    interval = [d_bar - hw, d_bar + hw]
    mde80 = (t_crit + float(_st.norm.ppf(0.8))) * factor * s
    report = {"s_log": s, "df": df, "mde80_log": mde80,
              "fallback_rates": fallback,
              "fallback_note": ("df is per-rate and near r-1; a fallback rate "
                                "certifies almost nothing") if fallback else None,
              "variance_screen": ratios, "residuals": residuals,
              "d_log": d, "d_bar_log": d_bar, "d_bar_interval_log": interval,
              "drift_spearman": drift, "wilcoxon_p": wilcoxon_p,
              "f_diagnostic": {"ratio": f_ratio, "p": f_p},
              "fresh_medians_ms": {key: medians[key] for key in medians},
              "persistent_medians_ms": {key: math.exp(y[key]) for key in y},
              "persistent_process": persistent["process"],
              "persistent_order": [entry["q256"] for entry in ordered],
              "persistent_ordinals": {str(entry["q256"]): entry["time_in_process"]
                                      for entry in ordered},
              "fresh_processes": phase_fresh_ids}
    return report, reasons


def _timing_gate(noise_band, selected):
    """Both phases plus the shared eps; one persistent process across phases."""
    raw = _band_raw(noise_band)
    eps_samples = [value for value in raw["eps"]["samples_ms"]
                   if type(value) in (int, float) and math.isfinite(value) and value > 0]
    if not eps_samples:
        return ({"gate_kind": GATE_KIND, "alpha": ALPHA, "r": FRESH_REPEATS,
                 "phases": {}},
                ["timer_cannot_resolve_process_noise: the timer produced no positive sample"])
    k = len(selected)
    K = 2 * k + 2
    eps_ms = min(eps_samples)
    reasons, phases, processes = [], {}, []
    fresh_ids = set()
    for phase in PHASES:
        report, phase_reasons = _phase_gate(phase, raw["phases"][phase],
                                            selected, eps_ms, K)
        reasons.extend(phase_reasons)
        if report is not None:
            phases[phase] = report
            processes.append(digest(report["persistent_process"]))
            fresh_ids.update(report["fresh_processes"])
    if len(processes) == len(PHASES) and len(set(processes)) != 1:
        reasons.append("persistent samples must come from one process")
    return ({"gate_kind": GATE_KIND, "alpha": ALPHA, "r": FRESH_REPEATS, "K": K,
             "eps_ms": eps_ms, "eps_source": raw["eps"]["eps_source"],
             "fresh_processes": sorted(fresh_ids), "phases": phases}, reasons)




def _persistent_rows(noise_band):
    """One persistent process and its per-rate raw samples, re-read from raw."""
    raw = _band_raw(noise_band)
    rows = {"process": None, "samples": {}, "ordinals": {}}
    for phase in PHASES:
        persistent = raw["phases"][phase]["persistent"]
        process = digest(persistent["process"])
        if rows["process"] not in (None, process):
            raise ValueError("persistent samples must come from one process")
        rows["process"] = process
        for entry in persistent["rates"]:
            rows["samples"].setdefault(entry["q256"], {})[phase] = entry["samples_ms"]
            known = rows["ordinals"].setdefault(entry["q256"], entry["time_in_process"])
            if known != entry["time_in_process"]:
                raise ValueError("persistent time in process disagrees between phases")
    return rows





def resolve_tessera_checkout(checkout, commit):
    """Refuse a producer checkout whose HEAD is not the declared commit."""
    checkout = Path(checkout)
    if not checkout.is_dir():
        raise ValueError(f"tessera checkout {str(checkout)!r} is not a directory")
    probe = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise ValueError(f"tessera checkout {str(checkout)!r} is not a git repository")
    head = probe.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("tessera producer commit must be a full 40-hex sha")
    if head != commit:
        raise ValueError(f"tessera checkout HEAD {head} is not the declared commit {commit}")
    analyzer = checkout / TRACE_ANALYZER
    if not analyzer.is_file():
        raise ValueError(f"pinned tessera checkout has no published analyzer at {TRACE_ANALYZER}")
    return checkout


def verify_run_report(report, *, report_sha256=None, trace_path=None, expected_runtime=None):
    """Verify a run report's own bindings; trust its windows as attested measurement.

    Re-hashes the report when a digest is supplied, refuses unknown schema
    versions, requires one runtime identity across both passes, and binds the
    trace (canonical digest and, when the trace file is given, its file sha)
    plus the collector library digest to every pass-R record's window.
    """
    if not isinstance(report, dict):
        raise ValueError("run report must be a JSON object")
    if report_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(report_sha256)):
            raise ValueError("report digest must be a 64-hex sha256")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"run report schema must be {REPORT_SCHEMA}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"run report schema version must be {REPORT_SCHEMA_VERSION}, "
                         f"refusing {report.get('schema_version')!r}")
    trace_binding = report.get("trace")
    if (not isinstance(trace_binding, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(trace_binding.get("sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(trace_binding.get("collector_library_sha256")))):
        raise ValueError("run report trace binding is incomplete")
    if expected_runtime is not None and report.get("runtime_identity") != expected_runtime:
        raise ValueError("run report runtime identity differs from the expected runtime")
    resource = report.get("pass_r")
    timing = report.get("pass_t")
    if (not isinstance(resource, dict) or not resource
            or not isinstance(timing, dict) or not timing
            or set(resource) != set(timing)):
        raise ValueError("run report passes must key the same nonempty roster")
    for rate, record in resource.items():
        window = record.get("window")
        if (not isinstance(window, dict)
                or window.get("trace_sha256") != trace_binding["sha256"]
                or window.get("interval") != f"rate:{rate}"):
            raise ValueError(f"pass-R record for {rate} is not bound to this report's trace")
        if record.get("collector", {}).get("library_sha256") != trace_binding["collector_library_sha256"]:
            raise ValueError(f"pass-R record for {rate} names another collector library")
    for rate, record in timing.items():
        if record.get("collector_started") is not False:
            raise ValueError(f"pass-T record for {rate} claims a started collector")
        samples = record.get("samples_ms")
        if not isinstance(samples, dict) or set(samples) != {"prefill", "decode"}:
            raise ValueError(f"pass-T record for {rate} has no per-phase samples")
    if trace_path is not None:
        trace_path = Path(trace_path)
        if not trace_path.is_file():
            raise ValueError(f"trace file {str(trace_path)!r} is missing")
        raw = trace_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != trace_binding.get("file_sha256"):
            raise ValueError("trace file bytes do not hash to the report's file digest")
        if digest(json.loads(raw)) != trace_binding["sha256"]:
            raise ValueError("trace object does not hash to the report's trace digest")
    return report


def rereport_windows(checkout, commit, report, trace_path):
    """Re-derive a report's windows through Tessera's published analyzer.

    Runs the pinned checkout's analyzer as a subprocess and refuses when any
    derived window differs from the report's attested window. This is the
    optional independent re-derivation of the verification model; the default
    path verifies bindings and trusts the derivation.
    """
    checkout = resolve_tessera_checkout(checkout, commit)
    resource = report["pass_r"]
    command = [sys.executable, str(checkout / TRACE_ANALYZER), str(trace_path),
               "--device-id", str(report["device_id"]),
               "--context-id", str(report["context_id"])]
    for rate in resource:
        command += ["--interval", f"rate:{rate}"]
    probe = subprocess.run(command, capture_output=True, text=True)
    if probe.returncode != 0:
        raise ValueError(f"tessera analyzer failed: {probe.stderr.strip()[:200]}")
    derived = json.loads(probe.stdout)["intervals"]
    for rate, record in resource.items():
        if derived[f"rate:{rate}"] != record["window"]:
            raise ValueError(f"re-derived window for {rate} differs from the report's")
    return derived


def _sampled_report_rates(rates, fresh_reports):
    selected = stratified_rates(rates, 9)
    if set(fresh_reports) != set(selected):
        raise ValueError("fresh evidence must cover exactly the stratified sample")
    return selected


def qualify_resource_transfer(runtime_identity, *, rates, persistent_report,
                              fresh_reports, noise_band, producer=None,
                              trace_path=None, report_sha256=None,
                              fresh_report_sha256s=None):
    """Decide whether persistent run-report measurements may substitute for fresh.

    Three-way, per stratified sampled rate: fresh-process ground truth (its
    own single-rate run report) against the persistent report's pass-R window
    (exact allocation-request multiset, bit-exact transient peak, one-time
    initialization multiset) and pass-T timing (raw band gate, r=5 fresh
    repeats). Every input is a verified report; window derivations are
    attested measurements, re-derived only through :func:`rereport_windows`.
    """
    if not isinstance(runtime_identity, dict) or runtime_identity.get("world_size") != 1:
        raise ValueError("resource transfer is bound per cut axis and rank: "
                         "refusing world_size != 1 until cases name theirs")
    if producer is not None:
        resolve_tessera_checkout(producer["checkout"], producer["commit"])
    persistent_report = verify_run_report(
        persistent_report, report_sha256=report_sha256, trace_path=trace_path,
        expected_runtime=runtime_identity)
    if persistent_report["runtime_identity"] != runtime_identity:
        raise ValueError("run report runtime identity differs from the qualification runtime")
    selected = _sampled_report_rates(rates, fresh_reports)
    verified_fresh = {}
    for rate in selected:
        entry = fresh_reports[rate]
        report = verify_run_report(entry["report"], report_sha256=entry.get("sha256"),
                                   expected_runtime=runtime_identity)
        if report["runtime_identity"] != runtime_identity:
            raise ValueError(f"fresh report for {rate} is another runtime")
        if len(report["pass_r"]) != 1:
            raise ValueError(f"fresh report for {rate} is not a single-rate run")
        verified_fresh[rate] = report

    reasons = []
    checks = []
    gate, gate_reasons = _timing_gate(noise_band, selected)
    reasons.extend(gate_reasons)
    persistent_resource_process = None
    timing_processes = set()
    for rate in selected:
        wire_rate = f"{runtime_identity['family']}_R{rate}"
        persistent_resource = persistent_report["pass_r"][wire_rate]
        persistent_timing = persistent_report["pass_t"][wire_rate]
        fresh = verified_fresh[rate]
        fresh_rate = next(iter(fresh["pass_r"]))
        fresh_resource = fresh["pass_r"][fresh_rate]
        fresh_timing = fresh["pass_t"][fresh_rate]
        failures = []
        fresh_window = fresh_resource["window"]
        window = persistent_resource["window"]
        if fresh_window.get("allocation_requests") != window.get("allocation_requests"):
            failures.append("allocation requests differ from fresh ground truth")
        if fresh_window.get("transient_peak_bytes") != window.get("transient_peak_bytes"):
            failures.append("transient peak differs from fresh ground truth")
        if fresh_window.get("initialization_requests") != window.get("initialization_requests"):
            failures.append("initialization requests differ from fresh ground truth")
        if fresh_resource.get("binding") != persistent_resource.get("binding"):
            failures.append("prepared identity binding differs between legs")
        if fresh_resource["process"] == persistent_resource["process"]:
            failures.append("fresh and persistent resource passes share a process")
        timing_processes.add(digest(persistent_timing["process"]))
        persistent_resource_process = digest(persistent_resource["process"])
        checks.append({"rate": rate, "failures": failures})
        if failures:
            reasons.extend(f"{wire_rate}: {failure}" for failure in failures)

    fresh_processes = [digest(next(iter(report["pass_r"].values()))["process"])
                       for report in verified_fresh.values()]
    if len(set(fresh_processes)) != len(fresh_processes):
        reasons.append("fresh reference process was reused across rates")
    persistent_processes = {persistent_resource_process} | timing_processes
    if set(fresh_processes) & persistent_processes:
        reasons.append("fresh repeats share a process with a persistent pass")

    for phase in PHASES:
        raw_phase = noise_band["raw"]["phases"][phase]
        persistent_rows = raw_phase["persistent"]["rates"]
        for row in persistent_rows:
            wire_rate = f"{runtime_identity['family']}_R{row['q256']}"
            record = persistent_report["pass_t"].get(wire_rate)
            if record is None:
                reasons.append(f"band names rate {row['q256']} the report does not carry")
                continue
            if row["samples_ms"] != record["samples_ms"][phase]:
                reasons.append(f"band persistent samples for {wire_rate} {phase} differ "
                               f"from the run report")
            if raw_phase["persistent"]["process"] != record["process"]:
                reasons.append(f"band persistent process differs from the run report ({phase})")

    status = "passed" if not reasons else "failed"
    artifact = {"schema": QUALIFICATION_SCHEMA, "status": status,
                "fallback_fresh_process": status != "passed",
                "runtime_identity": runtime_identity, "rates": list(rates),
                "sampled_rates": list(selected), "checks": checks,
                "timing_gate": gate, "reasons": sorted(set(reasons)),
                "producer": None if producer is None else {
                    "checkout": str(producer["checkout"]), "commit": producer["commit"]},
                "evidence": {"persistent_report_sha256": report_sha256,
                             "fresh_report_sha256s": {str(rate): fresh_reports[rate].get("sha256")
                                                      for rate in selected}}}
    artifact["qualification_id"] = digest({key: value for key, value in artifact.items()
                                           if key != "qualification_id"})
    return artifact


def require_resource_transfer(qualification, *, runtime_identity, persistent_report,
                              fresh_reports, noise_band, producer=None, trace_path=None,
                              report_sha256=None):
    """Recompute a qualification from its evidence; refuse anything that moved."""
    if not isinstance(qualification, dict) or qualification.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("resource transfer requires a qualification artifact")
    internal = digest({key: value for key, value in qualification.items()
                       if key != "qualification_id"})
    if internal != qualification.get("qualification_id"):
        raise ValueError("resource transfer qualification is not self-consistent: "
                         "its digest does not cover its own fields")
    recomputed = qualify_resource_transfer(
        runtime_identity, rates=qualification.get("rates", []),
        persistent_report=persistent_report, fresh_reports=fresh_reports,
        noise_band=noise_band, producer=producer, trace_path=trace_path,
        report_sha256=report_sha256)
    if recomputed["qualification_id"] != qualification.get("qualification_id"):
        raise ValueError("resource transfer qualification does not recompute "
                         "from its own evidence")
    if recomputed["status"] != "passed":
        raise ValueError("resource transfer qualification recompute failed: "
                         + "; ".join(recomputed["reasons"][:3]))
    return recomputed
