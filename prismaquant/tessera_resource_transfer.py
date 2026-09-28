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
RUNTIME_IDENTITY_SCHEMA = "tessera.native_resource_identity.v1"

def validate_runtime_identity(identity):
    """Full runtime-identity validation, one home beside the report consumer."""
    keys = {"schema", "image_digest", "tessera_package_sha256", "vllm_package_sha256",
            "nccl_version", "family", "world_size", "native_runtime_sha256"}
    if not isinstance(identity, dict) or set(identity) != keys \
            or identity.get("schema") != RUNTIME_IDENTITY_SCHEMA:
        raise ValueError("persistent run runtime identity is incomplete")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity["image_digest"]):
        raise ValueError("persistent run image digest is not immutable")
    for key in ("tessera_package_sha256", "vllm_package_sha256", "native_runtime_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError("persistent run package identity is invalid: " + key)
    version = identity["nccl_version"]
    if (not isinstance(version, list) or len(version) != 3 or
            any(type(v) is not int or v < 0 for v in version)):
        raise ValueError("persistent run NCCL version is missing")
    if type(identity["world_size"]) is not int or identity["world_size"] not in (1, 2):
        raise ValueError("persistent run world must be TP1 or TP2")
    if not re.fullmatch(r"TESSERA_(?:BF16|E4M3|E2M1)_K[12]", identity["family"]):
        raise ValueError("persistent run family identity is invalid")
    return identity
REPORT_SCHEMA_VERSION = 1
TRACE_ANALYZER = "experiments/native_resource_trace.py"
PRODUCER_CLI = "experiments/native_resource_passes.py"


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


def _load_report(path, label, report_sha256):
    """Read a report file, re-hash its bytes against a REQUIRED digest.

    Returns the parsed object; callers must pass it straight to
    :func:`_verify_parsed_report` with the same digest."""
    if not isinstance(report_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha256):
        raise ValueError(f"{label} requires its 64-hex sha256; the digest is the "
                         "evidence chain, not an optional annotation")
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label} file {str(path)!r}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != report_sha256:
        raise ValueError(f"{label} file {str(path)!r} does not hash to its digest")
    try:
        report = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{label} file {str(path)!r} is not JSON: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"{label} file {str(path)!r} must hold a JSON object")
    return report


def _verify_parsed_report(report, *, report_sha256, trace_path=None, expected_runtime=None):
    """Validate an already-parsed report against its REQUIRED bytes digest.

    The digest must have been produced by :func:`verify_run_report` (or
    :func:`_load_report`) from the file bytes; a caller-supplied digest over
    an arbitrary dict is self-referential and refused here by construction.
    """
    if not isinstance(report, dict):
        raise ValueError("run report must be a JSON object")
    if not isinstance(report_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha256):
        raise ValueError("run report verification requires its 64-hex sha256")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"run report schema must be {REPORT_SCHEMA}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"run report schema version must be {REPORT_SCHEMA_VERSION}, "
                         f"refusing {report.get('schema_version')!r}")
    identity = report.get("runtime_identity")
    validate_runtime_identity(identity)
    trace_binding = report.get("trace")
    if (not isinstance(trace_binding, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(trace_binding.get("sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(trace_binding.get("collector_library_sha256")))):
        raise ValueError("run report trace binding is incomplete")
    if expected_runtime is not None and identity != expected_runtime:
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
        process = record.get("process")
        if not isinstance(process, dict) or window.get("process_id") != process.get("pid"):
            raise ValueError(f"pass-R record for {rate} window belongs to another process")
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


def verify_run_report(report_path, *, report_sha256=None, trace_path=None,
                      expected_runtime=None):
    """Verify a run-report FILE: re-hash its bytes, parse, check bindings.

    The public entry point takes the path and does the hashing itself; the
    digest is REQUIRED. Windows are attested measurements: this verifies the
    report's own bindings (bytes -> digest -> schema -> trace sha ->
    collector sha -> per-record window digest/interval/process) and trusts
    the derivation, re-derived only through :func:`rereport_windows`.
    """
    report = _load_report(report_path, "run report", report_sha256)
    return _verify_parsed_report(report, report_sha256=report_sha256,
                                 trace_path=trace_path,
                                 expected_runtime=expected_runtime)


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
    probe = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        raise ValueError(f"tessera analyzer failed: {probe.stderr.strip()[:500]}")
    try:
        derived = json.loads(probe.stdout)["intervals"]
    except (KeyError, ValueError) as error:
        raise ValueError(f"tessera analyzer emitted no windows: "
                         f"{probe.stdout[:120]!r}") from error
    for rate, record in resource.items():
        if derived.get(f"rate:{rate}") != record["window"]:
            raise ValueError(f"re-derived window for {rate} differs from the report's")
    return derived


def _wire_rate(runtime_identity, rate):
    return f"{runtime_identity['family']}_R{rate}"


def qualify_resource_transfer(runtime_identity, *, rates, persistent_report,
                              report_sha256, fresh_reports, noise_band, producer,
                              trace_path=None):
    """Decide whether persistent run-report measurements may substitute for fresh.

    Three-way, per stratified sampled rate: fresh-process ground truth (its
    own single-rate run report) against the persistent report's pass-R window
    (exact allocation-request multiset, bit-exact transient peak, one-time
    initialization multiset) and pass-T timing (raw band gate, r=5 fresh
    repeats). Every input is a verified, digest-bound report; window
    derivations are attested measurements, re-derived only through
    :func:`rereport_windows`. The producer checkout is required and pinned.
    """
    validate_runtime_identity(runtime_identity)
    if runtime_identity["world_size"] != 1:
        raise ValueError("resource transfer is bound per cut axis and rank: "
                         "refusing world_size != 1 until cases name theirs")
    resolve_tessera_checkout(producer["checkout"], producer["commit"])
    if not isinstance(rates, list) or len(rates) < 5 or rates != list(range(rates[0], rates[-1] + 1)):
        raise ValueError("rates must be a consecutive integer domain of at least five")
    persistent_report = _verify_parsed_report(
        _load_report(persistent_report, "run report", report_sha256),
        report_sha256=report_sha256, trace_path=trace_path,
        expected_runtime=runtime_identity)
    # k derives from the band's own fresh evidence, never a magic constant.
    try:
        raw_fresh = noise_band["raw"]["phases"]["prefill"]["fresh"]
    except (KeyError, TypeError) as error:
        raise ValueError("noise band raw.phases.prefill.fresh is missing: the "
                         "band must carry per-rate fresh evidence") from error
    selected = stratified_rates(rates, len(raw_fresh))
    if set(fresh_reports) != set(selected):
        raise ValueError("fresh evidence must cover exactly the stratified sample")
    verified_fresh = {}
    for rate in selected:
        entry = fresh_reports[rate]
        report = _verify_parsed_report(
            _load_report(entry["report"], f"fresh report {rate}", entry["sha256"]),
            report_sha256=entry["sha256"], expected_runtime=runtime_identity)
        wire_rate = _wire_rate(runtime_identity, rate)
        if set(report["pass_r"]) != {wire_rate}:
            raise ValueError(f"fresh report for {rate} must key exactly {wire_rate}")
        verified_fresh[rate] = report

    reasons = []
    checks = []
    gate, gate_reasons = _timing_gate(noise_band, selected)
    reasons.extend(gate_reasons)
    resource_processes, timing_processes = set(), set()
    persistent_trace = persistent_report["trace"]["sha256"]
    for rate in selected:
        wire_rate = _wire_rate(runtime_identity, rate)
        if wire_rate not in persistent_report["pass_r"]:
            raise ValueError(f"run report is missing sampled rate {wire_rate}")
        persistent_resource = persistent_report["pass_r"][wire_rate]
        persistent_timing = persistent_report["pass_t"][wire_rate]
        fresh = verified_fresh[rate]
        fresh_resource = fresh["pass_r"][wire_rate]
        fresh_timing = fresh["pass_t"][wire_rate]
        failures = []
        fresh_window = fresh_resource["window"]
        window = persistent_resource["window"]
        if fresh_window.get("trace_sha256") == persistent_trace:
            failures.append("fresh and persistent windows derive from one trace")
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
        resource_processes.add(digest(persistent_resource["process"]))
        timing_processes.add(digest(persistent_timing["process"]))
        checks.append({"rate": rate, "failures": failures})
        if failures:
            reasons.extend(f"{wire_rate}: {failure}" for failure in failures)
    if len(resource_processes) != 1:
        reasons.append("pass-R sampled rates ran in more than one process")
    if len(timing_processes) != 1:
        reasons.append("pass-T sampled rates ran in more than one process")

    fresh_processes = [digest(next(iter(report["pass_r"].values()))["process"])
                       for report in verified_fresh.values()]
    if len(set(fresh_processes)) != len(fresh_processes):
        reasons.append("fresh reference process was reused across rates")
    persistent_processes = resource_processes | timing_processes
    if set(fresh_processes) & persistent_processes:
        reasons.append("fresh repeats share a process with a persistent pass")
    gate_fresh = set(gate.get("fresh_processes", []))
    if gate_fresh & (set(fresh_processes) | persistent_processes):
        reasons.append("band fresh repeats share a process with a report process")

    # Round-4 B3 restored: the drift axis is the pass-R trace order, and each
    # timing record's ordinal is bound to the band row it claims.
    window_order = [record["q256"] for record in
                    sorted(persistent_report["pass_r"].values(),
                           key=lambda record: record["window"]["begin_ns"])]
    for phase in PHASES:
        raw_phase = noise_band["raw"]["phases"][phase]
        band_rows = raw_phase["persistent"]["rates"]
        if [row["q256"] for row in band_rows] != window_order:
            reasons.append(f"persistent timing order disagrees with the pass-R "
                           f"windows: {phase}")
        for row in band_rows:
            wire_rate = _wire_rate(runtime_identity, row["q256"])
            record = persistent_report["pass_t"].get(wire_rate)
            if record is None:
                reasons.append(f"band names rate {row['q256']} the report does not carry")
                continue
            if row["samples_ms"] != record["samples_ms"][phase]:
                reasons.append(f"band persistent samples for {wire_rate} {phase} differ "
                               f"from the run report")
            if row.get("time_in_process") != record.get("time_in_process"):
                reasons.append(f"band ordinal for {wire_rate} {phase} differs from "
                               f"the run report")
            if raw_phase["persistent"]["process"] != record["process"]:
                reasons.append(f"band persistent process differs from the run report ({phase})")

    status = "passed" if not reasons else "failed"
    artifact = {"schema": QUALIFICATION_SCHEMA, "status": status,
                "fallback_fresh_process": status != "passed",
                "runtime_identity": runtime_identity, "rates": list(rates),
                "sampled_rates": list(selected), "checks": checks,
                "timing_gate": gate, "reasons": sorted(set(reasons)),
                "producer": {"checkout": str(producer["checkout"]),
                             "commit": producer["commit"]},
                "evidence": {"persistent_report_sha256": report_sha256,
                             "noise_band_sha256": digest(noise_band),
                             "fresh_report_sha256s": {str(rate): fresh_reports[rate]["sha256"]
                                                      for rate in selected}}}
    artifact["qualification_id"] = digest({key: value for key, value in artifact.items()
                                           if key != "qualification_id"})
    return artifact


def require_resource_transfer(qualification, *, runtime_identity, persistent_report,
                              report_sha256, fresh_reports, noise_band, producer,
                              trace_path=None):
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
        persistent_report=persistent_report, report_sha256=report_sha256,
        fresh_reports=fresh_reports, noise_band=noise_band, producer=producer,
        trace_path=trace_path)
    if recomputed["qualification_id"] != qualification.get("qualification_id"):
        raise ValueError("resource transfer qualification does not recompute "
                         "from its own evidence")
    if recomputed["status"] != "passed":
        raise ValueError("resource transfer qualification recompute failed: "
                         + "; ".join(recomputed["reasons"][:3]))
    return recomputed


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _main(argv=None):
    """Driver CLI: verify, re-derive and decide over pinned producer output.

    ``run-legs`` shells out to the pinned Tessera checkout's producer entry
    point (``pass-r`` / ``pass-t`` / ``assemble``) for the persistent roster
    and each sampled fresh rate, then writes the manifest the ``qualify``
    subcommand consumes. Exit 2 with a named refusal on any failure.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="prismaquant.tessera_resource_transfer",
        description="Allocator-side qualification over Tessera run reports.")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-report", help="verify one run report")
    verify.add_argument("--report", required=True)
    verify.add_argument("--sha", required=True, help="64-hex sha256 of the report file")
    verify.add_argument("--trace", default=None)
    verify.add_argument("--identity", default=None, help="expected identity JSON file")

    rere = commands.add_parser("rereport", help="re-derive windows via the pinned analyzer")
    rere.add_argument("--tessera-checkout", required=True)
    rere.add_argument("--tessera-commit", required=True)
    rere.add_argument("--report", required=True)
    rere.add_argument("--sha", required=True)
    rere.add_argument("--trace", required=True)

    qualify = commands.add_parser("qualify", help="decide the transfer question")
    qualify.add_argument("--tessera-checkout", required=True)
    qualify.add_argument("--tessera-commit", required=True)
    qualify.add_argument("--identity", required=True, help="runtime identity JSON file")
    qualify.add_argument("--rates", required=True, help="first:last consecutive q256 domain")
    qualify.add_argument("--report", required=True)
    qualify.add_argument("--report-sha", required=True)
    qualify.add_argument("--trace", default=None)
    qualify.add_argument("--fresh-manifest", required=True,
                         help="JSON {rate: {report, sha256}}")
    qualify.add_argument("--band", required=True, help="raw noise band JSON file")
    qualify.add_argument("--out", required=True)

    legs = commands.add_parser("run-legs", help="drive the pinned producer for all legs")
    legs.add_argument("--tessera-checkout", required=True)
    legs.add_argument("--tessera-commit", required=True)
    legs.add_argument("--identity", required=True)
    legs.add_argument("--rates", required=True, help="persistent roster, comma-separated wires")
    legs.add_argument("--fixtures", required=True,
                      help="JSON file mapping wire rates to fixture directories")
    legs.add_argument("--collector-library", required=True)
    legs.add_argument("--producer-python", required=True,
                      help="interpreter of the Tessera runtime the identity names")
    legs.add_argument("--fresh-rates", default=None,
                      help="comma-separated q256 integers (default: derived from --band)")
    legs.add_argument("--band", default=None,
                      help="raw noise band JSON; when given, the fresh sample is "
                           "derived with stratified_rates(rates, k) from the band")
    legs.add_argument("--device-id", type=int, required=True)
    legs.add_argument("--context-id", type=int, required=True)
    legs.add_argument("--out-dir", required=True)

    args = parser.parse_args(argv)

    def read(path, label):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"cannot read {label} file {str(path)!r}: {error}") from error

    if args.command == "verify-report":
        verify_run_report(args.report, report_sha256=args.sha,
                          trace_path=args.trace,
                          expected_runtime=read(args.identity, "identity") if args.identity else None)
        print(json.dumps({"status": "verified", "report": str(args.report),
                          "sha256": args.sha}, sort_keys=True))
    elif args.command == "rereport":
        report = _load_report(args.report, "run report", args.sha)
        windows = rereport_windows(args.tessera_checkout, args.tessera_commit,
                                   report, args.trace)
        print(json.dumps({"status": "rederived", "windows": windows}, sort_keys=True))
    elif args.command == "qualify":
        identity = read(args.identity, "identity")
        first, _, last = args.rates.partition(":")
        rates = list(range(int(first), int(last) + 1))
        manifest = read(args.fresh_manifest, "fresh manifest")
        fresh = {int(rate): entry for rate, entry in manifest.items()}
        band = read(args.band, "noise band")
        artifact = qualify_resource_transfer(
            identity, rates=rates, persistent_report=args.report,
            report_sha256=args.report_sha, fresh_reports=fresh, noise_band=band,
            producer={"checkout": args.tessera_checkout, "commit": args.tessera_commit},
            trace_path=args.trace)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, sort_keys=True)
        if artifact["status"] != "passed":
            raise ValueError("qualification failed: " + "; ".join(artifact["reasons"][:3]))
        print(json.dumps({"status": "passed", "qualification_id": artifact["qualification_id"],
                          "out": str(args.out)}, sort_keys=True))
    else:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        identity = read(args.identity, "identity")
        family = identity["family"]
        checkout = resolve_tessera_checkout(args.tessera_checkout, args.tessera_commit)

        def producer(*argv):
            command = [args.producer_python, "-m", "experiments.native_resource_passes",
                       *[str(arg) for arg in argv]]
            probe = subprocess.run(command, capture_output=True, text=True,
                                   timeout=600, cwd=str(checkout))
            if probe.returncode != 0:
                raise ValueError(f"tessera producer failed: {probe.stderr.strip()[:500]}")
            return probe

        def wire(rate):
            return f"{family}_R{rate}"

        if args.fresh_rates is not None:
            fresh_q256 = [int(value) for value in args.fresh_rates.split(",") if value]
        else:
            if args.band is None:
                raise ValueError("run-legs needs --fresh-rates or --band to derive the sample")
            band = read(args.band, "noise band")
            try:
                raw_fresh = band["raw"]["phases"]["prefill"]["fresh"]
            except (KeyError, TypeError) as error:
                raise ValueError("noise band raw.phases.prefill.fresh is missing") from error
            first_q = int(args.rates.split(",")[0].rsplit("_R", 1)[1])
            last_q = int(args.rates.split(",")[-1].rsplit("_R", 1)[1])
            fresh_q256 = stratified_rates(list(range(first_q, last_q + 1)), len(raw_fresh))

        producer("pass-r", "--rates", args.rates, "--fixtures", args.fixtures,
                 "--collector-library", args.collector_library,
                 "--runtime-identity", args.identity,
                 "--trace", out / "persistent_trace.json",
                 "--out", out / "persistent_records.json",
                 "--device-id", args.device_id, "--context-id", args.context_id)
        producer("pass-t", "--rates", args.rates, "--fixtures", args.fixtures,
                 "--records", out / "persistent_records.json",
                 "--runtime-identity", args.identity,
                 "--out", out / "persistent_timing.json")
        producer("assemble", "--resource", out / "persistent_records.json",
                 "--timing", out / "persistent_timing.json",
                 "--trace", out / "persistent_trace.json",
                 "--out", out / "persistent_report.json")
        manifest = {}
        for rate in fresh_q256:
            stem = f"fresh_{rate}"
            producer("pass-r", "--rates", wire(rate), "--fixtures", args.fixtures,
                     "--collector-library", args.collector_library,
                     "--runtime-identity", args.identity,
                     "--trace", out / f"{stem}_trace.json",
                     "--out", out / f"{stem}_records.json",
                     "--device-id", args.device_id, "--context-id", args.context_id)
            producer("pass-t", "--rates", wire(rate), "--fixtures", args.fixtures,
                     "--records", out / f"{stem}_records.json",
                     "--runtime-identity", args.identity,
                     "--out", out / f"{stem}_timing.json")
            producer("assemble", "--resource", out / f"{stem}_records.json",
                     "--timing", out / f"{stem}_timing.json",
                     "--trace", out / f"{stem}_trace.json",
                     "--out", out / f"{stem}_report.json")
            manifest[str(rate)] = {"report": str(out / f"{stem}_report.json"),
                                   "sha256": _file_sha256(out / f"{stem}_report.json")}
        summary = {"schema": "prismaquant.native_resource_transfer_run_legs.v1",
                   "producer_python": args.producer_python,
                   "persistent_report": str(out / "persistent_report.json"),
                   "persistent_report_sha256": _file_sha256(out / "persistent_report.json"),
                   "trace": str(out / "persistent_trace.json"),
                   "fresh": manifest}
        with open(out / "legs.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, sort_keys=True)
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    import sys

    try:
        _main()
    except ValueError as error:
        print(f"tessera-resource-transfer: {error}", file=sys.stderr)
        raise SystemExit(2) from error
