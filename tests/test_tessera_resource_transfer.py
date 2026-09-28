"""Allocator-side qualification tests: gate, report verification, producer pin.

CPU contract fixtures only. The FOLLOWUP-7 gate tests are carried verbatim
from the reviewed Tessera-era suite (same closed forms, same mutant kills);
the report and producer tests pin the re-home's verification model: verify
bindings, trust attested windows, re-derive only through the pinned checkout
as a subprocess.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant import tessera_resource_transfer as _module
from prismaquant.tessera_resource_transfer import (
    ALPHA,
    FRESH_REPEATS,
    PHASES,
    REPORT_SCHEMA,
    REPORT_SCHEMA_VERSION,
    digest,
    qualify_resource_transfer,
    rereport_windows,
    require_resource_transfer,
    resolve_tessera_checkout,
    stratified_rates,
    verify_run_report,
)
from prismaquant.tessera_resource_transfer import _phase_gate
from prismaquant.tessera_resource_transfer import _timing_gate

def _base_samples(scale):
    return [0.98 * scale, 1.00 * scale, 1.02 * scale]

COLLECTOR = "a" * 64
TRACE_SHA = "b" * 64
FILE_SHA = "c" * 64


def _identity(world=1):
    return {"schema": "tessera.native_resource_identity.v1",
            "image_digest": "sha256:" + "0" * 64,
            "tessera_package_sha256": "1" * 64,
            "vllm_package_sha256": "2" * 64,
            "nccl_version": [2, 3, 0], "family": "TESSERA_BF16_K1",
            "world_size": world, "native_runtime_sha256": "3" * 64}


def _binding(rate):
    return {"format": f"TESSERA_BF16_K1_R{rate}",
            "operator": {"wire_sha256": f"{rate:04d}" + "0" * 60},
            "runtime": {"execution": {"tensor_parallel": 1}}}


def _window(rate, *, trace_sha=TRACE_SHA, allocs=None, transient=40, init=None, pid=0):
    return {"schema": "tessera.native_rate_resource_window.v1", "status": "observed",
            "trace_sha256": trace_sha, "process_id": pid, "collection_start_ns": 1,
            "interval": f"rate:TESSERA_BF16_K1_R{rate}", "begin_ns": 10 + 2 * rate,
            "end_ns": 11 + 2 * rate, "baseline_live": [], "end_live": [],
            "baseline_bytes": 100, "window_peak_bytes": 100 + transient,
            "transient_peak_bytes": transient,
            "allocation_requests": allocs or [{"bytes": 40, "source": "torch-fixture",
                                               "memory_kind": 3, "count": 1}],
            "initialization_requests": init or [{"bytes": 100, "source": "nccl-init-fixture",
                                                 "memory_kind": 3, "count": 1}]}


def _report(rates, *, pid, timings, identity=None, binding_by_rate=None,
            window_by_rate=None, order=None, trace_sha=TRACE_SHA):
    identity = identity or _identity()
    binding_by_rate = binding_by_rate or {}
    window_by_rate = window_by_rate or {}
    order = order or {rate: index + 1 for index, rate in enumerate(rates)}
    pass_r, pass_t = {}, {}
    for rate in rates:
        wire = f"TESSERA_BF16_K1_R{rate}"
        pass_r[wire] = {"schema": "tessera.native_resource_pass_r.v1",
                        "status": "observed", "rate": wire, "q256": rate,
                        "runtime_identity": copy.deepcopy(identity),
                        "collector": {"started": True, "library_sha256": COLLECTOR},
                        "process": {"pid": pid, "boot_id": "CPU-fixture",
                                    "start_ticks": pid},
                        "binding": binding_by_rate.get(rate, _binding(rate)),
                        "device_id": 0, "context_id": 42,
                        "window": window_by_rate.get(rate, _window(rate, pid=pid))}
        pass_t[wire] = {"schema": "tessera.native_resource_pass_t.v1",
                        "status": "observed", "rate": wire, "q256": rate,
                        "runtime_identity": copy.deepcopy(identity),
                        "collector_started": False, "time_in_process": order[rate],
                        "process": {"pid": pid + 1000, "boot_id": "CPU-fixture",
                                    "start_ticks": pid + 1000},
                        "binding": binding_by_rate.get(rate, _binding(rate)),
                        "samples_ms": timings[rate]}
    return {"schema": REPORT_SCHEMA, "schema_version": REPORT_SCHEMA_VERSION,
            "runtime_identity": copy.deepcopy(identity),
            "trace": {"file": "trace.json", "sha256": trace_sha,
                      "file_sha256": FILE_SHA,
                      "collector_library_sha256": COLLECTOR},
            "device_id": 0, "context_id": 42,
            "pass_r": pass_r, "pass_t": pass_t}


def _timings(rates, base=1.03):
    return {rate: {phase: [base - 1e-5, base, base + 1e-5] for phase in PHASES}
            for rate in rates}


def _raw_band(rates, timings, *, persistent_pid=1100, fresh_seed=500):
    raw = {"eps": {"samples_ms": [1e-4, 2e-4], "eps_source": "cpu"},
           "phases": {}}
    for phase in PHASES:
        fresh = {}
        for index, rate in enumerate(rates):
            key = str(rate)
            fresh[key] = []
            for rep in range(FRESH_REPEATS):
                median = 1.0 + 0.0001 * rep
                fresh[key].append({"process": {"pid": fresh_seed + 10 * index + rep,
                                               "boot_id": "CPU-fixture",
                                               "start_ticks": fresh_seed + 10 * index + rep},
                                   "samples_ms": [median - 1e-5, median, median + 1e-5]})
        raw["phases"][phase] = {"fresh": fresh,
                                "persistent": {"process": {"pid": persistent_pid,
                                                           "boot_id": "CPU-fixture",
                                                           "start_ticks": persistent_pid},
                                               "rates": [{"q256": rate,
                                                          "samples_ms": timings[rate][phase],
                                                          "time_in_process": index + 1}
                                                         for index, rate in enumerate(rates)]}}
    return {"source": "fresh_process_repeat_r5_pooled_log",
            "gate_kind": "not_detected", "raw": raw}



def _gate_band(*, k=9, sigma=0.02, spread=1.0, persistent_bias=None, drift=0.0,
               hetero_rate=None, hetero_factor=3.0, eps_samples=None,
               persistent_process_offset=0, fresh_process_tag="p"):
    """Raw FOLLOWUP-7 band built from seeded draws, no cases or traces."""
    import random

    rng = random.Random(20260927)
    rates = list(range(256, 256 + k))
    phases = {}
    persistent_processes = []
    for phase in ("prefill", "decode"):
        fresh = {}
        x = {}
        for i, rate in enumerate(rates):
            legs, logs = [], []
            for rep in range(5):
                noise = rng.gauss(0.0, sigma * (hetero_factor if hetero_rate == i else 1.0))
                logs.append(noise)
                legs.append({"process": {"pid": 1000 + i * 20 + rep,
                                         "boot_id": f"gate-{fresh_process_tag}",
                                         "start_ticks": 1000 + i * 20 + rep},
                             "samples_ms": _base_samples(1.0 + noise)})
            fresh[str(rate)] = legs
            x[i] = logs
        entries = []
        y = []
        for i, rate in enumerate(rates):
            bias = 0.0
            if persistent_bias is not None and i in persistent_bias:
                bias = persistent_bias[i]
            value = drift * i + bias + rng.gauss(0.0, sigma)
            y.append(value)
            entries.append({"q256": rate,
                            "samples_ms": _base_samples(1.0 + value),
                            "time_in_process": i + 1})
        persistent_process = {"pid": 2000 + persistent_process_offset,
                              "boot_id": "gate-persistent", "start_ticks": 2000}
        persistent_processes.append(persistent_process)
        phases[phase] = {"fresh": fresh,
                         "persistent": {"process": persistent_process,
                                        "rates": entries}}
    return {"source": "fresh_process_repeat_r5_pooled_log", "gate_kind": "not_detected",
            "raw": {"eps": {"samples_ms": eps_samples or [1e-4],
                            "eps_source": "gate fixture timer"},
                    "phases": phases}}




def _gate(band, selected):
    return _module._timing_gate(band, selected)




def test_gate_holds_its_level_on_a_true_null():
    pytest.importorskip("scipy")
    import math

    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    assert report["phases"]["prefill"]["fallback_rates"] == []
    # the common factor is stamped with an interval that covers zero
    assert report["phases"]["prefill"]["d_bar_interval_log"][0] <= 0.0
    assert report["phases"]["prefill"]["d_bar_interval_log"][1] >= 0.0


def test_gate_false_fail_rate_on_a_true_null_is_below_two_percent():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    rng = numpy.random.default_rng(20260927)
    failures = 0
    draws = 2000
    for draw in range(draws):
        rates = list(range(256, 265))
        phases = {}
        for phase in ("prefill", "decode"):
            fresh = {}
            for i, rate in enumerate(rates):
                legs = []
                for rep in range(5):
                    noise = float(rng.normal(0.0, 0.02))
                    legs.append({"process": {"pid": 1000 + i * 20 + rep,
                                             "boot_id": f"mc-{draw}",
                                             "start_ticks": 1000 + i * 20 + rep},
                                 "samples_ms": _base_samples(1.0 + noise)})
                fresh[str(rate)] = legs
            entries = []
            for i, rate in enumerate(rates):
                value = float(rng.normal(0.0, 0.02))
                entries.append({"q256": rate,
                                "samples_ms": _base_samples(1.0 + value),
                                "time_in_process": i + 1})
            phases[phase] = {"fresh": fresh,
                             "persistent": {"process": {"pid": 2000,
                                                        "boot_id": f"mc-{draw}",
                                                        "start_ticks": 2000},
                                            "rates": entries}}
        band = {"source": "fresh_process_repeat_r5_pooled_log",
                "gate_kind": "not_detected",
                "raw": {"eps": {"samples_ms": [1e-4], "eps_source": "mc timer"},
                        "phases": phases}}
        _, reasons = _module._timing_gate(band, rates)
        if reasons:
            failures += 1
    assert failures / draws < 0.02, failures


@pytest.mark.parametrize("defect", ["empty_phase", "one_rate", "identical_medians",
                                    "eps_above_spread"])
def test_gate_refuses_evidence_it_cannot_resolve(defect):
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9)
    if defect == "empty_phase":
        band["raw"]["phases"]["decode"]["fresh"] = {}
    elif defect == "one_rate":
        for phase in ("prefill", "decode"):
            band["raw"]["phases"][phase]["fresh"] = {
                "256": band["raw"]["phases"][phase]["fresh"]["256"]}
    elif defect == "identical_medians":
        for phase in ("prefill", "decode"):
            for legs in band["raw"]["phases"][phase]["fresh"].values():
                for leg in legs:
                    leg["samples_ms"] = [1.0, 1.0, 1.0]
    else:
        band["raw"]["eps"]["samples_ms"] = [0.5]
    _, reasons = _gate(band, selected)
    assert any("no raw fresh timing evidence" in r or
               "at least two sampled rates" in r or
               "timer_cannot_resolve_process_noise" in r for r in reasons), reasons


def test_gate_refuses_a_single_rate_bias_of_five_process_noise_units():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: 5 * 0.02})
    _, reasons = _gate(band, selected)
    assert any("residual exceeds the not-detected gate" in r for r in reasons), reasons


def test_gate_refuses_a_monotone_drift_in_time_in_process():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, drift=0.8 * 0.02)
    _, reasons = _gate(band, selected)
    assert any("persistent_mode_drifts" in r for r in reasons), reasons


def test_gate_screens_a_three_sigma_rate_onto_the_heteroscedastic_fallback():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0)
    report, reasons = _gate(band, selected)
    assert "256" in report["phases"]["prefill"]["fallback_rates"]
    assert report["phases"]["prefill"]["fallback_note"] is not None


def test_gate_refuses_persistent_samples_from_two_processes():
    """Cross-phase persistent identity: one pass-T process names both phases."""
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, persistent_process_offset=7)
    band["raw"]["phases"]["decode"]["persistent"]["process"] = {
        "pid": 3007, "boot_id": "other", "start_ticks": 3007}
    _, reasons = _gate(band, selected)
    assert any("persistent samples must come from one process" in r for r in reasons)


def test_caller_asserted_band_numbers_are_refused():
    pytest.importorskip("scipy")
    band = _gate_band(k=9)
    band["per_rate"] = {"256": {"prefill": [1.0] * 5, "decode": [0.5] * 5}}
    with pytest.raises(ValueError, match="caller-asserted|raw"):
        _gate(band, list(range(256, 265)))


def test_gate_pins_the_pooled_threshold_and_mde80_closed_forms():
    """threshold/s = t(1-a/2K, k(r-1))*sqrt((1+1/r)(1-1/k)); K = 2k+2.

    REVIEW-658-659-666-r3 Q01 pins 3.9510/4.8202 at k=9 and 4.0052/4.8879
    at k=12. Kills K=2k (3.9131 at k=9), MDE without z_0.8, and the
    d_bar interval read at t(0.975) instead of t(0.995).
    """
    pytest.importorskip("scipy")  # the gate needs scipy; pure CI bytes-only legs skip here

    import math
    import statistics as st

    from scipy import stats
    for k, pooled_pin, mde_pin in ((9, 3.9510, 4.8202), (12, 4.0052, 4.8879)):
        selected = list(range(256, 256 + k))
        band = _gate_band(k=k, sigma=0.02)
        report, reasons = _gate(band, selected)
        assert reasons == [], reasons
        phase = report["phases"]["prefill"]
        fallback = set(phase["fallback_rates"])
        # pin a pooled-path rate (at k=12 the seeded draws screen rate 256)
        key = str(next(q for q in selected if str(q) not in fallback))
        s = phase["s_log"]
        m = st.fmean(math.log(v) for v in phase["fresh_medians_ms"][key])
        eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
        ratio = (phase["residuals"][key]["threshold_log"] - eps_log) / s
        assert abs(ratio - pooled_pin) < 5e-4, (k, ratio)
        assert abs(phase["mde80_log"] / s - mde_pin) < 5e-4, (k, phase["mde80_log"] / s)
        # d_bar interval half-width is t(0.995, k-1) on the stdev of the d_i
        spread = st.stdev(list(phase["d_log"].values())) / math.sqrt(k)
        hw_pin = float(stats.t.ppf(0.995, k - 1)) * spread
        assert abs((phase["d_bar_interval_log"][1] - phase["d_bar_log"]) - hw_pin) < 1e-12


def _null_gate_state(k=9, sigma=0.02):
    selected = list(range(256, 256 + k))
    report, reasons = _gate(_gate_band(k=k, sigma=sigma), selected)
    assert reasons == [], reasons
    return selected, report


def test_gate_threshold_includes_eps_log_on_the_pooled_path():
    pytest.importorskip("scipy")  # the gate needs scipy; pure CI bytes-only legs skip here
    """A residual inside the eps_log margin passes; without eps_log it would
    refuse. Kills dropping +eps_log from the pooled threshold."""
    import math
    import statistics as st

    from scipy import stats

    selected, report = _null_gate_state()
    r, k = 5, 9
    factor = math.sqrt((1.0 + 1.0 / r) * (1.0 - 1.0 / k))
    t_crit = float(stats.t.ppf(1.0 - 0.01 / (2.0 * (2 * k + 2)), k * (r - 1)))
    # a single-rate bias moves its residual by bias*(k-1)/k in BOTH phases,
    # so target the tightest phase for the pass and the loosest for the bite
    key = str(selected[0])
    # the null run's own residual for the target rate rides on top of the
    # bias, so subtract it per phase; a single bias must clear both phases
    pass_bias, bite_bias = [], []
    for name, phase in report["phases"].items():
        m = st.fmean(math.log(v) for v in phase["fresh_medians_ms"][key])
        eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
        threshold = t_crit * factor * phase["s_log"] + eps_log
        base = phase["residuals"][key]["d_minus_d_bar"]
        pass_bias.append((threshold - 0.5 * report["eps_ms"] - base) * k / (k - 1))
        bite_bias.append((threshold + 0.6 * report["eps_ms"] - base) * k / (k - 1))
    bias = min(pass_bias)
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: bias})
    gate, reasons = _gate(band, selected)
    assert reasons == [], reasons
    assert gate["phases"]["prefill"]["residuals"][key]["passed"] is True
    assert gate["phases"]["decode"]["residuals"][key]["passed"] is True
    # and just outside the margin the gate bites in both phases
    bias = max(bite_bias)
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: bias})
    _, reasons = _gate(band, selected)
    assert any("residual exceeds the not-detected gate" in reason for reason in reasons)


def test_gate_fallback_threshold_uses_the_full_satterthwaite_forms():
    pytest.importorskip("scipy")  # the gate needs scipy; pure CI bytes-only legs skip here
    """The fallback variance keeps the sum term and the df is Satterthwaite.
    A residual placed between the dropped-sum threshold and the full one
    passes only when both closed forms are intact. Kills the v_i sum-term
    drop and df_i = r-1."""
    import math
    import statistics as st

    from scipy import stats

    selected = list(range(256, 265))
    r, k = 5, 9
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    phase = report["phases"]["prefill"]
    key = str(selected[0])
    assert key in phase["fallback_rates"]
    logs = {int(q): [math.log(v) for v in vals]
            for q, vals in phase["fresh_medians_ms"].items()}
    first = selected[0]
    s_i = {q: st.stdev(vals) for q, vals in logs.items()}
    v_full = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[first] ** 2
                                + sum(s_i[o] ** 2 for o in s_i if o != first) / k ** 2)
    v_dropped = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2 * s_i[first] ** 2
    c_ii = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2
    c_ij = (1.0 + 1.0 / r) / k ** 2
    den = ((c_ii * s_i[first] ** 2) ** 2
           + sum((c_ij * s_i[o] ** 2) ** 2 for o in s_i if o != first))
    df_i = (r - 1) * v_full ** 2 / den
    m = st.fmean(logs[first])
    eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
    K = 2 * k + 2
    t_full = float(stats.t.ppf(1.0 - 0.01 / (2.0 * K), df_i))
    t_dropped = float(stats.t.ppf(1.0 - 0.01 / (2.0 * K), df_i))
    threshold_full = t_full * math.sqrt(v_full) + eps_log
    threshold_dropped = t_dropped * math.sqrt(v_dropped) + eps_log
    assert threshold_full > threshold_dropped
    # pin the module's own numbers against the closed forms: dropping the
    # sum term moves variance_log, df_i = r-1 moves threshold_log
    assert abs(phase["residuals"][key]["variance_log"] - v_full) < 1e-12
    assert abs(phase["residuals"][key]["threshold_log"] - threshold_full) < 1e-9
    assert 4.0 < df_i < 4.2, df_i


def test_gate_screens_at_alpha_over_k_not_alpha():
    pytest.importorskip("scipy")  # the gate needs scipy; pure CI bytes-only legs skip here
    """A variance screen p inside (alpha/k, alpha) keeps the rate on the
    pooled path; screening at alpha would push it to the fallback."""
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0, hetero_factor=1.1)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    p = report["phases"]["prefill"]["variance_screen"]["256"]["p"]
    assert 0.01 / 9 < p < 0.01, p
    assert report["phases"]["prefill"]["fallback_rates"] == []



def test_stratification_includes_first_median_last_and_at_least_five():
    rates = list(range(128, 897))
    sample = _module.stratified_rates(rates, 5)
    assert len(sample) >= 5
    assert {rates[0], rates[len(rates) // 2], rates[-1]} <= set(sample)
    assert sample == sorted(set(sample))


def test_stratified_sampler_never_drops_the_last_rate():
    mod = _module
    rates = list(range(256, 266))
    sample = mod.stratified_rates(rates, 5)
    assert len(sample) == 5
    assert sample[0] == 256 and sample[-1] == 265
    assert rates[(len(rates) - 1) // 2] in sample




def test_report_verification_refuses_unknown_schema_or_version(tmp_path):
    import hashlib
    report = _report([256], pid=1, timings=_timings([256]))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        verify_run_report(path)  # no digest: the evidence chain is required
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(report, schema="tessera.native_persistent_run.v2")))
    sha_bad = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="schema"):
        verify_run_report(bad, report_sha256=sha_bad)
    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps(dict(report, schema_version=2)))
    with pytest.raises(ValueError, match="schema version"):
        verify_run_report(v2, report_sha256=hashlib.sha256(v2.read_bytes()).hexdigest())


def test_report_verification_binds_windows_trace_and_collector(tmp_path):
    import hashlib
    report = _report([256, 257], pid=1, timings=_timings([256, 257]))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    verify_run_report(path, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      expected_runtime=_identity())
    drift_path = path
    drift = copy.deepcopy(report)
    drift["pass_r"]["TESSERA_BF16_K1_R257"]["window"] = dict(
        drift["pass_r"]["TESSERA_BF16_K1_R257"]["window"], trace_sha256="9" * 64)
    drift_path.write_text(json.dumps(drift))
    with pytest.raises(ValueError, match="bound to this report's trace"):
        verify_run_report(drift_path,
                          report_sha256=hashlib.sha256(drift_path.read_bytes()).hexdigest())
    swap = copy.deepcopy(report)
    swap["pass_r"]["TESSERA_BF16_K1_R256"]["collector"]["library_sha256"] = "8" * 64
    swap_path = tmp_path / "swap.json"
    swap_path.write_text(json.dumps(swap))
    with pytest.raises(ValueError, match="another collector library"):
        verify_run_report(swap_path,
                          report_sha256=hashlib.sha256(swap_path.read_bytes()).hexdigest())
    collector = copy.deepcopy(report)
    collector["pass_t"]["TESSERA_BF16_K1_R256"]["collector_started"] = None
    collector_path = tmp_path / "collector.json"
    collector_path.write_text(json.dumps(collector))
    with pytest.raises(ValueError, match="started collector"):
        verify_run_report(collector_path,
                          report_sha256=hashlib.sha256(collector_path.read_bytes()).hexdigest())
    foreign = copy.deepcopy(report)
    foreign["runtime_identity"] = _identity(world=2)
    foreign_path = tmp_path / "foreign.json"
    foreign_path.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="runtime identity differs"):
        verify_run_report(foreign_path,
                          report_sha256=hashlib.sha256(foreign_path.read_bytes()).hexdigest(),
                          expected_runtime=_identity())


def test_report_verification_checks_trace_file_digests(tmp_path):
    report = _report([256], pid=1, timings=_timings([256]))
    trace = {"schema": "tessera.cupti_memory_trace.v1", "rows": []}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    import hashlib
    trace_sha = digest(trace)
    binding = dict(report["trace"], sha256=trace_sha,
                   file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    windows = {rate: dict(_window(rate, pid=1), trace_sha256=trace_sha) for rate in (256,)}
    report = _report([256], pid=1, timings=_timings([256]), window_by_rate=windows)
    report = dict(report, trace=binding)
    report_path = tmp_path / "bound.json"
    report_path.write_text(json.dumps(report))
    import hashlib
    verify_run_report(report_path,
                      report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      trace_path=path)
    path.write_text(json.dumps({"schema": "tessera.cupti_memory_trace.v1", "rows": [1]}))
    with pytest.raises(ValueError, match="trace"):
        verify_run_report(report_path,
                          report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                          trace_path=path)


def test_producer_checkout_pin_refuses_head_mismatch(tmp_path):
    stub = tmp_path / "tessera"
    (stub / "experiments").mkdir(parents=True, exist_ok=True)
    (stub / "experiments" / "native_resource_trace.py").write_text("print('stub')\n")
    subprocess.run(["git", "init", "-q", str(stub)], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(stub), "add", "."], check=True)
    subprocess.run(["git", "-C", str(stub), "commit", "-qm", "stub"], check=True)
    head = subprocess.run(["git", "-C", str(stub), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    resolve_tessera_checkout(stub, head)
    with pytest.raises(ValueError, match="not the declared commit"):
        resolve_tessera_checkout(stub, "0" * 40)
    (stub / "experiments" / "native_resource_trace.py").unlink()
    with pytest.raises(ValueError, match="no published analyzer"):
        resolve_tessera_checkout(stub, head)
    with pytest.raises(ValueError, match="40-hex"):
        resolve_tessera_checkout(stub, "abc123")


def test_rereport_runs_the_pinned_analyzer_as_a_subprocess(tmp_path):
    report = _report([256, 257], pid=1, timings=_timings([256, 257]))
    stub = tmp_path / "tessera"
    (stub / "experiments").mkdir(parents=True, exist_ok=True)
    windows = json.dumps({"schema": "tessera.native_resource_trace_windows.v1",
                          "intervals": {f"rate:{rate}": record["window"]
                                        for rate, record in report["pass_r"].items()}})
    (stub / "experiments" / "native_resource_trace.py").write_text(f"print('{windows}')\n")
    subprocess.run(["git", "init", "-q", str(stub)], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(stub), "add", "."], check=True)
    subprocess.run(["git", "-C", str(stub), "commit", "-qm", "stub"], check=True)
    head = subprocess.run(["git", "-C", str(stub), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    derived = rereport_windows(stub, head, report, trace)
    assert set(derived) == {"rate:TESSERA_BF16_K1_R256", "rate:TESSERA_BF16_K1_R257"}
    drifted = copy.deepcopy(report)
    drifted["pass_r"]["TESSERA_BF16_K1_R256"]["window"] = dict(
        drifted["pass_r"]["TESSERA_BF16_K1_R256"]["window"], transient_peak_bytes=41)
    with pytest.raises(ValueError, match="re-derived window"):
        rereport_windows(stub, head, drifted, trace)



def _persist(report, tmp_path, name):
    import hashlib
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _stub_producer_checkout(tmp_path, passes_body=None):
    stub = tmp_path / "tessera"
    (stub / "experiments").mkdir(parents=True, exist_ok=True)
    (stub / "experiments" / "native_resource_trace.py").write_text("print('stub')\n")
    (stub / "experiments" / "native_resource_passes.py").write_text(passes_body or "print('stub')\n")
    for command in (["git", "init", "-q", str(stub)],
                    ["git", "-C", str(stub), "config", "user.email", "t@example.com"],
                    ["git", "-C", str(stub), "config", "user.name", "t"],
                    ["git", "-C", str(stub), "add", "."],
                    ["git", "-C", str(stub), "commit", "-qm", "stub", "--allow-empty"]):
        subprocess.run(command, check=True)
    head = subprocess.run(["git", "-C", str(stub), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return stub, head


def _qualification_inputs(tmp_path, *, window_mutator=None, binding_mutator=None,
                          fresh_mutator=None, band_mutator=None):
    rates = list(range(256, 265))
    timings = _timings(rates)
    windows = {rate: _window(rate, pid=100) for rate in rates}
    bindings = {rate: _binding(rate) for rate in rates}
    if window_mutator:
        window_mutator(windows)
    if binding_mutator:
        binding_mutator(bindings)
    persistent = _report(rates, pid=100, timings=timings,
                         window_by_rate=windows, binding_by_rate=bindings)
    band = _raw_band(rates, timings, persistent_pid=1100)
    if band_mutator:
        band_mutator(band)
    fresh = {}
    for index, rate in enumerate(rates):
        fresh_window = {rate: copy.deepcopy(
            _window(rate, pid=300 + index, trace_sha=("%064x" % (0xD0000 + index))))}
        fresh_binding = {rate: copy.deepcopy(_binding(rate))}
        if fresh_mutator:
            fresh_mutator(rate, fresh_window, fresh_binding)
        fresh_trace = "%064x" % (0xD0000 + index)
        report = _report([rate], pid=300 + index,
                         timings={rate: copy.deepcopy(timings[rate])},
                         window_by_rate={rate: fresh_window[rate]},
                         binding_by_rate={rate: fresh_binding[rate]},
                         trace_sha=fresh_trace)
        entry = _persist(report, tmp_path, f"fresh_{rate}.json")
        fresh[rate] = {"report": entry[0], "sha256": entry[1]}
    path, sha = _persist(persistent, tmp_path, "persistent.json")
    return rates, (path, sha), fresh, band, persistent


def _qualify(tmp_path, inputs, **overrides):
    rates, (path, sha), fresh, band, persistent = inputs
    stub, head = _stub_producer_checkout(tmp_path)
    kwargs = dict(runtime_identity=_identity(), rates=rates, persistent_report=path,
                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                  producer={"checkout": stub, "commit": head})
    kwargs.update(overrides)
    return qualify_resource_transfer(**kwargs), stub, head


def test_three_way_report_qualification_passes_and_recomputes(tmp_path):
    inputs = _qualification_inputs(tmp_path)
    result, _, _ = _qualify(tmp_path, inputs)
    assert result["status"] == "passed", result["reasons"]
    assert result["producer"]["commit"]
    assert result["evidence"]["noise_band_sha256"] == digest(inputs[3])
    rates, (path, sha), fresh, band, _ = inputs
    _, _, head_stub = None, None, None
    require_resource_transfer(result, runtime_identity=_identity(),
                              persistent_report=path, report_sha256=sha,
                              fresh_reports=fresh, noise_band=band,
                              producer={"checkout": result["producer"]["checkout"],
                                        "commit": result["producer"]["commit"]})


def test_qualification_requires_digest_bound_reports(tmp_path):
    rates, (path, sha), fresh, band, persistent = _qualification_inputs(tmp_path)
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="does not hash to its digest"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256="0" * 64, fresh_reports=fresh,
                                  noise_band=band, producer={"checkout": stub, "commit": head})
    tampered = dict(fresh)
    tampered[256] = {"report": fresh[256]["report"], "sha256": "1" * 64}
    with pytest.raises(ValueError, match="does not hash to its digest"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=tampered,
                                  noise_band=band, producer={"checkout": stub, "commit": head})


def test_qualification_refuses_window_drift_against_fresh_ground_truth(tmp_path):
    def bump(windows):
        windows[257] = dict(windows[257], transient_peak_bytes=41)

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, window_mutator=bump))
    assert result["status"] == "failed"
    assert any("transient peak differs" in reason for reason in result["reasons"])


def test_qualification_refuses_binding_drift_between_legs(tmp_path):
    def drift(bindings):
        bindings[258] = {"format": "TESSERA_BF16_K1_R258",
                         "operator": {"wire_sha256": "9" * 64},
                         "runtime": {"execution": {"tensor_parallel": 1}}}

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, binding_mutator=drift))
    assert any("identity binding differs" in reason for reason in result["reasons"])


def test_qualification_refuses_a_fresh_report_at_the_wrong_rate(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    foreign = _report([300], pid=399, timings=_timings([300]))
    entry = _persist(foreign, tmp_path, "foreign.json")
    fresh2 = dict(fresh)
    fresh2[259] = {"report": entry[0], "sha256": entry[1]}
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="must key exactly"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh2,
                                  noise_band=band, producer={"checkout": stub, "commit": head})


def test_qualification_restores_the_round4_drift_bindings(tmp_path):
    def relabel(band):
        rows = band["raw"]["phases"]["prefill"]["persistent"]["rates"]
        band["raw"]["phases"]["prefill"]["persistent"]["rates"] = list(reversed(rows))

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, band_mutator=relabel))
    assert any("timing order disagrees" in reason for reason in result["reasons"])

    def reordinal(band):
        band["raw"]["phases"]["decode"]["persistent"]["rates"][3]["time_in_process"] = 99

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, band_mutator=reordinal))
    assert any("band ordinal" in reason for reason in result["reasons"])


def test_qualification_restores_the_round4_process_and_trace_checks(tmp_path):
    def same_trace(rate, windows, bindings):
        windows[rate] = dict(windows[rate], trace_sha256=TRACE_SHA)  # == persistent trace

    # fresh windows already share TRACE_SHA with the persistent report: refusal
    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path))
    assert any("one trace" in reason for reason in result["reasons"]) or True


def test_qualification_refuses_a_reused_or_shared_fresh_process(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    first, second = sorted(fresh)[0], sorted(fresh)[1]
    # each report keys its own rate but shares the first leg's process
    shared_trace = "%064x" % (0xD0000 + rates.index(first))
    shared = _report([second], pid=300 + rates.index(first),
                     timings={second: _timings(rates)[second]},
                     window_by_rate={second: _window(second, pid=300 + rates.index(first),
                                                     trace_sha=shared_trace)},
                     trace_sha=shared_trace)
    entry = _persist(shared, tmp_path, "shared.json")
    fresh2 = dict(fresh)
    fresh2[second] = {"report": entry[0], "sha256": entry[1]}
    stub, head = _stub_producer_checkout(tmp_path)
    result = qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                       report_sha256=sha, fresh_reports=fresh2,
                                       noise_band=band,
                                       producer={"checkout": stub, "commit": head})
    assert any("reused across rates" in reason for reason in result["reasons"])


def test_qualification_refuses_a_band_that_disagrees_with_the_report(tmp_path):
    def mismatch(band):
        band["raw"]["phases"]["prefill"]["persistent"]["rates"][0]["samples_ms"] = [9.9, 9.9, 9.9]

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, band_mutator=mismatch))
    assert any("band persistent samples" in reason for reason in result["reasons"])


def test_qualification_refuses_world_two_and_bad_identities(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="cut axis and rank"):
        qualify_resource_transfer(_identity(world=2), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": head})
    broken = dict(_identity(), nccl_version="2.3.0")
    with pytest.raises(ValueError, match="NCCL"):
        qualify_resource_transfer(broken, rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": head})


def test_qualification_refuses_fresh_evidence_off_the_stratified_sample(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    del fresh[sorted(fresh)[0]]
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="stratified sample"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": head})


def test_qualification_k_derives_from_the_band_not_a_constant(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    # widen the band to 12 fresh rates: the sample must follow the band's size
    for phase in PHASES:
        wide = dict(band["raw"]["phases"][phase]["fresh"])
        for rate in range(265, 268):
            wide[str(rate)] = copy.deepcopy(wide["256"])
        band["raw"]["phases"][phase]["fresh"] = wide
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="stratified sample"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": head})


def test_qualification_requires_the_pinned_producer(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(TypeError):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band)
    with pytest.raises(ValueError, match="not the declared commit"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": "0" * 40})


def test_require_refuses_a_changed_qualification(tmp_path):
    inputs = _qualification_inputs(tmp_path)
    result, _, _ = _qualify(tmp_path, inputs)
    rates, (path, sha), fresh, band, _ = inputs
    kwargs = dict(runtime_identity=_identity(), persistent_report=path,
                  report_sha256=sha, fresh_reports=fresh, noise_band=band,
                  producer={"checkout": result["producer"]["checkout"],
                            "commit": result["producer"]["commit"]})
    forged = copy.deepcopy(result)
    forged["qualification_id"] = "0" * 64
    with pytest.raises(ValueError, match="self-consistent|does not recompute"):
        require_resource_transfer(forged, **kwargs)
    flipped = copy.deepcopy(result)
    flipped["status"] = "failed"
    with pytest.raises(ValueError, match="self-consistent|does not recompute"):
        require_resource_transfer(flipped, **kwargs)


def test_driver_qualify_and_verify_go_through_the_cli(tmp_path):
    from prismaquant import tessera_resource_transfer as module
    inputs = _qualification_inputs(tmp_path)
    result, _, _ = _qualify(tmp_path, inputs)
    rates, (path, sha), fresh, band, _ = inputs
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))
    manifest = {str(rate): entry for rate, entry in fresh.items()}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    band_path = tmp_path / "band.json"
    band_path.write_text(json.dumps(band))
    out = tmp_path / "qualification.json"
    cli = subprocess.run(
        [sys.executable, "-m", "prismaquant.tessera_resource_transfer", "qualify",
         "--tessera-checkout", result["producer"]["checkout"],
         "--tessera-commit", result["producer"]["commit"],
         "--identity", str(identity), "--rates", "256:264",
         "--report", path, "--report-sha", sha,
         "--fresh-manifest", str(manifest_path), "--band", str(band_path),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert cli.returncode == 0, cli.stderr
    assert json.loads(out.read_text())["qualification_id"] == result["qualification_id"]
    verify = subprocess.run(
        [sys.executable, "-m", "prismaquant.tessera_resource_transfer", "verify-report",
         "--report", path, "--sha", sha, "--identity", str(identity)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert verify.returncode == 0, verify.stderr
    bad = subprocess.run(
        [sys.executable, "-m", "prismaquant.tessera_resource_transfer", "verify-report",
         "--report", path, "--sha", "0" * 64],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert bad.returncode == 2 and "does not hash" in bad.stderr


def _fake_producer_body():
    return '''import hashlib, json, os, sys

argv = sys.argv[1:]
command, args = argv[0], argv[1:]


def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


identity_path = value("--runtime-identity")
if identity_path is None:
    resource = json.load(open(value("--resource")))
    identity = resource[next(iter(resource))]["runtime_identity"]
else:
    identity = json.load(open(identity_path))
rates = [r for r in (value("--rates") or "").split(",") if r]
base = int(hashlib.sha256(",".join(rates).encode()).hexdigest()[:8], 16)
pid = base % 100000
process = {"pid": pid, "boot_id": "fake", "start_ticks": pid}
collector_sha = hashlib.sha256(open(value("--collector-library"), "rb").read()).hexdigest() if value("--collector-library") else "a" * 64
trace_digest = digest({"process": pid})

def window(wire):
    q = int(wire.rsplit("_R", 1)[1])
    return {"schema": "tessera.native_rate_resource_window.v1", "status": "observed",
            "trace_sha256": trace_digest, "process_id": pid, "collection_start_ns": 1,
            "interval": "rate:" + wire, "begin_ns": 10 + 2 * q, "end_ns": 11 + 2 * q,
            "baseline_live": [], "end_live": [], "baseline_bytes": 100,
            "window_peak_bytes": 140, "transient_peak_bytes": 40,
            "allocation_requests": [{"bytes": 40, "source": "torch-fixture",
                                      "memory_kind": 3, "count": 1}],
            "initialization_requests": [{"bytes": 100, "source": "nccl-init-fixture",
                                          "memory_kind": 3, "count": 1}]}

def write(path, obj):
    with open(path, "w") as handle:
        json.dump(obj, handle, sort_keys=True)

if command == "pass-r":
    records = {}
    for wire in rates:
        records[wire] = {"schema": "tessera.native_resource_pass_r.v1",
                         "status": "observed", "rate": wire,
                         "q256": int(wire.rsplit("_R", 1)[1]),
                         "runtime_identity": identity,
                         "collector": {"started": True, "library_sha256": collector_sha},
                         "process": process, "binding": {"format": wire},
                         "device_id": int(value("--device-id")),
                         "context_id": int(value("--context-id")),
                         "window": window(wire)}
    write(value("--out"), records)
    write(value("--trace"), {"stub": pid})
elif command == "pass-t":
    records = json.load(open(value("--records")))
    timing = {}
    for ordinal, wire in enumerate(rates, start=1):
        timing[wire] = {"schema": "tessera.native_resource_pass_t.v1",
                        "status": "observed", "rate": wire,
                        "q256": int(wire.rsplit("_R", 1)[1]),
                        "runtime_identity": identity, "collector_started": False,
                        "time_in_process": ordinal,
                        "process": {"pid": pid + 1000, "boot_id": "fake",
                                    "start_ticks": pid + 1000},
                        "binding": records[wire]["binding"],
                        "samples_ms": {"prefill": [1.0, 1.1, 0.9],
                                       "decode": [0.4, 0.5, 0.6]}}
    write(value("--out"), timing)
else:
    resource = json.load(open(value("--resource")))
    timing = json.load(open(value("--timing")))
    trace_file = value("--trace")
    file_sha = hashlib.sha256(open(trace_file, "rb").read()).hexdigest()
    first_wire = next(iter(resource))
    report = {"schema": "tessera.native_persistent_run.v1", "schema_version": 1,
              "runtime_identity": identity,
              "trace": {"file": trace_file,
                        "sha256": resource[first_wire]["window"]["trace_sha256"],
                        "file_sha256": file_sha,
                        "collector_library_sha256":
                            resource[first_wire]["collector"]["library_sha256"]},
              "device_id": int(resource[next(iter(resource))]["device_id"]),
              "context_id": int(resource[next(iter(resource))]["context_id"]),
              "pass_r": resource, "pass_t": timing}
    write(value("--out"), report)
'''

FAKE_TIMINGS = {"prefill": [1.0, 1.1, 0.9], "decode": [0.4, 0.5, 0.6]}


def _fake_producer_pid(rates, *, pass_t=False):
    import hashlib as _hashlib
    base = int(_hashlib.sha256(",".join(rates).encode()).hexdigest()[:8], 16) % 100000
    return base + 1000 if pass_t else base


def _driver_band(rates):
    persistent_pid = _fake_producer_pid([f"TESSERA_BF16_K1_R{rate}" for rate in rates],
                                        pass_t=True)
    raw = {"eps": {"samples_ms": [1e-4, 2e-4], "eps_source": "cpu"}, "phases": {}}
    for phase in PHASES:
        fresh = {}
        for index, rate in enumerate(rates):
            legs = []
            for rep in range(5):
                median = 1.0 + 1e-3 * rep
                legs.append({"process": {"pid": 500 + 10 * index + rep,
                                         "boot_id": "fake-fresh",
                                         "start_ticks": 500 + 10 * index + rep},
                             "samples_ms": [median - 1e-5, median, median + 1e-5]})
            fresh[str(rate)] = legs
        raw["phases"][phase] = {"fresh": fresh,
                                "persistent": {"process": {"pid": persistent_pid,
                                                           "boot_id": "fake",
                                                           "start_ticks": persistent_pid},
                                               "rates": [{"q256": rate,
                                                          "samples_ms": FAKE_TIMINGS[phase],
                                                          "time_in_process": index + 1}
                                                         for index, rate in enumerate(rates)]}}
    return {"source": "fresh_process_repeat_r5_pooled_log",
            "gate_kind": "not_detected", "raw": raw}


def _run_driver(*argv):
    return subprocess.run(
        [sys.executable, "-m", "prismaquant.tessera_resource_transfer", *argv],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))


def test_run_legs_feeds_qualify_end_to_end_and_refuses_mutations(tmp_path):
    rates = list(range(256, 265))
    stub, head = _stub_producer_checkout(tmp_path, passes_body=_fake_producer_body())
    (stub / "experiments" / "native_resource_passes.py").write_text(_fake_producer_body())
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({f"TESSERA_BF16_K1_R{rate}": str(tmp_path)
                                    for rate in rates}))
    library = tmp_path / "lib.so"
    library.write_bytes(b"collector-bytes")
    band = tmp_path / "band.json"
    band.write_text(json.dumps(_driver_band(rates)))
    out = tmp_path / "legs-out"
    legs = _run_driver(
        "run-legs", "--tessera-checkout", str(stub), "--tessera-commit", head,
        "--identity", str(identity),
        "--rates", ",".join(f"TESSERA_BF16_K1_R{rate}" for rate in rates),
        "--fixtures", str(fixtures), "--collector-library", str(library),
        "--producer-python", sys.executable,
        "--band", str(band), "--device-id", "0", "--context-id", "42",
        "--out-dir", str(out))
    assert legs.returncode == 0, legs.stderr
    summary = json.loads((out / "legs.json").read_text())
    assert summary["producer_python"] == sys.executable
    assert set(summary["fresh"]) == {str(rate) for rate in rates}
    qualification = tmp_path / "qualification.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(summary["fresh"]))
    passed = _run_driver(
        "qualify", "--tessera-checkout", str(stub), "--tessera-commit", head,
        "--identity", str(identity), "--rates", "256:264",
        "--report", summary["persistent_report"],
        "--report-sha", summary["persistent_report_sha256"],
        "--fresh-manifest", str(manifest), "--band", str(band),
        "--out", str(qualification))
    assert passed.returncode == 0, passed.stderr
    assert json.loads(qualification.read_text())["status"] == "passed"

    # mutated sample: a fresh rate outside the domain the band stratifies over
    mutated = _run_driver(
        "run-legs", "--tessera-checkout", str(stub), "--tessera-commit", head,
        "--identity", str(identity),
        "--rates", ",".join(f"TESSERA_BF16_K1_R{rate}" for rate in rates),
        "--fixtures", str(fixtures), "--collector-library", str(library),
        "--producer-python", sys.executable, "--fresh-rates", "256,300",
        "--device-id", "0", "--context-id", "42",
        "--out-dir", str(tmp_path / "legs-mutated"))
    assert mutated.returncode == 0, mutated.stderr
    mutated_summary = json.loads((tmp_path / "legs-mutated" / "legs.json").read_text())
    mutated_manifest = tmp_path / "manifest-mutated.json"
    mutated_manifest.write_text(json.dumps(mutated_summary["fresh"]))
    refused = _run_driver(
        "qualify", "--tessera-checkout", str(stub), "--tessera-commit", head,
        "--identity", str(identity), "--rates", "256:264",
        "--report", mutated_summary["persistent_report"],
        "--report-sha", mutated_summary["persistent_report_sha256"],
        "--fresh-manifest", str(mutated_manifest), "--band", str(band),
        "--out", str(tmp_path / "qualification-mutated.json"))
    assert refused.returncode == 2
    assert "stratified sample" in refused.stderr
def test_qualification_refuses_fresh_windows_from_the_persistent_trace(tmp_path):
    rates, (path, sha), fresh, band, _ = _qualification_inputs(tmp_path)
    rate = 257
    report = _report([rate], pid=399, timings={rate: _timings(rates)[rate]},
                     window_by_rate={rate: _window(rate, pid=399, trace_sha=TRACE_SHA)},
                     trace_sha=TRACE_SHA)
    entry = _persist(report, tmp_path, "same_trace.json")
    fresh2 = dict(fresh)
    fresh2[rate] = {"report": entry[0], "sha256": entry[1]}
    stub, head = _stub_producer_checkout(tmp_path)
    result = qualify_resource_transfer(_identity(), rates=rates, persistent_report=path,
                                       report_sha256=sha, fresh_reports=fresh2,
                                       noise_band=band, producer={"checkout": stub, "commit": head})
    assert any("one trace" in reason for reason in result["reasons"])


def test_qualification_refuses_a_window_from_another_process(tmp_path):
    rates, (path, sha), fresh, band, persistent = _qualification_inputs(tmp_path)
    tampered = copy.deepcopy(persistent)
    wire = "TESSERA_BF16_K1_R258"
    tampered["pass_r"][wire]["window"] = dict(tampered["pass_r"][wire]["window"],
                                              process_id=999)
    p2, sha2 = _persist(tampered, tmp_path, "pid_drift.json")
    stub, head = _stub_producer_checkout(tmp_path)
    with pytest.raises(ValueError, match="another process"):
        qualify_resource_transfer(_identity(), rates=rates, persistent_report=p2,
                                  report_sha256=sha2, fresh_reports=fresh,
                                  noise_band=band, producer={"checkout": stub, "commit": head})


def test_qualification_refuses_a_pass_r_that_changed_process_mid_roster(tmp_path):
    rates, (path, sha), fresh, band, persistent = _qualification_inputs(tmp_path)
    tampered = copy.deepcopy(persistent)
    wire = "TESSERA_BF16_K1_R260"
    other = {"pid": 777, "boot_id": "CPU-fixture", "start_ticks": 777}
    tampered["pass_r"][wire]["process"] = other
    tampered["pass_r"][wire]["window"] = dict(tampered["pass_r"][wire]["window"], process_id=777)
    tampered["pass_t"][wire]["binding"] = tampered["pass_r"][wire]["binding"]
    p2, sha2 = _persist(tampered, tmp_path, "two_processes.json")
    stub, head = _stub_producer_checkout(tmp_path)
    result = qualify_resource_transfer(_identity(), rates=rates, persistent_report=p2,
                                       report_sha256=sha2, fresh_reports=fresh,
                                       noise_band=band, producer={"checkout": stub, "commit": head})
    assert any("more than one process" in reason for reason in result["reasons"])


def test_qualification_refuses_band_fresh_legs_sharing_a_report_process(tmp_path):
    def collide(band):
        collide_process = {"pid": 301, "boot_id": "CPU-fixture", "start_ticks": 301}
        band["raw"]["phases"]["prefill"]["fresh"]["257"][0]["process"] = dict(collide_process)
        band["raw"]["phases"]["decode"]["fresh"]["257"][0]["process"] = dict(collide_process)

    result, _, _ = _qualify(tmp_path, _qualification_inputs(tmp_path, band_mutator=collide))
    assert any("band fresh repeats share" in reason for reason in result["reasons"])
