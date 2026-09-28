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


def _window(rate, *, trace_sha=TRACE_SHA, allocs=None, transient=40, init=None):
    return {"schema": "tessera.native_rate_resource_window.v1", "status": "observed",
            "trace_sha256": trace_sha, "process_id": 0, "collection_start_ns": 1,
            "interval": f"rate:TESSERA_BF16_K1_R{rate}", "begin_ns": 10 + 2 * rate,
            "end_ns": 11 + 2 * rate, "baseline_live": [], "end_live": [],
            "baseline_bytes": 100, "window_peak_bytes": 100 + transient,
            "transient_peak_bytes": transient,
            "allocation_requests": allocs or [{"bytes": 40, "source": "torch-fixture",
                                               "memory_kind": 3, "count": 1}],
            "initialization_requests": init or [{"bytes": 100, "source": "nccl-init-fixture",
                                                 "memory_kind": 3, "count": 1}]}


def _report(rates, *, pid, timings, identity=None, binding_by_rate=None,
            window_by_rate=None, order=None):
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
                        "window": window_by_rate.get(rate, _window(rate))}
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
            "trace": {"file": "trace.json", "sha256": TRACE_SHA,
                      "file_sha256": FILE_SHA,
                      "collector_library_sha256": COLLECTOR},
            "device_id": 0, "context_id": 42,
            "pass_r": pass_r, "pass_t": pass_t}


def _timings(rates, base=1.03):
    return {rate: {phase: [base - 1e-5, base, base + 1e-5] for phase in PHASES}
            for rate in rates}


def _raw_band(rates, timings, *, persistent_pid=1100, fresh_seed=100):
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




def test_report_verification_refuses_unknown_schema_or_version():
    report = _report([256], pid=1, timings=_timings([256]))
    with pytest.raises(ValueError, match="schema"):
        verify_run_report(dict(report, schema="tessera.native_persistent_run.v2"))
    with pytest.raises(ValueError, match="schema version"):
        verify_run_report(dict(report, schema_version=2))
    with pytest.raises(ValueError, match="object"):
        verify_run_report([report])


def test_report_verification_binds_windows_trace_and_collector():
    report = _report([256, 257], pid=1, timings=_timings([256, 257]))
    verify_run_report(report, expected_runtime=_identity())
    drift = copy.deepcopy(report)
    drift["pass_r"]["TESSERA_BF16_K1_R257"]["window"] = dict(
        drift["pass_r"]["TESSERA_BF16_K1_R257"]["window"], trace_sha256="9" * 64)
    with pytest.raises(ValueError, match="bound to this report's trace"):
        verify_run_report(drift)
    swap = copy.deepcopy(report)
    swap["pass_r"]["TESSERA_BF16_K1_R256"]["collector"]["library_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="another collector library"):
        verify_run_report(swap)
    collector = copy.deepcopy(report)
    collector["pass_t"]["TESSERA_BF16_K1_R256"]["collector_started"] = True
    with pytest.raises(ValueError, match="started collector"):
        verify_run_report(collector)
    foreign = copy.deepcopy(report)
    foreign["runtime_identity"] = _identity(world=2)
    with pytest.raises(ValueError, match="runtime identity differs"):
        verify_run_report(foreign, expected_runtime=_identity())


def test_report_verification_checks_trace_file_digests(tmp_path):
    report = _report([256], pid=1, timings=_timings([256]))
    trace = {"schema": "tessera.cupti_memory_trace.v1", "rows": []}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    import hashlib
    trace_sha = digest(trace)
    binding = dict(report["trace"], sha256=trace_sha,
                   file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    windows = {rate: dict(_window(rate), trace_sha256=trace_sha) for rate in (256,)}
    report = _report([256], pid=1, timings=_timings([256]), window_by_rate=windows)
    report = dict(report, trace=binding)
    verify_run_report(report, trace_path=path)
    path.write_text(json.dumps({"schema": "tessera.cupti_memory_trace.v1", "rows": [1]}))
    with pytest.raises(ValueError, match="trace"):
        verify_run_report(report, trace_path=path)


def test_producer_checkout_pin_refuses_head_mismatch(tmp_path):
    stub = tmp_path / "tessera"
    (stub / "experiments").mkdir(parents=True)
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
    (stub / "experiments").mkdir(parents=True)
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


def _qualification_inputs(*, window_mutator=None, binding_mutator=None,
                          band_persistent_pid=1100):
    rates = list(range(256, 265))
    timings = _timings(rates)
    windows = {rate: _window(rate) for rate in rates}
    bindings = {rate: _binding(rate) for rate in rates}
    if window_mutator:
        window_mutator(windows)
    if binding_mutator:
        binding_mutator(bindings)
    persistent = _report(rates, pid=100, timings=timings,
                         window_by_rate=windows, binding_by_rate=bindings)
    band = _raw_band(rates, timings, persistent_pid=band_persistent_pid)
    fresh = {}
    for index, rate in enumerate(stratified_rates(rates, 9)):
        fresh[rate] = {"report": _report([rate], pid=300 + index,
                                         timings={rate: timings[rate]}),
                       "sha256": None}
    return rates, persistent, fresh, band


def test_three_way_report_qualification_passes_and_recomputes():
    rates, persistent, fresh, band = _qualification_inputs()
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band)
    assert result["status"] == "passed", result["reasons"]
    assert result["fallback_fresh_process"] is False
    assert result["schema"] == "prismaquant.native_resource_transfer_qualification.v1"
    require_resource_transfer(result, runtime_identity=_identity(),
                              persistent_report=persistent, fresh_reports=fresh,
                              noise_band=band)


def test_qualification_refuses_window_drift_against_fresh_ground_truth():
    def bump(windows):
        windows[257] = dict(windows[257], transient_peak_bytes=41)

    rates, persistent, fresh, band = _qualification_inputs(window_mutator=bump)
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band)
    assert result["status"] == "failed"
    assert any("transient peak differs" in reason for reason in result["reasons"])
    assert result["fallback_fresh_process"] is True


def test_qualification_refuses_binding_drift_between_legs():
    def drift(bindings):
        bindings[258] = {"format": "TESSERA_BF16_K1_R258",
                         "operator": {"wire_sha256": "9" * 64},
                         "runtime": {"execution": {"tensor_parallel": 1}}}

    rates, persistent, fresh, band = _qualification_inputs(binding_mutator=drift)
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band)
    assert any("identity binding differs" in reason for reason in result["reasons"])


def test_qualification_refuses_a_reused_or_shared_fresh_process():
    rates, persistent, fresh, band = _qualification_inputs()
    reused = {rate: copy.deepcopy(entry) for rate, entry in fresh.items()}
    first_rate = sorted(fresh)[0]
    reused[sorted(fresh)[1]]["report"] = copy.deepcopy(fresh[first_rate]["report"])
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=reused, noise_band=band)
    assert any("reused across rates" in reason for reason in result["reasons"])


def test_qualification_refuses_a_band_that_disagrees_with_the_report():
    rates, persistent, fresh, band = _qualification_inputs()
    band["raw"]["phases"]["prefill"]["persistent"]["rates"][0]["samples_ms"] = [9.9, 9.9, 9.9]
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band)
    assert any("band persistent samples" in reason for reason in result["reasons"])


def test_qualification_refuses_world_two_until_cases_name_axis_and_rank():
    rates, persistent, fresh, band = _qualification_inputs()
    with pytest.raises(ValueError, match="cut axis and rank"):
        qualify_resource_transfer(_identity(world=2), rates=rates,
                                  persistent_report=persistent,
                                  fresh_reports=fresh, noise_band=band)


def test_qualification_refuses_fresh_evidence_off_the_stratified_sample():
    rates, persistent, fresh, band = _qualification_inputs()
    del fresh[sorted(fresh)[0]]
    with pytest.raises(ValueError, match="stratified sample"):
        qualify_resource_transfer(_identity(), rates=rates,
                                  persistent_report=persistent,
                                  fresh_reports=fresh, noise_band=band)


def test_qualification_stamps_and_resolves_the_pinned_producer(tmp_path):
    stub = tmp_path / "tessera"
    (stub / "experiments").mkdir(parents=True)
    (stub / "experiments" / "native_resource_trace.py").write_text("print('stub')\n")
    subprocess.run(["git", "init", "-q", str(stub)], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(stub), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(stub), "add", "."], check=True)
    subprocess.run(["git", "-C", str(stub), "commit", "-qm", "stub"], check=True)
    head = subprocess.run(["git", "-C", str(stub), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    rates, persistent, fresh, band = _qualification_inputs()
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band,
                                       producer={"checkout": stub, "commit": head})
    assert result["producer"] == {"checkout": str(stub), "commit": head}
    with pytest.raises(ValueError, match="not the declared commit"):
        qualify_resource_transfer(_identity(), rates=rates,
                                  persistent_report=persistent,
                                  fresh_reports=fresh, noise_band=band,
                                  producer={"checkout": stub, "commit": "0" * 40})


def test_require_refuses_a_changed_qualification():
    rates, persistent, fresh, band = _qualification_inputs()
    result = qualify_resource_transfer(_identity(), rates=rates,
                                       persistent_report=persistent,
                                       fresh_reports=fresh, noise_band=band)
    forged = copy.deepcopy(result)
    forged["qualification_id"] = "0" * 64
    with pytest.raises(ValueError, match="self-consistent|does not recompute"):
        require_resource_transfer(forged, runtime_identity=_identity(),
                                  persistent_report=persistent,
                                  fresh_reports=fresh, noise_band=band)
    flipped = copy.deepcopy(result)
    flipped["status"] = "failed"
    flipped["reasons"] = ["hand-edited"]
    with pytest.raises(ValueError, match="self-consistent|does not recompute"):
        require_resource_transfer(flipped, runtime_identity=_identity(),
                                  persistent_report=persistent,
                                  fresh_reports=fresh, noise_band=band)
