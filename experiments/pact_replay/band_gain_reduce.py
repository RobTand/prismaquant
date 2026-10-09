"""In-domain band gains, paired sequence bootstrap and approved alpha stop rules.

Input is measured pact.band_observations.v1, never G3 final-window predictions.
Measured kinds include routed W_T16 and attention joint_T4_A4. When amplitudes
1, 2 and 4 all arrive, the approved (1,2) pair fit is kept, or (2,4) when the
null rule fails on (1,2); the third amplitude stays a diagnostic with a ready
(2,4) fallback. No clipping, price rescaling, energy extrapolation or invented
missing coefficient.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

COHORT = {"sample_range": [384, 448], "raw_tokens_per_sequence": 512,
          "prefix_ids": [154822, 154824], "input_contract": "prefixed_514",
          "local_prefix_rows": "excluded", "global_original_tokens": 32768,
          "scored_positions_per_sequence": 511}
BANDS = [(0, 3)] + [(start, start + 6) for start in range(3, 45, 6)]
KINDS = {"routed": {"W_T4", "W_T8", "W_T16", "A4", "A8"},
         "nonrouted": {"W_T8", "joint_T4_A4", "A8"},
         "attention": {"W_T8", "W_T16", "joint_T4_A4", "A8"}, "lm_head": {"W_T8", "W_T16", "A8"}}

def valid_stream(band,cls,kind):
    return cls in KINDS and kind in KINDS[cls] and (band == (45,46) if cls == "lm_head"
        else band in BANDS and not (band == (0,3) and cls == "routed"))


def actual_token_sha256(document):
    """Hash the actual original tokens after the exact prefix comparison."""
    import numpy as np
    tokens = np.asarray(document["input_token_ids"])
    if tokens.shape != (64, 514) or tokens.dtype.kind not in "iu":
        raise ValueError("The gain input must contain64 actual integer token sequences of length514")
    if not np.array_equal(tokens[:, :2], np.broadcast_to(COHORT["prefix_ids"], (64, 2))):
        raise ValueError("The actual gain prefix tokens differ")
    if (tokens < 0).any() or (tokens > np.iinfo(np.int64).max).any():
        raise ValueError("The actual gain tokens do not fit nonnegative int64")
    original = np.ascontiguousarray(tokens[:, 2:], dtype="<i8")
    return hashlib.sha256(original.tobytes()).hexdigest()


def anchor_response(candidate_energy, anchor_energy, exponent):
    """Approved A normalization; unit anchor shares are not exponentiated."""
    values = [float(v) for v in (candidate_energy, anchor_energy, exponent)]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite energy or exponent")
    candidate, anchor, alpha = values
    if min(candidate, anchor) < 0 or not 0.5 <= alpha <= 2:
        raise ValueError("Negative energy or exponent outside approved bounds")
    if 0.8 <= alpha <= 1.25:
        return candidate
    if anchor == 0:
        if candidate == 0:
            return 0.0
        raise ValueError("Positive candidate energy has no measured nonzero anchor")
    return anchor * (candidate / anchor) ** alpha


def aligned(records, fields):
    import numpy as np
    if len(records) != 64 or {row["sample_id"] for row in records} != set(range(384, 448)):
        raise ValueError("Exactly one observation for every prefixed held-out sample 384:448 is required")
    rows = sorted(records, key=lambda row: row["sample_id"])
    arrays = {field: np.asarray([row[field] for row in rows], dtype=np.float64) for field in fields}
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Nonfinite measured observations")
    if "E_sum" in arrays and (arrays["E_sum"] < 0).any():
        raise ValueError("Negative output-error energy")
    return arrays


def finite_summary(values):
    import numpy as np
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return {"se": None, "interval_95": None, "valid": int(len(finite)), "invalid": int(len(values) - len(finite))}
    return {"se": float(finite.std(ddof=1)),
            "interval_95": [float(v) for v in np.quantile(finite, [0.025, 0.975])],
            "valid": int(len(finite)), "invalid": int(len(values) - len(finite))}


def _fit_pair(first_records, second_records, null_records, bootstrap_indices,
              amplitude_first, amplitude_second):
    """Fit one measured amplitude pair with the shared sequence resamples."""
    import numpy as np
    first = aligned(first_records, ("E_sum", "KL_mean"))
    second = aligned(second_records, ("E_sum", "KL_mean"))
    null = aligned(null_records, ("KL_mean",))["KL_mean"]
    e1, e2 = first["E_sum"].sum(), second["E_sum"].sum()
    k1, k2 = first["KL_mean"].mean(), second["KL_mean"].mean()
    null_mean = float(null.mean())
    if min(e1, e2, k1, k2) <= 0 or e1 == e2:
        return {"status": "unidentifiable_signal", "energies": [float(e1), float(e2)],
                "KL_means": [float(k1), float(k2)], "null_KL_mean": null_mean}
    log_e = math.log(float(e2 / e1))
    log_k = math.log(float(k2 / k1))
    e1b, e2b = [row["E_sum"][bootstrap_indices].sum(axis=1) for row in (first, second)]
    k1b, k2b = [row["KL_mean"][bootstrap_indices].mean(axis=1) for row in (first, second)]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_eb = np.log(e2b / e1b)
        log_kb = np.log(k2b / k1b)
        alpha_bootstrap = log_kb / log_eb
        lambda_bootstrap = k1b / (e1b / COHORT["global_original_tokens"])
    alpha_bootstrap[(e1b <= 0) | (e2b <= 0) | (k1b <= 0) | (k2b <= 0) | (log_eb == 0)] = np.nan
    lambda_bootstrap[e1b <= 0] = np.nan
    energy_normalized = float(e1 / COHORT["global_original_tokens"])
    null_ratio = abs(null_mean) / min(float(k1), float(k2))
    return {"status": "measured" if null_ratio <= 0.2 else "needs_doubled_amplitude",
            "lambda": float(k1 / energy_normalized),
            "lambda_se": finite_summary(lambda_bootstrap)["se"],
            "alpha_component": log_k / log_e,
            "alpha_component_uncertainty": finite_summary(alpha_bootstrap),
            "anchor_amplitude": amplitude_first,
            "anchor_energy_global": energy_normalized,
            "anchor_KL": float(k1), "null_KL_mean": null_mean,
            "null_signal_ratio": null_ratio,
            "first_KL": float(k1), "second_KL": float(k2),
            "first_energy": float(e1), "second_energy": float(e2),
            "_log_e": log_e, "_log_k": log_k,
            "_log_eb": log_eb, "_log_kb": log_kb}


def _summarize_fit(fit):
    """JSON-safe fallback evidence without bootstrap arrays or per-sequence rows."""
    summary = {key: fit[key] for key in ("status", "lambda", "lambda_se", "alpha_component",
        "anchor_amplitude", "anchor_energy_global", "anchor_KL", "null_KL_mean",
        "null_signal_ratio", "first_KL", "second_KL", "first_energy", "second_energy") if key in fit}
    summary["alpha_component_uncertainty"] = fit.get("alpha_component_uncertainty")
    if "fit_pair" in fit:
        summary["fit_pair"] = fit["fit_pair"]
    summary["ready"] = "_log_e" in fit
    return summary


def curve(stream, bootstrap_indices):
    by_amplitude = {row["amplitude"]: row for row in stream["amplitudes"]}
    if len(by_amplitude) != len(stream["amplitudes"]) or any(
            type(amplitude) is not int for amplitude in by_amplitude):
        raise ValueError("A measured amplitude is duplicated or invalid")
    null_records = stream["null_per_sequence"]
    if set(by_amplitude) in ({1, 2}, {2, 4}):
        pair = sorted(by_amplitude)
        fit = _fit_pair(by_amplitude[pair[0]]["per_sequence"], by_amplitude[pair[1]]["per_sequence"],
                        null_records, bootstrap_indices, pair[0], pair[1])
        if "_log_e" not in fit:
            return dict(fit, fit_pair=pair)
        return _with_observations(stream, by_amplitude, pair, fit, None, None)
    if set(by_amplitude) == {1, 2, 4}:
        means = {amplitude: _mean_kl(by_amplitude[amplitude]["per_sequence"]) for amplitude in (1, 2, 4)}
        null_mean = _mean_kl(null_records)
        # The approved fit is (1,2), or (2,4) when the null rule fails on (1,2).
        # An unusable third point never changes the selected pair; its signal
        # stop applies only if the selected pair uses it.
        selectable = (all(math.isfinite(value) for value in (means[1], means[2], null_mean))
                      and means[1] > 0 and means[2] > 0
                      and abs(null_mean) / min(means[1], means[2]) <= 0.2)
        pair = [1, 2] if selectable else [2, 4]
        third = 4 if pair == [1, 2] else 1
        fit = _fit_pair(by_amplitude[pair[0]]["per_sequence"], by_amplitude[pair[1]]["per_sequence"],
                        null_records, bootstrap_indices, pair[0], pair[1])
        if "_log_e" not in fit:
            return dict(fit, fit_pair=pair, third_amplitude=_third_summary(by_amplitude[third]))
        other = [2, 4] if pair == [1, 2] else [1, 2]
        fallback = _fit_pair(by_amplitude[other[0]]["per_sequence"], by_amplitude[other[1]]["per_sequence"],
                             null_records, bootstrap_indices, other[0], other[1])
        return _with_observations(stream, by_amplitude, pair, fit, third,
                                  _summarize_fit({**fallback, "fit_pair": other}))
    raise ValueError("Each stream needs measured amplitudes 1,2 or the doubled pair 2,4")


def _mean_kl(records):
    import numpy as np
    return float(np.asarray([row["KL_mean"] for row in records], dtype=np.float64).mean())


def _third_summary(amplitude_row):
    """Retain the unused amplitude as diagnostic evidence, usable or not.

    Population, finite and nonnegative refusals stay. Zero signal is kept as
    an unusable diagnostic; it stops nothing unless the selected pair uses it."""
    import numpy as np
    records = amplitude_row["per_sequence"]
    if len(records) != 64 or {row["sample_id"] for row in records} != set(range(384, 448)):
        raise ValueError("The third-amplitude sample population differs")
    energy = np.asarray([row["E_sum"] for row in records], dtype=np.float64)
    kl = np.asarray([row["KL_mean"] for row in records], dtype=np.float64)
    if (energy < 0).any() or not np.isfinite(energy).all() or not np.isfinite(kl).all():
        raise ValueError("Nonfinite or negative third-amplitude observations")
    usable = bool(energy.sum() > 0 and kl.mean() > 0)
    return {"amplitude": amplitude_row["amplitude"], "E_sum": float(energy.sum()),
            "E_global_normalized": float(energy.sum() / COHORT["global_original_tokens"]),
            "KL_mean": float(kl.mean()),
            "status": "diagnostic" if usable else "zero_signal_unusable",
            "role": "diagnostic_and_ready_fallback"}


def _with_observations(stream, by_amplitude, pair, fit, third, fallback):
    del stream
    observations = []
    for amplitude in pair:
        data = aligned(by_amplitude[amplitude]["per_sequence"], ("E_sum", "KL_mean"))
        observations.append({"amplitude": amplitude,
            "E_sum": float(data["E_sum"].sum()),
            "E_global_normalized": float(data["E_sum"].sum() / COHORT["global_original_tokens"]),
            "KL_mean": float(data["KL_mean"].mean()),
            "KL_se": float(data["KL_mean"].std(ddof=1) / math.sqrt(64)),
            "per_sequence": by_amplitude[amplitude]["per_sequence"]})
    result = dict(fit, fit_pair=pair, amplitudes=observations)
    if third is not None:
        result["third_amplitude"] = _third_summary(by_amplitude[third])
    if fallback is not None:
        result["fallback_fit_" + "_".join(str(a) for a in fallback["fit_pair"])] = fallback
    return result


def normalize_declared_axes(declared):
    """Explicit phase axes as (band_start, band_stop, class, kind) triples.

    Only the canonical versioned pact.required_gain_axes.v1 declaration is
    accepted. Unknown or duplicated axes refuse; completeness never comes from
    present rows alone."""
    if not isinstance(declared, dict):
        raise ValueError("The declared axes must use the versioned required-axis declaration")
    if declared.get("schema") != "pact.required_gain_axes.v1":
        raise ValueError("The declared axis schema is unknown")
    entries = declared.get("required_axes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The declared phase axes are missing or invalid")
    triples = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("The declared phase axes are missing or invalid")
        band, cls, kind = entry.get("band"), entry.get("class"), entry.get("kind")
        if not isinstance(band, str) or band.count(":") != 1:
            raise ValueError("The declared phase axes are missing or invalid")
        try:
            start, stop = (int(part) for part in band.split(":"))
        except (TypeError, ValueError):
            raise ValueError("The declared phase axes are missing or invalid")
        triples.append((start, stop, cls, kind))
    axes = set()
    for axis in triples:
        if not valid_stream((axis[0], axis[1]), axis[2], axis[3]):
            raise ValueError("An unknown declared gain axis is present")
        if axis in axes:
            raise ValueError("A declared gain axis is duplicated")
        axes.add(axis)
    return axes


def reduce_observations(document, *, bootstrap_draws=2048, seed=20261006):
    import numpy as np
    if document.get("schema") != "pact.band_observations.v1" or any(
            document.get("cohort", {}).get(key) != value for key, value in COHORT.items()):
        raise ValueError("The measured prefixed-64 observation schema or cohort differs")
    token_sha = actual_token_sha256(document)
    cohort = dict(COHORT, token_sha256=token_sha)
    if bootstrap_draws < 2:
        raise ValueError("At least two bootstrap resamples are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 64, size=(bootstrap_draws, 64))
    rows, groups = [], {}
    identities = set()
    required = document.get("required_classes",["routed","nonrouted"])
    if not set(required) <= set(KINDS) or len(required) != len(set(required)):
        raise ValueError("The required gain classes are invalid")
    declared = document.get("required_axes")
    # Completeness comes from the explicit declared phase axes, never from the
    # kinds that happen to be present. First-ship declares T8/T16 axes; T4 and
    # attention axes belong to the full scope and must not block this phase.
    if declared is None:
        expected = {(*band, cls, kind) for band in [*BANDS, (45, 46)] for cls in required
                    for kind in KINDS[cls] if valid_stream(band, cls, kind)}
    else:
        expected = normalize_declared_axes(declared)
        if {axis[2] for axis in expected} != set(required):
            raise ValueError("The declared phase axes differ from the required classes")
    for stream in document["streams"]:
        band = (stream["band_start"], stream["band_stop"])
        cls, kind = stream["class"], stream["kind"]
        identity = (*band, cls, kind)
        if not valid_stream(band,cls,kind):
            raise ValueError("Unknown band, class or measured stream kind")
        if identity not in expected:
            raise ValueError("An unrequested gain axis is present; scope is declared, not inferred")
        if identity in identities:
            raise ValueError("Duplicated measured stream")
        identities.add(identity)
        result = {"band_start": band[0], "band_stop": band[1], "class": cls, "kind": kind,
                  "normalization": "global_original_token_energy", **curve(stream, indices)}
        rows.append(result)
        groups.setdefault((*band, cls), []).append(result)
    missing = sorted(expected - identities)
    stops = [{"reason": "missing_measurement", "stream": list(identity)} for identity in missing]
    band_alphas = []
    for identity, members in sorted(groups.items()):
        if any("_log_e" not in row for row in members):
            stops.append({"band": list(identity[:2]), "class": identity[2], "reason": "unidentifiable_signal"})
            continue
        denominator = sum(row["_log_e"] ** 2 for row in members)
        alpha = sum(row["_log_e"] * row["_log_k"] for row in members) / denominator
        log_e = np.stack([row["_log_eb"] for row in members])
        log_k = np.stack([row["_log_kb"] for row in members])
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_bootstrap = (log_e * log_k).sum(axis=0) / (log_e * log_e).sum(axis=0)
        uncertainty = finite_summary(alpha_bootstrap)
        entry = {"band_start": identity[0], "band_stop": identity[1], "class": identity[2],
                 "alpha": float(alpha), "uncertainty": uncertainty, "_bootstrap": alpha_bootstrap}
        band_alphas.append(entry)
        if not 0.5 <= alpha <= 2:
            stops.append({"band": list(identity[:2]), "class": identity[2], "reason": "alpha_outside_0.5_to_2", "alpha": float(alpha)})
        if uncertainty["se"] is None:
            stops.append({"band": list(identity[:2]), "class": identity[2], "reason": "alpha_uncertainty_unidentifiable"})
        for row in members:
            row.update(alpha_measured=float(alpha), alpha_se=uncertainty["se"],
                       energy_exponent=1.0 if 0.8 <= alpha <= 1.25 else float(alpha),
                       price_rule="linear" if 0.8 <= alpha <= 1.25 else "anchor_normalized_power")
            if row["status"] != "measured":
                stops.append({"stream": [*identity, row["kind"]], "reason": row["status"]})
    contrasts = []
    for first, second in itertools.combinations(band_alphas, 2):
        if first["class"] != second["class"]:
            continue
        delta = first["alpha"] - second["alpha"]
        uncertainty = finite_summary(first["_bootstrap"] - second["_bootstrap"])
        se = uncertainty["se"]
        both_linear = all(0.8 <= row["alpha"] <= 1.25 for row in (first, second))
        stop = not both_linear and se is not None and abs(delta) > 0.5 and abs(delta) > 2 * se
        contrast = {"class": first["class"], "first_band": [first["band_start"], first["band_stop"]],
                    "second_band": [second["band_start"], second["band_stop"]],
                    "delta_alpha": float(delta), "paired_bootstrap_se": se,
                    "both_linear_exemption": both_linear, "stop": stop,
                    "uncertainty": uncertainty}
        contrasts.append(contrast)
        if stop:
            stops.append({"reason": "alpha_band_heterogeneity", **contrast})
    clean = lambda row: {key: value for key, value in row.items() if not key.startswith("_")}
    if stops:
        for row in rows:
            row["status"] = "stop_and_report"
    return {"schema": "pact.band_gains.v1", "cohort": cohort,
            "gains": [clean(row) for row in rows], "band_alphas": [clean(row) for row in band_alphas],
            "alpha_contrasts": contrasts, "stop_reasons": stops,
            "status": "measured" if not stops else "stop_and_report",
            "bootstrap": {"draws": bootstrap_draws, "seed": seed,
                          "unit": "sequence", "same_resamples_across_all_bands": True},
            "decisions": ["dec-1006-222901-94f0", "dec-1006-224642-106f", "dec-1006-224708-1419"],
            "calibration_evidence": {gate: {"status": "pending_measurement", "evidence": None}
                for gate in ("T1_role_pooling", "T2_TCQ_body", "T3_far_anchor", "T4_nonrouted_T16",
                             "T5_panel_stack_ratio", "V1", "V2", "V3", "V4",
                             "prospective_quality_gate", "decisive_T16_band_arm")},
            "publication_policy": {"scientific_price_publication_authorized": False,
                "rule": "Every required gate must read passed with verifiable evidence; pending or failed evidence never authorizes scientific price publication.",
                "note": "CPU fixtures pass no gate; only measured GPU evidence changes a status."},
            "combined_uncertainty": {"combined_13_percent_assumed": False,
                "note": "Gain errors use the matched sequence bootstrap. Panel-ratio errors resample panel experts inside the same sequence draw at the panel stage. No independence is assumed here."},
            "T16_nonrouted_transfer_assumption": "T8 and T16 codec noise shapes have the same gain per unit of output-error energy; not established by these streams."}


def smoke():
    streams = []
    for band in BANDS:
        for cls in ("routed","nonrouted"):
            if band == (0, 3) and cls == "routed":
                continue
            for kind in sorted(KINDS[cls]):
                records = [{"sample_id": sample, "E_sum": 1 + (sample - 384) / 64,
                            "KL_mean": 0.001 * (1 + (sample - 384) / 64)} for sample in range(384, 448)]
                doubled = [{**row, "E_sum": row["E_sum"] * 4, "KL_mean": row["KL_mean"] * 4} for row in records]
                streams.append({"band_start": band[0], "band_stop": band[1], "class": cls, "kind": kind,
                    "amplitudes": [{"amplitude": 1, "per_sequence": records}, {"amplitude": 2, "per_sequence": doubled}],
                    "null_per_sequence": [{"sample_id": sample, "KL_mean": 0.0} for sample in range(384, 448)]})
    inputs = [[154822, 154824] + [0] * 512 for _ in range(64)]
    result = reduce_observations({"schema": "pact.band_observations.v1", "cohort": COHORT,
                                  "input_token_ids": inputs, "streams": streams}, bootstrap_draws=64)
    if result["status"] != "measured" or any(abs(row["alpha_measured"] - 1) > 1e-12 for row in result["gains"]):
        raise AssertionError("Matched linear amplitude response was not recovered")
    # Nonlinear pricing preserves measured anchor shares and scales a common amplitude.
    expected = (1 + 3) * 4 ** 1.6
    actual = anchor_response(4, 1, 1.6) + anchor_response(12, 3, 1.6)
    if not math.isclose(actual, expected, rel_tol=1e-12):
        raise AssertionError("Power normalization extrapolated across absolute unit energies")
    if anchor_response(7, 3, 1.1) != 7:
        raise AssertionError("Linear-band exponent was applied instead of the linear rule")
    try:
        anchor_response(1, 0, 1.6)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero anchor was replaced by an invented positive coefficient")
    triple = dict(streams[0])
    quadrupled = [{**row, "E_sum": row["E_sum"] * 16, "KL_mean": row["KL_mean"] * 16}
                  for row in triple["amplitudes"][0]["per_sequence"]]
    triple = {**triple, "amplitudes": [*triple["amplitudes"], {"amplitude": 4, "per_sequence": quadrupled}]}
    full = reduce_observations({"schema": "pact.band_observations.v1", "cohort": COHORT,
                                "input_token_ids": inputs, "streams": [triple, *streams[1:]]}, bootstrap_draws=64)
    first = full["gains"][0]
    if full["status"] != "measured" or first["fit_pair"] != [1, 2] or abs(first["alpha_measured"] - 1) > 1e-12:
        raise AssertionError("The third amplitude changed the accepted pair fit")
    if first.get("third_amplitude", {}).get("amplitude") != 4 or abs(
            first.get("fallback_fit_2_4", {}).get("alpha_component", 0) - 1) > 1e-12:
        raise AssertionError("The third amplitude was not retained as diagnostic and fallback evidence")
    return {"schema": "pact.band_gain_reduce_smoke.v1", "passed": True,
            "measured_synthetic_streams": len(result["gains"]),
            "limit": "Synthetic reducer behavior only; no measured band gain or GPU evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observations", type=Path)
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    result = smoke() if args.smoke else reduce_observations(json.loads(args.observations.read_bytes()), bootstrap_draws=args.bootstrap_draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result.get("passed") or result["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
