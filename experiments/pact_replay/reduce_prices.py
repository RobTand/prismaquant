"""Consume measured parent gains and actual own-family sequence energies.

Exactly 64 unique prefixed samples per required option; no partial/missing
row can be normalized as a full cohort. Nonrouted and attention T4 use the
direct joint T4+A4 gain. Routed T16 uses its own measured routed W_T16 gain
and own T16 anchor; a borrowed W_T8 coefficient stays refused. Approved
nonrouted W_T8 gain also prices T16 with the explicit codec-noise-shape
transfer assumption. No T16 encode occurs here. Each row reports its weight
and activation components separately with actual energy ratios; a one-eighth
bound applies to a weight component only, after an actual ratio <= 1/64 with
alpha >= 0.5, never to the complete price. Every row carries a matched price
uncertainty from 2048 shared sequence resamples at seed 20261006 over actual
raw sequence energies and retained gain observations; panel expert sets
resample inside the same draw when an approved panel roster is supplied, and
publication is refused when matched data or covariance are absent. T1-T5,
V1-V4, the prospective gate and the decisive T16 arm travel as calibration
evidence; pending or failed evidence never authorizes scientific prices, and
a claimed pass without verifiable evidence refuses publication. CPU fixtures
pass no gate.

Decisions dec-1006-222901-94f0, dec-1006-224642-106f and
 dec-1006-224708-1419: per-unit T(E)=E_anchor*(E/E_anchor)^alpha outside
[0.8,1.25]; inside retain linear. Single alpha outside [0.5,2] stops.
Same-class/kind bands stop only when difference >0.5 AND >2 paired
bootstrap SE, except both-linear pairs. All contrasts/SE are reported.
Zero own anchor with positive candidate is unidentifiable, never epsilon.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from itertools import combinations

EXPECTED = {"sample_range": [384,448], "raw_tokens_per_sequence": 512,
            "prefix_ids": [154822,154824], "input_contract": "prefixed_514",
            "local_prefix_rows": "excluded", "global_original_tokens": 32768,
            "scored_positions_per_sequence": 511}
ENERGIES = ("E_W_sum", "E_A_sum", "E_WA_sum", "E_W_sum_amp2", "E_A_sum_amp2", "E_WA_sum_amp2")


def number(v, field):
    if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v):
        raise ValueError("Not a finite numeric " + field)
    return float(v)


def read_energies(path):
    options, seen, tokens = {}, set(), set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "pact.stage1a_per_sequence_energy.v1" or row.get("dry_run_cpu") or row.get("missing"):
                raise ValueError("Incomplete, CPU-proof or unknown energy rows cannot be priced")
            if (row.get("input_contract") != "prefixed_514" or row.get("local_prefix_rows") != "excluded"
                or row.get("prefix_rows_enter_local_energies") is not False or row.get("prefix_ids") != EXPECTED["prefix_ids"]
                or row.get("global_original_tokens") != 32768 or row.get("original_tokens") != 512):
                raise ValueError("Energy row cohort differs from fixed clean prefixed64")
            if not row.get("amp2_measured"):
                raise ValueError("Amplitude two must be measured, not inferred four times joint energy")
            sample = row["sample_id"]
            if type(sample) is not int or not 384 <= sample < 448:
                raise ValueError("Energy sample outside fixed64 cohort")
            key = (row["qname"], row["weight_source"])
            if (key, sample) in seen:
                raise ValueError("Duplicate energy unit/option/sample")
            seen.add((key, sample))
            tokens.add(row["input_token_sha256"])
            for e in ENERGIES:
                if number(row[e], e) < 0:
                    raise ValueError("Measured squared output energy cannot be negative")
            entry = options.setdefault(key, {"meta": row, "samples": set(), "sequence": {},
                                              **{e:0.0 for e in ENERGIES}})
            for field in ("layer", "kind", "role", "family", "q256", "activation_contract",
                          "diagnostic_only", "passthrough", "weight_reference_definition", "canonical_T16_priceable", "anchor_source"):
                if entry["meta"].get(field) != row.get(field):
                    raise ValueError("Option metadata differs across paired sequences")
            entry["samples"].add(sample)
            entry["sequence"][sample] = {e: number(row[e], e) for e in ENERGIES}
            for e in ENERGIES:
                entry[e] += row[e]
    if not options or len(tokens) != 1:
        raise ValueError("No energies or inconsistent actual token distribution")
    for entry in options.values():
        if entry["samples"] != set(range(384,448)):
            raise ValueError("Each unit/option needs all64 unique sequences before normalization")
    names = {name for name,_ in options}
    for name in names:
        for source in ("A8S", "A4-q896", "EXL3"):
            required = ("A8S",) if next(entry["meta"]["kind"] for key,entry in options.items() if key[0] == name) in {"attention","lm_head"} else ("A8S","A4-q896","EXL3")
            if source in required and (name,source) not in options:
                raise ValueError("Missing primary/EXL3 option for " + name)
    return options, next(iter(tokens))


def load_gains(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "pact.band_gains.v1":
        raise ValueError("Parent gain schema must be pact.band_gains.v1")
    for key,value in EXPECTED.items():
        if doc["cohort"].get(key) != value:
            raise ValueError("Gain cohort differs: " + key)
    gains = {}
    for row in doc["gains"]:
        key = (row["band_start"], row["band_stop"], row["class"], row["kind"])
        if key in gains:
            raise ValueError("Duplicate measured band gain")
        if row.get("status") != "measured" or row.get("normalization") != "global_original_token_energy":
            raise ValueError("Unmeasured or wrong-normalization gain")
        alpha = number(row["alpha_measured"], "alpha")
        if not .5 <= alpha <= 2:
            raise ValueError("Single measured alpha outside [0.5,2]; stop/report")
        number(row["alpha_se"], "alpha SE")
        supplied_lambda = number(row["lambda"], "lambda")
        anchor_amplitude = row.get("anchor_amplitude")
        if type(anchor_amplitude) is not int or anchor_amplitude not in (1, 2):
            raise ValueError("The gain needs its actual amplitude-one or amplitude-two anchor")
        anchors = [a for a in row["amplitudes"] if a["amplitude"] == anchor_amplitude]
        if len(anchors) != 1:
            raise ValueError("The measured gain anchor is absent or duplicated")
        anchor_sum = number(anchors[0]["E_sum"], "band anchor sum")
        anchor_kl = number(anchors[0]["KL_mean"], "band anchor KL")
        if anchor_sum <= 0 or anchor_kl <= 0:
            raise ValueError("The gain needs positive measured anchor energy and KL")
        anchor_lambda = anchor_kl / (anchor_sum / 32768)
        if not math.isclose(supplied_lambda, anchor_lambda, rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError("Supplied gain differs from anchor KL divided by global-normalized anchor energy")
        pair = row.get("fit_pair")
        if list(pair or []) not in ([1, 2], [2, 4]):
            raise ValueError("The gain fit pair is not an accepted amplitude pair")
        kept = {a.get("amplitude"): a for a in row["amplitudes"]}
        if set(kept) != set(pair):
            raise ValueError("The retained gain observations differ from the accepted fit pair")
        series = {}
        for amplitude in pair:
            records = kept[amplitude].get("per_sequence")
            if not isinstance(records, list) or len(records) != 64:
                raise ValueError("The retained gain observations are incomplete; matched uncertainty is unavailable")
            ordered = {}
            for record in records:
                sample = record.get("sample_id")
                energy = number(record.get("E_sum"), "retained gain energy")
                kl = number(record.get("KL_mean"), "retained gain KL")
                if type(sample) is not int or not 384 <= sample < 448 or energy < 0 or sample in ordered:
                    raise ValueError("The retained gain observations are incomplete; matched uncertainty is unavailable")
                ordered[sample] = (energy, kl)
            if set(ordered) != set(range(384, 448)):
                raise ValueError("The retained gain observations are incomplete; matched uncertainty is unavailable")
            series[amplitude] = ordered
        row["_matched"] = {"pair": list(pair), "series": series}
        gains[key] = row
    if not gains:
        raise ValueError("No measured parent gains")
    contrasts = []
    for left,right in combinations(gains,2):
        if left[2:] != right[2:]:
            continue
        a,b = gains[left], gains[right]
        delta = a["alpha_measured"]-b["alpha_measured"]
        both_linear = all(.8 <= r["alpha_measured"] <= 1.25 for r in (a,b))
        supplied = next((c for c in doc.get("alpha_contrasts", [])
                         if c["class"] == left[2]
                         and {tuple(c["first_band"]), tuple(c["second_band"])} == {left[:2], right[:2]}), None)
        if supplied is None:
            raise ValueError("The gain producer omitted a paired class contrast")
        se = number(supplied["paired_bootstrap_se"], "paired alpha contrast SE")
        if se < 0:
            raise ValueError("The paired alpha contrast SE is negative")
        stop = not both_linear and abs(delta) > .5 and abs(delta) > 2 * se
        contrasts.append({"class": left[2], "kind": left[3],
                          "first_band": left[:2], "second_band": right[:2],
                          "delta_alpha": delta, "paired_bootstrap_se": se,
                          "both_linear_exemption": both_linear, "stop": stop})
    if any(c["stop"] for c in contrasts):
        raise ValueError("Measured same-class alpha heterogeneity exceeds 0.5 and two paired SE: " + json.dumps(contrasts))
    return doc, gains, contrasts


REQUIRED_GATES = ("T1_role_pooling", "T2_TCQ_body", "T3_far_anchor", "T4_nonrouted_T16",
                  "T5_panel_stack_ratio", "V1", "V2", "V3", "V4",
                  "prospective_quality_gate", "decisive_T16_band_arm")
MATCHED_DRAWS = 2048
MATCHED_SEED = 20261006


def publication_authorization(doc):
    """Machine-readable calibration gate for scientific price publication.

    Pending or failed evidence never authorizes scientific prices. A claimed
    pass without verifiable evidence refuses the whole publication."""
    raw = doc.get("calibration_evidence")
    if raw is None:
        legacy = doc.get("eligibility")
        if not isinstance(legacy, dict):
            raise ValueError("The gain document carries no calibration evidence")
        raw = {}
        for gate in REQUIRED_GATES:
            value = legacy.get(gate, legacy.get("V1_V2_V3_V4") if gate in ("V1", "V2", "V3", "V4") else None)
            if value != "pending_measurement":
                raise ValueError("The legacy eligibility claim is unverifiable")
            raw[gate] = {"status": "pending_measurement", "evidence": None}
    evidence = {}
    for gate in REQUIRED_GATES:
        entry = raw.get(gate)
        if not isinstance(entry, dict):
            raise ValueError("The calibration evidence is incomplete")
        status = entry.get("status")
        if status not in ("pending_measurement", "passed", "failed"):
            raise ValueError("The calibration status is unknown")
        proof = entry.get("evidence")
        if status == "passed":
            if not isinstance(proof, dict):
                raise ValueError("A claimed gate pass lacks verifiable evidence")
            digest = proof.get("digest_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in digest)):
                raise ValueError("A claimed gate pass lacks verifiable evidence")
            if not isinstance(proof.get("path"), str) or not proof.get("path"):
                raise ValueError("A claimed gate pass lacks verifiable evidence")
            import hashlib
            try:
                raw_proof = Path(proof["path"]).read_bytes()
            except OSError as error:
                raise ValueError("A claimed calibration proof cannot be read") from error
            if hashlib.sha256(raw_proof).hexdigest() != digest.lower():
                raise ValueError("The calibration proof differs from its actual byte digest")
            record = json.loads(raw_proof)
            axes = sorted([[row["band_start"], row["band_stop"], row["class"], row["kind"]]
                           for row in doc["gains"]])
            if (record.get("schema") != "pact.calibration_gate_proof.v1" or record.get("gate") != gate
                    or record.get("status") != "passed" or record.get("stop_reasons") != []):
                raise ValueError("The actual calibration proof does not pass this gate")
            if record.get("cohort") != doc["cohort"] or record.get("gain_axes") != axes:
                raise ValueError("The calibration proof does not bind the actual cohort and declared gain scope")
        elif status == "pending_measurement" and proof is not None:
            raise ValueError("Pending calibration carries no evidence")
        evidence[gate] = {"status": status, "evidence": proof}
    authorized = all(entry["status"] == "passed" for entry in evidence.values())
    return {"calibration_evidence": evidence, "scientific_price_publication_authorized": authorized}


def matched_gain_draws(gains, band, cls, indices):
    """Per-draw pooled alpha and per-kind lambda from retained observations.

    Every series is resampled with the same shared sequence indices, so gain
    and energy draws stay matched and their covariance enters the prices."""
    import numpy as np
    members = sorted(key for key in gains if key[:2] == tuple(band) and key[2] == cls)
    if not members:
        raise ValueError("No measured gains for the priced band and class")
    log_e, log_k, lambdas, pairs = [], [], {}, {}
    for key in members:
        row = gains[key]
        pair = row["_matched"]["pair"]
        series = row["_matched"]["series"]
        order = list(range(384, 448))
        first = np.array([series[pair[0]][s][0] for s in order])
        first_kl = np.array([series[pair[0]][s][1] for s in order])
        second = np.array([series[pair[1]][s][0] for s in order])
        second_kl = np.array([series[pair[1]][s][1] for s in order])
        e1b, e2b = first[indices].sum(axis=1), second[indices].sum(axis=1)
        k1b, k2b = first_kl[indices].mean(axis=1), second_kl[indices].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_eb = np.log(e2b / e1b)
            log_kb = np.log(k2b / k1b)
            lambda_b = k1b / (e1b / 32768)
        bad = ((e1b <= 0) | (e2b <= 0) | (k1b <= 0) | (k2b <= 0) | (log_eb == 0)
               | ~np.isfinite(log_eb) | ~np.isfinite(log_kb))
        log_eb, log_kb = log_eb.astype(float), log_kb.astype(float)
        log_eb[bad] = np.nan
        log_kb[bad] = np.nan
        lambda_b = lambda_b.astype(float)
        lambda_b[bad | ~np.isfinite(lambda_b)] = np.nan
        log_e.append(log_eb)
        log_k.append(log_kb)
        lambdas[key[3]] = lambda_b
        pairs[key[3]] = pair
    log_e = np.stack(log_e)
    log_k = np.stack(log_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (log_e * log_k).sum(axis=0) / (log_e * log_e).sum(axis=0)
    return {"alpha": alpha, "lambda": lambdas, "pairs": pairs}


def matched_row_uncertainty(parts, gain_draws, energy_series, indices, *, energy_draws=None):
    """Price draws with gain and energy resampled on the same indices.

    Each part prices one signed component from one gain kind with the shared
    pooled band alpha. A draw is valid only when every part is finite with a
    positive anchor where the nonlinear rule needs one. Fewer than two valid
    draws refuses the row."""
    import numpy as np
    order = list(range(384, 448))
    total = np.zeros(indices.shape[0])
    valid = np.ones(indices.shape[0], dtype=bool)
    for part in parts:
        if energy_draws is None:
            cand = np.array([energy_series[part["candidate"]][s] for s in order])[indices].sum(axis=1)
            anch = np.array([energy_series[part["anchor"]][s] for s in order])[indices].sum(axis=1)
        else:
            cand, anch = energy_draws[part["candidate"]], energy_draws[part["anchor"]]
        lam = gain_draws["lambda"][part["kind"]]
        alpha = gain_draws["alpha"]
        with np.errstate(divide="ignore", invalid="ignore"):
            if part["linear"]:
                draw = lam * cand / 32768
                good = np.isfinite(draw)
            else:
                draw = lam * anch * (cand / anch) ** alpha / 32768
                good = np.isfinite(draw) & (anch > 0) & (cand >= 0)
        zero_pair = (anch == 0) & (cand == 0)
        draw = np.where(zero_pair, 0.0, draw)
        good = (good & ((anch > 0) | zero_pair)) | zero_pair
        total = total + part.get("sign", 1) * np.where(good, draw, 0.0)
        valid = valid & good
    good_total = total[valid]
    if len(good_total) < 2:
        raise ValueError("Matched draws cannot support the price uncertainty")
    return {"se": float(good_total.std(ddof=1)),
            "interval_95": [float(v) for v in np.quantile(good_total, [0.025, 0.975])],
            "valid_draws": int(len(good_total))}


def draw_part(kind, gain_row, series, candidate_seq, anchor_seq, cand_field, anch_field, sign=1):
    """One signed draw component: series tags plus the gain's own rule."""
    field = anch_field if gain_row["anchor_amplitude"] == 1 else anch_field + "_amp2"
    candidate_tag, anchor_tag = kind + ":" + cand_field, kind + ":" + field + ":anchor"
    if candidate_tag not in series:
        series[candidate_tag] = {s: candidate_seq[s][cand_field] for s in candidate_seq}
    if anchor_tag not in series:
        series[anchor_tag] = {s: anchor_seq[s][field] for s in anchor_seq}
    return {"kind": kind, "linear": 0.8 <= gain_row["alpha_measured"] <= 1.25,
            "sign": sign, "candidate": candidate_tag, "anchor": anchor_tag}


def transformed(e, anchor, alpha):
    if anchor == 0:
        if e == 0:
            return 0.0
        raise ValueError("Zero own measured anchor with positive candidate is unidentifiable")
    if .8 <= alpha <= 1.25:
        return e
    return anchor * (e/anchor)**alpha


def anchor_energy(own_anchor, field, g):
    """Actual anchor energy at the gain's own anchor amplitude."""
    return own_anchor[field if g["anchor_amplitude"] == 1 else field + "_amp2"]


def component_bound_1_8(candidate, anchor, alpha):
    """Report-only one-eighth component bound from actual measured energies.

    A component ratio <= 1/8 is established only for that component, after an
    actual measured energy ratio <= 1/64 with alpha >= 0.5. It never caps a
    complete price, which keeps its activation term."""
    if anchor <= 0 or candidate < 0 or not alpha >= 0.5:
        return {"energy_ratio": candidate / anchor if anchor > 0 else None,
                "bound_1_8_established": False}
    ratio = candidate / anchor
    return {"energy_ratio": ratio, "bound_1_8_established": 0 < ratio <= 1 / 64}


def t8_reference_sources(options, anchors):
    """Resolve one actual four-bit T8 reference for every unit's rates."""
    references = {}
    for (qname, source), entry in options.items():
        meta = entry["meta"]
        if meta["family"] != "T8" or meta.get("quality_chord_check_only"):
            continue
        reference = meta.get("anchor_source")
        if reference is None:
            reference = "A8S"
        if not isinstance(reference, str) or not reference:
            raise ValueError("The T8 option has no valid measured reference name")
        if references.setdefault(qname, reference) != reference:
            raise ValueError("T8 rates for one unit must share one measured reference")
        anchor = anchors.get((qname, reference))
        if anchor is None:
            raise ValueError("Missing actual common T8 reference anchor for " + qname)
        anchor_meta = anchor["meta"]
        if anchor_meta["family"] != "T8" or anchor_meta["q256"] != 1024 or anchor_meta.get("passthrough"):
            raise ValueError("The common T8 reference must be an actual four-bit T8 anchor")
        for field in ("layer", "kind", "role", "expert", "activation_contract"):
            if anchor_meta.get(field) != meta.get(field):
                raise ValueError("The common T8 reference and candidate are incompatible: " + field)
    return references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energies", type=Path, required=True)
    p.add_argument("--anchors", type=Path,
                   help="Actual own-unit/family anchor rows; defaults to these existing primary anchor energies")
    p.add_argument("--gains", type=Path, required=True)
    p.add_argument("--panel-stack-scope", type=Path,
                   help="Approved routed stacks with actual full references and explicit panel units")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    args = p.parse_args()
    try:
        options,token_sha = read_energies(args.energies)
        anchors,anchor_token = read_energies(args.anchors or args.energies)
        if token_sha != anchor_token:
            raise ValueError("Anchor and candidate token distributions differ now")
        gain_doc,gains,contrasts = load_gains(args.gains)
        publication = publication_authorization(gain_doc)
        import numpy as np
        matched_indices = np.random.default_rng(MATCHED_SEED).integers(0, 64, size=(MATCHED_DRAWS, 64))
        independent_gain_indices = np.random.default_rng(MATCHED_SEED + 1).integers(0, 64, size=(MATCHED_DRAWS, 64))
        independent_energy_indices = np.random.default_rng(MATCHED_SEED + 2).integers(0, 64, size=(MATCHED_DRAWS, 64))
        gain_draw_cache = {}
        def draws_for(band, cls, indices, tag):
            key = (tuple(band), cls, tag)
            if key not in gain_draw_cache:
                gain_draw_cache[key] = matched_gain_draws(gains, band, cls, indices)
            return gain_draw_cache[key]
        if gain_doc["cohort"].get("token_sha256") != token_sha:
            raise ValueError("The actual gain and energy token populations differ")
        panel_contexts, panel_evidence = {}, []
        if args.panel_stack_scope:
            from panel_stack import materialize
            options, anchors, panel_contexts, panel_evidence = materialize(
                args.panel_stack_scope, options, anchors, gain_doc["cohort"], ENERGIES)
        t8_references = t8_reference_sources(options, anchors)
        priced,unpriced = [],[]
        denominator = 32768
        for (qname,source),entry in sorted(options.items()):
            meta = entry["meta"]
            if meta["family"] == "T16" and (meta.get("canonical_T16_priceable") is not True
                    or meta.get("weight_reference_definition") != "separate_values_and_row_scales_through_dot"):
                raise ValueError("The folded BF16 T16 render cannot supply canonical T16 prices")
            cls = meta["kind"] if meta["kind"] in {"routed","attention","lm_head"} else "nonrouted"
            bands = {(s,e) for s,e,c,k in gains if c == cls and s <= meta["layer"] < e}
            if len(bands) != 1:
                raise ValueError("No unique measured band for candidate class/layer")
            band = next(iter(bands))
            if source == "EXL3" or meta.get("diagnostic_only") and meta.get("quality_chord_check_only"):
                unpriced.append({"qname":qname,"option":source,"reason":"validation/chord diagnostic, not a serving candidate"})
                continue
            if meta.get("passthrough"):
                if entry["E_W_sum"] != 0 or entry["E_A_sum"] != 0 or entry["E_WA_sum"] != 0:
                    raise ValueError("SOURCE passthrough has nonzero energy")
                price = 0.0
                assumptions = []
                used = []
                weight_part, activation_part = 0.0, 0.0
                ratios, bounds, derivation = {}, {}, "passthrough_zero"
                uncertainty = {"predicted_KL_se": 0.0, "predicted_KL_interval_95": [0.0, 0.0],
                    "uncertainty_draws": MATCHED_DRAWS, "uncertainty_seed": MATCHED_SEED,
                    "uncertainty_valid_draws": MATCHED_DRAWS, "independence_reference_se": 0.0,
                    "scientific_price_authorized": publication["scientific_price_publication_authorized"]}
            else:
                family = meta["family"]
                own_source = t8_references[qname] if family == "T8" else meta.get("anchor_source") or (source if family == "T16" and cls in {"routed","attention","lm_head"} else "A4-q896" if family == "T4" else "A8S")
                anchor = anchors.get((qname,own_source))
                if anchor is None:
                    raise ValueError("Missing actual own-family anchor for candidate")
                if family == "T16" and cls == "routed":
                    anchor_meta = anchor["meta"]
                    if (anchor_meta["family"] != "T16" or anchor_meta.get("canonical_T16_priceable") is not True
                            or anchor_meta.get("weight_reference_definition") != "separate_values_and_row_scales_through_dot"):
                        raise ValueError("Routed T16 requires an actual canonical T16 anchor")
                    for field in ("layer", "kind", "role", "expert", "activation_contract", "tp_splits",
                                  "precision", "body", "outer_scheme"):
                        if anchor_meta.get(field) != meta.get(field):
                            raise ValueError("The routed T16 anchor and candidate are incompatible: " + field)
                def gain(kind):
                    row = gains.get((*band,cls,kind))
                    if row is None:
                        raise ValueError("Missing measured " + cls + " " + kind + " band gain")
                    return row
                def part(candidate, own_anchor, field, g):
                    return g["lambda"] * transformed(candidate[field], anchor_energy(own_anchor, field, g), g["alpha_measured"]) / denominator
                def track(candidate, own_anchor, field, g):
                    return component_bound_1_8(candidate[field], anchor_energy(own_anchor, field, g), g["alpha_measured"])
                used = []
                assumptions = []
                weight_part, activation_part = 0.0, 0.0
                ratios, bounds = {}, {}
                series, parts = {}, []
                if set(entry["sequence"]) != set(anchor["sequence"]):
                    raise ValueError("Candidate and anchor sequence populations differ")
                if cls in ("nonrouted", "attention") and family == "T4":
                    j = gain("joint_T4_A4")
                    price = part(entry, anchor, "E_WA_sum", j)
                    weight_part = price
                    used.append("joint_T4_A4 direct joint")
                    parts.append(draw_part("joint_T4_A4", j, series, entry["sequence"], anchor["sequence"], "E_WA_sum", "E_WA_sum"))
                    check = track(entry, anchor, "E_WA_sum", j)
                    ratios["E_WA_sum"], bounds["E_WA_sum"] = check["energy_ratio"], check["bound_1_8_established"]
                    span = (entry["E_WA_sum"], anchor_energy(anchor, "E_WA_sum", j))
                elif family in ("T4","T8","T16"):
                    if family == "T16":
                        weight_kind = "W_T16" if cls in {"routed","attention","lm_head"} else "W_T8"
                    else:
                        weight_kind = "W_T4" if family == "T4" else "W_T8"
                    if family == "T16" and cls == "routed" and (*band,cls,weight_kind) not in gains:
                        raise ValueError("No measured routed W_T16 band gain; a borrowed routed W_T8 coefficient is refused")
                    w = gain(weight_kind)
                    weight_part = part(entry, anchor, "E_W_sum", w)
                    price = weight_part
                    used.append(weight_kind)
                    parts.append(draw_part(weight_kind, w, series, entry["sequence"], anchor["sequence"], "E_W_sum", "E_W_sum"))
                    check = track(entry, anchor, "E_W_sum", w)
                    ratios["E_W_sum"], bounds["E_W_sum"] = check["energy_ratio"], check["bound_1_8_established"]
                    span = (entry["E_W_sum"], anchor_energy(anchor, "E_W_sum", w))
                    if family == "T16":
                        if cls == "routed":
                            assumptions.append("Routed T16 uses its own measured routed W_T16 gain and own T16 anchor; no borrowed T8 coefficient")
                        elif cls == "nonrouted":
                            assumptions.append("Approved nonrouted T8/T16 codec-noise shapes have same gain per error energy; cheap check only after existing T16 renders or prospective release")
                    else:
                        a = gain("A4" if family == "T4" else "A8")
                        activation_part = part(entry, anchor, "E_WA_sum", a) - part(entry, anchor, "E_W_sum", a)
                        price += activation_part
                        used.append("A4" if family == "T4" else "A8")
                        act_kind = "A4" if family == "T4" else "A8"
                        parts.append(draw_part(act_kind, a, series, entry["sequence"], anchor["sequence"], "E_WA_sum", "E_WA_sum"))
                        parts.append(draw_part(act_kind, a, series, entry["sequence"], anchor["sequence"], "E_W_sum", "E_W_sum", sign=-1))
                        check = track(entry, anchor, "E_WA_sum", a)
                        ratios["E_WA_sum"], bounds["E_WA_sum"] = check["energy_ratio"], check["bound_1_8_established"]
                else:
                    raise ValueError("Unknown family lacks measured own anchors/gains")
                derivation = "model_derived_below_measured_span" if span[0] < span[1] else "measured_span"
                energy_draws = independent_draws = None
                if qname in panel_contexts:
                    from panel_stack import matched_energy_draws
                    energy_draws = matched_energy_draws(panel_contexts[qname], source, parts, matched_indices, MATCHED_SEED + 3)
                    independent_draws = matched_energy_draws(panel_contexts[qname], source, parts, independent_energy_indices, MATCHED_SEED + 3)
                matched = matched_row_uncertainty(parts, draws_for(band, cls, matched_indices, "matched"),
                    series, matched_indices, energy_draws=energy_draws)
                independent = matched_row_uncertainty(parts, draws_for(band, cls, independent_gain_indices, "independent_gain"),
                    series, independent_energy_indices, energy_draws=independent_draws)
                uncertainty = {"predicted_KL_se": matched["se"], "predicted_KL_interval_95": matched["interval_95"],
                    "uncertainty_draws": MATCHED_DRAWS, "uncertainty_seed": MATCHED_SEED,
                    "uncertainty_valid_draws": matched["valid_draws"],
                    "independence_reference_se": independent["se"],
                    "scientific_price_authorized": publication["scientific_price_publication_authorized"]}
            if not math.isfinite(price):
                raise ValueError("Nonfinite predicted price")
            if not math.isfinite(weight_part) or not math.isfinite(activation_part):
                raise ValueError("Nonfinite price component")
            priced.append({"decision_unit_scope":meta.get("decision_unit_scope", "actual_unit"),
                "estimated_panel_stack":meta.get("estimated_panel_stack", False),
                "qname":qname,"layer":meta["layer"],"kind":meta["kind"],"role":meta["role"],
                "family":meta["family"],"q256":meta["q256"],"weight_source":source,
                "activation_contract":meta["activation_contract"],"predicted_KL":price,
                "predicted_KL_weight":weight_part,"predicted_KL_activation":activation_part,
                "component_energy_ratios":ratios,"component_bound_1_8":bounds,
                "derivation":derivation, **uncertainty,
                "gain_kinds":used,"band":band,"assumptions":assumptions,
                "diagnostic_only":meta.get("diagnostic_only",False),"global_original_tokens":denominator,
                **{e:entry[e] for e in ENERGIES}})
        result = {"schema":"pact.unit_price_rows.v2","cohort":gain_doc["cohort"],
                  "token_sha256":token_sha,"rows":priced,"unpriced":unpriced,
                  "panel_stack_evidence":panel_evidence,
                  "alpha_contrasts":contrasts,"clipped_or_rescaled":False,
                  "nonlinear_rule":"own-unit anchor-normalized powers; joint interaction counted once",
                  "calibration_evidence":publication["calibration_evidence"],
                  "scientific_price_publication_authorized":publication["scientific_price_publication_authorized"],
                  "combined_uncertainty":{"status":"matched_bootstrap","draws":MATCHED_DRAWS,"seed":MATCHED_SEED,
                      "same_sequence_resamples":True,
                      "panel_expert_resampling": "Experts are resampled inside each matched sequence draw for declared routed stacks." if panel_contexts else "Actual unit energies need no panel estimator.",
                      "independence_assumption":"not used; a per-row independence reference SE is reported for comparison only"}}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        report = {"schema":"pact.price_reduction_report.v2","status":"reduced",
                  "rows":len(priced),"alpha_contrasts":contrasts,"cohort":gain_doc["cohort"],
                  "calibration_evidence":publication["calibration_evidence"],
                  "scientific_price_publication_authorized":publication["scientific_price_publication_authorized"],
                  "combined_uncertainty":result["combined_uncertainty"]}
    except (ValueError,KeyError,TypeError) as exc:
        report = {"schema":"pact.price_reduction_report.v2","status":"refused",
                  "reason":str(exc),"valid_price_rows_published":False,
                  "scientific_price_publication_authorized":False}
        args.report_out.parent.mkdir(parents=True,exist_ok=True)
        args.report_out.write_text(json.dumps(report,indent=2)+"\n")
        raise
    args.report_out.parent.mkdir(parents=True,exist_ok=True)
    args.report_out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
