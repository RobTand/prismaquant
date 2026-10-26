"""Approved routed-stack energy estimates with matched expert/sequence draws."""
from __future__ import annotations
import copy
import json

from pathlib import Path


# Every expert of one panel role must share one numerical contract. The aggregate
# keeps the first expert metadata, so a later mismatch must refuse here.
COMPARABLE_FIELDS = ("family", "q256", "activation_contract", "weight_reference_definition",
                     "canonical_T16_priceable", "diagnostic_only", "body", "outer_scheme",
                     "tp_splits", "precision")


def require_comparable(entries, meta, label):
    """Refuse any entry whose numerical contract differs from ``meta``."""
    for entry in entries:
        for key in COMPARABLE_FIELDS:
            if entry["meta"].get(key) != meta.get(key):
                raise ValueError("The actual " + label + " entries are not comparable: " + key)


def materialize(scope_path, options, anchors, cohort, fields):
    import numpy as np
    scope = json.loads(Path(scope_path).read_bytes())
    if scope.get("schema") != "pact.panel_stack_scope.v1" or scope.get("cohort") != cohort:
        raise ValueError("The panel stack scope or actual cohort differs")
    groups = scope.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("The approved stack decision-unit groups are absent")
    candidates, own_anchors, contexts, evidence = dict(options), dict(anchors), {}, []
    samples = list(range(384, 448))
    seen, covered = set(), set()
    for group in groups:
        name = group["decision_unit"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("The stack decision unit is absent or duplicated")
        seen.add(name)
        reference_units, panel_units = group["reference_units"], group["panel_units"]
        if (not isinstance(reference_units, list) or not isinstance(panel_units, list)
                or len(reference_units) != len(set(reference_units))
                or len(panel_units) != len(set(panel_units)) or len(panel_units) < 2
                or not set(panel_units) <= set(reference_units)):
            raise ValueError("The actual stack reference or panel population differs")
        if covered & set(reference_units):
            raise ValueError("A measured expert belongs to more than one declared stack")
        covered.update(reference_units)
        candidates = {key: value for key, value in candidates.items() if key[0] not in covered}
        own_anchors = {key: value for key, value in own_anchors.items() if key[0] not in covered}
        if group.get("total_experts") != len(reference_units):
            raise ValueError("The declared expert count differs from the full measured reference")
        reference_source = group["reference_source"]
        table = anchors if group["reference_input"] == "anchors" else options if group["reference_input"] == "energies" else None
        if table is None:
            raise ValueError("The full reference input owner is not declared")
        def validate(entry):
            meta = entry["meta"]
            if meta["kind"] != "routed" or meta["layer"] != group["layer"] or meta["role"] != group["role"]:
                raise ValueError("The actual expert energy lies outside its stack scope")
            return entry
        full = [validate(table[(unit, reference_source)]) for unit in reference_units]
        if any(entry["meta"]["family"] != "T8" or entry["meta"]["q256"] != 1024 for entry in full):
            raise ValueError("The full stack reference must use comparable four-bit T8 energies")
        require_comparable(full, full[0]["meta"], "full stack reference")
        full_reference = {field: np.array([sum(entry["sequence"][sample][field] for entry in full)
                                          for sample in samples], dtype=np.float64) for field in fields}
        reference_panel = [validate(table[(unit, reference_source)]) for unit in panel_units]
        ref_panel = {field: np.array([[entry["sequence"][sample][field] for sample in samples]
                                     for entry in reference_panel], dtype=np.float64) for field in fields}
        source_names = group["price_sources"]
        anchor_sources = group["anchor_sources"]
        required = sorted(set(source_names) | set(anchor_sources.values()) | {reference_source})
        if not source_names or set(anchor_sources) != set(source_names):
            raise ValueError("Each declared price source needs exactly one own-anchor source")
        reference_table = table
        raw, zero_fields = {}, {}
        roles = [("candidate", source) for source in source_names]
        roles += [("anchor", source) for source in sorted(set(anchor_sources.values()) | {reference_source})]
        for role, source in roles:
            source_table = options if role == "candidate" else anchors
            if source == reference_source:
                if source_table is not reference_table:
                    try:
                        declared = [validate(source_table[(unit, source)]) for unit in panel_units]
                    except KeyError:
                        raise ValueError("The stack reference source is absent from its " + role + " table")
                    if any(own["sequence"][sample][field] != ref["sequence"][sample][field]
                           for own, ref in zip(declared, reference_panel)
                           for sample in samples for field in fields):
                        raise ValueError("The " + role + " table and the declared stack reference differ")
                    # The substituted reference panel must carry the same numerical contract.
                    require_comparable(declared, reference_panel[0]["meta"], role + " table reference")
                entries = reference_panel
            else:
                entries = [validate(source_table[(unit, source)]) for unit in panel_units]
            meta = copy.deepcopy(entries[0]["meta"])
            require_comparable(entries, meta, "panel option")
            if meta["family"] == "T16" and (meta.get("canonical_T16_priceable") is not True
                    or meta.get("weight_reference_definition") != "separate_values_and_row_scales_through_dot"):
                raise ValueError("The stack estimator refuses folded BF16 T16 inputs")
            key = (role, source)
            raw[key] = {field: np.array([[entry["sequence"][sample][field] for sample in samples]
                                         for entry in entries], dtype=np.float64) for field in fields}
            zero_fields[key] = set()
            if meta["activation_contract"] == "bf16_unquantized":
                for field in fields:
                    if field.startswith("E_A_sum"):
                        if np.any(raw[key][field] != 0):
                            raise ValueError("An unquantized activation contract has nonzero energy")
                        zero_fields[key].add(field)
            sequence, sums = {sample: {} for sample in samples}, {}
            for field in fields:
                numerator, denominator = float(raw[key][field].sum()), float(ref_panel[field].sum())
                if source == reference_source:
                    values = full_reference[field]
                elif field in zero_fields[key]:
                    values = np.zeros(len(samples), dtype=np.float64)
                elif denominator == 0:
                    raise ValueError("The stack ratio has an unidentifiable zero reference")
                else:
                    values = full_reference[field] * (numerator / denominator)
                sums[field] = float(values.sum())
                for index, sample in enumerate(samples):
                    sequence[sample][field] = float(values[index])
            meta.update(qname=name, expert=None, decision_unit_scope="routed_expert_stack",
                        estimated_panel_stack=source != reference_source,
                        panel_units=list(panel_units), full_reference_units=list(reference_units))
            if role == "candidate":
                meta["anchor_source"] = anchor_sources[source]
                candidates[(name, source)] = {"meta": meta, "samples": set(samples), "sequence": sequence, **sums}
            else:
                own_anchors[(name, source)] = {"meta": meta, "samples": set(samples), "sequence": sequence, **sums}
        contexts[name] = {"raw": raw, "reference_panel": ref_panel, "full_reference": full_reference,
                          "reference_source":reference_source, "zero_fields":zero_fields,
                          "anchor_sources": dict(anchor_sources), "panel_size": len(panel_units)}
        evidence.append({"decision_unit": name, "layer": group["layer"], "role": group["role"],
                         "panel_size": len(panel_units), "total_experts": len(reference_units),
                         "reference_source": reference_source, "reference_population_fully_measured": True,
                         "per_expert_rung_estimates_published": False})
    return candidates, own_anchors, contexts, evidence


def matched_energy_draws(context, source, parts, indices, seed):
    import numpy as np
    size = context["panel_size"]
    experts = np.random.default_rng(seed).integers(0, size, size=(len(indices), size))
    raw = context["raw"]
    ref = context["reference_panel"]
    full = context["full_reference"]
    anchor_source = context["anchor_sources"][source]
    results = {}
    for part in parts:
        for tag, key in ((part["candidate"], ("candidate", source)), (part["anchor"], ("anchor", anchor_source))):
            if tag in results:
                continue
            selected = key[1]
            field = tag.split(":")[1]
            output = np.empty(len(indices), dtype=np.float64)
            for draw, sequences in enumerate(indices):
                chosen = experts[draw]
                numerator = float(raw[key][field][chosen][:, sequences].sum())
                denominator = float(ref[field][chosen][:, sequences].sum())
                reference = float(full[field][sequences].sum())
                if selected == context["reference_source"]:
                    output[draw] = reference
                elif field in context["zero_fields"].get(key, ()):
                    output[draw] = 0.0
                elif denominator == 0:
                    output[draw] = np.nan
                else:
                    output[draw] = reference * numerator / denominator
            results[tag] = output
    return results
