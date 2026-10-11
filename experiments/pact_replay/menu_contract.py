"""Actual menu metadata and executed Linear views for PACT research prices."""
from __future__ import annotations
import json
from pathlib import Path

MLP_KINDS = {"routed", "shared", "dense"}
KINDS = MLP_KINDS | {"attention", "lm_head"}
CONTRACTS = {"T4": "e2m1_group16_ue4m3_static", "T8": "fp8_per_token_dynamic",
             "T16": "bf16_unquantized", "BF16": "bf16_unquantized"}


def extend_rows(renders, args):
    from g3_residency import read_file
    if args.added_inventory:
        inventory = json.loads(read_file(args.added_inventory))
        if inventory.get("schema") != "pact.added_unit_inventory.v2":
            raise ValueError("The added unit inventory schema differs")
        for unit in inventory["added_units"]:
            name = unit["qname"]
            if name in renders.rows:
                if unit["group"] != "shared" or renders.rows[name]["kind"] != "shared":
                    raise ValueError("The added inventory changes an existing unit")
                continue
            kind = unit["group"]
            layer = 45 if kind == "lm_head" else unit["structural_key"]["layer"]
            if kind not in KINDS or type(layer) is not int or not 0 <= layer <= 45:
                raise ValueError("The added unit has no executed layer or kind")
            shape = unit["shape"]
            if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("The added unit is not a Linear plane")
            renders.rows[name] = {"qname": name, "kind": kind, "role": unit["role"],
                "expert": None, "layer": layer, "checkpoint": unit["checkpoint"],
                "shape": shape, "pick": {"a8s": "SOURCE"}, "formats": {}}
            renders.references[name] = {"formats": {}}
    renders.extra_options = {}
    renders.menu_gaps = []
    if args.option_manifest:
        document = json.loads(read_file(args.option_manifest))
        if document.get("schema") != "pact.energy_options.v1":
            raise ValueError("The energy option schema differs")
        names = set()
        for entry in document["units"]:
            name = entry["qname"]
            if name in names or name not in renders.rows:
                raise ValueError("The option manifest duplicates or adds an unknown unit")
            names.add(name)
            options = entry["options"]
            if len({o["name"] for o in options}) != len(options):
                raise ValueError("The option manifest duplicates an option")
            for option in options:
                family = option["family"]
                if family not in CONTRACTS or option["contract"] != CONTRACTS[family]:
                    raise ValueError("The option family and activation contract differ")
                if family != "BF16":
                    if type(option["q256"]) is not int or option["q256"] <= 0:
                        raise ValueError("A rendered option needs its actual rate")
                    if "location" not in option and "wire" not in option:
                        raise ValueError("A rendered option needs its actual byte locator")
                if family == "T8" and renders.rows[name]["kind"] in {"attention","lm_head"} and option.get("tp_splits") not in (1,2):
                    raise ValueError("A new T8 option needs an explicit activation TP contract")
                if family == "T4":
                    if option.get("body") not in {"TCQ", "WINDOW"} or not option.get("scale"):
                        raise ValueError("A T4 option needs its actual body and served scale")
                option["unit_row"] = {key: renders.rows[name][key] for key in ("qname", "kind", "role", "expert", "layer")}
                option["weight_reference_definition"] = ("separate_values_and_row_scales_through_dot"
                    if family == "T16" else "source_bf16" if family == "BF16" else "decoded_bf16")
                option["canonical_T16_priceable"] = family == "T16"
            renders.extra_options[name] = options
        renders.menu_gaps.extend(document.get("gaps", []))
    for row in renders.rows.values():
        if row["kind"] not in MLP_KINDS:
            families = {o["family"] for o in renders.extra_options.get(row["qname"], [])}
            for family in ("T8", "T16"):
                if family not in families:
                    renders.menu_gaps.append({"qname": row["qname"], "family": family,
                        "q256": 2048 if family == "T16" else None,
                        "reason": "No actual research render in the supplied option manifest"})


def all_options(renders, row, result):
    primary = next(option for option in result if option["name"] == "A8S")
    for name, option in row["formats"].items():
        if option.get("wire") == primary.get("location") or name.startswith("v1::") or name == "EXL3":
            continue
        contract = option.get("contract")
        if contract not in {"fp8_per_token_dynamic", "bf16_unquantized"}:
            continue
        rate = int(name.rsplit("_R", 1)[1])
        if rate % 256:
            continue
        family = "T8" if contract == "fp8_per_token_dynamic" else "T16"
        result.append({"name": name, "family": family, "q256": int(name.rsplit("_R", 1)[1]),
            "contract": contract, "location": option["wire"], "rendered_shape": option["rendered_shape"],
            "unit_row": {key: row[key] for key in ("qname", "kind", "role", "expert", "layer")},
            "weight_reference_definition": "decoded_bf16"})
    result.extend(dict(option) for option in renders.extra_options.get(row["qname"], []))
    for option in result:
        if option["family"] == "T16":
            option["canonical_T16_priceable"] = True
            option["weight_reference_definition"] = "separate_values_and_row_scales_through_dot"
    if len({option["name"] for option in result}) != len(result):
        raise ValueError("The menu repeats an actual option")
    t8 = [option for option in result if option["family"] == "T8"]
    # Only priceable T8 options need the common R1024 reference. A chord-check-only
    # diagnostic is never priced (reduce_prices.t8_reference_sources skips it), so a
    # menu of only such diagnostics keeps them unpriced, with no reference.
    priceable = [option for option in t8 if not option.get("quality_chord_check_only")]
    if t8:
        references = {option["anchor_source"] for option in t8 if option.get("anchor_source")}
        if len(references) > 1:
            raise ValueError("The T8 menu has inconsistent common references")
        anchors = [option for option in t8 if option["q256"] == 1024
                   and not option.get("quality_chord_check_only") and not option.get("passthrough")]
        if references:
            reference = next(iter(references))
        elif any(option["name"] == "A8S" for option in anchors):
            reference = "A8S"
        elif len(anchors) == 1:
            reference = anchors[0]["name"]
        elif priceable:
            raise ValueError("The legacy T8 menu needs one actual R1024 reference")
        else:
            reference = None
        if reference is not None:
            for option in t8:
                option["anchor_source"] = reference
    return result


def executed_module(model, profile, row):
    """Resolve the actual module through the profile's name projection."""
    import torch
    matches = [module for name, module in model.named_modules()
               if name == row["qname"] or profile.live_to_recipe_name(name) == row["qname"]]
    if len(matches) != 1 or not isinstance(matches[0], torch.nn.Linear):
        raise ValueError("The unit has no unique executed Linear: " + row["qname"])
    return matches[0]


def selected_options(renders, row):
    options = renders.options(row)
    selected = renders.args.weight_source or ["A8S", "A4-q896", "EXL3"]
    found = {option["name"] for option in options}
    if set(selected) - found:
        raise ValueError("The selected unit lacks an actual requested render: " + row["qname"])
    return [option for option in options if option["name"] in selected]


def unit_weight(runner, row):
    if row["kind"] in MLP_KINDS:
        from g3_lib import unit_view
        return unit_view(runner.layers[row["layer"]], row)
    return executed_module(runner.model, runner.profile, row).weight.detach()
