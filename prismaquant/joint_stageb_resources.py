"""Derive Stage B resource caps without changing the captured BF16 science."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from pathlib import Path

from .cost_stage_checkpoint import canonical_json_sha256, publish_new_bytes
from .tessera_joint_allocation import _read_bound, _bound_stat_fence

SCHEMA = "prismaquant.joint_stageb_resource_policy.v1"
GIB = 1024 ** 3
_VERIFIED = {}


def _require(ok, message):
    if not ok:
        raise ValueError("Stage B resources: " + message)


#: Where a quantum holds its cotangent plane: in host memory, or in the
#: sealed local cotangent scratch (PRISMAQUANT_STAGE_B_COTANGENT_ROOT).
COTANGENT_PLANE_PLACEMENTS = ("host", "scratch")


def _cotangent_plane(value):
    """Validate a declared cotangent plane owner, or pass ``None`` through."""
    if value is None:
        return None
    _require(isinstance(value, dict) and set(value) == {"bytes", "placement"},
             "cotangent plane declares exactly bytes and placement")
    _require(type(value["bytes"]) is int and value["bytes"] > 0,
             "cotangent plane bytes must be a positive integer")
    _require(value["placement"] in COTANGENT_PLANE_PLACEMENTS,
             "cotangent plane placement must be host or scratch")
    return {"bytes": value["bytes"], "placement": value["placement"]}


def derive_policy(inputs, *, host_bytes=28 * GIB, physical_bytes=100 * GIB, gpu_bytes=72 * GIB,
                  candidate_files=None, cotangent_plane=None):
    """Use the existing statistics planner and retained-window budget owner.

    ``cotangent_plane`` (PQ #1141) declares the quantum's cotangent plane as an
    owner: ``{"bytes": N, "placement": "host" | "scratch"}``. A host plane is
    resident from checkpoint-load to the end of the quantum, so it is charged
    to both caps the planner holds renders against: the container's host cap
    and the aggregate physical limit. A scratch plane lives on local disk and
    charges neither. The declaration is recorded in ``limits`` and in the
    derivation only when given, so a policy derived without one is unchanged.
    """
    plane = _cotangent_plane(cotangent_plane)
    import torch
    from . import format_registry as fr
    from .aura_cost import _ZERO_COST_FORMATS
    from .joint_layer_quanta import qname_layer
    from .joint_retained_window_plan import (DECLARED_BUDGET_FIELDS, RetainedWindowBudget,
        derive_retained_window_budget, targets_from_statistics_plan)
    from .joint_statistics_plan import plan_joint_statistics_target_windows
    from .joint_served_activation import FORMAT, FORMAT_MAXIMA_KEY, verify_policy as verify_activation

    _require(set(inputs) == {"original_plan", "original_prepared", "candidate_overlay", "served_activation_policy"},
             "four exact source/catalog/activation bindings required")
    plan = json.loads(_read_bound(inputs["original_plan"], "resource original plan"))
    prepared = json.loads(_read_bound(inputs["original_prepared"], "resource original preparation"))
    catalog = json.loads(_read_bound(inputs["candidate_overlay"], "resource proposed candidate catalog"))
    _require(prepared["plan_sha256"] == inputs["original_plan"]["sha256"], "original preparation/plan differ")
    _require(catalog["old_prepared"] == inputs["original_prepared"], "catalog original preparation differs")
    activation = verify_activation(inputs["served_activation_policy"], original_prepared=inputs["original_prepared"])
    cache = pickle.loads(_read_bound(prepared["production_cache"], "resource qualified original PWC"))
    retained = plan["execution"]["retained_operator_windows"]
    old_budget = RetainedWindowBudget.from_dict(retained["budget"])
    _require(type(host_bytes) is int and 0 < host_bytes <= 28 * GIB, "host cap exceeds approved 28 GiB")
    _require(type(physical_bytes) is int and 0 < physical_bytes <= min(100 * GIB, old_budget.physical_limit_bytes),
             "aggregate cap exceeds approved or original bound")
    _require(type(gpu_bytes) is int and 0 < gpu_bytes <= min(72 * GIB, plan["max_gpu_bytes"]),
             "device cap exceeds approved or original bound")
    _require(host_bytes + gpu_bytes <= physical_bytes, "host/device reservations exceed aggregate cap")
    formats = {n: [f for f in fs if f not in _ZERO_COST_FORMATS] for n, fs in prepared["formats_by_qname"].items()}
    paths = {pair: str(Path(cache._path_for_value(path)).absolute()) for pair, path in cache.weights.items()}
    expected = {(n, FORMAT) for n in formats if FORMAT not in formats[n]}
    added = {(row["qname"], row["format"]): row for row in catalog["cells"]}
    _require(set(added) == expected and len(added) == len(catalog["cells"]), "added candidate roster differs")
    for pair, row in added.items():
        path = Path(row["render"])
        paths[pair] = str(path)
        formats[pair[0]].append(FORMAT)
    by_layer = {}
    for name in formats:
        layer = qname_layer(name)
        _require(layer is not None, "candidate has no decoder layer")
        by_layer.setdefault(layer, []).append(name)
    # Geometry comes from already qualified original source tensors. No new
    # render is described as qualified, and no render payload is read here.
    shapes = {}
    for (name, _), cell in cache.metadata["verified_cells"].items():
        shape = tuple(cell["source_weight"]["shape"])
        _require(shapes.setdefault(name, shape) == shape, "source geometry differs across original cells")
    file_rows, key_costs = [], {}
    recorded = None
    if candidate_files is not None:
        recorded = {tuple(row["member"]): row for row in candidate_files}
        _require(len(recorded) == len(candidate_files) and set(recorded) == set(paths),
                 "sealed resource file roster differs")
    from .cost_stage_checkpoint import _drive_ordered_units
    def observe_file(pair):
        path = paths[pair]
        if recorded is None:
            fence = _bound_stat_fence(Path(path))
        else:
            row = recorded[pair]
            _require(row["path"] == path, "sealed resource candidate path differs")
            fence = tuple(row["fence"])
            _require(len(fence) == 6 and all(type(n) is int for n in fence), "invalid sealed file fence")
        _require(fence[3] > 0, "empty candidate file")
        if pair in added:
            _require({"inode": fence[2], "bytes": fence[3], "mtime_ns": fence[4], "ctime_ns": fence[5]}
                     == added[pair]["render_stat"], "retained candidate file changed: " + path)
        return path, fence
    def record_file(pair, observed):
        path, fence = observed
        file_rows.append({"member": list(pair), "path": path, "fence": list(fence)})
        key_costs[pair] = {"incoming_storage_bytes": fence[3], "serialized_bytes": fence[3]}
    io_workers = min(8, max(1, 2 * len(os.sched_getaffinity(0))))
    if recorded is None:
        print(f"resource geometry: observing {len(paths)} candidate file sizes with {io_workers} bounded I/O readers", flush=True)
        _drive_ordered_units(sorted(paths), observe_file, record_file, workers=io_workers)
    else:
        for pair in sorted(paths):
            record_file(pair, observe_file(pair))
    mode = "observed" if recorded is None else "reused sealed observations for"
    print(f"resource geometry: {mode} {len(file_rows)} files; deriving {len(by_layer)} layers", flush=True)
    maxima = {**cache.activation_max_abs, FORMAT_MAXIMA_KEY: {FORMAT: activation["effective_max_abs"]}}
    targets = {}
    for layer, names in sorted(by_layer.items()):
        names.sort()
        modules = {n: torch.nn.Linear(shapes[n][1], shapes[n][0], bias=False, device="meta", dtype=torch.bfloat16)
                   for n in names}
        specs = {n: {fmt: fr.get_format(fmt) for fmt in formats[n]} for n in names}
        statistics = plan_joint_statistics_target_windows(modules, specs,
            max_statistics_bytes=physical_bytes, activation_max_abs=maxima)
        keys = {n: tuple((n, fmt) for fmt in formats[n]) for n in names}
        layer_costs = {key: key_costs[key] for values in keys.values() for key in values}
        targets[layer] = targets_from_statistics_plan(statistics, keys, layer_costs)
    host_plane = plane["bytes"] if plane is not None and plane["placement"] == "host" else 0
    _require(host_plane < host_bytes, "cotangent plane on the host exceeds the host cap")
    declared = {key: getattr(old_budget, key) for key in DECLARED_BUDGET_FIELDS}
    declared["physical_limit_bytes"] = physical_bytes - host_plane
    budget, derivation = derive_retained_window_budget(targets, declared=declared,
        source_bytes=retained["source_reserve_bytes"],
        prefetch_workers=plan["execution"]["operator_windows"]["prefetch_workers"],
        host_cap_bytes=host_bytes - host_plane, footprint_scope="pwc_serialized_upper_bound")
    limits = {"host_bytes": host_bytes, "physical_bytes": physical_bytes, "gpu_bytes": gpu_bytes}
    if plane is not None:
        limits["cotangent_plane"] = plane
        derivation["cotangent_plane"] = {
            **plane, "host_bytes_charged": host_plane,
            "peak_planned_bytes_with_plane": derivation["peak_planned_bytes"] + host_plane,
            "basis": ("a host plane is resident for the whole quantum, so the host cap "
                      "and the physical limit the windows are packed against are both "
                      "reduced by it; a scratch plane is on local disk")}
    return {"schema": SCHEMA, "inputs": copy.deepcopy(inputs),
        "limits": limits,
        "budget": budget.as_dict(), "derivation": derivation,
        "candidate_files": file_rows, "candidate_files_sha256": canonical_json_sha256(file_rows, where="resource candidate files"),
        "semantics": "resource_geometry_only; original_qualification_and_BF16_capture_unchanged"}


def verify_policy(bound, *, verify_files=False):
    key = (bound["path"], bound["sha256"], _bound_stat_fence(Path(bound["path"])))
    cached = _VERIFIED.get(key)
    if not verify_files and cached is not None and cached["fences"] == tuple(
            (b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in cached["dependencies"]):
        return cached["policy"]
    policy = json.loads(_read_bound(bound, "Stage B resource policy"))
    _require(policy.get("schema") == SCHEMA, "unknown resource policy")
    dependencies = [dict(bound), *policy["inputs"].values()]
    prepared = json.loads(_read_bound(policy["inputs"]["original_prepared"], "resource original preparation"))
    dependencies.append(prepared["production_cache"])
    activation = json.loads(_read_bound(policy["inputs"]["served_activation_policy"], "resource served activation"))
    dependencies.append(activation["census"])
    before = tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in dependencies)
    _require(before[0] == key, "resource policy changed while read")
    _require(policy == derive_policy(policy["inputs"], **policy["limits"],
        candidate_files=None if verify_files else policy["candidate_files"]), "independent resource derivation differs")
    _require(before == tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in dependencies),
             "resource metadata changed during derivation")
    # Policy creation observes actual sizes; workers independently rederive
    # the planner from those sealed observations. Optional verify_files repeats
    # the file audit when explicitly needed. Consumers use sealed upper bounds; executable readsets and PWC's existing
    # digest/size checks independently refuse changed render bytes before load.
    _VERIFIED.clear()
    _VERIFIED[key] = {"policy": policy, "dependencies": dependencies, "fences": before}
    return policy


def require_plan_resources(old_plan, new_plan, old_plan_binding, old_prepared_binding):
    bound = new_plan.get("stage_b_resource_policy")
    if bound is None:
        _require(new_plan["execution"] == old_plan["execution"], "execution changed without resource proof")
        _require(new_plan.get("max_gpu_bytes") == old_plan.get("max_gpu_bytes"), "device limit changed without resource proof")
        return None
    policy = verify_policy(bound)
    _require(policy["inputs"]["original_plan"] == old_plan_binding, "resource original plan differs")
    _require(policy["inputs"]["original_prepared"] == old_prepared_binding, "resource original preparation differs")
    _require(policy["inputs"]["candidate_overlay"] == new_plan["inputs"].get("candidate_overlay"), "resource candidate catalog differs")
    _require(policy["inputs"]["served_activation_policy"] == new_plan.get("served_activation_policy"), "resource activation policy differs")
    expected = copy.deepcopy(old_plan["execution"])
    expected["retained_operator_windows"]["budget"] = policy["budget"]
    _require(new_plan["execution"] == expected, "non-resource execution field changed")
    _require(new_plan["max_gpu_bytes"] == policy["limits"]["gpu_bytes"], "resource device ceiling differs")
    return policy


def enforce_device_policy(config, *, verified_limits=None):
    """Apply the policy's actual Torch ceiling before GPU preparation.

    ``verified_limits`` are the limits a Stage B head slice carries (PQ
    #1010): the metadata producer re-derived the policy once and the slice
    is bound to the record by digest, so the quantum applies them without
    re-reading the policy's campaign inputs.
    """
    bound = config.get("stage_b_resource_policy")
    if bound is None:
        _require(verified_limits is None, "verified limits without a resource policy")
        return None
    from .memory_management import enforce_device_envelope
    limits = (verify_policy(bound)["limits"] if verified_limits is None
              else verified_limits)
    _require(config["max_gpu_bytes"] == limits["gpu_bytes"], "device limit differs from policy")
    observed = enforce_device_envelope("cuda", limits["gpu_bytes"], where="Stage B device envelope")
    return {**observed, "policy": dict(bound), "limits": dict(limits)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cotangent-plane-bytes", type=int, default=None,
                        help="the quantum's cotangent plane bytes (PQ #1141)")
    parser.add_argument("--cotangent-plane-placement", choices=COTANGENT_PLANE_PLACEMENTS,
                        default=None, help="where the quantum holds that plane")
    args = parser.parse_args(argv)
    if (args.cotangent_plane_bytes is None) != (args.cotangent_plane_placement is None):
        parser.error("--cotangent-plane-bytes and --cotangent-plane-placement go together")
    plane = (None if args.cotangent_plane_bytes is None else
             {"bytes": args.cotangent_plane_bytes, "placement": args.cotangent_plane_placement})
    policy = derive_policy(json.loads(Path(args.inputs).read_bytes()), cotangent_plane=plane)
    raw = (json.dumps(policy, sort_keys=True) + "\n").encode()
    _require(publish_new_bytes(Path(args.out), raw), "policy output already exists")
    print(json.dumps({"status": "resource_geometry_derived", "out": args.out,
        "limits": policy["limits"], "budget": policy["budget"],
        "windows_by_layer": policy["derivation"]["windows_by_layer"]}, sort_keys=True))


if __name__ == "__main__":
    main()
