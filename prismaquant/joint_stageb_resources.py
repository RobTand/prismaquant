"""Derive Stage B resource caps without changing the captured BF16 science."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

from .cost_stage_checkpoint import canonical_json_sha256, publish_new_bytes
from .tessera_joint_allocation import _read_bound, _bound_stat_fence
from .schemas import Contract

SCHEMA = "prismaquant.joint_stageb_resource_policy.v1"
GIB = 1024 ** 3
_VERIFIED = {}


_require = Contract(ValueError, "Stage B resources: ").require


def capture_policy(capture):
    """A policy's ``capture`` block, checked: the batch and the measured workspace.

    ``{"capture_batch": B, "workspace_reserve_bytes": {"bytes", "receipt",
    "basis"}}`` (PQ #1151). The workspace is the measured bytes one stored
    batch's pass holds on the device, and the receipt names the measurement
    (action key, path, sha256). The receipt is a reference for readers: the
    derivation does not reread it, and nothing compares it at run time.
    """
    from .joint_retained_window_plan import MEASURED_BUDGET_FIELDS, _measured_owner

    _require(isinstance(capture, dict) and set(capture) == {"capture_batch", *MEASURED_BUDGET_FIELDS},
             "capture block needs exactly capture_batch and the measured owners")
    batch = capture["capture_batch"]
    _require(type(batch) is int and batch >= 1, "capture_batch must be a positive integer")
    try:
        for name in MEASURED_BUDGET_FIELDS:
            _measured_owner(name, capture[name])
    except ValueError as exc:
        raise ValueError("Stage B resources: " + str(exc)) from exc
    return batch, {name: capture[name] for name in MEASURED_BUDGET_FIELDS}


def chain_policy(chain):
    """A policy's ``chain`` block, checked: the regime and each shape's owner.

    ``{"chain_regime": {"batch_size", "probe_fusion"}, "shapes": {name: owner}}``
    (PQ #1163). Each owner is ``{layers, bytes, device_resident_bytes,
    host_committed_bytes, regime, source, basis}``, with a ``receipt`` when
    ``source`` is ``measured``; ``declared`` owners carry none and say so.
    The derivation checks each owner (``joint_retained_window_plan._chain_owners``).
    """
    _require(isinstance(chain, dict) and set(chain) == {"chain_regime", "shapes"},
             "chain block needs exactly chain_regime and shapes")
    return copy.deepcopy(chain["chain_regime"]), copy.deepcopy(chain["shapes"])


def chain_owner_from_receipt(path, *, action_key, layer, layers, basis=None):
    """The measured chain owner a ``--stop-after-chain`` receipt states for ``layer``.

    Reads the receipt once, here. The roll of ``layer`` must be complete; its
    workspace and resident bytes come from the roll's own readings
    (:func:`prismaquant.stage_b_workspace_profile.chain_roll_owner_bytes`),
    and the receipt's sha256 goes into the owner. ``layers`` are the layers of
    the same shape the owner prices.
    """
    from .stage_b_workspace_profile import CHAIN_SCHEMA, chain_roll_owner_bytes

    path = Path(path).absolute()
    raw = path.read_bytes()
    profile = json.loads(raw)
    _require(profile.get("schema") == CHAIN_SCHEMA, f"{path} is not a chain workspace profile")
    rolls = [roll for roll in profile["rolls"] if roll["layer"] == int(layer)]
    _require(len(rolls) <= 1, f"{path} records more than one chain roll of layer {layer}")
    _require(bool(rolls), f"{path} measured no chain roll of layer {layer}")
    _require(rolls[0]["failure"] is None,
             f"{path} records an incomplete roll of layer {layer}; a lower bound prices nothing")
    try:
        owned = chain_roll_owner_bytes(rolls[0])
    except ValueError as exc:
        raise ValueError(f"Stage B resources: {path}: {exc}") from exc
    _require(owned["bytes"] > 0, f"{path} measured no workspace for layer {layer}")
    return {"layers": sorted(int(value) for value in layers), **owned,
            "regime": dict(profile["identity"]["chain_regime"]), "source": "measured",
            "receipt": {"action_key": action_key, "path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest()},
            "basis": basis or (
                f"layer {int(layer)}'s chain roll: workspace = max(allocated delta, reserved "
                "delta) over the reading after its admission released the cache; "
                "device_resident_bytes = the CUDA reservation that admission read; "
                "host_committed_bytes = the larger of the cgroup committed bytes at that "
                "admission and at the roll's cgroup peak")}


def derive_policy(inputs, *, host_bytes=28 * GIB, physical_bytes=100 * GIB, gpu_bytes=72 * GIB, candidate_files=None,
                  capture=None, chain=None):
    """Use the existing statistics planner and retained-window budget owner.

    ``capture`` (PQ #1151, see :func:`capture_policy`) replaces the original
    plan's declared ``workspace_reserve_bytes`` with a measured one and plans
    the capture pass at its ``capture_batch``. ``chain`` (PQ #1163, see
    :func:`chain_policy`) plans the chain phase per chain layer shape, against
    the policy's device ceiling ``gpu_bytes`` and its physical bound. Without
    either the policy is the one before #1151, byte for byte.
    """
    import torch
    from . import format_registry as fr
    from .aura_cost import _ZERO_COST_FORMATS
    from .joint_layer_quanta import qname_layer
    from .joint_retained_window_plan import (DECLARED_BUDGET_FIELDS, MEASURED_BUDGET_FIELDS,
        RetainedWindowBudget, derive_retained_window_budget, targets_from_statistics_plan)
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
        print(f"resource geometry: observing {len(paths)} candidate file sizes with {io_workers} bounded I/O readers", file=sys.stderr, flush=True)
        _drive_ordered_units(sorted(paths), observe_file, record_file, workers=io_workers)
    else:
        for pair in sorted(paths):
            record_file(pair, observe_file(pair))
    mode = "observed" if recorded is None else "reused sealed observations for"
    print(f"resource geometry: {mode} {len(file_rows)} files; deriving {len(by_layer)} layers", file=sys.stderr, flush=True)
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
    capture_batch = measured = None
    if capture is not None:
        capture_batch, measured = capture_policy(capture)
    chain_kwargs = {}
    if chain is not None:
        regime, shapes = chain_policy(chain)
        chain_kwargs = {"chain_regime": regime, "chain_workspace": shapes,
                        "chain_device_limit_bytes": gpu_bytes}
    declared = {key: getattr(old_budget, key) for key in DECLARED_BUDGET_FIELDS
                if measured is None or key not in MEASURED_BUDGET_FIELDS}
    declared["physical_limit_bytes"] = physical_bytes
    budget, derivation = derive_retained_window_budget(targets, declared=declared,
        source_bytes=retained["source_reserve_bytes"],
        prefetch_workers=plan["execution"]["operator_windows"]["prefetch_workers"],
        host_cap_bytes=host_bytes, footprint_scope="pwc_serialized_upper_bound",
        measured=measured, capture_batch=capture_batch, **chain_kwargs)
    policy = {"schema": SCHEMA, "inputs": copy.deepcopy(inputs),
        "limits": {"host_bytes": host_bytes, "physical_bytes": physical_bytes, "gpu_bytes": gpu_bytes},
        "budget": budget.as_dict(), "derivation": derivation,
        "candidate_files": file_rows, "candidate_files_sha256": canonical_json_sha256(file_rows, where="resource candidate files"),
        "semantics": "resource_geometry_only; original_qualification_and_BF16_capture_unchanged"}
    if capture is not None:
        policy["capture"] = copy.deepcopy(capture)
    if chain is not None:
        policy["chain"] = copy.deepcopy(chain)
    return policy


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
        candidate_files=None if verify_files else policy["candidate_files"],
        capture=policy.get("capture"), chain=policy.get("chain")),
        "independent resource derivation differs")
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
    if verified_limits is None:
        limits = verify_policy(bound)["limits"]
        _require(config["max_gpu_bytes"] == limits["gpu_bytes"], "device limit differs from policy")
    else:
        # The slice's limits were sealed under the policy it was prepared
        # with; a plan re-declared from a measurement names another device
        # ceiling. That is a run seal (PQ #1147): dev mode prints both and
        # applies the plan's ceiling; certified mode refuses as before.
        from .dev_mode import seal_check

        limits = dict(verified_limits)
        if not seal_check(
                "Stage B device limit", limits["gpu_bytes"], config["max_gpu_bytes"],
                where="head slice limits against the plan's max_gpu_bytes",
                refusal=lambda: ValueError(
                    "Stage B resources: device limit differs from policy")):
            limits["gpu_bytes"] = config["max_gpu_bytes"]
    observed = enforce_device_envelope("cuda", limits["gpu_bytes"], where="Stage B device envelope")
    return {**observed, "policy": dict(bound), "limits": dict(limits)}


def workspace_from_receipt(path, *, action_key):
    """The measured workspace owner a #1151 profile receipt states.

    Reads the receipt once, here, to take its measured per-batch bytes and its
    sha256; the policy then carries both, and nothing rereads the receipt.
    """
    from .stage_b_workspace_profile import SCHEMA as PROFILE_SCHEMA

    path = Path(path).absolute()
    raw = path.read_bytes()
    profile = json.loads(raw)
    _require(profile.get("schema") == PROFILE_SCHEMA, f"{path} is not a capture workspace profile")
    _require(profile.get("ladder_complete") is True,
             f"{path} records an incomplete ladder; a refused or stopped step prices nothing")
    per_batch = profile["measured"]["workspace_per_batch_bytes"]
    _require(type(per_batch) is int and per_batch > 0, f"{path} measured no workspace")
    return {"bytes": per_batch,
            "receipt": {"action_key": action_key, "path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest()},
            "basis": profile["measured"]["basis"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host-bytes", type=int, default=28 * GIB)
    parser.add_argument("--physical-bytes", type=int, default=100 * GIB)
    parser.add_argument("--gpu-bytes", type=int, default=72 * GIB)
    parser.add_argument("--capture-batch", type=int,
                        help="plan the capture pass at this batch (PQ #1151); needs --workspace-receipt")
    parser.add_argument("--workspace-receipt",
                        help="a stage_b_workspace_profile receipt whose measured per-batch "
                             "workspace replaces the declared one")
    parser.add_argument("--workspace-action-key", help="the PrismaBuild action that wrote the receipt")
    parser.add_argument("--chain", help="a JSON file holding the policy's chain block (PQ #1163): "
                                        "the chain regime and each chain layer shape's owner")
    args = parser.parse_args(argv)
    capture = None
    if (args.capture_batch, args.workspace_receipt, args.workspace_action_key).count(None) not in (0, 3):
        parser.error("--capture-batch, --workspace-receipt and --workspace-action-key go together")
    if args.capture_batch is not None:
        capture = {"capture_batch": args.capture_batch,
                   "workspace_reserve_bytes": workspace_from_receipt(
                       args.workspace_receipt, action_key=args.workspace_action_key)}
    chain = None if args.chain is None else json.loads(Path(args.chain).read_bytes())
    policy = derive_policy(json.loads(Path(args.inputs).read_bytes()), host_bytes=args.host_bytes,
                           physical_bytes=args.physical_bytes, gpu_bytes=args.gpu_bytes,
                           capture=capture, chain=chain)
    raw = (json.dumps(policy, sort_keys=True) + "\n").encode()
    _require(publish_new_bytes(Path(args.out), raw), "policy output already exists")
    print(json.dumps({"status": "resource_geometry_derived", "out": args.out,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "limits": policy["limits"], "budget": policy["budget"],
        "capture": policy["derivation"].get("capture"),
        "chain": policy["derivation"].get("chain"),
        "peak_planned_bytes": policy["derivation"]["peak_planned_bytes"],
        "windows_by_layer": policy["derivation"]["windows_by_layer"]}, sort_keys=True))


if __name__ == "__main__":
    main()
