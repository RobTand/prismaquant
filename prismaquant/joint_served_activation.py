"""Explicit A4-only executed-group pricing over immutable qualification evidence."""
from __future__ import annotations

import argparse
from collections import ChainMap
import copy
import json
import pickle
from pathlib import Path
from .schemas import Contract

FORMAT = "TESSERA_E2M1_K2_R896"
SCHEMA = "prismaquant.joint_served_activation_policy.v1"
FORMAT_MAXIMA_KEY = "__prismaquant_joint_format_maxima_v1__"
_VERIFIED = {}
_GROUP_INDEX = {}


_require = Contract(ValueError, "served activation policy: ").require


def _policy_bytes(bound, label, read_bound):
    if read_bound is None:
        from .tessera_joint_allocation import _read_bound
        return _read_bound(bound,label)
    import hashlib
    raw=read_bound(bound)
    _require(isinstance(raw,(bytes,bytearray)) and hashlib.sha256(raw).hexdigest()==bound["sha256"],
             "alternate pinned reader changed bound content: "+label)
    return raw


def derive_policy(original_prepared, *, read_bound=None):
    """Derive the existing runtime's scale reduction from authenticated full draw."""
    from .tessera_joint_allocation import _read_bound
    from .cost_stage_checkpoint import canonical_json_sha256
    from .model_profiles import detect_profile
    from .nvfp4_activation_contract import routed_executed_max_abs, LEGACY_INPUT_GLOBAL_SCALE_POLICY
    prepared = json.loads(_policy_bytes(original_prepared, "served-group original prepared", read_bound))
    _require(prepared.get("status") == "complete" and prepared["calibration_input"]["shape"] == [512, 512],
             "requires the original complete 512x512 qualification")
    cache = pickle.loads(_policy_bytes(prepared["production_cache"], "served-group original PWC", read_bound))
    census_binding = cache.metadata["inputs"]["census"]
    census = json.loads(_policy_bytes(census_binding, "served-group source census", read_bound))
    _require(set(census["unit_shapes"]) == set(prepared["formats_by_qname"]), "source roster differs")
    maxima = cache.activation_max_abs
    effective, declaration = routed_executed_max_abs(
        maxima, profile=detect_profile(census["model"]),
        expected_members=prepared["formats_by_qname"], policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY)
    _require(declaration is not None and declaration["roster_complete"], "requires complete routed groups")
    members = {name for row in declaration["groups"].values() for name in row["members"]}
    # Only newly added routed A4 candidates change pricing semantics. Existing
    # dense A4 and every original candidate retain their exact old activation.
    _require(all(FORMAT not in prepared["formats_by_qname"][name] for name in members),
             "policy may only price newly added routed A4 candidates")
    result = {"schema": SCHEMA, "format": FORMAT, "original_prepared": dict(original_prepared),
        "original_cache": dict(prepared["production_cache"]), "census": dict(census_binding),
        "source_model_identity_sha256": canonical_json_sha256(prepared["source_model_identity"], where="served-group source"),
        "calibration_input": copy.deepcopy(prepared["calibration_input"]),
        "qualification_max_abs": {name: maxima[name] for name in sorted(members)},
        "effective_max_abs": {name: effective[name] for name in sorted(members)},
        "executed_grouping": declaration,
        "semantics": "new_joint_A4_prices_use_executed_group_scales; original_qualification_unchanged"}
    return result


def verify_policy(bound, *, original_prepared=None, read_bound=None):
    from .tessera_joint_allocation import _read_bound, _bound_stat_fence
    key = (bound["path"], bound["sha256"], _bound_stat_fence(Path(bound["path"])))
    cached = _VERIFIED.get(key)
    if cached is not None and cached["fences"] == tuple(
            (b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in cached["dependencies"]):
        policy = cached["policy"]
    else:
        policy = json.loads(_policy_bytes(bound, "served activation policy", read_bound))
        _require(policy.get("schema") == SCHEMA and policy.get("format") == FORMAT, "unknown policy scope")
        dependencies = [dict(bound)] + [policy[k] for k in ("original_prepared", "original_cache", "census")]
        before = tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in dependencies)
        _require(before[0] == key, "policy changed while it was read")
        _require(policy == derive_policy(policy["original_prepared"], **({} if read_bound is None else {"read_bound":read_bound})), "group maxima or calibrated source evidence changed")
        _require(before == tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"])))
                                 for b in dependencies), "calibrated policy inputs changed during verification")
        _VERIFIED.clear()
        _VERIFIED[key] = {"policy": policy, "dependencies": dependencies,
            "fences": before}
    if original_prepared is not None:
        _require(policy["original_prepared"] == original_prepared, "original prepared binding differs")
    return policy


def format_activation_maxima(maxima, spec):
    """Resolve an explicitly format-scoped transient view; never mutate maxima."""
    overrides = (maxima or {}).get(FORMAT_MAXIMA_KEY, {})
    if not overrides:
        return maxima
    selected = overrides.get(spec.name)
    if selected is None:
        return maxima
    _require(spec.name == FORMAT, "only the declared A4 format can override activation maxima")
    return ChainMap(selected, maxima)


def activate_policy(cache, bound):
    """Attach pricing-only policy; leave persisted maxima and all receipts intact."""
    return _attach_policy(cache, bound, verify_policy(bound))


def activate_verified_policy(cache, bound, policy):
    """Attach a policy whose derivation was verified once, at prepare.

    The Stage B head slice (PQ #1010) re-derives the policy in the metadata
    producer and binds its file by digest; the quantum passes the bytes it
    read against that digest. The scope and every cache-facing check below
    still run here; only the campaign-wide re-derivation does not repeat.
    """
    _require(isinstance(policy, dict) and policy.get("schema") == SCHEMA
             and policy.get("format") == FORMAT, "unknown policy scope")
    return _attach_policy(cache, bound, policy)


def _attach_policy(cache, bound, policy):
    from .nvfp4_activation_contract import resolve_input_global_scale_policy
    _require(resolve_input_global_scale_policy() == policy["executed_grouping"]["input_global_scale_policy"],
             "active scale arithmetic differs from the explicit policy")
    for name, maximum in policy["qualification_max_abs"].items():
        _require(cache.activation_max_abs.get(name) == maximum, "cache qualified maximum differs for " + name)
        _require((name, FORMAT) in cache.weights, "new A4 candidate missing for " + name)
    cache._joint_served_activation = (dict(bound), policy)
    cache._joint_activation_maxima_view = {
        **cache.activation_max_abs, FORMAT_MAXIMA_KEY: {FORMAT: policy["effective_max_abs"]}}
    return policy


def joint_activation_maxima(cache):
    active = getattr(cache, "_joint_served_activation", None)
    if active is None:
        return cache.activation_max_abs
    return cache._joint_activation_maxima_view


def policy_group(policy, name, fmt):
    if fmt != FORMAT or name not in policy["effective_max_abs"]:
        return None
    cached = _GROUP_INDEX.get(id(policy))
    if cached is None or cached[0] is not policy:
        index = {member: (key, group) for key, group in policy["executed_grouping"]["groups"].items()
                 for member in group["members"]}
        _GROUP_INDEX.clear()
        _GROUP_INDEX[id(policy)] = (policy, index)
    return _GROUP_INDEX[id(policy)][1][name]


def operator_policy_record(bound, policy, name, fmt, qualification_activation):
    group = policy_group(policy, name, fmt)
    if group is None:
        return None
    return {"schema": SCHEMA, "policy": dict(bound), "group": group[0],
            "qualification_activation": copy.deepcopy(qualification_activation)}


def require_priced_activation(bound, policy, name, fmt, qualification_activation, operator):
    """The sole exception to old qualification==priced-activation equality."""
    group = policy_group(policy, name, fmt)
    if group is None:
        _require(operator.get("served_activation_policy") is None, "unscoped activation override")
        _require(operator["activation"] == qualification_activation, "original activation changed")
        return None
    expected = copy.deepcopy(qualification_activation)
    _require(expected["act_bits"] == 4 and expected["static_contract"]["measured_as_served"] is True,
             "qualification is not the served static A4 contract")
    _require(expected["activation_max_abs"] == policy["qualification_max_abs"][name],
             "qualified activation maximum differs")
    expected["activation_max_abs"] = group[1]["max_abs"]
    expected["input_global_scale"] = group[1]["input_global_scale"]
    _require(operator["activation"] == expected, "priced activation differs from the executed group")
    _require(operator.get("served_activation_policy") == operator_policy_record(
        bound, policy, name, fmt, qualification_activation), "operator policy/qualification binding differs")
    return group[1]["input_global_scale"]


def main(argv=None):
    from .cost_stage_checkpoint import publish_new_bytes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-prepared", required=True)
    parser.add_argument("--original-prepared-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = derive_policy({"path": args.original_prepared, "sha256": args.original_prepared_sha256})
    raw = (json.dumps(result, sort_keys=True) + "\n").encode()
    _require(publish_new_bytes(Path(args.out), raw), "output already exists")
    print(json.dumps({"status": "proposed", "path": args.out,
                      "members": len(result["effective_max_abs"]),
                      "groups": len(result["executed_grouping"]["groups"])}))


if __name__ == "__main__":
    main()
