"""MoE aggregation and prune-manifest helpers for the allocator."""
from __future__ import annotations

import math
import re

from . import format_registry as fr
from .allocator_candidates import _passthrough_source_ok
from .allocator_solver import Candidate, _shape_from_stats, predicted_dloss


def _moe_group_and_projection(name: str) -> tuple[str, str] | None:
    """Return ``(experts_group_path, projection_suffix)`` for expert leaves."""
    m = re.search(r"^(.+\.experts)\.\d+\.(.+)$", name)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"^(.+\.experts)\.(gate_up_proj|down_proj)\.\d+$", name)
    if m:
        return m.group(1), m.group(2)
    return None


def _aggregate_candidate_memory_bits(
    members: list[str],
    spec: fr.FormatSpec,
    stats: dict,
) -> tuple[int, float]:
    total_params = sum(stats[m]["n_params"] for m in members)
    total_bytes = 0
    for m in members:
        shape = _shape_from_stats(stats[m])
        total_bytes += spec.memory_bytes_for_shape(shape)
    bits_per_param = 8.0 * total_bytes / max(total_params, 1)
    return total_bytes, bits_per_param


def _expert_ids_in_group(
    members: list[str],
    expert_info: dict[str, tuple[str, str]],
) -> tuple[str | None, dict[int, str]]:
    """Return router qname and ``{expert_id: member_qname}`` for a group."""
    router_qname: str | None = None
    by_eid: dict[int, str] = {}
    for m_ in members:
        info = expert_info.get(m_)
        if info is None:
            continue
        rq, eid_str = info
        if router_qname is None:
            router_qname = rq
        try:
            eid = int(eid_str)
        except (TypeError, ValueError):
            continue
        by_eid[eid] = m_
    return router_qname, by_eid


def _prune_cost_per_expert(
    saliency: float,
    h_trace: float,
    n_params: int,
    alpha: float,
) -> float:
    """Predicted loss for dropping one expert from its REAP saliency."""
    del h_trace, n_params
    if saliency <= 0:
        return 0.0
    return float(alpha) * float(saliency)


def _saliency_has_eid(saliency_map: dict, eid: int) -> bool:
    return eid in saliency_map or str(eid) in saliency_map


def _saliency_lookup(
    saliency_map: dict,
    eid: int,
    default: float = 0.0,
) -> float:
    if eid in saliency_map:
        return float(saliency_map[eid])
    s_eid = str(eid)
    if s_eid in saliency_map:
        return float(saliency_map[s_eid])
    return float(default)


def _saliency_complete_for_eids(saliency_map: dict, eids) -> bool:
    expected = [int(e) for e in eids]
    return bool(expected) and all(_saliency_has_eid(saliency_map, e) for e in expected)


def aggregate_moe_candidates(
    stats: dict, costs: dict, formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    granularity: str = "projection",
    calibrated_gains: dict[str, float] | None = None,
    expert_saliency: dict[str, dict[int, float]] | None = None,
    expert_info: dict[str, tuple[str, str]] | None = None,
    prune_ratios: tuple[float, ...] = (),
    prune_alpha: float = 0.5,
    source_manifest: dict[str, str] | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate per-expert Linears into per-layer MoE super-candidates."""
    gains = calibrated_gains or {}
    expert_leaves: dict[tuple[str, str], list[str]] = {}
    non_expert_names: list[str] = []
    for name in stats:
        grp_proj = _moe_group_and_projection(name)
        if grp_proj is None:
            non_expert_names.append(name)
            continue
        grp, projection = grp_proj
        if granularity == "layer":
            expert_leaves.setdefault((grp, "__all__"), []).append(name)
        else:
            expert_leaves.setdefault((grp, projection), []).append(name)

    stats_ext = {n: stats[n] for n in non_expert_names}
    costs_ext = {n: costs.get(n, {}) for n in non_expert_names}
    candidates_ext = {n: candidates[n] for n in non_expert_names
                      if n in candidates}

    for (grp, projection), members in expert_leaves.items():
        n_params = sum(stats[m_]["n_params"] for m_ in members)
        d_out = int(stats[members[0]]["out_features"])
        d_in = int(stats[members[0]]["in_features"])
        sum_h = sum(stats[m_]["h_trace"] for m_ in members)
        super_name = f"{grp}.__fused__.{projection}"

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m_].get("h_trace_raw", 0.0) for m_ in members),
            "h_w2_sum": sum(stats[m_].get("h_w2_sum", 0.0) for m_ in members),
            "w_max_abs": max(stats[m_]["w_max_abs"] for m_ in members),
            "w_norm_sq": sum(stats[m_]["w_norm_sq"] for m_ in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m_].get("n_tokens_seen", 0) for m_ in members),
            "route_prob": None,
            "router_path": None,
            "expert_id": None,
            "_fused_members": members,
            "_memory_bytes_by_format": {},
        }

        def _member_cost(m_: str, fmt: str) -> dict | None:
            m_costs = costs.get(m_, {})
            for alias in fr.aliases_for(fmt):
                entry = m_costs.get(alias)
                if entry is not None and "error" not in entry:
                    return entry
            return None

        super_cost = {}
        for spec in formats:
            available_members = [
                m_ for m_ in members if _member_cost(m_, spec.name) is not None
            ]
            if not available_members:
                super_cost[spec.name] = {"error": "partial"}
                continue
            sum_weight_mse_x_params = 0.0
            sum_params_avail = 0
            for m_ in available_members:
                p_i = stats[m_]["n_params"]
                sum_weight_mse_x_params += _member_cost(m_, spec.name)["weight_mse"] * p_i
                sum_params_avail += p_i
            mean_weight_mse = sum_weight_mse_x_params / max(sum_params_avail, 1)
            mean_output_mse = sum(
                _member_cost(m_, spec.name)["output_mse"]
                for m_ in available_members
            ) / len(available_members)

            sum_pred = 0.0
            for m_ in members:
                c = _member_cost(m_, spec.name)
                if c is None:
                    c = {"weight_mse": mean_weight_mse,
                         "output_mse": mean_output_mse}
                if "predicted_dloss" in c:
                    sum_pred += float(c["predicted_dloss"])
                else:
                    h_i = stats[m_]["h_trace"]
                    sum_pred += 0.5 * h_i * float(c["weight_mse"])

            if sum_h > 0:
                effective_mse = sum_pred / (0.5 * sum_h)
            else:
                effective_mse = 0.0

            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "output_mse": mean_output_mse,
                "rel_output_mse": mean_output_mse,
                "predicted_dloss": sum_pred,
            }
        costs_ext[super_name] = super_cost

        router_qname_for_grp, eid_to_member = (
            _expert_ids_in_group(members, expert_info or {})
        )
        saliency_map = (
            (expert_saliency or {}).get(router_qname_for_grp, {})
            if router_qname_for_grp is not None
            else {}
        )
        saliency_complete = _saliency_complete_for_eids(
            saliency_map, eid_to_member.keys(),
        )
        if prune_ratios and saliency_map and saliency_complete:
            effective_prune_ratios = tuple(
                sorted({0.0, *(r for r in prune_ratios if r > 0.0)})
            )
        else:
            effective_prune_ratios = (0.0,)

        prune_dloss_by_eid: dict[int, float] = {}
        for eid, member in eid_to_member.items():
            s_j = _saliency_lookup(saliency_map, eid, 0.0)
            h_j = float(stats[member].get("h_trace", 0.0))
            np_j = int(stats[member].get("n_params", 0) or 0)
            prune_dloss_by_eid[eid] = _prune_cost_per_expert(
                s_j, h_j, np_j, prune_alpha,
            )

        drop_order = sorted(prune_dloss_by_eid, key=prune_dloss_by_eid.get)
        num_experts_total = len(drop_order)

        member_format_sets = [
            {c.fmt for c in candidates.get(m_, [])} for m_ in members
            if candidates.get(m_)
        ]
        if member_format_sets:
            allowed_member_formats = set.intersection(*member_format_sets)
        else:
            allowed_member_formats = {spec.name for spec in formats}

        cands = []
        member_source_kinds: set[str | None] = set()
        if source_manifest is not None:
            for m_ in members:
                member_source_kinds.add(source_manifest.get(m_))
        for spec in formats:
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            if member_source_kinds and not all(
                _passthrough_source_ok(spec.name, sk)
                for sk in member_source_kinds
            ):
                continue
            if spec.name not in allowed_member_formats:
                continue
            gain = float(gains.get(spec.name, 1.0))
            per_member_dloss: dict[str, float] = {}
            for m_ in members:
                c = _member_cost(m_, spec.name)
                if c is None:
                    fb_weight_mse = entry["weight_mse"]
                    per_member_dloss[m_] = (
                        0.5 * float(stats[m_]["h_trace"]) * fb_weight_mse * gain
                    )
                else:
                    if "predicted_dloss" in c:
                        per_member_dloss[m_] = float(c["predicted_dloss"]) * gain
                    else:
                        per_member_dloss[m_] = (
                            0.5 * float(stats[m_]["h_trace"])
                            * float(c["weight_mse"]) * gain
                        )

            base_memory_bytes, base_bits_per_param = _aggregate_candidate_memory_bits(
                members, spec, stats
            )
            if spec.name not in stats_ext[super_name]["_memory_bytes_by_format"]:
                stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = base_memory_bytes

            for ratio in effective_prune_ratios:
                if ratio <= 0.0 or num_experts_total == 0:
                    predicted = predicted_dloss(sum_h, entry["weight_mse"], gain=gain)
                    cands.append(Candidate(
                        fmt=spec.name,
                        bits_per_param=base_bits_per_param,
                        memory_bytes=base_memory_bytes,
                        predicted_dloss=max(predicted, 0.0),
                        pruned_expert_ids=(),
                    ))
                    continue

                n_drop = min(num_experts_total, int(math.floor(num_experts_total * ratio)))
                if n_drop == 0:
                    continue
                dropped_eids = tuple(sorted(drop_order[:n_drop]))
                kept_eids = set(drop_order[n_drop:])

                pred_total = 0.0
                for m_, d in per_member_dloss.items():
                    info = (expert_info or {}).get(m_)
                    if info is None:
                        pred_total += d
                        continue
                    try:
                        eid = int(info[1])
                    except (TypeError, ValueError):
                        pred_total += d
                        continue
                    if eid in kept_eids:
                        pred_total += d
                for eid in dropped_eids:
                    pred_total += prune_dloss_by_eid.get(eid, 0.0)

                kept_frac = 1.0 - float(n_drop) / float(num_experts_total)
                prune_memory = int(base_memory_bytes * kept_frac)
                prune_bits_per_param = base_bits_per_param * kept_frac
                cands.append(Candidate(
                    fmt=spec.name,
                    bits_per_param=prune_bits_per_param,
                    memory_bytes=prune_memory,
                    predicted_dloss=max(pred_total, 0.0),
                    pruned_expert_ids=dropped_eids,
                ))

        if cands:
            candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


_PACKED_EXPERTS_PROJ_RE = re.compile(
    r"^(?P<parent>.+)\.experts\.(?P<proj>[A-Za-z_][A-Za-z0-9_]*)$"
)


def _packed_entry_router_qname(name: str) -> str | None:
    m = _PACKED_EXPERTS_PROJ_RE.match(name)
    if m is None:
        return None
    moe_block = m.group("parent")
    if not moe_block:
        return None
    return f"{moe_block}.gate"


def apply_global_prune_ratio(
    candidates: dict[str, list[Candidate]],
    stats: dict,
    expert_saliency: dict[str, dict[int, float]],
    global_ratio: float,
    prune_alpha: float = 1.0,
) -> int:
    """Rewrite packed-entry candidates at one global prune ratio."""
    R = float(global_ratio)
    if R <= 0.0 or not expert_saliency:
        return 0

    n_rewritten = 0
    for name, cs in list(candidates.items()):
        router_qname = _packed_entry_router_qname(name)
        if router_qname is None:
            continue
        saliency_map = expert_saliency.get(router_qname)
        if not saliency_map:
            continue
        s = stats.get(name, {})
        E = int(s.get("num_experts") or len(saliency_map))
        if E <= 0:
            continue
        if not _saliency_complete_for_eids(saliency_map, range(E)):
            continue
        n_drop = int(math.floor(E * R))
        if n_drop <= 0:
            continue
        prune_dloss_by_eid = {
            eid: prune_alpha * _saliency_lookup(saliency_map, eid, 0.0)
            for eid in range(E)
        }
        drop_order = sorted(
            range(E), key=lambda e: (prune_dloss_by_eid[e], e)
        )
        dropped = tuple(sorted(drop_order[:n_drop]))
        prune_dloss_total = sum(prune_dloss_by_eid[e] for e in dropped)
        kept_frac = 1.0 - float(n_drop) / float(E)

        n_params_total = int(s.get("n_params", 0) or 0)
        variants: list[Candidate] = []
        mem_by_fmt: dict[str, int] = {}
        for baseline_c in cs:
            pruned_mem = int(
                baseline_c.bits_per_param * n_params_total * kept_frac / 8.0
            )
            variants.append(Candidate(
                fmt=baseline_c.fmt,
                bits_per_param=baseline_c.bits_per_param * kept_frac,
                memory_bytes=pruned_mem,
                predicted_dloss=max(
                    baseline_c.predicted_dloss * kept_frac + prune_dloss_total,
                    0.0,
                ),
                pruned_expert_ids=dropped,
            ))
            mem_by_fmt[baseline_c.fmt] = pruned_mem
        candidates[name] = variants
        s["_memory_bytes_by_format"] = mem_by_fmt
        n_rewritten += 1
    return n_rewritten


def compute_max_prune_ratio(
    stats: dict,
    top_k: int,
) -> float:
    """Largest prune ratio that leaves at least ``top_k`` experts kept."""
    min_kept_ratio = 1.0
    for name, s in stats.items():
        if _packed_entry_router_qname(name) is None:
            continue
        E = int(s.get("num_experts", 0) or 0)
        if E <= 0:
            continue
        if E < top_k:
            raise ValueError(
                f"stat {name!r} has E={E} experts but top_k={top_k}; "
                "cannot prune without breaking routing."
            )
        max_r_here = (E - top_k) / E
        if max_r_here < min_kept_ratio:
            min_kept_ratio = max_r_here
    return min_kept_ratio


def expand_moe_assignment(
    assignment: dict[str, str],
    stats_ext: dict,
    pruned_map: dict[str, tuple[int, ...]] | None = None,
    expert_info: dict[str, tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Expand MoE super-Linear assignments back to individual experts."""
    out = {}
    pm = pruned_map or {}
    einfo = expert_info or {}
    for name, fmt in assignment.items():
        if ".__fused__." in name:
            members = stats_ext[name].get("_fused_members", [])
            dropped = set(pm.get(name, ()))
            for m_ in members:
                if dropped:
                    info = einfo.get(m_)
                    if info is not None:
                        try:
                            eid = int(info[1])
                        except (TypeError, ValueError):
                            eid = None
                        if eid is not None and eid in dropped:
                            continue
                out[m_] = fmt
        else:
            out[name] = fmt
    return out


def build_prune_manifest(
    pruned_map: dict[str, tuple[int, ...]],
    stats_ext: dict,
    expert_info: dict[str, tuple[str, str]],
    expert_saliency: dict[str, dict[int, float]] | None = None,
    uniform_kept: bool = False,
) -> tuple[dict[str, dict], list[str]]:
    """Build a router-keyed prune manifest for the exporter."""
    if not pruned_map:
        return {}, []

    by_router: dict[str, dict[str, set[int]]] = {}
    all_eids_by_router: dict[str, set[int]] = {}
    for source_name, dropped in pruned_map.items():
        members = stats_ext.get(source_name, {}).get("_fused_members", [])
        eids_here: set[int] = set()
        router: str | None = None
        if members:
            for m_ in members:
                info = expert_info.get(m_)
                if info is None:
                    continue
                r, eid_str = info
                router = router or r
                try:
                    eids_here.add(int(eid_str))
                except (TypeError, ValueError):
                    pass
        else:
            router = _packed_entry_router_qname(source_name)
            num_experts = int(
                stats_ext.get(source_name, {}).get("num_experts", 0) or 0
            )
            if router and num_experts > 0:
                eids_here = set(range(num_experts))
        if router is None:
            continue
        by_router.setdefault(router, {})[source_name] = set(dropped)
        all_eids_by_router.setdefault(router, set()).update(eids_here)

    manifest: dict[str, dict] = {}
    warnings: list[str] = []
    for router, super_to_dropped in by_router.items():
        sets = list(super_to_dropped.values())
        consensus = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        if union != consensus:
            disagree = union - consensus
            warnings.append(
                f"{router}: prune-set disagreement across projections; "
                f"consensus={sorted(consensus)}, additional-wanted="
                f"{sorted(disagree)} - honoring consensus only."
            )
        all_eids = all_eids_by_router.get(router, set())
        sorted_all = sorted(all_eids)
        kept = [eid for eid in sorted_all if eid not in consensus]
        orig_to_new = {eid: i for i, eid in enumerate(kept)}
        manifest[router] = {
            "num_experts_orig": len(sorted_all),
            "num_experts_kept": len(kept),
            "pruned_expert_ids": sorted(consensus),
            "kept_expert_ids": kept,
            "orig_to_new_eid": {str(k): v for k, v in orig_to_new.items()},
        }

    if uniform_kept and manifest:
        min_kept = min(int(e["num_experts_kept"]) for e in manifest.values())
        sal = expert_saliency or {}
        for router, entry in list(manifest.items()):
            cur_kept = int(entry["num_experts_kept"])
            if cur_kept <= min_kept:
                continue
            need_extra_drops = cur_kept - min_kept
            kept_now = list(entry["kept_expert_ids"])
            already_dropped = set(entry["pruned_expert_ids"])
            router_sal = sal.get(router, {})
            kept_ranked = sorted(
                kept_now,
                key=lambda eid: (_saliency_lookup(router_sal, eid, float("inf")), eid),
            )
            extra = set(kept_ranked[:need_extra_drops])
            new_dropped = sorted(already_dropped | extra)
            new_kept = [eid for eid in entry["kept_expert_ids"] if eid not in extra]
            entry["pruned_expert_ids"] = new_dropped
            entry["kept_expert_ids"] = new_kept
            entry["num_experts_kept"] = len(new_kept)
            entry["orig_to_new_eid"] = {
                str(eid): i for i, eid in enumerate(new_kept)
            }
            warnings.append(
                f"{router}: padded drops from {cur_kept}->{len(new_kept)} kept "
                f"(+{need_extra_drops} lowest-saliency) for uniform-kept "
                "config.json compatibility."
            )
        for router in sal:
            if router in manifest:
                continue
            if not router.endswith(".gate"):
                continue
            block = router[: -len(".gate")]
            num_orig = None
            for name, s in stats_ext.items():
                if name.startswith(f"{block}.experts.") and isinstance(s, dict):
                    n_e = int(s.get("num_experts", 0) or 0)
                    if n_e > 0:
                        num_orig = n_e
                        break
            if num_orig is None:
                continue
            router_sal = sal[router]
            need_drops = num_orig - min_kept
            if need_drops <= 0:
                continue
            all_eids = list(range(num_orig))
            ranked = sorted(
                all_eids,
                key=lambda eid: (_saliency_lookup(router_sal, eid, float("inf")), eid),
            )
            dropped = sorted(ranked[:need_drops])
            kept = [e for e in all_eids if e not in set(dropped)]
            manifest[router] = {
                "num_experts_orig": num_orig,
                "num_experts_kept": len(kept),
                "pruned_expert_ids": dropped,
                "kept_expert_ids": kept,
                "orig_to_new_eid": {str(e): i for i, e in enumerate(kept)},
            }
            warnings.append(
                f"{router}: DP chose no prune; added {need_drops} "
                f"lowest-saliency drops (->{len(kept)} kept) for "
                "uniform-kept config.json compatibility."
            )

    return manifest, warnings


def apply_consensus_prune(
    pruned_map: dict[str, tuple[int, ...]],
    manifest: dict[str, dict],
    stats_ext: dict,
    expert_info: dict[str, tuple[str, str]],
) -> dict[str, tuple[int, ...]]:
    """Coerce super-Linears to the router consensus drop set."""
    if not manifest:
        return pruned_map
    out: dict[str, tuple[int, ...]] = {}
    for source_name, dropped in pruned_map.items():
        members = stats_ext.get(source_name, {}).get("_fused_members", [])
        router: str | None = None
        if members:
            for m_ in members:
                info = expert_info.get(m_)
                if info is None:
                    continue
                router = info[0]
                break
        else:
            router = _packed_entry_router_qname(source_name)
        if router is None:
            out[source_name] = dropped
            continue
        entry = manifest.get(router)
        if entry is None:
            out[source_name] = dropped
            continue
        consensus = tuple(entry["pruned_expert_ids"])
        if consensus:
            out[source_name] = consensus
    return out
