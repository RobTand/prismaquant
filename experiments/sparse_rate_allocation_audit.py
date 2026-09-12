#!/usr/bin/env python3
"""Offline scalar-MSE allocation-regret replay for sparse E4M3 anchors.

This is deliberately an audit of a predictor's allocation decisions, not an
allocator input.  It uses measured scalar route MSE and measured wire bytes
only.  It is not joint AURA, KL, a runtime price, or serving evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiments.sparse_rate_study_seal import CURRENCY, SEAL_SCHEMA, StudySealError, load_json, roster, sha


SCHEMA = "prismaquant.sparse_rate_allocation_audit.v1"
DATASET_SCHEMA = "prismaquant.sparse_rate_dataset.v1"
STUDY_SCHEMA = "prismaquant.sparse_rate_model_study.v1"
FAMILY = "TESSERA_E4M3_K1"
RATES = (832, 960, 1088)
MAX_STATES = 200_000
MAX_TRANSITIONS = 8_000_000


class AllocationAuditError(ValueError):
    """The supplied research artifacts cannot support an honest replay."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AllocationAuditError(f"invalid JSON {path}") from exc
    if not isinstance(value, dict):
        raise AllocationAuditError(f"JSON object required: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationAuditError(message)


@dataclass(frozen=True)
class Option:
    rate: int
    bytes: int
    truth_uniform: float
    truth_counts: float
    predicted_uniform: float
    predicted_counts: float


@dataclass(frozen=True)
class Group:
    key: tuple[int, str]
    params: int
    options: tuple[Option, ...]


def _exact_mckp(groups: list[Group], costs: list[list[float]], budget: int):
    """Exact byte multi-choice DP, retaining one least-cost state per byte.

    This intentionally small research replay has an explicit refusal boundary.
    A future population that crosses it needs a reviewed exact decomposition;
    it must not acquire rounded bytes or a silently truncated frontier.
    """
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    transitions = 0
    for group, group_costs in zip(groups, costs):
        transitions += len(states) * len(group.options)
        if transitions > MAX_TRANSITIONS:
            raise AllocationAuditError("exact replay exceeds transition limit")
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (total, selected) in states.items():
            for option_index, (option, cost) in enumerate(zip(group.options, group_costs)):
                candidate_bytes = used + option.bytes
                if candidate_bytes > budget:
                    continue
                candidate = (total + float(cost), selected + (option_index,))
                old = next_states.get(candidate_bytes)
                if old is None or candidate[0] < old[0] or (
                        candidate[0] == old[0] and candidate[1] < old[1]):
                    next_states[candidate_bytes] = candidate
        states = next_states
        if not states:
            return None
        if len(states) > MAX_STATES:
            raise AllocationAuditError("exact replay exceeds state limit")
    used, (_, selected) = min(states.items(), key=lambda item: (item[1][0], item[0], item[1][1]))
    return selected, used


def _objective(groups: list[Group], selected: tuple[int, ...], mode: str) -> float:
    field = "truth_uniform" if mode == "uniform" else "truth_counts"
    return math.fsum(getattr(group.options[index], field)
                     for group, index in zip(groups, selected))


def _build_groups(data: dict[str, np.ndarray], predictions: np.ndarray,
                  study_truth: np.ndarray, *, stage: str, anchors: tuple[int, ...] = ()):
    rates = {int(rate): index for index, rate in enumerate(data["rates"])}
    _require(set(RATES) <= set(rates), "dataset lacks required rates 832/960/1088")
    n = len(data["qnames"])
    _require(predictions.shape == data["values"].shape == study_truth.shape,
             "study prediction/truth shape differs from dataset")
    is_final = np.asarray(data["layers"], dtype=np.int64) % 5 == 3
    eligible_stage = is_final if stage == "final" else ~is_final
    expert_family = ((np.asarray(data["families"]) == FAMILY)
                     & (np.asarray(data["roles"]) == "expert"))
    out_of_stage = expert_family & ~eligible_stage
    _require(not np.any(np.isfinite(predictions[out_of_stage]))
             and not np.any(np.isfinite(study_truth[out_of_stage])),
             "study contains prediction or truth outside its declared stage")
    selected = (expert_family
                & (np.asarray(data["roles"]) == "expert")
                & eligible_stage)
    for anchor in anchors:
        _require(anchor in rates, f"declared anchor {anchor} is absent from dataset")
        expected, actual = data["values"][selected, rates[anchor]], predictions[selected, rates[anchor]]
        finite = np.isfinite(expected)
        _require(np.array_equal(actual[finite], expected[finite])
                 and not np.any(np.isfinite(actual[~finite])),
                 f"declared anchor prediction differs from bound dataset at {anchor}")
    observed = np.isfinite(study_truth[selected])
    expected = data["values"][selected][observed]
    _require(np.array_equal(study_truth[selected][observed], expected),
             "study truth differs from the bound dataset")
    group_rows: dict[int, dict[str, dict[str, int]]] = {}
    omissions: list[dict] = []
    for index in np.flatnonzero(selected):
        structure = str(data["structures"][index])
        if structure not in {"gate_proj", "up_proj", "down_proj"}:
            omissions.append({"qname": str(data["qnames"][index]), "reason": "not_routed_expert_projection"})
            continue
        qname = str(data["qnames"][index])
        prefix = qname.rsplit(".", 1)[0]
        layer = int(data["layers"][index])
        triples = group_rows.setdefault(layer, {})
        _require(structure not in triples.setdefault(prefix, {}),
                 f"{qname}: duplicate sibling projection")
        triples[prefix][structure] = int(index)
    groups: list[Group] = []
    for layer, triples in sorted(group_rows.items()):
        incomplete = [prefix for prefix, members in triples.items()
                      if set(members) != {"gate_proj", "up_proj", "down_proj"}]
        if incomplete:
            raise AllocationAuditError(f"layer {layer}: incomplete expert sibling triple(s): {incomplete[:3]}")
        indices = [members[structure] for prefix, members in sorted(triples.items())
                   for structure in ("gate_proj", "up_proj", "down_proj")]
        activation_contracts = {str(data["activation_contracts"][i]) for i in indices}
        _require(len(activation_contracts) == 1,
                 f"layer {layer}: E4M3 activation contract differs across expert stack")
        options = []
        missing = []
        for rate in RATES:
            column = rates[rate]
            truth = data["values"][indices, column]
            wires = data["wire_bytes"][indices, column]
            predicted = predictions[indices, column]
            if (not np.all(np.isfinite(truth)) or not np.all(truth > 0)
                    or not np.all(np.isfinite(predicted)) or not np.all(predicted > 0)
                    or not np.all(wires > 0)):
                missing.append(rate)
                continue
            options.append(Option(rate=rate, bytes=int(np.sum(wires)),
                                  truth_uniform=float(np.sum(truth)),
                                  truth_counts=float(np.sum(truth * data["counts"][indices])),
                                  predicted_uniform=float(np.sum(predicted)),
                                  predicted_counts=float(np.sum(predicted * data["counts"][indices]))))
        if missing:
            raise AllocationAuditError(f"layer {layer}: missing measured or predicted rate(s) {missing}")
        params = int(sum(int(data["rows"][i]) * int(data["cols"][i]) for i in indices))
        groups.append(Group(key=(layer, f"{len(triples)}_experts"), params=params,
                            options=tuple(options)))
    _require(groups, "no complete predicted routed expert layers")
    return groups, omissions, {"candidate_expert_rows": int(np.sum(selected)),
                               "final_rows_excluded_by_stage": int(np.sum(is_final) if stage == "development" else 0),
                               "complete_layers": len(groups), "omitted_groups_or_rows": len(omissions),
                               "grouping": "whole_routed_layer__all_experts_gate_up_down"}


def _budgets(groups: list[Group]) -> list[tuple[float, int]]:
    params = sum(group.params for group in groups)
    floor = sum(min(option.bytes for option in group.options) for group in groups)
    ceiling = sum(max(option.bytes for option in group.options) for group in groups)
    requested = [3.25, 3.50, 3.75, 4.00, 4.25]
    rows = []
    for bpp in requested:
        budget = int(math.floor(bpp * params / 8.0))
        if floor <= budget <= ceiling:
            rows.append((bpp, budget))
    if not rows:
        rows = [(8.0 * floor / params, floor), (8.0 * ceiling / params, ceiling)]
    return list(dict.fromkeys(rows))


def audit_model(data: dict[str, np.ndarray], predictions: np.ndarray,
                study_truth: np.ndarray | None = None, *, stage: str,
                anchors: tuple[int, ...] = ()) -> dict:
    if study_truth is None:
        study_truth = np.full_like(predictions, np.nan)
    groups, omissions, coverage = _build_groups(data, predictions, study_truth, stage=stage, anchors=anchors)
    params = sum(group.params for group in groups)
    output = {"evaluated_groups": len(groups), "quantizable_params": params,
              "coverage": coverage, "omissions": omissions, "modes": {}}
    for mode in ("uniform", "routed_token_count_weighted"):
        predicted_field = "predicted_uniform" if mode == "uniform" else "predicted_counts"
        truth_field = "truth_uniform" if mode == "uniform" else "truth_counts"
        truth_costs = [[getattr(option, truth_field) for option in group.options] for group in groups]
        predicted_costs = [[getattr(option, predicted_field) for option in group.options] for group in groups]
        results = []
        for requested_bpp, budget in _budgets(groups):
            oracle = _exact_mckp(groups, truth_costs, budget)
            predicted = _exact_mckp(groups, predicted_costs, budget)
            if oracle is None or predicted is None:
                results.append({"requested_bpp": requested_bpp, "bytes_budget": budget, "infeasible": True})
                continue
            oracle_selection, oracle_bytes = oracle
            predicted_selection, predicted_bytes = predicted
            oracle_truth = _objective(groups, oracle_selection,
                                      "uniform" if mode == "uniform" else "counts")
            predicted_truth = _objective(groups, predicted_selection,
                                         "uniform" if mode == "uniform" else "counts")
            results.append({"requested_bpp": requested_bpp, "bytes_budget": budget,
                            "oracle_bytes": oracle_bytes, "predicted_choice_bytes": predicted_bytes,
                            "oracle_bpp": 8.0 * oracle_bytes / params,
                            "predicted_choice_bpp": 8.0 * predicted_bytes / params,
                            "oracle_scalar_mse": oracle_truth,
                            "predicted_choice_scalar_mse": predicted_truth,
                            "regret": predicted_truth - oracle_truth,
                            "regret_pct": 100.0 * (predicted_truth - oracle_truth) / oracle_truth,
                            "items_agree": sum(a == b for a, b in zip(oracle_selection, predicted_selection)),
                            "item_count": len(groups)})
        output["modes"][mode] = results
    return output


def _validate_seal(seal_path: Path, dataset: Path, study: Path,
                   manifest: dict, plan: dict) -> list[dict]:
    try:
        seal = load_json(seal_path)
        report_path = study / "report.json"
        report = load_json(report_path)
        current_roster = roster(study, report)
    except StudySealError as exc:
        raise AllocationAuditError(str(exc)) from exc
    _require(seal.get("schema") == SEAL_SCHEMA and seal.get("post_run_sealing") is True
             and seal.get("currency") == CURRENCY,
             "unsupported study evidence seal")
    _require(manifest.get("currency") == CURRENCY and plan.get("currency") == CURRENCY
             and report.get("currency") == CURRENCY,
             "dataset or study currency is not output_mse_under_route_activation_contract")
    _require(seal.get("dataset") == {"npz_sha256": _sha(dataset / "sparse_rate_dataset.npz"),
                                     "manifest_sha256": _sha(dataset / "manifest.json")},
             "study evidence seal dataset binding differs")
    sealed_study = seal.get("study")
    _require(isinstance(sealed_study, dict)
             and sealed_study.get("stage") == plan.get("stage")
             and sealed_study.get("plan_sha256") == _sha(study / "plan.json")
             and sealed_study.get("report_sha256") == _sha(report_path)
             and sealed_study.get("models") == current_roster,
             "study evidence seal study binding differs")
    if plan.get("stage") == "final":
        dependency = seal.get("final_dependency")
        _require(isinstance(dependency, dict)
                 and dependency.get("choice_sha256") == plan.get("choice_sha256")
                 and isinstance(dependency.get("development_report_sha256"), str),
                 "final study evidence seal lacks frozen choice linkage")
    return current_roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--seal", required=True,
                        help="post-run evidence seal created by sparse_rate_study_seal.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default=None,
                        help="Optional comma-separated study model stems to replay")
    arguments = parser.parse_args()
    dataset, study, seal, out = (Path(arguments.dataset), Path(arguments.study),
                                 Path(arguments.seal), Path(arguments.out))
    if out.exists():
        raise AllocationAuditError("--out already exists; preserve the prior audit")
    manifest = _json(dataset / "manifest.json")
    plan = _json(study / "plan.json")
    _require(manifest.get("schema") == DATASET_SCHEMA and manifest.get("currency") == CURRENCY,
             "unsupported dataset schema or currency")
    _require(plan.get("schema") == STUDY_SCHEMA and plan.get("stage") in {"development", "final"},
             "unsupported study schema or stage")
    dataset_npz = dataset / "sparse_rate_dataset.npz"
    _require(manifest.get("npz_sha256") == _sha(dataset_npz), "dataset NPZ hash differs from manifest")
    identity = plan.get("identity", {})
    _require(identity.get("dataset_npz_sha256") == _sha(dataset_npz)
             and identity.get("dataset_manifest_sha256") == _sha(dataset / "manifest.json"),
             "study does not bind this dataset")
    model_roster = _validate_seal(seal, dataset, study, manifest, plan)
    with np.load(dataset_npz, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    out.mkdir(parents=True)
    models = {}
    requested_models = None if arguments.models is None else {
        item.strip() for item in arguments.models.split(",") if item.strip()}
    _require(requested_models is None or requested_models,
             "--models must name at least one model")
    sealed_models = {row["name"]: tuple(row["anchors"]) for row in model_roster}
    _require(requested_models is None or requested_models <= set(sealed_models),
             "--models names an artifact absent from the sealed study")
    for name, anchors in sealed_models.items():
        if requested_models is not None and name not in requested_models:
            continue
        path = study / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            _require(set(archive.files) == {"predictions", "truth"}, f"invalid study model {path.name}")
            models[path.stem] = audit_model(data, archive["predictions"], archive["truth"],
                                            stage=plan["stage"], anchors=anchors)
    _require(models, "study has no prediction artifacts")
    _require(requested_models is None or set(models) == requested_models,
             "--models names an artifact absent from the study")
    payload = {"schema": SCHEMA, "research_only": True,
               "currency": CURRENCY,
               "scalar_aggregation": "sum_of_measured_output_mse_by_whole_routed_layer",
               "objective_label": "scalar_output_mse_allocation_regret_proxy",
               "not_joint_aura": True,
               "not_kl": True, "not_runtime": True,
               "byte_currency": "measured_wire_bytes_only",
               "study_stage": plan["stage"],
               "dataset": {"npz_sha256": _sha(dataset_npz),
                           "manifest_sha256": _sha(dataset / "manifest.json")},
               "study_evidence_seal_sha256": _sha(seal),
               "models": models}
    (out / "allocation_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
