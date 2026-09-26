#!/usr/bin/env python3
"""Run the pre-registered DSV4 layer-21 monotone min-chain pilot.

Free-fit errors and scale-search argmins are consumed from phase A.  The tool
never invokes the free scale sweep.  It materializes the banked free solution
from the matching warm argmin solely so a free winner can become the next
predecessor, then evaluates only the embed and cheap-refine chain arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open

from prismaquant.cb_minchain import (
    MINCHAIN_CONTEXT_VERSION,
    MINCHAIN_SCHEMA,
    chain_identity,
    embed_predecessor,
    refine_one_entry,
    select_arm,
)
from prismaquant.expert_empirical_cost import _cb_ladder_law
from prismaquant.layer_streaming import _build_fp8_scale_inv_map, _build_weight_map
from prismaquant.nvfp4_cb_formats import (
    ldlq_reassign_cb_fields,
    nvfp4_cb_fields,
    nvfp4_cb_reconstruct,
)
from tools.dsv4_ldlq_cost_campaign import (
    ANCHORS,
    COL_WEIGHTS,
    CONTEXT,
    HOLDOUTS,
    PILOT_LAYER,
    PROJECTIONS,
    RUNGS,
    RUN_ROOT,
    SOURCE,
    TOLERANCE,
    atomic_json,
    atomic_pickle,
    atomic_text,
    load_layer_identity,
    load_projection,
    per_slice_mse,
    percentile,
)


PILOT_EXPERTS = (121, 199, 97, 131, 240, 66, 84, 119,
                 126, 22, 80, 223, 6, 43, 167, 227)
CHAIN_ROOT = RUN_ROOT / "minchain-pilot"
CHAIN_SHARDS = CHAIN_ROOT / "shards"
CHAIN_STATE = CHAIN_ROOT / "state"
SCHEMA = "prismaquant.dsv4_minchain_pilot_report.v1"
STATE_SCHEMA = "prismaquant.dsv4_minchain_state.v1"
MATERIALIZE_SCHEMA = "prismaquant.dsv4_minchain_materialized_free.v1"


class FreeReplayError(RuntimeError):
    def __init__(self, projection: str, rung: int, detail: Mapping[str, Any]):
        super().__init__(
            f"banked free solution is not exactly replayable: "
            f"{projection} K{rung}"
        )
        self.projection = projection
        self.rung = rung
        self.detail = dict(detail)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device).contiguous()
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _local_fields(fields: Mapping[str, Any], expert: int, rows: int) -> dict:
    start, stop = expert * rows, (expert + 1) * rows
    local = dict(fields)
    for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
        if isinstance(local.get(key), torch.Tensor):
            local[key] = local[key][start:stop].contiguous()
    local["shape"] = (rows, int(fields["shape"][-1]))
    return local


def _compact(fields: Mapping[str, Any]) -> dict:
    out = dict(fields)
    for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
        if isinstance(out.get(key), torch.Tensor):
            out[key] = out[key].clone().contiguous()
    codebook = out.get("codebook")
    if isinstance(codebook, torch.Tensor):
        out["codebook"] = codebook.clone().contiguous()
    elif codebook is not None:
        out["codebook"] = tuple(table.clone().contiguous() for table in codebook)
    return out


def _warm_scales(path: Path, selected: Sequence[int], rows: int,
                 device: torch.device) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"scales"}:
            raise AssertionError(f"unexpected FP8 warm planes at {path}")
        scales = handle.get_tensor("scales")
    expected_rows = 256 * rows
    if tuple(scales.shape) != (expected_rows, 1):
        raise AssertionError(
            f"warm scale shape {tuple(scales.shape)} != {(expected_rows, 1)}"
        )
    selected_scales = torch.cat(
        [scales[expert * rows:(expert + 1) * rows] for expert in selected], dim=0
    )
    return {"scales": selected_scales.to(device=device).contiguous()}


def _materialize_free(
    *, weight: torch.Tensor, col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor], warm_path: Path, rung: int,
) -> tuple[list[dict], list[float], float]:
    rows = int(weight.shape[1])
    warm = _warm_scales(warm_path, PILOT_EXPERTS, rows, weight.device)
    # The dynamic compiled VQ argmin specializes on the product-table K.
    # Reusing that cache across ascending, differently-sized rungs was
    # observed to change the second warm replay even though a cold replay was
    # bit-for-error identical to phase A.  Free truth is already banked, so
    # fail closed and give every deterministic warm materialization the same
    # cold-specialization boundary as the verified standalone replay.  This
    # pilot-only reset is outside the timed free/chain arm regions.
    from prismaquant import nvfp4_cb_formats as cb_formats

    for name in (
        "_vq_dist_argmin_compiled",
        "_score_min_compiled",
        "_score_argmin_compiled",
        "_score_min_batched_compiled",
        "_score_minargmin_batched_compiled",
    ):
        getattr(cb_formats, name).cache_clear()
    with cb_formats._LDLQ_FACTOR_CACHE_LOCK:
        cb_formats._LDLQ_FACTOR_CACHE.clear()
    torch.compiler.reset()
    # Avoid a first-use lazy-wrapper race when the exact production replay
    # enters Cholesky concurrently from sixteen feeder threads.
    torch.linalg.cholesky(torch.eye(2, device=weight.device))
    torch.cuda.synchronize()
    started = time.perf_counter()
    fields = nvfp4_cb_fields(
        weight, rung, grid="fp8", mode="product",
        col_weights=col_weights, scale_sweep=True,
        warm_scale_state=warm, encode_tier="balanced",
    )
    fields = ldlq_reassign_cb_fields(
        weight, fields, col_weights, tuple(activation_rows),
        grid="fp8", mode="product", batch_experts=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    reconstruction = nvfp4_cb_reconstruct(
        fields, rung, grid="fp8", mode="product"
    ).to(weight.dtype)
    errors = per_slice_mse(weight, reconstruction)
    locals_ = [_local_fields(fields, index, rows) for index in range(len(weight))]
    del reconstruction, fields, warm
    return locals_, errors, elapsed


def _isolated_materialize_free(
    *, projection: str, rung: int, phase_path: Path,
    device: torch.device,
) -> tuple[list[dict], list[float], float]:
    """Replay one banked free solution in a process-cold encoder context."""
    temp = CHAIN_ROOT / "materialize-tmp" / (
        f"layer_021_{projection}_K{rung}_{sha256_file(phase_path)[:16]}.pt"
    )
    temp.parent.mkdir(parents=True, exist_ok=True)
    if temp.exists():
        temp.unlink()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PRISMAQUANT_CB_MINCHAIN_PILOT"] = "1"
    command = [
        sys.executable, str(Path(__file__).resolve()), "materialize-free",
        "--projection", projection, "--rung", str(rung),
        "--phase-path", str(phase_path), "--output", str(temp),
    ]
    last_code = None
    failure_detail: dict[str, Any] = {}
    for attempt in range(1, 4):
        result = subprocess.run(command, env=env, check=False)
        last_code = result.returncode
        if result.returncode == 0:
            break
        if temp.exists():
            failed = torch.load(temp, map_location="cpu", weights_only=False)
            if failed.get("status") == "REPLAY_MISMATCH":
                failure_detail = dict(failed)
            temp.unlink()
        print(
            f"[minchain] isolated free replay retry {attempt}/3 "
            f"{projection} K{rung}", flush=True,
        )
    if last_code:
        raise FreeReplayError(
            projection, rung, {
                **failure_detail,
                "attempts": 3,
                "last_exit_code": last_code,
            },
        )
    payload = torch.load(temp, map_location="cpu", weights_only=False)
    temp.unlink()
    if (
        payload.get("schema") != MATERIALIZE_SCHEMA
        or payload.get("projection") != projection
        or payload.get("rung") != rung
        or payload.get("phase_sha256") != sha256_file(phase_path)
    ):
        raise AssertionError(f"isolated free replay identity mismatch: {temp}")
    solutions = [_move(value, device) for value in payload["solutions"]]
    for fields in solutions:
        fields["indices"] = fields["indices"].to(torch.int64)
    return solutions, list(payload["errors"]), float(payload["elapsed_seconds"])


def _materialize_child(
    *, projection: str, rung: int, phase_path: Path, output: Path,
) -> int:
    """Cold-process worker used only by the pilot replay boundary."""
    if projection not in PROJECTIONS or rung not in RUNGS:
        raise ValueError(f"invalid materialization cell {projection} K{rung}")
    phase = pickle.loads(phase_path.read_bytes())
    if (
        phase.get("projection") != projection
        or phase.get("rung") != rung
        or phase.get("format") != f"FP8_CB_K{rung}"
    ):
        raise AssertionError(f"phase truth coordinate mismatch: {phase_path}")
    device = torch.device("cuda:0")
    _, layer_record = load_layer_identity(PILOT_LAYER)
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    full = load_projection(
        PILOT_LAYER, projection, device=device,
        identity=layer_record["identity"], all_col_weights=all_col_weights,
        model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
        scale_map=scale_map,
    )
    index = torch.tensor(PILOT_EXPERTS, device=device)
    weight = full["weight"].index_select(0, index).contiguous()
    col_weights = full["col_weights"].index_select(0, index).contiguous()
    activation_rows = tuple(
        full["activation_rows"][expert] for expert in PILOT_EXPERTS
    )
    del full
    solutions, errors, elapsed = _materialize_free(
        weight=weight, col_weights=col_weights,
        activation_rows=activation_rows,
        warm_path=Path(phase["warm_state_path"]), rung=rung,
    )
    banked = [
        float(phase["weight_mse_per_expert"][expert])
        for expert in PILOT_EXPERTS
    ]
    mismatches = []
    for expert, observed, truth in zip(PILOT_EXPERTS, errors, banked):
        relative = abs(observed - truth) / max(abs(truth), 1e-30)
        if relative > 2e-6:
            mismatches.append({
                "expert": expert, "observed": float(observed),
                "banked": float(truth), "absolute_excess": float(observed - truth),
                "relative_difference": float(relative),
            })
    if mismatches:
        worst = max(mismatches, key=lambda value: value["relative_difference"])
        _atomic_torch(output, {
            "schema": MATERIALIZE_SCHEMA,
            "status": "REPLAY_MISMATCH",
            "projection": projection, "rung": rung,
            "phase_sha256": sha256_file(phase_path),
            "tolerance_relative": 2e-6,
            "mismatch_count": len(mismatches), "worst": worst,
        })
        print(
            f"[minchain] {projection} K{rung} isolated replay mismatch "
            f"expert={worst['expert']} rel={worst['relative_difference']:.3e}",
            flush=True,
        )
        return 4
    packed = _move(solutions, torch.device("cpu"))
    for fields in packed:
        fields["indices"] = fields["indices"].to(torch.int16)
    _atomic_torch(output, {
        "schema": MATERIALIZE_SCHEMA,
        "projection": projection,
        "rung": rung,
        "phase_path": str(phase_path),
        "phase_sha256": sha256_file(phase_path),
        "solutions": packed,
        "errors": errors,
        "elapsed_seconds": elapsed,
    })
    return 0


def _one_error(weight: torch.Tensor, fields: Mapping[str, Any], rung: int) -> float:
    reconstruction = nvfp4_cb_reconstruct(
        dict(fields), rung, grid="fp8", mode="product"
    ).to(weight.dtype)
    return float((weight - reconstruction).float().square().mean().item())


def _evaluate_all(
    weight: torch.Tensor, solutions: Sequence[Mapping[str, Any]], rung: int,
) -> tuple[list[float], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = []
    for batch_start in range(0, len(solutions), 16):
        batch_stop = min(batch_start + 16, len(solutions))
        streams = [
            torch.cuda.Stream(device=weight.device)
            for _ in range(batch_stop - batch_start)
        ]

        def work(local: int) -> tuple[int, float]:
            index = batch_start + local
            with torch.cuda.device(weight.device), torch.cuda.stream(streams[local]):
                return index, _one_error(weight[index], solutions[index], rung)

        with ThreadPoolExecutor(max_workers=len(streams)) as pool:
            results.extend(pool.map(work, range(len(streams))))
        current = torch.cuda.current_stream(weight.device)
        for stream in streams:
            current.wait_stream(stream)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item[0])
    return [item[1] for item in results], elapsed


def _refine_all(
    *, weight: torch.Tensor, col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor], predecessors: Sequence[Mapping[str, Any]],
    rung: int,
) -> tuple[list[dict], list[float], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    results: list[tuple[int, dict, float]] = []
    for batch_start in range(0, len(predecessors), 16):
        batch_stop = min(batch_start + 16, len(predecessors))
        streams = [
            torch.cuda.Stream(device=weight.device)
            for _ in range(batch_stop - batch_start)
        ]

        def work(local: int) -> tuple[int, dict, float]:
            index = batch_start + local
            with torch.cuda.device(weight.device), torch.cuda.stream(streams[local]):
                fields = refine_one_entry(
                    weight[index], predecessors[index], rung,
                    col_weights=col_weights[index],
                    activation_rows=activation_rows[index],
                    iterations=3,
                )
                error = _one_error(weight[index], fields, rung)
            return index, fields, error

        with ThreadPoolExecutor(max_workers=len(streams)) as pool:
            results.extend(pool.map(work, range(len(streams))))
        current = torch.cuda.current_stream(weight.device)
        for stream in streams:
            current.wait_stream(stream)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item[0])
    return ([item[1] for item in results],
            [item[2] for item in results], elapsed)


def _load_state(path: Path, expected_sha: str, device: torch.device) -> list[dict]:
    if sha256_file(path) != expected_sha:
        raise AssertionError(f"min-chain state digest mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != STATE_SCHEMA:
        raise AssertionError(f"min-chain state schema mismatch: {path}")
    solutions = [_move(fields, device) for fields in payload["solutions"]]
    for fields in solutions:
        fields["indices"] = fields["indices"].to(torch.int64)
    return solutions


def _save_cell(
    projection: str, rung: int, record: Mapping[str, Any], solutions: Sequence[dict]
) -> None:
    state_path = CHAIN_STATE / f"layer_021_{projection}_K{rung}.pt"
    shard_path = CHAIN_SHARDS / f"layer_021_{projection}_K{rung}.pkl"
    packed = _move(list(solutions), torch.device("cpu"))
    for fields in packed:
        indices = fields["indices"]
        fields["indices"] = indices.to(torch.int16)
    _atomic_torch(state_path, {
        "schema": STATE_SCHEMA,
        "state_encoding": "indices-int16__runtime-int64",
        "projection": projection,
        "rung": rung,
        "expert_ids": list(PILOT_EXPERTS),
        "solutions": packed,
    })
    complete = dict(record)
    complete["state_path"] = str(state_path)
    complete["state_sha256"] = sha256_file(state_path)
    atomic_pickle(shard_path, complete)


def _fit_gate(records: Mapping[str, Mapping[int, Mapping]]) -> dict[str, Any]:
    kmap = {f"FP8_CB_K{k}": k for k in RUNGS}
    anchor_names = [f"FP8_CB_K{k}" for k in ANCHORS]
    target_names = [
        f"FP8_CB_K{k}" for k in RUNGS
        if k not in set(ANCHORS).union(HOLDOUTS)
    ]
    holdout_errors: dict[int, list[float]] = {k: [] for k in HOLDOUTS}
    prediction_errors: list[float] = []
    accepted: list[dict[str, Any]] = []
    by_projection: dict[str, Any] = {}
    for projection in PROJECTIONS:
        local_accepted = 0
        local_errors: list[float] = []
        for index, expert in enumerate(PILOT_EXPERTS):
            values = {
                f"FP8_CB_K{k}": float(records[projection][k]["selected_error"][index])
                for k in RUNGS
            }
            law = _cb_ladder_law(kmap, anchor_names, values)
            if law is None:
                continue
            holdouts = {}
            for k in HOLDOUTS:
                name = f"FP8_CB_K{k}"
                rel = abs(law.predict(name) - values[name]) / max(values[name], 1e-30)
                holdout_errors[k].append(rel)
                holdouts[k] = rel
            if all(holdouts[k] <= TOLERANCE for k in HOLDOUTS):
                local_accepted += 1
                accepted.append({"projection": projection, "expert": expert})
                for name in target_names:
                    rel = abs(law.predict(name) - values[name]) / max(values[name], 1e-30)
                    prediction_errors.append(rel)
                    local_errors.append(rel)
        by_projection[projection] = {
            "accepted": local_accepted,
            "total": len(PILOT_EXPERTS),
            "acceptance_rate": local_accepted / len(PILOT_EXPERTS),
            "median": statistics.median(local_errors) if local_errors else math.nan,
            "p95": percentile(local_errors, 0.95),
        }
    median = statistics.median(prediction_errors) if prediction_errors else math.nan
    p95 = percentile(prediction_errors, 0.95)
    return {
        "dual_holdouts": list(HOLDOUTS),
        "per_holdout": {
            f"K{k}": {
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
                "accepted_at_10pct": sum(value <= TOLERANCE for value in values),
            }
            for k, values in holdout_errors.items()
        },
        "accepted_slices": len(accepted),
        "total_slices": len(PILOT_EXPERTS) * len(PROJECTIONS),
        "accepted": accepted,
        "prediction_population": "missing rungs only (anchors and holdouts excluded)",
        "prediction_count": len(prediction_errors),
        "median": median,
        "p95": p95,
        "by_projection": by_projection,
        "pass": bool(prediction_errors and median <= 0.05 and p95 <= 0.15),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# DSV4 Monotone Min-Chain Pilot",
        "",
        f"- Mechanical decision: **{report['decision']}**",
        f"- Burn method: **{report['burn_method']}**",
        f"- Layer/expert cells: layer 21, 16 registered experts x 3 projections x {len(RUNGS)} rungs",
        "- Selection/report metric: production per-slice `weight_mse` (the allocator's predicted-dloss input)",
        "- Free arm: phase-A banked truth; scale sweep was not repeated",
        f"- Chain serialization version: `{MINCHAIN_CONTEXT_VERSION}`",
        "",
        "| Gate | Result | Measured value | Threshold |",
        "|---|---|---:|---:|",
        f"| G1 monotone selected curve | {'PASS' if gates['G1']['pass'] else 'FAIL'} | {gates['G1']['violations']} violations | 0 |",
        f"| G2 zero tax vs free | {'PASS' if gates['G2']['pass'] else 'FAIL'} | {gates['G2']['violations']} violations; max excess {gates['G2']['max_excess']:.3e} | 0 |",
        f"| G3 dual-holdout fit | {'PASS' if gates['G3']['pass'] else 'FAIL'} | median {gates['G3']['median']:.2%}; p95 {gates['G3']['p95']:.2%} | <=5%; <=15% |",
        f"| G4 encode overhead | {'PASS' if gates['G4']['pass'] else 'FAIL'} | {gates['G4']['total_over_free_ratio']:.3f}x | <=1.6x |",
        f"| G5 refine wins (informative) | INFO | {gates['G5']['wins']}/{gates['G5']['eligible_cells']} ({gates['G5']['win_rate']:.2%}); median gain {gates['G5']['median_gain']:.2%} | not gating |",
        "",
        "G4 compares banked full-fit seconds per expert-cell (phase-A batch time prorated 16/256) with measured embed+refine seconds per registered expert-cell. Free-solution warm materialization is separately reported and excluded because an integrated encoder already holds the just-finished free fields.",
        "",
        "## Fit detail",
        "",
        f"Dual-holdout accepted slices: {gates['G3']['accepted_slices']}/{gates['G3']['total_slices']}. Prediction errors are evaluated on missing rungs only, against measured chain truth.",
        "",
        "## Identity and assertions",
        "",
        "Every slice/rung records `winning_arm`, `solution_digest`, and (for embed/refine winners) `predecessor_digest`. The state file digest is content-checked before resume. G1 and G2 were runtime assertions; a violation aborts the pilot.",
        "",
    ]
    if report.get("failure"):
        failure = report["failure"]
        lines.extend([
            "## Mandatory replay-integrity failure",
            "",
            f"The executable chain stopped at `{failure['projection']} K{failure['rung']}` after {failure['attempts']} isolated cold attempts. Worst slice: expert {failure['worst']['expert']}, relative replay difference {failure['worst']['relative_difference']:.3%}, absolute difference {failure['worst']['absolute_excess']:.6e}.",
            "",
            "Phase A banked the reported free errors and warm scale argmins, but not the complete LDLQ assignment planes. Re-encoding the free arm was forbidden, so the missing verbatim predecessor cannot be substituted. G3 below includes a clearly labeled conservative banked running-min diagnostic; it is not promoted to an executable-chain PASS.",
            "",
        ])
    return "\n".join(lines)


def _replay_failure_report(failure: FreeReplayError) -> int:
    """Record a numerical NO-GO and mechanically select the incumbent."""
    simulated: dict[str, dict[int, dict[str, list[float]]]] = {}
    for projection in PROJECTIONS:
        simulated[projection] = {}
        previous = [math.inf] * len(PILOT_EXPERTS)
        for rung in RUNGS:
            phase_path = (
                RUN_ROOT / "pilot-shards"
                / f"layer_021_{projection}_K{rung}.pkl"
            )
            phase = pickle.loads(phase_path.read_bytes())
            free = [
                float(phase["weight_mse_per_expert"][expert])
                for expert in PILOT_EXPERTS
            ]
            selected = [min(prior, value) for prior, value in zip(previous, free)]
            simulated[projection][rung] = {"selected_error": selected}
            previous = selected
    diagnostic_fit = _fit_gate(simulated)
    diagnostic_pass = bool(diagnostic_fit["pass"])
    diagnostic_fit["diagnostic_threshold_pass"] = diagnostic_pass
    diagnostic_fit["pass"] = False
    diagnostic_fit["status"] = (
        "banked_running_min_diagnostic_only__executable_chain_incomplete"
    )

    partial_records = []
    for path in sorted(CHAIN_SHARDS.glob("layer_021_*_K*.pkl")):
        record = pickle.loads(path.read_bytes())
        if record.get("schema") == MINCHAIN_SCHEMA:
            partial_records.append(record)
    free_seconds = sum(
        float(record["timing"]["banked_free_prorated_seconds"])
        for record in partial_records
    )
    added_seconds = sum(
        float(record["timing"]["embed_seconds"])
        + float(record["timing"]["refine_seconds"])
        for record in partial_records
    )
    materialization = sum(
        float(record["timing"]["free_warm_materialization_seconds"])
        for record in partial_records
    )
    partial_ratio = (
        (free_seconds + added_seconds) / free_seconds
        if free_seconds else math.inf
    )
    arm_counts = {"free": 0, "embed": 0, "refine": 0}
    eligible_evaluated = 0
    refine_gains = []
    for record in partial_records:
        for local, arm in enumerate(record["winning_arm"]):
            arm_counts[arm] += 1
            if int(record["rung"]) > RUNGS[0]:
                eligible_evaluated += 1
            if arm == "refine":
                embed = float(record["embed_error"][local])
                refine = float(record["refine_error"][local])
                refine_gains.append(
                    (embed - refine) / max(abs(embed), 1e-30)
                )
    detail = failure.detail
    worst = dict(detail.get("worst") or {
        "expert": -1, "relative_difference": math.inf,
        "absolute_excess": math.inf,
    })
    positive_excess = max(0.0, float(worst["absolute_excess"]))
    report = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "decision": "ONE OR MORE MANDATORY GATES FAIL",
        "burn_method": "INCUMBENT",
        "expert_ids": list(PILOT_EXPERTS),
        "selection_metric": "weight_mse",
        "serialization": {
            "chain_schema": MINCHAIN_SCHEMA,
            "chain_version": MINCHAIN_CONTEXT_VERSION,
            "base_context": (
                CONTEXT.to_dict() if hasattr(CONTEXT, "to_dict") else str(CONTEXT)
            ),
        },
        "arm_counts": arm_counts,
        "failure": {
            "kind": "BANKED_FREE_VERBATIM_REPLAY_FAILURE",
            "projection": failure.projection, "rung": failure.rung,
            "attempts": int(detail.get("attempts", 3)), "worst": worst,
            "mismatch_count_last_attempt": int(detail.get("mismatch_count", 1)),
            "phase_sha256": detail.get("phase_sha256"),
            "policy": "no free-arm re-encode; mandatory NO-GO",
        },
        "gates": {
            "G1": {
                "pass": False, "violations": 1,
                "completed_chain_cells": len(partial_records),
                "required_chain_cells": len(PROJECTIONS) * len(RUNGS),
                "reason": "verbatim predecessor serialization unavailable",
            },
            "G2": {
                "pass": False, "violations": 1,
                "max_excess": positive_excess,
                "unproven_cells": 1,
                "replay_relative_difference": float(
                    worst["relative_difference"]
                ),
            },
            "G3": diagnostic_fit,
            "G4": {
                "pass": False,
                "free_prorated_seconds": free_seconds,
                "added_arm_seconds": added_seconds,
                "free_materialization_seconds_excluded": materialization,
                "added_over_free_ratio": (
                    added_seconds / free_seconds if free_seconds else math.inf
                ),
                "total_over_free_ratio": partial_ratio,
                "threshold": 1.6,
                "evaluated_chain_cells": len(partial_records),
                "required_chain_cells": len(PROJECTIONS) * len(RUNGS),
                "reason": "partial timing cannot pass registered full pilot",
            },
            "G5": {
                "gating": False, "wins": arm_counts["refine"],
                "eligible_cells": (
                    (len(RUNGS) - 1) * len(PILOT_EXPERTS) * len(PROJECTIONS)
                ),
                "evaluated_eligible_cells": eligible_evaluated,
                "win_rate": (
                    arm_counts["refine"] / eligible_evaluated
                    if eligible_evaluated else 0.0
                ),
                "median_gain": (
                    statistics.median(refine_gains) if refine_gains else 0.0
                ),
                "p95_gain": (
                    percentile(refine_gains, 0.95) if refine_gains else 0.0
                ),
                "status": "informative_partial",
            },
        },
    }
    atomic_json(CHAIN_ROOT / "MINCHAIN_PILOT.json", report)
    atomic_text(RUN_ROOT / "MINCHAIN_PILOT.md", _markdown(report))
    print(
        "[minchain] mandatory replay-integrity FAIL -> INCUMBENT", flush=True
    )
    return 2


def run() -> int:
    if os.environ.get("PRISMAQUANT_CB_MINCHAIN_PILOT") != "1":
        raise SystemExit("set PRISMAQUANT_CB_MINCHAIN_PILOT=1")
    if not torch.cuda.is_available():
        raise SystemExit("min-chain pilot requires CUDA")
    os.environ["PRISMAQUANT_CB_LDLQ"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"] = "16"
    os.environ["PRISMAQUANT_CB_ENCODE_TIER"] = "balanced"
    device = torch.device("cuda:0")
    _, layer_record = load_layer_identity(PILOT_LAYER)
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    from prismaquant.layer_streaming import _build_fp8_scale_inv_map
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))

    all_records: dict[str, dict[int, dict]] = {}
    free_seconds_prorated = 0.0
    added_seconds = 0.0
    materialization_seconds = 0.0
    monotone_violations = 0
    tax_violations = 0
    max_tax = 0.0
    refine_gains: list[float] = []
    arm_counts = {"free": 0, "embed": 0, "refine": 0}

    for projection in PROJECTIONS:
        print(f"[minchain] load layer 21 {projection}", flush=True)
        full = load_projection(
            PILOT_LAYER, projection, device=device,
            identity=layer_record["identity"], all_col_weights=all_col_weights,
            model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
            scale_map=scale_map,
        )
        index = torch.tensor(PILOT_EXPERTS, device=device)
        weight = full["weight"].index_select(0, index).contiguous()
        col_weights = full["col_weights"].index_select(0, index).contiguous()
        activation_rows = tuple(full["activation_rows"][expert] for expert in PILOT_EXPERTS)
        del full
        torch.cuda.empty_cache()
        all_records[projection] = {}
        predecessors: list[dict] | None = None
        predecessor_ids: list[dict] | None = None
        predecessor_errors: list[float] | None = None

        for rung in RUNGS:
            shard_path = CHAIN_SHARDS / f"layer_021_{projection}_K{rung}.pkl"
            if shard_path.is_file():
                record = pickle.loads(shard_path.read_bytes())
                if (record.get("schema") != MINCHAIN_SCHEMA
                        or record.get("expert_ids") != list(PILOT_EXPERTS)
                        or record.get("rung") != rung):
                    raise AssertionError(f"stale min-chain shard {shard_path}")
                predecessors = _load_state(
                    Path(record["state_path"]), record["state_sha256"], device
                )
                predecessor_ids = list(record["identity"])
                predecessor_errors = list(record["selected_error"])
                print(f"[minchain] resume {projection} K{rung}", flush=True)
            else:
                phase_path = RUN_ROOT / "pilot-shards" / f"layer_021_{projection}_K{rung}.pkl"
                if not phase_path.is_file():
                    raise AssertionError(f"phase-A truth is incomplete: {phase_path}")
                phase = pickle.loads(phase_path.read_bytes())
                free_solutions, materialized_errors, materialized_time = (
                    _isolated_materialize_free(
                        projection=projection, rung=rung,
                        phase_path=phase_path, device=device,
                    )
                )
                materialization_seconds += materialized_time
                banked_free = [
                    float(phase["weight_mse_per_expert"][expert])
                    for expert in PILOT_EXPERTS
                ]
                for expert, observed, banked in zip(
                    PILOT_EXPERTS, materialized_errors, banked_free
                ):
                    rel = abs(observed - banked) / max(abs(banked), 1e-30)
                    if rel > 2e-6:
                        raise AssertionError(
                            f"{projection} K{rung} expert {expert}: warm materialization "
                            f"differs from banked free truth by {rel:.3e}"
                        )
                if rung == RUNGS[0]:
                    selected = [_compact(solution) for solution in free_solutions]
                    selected_error = banked_free
                    identities = [
                        chain_identity(
                            winning_arm="free", solution=solution,
                            predecessor_digest=None,
                        ) for solution in selected
                    ]
                    arms = ["free"] * len(PILOT_EXPERTS)
                    embed_errors = [math.nan] * len(PILOT_EXPERTS)
                    refine_errors = [math.nan] * len(PILOT_EXPERTS)
                    embed_seconds = refine_seconds = 0.0
                else:
                    assert predecessors is not None
                    assert predecessor_ids is not None
                    assert predecessor_errors is not None
                    embedded = [
                        embed_predecessor(solution, rung)
                        for solution in predecessors
                    ]
                    embed_errors, embed_seconds = _evaluate_all(
                        weight, embedded, rung
                    )
                    for expert, observed, prior in zip(
                        PILOT_EXPERTS, embed_errors, predecessor_errors
                    ):
                        rel = abs(observed - prior) / max(abs(prior), 1e-30)
                        if rel > 2e-6:
                            raise AssertionError(
                                f"{projection} K{rung} expert {expert}: "
                                f"predecessor embed changed metric by {rel:.3e}"
                            )
                    refined, refine_errors, refine_seconds = _refine_all(
                        weight=weight, col_weights=col_weights,
                        activation_rows=activation_rows,
                        predecessors=predecessors, rung=rung,
                    )
                    added_seconds += embed_seconds + refine_seconds
                    selected, selected_error, identities, arms = [], [], [], []
                    for local, expert in enumerate(PILOT_EXPERTS):
                        arm, error = select_arm({
                            "free": banked_free[local],
                            "embed": embed_errors[local],
                            "refine": refine_errors[local],
                        })
                        candidates = {
                            "free": free_solutions[local],
                            "embed": embedded[local],
                            "refine": refined[local],
                        }
                        chosen = _compact(candidates[arm])
                        selected.append(chosen)
                        selected_error.append(error)
                        arms.append(arm)
                        predecessor_digest = (
                            predecessor_ids[local]["solution_digest"]
                            if arm in {"embed", "refine"} else None
                        )
                        identities.append(chain_identity(
                            winning_arm=arm, solution=chosen,
                            predecessor_digest=predecessor_digest,
                        ))
                        if error > predecessor_errors[local]:
                            monotone_violations += 1
                        excess = error - banked_free[local]
                        if excess > max(abs(banked_free[local]), 1e-30) * 1e-12:
                            tax_violations += 1
                            max_tax = max(max_tax, excess)
                        gain = (embed_errors[local] - refine_errors[local]) / max(
                            abs(embed_errors[local]), 1e-30
                        )
                        if arm == "refine":
                            refine_gains.append(gain)
                    del embedded, refined
                for arm in arms:
                    arm_counts[arm] += 1
                free_seconds_prorated += float(phase["elapsed_seconds"]) * (
                    len(PILOT_EXPERTS) / 256.0
                )
                record = {
                    "schema": MINCHAIN_SCHEMA,
                    "chain_version": MINCHAIN_CONTEXT_VERSION,
                    "created_at": utc_now(),
                    "layer": PILOT_LAYER,
                    "projection": projection,
                    "rung": rung,
                    "expert_ids": list(PILOT_EXPERTS),
                    "free_truth_shard": str(phase_path),
                    "free_truth_sha256": sha256_file(phase_path),
                    "free_error": banked_free,
                    "embed_error": embed_errors,
                    "refine_error": refine_errors,
                    "selected_error": selected_error,
                    "winning_arm": arms,
                    "identity": identities,
                    "serialization_context": {
                        "chain_version": MINCHAIN_CONTEXT_VERSION,
                        "slice_group_identities": identities,
                    },
                    "timing": {
                        "banked_free_full_stack_seconds": float(phase["elapsed_seconds"]),
                        "banked_free_prorated_seconds": float(phase["elapsed_seconds"]) * len(PILOT_EXPERTS) / 256.0,
                        "free_warm_materialization_seconds": materialized_time,
                        "embed_seconds": embed_seconds,
                        "refine_seconds": refine_seconds,
                    },
                }
                _save_cell(projection, rung, record, selected)
                predecessors = selected
                predecessor_ids = identities
                predecessor_errors = selected_error
                print(
                    f"[minchain] wrote {projection} K{rung} "
                    f"arms={dict((a, arms.count(a)) for a in set(arms))}",
                    flush=True,
                )
            all_records[projection][rung] = record
        del weight, col_weights, predecessors
        torch.cuda.empty_cache()

    # Recompute timing/count statistics from immutable shards so resumes are
    # numerically identical to uninterrupted execution.
    free_seconds_prorated = added_seconds = materialization_seconds = 0.0
    monotone_violations = tax_violations = 0
    max_tax = 0.0
    refine_gains = []
    arm_counts = {"free": 0, "embed": 0, "refine": 0}
    previous: dict[tuple[str, int], float] = {}
    for projection in PROJECTIONS:
        for rung in RUNGS:
            record = all_records[projection][rung]
            timing = record["timing"]
            free_seconds_prorated += timing["banked_free_prorated_seconds"]
            materialization_seconds += timing["free_warm_materialization_seconds"]
            added_seconds += timing["embed_seconds"] + timing["refine_seconds"]
            for local, arm in enumerate(record["winning_arm"]):
                arm_counts[arm] += 1
                selected = float(record["selected_error"][local])
                free = float(record["free_error"][local])
                if rung != RUNGS[0] and selected > previous[(projection, local)] + max(abs(previous[(projection, local)]), 1e-30) * 1e-12:
                    monotone_violations += 1
                excess = selected - free
                if excess > max(abs(free), 1e-30) * 1e-12:
                    tax_violations += 1
                    max_tax = max(max_tax, excess)
                if arm == "refine":
                    embed = float(record["embed_error"][local])
                    refine = float(record["refine_error"][local])
                    refine_gains.append((embed - refine) / max(abs(embed), 1e-30))
                previous[(projection, local)] = selected
    if monotone_violations or tax_violations:
        raise AssertionError(
            f"min-chain construction failed: monotone={monotone_violations}, "
            f"tax={tax_violations}"
        )
    fit = _fit_gate(all_records)
    overhead = (free_seconds_prorated + added_seconds) / free_seconds_prorated
    mandatory_pass = (
        monotone_violations == 0
        and tax_violations == 0
        and fit["pass"]
        and overhead <= 1.6
    )
    report = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "decision": "ALL MANDATORY GATES PASS" if mandatory_pass else "ONE OR MORE MANDATORY GATES FAIL",
        "burn_method": "MIN-CHAIN STRICT" if mandatory_pass else "INCUMBENT",
        "expert_ids": list(PILOT_EXPERTS),
        "selection_metric": "weight_mse",
        "serialization": {
            "chain_schema": MINCHAIN_SCHEMA,
            "chain_version": MINCHAIN_CONTEXT_VERSION,
            "base_context": CONTEXT.to_dict() if hasattr(CONTEXT, "to_dict") else str(CONTEXT),
        },
        "arm_counts": arm_counts,
        "gates": {
            "G1": {"pass": monotone_violations == 0, "violations": monotone_violations},
            "G2": {"pass": tax_violations == 0, "violations": tax_violations, "max_excess": max_tax},
            "G3": fit,
            "G4": {
                "pass": overhead <= 1.6,
                "free_prorated_seconds": free_seconds_prorated,
                "added_arm_seconds": added_seconds,
                "free_materialization_seconds_excluded": materialization_seconds,
                "added_over_free_ratio": added_seconds / free_seconds_prorated,
                "total_over_free_ratio": overhead,
                "threshold": 1.6,
            },
            "G5": {
                "gating": False,
                "wins": arm_counts["refine"],
                "eligible_cells": (len(RUNGS) - 1) * len(PILOT_EXPERTS) * len(PROJECTIONS),
                "win_rate": arm_counts["refine"] / ((len(RUNGS) - 1) * len(PILOT_EXPERTS) * len(PROJECTIONS)),
                "median_gain": statistics.median(refine_gains) if refine_gains else 0.0,
                "p95_gain": percentile(refine_gains, 0.95) if refine_gains else 0.0,
            },
        },
    }
    atomic_json(CHAIN_ROOT / "MINCHAIN_PILOT.json", report)
    atomic_text(RUN_ROOT / "MINCHAIN_PILOT.md", _markdown(report))
    print(f"[minchain] {report['decision']} -> {report['burn_method']}", flush=True)
    return 0 if mandatory_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    child = subparsers.add_parser("materialize-free")
    child.add_argument("--projection", required=True)
    child.add_argument("--rung", required=True, type=int)
    child.add_argument("--phase-path", required=True, type=Path)
    child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "materialize-free":
        return _materialize_child(
            projection=args.projection, rung=args.rung,
            phase_path=args.phase_path, output=args.output,
        )
    try:
        return run()
    except FreeReplayError as failure:
        return _replay_failure_report(failure)


if __name__ == "__main__":
    raise SystemExit(main())
