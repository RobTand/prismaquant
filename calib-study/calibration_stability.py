#!/usr/bin/env python3
"""CPU-only calibration stability study for the DSv4 Flash 92 GB run.

The activation cache stores input rows but not per-token Fisher contributions or
per-row quantized output errors.  Consequently this module resamples the stored
input second moment and uses its relative change to perturb the production
``0.5 * h_trace * output_mse`` price.  It never claims to recompute the CE
Fisher or output MSE exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FRACTIONS = (0.25, 0.50, 0.75)
HOT_K = (100, 500, 1775)
FORMATS = ("NVFP4_CB_K14", "NVFP4_CB_K15", "FP8_CB_K36", "BF16")
EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)$"
)


def jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    """Jaccard overlap, defining two empty sets as identical."""
    a, b = set(left), set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def allocation_churn(reference: Sequence[object], candidate: Sequence[object]) -> float:
    """Fraction of positions whose assignment differs."""
    if len(reference) != len(candidate):
        raise ValueError("assignments must have equal length")
    if not reference:
        return 0.0
    return sum(a != b for a, b in zip(reference, candidate)) / len(reference)


def subsample_means(
    values: Sequence[float], fraction: float, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """Random fixed-size subset means without replacement.

    The subset size is nearest-integer half-up, bounded to ``[1, n]``.  This
    intentionally estimates stability conditional on the rows present in the
    cache; route-presence uncertainty is handled separately by the coverage
    model.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    m = min(len(x), max(1, int(math.floor(len(x) * fraction + 0.5))))
    if m == len(x):
        return np.full(repeats, float(x.mean()), dtype=np.float64)
    out = np.empty(repeats, dtype=np.float64)
    for b in range(repeats):
        out[b] = float(x[rng.choice(len(x), size=m, replace=False)].mean())
    return out


def summarize(samples: Sequence[float]) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left), np.asarray(right)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def canonical_format(entry: dict | str | int) -> str:
    """Small torch-free subset of prismaquant.layer_config.canonicalize_format."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, int):
        return "BF16" if entry == 16 else str(entry)
    dt = entry.get("data_type")
    bits = int(entry.get("bits", 0))
    if dt == "nvfp4_cb":
        return f"NVFP4_CB_K{int(entry['cb_k'])}"
    if dt == "fp8_cb":
        return f"FP8_CB_K{int(entry['cb_k'])}"
    if dt == "fp4_e2m1" and bits == 4:
        return "MXFP4_SOURCE"
    if dt == "fp8_e4m3" and bits == 8 and int(entry.get("group_size", 0)) == 128:
        return "FP8_BLOCK_UE8M0_SOURCE"
    if dt in {"float", "bfloat16"} and bits in {0, 16}:
        return "BF16"
    raise ValueError(f"unsupported study layer-config entry: {entry!r}")


def format_payload_bits(fmt: str, stat: dict) -> float:
    """Exact per-tensor serialized bits for the four study menu formats.

    Shared codebook sidecars are fixed and excluded from the lambda solve.
    """
    n = int(stat["n_params"])
    out_features = int(stat["out_features"])
    if fmt == "NVFP4_CB_K14":
        return n * 2.03125 + 32.0
    if fmt == "NVFP4_CB_K15":
        return n * 2.15625 + 32.0
    if fmt == "FP8_CB_K36":
        return n * 4.5 + out_features * 32.0
    if fmt == "BF16":
        return n * 16.0
    raise KeyError(fmt)


def lambda_allocate(
    costs: np.ndarray, bits: np.ndarray, budget_bits: float, iterations: int = 48
) -> tuple[np.ndarray, float, float, float]:
    """Per-row Lagrangian argmin with bisection to the feasible side."""
    costs = np.asarray(costs, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.float64)
    if costs.shape != bits.shape or costs.ndim != 2:
        raise ValueError("costs and bits must have the same 2-D shape")

    def choose(lam: float) -> tuple[np.ndarray, float]:
        assignment = np.argmin(costs + lam * bits, axis=1)
        total = float(bits[np.arange(len(bits)), assignment].sum())
        return assignment, total

    lo, hi = 0.0, 1e-18
    feasible, feasible_bits = choose(hi)
    while feasible_bits > budget_bits:
        hi *= 2.0
        feasible, feasible_bits = choose(hi)
        if hi > 1e18:
            raise RuntimeError("could not bracket a feasible lambda")
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        candidate, candidate_bits = choose(mid)
        if candidate_bits <= budget_bits:
            hi, feasible, feasible_bits = mid, candidate, candidate_bits
        else:
            lo = mid
    objective = float(costs[np.arange(len(costs)), feasible].sum())
    return feasible, feasible_bits, objective, hi


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _quantile_rows(fraction: float, metric: str, values: Sequence[float]) -> dict:
    s = summarize(values)
    return {"fraction": fraction, "metric": metric, **s}


def _svg_line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    x_label: str,
    y_label: str,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Write a dependency-free SVG line chart."""
    width, height = 760, 440
    left, right, top, bottom = 78, 25, 48, 64
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points if math.isfinite(y)]
    xmin, xmax = min(xs), max(xs)
    ymin = min(ys) if y_min is None else y_min
    ymax = max(ys) if y_max is None else y_max
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    pad = 0.04 * (ymax - ymin)
    if y_min is None:
        ymin -= pad
    if y_max is None:
        ymax += pad

    def px(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def py(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    for tick in range(6):
        x = xmin + (xmax - xmin) * tick / 5
        y = ymin + (ymax - ymin) * tick / 5
        body.append(f'<text x="{px(x):.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.3g}</text>')
        body.append(f'<text x="{left-9}" y="{py(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>')
        body.append(f'<line x1="{left}" y1="{py(y):.1f}" x2="{width-right}" y2="{py(y):.1f}" stroke="#ddd"/>')
    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points if math.isfinite(y))
        body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            if math.isfinite(y):
                body.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        body.append(f'<text x="{width-right-5}" y="{top+16*idx}" text-anchor="end" font-family="sans-serif" font-size="11" fill="{color}">{label}</text>')
    body.extend([
        f'<text x="{(left+width-right)/2}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="12">{y_label}</text>',
        "</svg>",
    ])
    path.write_text("\n".join(body) + "\n")


def _load_reference_inputs(args: argparse.Namespace) -> dict:
    with args.cost.open("rb") as fh:
        cost = pickle.load(fh)
    with args.probe.open("rb") as fh:
        probe = pickle.load(fh)
    layer_config = json.loads(args.layer_config.read_text())
    selection = json.loads(args.selection.read_text())
    return {"cost": cost, "probe": probe, "layer_config": layer_config, "selection": selection}


def _scan_activation_energy(act_dir: Path, expected_names: set[str]) -> tuple[dict[str, np.ndarray], dict]:
    # Hide every GPU before importing torch.  The study uses CPU tensors only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    energies: dict[str, np.ndarray] = {}
    inventory_hash = hashlib.sha256()
    files = sorted(act_dir.glob("*.pt"))
    total_bytes = 0
    row_index_nonmonotonic = 0
    for number, path in enumerate(files, 1):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "inputs" not in payload or "name" not in payload:
            raise ValueError(f"unexpected activation payload: {path}")
        name = str(payload["name"])
        x = payload["inputs"]
        if name in energies:
            raise ValueError(f"duplicate activation qname: {name}")
        row_energy = x.float().square().mean(dim=1).numpy().astype(np.float64, copy=True)
        row_indices = payload.get("row_indices")
        if row_indices is not None:
            idx = row_indices.numpy()
            if len(idx) > 1 and np.any(idx[1:] < idx[:-1]):
                row_index_nonmonotonic += 1
        energies[name] = row_energy
        size = path.stat().st_size
        total_bytes += size
        inventory_hash.update(f"{path.name}\0{size}\0{name}\0{len(row_energy)}\n".encode())
        inventory_hash.update(row_energy.tobytes())
        if number % 2000 == 0:
            print(f"[calib-study] activation energy {number}/{len(files)}", flush=True)
    unexpected = sorted(set(energies) - expected_names)
    if unexpected:
        raise ValueError(f"activation cache has {len(unexpected)} unexpected qnames")
    return energies, {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "derived_inventory_sha256": inventory_hash.hexdigest(),
        "qnames_missing_files": len(expected_names - set(energies)),
        "row_index_nonmonotonic_files": row_index_nonmonotonic,
    }


def _build_baseline(data: dict) -> dict:
    costs = data["cost"]["costs"]
    stats = data["probe"]["stats"]
    layer_config = data["layer_config"]
    names = sorted(costs)
    n = len(names)
    h_trace = np.array([float(stats[name]["h_trace"]) for name in names])
    n_params = np.array([int(stats[name]["n_params"]) for name in names], dtype=np.int64)
    shipped_formats: list[str] = []
    shipped_dloss = np.zeros(n, dtype=np.float64)
    cost_matrix = np.zeros((n, len(FORMATS)), dtype=np.float64)
    bits_matrix = np.zeros_like(cost_matrix)
    activation_rows = np.zeros(n, dtype=np.int64)
    routed_rows = np.array([int(stats[name].get("n_tokens_seen", 0)) for name in names], dtype=np.int64)
    for i, name in enumerate(names):
        shipped = canonical_format(layer_config[name])
        shipped_formats.append(shipped)
        for j, fmt in enumerate(FORMATS):
            entry = costs[name][fmt]
            cost_matrix[i, j] = 0.5 * h_trace[i] * float(entry.get("output_mse", 0.0))
            bits_matrix[i, j] = format_payload_bits(fmt, stats[name])
        first = costs[name][FORMATS[0]]
        activation_rows[i] = int(first.get("n_activation_rows", 0))
        if shipped in costs[name]:
            shipped_dloss[i] = 0.5 * h_trace[i] * float(costs[name][shipped].get("output_mse", 0.0))
    return {
        "names": names,
        "name_index": {name: i for i, name in enumerate(names)},
        "h_trace": h_trace,
        "n_params": n_params,
        "shipped_formats": shipped_formats,
        "shipped_dloss": shipped_dloss,
        "cost_matrix": cost_matrix,
        "bits_matrix": bits_matrix,
        "activation_rows": activation_rows,
        "routed_rows": routed_rows,
    }


def _bootstrap_ratios(
    energies: dict[str, np.ndarray],
    names: list[str],
    routed_rows: np.ndarray,
    fraction: float,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized fixed-size row subsets plus binomial route thinning."""
    rng = np.random.default_rng(seed)
    ratios = np.ones((repeats, len(names)), dtype=np.float64)
    relative_std = np.zeros(len(names), dtype=np.float64)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, name in enumerate(names):
        if name in energies:
            groups[len(energies[name])].append(i)
    for n_rows, indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        matrix = np.stack([energies[names[i]] for i in indices])
        full = matrix.mean(axis=1)
        subset_n = min(n_rows, max(1, int(math.floor(n_rows * fraction + 0.5))))
        means = np.empty((repeats, len(indices)), dtype=np.float64)
        if subset_n == n_rows:
            means[:] = full
        else:
            for b in range(repeats):
                keys = rng.random(matrix.shape)
                chosen = np.argpartition(keys, subset_n - 1, axis=1)[:, :subset_n]
                means[b] = np.take_along_axis(matrix, chosen, axis=1).mean(axis=1)
        safe = np.where(full != 0, full, 1.0)
        group_ratios = means / safe
        group_ratios[:, full == 0] = 1.0
        ratios[:, indices] = group_ratios
        denom = np.abs(means.mean(axis=0))
        relative_std[indices] = np.divide(
            means.std(axis=0, ddof=1), denom,
            out=np.zeros_like(denom), where=denom > 0,
        )
    counts = np.asarray(routed_rows, dtype=np.int64)
    route_ratios = np.ones_like(ratios)
    positive = counts > 0
    thinned = rng.binomial(counts[positive][None, :], fraction, size=(repeats, int(positive.sum())))
    route_ratios[:, positive] = thinned / (fraction * counts[positive][None, :])
    combined = ratios * route_ratios
    combined_mean = combined.mean(axis=0)
    combined_relative_std = np.divide(
        combined.std(axis=0, ddof=1), np.abs(combined_mean),
        out=np.zeros_like(combined_mean), where=combined_mean != 0,
    )
    return combined, relative_std, combined_relative_std


def _hotset_results(base: dict, ratios: np.ndarray, fraction: float) -> tuple[dict, list[dict]]:
    baseline = base["shipped_dloss"]
    names = base["names"]
    full_order = np.lexsort((np.asarray(names), -baseline))
    by_k = {}
    rows: list[dict] = []
    for k in HOT_K:
        full = set(full_order[:k].tolist())
        overlaps, concentrations = [], []
        for b in range(len(ratios)):
            loss = baseline * ratios[b]
            order = np.lexsort((np.asarray(names), -loss))
            overlaps.append(jaccard(full, set(order[:k].tolist())))
            concentrations.append(float(loss[order[:k]].sum() / loss.sum()))
        by_k[str(k)] = {
            "jaccard": summarize(overlaps),
            "concentration_fraction": summarize(concentrations),
        }
        rows.append({"fraction": fraction, "k": k, **{f"jaccard_{x}": y for x, y in summarize(overlaps).items()}, **{f"concentration_{x}": y for x, y in summarize(concentrations).items()}})
    return by_k, rows


def _allocation_results(
    base: dict,
    ratios: np.ndarray,
    fraction: float,
    budget_bits: float,
    full_assignment: np.ndarray,
    full_objective: float,
) -> tuple[dict, list[dict]]:
    churn, delta, relative_delta, achieved, proxy_delta = [], [], [], [], []
    arange = np.arange(len(full_assignment))
    full_costs = base["cost_matrix"]
    bits = base["bits_matrix"]
    for ratio in ratios:
        proxy_costs = full_costs * ratio[:, None]
        assignment, total_bits, proxy_objective, _ = lambda_allocate(proxy_costs, bits, budget_bits)
        evaluated = float(full_costs[arange, assignment].sum())
        churn.append(float(np.mean(assignment != full_assignment)))
        delta.append(evaluated - full_objective)
        relative_delta.append((evaluated - full_objective) / full_objective if full_objective else 0.0)
        achieved.append(total_bits)
        proxy_delta.append(proxy_objective - float(proxy_costs[arange, full_assignment].sum()))
    result = {
        "churn_fraction": summarize(churn),
        "full_data_predicted_dloss_delta": summarize(delta),
        "full_data_predicted_dloss_relative_delta": summarize(relative_delta),
        "achieved_payload_bits": summarize(achieved),
        "resampled_objective_delta_vs_full_assignment": summarize(proxy_delta),
    }
    row = {"fraction": fraction}
    for metric, summary in result.items():
        for key, value in summary.items():
            row[f"{metric}_{key}"] = value
    return result, [row]


def _nb_log_likelihood(hist: Counter[int], mean: float, r: float) -> float:
    p = r / (r + mean)
    logp, logq = math.log(p), math.log1p(-p)
    total = 0.0
    for n, count in hist.items():
        total += count * (
            math.lgamma(n + r) - math.lgamma(r) - math.lgamma(n + 1)
            + r * logp + n * logq
        )
    return total


def fit_negative_binomial(counts: Sequence[int]) -> tuple[float, float]:
    """Fit NB(mean, dispersion r) by 1-D MLE; counts are uncensored."""
    x = np.asarray(counts, dtype=np.int64)
    mean = float(x.mean())
    hist = Counter(int(v) for v in x)
    lo, hi = -10.0, 12.0
    phi = (1 + math.sqrt(5)) / 2
    c = hi - (hi - lo) / phi
    d = lo + (hi - lo) / phi
    fc = _nb_log_likelihood(hist, mean, math.exp(c))
    fd = _nb_log_likelihood(hist, mean, math.exp(d))
    for _ in range(80):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) / phi
            fc = _nb_log_likelihood(hist, mean, math.exp(c))
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) / phi
            fd = _nb_log_likelihood(hist, mean, math.exp(d))
    return mean, math.exp((lo + hi) / 2)


def nb_cdf(k: int, mean: float, r: float) -> float:
    if k < 0:
        return 0.0
    p = r / (r + mean)
    pmf = p**r
    total = pmf
    for n in range(k):
        pmf *= (n + r) / (n + 1) * (1 - p)
        total += pmf
    return min(1.0, total)


def nb_coverage(mean: float, r: float, multiplier: float, threshold: int) -> float:
    return 1.0 - nb_cdf(threshold - 1, mean * multiplier, r)


def required_multiplier(mean: float, r: float, threshold: int = 8, target: float = 0.95) -> float:
    lo, hi = 0.0, 1.0
    while nb_coverage(mean, r, hi, threshold) < target:
        hi *= 2
        if hi > 1e6:
            return float("inf")
    for _ in range(80):
        mid = (lo + hi) / 2
        if nb_coverage(mean, r, mid, threshold) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _binomial_tail(n: int, probability: float, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    if n < threshold:
        return 0.0
    if probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return 0.0
    # The study needs thresholds 1 and 8.  Sum the short lower tail instead
    # of materialising thousands of huge binomial coefficients.
    q = 1.0 - probability
    pmf = q**n
    lower = pmf
    for k in range(threshold - 1):
        pmf *= (n - k) / (k + 1) * probability / q
        lower += pmf
    return max(0.0, min(1.0, 1.0 - lower))


def _coverage_results(base: dict, repeats: int, seed: int) -> tuple[dict, list[dict]]:
    units: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for name, count in zip(base["names"], base["routed_rows"]):
        match = EXPERT_RE.fullmatch(name)
        if match:
            units[(int(match[1]), int(match[2]))][match[3]] = int(count)
    if len(units) != 43 * 256 or any(len(v) != 3 for v in units.values()):
        raise ValueError("expected 43 x 256 complete expert units")
    projection_mismatches = sum(len(set(v.values())) != 1 for v in units.values())
    matrix = np.array([[units[(layer, expert)]["gate_proj"] for expert in range(256)] for layer in range(43)], dtype=np.int64)
    counts = matrix.ravel()
    mean, dispersion = fit_negative_binomial(counts)
    required = required_multiplier(mean, dispersion)

    rng = np.random.default_rng(seed)
    required_samples = []
    fit_means, fit_dispersion = [], []
    for _ in range(max(200, repeats)):
        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))].ravel()
        sample_mean, sample_dispersion = fit_negative_binomial(sampled)
        fit_means.append(sample_mean)
        fit_dispersion.append(sample_dispersion)
        required_samples.append(required_multiplier(sample_mean, sample_dispersion))

    multipliers = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    while multipliers[-1] < required * 1.25:
        multipliers.append(multipliers[-1] * 2)
    curve: list[dict] = []
    for scale in multipliers:
        row = {
            "calibration_multiplier": scale,
            "samples_512": scale * 16,
            "tokens": scale * 8192,
            "nb_fraction_ge_1": nb_coverage(mean, dispersion, scale, 1),
            "nb_fraction_ge_8": nb_coverage(mean, dispersion, scale, 8),
        }
        if scale <= 1:
            row["empirical_binomial_thinning_fraction_ge_1"] = float(np.mean([_binomial_tail(int(n), scale, 1) for n in counts]))
            row["empirical_binomial_thinning_fraction_ge_8"] = float(np.mean([_binomial_tail(int(n), scale, 8) for n in counts]))
        else:
            row["empirical_binomial_thinning_fraction_ge_1"] = None
            row["empirical_binomial_thinning_fraction_ge_8"] = None
        curve.append(row)

    req_summary = summarize(required_samples)
    result = {
        "expert_units": len(counts),
        "projection_count_mismatches": projection_mismatches,
        "route_count_source": "probe.stats[n_tokens_seen] (uncapped; superior to capped n_activation_rows)",
        "current": {
            "samples": 16,
            "seqlen": 512,
            "tokens": 8192,
            "mean_routed_rows_per_expert": float(counts.mean()),
            "median_routed_rows_per_expert": float(np.median(counts)),
            "max_routed_rows_per_expert": int(counts.max()),
            "fraction_ge_1": float(np.mean(counts >= 1)),
            "fraction_ge_8": float(np.mean(counts >= 8)),
            "fraction_ge_64": float(np.mean(counts >= 64)),
        },
        "negative_binomial_fit": {
            "mean": mean,
            "dispersion_r": dispersion,
            "observed_fraction_zero": float(np.mean(counts == 0)),
            "fit_fraction_zero": 1.0 - nb_coverage(mean, dispersion, 1.0, 1),
            "observed_fraction_ge_8": float(np.mean(counts >= 8)),
            "fit_fraction_ge_8": nb_coverage(mean, dispersion, 1.0, 8),
            "adequacy_warning": "A single NB overpredicts zero experts and does not identify structural zeros; the required-size estimate is conditional and not a direct recipe.",
        },
        "required_for_95pct_ge_8": {
            "multiplier_point": required,
            "multiplier_layer_bootstrap": req_summary,
            "samples_512_point": math.ceil(16 * required),
            "samples_512_p05": math.ceil(16 * req_summary["p05"]),
            "samples_512_p95": math.ceil(16 * req_summary["p95"]),
            "tokens_point": math.ceil(8192 * required),
            "conditional_on_model": "Gamma-Poisson / negative-binomial expert-rate population; assumes no structural-zero experts and stationary corpus composition",
            "identifiability": "If more than 5% of experts are structural zeros for this corpus, 95% coverage is unattainable at any sample size. One aggregate count per expert cannot distinguish structural zeros from very rare experts.",
        },
        "fit_layer_bootstrap": {
            "mean": summarize(fit_means),
            "dispersion_r": summarize(fit_dispersion),
            "repeats": max(200, repeats),
        },
        "curve": curve,
        "cap_caveat": "The activation cache caps rows at 64, but coverage uses uncapped probe n_tokens_seen. Activation-side stability above 64 remains censored by the cache.",
    }
    return result, curve


def _corpus_inventory(calibration_dir: Path, recipe_dataset: Path) -> dict:
    files = []
    recipe_manifest = None
    for path in sorted(calibration_dir.glob("*")):
        if not path.is_file():
            continue
        item = {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if path.suffix == ".jsonl":
            records = 0
            chars = 0
            words = 0
            manifest = None
            tagged_records = 0
            with path.open() as fh:
                for line in fh:
                    obj = json.loads(line)
                    if "__manifest__" in obj:
                        manifest = obj["__manifest__"]
                        continue
                    text = obj.get("text")
                    if isinstance(text, str):
                        records += 1
                        chars += len(text)
                        words += len(text.split())
                    if any(key in obj for key in ("domain", "source", "source_id", "sample_id")):
                        tagged_records += 1
            item.update({"text_records": records, "characters": chars, "whitespace_words": words, "tagged_records": tagged_records, "manifest": manifest})
            if path.resolve() == recipe_dataset.resolve():
                recipe_manifest = manifest
        files.append(item)
    return {
        "directory": str(calibration_dir),
        "files": files,
        "recipe_dataset": str(recipe_dataset),
        "recipe_manifest": recipe_manifest,
        "recoverable_domain_analysis": False,
        "reason": "The corpus manifest records only aggregate mix; text records and activation payloads have no domain/source/sample identifier, and expert row_indices are local to each Linear rather than global sample-token ids.",
        "future_probe_metadata": [
            "stable sample_id and source_id/domain on every corpus record",
            "for each cached row: sample_id, sequence index, global token offset, source/domain",
            "for each expert route: layer, expert_id, sample_id, token offset, router probability/rank",
            "per-token Fisher contribution (or sufficient x^2 and output-gradient^2 factors)",
            "per-row output squared error for every candidate format",
        ],
    }


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    curves_dir = args.output / "curves"
    curves_dir.mkdir(exist_ok=True)
    data = _load_reference_inputs(args)
    base = _build_baseline(data)
    expected = set(base["names"])
    energies, act_inventory = _scan_activation_energy(args.act_dir, expected)

    selection_predicted = float(data["selection"]["predicted_dloss"])
    baseline_sum = float(base["shipped_dloss"].sum())
    full_order = np.lexsort((np.asarray(base["names"]), -base["shipped_dloss"]))
    concentration = {
        str(k): float(base["shipped_dloss"][full_order[:k]].sum() / baseline_sum)
        for k in HOT_K
    }
    hot_rows = [
        {
            "rank": rank + 1,
            "qname": base["names"][i],
            "format": base["shipped_formats"][i],
            "dloss": float(base["shipped_dloss"][i]),
            "h_trace": float(base["h_trace"][i]),
            "n_activation_rows": int(base["activation_rows"][i]),
            "n_tokens_seen": int(base["routed_rows"][i]),
        }
        for rank, i in enumerate(full_order[:100])
    ]
    hotset_route_support = {}
    for k in HOT_K:
        selected = full_order[:k]
        hotset_route_support[str(k)] = {
            "minimum_n_tokens_seen": int(base["routed_rows"][selected].min()),
            "median_n_tokens_seen": float(np.median(base["routed_rows"][selected])),
            "fraction_n_tokens_seen_ge_8": float(np.mean(base["routed_rows"][selected] >= 8)),
            "fraction_activation_cache_capped_at_64": float(np.mean(base["activation_rows"][selected] >= 64)),
        }

    full_energy = np.full(len(base["names"]), np.nan)
    for name, values in energies.items():
        full_energy[base["name_index"][name]] = float(values.mean())
    shared = np.isfinite(full_energy) & (base["h_trace"] > 0)
    proxy_spearman = spearman(full_energy[shared], base["h_trace"][shared])

    inventory = {
        "cpu_only": True,
        "reference_inputs": {
            "cost": {"path": str(args.cost), "bytes": args.cost.stat().st_size, "sha256": _sha256(args.cost)},
            "probe": {"path": str(args.probe), "bytes": args.probe.stat().st_size, "sha256": _sha256(args.probe)},
            "layer_config": {"path": str(args.layer_config), "bytes": args.layer_config.stat().st_size, "sha256": _sha256(args.layer_config)},
            "selection": {"path": str(args.selection), "bytes": args.selection.stat().st_size, "sha256": _sha256(args.selection)},
            "activation_dir": {"path": str(args.act_dir), **act_inventory},
        },
        "cost_rows": len(base["names"]),
        "cost_formats": list(data["cost"]["formats"]),
        "activation_rows_limit": int(data["probe"]["meta"]["activation_rows_limit"]),
        "activation_row_count_histogram": {str(k): v for k, v in sorted(Counter(base["activation_rows"]).items())},
        "probe_recipe": {k: data["probe"]["meta"].get(k) for k in ("dataset", "nsamples", "seqlen", "calib_hash", "dtype", "top_k", "fisher_norm_tokens", "packed_fisher_estimator")},
        "full_decomposition": {
            "formula": "0.5 * probe.stats[qname].h_trace * cost[assigned_format].output_mse; source passthroughs contribute zero",
            "sum": baseline_sum,
            "selection_json_predicted_dloss": selection_predicted,
            "absolute_reproduction_error": abs(baseline_sum - selection_predicted),
            "concentration_fraction": concentration,
            "hotset_route_support": hotset_route_support,
            "top_100_rows": hot_rows,
        },
        "activation_second_moment_vs_h_trace_spearman": proxy_spearman,
    }
    _json_dump(args.output / "data_inventory.json", inventory)

    meta = data["layer_config"]["__prismaquant__"]
    budget_bits = float(meta["body_assignment_payload_bits_total"] - meta.get("body_shared_cb_sidecar_bits", 0.0))
    full_assignment, full_bits, full_objective, full_lambda = lambda_allocate(base["cost_matrix"], base["bits_matrix"], budget_bits)

    estimator = {
        "repeats": args.repeats,
        "fractions": {},
        "proxy_definition": {
            "h_proxy": "mean over cached activation rows of mean(input**2 across features)",
            "subsample": "fixed-size random subset without replacement; m=max(1, nearest_half_up(fraction*n_cached))",
            "route_factor": "Binomial(n_tokens_seen, fraction) / (fraction * n_tokens_seen), using uncapped probe counts",
            "loss_proxy": "full production assigned-row dloss multiplied by activation-energy ratio and route_factor",
            "not_recomputed": ["CE-Fisher output-gradient factor", "per-row output_mse", "sample-correlated routing pattern"],
        },
        "full_proxy_vs_production_h_trace_spearman": proxy_spearman,
    }
    hotset = {
        "repeats": args.repeats,
        "full_data_concentration_fraction": concentration,
        "fractions": {},
    }
    allocation = {
        "repeats": args.repeats,
        "formats": list(FORMATS),
        "budget": {
            "payload_bits_excluding_fixed_shared_sidecars": budget_bits,
            "source": "b-92 layer_config body_assignment_payload_bits_total - body_shared_cb_sidecar_bits",
        },
        "full_simplified_allocation": {
            "payload_bits": full_bits,
            "predicted_dloss": full_objective,
            "lambda": full_lambda,
            "format_counts": {FORMATS[i]: int(np.sum(full_assignment == i)) for i in range(len(FORMATS))},
            "note": "Independent per-row Lagrangian solve; intentionally omits serving-atomic aggregation and fixed shared sidecars.",
        },
        "fractions": {},
    }
    estimator_fraction_rows: list[dict] = []
    estimator_n_rows: list[dict] = []
    hotset_rows: list[dict] = []
    allocation_rows: list[dict] = []

    for fraction_number, fraction in enumerate(FRACTIONS):
        print(f"[calib-study] bootstrap fraction={fraction:.2f}", flush=True)
        ratios, energy_relative_std, combined_relative_std = _bootstrap_ratios(
            energies, base["names"], base["routed_rows"], fraction, args.repeats,
            args.seed + fraction_number * 100003,
        )
        cached = base["activation_rows"] > 0
        positive_loss = base["shipped_dloss"] > 0
        fraction_summary = {
            "activation_energy_relative_std_all_cached": summarize(energy_relative_std[cached]),
            "activation_energy_relative_std_positive_dloss": summarize(energy_relative_std[positive_loss]),
            "combined_route_and_energy_relative_std_all_cached": summarize(combined_relative_std[cached]),
            "combined_route_and_energy_relative_std_positive_dloss": summarize(combined_relative_std[positive_loss]),
            "dloss_weighted_mean_combined_relative_std": float(np.average(combined_relative_std[positive_loss], weights=base["shipped_dloss"][positive_loss])),
            "singleton_cached_rows": int(np.sum(base["activation_rows"] == 1)),
            "singleton_note": "n=1 has conditional activation-energy RSD 0, but combined RSD includes route disappearance/count variation. Per-route contribution variance remains unidentifiable.",
        }
        estimator["fractions"][str(fraction)] = fraction_summary
        estimator_fraction_rows.append({
            "fraction": fraction,
            "median_activation_energy_relative_std": fraction_summary["activation_energy_relative_std_all_cached"]["median"],
            "p90_activation_energy_relative_std": float(np.quantile(energy_relative_std[cached], 0.90)),
            "median_combined_relative_std": fraction_summary["combined_route_and_energy_relative_std_all_cached"]["median"],
            "p90_combined_relative_std": float(np.quantile(combined_relative_std[cached], 0.90)),
            "dloss_weighted_mean_combined_relative_std": fraction_summary["dloss_weighted_mean_combined_relative_std"],
        })
        for n_rows in sorted(set(base["activation_rows"][cached])):
            mask = base["activation_rows"] == n_rows
            energy_vals = energy_relative_std[mask]
            combined_vals = combined_relative_std[mask]
            row = {
                "fraction": fraction,
                "n_activation_rows": int(n_rows),
                "unit_count": int(mask.sum()),
                "median_activation_energy_relative_std": float(np.median(energy_vals)),
                "p90_activation_energy_relative_std": float(np.quantile(energy_vals, 0.90)),
                "mean_activation_energy_relative_std": float(energy_vals.mean()),
                "median_combined_relative_std": float(np.median(combined_vals)),
                "p90_combined_relative_std": float(np.quantile(combined_vals, 0.90)),
                "mean_combined_relative_std": float(combined_vals.mean()),
            }
            estimator_n_rows.append(row)

        h_result, h_rows = _hotset_results(base, ratios, fraction)
        hotset["fractions"][str(fraction)] = h_result
        hotset_rows.extend(h_rows)
        a_result, a_rows = _allocation_results(
            base, ratios, fraction, budget_bits, full_assignment, full_objective
        )
        allocation["fractions"][str(fraction)] = a_result
        allocation_rows.extend(a_rows)

    _json_dump(args.output / "estimator_stability.json", estimator)
    _json_dump(args.output / "hotset_stability.json", hotset)
    _json_dump(args.output / "allocation_churn.json", allocation)
    _write_csv(curves_dir / "estimator_fraction.csv", estimator_fraction_rows)
    _write_csv(curves_dir / "estimator_by_n.csv", estimator_n_rows)
    _write_csv(curves_dir / "hotset_jaccard.csv", hotset_rows)
    _write_csv(curves_dir / "allocation_churn.csv", allocation_rows)

    coverage, coverage_rows = _coverage_results(base, args.repeats, args.seed + 700001)
    _json_dump(args.output / "coverage_model.json", coverage)
    _write_csv(curves_dir / "coverage.csv", coverage_rows)
    composition = _corpus_inventory(args.calibration_dir, Path(data["probe"]["meta"]["dataset"]))
    _json_dump(args.output / "composition.json", composition)

    _svg_line_chart(
        curves_dir / "hotset_jaccard.svg", "Hot-set identity stability",
        [(f"top-{k}", [(f, hotset["fractions"][str(f)][str(k)]["jaccard"]["median"]) for f in FRACTIONS]) for k in HOT_K],
        "fraction of cached rows", "median Jaccard", y_min=0.0, y_max=1.0,
    )
    _svg_line_chart(
        curves_dir / "allocation_churn.svg", "Simplified allocation churn",
        [("row churn", [(f, allocation["fractions"][str(f)]["churn_fraction"]["median"]) for f in FRACTIONS])],
        "fraction of cached rows", "fraction of formats changed", y_min=0.0,
    )
    _svg_line_chart(
        curves_dir / "estimator_fraction.svg", "Activation second-moment stability",
        [
            ("median combined RSD", [(r["fraction"], r["median_combined_relative_std"]) for r in estimator_fraction_rows]),
            ("p90 combined RSD", [(r["fraction"], r["p90_combined_relative_std"]) for r in estimator_fraction_rows]),
            ("dloss-weighted combined RSD", [(r["fraction"], r["dloss_weighted_mean_combined_relative_std"]) for r in estimator_fraction_rows]),
        ],
        "fraction of cached rows", "relative standard deviation", y_min=0.0,
    )
    _svg_line_chart(
        curves_dir / "coverage.svg", "Expert coverage model",
        [
            (">=1 row (NB)", [(r["calibration_multiplier"], r["nb_fraction_ge_1"]) for r in coverage_rows]),
            (">=8 rows (NB)", [(r["calibration_multiplier"], r["nb_fraction_ge_8"]) for r in coverage_rows]),
        ],
        "calibration size / current size", "fraction of expert units", y_min=0.0, y_max=1.0,
    )

    headline = hotset["fractions"]["0.5"]["100"]["jaccard"]
    churn50 = allocation["fractions"]["0.5"]["churn_fraction"]
    summary = {
        "top100_jaccard_at_50pct": headline,
        "allocation_churn_at_50pct": churn50,
        "current_expert_coverage_ge8": coverage["current"]["fraction_ge_8"],
        "required_samples_512_for_95pct_ge8": coverage["required_for_95pct_ge_8"],
        "proxy_spearman_vs_h_trace": proxy_spearman,
        "concentration_full": concentration,
        "verdict_inputs": {
            "hotset_threshold": 0.8,
            "hotset_pass_by_median": headline["median"] >= 0.8,
            "coverage_gate_95pct_ge8_currently_passes": coverage["current"]["fraction_ge_8"] >= 0.95,
            "exact_fisher_resampling_possible": False,
            "domain_resampling_possible": False,
        },
    }
    _json_dump(args.output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", type=Path, default=Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6/act"))
    parser.add_argument("--cost", type=Path, default=root / "artifacts/cost_full.pkl")
    parser.add_argument("--probe", type=Path, default=root / "artifacts-mxfp4/probe.pkl")
    parser.add_argument("--layer-config", type=Path, default=root / "artifacts-mxfp4/oldmenu-grid/b-92/layer_config.json")
    parser.add_argument("--selection", type=Path, default=root / "artifacts-mxfp4/oldmenu-grid/b-92/selection.json")
    parser.add_argument("--calibration-dir", type=Path, default=Path("/home/rob/dq-runs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("calib-study"))
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)
    if args.repeats < 30:
        parser.error("--repeats must be >= 30")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
