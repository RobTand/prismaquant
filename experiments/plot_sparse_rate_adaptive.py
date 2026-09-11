#!/usr/bin/env python3
"""Plot the measured sparse-rate curves and the adaptive interpolation audit.

This is a research figure: rates are one Linear's finite-band measurements and
the ordinate is the measured output MSE under the route activation contract.
It does not claim KL, serving throughput, or a continuous error bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCREEN_P99 = 1.0
SCREEN_MAX = 5.0
PRIMARY = "value interpolation; relative tolerance 0.005; 2 checks/interval"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def curve_values(path: Path, *, allow_singleton: bool = False) -> tuple[np.ndarray, np.ndarray]:
    value = read_json(path)
    rates, measurements = value.get("rates"), value.get("values")
    if not isinstance(rates, list) or not isinstance(measurements, list) or len(rates) != len(measurements):
        raise ValueError(f"{path}: curve must contain matching rates and values")
    if len(rates) < (1 if allow_singleton else 2):
        needed = "one" if allow_singleton else "two"
        raise ValueError(f"{path}: curve needs at least {needed} measured points")
    return np.asarray(rates, dtype=float), np.asarray(measurements, dtype=float)


def validate_report(report: dict, report_path: Path, curve_path: Path) -> None:
    expected = sha256(curve_path)
    if report.get("curve_sha256") != expected:
        raise ValueError(f"{report_path}: curve_sha256 does not match supplied curve")
    result = report.get("result", {})
    policy = (result.get("mode"), result.get("tolerance"), result.get("checks_per_interval"))
    try:
        frozen = policy[0] == "value" and abs(float(policy[1]) - .005) <= 1e-12 and policy[2] == 2
    except (TypeError, ValueError):
        frozen = False
    if not frozen:
        raise ValueError(f"{report_path}: report is not the frozen primary policy")


def final_state(report: dict) -> tuple[dict | None, str | None]:
    result = report.get("result", {})
    status = result.get("status")
    snapshots = result.get("snapshots", [])
    if status in {"invalid_measurement", "invalid_endpoints", "no_usable_final_state"}:
        return None, f"{status}: requires measurement"
    if not snapshots:
        return None, "no usable final state: requires measurement"
    return snapshots[-1], None


def interpolate(rates: np.ndarray, values: np.ndarray, anchors: list[int]) -> np.ndarray:
    """Primary value interpolation, with measured anchors retained exactly."""
    points = sorted(set(int(x) for x in anchors))
    if len(points) < 2:
        raise ValueError("a final state needs at least two anchors")
    return np.interp(rates, np.asarray(points, dtype=float),
                     np.interp(np.asarray(points, dtype=float), rates, values))


def metrics(snapshot: dict) -> tuple[float | None, float | None]:
    m = snapshot.get("never_revealed_metrics", {})
    p99 = m.get("p99_relative")
    maximum = m.get("max_relative")
    return (None if p99 is None else 100.0 * float(p99),
            None if maximum is None else 100.0 * float(maximum))


def plot_family(ax_top, ax_bottom, label: str, curve_path: Path, report_path: Path,
                color: str, terminal_path: Path | None = None) -> None:
    rates, values = curve_values(curve_path)
    normalized = values / values[0]
    ax_top.plot(rates, normalized, color=color, lw=1.4, alpha=.72, label=f"{label} measured")
    ax_top.scatter(rates, normalized, s=8, color=color, alpha=.45, zorder=3)

    report = read_json(report_path)
    validate_report(report, report_path, curve_path)
    snapshot, invalid = final_state(report)
    snapshots = report.get("result", {}).get("snapshots", [])
    if invalid:
        ax_top.text(.04, .93, invalid, transform=ax_top.transAxes, color="#a33",
                    fontsize=7.5, va="top", ha="left", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        # Snapshots remain useful audit evidence, but are explicitly rejected
        # as a final qualification when the run stopped invalidly.
        if snapshots:
            rejected = snapshots[-1]
            try:
                predicted = interpolate(rates, values, rejected.get("measured_rates", [])) / values[0]
                ax_top.plot(rates, predicted, color="#777", lw=1.1, ls="--",
                            label=f"{label} last snapshot (rejected final)")
            except (TypeError, ValueError):
                pass
        ax_bottom.text(.04, .94, "invalid / requires measurement; final rejected",
                       transform=ax_bottom.transAxes, ha="left", va="top", fontsize=7.5, color="#a33")
    else:
        anchors = snapshot.get("measured_rates", [])
        try:
            predicted = interpolate(rates, values, anchors) / values[0]
            ax_top.plot(rates, predicted, color=color, lw=2.0, alpha=.95,
                        label=f"{label} primary final anchors ({len(set(anchors))})")
            anchor_set = sorted(set(int(x) for x in anchors))
            pos = {int(rate): i for i, rate in enumerate(rates)}
            ai = [pos[r] for r in anchor_set if r in pos]
            ax_top.scatter(rates[ai], normalized[ai], s=26, facecolors="white",
                           edgecolors=color, linewidths=1.1, zorder=4)
        except (TypeError, ValueError):
            ax_top.text(.04, .93, "no usable final state: requires measurement",
                        transform=ax_top.transAxes, color="#a33", fontsize=7.5, va="top")

        final_p99, final_max = metrics(snapshot)
        screen = snapshot.get("never_revealed_metrics", {}).get("screen_pass")
        verdict = "PASS" if screen is True else "FAIL" if screen is False else "UNAVAILABLE"
        if final_p99 is not None and final_max is not None:
            ax_top.text(.97, .04,
                        f"final anchors: {len(set(anchors))}\n"
                        f"held-out screen: {verdict}\n"
                        f"p99 {final_p99:.2f}% / max {final_max:.2f}%",
                        transform=ax_top.transAxes, ha="right", va="bottom", fontsize=7.5,
                        color="#176b4d" if verdict == "PASS" else "#a33")
    counts, p99s, maxs = [], [], []
    for item in snapshots:
        p99, maximum = metrics(item)
        if p99 is not None and maximum is not None:
            counts.append(int(item.get("measurement_count", len(item.get("measured_rates", [])))))
            p99s.append(p99); maxs.append(maximum)
    if counts:
        ax_bottom.plot(counts, p99s, "o-", color=color, lw=1.2, ms=3.5, label="p99")
        ax_bottom.plot(counts, maxs, "s--", color=color, lw=1.1, ms=3.2, alpha=.8, label="max")
    elif not invalid:
        ax_bottom.text(.5, .5, "no audit snapshots", transform=ax_bottom.transAxes,
                       ha="center", va="center", fontsize=8, color="#666")

    if any(item.get("all_candidates_measured") is True for item in snapshots):
        ax_bottom.text(.04, .08, "all candidates measured: empty held-out audit",
                       transform=ax_bottom.transAxes, fontsize=7, color="#555")

    if terminal_path is not None:
        tr, tv = curve_values(terminal_path, allow_singleton=True)
        if len(tr) != 1:
            raise ValueError(f"{terminal_path}: separate terminal curve must contain exactly one point")
        ax_top.scatter(tr, tv / values[0], marker="D", s=25, color=color, edgecolors="black",
                       linewidths=.4, zorder=5, label=f"{label} terminal measured (separate)")
    ax_top.set_title(label)
    ax_top.grid(True, alpha=.22)
    ax_bottom.grid(True, alpha=.22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bf16", "e4m3", "e2m1"):
        parser.add_argument(f"--{name}-curve", type=Path, required=True)
        parser.add_argument(f"--{name}-report", type=Path, required=True)
    parser.add_argument("--e2m1-terminal-curve", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    families = [("BF16", args.bf16_curve, args.bf16_report, "#345995", None),
                ("E4M3", args.e4m3_curve, args.e4m3_report, "#e76f51", None),
                ("E2M1 window", args.e2m1_curve, args.e2m1_report, "#2a9d8f", args.e2m1_terminal_curve)]
    for _, curve, report, _, terminal in families:
        if not curve.is_file() or not report.is_file() or (terminal and not terminal.is_file()):
            raise FileNotFoundError(f"missing input: {curve} / {report} / {terminal}")
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharey="row", constrained_layout=False)
    for col, (label, curve, report, color, terminal) in enumerate(families):
        plot_family(axes[0, col], axes[1, col], label, curve, report, color, terminal)
    for ax in axes[0]:
        ax.set_xlabel("sparse rate (measured legal rung)")
    for ax in axes[1]:
        ax.set_xlabel("actual snapshot anchor count")
        ax.set_yscale("log")
    axes[0, 0].set_ylabel("output MSE / first measured point")
    axes[1, 0].set_ylabel("relative error (%)")
    axes[1, 0].axhline(SCREEN_P99, color="#555", lw=.9, ls=":", label="p99 screen 1%")
    axes[1, 0].axhline(SCREEN_MAX, color="#555", lw=.9, ls="--", label="max screen 5%")
    for ax in axes[1, 1:]:
        ax.axhline(SCREEN_P99, color="#555", lw=.9, ls=":")
        ax.axhline(SCREEN_MAX, color="#555", lw=.9, ls="--")
    axes[0, 0].legend(fontsize=7, loc="best")
    axes[1, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Adaptive sparse-rate interpolation audit (research measurement)", fontsize=14)
    fig.text(.5, .018, PRIMARY + ". One Linear; finite measured bands; ordinate is output MSE, not KL or serving quality.",
             ha="center", fontsize=8.5, color="#444")
    fig.subplots_adjust(top=.88, bottom=.12, wspace=.08, hspace=.28)
    png, pdf = args.out / "sparse_rate_adaptive.png", args.out / "sparse_rate_adaptive.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    inputs = []
    for _, curve, report, _, terminal in families:
        for path, role in ((curve, "curve"), (report, "primary_policy_report")):
            inputs.append({"role": role, "path": str(path), "sha256": sha256(path)})
        if terminal:
            inputs.append({"role": "separate_terminal_curve", "path": str(terminal), "sha256": sha256(terminal)})
    manifest = {"schema": "prismaquant.sparse_rate_adaptive_figure_manifest.v1",
                "primary_policy": PRIMARY, "inputs": inputs,
                "outputs": [{"path": str(png), "sha256": sha256(png)},
                             {"path": str(pdf), "sha256": sha256(pdf)}]}
    (args.out / "source_hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "manifest": str(args.out / 'source_hash_manifest.json')}))


if __name__ == "__main__":
    main()
