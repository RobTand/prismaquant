#!/usr/bin/env python3
"""Render the receipt-bound prospective sparse-rate family transfer audit.

The figure consumes the frozen protocol, measured curve pieces, prediction
seals, and audits from one exact artifact directory.  It deliberately draws
raw points and straight prediction rules: no smoothing or monotonic repair is
applied to measured values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/sparse-rate-20260911/prospective-shared-down-l20-01")
FILES = {
    "protocol_e4": "protocol-e4-01.json",
    "protocol_e2": "protocol-e2-01.json",
    "preselection": "preselection-01.json",
    "bf_curve": "l20_shared_down_bf16_full.curve.json",
    "e4_left": "l20_shared_down_e4m3_endpoint_r832.curve.json",
    "e4_right": "l20_shared_down_e4m3_endpoint_r1088.curve.json",
    "e4_interior": "l20_shared_down_e4m3_hidden_interior.curve.json",
    "e4_seal": "e4-seal-01.json",
    "e4_audit": "e4-audit-01.json",
    "e2_left": "l20_shared_down_e2m1_window_endpoint_r832.curve.json",
    "e2_right": "l20_shared_down_e2m1_window_endpoint_r895.curve.json",
    "e2_interior": "l20_shared_down_e2m1_window_hidden_interior.curve.json",
    "e2_terminal": "l20_shared_down_e2m1_terminal_r896.curve.json",
    "e2_seal": "e2-seal-01.json",
    "e2_audit": "e2-audit-01.json",
}
SCREEN_P99, SCREEN_MAX = 1.0, 5.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    value = load(path)
    rates, values = value.get("rates"), value.get("values")
    if not isinstance(rates, list) or not isinstance(values, list) or len(rates) != len(values):
        raise ValueError(f"{path}: expected matching rates/values arrays")
    if not rates or len(set(rates)) != len(rates) or rates != sorted(rates):
        raise ValueError(f"{path}: rates must be nonempty and sorted")
    return np.asarray(rates, dtype=float), np.asarray(values, dtype=float)


def join_curves(*paths: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [curve(path) for path in paths]
    rates = np.concatenate([item[0] for item in rows])
    values = np.concatenate([item[1] for item in rows])
    order = np.argsort(rates)
    rates, values = rates[order], values[order]
    if len(np.unique(rates)) != len(rates):
        raise ValueError("curve pieces overlap")
    return rates, values


def predictions(path: Path, forms: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    seal = load(path)
    rows = seal.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: missing prediction rows")
    result = {}
    rates = np.asarray([row["rate"] for row in rows], dtype=float)
    for form in forms:
        result[form] = (rates, np.asarray([row[form] for row in rows], dtype=float))
    return result


def metric_text(audit: dict, forms: list[str]) -> str:
    lines = []
    metrics = audit.get("metrics", {})
    for form in forms:
        item = metrics.get(form, {})
        all_m, fresh_m = item.get("all_interior", {}), item.get("fresh_only", {})
        def fmt(m):
            return f"p99 {100*float(m['p99_relative']):.2f}% / max {100*float(m['max_relative']):.2f}%"
        label = {"affine_bf_from_paired_endpoints": "Affine BF transfer",
                 "rate_linear_delta_from_bf": "Rate-linear BF delta",
                 "endpoint_value_linear": "Endpoint-value linear"}.get(form, form)
        def verdict(m):
            return "PASS" if m.get("screen_pass") is True else "FAIL" if m.get("screen_pass") is False else "?"
        lines.append(f"{label}\nall ({all_m.get('count', '?')}): {fmt(all_m)} [{verdict(all_m)}]\nfresh ({fresh_m.get('count', '?')}): {fmt(fresh_m)} [{verdict(fresh_m)}]")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.artifact_dir
    paths = {key: base / name for key, name in FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required measured/seal/audit inputs:\n" + "\n".join(missing))

    protocols = {family: load(paths[f"protocol_{family}"]) for family in ("e4", "e2")}
    for family, protocol in protocols.items():
        if protocol.get("schema") != "prismaquant.prospective_paired_family_transfer_protocol.v1":
            raise ValueError(f"unsupported {family} prospective transfer protocol")
    preselection = load(paths["preselection"])
    e4_protocol, e2_protocol = protocols["e4"], protocols["e2"]
    e4_forms = e4_protocol["prediction_families"]["e4"]["forms"]
    e2_forms = e2_protocol["prediction_families"]["e2"]["forms"]
    # The protocol is E4-root verified and names the exact complementary pieces.
    bf_r, bf_v = curve(paths["bf_curve"])
    e4_r, e4_v = join_curves(paths["e4_left"], paths["e4_interior"], paths["e4_right"])
    e2_r, e2_v = join_curves(paths["e2_left"], paths["e2_interior"], paths["e2_right"])
    terminal_r, terminal_v = curve(paths["e2_terminal"])
    if len(terminal_r) != 1 or terminal_r[0] in e2_r:
        raise ValueError("E2 terminal must be one separate rate")
    e4_pred = predictions(paths["e4_seal"], e4_forms)
    e2_pred = predictions(paths["e2_seal"], e2_forms)
    e4_audit, e2_audit = load(paths["e4_audit"]), load(paths["e2_audit"])
    if e4_audit.get("family") != "e4" or e2_audit.get("family") != "e2":
        raise ValueError("audit family labels do not match")
    for family, audit, protocol, seal_key, interior_key in (
            ("e4", e4_audit, e4_protocol, "e4_seal", "e4_interior"),
            ("e2", e2_audit, e2_protocol, "e2_seal", "e2_interior")):
        if audit.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise ValueError(f"{family} audit protocol hash does not match protocol")
        if audit.get("seal_sha256") != load(paths[seal_key]).get("seal_sha256"):
            raise ValueError(f"{family} audit seal hash does not match seal")
        if audit.get("interior_curve_sha256") != digest(paths[interior_key]):
            raise ValueError(f"{family} audit interior hash does not match plotted curve")

    # Preserve the measured MSE scale; every line is a raw polyline or a
    # sealed prediction, with no smoothing or monotonic repair.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=False)
    colors = {"bf": "#345995", "e4": "#e76f51", "e2": "#2a9d8f"}
    ax = axes[0, 0]
    ax.plot(bf_r, bf_v, color=colors["bf"], lw=1.1, alpha=.8, label="BF16 measured (257)")
    ax.scatter(e4_r, e4_v, color=colors["e4"], s=10, alpha=.65, label="E4M3 measured (257)")
    for form, (r, v) in e4_pred.items():
        ax.plot(r, v, lw=1.8, label={"affine_bf_from_paired_endpoints": "Affine BF transfer", "rate_linear_delta_from_bf": "Rate-linear BF delta"}[form])
    ax.set_title("E4M3 transfer: measured curve and sealed predictions")
    ax.set_xlabel("sparse rate (R832–R1088)"); ax.set_ylabel("raw output MSE")
    ax.grid(alpha=.22); ax.legend(fontsize=7, loc="best")
    ax.text(.02, .02, metric_text(e4_audit, e4_forms), transform=ax.transAxes, fontsize=7, va="bottom",
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#ddd"})

    ax = axes[0, 1]
    ax.scatter(e2_r, e2_v, color=colors["e2"], s=14, alpha=.7, label="E2M1 measured (64 incl. endpoints)")
    for form, (r, v) in e2_pred.items():
        ax.plot(r, v, color="#264653", lw=1.8, label="Endpoint-value linear prediction")
    ax.scatter(terminal_r, terminal_v, marker="D", s=34, color="#f4a261", edgecolor="black",
               lw=.5, label="R896 terminal measured (separate)")
    ax.set_title("E2M1 window: prediction ends at R895")
    ax.set_xlabel("sparse rate"); ax.set_ylabel("raw output MSE")
    ax.grid(alpha=.22); ax.legend(fontsize=7, loc="best")
    ax.text(.02, .02, metric_text(e2_audit, e2_forms), transform=ax.transAxes, fontsize=7, va="bottom",
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#ddd"})

    def residual_panel(ax, measured_r, measured_v, preds, title):
        truth = dict(zip(measured_r.astype(int), measured_v))
        for form, (rates, values) in preds.items():
            common = [int(r) for r in rates if int(r) in truth]
            residual = [100 * (values[np.where(rates == rate)[0][0]] / truth[rate] - 1) for rate in common]
            ax.plot(common, residual, lw=1.0, marker=".", ms=2.5, label=form.replace("_", " "))
        ax.axhline(0, color="#555", lw=.7)
        ax.axhline(SCREEN_P99, color="#777", ls=":", lw=.9, label="p99 screen 1%")
        ax.axhline(-SCREEN_P99, color="#777", ls=":", lw=.9)
        ax.axhline(SCREEN_MAX, color="#999", ls="--", lw=.8, label="max screen 5%")
        ax.axhline(-SCREEN_MAX, color="#999", ls="--", lw=.8)
        ax.set_title(title); ax.set_xlabel("sparse rate"); ax.set_ylabel("relative residual (%)")
        ax.grid(alpha=.22); ax.legend(fontsize=7, loc="best")

    residual_panel(axes[1, 0], e4_r, e4_v, e4_pred, "E4M3 pointwise relative residual")
    residual_panel(axes[1, 1], e2_r, e2_v, e2_pred, "E2M1 pointwise relative residual (R896 excluded)")
    fig.suptitle("Prospective sparse-rate family transfer audit · layer 20 shared down projection", fontsize=14)
    fig.text(.5, .012, "Output MSE under route activation contract · raw measured points · no smoothing or monotonic enforcement · research audit only\n"
             "Candidate 262 and complete reference 579 are point counts; they are not a time-savings claim.",
             ha="center", fontsize=8.5, color="#444")
    fig.subplots_adjust(top=.90, bottom=.11, wspace=.16, hspace=.30)
    args.out.mkdir(parents=True, exist_ok=True)
    png, pdf = args.out / "sparse_rate_family_transfer.png", args.out / "sparse_rate_family_transfer.pdf"
    fig.savefig(png, dpi=220); fig.savefig(pdf); plt.close(fig)

    measured_roles = ["bf_curve", "e4_left", "e4_right", "e4_interior", "e2_left", "e2_right", "e2_interior", "e2_terminal"]
    derived_roles = ["protocol_e4", "protocol_e2", "preselection", "e4_audit", "e2_audit"]
    predicted_roles = ["e4_seal", "e2_seal"]
    manifest = {"schema": "prismaquant.sparse_rate_family_transfer_figure_manifest.v1",
                "research_only": True, "currency": e4_protocol.get("currency"),
                "protocol_sha256": {family: digest(paths[f"protocol_{family}"]) for family in ("e4", "e2")},
                "source": {"qname": preselection.get("qname"), "qshape": preselection.get("qshape"),
                           "protocol_schema": {family: protocols[family].get("schema") for family in ("e4", "e2")},
                           "protocol_sha256": {family: protocols[family].get("protocol_sha256") for family in ("e4", "e2")}},
                "inputs": [{"role": role, "kind": "measured" if role in measured_roles else "derived-audit" if role.endswith("_audit") else "protocol" if role in derived_roles else "predicted",
                            "path": str(paths[role]), "sha256": digest(paths[role])} for role in measured_roles + derived_roles + predicted_roles],
                "outputs": [{"role": "png", "path": str(png), "sha256": digest(png)},
                            {"role": "pdf", "path": str(pdf), "sha256": digest(pdf)}],
                "screen": {"p99_percent": SCREEN_P99, "max_percent": SCREEN_MAX,
                           "metric_labels": {"e4": {"all": e4_audit["metrics"][e4_forms[0]]["all_interior"].get("count"), "fresh": e4_audit["metrics"][e4_forms[0]]["fresh_only"].get("count")},
                                             "e2": {"all": e2_audit["metrics"][e2_forms[0]]["all_interior"].get("count"), "fresh": e2_audit["metrics"][e2_forms[0]]["fresh_only"].get("count")}}}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "manifest": str(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
