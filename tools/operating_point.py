"""Fit the measured rate-distortion frontier and price the operating-point choice.

Validated sweeps on these models come out log-linear in measured held-out KL
(Qwen3.8-27B: R^2 0.995 over 4.5..8.25 bpp, x1.72 KL per bpp). A log-linear
frontier is SCALE-FREE: the relative return per bpp is a constant, so the
curve itself distinguishes no interior point, and every "knee" is an axis
artifact -- the same sweep's kneedle picked 4.75 on the log axis and 6.0 on
the linear axis. Choosing a bpp therefore requires an anchor from OUTSIDE the
(bytes, KL) plane.

This tool is the reproducible half of the choice: the fit, its validity gate
(R^2), the per-bpp exchange rate, the axis-dependence report, and the bpp at
which any externally chosen KL target lands. The anchor doctrine -- what the
KL target should BE (task-parity saturation, card budget, ...) -- is a
decision, not a computation; it lives in
docs/design/operating_point_selection.md and is NOT encoded here.

Usage:
  python tools/operating_point.py --selection WORK/artifacts/validated_frontier_selection.json \
      [--kl-target 0.05 0.03 ...] [--qparams 24350556160]
  python tools/operating_point.py --csv points.csv   # lines of "bpp,kl[,kl_max]"
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _collect_selection(path: str) -> list[tuple[float, float, float | None]]:
    with open(path) as fh:
        d = json.load(fh)
    rows: dict[float, tuple[float, float | None]] = {}

    def add(r: object) -> None:
        if not isinstance(r, dict):
            return
        b = r.get("bpp") or r.get("achieved_bits")
        kl = r.get("kl") or r.get("kl_mean")
        if isinstance(b, (int, float)) and isinstance(kl, (int, float)) and kl > 0:
            rows.setdefault(round(float(b), 6), (float(kl), r.get("kl_max")))

    for r in d.get("frontier") or []:
        add(r)
    # Vetoed / dominated rows are still valid MEASUREMENTS: the veto governs
    # what may ship, not what may inform the fit.
    for v in d.get("vetoed_rows") or []:
        add(v.get("row", v))
    vj = d.get("validation_json")
    if isinstance(vj, str):
        try:
            with open(vj) as fh:
                raw = json.load(fh)
            for r in raw if isinstance(raw, list) else (
                    raw.get("results") or raw.get("rows") or []):
                add(r)
        except OSError:
            pass
    return sorted((b, kl, km) for b, (kl, km) in rows.items())


def _collect_csv(path: str) -> list[tuple[float, float, float | None]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            b, kl = float(parts[0]), float(parts[1])
            km = float(parts[2]) if len(parts) > 2 and parts[2] else None
            out.append((b, kl, km))
    return sorted(out)


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n
    ybar = sy / n
    ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return alpha, beta, (1.0 - ss_res / ss_tot if ss_tot else float("nan"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--selection", help="validated_frontier_selection.json")
    src.add_argument("--csv", help="CSV of bpp,kl[,kl_max] lines")
    ap.add_argument("--kl-target", type=float, nargs="*", default=[],
                    help="externally chosen KL anchors to place on the bpp axis")
    ap.add_argument("--qparams", type=int, default=None,
                    help="quantizable body params, to price bpp in GB")
    ap.add_argument("--r2-gate", type=float, default=0.99,
                    help="below this R^2 the fit is not a valid interpolator")
    args = ap.parse_args(argv)

    pts = (_collect_selection(args.selection) if args.selection
           else _collect_csv(args.csv))
    if len(pts) < 3:
        print(f"REFUSE: {len(pts)} measured points; need >= 3 to say anything "
              "about the frontier's shape.", file=sys.stderr)
        return 2

    xs = [b for b, _, _ in pts]
    ys = [math.log10(kl) for _, kl, _ in pts]
    alpha, beta, r2 = _fit(xs, ys)

    print(f"{len(pts)} measured points, bpp {xs[0]:.3f}..{xs[-1]:.3f}")
    for b, kl, km in pts:
        pred = 10 ** (alpha + beta * b)
        tail = f"  kl_max={km:.4f}" if km else ""
        print(f"  bpp={b:7.4f}  KL={kl:.6f}  fit={pred:.6f}  "
              f"resid={100 * (kl / pred - 1):+6.1f}%{tail}")

    print(f"\nfit: log10(KL) = {alpha:+.4f} {beta:+.4f}*bpp   R^2 = {r2:.4f}")
    if r2 >= args.r2_gate:
        print(f"SCALE-FREE REGIME (R^2 >= {args.r2_gate}): constant relative "
              f"return, KL reduction factor {10 ** (-beta):.3f} per +1 bpp "
              f"({10 ** (-beta * 0.5):.3f} per +0.5). No interior optimum "
              "exists on this curve; a knee reported on it is an axis "
              "artifact. Pick the operating point from an external anchor "
              "(see docs/design/operating_point_selection.md).")
    else:
        print(f"R^2 {r2:.4f} < {args.r2_gate}: the frontier has structure; "
              "this fit is NOT a valid interpolator. Inspect residuals "
              "before using any number below.")

    kms = [(b, km) for b, _, km in pts if km]
    if len(kms) == len(pts):
        a2, b2, r22 = _fit([b for b, _ in kms],
                           [math.log10(km) for _, km in kms])
        print(f"tail (kl_max) slope: {b2:+.4f}/bpp "
              f"(KL reduction factor {10 ** (-b2):.3f} per +1 bpp, R^2 {r22:.3f}) -- "
              f"{'tracks the mean' if abs(b2 - beta) < 0.05 else 'DIVERGES from the mean: the tail has its own regime; principle 4 applies'}")

    for klt in args.kl_target:
        b = (math.log10(klt) - alpha) / beta
        gb = f"  ({b * args.qparams / 8 / 1e9:.2f} GB body)" if args.qparams else ""
        inside = "" if xs[0] <= b <= xs[-1] else "  [EXTRAPOLATED beyond the sweep]"
        print(f"KL target {klt:g} -> bpp {b:.2f}{gb}{inside}")

    if args.selection:
        with open(args.selection) as fh:
            kc = json.load(fh).get("kneedle_comparison") or {}
        le, rl = kc.get("log_error") or {}, kc.get("raw_linear") or {}
        if le.get("bpp") and rl.get("bpp"):
            print(f"\nkneedle axis-dependence on THIS data: log-error picks "
                  f"{le['bpp']:.3f}, raw-linear picks {rl['bpp']:.3f} "
                  f"(gap {abs(rl['bpp'] - le['bpp']):.2f} bpp). The same tool, "
                  "two axes, two answers: the knee is not a property of the "
                  "frontier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
