"""CPU-only empirical check for the two-tier scale spec
(docs/lanes/nvfp4-cb/two-tier-scale-spec.md §3).

Question: constraining the NVFP4-CB per-group-16 scale sweep to the two-tier
REACHABLE set {T[c] x 2^E_sb : legality mask} (per-256 E8M0 super + 4/5-bit
sub-code) — instead of the free E4M3 plane — costs how much weighted
reconstruction error on real Qwen3-0.6B tensors?

Design notes (mirrors the spec):
  * NO GPU — CUDA_VISIBLE_DEVICES cleared before torch import, asserted off.
    (A timing-sensitive serving benchmark owns the GPU.)
  * Instrument: product-k16 with the FIXED deterministic lattice, same codebook
    in every arm, so deltas isolate SCALE CODING only (codebook quality is an
    orthogonal, already-studied axis).
  * Weights: the real exp-1 imatrix (E[x^2] per column, seed 0) — genuinely
    weighted error, loaded from the exp-1 cache, no recomputation.
  * Baseline arm = the shipping encoder (free 16-candidate E4M3 sweep + 2 WLS
    refits, `_sweep_encode`). Two-tier arms get the same machinery with the
    candidate set restricted + a snap-to-reachable WLS refit — the exact
    "existing sweep becomes the two-tier encoder" claim under test.
  * E_sb sweep window: E0 + [-3..+1] where E0 places the table top at the
    superblock's max ideal scale; edge-hit rate reported (must be ~0 for the
    windowed sweep to stand in for the exhaustive one).

Run: PYTHONPATH=. python scripts/two_tier_scale_check.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # hard no-GPU guard

import json                                        # noqa: E402
import pickle                                      # noqa: E402
from pathlib import Path                           # noqa: E402

import torch                                       # noqa: E402

assert not torch.cuda.is_available(), "this check must not touch the GPU"

import prismaquant.nvfp4_cb_formats as F           # noqa: E402

QWEN = "/home/rob/models/Qwen3-0.6B/model.safetensors"
IMATRIX = "/home/rob/dq-runs/nvfp4-cb-phase0/exp1/results/imatrix_seed0.pkl"
TENSORS = [
    "model.layers.6.self_attn.q_proj.weight",
    "model.layers.6.mlp.gate_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
]
ROWS = 512                 # deterministic leading-row slice per tensor
K, GRID, MODE = 16, "fp4", "product"
E4M3_MAX = 448.0
OUT_JSON = Path(__file__).resolve().parent.parent / "docs" / "nvfp4-cb-plan" / "two_tier_scale_check.json"

# Sub-code tables (values are e4m3-exact by construction: (8+j)/8 * 2^i).
TABLES = {
    # 8 mantissa steps x 2 octaves — full e4m3 granularity, [1, 3.75] span.
    "T4_2oct8m": [(8 + j) / 8 * (1 << i) for i in range(2) for j in range(8)],
    # 4 mantissa steps x 4 octaves — coarser, [1, 14] span.
    "T4_4oct4m": [(8 + 2 * j) / 8 * (1 << i) for i in range(4) for j in range(4)],
    # 5-bit fallback: 8 mantissa steps x 4 octaves, [1, 15] span (32 entries).
    "T5_4oct8m": [(8 + j) / 8 * (1 << i) for i in range(4) for j in range(8)],
}


def e4m3_exact(x: torch.Tensor) -> torch.Tensor:
    """True where x is exactly representable in float8_e4m3fn."""
    r = x.to(torch.float8_e4m3fn).to(torch.float32)
    return (r == x) & x.isfinite()


def weighted_err(w2d, wq, s):
    """Total weighted + unweighted recon error at per-group scales s."""
    err_g, _, _ = F._eval_candidate(w2d, wq, s, GRID, MODE, CB)
    err_u, _, _ = F._eval_candidate(w2d, None, s, GRID, MODE, CB)
    return float(err_g.sum()), float(err_u.sum())


def two_tier_encode(w2d, wq, table):
    """Windowed-exhaustive two-tier scale selection + snap-refit.

    Returns (scales (rows, ng), diag dict). Scales are guaranteed reachable:
    s = table[c] * 2^E_sb with per-(E, entry) e4m3-exactness enforced.
    """
    rows, in_f = w2d.shape
    ng = in_f // F.FP4_GROUP
    n_sb = in_f // F.SUPERBLOCK
    gps = F.SUPERBLOCK // F.FP4_GROUP                          # 16 groups/sb
    T = torch.tensor(sorted(table), dtype=torch.float32)
    ideal = F._group_amax(w2d, GRID) / F.NVFP4_GRID_MAX        # (rows, ng)
    ideal_sb = ideal.reshape(rows, n_sb, gps)
    imax = ideal_sb.amax(-1)                                   # (rows, n_sb)
    # E0 places the table top at the sb's max ideal (largest group reachable
    # without forced clipping); all-zero sb -> deterministic floor.
    E0 = torch.where(imax > 0, (imax / T[-1]).log2().ceil(),
                     torch.full_like(imax, -9.0))
    E_OFFS = list(range(-3, 2))
    best_err = None
    best_s = None
    best_eoff = None
    for eoff in E_OFFS:
        E = E0 + eoff                                          # (rows, n_sb)
        p2 = torch.pow(2.0, E)
        # candidates (|T|, rows, ng): table entry x 2^E_sb, per-entry legality.
        err_best_e = None
        s_best_e = None
        for t in T.tolist():
            s_c = (t * p2).repeat_interleave(gps, dim=1)       # (rows, ng)
            legal = e4m3_exact(s_c) & (s_c <= E4M3_MAX) & (s_c > 0)
            s_eval = torch.where(legal, s_c, torch.full_like(s_c, 1.0))
            err_g, _, _ = F._eval_candidate(w2d, wq, s_eval, GRID, MODE, CB)
            err_g = torch.where(legal, err_g,
                                torch.full_like(err_g, float("inf")))
            if err_best_e is None:
                err_best_e, s_best_e = err_g, torch.where(legal, s_c, s_eval)
            else:
                better = err_g < err_best_e
                err_best_e = torch.where(better, err_g, err_best_e)
                s_best_e = torch.where(better, s_c, s_best_e)
        # pick E per superblock by total weighted error.
        err_sb = err_best_e.reshape(rows, n_sb, gps).sum(-1)   # (rows, n_sb)
        if best_err is None:
            best_err, best_s = err_sb, s_best_e
            best_eoff = torch.full_like(err_sb, eoff)
        else:
            better = err_sb < best_err
            best_err = torch.where(better, err_sb, best_err)
            bexp = better.repeat_interleave(gps, dim=1)
            best_s = torch.where(bexp, s_best_e, best_s)
            best_eoff = torch.where(better, torch.full_like(err_sb, eoff),
                                    best_eoff)
    edge_hit = float(((best_eoff == E_OFFS[0]) | (best_eoff == E_OFFS[-1]))
                     .float().mean())
    # snap-to-reachable WLS refit (2 iters), reachable set frozen at best E.
    E_best = E0 + best_eoff                                    # (rows, n_sb)
    reach = T.reshape(1, 1, -1) * torch.pow(2.0, E_best).unsqueeze(-1)
    reach_legal = e4m3_exact(reach) & (reach <= E4M3_MAX) & (reach > 0)
    reach = torch.where(reach_legal, reach,
                        torch.full_like(reach, float("nan")))
    for _ in range(F._SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = F._eval_candidate(w2d, wq, best_s, GRID, MODE, CB)
        wcol = (wq.reshape(rows, in_f) if wq is not None
                else torch.ones_like(g))
        num = F._group_reduce(wcol * g * w2d, GRID)
        den = F._group_reduce(wcol * g * g, GRID)
        s_star = torch.where(den > 0, num / den.clamp_min(1e-30), best_s)
        # nearest legal reachable value per group.
        s_sb = s_star.reshape(rows, n_sb, gps)
        d = (reach.unsqueeze(2) - s_sb.unsqueeze(-1)).abs()    # (r,sb,gps,|T|)
        d = torch.where(d.isnan(), torch.full_like(d, float("inf")), d)
        nearest = reach.unsqueeze(2).expand_as(d).gather(
            -1, d.argmin(-1, keepdim=True)).squeeze(-1)
        s_snap = nearest.reshape(rows, ng)
        err_snap, _, _ = F._eval_candidate(w2d, wq, s_snap, GRID, MODE, CB)
        best_s = torch.where(err_snap < err_cur, s_snap, best_s)
    assert bool(e4m3_exact(best_s).all()), "two-tier scale left the e4m3 grid"
    return best_s, {"edge_hit_rate": edge_hit}


def main() -> None:
    from safetensors import safe_open
    global CB
    CB = F._resolve_codebook(K, GRID, MODE, None, torch.device("cpu"))
    with open(IMATRIX, "rb") as f:
        imatrix = pickle.load(f)
    results = {}
    with safe_open(QWEN, "pt") as st:
        for name in TENSORS:
            q = name[: -len(".weight")]
            w2d = st.get_tensor(name)[:ROWS].to(torch.float32)
            rows, in_f = w2d.shape
            cw = imatrix[q]["e_x2"].to(torch.float32)
            cw2d = torch.broadcast_to(cw, w2d.shape).contiguous()
            wq = F._col_weight_vectors(cw2d.reshape(rows, in_f))
            r: dict = {"shape": [rows, in_f]}

            # scale-plane statistics (free ideal scales).
            ideal = F._group_amax(w2d, GRID) / F.NVFP4_GRID_MAX
            n_sb = in_f // F.SUPERBLOCK
            isb = ideal.reshape(rows, n_sb, 16)
            spread = (isb.amax(-1) / isb.amin(-1).clamp_min(1e-30)).log2()
            r["ideal_scale_stats"] = {
                "subnormal_frac": float((ideal < 2.0 ** -6).float().mean()),
                "within_sb_spread_log2_p50": float(spread.quantile(0.5)),
                "within_sb_spread_log2_p90": float(spread.quantile(0.9)),
                "within_sb_spread_log2_p99": float(spread.quantile(0.99)),
                "within_sb_spread_log2_max": float(spread.max()),
            }

            # arm 0: one-shot (candidate 0 = amax/6 snapped).
            s0 = F._snap_scale(ideal, GRID)
            r["oneshot"] = weighted_err(w2d, wq, s0)
            # arm 1: shipping free sweep (16 cands + 2 WLS refits).
            s_free, _ = F._sweep_encode(w2d, GRID, MODE, CB, wq)
            r["free_sweep"] = weighted_err(w2d, wq, s_free)
            r["free_sweep_subnormal_frac"] = float(
                (s_free < 2.0 ** -6).float().mean())
            # arms 2..: two-tier variants.
            for tname, table in TABLES.items():
                s_tt, diag = two_tier_encode(w2d, wq, table)
                we, ue = weighted_err(w2d, wq, s_tt)
                fw = r["free_sweep"]
                r[tname] = {
                    "weighted": we, "unweighted": ue,
                    "tax_vs_free_weighted_pct": (we / fw[0] - 1) * 100,
                    "tax_vs_free_unweighted_pct": (ue / fw[1] - 1) * 100,
                    **diag,
                }
            results[q] = r
            print(f"== {q} {tuple(w2d.shape)}")
            print(f"   ideal-scale: {r['ideal_scale_stats']}")
            print(f"   oneshot weighted {r['oneshot'][0]:.6g} | "
                  f"free_sweep {r['free_sweep'][0]:.6g} "
                  f"(subnormal frac {r['free_sweep_subnormal_frac']:.3f})")
            for tname in TABLES:
                t = r[tname]
                print(f"   {tname}: weighted {t['weighted']:.6g} "
                      f"tax {t['tax_vs_free_weighted_pct']:+.2f}% "
                      f"(unw {t['tax_vs_free_unweighted_pct']:+.2f}%) "
                      f"edge_hit {t['edge_hit_rate']:.4f}")
    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
