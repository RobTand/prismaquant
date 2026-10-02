"""Compare two a_side_diag runs on identical tokens and probes, e.g. fp32 against bf16 (CPU only).

usage: compare_dtype_runs.py STEM_REF STEM_TEST [--boot 4000] [--top 12]

It refuses unless the units, probe seeds and token ids match. Per spec (A, A without position 0,
W4) it reports each run's full-estimator total, sum_u 0.5 mean_p (sum_seq c_pu,seq)^2, and the
per-sequence diagonal price D[u, seq] = 0.5 mean_p c_pu,seq^2. The diagonal is unbiased for the same
quantity (probe seeds are independent per sequence), so a paired bootstrap over sequences of
sum D_test / sum D_ref gives the ratio a CI. It lists the units that carry the test run's excess,
with their position-0 density share in each run. For the A and W arms it also reports the
reference run's KL_true against each run's P_add, with a sequence bootstrap of the ratio.
The test run's own KL_true is printed but not used as truth: a bf16 forward inflates differences.
"""
import argparse
import json
import sys

import torch

SPECS = ("FP8_E4M3", "FP8_E4M3_NOPOS0", "NVFP4A16")
ARM_SPEC = {"A_all": "FP8_E4M3", "A_attn": "FP8_E4M3", "A_mlp": "FP8_E4M3",
            "A_all_nopos0": "FP8_E4M3_NOPOS0", "W4_all": "NVFP4A16", "W4_attn": "NVFP4A16",
            "W4_mlp": "NVFP4A16"}


def load(stem):
    meta = json.load(open(stem + ".json"))
    tensors = torch.load(stem + ".pt", weights_only=False)
    return meta, tensors


def boot_ratio(num, den, n_boot, gen):
    """num, den: [n_seq] per-sequence parts of two totals; paired bootstrap of sum(num)/sum(den)."""
    n = num.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=gen)
    r = num[idx].sum(1) / den[idx].sum(1)
    q = torch.quantile(r, torch.tensor([0.025, 0.5, 0.975], dtype=r.dtype))
    return {"ratio": float(num.sum() / den.sum()), "ci95": [float(q[0]), float(q[2])],
            "boot_median": float(q[1]), "se": float(r.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    (mr, tr), (mt, tt) = load(args.ref), load(args.test)
    for key in ("units", "seeds", "ids_sha256", "n_global"):
        if mr[key] != mt[key]:
            raise SystemExit(f"REFUSED: the runs differ in {key}")
    names = mr["units"]
    gen = torch.Generator().manual_seed(20261001)
    cr, ct = tr["comps"].double(), tt["comps"].double()          # [spec, unit, probe, seq]
    report = {"ref": args.ref, "test": args.test,
              "dtypes": [mr["args"]["dtype"], mt["args"]["dtype"]], "specs": {}, "arms": {}}
    for si, spec in enumerate(SPECS):
        full_r = 0.5 * cr[si].sum(-1).pow(2).mean(-1)              # [unit]
        full_t = 0.5 * ct[si].sum(-1).pow(2).mean(-1)
        dr = 0.5 * cr[si].pow(2).mean(1)                            # [unit, seq]
        dt = 0.5 * ct[si].pow(2).mean(1)
        rec = {"full_ref": float(full_r.sum()), "full_test": float(full_t.sum()),
               "full_ratio": float(full_t.sum() / full_r.sum()),
               "diag_ref": float(dr.sum()), "diag_test": float(dt.sum()),
               "diag_ratio": boot_ratio(dt.sum(0), dr.sum(0), args.boot, gen)}
        excess = dt.sum(1) - dr.sum(1)
        tot_excess = float(excess.sum())
        order = torch.argsort(excess, descending=True)
        pos0 = {}
        for tag, t in (("ref", tr), ("test", tt)):
            dens = t.get("dens_unit_pos")
            if dens is not None:
                d = dens[si].double()
                pos0[tag] = d[:, 0] / d.sum(-1).clamp_min(1e-300)
        top = []
        for i in order[: args.top].tolist():
            row = {"unit": names[i], "diag_ref": float(dr[i].sum()), "diag_test": float(dt[i].sum()),
                   "excess_share": float(excess[i]) / tot_excess if tot_excess else None}
            for tag, share in pos0.items():
                row[f"pos0_share_{tag}"] = float(share[i])
            top.append(row)
        rec["excess_total"] = tot_excess
        rec["top5_excess_share"] = (float(excess[order[:5]].sum()) / tot_excess
                                    if tot_excess else None)
        rec["top_excess_units"] = top
        report["specs"][spec] = rec
    for arm, spec in ARM_SPEC.items():
        ar, at = mr["arms"].get(arm), mt["arms"].get(arm)
        if ar is None or at is None:
            continue
        kl_seq = torch.tensor(ar["KL_seq"], dtype=torch.float64)
        group = list(range(len(names))) if arm.endswith(("_all", "_nopos0")) else None
        si = SPECS.index(spec)
        rec = {"KL_true_ref": ar["KL_true"], "KL_true_test_not_truth": at["KL_true"],
               "P_add_ref": ar["P_add"], "P_add_test": at["P_add"],
               "KL_ref_over_Padd_ref": ar["KL_true"] / ar["P_add"] if ar["P_add"] else None,
               "KL_ref_over_Padd_test": ar["KL_true"] / at["P_add"] if at["P_add"] else None}
        if group is not None:
            # per-sequence diagonal price of the arm's units, against the reference KL per sequence
            for tag, c in (("ref", cr), ("test", ct)):
                d = (0.5 * c[si][group].pow(2).mean(1)).sum(0)      # [seq]
                rec[f"KL_ref_over_diag_{tag}"] = boot_ratio(kl_seq, d, args.boot, gen)
        report["arms"][arm] = rec
    out = args.test + ".vs_ref.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    for spec, rec in report["specs"].items():
        print(f"{spec}: full {rec['full_ref']:.4g} -> {rec['full_test']:.4g} "
              f"(x{rec['full_ratio']:.3f}); diag x{rec['diag_ratio']['ratio']:.3f} "
              f"CI {rec['diag_ratio']['ci95'][0]:.3f}-{rec['diag_ratio']['ci95'][1]:.3f}; "
              f"top5 excess share {rec['top5_excess_share']}")
        for row in rec["top_excess_units"][:6]:
            print("   ", json.dumps(row))
    for arm, rec in report["arms"].items():
        print(arm, json.dumps({k: v for k, v in rec.items()}))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
