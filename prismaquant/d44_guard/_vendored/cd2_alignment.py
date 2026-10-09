"""Unchanged CD2 alignment functions, not a capture or G3 instrument.

Recovered g3_cdiag.py lines 85-115 from campaign g3job head
2627750ff631077ae8b96da3467e050d6060beea on sparky.
"""
import torch
import torch.nn.functional as F


def recover_perm(g_dec, g_src, u_dec, u_src):
    """EXL3's intermediate-channel permutation for one expert: stored row i is source row perm[i].

    Returns (perm or None, diagnostics).  Gated: top-1 cosine is a bijection, gate and up agree,
    and the weakest top-1 match clears the strongest runner-up."""
    def top(dec, src):
        a = F.normalize(dec.float(), dim=1)
        b = F.normalize(src.float(), dim=1)
        c = a @ b.T
        v, i = c.topk(2, dim=1)
        return i[:, 0], v[:, 0], v[:, 1]
    gi, g1, g2 = top(g_dec, g_src)
    ui, u1, u2 = top(u_dec, u_src)
    bij = int(torch.unique(gi).numel()) == gi.numel()
    agree = bool(torch.equal(gi, ui))
    diag = {"min_top1": float(torch.minimum(g1, u1).min()), "max_second": float(torch.maximum(g2, u2).max()),
            "bijective": bij, "gate_up_agree": agree}
    ok = bij and agree and diag["min_top1"] > diag["max_second"]
    diag["ok"] = ok
    return (gi if ok else None), diag


def aligned(perm, wg, wu, wd):
    """Source-basis weights from EXL3's stored P.Wg, P.Wu, Wd.P^T (row i of the store is source row perm[i])."""
    g = torch.empty_like(wg)
    u = torch.empty_like(wu)
    d = torch.empty_like(wd)
    g[perm] = wg
    u[perm] = wu
    d[:, perm] = wd
    return g, u, d
