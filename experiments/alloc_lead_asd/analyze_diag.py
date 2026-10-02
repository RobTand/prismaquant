"""Summarize an a_side_diag.py result: the ratio chain per arm, per-layer and single-unit
ratios, per-sequence agreement and the position profile.  CPU only.

usage: analyze_diag.py STEM [STEM ...]     (reads STEM.json and STEM.pt)
"""
import json
import math
import sys

import torch


def ratio(a, b):
    return a / b if b else float("nan")


def spearman(x, y):
    rx = torch.tensor(x).argsort().argsort().double()
    ry = torch.tensor(y).argsort().argsort().double()
    return float(torch.corrcoef(torch.stack((rx, ry)))[0, 1])


def pearson(x, y):
    return float(torch.corrcoef(torch.stack((torch.tensor(x).double(),
                                              torch.tensor(y).double())))[0, 1])


def summarize(stem):
    d = json.load(open(stem + ".json"))
    t = torch.load(stem + ".pt")
    arms = d["arms"]
    print(f"== {stem}  model={d['args']['model'].rsplit('/', 1)[-1]} text={d['args']['text']} "
          f"dtype={d['args']['dtype']} N={d['n_global']} probes={len(d['seeds'])} "
          f"ids={d['ids_sha256'][:12]}")
    hdr = (f"{'arm':34s} {'P_add':>10s} {'P_joint':>10s} {'S_real':>10s} {'Q_real':>10s} "
           f"{'KL':>10s} {'KL/Padd':>8s} {'Pj/Pa':>6s} {'S/Pj':>6s} {'Q/S':>6s} {'KL/Q':>6s} "
           f"{'corr':>6s} {'slope':>6s}")
    print(hdr)
    for label, a in arms.items():
        if label.startswith(("A_unit:", "W4_unit:", "A_layer")):
            continue
        print(f"{label:34s} {a['P_add']:10.4g} {a['P_joint']:10.4g} {a['S_real']:10.4g} "
              f"{a['Q_real']:10.4g} {a['KL_true']:10.4g} {ratio(a['KL_true'], a['P_add']):8.3f} "
              f"{ratio(a['P_joint'], a['P_add']):6.2f} {ratio(a['S_real'], a['P_joint']):6.2f} "
              f"{ratio(a['Q_real'], a['S_real']):6.2f} {ratio(a['KL_true'], a['Q_real']):6.2f} "
              f"{a['corr_slin_sreal']:6.3f} {a['slope_slin_on_sreal']:6.2f}")
    # A increment on top of W, as G3 measures it
    if "W4A8_all" in arms and "W4_all" in arms:
        inc = arms["W4A8_all"]["KL_true"] - arms["W4_all"]["KL_true"]
        print(f"A increment on W4 (KL W4A8 - KL W4) = {inc:.4g}; A alone KL = "
              f"{arms['A_all']['KL_true']:.4g}; P_add(A) = {arms['A_all']['P_add']:.4g}; "
              f"increment/P_add = {ratio(inc, arms['A_all']['P_add']):.3f}")
    layers = sorted(k for k in arms if k.startswith("A_layer"))
    if layers:
        print("per-layer A: layer P_add KL KL/P_add corr")
        tot_p = tot_k = 0.0
        for k in layers:
            a = arms[k]
            tot_p += a["P_add"]
            tot_k += a["KL_true"]
            print(f"  {k:12s} {a['P_add']:10.4g} {a['KL_true']:10.4g} "
                  f"{ratio(a['KL_true'], a['P_add']):7.3f} {a['corr_slin_sreal']:6.3f}")
        print(f"  sum over layers: P_add {tot_p:.4g} KL {tot_k:.4g} ratio {ratio(tot_k, tot_p):.3f}; "
              f"A_all KL {arms['A_all']['KL_true']:.4g} (layer additivity {ratio(arms['A_all']['KL_true'], tot_k):.3f})")
    units = sorted(k for k in arms if k.startswith("A_unit:"))
    if units:
        print("single units: name A[P_add KL ratio] W4[P_add KL ratio]")
        ra, rw = [], []
        for k in units:
            name = k.split(":", 1)[1]
            a, w = arms[k], arms.get("W4_unit:" + name)
            rA = ratio(a["KL_true"], a["P_add"])
            rW = ratio(w["KL_true"], w["P_add"]) if w else float("nan")
            ra.append(rA)
            rw.append(rW)
            print(f"  {name:42s} A {a['P_add']:9.3g} {a['KL_true']:9.3g} {rA:6.3f} | "
                  f"W {w['P_add'] if w else float('nan'):9.3g} {w['KL_true'] if w else float('nan'):9.3g} {rW:6.3f}")
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"  median ratio A {med(ra):.3f}  W {med(rw):.3f}")
    for label in ("A_all", "W4_all"):
        a = arms[label]
        print(f"{label} per-sequence: pearson(P_joint_seq, KL_seq) {pearson(a['P_joint_seq'], a['KL_seq']):.3f} "
              f"spearman {spearman(a['P_joint_seq'], a['KL_seq']):.3f}; pearson(Q_seq, KL_seq) "
              f"{pearson(a['Q_seq'], a['KL_seq']):.3f}; top-4 seq share of KL "
              f"{sum(sorted(a['KL_seq'])[-4:]) / sum(a['KL_seq']):.3f}, of P_joint "
              f"{sum(sorted(a['P_joint_seq'])[-4:]) / sum(a['P_joint_seq']):.3f}")
    for label in ("A_all", "W4_all", "A_all_nopos0"):
        key = f"kl_pos:{label}"
        if key in t:
            kp = t[key].double()
            tot = float(kp.sum())
            bands = [(0, 1), (1, 16), (16, 128), (128, 512)]
            shares = [float(kp[a:b].sum()) / tot for a, b in bands]
            print(f"{label} KL share by position band {bands}: " + " ".join(f"{s:.3f}" for s in shares))


if __name__ == "__main__":
    for stem in sys.argv[1:]:
        summarize(stem)
