#!/usr/bin/env python3
"""CD2: Tessera T-8 R1024 encoder factorial on the CD1 sampled experts (codec-decomp, 2026-09-30).

Question: EXL3 beats our routed T-8 R1024 on the G3 finals at matched bytes while T-8 has the lower
distribution-free weight error (REPORT 4.2).  Which encoder input closes it?  Three candidates, each
changed alone and in combination, everything else the shipped encode:
  * H   -- the Hessian the LDLQ and the row-scale refit see:
           wiki   = the production capture (wikitext-2 draw, 262,144 tokens; digest-gated against
                    the encoder-committed identity in cached_units.pact-release-t8-20260928);
           panel_all / panel_na4 = the unit's own X^T X over the panel's selection+confirmation
                    windows (document-disjoint from the finals; CD1 fitH), all / non-axis4 only;
           finals = X^T X over the finals' own routed rows (IN-SAMPLE: a P-label bound on what any
                    H draw can buy, never a candidate);
  * sigma -- the LDLQ regulariser (1.0 shipped; 0.025 is EXL3's);
  * rot -- RotationState.NONE (shipped) or R_IN_ONLY (block Hadamard on the input axis; the
           decoder undoes it, so the render is in the source basis).
v0 = (wiki, 1.0, NONE) is the shipped encode and must reproduce the A8 export's rendered sha256 for
every unit (--allow-v0-drift records a mismatch instead of refusing).

Evaluated on the finals' routed rows of each sampled expert (CD1 eval/, the x, router weight and token
index the executed experts module saw), fp32, per window (25):
    full = sum_t ||w_t (f_c(x_t) - f_s(x_t))||^2        codec gate/up/down
    gu   = sum_t ||w_t (h_c(x_t) Wd_s^T - f_s(x_t))||^2   codec gate/up, source down
    d    = sum_t ||w_t (h_s(x_t) Wd_c^T - f_s(x_t))||^2   source gate/up, codec down
for every variant, and for the decoded served bytes: REF (A8 T-8 R1024), EXL3 (aligned by recovered P),
the R960/R1088 bracket (the local rate-distortion slope, to convert an error change into bytes).
Every decode is hash-gated against manifest v3; every source tensor against its source_sha256.
This is a measurement instrument, not a price: nothing here feeds an allocator.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("TESSERA_SRC", "/tessera/src"))
sys.path.insert(0, os.environ.get("G3_PQ_ROOT", "/pq"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import g3_lib as L  # noqa: E402

FMT = "TESSERA_E4M3_K1_R1024"
BRACKET = ("TESSERA_E4M3_K1_R960", "TESSERA_E4M3_K1_R1088")
EXL3 = "EXL3"
ROLES = ("gate_proj", "up_proj", "down_proj")
CTX = 2048
H_SETS = ("wiki", "panel_all", "panel_na4", "finals")
SIGMAS = (1.0, 0.025)
ROTS = ("NONE", "R_IN_ONLY")
V0 = ("wiki", 1.0, "NONE")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def variants(h_sets=H_SETS, sigmas=SIGMAS, rots=ROTS):
    return [(h, s, r) for r in rots for h in h_sets for s in sigmas]


def vkey(v):
    return f"{v[0]}|s{v[1]:g}|{v[2]}"


# ------------------------------------------------------------------ the expert metric
def act(gu):
    from g3_cdiag import act as _act
    return _act(gu)


@torch.no_grad()
def expert_errors(x, w, win, nwin, src, cod, rows_out=None, key=None):
    """Per-window (full, gu, d) sums for one expert; src/cod = {role: [out, in] tensor}.

    With rows_out, also stores the per-row full error (float32) under rows_out[key], so the
    analysis can read the tail of a variant on the same routed rows, not only its window sums.
    """
    xf = x.float()
    wf = w.float()[:, None]
    gsrc = torch.cat([src["gate_proj"], src["up_proj"]], 0).float()
    dsrc = src["down_proj"].float()
    gcod = torch.cat([cod["gate_proj"], cod["up_proj"]], 0).float()
    dcod = cod["down_proj"].float()
    h_s = act(xf @ gsrc.T)
    y_s = h_s @ dsrc.T
    h_c = act(xf @ gcod.T)
    cols = [(wf * (h_c @ dcod.T - y_s)).pow(2).sum(1),
            (wf * (h_c @ dsrc.T - y_s)).pow(2).sum(1),
            (wf * (h_s @ dcod.T - y_s)).pow(2).sum(1),
            (wf * y_s).pow(2).sum(1)]
    out = torch.zeros(nwin, len(cols), dtype=torch.float64, device=x.device)
    out.index_add_(0, win, torch.stack(cols, 1).double())
    if rows_out is not None:
        rows_out[key] = cols[0].cpu().numpy().astype(np.float32)
    return out.cpu().numpy()


def load_cdiag(run, allow_partial=False):
    """(records, number of final windows, the bytes the panel H provenance hashes, partial?).

    A finished CD1 run is read from result.json.  With allow_partial, a run that never finished
    (withdrawn) is read from layers.partial.json: a layer counts only when its record was written,
    which is after its arrays, eval rows and fit Hessians.  The window count is then read off a
    written layer's arrays, never assumed.
    """
    run = Path(run)
    if (run / "result.json").is_file():
        raw = (run / "result.json").read_bytes()
        cd = json.loads(raw)
        return cd["layers"], len(cd["final_windows"]), raw, False
    if not allow_partial:
        raise SystemExit(f"{run} has no result.json; CD1 did not finish")
    raw = (run / "layers.partial.json").read_bytes()
    recs = json.loads(raw)["layers"]
    done = [r for r in recs if "sampled_experts" in r.get("cdiag", {})]
    if not done:
        raise SystemExit(f"{run}: no sampled layer finished")
    n = np.load(run / "layers" / f"L{int(done[0]['layer']):02d}.npz")[f"full__{FMT}"].size
    if n % CTX:
        raise SystemExit(f"{run}: {n} per-token rows is not a whole number of {CTX}-token windows")
    return recs, n // CTX, raw, True


# ------------------------------------------------------------------ bytes in, identities checked
def read_wire(ent, roots):
    loc = ent["wire"]
    root = roots[loc["root"]]
    if "ranges" in loc:
        blob = b"".join(L.read_range(os.path.join(root, sh), off, n) for sh, off, n in loc["ranges"])
        if len(blob) != loc["member_bytes"] or sha_bytes(blob) != loc["wire_sha256"]:
            raise L.HashGateError(f"EXL3 wire ({len(blob)} B) differs from the pre-pass framing")
        return blob
    off, n = L.member_location(loc)
    return L.read_range(os.path.join(root, loc["shard"]), off, n)


def decode_served(fmt, ent, roots, device, what):
    blob = read_wire(ent, roots)
    out_f, in_f = ent["rendered_shape"]
    if fmt == EXL3:
        import exl3_torch
        dec = exl3_torch.decode_wire(blob, in_f, out_f, device=device)
    else:
        from tessera.unit_artifact import read_unit_artifact
        dec = read_unit_artifact(blob, device=device)
    dec = dec.to(torch.bfloat16)
    L.check_identity(dec, ent["rendered_sha256"], f"{what} [{fmt}]")
    return dec, len(blob)


class Encoder:
    """The shipped T-8 R1024 encode (PrismaQuant's tessera_render seam: grid, q256, name, and the
    ActivationSource.for_unit keywords on the recipe's own scale plane), with the three factors
    exposed: the Hessian handed to ActivationSource, its ldlq_sigma, and encode_linears' rotation."""

    def __init__(self, fmt=FMT):
        from prismaquant.tessera_formats import parse_tessera_format_name, tessera_wire_recipe
        from prismaquant.tessera_render import _grid_for
        import tessera.export as tx
        family, rung = parse_tessera_format_name(fmt)
        self.fmt, self.q256 = fmt, int(rung)
        self.grid = _grid_for(family)
        self.recipe = tessera_wire_recipe(family, rung)
        self.tx = tx
        self.walls = []

    def kwargs(self, name, H, provenance, sigma, rot, in_features, device):
        from tessera.manifest import RotationState
        src = self.tx.ActivationSource({name: H}, dict(provenance), ldlq_sigma=float(sigma))
        return src.for_unit(name, int(in_features), device, scale_plane=self.recipe.scale_plane,
                            rotation=RotationState[rot])

    def encode(self, weights, per_unit, rot):
        """One batch at one rotation: same-shape weights and their for_unit kwargs, through
        encode_linears(per_unit=...) -- each blob byte-identical to encode_linear alone (tessera#385;
        test_factorial checks it here).  Returns [(render bf16, blob)], render = the bytes decoded."""
        from tessera.manifest import RotationState
        from tessera.unit_artifact import read_unit_artifact
        t0 = time.perf_counter()
        units = self.tx.encode_linears(list(weights), grid=self.grid, q256=self.q256,
                                       names=[self.fmt] * len(weights), per_unit=list(per_unit),
                                       verify=False, rotation=RotationState[rot])
        if weights[0].is_cuda:
            torch.cuda.synchronize()
        self.walls.append({"units": len(weights), "shape": list(weights[0].shape), "rot": rot,
                           "s": time.perf_counter() - t0})
        dev = str(weights[0].device)
        return [(read_unit_artifact(u.blob, device=dev).to(torch.bfloat16), u.blob) for u in units]


def load_wiki_hessian(capture, qname, cached_units):
    from tessera.cached_unit import digest_host_tensor
    d = torch.load(Path(capture) / f"{qname.replace('.', '__')}.pt", map_location="cpu", weights_only=True)
    H = d["hessian"].contiguous()
    ident = cached_units[qname]["identity"]
    want = ident["calibration"]["hessian"]["sha256"]
    got = digest_host_tensor(H)
    if got != want:
        raise SystemExit(f"{qname}: wikitext Hessian digest {got[:16]} != encoder-committed {want[:16]}")
    prov = dict(ident["calibration"]["settings"]["hessian"])
    return H, int(d["count"]), prov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdiag", required=True, help="the CD1 run dir (eval/, fitH/, result.json)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--source", required=True, help="GLM-5.3-Flash-BF16")
    ap.add_argument("--capture", required=True, help="the census calibration-cache/inputs dir")
    ap.add_argument("--cached-units", required=True)
    ap.add_argument("--a8-export", required=True)
    ap.add_argument("--wire-cache", required=True)
    ap.add_argument("--exl3-dir", required=True)
    ap.add_argument("--t8r-export", required=True)
    ap.add_argument("--layers", help="comma list; default every sampled layer in the CD1 run")
    ap.add_argument("--experts-per-layer", type=int, default=0, help="0 = all sampled")
    ap.add_argument("--h-sets", default=",".join(H_SETS))
    ap.add_argument("--sigmas", default=",".join(f"{s:g}" for s in SIGMAS))
    ap.add_argument("--rots", default=",".join(ROTS))
    ap.add_argument("--allow-v0-drift", action="store_true")
    ap.add_argument("--allow-partial-cdiag", action="store_true",
                    help="read a withdrawn CD1 run's finished sampled layers (layers.partial.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import tessera
    tessera_src = os.environ.get("TESSERA_SRC", "/tessera/src")
    if not tessera.__file__.startswith(tessera_src):
        raise SystemExit(f"tessera imported from {tessera.__file__}, not the pin {tessera_src}")
    from prismaquant.gpu_guard import require_cuda_hot_path
    device = require_cuda_hot_path("codec_factorial", "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tok").mkdir()
    raw = Path(a.manifest).read_bytes()
    if sha_bytes(raw) != a.manifest_sha256:
        raise SystemExit("manifest sha256 differs")
    m = json.loads(raw)
    if m.get("schema") != "surrogate-diag.g3.unit_manifest.v3":
        raise SystemExit("the factorial reads the manifest v3 per-row formats tables")
    rows = {(int(r["layer"]), int(r["expert"]), r["role"]): r for r in m["rows"] if r["kind"] == "routed"}
    cached_units = json.loads((Path(a.cached_units)).read_bytes())["units"]
    wm = json.loads((Path(a.source) / "model.safetensors.index.json").read_bytes())["weight_map"]
    roots = {"a8": a.a8_export, "wirecache": a.wire_cache, "exl3": a.exl3_dir, "t8r": a.t8r_export}
    cd_layers, nwin, cd_raw, cd_partial = load_cdiag(a.cdiag, a.allow_partial_cdiag)
    sampled = {int(r["layer"]): r["cdiag"]["sampled_experts"] for r in cd_layers if "sampled_experts" in r.get("cdiag", {})}
    if a.layers and not set(int(x) for x in a.layers.split(",")) <= set(sampled):
        raise SystemExit(f"--layers {a.layers}: the CD1 run has sampled layers {sorted(sampled)} only")
    layers = sorted(sampled) if not a.layers else [int(x) for x in a.layers.split(",")]
    vs = variants(a.h_sets.split(","), [float(s) for s in a.sigmas.split(",")], a.rots.split(","))
    if V0 not in vs:
        raise SystemExit(f"the factorial must include v0 {V0}")
    enc = Encoder()
    from safetensors import safe_open
    env = {"schema": "codec-decomp.factorial.v1", "cdiag": str(a.cdiag),
           "cdiag_result_sha256": sha_bytes(cd_raw), "cdiag_partial": cd_partial,
           "manifest_sha256": a.manifest_sha256, "tessera": tessera.__file__, "torch": torch.__version__,
           "host": os.uname().nodename, "argv": sys.argv, "fmt": FMT, "variants": [vkey(v) for v in vs],
           "v0": vkey(V0), "recipe_scale_plane": str(enc.recipe.scale_plane), "tf32": False,
           "container_content_sha256": os.environ.get("PRISMAQUANT_CONTAINER_CONTENT_SHA256"),
           "definitions": {"cols": ["full", "gu", "d", "ref_out"], "per": "window (finals index)",
                           "finals": "IN-SAMPLE H: P-label bound only"}}
    experts = []
    t_start = time.time()
    for layer in layers:
        chosen = sampled[layer][: a.experts_per_layer or None]
        for e in chosen:
            t0 = time.time()
            tag = f"L{layer:02d}_E{e:03d}"
            ev = torch.load(Path(a.cdiag) / "eval" / f"{tag}.pt", map_location="cpu", weights_only=True)
            fh = torch.load(Path(a.cdiag) / "fitH" / f"{tag}.pt", map_location="cpu", weights_only=True)
            x = ev["x"].to(device)
            w = ev["w"].to(device)
            win = (ev["tok"] // CTX).to(device)
            rec = {"layer": layer, "expert": e, "finals_rows": int(x.shape[0]), "fit_tokens": fh["n"],
                   "units": {}, "served": {}, "variants": {}}
            tok_rows = {}
            src = {}
            for role in ROLES:
                r = rows[(layer, e, role)]
                q = r["qname"]
                with safe_open(str(Path(a.source) / wm[q + ".weight"]), framework="pt", device="cpu") as f:
                    t = f.get_tensor(q + ".weight")
                L.check_identity(t, r["source_sha256"], f"{q} source")
                src[role] = t.to(device)
            # the served bytes, decoded and gated
            served = {}
            for fmt in (FMT, EXL3) + BRACKET:
                dec = {}
                nbytes = 0
                for role in ROLES:
                    r = rows[(layer, e, role)]
                    d, nb = decode_served(fmt, r["formats"][fmt], roots, device, r["qname"])
                    dec[role] = d
                    nbytes += nb
                if fmt == EXL3:
                    from g3_cdiag import aligned, recover_perm
                    perm, diag = recover_perm(dec["gate_proj"], src["gate_proj"], dec["up_proj"], src["up_proj"])
                    rec["exl3_align"] = {k: v for k, v in diag.items()}
                    if perm is None:
                        raise SystemExit(f"{tag}: EXL3 permutation not recovered {diag}")
                    dec["gate_proj"], dec["up_proj"], dec["down_proj"] = aligned(perm, dec["gate_proj"], dec["up_proj"], dec["down_proj"])
                served[fmt] = dec
                rec["served"][fmt] = {"bytes": nbytes, "err": expert_errors(x, w, win, nwin, src, dec, tok_rows,
                                                                             f"served__{fmt}").tolist()}
            # the Hessians, per input group
            H = {}
            for grp, role_src in (("gu", "gate_proj"), ("d", "down_proj")):
                q = rows[(layer, e, role_src)]["qname"]
                wiki, count, prov = load_wiki_hessian(a.capture, q, cached_units)
                if grp == "gu":
                    up_q = rows[(layer, e, "up_proj")]["qname"]
                    wiki_up, _c, _p = load_wiki_hessian(a.capture, up_q, cached_units)
                    rec["wiki_gate_up_identical"] = bool(torch.equal(wiki, wiki_up))
                    if not rec["wiki_gate_up_identical"]:
                        raise SystemExit(f"{tag}: gate and up wikitext Hessians differ; one H per group is wrong")
                    xf = x.float()
                    fin = (xf.T @ xf).cpu()
                else:
                    hs = act(x.float() @ torch.cat([src["gate_proj"], src["up_proj"]], 0).float().T)
                    fin = (hs.T @ hs).cpu()
                ph = fh["H_gu" if grp == "gu" else "H_d"]
                n = fh["n"]
                H[grp] = {"wiki": (wiki, prov),
                          "panel_all": (ph[0] + ph[1], {"text_sha256": env["cdiag_result_sha256"], "fit_tokens": n[0] + n[1],
                                                        "fit_ids_sha256": "cdiag-fit-windows:all"}),
                          "panel_na4": (ph[0].clone(), {"text_sha256": env["cdiag_result_sha256"], "fit_tokens": n[0],
                                                        "fit_ids_sha256": "cdiag-fit-windows:non-axis4"}),
                          "finals": (fin, {"text_sha256": env["cdiag_result_sha256"], "fit_tokens": int(x.shape[0]),
                                           "fit_ids_sha256": "finals-in-sample"})}
                rec.setdefault("wiki_count", {})[grp] = count
            # encode: per rotation, gate+up in one batch (same shape, same H per variant), down in another
            dec_v = {vkey(v): {} for v in vs}
            for rot in sorted({v[2] for v in vs}, key=ROTS.index):
                vr = [v for v in vs if v[2] == rot]
                for roles_b, grp in ((("gate_proj", "up_proj"), "gu"), (("down_proj",), "d")):
                    weights, per, keys = [], [], []
                    for role in roles_b:
                        q = rows[(layer, e, role)]["qname"]
                        for v in vr:
                            Hm, prov = H[grp][v[0]]
                            weights.append(src[role])
                            per.append(enc.kwargs(q, Hm, prov, v[1], rot, src[role].shape[1], device))
                            keys.append((role, v))
                    got = enc.encode(weights, per, rot)
                    for (role, v), (render, blob) in zip(keys, got):
                        dec_v[vkey(v)][role] = render
                        rec["units"].setdefault(vkey(v), {})[role] = {"bytes": len(blob), "blob_sha256": sha_bytes(blob),
                                                                       "rendered_sha256": L.tensor_sha256(render)}
                    del per
            # v0 must be the shipped bytes
            v0 = vkey(V0)
            rec["v0_equals_a8"] = {role: rec["units"][v0][role]["rendered_sha256"] == rows[(layer, e, role)]["formats"][FMT]["rendered_sha256"]
                                   for role in ROLES}
            if not all(rec["v0_equals_a8"].values()) and not a.allow_v0_drift:
                json.dump({"env": env, "experts": experts + [rec], "complete": False, "refused": "v0 != A8"},
                          open(out / "factorial.partial.json", "w"))
                raise SystemExit(f"{tag}: v0 {v0} does not reproduce the A8 rendered bytes {rec['v0_equals_a8']}")
            for k, dec in dec_v.items():
                rec["variants"][k] = {"bytes": sum(u["bytes"] for u in rec["units"][k].values()),
                                      "err": expert_errors(x, w, win, nwin, src, dec, tok_rows,
                                                           f"variant__{k}").tolist()}
            # per-row full error of every decode on the same routed rows (the tail readout)
            np.savez(out / "tok" / f"{tag}.npz", tok=ev["tok"].numpy(), **tok_rows)
            rec["tok_rows"] = f"tok/{tag}.npz"
            rec["seconds"] = time.time() - t0
            experts.append(rec)
            ref = np.asarray(rec["served"][FMT]["err"]).sum(0)
            ex = np.asarray(rec["served"][EXL3]["err"]).sum(0)
            brief = {k: round(float(np.asarray(v["err"]).sum(0)[0] / ref[0]), 4) for k, v in rec["variants"].items()}
            log(tag, f"{rec['seconds']:.0f}s rows {rec['finals_rows']} EXL3/REF {ex[0] / ref[0]:.4f} v0==A8 {rec['v0_equals_a8']}", brief)
            json.dump({"env": env, "experts": experts, "encode_walls": enc.walls, "complete": False},
                      open(out / "factorial.partial.json", "w"))
            del x, w, win, src, served, dec_v, H
            torch.cuda.empty_cache()
    json.dump({"env": env, "experts": experts, "encode_walls": enc.walls, "complete": True,
               "seconds": time.time() - t_start}, open(out / "factorial.json", "w"), indent=1)
    log("DONE", len(experts), "experts", round(time.time() - t_start), "s")


if __name__ == "__main__":
    main()
