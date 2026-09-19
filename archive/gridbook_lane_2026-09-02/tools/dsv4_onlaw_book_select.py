#!/usr/bin/env python
"""Author and verify CB_ROUTED_MOE_BOOK_SELECTION for the on-law routed rungs.

The selection is the operator's explicit, fail-closed statement of which banked
burn shard backs each routed `(layer, projection, rung)`.  `load_banked_cbl_book`
deliberately never scans a directory and never falls back to a lattice, so this
file -- not the contents of the bank -- is what makes a rung renderable.

Digests deliberately do NOT live in the selection: the loader compares the
selected shard's `content_guard` against the producer's tensors at the moment
the bundle is built.  This tool therefore verifies the same way the exporter
will, by re-deriving `tensor_value_identity` from the burn's own population
loader (`dsv4_onlaw_book_burn.load_population`).

Verification modes:
  none    structural only -- every cell has an acceptable shard
  sample  + re-derive digests from source for a stratified sample (default)
  full    + re-derive for all 258 cells (~1 h, dominated by weight I/O)

Cross-rung consistency is checked in every mode and needs no GPU: the two rungs
of one (layer, projection) must carry byte-identical source and col_weights
digests, because they are the same tensors.  A mismatch means the two cells were
burned against different source state and the pair must not ship.

`--shard-root` / `--col-weights` / `--act-root` must name the SAME calibration
generation the burn used (see the CALIBRATION REBASE section of
dsv4_onlaw_book_burn).  Verification is exactly the check that fails when they
do not: `col_weights_digest` is derived from the imatrix passed here and
compared against the one the book was learned under.

`--keying` must name the keying the burn used (see KEYING in
dsv4_onlaw_book_burn): the census enumerates `(layer, gate_up_proj | down_proj)`
under stack keying and `(layer, gate_proj | up_proj | down_proj)` under role
keying, and the bundle builder binds a selection only under the keying it was
burned with.  The two forms write differently named selections in the same
shard root (`CB_ROUTED_MOE_BOOK_SELECTION.json` for role, the historical name;
`CB_ROUTED_MOE_BOOK_SELECTION.stack.json` for stack) so neither can be handed
to a build expecting the other by accident.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from prismaquant.cb_banked_books import (
    ROUTED_MOE_CBL_SELECTION_SCHEMA,
    BankedCBLBookRequest,
    load_banked_cbl_book,
    load_routed_moe_cbl_selection,
)
from tools import dsv4_afast_burn as burn
from tools import dsv4_ldlq_cost_campaign as ldlq_campaign
from tools import dsv4_onlaw_book_burn as ob
from prismaquant.routed_moe_codebooks import (
    DEFAULT_ROUTED_BOOK_KEYING,
    ROUTED_BOOK_KEYINGS,
    ROUTED_BOOK_KEYING_STACK,
    normalize_routed_book_keying,
)

BOOK_ROOT = burn.RUN_ROOT / "bucket-books"


def selection_filename(keying: str) -> str:
    keying = normalize_routed_book_keying(keying)
    if keying == ROUTED_BOOK_KEYING_STACK:
        return "CB_ROUTED_MOE_BOOK_SELECTION.stack.json"
    return "CB_ROUTED_MOE_BOOK_SELECTION.json"


def _guard(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (payload.get("identity") or {}).get("content_guard") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", default="28,32")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--verify", choices=("none", "sample", "full"), default="sample",
    )
    parser.add_argument("--sample-size", type=int, default=8)
    # Must match the burn's calibration generation; see the module docstring.
    parser.add_argument("--shard-root", default=None)
    parser.add_argument("--col-weights", default=None)
    parser.add_argument("--act-root", default=None)
    # Restrict the census to a layer subset.  The use for this is checking the
    # first cells of a running burn end-to-end, so a wrong-digest generation is
    # caught at minute two rather than after the whole bank is written.
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated layer subset; default every layer",
    )
    parser.add_argument(
        "--keying", choices=ROUTED_BOOK_KEYINGS,
        default=DEFAULT_ROUTED_BOOK_KEYING,
        help="the keying the burn used; see dsv4_onlaw_book_burn KEYING",
    )
    args = parser.parse_args()
    keying = normalize_routed_book_keying(args.keying)
    keys = ob.populations(keying)

    if bool(args.col_weights) != bool(args.act_root):
        raise SystemExit(
            "REFUSE: --col-weights and --act-root are one calibration and must "
            "be given together"
        )
    if args.shard_root:
        burn.BURN_CELL_ROOT = Path(args.shard_root)
    if args.col_weights:
        burn.COL_WEIGHTS = Path(args.col_weights)
    # ACT_ROOT is NOT installed here: it is installed by `resolve_identities`,
    # after the by-layer stores have validated against their own cache.
    # Keep a rebased selection out of the previous generation's filename.
    out = Path(args.out) if args.out else (
        burn.BURN_CELL_ROOT / selection_filename(keying)
        if args.shard_root
        else burn.RUN_ROOT / "onlaw-books" / selection_filename(keying)
    )

    layers = (
        [int(v) for v in args.layers.split(",") if v.strip()]
        if args.layers else list(range(burn.LAYER_COUNT))
    )
    rungs = tuple(int(r) for r in args.rungs.split(",") if r.strip())
    illegal = [r for r in rungs if r not in ob.ONLAW_ROUTED_RUNGS]
    if illegal:
        raise SystemExit(
            f"REFUSE: {illegal} is not an on-law bankable routed rung"
        )

    cells: list[dict] = []
    missing: list[str] = []
    for layer in layers:
        for projection in keys:
            for rung in rungs:
                shard = ob.accepted_shard(layer, projection, rung)
                if shard is None:
                    missing.append(f"L{layer:02d} {projection} K{rung}")
                    continue
                cells.append({
                    "layer": layer, "projection": projection,
                    "rung": rung, "burn_shard": str(shard),
                })
    expected = len(layers) * len(keys) * len(rungs)
    print(f"[select] keying={keying} populations={list(keys)}")
    print(f"[select] {len(cells)}/{expected} cells have an acceptable shard")
    if missing:
        print(f"[select] REFUSE: {len(missing)} cells unbanked, e.g. "
              f"{missing[:5]}"
              + (" (burn them with dsv4_onlaw_book_burn --keying stack)"
                 if keying == ROUTED_BOOK_KEYING_STACK else ""))
        return 1

    # Cross-rung digest consistency (no GPU): same tensors, same digests.
    by_cell = {(c["layer"], c["projection"], c["rung"]): c for c in cells}
    inconsistent = []
    for layer in layers:
        for projection in keys:
            guards = [
                _guard(Path(by_cell[(layer, projection, r)]["burn_shard"]))
                for r in rungs
            ]
            digests = {
                (g.get("source_digest"), g.get("col_weights_digest"))
                for g in guards
            }
            if len(digests) > 1:
                inconsistent.append(f"L{layer:02d} {projection}")
    if inconsistent:
        print(f"[select] REFUSE: {len(inconsistent)} (layer, population) pairs "
              f"disagree on source digest across rungs: {inconsistent[:5]}")
        return 1
    print(f"[select] cross-rung digest consistency OK "
          f"({len(layers) * len(keys)} pairs)")

    # A partial census must never be written: a selection file is read as the
    # complete accepted set, and one covering 3 of 129 pairs would render the
    # rest by silently missing rather than by refusing.
    if len(layers) == burn.LAYER_COUNT:
        if out.is_file():
            previous = json.loads(out.read_text())
            previous_keys = sorted({
                str(cell.get("projection"))
                for cell in (previous.get("cells") or [])
            })
            if previous_keys != sorted(keys):
                raise SystemExit(
                    f"REFUSE: {out} holds a selection over {previous_keys}, "
                    f"not the {keying} populations {sorted(keys)}; pass "
                    "--out rather than overwriting the other keying's "
                    "selection"
                )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": ROUTED_MOE_CBL_SELECTION_SCHEMA,
            "book_root": str(BOOK_ROOT),
            "cells": cells,
        }, indent=2, sort_keys=True) + "\n")
        selection = load_routed_moe_cbl_selection(out)
        print(f"[select] wrote {out}")
        print(f"[select] selection sha256: {selection.content_sha256}")
    else:
        print(f"[select] layer subset ({len(layers)}/{burn.LAYER_COUNT}); "
              "verifying only, no selection written")

    if args.verify == "none":
        return 0

    targets = sorted({(c["layer"], c["projection"]) for c in cells})
    if args.verify == "sample":
        step = max(1, len(targets) // args.sample_size)
        targets = targets[::step][:args.sample_size]
    print(f"[select] verifying {len(targets)} (layer, population) loads "
          f"x {len(rungs)} rungs against freshly loaded source")

    if not torch.cuda.is_available():
        raise SystemExit("REFUSE: digest verification requires CUDA")
    burn._configure()
    from prismaquant.cb_warm_state import tensor_value_identity

    with burn.COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    model_to_shard, model_to_ckpt = burn._build_weight_map(str(burn.SOURCE))
    scale_map = burn._build_fp8_scale_inv_map(str(burn.SOURCE))

    identities = ob.resolve_identities(
        {layer for layer, _ in targets},
        all_col_weights,
        act_root=args.act_root,
    )
    profile = None
    if keying == ROUTED_BOOK_KEYING_STACK:
        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(str(burn.SOURCE))

    verified = 0
    for layer, projection in targets:
        data = ob.load_population(
            layer, projection, keying=keying, profile=profile,
            device=torch.device("cuda:0"),
            identity=identities[layer], all_col_weights=all_col_weights,
            model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
            scale_map=scale_map,
        )
        try:
            source_identity = tensor_value_identity(data["weight"])
            col_identity = tensor_value_identity(data["col_weights"])
            for rung in rungs:
                cell = by_cell[(layer, projection, rung)]
                load_banked_cbl_book(
                    BankedCBLBookRequest(
                        burn_shard_path=Path(cell["burn_shard"]),
                        layer=layer, projection=projection, rung=rung,
                        source_digest=source_identity[1],
                        col_weights_digest=col_identity[1],
                        source_shape=source_identity[0],
                        col_weights_shape=col_identity[0],
                    ),
                    book_root=BOOK_ROOT,
                )
                verified += 1
        finally:
            del data
            torch.cuda.empty_cache()
        print(f"[select] verified L{layer:02d} {projection} "
              f"({verified} cells)", flush=True)

    print(f"[select] VERIFIED {verified} cells, zero fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
