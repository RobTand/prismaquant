#!/usr/bin/env python3
"""Real-data gate for the activation-row fix.

Runs ``measure_batched_gpu`` over real DSv4-Flash layer tensors and dumps the
full cost rows as JSON. Run it against the pristine checkout and the fixed one
and diff:

  * LAYER 0 (every Linear has 64 cached rows -> one row bucket): every field
    must be byte-identical, and must match the shipped shard.
  * A LATER LAYER (ragged rows): weight_mse identical (the encoder is
    untouched), output_mse DIFFERENT BY DESIGN -- that is the defect being
    fixed -- and n_activation_rows present in the new arm only.
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
from pathlib import Path

import torch
from torch import nn

_spec = ilu.spec_from_file_location(
    "cb_encode_replica",
    str(Path(__file__).resolve().parent / "cb_encode_replica.py"))
R = ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)


class _ActIndex:
    def __init__(self, entries):
        self._a = {}
        for name, path in entries.items():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            self._a[name] = (blob["inputs"], blob.get("row_indices"))

    def __contains__(self, n):
        return n in self._a

    def __len__(self):
        return len(self._a)

    def load(self, n):
        return self._a[n][0]

    def load_with_row_indices(self, n):
        return self._a[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--formats", default="NVFP4_CB_K14,FP8_CB_K36")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R.apply_prod_env()
    from prismaquant import format_registry as fr
    from prismaquant.measure_quant_cost import measure_batched_gpu

    model = nn.Module()
    acts, names = {}, []
    e = -1
    while len(names) < args.experts and e < 255:
        e += 1
        ap_ = (R.ACT / f"model__layers__{args.layer}__mlp__experts__{e}__"
                       f"{args.proj}.pt")
        if not ap_.exists():
            continue        # never-routed expert: no activation rows, skipped
        w = R.load_expert_weight(args.layer, e, args.proj)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(w.float())
        nm = f"e{e}"
        model.add_module(nm, lin)
        acts[nm] = ap_
        names.append(nm)

    idx = _ActIndex(acts)
    rows = {n: int(idx.load(n).shape[0]) for n in names}
    print(f"layer {args.layer} {args.proj}: rows per expert = "
          f"{[rows[n] for n in names]}")
    specs = [fr.get_format(f) for f in args.formats.split(",")]
    got = measure_batched_gpu(model, idx, set(names), specs, "cuda",
                              torch.bfloat16, chunk_size=256)
    payload = {n: {f: dict(v) for f, v in got[n].items()} for n in names}
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    for n in names[:4]:
        for f in sorted(payload[n]):
            print(f"  {n:4s} {f:14s} {payload[n][f]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
