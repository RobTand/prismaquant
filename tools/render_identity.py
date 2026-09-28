"""Is a production render bit-identical across two GB10 boxes?

This is the gate on whether PrismaBuild can carry quantization work at all.  A
CAS receipt keyed by an action key promises that the same action yields the same
bytes anywhere.  If a render differs between sparky and sparklina, that promise
is false and the cache would silently serve one box's result for the other's.

Renders the real production path (GPTQ + static_act_order + JSO, the shipping
levers) on real weights and prints a digest per tensor.  Run on both boxes, diff
the output::

    python3 tools/render_identity.py --out /home/rob/tmp/render_$(hostname).json

Moved from PrismaBuild's ``tools/fleet/`` on 2026-09-28 (PB #1076): it renders
through PrismaQuant, so it lives here, and PrismaBuild imports no client.

**The activations mapping is keyed by qname, not by "input".**  A wrong key is
not an error: `render_production_weight` does `activations.get(qname)`, gets
None, and silently renders the RTN path -- plausible digests that would compare
equal across boxes while measuring nothing the shipping recipe does.  The first
version of this script had exactly that bug.  So the script now proves the
levers engaged before it reports identity: it renders one tensor with levers={}
and refuses to write a result unless that digest DIFFERS from the levered one.

Everything that renders or writes is inside ``main``. Importing this file must
not run the experiment: the body used to load the model, occupy the GPU and
overwrite ``render_<host>.json`` in the live store, so any importer silently
replaced a recorded cross-box result with a fresh one. That is how the
2026-08-31 sparky result was overwritten on 2026-09-05. For the same reason
``--out`` has no default: the recorded results under the fleet directory are
never a destination this tool picks for itself.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import socket

os.environ.setdefault("PYTHONHASHSEED", "0")

MODEL = "/mnt/shared/models/GLM-5.3-Flash-4layer"
LEVERS = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}


def digest(t) -> str:
    # view as uint8 to hash the exact bits: bfloat16 has no numpy dtype, and
    # casting to float32 would hide a low-bit difference, which is the entire
    # thing this script exists to detect.
    import torch

    flat = t.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL,
                        help="checkpoint directory whose layer-0 weights are rendered")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="JSON file this box's digests are written to; run on "
                             "each box with its own path and diff the two")
    args = parser.parse_args(argv)
    # Rendering needs torch and the production cache; ``--help`` does not.
    import torch
    from safetensors import safe_open
    from prismaquant.production_weight_cache import render_production_weight

    torch.manual_seed(0)
    shards = sorted(glob.glob(f"{args.model}/*.safetensors"))
    rows = []
    picked = 0
    lever_proof = None
    for path in shards:
        with safe_open(path, "pt") as f:
            for key in sorted(f.keys()):
                if not key.endswith(".weight") or ".layers.0." not in key:
                    continue
                if not any(r in key for r in ("q_proj", "gate_proj", "down_proj")):
                    continue
                W = f.get_tensor(key)
                if W.ndim != 2 or min(W.shape) < 256:
                    continue
                W = W[:512, :1024].to("cuda").to(torch.bfloat16).contiguous()
                # Deterministic synthetic activations: same seed -> same bytes on
                # both boxes, so any digest difference is the RENDER, not the input.
                g = torch.Generator(device="cuda"); g.manual_seed(1234)
                X = torch.randn(256, W.shape[1], generator=g, device="cuda", dtype=torch.bfloat16)
                acts = {key: X}          # keyed by qname -- see the module docstring
                for fmt in ("NVFP4", "FP8_E4M3"):
                    try:
                        out = render_production_weight(
                            W, fmt, qname=key, activations=acts, levers=LEVERS
                        )
                        rows.append({"qname": key, "fmt": fmt, "digest": digest(out),
                                     "shape": list(out.shape), "dtype": str(out.dtype)})
                    except Exception as exc:
                        rows.append({"qname": key, "fmt": fmt, "error": repr(exc)[:200]})
                if lever_proof is None:
                    # Prove the levers changed the bytes.  If this comes back equal,
                    # the render fell through to RTN and the identity result below
                    # would be measuring the wrong code path.
                    rtn = digest(render_production_weight(
                        W, "NVFP4", qname=key, activations=acts, levers={}))
                    levered = next(r["digest"] for r in rows
                                   if r["qname"] == key and r["fmt"] == "NVFP4")
                    lever_proof = {"qname": key, "levered": levered, "rtn": rtn,
                                   "engaged": levered != rtn}
                picked += 1
                if picked >= 3:
                    break
        if picked >= 3:
            break

    if not (lever_proof and lever_proof["engaged"]):
        raise SystemExit(
            f"REFUSING to report identity: levers did not engage -- {lever_proof}. "
            "The levered render is byte-equal to the RTN render, so this run does "
            "not measure the shipping path."
        )

    payload = json.dumps({"host": socket.gethostname(),
                      "torch": torch.__version__,
                      "device": torch.cuda.get_device_name(0),
                      "input_digest": digest(X),
                      "lever_proof": lever_proof,
                      "rows": rows}, indent=1)
    args.out.write_text(payload)
    print(f"wrote {args.out}  (levers engaged: {lever_proof['engaged']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
