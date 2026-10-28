#!/usr/bin/env python3
"""Reshard a single large .safetensors file into target-sized shards + HF index.

Streaming byte-copy per tensor (no torch, no GPU, RAM ~ the copy buffer): the
Hy3 CB export writes one 110 GB model.safetensors, but the HF Hub caps files at
50 GB. Produces model-0000N-of-0000M.safetensors + model.safetensors.index.json
in --out; tensor bytes are copied verbatim (bit-identical), source order kept.
Tensors are indivisible, so one tensor can exceed the requested shard size.

  reshard_safetensors.py SRC OUT_DIR [--shard-gb 25]
"""
import argparse
import json
import os
import struct
import sys

BUF = 64 * 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--shard-gb", type=float, default=25.0)
    args = ap.parse_args()
    cap = int(args.shard_gb * 1e9)

    with open(args.src, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    meta = hdr.pop("__metadata__", None)
    data_base = 8 + hlen

    # Greedy source-order packing into shards.
    shards: list[list[str]] = [[]]
    used = 0
    for name, e in hdr.items():
        sz = e["data_offsets"][1] - e["data_offsets"][0]
        if used and used + sz > cap:
            shards.append([])
            used = 0
        shards[-1].append(name)
        used += sz
    n = len(shards)
    os.makedirs(args.out_dir, exist_ok=True)

    weight_map: dict[str, str] = {}
    total = 0
    with open(args.src, "rb") as src:
        for i, names in enumerate(shards, 1):
            fname = f"model-{i:05d}-of-{n:05d}.safetensors"
            sh_hdr: dict = {}
            off = 0
            for name in names:
                e = hdr[name]
                sz = e["data_offsets"][1] - e["data_offsets"][0]
                sh_hdr[name] = {"dtype": e["dtype"], "shape": e["shape"],
                                "data_offsets": [off, off + sz]}
                off += sz
                weight_map[name] = fname
            if meta is not None:
                sh_hdr["__metadata__"] = meta
            hb = json.dumps(sh_hdr, separators=(",", ":")).encode()
            hb += b" " * (-len(hb) % 8)          # 8-byte alignment, spec-legal
            with open(os.path.join(args.out_dir, fname), "wb") as out:
                out.write(struct.pack("<Q", len(hb)))
                out.write(hb)
                for name in names:
                    e = hdr[name]
                    src.seek(data_base + e["data_offsets"][0])
                    left = e["data_offsets"][1] - e["data_offsets"][0]
                    while left:
                        chunk = src.read(min(BUF, left))
                        if not chunk:
                            raise IOError(f"short read in {name}")
                        out.write(chunk)
                        left -= len(chunk)
            total += off
            print(f"[reshard] {fname}: {len(names)} tensors, "
                  f"{off/1e9:.2f} GB", flush=True)

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(args.out_dir,
                           "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"[reshard] {n} shards, {total/1e9:.2f} GB, index written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
