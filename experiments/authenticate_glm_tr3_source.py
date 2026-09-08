#!/usr/bin/env python3
"""Explicit offline CPU authentication pass; submit through PrismaBuild.

This reads reference or candidate shard bodies once using the existing checkpoint mechanism.
It never pairs historical hashes with newly invented mutation fingerprints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

from experiments.build_glm_tr3_teacher import require_reference_binding
from experiments.glm_tr3_full_vocab import bound_json, sha256
from prismaquant.cost_streaming import build_source_checkpoint_identity
from tools.full_kl_teacher_payload import atomic_json_write, canonical_sha256


def require_checksum_binding(path, expected, identity, model):
    import hashlib
    import re
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("upstream checksum manifest digest mismatch")
    rows = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None or match[2] in rows:
            raise ValueError("upstream checksum manifest is malformed or repeats a path")
        name = match[2]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("upstream checksum manifest has an unsafe path")
        rows[name] = match[1]
    shards = identity["shards"]
    expected_shards = {name for name in rows if name.endswith(".safetensors")}
    if expected_shards != {row["name"] for row in shards}:
        raise ValueError("current candidate does not cover the upstream shard set")
    for row in shards:
        if row["sha256"] != rows[row["name"]]:
            raise ValueError("current candidate shard differs from upstream checksum")
    # Include every actual non-shard artifact file; upstream research/runtime
    # directories absent from this serving artifact are not required payloads.
    for file in Path(model).rglob("*"):
        if not file.is_file() or file.suffix == ".safetensors":
            continue
        name = file.relative_to(model).as_posix()
        if name.startswith(".cache/") or name == "SHA256SUMS":
            continue
        if name not in rows or sha256(file) != rows[name]:
            raise ValueError(f"candidate metadata differs from upstream checksum: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "digest-cache", "output"):
        parser.add_argument("--" + name, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference-binding")
    group.add_argument("--sha256sums")
    parser.add_argument("--reference-binding-sha256")
    parser.add_argument("--sha256sums-sha256")
    args = parser.parse_args()
    if args.reference_binding and not args.reference_binding_sha256:
        parser.error("reference binding requires its independent SHA256")
    if args.sha256sums and not args.sha256sums_sha256:
        parser.error("SHA256SUMS requires its independent SHA256")
    binding = bound_json(args.reference_binding, args.reference_binding_sha256) if args.reference_binding else None
    identity = build_source_checkpoint_identity(
        args.model, extra_shard_paths=list(Path(args.model).rglob("*.safetensors")),
        digest_cache_path=args.digest_cache)
    if binding is not None:
        require_reference_binding(binding, identity, args.model)
    else:
        require_checksum_binding(args.sha256sums, args.sha256sums_sha256, identity, args.model)
    atomic_json_write({"schema": "prismaquant.glm_tr3_source_authentication/1",
                       "identity": identity, "identity_sha256": canonical_sha256(identity),
                       "reference_binding_sha256": args.reference_binding_sha256,
                       "sha256sums_sha256": args.sha256sums_sha256,
                       "digest_cache_sha256": sha256(args.digest_cache)}, args.output)


if __name__ == "__main__":
    main()
