"""One declared staged range in the real G3 container, never a model quality arm.

Launch with g3_launch --reader-smoke JSON and pbrun's data manifest.
Use stage/share auto and qualified RAM auto.
The JSON fields are path, offset, bytes, and sha256.
The host path must match an existing G3 input mount.
This smoke qualifies only the declared range and its device consumer.
It proves no speed, energy, or whole-model residency result.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from g3_offline_decoded_kl import HashPool
from g3_pq_policy.digests import bytes_sha256hex
from g3_residency import staged_range, resolve_g3_origin, receipt

def prepare_g3_range(manifest_path, out, source_root):
    """Select ONE actually declared used-source range; write metadata, never copy/stage bytes."""
    import gzip
    from g3_lib import read_range
    with open(manifest_path, "rb") as f:
        opener = gzip.open if f.read(2) == b"\x1f\x8b" else open
    with opener(manifest_path, "rb") as f:
        manifest = json.load(f)
    source = Path(source_root)
    entry = next(e for e in manifest["entries"] if Path(e["path"]).is_relative_to(source)
                 and e["path"].endswith(".safetensors") and e["offset"] > 0
                 and (10 << 20) < e["bytes"] <= (128 << 20))
    raw = read_range(entry["path"], entry["offset"], entry["bytes"])
    entry = {**entry, "sha256": bytes_sha256hex(raw)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = {"path": "/source/" + Path(entry["path"]).relative_to(source).as_posix(),
            "offset": entry["offset"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
    declaration = {"schema": "prismaquant.prismabuild.data_manifest.v2",
                   "produced_by": {"tool": "g3_residency_smoke"}, "annotations": {},
                   "mount_prefix": manifest["mount_prefix"], "entries": [entry],
                   "entry_count": 1, "total_bytes": entry["bytes"],
                   "read_plan": {"phases": [{"name": "smoke", "entry_indices": [0],
                        "bytes": entry["bytes"], "cumulative_bytes": entry["bytes"]}],
                        "read_bytes": entry["bytes"]}}
    (out / "data-manifest.json").write_text(json.dumps(declaration))
    (out / "range.json").write_text(json.dumps(spec))
    print(json.dumps({"directory": str(out), "range": spec}), flush=True)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec")
    p.add_argument("--prepare-manifest", help="CPU-only: select one used-source range from the real G3 manifest")
    p.add_argument("--source-root", default="/mnt/shared/models/GLM-5.3-Flash-BF16")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.prepare_manifest:
        return prepare_g3_range(args.prepare_manifest, args.out, args.source_root)
    if not args.spec:
        p.error("a container smoke requires --spec")
    spec = json.loads(args.spec)
    torch.set_num_threads(1)
    raw = staged_range(spec['path'], spec['offset'], spec['bytes'])
    if raw is None:
        raise RuntimeError('smoke range is absent from the admitted map; no origin fallback qualifies it')
    if bytes_sha256hex(raw) != spec['sha256']:
        raise RuntimeError('smoke range differs from its own expected digest')
    if not torch.cuda.is_available():
        raise RuntimeError('G3 container smoke requires an admitted CUDA device')
    torch.cuda.reset_peak_memory_stats()
    tensor = torch.frombuffer(raw, dtype=torch.uint8).to('cuda')
    pool = HashPool(max_inflight_bytes=256 << 20)
    pool.submit(tensor, spec['sha256'], 'staged smoke GPU roundtrip')
    verified = pool.drain()
    pool.close()
    from g3_one_arm_smoke import compare_one_arm
    comparison = compare_one_arm(Path(args.out).with_suffix(".one-arm"), device="cuda", profiles=True)
    result = {'schema': 'campaign.g3.staged_consumer_smoke.v1', 'argv': sys.argv,
              'host_path': resolve_g3_origin(spec['path']), 'range': spec, 'own_digest_verified': True,
              'gpu_roundtrip_verified': verified, 'hash_copy_profile': pool.profile,
              'cuda_max_allocated': torch.cuda.max_memory_allocated(),
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'residency': receipt(), 'one_arm_comparison': comparison,
              'scope': 'one staged range, GPU hash copies, and a tiny one-arm KL comparison'}
    with Path(args.out).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
