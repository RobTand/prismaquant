"""End-to-end proof that a shard read is served off PrismaBuild's stage tier.

Not a unit test: this uses the LIVE residency map composed for the GLM-5.3-Flash
joint run, the real staged ranges on /stage/prewarm, and the real seam
(layer_streaming._source_safe_open).  It measures bytes per NFS mount, so a
redirect that silently fell back to the pool cannot pass.

Read-only.  Writes nothing anywhere.
"""
import json, os, sys, time

MAP = ("/mnt/shared/prismabuild-fleet/pb-queue/residency/"
       "ad8803aa32333dffc52b5d35d7203e0a544a62c88babb013fe763312c35fefe4.map.json")


def delta(a, b, mnt):
    if mnt not in a or mnt not in b:
        return (0, 0)
    return (b[mnt][0] - a[mnt][0], b[mnt][1] - a[mnt][1])


def main():
    os.environ["PRISMABUILD_RESIDENCY_MAP"] = MAP
    from safetensors import safe_open
    from prismaquant.io_spans import nfs_read_bytes
    from prismaquant.layer_streaming import _source_safe_open
    from prismaquant.residency_map import residency_resolver, bind_residency_manifest

    m = json.load(open(MAP))
    entries = m["entries"]
    # The pass must say which read set it is entitled to; the joint run does this
    # at tessera_joint_aura.py:1894 from --data-manifest-sha256.
    bind_residency_manifest(m["manifest_sha256"])
    print(f"bound data manifest {m['manifest_sha256'][:16]}...")
    print(f"map: {len(entries)} entries, stage_root={m['stage_root']}")

    # A shard staged as two ranges with a gap between them: some tensors are on
    # the stage, some are not, which is the half-staged state the reader must
    # handle without ever serving partial bytes.
    declared = "/mnt/shared/models/GLM-5.3-Flash-BF16/model-00001-of-00120.safetensors"
    ranges = sorted(((v["offset"], v["bytes"], v["stage_path"])
                     for k, v in entries.items() if k.endswith(":" + declared)),
                    key=lambda r: r[0])
    print(f"\n{os.path.basename(declared)} is named by {len(ranges)} map range(s):")
    present = []
    for off, nb, sp in ranges:
        here = os.path.isfile(sp)
        size = os.path.getsize(sp) if here else -1
        flag = "on stage" if here and size == nb else ("SIZE %d != %d" % (size, nb) if here else "NOT ON STAGE")
        print(f"   [{off:>12} .. {off + nb:>12})  {nb / 2**30:6.2f} GiB  {flag}")
        if here and size == nb:
            present.append((off, nb, sp))
    print(f"   -> {len(present)} of {len(ranges)} usable; the rest must fall back to the pool")
    if not present:
        print("no usable staged range — cannot demonstrate"); return 2
    ranges = present

    with safe_open(declared, framework="pt") as f:
        header = json.loads(open(declared, "rb").read(8 + int.from_bytes(
            open(declared, "rb").read(8), "little"))[8:])
    base = 8 + int.from_bytes(open(declared, "rb").read(8), "little")
    rows = {k: v for k, v in header.items() if k != "__metadata__"}

    def covered(v):
        b, e = base + v["data_offsets"][0], base + v["data_offsets"][1]
        return any(off <= b and e <= off + nb for off, nb, _ in ranges)

    inside = [k for k, v in rows.items() if covered(v) and v["data_offsets"][1] > v["data_offsets"][0]]
    outside = [k for k, v in rows.items() if not covered(v) and v["data_offsets"][1] > v["data_offsets"][0]]
    print(f"\ntensors: {len(inside)} inside a staged range, {len(outside)} outside it")
    if not inside:
        print("no tensor is covered — cannot demonstrate"); return 2

    inside.sort(key=lambda k: rows[k]['data_offsets'][1] - rows[k]['data_offsets'][0], reverse=True)
    picked = inside[:6]
    picked_bytes = sum(rows[k]["data_offsets"][1] - rows[k]["data_offsets"][0] for k in picked)
    print(f"reading {len(picked)} staged tensors, {picked_bytes / 2**20:.1f} MiB of payload")

    resolver = residency_resolver()
    print(f"resolver active: {resolver is not None and getattr(resolver, 'active', True)}")

    before = nfs_read_bytes()
    t0 = time.time()
    served = {}
    with _source_safe_open(declared, framework="pt") as fh:
        for k in picked:
            served[k] = fh.get_tensor(k)
    read_s = time.time() - t0
    after = nfs_read_bytes()

    stage_d = delta(before, after, "/stage/prewarm")
    pool_d = delta(before, after, "/mnt/shared")
    print("\n=== bytes moved during the staged read ===")
    print(f"  /stage/prewarm  client {stage_d[0] / 2**20:9.1f} MiB   server {stage_d[1] / 2**20:9.1f} MiB")
    print(f"  /mnt/shared     client {pool_d[0] / 2**20:9.1f} MiB   server {pool_d[1] / 2**20:9.1f} MiB")
    print(f"  wall {read_s:.2f}s")

    rep = resolver.report() if resolver is not None else {}
    print("\n=== resolver report ===")
    for k in sorted(rep):
        print(f"  {k} = {rep[k]}")

    print("\n=== byte-for-byte against the pool file ===")
    bad = []
    with safe_open(declared, framework="pt") as fh:
        for k in picked:
            if not fh.get_tensor(k).equal(served[k]):
                bad.append(k)
    print(f"  {len(picked) - len(bad)} of {len(picked)} identical" + (f"  MISMATCH: {bad}" if bad else ""))

    ok = (not bad
          and stage_d[0] >= picked_bytes * 0.9
          and rep.get("range_hits", 0) + rep.get("hits", 0) > 0)
    print("\nVERDICT:", "STAGE-FED AND CORRECT" if ok else "NOT DEMONSTRATED")
    if not ok:
        print("  reason: " + ", ".join(filter(None, [
            f"{len(bad)} tensor(s) differ" if bad else "",
            f"stage client bytes {stage_d[0]} < 90% of {picked_bytes}" if stage_d[0] < picked_bytes * 0.9 else "",
            "resolver recorded no hit" if rep.get("range_hits", 0) + rep.get("hits", 0) == 0 else ""])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
