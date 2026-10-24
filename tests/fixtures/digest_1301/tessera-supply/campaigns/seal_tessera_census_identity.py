def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="census cost.anchors.json")
    parser.add_argument("--out", required=True, help="new roster JSON path")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        raw = handle.read()
    digest.update(raw)
    manifest = json.loads(raw)
    del raw
    roster = seal_roster(manifest)
    roster["checkpoint_manifest"] = {"path": str(manifest_path), "sha256": digest.hexdigest(),
                                     "bytes": manifest_path.stat().st_size}
    encoded = (json.dumps(roster, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    out = Path(args.out)
    _atomic_write_new_bytes(out, encoded)
    sha = hashlib.sha256(encoded).hexdigest()
    _atomic_write_new_bytes(out.with_name(out.name + ".sha256"), f"{sha}  {out.name}\n".encode())
    print(json.dumps({"roster": str(out.resolve()), "roster_sha256": sha,
                      "identity_sha256": roster["identity_sha256"],
                      "units": len(roster["units"]),
                      "checkpoint_manifest_sha256": roster["checkpoint_manifest"]["sha256"]},
                     sort_keys=True))
    return 0
