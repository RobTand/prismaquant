# Original source excerpts, independently captured from 6d8288adc94b3c4b624c143ed372fb6513f32784
# Source: tools/build_stagea_seed_package.py; function bodies are verbatim.

def manifest_wire(manifest) -> bytes:
    """The manifest's sealed bytes: sorted compact JSON, gzip without a timestamp."""
    return gzip.compress((json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                          + "\n").encode(), mtime=0)


def _pin(path, sha256, what) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SeedPackageRefused(f"the {what} {path} does not have the pinned digest {sha256}")
    return {"path": str(path), "sha256": sha256}


def build(*, original_manifest, original_manifest_sha256, plan, checkpoint, capsule,
          through, implementation_from, implementation_to, compare, output,
          validate=None) -> dict:
    """Write the package into ``output``, a directory that must not exist.

    ``plan`` is the pinned ``{path, sha256}`` of the plan the source manifest
    names; its ``inputs`` say which head entries are the head walk's.
    """
    wire = Path(original_manifest).read_bytes()
    if hashlib.sha256(wire).hexdigest() != original_manifest_sha256:
        raise SeedPackageRefused("the source manifest does not have the pinned digest")
    original = json.loads(gzip.decompress(wire) if wire[:2] == b"\x1f\x8b" else wire)
    try:
        original, dropped = drop_source_head_walk_reads(original, plan)
    except ValueError as error:
        raise SeedPackageRefused(str(error)) from error
    declared = (None if implementation_from is None and implementation_to is None
                else {"from": implementation_from, "to": implementation_to})
    spec = normalize_seed_spec({
        "schema": SEED_SPEC_SCHEMA, "checkpoint": checkpoint, "capsule": capsule,
        "through": through, "implementation_compatibility": declared,
        "compare": compare})
    record = load_pinned_checkpoint(spec["checkpoint"], "checkpoint")
    reference = (None if spec["compare"] is None
                 else load_pinned_checkpoint(spec["compare"], "compare checkpoint"))
    spec_bytes = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    root = Path(output)
    spec_entry = _entry(root / SPEC_NAME, len(spec_bytes),
                        hashlib.sha256(spec_bytes).hexdigest())
    manifest = seed_manifest(original, spec=spec, spec_entry=spec_entry,
                             checkpoint=record, compare=reference)
    if validate is not None:
        validate(manifest)
    manifest_bytes = manifest_wire(manifest)
    phases = manifest["read_plan"]["phases"]
    package = {
        "schema": PACKAGE_SCHEMA,
        "seed_spec": {"path": spec_entry["path"], "sha256": spec_entry["sha256"]},
        "data_manifest": {"path": str(root / MANIFEST_NAME),
                          "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "source_manifest": {"path": str(original_manifest),
                            "sha256": original_manifest_sha256},
        "phases": [{"name": phase["name"], "bytes": phase["bytes"],
                    "entries": len(phase["entry_indices"])} for phase in phases],
        "read_bytes": manifest["read_plan"]["read_bytes"],
        "head_walk_reads_dropped": dropped,
        "peak_consecutive_phase_bytes": peak_consecutive_bytes(phases),
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / SPEC_NAME).write_bytes(spec_bytes)
    (root / MANIFEST_NAME).write_bytes(manifest_bytes)
    (root / PACKAGE_NAME).write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package
