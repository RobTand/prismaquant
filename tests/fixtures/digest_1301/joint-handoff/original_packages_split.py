# Original source excerpts, independently captured from 6d8288adc94b3c4b624c143ed372fb6513f32784
# Source: tools/build_stagea_split_package.py; function bodies are verbatim.

def load_round(output_root, chain_state_sha256, checkpoint_sha256) -> dict:
    """The run's chain state and sealed checkpoints, checked the way a relaunch checks them.

    Returns ``{space, state, state_entry, sealed, boundary}``: ``sealed``
    maps each sealed checkpoint's boundary to its record, and ``boundary``
    is the lowest, which must be the checkpoint ``checkpoint_sha256`` pins.
    """
    from prismaquant.joint_adjoint_checkpoints import adjoint_space
    from prismaquant.stage_a_chain_resume import (
        ChainResumeRefused, _sealed_checkpoints, chain_state_path, load_chain_state)

    space = adjoint_space(output_root)
    try:
        state = load_chain_state(space, chain_state_sha256)
        session = state["boundary_storage"]["session"]
        marker = {"generation": session["generation"], "kind": "adjoint_checkpoint",
                  "run_identity_sha256": session["run_identity_sha256"]}
        sealed, _partial = _sealed_checkpoints(
            space, marker, [int(mark) for mark in state["stride"]["boundaries"]])
    except ChainResumeRefused as exc:
        raise SplitPackageRefused(str(exc)) from exc
    if not sealed:
        raise SplitPackageRefused("the run has no sealed checkpoint to split from")
    boundary = min(sealed)
    path = Path(sealed[boundary][1]) / "checkpoint.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_sha256:
        raise SplitPackageRefused(
            f"the lowest sealed checkpoint {path} does not have the pinned digest "
            f"{checkpoint_sha256}")
    raw = chain_state_path(space).read_bytes()
    return {"space": space, "state": state, "boundary": boundary,
            "sealed": {mark: record for mark, (record, _directory) in sealed.items()},
            "state_entry": _entry(chain_state_path(space), len(raw), chain_state_sha256)}


def _describe(path, wire, manifest, stat, prefix="chain-") -> dict:
    phases = manifest["read_plan"]["phases"]
    return {"data_manifest": {"path": str(path), "sha256": hashlib.sha256(wire).hexdigest()},
            "phases": [{"name": phase["name"], "bytes": phase["bytes"],
                        "entries": len(phase["entry_indices"])} for phase in phases],
            "entry_count": manifest["entry_count"],
            "read_bytes": manifest["read_plan"]["read_bytes"],
            "source_bytes": source_bytes(manifest, prefix),
            "peak_consecutive_phase_bytes": peak_consecutive_bytes(phases),
            "stat_check": stat}


def build(*, original_manifest, original_manifest_sha256, plan, output_root,
          chain_state_sha256, checkpoint_sha256, through, ranges, output,
          validate=None) -> dict:
    """Write the round's package into ``output``, a directory that must not exist."""
    original, dropped = _load_original(original_manifest, original_manifest_sha256, plan)
    loaded = load_round(output_root, chain_state_sha256, checkpoint_sha256)
    prep, quanta = round_manifests(original, loaded, through=through, ranges=ranges)
    original_paths = {(entry["path"], entry["offset"]) for entry in original["entries"]}
    root = Path(output)
    written = [(root / PREP_MANIFEST_NAME, prep)]
    written += [(root / quantum_manifest_name(*pair), manifest)
                for pair, manifest in quanta.items()]
    described = []
    for path, manifest in written:
        stat = stat_check(manifest, original_paths)
        if validate is not None:
            validate(manifest)
        described.append((path, manifest_wire(manifest), manifest, stat))
    record = loaded["sealed"][loaded["boundary"]]
    package = {
        "schema": PACKAGE_SCHEMA,
        "source_manifest": {"path": str(original_manifest),
                            "sha256": original_manifest_sha256},
        "chain_state": {"path": loaded["state_entry"]["path"],
                        "sha256": chain_state_sha256,
                        "chain_state_sha256": loaded["state"]["chain_state_sha256"]},
        "checkpoint": {"boundary": loaded["boundary"], "sha256": checkpoint_sha256,
                       "cotangent_sha256": record["cotangent_sha256"]},
        "from": loaded["boundary"], "through": int(through),
        "boundaries": prep["annotations"]["chain_split"]["boundaries"],
        "ranges": [list(pair) for pair in sorted(tuple(pair) for pair in ranges)],
        "n_batches": int(loaded["state"]["n_batches"]),
        "head_walk_reads_dropped": dropped,
        "prep": _describe(*described[0]),
        "quanta": [{"samples": list(pair), **_describe(*item)}
                   for pair, item in zip(quanta, described[1:])],
    }
    package["round_source_bytes"] = sum(item["source_bytes"] for item in package["quanta"])
    root.mkdir(parents=True, exist_ok=False)
    for path, data, _manifest, _stat in described:
        path.write_bytes(data)
    (root / PACKAGE_NAME).write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package
