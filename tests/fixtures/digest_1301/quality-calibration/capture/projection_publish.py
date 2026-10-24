    raw = (json.dumps(carried, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    from prismaquant.cost_stage_checkpoint import publish_new_bytes

    destination = path.parent.resolve(strict=False) / path.name
    if not publish_new_bytes(destination, raw, nofollow=True):
        raise RuntimeError(f"{path} exists; the projection publishes once")
    return {
        "phase": "projection",
        "projection": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "plan": {"path": str(Path(args.plan).resolve()), "sha256": plan_sha256},
        "stacks": sorted(carried["stacks"]),
        "units": sum(len(units) for units in carried["stacks"].values()),
        "request": carried["request"], "tool": carried["tool"],
        "seconds": {"total": time.monotonic() - started},
    }
