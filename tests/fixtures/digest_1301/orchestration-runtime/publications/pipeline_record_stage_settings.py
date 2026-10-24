def _record_stage_settings(
    manifest_path: Path,
    stage: str,
    projection: Mapping[str, str],
) -> None:
    if manifest_path.exists():
        payload = _load_stage_manifest(manifest_path)
    else:
        payload = {"schema": STAGE_MANIFEST_SCHEMA, "stages": {}}
    stages = dict(payload.get("stages") or {})
    stages[stage] = dict(projection)
    payload["stages"] = stages
    payload["schema"] = STAGE_MANIFEST_SCHEMA
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
