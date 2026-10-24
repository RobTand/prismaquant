def _record_unverified_reuse(
    artifact_path: Path,
    manifest_path: Path,
    stage: str,
    requested: Mapping[str, str],
    reason: str,
) -> str | None:
    """Record that ``stage`` reuses an artifact whose identity stays unknown.

    The requested projection is filed under ``observed_current_request`` -- it
    is this run's request, not the artifact's settings and not production
    proof. The stage's settings record is never written here, so an artifact
    that was reused unverified can never become indistinguishable from one
    whose projection was actually compared. A verified record already stored
    for the stage is left untouched. Returns an error string when the
    admission could not be persisted.

    ``tessera-plan`` never reaches this function by construction:
    ``check_stage_settings`` refuses an unrecorded plan outright, because a
    translated plan needs an allocation-content binding and an admission is
    not one.
    """
    try:
        if manifest_path.exists():
            payload = _load_stage_manifest(manifest_path)
        else:
            payload = {"schema": STAGE_MANIFEST_SCHEMA, "stages": {}}
        stages = dict(payload.get("stages") or {})
        existing = stages.get(stage)
        if existing is not None:
            # Either the admission is already on file (keep its first
            # observation) or a verified projection owns the stage. Never
            # overwrite either with a fresh request.
            return None
        stages[stage] = {
            UNVERIFIED_SETTINGS_KEY: {
                "settings_identity": "unknown",
                "attests_this_artifact": False,
                "artifact": str(artifact_path),
                "reason": reason,
                "observed_current_request": dict(requested),
                "first_observed_unix": int(time.time()),
            }
        }
        payload["stages"] = stages
        payload["schema"] = STAGE_MANIFEST_SCHEMA
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    except OSError as exc:
        return f"{manifest_path}: {exc}"
    return None
