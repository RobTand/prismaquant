import json

class StreamedBoundaryArtifacts:

    def _publish_status(self):
        if self.directory is None or self._readonly:
            return
        from .cost_stage_checkpoint import atomic_write_bytes
        data = {"schema": self.config["schema"], "session": self.session,
                "policy": self.identity, "status": self._status,
                "working_artifacts_reusable": False, "telemetry": self.telemetry}
        if self._owner_label is not None:
            data["owner"] = {"label": self._owner_label, **self._owner_fields}
        retained = self._retained_debt()
        if retained is not None:
            data["retained"] = retained
        path = self.status_path()
        path.parent.mkdir(exist_ok=True)
        atomic_write_bytes(path,
            (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())

