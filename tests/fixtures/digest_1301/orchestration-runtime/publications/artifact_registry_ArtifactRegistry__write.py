def _write(self, records: list[ArtifactRecord]) -> None:
    payload = {"records": [record.to_dict() for record in records]}
    tmp = self.path.with_name(
        f".{self.path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
    finally:
        if tmp.exists():
            tmp.unlink()
