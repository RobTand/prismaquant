def _write_overlay(path: Path, names: Sequence[str]) -> None:
    payload = {name: "BF16" for name in sorted(names)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
