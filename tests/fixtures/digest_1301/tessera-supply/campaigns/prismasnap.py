def _print(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
