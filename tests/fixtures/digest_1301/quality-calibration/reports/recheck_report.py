(args.out / "recheck.json").write_text(json.dumps({
    "inputs": {name: {"path": str(paths[name]), "sha256": hashlib.sha256(data).hexdigest()}
               for name, data in raw.items()},
    "normalization": "canonical_refusals only, on a separate monolith comparison copy",
    "historical_refusals_multiset_equal": True,
    "no_new_encodes": True,
    "comparison_returncode": result.returncode,
    "comparison_command": command,
}, indent=2) + "\n")
