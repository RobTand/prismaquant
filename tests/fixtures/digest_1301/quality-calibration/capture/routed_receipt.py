    raw = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    if not publish_new_bytes(root / "receipt.json", raw):
        raise ValueError("routed capture receipt already exists")
    print(json.dumps({"status": "complete", "captures": len(entries),
                      "receipt": str(root / "receipt.json")}), flush=True)
    return 0
