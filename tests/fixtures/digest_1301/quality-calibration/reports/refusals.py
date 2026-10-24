counts = {name: Counter(json.dumps(item, sort_keys=True) for item in payload["non_interpolable"])
          for name, payload in payloads.items()}
if counts["monolith"] != counts["historical_merge"]:
    raise RuntimeError("historical refusal records differ in content or multiplicity")
