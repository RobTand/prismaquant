def result_json(record):
    return json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
