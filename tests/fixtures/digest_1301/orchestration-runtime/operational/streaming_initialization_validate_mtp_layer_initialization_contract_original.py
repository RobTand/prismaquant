def validate_mtp_layer_initialization_contract(value):
    """Read a completed checkpoint load of one MTP layer.

    The MTP layer is outside the streamed text forward (see
    ``_StreamingInitializationAudit``), so a capture over it cannot carry that
    forward's witness. This contract says what that capture ran instead:

    * ``state``: every tensor of the loaded layer, with a SHA-256 of its bytes,
      so a load that packed or cast anything differently is a different
      contract;
    * ``source_map_sha256``: the checkpoint keys the load read and the shard
      of each, whose bytes the capture's source roster authenticates;
    * ``input``: the hash-bound manifest of the hidden states the layer ran
      on, which carries the witness of the run that produced them.
    """
    keys = {"schema", "scope", "status", "transformers_version", "model_class",
            "dtype", "layer_prefix", "experts_implementation", "checkpoint_tensors",
            "state", "state_sha256", "source_map_sha256", "input"}
    hex64 = re.compile(r"[0-9a-f]{64}")
    if (not isinstance(value, dict) or set(value) != keys or
            value.get("schema") != _MTP_LAYER_INITIALIZATION_SCHEMA or
            value.get("scope") != "mtp_layer_checkpoint_load" or
            value.get("status") != "completed" or
            any(not isinstance(value.get(k), str) or not value[k]
                for k in ("transformers_version", "model_class", "dtype", "layer_prefix",
                          "experts_implementation")) or
            type(value.get("checkpoint_tensors")) is not int or value["checkpoint_tensors"] < 1 or
            not isinstance(value.get("source_map_sha256"), str) or
            not hex64.fullmatch(value["source_map_sha256"])):
        raise ValueError("Missing or incomplete MTP layer initialization contract")
    state = value["state"]
    if not isinstance(state, dict) or not state:
        raise ValueError("MTP layer initialization contract names no loaded state")
    for name, record in state.items():
        if (not isinstance(name, str) or not name or not isinstance(record, dict) or
                set(record) != {"shape", "dtype", "sha256"} or
                not isinstance(record["shape"], list) or
                any(type(n) is not int or n < 0 for n in record["shape"]) or
                not isinstance(record["dtype"], str) or not record["dtype"] or
                not isinstance(record["sha256"], str) or not hex64.fullmatch(record["sha256"])):
            raise ValueError("Invalid MTP layer tensor witness")
    if value["state_sha256"] != _initialization_digest(state):
        raise ValueError("MTP layer state digest differs from its tensor witnesses")
    source = value["input"]
    if (not isinstance(source, dict) or set(source) != {"schema", "path", "sha256"} or
            any(not isinstance(source.get(k), str) or not source[k] for k in source) or
            not hex64.fullmatch(source["sha256"])):
        raise ValueError("MTP layer initialization contract has no hash-bound input")
    return json.loads(json.dumps(value))
