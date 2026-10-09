"""Deterministic bounded perturbed-window checkpoints for one band.

One replay extent splits into contiguous windows. Each admitted call runs
exactly one declared window and commits one atomic checkpoint with exact
per-stream hidden state, full pass state and ownership receipts. The
perturbed band and the replay extent stay fixed across all windows. Only
the window coordinates and the bound source window advance. A resume
validates actual bytes and batch comparability. Generation labels follow the
shared development policy. Clean state never substitutes for perturbed state.

The checkpoint schema is ``pact.perturbed_stream_checkpoint.v1``. Working
boundary generations stay disposable: resume bytes live in the batch state
file, while exact hidden references give provenance back to the generation
that wrote them. The caller republishes resumed hidden tensors into its own
fresh generation. No second cache exists. No source read happens here.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path


CHECKPOINT_SCHEMA = "pact.perturbed_stream_checkpoint.v1"
COORDINATE_FIELDS = ("band_start", "band_stop", "replay_start", "replay_stop", "next_layer",
                     "window_start", "window_stop")
FROZEN_ROSTER_FIELDS = ("stream_id", "class", "kind", "weight_source", "amplitude", "null_replay")
PER_STREAM_FIELDS = ("sample_ids", "exact_hidden_references", "owned_boundary_generation",
                     "batch_state_path", "full_forward_pass_state")
COHORT_FIELDS = ("sample_range", "raw_tokens_per_sequence", "prefix_ids", "input_contract",
                 "local_prefix_rows", "global_original_tokens", "scored_positions_per_sequence",
                 "token_sha256", "sequences")
TELEMETRY_FIELDS = ("injections", "empty_lease_telemetry")
SOURCE_FIELDS = ("receipt_sha256", "source_window", "start_generation", "stop_generation")
# The teacher receipt fields that carry actual teacher data: cohort, layers,
# samples, token state location and the teacher arrays. Session, boundary
# references, timing and read telemetry are historical identity only.
TEACHER_COMPARABLE_FIELDS = ("schema", "cohort", "layer_start", "layer_stop", "sample_ids", "dry_run",
                             "batch_state_path", "teacher_arrays")


def _fail(reason):
    raise ValueError("Perturbed window checkpoint rejected: " + reason)


def _logical_tensor_bytes(tensor):
    """Return the exact logical bytes of one tensor without unrelated storage.

    Exact tensors share their storage read-only with no copy. Only a
    nonzero-offset or oversized backing takes one logical-size compact copy.
    Dtype and shape stay with the caller and enter the digest beside the
    bytes, so BF16 tensors keep their exact values.
    """
    import torch
    if not isinstance(tensor, torch.Tensor):
        _fail("hidden state must be a tensor")
    cpu = tensor.detach().to(device="cpu")
    cpu = cpu.contiguous() if not cpu.is_contiguous() else cpu
    nbytes = cpu.numel() * cpu.element_size()
    storage = cpu.untyped_storage()
    if cpu.storage_offset() == 0 and storage.nbytes() == nbytes:
        return bytes(storage), cpu
    compact = cpu.clone(memory_format=torch.contiguous_format)
    raw = bytes(compact.untyped_storage())
    if len(raw) != nbytes:
        _fail("hidden state storage is not compact")
    return raw, compact


_AMPLITUDES = (1, 2, 4)


def _require_int(value, name):
    if type(value) is not int:
        _fail(name + " is not an integer")
    return value


def _require_hex(value, name):
    if type(value) is not str or len(value) != 64:
        _fail(name + " is not a 64 character digest")
    try:
        bytes.fromhex(value)
    except ValueError:
        _fail(name + " is not hexadecimal")
    return value


def check_roster_entry(entry):
    """Validate one frozen stream roster entry and return a copy."""
    if not isinstance(entry, dict) or set(entry) != set(FROZEN_ROSTER_FIELDS):
        _fail("roster entry must carry exactly the frozen fields")
    if type(entry["stream_id"]) is not str or not entry["stream_id"]:
        _fail("roster stream_id must be a nonempty string")
    for key in ("class", "kind", "weight_source"):
        if type(entry[key]) is not str or not entry[key]:
            _fail("roster " + key + " must be a nonempty string")
    if entry["amplitude"] not in _AMPLITUDES:
        _fail("roster amplitude must be one of 1, 2, 4")
    if type(entry["null_replay"]) is not bool:
        _fail("roster null_replay must be a boolean")
    return dict(entry)


def check_source(source):
    """Validate one source binding and return a copy."""
    if not isinstance(source, dict) or set(source) != set(SOURCE_FIELDS):
        _fail("source must bind receipt, window and both generations")
    _require_hex(source["receipt_sha256"], "source receipt_sha256")
    window = source["source_window"]
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or not all(type(v) is int for v in window) or not window[0] < window[1]):
        _fail("source window must be an increasing integer pair")
    for key in ("start_generation", "stop_generation"):
        generation = source[key]
        if (not isinstance(generation, dict) or set(generation) != {"generation", "run_identity_sha256"}
                or not generation["generation"] or not generation["run_identity_sha256"]):
            _fail("source " + key + " must name its ownership generation")
    return {"receipt_sha256": source["receipt_sha256"], "source_window": [window[0], window[1]],
            "start_generation": dict(source["start_generation"]),
            "stop_generation": dict(source["stop_generation"])}


def check_cohort(cohort):
    """Validate one cohort mapping and return a JSON-stable copy."""
    if not isinstance(cohort, dict):
        _fail("cohort must be a mapping")
    for key in COHORT_FIELDS:
        if key not in cohort:
            _fail("cohort omits " + key)
    if (not isinstance(cohort["sample_range"], (list, tuple)) or len(cohort["sample_range"]) != 2):
        _fail("cohort sample_range must be an integer pair")
    _require_hex(cohort["token_sha256"], "cohort token_sha256")
    normalized = dict(cohort)
    normalized["sample_range"] = [cohort["sample_range"][0], cohort["sample_range"][1]]
    if isinstance(cohort["prefix_ids"], tuple):
        normalized["prefix_ids"] = list(cohort["prefix_ids"])
    return normalized


def check_coordinates(band_start, band_stop, replay_start, replay_stop, next_layer, window_start,
                      window_stop):
    """Validate band, replay and window coordinates and return them as integers.

    The perturbed band stays fixed across all windows. The replay extent
    covers the band and its tail. Each admitted window executes exactly one
    slice of the replay extent.
    """
    for name, value in (("band_start", band_start), ("band_stop", band_stop),
                        ("replay_start", replay_start), ("replay_stop", replay_stop),
                        ("next_layer", next_layer), ("window_start", window_start),
                        ("window_stop", window_stop)):
        _require_int(value, name)
    if not 0 <= band_start < band_stop <= 45:
        _fail("band range must sit inside the actual source graph")
    if not 0 <= replay_start <= band_start < band_stop <= replay_stop <= 45:
        _fail("replay extent must cover the perturbed band and its tail")
    if not replay_start <= window_start < window_stop <= replay_stop:
        _fail("window must sit inside its replay extent")
    if next_layer != window_stop:
        _fail("next_layer must equal the completed window stop")
    return (band_start, band_stop, replay_start, replay_stop, next_layer, window_start,
            window_stop)


def plan_windows(replay_start, replay_stop, *, window_layers):
    """Split one replay extent into deterministic contiguous windows.

    The split is a pure function of its inputs. No scheduling, no IO and no
    fallback exist. The last window may be shorter than the stride.
    """
    _require_int(replay_start, "replay_start")
    _require_int(replay_stop, "replay_stop")
    _require_int(window_layers, "window_layers")
    if not 0 <= replay_start < replay_stop <= 45:
        _fail("replay extent must sit inside the actual source graph")
    if window_layers < 1:
        _fail("window stride must be at least one layer")
    windows = []
    cursor = replay_start
    while cursor < replay_stop:
        stop = min(cursor + window_layers, replay_stop)
        windows.append({"window_start": cursor, "window_stop": stop})
        cursor = stop
    return windows


TEACHER_CONTENT_SCHEMA = "pact.prefixed_teacher_validation.v1"


def bind_teacher_content(receipt_bytes, content_bytes, *, content_sha256, cohort):
    """Validate one teacher content proof against its receipt and the actual cohort.

    The proof names one SHA-256 per teacher array and the batch-state digest.
    Its own bytes, sample population, cohort, tokens and array geometry must
    match. These checks refuse in both modes. The frontier receipt digest in
    the proof is historical identity and goes through seal_check. Returns the
    content digests that the reader and the strict teacher digest consume.
    """
    raw = bytes(content_bytes)
    if hashlib.sha256(raw).hexdigest() != _require_hex(content_sha256, "teacher content sha256"):
        _fail("teacher content proof bytes differ from their declared digest")
    try:
        content = json.loads(raw.decode())
        receipt = json.loads(bytes(receipt_bytes).decode())
    except ValueError:
        _fail("teacher content proof or receipt is not valid JSON")
    if not isinstance(content, dict) or content.get("schema") != TEACHER_CONTENT_SCHEMA:
        _fail("teacher content proof schema differs")
    samples = receipt.get("sample_ids")
    if content.get("sample_ids") != samples or not isinstance(samples, list) or not samples:
        _fail("teacher content proof names another sample population")
    for key, value in content.get("cohort", {}).items():
        if receipt["cohort"].get(key) != value or (key in cohort and cohort[key] != value):
            _fail("teacher content proof binds another cohort: " + key)
    if content.get("original_token_sha256") != cohort.get("token_sha256"):
        _fail("teacher content proof binds other tokens")
    if content.get("all_finite") is not True or content.get("arrays_verified") != len(samples):
        _fail("teacher content proof does not verify every finite array")
    declared = {row["sample_id"]: row for row in receipt.get("teacher_arrays", [])}
    rows = content.get("teacher_arrays")
    if not isinstance(rows, list) or len(rows) != len(samples):
        _fail("teacher content proof omits teacher arrays")
    arrays, recorded_paths, current_paths = {}, {}, {}
    for row in rows:
        sample = row.get("sample_id")
        own = declared.get(sample)
        if own is None or sample in arrays:
            _fail("teacher content proof names another or a repeated sample")
        if (row.get("file_bytes") != own["bytes"] or row.get("shape") != own["shape"]
                or row.get("dtype") != own["dtype"]):
            _fail("teacher content proof array %d geometry differs from its receipt row" % sample)
        arrays[sample] = _require_hex(row.get("sha256"), "teacher array sha256")
        recorded_paths[sample], current_paths[sample] = row.get("path"), own["path"]
    if set(arrays) != set(declared) or set(arrays) != set(samples):
        _fail("teacher content proof does not cover every teacher array")
    from g3_pq_policy.dev_mode import seal_check
    # A recorded path is historical identity. The bytes at the current path
    # still refuse in both modes when they differ from their content digest.
    seal_check("teacher content proof array paths", recorded_paths, current_paths,
        where="teacher content binding", refusal=ValueError("Recorded teacher array paths differ"))
    seal_check("teacher content proof receipt identity", content.get("frontier_sha256"),
        hashlib.sha256(bytes(receipt_bytes)).hexdigest(), where="teacher content binding",
        refusal=ValueError("Recorded teacher content proof names another receipt"))
    return {"content_sha256": content_sha256, "arrays": arrays,
            "batch_state_sha256": _require_hex(content.get("batch_state_sha256"), "batch_state_sha256")}


def teacher_bindings(receipt_bytes, content=None):
    """Split one teacher receipt into its strict data digest and its identity digest.

    ``teacher_digest`` covers TEACHER_COMPARABLE_FIELDS. With a bound content
    proof it covers every teacher array SHA-256 and the batch-state digest
    instead of the file paths. It always refuses on change. ``teacher_receipt_sha256`` covers the complete
    receipt, session included; a resume compares it through seal_check.
    """
    raw = bytes(receipt_bytes)
    try:
        document = json.loads(raw.decode())
    except ValueError:
        _fail("teacher receipt is not valid JSON")
    if not isinstance(document, dict) or any(key not in document for key in TEACHER_COMPARABLE_FIELDS):
        _fail("teacher receipt omits its comparable teacher data")
    comparable = {key: document[key] for key in TEACHER_COMPARABLE_FIELDS}
    if content is not None:
        # Content digests cover the bytes, so file locations are identity only (D32).
        # The complete receipt digest, paths included, goes through seal_check.
        del comparable["batch_state_path"]
        comparable["teacher_arrays"] = [{key: value for key, value in row.items() if key != "path"}
                                        for row in document["teacher_arrays"]]
        comparable["teacher_content"] = {"arrays": {str(sample): digest for sample, digest
                                                    in sorted(content["arrays"].items())},
                                         "batch_state_sha256": content["batch_state_sha256"]}
    encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"teacher_digest": hashlib.sha256(encoded).hexdigest(),
            "teacher_receipt_sha256": hashlib.sha256(raw).hexdigest()}


def checkpoint_path(checkpoint_dir, window_start, window_stop):
    """Name one window checkpoint deterministically."""
    _require_int(window_start, "window_start")
    _require_int(window_stop, "window_stop")
    return Path(checkpoint_dir) / ("perturbed-window-%02d-%02d.json" % (window_start, window_stop))




def _hidden_digest(tensor):
    raw, compact = _logical_tensor_bytes(tensor)
    digest = hashlib.sha256()
    digest.update(str(compact.dtype).encode())
    digest.update((",".join(str(dim) for dim in compact.shape)).encode())
    digest.update(raw)
    return digest.hexdigest(), compact


def _canonical_state_bytes(state):
    """Encode pass state by value so the digest survives a save/load round trip.

    Only the closed grammar of profile pass state is accepted: tensors,
    mappings, tuples, lists, None and str/bool/int/float/complex leaves.
    Anything opaque fails closed.
    """
    import torch
    from collections.abc import Mapping
    digest = hashlib.sha256()
    if isinstance(state, torch.Tensor):
        raw, compact = _logical_tensor_bytes(state)
        digest.update(b"T" + str(compact.dtype).encode() + b"|"
                      + (",".join(str(dim) for dim in compact.shape)).encode() + b"|" + raw)
    elif isinstance(state, Mapping):
        digest.update(b"D" + str(len(state)).encode() + b"|")
        for key in sorted(state, key=lambda item: (type(item).__name__, repr(item))):
            digest.update(_canonical_state_bytes(key) + _canonical_state_bytes(state[key]))
    elif isinstance(state, (tuple, list)):
        digest.update((b"P" if isinstance(state, tuple) else b"L") + str(len(state)).encode()
                      + b"|")
        for item in state:
            digest.update(_canonical_state_bytes(item))
    elif state is None or type(state) in (str, bool, int, float, complex):
        digest.update(b"V" + type(state).__name__.encode() + b"|" + repr(state).encode())
    else:
        _fail("pass state holds an opaque value of type " + type(state).__name__)
    return digest.digest()


def _state_digest(state):
    return hashlib.sha256(_canonical_state_bytes(state)).hexdigest()


def validate_forward_state(forward, *, width=None, input_ids=None, expected=None):
    """Bind the complete restored forward state to actual current inputs."""
    import torch
    fields = {"input_ids", "position_ids", "position_embeddings", "attention_mask"}
    if not isinstance(forward, dict) or set(forward) != fields:
        _fail("forward state must carry the complete batch contract")
    ids = forward["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1:
        _fail("forward input IDs have wrong dtype or shape")
    if width is not None and ids.shape[1] != width:
        _fail("forward input IDs have wrong token width")
    positions = forward["position_ids"]
    if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64 or positions.ndim < 1 or positions.shape[-1] != ids.shape[1]:
        _fail("forward positions have wrong dtype or token width")
    _state_digest(forward)
    if input_ids is not None and (ids.dtype != input_ids.dtype or ids.shape != input_ids.shape or not torch.equal(ids, input_ids)):
        _fail("restored forward tokens differ from actual current inputs")
    if expected is not None and any(_state_digest(forward[key]) != _state_digest(expected[key]) for key in fields):
        _fail("restored forward state differs from the actual batch contract")


def validate_forward_cohort(payload, samples, cohort):
    import torch
    prefix = len(cohort["prefix_ids"])
    tokens = torch.cat([payload[sample]["forward"]["input_ids"][:, prefix:] for sample in samples])
    digest = hashlib.sha256(tokens.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if digest != cohort["token_sha256"]:
        _fail("forward token bytes differ from the actual cohort")


def _atomic_write_bytes(path, data, *, guard=None):
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial." + uuid.uuid4().hex)
    created = False
    try:
        if guard is not None:
            guard("perturbed checkpoint write")
        with open(temporary, "xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        created = True
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def check_replay_telemetry(telemetry, stream_id):
    """Validate cumulative per-stream lease telemetry and return a JSON copy.

    None means that the producer records no telemetry. Otherwise the value
    carries exactly the lease lists of every completed window so far.
    """
    if telemetry is None:
        return None
    if not isinstance(telemetry, dict) or set(telemetry) != set(TELEMETRY_FIELDS):
        _fail("stream " + stream_id + " telemetry must carry exactly the lease lists")
    for key in TELEMETRY_FIELDS:
        if not isinstance(telemetry[key], list) or not all(isinstance(row, dict) for row in telemetry[key]):
            _fail("stream " + stream_id + " telemetry " + key + " must be a list of records")
    try:
        return json.loads(json.dumps(telemetry, allow_nan=False))
    except (TypeError, ValueError):
        _fail("stream " + stream_id + " telemetry is not finite JSON")


def _entry_record(reference):
    if isinstance(reference, dict):
        for key in ("name", "path", "sha256", "tensor_bytes", "file_bytes",
                    "shape", "dtype", "metadata"):
            if key not in reference:
                _fail("hidden reference record omits " + key)
        return dict(reference)
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record
    return exact_entry_record(reference)


def commit_stream_window(path, *, band_start, band_stop, replay_start, replay_stop, next_layer,
                         window_start, window_stop, window_layers, cohort, stream_roster,
                         source, teacher_digest, input_digest, streams, perturbed=True,
                         guard=None, teacher_receipt_sha256=None):
    """Validate all streams, then publish one immutable checkpoint atomically."""
    import torch
    import uuid
    path = Path(path)
    if path.exists():
        _fail("a committed window cannot be replaced")
    if perturbed is not True:
        _fail("only perturbed state may commit a perturbed checkpoint")
    check_coordinates(band_start, band_stop, replay_start, replay_stop, next_layer, window_start, window_stop)
    _require_int(window_layers, "window_layers")
    if window_layers < 1:
        _fail("window stride must be at least one layer")
    cohort = check_cohort(cohort)
    if not isinstance(stream_roster, (list, tuple)) or not stream_roster:
        _fail("stream roster must be a nonempty ordered list")
    roster = [check_roster_entry(entry) for entry in stream_roster]
    identifiers = [entry["stream_id"] for entry in roster]
    if len(set(identifiers)) != len(identifiers):
        _fail("stream roster repeats a stream_id")
    source = check_source(source)
    if source["source_window"] != [window_start, window_stop]:
        _fail("source window must equal the one executed window")
    _require_hex(teacher_digest, "teacher_digest")
    _require_hex(input_digest, "input_digest")
    if teacher_receipt_sha256 is not None:
        _require_hex(teacher_receipt_sha256, "teacher_receipt_sha256")
    if not isinstance(streams, dict) or set(streams) != set(identifiers):
        _fail("stream state must cover exactly the roster")
    generation_id = uuid.uuid4().hex
    stored, pending = {}, {}
    for entry in roster:
        stream_id = entry["stream_id"]
        state = streams[stream_id]
        if not isinstance(state, dict):
            _fail("stream " + stream_id + " state must be a mapping")
        sample_ids = state.get("sample_ids")
        if sample_ids != list(range(*cohort["sample_range"])):
            _fail("stream " + stream_id + " sample_ids must cover the ordered cohort")
        for key in ("hidden", "pass_state", "forward", "references"):
            if not isinstance(state.get(key), dict) or set(state[key]) != set(sample_ids):
                _fail("stream " + stream_id + " " + key + " must cover exactly its samples")
        generation = state.get("owned_boundary_generation")
        if not isinstance(generation, dict) or set(generation) != {"generation", "run_identity_sha256"}:
            _fail("stream " + stream_id + " must name its owned boundary generation")
        hidden_digests, pass_digests, forward_digests, payload, references = {}, {}, {}, {}, {}
        for sample in sample_ids:
            if state["pass_state"][sample] is None:
                _fail("stream " + stream_id + " sample %d misses pass state" % sample)
            forward = state["forward"][sample]
            validate_forward_state(forward, width=cohort["raw_tokens_per_sequence"] + len(cohort["prefix_ids"]))
            digest, compact = _hidden_digest(state["hidden"][sample])
            hidden_digests[str(sample)] = digest
            pass_digests[str(sample)] = _state_digest(state["pass_state"][sample])
            forward_digests[str(sample)] = _state_digest(forward)
            payload[sample] = {"hidden": compact, "pass_state": state["pass_state"][sample], "forward": forward}
            record = _entry_record(state["references"][sample])
            metadata = record["metadata"]
            coordinates = metadata["identity"]["coordinates"]
            from g3_pq_policy.dev_mode import seal_check
            seal_check("checkpoint boundary generation", generation, metadata["identity"]["session"],
                where="perturbed checkpoint stream " + stream_id,
                refusal=ValueError("Recorded checkpoint boundary generation differs"))
            if coordinates["batch"] != sample or coordinates["boundary"] != window_stop:
                _fail("stream " + stream_id + " reference names another sample or layer")
            if list(record["shape"]) != list(compact.shape) or record["dtype"] != str(compact.dtype):
                _fail("stream " + stream_id + " reference shape differs from its hidden bytes")
            references[str(sample)] = record
        validate_forward_cohort(payload, sample_ids, cohort)
        stream_name = hashlib.sha256(stream_id.encode()).hexdigest()
        batch_path = Path(str(path) + "." + generation_id + "." + stream_name + ".states.pt")
        stored[stream_id] = {"sample_ids": list(sample_ids), "exact_hidden_references": references,
            "owned_boundary_generation": dict(generation), "batch_state_path": str(batch_path),
            "full_forward_pass_state": {key: {"pass_state_sha256": value} for key, value in pass_digests.items()},
            "hidden_sha256": hidden_digests, "forward_sha256": forward_digests,
            "replay_telemetry": check_replay_telemetry(state.get("telemetry"), stream_id)}
        pending[stream_id] = payload
    written = []
    try:
        for stream_id, payload in pending.items():
            if guard is not None:
                guard("perturbed checkpoint serialization")
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            batch_bytes = buffer.getvalue()
            stored[stream_id]["batch_state_sha256"] = hashlib.sha256(batch_bytes).hexdigest()
            batch_path = Path(stored[stream_id]["batch_state_path"])
            _atomic_write_bytes(batch_path, batch_bytes, guard=guard)
            written.append(batch_path)
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "band_start": band_start, "band_stop": band_stop,
            "replay_start": replay_start, "replay_stop": replay_stop, "next_layer": next_layer,
            "window_start": window_start, "window_stop": window_stop, "window_layers": window_layers,
            "cohort": cohort, "stream_roster": roster, "source": source, "teacher_digest": teacher_digest,
            "input_digest": input_digest, "teacher_receipt_sha256": teacher_receipt_sha256,
            "perturbed": True, "streams": stored}
        raw = (json.dumps(checkpoint, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _atomic_write_bytes(path, raw, guard=guard)
        return checkpoint
    except BaseException:
        for batch_path in written:
            batch_path.unlink(missing_ok=True)
        raise


def _read_checkpoint_file(path):
    from g3_residency import read_file
    try:
        raw = read_file(Path(path))
    except OSError:
        _fail("checkpoint file is missing: " + str(path))
    try:
        checkpoint = json.loads(raw.decode())
    except ValueError:
        _fail("checkpoint file is not valid JSON")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        _fail("checkpoint schema differs")
    return checkpoint


def _check_stored_stream(checkpoint, stream_id, *, guard=None):
    import torch
    stored = checkpoint["streams"][stream_id]
    for key in PER_STREAM_FIELDS + ("batch_state_sha256", "hidden_sha256", "forward_sha256",
                                    "replay_telemetry"):
        if key not in stored:
            _fail("stream " + stream_id + " checkpoint omits " + key)
    check_replay_telemetry(stored["replay_telemetry"], stream_id)
    sample_ids = stored["sample_ids"]
    batch_path = Path(stored["batch_state_path"])
    if guard is not None:
        guard("perturbed checkpoint read")
    from g3_residency import read_file
    try:
        batch_bytes = read_file(batch_path)
    except OSError:
        _fail("stream " + stream_id + " batch state is missing")
    if hashlib.sha256(batch_bytes).hexdigest() != stored["batch_state_sha256"]:
        _fail("stream " + stream_id + " batch bytes differ")
    try:
        payload = torch.load(io.BytesIO(batch_bytes), map_location="cpu",
                             weights_only=True)
    except Exception:
        _fail("stream " + stream_id + " batch state does not load")
    if set(payload) != set(sample_ids):
        _fail("stream " + stream_id + " batch state changes its samples")
    for sample in sample_ids:
        row = payload[sample]
        if not isinstance(row, dict) or row.get("pass_state") is None:
            _fail("stream " + stream_id + " sample %d misses pass state" % sample)
        if "hidden" not in row or "forward" not in row:
            _fail("stream " + stream_id + " sample %d misses hidden or forward state" % sample)
        validate_forward_state(row["forward"], width=checkpoint["cohort"]["raw_tokens_per_sequence"] + len(checkpoint["cohort"]["prefix_ids"]))
        if _state_digest(row["forward"]) != stored["forward_sha256"].get(str(sample)):
            _fail("stream " + stream_id + " sample %d forward bytes differ" % sample)
        digest, _ = _hidden_digest(row["hidden"])
        if type(stored["hidden_sha256"]) is not dict or digest != stored["hidden_sha256"].get(
                str(sample)):
            _fail("stream " + stream_id + " sample %d hidden bytes differ" % sample)
        pass_entry = stored["full_forward_pass_state"].get(
            str(sample)) if isinstance(stored["full_forward_pass_state"], dict) else None
        if not isinstance(pass_entry, dict) or _state_digest(row["pass_state"]) != pass_entry.get(
                "pass_state_sha256"):
            _fail("stream " + stream_id + " sample %d pass bytes differ" % sample)
        record = stored["exact_hidden_references"].get(
            str(sample)) if isinstance(stored["exact_hidden_references"], dict) else None
        if not isinstance(record, dict):
            _fail("stream " + stream_id + " sample %d reference is missing" % sample)
        from prismaquant.joint_adjoint_checkpoints import reference_from_record
        try:
            reference_from_record(record)
        except Exception:
            _fail("stream " + stream_id + " sample %d reference does not rebuild" % sample)
        metadata = record.get("metadata")
        identity = metadata.get("identity") if isinstance(metadata, dict) else None
        coordinates = identity.get("coordinates") if isinstance(identity, dict) else None
        if not isinstance(coordinates, dict):
            _fail("stream " + stream_id + " sample %d reference names nothing" % sample)
        from g3_pq_policy.dev_mode import seal_check
        seal_check("checkpoint boundary generation", stored["owned_boundary_generation"], identity.get("session"),
            where="perturbed checkpoint stream " + stream_id,
            refusal=ValueError("Recorded checkpoint boundary generation differs"))
        if coordinates.get("batch") != sample or coordinates.get("boundary") != checkpoint[
                "window_stop"]:
            _fail("stream " + stream_id + " reference names another sample or layer")
    validate_forward_cohort(payload, sample_ids, checkpoint["cohort"])
    return payload


def load_window_checkpoint(path, *, expected, guard=None):
    """Validate one checkpoint against its expected run bindings.

    ``expected`` carries band_start, band_stop, replay_start, replay_stop,
    next_layer, window_start, window_stop, window_layers, cohort,
    stream_roster, source (with this window source_window), teacher_digest,
    teacher_receipt_sha256, input_digest, perturbed and streams mapping each stream_id to its
    sample_ids. Band, replay, cohort, roster and digests stay fixed across
    all windows. Only the window coordinates and the bound source window
    advance. Generation identities follow the shared development policy.
    Actual bytes, windows, tokens and batch values always refuse on mismatch.
    """
    if not isinstance(expected, dict):
        _fail("resume needs its expected run bindings")
    checkpoint = _read_checkpoint_file(path)
    check_coordinates(checkpoint.get("band_start"), checkpoint.get("band_stop"),
                      checkpoint.get("replay_start"), checkpoint.get("replay_stop"),
                      checkpoint.get("next_layer"), checkpoint.get("window_start"),
                      checkpoint.get("window_stop"))
    for key in ("band_start", "band_stop", "replay_start", "replay_stop"):
        if checkpoint.get(key) != expected.get(key):
            _fail("checkpoint binds changed global band or replay bindings")
    for key in ("next_layer", "window_start", "window_stop", "window_layers"):
        if checkpoint.get(key) != expected.get(key):
            _fail("checkpoint binds another window")
    if checkpoint.get("perturbed") is not True or expected.get("perturbed", True) is not True:
        _fail("clean state can never resume a perturbed window")
    if checkpoint.get("cohort") != check_cohort(expected.get("cohort")):
        _fail("checkpoint binds another cohort")
    roster = [check_roster_entry(entry) for entry in expected.get("stream_roster", [])]
    if checkpoint.get("stream_roster") != roster:
        _fail("checkpoint binds an altered stream roster")
    expected_source = check_source(expected.get("source"))
    actual_source = check_source(checkpoint.get("source"))
    if actual_source["source_window"] != expected_source["source_window"]:
        _fail("checkpoint binds another actual source window")
    from g3_pq_policy.dev_mode import seal_check
    seal_check("checkpoint source identity",
        {key: expected_source[key] for key in ("receipt_sha256", "start_generation", "stop_generation")},
        {key: actual_source[key] for key in ("receipt_sha256", "start_generation", "stop_generation")},
        where="perturbed checkpoint resume", refusal=ValueError("Recorded checkpoint source identities differ"))
    for key in ("teacher_digest", "input_digest"):
        if checkpoint.get(key) != _require_hex(expected.get(key), key):
            _fail("checkpoint binds another teacher or input digest")
    seal_check("checkpoint teacher receipt identity", expected.get("teacher_receipt_sha256"),
        checkpoint.get("teacher_receipt_sha256"), where="perturbed checkpoint resume",
        refusal=ValueError("Recorded checkpoint teacher receipt identities differ"))
    expected_streams = expected.get("streams")
    if not isinstance(expected_streams, dict) or set(checkpoint.get("streams", {})) != set(
            expected_streams):
        _fail("checkpoint binds another stream population")
    for stream_id, spec in expected_streams.items():
        if (not isinstance(spec, dict) or checkpoint["streams"][stream_id]["sample_ids"]
                != spec.get("sample_ids")):
            _fail("stream " + stream_id + " binds another sample population")
    for stream_id in expected_streams:
        _check_stored_stream(checkpoint, stream_id, guard=guard)
    return checkpoint


def resume_stream_window(checkpoint_or_path, stream_id, *, window_start, window_stop, guard=None):
    """Restore one stream exact hidden plus pass state from its checkpoint.

    Accepts a validated checkpoint mapping or a checkpoint path. A path still
    verifies schema, coordinates, perturbed marking and every stored byte. It
    cannot verify run bindings; call load_window_checkpoint first for that.
    The returned tensors are fresh CPU copies owned by the caller.
    """
    checkpoint = _read_checkpoint_file(checkpoint_or_path) if isinstance(
        checkpoint_or_path, (str, Path)) else checkpoint_or_path
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        _fail("checkpoint schema differs")
    if checkpoint.get("perturbed") is not True:
        _fail("clean state can never resume a perturbed window")
    if checkpoint.get("window_start") != window_start or checkpoint.get("window_stop") != window_stop:
        _fail("checkpoint binds another window")
    if not isinstance(checkpoint.get("streams"), dict) or stream_id not in checkpoint["streams"]:
        _fail("checkpoint misses stream " + str(stream_id))
    payload = _check_stored_stream(checkpoint, stream_id, guard=guard)
    stored = checkpoint["streams"][stream_id]
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    hidden, states, forward, references = {}, {}, {}, {}
    for sample in stored["sample_ids"]:
        row = payload[sample]
        tensor = row["hidden"].detach().to(device="cpu", copy=True)
        hidden[sample] = tensor
        states[sample] = row["pass_state"]
        forward[sample] = row["forward"]
        references[sample] = reference_from_record(stored["exact_hidden_references"][str(sample)])
        del tensor
    return {"stream_id": stream_id, "sample_ids": list(stored["sample_ids"]), "hidden": hidden,
            "pass_state": states, "forward": forward, "references": references,
            "owned_boundary_generation": dict(stored["owned_boundary_generation"]),
            "telemetry": check_replay_telemetry(stored["replay_telemetry"], stream_id)}


def next_window_dependencies(checkpoint, *, checkpoint_file):
    """Name the explicit code-owned next-window dependencies.

    The next window starts exactly at next_layer with the same stride over
    the replay extent. A complete replay names no further window. The caller
    supplies no schedule.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        _fail("checkpoint schema differs")
    check_coordinates(checkpoint.get("band_start"), checkpoint.get("band_stop"),
                      checkpoint.get("replay_start"), checkpoint.get("replay_stop"),
                      checkpoint.get("next_layer"), checkpoint.get("window_start"),
                      checkpoint.get("window_stop"))
    stride = checkpoint.get("window_layers")
    _require_int(stride, "window_layers")
    band_start, band_stop, replay_start, replay_stop, next_layer = (
        checkpoint["band_start"], checkpoint["band_stop"], checkpoint["replay_start"],
        checkpoint["replay_stop"], checkpoint["next_layer"])
    if next_layer >= replay_stop:
        return {"complete": True, "band_start": band_start, "band_stop": band_stop,
                "replay_start": replay_start, "replay_stop": replay_stop,
                "next_layer": next_layer, "window_start": None, "window_stop": None,
                "requires": None}
    window_stop = min(next_layer + stride, replay_stop)
    return {"complete": False, "band_start": band_start, "band_stop": band_stop,
            "replay_start": replay_start, "replay_stop": replay_stop,
            "next_layer": next_layer, "window_start": next_layer, "window_stop": window_stop,
            "requires": {
                "checkpoint_path": str(Path(checkpoint_file)),
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "batch_state_paths": [checkpoint["streams"][stream_id]["batch_state_path"]
                                      for stream_id in [entry["stream_id"]
                                                        for entry in checkpoint["stream_roster"]]],
                "source": dict(checkpoint["source"]),
                "owned_boundary_generations": {
                    stream_id: dict(checkpoint["streams"][stream_id][
                        "owned_boundary_generation"])
                    for stream_id in [entry["stream_id"] for entry in checkpoint["stream_roster"]]}}}
