"""Pure streamed initialization metadata policy; runtime audit owns production."""

from __future__ import annotations

import re
import json
from collections.abc import Iterable
from typing import Any, TypedDict

from .digests import DIRECT_ASCII_STRICT


class StreamingInitializationContract(TypedDict):
    """The existing complete census contract, required by the metadata merge."""

    schema: str
    scope: str
    status: str
    transformers_version: str
    model_class: str
    dtype: str
    layers_prefix: str
    num_layers: int
    persistent_tensors: int
    derived_buffers: int
    state_sha256: str
    source_map_sha256: str


_STREAMING_INITIALIZATION_SCHEMA = "prismaquant.streaming_initialization.v1"


_initialization_digest = DIRECT_ASCII_STRICT.sha256


def _prefix_layer_index(name: str, prefix: str) -> int | None:
    """The decoder-layer index of a ``<prefix><N>.`` state name, else None."""
    match = re.fullmatch(re.escape(prefix) + r"(\d+)\..+", name)
    return None if match is None else int(match[1])


def validate_streaming_initialization_contract(value):
    """Read completed source-forward coverage; never accept a pending skeleton."""
    keys = {"schema", "scope", "status", "transformers_version", "model_class",
            "dtype", "layers_prefix", "num_layers", "persistent_tensors",
            "derived_buffers", "state_sha256", "source_map_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or
            value.get("schema") != _STREAMING_INITIALIZATION_SCHEMA or
            value.get("scope") != "streamed_text_source_forward" or
            value.get("status") != "completed" or
            any(not isinstance(value.get(k), str) or not value[k]
                for k in ("transformers_version", "model_class", "dtype", "layers_prefix")) or
            any(type(value.get(k)) is not int or value[k] < minimum
                for k, minimum in (("num_layers", 1), ("persistent_tensors", 1), ("derived_buffers", 0))) or
            any(not isinstance(value.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", value[k])
                for k in ("state_sha256", "source_map_sha256"))):
        raise ValueError("Missing or incomplete streaming initialization contract")
    return dict(value)


def validate_streaming_prefix_initialization_contract(value):
    """Validate an observed proper prefix without admitting it as a full load."""
    keys = {"schema", "scope", "status", "transformers_version", "model_class",
            "dtype", "layers_prefix", "total_model_layers", "observed_layers",
            "head_state_names", "state", "persistent_tensors", "derived_buffers",
            "state_sha256", "source_map_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or
            value.get("schema") != "prismaquant.streaming_prefix_initialization.v1" or
            value.get("scope") != "streamed_text_source_prefix" or
            value.get("status") != "completed"):
        raise ValueError("Missing or invalid streaming prefix initialization contract")
    for key in ("transformers_version", "model_class", "dtype", "layers_prefix"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError("Invalid streaming prefix identity")
    layers = value["observed_layers"]
    if (type(value["total_model_layers"]) is not int or
            not isinstance(layers, list) or not layers or
            any(type(layer) is not int for layer in layers) or
            layers != list(range(len(layers))) or
            len(layers) >= value["total_model_layers"]):
        raise ValueError("Streaming prefix must cover every predecessor and remain a proper prefix")
    state, heads = value["state"], value["head_state_names"]
    if (not isinstance(state, dict) or not state or not isinstance(heads, list)
            or not heads or any(not isinstance(name, str) for name in heads)
            or heads != sorted(set(heads)) or not set(heads) <= set(state)):
        raise ValueError("Streaming prefix has incomplete head coverage")
    prefix = value["layers_prefix"]
    seen, checkpoint, derived = set(), 0, 0
    for name, record in state.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("Invalid streaming prefix state")
        kind = record.get("kind")
        expected = {"shape", "dtype", "kind"} | ({"sha256"} if kind == "derived_buffer" else set())
        if (kind not in {"checkpoint", "derived_buffer"} or set(record) != expected
                or not isinstance(record["shape"], list)
                or any(type(n) is not int or n < 0 for n in record["shape"])
                or not isinstance(record["dtype"], str) or not record["dtype"]):
            raise ValueError("Invalid streaming prefix tensor witness")
        if name in heads:
            if name.startswith(prefix):
                raise ValueError("Body state cannot stand in for prefix head coverage")
        else:
            index = _prefix_layer_index(name, prefix)
            if index is None or index not in layers:
                raise ValueError("Streaming prefix includes state outside its observed scope")
            seen.add(index)
        checkpoint += kind == "checkpoint"
        derived += kind == "derived_buffer"
        if kind == "derived_buffer" and not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            raise ValueError("Invalid derived-buffer digest")
    if seen != set(layers) or not any(state[name]["kind"] == "checkpoint" for name in heads):
        raise ValueError("Streaming prefix omitted a head or body layer")
    if (type(value["persistent_tensors"]) is not int or value["persistent_tensors"] != checkpoint
            or type(value["derived_buffers"]) is not int or value["derived_buffers"] != derived
            or value["state_sha256"] != _initialization_digest(state)
            or not re.fullmatch(r"[0-9a-f]{64}", str(value["source_map_sha256"]))):
        raise ValueError("Streaming prefix state digest/counts differ")
    return dict(value)


_SELECTED_INITIALIZATION_SCHEMA = "prismaquant.streaming_selected_initialization.v1"


def validate_streaming_selected_initialization_witness(value):
    """Validate the state a traversal of selected layers installed.

    A witness for a run that installs a few decoder layers and the head,
    not the whole source. It is provenance for the tensors that run produced
    and is never a source-initialization contract, so
    :func:`prismaquant.validate_source_initialization_contract` does not
    accept it.
    """
    keys = {"schema", "scope", "status", "transformers_version", "model_class",
            "dtype", "layers_prefix", "total_model_layers", "observed_layers",
            "head_state_names", "state", "persistent_tensors", "derived_buffers",
            "state_sha256", "source_map_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or
            value.get("schema") != _SELECTED_INITIALIZATION_SCHEMA or
            value.get("scope") != "streamed_text_source_selected" or
            value.get("status") != "completed"):
        raise ValueError("Missing or invalid streaming selected-layer witness")
    for key in ("transformers_version", "model_class", "dtype", "layers_prefix"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError("Invalid streaming selected-layer identity")
    layers, total = value["observed_layers"], value["total_model_layers"]
    if (type(total) is not int or not isinstance(layers, list) or not layers or
            any(type(layer) is not int or not 0 <= layer < total for layer in layers) or
            layers != sorted(set(layers))):
        raise ValueError("Streaming selected-layer witness names no valid layers")
    state, heads = value["state"], value["head_state_names"]
    if (not isinstance(state, dict) or not state or not isinstance(heads, list)
            or not heads or heads != sorted(set(heads)) or not set(heads) <= set(state)):
        raise ValueError("Streaming selected-layer witness has incomplete head coverage")
    prefix = value["layers_prefix"]
    seen, checkpoint, derived = set(), 0, 0
    for name, record in state.items():
        kind = record.get("kind") if isinstance(record, dict) else None
        expected = {"shape", "dtype", "kind"} | ({"sha256"} if kind == "derived_buffer" else set())
        if (kind not in {"checkpoint", "derived_buffer"} or set(record) != expected
                or not isinstance(record["shape"], list)
                or any(type(n) is not int or n < 0 for n in record["shape"])
                or not isinstance(record["dtype"], str) or not record["dtype"]):
            raise ValueError("Invalid streaming selected-layer tensor witness")
        if kind == "derived_buffer" and not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            raise ValueError("Invalid derived-buffer digest")
        if name not in heads:
            index = _prefix_layer_index(name, prefix)
            if index is None or index not in layers:
                raise ValueError("Streaming selected-layer state is outside its observed layers")
            seen.add(index)
        elif name.startswith(prefix):
            raise ValueError("Body state cannot stand in for head coverage")
        checkpoint += kind == "checkpoint"
        derived += kind == "derived_buffer"
    if seen != set(layers):
        raise ValueError("Streaming selected-layer witness omitted an observed layer")
    if (type(value["persistent_tensors"]) is not int or value["persistent_tensors"] != checkpoint
            or type(value["derived_buffers"]) is not int or value["derived_buffers"] != derived
            or value["state_sha256"] != _initialization_digest(state)
            or not re.fullmatch(r"[0-9a-f]{64}", str(value["source_map_sha256"]))):
        raise ValueError("Streaming selected-layer state digest/counts differ")
    return dict(value)


def merge_streaming_selected_initialization_witnesses(
    witnesses: Iterable[dict[str, Any]],
    *,
    expected_contract: StreamingInitializationContract,
) -> StreamingInitializationContract:
    """Reconstruct metadata only when it equals the complete census contract.

    Each completed selection must be contiguous; disjoint selections must tile
    every model layer. Head state is counted once, with every record agreeing.
    This does not authenticate checkpoint bytes or activate a capture chain.
    """
    expected = validate_streaming_initialization_contract(expected_contract)
    try:
        selections = [validate_streaming_selected_initialization_witness(w) for w in witnesses]
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError("Malformed streaming selected-layer witness collection") from exc
    if not selections:
        raise ValueError("Streaming initialization merge requires completed selections")
    selections.sort(key=lambda w: w["observed_layers"][0])
    heads = selections[0]["head_state_names"]
    state = {name: selections[0]["state"][name] for name in heads}
    seen = set()
    for witness in selections:
        if (witness["total_model_layers"] != expected["num_layers"] or
                any(witness[key] != expected[key] for key in (
                    "transformers_version", "model_class", "dtype", "layers_prefix", "source_map_sha256"))):
            raise ValueError("Streaming initialization merge identity differs from complete census")
        layers = witness["observed_layers"]
        if layers != list(range(layers[0], layers[-1] + 1)):
            raise ValueError("Streaming initialization merge requires contiguous selections")
        if seen.intersection(layers):
            raise ValueError("Streaming initialization merge selections overlap")
        seen.update(layers)
        if (witness["head_state_names"] != heads or
                any(witness["state"][name] != state[name] for name in heads)):
            raise ValueError("Streaming initialization merge head state differs")
        for name, record in witness["state"].items():
            if name in heads:
                continue
            if name in state:
                raise ValueError("Streaming initialization merge repeats body state")
            state[name] = record
    if seen != set(range(expected["num_layers"])):
        raise ValueError("Streaming initialization merge omitted model layers")
    merged: StreamingInitializationContract = {
        "schema": _STREAMING_INITIALIZATION_SCHEMA,
        "scope": "streamed_text_source_forward", "status": "completed",
        "transformers_version": expected["transformers_version"],
        "model_class": expected["model_class"], "dtype": expected["dtype"],
        "layers_prefix": expected["layers_prefix"], "num_layers": expected["num_layers"],
        "persistent_tensors": sum(r["kind"] == "checkpoint" for r in state.values()),
        "derived_buffers": sum(r["kind"] == "derived_buffer" for r in state.values()),
        "state_sha256": _initialization_digest(state),
        "source_map_sha256": expected["source_map_sha256"],
    }
    validate_streaming_initialization_contract(merged)
    if merged != expected:
        raise ValueError("Merged streaming initialization differs from complete census contract")
    return merged


_INITIALIZATION_CONTRACT_ATTRIBUTE = "_prismaquant_pretrained_initialization_contract"
_MTP_LAYER_INITIALIZATION_SCHEMA = "prismaquant.mtp_layer_initialization.v1"


def clear_pretrained_initialization_contract(model):
    """Retire an earlier completion before checkpoint finalization starts."""
    model.__dict__.pop(_INITIALIZATION_CONTRACT_ATTRIBUTE, None)


def record_pretrained_initialization(model, *, transformers_version):
    """Record successful missing-state initialization on the loaded model."""
    setattr(model, _INITIALIZATION_CONTRACT_ATTRIBUTE, {
        "schema": "prismaquant.pretrained_initialization.v1",
        "scope": "checkpoint_missing_state",
        "status": "completed",
        "transformers_version": transformers_version,
    })


def validate_pretrained_initialization_contract(value):
    """Validate and copy a checkpoint initialization provenance descriptor."""
    if not isinstance(value, dict) or set(value) != {
        "schema", "scope", "status", "transformers_version"
    } or value.get("schema") != "prismaquant.pretrained_initialization.v1" \
            or value.get("scope") != "checkpoint_missing_state" \
            or value.get("status") != "completed" \
            or not isinstance(value.get("transformers_version"), str) \
            or not value["transformers_version"].strip():
        raise ValueError("Missing or invalid pretrained initialization contract")
    return dict(value)


def pretrained_initialization_contract(model):
    """Require evidence that this model completed checkpoint initialization.

    From-config skeletons have no such descriptor. This records the load
    phase, not a general certification of subsequent model mutations.
    """
    return validate_pretrained_initialization_contract(
        getattr(model, _INITIALIZATION_CONTRACT_ATTRIBUTE, None))


def validate_source_initialization_contract(value):
    """Read a qualified source-loading route without conflating the routes.

    The streamed text forward, a checkpoint load through ``from_pretrained``,
    and the checkpoint load of one MTP layer outside that forward.
    """
    if isinstance(value, dict) and value.get("schema") == "prismaquant.streaming_initialization.v1":
        from .streaming_initialization import validate_streaming_initialization_contract
        return validate_streaming_initialization_contract(value)
    if isinstance(value, dict) and value.get("schema") == "prismaquant.mtp_layer_initialization.v1":
        from .streaming_initialization import validate_mtp_layer_initialization_contract
        return validate_mtp_layer_initialization_contract(value)
    return validate_pretrained_initialization_contract(value)


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
