"""Pure streamed initialization metadata policy; runtime audit owns production."""

from __future__ import annotations

import re
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
            match = re.fullmatch(re.escape(prefix) + r"(\d+)\..+", name)
            if match is None or int(match[1]) not in layers:
                raise ValueError("Streaming prefix includes state outside its observed scope")
            seen.add(int(match[1]))
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
            match = re.fullmatch(re.escape(prefix) + r"(\d+)\..+", name)
            if match is None or int(match[1]) not in layers:
                raise ValueError("Streaming selected-layer state is outside its observed layers")
            seen.add(int(match[1]))
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
