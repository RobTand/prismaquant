"""GLM MTP layer 45: calibration capture (PQ #1290, part of #1271).

The MTP layer's input is the target's post-final-norm hidden state ``h``. The
body pipeline never materializes it: Stage A keeps layer boundaries, and its
last one (``boundary-<i>-44-at-44``) is the *input* of layer 44. So the capture
runs in two phases:

1. :func:`final_hidden` runs the last backbone layer from that boundary on the
   sealed streamed runner, then the model's own mHC collapse and final norm.
   These are the steps of ``StreamedCausalLM._finish`` before its head.
2. :class:`MtpCaptureFeed` runs the MTP layer on ``h`` for each calibration
   sequence. The body campaign's collector (``tessera_campaign._collect_activations``)
   hooks the layer through :class:`~prismaquant.glm_mtp.MtpCheckpointModel`,
   unchanged, so the rows, Hessians, counts and maxima mean what the body's do.

The collector calls its forward once per calibration batch and passes no
index. The feed keeps its own index and refuses a batch that is not the next
calibration row, because a hidden state paired with the wrong tokens produces
plausible tensors that nothing downstream can catch.

The capture is published in the canonical format
(``tessera_calibration_cache.v2``) over its own census, which
:func:`mtp_census` derives from the body census: the same model, draw and
producer source roster, and the 867 MTP units. Its identity inherits the
canonical capture's hash-bound source roster
(``CaptureSourceAuthentication.admit_derived_census``), and every shard the
two phases read is authenticated through that owner. Its initialization
contract is the MTP layer's own checkpoint load
(``prismaquant.mtp_layer_initialization.v1``), bound to the hidden states it
ran on, because the layer is outside the streamed text forward whose witness
the body capture carries.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import closing
from pathlib import Path

import torch

from . import glm_mtp

FINAL_HIDDEN_SCHEMA = "prismaquant.glm_mtp.final_hidden.v1"
CENSUS_EXTENSION_SCHEMA = "prismaquant.glm_mtp.census_extension.v1"


def final_hidden(runner, input_ids, boundary, *, layer: int):
    """The target's post-final-norm hidden state from the last layer's input.

    ``runner`` is a ``StreamedCausalLM`` whose ``layer`` the caller has
    installed. ``boundary`` is that layer's input in the runner's stream
    layout, as Stage A stored it. The batch metadata (positions, masks) comes
    from the runner's own ``_prepare`` on the same ids, as the quanta rebuild
    it. Refuses any layer but the last, since only the last feeds the norm.
    """
    from .cost_streaming import StreamedForwardBoundaries
    from .layer_streaming import _get_final_norm

    if int(layer) != int(runner.num_layers) - 1:
        raise ValueError(
            f"final hidden state comes from the last backbone layer "
            f"({runner.num_layers - 1}), not layer {layer}")
    ids, position_ids, hidden, embeddings, mask = runner._prepare(input_ids)
    del hidden
    batch = StreamedForwardBoundaries(ids, position_ids, embeddings, mask, [], None)
    pass_state = runner.profile.isolated_layer_pass_state(None, runner.layers[layer])
    out = runner.isolated_layer(
        batch, int(layer), boundary.to(device=runner.device, dtype=runner.dtype),
        pass_state=pass_state)
    out = runner.profile.collapse_hidden_after_layers(out, runner.base_model)
    norm = _get_final_norm(runner.base_model)
    return norm(out) if norm is not None else out


class MtpCaptureFeed:
    """The collector's ``forward_batch`` for the MTP layer.

    ``model`` is an :class:`~prismaquant.glm_mtp.MtpCheckpointModel`.
    ``calibration_ids`` is the ``[n, T]`` draw on CPU. ``read_final_hidden(i)``
    returns sequence ``i``'s ``[1, T, H]`` post-final-norm hidden state.
    :meth:`batches` yields the draw one sequence at a time. Each call must
    receive the next of those, in order.
    """

    def __init__(self, model, embed_tokens, calibration_ids, *, read_final_hidden):
        if calibration_ids.ndim != 2:
            raise ValueError("MTP capture needs [n, T] calibration ids")
        self.model = model
        self.embed_tokens = embed_tokens
        self.calibration_ids = calibration_ids.detach().to("cpu")
        self.read_final_hidden = read_final_hidden
        self.fed = 0

    def batches(self):
        for index in range(int(self.calibration_ids.shape[0])):
            yield self.calibration_ids[index:index + 1]

    def __call__(self, input_ids):
        index = self.fed
        total = int(self.calibration_ids.shape[0])
        if index >= total or not torch.equal(
                input_ids.detach().to("cpu"), self.calibration_ids[index:index + 1]):
            raise RuntimeError(
                f"MTP capture expected calibration row {index} of {total}; the batch "
                "it received is not that row")
        hidden = self.read_final_hidden(index)
        if tuple(hidden.shape[:2]) != tuple(input_ids.shape):
            raise RuntimeError(
                f"final hidden state for row {index} has shape {tuple(hidden.shape)}, "
                f"not [1, {input_ids.shape[1]}, H]")
        # The layer runs at its own weights' dtype, as the served MTP head
        # does, whatever dtype phase 1 stored the hidden states in.
        dtype = self.model.layer.eh_proj.weight.dtype
        hidden = hidden.to(device=input_ids.device, dtype=dtype)
        next_ids, previous, positions = glm_mtp.mtp_rows(input_ids, hidden)
        out = self.model(self.embed_tokens(next_ids).to(dtype), previous, positions)
        self.fed += 1
        return out

    def require_complete(self):
        total = int(self.calibration_ids.shape[0])
        if self.fed != total:
            raise RuntimeError(f"MTP capture fed {self.fed} of {total} calibration rows")


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_bound_json(path, expected_sha256):
    """``(document, sha256)`` of a JSON file whose bytes must hash as expected."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"{path}: sha256 {digest} is not the bound {expected_sha256}")
    return json.loads(raw), digest


def ordered_boundary_records(manifest, n_sequences, *, layer):
    """Layer ``layer``'s input for sequences ``0..n-1``, in order.

    Each record is checked before anything is read: its name and its own
    ``coordinates`` (batch and boundary) must both say it is that sequence's
    input to that layer. The exact reader checks each entry's session.
    """
    records = manifest["records"]
    by_name = {record["name"]: record for record in records}
    if len(by_name) != len(records):
        raise RuntimeError("boundary manifest names one entry twice")
    ordered = []
    for index in range(int(n_sequences)):
        name = f"boundary-{index}-{layer}-at-{layer}"
        record = by_name.get(name)
        if record is None:
            raise RuntimeError(f"boundary manifest has no layer-{layer} input for sequence {index}")
        identity = (record.get("metadata") or {}).get("identity") or {}
        coordinates = identity.get("coordinates") or {}
        if coordinates.get("batch") != index or coordinates.get("boundary") != layer:
            raise RuntimeError(
                f"boundary entry {name} is not sequence {index}'s layer-{layer} input "
                f"(coordinates {coordinates})")
        ordered.append(record)
    return ordered


def write_final_hidden(runner, calibration_ids, boundary_records, *, boundary_session,
                       layer, out_dir, session, read_ahead_bytes, head=None, progress=None):
    """Phase 1: every sequence's post-final-norm hidden state, as exact entries.

    ``boundary_records`` come from :func:`ordered_boundary_records`. They are
    read in order with a bounded read-ahead, each one hash-verified before
    it is used. ``head`` is the target's ``lm_head``: when given, each
    sequence's next-token top-1 agreement and NLL are recorded beside its
    entry, a check that the hidden state is the one the head reads.

    ``progress``, when given, is called with the count of sequences written
    after each one.

    Returns ``(records, head_check)``. Refuses an output directory that
    already holds entries: a retry writes to a new one.
    """
    from .joint_adjoint_checkpoints import exact_entry_record, stream_exact_entry_tensors
    from .perturbed_x_cache import write_exact_activation_cache_entry

    n_sequences = int(calibration_ids.shape[0])
    if len(boundary_records) != n_sequences:
        raise RuntimeError(
            f"{len(boundary_records)} boundary entries for {n_sequences} calibration sequences")
    entries = Path(out_dir) / "entries"
    if entries.exists() and any(entries.iterdir()):
        raise RuntimeError(f"{entries} already holds entries; write to a new directory")
    entries.mkdir(parents=True, exist_ok=True)
    records, head_check = [], []
    stream = stream_exact_entry_tensors(
        boundary_records, expected_session=boundary_session,
        max_resident_bytes=read_ahead_bytes)
    with closing(stream):
        for index, (record, boundary) in enumerate(stream):
            if record["name"] != boundary_records[index]["name"]:
                raise RuntimeError(f"boundary stream yielded {record['name']} at sequence {index}")
            ids = calibration_ids[index:index + 1]
            with torch.inference_mode():
                hidden = final_hidden(runner, ids, boundary, layer=layer)
                if head is not None:
                    logits = head(hidden[0, :-1]).float()
                    target = ids[0, 1:].to(logits.device)
                    head_check.append({
                        "sequence": index,
                        "top1": float((logits.argmax(-1) == target).float().mean()),
                        "nll": float(torch.nn.functional.cross_entropy(logits, target)),
                    })
                    del logits
            value = hidden.detach().to("cpu").contiguous()
            del hidden, boundary
            name = f"final-hidden-{index}"
            reference = write_exact_activation_cache_entry(
                entries, name, value,
                identity={"kind": "final_hidden", "session": session, "slot": name,
                          "coordinates": {"batch": index, "layer": int(layer)}},
                max_tensor_bytes=value.numel() * value.element_size(),
                max_file_bytes=value.numel() * value.element_size() + (1 << 16))
            records.append(exact_entry_record(reference))
            if progress is not None:
                progress(index + 1)
    return records, head_check


def publish_final_hidden(out_dir, *, session, records, layer, inputs, source_witness,
                         source_authentication, head_check):
    """Write phase 1's manifest; returns ``{schema, path, sha256}``."""
    manifest = {
        "schema": FINAL_HIDDEN_SCHEMA, "session": session, "layer": int(layer),
        "inputs": inputs, "source_witness": source_witness,
        "source_authentication": source_authentication,
        "head_check": head_check, "records": records,
    }
    from .cost_stage_checkpoint import atomic_write_bytes

    path = Path(out_dir) / "manifest.json"
    if path.exists():
        raise RuntimeError(f"{path} exists; phase 1 publishes once")
    raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(path, raw)
    return {"schema": FINAL_HIDDEN_SCHEMA, "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest()}


def final_hidden_session(run_identity):
    """A fresh session for one phase-1 run, bound to what it read."""
    return {"generation": uuid.uuid4().hex, "run_identity_sha256": _json_sha256(run_identity)}


def final_hidden_stream(manifest, n_sequences, *, read_ahead_bytes):
    """``read(index)`` over phase 1's entries, in order, hash-verified.

    Records are checked against their sequence before any read, and the
    stream refuses an index that is not the next one.
    """
    from .joint_adjoint_checkpoints import stream_exact_entry_tensors

    if manifest.get("schema") != FINAL_HIDDEN_SCHEMA:
        raise RuntimeError("not a GLM MTP final-hidden manifest")
    by_name = {record["name"]: record for record in manifest["records"]}
    ordered = []
    for index in range(int(n_sequences)):
        record = by_name.get(f"final-hidden-{index}")
        coordinates = ((record or {}).get("metadata") or {}).get("identity", {}).get("coordinates")
        if record is None or (coordinates or {}).get("batch") != index:
            raise RuntimeError(f"final-hidden manifest has no entry for sequence {index}")
        ordered.append(record)
    stream = stream_exact_entry_tensors(ordered, expected_session=manifest["session"],
                                        max_resident_bytes=read_ahead_bytes)
    state = {"next": 0}

    def read(index):
        if index != state["next"]:
            raise RuntimeError(f"final-hidden stream expected sequence {state['next']}, not {index}")
        record, tensor = next(stream)
        if record["name"] != f"final-hidden-{index}":
            raise RuntimeError(f"final-hidden stream yielded {record['name']} for sequence {index}")
        state["next"] += 1
        return tensor

    return read, stream


def capture_mtp_layer(model, embed_tokens, calibration_ids, read_final_hidden, *, units,
                      profile, device, max_act_rows, resource_check=None):
    """Phase 2: the body's collector over the MTP layer, one sequence at a time.

    Returns the collector's ``(rows, hessians, counts, max_abs)`` for ``units``.
    """
    from .tessera_campaign import _collect_activations

    feed = MtpCaptureFeed(model, embed_tokens, calibration_ids,
                          read_final_hidden=read_final_hidden)
    result = _collect_activations(
        model, sorted(units), feed.batches(), int(max_act_rows), device,
        want_hessian=True, profile=profile, forward_batch=feed,
        resource_check=resource_check)
    feed.require_complete()
    return result


def mtp_anchor_groups(model, units, profile):
    """The body's anchor grouping (``resolve_anchor_groups``) over the MTP units."""
    from .routed_experts import profile_declared_packed_expert_projections
    from .tessera_campaign import resolve_anchor_groups

    members = {member.qname: member
               for member in profile_declared_packed_expert_projections(model, profile)}
    return resolve_anchor_groups(sorted(units), profile=profile, expert_members=members)


def mtp_projection_request(base_census, stacks):
    """``{stack: (grid, q256)}``: the question the body census asked, per stack.

    The producer's unit records do not depend on the nominal rung it is asked
    at (``tessera_expert_projection.stack_plan_request``), so this is not a
    choice made here: it is the one rung the body census's own request named
    for every stack. A census that recorded none, or several, refuses.
    """
    asked = ((base_census.get("expert_projection") or {}).get("request") or {})
    nominal = {(str(entry["grid"]), int(entry["q256"])) for entry in asked.values()}
    if len(nominal) != 1:
        raise RuntimeError(
            f"the base census recorded {len(nominal)} nominal producer requests "
            f"{sorted(nominal)}; the MTP projection asks the one the body asked")
    grid, q256 = nominal.pop()
    return {str(stack): (grid, q256) for stack in stacks}


def mtp_expert_projection(model_path, model, profile, *, base_census, out_path):
    """The producer's projection of the MTP layer's routed stack, as a census carries it.

    ``model`` is an :class:`~prismaquant.glm_mtp.MtpCheckpointModel`; its
    tensors may be on the meta device, since only the declared population is
    read. The producer is asked once (it hashes the whole checkpoint to seal
    its source) and its answer is bound exactly to the profile-declared
    units, the campaign's own binding. Returns the carried block
    (``tessera_expert_projection.carried_projection``) with its one attempt.
    """
    from .tessera_campaign import _require_campaign_population
    from .tessera_expert_projection import (bind_expert_projection, carried_projection,
                                            producer_plan_tool, request_expert_projection,
                                            stack_plan_request)

    population = _require_campaign_population(model, profile, int(base_census["layer_stride"]))
    if not population.declared:
        raise RuntimeError("the MTP layer declares no routed expert stack to project")
    stacks = mtp_projection_request(base_census, population.declared)
    tool = producer_plan_tool()
    answer = request_expert_projection(model_path, stacks, out_path=out_path)
    bound = bind_expert_projection(answer, declared=population.declared)
    request = stack_plan_request(stacks)
    carried = carried_projection(answer, bound, request=request, tool=str(tool))
    carried["plan_attempts"] = [{"request": request, "refused": None}]
    return carried


def check_mtp_expert_projection(carried, model, profile, *, model_path,
                                source_authentication=None, layer_stride=1):
    """Bind a carried projection to the loaded MTP layer and check its bytes.

    The census path's check (``tessera_campaign._checked_projected_units``):
    every routed unit's source tensor, read from the shard the producer
    hashed, must equal the loaded layer's view of it. Returns
    ``{qname: unit record}`` for every routed unit.
    """
    from .tessera_campaign import _checked_projected_units, _require_campaign_population
    from .tessera_expert_projection import bind_expert_projection

    population = _require_campaign_population(model, profile, int(layer_stride))
    bound = bind_expert_projection(carried["producer"], declared=population.declared)
    return _checked_projected_units(
        bound, weights={member.qname: member.weight for member in population.members},
        model_path=model_path, source=carried["producer"]["source"],
        **({"source_authentication": source_authentication}
           if source_authentication is not None else {}))


def mtp_census(*, base_census, base_census_ref, canonical_capture_ref, final_hidden_ref,
               layer, units, counts, max_abs, groups, model_load_contract,
               attention_implementation, capture_runtime, expert_projection):
    """The MTP capture's census: the body census's schema over the MTP units.

    Model and draw are the body census's. ``expert_projection`` is the
    producer's projection of the MTP stack (:func:`mtp_expert_projection`),
    which the campaign reads instead of asking the producer per row; its
    source seal is the body producer's roster, so the capture inherits that
    roster (``admit_derived_census``). ``mtp_extension`` names what the census
    was derived from.
    """
    from types import SimpleNamespace

    from .tessera_campaign import calibration_census

    if set(counts) != set(units) or set(max_abs) != set(units):
        raise RuntimeError("MTP census counts and maxima must cover exactly its units")
    routed = sorted(name for name in units if ".mlp.experts." in name)
    dense = sorted(name for name in units if name not in set(routed))
    args = SimpleNamespace(**{key: base_census[key] for key in (
        "model", "nsamples", "seqlen", "seed", "layer_stride")})
    census = calibration_census(
        counts, max_abs, args=args, groups=groups, dense_targets=dense,
        expert_targets=routed, shapes=units,
        identity={key: base_census[key] for key in ("text_sha256", "fit_ids_sha256")},
        expert_projection=dict(expert_projection),
        model_load_contract=model_load_contract,
        attention_implementation=attention_implementation, capture_runtime=capture_runtime)
    census["mtp_extension"] = {
        "schema": CENSUS_EXTENSION_SCHEMA, "layer": int(layer),
        "base_census": dict(base_census_ref), "canonical_capture": dict(canonical_capture_ref),
        "final_hidden": dict(final_hidden_ref),
        "source_files": "inherited from canonical_capture; each shard read is authenticated",
    }
    return census


def publish_mtp_capture(root, *, census, census_path, source_authentication, calibration,
                        max_act_rows, rows, hessians, counts, max_abs,
                        completed_contract):
    """Write the census, then the capture through the canonical writer.

    ``completed_contract`` is the MTP layer's initialization contract taken
    again after the forward. The writer refuses it unless it equals the
    census's, the contract the capture identity names.
    Returns ``(identity, census_sha256, capture receipt)``.
    """
    from . import tessera_calibration_cache as cc
    from .cost_stage_checkpoint import atomic_write_bytes

    raw = (json.dumps(census, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(Path(census_path), raw)
    census_sha256 = hashlib.sha256(raw).hexdigest()
    admitted = source_authentication.admit_derived_census(census_path)
    if admitted != census_sha256:
        raise RuntimeError("MTP census changed between its write and its admission")
    identity = cc.capture_identity(
        census_path, calibration=calibration, max_act_rows=int(max_act_rows),
        model_load_contract=census["model_load_contract"],
        attention_implementation=census["attention_implementation"],
        source_authentication=source_authentication)
    writer = cc.CaptureWriter(root, census_path=census_path, identity=identity)
    writer.write(acts=rows, hessians=hessians, counts=counts, maxima=max_abs)
    receipt = writer.finish(model_load_contract=completed_contract)
    return identity, census_sha256, receipt


__all__ = [
    "CENSUS_EXTENSION_SCHEMA",
    "FINAL_HIDDEN_SCHEMA",
    "MtpCaptureFeed",
    "capture_mtp_layer",
    "check_mtp_expert_projection",
    "final_hidden",
    "final_hidden_session",
    "final_hidden_stream",
    "mtp_anchor_groups",
    "mtp_census",
    "mtp_expert_projection",
    "mtp_projection_request",
    "ordered_boundary_records",
    "publish_final_hidden",
    "publish_mtp_capture",
    "read_bound_json",
    "write_final_hidden",
]
