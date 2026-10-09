"""Consume pact.prefixed_clean_frontier.v1 through existing exact owners only.

The start reference may belong to a predecessor generation. It is attached
only to start_reference_owner, never to the receipt's stop generation.
All parent inputs remain borrowed/read-only. Clean intermediate outputs
use a private working generation of the SAME StreamedBoundaryArtifacts
owner; only these owned rolling entries are retired. No activation store,
foreign mutation, source seal or private residency system is introduced.

run_frontier_window executes exactly one declared window per admitted call
and commits its band_continuation checkpoint. The first window starts from
borrowed clean start boundaries. Later windows resume the exact prerequisite
checkpoint into a fresh working generation. Only owned rolling entries are
retired. Borrowed start boundaries are always retained.
"""
from __future__ import annotations
from contextlib import ExitStack
import io
import json
import hashlib
from pathlib import Path


def move_state(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key:move_state(item,device) for key,item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_state(item,device) for item in value)
    if isinstance(value, list):
        return [move_state(item,device) for item in value]
    return value


def load_frontier(args, cohort, inputs, *, cpu_proof=False):
    import torch
    from g3_residency import read_file
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    receipt = json.loads(read_file(args.boundaries))
    if receipt.get("schema") != "pact.prefixed_clean_frontier.v1":
        raise ValueError("The clean frontier schema differs")
    if cpu_proof and not args.dry_run_cpu:
        raise ValueError("Only the CPU proof can use the proof frontier")
    if receipt.get("dry_run") and not cpu_proof:
        raise ValueError("A proof frontier cannot supply source prices")
    declared = receipt["cohort"]
    for key in ("sample_range","raw_tokens_per_sequence","prefix_ids","input_contract",
                "local_prefix_rows","global_original_tokens","scored_positions_per_sequence"):
        if declared.get(key) != cohort[key]:
            raise ValueError("Clean frontier and energy cohort differ now: " + key)
    samples = list(range(*cohort["sample_range"]))
    if receipt.get("dry_run"):
        if receipt["sample_ids"] != [samples[0]]:
            raise ValueError("The CPU frontier must name the first held-out sample")
        samples = receipt["sample_ids"]
    elif receipt["sample_ids"] != samples:
        raise ValueError("The clean frontier omits or repeats held-out samples")
    start,stop = receipt["layer_start"],receipt["layer_stop"]
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 45:
        raise ValueError("Clean frontier has an invalid actual layer range")
    lo,hi = map(int,args.layer_range.split(":"))
    if not cpu_proof and not start <= lo < hi <= stop:
        raise ValueError("The energy layers lie outside the clean frontier")
    state_bytes = read_file(receipt["start_batch_state_path"])
    states = torch.load(io.BytesIO(state_bytes),map_location="cpu",weights_only=True)
    if set(states) != set(samples) or set(receipt["start_references"]) != {str(s) for s in samples}:
        raise ValueError("Clean start states/references do not cover exactly the heldout cohort")
    refs = {}
    token_width = cohort["raw_tokens_per_sequence"] + len(cohort["prefix_ids"])
    for sample,ids in zip(samples,inputs):
        state = states[sample]
        if not torch.equal(state["input_ids"],ids) or list(ids.shape) != [1,token_width]:
            raise ValueError("Incoming original/prefix tokens or shape differ for sample %d"%sample)
        reference = reference_from_record(receipt["start_references"][str(sample)])
        metadata = json.loads(reference.metadata_json)
        coordinates = metadata["identity"]["coordinates"]
        # Current sample/layer correctness, not a recorded source identity.
        if coordinates["batch"] != sample or coordinates["boundary"] != start:
            raise ValueError("Clean start reference names another sample/layer")
        if len(reference.shape) < 2 or reference.shape[0] != 1 or reference.shape[1] != token_width:
            raise ValueError("Clean start hidden has wrong sample/token axes")
        refs[sample] = reference
    return receipt,states,refs,samples


def run_frontier(runner, args, cohort, inputs, measure, guard, *, stop_layer=None, on_tail=None):
    import torch
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts, StreamedForwardBoundaries
    receipt,states,refs,samples = load_frontier(args,cohort,inputs)
    start = receipt["layer_start"]
    stop = receipt["layer_stop"] if stop_layer is None else stop_layer
    if type(stop) is not int or not start < stop <= 45:
        raise ValueError("The replay stop must lie inside the actual source graph")
    incoming_owner = receipt["start_reference_owner"]
    with ExitStack() as stack:
        borrowed = stack.enter_context(StreamedBoundaryArtifacts(incoming_owner["boundary_config"]))
        borrowed.attach(incoming_owner["session"],n_probes=0)
        config = dict(receipt["boundary_config"],directory=str(args.boundary_dir/"energy-working"))
        rolling = stack.enter_context(StreamedBoundaryArtifacts(config))
        rolling.bind({"cohort":cohort,"layer_range":[start,stop]},n_probes=0,check_memory=guard,published=False)
        replay_state = {"states":states,"borrowed_owner":borrowed,"working_owner":rolling,"current":{}}
        for layer in range(start,stop):
            guard("clean frontier source prefetch")
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer,require_prefetched=True,prefetch_following=False)
            if layer+1 < stop:
                runner.context.schedule_prefetch(layer+1)
            output_refs = {}
            cursor = 0
            owner = borrowed if layer == start else rolling
            def forward(ids):
                nonlocal cursor
                if cursor >= len(samples):
                    raise ValueError("Energy replay repeated a sequence")
                sample = samples[cursor]
                if not torch.equal(ids,inputs[cursor]):
                    raise ValueError("Energy replay changed ordered actual tokens")
                row = move_state(states[sample],runner.device)
                pass_state = row["pass_state"]
                batch = StreamedForwardBoundaries(row["input_ids"],row["position_ids"],
                    row["position_embeddings"],row["attention_mask"],[],None)
                with owner.prefetch([refs[sample]]) as window:
                    hidden = owner.get(window,refs[sample]).to(runner.device,dtype=runner.dtype)
                    replay_state["current"] = {"row":row,"hidden":hidden,"pass_state":pass_state}
                    guard("resident clean frontier call")
                    with torch.inference_mode():
                        output = runner.isolated_layer(batch,layer,hidden,pass_state=pass_state)
                    replay_state["current"]["output"] = output
                    if layer+1 < stop:
                        output_refs[sample] = rolling.write(output,batch_index=sample,boundary_index=layer+1)
                    elif on_tail is not None:
                        on_tail(sample, batch, output, pass_state)
                    states[sample]["pass_state"] = move_state(pass_state,"cpu")
                    replay_state["current"].clear()
                    del hidden,output
                cursor += 1
            try:
                measure(layer,forward,replay_state)
                if cursor != len(samples):
                    raise ValueError("Energy replay omitted actual heldout sequences")
            finally:
                runner.context.unload(layer)
            if layer > start:
                for reference in refs.values():
                    rolling.retire(reference)
            refs = output_refs
    return {"schema":receipt["schema"],"layer_range":[start,stop],
            "start_generation":incoming_owner["session"],"stop_generation":receipt["session"],
            "foreign_entries_retired":0,"sample_ids":samples}


def _read_back_hidden(owner, refs, *, guard):
    """Read every window hidden tensor through the owner in admitted windows.

    One prefetch never asks for more than the owner resident byte budget
    has free. A single tensor above that budget refuses before any read.
    """
    guard("perturbed window hidden readback")
    budget = owner.config["max_resident_bytes"] - owner.telemetry["resident_tensor_bytes"]
    ordered, hidden, batch, size = sorted(refs), {}, [], 0
    batches = []
    for sample in ordered:
        nbytes = refs[sample].tensor_bytes
        if nbytes > budget:
            raise RuntimeError("One hidden tensor exceeds the owner resident byte budget")
        if batch and size + nbytes > budget:
            batches.append(batch)
            batch, size = [], 0
        batch.append(sample)
        size += nbytes
    if batch:
        batches.append(batch)
    for batch in batches:
        guard("perturbed window hidden readback")
        with owner.prefetch([refs[sample] for sample in batch]) as window:
            for sample in batch:
                hidden[sample] = owner.get(window, refs[sample]).detach().to(device="cpu", copy=True)
    return hidden


def run_frontier_window(runner, args, cohort, inputs, measure, guard, *, band_start, band_stop,
                        replay_start, replay_stop, window_start, window_stop, window_layers,
                        checkpoint_dir, stream, source, teacher_digest, input_digest, on_tail=None,
                        teacher_receipt_sha256=None):
    """Execute exactly one declared perturbed window and commit its checkpoint.

    The admitted call installs only its declared layers. The first window of
    the replay extent starts from the borrowed clean start boundaries. Any
    later window resumes its exact prerequisite checkpoint into a fresh
    working generation and never restarts from the clean frontier. The per
    layer replay matches run_frontier exactly: the existing pass-state
    producer and the same isolated_layer calls. The call commits one
    band_continuation checkpoint bound to the one executed window, retires
    every owned entry it created and returns the committed checkpoint with
    the code-owned next-window dependencies. Borrowed start boundaries are
    always retained. No automatic source read and no fallback exist.
    """
    import torch
    import band_continuation as continuation
    from band_continuation import _require_hex
    from g3_residency import read_file
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts, StreamedForwardBoundaries
    continuation.check_coordinates(band_start, band_stop, replay_start, replay_stop, window_stop,
                                   window_start, window_stop)
    receipt, start_states, start_refs, samples = load_frontier(args, cohort, inputs)
    if replay_start != receipt["layer_start"]:
        raise ValueError("The replay must start at the actual borrowed starting boundary")
    if not replay_start < replay_stop <= 45:
        raise ValueError("The replay stop must lie inside the actual source graph")
    stream = continuation.check_roster_entry(stream)
    source = continuation.check_source(source)
    if source["source_window"] != [window_start, window_stop]:
        raise ValueError("The source window must equal the one executed window")
    _require_hex(teacher_digest, "teacher_digest")
    _require_hex(input_digest, "input_digest")
    receipt_sha = hashlib.sha256(bytes(read_file(args.boundaries))).hexdigest()
    if source["receipt_sha256"] != receipt_sha:
        raise ValueError("The source receipt bytes differ")
    windows = continuation.plan_windows(replay_start, replay_stop, window_layers=window_layers)
    if {"window_start": window_start, "window_stop": window_stop} not in windows:
        raise ValueError("The window is not a declared plan window")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_checkpoint = continuation.checkpoint_path(directory, window_start, window_stop)
    if output_checkpoint.exists():
        raise ValueError("A committed window blocks its replay range")
    incoming_owner = receipt["start_reference_owner"]

    def global_bindings():
        return {"band_start": band_start, "band_stop": band_stop, "replay_start": replay_start,
                "replay_stop": replay_stop, "cohort": cohort, "stream_roster": [stream],
                "teacher_digest": teacher_digest, "input_digest": input_digest,
                "teacher_receipt_sha256": teacher_receipt_sha256,
                "perturbed": True, "streams": {stream["stream_id"]: {"sample_ids": samples}}}

    def expected_for(window, window_source):
        return {**global_bindings(), "next_layer": window["window_stop"],
                "window_start": window["window_start"], "window_stop": window["window_stop"],
                "window_layers": window_layers, "source": window_source}

    def forward_inputs(sample, states):
        return {key: move_state(states[sample][key], "cpu") for key in (
            "input_ids", "position_ids", "position_embeddings", "attention_mask")}

    with ExitStack() as stack:
        borrowed = stack.enter_context(StreamedBoundaryArtifacts(incoming_owner["boundary_config"]))
        borrowed.attach(incoming_owner["session"], n_probes=0)
        config = dict(receipt["boundary_config"], directory=str(args.boundary_dir / "energy-working"))
        rolling = stack.enter_context(StreamedBoundaryArtifacts(config))
        rolling.bind({"cohort": cohort, "layer_range": [replay_start, replay_stop],
                      "window": [window_start, window_stop]}, n_probes=0,
                     check_memory=guard, published=False)
        replay_state = {"states": None, "borrowed_owner": borrowed, "working_owner": rolling,
                        "current": {}}
        resumed_from = None
        if window_start == replay_start:
            states, live_refs, live_owned = start_states, start_refs, False
        else:
            previous = None
            for window in windows:
                if window["window_stop"] == window_start:
                    previous = window
            if previous is None:
                raise ValueError("The window has no declared prerequisite window")
            candidate = continuation.checkpoint_path(directory, previous["window_start"],
                                                     previous["window_stop"])
            if not candidate.exists():
                raise ValueError("The prerequisite window checkpoint is missing")
            previous_source = dict(source, source_window=[previous["window_start"],
                                                          previous["window_stop"]])
            checkpoint = continuation.load_window_checkpoint(
                candidate, expected=expected_for(previous, previous_source), guard=guard)
            restored = continuation.resume_stream_window(
                checkpoint, stream["stream_id"], window_start=previous["window_start"],
                window_stop=previous["window_stop"], guard=guard)
            for sample, ids in zip(samples, inputs):
                continuation.validate_forward_state(restored["forward"][sample], input_ids=ids,
                    expected=forward_inputs(sample, start_states))
            states = {sample: {**start_states[sample], **restored["forward"][sample],
                               "pass_state": restored["pass_state"][sample]} for sample in samples}
            live_refs = {sample: rolling.write(restored["hidden"][sample], batch_index=sample,
                                                boundary_index=window_start) for sample in samples}
            live_owned = True
            resumed_from = str(candidate)
        replay_state["states"] = states
        executed = []
        for layer in range(window_start, window_stop):
            guard("clean frontier source prefetch")
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=True, prefetch_following=False)
            if layer + 1 < window_stop:
                runner.context.schedule_prefetch(layer + 1)
            output_refs = {}
            cursor = 0
            owner = borrowed if layer == replay_start and not live_owned else rolling
            input_refs, input_owned = live_refs, live_owned

            def forward(ids):
                nonlocal cursor
                if cursor >= len(samples):
                    raise ValueError("Energy replay repeated a sequence")
                sample = samples[cursor]
                if not torch.equal(ids, inputs[cursor]):
                    raise ValueError("Energy replay changed ordered actual tokens")
                row = move_state(states[sample], runner.device)
                pass_state = row["pass_state"]
                batch = StreamedForwardBoundaries(row["input_ids"], row["position_ids"],
                    row["position_embeddings"], row["attention_mask"], [], None)
                with owner.prefetch([input_refs[sample]]) as resident:
                    hidden = owner.get(resident, input_refs[sample]).to(
                        runner.device, dtype=runner.dtype)
                    replay_state["current"] = {"row": row, "hidden": hidden,
                                               "pass_state": pass_state}
                    guard("resident clean frontier call")
                    with torch.inference_mode():
                        output = runner.isolated_layer(batch, layer, hidden,
                                                       pass_state=pass_state)
                    replay_state["current"]["output"] = output
                    output_refs[sample] = rolling.write(output, batch_index=sample,
                                                        boundary_index=layer + 1)
                    if layer + 1 == replay_stop and on_tail is not None:
                        on_tail(sample, batch, output, pass_state)
                    states[sample]["pass_state"] = move_state(pass_state, "cpu")
                    replay_state["current"].clear()
                    del hidden, output
                cursor += 1
            try:
                measure(layer, forward, replay_state)
                if cursor != len(samples):
                    raise ValueError("Energy replay omitted actual heldout sequences")
            finally:
                runner.context.unload(layer)
            if input_owned:
                for reference in input_refs.values():
                    rolling.retire(reference)
            live_refs, live_owned = output_refs, True
            executed.append(layer)
        hidden = _read_back_hidden(rolling, live_refs, guard=guard)
        checkpoint_file = continuation.checkpoint_path(directory, window_start, window_stop)
        checkpoint = continuation.commit_stream_window(
            checkpoint_file, band_start=band_start, band_stop=band_stop,
            replay_start=replay_start, replay_stop=replay_stop, next_layer=window_stop,
            window_start=window_start, window_stop=window_stop, window_layers=window_layers,
            cohort=cohort, stream_roster=[stream], source=source, teacher_digest=teacher_digest,
            input_digest=input_digest, teacher_receipt_sha256=teacher_receipt_sha256,
            streams={stream["stream_id"]: {
                "sample_ids": samples, "hidden": hidden,
                "pass_state": {sample: states[sample]["pass_state"] for sample in samples},
                "forward": {sample: forward_inputs(sample, states) for sample in samples},
                "references": dict(live_refs),
                "owned_boundary_generation": dict(rolling.session)}}, guard=guard)
        for reference in live_refs.values():
            rolling.retire(reference)
    return {"schema": receipt["schema"], "band_start": band_start, "band_stop": band_stop,
            "replay_start": replay_start, "replay_stop": replay_stop, "window_start": window_start,
            "window_stop": window_stop, "next_layer": window_stop, "sample_ids": samples,
            "executed_layers": executed, "checkpoint_path": str(checkpoint_file),
            "checkpoint": checkpoint, "resumed_from": resumed_from, "final_hidden": hidden,
            "windows": windows, "next": continuation.next_window_dependencies(checkpoint, checkpoint_file=checkpoint_file)}
