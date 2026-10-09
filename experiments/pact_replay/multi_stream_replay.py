"""Batched multi-stream band replay on the fixed clean prefixed cohort.

One resident source layer serves every stream of one window step. Each
stream keeps its own independent exact hidden state and pass state from
the same captured input cohort; no stream reuses another stream's mutable
state. Amplitudes 1, 2 and 4 are recorded for each kind. The accepted
(1, 2) or (2, 4) fit pair is still selected by the gain reducer; this
entry never changes that fit.

Bounded windows use the exact E1 helpers from band_continuation. Every
completed window commits one checkpoint across all roster streams; a resume
republishes exact perturbed hidden state into a fresh generation. Checkpoint
mode refuses without the helpers. The standalone E2 CPU proof runs the same
windowed API on a tensor fixture with the real boundary owner.
"""
from __future__ import annotations
import copy
import json
import os
from contextlib import ExitStack
from pathlib import Path
from replay_progress import DEFAULT_STALL_SECONDS, ReplayProgress, require_stall_seconds, supervise


def _isolate(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _isolate(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_isolate(item) for item in value)
    if isinstance(value, list):
        return [_isolate(item) for item in value]
    return copy.deepcopy(value)


def require_helpers():
    """Confirmed E1 helpers in band_continuation.py. Raises when absent.

    Checkpoint mode has no silent fallback: without the exact helpers the
    run refuses instead of replaying the full path.
    """
    try:
        from band_continuation import (plan_windows, commit_stream_window, load_window_checkpoint,
            resume_stream_window, next_window_dependencies, checkpoint_path,
            check_roster_entry, check_source)
    except ImportError:
        raise RuntimeError("Multi-stream checkpoint mode needs band_continuation.py after parent integration")
    return {"plan_windows": plan_windows, "commit_stream_window": commit_stream_window,
        "load_window_checkpoint": load_window_checkpoint, "resume_stream_window": resume_stream_window,
        "next_window_dependencies": next_window_dependencies, "checkpoint_path": checkpoint_path,
        "check_roster_entry": check_roster_entry, "check_source": check_source}


def prepare_streams(renders, rows, manifest_streams, *, energy_path, cohort):
    """Validate roster, select units and resolve actual options per stream."""
    from band_replay import class_kinds, resolve_replay_options, check_stream_family, measured_energy
    prepared = {}
    for stream in manifest_streams:
        kinds = class_kinds(stream["class"])
        selected = [row for row in rows if row["kind"] in kinds]
        if not selected:
            raise ValueError("The replay class has no actual units in this band: " + stream["stream_id"])
        replay_options = resolve_replay_options(renders, selected, stream["weight_source"])
        for row in selected:
            check_stream_family(replay_options[row["qname"]], stream["kind"])
        energies = measured_energy(energy_path, selected, stream["weight_source"], cohort,
            stream["kind"], stream["amplitude"], replay_options=replay_options)
        prepared[stream["stream_id"]] = {"spec": stream, "selected": selected,
            "replay_options": replay_options, "energies": energies,
            "records": [], "injections": [], "empty_lease_telemetry": []}
    return prepared


def stream_specs_for_layer(prepared_entry, layer, *, runner, start, stop, decoded_by_name):
    """Build this stream's independent UnitSpec list and decoded weights."""
    from menu_contract import executed_module
    from mixed_injection import UnitSpec, StaticScale
    spec = prepared_entry["spec"]
    replay_options = prepared_entry["replay_options"]
    specs, decoded = [], {}
    if start <= layer < stop and not spec["null_replay"]:
        for row in prepared_entry["selected"]:
            if row["layer"] != layer:
                continue
            option = replay_options[row["qname"]]
            expected = "T16" if spec["kind"] == "W_T16" else "T4" if spec["kind"] in ("W_T4", "A4", "joint_T4_A4") else "T8"
            scale = StaticScale(**option["scale"]) if expected == "T4" else None
            module = executed_module(runner.model, runner.profile, row) if row["kind"] in {"attention", "lm_head"} or expected == "T16" else None
            specs.append(UnitSpec(row["qname"], row["kind"], row["role"], row["expert"],
                "BF16" if expected == "T16" else expected, option["contract"], scale, module, option.get("tp_splits")))
            if spec["kind"].startswith("W_") or spec["kind"] == "joint_T4_A4":
                if row["qname"] not in decoded_by_name:
                    raise ValueError("The weight stream lacks its resident cache-owned plane")
                decoded[row["qname"]] = decoded_by_name[row["qname"]]
    return specs, decoded


def _read_back_hidden(owner, refs, *, guard):
    from frontier_replay import _read_back_hidden as bounded
    return bounded(owner, refs, guard=guard)


def _forward_inputs(clean_states, sample):
    from frontier_replay import move_state
    return {key: move_state(clean_states[sample][key], "cpu") for key in (
        "input_ids", "position_ids", "position_embeddings", "attention_mask")}


def run_band_stream_window(runner, args, cohort, inputs, roster, prepared, *, band_start, band_stop,
                           replay_start, replay_stop, window_start, window_stop, window_layers,
                           checkpoint_dir, source, teacher_digest, input_digest, measure_hook,
                           on_sample, check, progress, resume=True, teacher_receipt_sha256=None):
    """Execute one declared window for every roster stream with one install per layer.

    Each stream keeps its own isolated hidden state and pass state from the same
    captured input cohort; no stream reuses another stream's mutable state. Each
    stream replays through its own fresh working generation plus the borrowed
    start owner. The first window starts clean; a later window resumes its exact
    prerequisite checkpoint, never clean state. The completed window commits one
    E1 checkpoint across all roster streams. Amplitudes 1, 2 and 4 are recorded;
    the accepted pair fit stays with the gain reducer. measure_hook(entry, layer,
    forward, replay_state) drives one stream's perturbation lease over the ordered
    inputs. on_sample fires for each stream sample at the replay stop layer.
    """
    import torch
    from bounded_profile import capture_item
    from frontier_replay import load_frontier, move_state
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts, StreamedForwardBoundaries
    from footprint import FootprintRecorder
    helpers = require_helpers()
    receipt, clean_states, refs, samples = load_frontier(args, cohort, inputs)
    if receipt["layer_start"] != replay_start:
        raise ValueError("The replay must start at the actual borrowed starting boundary")
    if not replay_start < replay_stop <= 45:
        raise ValueError("The replay stop must lie inside the actual source graph")
    if receipt["layer_stop"] < band_stop:
        raise ValueError("The clean receipt must cover the perturbed band")
    roster = [helpers["check_roster_entry"](entry) for entry in roster]
    identifiers = [entry["stream_id"] for entry in roster]
    if not identifiers or set(prepared) != set(identifiers):
        raise ValueError("Prepared streams must cover exactly the roster")
    source = helpers["check_source"](source)
    if source["source_window"] != [window_start, window_stop]:
        raise ValueError("The source window must equal the one executed window")
    plan = helpers["plan_windows"](replay_start, replay_stop, window_layers=window_layers)
    if {"window_start": window_start, "window_stop": window_stop} not in plan:
        raise ValueError("The declared window leaves its replay plan")
    directory = Path(checkpoint_dir)
    incoming_owner = receipt["start_reference_owner"]
    footprint = FootprintRecorder()
    candidate = helpers["checkpoint_path"](directory, window_start, window_stop)
    if candidate.exists():
        raise ValueError("A committed window blocks its replay range")

    def expected_for(started, stopped, window_source):
        return {"band_start": band_start, "band_stop": band_stop, "replay_start": replay_start,
                "replay_stop": replay_stop, "next_layer": stopped,
                "window_start": started, "window_stop": stopped,
                "window_layers": window_layers, "cohort": cohort, "stream_roster": roster,
                "source": window_source, "teacher_digest": teacher_digest, "input_digest": input_digest,
                "teacher_receipt_sha256": teacher_receipt_sha256, "perturbed": True, "streams": {sid: {"sample_ids": list(samples)} for sid in identifiers}}
    with ExitStack() as stack:
        borrowed = stack.enter_context(StreamedBoundaryArtifacts(incoming_owner["boundary_config"]))
        borrowed.attach(incoming_owner["session"], n_probes=0)
        rollings = {}
        for sid in identifiers:
            config = dict(receipt["boundary_config"],
                          directory=str(args.boundary_dir / ("energy-working-" + sid)))
            owner = stack.enter_context(StreamedBoundaryArtifacts(config))
            owner.bind({"cohort": cohort, "layer_range": [replay_start, replay_stop]}, n_probes=0,
                       check_memory=check, published=False)
            rollings[sid] = owner
        replay_state = {"borrowed_owner": borrowed, "working_owners": rollings, "current": {}}
        continent = {sid: {"refs": dict(refs), "owned": False,
                           "states": {sample: _isolate(clean_states[sample]) for sample in samples}}
                     for sid in identifiers}
        resumed_from, installed = None, []
        from band_continuation import TELEMETRY_FIELDS
        prior_telemetry = {sid: {key: [] for key in TELEMETRY_FIELDS} for sid in identifiers}
        telemetry_marks = {sid: {key: len(prepared[sid].setdefault(key, [])) for key in TELEMETRY_FIELDS}
                           for sid in identifiers}
        if window_start > replay_start:
            if not resume:
                raise ValueError("A later window needs its prerequisite checkpoint")
            previous = next(window for window in plan if window["window_stop"] == window_start)
            prerequisite = helpers["checkpoint_path"](directory, previous["window_start"],
                                                       previous["window_stop"])
            if not prerequisite.exists():
                raise ValueError("The checkpoint chain leaves a window gap")
            prior_source = dict(source, source_window=[previous["window_start"], previous["window_stop"]])
            checkpoint = helpers["load_window_checkpoint"](
                prerequisite, expected=expected_for(previous["window_start"], previous["window_stop"],
                                                    prior_source), guard=check)
            for sid in identifiers:
                restored = helpers["resume_stream_window"](
                    checkpoint, sid, window_start=previous["window_start"],
                    window_stop=previous["window_stop"], guard=check)
                from band_continuation import validate_forward_state
                for sample, ids in zip(samples, inputs):
                    validate_forward_state(restored["forward"][sample], input_ids=ids,
                        expected=_forward_inputs(clean_states, sample))
                continent[sid]["states"] = {
                    sample: {**restored["forward"][sample],
                             "pass_state": restored["pass_state"][sample]} for sample in samples}
                continent[sid]["refs"] = {
                    sample: rollings[sid].write(restored["hidden"][sample], batch_index=sample,
                                                boundary_index=previous["window_stop"]) for sample in samples}
                continent[sid]["owned"] = True
                if restored["telemetry"] is None:
                    raise ValueError("The prerequisite checkpoint lacks its lease telemetry: " + sid)
                prior_telemetry[sid] = restored["telemetry"]
            resumed_from = str(prerequisite)
        for layer in range(window_start, window_stop):
            check("clean frontier source prefetch")
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=True, prefetch_following=False)
            installed.append(layer)
            if layer + 1 < window_stop:
                runner.context.schedule_prefetch(layer + 1)
            output_refs = {sid: {} for sid in identifiers}
            counts = {sid: 0 for sid in identifiers}
            entry_refs = {sid: dict(continent[sid]["refs"]) for sid in identifiers}
            entry_owned = {sid: continent[sid]["owned"] for sid in identifiers}

            def make_forward(sid):
                def forward(ids):
                    if counts[sid] >= len(samples):
                        raise ValueError("Multi-stream replay repeated a sequence")
                    sample = samples[counts[sid]]
                    if not torch.equal(ids, inputs[counts[sid]]):
                        raise ValueError("Multi-stream replay changed ordered actual tokens")
                    row = move_state(continent[sid]["states"][sample], runner.device)
                    pass_state = row["pass_state"]
                    batch = StreamedForwardBoundaries(row["input_ids"], row["position_ids"],
                        row["position_embeddings"], row["attention_mask"], [], None)
                    owner = borrowed if layer == replay_start and not entry_owned[sid] else rollings[sid]
                    with capture_item():
                        with owner.prefetch([entry_refs[sid][sample]]) as resident:
                            hidden = _isolate(owner.get(resident, entry_refs[sid][sample]).to(
                                runner.device, dtype=runner.dtype))
                            check("resident perturbed call")
                            with torch.inference_mode():
                                output = runner.isolated_layer(batch, layer, hidden, pass_state=pass_state)
                            output_refs[sid][sample] = rollings[sid].write(output, batch_index=sample,
                                boundary_index=layer + 1)
                            if layer + 1 == replay_stop:
                                on_sample(sid, sample, batch, output, pass_state)
                            continent[sid]["states"][sample]["pass_state"] = move_state(pass_state, "cpu")
                            del hidden, output
                    counts[sid] += 1
                    unit = "complete teacher score %d" if layer + 1 == replay_stop else "complete band sequence %d"
                    progress("layer-%02d-stream-%s" % (layer, sid), unit % sample)
                return forward

            try:
                for sid in identifiers:
                    measure_hook(prepared[sid], layer, make_forward(sid), replay_state)
                    if counts[sid] != len(samples):
                        raise ValueError("Multi-stream replay omitted actual heldout sequences: " + sid)
                    progress("layer-%02d-stream-%s" % (layer, sid), "complete band replay layer stream")
            finally:
                runner.context.unload(layer)
            for sid in identifiers:
                if entry_owned[sid]:
                    for reference in entry_refs[sid].values():
                        rollings[sid].retire(reference)
                continent[sid]["refs"], continent[sid]["owned"] = output_refs[sid], True
        footprint.record("window-%02d-%02d-complete" % (window_start, window_stop),
                         state={"streams": sorted(identifiers)})
        hidden = {sid: _read_back_hidden(rollings[sid], continent[sid]["refs"], guard=check)
                  for sid in identifiers}
        telemetry = {sid: {key: prior_telemetry[sid][key] + prepared[sid][key][telemetry_marks[sid][key]:]
                           for key in TELEMETRY_FIELDS} for sid in identifiers}
        commit_streams = {sid: {"sample_ids": list(samples), "hidden": hidden[sid],
                                "pass_state": {sample: continent[sid]["states"][sample]["pass_state"]
                                               for sample in samples},
                                "forward": {sample: _forward_inputs(clean_states, sample)
                                            for sample in samples},
                                "references": dict(continent[sid]["refs"]),
                                "owned_boundary_generation": dict(rollings[sid].session),
                                "telemetry": telemetry[sid]}
                          for sid in identifiers}
        checkpoint = helpers["commit_stream_window"](
            candidate, band_start=band_start, band_stop=band_stop, replay_start=replay_start,
            replay_stop=replay_stop, next_layer=window_stop, window_start=window_start,
            window_stop=window_stop, window_layers=window_layers, cohort=cohort, stream_roster=roster,
            source=source, teacher_digest=teacher_digest, input_digest=input_digest,
            streams=commit_streams, guard=check, teacher_receipt_sha256=teacher_receipt_sha256)
        final_hidden = hidden
        for sid in identifiers:
            for key in TELEMETRY_FIELDS:
                prepared[sid][key][:] = checkpoint["streams"][sid]["replay_telemetry"][key]
    return {"schema": receipt["schema"], "layer_range": [replay_start, replay_stop],
            "band_range": [band_start, band_stop], "window": {"window_start": window_start,
            "window_stop": window_stop}, "start_generation": incoming_owner["session"],
            "stop_generation": receipt["session"], "foreign_entries_retired": 0,
            "sample_ids": list(samples), "checkpoint_path": str(candidate), "checkpoint": checkpoint,
            "resumed_from": resumed_from, "final_hidden": final_hidden, "installed_layers": installed,
            "multi_stream": True, "streams": sorted(identifiers), "footprint": footprint.finish(),
            "next": helpers["next_window_dependencies"](checkpoint, checkpoint_file=candidate)}


def production_measure(entry, layer, forward, replay_state, *, ctx):
    """Drive one stream's real injection lease over the ordered inputs."""
    from contextlib import nullcontext
    from mixed_injection import inject_layer
    from menu_contract import unit_weight
    check, runner, renders = ctx["check"], ctx["runner"], ctx["renders"]
    start, stop, inputs = ctx["start"], ctx["stop"], ctx["inputs"]
    spec = entry["spec"]
    check("band layer")
    weight = spec["kind"].startswith("W_") or spec["kind"] == "joint_T4_A4"
    selected = [row for row in entry["selected"] if row["layer"] == layer]
    if weight and not spec["null_replay"] and start <= layer < stop:
        sources = {row["qname"]: unit_weight(runner, row) for row in selected}
        options = {row["qname"]: [entry["replay_options"][row["qname"]]] for row in selected}
        render_context = renders.retained_decoded_window(sources, options, device=runner.device,
            max_resident_bytes=ctx["rendered_budget_bytes"], guard=check)
    else:
        render_context = nullcontext(({}, {"owner": "existing streamed source cache", "actual_resident_bytes": 0}))
    with render_context as (weights, residency):
        decoded_by_name = {name: value["options"][0][1] for name, value in weights.items()}
        ctx["rendered_residency"].append({"layer": layer, "stream_id": spec["stream_id"], **residency})
        specs, decoded = stream_specs_for_layer(entry, layer, runner=runner,
            start=start, stop=stop, decoded_by_name=decoded_by_name)
        lease = inject_layer(runner.layers[layer], specs, decoded, amplitude=spec["amplitude"],
            weight=weight, activation=not spec["kind"].startswith("W_"),
            allow_empty=spec["null_replay"]) if specs or spec["null_replay"] and start <= layer < stop else nullcontext()
        try:
            from band_replay import forward_with_lease
            forward_with_lease(layer, lease, inputs, forward, check,
                null_replay=spec["null_replay"], injections=entry["injections"],
                empty_lease_telemetry=entry["empty_lease_telemetry"])
        finally:
            decoded.clear()
            decoded_by_name.clear()


def production_on_sample(runner, teachers, prepared, check, sid, sample, batch, output, pass_state):
    """Score one stream sample against the global teacher at the replay stop layer."""
    import torch
    from band_replay import score_teacher, sequence_record
    with torch.inference_mode():
        logits = runner.tail_logits(batch, output)
    check("band teacher score")
    positions = score_teacher(logits, teachers[sample], check)
    entry = prepared[sid]
    spec = entry["spec"]
    record = sequence_record(sample, positions, null_replay=spec["null_replay"],
        energies=None if spec["null_replay"] else entry["energies"])
    journal = entry.get("score_stream")
    if journal is not None:
        journal.write(json.dumps(record, allow_nan=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
    entry["records"].append(record)


TEACHER_STOP = 45


def cli_parser():
    """The actual multi-stream command parser shared by main and its proof."""
    from stage1a_energy import parser
    p = parser()
    p.add_argument("--band", required=True, help="The perturbed layer range")
    p.add_argument("--stream-manifest", type=Path, required=True, help="Adopted explicit roster pact.stream_roster.v1")
    p.add_argument("--energy-rows", type=Path, required=True)
    p.add_argument("--teacher-frontier", type=Path, required=True)
    p.add_argument("--teacher-content", type=Path, required=True,
        help="Teacher validation proof pact.prefixed_teacher_validation.v1 with one SHA-256 per array")
    p.add_argument("--teacher-content-sha256", required=True,
        help="Declared SHA-256 of the teacher validation proof bytes")
    p.add_argument("--stall-seconds", type=float, default=DEFAULT_STALL_SECONDS,
        help="Maximum silence between completed steps, in seconds. No total duration limit.")
    p.add_argument("--window-layers", type=int, default=None,
        help="Committed window stride in layers. Defaults to one window for the whole replay.")
    p.add_argument("--window-start", type=int, required=True)
    p.add_argument("--window-stop", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True,
        help="E1 window checkpoint directory. The admitted call commits its one declared window.")
    return p


def lease_scope(args):
    """Mirror the original strict input lease contract for every device."""
    return bool((args.device == "cuda" or args.require_staged_inputs) and not args.prepare_readset)


def digest_staged(path):
    """Digest bytes through the existing staged reader, never a direct read."""
    import hashlib
    from g3_residency import read_file
    return hashlib.sha256(bytes(read_file(Path(path)))).hexdigest()


def resolve_cli_run(args, receipt_doc, receipt_bytes, roster_doc):
    """Admit exactly one declared window from actual CLI arguments.

    Returns the band, the replay extent to the global teacher stop, the one
    window, its plan and the receipt generation bindings. Any other shape
    raises before any source read.
    """
    import hashlib
    from band_replay import validate_stream_manifest
    from band_continuation import plan_windows as plan_replay_windows
    band_start, band_stop = map(int, args.band.split(":"))
    if not 0 <= band_start < band_stop <= 45 or args.layer_range != args.band:
        raise ValueError("The replay needs its exact band and fixed prefixed64 cohort")
    if args.input_contract != "prefixed_514":
        raise ValueError("The replay needs its fixed prefixed64 cohort")
    manifest_streams = validate_stream_manifest(roster_doc)
    if roster_doc.get("band") != args.band:
        raise ValueError("The roster band differs from the replay band")
    for stream in manifest_streams:
        if not isinstance(stream["weight_source"], str) or not stream["weight_source"]:
            raise ValueError("A roster stream lacks its actual weight source")
        if args.weight_source and stream["weight_source"] not in args.weight_source:
            raise ValueError("A roster weight source differs from the declared sources")
    replay_start, receipt_stop = receipt_doc["layer_start"], receipt_doc["layer_stop"]
    replay_stop = TEACHER_STOP
    if not replay_start <= band_start < band_stop <= replay_stop:
        raise ValueError("The perturbed band must sit inside its replay extent")
    if not receipt_stop >= band_stop:
        raise ValueError("The clean receipt must cover the perturbed band")
    window_layers = args.window_layers or (replay_stop - replay_start)
    plan = plan_replay_windows(replay_start, replay_stop, window_layers=window_layers)
    if {"window_start": args.window_start, "window_stop": args.window_stop} not in plan:
        raise ValueError("The CLI window must sit on the planned window grid")
    require_stall_seconds(args.stall_seconds)
    generations = {"receipt_sha256": hashlib.sha256(bytes(receipt_bytes)).hexdigest(),
                   "source_window": [receipt_doc["layer_start"], receipt_doc["layer_stop"]],
                   "start_generation": receipt_doc["start_reference_owner"]["session"],
                   "stop_generation": receipt_doc["session"]}
    return {"band_start": band_start, "band_stop": band_stop, "replay_start": replay_start,
            "replay_stop": replay_stop, "window_start": args.window_start,
            "window_stop": args.window_stop, "window_layers": window_layers, "plan": plan,
            "generations": generations, "manifest_streams": manifest_streams}


def main():
    args = cli_parser().parse_args()
    require_stall_seconds(args.stall_seconds)
    if args.dry_run_cpu or args.prepare_readset:
        return run_replay(args)
    return supervise(lambda notify: run_replay(args, notify), args.stall_seconds)


def run_replay(args, notify=lambda record: None):
    from stage1a_energy import setup, actual_cpu, available, guard
    from band_replay import load_teacher, publish_stream_document
    import time
    from strict_leases import strict_input_leases
    with strict_input_leases(lease_scope(args)) as leases:
        from g3_residency import read_file
        receipt_bytes = bytes(read_file(args.boundaries))
        receipt_doc = json.loads(receipt_bytes.decode())
        document = json.loads(bytes(read_file(args.stream_manifest)))
        run = resolve_cli_run(args, receipt_doc, receipt_bytes, document)
        band_start, band_stop = run["band_start"], run["band_stop"]
        replay_start, replay_stop = run["replay_start"], run["replay_stop"]
        window_start, window_stop = run["window_start"], run["window_stop"]
        window_layers, plan = run["window_layers"], run["plan"]
        manifest_streams, generations = run["manifest_streams"], run["generations"]
        renders, cohort, inputs, rows = setup(args)
        if args.prepare_readset:
            from build_manifest import prepare
            prepare(renders, cohort, rows, args)
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.run_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        from band_continuation import bind_teacher_content, teacher_bindings
        teacher_receipt = bytes(read_file(args.teacher_frontier))
        teacher_content = bind_teacher_content(teacher_receipt, read_file(args.teacher_content),
            content_sha256=args.teacher_content_sha256, cohort=cohort)
        teachers = load_teacher(args.teacher_frontier, cohort, inputs, small_read=args.dry_run_cpu,
                                content=teacher_content)
        prepared = prepare_streams(renders, rows, manifest_streams, energy_path=args.energy_rows, cohort=cohort)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run_cpu:
            with args.output.open("w") as stream:
                proof = actual_cpu(renders, cohort, inputs, rows, args, stream)
            proof.update(band=args.band, teacher_header_read=True,
                streams=sorted(prepared), stream_manifest=document["streams"], input_leases=leases)
            args.run_manifest_out.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"D38_CPU_passed": True, "full_band_measured": False,
                "streams": sorted(prepared)}), flush=True)
            return
        import torch
        from energy_inputs import source_reader
        from prismaquant.model_profiles import detect_profile
        from prismaquant.cost_streaming import build_streamed_causal_lm
        if args.startup_need_gib is None or args.startup_need_gib < 3 or available() < args.startup_need_gib * 2**30:
            raise MemoryError("D30 measured replay footprint plus margin is unavailable")
        if type(args.rendered_resident_budget_gib) is not int or args.rendered_resident_budget_gib <= 0:
            raise ValueError("The GPU replay path needs its admitted rendered-resident byte budget")
        began = time.monotonic()
        progress = ReplayProgress(
            args.output_dir / ("progress-%02d-%02d.jsonl" % (window_start, window_stop)),
            args.stall_seconds, notify=notify)

        def check(label):
            guard(label)
            progress.check(label)

        teacher = teacher_bindings(teacher_receipt, teacher_content)
        ctx = {"check": check, "runner": None, "renders": renders,
               "start": band_start, "stop": band_stop, "inputs": inputs,
               "rendered_budget_bytes": args.rendered_resident_budget_gib * 2**30, "rendered_residency": []}

        def measure_hook(entry, layer, forward, replay_state):
            production_measure(entry, layer, forward, replay_state, ctx=ctx)

        def on_sample(sid, sample, batch, output, pass_state):
            production_on_sample(ctx["runner"], teachers, prepared, check,
                                 sid, sample, batch, output, pass_state)

        window_source = dict(generations, source_window=[window_start, window_stop])
        with source_reader(args) as reads, ExitStack() as scores:
            for sid, entry in prepared.items():
                journal = args.output_dir / ("scores-%02d-%02d-%s.jsonl" % (window_start, window_stop, sid))
                entry["score_stream"] = scores.enter_context(journal.open("a"))
            runner = build_streamed_causal_lm(str(args.source_root), device=torch.device("cuda"), dtype=torch.bfloat16,
                profile=detect_profile(str(args.source_root)), offload_folder=str(args.boundary_dir / "offload"),
                max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1, cache_headroom_gb=3,
                prefetch_min_available_gb=2, require_prefetched_residency=True, attn_implementation="eager")
            runner.model.eval()
            ctx["runner"] = runner
            from mixed_injection import interceptable_packed_experts
            try:
                with interceptable_packed_experts(runner.model) as experts_backends:
                    call = run_band_stream_window(
                        runner, args, cohort, inputs, manifest_streams, prepared,
                        band_start=band_start, band_stop=band_stop, replay_start=replay_start,
                        replay_stop=replay_stop, window_start=window_start,
                        window_stop=window_stop, window_layers=window_layers,
                        checkpoint_dir=args.checkpoint_dir, source=window_source,
                        teacher_digest=teacher["teacher_digest"], input_digest=cohort["token_sha256"],
                        teacher_receipt_sha256=teacher["teacher_receipt_sha256"],
                        measure_hook=measure_hook, on_sample=on_sample, check=check,
                        progress=progress)
            finally:
                runner.shutdown()
                ctx["runner"] = None
            source_evidence = dict(reads.stats)
        evidence = {key: value for key, value in call.items() if key != "final_hidden"}
        evidence["experts_backend"] = {"pinned": "eager", "previous": experts_backends}
        evidence["windows"] = plan
        evidence["checkpoints"] = [call["checkpoint_path"]]
        outputs = {}
        if window_stop < replay_stop:
            args.run_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.run_manifest_out.write_text(json.dumps({"outputs": outputs, "streams": sorted(outputs),
                "windows": plan, "checkpoints": evidence["checkpoints"],
                "next": call["next"], "complete": False,
                "elapsed_seconds": time.monotonic() - began}, indent=2) + "\n")
            print(json.dumps({"outputs": outputs, "streams": sorted(outputs),
                "checkpoints": evidence["checkpoints"], "full_band_measured": False}), flush=True)
            return
        for sid, entry in prepared.items():
            spec = entry["spec"]
            if len(entry["records"]) != 64 or {row["sample_id"] for row in entry["records"]} != set(range(384, 448)):
                raise ValueError("The band replay omits or repeats actual samples: " + sid)
            result = {"schema": "pact.band_stream.v1", "cohort": cohort,
                "input_token_ids": torch.cat(inputs).tolist(),
                "band_start": band_start, "band_stop": band_stop, "class": spec["class"], "kind": spec["kind"],
                "amplitude": spec["amplitude"], "null_replay": spec["null_replay"],
                "stream_id": spec["stream_id"], "stream_roster": document["streams"],
                "per_sequence": sorted(entry["records"], key=lambda row: row["sample_id"]),
                "rendered_residency": ctx["rendered_residency"],
                "replay": evidence, "injections": entry["injections"],
                "empty_lease_telemetry":entry["empty_lease_telemetry"], "source_reads": source_evidence,
                "input_leases": leases,
                "elapsed_seconds": time.monotonic() - began,
                "limit": "Batched multi-stream measurement. Amplitudes 1/2/4 are recorded; the accepted pair fit stays with the gain reducer."}
            out_path = args.output_dir / (sid + ".json")
            publish_stream_document(out_path, result)
            outputs[sid] = str(out_path)
        args.run_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.run_manifest_out.write_text(json.dumps({"outputs": outputs, "streams": sorted(outputs),
            "windows": plan, "checkpoints": evidence["checkpoints"],
            "next": call["next"], "complete": True,
            "elapsed_seconds": time.monotonic() - began}, indent=2) + "\n")
        progress("energy", "durable complete band streams")
        print(json.dumps({"outputs": outputs, "streams": sorted(outputs),
            "full_band_measured": True}), flush=True)


if __name__ == "__main__":
    main()
