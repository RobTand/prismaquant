"""One bounded measured band stream on the fixed clean prefixed cohort."""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import time


def apply_teacher_content(entries, state_bytes, content):
    """Bind actual teacher bytes to the content digests of one validated proof.

    The batch-state bytes must equal their digest now. Each teacher entry
    takes its array digest, and score_teacher refuses bytes that differ.
    """
    if hashlib.sha256(bytes(state_bytes)).hexdigest() != content["batch_state_sha256"]:
        raise ValueError("The teacher batch-state bytes differ from their content digest")
    if set(entries) != set(content["arrays"]):
        raise ValueError("The teacher content digests cover another sample population")
    for sample, entry in entries.items():
        entry["sha256"] = content["arrays"][sample]
    return entries


def load_teacher(path, cohort, inputs, *, small_read=False, content=None):
    import numpy as np
    from g3_residency import read_file
    document = json.loads(read_file(path))
    if document.get("schema") != "pact.prefixed_clean_frontier.v1" or document.get("layer_stop") != 45 or document.get("dry_run"):
        raise ValueError("The band teacher needs a completed clean source frontier at layer45")
    for key, value in cohort.items():
        if key in document["cohort"] and document["cohort"][key] != value:
            raise ValueError("The band teacher cohort differs: " + key)
    entries = {row["sample_id"]: row for row in document["teacher_arrays"]}
    if len(entries) != 64 or set(entries) != set(range(384,448)):
        raise ValueError("The teacher omits or repeats held-out samples")
    state = read_file(document["batch_state_path"])
    import torch
    states = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
    for sample, ids in zip(range(384,448),inputs):
        if not torch.equal(states[sample]["input_ids"], ids):
            raise ValueError("The actual teacher and stream tokens differ")
    for row in entries.values():
        if row["shape"] != [511,154880] or row["dtype"] != "float32":
            raise ValueError("The teacher has wrong scored geometry or dtype")
    if content is not None:
        apply_teacher_content(entries, state, content)
    if small_read:
        first = entries[384]
        from g3_lib import read_range
        raw = read_range(first["path"],0,min(first["bytes"],4096))
        if not raw:
            raise ValueError("The real teacher header is empty")
    return entries


def measured_energy(path, selected, source_name, cohort, kind, amplitude, *, replay_options):
    from g3_residency import read_file
    field = "E_WA_sum" if kind == "joint_T4_A4" else "E_W_sum" if kind.startswith("W_") else "E_A_sum"
    if amplitude != 1:
        field += "_amp" + str(amplitude)
    units = {row["qname"]: row for row in selected}
    names = set(units)
    if set(replay_options) != names or any(option["name"] != source_name for option in replay_options.values()):
        raise ValueError("The resolved replay options do not cover the selected unit source")
    seen, sums = set(), {sample:0.0 for sample in range(384,448)}
    for line in bytes(read_file(path)).decode().splitlines():
        row = json.loads(line)
        if row["qname"] not in names or row["weight_source"] != source_name:
            continue
        if row.get("dry_run_cpu") or row.get("input_token_sha256") != cohort["token_sha256"]:
            raise ValueError("The stream lacks comparable full clean energy")
        if row.get("prefix_rows_enter_local_energies") is not False or row.get("global_original_tokens") != 32768:
            raise ValueError("The local energy token scope differs")
        option = replay_options[row["qname"]]
        for energy_field, option_field in (("family", "family"), ("q256", "q256"),
                                           ("activation_contract", "contract")):
            if row.get(energy_field) != option[option_field]:
                raise ValueError("The actual energy and replay configuration differ: " + energy_field)
        expected_tp = option.get("tp_splits", 2 if units[row["qname"]]["role"] == "down_proj" else 1)
        if type(row.get("tp_splits")) is not int or row["tp_splits"] != expected_tp:
            raise ValueError("The actual energy and replay TP split differ")
        expected_scale = (option.get("scale") or {}).get("effective")
        actual_scale = (row.get("scale") or {}).get("effective")
        if actual_scale != expected_scale or isinstance(actual_scale, bool):
            raise ValueError("The actual energy and replay effective static scale differ")
        key = row["qname"],row["sample_id"]
        if key in seen or row["sample_id"] not in sums:
            raise ValueError("The local energy repeats or changes a sample")
        value = row[field]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError("The measured stream energy is invalid")
        sums[row["sample_id"]] += value
        seen.add(key)
    if seen != {(name,sample) for name in names for sample in sums}:
        raise ValueError("The stream lacks exact per-unit energy coverage")
    return sums


def score_teacher(logits, entry, guard):
    import numpy as np
    import torch
    from g3_residency import read_file
    raw = read_file(entry["path"])
    if len(raw) != entry["bytes"]:
        raise ValueError("The teacher file length differs")
    if "sha256" in entry and hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("The teacher bytes differ from their content digest")
    teacher = np.load(io.BytesIO(raw),allow_pickle=False)
    if teacher.shape != (511,154880) or teacher.dtype != np.float32:
        raise ValueError("The actual teacher scored plane differs")
    student = logits[0,2:-1]
    if tuple(student.shape) != teacher.shape:
        raise ValueError("The student scored token positions differ")
    positions = []
    with torch.inference_mode():
        for start in range(0,511,16):
            guard("teacher KL rows")
            a = torch.from_numpy(teacher[start:start+16]).to(student.device)
            b = student[start:start+16].float()
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError("The scored logits are not finite")
            logp, logq = a.log_softmax(-1), b.log_softmax(-1)
            values = (logp.exp() * (logp-logq)).sum(-1,dtype=torch.float64)
            positions.extend(values.cpu().tolist())
    return positions


STREAM_MANIFEST_SCHEMA = "pact.stream_roster.v1"
FROZEN_STREAM_FIELDS = ("stream_id", "class", "kind", "weight_source", "amplitude", "null_replay")
STREAM_CLASSES = ("routed", "nonrouted", "attention", "lm_head")
STREAM_KINDS = ("W_T4", "W_T8", "W_T16", "A4", "A8", "joint_T4_A4")


def class_kinds(stream_class):
    """Actual unit kinds served by one stream class."""
    if stream_class == "routed":
        return {"routed"}
    if stream_class == "nonrouted":
        return {"shared", "dense"}
    return {stream_class}


def resolve_replay_options(renders, selected, weight_source):
    """Resolve the exact declared option for every selected unit."""
    resolved = {}
    for row in selected:
        matches = [option for option in renders.options(row) if option["name"] == weight_source]
        if len(matches) != 1:
            raise ValueError("The stream needs exactly one actual declared weight source for " + row["qname"])
        resolved[row["qname"]] = matches[0]
    return resolved


def options_for_roster(renders, row, streams):
    """Resolve only options declared for this unit's actual stream class."""
    resolved = {}
    for stream in streams:
        if row["kind"] not in class_kinds(stream["class"]):
            continue
        name = stream["weight_source"]
        if name not in resolved:
            option = resolve_replay_options(renders, [row], name)[row["qname"]]
            check_stream_family(option, stream["kind"])
            resolved[name] = option
    if not resolved:
        raise ValueError("The stream roster declares no option for " + row["qname"])
    return list(resolved.values())


def check_stream_family(option, stream_kind):
    """Refuse a replay option whose anchor family differs from its stream kind."""
    expected = "T16" if stream_kind == "W_T16" else "T4" if stream_kind in ("W_T4", "A4", "joint_T4_A4") else "T8"
    if option["family"] != expected:
        raise ValueError("The stream does not match its actual anchor family")
    return expected


def sequence_record(sample, kl_positions, *, null_replay, energies=None):
    """One per-sequence stream record. Null replays carry no energy value."""
    record = {"sample_id": sample, "KL_mean": sum(kl_positions) / 511, "per_position_KL": kl_positions}
    if null_replay:
        if energies is not None:
            raise ValueError("A null replay must not carry an actual energy")
        return record
    if energies is None or sample not in energies:
        raise ValueError("An amplitude stream needs its actual per-sample energy")
    return {"sample_id": sample, "E_sum": energies[sample], **record}


def validate_stream_manifest(document):
    """Validate an adopted explicit stream roster. Returns the stream list."""
    if document.get("schema") != STREAM_MANIFEST_SCHEMA:
        raise ValueError("The stream roster schema differs")
    band = document.get("band")
    streams = document.get("streams")
    if not isinstance(band, str) or not isinstance(streams, list) or not streams:
        raise ValueError("The stream roster needs its band and a nonempty stream list")
    seen_ids, seen_configs = set(), set()
    null_keys = set()
    for stream in streams:
        for field in FROZEN_STREAM_FIELDS:
            if field not in stream:
                raise ValueError("A roster stream omits frozen field " + field)
        if stream["class"] not in STREAM_CLASSES or stream["kind"] not in STREAM_KINDS:
            raise ValueError("A roster stream names an unknown class or kind")
        if stream["stream_id"] in seen_ids:
            raise ValueError("The roster repeats a stream id")
        seen_ids.add(stream["stream_id"])
        if stream["null_replay"]:
            if stream["amplitude"] != 1:
                raise ValueError("A null replay must use amplitude one as its placeholder")
        elif stream["amplitude"] not in (1, 2, 4):
            raise ValueError("A measured amplitude is duplicated or invalid")
        config = (stream["class"], stream["kind"], stream["weight_source"], stream["amplitude"], stream["null_replay"])
        if config in seen_configs:
            raise ValueError("The roster repeats a stream configuration")
        seen_configs.add(config)
        if stream["null_replay"]:
            if null_keys:
                raise ValueError("The clean null replay is duplicated for the band")
            null_keys.add(band)
    return streams


def forward_with_lease(layer, lease, inputs, forward, check, *,
                       null_replay, injections, empty_lease_telemetry):
    """Keep empty null leases separate from actual perturbation telemetry."""
    with lease as telemetry:
        if telemetry is not None:
            target = empty_lease_telemetry if null_replay else injections
            target.append({"layer": layer, **telemetry})
        for ids in inputs:
            check("band sequence")
            forward(ids)


def publish_stream_document(path, document):
    """Atomically publish one pact.band_stream.v1 document, fail-closed."""
    if document.get("schema") != "pact.band_stream.v1":
        raise ValueError("A completed measured band stream is required")
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(document, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    from stage1a_energy import parser, setup, actual_cpu, available, guard, progress
    p = parser()
    p.add_argument("--band", required=True, help="The perturbed layer range")
    p.add_argument("--stream-class", choices=("routed","nonrouted","attention","lm_head"),required=True)
    p.add_argument("--stream-kind", choices=("W_T4","W_T8","W_T16","A4","A8","joint_T4_A4"),required=True)
    p.add_argument("--amplitude", type=int,choices=(1,2,4),required=True)
    p.add_argument("--null-replay",action="store_true")
    p.add_argument("--energy-rows",type=Path,required=True)
    p.add_argument("--teacher-frontier",type=Path,required=True)
    p.add_argument("--deadline-seconds",type=int,default=1700)
    args = p.parse_args()
    start,stop = map(int,args.band.split(":"))
    if not (0 <= start < stop <= 45 or (start,stop) == (45,46)) or args.layer_range != args.band or args.input_contract != "prefixed_514":
        raise ValueError("The replay needs its exact band and fixed prefixed64 cohort")
    if not 0 < args.deadline_seconds <= 1700 or len(args.weight_source or []) != 1:
        raise ValueError("The replay needs one actual weight source and a bounded deadline")
    from strict_leases import strict_input_leases
    with strict_input_leases(args.device == "cuda") as leases:
        renders,cohort,inputs,rows = setup(args)
        kinds = class_kinds(args.stream_class)
        selected = [row for row in rows if row["kind"] in kinds]
        if not selected:
            raise ValueError("The replay class has no actual units in this band")
        replay_options = resolve_replay_options(renders, selected, args.weight_source[0])
        teachers = load_teacher(args.teacher_frontier,cohort,inputs,small_read=args.dry_run_cpu)
        energies = measured_energy(args.energy_rows,selected,args.weight_source[0],cohort,args.stream_kind,args.amplitude,
                                   replay_options=replay_options)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.dry_run_cpu:
            with args.output.open("w") as stream:
                proof = actual_cpu(renders,cohort,inputs,rows,args,stream)
            proof.update(band=args.band,teacher_header_read=True,stream_energy_samples=len(energies),input_leases=leases)
            args.run_manifest_out.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
            print(json.dumps({"D38_CPU_passed":True,"full_band_measured":False}),flush=True)
            return
        import torch
        from energy_inputs import source_reader
        from frontier_replay import run_frontier
        from menu_contract import selected_options,unit_weight,executed_module
        from mixed_injection import UnitSpec,StaticScale,inject_layer
        from prismaquant.model_profiles import detect_profile
        from prismaquant.cost_streaming import build_streamed_causal_lm
        from footprint import FootprintRecorder
        if args.startup_need_gib is None or args.startup_need_gib < 3 or available() < args.startup_need_gib*2**30:
            raise MemoryError("D30 measured replay footprint plus margin is unavailable")
        began = time.monotonic()
        def check(label):
            guard(label)
            if time.monotonic()-began >= args.deadline_seconds:
                raise TimeoutError("The band quantum reached its declared deadline")
        records, injections, empty_lease_telemetry = [], [], []
        footprint = FootprintRecorder()
        with source_reader(args) as reads:
            runner = build_streamed_causal_lm(str(args.source_root),device=torch.device("cuda"),dtype=torch.bfloat16,
                profile=detect_profile(str(args.source_root)),offload_folder=str(args.boundary_dir/"offload"),
                max_cache_slots=2,prefetch_workers=1,prefetch_lookahead=1,cache_headroom_gb=3,
                prefetch_min_available_gb=2,require_prefetched_residency=True,attn_implementation="eager")
            runner.model.eval()
            replay_current = {}
            def measure(layer,forward,replay_state):
                nonlocal replay_current
                replay_current = replay_state
                check("band layer")
                specs, decoded = [], {}
                if start <= layer < stop and not args.null_replay:
                    for row in selected:
                        if row["layer"] != layer:
                            continue
                        option = replay_options[row["qname"]]
                        expected_family = check_stream_family(option, args.stream_kind)
                        scale = StaticScale(**option["scale"]) if expected_family == "T4" else None
                        module = executed_module(runner.model,runner.profile,row) if row["kind"] in {"attention","lm_head"} or expected_family == "T16" else None
                        spec = UnitSpec(row["qname"],row["kind"],row["role"],row["expert"],"BF16" if expected_family == "T16" else expected_family,option["contract"],scale,module,option.get("tp_splits"))
                        specs.append(spec)
                        if args.stream_kind.startswith("W_") or args.stream_kind == "joint_T4_A4":
                            source = unit_weight(runner,row)
                            decoded[row["qname"]] = renders.decode(option,source,device=runner.device)
                footprint.record("layer-%02d-ready"%layer,runner=runner,weights={name:{"source":unit_weight(runner,next(row for row in selected if row["qname"] == name)),"options":[({},value)]} for name,value in decoded.items()},state=replay_state)
                lease = inject_layer(runner.model if layer == 45 else runner.layers[layer],specs,decoded,amplitude=args.amplitude,
                    weight=args.stream_kind.startswith("W_") or args.stream_kind == "joint_T4_A4",
                    activation=not args.stream_kind.startswith("W_"),allow_empty=args.null_replay) if specs or args.null_replay and start <= layer < stop else nullcontext()
                forward_with_lease(layer, lease, inputs, forward, check,
                    null_replay=args.null_replay, injections=injections,
                    empty_lease_telemetry=empty_lease_telemetry)
                footprint.record("layer-%02d-restored"%layer,runner=runner,state=replay_state)
                decoded.clear()
                progress("layer-%02d"%layer,"complete band replay layer")
            def scored_logits(sample,logits):
                check("band teacher score")
                positions = score_teacher(logits,teachers[sample],check)
                records.append(sequence_record(sample, positions, null_replay=args.null_replay,
                    energies=None if args.null_replay else energies))
                footprint.record("sample-%04d-score"%sample,runner=runner,state={"logits":logits,"replay":replay_current})
            def tail(sample,batch,hidden,state):
                with torch.inference_mode():
                    logits = runner.tail_logits(batch,hidden)
                scored_logits(sample,logits)
            try:
                if (start,stop) == (45,46):
                    from tail_replay import run_tail_frontier
                    replay = run_tail_frontier(runner,args,cohort,inputs,measure,check,on_logits=scored_logits)
                else:
                    from mixed_injection import interceptable_packed_experts
                    with interceptable_packed_experts(runner.model):
                        replay = run_frontier(runner,args,cohort,inputs,measure,check,stop_layer=45,on_tail=tail)
            finally:
                runner.shutdown()
            source_evidence = dict(reads.stats)
        if len(records) != 64 or {row["sample_id"] for row in records} != set(range(384,448)):
            raise ValueError("The band replay omits or repeats actual samples")
        result = {"schema":"pact.band_stream.v1","cohort":cohort,"input_token_ids":torch.cat(inputs).tolist(),
            "band_start":start,"band_stop":stop,"class":args.stream_class,"kind":args.stream_kind,
            "amplitude":args.amplitude,"null_replay":args.null_replay,"per_sequence":records,
            "replay":replay,"injections":injections,"empty_lease_telemetry":empty_lease_telemetry,
            "source_reads":source_evidence,"input_leases":leases,
            "footprint":footprint.finish(),"elapsed_seconds":time.monotonic()-began,
            "limit":"One measured stream. The complete observation reducer decides price eligibility."}
        publish_stream_document(args.output, result)
        args.run_manifest_out.write_text(json.dumps({"output":str(args.output),"samples":64,"elapsed_seconds":result["elapsed_seconds"]},indent=2)+"\n")
        progress("energy","durable complete band stream")
        print(json.dumps({"output":str(args.output),"samples":64,"full_band_measured":True}),flush=True)


if __name__ == "__main__":
    main()
