"""Build exact CPU proof and GPU readsets on the admitted CPU worker.

No local heavyweight plan/analysis. Called by the same entry point with
--prepare-readset, after argument/import/cohort preparation. Every read
phase is named for PrismaBuild progress. Source streaming, canonical
wires, complete static G groups and exact boundary inputs are declared.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


class ReadPlan:
    def __init__(self):
        self.entries, self.registry, self.phases = [], {}, []
        self.cumulative = 0

    def add(self, path, offset=0, size=None, sha=None):
        path = str(Path(path))
        size = os.stat(path).st_size if size is None else size
        key = (path, offset)
        if key in self.registry:
            old = self.entries[self.registry[key]]
            if old["bytes"] != size:
                raise ValueError("Contradictory exact read range")
            return self.registry[key]
        index = len(self.entries)
        self.entries.append({"path":path,"offset":offset,"bytes":size,"sha256":sha})
        self.registry[key] = index
        return index

    def phase(self, name, indices):
        indices = list(dict.fromkeys(indices))
        size = sum(self.entries[i]["bytes"] for i in indices)
        self.cumulative += size
        self.phases.append({"name":name,"entry_indices":indices,"bytes":size,"cumulative_bytes":self.cumulative})

    def finish(self, path, annotations):
        value = {"schema":"prismaquant.prismabuild.data_manifest.v2","mount_prefix":"/mnt/shared",
            "produced_by":{"tool":"pact stage1a existing-render readset"},"annotations":annotations,
            "entries":self.entries,"entry_count":len(self.entries),
            "total_bytes":sum(e["bytes"] for e in self.entries),
            "read_plan":{"phases":self.phases,"read_bytes":self.cumulative}}
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(value,indent=2)+"\n")
        return value


def metadata(plan,args,rows):
    refs = [plan.add(p) for p in (args.manifest,args.reference_manifest,args.draw,
        args.source_root/"config.json",args.source_root/"model.safetensors.index.json",
        args.a4_root/"config.json",args.a4_root/"model.safetensors.index.json",
        args.capture_root/"split-manifest.json")]
    for layer in sorted({r["layer"] for r in rows}):
        path = args.capture_root/"layers"/("L%03d"%layer)/"manifest.json"
        if path.exists():
            refs.append(plan.add(path))
    refs.extend(plan.add(path) for path in (args.units_file, args.added_inventory, args.option_manifest, args.t4_inventory) if path is not None)
    return refs


def option_ranges(plan, renders, option):
    from g3_lib import member_location
    if option.get("passthrough"):
        return []
    if "wire" in option:
        wire = option["wire"]
        return [plan.add(wire["path"],wire["offset"],wire["bytes"])]
    loc = option["location"]
    roots = {"a8":renders.args.a8_root,"t8r":renders.args.t8r_root,"exl3":renders.args.exl3_root}
    root = Path(roots[loc.get("root","a8")])
    if "ranges" in loc:
        return [plan.add(root/shard,offset,size) for shard,offset,size in loc["ranges"]]
    offset,size = member_location(loc)
    return [plan.add(root/loc["shard"],offset,size)]


def static_ranges(plan,renders,layers):
    refs = []
    for layer in layers:
        renders.static_layer(layer)
    for fact in renders.a4.reads:
        refs.append(plan.add(fact["path"],fact["offset"],fact["bytes"],fact["sha256"]))
    for shard,n in renders.a4.header_sizes.items():
        refs += [plan.add(renders.a4.root/shard,0,8),plan.add(renders.a4.root/shard,8,n)]
    return refs


def band_ranges(plan, args, *, small_read, cohort=None):
    """Declare actual band inputs through the existing read-plan owner."""
    if not getattr(args, "stream_manifest", None):
        return []
    from g3_residency import read_file
    from band_continuation import checkpoint_path, plan_windows
    refs = [plan.add(args.stream_manifest), plan.add(args.energy_rows), plan.add(args.teacher_frontier)]
    teacher = json.loads(read_file(args.teacher_frontier))
    content = None
    if getattr(args, "teacher_content", None):
        if cohort is None:
            raise ValueError("The teacher content binding needs the actual cohort")
        # The staged reader then verifies each teacher read against its content digest.
        from band_continuation import bind_teacher_content
        refs.append(plan.add(args.teacher_content, sha=args.teacher_content_sha256))
        content = bind_teacher_content(read_file(args.teacher_frontier), read_file(args.teacher_content),
            content_sha256=args.teacher_content_sha256, cohort=cohort)
    refs.append(plan.add(teacher["batch_state_path"], sha=content and content["batch_state_sha256"]))
    if small_read:
        first = next(row for row in teacher["teacher_arrays"] if row["sample_id"] == 384)
        refs.append(plan.add(first["path"], 0, min(first["bytes"], 4096)))
    elif args.window_stop == teacher["layer_stop"]:
        refs.extend(plan.add(row["path"], 0, row["bytes"], content and content["arrays"][row["sample_id"]])
                    for row in teacher["teacher_arrays"])
    frontier = json.loads(read_file(args.boundaries))
    replay_start = frontier["layer_start"]
    if args.window_start > replay_start:
        stride = args.window_layers or (teacher["layer_stop"] - replay_start)
        windows = plan_windows(replay_start, teacher["layer_stop"], window_layers=stride)
        prior = next(window for window in windows if window["window_stop"] == args.window_start)
        path = checkpoint_path(args.checkpoint_dir, prior["window_start"], prior["window_stop"])
        checkpoint = json.loads(read_file(path))
        if checkpoint["next_layer"] != args.window_start:
            raise ValueError("The next read plan requires its exact prior checkpoint")
        refs.append(plan.add(path))
        for stored in checkpoint["streams"].values():
            refs.append(plan.add(stored["batch_state_path"], sha=stored["batch_state_sha256"]))
    return refs

def prepare(renders,cohort,rows,args):
    from g3_readset import shard_header
    from stage1a_energy import progress
    directory = args.prepare_readset
    directory.mkdir(parents=True,exist_ok=True)
    dry = ReadPlan()
    dry_setup = metadata(dry,args,rows)
    dry_setup += static_ranges(dry,renders,sorted({r["layer"] for r in rows}))
    dry_setup += band_ranges(dry, args, small_read=True, cohort=cohort)
    if args.boundaries:
        from g3_residency import read_file
        frontier_probe = json.loads(read_file(args.boundaries))
        dry_setup += [dry.add(args.boundaries), dry.add(frontier_probe["start_batch_state_path"])]
        first_ref = frontier_probe["start_references"][str(cohort["sample_range"][0])]
        dry_setup.append(dry.add(first_ref["path"], 0, first_ref["file_bytes"], first_ref["sha256"]))
    probe = rows[0]
    name = probe.get("checkpoint", probe["qname"]+".weight")
    shard = renders.source_index[name]
    header,n = shard_header(args.source_root/shard,with_size=True)
    extent = header[name]
    refs = [dry.add(args.source_root/shard,0,8),dry.add(args.source_root/shard,8,n),
            dry.add(args.source_root/shard,extent["offset"],extent["bytes"])]
    from menu_contract import selected_options
    if getattr(args, "stream_manifest", None):
        from g3_residency import read_file
        from band_replay import options_for_roster
        stream_roster = json.loads(read_file(args.stream_manifest))["streams"]
        def planned_options(row):
            return options_for_roster(renders, row, stream_roster)
    else:
        def planned_options(row):
            return selected_options(renders, row)
    for option in planned_options(probe):
        refs += option_ranges(dry,renders,option)
    capture_manifest = json.loads((args.capture_root/"layers"/("L%03d"%probe["layer"])/"manifest.json").read_text())
    capture = capture_manifest["units"][probe["qname"]]["heldout"]
    refs.append(dry.add(args.capture_root/capture["file"],0,capture["bytes"],capture["sha256"]))
    # Keep the complete CPU input scope resident through the actual consumer.
    dry.phase("setup",dry_setup+refs)
    dry.phases[0]["resident_before_launch"] = True
    dry_doc = dry.finish(directory/"cpu-data.json",{"scope":"exact real input CPU proof, not full heldout",
                      "output_bound_gib":0.1})

    gpu = ReadPlan()
    setup_refs = metadata(gpu,args,rows)
    setup_refs += static_ranges(gpu,renders,sorted({r["layer"] for r in rows}))
    setup_refs += band_ranges(gpu, args, small_read=False, cohort=cohort)
    source_by_layer = {}
    source_headers = {}
    frontier = None
    if args.boundaries:
        from frontier_replay import load_frontier
        # Same token/shape/layer contract as the actual replay, no guessed paths.
        import torch
        from g3_residency import read_file
        from safetensors.torch import load
        draw = [v for v in load(bytes(read_file(args.draw))).values() if list(v.shape) == [512,512]][0]
        a,b = cohort["sample_range"]
        original = draw[a:b]
        prefix = torch.tensor([cohort["prefix_ids"]],dtype=original.dtype).repeat(b-a,1)
        batches = torch.cat((prefix,original),dim=1)
        frontier,_,_,_ = load_frontier(args,cohort,[batches[i:i+1] for i in range(b-a)])
        setup_refs += [gpu.add(args.boundaries),gpu.add(frontier["start_batch_state_path"])]
    for shard in dict.fromkeys(renders.source_index.values()):
        source_headers[shard], size = shard_header(args.source_root/shard, with_size=True)
        setup_refs += [gpu.add(args.source_root/shard, 0, 8), gpu.add(args.source_root/shard, 8, size)]
    from prismaquant.model_profiles import detect_profile
    profile = detect_profile(str(args.source_root))
    selected_layers = set(range(args.window_start, args.window_stop)) if getattr(args, "stream_manifest", None) else set(range(frontier["layer_start"], frontier["layer_stop"])) if frontier else {r["layer"] for r in rows}
    for tensor,shard in renders.source_index.items():
        live = profile.checkpoint_to_live_name(tensor,multimodal=False)
        import re
        match = re.search(r"\.layers\.(\d+)\.",live) if live is not None else None
        layer = int(match.group(1)) if match and int(match.group(1)) < 45 else None
        if args.boundaries and layer is not None and layer not in selected_layers:
            continue
        if shard not in source_headers:
            source_headers[shard],n = shard_header(args.source_root/shard,with_size=True)
            setup_refs += [gpu.add(args.source_root/shard,0,8),gpu.add(args.source_root/shard,8,n)]
        extent = source_headers[shard][tensor]
        index = gpu.add(args.source_root/shard,extent["offset"],extent["bytes"])
        if layer is None:
            setup_refs.append(index)
        else:
            source_by_layer.setdefault(layer,[]).append(index)
    # Require the complete GPU input phase before source or lookahead reads.
    for layer in sorted(source_by_layer):
        refs = list(source_by_layer[layer])
        for row in rows:
            if row["layer"] == layer:
                for option in planned_options(row):
                    refs += option_ranges(gpu,renders,option)
        if frontier and layer == min(selected_layers):
            for sample in frontier["sample_ids"]:
                ref = frontier["start_references"][str(sample)]
                refs.append(gpu.add(ref["path"],0,ref["file_bytes"],ref["sha256"]))
    gpu.phase("source",range(len(gpu.entries)))
    gpu.phases[0]["resident_before_launch"] = True
    gpu.finish(directory/"gpu-data.json",{"scope":"complete declared source/decoded/boundary reads for selected layers",
        "cohort":cohort,"no_gpu_submitted":True,
        "pilot_bounds":"Use one selected layer with exact parent boundaries. Keep two source slots and one lookahead. Use current-layer BF16 decoded weights and chunk256 FP32 products. Supply the measured footprint plus3GiB. Submit jobs of1800 seconds or less at priority zero. This declaration makes no runtime throughput claim."})
    progress("setup","readsets durable")
    print(json.dumps({"cpu_data":str(directory/"cpu-data.json"),"gpu_data":str(directory/"gpu-data.json"),
                      "cpu_bytes":dry_doc["total_bytes"]}),flush=True)
