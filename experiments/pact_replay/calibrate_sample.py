"""Actual FIT calibration and canonical research samples for added PACT units."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("calibrate","merge","sample"),required=True)
    for name in ("pq-root","tessera-src","g3-source","source-root","inventory","draw"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--draw-sha256",required=True)
    p.add_argument("--unit",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--dry-run-cpu",action="store_true")
    p.add_argument("--require-staged-inputs",action="store_true")
    p.add_argument("--fit-sample-range",default="0:384")
    p.add_argument("--fit-capture",type=Path)
    p.add_argument("--captures",type=Path,nargs="+")
    p.add_argument("--family",choices=("T8","T16"))
    p.add_argument("--q256",type=int)
    p.add_argument("--tp-splits",type=int,choices=(1,2))
    p.add_argument("--startup-need-gib",type=float)
    p.add_argument("--gpu-seconds-remaining",type=float)
    p.add_argument("--deadline-seconds",type=int,default=1700)
    p.add_argument("--boundary-dir",type=Path,required=True)
    p.add_argument("--prepare-readset",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--option-manifest-out",type=Path)
    p.add_argument("--run-manifest-out",type=Path,required=True)
    return p


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(path)+".partial")
    with temporary.open("w") as stream:
        json.dump(value,stream,allow_nan=False,indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def setup(args):
    sys.path[:0] = [str(Path(__file__).resolve().parent),str(args.pq_root),str(args.tessera_src),str(args.g3_source)]
    import torch
    from g3_residency import read_file
    from safetensors.torch import load
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not 0 < args.deadline_seconds <= 1700:
        raise ValueError("The sample action needs a deadline within1700 seconds")
    if args.mode != "merge" and (args.device == "cpu") != args.dry_run_cpu:
        raise ValueError("CPU calibration and samples are diagnostic dry runs only")
    inventory = json.loads(read_file(args.inventory))
    if inventory.get("schema") != "pact.added_unit_inventory.v2":
        raise ValueError("The added unit inventory schema differs")
    matches = [unit for unit in inventory["added_units"] if unit["qname"] == args.unit]
    if len(matches) != 1 or matches[0]["group"] not in {"attention","shared","lm_head"}:
        raise ValueError("The sample must name one actual added nonrouted unit")
    unit = matches[0]
    unit["kind"] = unit["group"]
    unit["layer"] = 45 if unit["kind"] == "lm_head" else unit["structural_key"]["layer"]
    unit["expert"] = None
    raw = read_file(args.draw)
    if hashlib.sha256(raw).hexdigest() != args.draw_sha256:
        raise ValueError("The saved draw fails its own byte digest")
    candidates = [value for value in load(bytes(raw)).values() if tuple(value.shape) == (512,512)]
    if len(candidates) != 1 or candidates[0].dtype != torch.int64:
        raise ValueError("The draw needs one actual int64 token plane")
    fit = candidates[0][:384]
    return unit,fit


def source_extent(args,unit):
    from g3_readset import shard_header
    from g3_residency import read_file
    index = json.loads(read_file(args.source_root/"model.safetensors.index.json"))["weight_map"]
    path = args.source_root/index[unit["checkpoint"]]
    header,size = shard_header(path,with_size=True)
    extent = header[unit["checkpoint"]]
    declared = unit["source_location"]
    if (list(extent["shape"]) != unit["shape"] or extent["offset"] != declared["header_bytes"]+declared["data_offsets"][0]
            or extent["bytes"] != declared["byte_size"]):
        raise ValueError("The actual source header and inventory geometry or extent differ")
    return path,extent,size


def read_source(args,unit):
    import torch
    from g3_lib import read_range
    path,extent,_ = source_extent(args,unit)
    if extent["dtype"] != torch.bfloat16:
        raise ValueError("The canonical source sample needs actual BF16 weights")
    raw = read_range(str(path),extent["offset"],extent["bytes"])
    value = torch.frombuffer(raw,dtype=torch.bfloat16).reshape(extent["shape"]).to(args.device)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("The actual source sample weights are not finite")
    return value,{"path":str(path),"offset":extent["offset"],"bytes":extent["bytes"]}


def prepare(args,unit):
    from build_manifest import ReadPlan
    from g3_readset import shard_header
    from g3_residency import read_file
    from prismaquant.model_profiles import detect_profile
    plan = ReadPlan()
    setup_refs = [plan.add(path) for path in (args.inventory,args.draw,args.source_root/"config.json",args.source_root/"model.safetensors.index.json")]
    if args.fit_capture:
        setup_refs.append(plan.add(args.fit_capture))
    if args.captures:
        setup_refs.extend(plan.add(path) for path in args.captures)
    source_refs = []
    index = json.loads(read_file(args.source_root/"model.safetensors.index.json"))["weight_map"]
    if args.mode == "calibrate":
        profile = detect_profile(str(args.source_root))
        stop = min(unit["layer"]+1,44)
        headers = {}
        for shard in dict.fromkeys(index.values()):
            headers[shard],size = shard_header(args.source_root/shard,with_size=True)
            setup_refs.extend((plan.add(args.source_root/shard,0,8),plan.add(args.source_root/shard,8,size)))
        for name,shard in index.items():
            live = profile.checkpoint_to_live_name(name,multimodal=False)
            if live is not None and ".layers." in live:
                layer = int(live.split(".layers.",1)[1].split(".",1)[0])
                if 0 <= layer < 45 and layer > stop:
                    continue
            extent = headers[shard][name]
            source_refs.append(plan.add(args.source_root/shard,extent["offset"],extent["bytes"]))
    elif args.mode == "sample":
        path,extent,size = source_extent(args,unit)
        setup_refs.extend((plan.add(path,0,8),plan.add(path,8,size)))
        source_refs.append(plan.add(path,extent["offset"],extent["bytes"]))
    plan.phase("source",[*setup_refs,*source_refs])
    result = plan.finish(args.prepare_readset,{"mode":args.mode,"unit":args.unit,"diagnostic_only":args.dry_run_cpu})
    print(json.dumps({"data_manifest":str(args.prepare_readset),"bytes":result["total_bytes"]}),flush=True)


def calibrate(args,unit,fit,guard,footprint):
    import torch
    from energy_inputs import source_reader
    from menu_contract import executed_module
    from prismaquant.model_profiles import detect_profile
    from prismaquant.cost_streaming import build_streamed_causal_lm,StreamedBoundaryArtifacts
    from prismaquant.tessera_campaign import _collect_activations
    from stage1a_energy import progress
    start,stop = map(int,args.fit_sample_range.split(":"))
    if not 0 <= start < stop <= 384:
        raise ValueError("Calibration must use an explicit subset of the FIT population")
    actual_stop = start+1 if args.dry_run_cpu else stop
    ids = fit[start:actual_stop,:8 if args.dry_run_cpu else 512]
    batches = [row[None] for row in ids]
    result = None
    class Complete(Exception):
        pass
    with source_reader(args) as reads:
        runner = build_streamed_causal_lm(str(args.source_root),device=torch.device(args.device),dtype=torch.bfloat16,
            profile=detect_profile(str(args.source_root)),offload_folder=str(args.boundary_dir/"offload"),
            max_cache_slots=2,prefetch_workers=1,prefetch_lookahead=1,cache_headroom_gb=3,
            prefetch_min_available_gb=2,require_prefetched_residency=True,attn_implementation="eager")
        runner.model.eval()
        module = executed_module(runner.model,runner.profile,unit)
        name = next(name for name,value in runner.model.named_modules() if value is module)
        config = {"schema":"prismaquant.aura.boundary_storage.v2","capture_order":"layer_major",
            "directory":str(args.boundary_dir/"calibration"),"max_resident_bytes":256*2**20,
            "max_auxiliary_bytes":2*2**30,"max_artifact_bytes":140*2**30,"prefetch_batches":1}
        try:
            with StreamedBoundaryArtifacts(config) as owner:
                owner.bind({"unit":args.unit,"sample_range":[start,actual_stop],"input_contract":"raw_8_CPU_probe" if args.dry_run_cpu else "raw_512"},n_probes=0,check_memory=guard,published=False)
                def visit(layer,forward):
                    nonlocal result
                    guard("calibration source layer")
                    if layer == unit["layer"]:
                        result = _collect_activations(runner.model,[name],batches,1,runner.device,
                            want_hessian=True,profile=runner.profile,forward_batch=forward,resource_check=guard)
                        footprint.record("calibration-unit-complete",runner=runner,state={"H":result[1][name]})
                        raise Complete()
                    for batch in batches:
                        guard("calibration source sequence")
                        forward(batch)
                    progress("capture","durable clean calibration layer")
                if unit["layer"] == 45:
                    def traverse(all_ids):
                        return runner.visit_layer_batches([row[None] for row in all_ids],visit,
                            output_consumer=lambda _index,_logits:guard("head calibration output"),boundary_storage=owner,
                            source_phase=lambda stage,_layer,_size:guard(stage))
                    result = _collect_activations(runner.model,[name],[ids],1,runner.device,want_hessian=True,
                        profile=runner.profile,forward_batch=traverse,resource_check=guard)
                else:
                    try:
                        runner.visit_layer_batches(batches,visit,boundary_storage=owner,source_phase=lambda stage,_layer,_size:guard(stage))
                    except Complete:
                        pass
        finally:
            runner.shutdown()
    if result is None or result[2][name] <= 0 or result[1][name] is None:
        raise ValueError("The actual source forward supplied no calibration rows")
    H = result[1][name]
    if tuple(H.shape) != (unit["shape"][1],)*2 or not bool(torch.isfinite(H).all()):
        raise ValueError("The actual FIT Hessian has wrong geometry or nonfinite values")
    provenance = {"hessian_role":"fit","sample_range":[start,actual_stop],"input_contract":"raw_8_CPU_probe" if args.dry_run_cpu else "raw_512",
        "text_sha256":args.draw_sha256,"fit_tokens":int(ids.numel()),
        "fit_ids_sha256":hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
        "actual_token_ids":ids.tolist(),"model":str(args.source_root),"dry_run_cpu":args.dry_run_cpu,
        "H_definition":"FP32 Gram over every actual unit input row; no Hessian row cap"}
    payload = {"H":{args.unit:H},"counts":{args.unit:result[2][name]},"provenance":provenance}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(args.output)+".partial")
    torch.save(payload,temporary)
    temporary.replace(args.output)
    return {"schema":"pact.fit_capture_result.v1","output":str(args.output),"provenance":provenance,
        "counts":payload["counts"],"source_reads":reads.stats,"full_fit_capture":not args.dry_run_cpu and [start,actual_stop] == [0,384]}


def load_capture(path):
    import torch
    from g3_residency import read_file
    return torch.load(io.BytesIO(read_file(path)),map_location="cpu",weights_only=True)


def merge(args,unit,fit,guard):
    import torch
    if args.device != "cpu" or not args.captures or args.dry_run_cpu:
        raise ValueError("Capture assembly is CPU-only and needs actual complete quanta")
    H,count,seen = None,0,set()
    for path in args.captures:
        guard("merge FIT Hessian")
        payload = load_capture(path)
        provenance = payload["provenance"]
        a,b = provenance["sample_range"]
        if provenance.get("dry_run_cpu") or provenance.get("hessian_role") != "fit" or not 0 <= a < b <= 384:
            raise ValueError("A FIT quantum is proof-only or has invalid sample scope")
        if seen.intersection(range(a,b)) or provenance["actual_token_ids"] != fit[a:b].tolist():
            raise ValueError("The actual FIT populations repeat or differ")
        if provenance["input_contract"] != "raw_512" or set(payload["H"]) != {args.unit}:
            raise ValueError("The FIT unit or token contract differs")
        value = payload["H"][args.unit]
        if tuple(value.shape) != (unit["shape"][1],)*2 or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("A FIT quantum Hessian is invalid")
        rows = payload["counts"][args.unit]
        if type(rows) is not int or rows <= 0:
            raise ValueError("A FIT quantum has no actual unit rows")
        H = value.clone() if H is None else H.add_(value)
        count += rows
        seen.update(range(a,b))
    if seen != set(range(384)):
        raise ValueError("The calibration lacks the complete FIT population0:384")
    provenance = {"hessian_role":"fit","sample_range":[0,384],"input_contract":"raw_512", "dry_run_cpu":False,
        "text_sha256":args.draw_sha256,"fit_tokens":196608,"fit_ids_sha256":hashlib.sha256(fit.contiguous().numpy().tobytes()).hexdigest(),
        "model":str(args.source_root),"H_definition":"Sum of actual FP32 Gram quanta over all FIT inputs"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(args.output)+".partial")
    torch.save({"H":{args.unit:H},"counts":{args.unit:count},"provenance":provenance},temporary)
    temporary.replace(args.output)
    return {"schema":"pact.fit_capture_result.v1","output":str(args.output),"provenance":provenance,"full_fit_capture":True,"actual_rows":count}


def sample_option(family, q256, *, wire, format_facts, tp_splits):
    """Bind every rate to this unit's fixed family reference sample."""
    reference_rate = 1024 if family == "T8" else 2048
    return {"name": family+"-R"+str(q256)+"-sample", "family": family, "q256": q256,
        "contract": "bf16_unquantized" if family == "T16" else "fp8_per_token_dynamic",
        "body": format_facts["body"], "outer_scheme": format_facts["plane"], "wire": wire,
        "anchor_source": family+"-R"+str(reference_rate)+"-sample",
        "tp_splits": tp_splits if family == "T8" else 1,
        "canonical_T16_priceable": family == "T16",
        "weight_reference_definition": "separate_values_and_row_scales_through_dot" if family == "T16" else "decoded_bf16",
        "serving_admission": False}


def sample(args,unit,fit,guard,footprint):
    import torch
    from tessera.alphabet import E4M3_GRID,BF16_GRID
    from tessera.export import ActivationSource,encode_linear,wire_recipe
    from tessera.unit_artifact import parse_unit_metadata
    from canonical_weight import read_canonical_weight
    from mixed_manifest import read_unit_blob
    from g3_residency import host_path
    if args.family is None or args.q256 is None or args.fit_capture is None:
        raise ValueError("A sample needs its explicit family, whole-bit rate, and actual FIT capture")
    if args.family == "T16" and args.q256 != 2048 or args.family == "T8" and args.q256 not in (512,768,1024,1280):
        raise ValueError("The requested sample rate lies outside this research commission")
    if args.family == "T8" and args.tp_splits is None:
        raise ValueError("A new T8 unit needs an explicit activation TP contract")
    payload = load_capture(args.fit_capture)
    provenance = payload["provenance"]
    if provenance.get("hessian_role") != "fit" or set(payload["H"]) != {args.unit}:
        raise ValueError("The sample needs its actual own-unit FIT Hessian")
    full_sha = hashlib.sha256(fit.contiguous().numpy().tobytes()).hexdigest()
    if not args.dry_run_cpu and (provenance.get("dry_run_cpu") or provenance.get("sample_range") != [0,384]
            or provenance.get("fit_ids_sha256") != full_sha or provenance.get("fit_tokens") != 196608):
        raise ValueError("The sample lacks the comparable complete FIT population")
    source,extent = read_source(args,unit)
    H = payload["H"][args.unit]
    if tuple(H.shape) != (source.shape[1],)*2 or H.dtype != torch.float32 or not bool(torch.isfinite(H).all()):
        raise ValueError("The actual sample source and FIT Hessian differ")
    original_shape = list(source.shape)
    if args.dry_run_cpu:
        source = source[:16,:64].contiguous()
        H = H[:64,:64].contiguous()
    grid = BF16_GRID if args.family == "T16" else E4M3_GRID
    recipe = wire_recipe(grid,args.q256)
    activation = ActivationSource({args.unit:H},provenance)
    guard("activation-aware encoder")
    kwargs = activation.for_unit(args.unit+".weight",source.shape[1],device=args.device,scale_plane=recipe.scale_plane,weight=source)
    exported = encode_linear(source,grid=grid,q256=args.q256,name=args.unit,verify=True,**kwargs)
    guard("canonical sample decode")
    row = {key:unit[key] for key in ("qname","kind","role","expert","layer")}
    raw,fact = read_unit_blob(row,expected_shape=tuple(source.shape),data=exported.blob)
    decoded = read_canonical_weight(raw,args.device,tuple(source.shape)) if args.family == "T16" else None
    footprint.record("encoded-sample-ready",weights={args.unit:{"source":source,"options":[({},decoded)]}},state={"H":H})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(args.output)+".partial")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    result = {"schema":"pact.canonical_sample_result.v1","qname":args.unit,"family":args.family,"q256":args.q256,
        "actual_source_shape":original_shape,"sample_shape":list(source.shape),"dry_run_cpu":args.dry_run_cpu,
        "full_fit_capture":not args.dry_run_cpu,"source_extent":extent,"wire_facts":fact,"serialized_bytes":len(raw),
        "route_status":"research_unbacked","serving_admission":False,"price_measured":False}
    if not args.dry_run_cpu:
        if args.option_manifest_out is None:
            raise ValueError("A real sample needs its actual option locator output")
        option = sample_option(args.family,args.q256,
            wire={"path":host_path(str(args.output)),"offset":0,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()},
            format_facts=fact["format"],tp_splits=args.tp_splits)
        atomic_json(args.option_manifest_out,{"schema":"pact.energy_options.v1","units":[{"qname":args.unit,"options":[option]}],"gaps":[]})
        result["option_manifest"] = host_path(str(args.option_manifest_out))
    return result


def main():
    args = parser().parse_args()
    sys.path.insert(0,str(args.g3_source))
    from strict_leases import strict_input_leases
    from stage1a_energy import available,guard as memory_guard,progress
    from footprint import FootprintRecorder
    began = time.monotonic()
    limit = args.deadline_seconds
    if args.device == "cuda" and not args.prepare_readset:
        if args.startup_need_gib is None or args.startup_need_gib < 3 or available() < args.startup_need_gib*2**30:
            raise MemoryError("D30 measured sample footprint plus margin is unavailable")
        if args.mode == "sample" and args.family == "T16":
            if args.gpu_seconds_remaining is None or not 0 < args.gpu_seconds_remaining <= 3600:
                raise ValueError("The T16 sample needs the remaining one-hour campaign allowance")
            limit = min(limit,args.gpu_seconds_remaining)
    def guard(label):
        memory_guard(label)
        if time.monotonic()-began >= limit:
            raise TimeoutError("The calibration or sample quantum reached its declared deadline")
    with strict_input_leases((args.device == "cuda" or args.require_staged_inputs) and not args.prepare_readset) as leases:
        unit,fit = setup(args)
        if args.prepare_readset:
            prepare(args,unit)
            return
        progress("setup","actual sample input metadata ready")
        footprint = FootprintRecorder()
        if args.mode == "calibrate":
            result = calibrate(args,unit,fit,guard,footprint)
        elif args.mode == "merge":
            result = merge(args,unit,fit,guard)
        else:
            result = sample(args,unit,fit,guard,footprint)
        result.update(input_leases=leases,footprint=footprint.finish(),elapsed_seconds=time.monotonic()-began)
        atomic_json(args.run_manifest_out,result)
        progress("capture" if args.mode != "sample" else "sample","durable actual sample producer result")
        print(json.dumps({"result":str(args.run_manifest_out),"mode":args.mode,"dry_run_cpu":args.dry_run_cpu}),flush=True)


if __name__ == "__main__":
    main()
