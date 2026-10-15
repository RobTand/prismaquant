"""Compare actual native PWC renders under one fixed calibration recipe.

Run only as a PrismaBuild-admitted container action. No standalone cache or
residency implementation: this drives the production cache and its prefetch.
"""

import prismaquant.native_render_codec as _native_render_codec
import argparse
import json
import os
from pathlib import Path
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.workspace_netdata import NetdataWriter, sample_netdata
from prismaquant.calibration_data import load_wikitext_calibration_windowed, _sample_token_windows_from_texts
from prismaquant.model_profiles import profile_from_model
from prismaquant.model_walk import walk_model
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.digests import indent2_json_file_bytes, file_sha256hex
from prismaquant.tensor_digests import tensor_identity
from prismaquant.io_engine import ENGINE
import prismaquant.export_native_compressed as exporter


def identity(tensor):
    return {**tensor_identity(tensor), "bytes": tensor.numel() * tensor.element_size()}


def compare_records(before_dir: Path, after_dir: Path, output: Path, baseline_options: Path | None = None):
    before = json.loads((before_dir / "result.json").read_text())
    after = json.loads((after_dir / "result.json").read_text())
    baseline_recipe_attestation = None
    if baseline_options is not None:
        from prismaquant.native_render_levers import NativeRenderLevers
        attestation = json.loads(baseline_options.read_text())
        if attestation["result_sha256"] != file_sha256hex(before_dir / "result.json"):
            raise ValueError("baseline recipe attestation names a different result")
        expanded = NativeRenderLevers.from_record(attestation["native_render_levers"]).record()
        for key, value in before["levers"].items():
            if expanded.get(key) != value:
                raise ValueError(f"baseline native option {key} disagrees with explicit attestation")
        baseline_recipe_attestation = {"path": str(baseline_options),
                                      "sha256": file_sha256hex(baseline_options),
                                      "captured_levers": before["levers"]}
        before["levers"] = expanded
    fields = ("schema", "calibration", "nsamples", "seqlen", "max_act_rows",
              "profiler_options", "hooked_qnames", "assignment", "formats",
              "rendered_qnames", "levers", "prefetch", "rendered_tensors",
              "wire", "scores", "gates")
    differing = [field for field in fields if before[field] != after[field]]
    if differing:
        raise ValueError(f"native render comparison differs in {differing}")
    evidence = {}
    for label, directory in (("before", before_dir), ("after", after_dir)):
        rows = [json.loads(line) for line in (directory / "netdata.jsonl").read_text().splitlines()]
        counts = {host: sum(row["host"] == host for row in rows)
                  for host in ("sparky", "sparklina")}
        if min(counts.values()) < 2:
            raise ValueError(f"missing paired Netdata series: {label} {counts}")
        power = {}
        for host in ("sparky", "sparklina"):
            points = []
            for row in rows:
                if row["host"] != host:
                    continue
                charts = [chart for chart in row["metrics"].values()
                          if chart.get("context") == "nvidia_smi.gpu_power_draw"]
                if len(charts) != 1:
                    continue
                value = charts[0].get("dimensions", {}).get("power_draw", {}).get("value")
                if isinstance(value, (int, float)) and value >= 0:
                    points.append((float(row["time"]), float(value)))
            points.sort()
            if len(points) > 1:
                joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                             for a, b in zip(points, points[1:]))
                power[host] = {"samples": len(points), "sampled_seconds": points[-1][0] - points[0][0],
                               "mean_watts": sum(p[1] for p in points) / len(points),
                               "peak_sample_watts": max(p[1] for p in points),
                               "sampled_gpu_joules": joules,
                               "rendered_entries_per_sampled_gpu_joule":
                                   len(before["rendered_tensors"]) / joules
                                   if host == "sparklina" and joules > 0 else None}
        evidence[label] = {"path": str(directory), "netdata_counts": counts,
                           "gpu_power": power,
                           "power_scope": "Whole instrumented workflow, including trace serialization. Sparklina is the admitted image-qualified worker; Sparky is external load. Coarse sampling cannot attribute energy to individual render kernels.",
                           "profile_sha256": file_sha256hex(directory / "profile.json"),
                           "netdata_sha256": file_sha256hex(directory / "netdata.jsonl"),
                           "result_sha256": file_sha256hex(directory / "result.json")}
    result = {"schema": "prismaquant.native_render_refactor_comparison.v1",
              "equal_fields": list(fields), "rendered_entries": len(before["rendered_tensors"]),
              "calibration": before["calibration"], "evidence": evidence,
              "performance_claim": None,
              "baseline_recipe_attestation": baseline_recipe_attestation}
    output.write_bytes(indent2_json_file_bytes(result))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--compare-before", type=Path)
    parser.add_argument("--compare-after", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--baseline-options-record", type=Path)
    parser.add_argument("--root", default="/mnt/shared/pq-refactor-20261014-native")
    parser.add_argument("--prepare-calibration", action="store_true")
    parser.add_argument("--calibration-arrow")
    args = parser.parse_args()
    if args.compare_before is not None or args.compare_after is not None:
        if args.compare_before is None or args.compare_after is None or args.comparison_output is None:
            parser.error("comparison requires both directories and an output path")
        compare_records(args.compare_before, args.compare_after, args.comparison_output,
                        args.baseline_options_record)
        return
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    root = Path(args.root)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model_path = root / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    draw_path = root / "calibration.pt"
    if draw_path.exists():
        draw = torch.load(draw_path, weights_only=True)
    else:
        if args.calibration_arrow:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            with pa.memory_map(args.calibration_arrow, "r") as source:
                rows = ipc.open_stream(source).read_all().column("text").to_pylist()
            texts = [text for text in rows if str(text or "").strip()]
            draw = _sample_token_windows_from_texts(texts, tokenizer, 4, 256, seed=42)
        else:
            draw = load_wikitext_calibration_windowed(tokenizer, 4, 256, seed=42)
        torch.save(draw, draw_path)
    if tuple(draw.shape) != (4, 256):
        raise ValueError("calibration recipe differs")
    if args.prepare_calibration:
        receipt = {"dataset": "WikiText-2-raw", "split": "train", "seed": 42,
                   "nsamples": 4, "seqlen": 256, "calibration": identity(draw)}
        if args.calibration_arrow:
            receipt["corpus_artifact"] = {"path": args.calibration_arrow,
                                          "sha256": file_sha256hex(args.calibration_arrow)}
        (root / "calibration-provenance.json").write_bytes(indent2_json_file_bytes(receipt))
        print(json.dumps(receipt), flush=True)
        return
    output = root / args.label
    output.mkdir()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to("cuda").eval()
    profile = profile_from_model(model)
    walk = walk_model(model, claim_rules=profile.walk_claim_rules())
    walk.raise_if_failed()
    qnames = [node.name.removesuffix(".weight") for node in walk.nodes
              if walk.claims[node.name].disposition == "decide"
              and isinstance(model.get_submodule(node.name.removesuffix(".weight")), torch.nn.Linear)]
    selected = [name for name in qnames if name.startswith("model.layers.0.")]
    if len(selected) != 7 or len(qnames) < 100:
        raise ValueError("unexpected Qwen3 linear roster")
    assignment = {name: "NVFP4" for name in qnames}
    formats = ("NVFP4", "FP8_E4M3")
    errors = []
    stop = threading.Event()
    telemetry = []
    def collect():
        with (output / "netdata.jsonl").open("w") as handle:
            writer = NetdataWriter(handle)
            while not stop.is_set():
                for host in ("sparky", "sparklina"):
                    try:
                        record = sample_netdata(host)
                        writer.write(record)
                        handle.flush()
                        telemetry.append((host, record["time"]))
                    except Exception as exc:
                        errors.append(f"{host}: {type(exc).__name__}: {exc}")
                stop.wait(2)
    sampler = ENGINE.submit(collect)
    started = time.time()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False) as profiler:
            cache = fill_production_weight_cache(
                model, draw, qnames, formats=formats,
                render_qnames=selected, max_act_rows=256, progress=True,
                cache_dir=output / "weights")
        torch.cuda.synchronize()
        profiler.export_chrome_trace(str(output / "profile.json"))
        (output / "profile.txt").write_text(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=50))
        if cache.failed:
            raise ValueError(f"render failures: {cache.failed}")
        prefetch = {fmt: cache.prefetch_assignment(
            {name: fmt for name in selected}, require=True, max_workers=1,
            max_resident_bytes=512 * 1024**2) for fmt in formats}
        exporter._PRODUCTION_WEIGHT_CACHE = cache
        exporter._INPUT_GLOBAL_SCALES = exporter._unify_input_global_scales_across_fused_siblings(
            {name: _native_render_codec._nvfp4_input_global_scale_from_max_abs(value)
             for name, value in cache.activation_max_abs.items()}, profile=profile)
        joint = exporter._compute_nvfp4_joint_global(model, assignment, profile=profile)
        tensors, wire = {}, {}
        for name in selected:
            for fmt in formats:
                coordinate = f"{name}|{fmt}"
                tensors[coordinate] = identity(cache.get_resident(name, fmt))
                encoded = exporter._pack_production_cached_2d(
                    name, fmt, nvfp4_global_real_override=joint.get(name),
                    device=torch.device("cuda"))
                if encoded is None:
                    raise ValueError(f"native packing failed: {coordinate}")
                wire[coordinate] = {key: identity(value) for key, value in encoded.items()}
        result = {"schema": "prismaquant.native_render_refactor_evidence.v1",
                  "label": args.label, "started": started, "ended": time.time(),
                  "affinity": sorted(os.sched_getaffinity(0)),
                  "calibration": identity(draw), "nsamples": 4, "seqlen": 256,
                  "max_act_rows": 256, "profiler_options": {"record_shapes": True, "profile_memory": False}, "hooked_qnames": qnames,
                  "assignment": assignment, "formats": formats, "rendered_qnames": selected,
                  "levers": cache.levers, "prefetch": prefetch,
                  "rendered_tensors": tensors, "wire": wire,
                  "scores": cache.metadata["render_scores"],
                  "gates": cache.metadata["render_gates"]}
        (output / "result.json").write_bytes(indent2_json_file_bytes(result))
        print(json.dumps({"label": args.label, "output": str(output),
                          "rendered": len(tensors), "calibration": result["calibration"]}), flush=True)
    finally:
        stop.set()
        sampler.result()
    if errors:
        raise ValueError(f"missing Netdata series: {errors}")
    for host in ("sparky", "sparklina"):
        if sum(name == host for name, _ in telemetry) < 2:
            raise ValueError(f"insufficient Netdata samples for {host}")


if __name__ == "__main__":
    main()
