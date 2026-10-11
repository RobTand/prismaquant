"""Existing-render Stage1a energy and same-entry real-input CPU proof.

Raw128 and prefixed64 are distinct cohorts. Prefix positions condition the
clean forward but are excluded BEFORE profile-owned routed derivation.
Energy consumers are synchronous on the resident measurement device. Only
sequence FP64 sums persist, never full activation rows. Two source cache
slots, one lookahead, existing exact boundary owner and D30 guards.

CPU dry-run reads the actual draw, both complete manifests, source header
and weight, A8/A4/EXL3 wires, complete layer static scales and a digest-
checked heldout capture. It forwards eight real rows through a real source
Linear with the SAME LiveRows/SequenceEnergy consumer as the GPU path.
It is not a full model forward or native GPU qualification, and its
retained diagnostic rows never become prices.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import runpy
import sys


RAW = {"sample_range": [384, 512], "raw_tokens_per_sequence": 512,
       "prefix_ids": [], "input_contract": "raw_512", "local_prefix_rows": "absent",
       "global_original_tokens": 65536, "scored_positions_per_sequence": 511}
PREFIXED = {"sample_range": [384, 448], "raw_tokens_per_sequence": 512,
            "prefix_ids": [154822, 154824], "input_contract": "prefixed_514",
            "local_prefix_rows": "excluded", "global_original_tokens": 32768,
            "scored_positions_per_sequence": 511}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pq-root", type=Path, required=True)
    p.add_argument("--tessera-src", type=Path, required=True)
    p.add_argument("--g3-source", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--a8-root", type=Path, required=True)
    p.add_argument("--t8r-root", type=Path, required=True)
    p.add_argument("--a4-root", type=Path, required=True)
    p.add_argument("--exl3-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reference-manifest", type=Path, required=True)
    p.add_argument("--added-inventory", type=Path)
    p.add_argument("--t4-inventory", type=Path)
    p.add_argument("--option-manifest", type=Path)
    p.add_argument("--weight-source", action="append", help="Actual option name for this bounded energy quantum")
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--draw", type=Path, required=True)
    p.add_argument("--draw-sha256", required=True)
    p.add_argument("--input-contract", choices=("raw_512", "prefixed_514"), required=True)
    p.add_argument("--sample-range", required=True)
    p.add_argument("--prefix-ids", nargs=2, type=int, default=[154822, 154824])
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--layer-range", default="0:45")
    p.add_argument("--units-file", type=Path)
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument("--cpu-rows", type=int, default=8)
    p.add_argument("--dry-run-cpu", action="store_true")
    p.add_argument("--require-staged-inputs", action="store_true")
    p.add_argument("--prepare-readset", type=Path)
    p.add_argument("--boundaries", type=Path,
                   help="Parent pact.prefixed_clean_frontier.v1 receipt, with separate start-reference owner and CPU start state")
    p.add_argument("--boundary-dir", type=Path, default=Path("/tmp/pact-energy-boundaries"))
    p.add_argument("--startup-need-gib", type=float,
                   help="Host measured peak (GPU+RSS+context) plus 3 GiB per D30; required for GPU")
    p.add_argument("--rendered-resident-budget-gib", type=int,
                   help="Admitted whole-lifetime decoded tensor budget; required for GPU work")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-manifest-out", type=Path, required=True)
    return p


_PROGRESS_UNITS = 0


def progress(phase, unit):
    global _PROGRESS_UNITS
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper:
        _PROGRESS_UNITS += 1
        if not runpy.run_path(helper)["commit"](_PROGRESS_UNITS, phase=phase, unit=unit):
            raise RuntimeError("PrismaBuild rejected durable progress")


def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read D30 MemAvailable")


def guard(label):
    if available() < 2 * 2**30:
        raise MemoryError("D30 abort before " + label + ": MemAvailable below 2 GiB")


def setup(args):
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(args.pq_root),
                   str(args.tessera_src), str(args.g3_source)]
    import torch
    from g3_residency import read_file
    from energy_inputs import ExistingRenders
    from frontier_replay import load_frontier, run_frontier
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts, StreamedCausalLM, build_streamed_causal_lm
    from prismaquant.production_weight_cache import _PackedExpertActivationCollector
    from prismaquant.measure_quant_cost import derive_per_expert_activations
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if (args.device == "cpu") != args.dry_run_cpu:
        raise ValueError("CPU is a dry-run only; GPU is not a CPU dry-run")
    if args.chunk_rows <= 0 or not 0 < args.cpu_rows <= 512:
        raise ValueError("Invalid row chunk or bounded CPU probe rows")
    cohort = dict(RAW if args.input_contract == "raw_512" else PREFIXED)
    expected_range = "%d:%d" % tuple(cohort["sample_range"])
    if args.sample_range != expected_range:
        raise ValueError("Declared cohort needs exact sample range " + expected_range)
    if args.input_contract == "prefixed_514" and args.prefix_ids != PREFIXED["prefix_ids"]:
        raise ValueError("Serving prefix differs from parent fixed cohort")
    raw = read_file(args.draw)
    if hashlib.sha256(raw).hexdigest() != args.draw_sha256:
        raise ValueError("Saved draw differs from its own byte digest")
    from safetensors.torch import load
    candidates = [value for value in load(bytes(raw)).values() if list(value.shape) == [512, 512]]
    if len(candidates) != 1 or candidates[0].dtype != torch.int64:
        raise ValueError("Draw must contain one actual int64 512x512 token plane")
    start, stop = cohort["sample_range"]
    original = candidates[0][start:stop]
    cohort["token_sha256"] = hashlib.sha256(original.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    ids = original
    if cohort["prefix_ids"]:
        ids = torch.cat((torch.tensor([cohort["prefix_ids"]], dtype=ids.dtype).repeat(len(ids), 1), ids), dim=1)
    input_batches = [ids[i:i+1] for i in range(len(ids))]
    cohort["sequences"] = len(input_batches)
    renders = ExistingRenders(args)
    lo, hi = map(int, args.layer_range.split(":"))
    if not 0 <= lo < hi <= 46:
        raise ValueError("Layer range must cover source layers or the tail at index45")
    rows = [r for r in renders.rows.values() if lo <= r["layer"] < hi]
    if args.units_file:
        names = bytes(read_file(args.units_file)).decode().splitlines()
        if not names or len(names) != len(set(names)) or set(names) - renders.rows.keys():
            raise ValueError("Selected unit roster is empty, duplicated or unknown")
        rows = [r for r in rows if r["qname"] in names]
        if {r["qname"] for r in rows} != set(names):
            raise ValueError("Selected units leave the declared layer range")
    if not rows:
        raise ValueError("No pricing units selected")
    return renders, cohort, input_batches, sorted(rows, key=lambda r: (r["layer"], r["qname"]))


def actual_cpu(renders, cohort, inputs, rows, args, stream):
    import torch
    from energy_consumer import LiveRows, SequenceEnergy
    from prismaquant.model_profiles import detect_profile
    # No token or geometry substitute: one real source/decoded [N,K] unit.
    if len(rows) != 1:
        raise ValueError("D38 CPU proof must select one actual unit (complete layer scales still read)")
    row = rows[0]
    source, source_range = renders.read_source(row, torch.device("cpu"))
    from menu_contract import selected_options
    if getattr(args, "stream_manifest", None):
        from g3_residency import read_file
        from band_replay import options_for_roster
        options = options_for_roster(renders, row, json.loads(read_file(args.stream_manifest))["streams"])
    else:
        options = selected_options(renders, row)
    decoded = []
    for option in options:
        prefetched = None if option.get("passthrough") else renders.read_weight(option, source.shape)
        decoded.append((option, renders.decode(option, source, device="cpu", prefetched=prefetched)))
    x, capture_receipt = renders.capture_probe(row)
    if x.shape[1] != source.shape[1]:
        raise ValueError("Actual capture/source width mismatch")
    # Actual Linear forward triggers the SAME synchronous live consumer.
    model = torch.nn.Module()
    parts = row["qname"].split(".")
    parent = model
    for name in parts[:-1]:
        child = torch.nn.Module()
        parent.add_module(name, child)
        parent = child
    linear = torch.nn.Linear(source.shape[1], source.shape[0], bias=False)
    linear.weight = torch.nn.Parameter(source, requires_grad=False)
    parent.add_module(parts[-1], linear)
    profile = detect_profile(str(args.source_root))
    weights = {row["qname"]: {"source": source, "options": decoded}}
    from footprint import FootprintRecorder
    footprint = FootprintRecorder()
    footprint.record("CPU-source-and-renders", weights=weights, state={"inputs":x})
    consumer = SequenceEnergy(rows, weights, cohort=cohort, device=torch.device("cpu"),
                              chunk_rows=args.chunk_rows, guard=guard, stream=stream)
    # Captured eight rows are raw diagnostic inputs, not a fake 514-token pass.
    from bounded_profile import capture_item
    with capture_item():
        consumer.begin(cohort["sample_range"][0])
        tap_row = dict(row, kind="dense")
        with torch.inference_mode(), LiveRows(model, profile, [tap_row], consumer.consume):
            linear(x.to(source.dtype))
        consumer.end()
    consumer.finish([cohort["sample_range"][0]])
    footprint.record("CPU-consumer-complete", weights=weights, state={"inputs":x})
    frontier_probe = None
    if args.boundaries:
        from frontier_replay import load_frontier
        from prismaquant.cost_streaming import StreamedBoundaryArtifacts
        frontier, frontier_states, refs, samples = load_frontier(args, cohort, inputs, cpu_proof=True)
        incoming = frontier["start_reference_owner"]
        with StreamedBoundaryArtifacts(incoming["boundary_config"]) as owner:
            owner.attach(incoming["session"], n_probes=0)
            with owner.prefetch([refs[samples[0]]]) as window:
                actual = owner.get(window, refs[samples[0]])
                footprint.record("CPU-frontier-window",weights=weights,state={"owner":owner,"hidden":actual,"states":frontier_states})
                frontier_probe = {"sample_id": samples[0], "shape": list(actual.shape),
                                  "start_generation": incoming["session"], "foreign_entries_retired": 0,
                                  "cpu_proof_only": True}
    return {"schema": "pact.same_entry_real_cpu_proof.v2", "actual_consumer_passed": True,
            "source_shape": list(source.shape), "footprint": footprint.finish(), "decoded_shapes": [list(w.shape) for _, w in decoded],
            "source_range": source_range, "capture": capture_receipt, "frontier_probe": frontier_probe,
            "cohort": cohort, "prefix_energy_measured": False,
            "partial_diagnostic_only": True, "full_heldout_measured": False,
            "same_consumer": "LiveRows -> SequenceEnergy -> batched_option_sums, real source Linear",
            "activation_backend": "attested CPU numerical oracle, not GPU op evidence",
            "scale_reduction": renders.static_layer(row["layer"])[1],
            "limit": "Real source/decoded/capture reads and consumer plumbing, not clean full-model or GPU equivalence proof"}


def run_gpu(renders, cohort, inputs, rows, args, stream):
    from energy_inputs import source_reader
    with source_reader(args) as reader:
        result = _run_gpu(renders, cohort, inputs, rows, args, stream)
        result["source_reads"] = reader.stats
        return result


def _run_gpu(renders, cohort, inputs, rows, args, stream):
    import torch
    from bounded_profile import capture_item
    from energy_consumer import LiveRows, SequenceEnergy
    from menu_contract import unit_weight, selected_options
    from prismaquant.model_profiles import detect_profile
    from prismaquant.cost_streaming import build_streamed_causal_lm, StreamedBoundaryArtifacts
    if not args.boundaries:
        raise ValueError("The GPU energy quantum needs its actual bounded source frontier")
    if args.startup_need_gib is None or args.startup_need_gib < 3:
        raise ValueError("D30 requires the parent measured peak plus 3 GiB via --startup-need-gib")
    if available() < args.startup_need_gib * 2**30:
        raise MemoryError("D30 measured startup footprint plus margin is unavailable")
    if type(args.rendered_resident_budget_gib) is not int or args.rendered_resident_budget_gib <= 0:
        raise ValueError("The GPU energy path needs its admitted rendered-resident byte budget")
    profile = detect_profile(str(args.source_root))
    runner = build_streamed_causal_lm(str(args.source_root), device=torch.device("cuda"),
        dtype=torch.bfloat16, profile=profile, offload_folder=str(args.boundary_dir / "offload"),
        max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1,
        cache_headroom_gb=2, prefetch_min_available_gb=2, require_prefetched_residency=True,
        attn_implementation="eager")
    runner.model.eval()
    from footprint import FootprintRecorder
    footprint = FootprintRecorder()
    footprint.record("source-model-ready", runner=runner)
    aggregates, rendered_residency = [], []
    by_layer = {layer: [r for r in rows if r["layer"] == layer] for layer in sorted({r["layer"] for r in rows})}
    def measure(layer, forward, replay_state):
        selected = by_layer.get(layer, [])
        if not selected:
            for index, ids in enumerate(inputs):
                guard("unpriced clean source layer")
                with capture_item():
                    forward(ids)
                progress("layer-%02d" % layer,
                         "complete unpriced sequence %d" % (cohort["sample_range"][0] + index))
            progress("layer-%02d" % layer, "unpriced clean layer complete")
            return
        guard("resident source layer")
        sources = {row["qname"]: unit_weight(runner, row) for row in selected}
        options = {row["qname"]: selected_options(renders, row) for row in selected}
        with renders.retained_decoded_window(sources, options, device=runner.device,
                max_resident_bytes=args.rendered_resident_budget_gib * 2**30, guard=guard) as (weights, residency):
            rendered_residency.append({"layer": layer, **residency})
            footprint.record("layer-%02d-all-renders-ready" % layer, runner=runner, weights=weights, state=replay_state)
            consumer = SequenceEnergy(selected, weights, cohort=cohort, device=runner.device,
                chunk_rows=args.chunk_rows, guard=guard, stream=stream)
            try:
                with LiveRows(runner.model, runner.profile, selected, consumer.consume,
                              prefix_rows=len(cohort["prefix_ids"])):
                    for index, ids in enumerate(inputs):
                        guard("clean heldout sequence")
                        with capture_item():
                            consumer.begin(cohort["sample_range"][0] + index)
                            forward(ids)
                            consumer.end()
                        stream.flush()
                        os.fsync(stream.fileno())
                        progress("layer-%02d" % layer,
                                 "complete energy sequence %d" % (cohort["sample_range"][0] + index))
                        if index in (0, len(inputs) - 1):
                            footprint.record("layer-%02d-sequence-%d-complete" % (layer, index),
                                runner=runner, weights=weights, state=replay_state)
                aggregates.extend(consumer.finish(range(*cohort["sample_range"])))
            finally:
                weights.clear()
        stream.flush()
        progress("layer-%02d" % layer, "complete layer %d" % layer)

    try:
        if set(by_layer) == {45}:
            from tail_replay import run_tail_frontier
            frontier_evidence = run_tail_frontier(runner,args,cohort,inputs,measure,guard)
        else:
            from frontier_replay import run_frontier
            frontier_evidence = run_frontier(runner,args,cohort,inputs,measure,guard)
        return {"schema": "pact.stage1a_energy_run_manifest.v2", "cohort": cohort,
                "full_heldout_measured": True, "device": "cuda", "aggregates": aggregates,
                "normalization": "Each unit sum divided once by global original tokens; sequence rows retain raw sums",
                "source_cache_slots": 2, "lookahead": 1, "frontier_replay": frontier_evidence, "footprint": footprint.finish(),
                "prefix_exclusion": "clean module input positions excluded before profile-owned routed derivation",
                "amp2_measured": True, "scale_evidence": {str(l): renders.static_layer(l)[1] for l, members in by_layer.items() if any(r["kind"] in {"routed","shared","dense"} for r in members)},
                "rendered_residency": rendered_residency,
                "startup_need_gib": args.startup_need_gib,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_cuda_allocated": torch.cuda.max_memory_allocated()}
    finally:
        runner.shutdown()


def main():
    args = parser().parse_args()
    guard("setup")
    from strict_leases import strict_input_leases
    with strict_input_leases((args.device == "cuda" or args.require_staged_inputs) and not args.prepare_readset) as lease_audit:
        return run(args, lease_audit)


def run(args, lease_audit):
    renders, cohort, inputs, rows = setup(args)
    if args.prepare_readset:
        from build_manifest import prepare
        prepare(renders, cohort, rows, args)
        if not args.dry_run_cpu:
            return
    progress("setup", "actual metadata and cohort ready")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    try:
        with temporary.open("w") as stream:
            result = actual_cpu(renders, cohort, inputs, rows, args, stream) if args.dry_run_cpu else run_gpu(renders, cohort, inputs, rows, args, stream)
            stream.flush()
            os.fsync(stream.fileno())
        from g3_residency import receipt
        result["residency"] = receipt()
        result["input_leases"] = dict(lease_audit)
        result["menu_gaps"] = renders.menu_gaps
        result["selected_weight_sources"] = args.weight_source or ["A8S", "A4-q896", "EXL3"]
        args.run_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.run_manifest_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        progress("energy", "complete instrument")
        print(json.dumps({"output": str(args.output), "result": str(args.run_manifest_out),
                          "actual_consumer_passed": result.get("actual_consumer_passed"),
                          "full_heldout_measured": result["full_heldout_measured"]}), flush=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
