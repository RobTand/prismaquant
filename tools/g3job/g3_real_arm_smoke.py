"""Compare one real G3 window through four pinned layers, with and without omission.

This diagnostic uses the real source loader, expert packer, unit_view, decoder,
head and FP64 KL. Its truncated logits are not full-model quality evidence.
The production arm and pinned /pq files stay unchanged.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import g3_offline_decoded_kl as G

MODES = ("source-read", "omission")



def progress(mode, phase, *, scored=False):
    print("G3_COMPARE_PROGRESS " + json.dumps({"phase": f"{mode}-{phase}", "scored": scored}), flush=True)


def prepare_real_arm_plan(spec, out):
    """Build the union of both diagnostic read sets, without source payload reads."""
    sys.path.insert(0, str(G.PQ))
    from g3_readset import SourceReads, build_manifest
    m, by_layer, _ = G.read_g3_manifest(spec["manifest"], spec["manifest_sha256"], None)
    reads = SourceReads(spec["model"], [spec["arm"]], by_layer)
    reads.omitted.clear()  # The union includes the baseline's source payloads.
    teacher_specs = [(root, json.loads((Path(root) / "teacher.json").read_bytes()))
                     for _name, root, _digest in spec["teachers"]]
    panel = json.loads(Path(spec["panel"]).read_bytes())
    arrays = Path(spec["arrays_root"])
    setup = [spec["manifest"], spec["panel"], arrays / Path(panel["causal_mask_array"]).name,
             *(arrays / Path(w["tokens_path"]).name for w in panel["windows"]), *spec["setup_files"]]
    plan = build_manifest(reads, [spec["arm"]], by_layer, spec["roots"], teacher_specs,
                          [panel["windows"][spec["window"]]["window_id"]], setup_files=setup)
    selected = [phase for phase in plan["read_plan"]["phases"]
                if phase["name"] in {"setup", "teachers", *(f"layer-{i:02d}" for i in range(spec["layers"]))}]
    indices = list(dict.fromkeys(i for phase in selected for i in phase["entry_indices"]))
    positions = {old: new for new, old in enumerate(indices)}
    plan["entries"] = [plan["entries"][i] for i in indices]
    plan["entry_count"] = len(indices)
    plan["total_bytes"] = sum(entry["bytes"] for entry in plan["entries"])
    for phase in selected:
        phase["entry_indices"] = [positions[i] for i in phase["entry_indices"]]
    phases, cumulative = [], 0
    for mode in MODES:
        for phase in selected:
            cumulative += phase["bytes"]
            phases.append({**phase, "name": mode + "-" + phase["name"], "cumulative_bytes": cumulative})
    plan["read_plan"] = {"phases": phases, "read_bytes": cumulative}
    plan["annotations"]["teacher_repeats"] = 2
    with gzip.open(out, "xb") as stream:
        stream.write(json.dumps(plan, separators=(",", ":")).encode())
    print(json.dumps({"manifest": out, "phases": len(phases), "entries": plan["entry_count"]}), flush=True)


def run_pass(spec, out, mode):
    import importlib
    import numpy as np
    import torch
    sys.path.insert(0, str(G.PQ))
    sys.path.insert(0, str(G.PQ / "tools"))
    sys.path.insert(0, str(G.PQ / "experiments"))
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.model_profiles import detect_profile
    from tools.build_streamed_full_kl_teacher import _require_source_execution_policy, _source_derivative_policy
    from experiments.glm_tr3_full_vocab import load_panel, token_kl, CONTEXT_LENGTH, VOCAB_SIZE
    importlib.import_module("experiments.build_glm_tr3_teacher")  # Activate the pinned source bootstrap.
    from g3_prepared_source import cached_checkpoint_identity
    from g3_readset import SourceReads
    from g3_residency import finish_phase, receipt

    os.environ["G3_PHASE_PREFIX"] = mode + "-"
    out = Path(out)
    out.mkdir(exist_ok=False)
    started = G.time.time()
    progress(mode, "setup")
    m, by_layer, sha = G.read_g3_manifest(spec["manifest"], spec["manifest_sha256"], None)
    panel, inputs = load_panel(spec["panel"], arrays_root=spec["arrays_root"])
    inputs = [inputs[spec["window"]]]  # Keep the complete original 2048-token window.
    wid = panel["windows"][spec["window"]]["window_id"]
    teachers = G.Teachers([(name, root, digest) for name, root, digest in spec["teachers"]], [wid])
    t04 = teachers.specs[0][4]
    identity = cached_checkpoint_identity(spec["model"], None, None, stored_identity=t04["source_model_identity"])
    G.source_identity_check(t04, t04["source_model_identity_sha256"], spec["reference_binding"],
                            spec["reference_binding_sha256"], identity, Path(spec["model"]))
    policy = _source_derivative_policy(argparse.Namespace(source_derivative_json=spec["source_derivative_json"],
                                      source_derivative_sha256=spec["source_derivative_sha256"]))
    device = require_cuda_hot_path("g3_real_arm_smoke", "cuda")
    reads = SourceReads(spec["model"], [spec["arm"]], by_layer)
    omitted_roster = len(reads.omitted)
    if mode == "source-read":
        reads.omitted.clear()
        G.source_indices = lambda arms, rows: list(range(len(rows)))
    reads.install()
    runner = build_streamed_causal_lm(spec["model"], device=device, dtype=torch.bfloat16,
        offload_folder=str(out / "offload"), profile=detect_profile(spec["model"]), cache_headroom_gb=12.,
        max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager", source_derivative=policy)
    # Limit only this diagnostic traversal. Do not edit the pinned numerical code.
    assert runner.num_layers == 45
    runner.num_layers = spec["layers"]
    schedule = runner.context.schedule_prefetch
    runner.context.schedule_prefetch = lambda layer: schedule(layer) if layer < spec["layers"] else False
    visited = {k: v for k, v in by_layer.items() if k < spec["layers"]}
    sub = G.Substitution(runner, spec["arm"], visited, spec["roots"], spec["decoder"], out, -1)
    results = {}
    try:
        runner.model.eval()
        runner.context.begin_source_initialization_audit()
        _require_source_execution_policy(source_execution_identity(runner.model), policy)
        finish_phase("setup")
        progress(mode, "layer-00")

        def consume(index, logits):
            assert index == 0 and tuple(logits.shape) == (1, CONTEXT_LENGTH, VOCAB_SIZE)
            raw = logits[0, :-1].float()
            assert raw.device.type == "cuda" and bool(torch.isfinite(raw).all())
            np.save(out / "logits.npy", raw.cpu().numpy(), allow_pickle=False)
            loaded = teachers.get(index)
            for (name, *_rest), (arr, _sha) in zip(teachers.specs, loaded):
                kl = token_kl(torch.from_numpy(arr).to(device), raw)
                assert kl.dtype == torch.float64
                np.save(out / f"kl.{name}.npy", kl.cpu().numpy(), allow_pickle=False)
            progress(mode, "teachers", scored=True)

        def visitor(layer, forward_batch):
            rec = sub.before(layer)
            try:
                forward_batch(inputs[0])
            finally:
                sub.after(layer, rec)
            progress(mode, f"layer-{layer + 1:02d}" if layer + 1 < spec["layers"] else "teachers")

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False) as profiler:
            with torch.inference_mode():
                runner.visit_layer_batches(inputs, visitor, output_consumer=consume)
        profiler.export_chrome_trace(str(out / "runner.trace.json"))
        assert len(sub.records) == spec["layers"] and sub.hash.all_matched
        assert sub.copied == sum(map(len, visited.values()))
        assert any(r["kind"] == "routed" for rows in visited.values() for r in rows)
        results = {"window_id": wid, "layers": spec["layers"], "manifest_sha256": sha,
                   "source_readset": reads.stats, "omitted_roster": omitted_roster,
                   "source_verified": sub.source_checked, "rendered_verified": sub.rendered_checked,
                   "copied": sub.copied, "all_hashes_matched": sub.hash.all_matched,
                   "layers_detail": sub.records, "hash_copy_profile": sub.hash.profile,
                   "runtime_stamp": G.run_metadata_stamp()}
    finally:
        G.close_arm_inputs(runner, sub.hash, teachers, [sub.reader])
    results["residency"] = receipt()
    results.update(start_unix=started, end_unix=G.time.time())
    G.write_g3_record(results, out / "result.json")


def compare_real_arm_passes(spec, out):
    import numpy as np
    import time
    out = Path(out)
    out.mkdir(exist_ok=False)
    started = time.time()
    for mode in MODES:
        subprocess.run([sys.executable, __file__, "--pass-mode", mode, "--spec", json.dumps(spec),
                        "--out", str(out / mode)], check=True)
    equality = {}
    for name in ("logits.npy", "kl.teacher04.npy", "kl.teacher2.npy"):
        a = np.load(out / MODES[0] / name, mmap_mode="r", allow_pickle=False)
        b = np.load(out / MODES[1] / name, mmap_mode="r", allow_pickle=False)
        equality[name] = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))
        assert equality[name], f"real runner bytes differ: {name}"
    passes = {mode: json.loads((out / mode / "result.json").read_bytes()) for mode in MODES}
    assert passes["source-read"]["source_readset"]["omitted_tensors"] == 0
    assert passes["omission"]["source_readset"]["omitted_tensors"] > 0
    assert passes["source-read"]["copied"] == passes["omission"]["copied"]
    result = {"schema": "campaign.g3.real_arm_comparison.v1", "arm": spec["arm"],
              "start_unix": started, "end_unix": time.time(), "bitwise_equal": equality, "passes": passes,
              "scope": "One original TR3 window through four real GLM layers and the pinned head. Not full-model quality.",
              "work_per_joule": None, "performance_claim": None}
    G.write_g3_record(result, out / "comparison.json")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pass-mode", choices=MODES)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec[1:]).read_bytes()) if args.spec.startswith("@") else json.loads(args.spec)
    if spec["arm"] != "a8_w" or spec["layers"] != 4 or not 0 <= spec["window"] < 25:
        parser.error("the real comparison uses a8_w, four layers, and one original panel window")
    if args.prepare:
        prepare_real_arm_plan(spec, args.out)
    elif args.pass_mode:
        from tessera.unit_artifact import read_unit_artifact
        spec["decoder"] = read_unit_artifact
        run_pass(spec, args.out, args.pass_mode)
    else:
        compare_real_arm_passes(spec, args.out)


if __name__ == "__main__":
    main()
