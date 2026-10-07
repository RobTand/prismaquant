"""Measure synthetic paired-trade reports through the actual allocator command."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import pstats
import resource
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prismaquant import format_registry as fr, joint_aura as joint
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX, DIRECT_UTF8_STRICT, bytes_sha256hex, file_sha256hex
from prismabuild import client as pb
from prismaquant.layer_config import load_assignment

LOW, HIGH = "FP8_E4M3", "FP8_E5M2"
ROLES = ("gate", "up", "down")


def arrays(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"difference_per_probe", "group_differences", "assignment_a", "assignment_b", "probe_ids"}:
                yield key
            yield from arrays(value)
    elif isinstance(node, list):
        for value in node:
            yield from arrays(value)


def samples(layer, expert, role, probes):
    base = 1 + ((layer * 13 + expert * 3 + role * 5) % 17) / 16
    low = [base + ((k * 7 + layer * 11 + expert * 13 + role * 17) % 19 - 9) / 32
           for k in range(probes)]
    return low, [x + 2 + role / 16 for x in low]


def fixture(directory, config, probes, layers):
    import torch
    c = config["text_config"]
    source_content = {"config": config, "weight_map": {"synthetic": "none"},
                      "shards": [{"path": "/synthetic/no-weights", "size": 1, "sha256": "a" * 64}]}
    source_model = {"schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic GLM-5.3-Flash dimensions",
                    "resolved_commit": None, "content_sha256": joint.identity_sha256(source_content), **source_content}
    arithmetic = joint.arithmetic_identity(torch.float32)
    probe = {"schema": "prismaquant.joint_aura.probes.v2", "seed_base": 228600,
             "n_probes": probes, "calibration_sha256": "c" * 64, "producer_source_sha256": "d" * 64,
             "source_model": source_model, "distribution": "rademacher", "normalization": "global_kl_fisher",
             "temperature": 1.0, "arithmetic": arithmetic}
    costs, stats, identities = {}, {}, {LOW: {}, HIGH: {}}
    for layer in layers:
        for expert in range(c["n_routed_experts"]):
            for role, label in enumerate(ROLES):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{label}_proj"
                shape = [c["moe_intermediate_size"], c["hidden_size"]]
                if label == "down":
                    shape.reverse()
                n = math.prod(shape)
                stats[name] = {"h_trace": 1.0, "n_params": n, "in_features": shape[1], "out_features": shape[0]}
                rows = {}
                for fmt, signed in zip((LOW, HIGH), samples(layer, expert, role, probes)):
                    operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": name, "format": fmt,
                                "probe_identity_sha256": joint.identity_sha256(probe),
                                "source_weight": {"content_sha256": joint.identity_sha256({"synthetic_source": name}),
                                                  "shape": shape, "dtype": "torch.float32", "logical_bytes": n * 4},
                                "rendered_weight": {"content_sha256": joint.identity_sha256({"synthetic_render": name, "format": fmt}),
                                                    "shape": shape, "dtype": "torch.float32", "logical_bytes": n * 4},
                                "activation": joint.activation_identity(fr.get_format(fmt), {}, name), "arithmetic": arithmetic}
                    rows[fmt] = joint.make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
                        signed_components=[{"weight": x, "activation": 0.0, "mixed": 0.0, "total": x} for x in signed])
                    identities[fmt][name] = rows[fmt]["joint_operator_identity_sha256"]
                costs[name] = rows
    payload = {"costs": costs, "meta": {"formats": [LOW, HIGH]},
               "provenance": {"cost_mode": "aura", "joint_activation": True, "cost_currency": joint.JOINT_CURRENCY}}
    (directory / "probe.pkl").write_bytes(pickle.dumps({"stats": stats, "meta": {"model": None}}))
    (directory / "costs.pkl").write_bytes(pickle.dumps(payload))
    (directory / "baseline.json").write_text(json.dumps(dict.fromkeys(costs, HIGH)))
    return identities


def profile_allocator_command(directory, fixture_dir, refusal=False):
    directory.mkdir()
    launcher = "import cProfile, runpy, sys\nprofile_path = sys.argv.pop(1)\nsys.argv[0] = 'prismaquant.allocator'\nprofile = cProfile.Profile()\ntry:\n    profile.enable()\n    runpy.run_module('prismaquant.allocator', run_name='__main__')\nfinally:\n    profile.disable()\n    profile.dump_stats(profile_path)\n"
    command = [sys.executable, "-c", launcher, str(directory / "profile.pstats"),
               "--probe", str(fixture_dir / "probe.pkl"), "--costs", str(fixture_dir / "costs.pkl"),
               "--formats", f"{LOW},{HIGH}", "--allow-default-profile", "--target-bits", "9", "--pareto-targets", "9",
               "--bit-precision", "0.001", "--cost-baseline-assignment", str(fixture_dir / "baseline.json"),
               "--layer-config", str(directory / "layer.json"), "--pareto-csv", str(directory / "pareto.csv"),
               "--pareto-output-dir", str(directory / "seeds")]
    if refusal:
        command += ["--no-packed-aggregation", "--no-fused-aggregation"]
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        result = subprocess.run(["/usr/bin/time", "-v", "-o", str(directory / "time.txt"), *command],
                                stdout=stdout, stderr=stderr, env={**os.environ, "PRISMAQUANT_COST_UCB_Z": "0" if refusal else "1"})
    peak = next(int(line.split(":", 1)[1]) for line in (directory / "time.txt").read_text().splitlines()
                if "Maximum resident set size" in line)
    profile = pstats.Stats(str(directory / "profile.pstats"))
    owners = [{"file": Path(key[0]).name, "line": key[1], "function": key[2], "calls": value[1],
               "self_seconds": value[2], "cumulative_seconds": value[3]}
              for key, value in profile.stats.items()
              if key[2] in {"price_paired_rate_trade", "summarize_paired_rate_trade", "_solve_for_target_uncached",
                            "_write_layer_config", "solve_with_promotion"}]
    return {"returncode": result.returncode, "process_peak_kib": peak, "profile_owners": owners,
            "command": command, "output_bytes": {str(path.relative_to(directory)): path.stat().st_size
             for path in directory.rglob("*") if path.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "tools/pq2286_target_config.json")
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--layer-count", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    config_bytes = args.model_config.read_bytes()
    config = json.loads(config_bytes)
    c = config["text_config"]
    layers = list(range(c["first_k_dense_replace"], c["num_hidden_layers"]))
    if args.layer_count is not None:
        layers = layers[:args.layer_count]
    fixture_dir = args.output / "fixture"
    fixture_dir.mkdir()
    identities = fixture(fixture_dir, config, args.probes, layers)
    fixture_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gc.collect()
    normal = profile_allocator_command(args.output / "allocation", fixture_dir)
    assert normal["returncode"] == 0, normal
    layer_path = args.output / "allocation/layer.json"
    selected = load_assignment(layer_path)
    assert selected == dict.fromkeys(identities[LOW], LOW)
    trade = json.loads(layer_path.read_text())["__prismaquant__"]["paired_rate_trade"]
    assert trade["probe_ids"] == list(range(228600, 228600 + args.probes))
    for key, fmt in (("assignment_a", LOW), ("assignment_b", HIGH)):
        assert trade[key]["operator_identity_sha256_by_unit"] == identities[fmt]
        assert trade[key]["assignment_identity_sha256"] == joint.identity_sha256(identities[fmt])
    expected = [math.fsum(sign * 0.5 * x[k] ** 2 for layer in layers for expert in range(c["n_routed_experts"])
                          for role in range(3) for sign, x in zip((1, -1), samples(layer, expert, role, args.probes)))
                for k in range(args.probes)]
    assert trade["difference_per_probe"] == expected
    assert trade["refused"] is False
    normal["assignment_sha256"] = DIRECT_UTF8_STRICT.sha256(selected)
    normal["trade_sha256"] = DIRECT_UTF8_STRICT.sha256(trade)
    normal["difference_per_probe"] = trade["difference_per_probe"]
    normal["math"] = {key: trade[key] for key in ("mean_difference", "paired_standard_error", "predicted_dloss", "hedged_difference")}
    report = json.loads((args.output / "allocation/format_applicability.json").read_text())
    normal["report_full_array_count"] = len(list(arrays(report["paired_rate_trades"])))
    del report, trade, selected
    with (fixture_dir / "costs.pkl").open("rb") as source:
        payload = pickle.load(source)
    for name, rows in payload["costs"].items():
        if int(name.split(".experts.")[1].split(".")[0]) != 0:
            row = rows[HIGH]
            signed = [component["total"] + 1 / 4096 for component in rows[LOW]["signed_components_per_probe"]]
            rows[HIGH] = joint.make_joint_aura_entry(operator_identity=row["joint_operator_identity"],
                probe_identity=row["probe_identity"], signed_components=[
                    {"weight": x, "activation": 0.0, "mixed": 0.0, "total": x} for x in signed])
    (fixture_dir / "costs.pkl").write_bytes(pickle.dumps(payload))
    del payload
    gc.collect()
    refusal = profile_allocator_command(args.output / "refusal", fixture_dir, refusal=True)
    assert refusal["returncode"] != 0
    assert not (args.output / "refusal/layer.json").exists()
    stderr = (args.output / "refusal/stderr.txt").read_text()
    refused = json.loads(stderr.split("routed layer rate trade refused: ", 1)[1])
    assert set(refused) == {f"model.layers.{layer}.mlp.experts" for layer in layers}
    assert all(row["refused"] and row["dominant_experts"] == ["0"] for row in refused.values())
    refusal["report_full_array_count"] = len(list(arrays(refused)))
    refusal["verdict"] = {layer: {key: row[key] for key in ("refused", "refusal_reason", "dominant_experts", "mean_difference", "paired_standard_error")}
                          for layer, row in refused.items()}
    proof = {"schema": "pq2286.actual_cli_structural_proof.v1", "synthetic": True, "scientific_allocation": False,
             "model": "GLM-5.3-Flash-BF16", "source_config": str(args.model_config),
             "config_sha256": bytes_sha256hex(config_bytes), "dimensions": {
                 "total_layers": c["num_hidden_layers"], "dense_layers": c["first_k_dense_replace"], "routed_layers": layers,
                 "experts_per_layer": c["n_routed_experts"], "roles": list(ROLES), "units": len(identities[LOW]),
                 "hidden_size": c["hidden_size"], "moe_intermediate_size": c["moe_intermediate_size"], "probes": args.probes},
             "fixture_process_peak_kib": fixture_peak, "allocation": normal, "refusal": refusal,
             "native_threads": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    (args.output / "proof.json").write_text(json.dumps(proof, indent=2))
    archive = args.output / "proof.tar.gz"
    queue = pb.PoolQueue(os.environ["PRISMABUILD_QUEUE_ROOT"])
    owner = os.environ["PRISMABUILD_ACTION_KEY"]
    template = pb.declared_template(queue, owner)
    assert template.get("write_only") is True
    assert len(template["permitted_tiers"]) == 1
    tier = template["permitted_tiers"][0]
    instance = pb.bind_declared_instance(
        queue, owner_action_key=owner, claim_snapshot=pb.read_claimed_record(queue, owner))
    pb.declare_instance(queue.root, instance)
    admitted = pb.admit_instance(queue, instance, template)
    assert admitted["ok"], admitted
    prewrite = pb.require_prewrite(
        queue, instance, template, batch_id="cli-proof", tier=tier,
        class_bytes={"payload": pb.checked_instance_maxima(template)["payload_max_bytes"],
                     "checkpoint": 0, "temp": 0}, paths=[str(archive.resolve())])
    assert prewrite["ok"], prewrite
    with tarfile.open(archive, "w:gz") as target:
        for child in ("allocation", "refusal", "proof.json"):
            target.add(args.output / child, arcname=child)
    descriptor = pb.validate_descriptor({
        "schema": pb.DESCRIPTOR_SCHEMA_V2, "slot": "cli_proof", "artifact_class": "payload",
        "path": str(archive.resolve()), "bytes": archive.stat().st_size,
        "sha256": file_sha256hex(archive), "producer_generation": instance["owner_attempt"]["nonce"],
        "owner_action_key": owner, "owner_attempt": instance["owner_attempt"],
    }, template, instance)
    publication = pb.commit_origin_batch(
        queue, instance, template, [descriptor], batch_id="cli-proof", lifetime="retain")
    assert publication["ok"], publication
    proof["artifact_bundle"] = descriptor
    proof["artifact_publication"] = publication
    print(DIRECT_ASCII_SPACED_LAX.text(proof))


if __name__ == "__main__":
    main()
