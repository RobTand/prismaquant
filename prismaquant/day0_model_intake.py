"""Day-zero source metadata intake. A draft is not a registered profile."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .cost_stage_checkpoint import atomic_write_bytes
from .digests import indent2_json_file_bytes
from .export_structure import _read_metadata, verify_export_structure
from .model_profiles.default import DefaultProfile
from .model_profiles.registry import detect_profile
from .model_profiles.structure import ModelStructureSpec, SCHEMA
from .name_projection import DECLARED_OUT_OF_GRAPH, NameProjection, is_packed_expert_qname, strip_weight_leaf
from .shipcard import _strict_json_object, safetensors_header_spans
from .source_read_plan import read_safetensors_header, roster_layers_prefix


def _config(root: Path) -> tuple[dict, dict, dict]:
    config = _strict_json_object((root / "config.json").read_bytes(), where="intake config.json")
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("config.model_type must be a nonempty string")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures or any(
        not isinstance(name, str) or not name for name in architectures
    ):
        raise ValueError("config.architectures must be a nonempty list of strings")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("config.text_config must be an object")
    fields = {"layers": "num_hidden_layers", "hidden_size": "hidden_size", "vocab_size": "vocab_size"}
    dimensions = {}
    for name, field in fields.items():
        value = text.get(field)
        if type(value) is not int or value < 1:
            raise ValueError(f"config.{field} must be a positive integer")
        dimensions[name] = value
    return config, text, dimensions


def _source(root: Path, *, metadata_only: bool) -> tuple[dict, dict, dict]:
    """Use the existing partial-index and physical-header owners."""
    index = root / "model.safetensors.index.json"
    if index.exists():
        from .prismasnap_checkpoint import _Checkpoint

        # Strict JSON refusal precedes the checkpoint owner's index grammar.
        _strict_json_object(index.read_bytes(), where="intake source index")
        source = _Checkpoint(root, require_all_shards=not metadata_only)
        weight_map = source.weight_map
        document = source.index
        available = sorted(source.available_shards)
    elif (root / "model.safetensors").exists():
        verify_export_structure(root)
        header, _, _ = read_safetensors_header(str(root / "model.safetensors"))
        weight_map = {name: "model.safetensors" for name in header if name != "__metadata__"}
        document = {"weight_map": weight_map}
        available = ["model.safetensors"]
    elif metadata_only:
        return {}, {}, {"scope": "config_only", "shards": 0, "tensors": 0, "headers_read": 0}
    else:
        raise ValueError("checkpoint lacks model.safetensors or its source index")
    if any(not name.endswith(".safetensors") for name in weight_map.values()):
        raise ValueError("source weight_map must name safetensors shards")
    complete = set(available) == set(weight_map.values())
    if complete:
        metadata = document.get("metadata", {})
        size_basis = "data" if isinstance(metadata, dict) and "total_size" in metadata else None
        verify_export_structure(root, index_size_basis=size_basis)
    tensors = {}
    for shard in available:
        raw, base, observed = _read_metadata(root / shard, shard=True)
        safetensors_header_spans(raw, data_bytes=observed["bytes"] - base, where=str(root / shard))
        header = _strict_json_object(raw, where=str(root / shard))
        rows = {name: row for name, row in header.items() if name != "__metadata__"}
        expected = {name for name, owner in weight_map.items() if owner == shard}
        if set(rows) != expected:
            raise ValueError(f"source index/header tensor roster differs in {shard}")
        tensors.update(rows)
    return document, tensors, {
        "scope": "headers_and_index_only" if complete else "index_only",
        "shards": len(set(weight_map.values())), "tensors": len(weight_map), "headers_read": len(available),
    }


def inspect_checkpoint(root: Path, *, metadata_only: bool = False) -> tuple[dict, dict]:
    """Return a grammar-valid draft and a report without loading tensor data."""
    config, text, dimensions = _config(root)
    document, tensors, source = _source(root, metadata_only=metadata_only)
    profile = detect_profile(str(root), config=config)
    if document:
        profile._declare_checkpoint_index(document)
    registered = not isinstance(profile, DefaultProfile)
    spec = profile.structure_spec() if registered else None
    unsupported = []
    if not registered:
        unsupported.append({"kind": "unregistered_architecture", "architectures": config["architectures"]})
    if not document:
        unsupported.append({"kind": "source_metadata_unavailable", "reason": "the repository has no safetensors index"})
    weight_map = document.get("weight_map", {})
    projection = NameProjection(profile, checkpoint_keys=tuple(weight_map))
    live_sources = {}
    groups = Counter()
    body_names = []
    for name in weight_map:
        mapped = projection.checkpoint_to_live(name)
        if mapped.outcome == DECLARED_OUT_OF_GRAPH:
            groups["declared_out_of_graph"] += 1
            continue
        live = mapped.target
        if live in live_sources and live_sources[live] != name:
            raise ValueError(f"source names collide at live tensor {live}: {live_sources[live]} and {name}")
        live_sources[live] = name
        # MTP and visual namespaces are separate profile contracts.
        if registered:
            if name.startswith(profile.source_tensor_name(profile.body_layer_prefix() + ".")):
                body_names.append(name)
        elif ".layers." in name:
            body_names.append(name)
        unit = strip_weight_leaf(live)
        recipe = profile.live_to_recipe_name(unit)
        group = projection.serving_group(recipe)
        groups[group.kind] += 1
        row = tensors.get(name)
        if row is not None:
            shape = row["shape"]
            if len(shape) >= 3 and not (
                is_packed_expert_qname(recipe) and profile.packed_expert_role_group(recipe) is not None
            ):
                unsupported.append({"kind": "unclassified_parameter", "tensor": name, "shape": shape})
            if is_packed_expert_qname(recipe) and profile.packed_expert_role_group(recipe) is None:
                unsupported.append({"kind": "undeclared_expert_projection", "tensor": name, "shape": shape})
    prefix = None
    if weight_map:
        prefix = roster_layers_prefix(body_names)
        layers = {int(name[len(prefix):].split(".", 1)[0]) for name in body_names}
        if layers != set(range(dimensions["layers"])):
            raise ValueError(f"config.num_hidden_layers={dimensions['layers']} differs from source layers {sorted(layers)}")
    for accessor in (profile.embedding_name, profile.lm_head_name):
        key = profile.source_tensor_name(accessor() + ".weight")
        row = tensors.get(key)
        if row is not None and row["shape"] != [dimensions["vocab_size"], dimensions["hidden_size"]]:
            raise ValueError(f"config.vocab_size/hidden_size differs from {key} shape {row['shape']}")
    if spec is not None:
        # The shipped grammar owns the representation; preserve its naming rules.
        resource = Path(__file__).parent / "model_profiles/specs" / (spec.id + ".json")
        draft = _strict_json_object(resource.read_bytes(), where="registered structure spec")
    else:
        draft = {"schema": SCHEMA, "id": config["model_type"], "naming": {}, "fused_groups": [],
                 "pinned_names": [], "passthrough_prefixes": []}
    draft = {key: value for key, value in draft.items() if not key.startswith("_")}
    draft["match"] = {"model_type": [config["model_type"]], "architectures": config["architectures"]}
    draft["supported_lanes"] = []
    draft.pop("preferred_lane", None)
    draft.pop("default_serving_profile", None)
    if prefix is not None:
        draft["shard_regexes"] = {**draft.get("shard_regexes", {}), "body_layer_prefix": prefix.rstrip(".")}
    ModelStructureSpec.from_dict(draft)
    report = {
        "schema": "prismaquant.day0_intake.v1", "status": "draft_only", "checkpoint": str(root),
        "model_type": config["model_type"], "architectures": config["architectures"],
        "profile": profile.name, "profile_registered": registered, "native_serving_qualified": False,
        "dimensions": dimensions, "source": source, "profile_groups": dict(sorted(groups.items())),
        "unsupported_module_kinds": unsupported, "runtime_checks_executed": [],
        "runtime_checks_not_executed": ["BF16 tensor-parallel degree two load and generation", "native serving qualification"],
        "bf16_tp2": {"status": "not_requested"},
    }
    return draft, report


def _download(args, *, full: bool) -> Path:
    from huggingface_hub import snapshot_download

    options = {"repo_id": args.model_id, "revision": args.revision, "local_dir": str(args.download_dir),
               "max_workers": 1}
    if args.hub_endpoint:
        options["endpoint"] = args.hub_endpoint
    if not full:
        options["allow_patterns"] = ["config.json", "model.safetensors.index.json"]
    else:
        options["allow_patterns"] = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"]
    return Path(snapshot_download(**options)).resolve()


def bf16_tp2_command(args, root: Path) -> list[str]:
    """Use Docker and the existing vLLM prompt check, not another scheduler."""
    code = Path(__file__).resolve().parents[1]
    command = ["docker", "run", "--rm", "--gpus", args.runtime_gpus, "--network=host", "--ipc=host",
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--mount", f"type=bind,src={root},dst=/model,readonly",
               "--mount", f"type=bind,src={code},dst=/source,readonly",
               "--mount", f"type=bind,src={args.output.resolve()},dst=/check",
               "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1"]
    if args.ray_address:
        command += ["--env", f"RAY_ADDRESS={args.ray_address}"]
    command += ["--entrypoint", args.runtime_python, args.runtime_image,
                "/source/tools/vllm_prompt_smoke.py", "--model", "/model", "--dtype", "bfloat16",
                "--tensor-parallel-size", "2", "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--output-json", "/check/bf16-tp2.json"]
    if args.ray_address:
        command += ["--distributed-executor-backend", "ray"]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model-id")
    parser.add_argument("--revision", help="Full model repository commit, not a branch or tag")
    parser.add_argument("--download-dir", type=Path, help="Hugging Face local_dir; no separate artifact cache")
    parser.add_argument("--hub-endpoint", help="Optional Hugging Face Hub endpoint")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bf16-tp2", action="store_true", help="Download remaining weights and execute the real runtime check")
    mode.add_argument("--bf16-tp2-preflight", action="store_true", help="Validate the same invocation on CPU; do not launch")
    parser.add_argument("--runtime-image")
    parser.add_argument("--runtime-python", default="/usr/bin/python3")
    parser.add_argument("--runtime-gpus", help="Docker GPU selection; supplied only for the explicit runtime path")
    parser.add_argument("--ray-address", help="Address of an existing Ray cluster, if degree two spans hosts")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    args = parser.parse_args(argv)
    if args.model_id and (not args.revision or re.fullmatch(r"[0-9a-fA-F]{40}", args.revision) is None):
        parser.error("--revision must be a full forty-character repository commit")
    if args.model_id and not args.download_dir:
        parser.error("--download-dir is required with --model-id")
    if args.checkpoint and (args.revision or args.download_dir or args.hub_endpoint):
        parser.error("local checkpoint inputs cannot include remote download inputs")
    runtime = args.bf16_tp2 or args.bf16_tp2_preflight
    if runtime and not args.runtime_image:
        parser.error("--runtime-image is required for the BF16 degree-two check")
    if runtime and not args.runtime_gpus:
        parser.error("--runtime-gpus is required for the BF16 degree-two check")
    if not runtime and (args.runtime_image or args.runtime_gpus or args.ray_address):
        parser.error("runtime inputs require --bf16-tp2 or --bf16-tp2-preflight")
    if args.max_model_len < 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("runtime length and memory utilization must be positive and within their ranges")
    try:
        root = _download(args, full=False) if args.model_id else args.checkpoint.resolve(strict=True)
        draft, report = inspect_checkpoint(root, metadata_only=bool(args.model_id))
        if runtime:
            config, text, _ = _config(root)
            if config.get("quantization_config") or text.get("quantization_config"):
                raise ValueError("the BF16 check requires an unquantized source checkpoint")
            limit = text.get("max_position_embeddings")
            if type(limit) is int and args.max_model_len > limit:
                raise ValueError("max-model-len exceeds config.max_position_embeddings")
        if args.bf16_tp2:
            if args.model_id:
                root = _download(args, full=True)
                draft, report = inspect_checkpoint(root)
            document, tensors, _ = _source(root, metadata_only=False)
            for name, row in tensors.items():
                if len(row["shape"]) >= 2 and row["dtype"] != "BF16":
                    raise ValueError(f"the BF16 check requires BF16 source weights: {name} is {row['dtype']}")
        report.update({"model_id": args.model_id, "revision": args.revision})
        if runtime:
            report["bf16_tp2"] = {"status": "cpu_preflight_only" if args.bf16_tp2_preflight else "requested",
                                   "command": bf16_tp2_command(args, root)}
        if args.bf16_tp2_preflight:
            smoke = Path(__file__).resolve().parents[1] / "tools/vllm_prompt_smoke.py"
            command = report["bf16_tp2"]["command"]
            invocation = command[command.index("/source/tools/vllm_prompt_smoke.py") + 1:]
            invocation[invocation.index("--model") + 1] = str(root)
            output_arg = invocation.index("--output-json")
            del invocation[output_arg:output_arg + 2]
            checked = subprocess.run([sys.executable, str(smoke), "--cpu-preflight", *invocation],
                                     capture_output=True, text=True, check=True)
            report["bf16_tp2"]["preflight"] = json.loads(checked.stdout)
        atomic_write_bytes(args.output / "structure.draft.json", indent2_json_file_bytes(draft))
        atomic_write_bytes(args.output / "intake.json", indent2_json_file_bytes(report))
        if args.bf16_tp2:
            with (args.output / "bf16-tp2.log").open("wb") as log:
                result = subprocess.run(report["bf16_tp2"]["command"], stdout=log, stderr=subprocess.STDOUT)
            report["bf16_tp2"].update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
            report["runtime_checks_executed"] = ["vllm"]
            report["runtime_checks_not_executed"] = ["native serving qualification"]
            atomic_write_bytes(args.output / "intake.json", indent2_json_file_bytes(report))
            if result.returncode:
                return result.returncode
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"day-zero intake refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
