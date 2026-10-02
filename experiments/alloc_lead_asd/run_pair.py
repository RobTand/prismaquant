"""Bounded issue1962 FP32/BF16 screen; no production estimator correction.

The dtype pair changes both primal and backward paths. An elevated BF16 price
is a lead for a backward-only causal control, never proof of cotangent noise.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.initialization import no_init_weights
from safetensors.torch import load as load_safetensors

from experiments.alloc_lead_asd import a_side_diag as diag
from prismaquant.joint_aura import source_execution_identity
from prismaquant.residency_map import bind_residency_manifest, residency_report
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_whole_file
from prismaquant.prismabuild_progress import commit as report_progress


MODEL_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
INPUT_SHA = "74ddf32461f4c041d3743db280aa0df5a7de69937a5dc92208623a1173bdbd68"


def file_sha(path):
    with open(path, "rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = os.fstat(handle.fileno())
    fence = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if fence(before) != fence(after) or fence(after) != fence(os.stat(path)):
        raise RuntimeError(f"source changed while reading {path}")
    return digest


def runtime_identity(model):
    forwards = {}
    for module in model.modules():
        cls = type(module)
        key = f"{cls.__module__}.{cls.__qualname__}"
        if key in forwards or not cls.__module__.startswith("transformers"):
            continue
        forward = module.forward
        function = getattr(forward, "__func__", forward)
        path = inspect.getsourcefile(function)
        forwards[key] = {
            "effective_forward": f"{function.__module__}.{function.__qualname__}",
            "source_path": path,
            "source_sha256": file_sha(path) if path else None,
        }
    return {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "source_execution": source_execution_identity(model), "forwards": forwards,
        "config": model.config.to_dict(),
    }


def load_source_model(config, state_dict, *, dtype, device):
    # Meta construction leaves nonpersistent rotary buffers unmaterialized.
    # Ordinary construction under the library's no-init context creates those
    # buffers while avoiding random weight fills. Copy into the requested
    # parameter dtype, then move devices without casting FP32 rotary buffers.
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, attn_implementation="sdpa")
    loaded = model.load_state_dict(state_dict, strict=False)
    expected_missing = (["lm_head.weight"] if config.tie_word_embeddings
                        and "lm_head.weight" not in state_dict else [])
    if loaded.missing_keys != expected_missing or loaded.unexpected_keys:
        raise RuntimeError(f"source state coverage differs: {loaded}")
    model.tie_weights()
    if any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()]):
        raise RuntimeError("source state left a meta tensor")
    return model.to(device=device).eval()


def load_staged_inputs(manifest_sha256):
    """Use the existing PB lease reader once for the diagnostic's source inputs."""
    model_path = Path("/mnt/shared/models/qwen3-small-alloc-lead/Qwen3-0.6B")
    input_path = Path("/mnt/shared/tessera-measurements/alloc-lead-asd/inputs_qwen3.safetensors")
    activate_staged_tier_policy("ram,ssd")
    bind_residency_manifest(manifest_sha256)
    manifest_path = Path(__file__).with_name("recipes") / "pair-inputs.json"
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
        raise RuntimeError("sealed data manifest differs from diagnostic recipe")
    manifest = json.loads(manifest_raw)
    bindings = {row["path"]: row["sha256"] for row in manifest["entries"]}

    def read(path):
        raw = read_staged_whole_file(path, bindings[str(path)], label="surrogate-screen-input")
        if hashlib.sha256(raw).hexdigest() != bindings[str(path)]:
            raise RuntimeError(f"staged source digest differs: {path}")
        return raw

    config_dict = json.loads(read(model_path / "config.json"))
    if config_dict.get("model_type") != "qwen3":
        raise RuntimeError("model source config is not Qwen3")
    config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
    generation_config = json.loads(read(model_path / "generation_config.json"))
    raw = read(model_path / "model.safetensors")
    state_dict = load_safetensors(raw)
    del raw
    raw = read(input_path)
    tokens = load_safetensors(raw)
    del raw
    ids = tokens["fit_s42"][:4].contiguous()
    del tokens
    if transformers.__version__ != "5.16.1":
        raise RuntimeError("producer Transformers version differs from recovered runtime")
    return config, generation_config, state_dict, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--only-dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--deterministic-backward", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, generation_config, state_dict, ids = load_staged_inputs(args.data_manifest_sha256)
    model_path = Path("/mnt/shared/models/qwen3-small-alloc-lead/Qwen3-0.6B")
    input_path = Path("/mnt/shared/tessera-measurements/alloc-lead-asd/inputs_qwen3.safetensors")
    identity = {
        "schema": "prismaquant.research.surrogate_dtype_pair.v1",
        "screen_only": True, "n_sequences": 4, "sequence_length": 512,
        "n_probes": 8, "seeds": list(range(7000, 7008)), "text": "fit_s42",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "model_sha256": MODEL_SHA, "inputs_file_sha256": INPUT_SHA,
        "data_manifest_sha256": args.data_manifest_sha256,
        "input_residency": residency_report(),
        "generation_config": generation_config,
        "source_commit": os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"),
        "container_content_sha256": os.environ.get("PRISMAQUANT_CONTAINER_CONTENT_SHA256"),
        "legs": {}, "started_unix": time.time(),
    }
    dtypes = [args.only_dtype] if args.only_dtype else ["float32", "bfloat16"]
    identity["requested_dtypes"] = dtypes
    identity["completion_scope"] = "requested_dtype_legs"
    identity["deterministic_backward"] = args.deterministic_backward
    if tuple(ids.shape) != (4, 512):
        raise RuntimeError("input prefix geometry mismatch")
    identity["input_prefix_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    diag.atomic_json_dump(identity, str(output / "inputs.ready.json"))
    report_progress(1, "startup", "source_input_identity")
    for leg_index, dtype in enumerate(dtypes):
        def progress(phase, count):
            base = 1 + leg_index * 8 + (4 if phase == "arms" else 0)
            report_progress(base + count, f"{phase}_{dtype}", "durable_sequence_blocks")

        torch.manual_seed(0)
        model = load_source_model(config, state_dict, dtype=getattr(torch, dtype), device="cuda")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        hook = model.get_input_embeddings().register_forward_hook(
            lambda module, inputs, result: result.requires_grad_(True))
        runtime = runtime_identity(model)
        with torch.no_grad():
            z0 = model(input_ids=ids[:1].to("cuda"), use_cache=False).logits
            z1 = model(input_ids=ids[:1].to("cuda"), use_cache=False).logits
            finite = bool(torch.isfinite(z0).all() & torch.isfinite(z1).all())
            identical = torch.equal(z0, z1)
            maxdiff = float((z0.float() - z1.float()).abs().max())
        del z0, z1
        if not finite or not identical:
            raise RuntimeError(f"{dtype} repeated clean forward fails null gate")
        stem = str(output / dtype)
        job = argparse.Namespace(model=str(model_path), inputs=str(input_path), text="fit_s42",
            n_seqs=4, n_probes=8, seed_base=7000, dtype=dtype, n_single=0,
            layer_arms=False, dz_dtype="float32", pricing_from=None, profile=True,
            smoke_first=False, output=stem, deterministic_backward=args.deterministic_backward)
        diag.run(job, model, ids=ids, progress_callback=progress)
        meta = json.loads(Path(stem + ".json").read_text())
        if (meta.get("complete") is not True or meta["ids_sha256"] != identity["input_prefix_sha256"]
                or meta["seeds"] != identity["seeds"] or meta["n_global"] != 2048):
            raise RuntimeError("completed leg identity mismatch")
        identity["legs"][dtype] = {
            "stem": stem, "runtime": runtime, "units": meta["units"],
            "null_forward": {"finite": finite, "bitwise_equal": identical, "max_abs_diff": maxdiff},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        hook.remove()
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        diag.atomic_json_dump(identity, str(output / "pair.partial.json"))
    if len(dtypes) == 2 and identity["legs"]["float32"]["units"] != identity["legs"]["bfloat16"]["units"]:
        raise RuntimeError("dtype pair unit roster mismatch")
    identity["finished_unix"] = time.time()
    identity["complete"] = True
    diag.atomic_json_dump(identity, str(output / "pair.json"))
    report_progress(2 + len(dtypes) * 8, "publish", "requested_dtype_legs")
    files = sorted(p for p in output.iterdir() if p.is_file())
    print(json.dumps({"complete": True, "result": str(output / "pair.json"),
                      "files": [{"name": p.name, "size": p.stat().st_size,
                                 "sha256": file_sha(p)} for p in files]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
