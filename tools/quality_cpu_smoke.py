"""Exercise both quality entry points with a real small CPU checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import torch

from prismaquant.digests import bytes_sha256hex, file_sha256hex
from prismaquant.quality_stage import artifact, write_result


def _smoke_json(path, value):
    write_result(path, value)
    return artifact(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model_path = args.model.resolve(strict=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    from prismaquant.production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity, render_production_weight
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="eager").eval()
    texts = ["The capital of France is Paris. The capital of Spain is Madrid. "*8,
             "Water is made of hydrogen and oxygen. Ice is frozen water. "*8]
    windows, arrays = [], []
    length = 32
    for index, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :length]
        if ids.shape != (1, length):
            raise ValueError("real smoke text has too few tokens")
        tokens_path = root/f"tokens-{index}.npy"
        np.save(tokens_path, ids[0].numpy().astype(np.int32), allow_pickle=False)
        with torch.inference_mode():
            logits = model(ids, use_cache=False).logits[0, :-1].float().numpy()
        path = root/f"teacher-{index}.npy"
        np.save(path, logits, allow_pickle=False)
        window_id = f"cpu-{index}"
        windows.append({"window_id": window_id, "tokens_path": str(tokens_path), "tokens_sha256": file_sha256hex(tokens_path)})
        arrays.append({"window_id": window_id, "path": str(path), "file_sha256": file_sha256hex(path),
                       "array_sha256": bytes_sha256hex(memoryview(logits).cast("B"))})
    qname = "model.layers.0.mlp.down_proj"
    source = model.get_submodule(qname).weight.detach()
    rendered = render_production_weight(source, "FP8_E4M3", qname=qname, activations={},
                                        levers={"gptq": False, "scale_sweep": False})
    row = {"qname": qname, "format": "FP8_E4M3", "source_sha256": _cb_cache_tensor_identity(source)["content_sha256"],
           "rendered_sha256": _cb_cache_tensor_identity(rendered)["content_sha256"]}
    cache = ProductionWeightCache(weights={(qname, "FP8_E4M3"): rendered.cpu()}, levers={"gptq": False, "scale_sweep": False})
    cache_path = root/"production-cache.pkl"
    cache_path.write_bytes(pickle.dumps(cache))
    identity = build_source_checkpoint_identity(model_path)
    protocol = _smoke_json(root/"protocol.json", {"schema": "prismaquant.g3_protocol/2", "name": "small-cpu-smoke",
        "context_length": length, "window_count": len(windows), "vocab_size": int(logits.shape[1]),
        "prefix_tokens": [], "prefix_ids": [], "tensor_parallel_size": 1, "tile_rows": 32})
    panel = _smoke_json(root/"panel.json", {"schema": "prismaquant.g3_panel/1", "windows": windows})
    teacher = _smoke_json(root/"teacher.json", {"windows": [row["window_id"] for row in windows],
        "prefix": {"ids": []}, "arrays": arrays, "source_model_identity": identity})
    g3 = {"schema": "prismaquant.g3_v2/1", "protocol": protocol,
        "tokenizer": artifact(model_path/"tokenizer.json"), "panel": panel, "teacher": teacher,
        "candidate": {"backend": "streamed", "model": str(model_path), "offload_folder": str(root/"offload"),
            "assignments": [row], "production_cache": artifact(cache_path), "max_resident_bytes": rendered.nbytes,
            "source_prefetch": {"max_cache_slots": 2, "prefetch_workers": 1, "prefetch_lookahead": 1,
                "cache_headroom_gb": 1, "prefetch_min_available_gb": 1, "require_prefetched_residency": True}},
        "device": "cpu", "criteria": []}
    _smoke_json(root/"g3.config.json", g3)
    del model, source, rendered, cache
    gc.collect()
    documents = root/"task-documents.jsonl"
    documents.write_text('\n'.join(json.dumps(row) for row in [
        {"prompt": "The capital of France is", "choices": [" Paris", " London"], "answer": 0},
        {"prompt": "Water freezes into", "choices": [" ice", " steam"], "answer": 0}])+"\n")
    task = {"schema": "prismaquant.task_suite/1", "backend": {"name": "hf", "pretrained": str(model_path),
        "tokenizer": str(model_path), "device": "cpu", "dtype": "float32", "batch_size": 1,
        "max_length": 128, "trust_remote_code": False}, "tasks": [{"task": "small_cpu_facts",
        "dataset_path": "json", "dataset_kwargs": {"data_files": {"train": str(documents)}},
        "test_split": "train", "output_type": "multiple_choice", "doc_to_text": "prompt",
        "doc_to_choice": "choices", "doc_to_target": "answer", "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True}]}],
        "sampling": {"limit": 2, "num_fewshot": 0, "random_seed": 0, "numpy_seed": 0,
                     "torch_seed": 0, "fewshot_seed": 0}, "criteria": []}
    _smoke_json(root/"tasks.config.json", task)
    if args.prepare_only:
        _smoke_json(root/"inputs.json", {"g3_config": artifact(root/"g3.config.json"),
                                      "task_config": artifact(root/"tasks.config.json"), "model": identity})
        return 0
    for module, config, output in (("g3_v2", "g3.config.json", "g3.json"),
                                    ("task_suite", "tasks.config.json", "tasks.json")):
        subprocess.run([sys.executable, "-m", "prismaquant."+module, "--config", str(root/config),
                        "--output", str(root/output)], check=True)
    _smoke_json(root/"smoke.json", {"schema": "prismaquant.quality_cpu_smoke/1", "model": identity,
        "g3": artifact(root/"g3.json"), "tasks": artifact(root/"tasks.json"),
        "device": "cpu", "skips": [], "qualification": "Execution only. No GLM quality or native kernel claim."})
    print(json.dumps({"smoke": str(root/"smoke.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
