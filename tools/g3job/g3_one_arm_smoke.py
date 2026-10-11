"""Compare one tiny arm on retained source bytes and on omitted source bytes.

This comparison uses the pinned FP64 loss function. It is not a promotion gate.
It does not qualify a whole-model arm, a decoder, or a performance result.
"""
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

import g3_lib as L
from g3_offline_decoded_kl import HashPool, PQ
from g3_readset import SourceReads


def compare_one_arm(root, *, device="cpu", profiles=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    source = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
    save_file({name + ".weight": source}, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name + ".weight": "model.safetensors"}}))
    replacement = source.flip(0).to(device)
    row = {"qname": name, "kind": "routed", "role": "gate_proj", "expert": 0,
           "source_sha256": L.tensor_sha256(source),
           "a8_rendered_shape": [2, 4], "a8_rendered_sha256": L.tensor_sha256(replacement)}
    inputs = torch.arange(12, device=device, dtype=torch.bfloat16).reshape(3, 4)
    teacher = torch.tensor([[0.3, -0.1], [0.9, 0.4], [-0.5, 0.7]], device=device)
    sys.path.insert(0, str(PQ))
    from experiments.glm_tr3_full_vocab import token_kl
    outputs, losses, receipts = [], [], []
    for label, arms in (("source-read", ["null", "a8_w"]), ("source-omission", ["a8_w"])):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as profiler:
            reads = SourceReads(root, arms, {0: [row]})
            with reads.open(str(root / "model.safetensors"), framework="pt", device=device) as handle:
                weight = handle.get_tensor(name + ".weight")
            pool = HashPool(max_inflight_bytes=256)
            try:
                if not reads.omitted:
                    pool.submit(weight, row["source_sha256"], "used source")
                    pool.drain()
                weight.copy_(replacement)
                pool.submit(weight, row["a8_rendered_sha256"], "installed candidate")
                pool.drain()
                logits = inputs.float() @ weight.float().T
                loss = token_kl(teacher, logits, require_cuda=device == "cuda")
                if not pool.all_matched:
                    raise RuntimeError("The one-arm digest checks did not complete")
            finally:
                pool.close()
            outputs.append(logits)
            losses.append(loss)
            receipts.append({"path": label, "source_reads": reads.stats,
                             "hashes": pool.checked, "hash_copy_profile": pool.profile})
        if profiles:
            profiler.export_chrome_trace(str(root / (label + ".trace.json")))
    equal_logits = torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    equal_kl = torch.equal(losses[0].view(torch.uint8), losses[1].view(torch.uint8))
    if not equal_logits or not equal_kl:
        raise RuntimeError("Source omission changed the one-arm logits or FP64 KL bytes")
    return {"logits_bitwise_equal": equal_logits, "kl_bitwise_equal": equal_kl,
            "positions": 3, "vocabulary": 2, "device": device, "paths": receipts,
            "scope": "one tiny Linear arm; no whole-model or performance qualification"}
