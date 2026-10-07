"""Run an offline decoded G3 screen. This is not served KL."""
from __future__ import annotations

from contextlib import nullcontext
import io
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .digests import bytes_sha256hex, file_sha256hex
from .g3_numerics import token_kl
from .quality_stage import artifact, cli, read_binding, g3_candidate_binding


def _g3_json(binding, label):
    return json.loads(read_binding(binding, label))


def _g3_positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _array(row, shape, *, verify):
    path = Path(row["path"])
    before = path.stat()
    if verify and file_sha256hex(path) != row["file_sha256"]:
        raise ValueError(f"array checksum differs: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or tuple(array.shape) != tuple(shape):
        raise ValueError(f"array geometry or dtype differs: {path}")
    if verify:
        if not np.isfinite(array).all():
            raise ValueError(f"array contains nonfinite logits: {path}")
        if row.get("array_sha256") and bytes_sha256hex(memoryview(array).cast("B")) != row["array_sha256"]:
            raise ValueError(f"array data checksum differs: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"array changed during read: {path}")
    return array


def _g3_protocol(config):
    protocol = _g3_json(config["protocol"], "G3 protocol")
    if protocol.get("schema") != "prismaquant.g3_protocol/2":
        raise ValueError("unsupported G3 protocol schema")
    for name in ("context_length", "window_count", "vocab_size", "tensor_parallel_size", "tile_rows"):
        _g3_positive(protocol[name], name)
    length, count, vocab = (protocol[name] for name in ("context_length", "window_count", "vocab_size"))
    if length < 2 or vocab < 2:
        raise ValueError("G3 needs causal positions and more than one vocabulary column")
    return protocol


def _g3_panel_inputs(config, protocol):
    length, count, vocab = (protocol[name] for name in ("context_length", "window_count", "vocab_size"))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(read_binding(config["tokenizer"], "G3 tokenizer").decode("utf-8"))
    prefixes = protocol["prefix_tokens"]
    if not isinstance(prefixes, list) or not all(isinstance(x, str) for x in prefixes):
        raise ValueError("prefix_tokens must be an explicit token list")
    actual = [tokenizer.token_to_id(name) for name in prefixes]
    if actual != protocol["prefix_ids"] or any(type(value) is not int for value in actual):
        raise ValueError("actual tokenizer prefix IDs differ from the protocol")
    panel = _g3_json(config["panel"], "G3 panel")
    for name, expected in (("context_length", length), ("window_count", count), ("vocab_size", vocab)):
        if name in panel and panel[name] != expected:
            raise ValueError(f"panel {name} differs from protocol")
    windows = panel["windows"]
    names = [row["window_id"] for row in windows]
    if len(names) != count or len(set(names)) != count:
        raise ValueError("G3 window population is incomplete or duplicate")
    inputs = []
    for row in windows:
        raw = read_binding({"path": row["tokens_path"], "sha256": row["tokens_sha256"]}, "G3 tokens")
        tokens = np.load(io.BytesIO(raw), allow_pickle=False)
        if tokens.dtype != np.dtype("int32") or tokens.shape != (length,) or np.any(tokens < 0) or np.any(tokens >= vocab):
            raise ValueError("panel token geometry, dtype or vocabulary differs")
        inputs.append(torch.tensor([actual + tokens.tolist()], dtype=torch.int64))
    if "causal_mask_array" in panel:
        raw = read_binding({"path": panel["causal_mask_array"], "sha256": panel["causal_mask_sha256"]}, "G3 causal mask")
        mask = np.load(io.BytesIO(raw), allow_pickle=False)
        if mask.dtype != np.uint8 or mask.shape != (length,) or not np.all(mask == 1):
            raise ValueError("G3 requires unpadded causal inputs")
    return inputs, names, actual


def _g3_teacher(config, names, actual):
    teacher = _g3_json(config["teacher"], "G3 emitted teacher")
    for name in ("panel", "tokenizer"):
        if teacher.get(name, {}).get("sha256") != config[name]["sha256"]:
            raise ValueError(f"teacher {name} pairing differs from the current inputs")
    if teacher["windows"] != names or teacher["prefix"]["ids"] != actual:
        raise ValueError("teacher/panel prefix and window pairing differs")
    if [row["window_id"] for row in teacher["arrays"]] != names:
        raise ValueError("teacher array order differs from paired windows")
    return teacher


def _g3_fidelity(protocol):
    from tools.gold_measurement_fidelity import tr3_kl_fidelity
    fidelity = tr3_kl_fidelity(vocab_size=protocol["vocab_size"], n_windows=protocol["window_count"],
                             seqlen=protocol["context_length"])
    fidelity.update(instrument="prismaquant.g3_v2", execution="offline_decoded_forward",
        prefix_ids=protocol["prefix_ids"], scored_slice=[len(protocol["prefix_ids"]),
        protocol["context_length"]+len(protocol["prefix_ids"])-1])
    return fidelity


def prepare_g3(config, *, verify=False, check_device=True):
    """Read the declared population, actual prefix IDs and paired teacher."""
    protocol = _g3_protocol(config)
    length, count, vocab = (protocol[name] for name in ("context_length", "window_count", "vocab_size"))
    inputs, names, actual = _g3_panel_inputs(config, protocol)
    teacher = _g3_teacher(config, names, actual)
    for row in teacher["arrays"]:
        _array(row, (length-1, vocab), verify=verify)
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("G3 device must be CPU or CUDA")
    if check_device and device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("G3 requested CUDA without a visible device")
    candidate = config["candidate"]
    if candidate["backend"] == "retained_logits":
        retained = _g3_json(candidate["receipt"], "G3 candidate receipt")
        if retained["windows"] != names or [row["window_id"] for row in retained["arrays"]] != names:
            raise ValueError("candidate array order differs from paired windows")
        for row in retained["arrays"]:
            _array(row, (length-1, vocab), verify=False)
    elif candidate["backend"] == "streamed":
        retained = None
        from .stage_inputs import source_prefetch
        source_prefetch(candidate)
        _g3_positive(candidate["max_resident_bytes"], "max_resident_bytes")
        if not isinstance(candidate["assignments"], list):
            raise ValueError("assignments must be an explicit array; empty denotes the source control")
        if len({row["qname"] for row in candidate["assignments"]}) != len(candidate["assignments"]):
            raise ValueError("duplicate G3 weight assignment")
    else:
        raise ValueError("unsupported G3 candidate backend")
    return protocol, teacher, inputs, names, device, retained


def preflight_g3(config):
    protocol, teacher, inputs, names, device, retained = prepare_g3(config, check_device=False)
    return {"identity": {"teacher": config["teacher"], "tokenizer": config["tokenizer"],
                         "candidate_inputs": g3_candidate_binding(config),
                         "source_model_identity": teacher.get("source_model_identity", teacher.get("source"))},
            "population": {"window_ids": names, "windows": len(inputs),
                "input_tokens": len(inputs[0][0]), "scored_positions_per_window": protocol["context_length"]-1,
                "prefix_ids": protocol["prefix_ids"], "vocab_size": protocol["vocab_size"],
                "device": "cpu", "requested_device": str(device), "backend": config["candidate"]["backend"]},
            "limitations": ["Preflight reads configuration and array shapes. It does not measure quality."]}


def decode_render(row, *, device, reader=None):
    """Decode actual wire bytes through the accepted or public reader."""
    render = row["render"]
    raw = read_binding(render["wire"], "G3 rendered wire")
    if render["kind"] == "exl3":
        from .g3_exl3 import decode_wire
        decoded = decode_wire(raw, render["in_features"], render["out_features"], device=device)
    elif render["kind"] == "tessera":
        if reader is None:
            from tessera.unit_artifact import read_unit_artifact
        else:
            read_unit_artifact = reader.read_unit_artifact
        decoded = read_unit_artifact(raw, device=device)
    else:
        raise ValueError("unsupported G3 rendered wire kind")
    return decoded.to(torch.bfloat16)


def static_scale(candidate, item):
    """Read actual selected calibration scales. Preserve the accepted reduction."""
    from safetensors.torch import load
    from .g3_activation import StaticScale
    receipt = _g3_json(candidate["static_scales"], "G3 selected scale receipt")
    if receipt.get("schema") != "prismaquant.selected_priced_scales.v1":
        raise ValueError("G3 static scales need the selected calibration receipt")
    scales = load(read_binding(receipt["input_scales"], "G3 actual calibration scale tensors"))
    declaration = item["scale"]
    source, group = declaration["source"], declaration["group"]
    members = tuple(declaration.get("members", []))
    raw = scales[source].float()
    if raw.numel() != 1:
        raise ValueError("G3 input global scale must be scalar")
    effective = raw
    if item["kind"] == "routed":
        groups = (receipt.get("activation_scale_grouping_declaration") or {}).get("groups", {})
        if not members or not any(set(record["members"]) == set(members) for record in groups.values()):
            raise ValueError("G3 routed static scale cohort differs from the calibration owner")
        actual = torch.stack([scales[name+".input_global_scale"].float().reshape(()) for name in members])
        effective = torch.reciprocal(torch.reciprocal(actual).max())
    return StaticScale(float(raw), float(effective), source, group, members)

def _stream(config, protocol, inputs, consume):
    from .cost_streaming import build_streamed_causal_lm
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity
    from .routed_experts import profile_declared_packed_expert_projections, refresh_packed_expert_projections
    from .stage_inputs import source_prefetch
    from .g3_activation import UnitSpec, StaticScale, inject_layer
    from .dev_mode import seal_check
    from .joint_aura import source_execution_identity

    candidate = config["candidate"]
    profile = detect_profile(candidate["model"])
    policy = _g3_json(candidate["source_derivative"], "G3 source derivative") if candidate.get("source_derivative") else None
    runner = build_streamed_causal_lm(candidate["model"], device=torch.device(config["device"]),
        dtype=torch.bfloat16, offload_folder=candidate["offload_folder"], profile=profile,
        attn_implementation="eager", source_derivative=policy, **source_prefetch(candidate))
    try:
        runner.model.eval()
        modules = dict(runner.model.named_modules())
        projected = {member.qname: member for member in profile_declared_packed_expert_projections(runner.model, profile)}
        rows_by_layer = {}
        for row in candidate["assignments"]:
            rows_by_layer.setdefault(runner.layer_index_for_qname(row["qname"]), []).append(row)
        cache = (pickle.loads(read_binding(candidate["production_cache"], "G3 production cache"))
                 if candidate.get("production_cache") else ProductionWeightCache(weights={}, levers={}))
        if not isinstance(cache, ProductionWeightCache):
            raise ValueError("G3 requires the public ProductionWeightCache")
        if candidate.get("cache_dir"):
            cache.relocate(candidate["cache_dir"])
        specs_by_layer = {}
        for item in candidate.get("activation_specs", []):
            item = dict(item)
            layer = item.pop("layer")
            if item.get("scale") is not None:
                item["scale"] = static_scale(candidate, item)
            specs_by_layer.setdefault(layer, []).append(UnitSpec(**item))
        execution = source_execution_identity(runner.model)
        if config.get("recorded_source_execution") is not None:
            seal_check("G3 source execution", config["recorded_source_execution"], execution,
                       where="G3", refusal=lambda: ValueError("recorded source execution differs"))
        visited = []
        def visitor(layer, forward_batch):
            rows = rows_by_layer.get(layer, [])
            live = {member.qname: member.weight for member in refresh_packed_expert_projections(
                [projected[row["qname"]] for row in rows if row["qname"] in projected], profile)}
            views = {row["qname"]: live[row["qname"]] if row["qname"] in live else modules[row["qname"]].weight.data
                     for row in rows}
            keys = [(row["qname"], row["format"]) for row in rows if row["format"] != "SOURCE"]
            created = []
            for row in rows:
                key = (row["qname"], row["format"])
                if row["format"] != "SOURCE" and row.get("render") is not None:
                    if cache.resolve_key(*key) is not None:
                        raise ValueError("G3 assignment names both cache and wire render")
                    if views[row["qname"]].numel()*2 > candidate["max_resident_bytes"]:
                        raise ValueError("G3 decoded weight exceeds its resident budget")
                    from .tessera_reader import load_declared_reader
                    reader = load_declared_reader(candidate.get("reader")) if row["render"]["kind"] == "tessera" else None
                    cache.weights[key] = decode_render(row, device=views[row["qname"]].device, reader=reader)
                    created.append(key)
            window = cache.resident_window(keys, max_resident_bytes=candidate["max_resident_bytes"], max_workers=1) if keys else nullcontext()
            with window:
                jobs = []
                for row in rows:
                    view = views[row["qname"]]
                    if view.is_meta or _cb_cache_tensor_identity(view)["content_sha256"] != row["source_sha256"]:
                        raise ValueError(f"{row['qname']}: installed source bytes differ")
                    if row["format"] != "SOURCE":
                        rendered = cache.get_resident(row["qname"], row["format"]).to(device=view.device, dtype=torch.bfloat16)
                        if rendered.shape != view.shape or not torch.isfinite(rendered).all():
                            raise ValueError(f"{row['qname']}: rendered geometry or values differ")
                        if _cb_cache_tensor_identity(rendered)["content_sha256"] != row["rendered_sha256"]:
                            raise ValueError(f"{row['qname']}: decoded rendered bytes differ")
                        jobs.append((view, rendered))
                for view, rendered in jobs:
                    view.copy_(rendered)
                packed = {projected[spec.qname].module for spec in specs_by_layer.get(layer, [])
                          if spec.kind == "routed" and spec.qname in projected}
                if len(packed) > 1:
                    raise ValueError("G3 activation plan has more than one packed module in a layer")
                hooks = inject_layer(runner.layers[layer], specs_by_layer[layer], tp=protocol["tensor_parallel_size"],
                    unit_views=views, unit_modules=modules, packed_module=next(iter(packed), None),
                    allow_empty=True) if layer in specs_by_layer else nullcontext()
                with hooks:
                    for tokens in inputs:
                        forward_batch(tokens)
            for key in created:
                cache.weights.pop(key)
            visited.append(layer)
        with torch.inference_mode():
            runner.visit_layer_batches(inputs, visitor, output_consumer=consume)
        if visited != list(range(runner.num_layers)):
            raise ValueError("G3 traversal omitted or reordered layers")
        return {"profile": type(profile).__name__, "source_execution": execution, "layers": visited,
                "source_prefetch": runner.context.prefetch_summary()}
    finally:
        runner.shutdown()


def measure_g3(config, output):
    protocol, teacher, inputs, names, device, retained = prepare_g3(config)
    rows, agreements, details = [], [], []
    next_window = 0
    def consume(index, logits):
        nonlocal next_window
        if index != next_window:
            raise ValueError("candidate output order changed")
        if retained is None:
            shape = (1, protocol["context_length"]+len(protocol["prefix_ids"]), protocol["vocab_size"])
            if tuple(logits.shape) != shape:
                raise ValueError("candidate logit geometry differs")
            raw = logits[0, len(protocol["prefix_ids"]):-1].float()
        else:
            raw = logits.float()
        trow = teacher["arrays"][index]
        array = _array(trow, (protocol["context_length"]-1, protocol["vocab_size"]), verify=True)
        target = torch.from_numpy(np.array(array, copy=True)).to(device)
        vector = token_kl(target, raw, tile_rows=protocol["tile_rows"], require_cuda=device.type == "cuda")
        rows.append(vector)
        agreements.append(target.argmax(-1) == raw.argmax(-1))
        details.append({"window_id": names[index],
            "input_tokens_sha256": bytes_sha256hex(memoryview(inputs[index][0].detach().cpu().numpy()).cast("B")),
            "teacher_file_sha256": trow["file_sha256"],
            "teacher_array_sha256": bytes_sha256hex(memoryview(array).cast("B")),
            "candidate_array_sha256": bytes_sha256hex(memoryview(raw.cpu().numpy()).cast("B"))})
        next_window += 1
    with torch.inference_mode():
        if retained is not None:
            for index, row in enumerate(retained["arrays"]):
                array = _array(row, (protocol["context_length"]-1, protocol["vocab_size"]), verify=True)
                consume(index, torch.from_numpy(np.array(array, copy=True)).to(device))
            execution = {"backend": "retained_logits", "candidate_receipt": config["candidate"]["receipt"]}
        else:
            execution = _stream(config, protocol, inputs, consume)
    if next_window != protocol["window_count"]:
        raise ValueError("candidate omitted paired windows")
    from .g3_numerics import g3_summary
    metrics, values, agreement_values, means, top1_means = g3_summary(rows, agreements)
    for row, mean, top1 in zip(details, means, top1_means, strict=True):
        row.update(mean_kl=mean, top1_agreement=top1)
    array_path = Path(output).with_suffix(".per_position_kl.npy")
    from .cost_stage_checkpoint import publish_new_bytes
    payload = io.BytesIO()
    np.save(payload, values, allow_pickle=False)
    if not publish_new_bytes(array_path, payload.getvalue()):
        raise ValueError("G3 per-position output already exists")
    agreement_path = Path(output).with_suffix(".per_position_top1_agreement.npy")
    payload = io.BytesIO()
    np.save(payload, agreement_values, allow_pickle=False)
    if not publish_new_bytes(agreement_path, payload.getvalue()):
        raise ValueError("G3 per-position agreement output already exists")
    return {"metrics": metrics,
        "identity": {"teacher": config["teacher"], "tokenizer": config["tokenizer"], "protocol": config["protocol"],
                     "panel": config["panel"],
                     "candidate_inputs": g3_candidate_binding(config),
                     "source_model_identity": teacher.get("source_model_identity", teacher.get("source")), "execution": execution,
                     "torch": torch.__version__},
        "population": {"device": str(device), "cuda_available": torch.cuda.is_available(), "skips": [],
            "window_ids": names, "windows": len(rows), "positions": int(values.size), "per_window": details,
            "measurement_fidelity": _g3_fidelity(protocol)}, "artifacts": [artifact(array_path), artifact(agreement_path)],
        "limitations": ["Offline decoded KL is not served KL.", "CPU execution does not qualify GLM or native kernels."]}


def verify_g3_result(result, config):
    """Replay owned arrays and mathematical inputs without model inference."""
    from .g3_numerics import g3_summary
    from .stage_inputs import read_bound
    protocol = _g3_protocol(config)
    inputs, names, actual = _g3_panel_inputs(config, protocol)
    teacher = _g3_teacher(config, names, actual)
    artifacts = result["artifacts"]
    if len(artifacts) != 2:
        raise ValueError("G3 result requires owned KL and agreement arrays")
    arrays = [np.load(io.BytesIO(read_bound(binding, "G3 owned result array")), allow_pickle=False)
              for binding in artifacts]
    shape = (protocol["window_count"], protocol["context_length"] - 1)
    if any(not isinstance(array, np.ndarray) or array.shape != shape for array in arrays):
        raise ValueError("G3 owned array population or geometry differs")
    values, agreements = arrays
    if values.dtype != np.float64 or agreements.dtype != np.bool_:
        raise ValueError("G3 owned array dtype differs")
    population = result["population"]
    device = torch.device(population["device"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("G3 recorded numerical device is unavailable")
    metrics, _, _, means, top1_means = g3_summary(torch.from_numpy(values).to(device),
                                                 torch.from_numpy(agreements).to(device))
    if result["measurement"]["metrics"] != metrics:
        raise ValueError("G3 metrics differ from the owned arrays")
    expected = {"window_ids": names, "windows": len(values), "positions": int(values.size),
                "measurement_fidelity": _g3_fidelity(protocol)}
    if any(type(population.get(name)) is not type(value) or population.get(name) != value
           for name, value in expected.items()):
        raise ValueError("G3 recorded population differs from owned data")
    windows = population.get("per_window")
    if not isinstance(windows, list) or len(windows) != len(values):
        raise ValueError("G3 recorded per-window population differs")
    for index, (row, mean, top1) in enumerate(zip(windows, means, top1_means, strict=True)):
        if row.get("window_id") != names[index] or row.get("mean_kl") != mean or row.get("top1_agreement") != top1:
            raise ValueError("G3 window summaries differ from owned data")
        tokens = bytes_sha256hex(memoryview(inputs[index][0].numpy()).cast("B"))
        if row.get("input_tokens_sha256") != tokens:
            raise ValueError("G3 window token population differs from the measurement")
        current = teacher["arrays"][index]
        if row.get("teacher_file_sha256") != current["file_sha256"]:
            raise ValueError("G3 paired teacher array differs from the measurement")
        if current.get("array_sha256") and row.get("teacher_array_sha256") != current["array_sha256"]:
            raise ValueError("G3 paired teacher numeric data differs")


def main(argv=None):
    return cli("g3_v2", measure_g3, preflight_g3, argv)


if __name__ == "__main__":
    raise SystemExit(main())
