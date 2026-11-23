import json
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import day0_model_intake, digests, gpu_guard, production_recache
from prismaquant import export_native_compressed as exporter
from prismaquant import tessera_plan_writer as writer
from prismaquant.production_weight_cache import ProductionWeightCache


def _intake_stdout(root, capsys):
    source = root / "intake-source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 1, "hidden_size": 4, "vocab_size": 8,
        "num_attention_heads": 2, "num_key_value_heads": 2, "intermediate_size": 6,
    }))
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4, dtype=torch.bfloat16)},
              str(source / "model.safetensors"))
    assert day0_model_intake.main(["--checkpoint", str(source), "--output", str(root / "intake")]) == 0
    return capsys.readouterr().out.encode("utf-8")


def _export_writes(root, monkeypatch):
    shards = root / "shards"
    shard_writer = exporter.IncrementalSafetensorsWriter(shards, shard_bytes=32)
    shard_writer.add_tensors({"b.weight": torch.ones(8), "é.weight": torch.ones(8)})
    shard_writer.finalize()

    source = root / "export-source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"num_hidden_layers": 1},
    }))
    vision = "model.vision_tower.vision_model.encoder.layers.0.attn.qkv"
    save_file({vision + ".weight": torch.ones(32, 32, dtype=torch.bfloat16) / 64},
              str(source / "model.safetensors"))
    recipe = root / "layer_config.json"
    recipe.write_text(json.dumps({vision: "FP8_E4M3"}))
    output = root / "export"
    # Replace the text-body GPU sweep. Keep the export writes on CPU.
    monkeypatch.setattr(exporter, "materialize_tensors_streaming", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(exporter, "_ALLOCATOR_TARGET_PROFILE", "research")
    exporter.main([
        "--model", str(source), "--layer-config", str(recipe), "--output", str(output),
        "--no-gptq", "--no-scale-sweep",
    ])
    return (shards / "model.safetensors.index.json", output / "config.json",
            output / "mixed_native_manifest.json")


def _recache_writes(root, monkeypatch):
    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(root),
                                  activation_max_abs={"l1": 1.0})
    monkeypatch.setattr(production_recache, "measure_production_activation_max_abs",
                        lambda *a, **k: {"l1": 2.0})
    production_recache.recache_production_weight_cache(
        torch.nn.Linear(1, 1), torch.ones(1, 1, dtype=torch.long),
        {"l1": "BF16"}, cache, progress=False,
    )
    return (root / "activation_max_abs.json", root / "activation_max_abs_delta.json")


def _plan_writes(root, monkeypatch):
    module = "model.layers.0.self_attn"
    qkv = tuple(f"{module}.{role}_proj.weight" for role in "qkv")
    shapes = dict.fromkeys(qkv, (32, 32))
    surface = SimpleNamespace(
        MOE_ROUTER=re.compile(r"^.*\.mlp\.(?:gate|router)\.weight$"),
        quantizable=lambda model: ([], shapes, {}, {}),
        expert_stacks=lambda routed: {},
        packed_expert_stacks=lambda packed: {},
        body_layer=lambda tensor: 0,
        fused_module=lambda tensor: (module, qkv) if tensor in qkv else None,
        grid_for_name=lambda name: name,
        module_scheme_key=lambda grid, q256: grid,
        validate_serving_plan=lambda plan: None,
        units_from_plan=lambda plan, shapes: list(plan.values()),
        selection_requirement=lambda units: {"requires_validation": False},
    )
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    assignment = root / "assignment.json"
    plan = root / "plan.json"
    config = {name.removesuffix(".weight"): {"tessera_format": "TESSERA_E4M3_K1_R1024"}
              for name in qkv}
    assignment.write_text(json.dumps(config))
    assert writer.main([str(assignment), str(root), str(plan), "--no-uniform-control"]) == 0
    config[qkv[1].removesuffix(".weight")] = {"tessera_format": "TESSERA_E2M1_K2_R896"}
    with pytest.raises(writer.PlanError, match=r"^1 fused module\(s\) do not share one scheme") as exc:
        writer.build_serving_plan(config, shapes, cover="as-allocated", allow_disagreement=False,
                                  surface=surface, with_control=False)
    fragment = str(exc.value).split(": ", 1)[1].split("vLLM builds ONE", 1)[0]
    return plan.with_suffix(".json.provenance.json"), fragment.encode("utf-8")


def test_listed_calls_use_insertion_ascii_indent2_lax(tmp_path, monkeypatch, capsys):
    real_text = digests.JsonProfile.text
    expected = []

    def observe(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT2_LAX:
            prefix = marker * (len(expected) + 1)
            expected.append((prefix + json.dumps(value, indent=2)).encode("utf-8"))
            return prefix + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe)
    # Add a different number of spaces for each call on the second run.
    for run, marker in enumerate(("", " ")):
        expected.clear()
        root = tmp_path / str(run)
        root.mkdir()
        capsys.readouterr()
        stdout = _intake_stdout(root, capsys)
        files = (*_export_writes(root, monkeypatch), *_recache_writes(root, monkeypatch))
        sidecar, refusal = _plan_writes(root, monkeypatch)
        actual = {"intake stdout": stdout,
                  **{path.name: path.read_bytes() for path in (*files, sidecar)},
                  "plan refusal": refusal}
        assert len(expected) == len(actual), "The profile must supply all eight outputs."
        for (name, output), encoded in zip(actual.items(), expected, strict=True):
            lf = b"\n" if name in {"intake stdout", "plan refusal"} else b""
            assert output == encoded + lf, name
