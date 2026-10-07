"""Metadata plans preserve source ownership and do not qualify a runtime."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def source_fixture(root, layout):
    source = root / layout
    source.mkdir()
    if layout == "dense":
        tensors = {f"model.layers.{i}.mlp.down_proj.weight": torch.zeros(32, 32)
                   for i in range(3)}
        tensors["model.embed_tokens.weight"] = torch.zeros(32, 32)
        save_file(tensors, str(source / "model.safetensors"))
        owners = {name: "model.safetensors" for name in tensors}
        config = {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 3}
    else:
        tensors = {"model.layers.0.mlp.down_proj.weight": torch.zeros(32, 32)}
        tensors.update({f"model.layers.{i}.mlp.experts.{e}.{role}.weight": torch.zeros(32, 32)
                        for i in (2, 5) for e in range(2)
                        for role in ("gate_proj", "up_proj", "down_proj")})
        tensors["model.embed_tokens.weight"] = torch.zeros(32, 32)
        owners = {name: f"weights-{i % 2}.safetensors" for i, name in enumerate(tensors)}
        for shard in set(owners.values()):
            save_file({n: t for n, t in tensors.items() if owners[n] == shard}, str(source / shard))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": owners}))
        config = {"architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 6,
                  "num_experts": 2, "hidden_size": 32, "moe_intermediate_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    plan = root / f"{layout}-plan.json"
    plan.write_text(json.dumps({name: "BF16" for name in tensors if name.endswith("down_proj.weight")
                               and ".experts." not in name}))
    return source, plan, owners


def metadata_cli(source, plan):
    return subprocess.run([sys.executable, "-m", "tools.tessera_fleet.dispatch_model",
                           "--dry-run", "--source", str(source), "--plan", str(plan),
                           "--cpus", "2", "--mem-gb", "6"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("layout", ["dense", "indexed_experts"])
def test_real_metadata_dispatch_covers_source_once_and_plans_census(tmp_path, layout):
    source, plan, owners = source_fixture(tmp_path, layout)
    before = {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    result = metadata_cli(source, plan)
    assert result.returncode == 0, result.stderr
    setup = json.loads(result.stdout)
    assert {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before
    covered = [name for part in setup["partitions"] for name in part["source_tensors"]]
    assert len(covered) == len(set(covered)) == len(owners)
    assert set(covered) == set(owners)
    assert all(part["source_tensors"] for part in setup["partitions"])
    from tessera.serving_parts import partition_owner
    assert all(partition_owner(name, setup["partition_count"]) == part["index"]
               for part in setup["partitions"] for name in part["source_tensors"])
    assert setup["partition_count"] == (3 if layout == "dense" else 2)
    assert all(part["source_shards"] == sorted({owners[n] for n in part["source_tensors"]})
               for part in setup["partitions"])
    census = setup["construction_census"]
    assert census["status"] == "not_run"
    assert census["argv"] == ["python3", "tools/tessera_construction_census.py",
                              str(source.resolve()), "construction-census.json", "--device", "meta"]
    assert census["architectures"] == json.loads((source / "config.json").read_text())["architectures"]
    assert setup["runtime_qualification"] == "not_run"
    assert len(setup["encode_rows"]) == setup["partition_count"]
    assert all(row["demand"]["cpu"] == 2 and row["demand"]["mem_gb"] == 6
               for row in setup["encode_rows"])
    assert {unit["kind"] for unit in setup["units"]} == (
        {"dense"} if layout == "dense" else {"dense", "routed_moe"})


@pytest.mark.parametrize("defect,reason", [("index", "does not exactly cover"),
                                           ("kind", "unsupported module kind"),
                                           ("shape", "shape"),
                                           ("architecture", "model profile")])
def test_metadata_dispatch_refuses_unsupported_source_or_plan(tmp_path, defect, reason):
    source, plan, owners = source_fixture(tmp_path, "dense")
    if defect == "index":
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict(owners, missing="model.safetensors")}))
    elif defect == "kind":
        tensor = "model.layers.0.mlp.conv.weight"
        weights = {name: torch.zeros(32, 32) for name in owners}
        weights[tensor] = torch.zeros(32, 32, 3)
        save_file(weights, str(source / "model.safetensors"))
        plan.write_text(json.dumps({tensor: {"grid": "E4M3", "q256": 1024}}))
    elif defect == "shape":
        save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(32, 33)},
                  str(source / "model.safetensors"))
        plan.write_text(json.dumps({"model.layers.0.mlp.down_proj.weight": {"grid": "E4M3", "q256": 1024}}))
    else:
        (source / "config.json").write_text(json.dumps({"architectures": ["UnknownForCausalLM"]}))
    result = metadata_cli(source, plan)
    assert result.returncode != 0
    assert reason in result.stderr, result.stderr


def test_real_plan_writer_publishes_the_same_metadata_setup(tmp_path):
    source, _plan, owners = source_fixture(tmp_path, "dense")
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({n.removesuffix(".weight"): "BF16"
                                      for n in owners if ".layers." in n}))
    plan = tmp_path / "written-plan.json"
    setup_path = tmp_path / "setup.json"
    result = subprocess.run([sys.executable, "-m", "prismaquant.tessera_plan_writer",
                             str(assignment), str(source), str(plan), "--no-uniform-control",
                             "--export-setup-json", str(setup_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    setup = json.loads(setup_path.read_text())
    assert setup["source_tensors"] == owners
    assert setup["construction_census"]["status"] == "not_run"
    assert setup["partition_count"] == 3
