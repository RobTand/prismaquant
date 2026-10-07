"""CPU intake and refusal tests. These do not qualify serving gates."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from prismaquant import shipcard
from tools import dsv4_wikitext_inputs as inputs
from tools.full_kl_teacher_payload import tokenizer_identity


def small_inputs(root: Path) -> tuple[Path, Path, Path]:
    """Build small tensors and run the real generic input sealer."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    root.mkdir(parents=True, exist_ok=True)
    model = root / "artifact"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "vocab_size": 8,
    }))
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    (model / "tokenizer_config.json").write_text('{"tokenizer_class":"TokenizersBackend"}')
    save_file({"model.embed_tokens.weight": torch.zeros((8, 4), dtype=torch.bfloat16)},
              str(model / "model.safetensors"))
    card = shipcard.build_shipcard(model, build={})
    shipcard.write_shipcard(model / "shipcard.json", card)
    teacher = root / "teacher.pt"
    torch.save({"calib_ids": torch.tensor([[1, 2, 3, 1]], dtype=torch.long),
                "teacher_logprobs": torch.log_softmax(torch.zeros((1, 8)), dim=-1),
                "topk_ids": torch.arange(8, dtype=torch.int32).expand(1, 3, 8).clone(),
                "topk_lps": torch.log_softmax(torch.zeros((1, 3, 8)), dim=-1),
                "prompt_top_k": 8,
                "vocab_size": 8, "seqlen": 4, "score_positions": "all"}, teacher)
    tokenizer_evidence = tokenizer_identity(model)
    train_count = 1024
    windows = [[1, 2] * (inputs.FULL_KL_SEQLEN // 2) for _ in range(inputs.FULL_KL_N_SAMPLES)]
    ppl_ids = [1, 2] * (inputs.PPL_N_TOKENS // 2)
    def dataset(split, count):
        return {**inputs._expected_dataset(split=split), "total_tokens": count,
                "fingerprint": inputs.MODEL_DATASET_FINGERPRINTS[split]}
    config_bytes = (model / "config.json").read_bytes()
    payload = inputs.seal_model_wikitext_inputs({
        "schema": inputs.MODEL_WIKITEXT_INPUTS_SCHEMA,
        "datasets_distribution": {"name": inputs.DATASETS_DISTRIBUTION,
                                  "version": inputs.DATASETS_VERSION},
        "corpus_construction": dict(inputs.CORPUS_CONSTRUCTION),
        "tokenizer": tokenizer_evidence, "model": inputs.wikitext_model_identity(model),
        "source_config": {"bytes": len(config_bytes), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "full_kl": {"dataset": dataset("train", train_count),
                    "selection": {"sampler": "python.random.Random(seed).sample(range(max_start), n_samples)/v1",
                                  "window_seed": inputs.FULL_KL_WINDOW_SEED,
                                  "n_samples": inputs.FULL_KL_N_SAMPLES,
                                  "seqlen": inputs.FULL_KL_SEQLEN,
                                  "starts": random.Random(inputs.FULL_KL_WINDOW_SEED).sample(
                                      range(train_count - inputs.FULL_KL_SEQLEN), inputs.FULL_KL_N_SAMPLES)},
                    "token_ids": windows,
                    "token_ids_tensor_sha256": inputs._tensor_sha256(torch.tensor(windows, dtype=torch.long))},
        "ppl": {"dataset": dataset("test", inputs.PPL_N_TOKENS),
                "selection": {"strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
                              "n_tokens": inputs.PPL_N_TOKENS},
                "token_ids": ppl_ids, "token_ids_sha256": inputs.canonical_sha256(ppl_ids)},
    }, expected_tokenizer_identity=tokenizer_evidence,
       expected_model_identity=inputs.wikitext_model_identity(model))
    tokens = root / "wikitext.json"
    tokens.write_text(json.dumps(payload))
    return model, teacher, tokens


def job_config(root: Path) -> Path:
    model, teacher, tokens = small_inputs(root)
    quality = root / "quality.json"
    quality.write_text(json.dumps({"schema": "not-a-quality-configuration"}))
    stages = [
        {"id": "gold.kl", "args": ["--teacher-payload", str(teacher), "--score-positions", "all"],
         "output": str(root / "gold.kl.json")},
        {"id": "gold.ppl", "args": ["--wikitext-inputs", str(tokens), "--wikitext-inputs-sha256",
                                    hashlib.sha256(tokens.read_bytes()).hexdigest()],
         "output": str(root / "gold.ppl.json")},
    ]
    for slot in shipcard.required_slots(shipcard.load_shipcard(model / "shipcard.json"), model_dir=model):
        if slot.startswith("gold."):
            continue
        stages.append({"id": slot, "argv": [sys.executable, "-m", "prismaquant.shipcard_cli",
                                           "verify", "{shipcard}", "--model-dir", "{artifact}"],
                       "record": "{shipcard}", "output": str(root / (slot + ".json"))})
    for slot in ("offline.g3", "task_suite"):
        stages.append({"id": slot, "config": str(quality), "output": str(root / (slot + ".json"))})
    config = root / "job.json"
    config.write_text(json.dumps({"schema": "prismaquant.ship_gates/1", "artifact": str(model),
                                 "topology": {"tensor_parallel_size": 2, "nnodes": 1},
                                 "serve_image": "cpu-preflight-only",
                                 "stages": stages, "inputs": [str(teacher), str(tokens), str(quality)]}))
    return config


def _cli(module, args):
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True)


def test_real_gold_kl_preflight_never_imports_vllm(tmp_path):
    model, teacher, _ = small_inputs(tmp_path)
    output = tmp_path / "preflight.json"
    process = _cli("tools.measure_vllm_full_kl", ["--mode", "student", "--model", str(model),
        "--teacher-payload", str(teacher), "--serve-image", "cpu", "--output", str(output), "--preflight"])
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["runtime_qualification"] == "not_run"
    assert result["n_samples"] == 1 and result["vocab_size"] == 8


def test_real_ppl_preflight_uses_generic_value_closed_inputs(tmp_path):
    model, _, tokens = small_inputs(tmp_path)
    output = tmp_path / "preflight.json"
    process = _cli("tools.measure_vllm_wikitext_ppl", ["--model", str(model), "--output", str(output),
        "--serve-image", "cpu", "--wikitext-inputs", str(tokens), "--wikitext-inputs-sha256",
        hashlib.sha256(tokens.read_bytes()).hexdigest(), "--preflight"])
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["runtime_qualification"] == "not_run"
    assert result["n_tokens_scored"] == 8176


@pytest.mark.parametrize("damage", ["shape", "token"])
def test_kl_preflight_refuses_invalid_teacher_values(tmp_path, damage):
    model, teacher, _ = small_inputs(tmp_path)
    payload = torch.load(teacher, weights_only=True)
    if damage == "shape":
        payload["seqlen"] = 5
    else:
        payload["calib_ids"][0, 0] = 8
    torch.save(payload, teacher)
    process = _cli("tools.measure_vllm_full_kl", ["--mode", "student", "--model", str(model),
        "--teacher-payload", str(teacher), "--serve-image", "cpu", "--output", str(tmp_path / "out.json"), "--preflight"])
    assert process.returncode != 0
    assert "token shape or vocabulary is invalid" in process.stderr


def test_runner_keeps_every_missing_evidence_failure(tmp_path):
    from prismaquant.ship_gates import run
    config = job_config(tmp_path)
    result = run(config, tmp_path / "refusal.json", verify_only=True)
    assert result["status"] == "refused"
    assert result["runtime_qualification"] == "refused"
    assert all(stage["status"] == "failed" for stage in result["stages"])
    assert {stage["id"] for stage in result["stages"]} >= set(shipcard.REQUIRED_SLOTS)
    assert any("UNFILLED" in problem for problem in result["problems"])


@pytest.mark.parametrize("damage", ["omit_graph", "omit_gold", "duplicate", "tp", "nested", "identity"])
def test_runner_refuses_incomplete_or_inconsistent_configuration(tmp_path, damage):
    from prismaquant.ship_gates import run
    path = job_config(tmp_path)
    config = json.loads(path.read_text())
    if damage in {"omit_graph", "omit_gold"}:
        omit = "native_export.graph" if damage == "omit_graph" else "gold.kl"
        config["stages"] = [stage for stage in config["stages"] if stage["id"] != omit]
    elif damage == "duplicate":
        config["stages"].append(copy.deepcopy(config["stages"][0]))
    elif damage == "tp":
        config["topology"] = {"tensor_parallel_size": 3, "nnodes": 2}
    elif damage == "nested":
        next(stage for stage in config["stages"] if "argv" in stage)["argv"] = ["pbrun.py"]
    else:
        config["stages"][0]["args"] += ["--model=other-model"]
    path.write_text(json.dumps(config))
    assert run(path, tmp_path / "refusal.json", preflight=True)["status"] == "refused"


def test_preflight_runs_real_gold_producers_but_refuses_absent_quality(tmp_path):
    from prismaquant.ship_gates import run
    config = job_config(tmp_path)
    report = run(config, tmp_path / "preflight.json", preflight=True)
    assert report["status"] == "refused"
    assert report["runtime_qualification"] == "not_run"
    stages = {stage["id"]: stage for stage in report["stages"]}
    assert stages["gold.kl"]["status"] == stages["gold.ppl"]["status"] == "preflight"
    assert stages["native_export.graph"]["status"] == "not_run"
    assert stages["offline.g3"]["status"] == "failed"
    assert stages["task_suite"]["status"] == "not_run"


def test_runner_never_overwrites_its_configuration(tmp_path):
    from prismaquant.ship_gates import run
    config = job_config(tmp_path)
    before = config.read_bytes()
    report = run(config, config, preflight=True)
    assert report["status"] == "refused"
    assert config.read_bytes() == before


def test_runner_keeps_an_existing_result(tmp_path):
    from prismaquant.ship_gates import run
    config = job_config(tmp_path)
    output = tmp_path / "prior-result.json"
    output.write_text('{"retained_result":true}')
    before = output.read_bytes()
    assert run(config, output, verify_only=True)["status"] == "refused"
    assert output.read_bytes() == before


@pytest.mark.parametrize("protected", [True, False], ids=["protected-input", "retained-log"])
def test_preflight_preserves_actual_log_bytes(tmp_path, protected):
    from prismaquant.ship_gates import run
    config_path = job_config(tmp_path)
    report_path = tmp_path / "preflight.json"
    log_path = tmp_path / "preflight.stages" / "gold.kl.log"
    log_path.parent.mkdir()
    retained = b"retained input or log bytes\n"
    log_path.write_bytes(retained)
    if protected:
        config = json.loads(config_path.read_text())
        config["inputs"].append(str(log_path))
        config_path.write_text(json.dumps(config))
    report = run(config_path, report_path, preflight=True)
    assert report["status"] == "refused"
    assert log_path.read_bytes() == retained
    assert not log_path.with_suffix(".json").exists()


def _task_replay_job(root):
    from test_task_quality_replay import make_task_replay_fixture
    from prismaquant.cost_streaming import build_source_checkpoint_identity
    job_path = job_config(root / "job")
    job = json.loads(job_path.read_text())
    config, result, raw = make_task_replay_fixture(root / "quality")
    model = Path(job["artifact"])
    config["backend"]["pretrained"] = str(model)
    result["identity"]["model"] = str(model)
    result["identity"]["model_artifact"] = build_source_checkpoint_identity(model)
    config_path = root / "task-replay.config.json"
    config_path.write_text(json.dumps(config))
    result["configuration"].update(path=str(config_path),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    stage = next(stage for stage in job["stages"] if stage["id"] == "task_suite")
    stage.update(config=str(config_path), output=str(root / "task-result.json"))
    Path(stage["output"]).write_text(json.dumps(result))
    job["inputs"].extend((str(config_path), str(raw)))
    job_path.write_text(json.dumps(job))
    return job_path, config, result, raw, Path(stage["output"])


def _verify_replay_stage(job_path, root, identity):
    output = root / "verify.json"
    process = _cli("prismaquant.ship_gates", ["--config", str(job_path),
        "--output", str(output), "--verify-only"])
    assert process.returncode == 1, process.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "refused"
    stage = next(stage for stage in report["stages"] if stage["id"] == identity)
    return report, stage


@pytest.mark.parametrize("kind", ["config", "weight", "tokenizer"])
def test_task_replay_consumer_certified_replacement_refuses(tmp_path, monkeypatch, kind):
    from test_task_quality_replay import replace_task_fixture
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    job, config, _, _, _ = _task_replay_job(tmp_path)
    replace_task_fixture(config, kind)
    _, task = _verify_replay_stage(job, tmp_path, "task_suite")
    assert task["status"] == "failed"
    assert "provenance" in task["error"]


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("damage", ["corrupt", "metrics", "sampling"])
def test_task_replay_consumer_owned_bytes_and_math_refuse(tmp_path, monkeypatch, dev, damage):
    from prismaquant.quality_stage import artifact
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    job, _, result, raw_path, output = _task_replay_job(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    if damage == "corrupt":
        raw_path.write_bytes(b"corrupt owned result")
    else:
        raw = json.loads(raw_path.read_text())
        if damage == "metrics":
            raw["results"]["facts"]["acc,none"] = 0.0
        else:
            raw["config"]["torch_seed"] = 1
        raw_path.write_text(json.dumps(raw))
        result["artifacts"] = [artifact(raw_path)]
        output.write_text(json.dumps(result))
    _, task = _verify_replay_stage(job, tmp_path, "task_suite")
    assert task["status"] == "failed"


def test_task_replay_consumer_dev_preserves_metrics_and_stamp(tmp_path, monkeypatch):
    from test_task_quality_replay import replace_task_fixture
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    job, config, result, _, _ = _task_replay_job(tmp_path)
    stored = copy.deepcopy(result["measurement"])
    replace_task_fixture(config, "weight")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE")
    report, task = _verify_replay_stage(job, tmp_path, "task_suite")
    assert task["status"] == "passed"
    assert task["result"]["measurement"] == stored
    assert task["result"].get("dev_uncertified") is True
    assert report.get("dev_uncertified") is True


def _g3_replay_job(root):
    from test_g3_quality_replay import make_g3_replay_fixture
    job_path = job_config(root / "job")
    job = json.loads(job_path.read_text())
    config, result, kl, agreement = make_g3_replay_fixture(root / "quality")
    config_path = root / "g3-replay.config.json"
    config_path.write_text(json.dumps(config))
    result["configuration"].update(path=str(config_path),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    stage = next(stage for stage in job["stages"] if stage["id"] == "offline.g3")
    stage.update(config=str(config_path), output=str(root / "g3-result.json"))
    Path(stage["output"]).write_text(json.dumps(result))
    job["inputs"].append(str(config_path))
    job_path.write_text(json.dumps(job))
    return job_path, config, result, kl, agreement, Path(stage["output"])


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("damage", ["metric_and_gate", "window_summary", "missing_kl",
                                    "missing_agreement", "corrupt_kl"])
def test_g3_replay_consumer_owned_arrays_and_summaries_refuse(tmp_path, monkeypatch, dev, damage):
    from prismaquant.quality_stage import evaluate_criteria
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    job, config, result, kl, agreement, output = _g3_replay_job(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    if damage == "metric_and_gate":
        result["measurement"]["metrics"]["mean_kl"] += 0.25
        result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
        assert result["gate"]["status"] == "passed"
    elif damage == "window_summary":
        result["population"]["per_window"][0]["mean_kl"] += 0.25
    elif damage == "missing_kl":
        kl.unlink()
    elif damage == "missing_agreement":
        agreement.unlink()
    else:
        kl.write_bytes(b"corrupt owned KL array")
    output.write_text(json.dumps(result))
    _, g3 = _verify_replay_stage(job, tmp_path, "offline.g3")
    assert g3["status"] == "failed"
