"""Run explicit small tasks through the real lm-eval Hugging Face backend."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path

from .quality_stage import artifact, cli, finite_number, write_result


def validate_config(config):
    backend, sampling, tasks = config.get("backend"), config.get("sampling"), config.get("tasks")
    required = {"name", "pretrained", "tokenizer", "device", "dtype", "batch_size", "max_length", "trust_remote_code"}
    if not isinstance(backend, dict) or not required <= backend.keys() or backend["name"] != "hf":
        raise ValueError("task backend requires explicit hf model, tokenizer, device, dtype and batch settings")
    for name in ("pretrained", "tokenizer", "device", "dtype"):
        if not isinstance(backend[name], str) or not backend[name]:
            raise ValueError(f"task backend requires {name}")
    if backend["device"] not in ("cpu", "cuda") and not backend["device"].startswith("cuda:"):
        raise ValueError("task device must be CPU or CUDA")
    for name in ("batch_size", "max_length"):
        if type(backend[name]) is not int or backend[name] <= 0:
            raise ValueError(f"task backend {name} must be a positive integer")
    if type(backend["trust_remote_code"]) is not bool:
        raise ValueError("trust_remote_code must be explicit Boolean")
    fields = {"limit", "num_fewshot", "random_seed", "numpy_seed", "torch_seed", "fewshot_seed"}
    if not isinstance(sampling, dict) or set(sampling) != fields:
        raise ValueError("task sampling requires limit, few-shot count and all four seeds")
    limit = sampling["limit"]
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("task limit must be a positive sample count or explicit null for all samples")
    for name in fields - {"limit"}:
        if type(sampling[name]) is not int or sampling[name] < 0:
            raise ValueError(f"task sampling {name} must be a nonnegative integer")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, (str, dict)) for task in tasks):
        raise ValueError("tasks must be an explicit nonempty list of lm-eval names or task configurations")
    names = [task if isinstance(task, str) else task.get("task") for task in tasks]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError("task names must be nonempty and unique")
    return config


def task_metrics(raw):
    results = raw.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("task backend returned no task population")
    metrics = {}
    for task, values in results.items():
        task_count = 0
        for name, value in values.items():
            if name in ("name", "alias", "sample_len", "sample_count") or "_stderr" in name:
                continue
            metrics[f"{task}/{name}"] = finite_number(value, f"{task}/{name}")
            task_count += 1
        if not task_count:
            raise ValueError(f"task backend returned no measured metric for {task}")
    return metrics


def _task_model_artifact(backend, *, model_config=None):
    from .cost_streaming import build_source_checkpoint_identity
    path = Path(backend["pretrained"])
    if path.is_dir():
        return build_source_checkpoint_identity(path)
    if model_config is None:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(backend["pretrained"], revision=backend.get("revision"),
                                                  trust_remote_code=backend["trust_remote_code"])
    return {"repository": str(path), "resolved_commit": getattr(model_config, "_commit_hash", None)}


def _task_tokenizer_files(tokenizer):
    from tools.full_kl_teacher_payload import tokenizer_identity
    root = Path(tokenizer)
    if not root.is_dir():
        return []
    files = tokenizer_identity(root)["files"]
    return [{"path": str((root / name).resolve()), "sha256": row["sha256"]}
            for name, row in sorted(files.items())]


def task_population(raw, config):
    """Replay actual task counts and the configured mathematical sample policy."""
    expected = {task if isinstance(task, str) else task["task"] for task in config["tasks"]}
    if not expected <= set(raw["results"]):
        raise ValueError("task backend omitted a requested task")
    samples = raw.get("n-samples")
    if not isinstance(samples, dict) or any(not isinstance(samples.get(name), dict)
            or type(samples[name].get("effective")) is not int or samples[name]["effective"] <= 0 for name in expected):
        raise ValueError("task backend did not measure every requested task")
    sampling = config["sampling"]
    for name in ("limit", "random_seed", "numpy_seed", "torch_seed", "fewshot_seed"):
        value = raw.get("config", {}).get(name)
        if type(value) is not type(sampling[name]) or value != sampling[name]:
            raise ValueError("task raw sampling differs: " + name)
    if any(type(raw.get("n-shot", {}).get(name)) is not int
            or raw["n-shot"][name] != sampling["num_fewshot"] for name in expected):
        raise ValueError("task raw sampling differs: num_fewshot")
    return {"device": config["backend"]["device"], "tasks": sorted(expected),
            "samples": samples, "sampling": sampling}


def verify_task_result(result, config):
    """Verify owned raw bytes. Stamp recorded provenance in default dev mode."""
    from .dev_mode import NOT_COMPUTED, dev_mode_enabled, dev_stamp, seal_check
    from .schemas import strict_json_loads
    from .stage_inputs import read_bound
    validate_config(config)
    artifacts = result["artifacts"]
    if len(artifacts) != 1:
        raise ValueError("task result requires its raw lm-eval artifact")
    raw = strict_json_loads(read_bound(artifacts[0], "task raw result"),
        duplicate=lambda key: ValueError("duplicate task raw result key: " + key),
        constant=lambda value: ValueError("nonfinite task raw result value: " + value))
    if task_metrics(raw) != result["measurement"]["metrics"]:
        raise ValueError("task raw metrics differ from the stored measurement")
    for name, value in task_population(raw, config).items():
        if result["population"].get(name) != value:
            raise ValueError("task mathematical population differs: " + name)
    identity = result["identity"]
    if identity.get("task_configs") != raw.get("configs") or identity.get("task_versions") != raw.get("versions"):
        raise ValueError("task definitions or versions differ from the owned raw result")
    backend = config["backend"]
    model = {"path": identity.get("model"), "artifact": identity.get("model_artifact")}
    tokenizer = {"path": identity.get("tokenizer"), "files": identity.get("tokenizer_files"),
                 "commit": identity.get("tokenizer_commit")}
    dev = dev_mode_enabled()
    if dev:
        current_model = current_tokenizer = NOT_COMPUTED
    else:
        current_model = {"path": backend["pretrained"], "artifact": _task_model_artifact(backend)}
        commit = None
        if not Path(backend["tokenizer"]).is_dir():
            from transformers import AutoTokenizer
            current = AutoTokenizer.from_pretrained(backend["tokenizer"],
                revision=backend.get("tokenizer_revision"), trust_remote_code=backend["trust_remote_code"])
            commit = current.init_kwargs.get("_commit_hash")
        current_tokenizer = {"path": backend["tokenizer"], "files": _task_tokenizer_files(backend["tokenizer"]),
                             "commit": commit}
    seal_check("task model provenance", model, current_model, where="task replay",
               refusal=lambda: ValueError("task model provenance differs"))
    seal_check("task tokenizer provenance", tokenizer, current_tokenizer, where="task replay",
               refusal=lambda: ValueError("task tokenizer provenance differs"))
    if dev:
        result.update(dev_stamp(timestamped=False))
        limit = "Stored dev metrics do not measure replacement model or tokenizer bytes."
        if limit not in result["limitations"]:
            result["limitations"].append(limit)


def _versions():
    return {name: version(name) for name in ("lm-eval", "torch", "transformers", "datasets")}


def preflight_tasks(config):
    validate_config(config)
    from transformers import AutoConfig, AutoTokenizer
    from lm_eval.tasks import TaskManager, get_task_dict
    backend = config["backend"]
    model_config = AutoConfig.from_pretrained(backend["pretrained"], revision=backend.get("revision"),
                                            trust_remote_code=backend["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(backend["tokenizer"], revision=backend.get("tokenizer_revision"),
                                             trust_remote_code=backend["trust_remote_code"])
    tasks = get_task_dict(config["tasks"], task_manager=TaskManager())
    return {"identity": {"backend": _versions(), "model_type": model_config.model_type,
                         "model_commit": getattr(model_config, "_commit_hash", None),
                         "tokenizer": backend["tokenizer"], "tokenizer_vocab_size": len(tokenizer)},
            "population": {"tasks": sorted(tasks), "sampling": config["sampling"], "device": "cpu", "skips": []},
            "limitations": ["Preflight loads task and tokenizer metadata. It does not run model inference."]}


def measure_tasks(config, output):
    validate_config(config)
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    backend = dict(config["backend"])
    backend.pop("name")
    tokenizer_revision = backend.pop("tokenizer_revision", None)
    if tokenizer_revision is not None:
        from transformers import AutoTokenizer
        backend["tokenizer"] = AutoTokenizer.from_pretrained(backend["tokenizer"], revision=tokenizer_revision,
                                                            trust_remote_code=backend["trust_remote_code"])
    if backend["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("task backend requested CUDA without a visible device")
    lm = HFLM(**backend)
    sample = config["sampling"]
    raw = simple_evaluate(model=lm, tasks=config["tasks"], num_fewshot=sample["num_fewshot"],
        limit=sample["limit"], random_seed=sample["random_seed"], numpy_random_seed=sample["numpy_seed"],
        torch_random_seed=sample["torch_seed"], fewshot_random_seed=sample["fewshot_seed"],
        log_samples=True, bootstrap_iters=0)
    if raw is None:
        raise ValueError("task backend returned no result")
    metrics = task_metrics(raw)
    population = task_population(raw, config)
    payload = json.loads(json.dumps(raw, default=handle_non_serializable, allow_nan=False))
    raw_path = Path(output).with_suffix(".lm_eval.json")
    write_result(raw_path, payload)
    source = _task_model_artifact(config["backend"], model_config=lm.model.config)
    tokenizer = lm.tokenizer
    identity = {"backend": _versions(), "model_artifact": source, "model": config["backend"]["pretrained"],
                "tokenizer": config["backend"]["tokenizer"], "tokenizer_vocab_size": len(tokenizer),
                "tokenizer_files": _task_tokenizer_files(config["backend"]["tokenizer"]),
                "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
                "task_versions": raw.get("versions"), "task_configs": payload.get("configs")}
    return {"metrics": metrics, "identity": identity,
            "population": {**population, "cuda_available": torch.cuda.is_available(), "skips": []},
            "artifacts": [artifact(raw_path)],
            "limitations": ["A small CPU smoke proves backend execution. It does not qualify GLM quality."]}


def main(argv=None):
    return cli("task_suite", measure_tasks, preflight_tasks, argv)


if __name__ == "__main__":
    raise SystemExit(main())
