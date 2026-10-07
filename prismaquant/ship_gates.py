"""Run one configured ship-gates sequence inside an admitted action.

This module starts stage processes, not jobs. PrismaBuild owns admission and
multi-host placement. Existing producers and the shipcard verifier own gates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from prismaquant.cost_stage_checkpoint import atomic_write_bytes
from prismaquant.digests import file_sha256hex
from prismaquant.model_profiles import detect_profile
from prismaquant.shipcard import _strict_json_object, load_shipcard, required_slots, verify

CONFIG_SCHEMA = "prismaquant.ship_gates/1"
RESULT_SCHEMA = "prismaquant.ship_gates_result/1"
QUALITY_STAGES = {"offline.g3": "g3_v2", "task_suite": "task_suite"}
GOLD_TOOLS = {
    "gold.kl": "tools.measure_vllm_full_kl",
    "gold.ppl": "tools.measure_vllm_wikitext_ppl",
}


class OutputConflict(ValueError):
    """An output path would change an input or retained result."""


def _read(path: Path) -> dict:
    value = dict(_strict_json_object(path.read_bytes(), where=str(path)))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    atomic_write_bytes(path, (json.dumps(value, sort_keys=True, indent=2,
                                       allow_nan=False) + "\n").encode())


def _validate_destinations(destinations: list[Path], protected: set[Path], artifact: Path) -> None:
    if len(set(destinations)) != len(destinations):
        raise OutputConflict("stage, log and job output paths must be distinct")
    if any(destination in protected or artifact == destination or artifact in destination.parents
           for destination in destinations):
        raise OutputConflict("outputs must not overwrite inputs or enter the artifact")


def load_config(path: str | Path, *, report_output: Path | None = None,
                preflight: bool = False) -> tuple[dict, dict]:
    """Resolve the existing profile and the exact card's required slot set."""
    path = Path(path).resolve(strict=True)
    config = _read(path)
    fields = {"schema", "artifact", "topology", "serve_image", "stages", "inputs"}
    if set(config) != fields or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("ship-gates configuration fields or schema differ")
    artifact = Path(config["artifact"]).resolve(strict=True)
    if not artifact.is_dir():
        raise ValueError("artifact must be a directory")
    if report_output is not None and (artifact == report_output or artifact in report_output.parents):
        raise OutputConflict("job output must stay outside the artifact")
    profile = detect_profile(artifact)
    card = load_shipcard(artifact / "shipcard.json")
    from prismaquant.dev_mode import seal_check
    seal_check("artifact_path", str(Path(card.get("model_dir", "")).resolve()),
               str(artifact), where="ship-gates card", refusal=ValueError)
    topology = config["topology"]
    if not isinstance(topology, dict):
        raise ValueError("topology must be an object")
    from tools.gold_engine_options import gold_engine_kwargs
    engine = gold_engine_kwargs(argparse.Namespace(**topology))
    if "tensor_parallel_size" not in topology or "nnodes" not in topology:
        raise ValueError("topology requires explicit tensor_parallel_size and nnodes")
    allowed = {"tensor_parallel_size", "nnodes", "node_rank", "master_addr", "master_port",
               "distributed_executor_backend", "data_parallel_backend", "moe_backend",
               "kv_cache_memory_bytes"}
    if set(topology) - allowed:
        raise ValueError("unknown topology fields")
    if not isinstance(config["serve_image"], str) or not config["serve_image"]:
        raise ValueError("serve_image must be explicit")
    stages = config["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a nonempty ordered list")
    owed = set(required_slots(card, model_dir=artifact)) | set(QUALITY_STAGES)
    ids = [stage.get("id") for stage in stages if isinstance(stage, dict)]
    if len(ids) != len(stages) or len(set(ids)) != len(ids) or set(ids) != owed:
        raise ValueError(f"stage set must close every required slot plus quality: {sorted(owed)}")
    for stage in stages:
        identity = stage["id"]
        expected = {"id", "config", "output"} if identity in QUALITY_STAGES else (
            {"id", "args", "output"} if identity in GOLD_TOOLS else {"id", "argv", "record", "output"})
        if set(stage) != expected:
            raise ValueError(f"{identity}: stage fields differ from {sorted(expected)}")
        if identity in QUALITY_STAGES:
            _read(Path(stage["config"]).resolve(strict=True))
        else:
            argv = stage.get("args", stage.get("argv"))
            if not isinstance(argv, list) or not all(isinstance(word, str) for word in argv):
                raise ValueError(f"{identity}: arguments must be a string array")
            if identity not in GOLD_TOOLS and not argv:
                raise ValueError(f"{identity}: argv is empty")
            if any(Path(word).name in {"pbrun.py", "pbtest.py", "pbgang.py", "pbcampaign.py"}
                   for word in argv):
                raise ValueError("stage processes must not submit nested jobs")
            if identity in GOLD_TOOLS:
                reserved = {"--model", "--output", "--preflight", "--serve-image", "--mode"}
                reserved.update("--" + name.replace("_", "-") for name in allowed)
                if any(word.split("=", 1)[0] in reserved for word in argv):
                    raise ValueError(f"{identity}: model, output and topology are runner-owned")
    inputs = config["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must name the files the job reads")
    input_evidence = []
    for item in inputs:
        if not isinstance(item, str):
            raise ValueError("input path must be a string")
        source = Path(item).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"input is not a file: {source}")
        input_evidence.append({"path": str(source), "bytes": source.stat().st_size,
                               "sha256": file_sha256hex(source)})
    protected = {path} | {Path(item).resolve() for item in inputs}
    protected.update(Path(stage["config"]).resolve() for stage in stages if "config" in stage)
    configured_outputs = [Path(stage["output"]).resolve() for stage in stages]
    if preflight and report_output is None:
        raise ValueError("preflight requires a job output path")
    actual_outputs = [(report_output.parent / (report_output.stem + ".stages")
                       / (stage["id"] + ".json")).resolve() if preflight else configured
                      for stage, configured in zip(stages, configured_outputs)]
    actual_logs = [destination.with_suffix(".log") for destination in actual_outputs]
    destinations = configured_outputs + [path.with_suffix(".log") for path in configured_outputs]
    for configured, actual, log in zip(configured_outputs, actual_outputs, actual_logs):
        if actual != configured:
            destinations.extend((actual, log))
    if report_output is not None:
        destinations.append(report_output)
    _validate_destinations(destinations, protected, artifact)
    stage_destinations = {stage["id"]: {"output": str(actual), "log": str(log)}
                          for stage, actual, log in zip(stages, actual_outputs, actual_logs)}

    return config, {"artifact": str(artifact), "profile": profile.name,
                    "architectures": list(profile.declared_architectures()),
                    "lane": card.get("lane"), "model_sha": card["model_sha"],
                    "topology": engine, "required_slots": sorted(owed - set(QUALITY_STAGES)),
                    "inputs": input_evidence, "config_sha256": file_sha256hex(path),
                    "stage_destinations": stage_destinations}


def _command(stage: dict, context: dict, output: Path, image: str,
             *, preflight: bool) -> list[str]:
    identity = stage["id"]
    if identity in QUALITY_STAGES:
        command = [sys.executable, "-m", "prismaquant." + QUALITY_STAGES[identity],
                   "--config", stage["config"], "--output", str(output)]
    elif identity in GOLD_TOOLS:
        command = [sys.executable, "-m", GOLD_TOOLS[identity],
                   "--model", context["artifact"], "--output", str(output),
                   "--serve-image", image]
        if identity == "gold.kl":
            command += ["--mode", "student"]
        for name, value in context["topology"].items():
            command += ["--" + name.replace("_", "-"), str(value)]
        command += stage["args"]
    else:
        values = {"artifact": context["artifact"],
                  "shipcard": str(Path(context["artifact"]) / "shipcard.json"),
                  "output": str(output), "image": image,
                  "tp": str(context["topology"]["tensor_parallel_size"]),
                  "nnodes": str(context["topology"]["nnodes"])}
        # Tokens are whole arguments. JSON braces and shell text stay literal.
        return [values.get(word[1:-1], word) if word.startswith("{") and word.endswith("}")
                else word for word in stage["argv"]]
    if preflight:
        command.append("--preflight")
    return command


def _replay_quality(stage: dict, output: Path, *, preflight: bool, artifact: str) -> dict:
    from prismaquant.quality_stage import verify_result
    config_path = Path(stage["config"])
    config = _read(config_path)
    if stage["id"] == "task_suite":
        measured = config.get("backend", {}).get("pretrained")
        if not isinstance(measured, str) or Path(measured).resolve() != Path(artifact).resolve():
            raise ValueError("task backend must measure the configured current artifact")
    result = _read(output)
    gate = verify_result(result, config, config_sha256=file_sha256hex(config_path))
    if result.get("stage") != QUALITY_STAGES[stage["id"]]:
        raise ValueError("quality result stage differs")
    if stage["id"] == "offline.g3" and result["measurement"]["metric_kind"] != "offline_decoded_kl":
        raise ValueError("offline G3 must not claim served KL")
    expected_measurement = "preflight" if preflight else "succeeded"
    expected_gate = "not_evaluated" if preflight else "passed"
    if gate.get("status") != expected_gate:
        raise ValueError("replayed quality criteria did not pass")
    if result["measurement"]["status"] != expected_measurement or result["gate"]["status"] != expected_gate:
        raise ValueError(f"quality measurement/gate must be {expected_measurement}/{expected_gate}")
    return result


def _replay_slot(stage: dict, context: dict, output: Path, *, produced: bool) -> dict:
    from prismaquant.shipcard import fill_slot
    from prismaquant.shipcard_cli import main as card_cli
    artifact = Path(context["artifact"])
    card_path = artifact / "shipcard.json"
    identity = stage["id"]
    if produced and identity in GOLD_TOOLS:
        rc = card_cli(["fill", str(card_path), "--slot", identity, "--record", str(output),
                       "--model-dir", str(artifact), "--tool", GOLD_TOOLS[identity]])
        if rc:
            raise ValueError(f"{identity}: gold fill refused with exit {rc}")
    card = load_shipcard(card_path)
    if produced and identity not in GOLD_TOOLS:
        record_path = stage["record"]
        source = card_path if record_path == "{shipcard}" else (
            output if record_path == "{output}" else Path(record_path))
        document = _read(source)
        record = document.get("slots", {}).get(identity) if source == card_path else document
        candidate = copy.deepcopy(card)
        candidate["slots"][identity] = record
        problems = verify(candidate, model_dir=artifact, required=[identity])
        if problems:
            raise ValueError("; ".join(problems))
        if source != card_path:
            fill_slot(card_path, identity, record)
            card = load_shipcard(card_path)
    problems = verify(card, model_dir=artifact, required=[identity])
    if problems:
        raise ValueError("; ".join(problems))
    return card["slots"][identity]


def run(config_path: str | Path, output: str | Path, *, preflight: bool = False,
        verify_only: bool = False) -> dict:
    """Keep every stage outcome. A CPU preflight can never pass qualification."""
    output = Path(output).resolve()
    report = {"schema": RESULT_SCHEMA, "status": "running", "mode":
              "preflight" if preflight else ("verify_only" if verify_only else "execute"),
              "runtime_qualification": "not_run", "stages": [], "problems": []}
    try:
        if output.exists():
            raise OutputConflict("job output already exists; use a new result path")
        config, context = load_config(config_path, report_output=output, preflight=preflight)
        report["context"] = context
        import torch
        from prismaquant.shipcard import git_provenance
        report["source"] = git_provenance()
        report["population"] = {"torch": str(torch.__version__),
                                "cuda_available": torch.cuda.is_available(),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "devices": [torch.cuda.get_device_name(index)
                                            for index in range(torch.cuda.device_count())]}
        report["stages"] = [{"id": stage["id"], "status": "not_run"}
                            for stage in config["stages"]]
        if not preflight and not verify_only and context["topology"]["nnodes"] > 1:
            raise ValueError("multi-host ordered stages need the serving owner's rank lifecycle driver")
        if not verify_only and any(Path(paths["log"]).exists()
                                   for paths in context["stage_destinations"].values()):
            raise OutputConflict("stage log already exists; use a new result path")
        _write(output, report)
        stopped = False
        for stage, outcome in zip(config["stages"], report["stages"]):
            identity = stage["id"]
            stage_output = Path(context["stage_destinations"][identity]["output"])
            stage_output.parent.mkdir(parents=True, exist_ok=True)
            outcome["output"] = str(stage_output)
            if stopped and not verify_only:
                outcome["reason"] = "An earlier stage failed."
                continue
            if preflight and identity not in QUALITY_STAGES and identity not in GOLD_TOOLS:
                outcome.update(status="not_run", reason="This gate requires serving evidence.")
                _write(output, report)
                continue
            try:
                if not verify_only:
                    if stage_output.exists():
                        raise ValueError("stage output already exists; use a new result path")
                    command = _command(stage, context, stage_output, config["serve_image"],
                                       preflight=preflight)
                    outcome["argv"] = command
                    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1")
                    if preflight:
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    log_path = Path(context["stage_destinations"][identity]["log"])
                    outcome["log"] = str(log_path)
                    with log_path.open("xb") as log:
                        completed = subprocess.run(command, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, check=False)
                    outcome["returncode"] = completed.returncode
                    if stage_output.is_file():
                        outcome["sha256"] = file_sha256hex(stage_output)
                        outcome["result"] = _read(stage_output)
                    if completed.returncode:
                        raise ValueError(f"stage process exited {completed.returncode}")
                if identity in QUALITY_STAGES:
                    result = _replay_quality(stage, stage_output, preflight=preflight,
                                             artifact=context["artifact"])
                elif preflight:
                    result = _read(stage_output)
                    if (result.get("schema") != "prismaquant.gold_preflight/1"
                            or result.get("stage") != identity
                            or result.get("status") != "preflight"
                            or result.get("runtime_qualification") != "not_run"):
                        raise ValueError("gold preflight cannot stand in for a served result")
                else:
                    result = _replay_slot(stage, context, stage_output, produced=not verify_only)
                outcome.update(status="preflight" if preflight else "passed", result=result)
                if result.get("dev_uncertified") is True:
                    from prismaquant.dev_mode import dev_stamp
                    report.update(dev_stamp(timestamped=False))
                if stage_output.is_file():
                    outcome["sha256"] = file_sha256hex(stage_output)
            except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                outcome.update(status="failed", error=str(exc))
                report["problems"].append(f"{identity}: {exc}")
                stopped = True
            _write(output, report)
        if not preflight:
            problems = verify(load_shipcard(Path(context["artifact"]) / "shipcard.json"),
                              model_dir=context["artifact"])
            report["problems"].extend(problems)
            report["runtime_qualification"] = "passed" if not report["problems"] else "refused"
        report["status"] = ("refused" if report["problems"] else
                            ("preflight" if preflight else "passed"))
    except OutputConflict as exc:
        report["problems"].append(str(exc))
        report["status"] = "refused"
        return report
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        report["problems"].append(str(exc))
        report["status"] = "refused"
    _write(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.config, args.output, preflight=args.preflight, verify_only=args.verify_only)
    evidence = []
    result_path = Path(args.output).resolve()
    if result_path.is_file():
        evidence.append({"path": str(result_path), "sha256": file_sha256hex(result_path)})
    for stage in report["stages"]:
        for key in ("output", "log"):
            path = Path(stage[key]) if key in stage else None
            if path is not None and path.is_file():
                evidence.append({"path": str(path), "sha256": file_sha256hex(path)})
    print(json.dumps({"status": report["status"], "evidence": evidence,
                      "problems": report["problems"]}))
    return 1 if report["status"] == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
