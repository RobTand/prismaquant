"""Replay retained CPU API protocol evidence without contacting an endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismaquant.digests import bytes_sha256hex
from prismaquant.quality_stage import finite_number
from prismaquant.schemas import strict_json_loads
from prismaquant.task_suite import task_metrics, task_population


def _proof_json(raw, label):
    return strict_json_loads(raw,
        duplicate=lambda key: ValueError(f"{label}: duplicate key {key}"),
        constant=lambda value: ValueError(f"{label}: nonfinite {value}"))


def _retained_api_files(directory):
    root = Path(directory).resolve()
    manifest = _proof_json((root / "manifest.json").read_bytes(), "manifest")
    if manifest["format_version"] != 1:
        raise ValueError("unsupported retained fixture format")
    files, originals = {}, {}
    for name, row in manifest["files"].items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or root not in path.parents:
            raise ValueError("evidence path escapes the fixture")
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"missing evidence: {name}") from exc
        if len(raw) != row["bytes"] or bytes_sha256hex(raw) != row["sha256"]:
            raise ValueError(f"evidence bytes differ: {name}")
        if row["original_path"] in originals:
            raise ValueError("duplicate original evidence path")
        files[name] = raw
        originals[row["original_path"]] = raw
    return manifest, files, originals


def _proof_bound(reference, originals):
    raw = originals.get(reference["path"])
    if raw is None or len(raw) != reference["bytes"] or bytes_sha256hex(raw) != reference["sha256"]:
        raise ValueError("retained evidence binding differs: " + reference["path"])
    return raw


def _proof_task(kind, files, inputs):
    prefix = "proof3/llama_cpp/" + kind
    raw = _proof_json(files[prefix + ".result.json"], kind + " raw result")
    config = _proof_json(files[prefix + ".config.json"], kind + " configuration")
    task = config["tasks"][0]
    name = inputs[kind]["task"]
    if len(config["tasks"]) != 1 or any(task.get(key) != value for key, value in inputs[kind].items()):
        raise ValueError("task definitions differ from the retained inputs")
    declared = raw["configs"][name]
    if any(declared.get(key) != value for key, value in task.items() if key != "metadata"):
        raise ValueError("raw task definitions differ")
    if any(declared["metadata"].get(key) != value for key, value in task["metadata"].items()):
        raise ValueError("raw task metadata differs")
    if raw["versions"] != {name: task["metadata"]["version"]}:
        raise ValueError("raw task versions differ")
    sampling = {"limit": inputs["limit"], "num_fewshot": inputs["num_fewshot"],
                **{key: inputs["seed"] for key in ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")}}
    for key, value in sampling.items():
        configured_key = {"numpy_seed": "numpy_random_seed", "torch_seed": "torch_random_seed",
                          "fewshot_seed": "fewshot_random_seed"}.get(key, key)
        if config[configured_key] != value:
            raise ValueError("task configuration sampling differs: " + key)
    population = task_population(raw, {"backend": {"device": "cpu"}, "tasks": [task], "sampling": sampling})
    samples = raw["samples"][name]
    if (set(raw["results"]) != {name} or len(samples) != inputs["limit"]
            or population["samples"][name] != {"original": len(inputs["documents"]), "effective": len(samples)}
            or [row["doc_id"] for row in samples] != inputs["sample_ids"]):
        raise ValueError("task population differs from the retained inputs")
    for sample in samples:
        if sample["doc"] != inputs["documents"][sample["doc_id"]] or sample["filter"] != "none":
            raise ValueError("task sample definition or filter differs")
    return raw, samples, task_metrics(raw)


def _proof_score(exchange, context, continuation):
    request, response = exchange["request_json"], exchange["response_json"]
    prompt = context + continuation
    probs = response["choices"][0]["logprobs"]
    tokens, offsets, values = (probs[key] for key in ("tokens", "text_offset", "token_logprobs"))
    if (request["prompt"] != prompt or not continuation or len(tokens) < 2
            or len(values) != len(tokens) or len(offsets) != len(tokens)
            or len(probs["top_logprobs"]) != len(tokens) or "".join(tokens[:-1]) != prompt):
        raise ValueError("scoring token alignment differs from the request")
    position = 0
    for token, offset in zip(tokens, offsets):
        if type(offset) is not int or offset != position:
            raise ValueError("scoring token alignment has invalid offsets")
        position += len(token)
    if offsets.count(len(context)) != 1:
        raise ValueError("scoring token alignment has no unique continuation boundary")
    start = offsets.index(len(context))
    selected = [finite_number(value, "continuation logprob") for value in values[start:-1]]
    if not selected or "".join(tokens[start:-1]) != continuation:
        raise ValueError("scoring token alignment has no continuation tokens")
    total = sum(selected)
    tops = probs["top_logprobs"][start:-1]
    greedy = all(token == max(top, key=top.get) for token, top in zip(tokens[start:-1], tops))
    return {"http_response_sha256": exchange["response"]["sha256"],
            "context_tokens": start, "continuation_tokens": len(selected),
            "selected_logprobs": selected, "sum_logprobs": total}, [total, greedy]


def replay_proof(directory):
    """Verify fixture custody and independently replay its saved task results."""
    manifest, files, originals = _retained_api_files(directory)
    load = lambda name: _proof_json(files[name], name)
    try:
        recipe = load("reusable-recipe.json")
        inputs = load("task-inputs.json")
        capabilities = load("proof3/capabilities.json")
        transcript = load("proof3/transcript.json")
        client = load("proof3/llama_cpp/client.json")
        server = capabilities["servers"]["llama_cpp"]
        if (recipe["accepted_source_base"] != manifest["historical"]["source_base"]
                or capabilities["source"] != {"accepted_base": manifest["historical"]["source_base"],
                    "snapshot_head": manifest["historical"]["snapshot_commit"]}
                or recipe["llama_cpp"]["proof_action_key"] != manifest["historical"]["action_key"]
                or recipe["native_evidence"]["receipt"]["receipt_sha256"] != manifest["historical"]["receipt_sha256"]):
            raise ValueError("historical source or native receipt reference differs")
        for reference in (recipe["task_inputs"], recipe["task_data"], capabilities["inputs"],
                          capabilities["data"], capabilities["transcript"]):
            _proof_bound(reference, originals)
        documents = [_proof_json(line, "task data") for line in files["task-data.jsonl"].splitlines()]
        if documents != inputs["documents"]:
            raise ValueError("task population data differs")
        if (client != recipe["llama_cpp"]["client_args"] or client != server["client_args"]
                or client["base_url"] != server["endpoint"] or client["model"] != server["model_alias"]
                or client["seed"] != inputs["seed"]):
            raise ValueError("retained public client configuration differs")
        if capabilities["devices"]["cuda_device_count"] != 0 or capabilities["skips"]:
            raise ValueError("retained CPU population differs")
        generation, scoring = {}, {}
        for exchange in transcript:
            response = _proof_json(_proof_bound(exchange["response"], originals), "HTTP response")
            request_bytes = _proof_bound(exchange["request"], originals)
            if response != exchange["response_json"] or exchange["http_status"] != 200:
                raise ValueError("HTTP response differs from its transcript")
            if exchange["method"] == "GET":
                if request_bytes or [row["id"] for row in response["data"]] != [client["model"]]:
                    raise ValueError("retained model-list response differs")
                continue
            request = _proof_json(request_bytes, "HTTP request")
            if request != exchange["request_json"] or exchange["url"] != client["base_url"]:
                raise ValueError("HTTP request differs from its transcript")
            if request["model"] != client["model"] or request["seed"] != inputs["seed"]:
                raise ValueError("HTTP model or sampling differs")
            target = scoring if request.get("echo") is True else generation
            if request["prompt"] in target:
                raise ValueError("duplicate task request population")
            target[request["prompt"]] = exchange
        _, score_samples, metrics = _proof_task("scoring", files, inputs)
        _, gen_samples, gen_metrics = _proof_task("generation", files, inputs)
        metrics.update(gen_metrics)
        scored, correct = {}, []
        for sample in score_samples:
            arguments, scores = sample["arguments"], []
            expected = [[sample["doc"]["question"], " " + choice] for choice in sample["doc"]["choices"]]
            if arguments != expected or sample["target"] != sample["doc"]["gold"]:
                raise ValueError("scoring task definitions differ")
            for index, (context, continuation) in enumerate(arguments):
                exchange = scoring.pop(context + continuation)
                request = exchange["request_json"]
                if request["max_tokens"] != 1 or request["logprobs"] != 1:
                    raise ValueError("scoring capability request differs")
                summary, result = _proof_score(exchange, context, continuation)
                if sample["resps"][index] != [result] or sample["filtered_resps"][index] != result:
                    raise ValueError("raw scoring result differs from response logprobs")
                scored[summary["http_response_sha256"]] = summary
                scores.append(result[0])
            accuracy = float(max(range(len(scores)), key=scores.__getitem__) == sample["target"])
            if sample["acc"] != accuracy:
                raise ValueError("sample accuracy metric differs")
            correct.append(accuracy)
        name = inputs["scoring"]["task"]
        if metrics[name + "/acc,none"] != sum(correct) / len(correct):
            raise ValueError("aggregate scoring metric differs")
        matches = []
        for sample in gen_samples:
            prompt = sample["doc"]["question"]
            exchange = generation.pop(prompt)
            request = exchange["request_json"]
            text = exchange["response_json"]["choices"][0]["text"]
            if (sample["arguments"] != [[prompt, inputs["generation"]["generation_kwargs"]]]
                    or request["max_tokens"] != inputs["generation_max_tokens"]
                    or sample["target"] != sample["doc"]["answer"]
                    or sample["resps"] != [[text]] or sample["filtered_resps"] != [text]):
                raise ValueError("generation task result differs from response")
            match = float(text == sample["target"])
            if sample["exact_match"] != match:
                raise ValueError("generation sample metric differs")
            matches.append(match)
        if metrics[inputs["generation"]["task"] + "/exact_match,none"] != sum(matches) / len(matches):
            raise ValueError("aggregate generation metric differs")
        if scoring or generation:
            raise ValueError("unmatched HTTP task population")
        ordered = [scored[row["response"]["sha256"]] for row in transcript
                   if row.get("request_json", {}).get("echo") is True]
        return {"execution": "historical_retained_replay", "scope": manifest["scope"],
                "historical": manifest["historical"], "runtime_witness_qualification": "not_run",
                "receipt_attestation_reverified": False,
                "population": {"device": "cpu", "documents": len(documents),
                               "scoring_choices": len(ordered), "generation_requests": len(matches)},
                "metrics": metrics, "scoring": ordered}
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("incomplete retained task API evidence: " + str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = replay_proof(args.evidence_dir)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
