#!/usr/bin/env python3
"""Materialize the two DSv4 gold token workloads in a CPU environment."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.dsv4_wikitext_inputs import (
    CORPUS_CONSTRUCTION,
    DATASETS_DISTRIBUTION,
    DATASETS_VERSION,
    DSV4_WIKITEXT_INPUTS_SCHEMA,
    MODEL_WIKITEXT_INPUTS_SCHEMA,
    seal_model_wikitext_inputs,
    wikitext_model_identity,
    FULL_KL_N_SAMPLES,
    FULL_KL_SEQLEN,
    FULL_KL_SPLIT,
    FULL_KL_WINDOW_SEED,
    PPL_N_TOKENS,
    PPL_SPLIT,
    WIKITEXT_CONFIG,
    WIKITEXT_DATASET,
    WIKITEXT_REVISION,
    seal_dsv4_wikitext_inputs,
)
from tools.full_kl_teacher_payload import atomic_json_write, tokenizer_identity


def _load_corpus(*, split: str, cache_dir: str, dataset_repo: str = WIKITEXT_DATASET) -> tuple[object, str, dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_repo,
        WIKITEXT_CONFIG,
        split=split,
        cache_dir=cache_dir,
        revision=WIKITEXT_REVISION,
    )
    rows = [
        row["text"] for row in dataset
        if isinstance(row.get("text"), str) and row["text"].strip()
    ]
    text = "\n\n".join(rows)
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError("WikiText dataset exposes no immutable fingerprint")
    return dataset, text, {
        "name": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "split": split,
        "revision": WIKITEXT_REVISION,
        "fingerprint": fingerprint,
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _build_payload(*, model: Path, cache_dir: str, input_schema: str = "dsv4-v1") -> dict:
    if input_schema not in {"dsv4-v1", "model-v2"}:
        raise ValueError("unsupported WikiText materialization schema")
    observed_datasets = version(DATASETS_DISTRIBUTION)
    if observed_datasets != DATASETS_VERSION:
        raise RuntimeError(
            f"DSv4 token materialization requires datasets=={DATASETS_VERSION}, "
            f"got {observed_datasets}"
        )
    source_config_bytes = (model / "config.json").read_bytes() if input_schema == "model-v2" else None
    tokenizer_attestation = tokenizer_identity(model)
    # The source declares PreTrainedTokenizerFast in tokenizer_config.json.
    # Constructing that class directly avoids importing the DSv4 model config
    # in this CPU-only environment while using the same tokenizer.json backend
    # as the exact serving image's AutoTokenizer.
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    constructor = AutoTokenizer if input_schema == "model-v2" else PreTrainedTokenizerFast
    tokenizer = constructor.from_pretrained(
        model,
        local_files_only=True,
    )
    if len(tokenizer) <= 0:
        raise RuntimeError("DSv4 tokenizer has no vocabulary")

    corpus_options = {"dataset_repo": "Salesforce/wikitext"} if input_schema == "model-v2" else {}
    _train, train_text, train_evidence = _load_corpus(
        split=FULL_KL_SPLIT,
        cache_dir=cache_dir,
        **corpus_options,
    )
    train_ids = tokenizer(
        train_text,
        add_special_tokens=False,
    ).input_ids
    train_total = len(train_ids)
    max_start = train_total - FULL_KL_SEQLEN
    if max_start < FULL_KL_N_SAMPLES:
        raise RuntimeError("WikiText train corpus has too few token windows")
    starts = random.Random(FULL_KL_WINDOW_SEED).sample(
        range(max_start), FULL_KL_N_SAMPLES
    )
    windows = [
        train_ids[start : start + FULL_KL_SEQLEN]
        for start in starts
    ]
    train_evidence["total_tokens"] = train_total

    _test, test_text, test_evidence = _load_corpus(
        split=PPL_SPLIT,
        cache_dir=cache_dir,
        **corpus_options,
    )
    test_ids = tokenizer(
        test_text,
        add_special_tokens=False,
    ).input_ids
    if len(test_ids) < PPL_N_TOKENS:
        raise RuntimeError("WikiText test corpus cannot satisfy the PPL prefix")
    test_evidence["total_tokens"] = len(test_ids)

    import torch

    window_tensor = torch.tensor(windows, dtype=torch.long).contiguous()
    window_sha = hashlib.sha256(
        window_tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()
    ppl_ids = [int(value) for value in test_ids[:PPL_N_TOKENS]]
    ppl_sha = hashlib.sha256(json.dumps(
        ppl_ids,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    payload = {
        "schema": DSV4_WIKITEXT_INPUTS_SCHEMA,
        "datasets_distribution": {
            "name": DATASETS_DISTRIBUTION,
            "version": observed_datasets,
        },
        "corpus_construction": dict(CORPUS_CONSTRUCTION),
        "tokenizer": tokenizer_attestation,
        "full_kl": {
            "dataset": train_evidence,
            "selection": {
                "sampler": (
                    "python.random.Random(seed).sample(range(max_start), "
                    "n_samples)/v1"
                ),
                "window_seed": FULL_KL_WINDOW_SEED,
                "n_samples": FULL_KL_N_SAMPLES,
                "seqlen": FULL_KL_SEQLEN,
                "starts": starts,
            },
            "token_ids": windows,
            "token_ids_tensor_sha256": window_sha,
        },
        "ppl": {
            "dataset": test_evidence,
            "selection": {
                "strategy": (
                    "contiguous_prefix_after_full_corpus_tokenization/v1"
                ),
                "n_tokens": PPL_N_TOKENS,
            },
            "token_ids": ppl_ids,
            "token_ids_sha256": ppl_sha,
        },
    }
    if tokenizer_identity(model) != tokenizer_attestation:
        raise RuntimeError("tokenizer files changed during materialization")
    if input_schema == "model-v2":
        identity = wikitext_model_identity(model)
        if len(tokenizer) > identity["vocab_size"]:
            raise RuntimeError("tokenizer vocabulary exceeds model token domain")
        config_bytes = (model / "config.json").read_bytes()
        if config_bytes != source_config_bytes:
            raise RuntimeError("model config changed during materialization")
        payload.update(schema=MODEL_WIKITEXT_INPUTS_SCHEMA, model=identity,
            source_config={"bytes": len(config_bytes),
                           "sha256": hashlib.sha256(config_bytes).hexdigest()})
        return seal_model_wikitext_inputs(payload,
            expected_tokenizer_identity=tokenizer_attestation,
            expected_model_identity=identity)
    return seal_dsv4_wikitext_inputs(payload,
        expected_tokenizer_identity=tokenizer_attestation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-schema", choices=("dsv4-v1", "model-v2"), default="dsv4-v1")
    args = parser.parse_args()
    model = Path(args.model).resolve(strict=True)
    output = Path(args.output)
    if not model.is_dir():
        parser.error("--model must be a local model directory")
    if output.exists():
        parser.error("refusing to overwrite existing WikiText inputs")
    payload = _build_payload(model=model, cache_dir=args.dataset_cache_dir, input_schema=args.input_schema)
    atomic_json_write(payload, output)
    print(json.dumps({
        "output": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "schema": payload["schema"],
        "semantic_sha256": payload["semantic_sha256"],
        "full_kl_token_ids_tensor_sha256": (
            payload["full_kl"]["token_ids_tensor_sha256"]
        ),
        "ppl_token_ids_sha256": payload["ppl"]["token_ids_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
