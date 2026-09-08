"""Offline, value-closed WikiText token inputs for gold measurements.

The exact Spark serving image intentionally contains no Hugging Face
``datasets`` package.  Dataset materialization is CPU preprocessing, not part
of a GPU-bound measurement.  This contract lets a CPU environment carrying
the one pinned ``datasets`` version produce the two small token selections
once; the streamed teacher and in-process PPL tools then verify exact corpus,
tokenizer, sampling, and token-value identities without importing
``datasets``.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

import torch


DSV4_WIKITEXT_INPUTS_SCHEMA = "prismaquant.dsv4_wikitext_inputs/1"
MODEL_WIKITEXT_INPUTS_SCHEMA = "prismaquant.model_wikitext_inputs/2"
WIKITEXT_INPUT_MODEL_SCHEMA = "prismaquant.wikitext_input_model/1"
# Canonical Salesforce cache fingerprints; corpus bytes remain the v1 pins.
MODEL_DATASET_FINGERPRINTS = {"train": "5d4fb603254a7a5b", "test": "a46124b21ac53738"}
DSV4_WIKITEXT_INPUTS_MAX_BYTES = 1_048_576
DATASETS_DISTRIBUTION = "datasets"
DATASETS_VERSION = "4.6.0"
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CORPUS_CONSTRUCTION = {
    "row_filter": "include iff bool(text.strip()); preserve text verbatim",
    "join_separator": "\n\n",
    "normalization": "none",
}

TOKENIZER_IDENTITY_SHA256 = (
    "9f7ee7cb93b58bf30f278965547e7584b89c848e76c3adfeb92c070a88492de0"
)
TOKENIZER_VOCAB_SIZE = 129_280

FULL_KL_SPLIT = "train"
FULL_KL_DATASET_FINGERPRINT = "7c4dea6941cc4a0a"
FULL_KL_CORPUS_SHA256 = (
    "fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047"
)
FULL_KL_TOTAL_TOKENS = 2_423_186
FULL_KL_N_SAMPLES = 8
FULL_KL_SEQLEN = 512
FULL_KL_WINDOW_SEED = 42
FULL_KL_STARTS = (
    466_956, 104_902, 1_153_556, 1_027_150,
    936_213, 585_264, 429_895, 2_287_433,
)
FULL_KL_TOKEN_IDS_TENSOR_SHA256 = (
    "b3426e9bab87a1c444b04d0ce01fa9cba5ace313b91db2c3f77fc3525e732b22"
)

PPL_SPLIT = "test"
PPL_DATASET_FINGERPRINT = "7ccd6deaa4fc56e5"
PPL_CORPUS_SHA256 = (
    "c5b5caea5bd655cb221545a484f2f0f59d35092a17a66840d7b9513d0b99687d"
)
PPL_TOTAL_TOKENS = 287_597
PPL_N_TOKENS = 8_192
PPL_TOKEN_IDS_SHA256 = (
    "6c23cefbd78c327d6edac566a5c6b419871021b6cf9890ec830713c1de704961"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DSv4WikiTextInputsError(ValueError):
    """The offline WikiText token payload is not the exact release input."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DSv4WikiTextInputsError(
            "WikiText inputs are not strict canonical JSON"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.long).contiguous()
    return hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _strict_json_load(
    path: str | Path, *, expected_sha256: str | None = None,
) -> object:
    def reject_constant(value: str) -> None:
        raise DSv4WikiTextInputsError(
            f"WikiText inputs contain non-JSON constant {value}"
        )

    def reject_duplicate_members(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DSv4WikiTextInputsError(
                    f"WikiText inputs contain duplicate object member {key!r}"
                )
            result[key] = value
        return result

    source = Path(path).resolve(strict=True)
    try:
        size = source.stat().st_size
        if size <= 0 or size > DSV4_WIKITEXT_INPUTS_MAX_BYTES:
            raise DSv4WikiTextInputsError(
                "WikiText inputs size is outside the closed "
                f"1..{DSV4_WIKITEXT_INPUTS_MAX_BYTES}-byte bound"
            )
        raw = source.read_bytes()
        if len(raw) != size:
            raise DSv4WikiTextInputsError("WikiText inputs changed while reading")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise DSv4WikiTextInputsError("WikiText input file SHA256 differs")
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_members,
        )
    except DSv4WikiTextInputsError:
        raise
    except Exception as exc:
        raise DSv4WikiTextInputsError(
            f"could not read DSv4 WikiText inputs: {source}"
        ) from exc


def _validate_tokenizer_identity(
    value: object, *, expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "content_sha256", "files"
    }:
        raise DSv4WikiTextInputsError("tokenizer identity is not closed")
    if value.get("schema") != "prismaquant.tokenizer_identity/1" or (
        value.get("content_sha256") != (
            TOKENIZER_IDENTITY_SHA256 if expected_content_sha256 is None
            else expected_content_sha256
        )
    ):
        raise DSv4WikiTextInputsError("tokenizer value identity differs")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise DSv4WikiTextInputsError("tokenizer file identity is empty")
    for name, descriptor in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"bytes", "sha256"}
            or isinstance(descriptor.get("bytes"), bool)
            or not isinstance(descriptor.get("bytes"), int)
            or int(descriptor["bytes"]) <= 0
            or not isinstance(descriptor.get("sha256"), str)
            or _SHA256_RE.fullmatch(str(descriptor["sha256"])) is None
        ):
            raise DSv4WikiTextInputsError(
                f"tokenizer file descriptor is malformed: {name!r}"
            )
    if canonical_sha256({"files": dict(files)}) != value.get(
        "content_sha256"
    ):
        raise DSv4WikiTextInputsError("tokenizer file digest differs")
    return dict(value)


def _expected_dataset(*, split: str) -> dict[str, object]:
    if split == FULL_KL_SPLIT:
        fingerprint = FULL_KL_DATASET_FINGERPRINT
        corpus_sha256 = FULL_KL_CORPUS_SHA256
        total_tokens = FULL_KL_TOTAL_TOKENS
    elif split == PPL_SPLIT:
        fingerprint = PPL_DATASET_FINGERPRINT
        corpus_sha256 = PPL_CORPUS_SHA256
        total_tokens = PPL_TOTAL_TOKENS
    else:  # pragma: no cover - internal misuse
        raise AssertionError(split)
    return {
        "name": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "split": split,
        "revision": WIKITEXT_REVISION,
        "fingerprint": fingerprint,
        "corpus_sha256": corpus_sha256,
        "total_tokens": total_tokens,
    }


def _token(value: object, *, where: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= TOKENIZER_VOCAB_SIZE
    ):
        raise DSv4WikiTextInputsError(f"{where} contains an invalid token id")
    return int(value)


def validate_dsv4_wikitext_inputs(
    payload: object,
    *,
    expected_tokenizer_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize both pre-tokenized DSv4 gold workloads."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "datasets_distribution", "corpus_construction",
        "tokenizer", "full_kl", "ppl", "semantic_sha256",
    }:
        raise DSv4WikiTextInputsError("WikiText input fields are not closed")
    if payload.get("schema") != DSV4_WIKITEXT_INPUTS_SCHEMA:
        raise DSv4WikiTextInputsError("unsupported WikiText input schema")
    if payload.get("datasets_distribution") != {
        "name": DATASETS_DISTRIBUTION,
        "version": DATASETS_VERSION,
    }:
        raise DSv4WikiTextInputsError("datasets producer version differs")
    if payload.get("corpus_construction") != CORPUS_CONSTRUCTION:
        raise DSv4WikiTextInputsError("WikiText corpus construction differs")
    tokenizer = _validate_tokenizer_identity(payload.get("tokenizer"))
    if (
        expected_tokenizer_identity is not None
        and tokenizer != dict(expected_tokenizer_identity)
    ):
        raise DSv4WikiTextInputsError(
            "WikiText inputs tokenizer differs from the current model"
        )

    full_kl = payload.get("full_kl")
    if not isinstance(full_kl, Mapping) or set(full_kl) != {
        "dataset", "selection", "token_ids", "token_ids_tensor_sha256"
    } or full_kl.get("dataset") != _expected_dataset(split=FULL_KL_SPLIT):
        raise DSv4WikiTextInputsError("full-KL dataset identity differs")
    expected_selection = {
        "sampler": (
            "python.random.Random(seed).sample(range(max_start), n_samples)/v1"
        ),
        "window_seed": FULL_KL_WINDOW_SEED,
        "n_samples": FULL_KL_N_SAMPLES,
        "seqlen": FULL_KL_SEQLEN,
        "starts": list(FULL_KL_STARTS),
    }
    if full_kl.get("selection") != expected_selection:
        raise DSv4WikiTextInputsError("full-KL window selection differs")
    raw_windows = full_kl.get("token_ids")
    if not isinstance(raw_windows, list) or len(raw_windows) != FULL_KL_N_SAMPLES:
        raise DSv4WikiTextInputsError("full-KL token windows are malformed")
    windows: list[list[int]] = []
    for index, row in enumerate(raw_windows):
        if not isinstance(row, list) or len(row) != FULL_KL_SEQLEN:
            raise DSv4WikiTextInputsError(
                f"full-KL token window {index} has the wrong length"
            )
        windows.append([
            _token(value, where=f"full-KL token window {index}")
            for value in row
        ])
    full_tensor = torch.tensor(windows, dtype=torch.long)
    if (
        full_kl.get("token_ids_tensor_sha256")
        != FULL_KL_TOKEN_IDS_TENSOR_SHA256
        or _tensor_sha256(full_tensor) != FULL_KL_TOKEN_IDS_TENSOR_SHA256
    ):
        raise DSv4WikiTextInputsError("full-KL token values differ")

    ppl = payload.get("ppl")
    if not isinstance(ppl, Mapping) or set(ppl) != {
        "dataset", "selection", "token_ids", "token_ids_sha256"
    } or ppl.get("dataset") != _expected_dataset(split=PPL_SPLIT):
        raise DSv4WikiTextInputsError("PPL dataset identity differs")
    if ppl.get("selection") != {
        "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
        "n_tokens": PPL_N_TOKENS,
    }:
        raise DSv4WikiTextInputsError("PPL token selection differs")
    raw_ppl_ids = ppl.get("token_ids")
    if not isinstance(raw_ppl_ids, list) or len(raw_ppl_ids) != PPL_N_TOKENS:
        raise DSv4WikiTextInputsError("PPL token prefix is malformed")
    ppl_ids = [_token(value, where="PPL token prefix") for value in raw_ppl_ids]
    if (
        ppl.get("token_ids_sha256") != PPL_TOKEN_IDS_SHA256
        or canonical_sha256(ppl_ids) != PPL_TOKEN_IDS_SHA256
    ):
        raise DSv4WikiTextInputsError("PPL token values differ")

    unsigned = {key: value for key, value in payload.items()
                if key != "semantic_sha256"}
    if (
        not isinstance(payload.get("semantic_sha256"), str)
        or payload.get("semantic_sha256") != canonical_sha256(unsigned)
    ):
        raise DSv4WikiTextInputsError("WikiText input semantic digest differs")
    normalized = dict(payload)
    normalized["tokenizer"] = tokenizer
    normalized["full_kl"] = dict(full_kl)
    normalized["full_kl"]["token_ids"] = windows
    normalized["ppl"] = dict(ppl)
    normalized["ppl"]["token_ids"] = ppl_ids
    return normalized


def seal_dsv4_wikitext_inputs(
    payload: Mapping[str, Any],
    *,
    expected_tokenizer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the semantic digest and validate a freshly materialized payload."""
    if "semantic_sha256" in payload:
        raise DSv4WikiTextInputsError("unsealed payload already has a digest")
    sealed = dict(payload)
    sealed["semantic_sha256"] = canonical_sha256(sealed)
    return validate_dsv4_wikitext_inputs(
        sealed,
        expected_tokenizer_identity=expected_tokenizer_identity,
    )


def load_dsv4_wikitext_inputs(
    path: str | Path,
    *,
    expected_tokenizer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_dsv4_wikitext_inputs(
        _strict_json_load(path),
        expected_tokenizer_identity=expected_tokenizer_identity,
    )


def _validate_model_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "model_type", "text_model_type", "vocab_size"
    } or value.get("schema") != WIKITEXT_INPUT_MODEL_SCHEMA:
        raise DSv4WikiTextInputsError("WikiText input model identity is not closed")
    if any(not isinstance(value.get(key), str) or not value[key]
           for key in ("model_type", "text_model_type")) or (
        type(value.get("vocab_size")) is not int or value["vocab_size"] <= 0
    ):
        raise DSv4WikiTextInputsError("WikiText input model token domain is invalid")
    return dict(value)


def normalize_wikitext_model_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project source or candidate config onto its shared token domain."""
    if not isinstance(config, Mapping):
        raise DSv4WikiTextInputsError("WikiText model config is not an object")
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise DSv4WikiTextInputsError("WikiText text config is not an object")
    return _validate_model_identity({
        "schema": WIKITEXT_INPUT_MODEL_SCHEMA,
        "model_type": config.get("model_type"),
        "text_model_type": text.get("model_type", config.get("model_type")),
        "vocab_size": text.get("vocab_size"),
    })


def wikitext_model_identity(model: str | Path) -> dict[str, Any]:
    """Read the token domain; quantization config bytes are provenance only."""
    try:
        return normalize_wikitext_model_identity(json.loads(
            (Path(model) / "config.json").read_text(encoding="utf-8")))
    except DSv4WikiTextInputsError:
        raise
    except (OSError, ValueError, AttributeError) as exc:
        raise DSv4WikiTextInputsError("could not read WikiText model config") from exc


def _model_dataset(value: object, *, split: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DSv4WikiTextInputsError("WikiText dataset identity differs")
    total = value.get("total_tokens")
    if type(total) is not int or total <= 0:
        raise DSv4WikiTextInputsError("WikiText dataset token count is invalid")
    expected = {**_expected_dataset(split=split), "total_tokens": total,
                "fingerprint": MODEL_DATASET_FINGERPRINTS[split]}
    if dict(value) != expected:
        raise DSv4WikiTextInputsError("WikiText dataset identity differs")
    return dict(value)


def validate_model_wikitext_inputs(
    payload: object, *, expected_tokenizer_identity: Mapping[str, Any],
    expected_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v2 values; file intake additionally requires an external SHA."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "datasets_distribution", "corpus_construction", "tokenizer",
        "model", "source_config", "full_kl", "ppl", "semantic_sha256",
    }:
        raise DSv4WikiTextInputsError("WikiText input fields are not closed")
    if payload.get("schema") != MODEL_WIKITEXT_INPUTS_SCHEMA:
        raise DSv4WikiTextInputsError("unsupported WikiText input schema")
    if payload.get("datasets_distribution") != {
        "name": DATASETS_DISTRIBUTION, "version": DATASETS_VERSION,
    }:
        raise DSv4WikiTextInputsError("datasets producer version differs")
    if payload.get("corpus_construction") != CORPUS_CONSTRUCTION:
        raise DSv4WikiTextInputsError("WikiText corpus construction differs")
    model = _validate_model_identity(payload.get("model"))
    if model != _validate_model_identity(expected_model_identity):
        raise DSv4WikiTextInputsError("WikiText input model differs from the current model")
    if not isinstance(expected_tokenizer_identity, Mapping):
        raise DSv4WikiTextInputsError("expected tokenizer identity is missing")
    expected_tokenizer_sha = expected_tokenizer_identity.get("content_sha256")
    if not isinstance(expected_tokenizer_sha, str) or not _SHA256_RE.fullmatch(expected_tokenizer_sha):
        raise DSv4WikiTextInputsError("expected tokenizer identity is malformed")
    tokenizer = _validate_tokenizer_identity(payload.get("tokenizer"),
        expected_content_sha256=expected_tokenizer_sha)
    if tokenizer != dict(expected_tokenizer_identity):
        raise DSv4WikiTextInputsError("WikiText inputs tokenizer differs from the current model")
    source_config = payload.get("source_config")
    if not isinstance(source_config, Mapping) or set(source_config) != {"bytes", "sha256"} or (
        type(source_config.get("bytes")) is not int or source_config["bytes"] <= 0
        or not isinstance(source_config.get("sha256"), str)
        or _SHA256_RE.fullmatch(source_config["sha256"]) is None
    ):
        raise DSv4WikiTextInputsError("source config provenance is malformed")

    def token(value, *, where):
        if type(value) is not int or not 0 <= value < model["vocab_size"]:
            raise DSv4WikiTextInputsError(f"{where} contains an invalid token id")
        return value

    full = payload.get("full_kl")
    if not isinstance(full, Mapping) or set(full) != {
        "dataset", "selection", "token_ids", "token_ids_tensor_sha256",
    }:
        raise DSv4WikiTextInputsError("full-KL fields are not closed")
    train = _model_dataset(full["dataset"], split=FULL_KL_SPLIT)
    max_start = train["total_tokens"] - FULL_KL_SEQLEN
    if max_start < FULL_KL_N_SAMPLES:
        raise DSv4WikiTextInputsError("full-KL corpus cannot satisfy the window selection")
    selection = {
        "sampler": "python.random.Random(seed).sample(range(max_start), n_samples)/v1",
        "window_seed": FULL_KL_WINDOW_SEED, "n_samples": FULL_KL_N_SAMPLES,
        "seqlen": FULL_KL_SEQLEN,
        "starts": random.Random(FULL_KL_WINDOW_SEED).sample(range(max_start), FULL_KL_N_SAMPLES),
    }
    if full.get("selection") != selection or any(
        type(start) is not int for start in full.get("selection", {}).get("starts", [])
    ):
        raise DSv4WikiTextInputsError("full-KL window selection differs")
    raw_windows = full.get("token_ids")
    if not isinstance(raw_windows, list) or len(raw_windows) != FULL_KL_N_SAMPLES:
        raise DSv4WikiTextInputsError("full-KL token windows are malformed")
    windows = []
    for index, row in enumerate(raw_windows):
        if not isinstance(row, list) or len(row) != FULL_KL_SEQLEN:
            raise DSv4WikiTextInputsError("full-KL token window has the wrong length")
        windows.append([token(value, where=f"full-KL token window {index}") for value in row])
    if full.get("token_ids_tensor_sha256") != _tensor_sha256(torch.tensor(windows, dtype=torch.long)):
        raise DSv4WikiTextInputsError("full-KL token values differ")

    ppl = payload.get("ppl")
    if not isinstance(ppl, Mapping) or set(ppl) != {
        "dataset", "selection", "token_ids", "token_ids_sha256",
    }:
        raise DSv4WikiTextInputsError("PPL fields are not closed")
    test = _model_dataset(ppl["dataset"], split=PPL_SPLIT)
    if test["total_tokens"] < PPL_N_TOKENS or ppl.get("selection") != {
        "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
        "n_tokens": PPL_N_TOKENS,
    }:
        raise DSv4WikiTextInputsError("PPL token selection differs")
    raw_ppl = ppl.get("token_ids")
    if not isinstance(raw_ppl, list) or len(raw_ppl) != PPL_N_TOKENS:
        raise DSv4WikiTextInputsError("PPL token prefix is malformed")
    ppl_ids = [token(value, where="PPL token prefix") for value in raw_ppl]
    if ppl.get("token_ids_sha256") != canonical_sha256(ppl_ids):
        raise DSv4WikiTextInputsError("PPL token values differ")
    unsigned = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    if payload.get("semantic_sha256") != canonical_sha256(unsigned):
        raise DSv4WikiTextInputsError("WikiText input semantic digest differs")
    return {**payload, "model": model, "tokenizer": tokenizer,
            "full_kl": {**full, "token_ids": windows}, "ppl": {**ppl, "token_ids": ppl_ids}}


def seal_model_wikitext_inputs(
    payload: Mapping[str, Any], *, expected_tokenizer_identity: Mapping[str, Any],
    expected_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if "semantic_sha256" in payload:
        raise DSv4WikiTextInputsError("unsealed payload already has a digest")
    sealed = {**payload, "semantic_sha256": canonical_sha256(payload)}
    return validate_model_wikitext_inputs(sealed,
        expected_tokenizer_identity=expected_tokenizer_identity,
        expected_model_identity=expected_model_identity)


def load_wikitext_inputs(
    path: str | Path, *, expected_tokenizer_identity: Mapping[str, Any],
    expected_model_identity: Mapping[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read recognized versions, requiring an independently pinned v2 file."""
    payload = _strict_json_load(path, expected_sha256=expected_sha256)
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    if schema == DSV4_WIKITEXT_INPUTS_SCHEMA:
        return validate_dsv4_wikitext_inputs(payload,
            expected_tokenizer_identity=expected_tokenizer_identity)
    if schema != MODEL_WIKITEXT_INPUTS_SCHEMA:
        raise DSv4WikiTextInputsError("unsupported WikiText input schema")
    if expected_sha256 is None:
        raise DSv4WikiTextInputsError("model-v2 inputs require an independent file SHA256")
    return validate_model_wikitext_inputs(payload,
        expected_tokenizer_identity=expected_tokenizer_identity,
        expected_model_identity=expected_model_identity)


__all__ = [name for name in globals() if name.startswith(("DSV4_", "FULL_", "PPL_", "TOKENIZER_", "WIKITEXT_"))] + [
    "CORPUS_CONSTRUCTION",
    "DATASETS_DISTRIBUTION",
    "DATASETS_VERSION",
    "DSv4WikiTextInputsError",
    "canonical_sha256",
    "load_dsv4_wikitext_inputs",
    "seal_dsv4_wikitext_inputs",
    "validate_dsv4_wikitext_inputs",
    "MODEL_WIKITEXT_INPUTS_SCHEMA",
    "wikitext_model_identity",
    "normalize_wikitext_model_identity",
    "load_wikitext_inputs",
    "seal_model_wikitext_inputs",
    "validate_model_wikitext_inputs",
]
